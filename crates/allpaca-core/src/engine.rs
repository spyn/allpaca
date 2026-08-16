use std::sync::{Arc, Mutex};
use std::time::Instant;

use allpaca_gguf::{extract_tokenizer, GgufFile};
use allpaca_types::{
    AllpacaError, Config, DraftProviderKind, FatalError, FeatureFlags, FinishReason,
    GenerationOutput, GenerationParams, StatsSnapshot, StepResult, TokenEvent, TransientError,
};
use futures::Stream;
use rand::rngs::StdRng;

use crate::constraint::JsonConstraint;
use crate::device_manager::DeviceInventory;
use crate::draft::{DraftModelProvider, DraftProvider, NGramProvider};
use crate::kv_tier::KvTierManager;
use crate::logits_ring::LogitRing;
use crate::model::{alloc_pages_for_seq, ensure_pages, ModelConfig, TinyModel};
use crate::sampling::{rng_from_params, sample};
use crate::scheduler::Scheduler;
use crate::telemetry::Metrics;
use crate::tokenizer::EngineTokenizer;

struct ActiveRequest {
    id: u64,
    tokens: Vec<u32>,
    prompt_len: usize,
    generated: usize,
    params: GenerationParams,
    pages: Vec<Vec<u64>>,
    pending_logits: Vec<f32>,
    emit_queue: Vec<TokenEvent>,
    json_prefix: String,
    rng: StdRng,
    started: Instant,
    last_token_at: Instant,
    first_emitted: bool,
    finished: bool,
}

struct EngineInner {
    flags: FeatureFlags,
    scheduler: Scheduler,
    kv: KvTierManager,
    ring: LogitRing,
    tokenizer: Option<EngineTokenizer>,
    target: Option<Arc<TinyModel>>,
    draft_model: Option<Arc<TinyModel>>,
    ngram: NGramProvider,
    active: Option<ActiveRequest>,
    next_request_id: u64,
    model_id: String,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<EngineInner>>,
    config: Config,
    inventory: DeviceInventory,
    metrics: Arc<Metrics>,
}

pub struct RequestHandle {
    id: u64,
    inner: Arc<Mutex<EngineInner>>,
    metrics: Arc<Metrics>,
}

impl std::fmt::Debug for RequestHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestHandle").field("id", &self.id).finish()
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if guard.active.as_ref().map(|a| a.id) == Some(self.id) {
                if let Some(active) = guard.active.take() {
                    for layer in active.pages {
                        for page in layer {
                            guard.kv.release(page, &self.metrics);
                        }
                    }
                }
                self.metrics
                    .active_requests
                    .store(0, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

impl Engine {
    pub fn new(config: Config) -> Result<Self, AllpacaError> {
        config.validate()?;
        let inventory = DeviceInventory::discover(&config);
        let mut flags = FeatureFlags::from_mode(inventory.pipeline_mode);
        flags.overlay(&config.features);
        apply_provider_flags(&mut flags, &config);
        let scheduler = Scheduler::new(inventory.pipeline_mode, flags.clone(), &config.speculative);
        let kv = KvTierManager::new(
            config.kv.vram_bytes,
            config.kv.dram_bytes,
            config.kv.hot_fraction,
        );
        let ring = LogitRing::new(config.sampling.pinned_ring_slots, 64);
        let inner = EngineInner {
            flags,
            scheduler,
            kv,
            ring,
            tokenizer: None,
            target: None,
            draft_model: None,
            ngram: NGramProvider::new(config.speculative.ngram_n),
            active: None,
            next_request_id: 1,
            model_id: config.model.id.clone(),
        };
        let engine = Self {
            inner: Arc::new(Mutex::new(inner)),
            config: config.clone(),
            inventory,
            metrics: Arc::new(Metrics::default()),
        };
        if config.model.synthetic {
            engine.load_synthetic(ModelConfig::tiny())?;
        } else if let Some(path) = &config.model.path {
            engine.load_gguf(path)?;
        }
        Ok(engine)
    }

    pub fn inventory(&self) -> &DeviceInventory {
        &self.inventory
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn stats(&self) -> StatsSnapshot {
        let model_id = self
            .lock()
            .ok()
            .and_then(|g| g.target.as_ref().map(|m| m.name.clone()))
            .unwrap_or_else(|| self.config.model.id.clone());
        self.metrics.snapshot(
            &format!("{:?}", self.inventory.pipeline_mode),
            &format!("{:?}", self.inventory.transfer_path),
            self.inventory.target_device,
            self.inventory.draft_device,
            model_id,
        )
    }

    pub fn model_id(&self) -> String {
        self.lock()
            .ok()
            .map(|g| g.model_id.clone())
            .unwrap_or_else(|| self.config.model.id.clone())
    }

    pub fn has_model(&self) -> bool {
        self.lock()
            .map(|g| g.target.is_some() && g.tokenizer.is_some())
            .unwrap_or(false)
    }

    pub fn load_synthetic(&self, cfg: ModelConfig) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let seed = self.config.engine.determinism_seed.unwrap_or(42);
        let tokenizer = EngineTokenizer::synthetic(cfg.vocab_size)?;
        inner.ring.resize_vocab(cfg.vocab_size);
        inner.target = Some(Arc::new(TinyModel::synthetic(cfg.clone(), seed, "synthetic")));
        if inner.flags.draft_model_speculation {
            inner.draft_model = Some(Arc::new(TinyModel::synthetic(
                ModelConfig::draft_tiny(),
                seed.wrapping_add(1),
                "synthetic-draft",
            )));
        }
        inner.tokenizer = Some(tokenizer);
        inner.model_id = "synthetic".into();
        tracing::info!(vocab = cfg.vocab_size, layers = cfg.n_layers, "loaded synthetic model");
        Ok(())
    }

    pub fn load_gguf(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = GgufFile::open(path)?;
        let tok = extract_tokenizer(&file)?;
        let model = TinyModel::from_gguf(&file)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        inner.tokenizer = Some(EngineTokenizer::from_gguf(tok));
        if let Some(draft_path) = &self.config.model.draft_path {
            match GgufFile::open(draft_path).and_then(|f| TinyModel::from_gguf(&f)) {
                Ok(draft) => inner.draft_model = Some(Arc::new(draft)),
                Err(e) => {
                    tracing::warn!(error = %e, "draft GGUF failed; disabling draft-model speculation");
                    inner.flags.draft_model_speculation = false;
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        Ok(())
    }

    pub fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationOutput, AllpacaError> {
        let mut handle = self.begin_request(prompt, params)?;
        let mut token_ids = Vec::new();
        let finish = loop {
            match self.step(&mut handle)? {
                StepResult::Token(ev) => {
                    token_ids.push(ev.token_id);
                    if let Some(reason) = ev.finish_reason {
                        break reason;
                    }
                }
                StepResult::Finished(reason) => break reason,
            }
        };
        let text = {
            let inner = self.lock()?;
            inner
                .tokenizer
                .as_ref()
                .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
                .decode(&token_ids)?
        };
        let prompt_tokens = handle_prompt_len(&handle, &self.inner)?;
        Ok(GenerationOutput {
            text,
            completion_tokens: token_ids.len(),
            token_ids,
            prompt_tokens,
            finish_reason: finish,
        })
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> impl Stream<Item = Result<TokenEvent, AllpacaError>> + Send {
        let engine = self.clone();
        async_stream::stream! {
            let mut handle = match engine.begin_request(&prompt, params) {
                Ok(h) => h,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            loop {
                match engine.step(&mut handle) {
                    Ok(StepResult::Token(ev)) => {
                        let done = ev.finish_reason.is_some();
                        yield Ok(ev);
                        if done {
                            break;
                        }
                    }
                    Ok(StepResult::Finished(_)) => break,
                    Err(e) => {
                        yield Err(e);
                        break;
                    }
                }
            }
        }
    }

    pub fn begin_request(
        &self,
        prompt: &str,
        mut params: GenerationParams,
    ) -> Result<RequestHandle, AllpacaError> {
        if params.max_tokens == 0 {
            params.max_tokens = 1;
        }
        let mut inner = self.lock()?;
        if inner.active.is_some() {
            return Err(AllpacaError::Transient(TransientError::RequestBusy));
        }
        let tokenizer = inner
            .tokenizer
            .as_ref()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let mut tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            if let Some(bos) = tokenizer.bos_id {
                tokens.push(bos);
            } else {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(
                    "empty prompt encoding".into(),
                )));
            }
        }
        if let Some(eos) = tokenizer.eos_id {
            if !params.stop_token_ids.contains(&eos) {
                params.stop_token_ids.push(eos);
            }
        }
        let vocab = tokenizer.vocab_size();
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let page_tokens = self.config.kv.page_tokens;
        let pages = alloc_pages_for_seq(
            &mut inner.kv,
            &target.config,
            page_tokens,
            tokens.len() + params.max_tokens,
            &self.metrics,
        )?;
        let mut pending_logits = vec![0.0f32; vocab];
        for (pos, tok) in tokens.iter().enumerate() {
            // TODO: HIP prefill kernel dispatch when target_device is HIP.
            target.forward_token(
                *tok,
                pos,
                &pages,
                &mut inner.kv,
                &self.metrics,
                &mut pending_logits,
            )?;
        }
        inner.ngram.observe(&tokens);
        let published = inner.ring.publish(&pending_logits)?;
        inner.ring.release_current();
        let rng = rng_from_params(&params, self.config.engine.determinism_seed);
        let id = inner.next_request_id;
        inner.next_request_id += 1;
        inner.active = Some(ActiveRequest {
            id,
            prompt_len: tokens.len(),
            tokens,
            generated: 0,
            params,
            pages,
            pending_logits: published,
            emit_queue: Vec::new(),
            json_prefix: String::new(),
            rng,
            started: Instant::now(),
            last_token_at: Instant::now(),
            first_emitted: false,
            finished: false,
        });
        self.metrics
            .requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .active_requests
            .store(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics.prompt_tokens.fetch_add(
            inner.active.as_ref().map(|a| a.prompt_len as u64).unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(RequestHandle {
            id,
            inner: Arc::clone(&self.inner),
            metrics: Arc::clone(&self.metrics),
        })
    }

    pub fn step(&self, handle: &mut RequestHandle) -> Result<StepResult, AllpacaError> {
        let mut inner = self.lock()?;
        {
            let active = inner
                .active
                .as_mut()
                .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
            if active.id != handle.id {
                return Err(AllpacaError::Transient(TransientError::RequestBusy));
            }
            if !active.emit_queue.is_empty() {
                return Ok(StepResult::Token(active.emit_queue.remove(0)));
            }
            if active.finished {
                return Ok(StepResult::Finished(FinishReason::Stop));
            }
            if active.generated >= active.params.max_tokens {
                active.finished = true;
                return Ok(StepResult::Finished(FinishReason::Length));
            }
        }

        let vocab_tokens = inner
            .tokenizer
            .as_ref()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
            .tokens
            .clone();
        let json_constraints = inner.flags.json_constraints;
        let speculate = self.config.speculative.enabled
            && (inner.flags.ngram_speculation || inner.flags.draft_model_speculation);
        let (json_mode, json_prefix, temperature, remaining, depth) = {
            let active = inner.active.as_ref().unwrap();
            let remaining = active.params.max_tokens.saturating_sub(active.generated);
            let depth = inner.scheduler.draft_depth.min(remaining.saturating_sub(1));
            (
                active.params.json_mode
                    && json_constraints
                    && self.config.structured_output.enabled,
                active.json_prefix.clone(),
                active.params.temperature,
                remaining,
                depth,
            )
        };

        let mask = if json_mode {
            let constraint = JsonConstraint::new(vocab_tokens);
            let mut m = constraint.mask(&json_prefix);
            if !m.any() {
                tracing::warn!("JSON constraint mask empty; allowing all tokens (degraded)");
                self.metrics
                    .degraded_events
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                m = allpaca_types::TokenMask::allow_all(inner.tokenizer.as_ref().unwrap().vocab_size());
            }
            Some(m)
        } else {
            None
        };

        let drafts = if speculate && depth > 0 {
            propose_drafts(&mut inner, depth, temperature)?
        } else {
            Vec::new()
        };
        let proposed = drafts.len() as u64;
        let produced = verify_and_sample(&mut inner, drafts, mask.as_ref(), remaining, &self.metrics)?;
        let accepted = produced.accepted;
        inner.scheduler.record_acceptance(accepted, proposed);
        self.metrics
            .draft_proposed
            .fetch_add(proposed, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .draft_accepted
            .fetch_add(accepted, std::sync::atomic::Ordering::Relaxed);

        let last_idx = produced.tokens.len().saturating_sub(1);
        for (i, tok) in produced.tokens.into_iter().enumerate() {
            let piece = inner
                .tokenizer
                .as_ref()
                .map(|t| t.decode_one(tok))
                .unwrap_or_default();
            let (stop, maxed, first) = {
                let active = inner.active.as_ref().unwrap();
                (
                    is_stop(tok, &active.params),
                    active.generated + 1 >= active.params.max_tokens && i == last_idx,
                    !active.first_emitted,
                )
            };
            let finish = if stop {
                Some(FinishReason::Stop)
            } else if maxed {
                Some(FinishReason::Length)
            } else {
                None
            };
            let now = Instant::now();
            if let Some(active) = inner.active.as_mut() {
                if !is_special(&piece) {
                    active.json_prefix.push_str(&piece);
                }
                if first {
                    let ttft = now.duration_since(active.started).as_millis() as u64;
                    self.metrics
                        .ttft_ms_last
                        .store(ttft, std::sync::atomic::Ordering::Relaxed);
                    active.first_emitted = true;
                } else {
                    let itl = now.duration_since(active.last_token_at).as_millis() as u64;
                    self.metrics.record_itl_ms(itl);
                }
                active.last_token_at = now;
                active.generated += 1;
                if finish.is_some() {
                    active.finished = true;
                }
                active.emit_queue.push(TokenEvent {
                    token_id: tok,
                    text: piece,
                    finish_reason: finish,
                });
            }
            self.metrics
                .tokens_generated
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let ev = inner
            .active
            .as_mut()
            .and_then(|a| {
                if a.emit_queue.is_empty() {
                    None
                } else {
                    Some(a.emit_queue.remove(0))
                }
            })
            .ok_or_else(|| AllpacaError::Fatal(FatalError::InvalidConfig("empty step".into())))?;
        Ok(StepResult::Token(ev))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, EngineInner>, AllpacaError> {
        self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
        })
    }
}

fn apply_provider_flags(flags: &mut FeatureFlags, config: &Config) {
    if !config.speculative.enabled {
        flags.ngram_speculation = false;
        flags.draft_model_speculation = false;
        return;
    }
    let providers = &config.speculative.providers;
    if !providers.contains(&DraftProviderKind::Ngram) {
        flags.ngram_speculation = false;
    }
    if !providers.contains(&DraftProviderKind::DraftModel) {
        flags.draft_model_speculation = false;
    }
}

fn propose_drafts(
    inner: &mut EngineInner,
    depth: usize,
    temperature: f32,
) -> Result<Vec<u32>, AllpacaError> {
    let ctx = inner
        .active
        .as_ref()
        .map(|a| a.tokens.clone())
        .unwrap_or_default();
    if inner.flags.draft_model_speculation {
        if let Some(model) = inner.draft_model.clone() {
            let mut provider = DraftModelProvider::new((*model).clone());
            let draft = provider.propose(&ctx, depth, temperature)?;
            if !draft.is_empty() {
                return Ok(draft);
            }
        }
    }
    if inner.flags.ngram_speculation {
        return inner.ngram.propose(&ctx, depth, temperature);
    }
    Ok(Vec::new())
}

struct Produced {
    tokens: Vec<u32>,
    accepted: u64,
}

fn verify_and_sample(
    inner: &mut EngineInner,
    drafts: Vec<u32>,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
) -> Result<Produced, AllpacaError> {
    let mut produced = Vec::new();
    let mut accepted = 0u64;
    let mut draft_iter = drafts.into_iter().peekable();
    while produced.len() < remaining {
        let (mut logits, params) = {
            let active = inner
                .active
                .as_ref()
                .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
            (active.pending_logits.clone(), active.params.clone())
        };
        let sampled = {
            let active = inner.active.as_mut().unwrap();
            sample(&mut logits, mask, &params, &mut active.rng)
        };
        produced.push(sampled);
        let matched = draft_iter.peek().copied() == Some(sampled);
        if matched {
            accepted += 1;
            draft_iter.next();
        }
        append_and_forward(inner, sampled, metrics)?;
        if is_stop(sampled, &params) || !matched || draft_iter.peek().is_none() {
            break;
        }
    }
    Ok(Produced {
        tokens: produced,
        accepted,
    })
}

fn append_and_forward(
    inner: &mut EngineInner,
    token: u32,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let pos = inner
        .active
        .as_ref()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
        .tokens
        .len();
    let page_tokens = inner
        .active
        .as_ref()
        .and_then(|a| a.pages.first())
        .and_then(|layer| layer.first())
        .and_then(|id| inner.kv.get(*id))
        .map(|p| p.capacity_tokens)
        .unwrap_or(128);
    let cfg = inner
        .target
        .as_ref()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
        .config
        .clone();
    let mut pages = inner.active.as_ref().unwrap().pages.clone();
    ensure_pages(&mut inner.kv, &cfg, page_tokens, &mut pages, pos, metrics)?;
    if let Some(active) = inner.active.as_mut() {
        active.pages = pages;
        active.tokens.push(token);
    }
    let ctx = inner.active.as_ref().unwrap().tokens.clone();
    inner.ngram.observe(&ctx);
    let vocab = inner.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(1);
    let mut logits = vec![0.0f32; vocab];
    let pages = inner.active.as_ref().unwrap().pages.clone();
    let target = inner
        .target
        .clone()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    target.forward_token(token, pos, &pages, &mut inner.kv, metrics, &mut logits)?;
    let published = inner.ring.publish(&logits)?;
    inner.ring.release_current();
    if let Some(active) = inner.active.as_mut() {
        active.pending_logits = published;
    }
    Ok(())
}

fn is_stop(token: u32, params: &GenerationParams) -> bool {
    params.stop_token_ids.contains(&token)
}

fn is_special(piece: &str) -> bool {
    piece.starts_with('<') && piece.ends_with('>')
}

fn handle_prompt_len(
    handle: &RequestHandle,
    inner: &Arc<Mutex<EngineInner>>,
) -> Result<usize, AllpacaError> {
    let guard = inner.lock().map_err(|_| {
        AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
    })?;
    if let Some(active) = guard.active.as_ref() {
        if active.id == handle.id {
            return Ok(active.prompt_len);
        }
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_types::CliOverrides;

    fn test_engine() -> Engine {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(7),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.kv.page_tokens = 8;
        cfg.kv.vram_bytes = 8 * 1024;
        cfg.kv.dram_bytes = 64 * 1024;
        Engine::new(cfg).unwrap()
    }

    #[test]
    fn generate_produces_tokens() {
        let engine = test_engine();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        assert!(out.completion_tokens <= 8);
        assert!(out.prompt_tokens >= 1);
    }

    #[test]
    fn begin_step_matches_generate() {
        let engine = test_engine();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut handle = engine.begin_request("hello", params.clone()).unwrap();
        let mut n = 0;
        loop {
            match engine.step(&mut handle).unwrap() {
                StepResult::Token(_) => n += 1,
                StepResult::Finished(_) => break,
            }
            if n > 8 {
                break;
            }
        }
        assert!(n >= 1);
    }

    #[test]
    fn second_request_is_busy() {
        let engine = test_engine();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let _first = engine.begin_request("hello", params.clone()).unwrap();
        let err = engine.begin_request("world", params).unwrap_err();
        assert!(matches!(err, AllpacaError::Transient(TransientError::RequestBusy)));
    }

    #[test]
    fn json_mode_does_not_panic() {
        let engine = test_engine();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 6,
                    temperature: 0.0,
                    json_mode: true,
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
    }
}
