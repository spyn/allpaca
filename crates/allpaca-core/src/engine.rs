use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use allpaca_gguf::{extract_tokenizer, GgufFile};
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{
    load_hf_config, load_vocab_sidecar, resolve_sidecar, SafetensorsFile,
};
use allpaca_types::{
    AllpacaError, Config, DraftProviderKind, FatalError, FeatureFlags, FinishReason,
    GenerationOutput, GenerationParams, KvHandle, StatsSnapshot, StepResult, TokenEvent,
    TransientError,
};
use futures::Stream;
use rand::rngs::StdRng;

use crate::constraint::JsonConstraint;
use crate::device_manager::DeviceInventory;
use crate::draft::{DraftModelProvider, DraftProvider, NGramProvider};
use crate::forest::{build_token_forest, TokenForest};
use crate::kv_tier::KvTierManager;
use crate::logits_ring::LogitRing;
use crate::model::{
    alloc_pages_for_seq, ensure_pages, DenseKv, ModelConfig, ResidentModel, TinyModel,
};
use crate::sampling::{rng_from_params, sample};
use crate::scheduler::Scheduler;
use crate::telemetry::Metrics;
use crate::tokenizer::EngineTokenizer;
use allpaca_hip::HipWorkspace;

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
    active: HashMap<u64, ActiveRequest>,
    next_request_id: u64,
    model_id: String,
    target_resident: Option<Arc<ResidentModel>>,
    draft_resident: Option<Arc<ResidentModel>>,
    draft_kv: DenseKv,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<EngineInner>>,
    config: Config,
    inventory: DeviceInventory,
    metrics: Arc<Metrics>,
    hip: Option<Arc<HipWorkspace>>,
    draft_hip: Option<Arc<HipWorkspace>>,
}

pub struct RequestHandle {
    id: u64,
    inner: Arc<Mutex<EngineInner>>,
    metrics: Arc<Metrics>,
}

impl RequestHandle {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn kv_handle(&self) -> Result<KvHandle, AllpacaError> {
        let guard = self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
        })?;
        let active = guard
            .active
            .get(&self.id)
            .ok_or(AllpacaError::Transient(TransientError::RequestBusy))?;
        Ok(KvHandle {
            request_id: active.id,
            epoch: 0,
            token_count: active.tokens.len(),
            pages: active.pages.iter().flatten().copied().collect(),
        })
    }
}

impl std::fmt::Debug for RequestHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestHandle").field("id", &self.id).finish()
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(active) = guard.active.remove(&self.id) {
                for layer in active.pages {
                    for page in layer {
                        guard.kv.release(page, &self.metrics);
                    }
                }
                self.metrics.active_requests.store(
                    guard.active.len() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }
    }
}

impl Engine {
    pub fn new(config: Config) -> Result<Self, AllpacaError> {
        let mut config = config;
        if config.model.draft_path.is_none() {
            if let Ok(raw) = std::env::var("ALLPACA_DRAFT_GGUF") {
                let p = std::path::PathBuf::from(raw);
                if p.is_file() {
                    config.model.draft_path = Some(p);
                }
            }
        }
        config.validate()?;
        let inventory = DeviceInventory::discover(&config);
        let mut flags = FeatureFlags::from_mode(inventory.pipeline_mode);
        flags.overlay(&config.features);
        apply_provider_flags(&mut flags, &config);
        let scheduler = Scheduler::new(inventory.pipeline_mode, flags.clone(), &config.speculative);
        let hip = open_hip_workspace(&config, &inventory).map(Arc::new);
        let draft_hip = open_draft_hip_workspace(&config, &inventory).map(Arc::new);
        let kv = KvTierManager::with_hip(
            config.kv.vram_bytes,
            config.kv.dram_bytes,
            config.kv.hot_fraction,
            hip.as_ref().map(|h| h.device),
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
            active: HashMap::new(),
            next_request_id: 1,
            model_id: config.model.id.clone(),
            target_resident: None,
            draft_resident: None,
            draft_kv: DenseKv::default(),
        };
        let engine = Self {
            inner: Arc::new(Mutex::new(inner)),
            config: config.clone(),
            inventory,
            metrics: Arc::new(Metrics::default()),
            hip,
            draft_hip,
        };
        if config.model.synthetic {
            engine.load_synthetic(ModelConfig::tiny())?;
        } else if let Some(path) = &config.model.path {
            engine.load_model_path(path)?;
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
        let (model_id, kv_h2d, kv_d2h, draft_kv_bytes) = match self.lock() {
            Ok(g) => (
                g.target
                    .as_ref()
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| self.config.model.id.clone()),
                g.kv.activation_h2d_bytes(),
                g.kv.activation_d2h_bytes(),
                g.draft_kv.device_bytes(),
            ),
            Err(_) => (self.config.model.id.clone(), 0, 0, 0),
        };
        let (backend, device_memory_bytes) = if self.hip.is_some() {
            ("hip".into(), allpaca_hip::allocated_bytes())
        } else {
            ("cpu".into(), 0)
        };
        let hip_h2d = self
            .hip
            .as_ref()
            .map(|h| h.activation_h2d_bytes())
            .unwrap_or(0);
        let hip_d2h = self
            .hip
            .as_ref()
            .map(|h| h.activation_d2h_bytes())
            .unwrap_or(0);
        let draft_h2d = self
            .draft_hip
            .as_ref()
            .map(|h| h.activation_h2d_bytes())
            .unwrap_or(0);
        let draft_d2h = self
            .draft_hip
            .as_ref()
            .map(|h| h.activation_d2h_bytes())
            .unwrap_or(0);
        self.metrics
            .device_memory_bytes
            .store(device_memory_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .draft_kv_bytes_on_device
            .store(draft_kv_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .activation_h2d_bytes
            .store(hip_h2d + kv_h2d + draft_h2d, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .activation_d2h_bytes
            .store(hip_d2h + kv_d2h + draft_d2h, std::sync::atomic::Ordering::Relaxed);
        let bench = self.inventory.transfer_benchmark;
        let p2p_copy_ok = bench.map(|b| b.p2p_copy_ok).unwrap_or(false);
        let p2p_status = if p2p_copy_ok {
            "measured"
        } else {
            "p2p_unavailable"
        };
        self.metrics.snapshot(
            &format!("{:?}", self.inventory.pipeline_mode),
            &format!("{:?}", self.inventory.transfer_path),
            self.inventory.p2p_capable,
            self.inventory.p2p_enabled,
            p2p_status,
            bench.map(|b| b.p2p_gbps).unwrap_or(0.0),
            bench.map(|b| b.host_staging_gbps).unwrap_or(0.0),
            bench.map(|b| b.p2p_us).unwrap_or(0.0),
            bench.map(|b| b.host_staging_us).unwrap_or(0.0),
            self.inventory.target_device,
            self.inventory.draft_device,
            model_id,
            backend,
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
        drop(inner);
        self.upload_resident_weights()?;
        tracing::info!(vocab = cfg.vocab_size, layers = cfg.n_layers, "loaded synthetic model");
        Ok(())
    }

    fn load_model_path(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => self.load_safetensors(path),
            Some("onnx") => self.load_onnx(path),
            _ => self.load_gguf(path),
        }
    }

    pub fn load_safetensors(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = SafetensorsFile::open(path)?;
        let config_path = resolve_sidecar(
            self.config.model.config_path.as_deref(),
            path,
            "config.json",
        );
        let tokenizer_path = resolve_sidecar(
            self.config.model.tokenizer_path.as_deref(),
            path,
            "tokenizer.json",
        );
        let hf = load_hf_config(&config_path)?;
        let vocab = load_vocab_sidecar(&tokenizer_path)?;
        let model = TinyModel::from_safetensors(&file, &hf)?;
        let tokenizer = EngineTokenizer::from_vocab(vocab.tokens, vocab.bos_id, vocab.eos_id)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        inner.tokenizer = Some(tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()
    }

    pub fn load_onnx(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = OnnxFile::open(path)?;
        file.ensure_weight_export()?;
        let config_path = resolve_sidecar(
            self.config.model.config_path.as_deref(),
            path,
            "config.json",
        );
        let tokenizer_path = resolve_sidecar(
            self.config.model.tokenizer_path.as_deref(),
            path,
            "tokenizer.json",
        );
        let hf = load_hf_config(&config_path)?;
        let vocab = load_vocab_sidecar(&tokenizer_path)?;
        let model = TinyModel::from_onnx(&file, &hf)?;
        let tokenizer = EngineTokenizer::from_vocab(vocab.tokens, vocab.bos_id, vocab.eos_id)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        inner.tokenizer = Some(tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()
    }

    pub fn load_gguf(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = GgufFile::open(path)?;
        let tok = extract_tokenizer(&file)?;
        let model = TinyModel::from_gguf(&file)?;
        if !model.host_weights_ready() && self.hip.is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "large GGUF requires HIP (host F32 materialize is skipped to avoid OOM)".into(),
            )));
        }
        let mut draft_file = None;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        inner.tokenizer = Some(EngineTokenizer::from_gguf(tok));
        if let Some(draft_path) = &self.config.model.draft_path {
            match GgufFile::open(draft_path) {
                Ok(f) => match TinyModel::from_gguf(&f) {
                    Ok(draft) => {
                        inner.draft_model = Some(Arc::new(draft));
                        draft_file = Some(f);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "draft GGUF failed; disabling draft-model speculation");
                        inner.flags.draft_model_speculation = false;
                        self.metrics
                            .degraded_events
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                },
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
        drop(inner);
        self.upload_resident_from_gguf(Some(&file), draft_file.as_ref())
    }

    fn upload_resident_weights(&self) -> Result<(), AllpacaError> {
        self.upload_resident_from_gguf(None, None)
    }

    fn upload_resident_from_gguf(
        &self,
        target_gguf: Option<&GgufFile>,
        draft_gguf: Option<&GgufFile>,
    ) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let mut weight_bytes = 0u64;
        let mut h2d = 0u64;
        let target = inner.target.clone();
        let draft = inner.draft_model.clone();
        inner.target_resident = if let (Some(hip), Some(target)) = (&self.hip, target) {
            let uploaded = if target.host_weights_ready() {
                ResidentModel::upload(&target, hip.device)
            } else if let Some(file) = target_gguf {
                ResidentModel::upload_gguf(file, &target.config, hip.device)
            } else {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "large GGUF missing file handle for HIP upload".into(),
                )))
            };
            match uploaded {
                Ok(resident) => {
                    let n = resident.bytes();
                    weight_bytes += n;
                    h2d += n;
                    Some(Arc::new(resident))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "HIP target weight upload failed; GEMM will H2D per op");
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        } else {
            None
        };
        inner.draft_resident = if let (Some(hip), Some(draft)) = (&self.draft_hip, draft) {
            let uploaded = if draft.host_weights_ready() {
                ResidentModel::upload(&draft, hip.device)
            } else if let Some(file) = draft_gguf {
                ResidentModel::upload_gguf(file, &draft.config, hip.device)
            } else {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "large draft GGUF missing file handle for HIP upload".into(),
                )))
            };
            match uploaded {
                Ok(resident) => {
                    let n = resident.bytes();
                    weight_bytes += n;
                    h2d += n;
                    Some(Arc::new(resident))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "HIP draft weight upload failed; GEMM will H2D per op");
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        } else {
            None
        };
        self.metrics
            .weight_bytes_on_device
            .store(weight_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .weight_h2d_bytes
            .store(h2d, std::sync::atomic::Ordering::Relaxed);
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
        if inner.active.len() >= self.config.engine.max_active_requests {
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
        let resident = inner.target_resident.clone();
        for (pos, tok) in tokens.iter().enumerate() {
            // TODO: HIP prefill kernel dispatch when target_device is HIP.
            target.forward_token(
                *tok,
                pos,
                &pages,
                &mut inner.kv,
                &self.metrics,
                &mut pending_logits,
                self.hip.as_deref(),
                resident.as_deref(),
            )?;
        }
        inner.ngram.observe(&tokens);
        let published = inner.ring.publish(&pending_logits)?;
        inner.ring.release_current();
        let rng = rng_from_params(&params, self.config.engine.determinism_seed);
        let id = inner.next_request_id;
        inner.next_request_id += 1;
        let prompt_len = tokens.len();
        inner.active.insert(
            id,
            ActiveRequest {
                id,
                prompt_len,
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
            },
        );
        self.metrics
            .requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics.active_requests.store(
            inner.active.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.metrics.prompt_tokens.fetch_add(
            prompt_len as u64,
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
        let id = handle.id;
        {
            let active = request_mut(&mut inner, id)?;
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
            let active = request(&inner, id)?;
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

        let use_tree = speculate && inner.flags.tree_speculation && depth > 0;
        self.metrics
            .target_evals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let produced = if use_tree {
            let ctx = request(&inner, id)?.tokens.clone();
            let max_nodes = self.config.speculative.tree.max_nodes.max(1);
            let draft = if inner.flags.draft_model_speculation {
                inner.draft_model.as_deref()
            } else {
                None
            };
            let forest = build_token_forest(
                &inner.ngram,
                draft,
                &ctx,
                max_nodes,
                depth.max(1),
            )?;
            self.metrics
                .tree_nodes
                .fetch_add(forest.nodes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            verify_forest(&mut inner, id, &forest, mask.as_ref(), remaining, &self.metrics, self.hip.as_deref())?
        } else {
            let drafts = if speculate && depth > 0 {
                propose_drafts(
                    &mut inner,
                    id,
                    depth,
                    temperature,
                    self.draft_hip.as_deref(),
                    &self.metrics,
                )?
            } else {
                Vec::new()
            };
            verify_and_sample(&mut inner, id, drafts, mask.as_ref(), remaining, &self.metrics, self.hip.as_deref())?
        };
        let accepted = produced.accepted;
        let proposed = produced.proposed;
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
                let active = request(&inner, id)?;
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
            if let Some(active) = inner.active.get_mut(&id) {
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
            .get_mut(&id)
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

fn open_hip_workspace(config: &Config, inventory: &DeviceInventory) -> Option<HipWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_hip::runtime_linked() {
        return None;
    }
    let gpus: Vec<_> = inventory.devices.iter().filter(|d| d.is_gpu()).collect();
    if gpus.is_empty() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "hip" {
        return None;
    }
    let device = if inventory.target_device >= 0 {
        inventory.target_device
    } else {
        gpus[0].id
    };
    match HipWorkspace::new(device, 8 * 1024 * 1024, 256) {
        Ok(ws) => {
            tracing::info!(device, bytes = ws.allocated_bytes(), "HIP workspace allocated");
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "HIP workspace failed; CPU fallback");
            None
        }
    }
}

/// Resident draft workspace on the slow GPU. Absent on 1 GPU / CPU (degraded).
fn open_draft_hip_workspace(config: &Config, inventory: &DeviceInventory) -> Option<HipWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_hip::runtime_linked() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "hip" {
        return None;
    }
    let draft = inventory.draft_device;
    let target = inventory.target_device;
    if draft < 0 || draft == target {
        return None;
    }
    match HipWorkspace::new(draft, 8 * 1024 * 1024, 256) {
        Ok(ws) => {
            tracing::info!(
                device = draft,
                bytes = ws.allocated_bytes(),
                "HIP draft workspace allocated (resident; P2P not required)"
            );
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "HIP draft workspace failed; CPU draft fallback");
            None
        }
    }
}

fn apply_provider_flags(flags: &mut FeatureFlags, config: &Config) {
    if config.features.tree_speculation.is_none() {
        flags.tree_speculation = config.speculative.enabled && config.speculative.tree.enabled;
    }
    if !config.speculative.enabled {
        flags.ngram_speculation = false;
        flags.draft_model_speculation = false;
        flags.tree_speculation = false;
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

fn request(inner: &EngineInner, id: u64) -> Result<&ActiveRequest, AllpacaError> {
    inner
        .active
        .get(&id)
        .ok_or(AllpacaError::Transient(TransientError::RequestBusy))
}

fn request_mut(inner: &mut EngineInner, id: u64) -> Result<&mut ActiveRequest, AllpacaError> {
    inner
        .active
        .get_mut(&id)
        .ok_or(AllpacaError::Transient(TransientError::RequestBusy))
}

fn propose_drafts(
    inner: &mut EngineInner,
    id: u64,
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<Vec<u32>, AllpacaError> {
    let ctx = request(inner, id)?.tokens.clone();
    if inner.flags.draft_model_speculation {
        if let Some(model) = inner.draft_model.clone() {
            let mut provider = DraftModelProvider::new((*model).clone());
            let resident = inner.draft_resident.clone();
            let draft = provider.propose_on(
                &ctx,
                depth,
                temperature,
                &mut inner.draft_kv,
                draft_hip,
                resident.as_deref(),
                Some(metrics),
            )?;
            if !draft.is_empty() {
                host_stage_draft_tokens(&draft, draft_hip, metrics);
                return Ok(draft);
            }
        }
    }
    if inner.flags.ngram_speculation {
        return inner.ngram.propose(&ctx, depth, temperature);
    }
    Ok(Vec::new())
}

/// Tokens (and only tokens) cross the device boundary via host. Weights stay put.
fn host_stage_draft_tokens(tokens: &[u32], draft_hip: Option<&HipWorkspace>, metrics: &Metrics) {
    if tokens.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(tokens) as u64;
    metrics
        .draft_tokens_copied
        .fetch_add(tokens.len() as u64, std::sync::atomic::Ordering::Relaxed);
    metrics
        .host_staging_bytes
        .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    if let Some(hip) = draft_hip {
        let _ = hip.bind();
    }
}

struct Produced {
    tokens: Vec<u32>,
    accepted: u64,
    proposed: u64,
}

fn verify_and_sample(
    inner: &mut EngineInner,
    id: u64,
    drafts: Vec<u32>,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
) -> Result<Produced, AllpacaError> {
    let drafts_len = drafts.len() as u64;
    let mut produced = Vec::new();
    let mut accepted = 0u64;
    let mut draft_iter = drafts.into_iter().peekable();
    while produced.len() < remaining {
        let (mut logits, params) = {
            let active = request(inner, id)?;
            (active.pending_logits.clone(), active.params.clone())
        };
        let sampled = {
            let active = request_mut(inner, id)?;
            sample(&mut logits, mask, &params, &mut active.rng)
        };
        produced.push(sampled);
        let matched = draft_iter.peek().copied() == Some(sampled);
        if matched {
            accepted += 1;
            draft_iter.next();
        }
        append_and_forward(inner, id, sampled, metrics, hip)?;
        if is_stop(sampled, &params) || !matched || draft_iter.peek().is_none() {
            break;
        }
    }
    Ok(Produced {
        tokens: produced,
        accepted,
        proposed: drafts_len,
    })
}

fn verify_forest(
    inner: &mut EngineInner,
    id: u64,
    forest: &TokenForest,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
) -> Result<Produced, AllpacaError> {
    let vocab = inner.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(1);
    let mut node_logits = vec![vec![0.0f32; vocab]; forest.nodes.len()];
    if !forest.nodes.is_empty() {
        let pages = request(inner, id)?.pages.clone();
        let prompt_len = request(inner, id)?.tokens.len();
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        target.forward_forest(
            forest,
            &pages,
            &inner.kv,
            prompt_len,
            &mut node_logits,
            hip,
            Some(metrics),
        )?;
    }

    let mut produced = Vec::new();
    let mut accepted_tokens = Vec::new();
    let mut parent = -1i32;
    while produced.len() < remaining {
        let (mut logits, params) = {
            let active = request(inner, id)?;
            let mut logits = active.pending_logits.clone();
            if parent >= 0 {
                if let Some(nl) = node_logits.get(parent as usize) {
                    if nl.len() == logits.len() {
                        logits.copy_from_slice(nl);
                    }
                }
            }
            (logits, active.params.clone())
        };
        let sampled = {
            let active = request_mut(inner, id)?;
            sample(&mut logits, mask, &params, &mut active.rng)
        };
        produced.push(sampled);
        let child = forest.children(parent).find(|&i| forest.nodes[i].token_id as u32 == sampled);
        if let Some(idx) = child {
            accepted_tokens.push(sampled);
            parent = idx as i32;
        }
        append_and_forward(inner, id, sampled, metrics, hip)?;
        if is_stop(sampled, &params) || child.is_none() || forest.children(parent).next().is_none()
        {
            break;
        }
    }
    let path_valid = forest.is_valid_path(&accepted_tokens);
    if !path_valid {
        metrics
            .tree_path_invalid
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    metrics.tree_accepted_path_len.fetch_add(
        accepted_tokens.len() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok(Produced {
        tokens: produced,
        accepted: accepted_tokens.len() as u64,
        proposed: forest.nodes.len() as u64,
    })
}

fn append_and_forward(
    inner: &mut EngineInner,
    id: u64,
    token: u32,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
) -> Result<(), AllpacaError> {
    let pos = request(inner, id)?.tokens.len();
    let page_tokens = request(inner, id)?
        .pages
        .first()
        .and_then(|layer| layer.first())
        .and_then(|page_id| inner.kv.get(*page_id))
        .map(|p| p.capacity_tokens)
        .unwrap_or(128);
    let cfg = inner
        .target
        .as_ref()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
        .config
        .clone();
    let mut pages = request(inner, id)?.pages.clone();
    ensure_pages(&mut inner.kv, &cfg, page_tokens, &mut pages, pos, metrics)?;
    {
        let active = request_mut(inner, id)?;
        active.pages = pages;
        active.tokens.push(token);
    }
    let ctx = request(inner, id)?.tokens.clone();
    inner.ngram.observe(&ctx);
    let vocab = inner.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(1);
    let mut logits = vec![0.0f32; vocab];
    let pages = request(inner, id)?.pages.clone();
    let target = inner
        .target
        .clone()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    let resident = inner.target_resident.clone();
    target.forward_token(
        token,
        pos,
        &pages,
        &mut inner.kv,
        metrics,
        &mut logits,
        hip,
        resident.as_deref(),
    )?;
    let published = inner.ring.publish(&logits)?;
    inner.ring.release_current();
    request_mut(inner, id)?.pending_logits = published;
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
    if let Some(active) = guard.active.get(&handle.id) {
        return Ok(active.prompt_len);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_types::CliOverrides;

    fn test_config() -> Config {
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
        cfg
    }

    fn test_engine() -> Engine {
        Engine::new(test_config()).unwrap()
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
        assert_eq!(out.completion_tokens, out.token_ids.len());
        assert!(out.completion_tokens <= 8);
        assert!(out.prompt_tokens >= 1);
        assert!(matches!(
            out.finish_reason,
            FinishReason::Length | FinishReason::Stop
        ));
    }

    #[tokio::test]
    async fn generate_stream_emits_tokens() {
        use futures::StreamExt;

        let engine = test_engine();
        let mut stream = std::pin::pin!(engine.generate_stream(
            "hello".into(),
            GenerationParams {
                max_tokens: 6,
                temperature: 0.0,
                seed: Some(7),
                ..GenerationParams::default()
            },
        ));
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            let ev = item.expect("stream token");
            events.push(ev.clone());
            if ev.finish_reason.is_some() {
                break;
            }
        }
        assert!(!events.is_empty());
        assert!(events.iter().all(|e| e.token_id < 64));
        assert!(events.last().and_then(|e| e.finish_reason).is_some());
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
        while let StepResult::Token(_) = engine.step(&mut handle).unwrap() {
            n += 1;
            if n > 8 {
                break;
            }
        }
        assert!(n >= 1);
    }

    #[test]
    fn exceeding_cap_is_transient_and_drop_frees_slot() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let first = engine.begin_request("hello", params.clone()).unwrap();
        let second = engine.begin_request("world", params.clone()).unwrap();
        let err = engine.begin_request("third", params.clone()).unwrap_err();
        assert!(
            matches!(
                err,
                AllpacaError::Transient(TransientError::RequestBusy)
                    | AllpacaError::Transient(TransientError::Backpressure(_))
            ),
            "saturated scheduler must be Transient, got {err:?}"
        );
        assert_eq!(err.class(), allpaca_types::ErrorClass::Transient);
        drop(first);
        engine
            .begin_request("third", params.clone())
            .expect("dropping a Request must free a scheduler slot");
        drop(second);
        let out = engine
            .generate("hello", params)
            .expect("completed generate must free its slot");
        assert!(!out.token_ids.is_empty());
    }

    #[test]
    fn each_request_keeps_its_own_kv_pages() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.kv.page_tokens = 2;
        cfg.kv.vram_bytes = 1024;
        cfg.kv.dram_bytes = 256 * 1024;
        cfg.kv.hot_fraction = 0.35;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut first = engine.begin_request("hello", params.clone()).unwrap();
        let mut second = engine.begin_request("world", params).unwrap();
        assert!(matches!(engine.step(&mut first).unwrap(), StepResult::Token(_)));
        assert!(matches!(engine.step(&mut second).unwrap(), StepResult::Token(_)));

        let kv_a = first.kv_handle().expect("first Request KvHandle");
        let kv_b = second.kv_handle().expect("second Request KvHandle");
        assert_ne!(kv_a.request_id, kv_b.request_id);
        assert!(!kv_a.pages.is_empty());
        assert!(!kv_b.pages.is_empty());
        for page in &kv_a.pages {
            assert!(
                !kv_b.pages.contains(page),
                "Requests must not share KV Pages; a={kv_a:?} b={kv_b:?}"
            );
        }

        let pages_both = engine.stats().vram_pages + engine.stats().dram_pages;
        assert!(pages_both > 0);
        drop(first);
        let pages_after = engine.stats().vram_pages + engine.stats().dram_pages;
        assert!(
            pages_after > 0,
            "finishing one Request must not wipe the other's KV; stats={:?}",
            engine.stats()
        );
        assert!(
            pages_after < pages_both,
            "dropped Request must release its KV Pages; before={pages_both} after={pages_after}"
        );
        assert!(matches!(
            engine.step(&mut second).unwrap(),
            StepResult::Token(_) | StepResult::Finished(_)
        ));
        let stats = engine.stats();
        assert!(
            stats.dram_pages > 0 || stats.page_migrations > 0 || stats.vram_pages > 0,
            "KV spilling/stats must still work with two Requests; stats={stats:?}"
        );
    }

    #[tokio::test]
    async fn two_concurrent_streams_progress_independently() {
        use futures::StreamExt;

        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 6,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut a = std::pin::pin!(engine.generate_stream("hello".into(), params.clone()));
        let mut b = std::pin::pin!(engine.generate_stream("world".into(), params));
        let ev_a = a
            .next()
            .await
            .expect("stream a")
            .expect("first Request stream must not be RequestBusy");
        let ev_b = b
            .next()
            .await
            .expect("stream b")
            .expect("second Request stream must not be RequestBusy");
        assert!(ev_a.token_id < 64);
        assert!(ev_b.token_id < 64);
        let mut n_a = 1usize;
        let mut n_b = 1usize;
        while let Some(item) = a.next().await {
            item.expect("stream a token");
            n_a += 1;
        }
        while let Some(item) = b.next().await {
            item.expect("stream b token");
            n_b += 1;
        }
        assert!(n_a >= 1);
        assert!(n_b >= 1);
    }

    #[test]
    fn two_concurrent_generate_progress_independently() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 6,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let a = {
            let engine = engine.clone();
            let params = params.clone();
            std::thread::spawn(move || engine.generate("hello", params))
        };
        let b = {
            let engine = engine.clone();
            std::thread::spawn(move || engine.generate("world", params))
        };
        let out_a = a
            .join()
            .expect("thread a")
            .expect("first Request must not be RequestBusy");
        let out_b = b
            .join()
            .expect("thread b")
            .expect("second Request must not be RequestBusy");
        assert!(!out_a.token_ids.is_empty());
        assert!(!out_b.token_ids.is_empty());
        assert!(out_a.prompt_tokens >= 1);
        assert!(out_b.prompt_tokens >= 1);
        assert_ne!(
            out_a.token_ids, out_b.token_ids,
            "different prompts must produce independently progressing Requests"
        );
    }

    #[test]
    fn two_begin_request_succeed_when_max_active_requests_is_two() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let first = engine.begin_request("hello", params.clone()).unwrap();
        let second = engine
            .begin_request("world", params)
            .expect("second Request must be admitted when under the cap");
        assert_eq!(engine.stats().active_requests, 2);
        drop(first);
        drop(second);
    }

    #[test]
    fn second_request_is_busy() {
        let engine = test_engine();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let first = engine.begin_request("hello", params.clone()).unwrap();
        let err = engine.begin_request("world", params.clone()).unwrap_err();
        assert!(matches!(
            err,
            AllpacaError::Transient(TransientError::RequestBusy)
        ));
        assert_eq!(err.class(), allpaca_types::ErrorClass::Transient);
        drop(first);
        engine
            .begin_request("world", params)
            .expect("lock released after Request handle drop");
    }

    #[test]
    fn generate_spills_kv_pages_to_pinned_dram() {
        let mut cfg = test_config();
        cfg.kv.page_tokens = 2;
        cfg.kv.vram_bytes = 1024;
        cfg.kv.dram_bytes = 256 * 1024;
        cfg.kv.hot_fraction = 0.35;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 16,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut handle = engine.begin_request("hello", params).unwrap();
        let mut tokens = 0usize;
        while let StepResult::Token(ev) = engine.step(&mut handle).unwrap() {
            tokens += 1;
            if ev.finish_reason.is_some() {
                break;
            }
        }
        assert!(tokens >= 1);
        let stats = engine.stats();
        assert!(
            stats.dram_pages > 0 || stats.page_migrations > 0,
            "KV Pages must move VRAM → Pinned DRAM under pressure; stats={stats:?}"
        );
        drop(handle);
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
        assert!(!out.token_ids.is_empty(), "generate still works after KV spill");
    }

    #[test]
    fn disabling_tree_falls_back_to_linear_draft() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut cfg = test_config();
        cfg.speculative.tree.enabled = true;
        cfg.features.tree_speculation = Some(false);
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(true);
        let engine = Engine::new(cfg).expect("disabling Token Forest must not be Fatal");
        let out = engine.generate("hello", params).expect("linear Draft Provider fallback");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert_eq!(
            stats.tree_nodes, 0,
            "disabled tree Feature Flag must not build a Token Forest; stats={stats:?}"
        );
        assert!(
            stats.draft_proposed > 0,
            "Engine must fall back to linear NGram/DraftModel providers; stats={stats:?}"
        );
    }

    #[test]
    fn tree_accepted_prefix_is_valid_forest_path() {
        let mut cfg = test_config();
        cfg.speculative.tree.enabled = true;
        cfg.speculative.tree.max_nodes = 16;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(true);
        let engine = Engine::new(cfg).unwrap();
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
        let stats = engine.stats();
        assert!(
            stats.tree_nodes > 0,
            "Token Forest must be built when the Feature Flag is on"
        );
        assert!(
            stats.tree_path_valid,
            "accepted prefix must be a valid ancestry path; stats={stats:?}"
        );
        assert!(
            stats.tree_accepted_path_len <= stats.tree_nodes,
            "accepted path cannot be longer than the forest; stats={stats:?}"
        );
    }

    #[test]
    fn tree_speculation_proposes_multiple_branches() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut linear_cfg = test_config();
        linear_cfg.speculative.tree.enabled = false;
        linear_cfg.features.draft_model_speculation = Some(true);
        linear_cfg.features.ngram_speculation = Some(true);
        let linear = Engine::new(linear_cfg).unwrap();
        linear.generate("hello", params.clone()).unwrap();
        let linear_stats = linear.stats();

        let mut tree_cfg = test_config();
        tree_cfg.speculative.tree.enabled = true;
        tree_cfg.speculative.tree.max_nodes = 16;
        tree_cfg.features.draft_model_speculation = Some(true);
        tree_cfg.features.ngram_speculation = Some(true);
        let tree = Engine::new(tree_cfg).unwrap();
        tree.generate("hello", params).unwrap();
        let tree_stats = tree.stats();

        assert!(
            tree_stats.tree_nodes > 1,
            "Token Forest must propose multiple candidate nodes; stats={tree_stats:?}"
        );
        assert!(
            tree_stats.tree_nodes > linear_stats.draft_proposed
                || tree_stats.draft_proposed > linear_stats.draft_proposed
                || tree_stats.accepted_tokens_per_target_eval
                    > linear_stats.accepted_tokens_per_target_eval,
            "tree must expose more candidates than linear-only; tree={tree_stats:?} linear={linear_stats:?}"
        );
    }

    #[test]
    fn tree_speculation_generate_matches_greedy_without_tree() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut with_tree = test_config();
        with_tree.speculative.tree.enabled = true;
        with_tree.speculative.tree.max_nodes = 16;
        with_tree.features.draft_model_speculation = Some(true);
        with_tree.features.ngram_speculation = Some(true);
        let tree_engine = Engine::new(with_tree).unwrap();
        let tree_out = tree_engine.generate("hello", params.clone()).unwrap();
        assert!(
            !tree_out.token_ids.is_empty(),
            "tree-enabled generate must still produce tokens"
        );

        let mut no_tree = test_config();
        no_tree.speculative.tree.enabled = false;
        no_tree.features.draft_model_speculation = Some(true);
        no_tree.features.ngram_speculation = Some(true);
        let baseline = Engine::new(no_tree).unwrap();
        let base_out = baseline.generate("hello", params).unwrap();
        assert_eq!(
            tree_out.token_ids, base_out.token_ids,
            "Target Model must decide tokens when Token Forest is enabled"
        );
    }

    #[test]
    fn target_decides_tokens_with_speculation() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut with_spec = test_config();
        with_spec.features.draft_model_speculation = Some(true);
        with_spec.features.ngram_speculation = Some(true);
        let speculated = Engine::new(with_spec).unwrap();
        let with_out = speculated.generate("hello", params.clone()).unwrap();
        assert!(
            speculated.stats().draft_proposed > 0,
            "Draft Provider must propose under speculation"
        );

        let mut no_spec = test_config();
        no_spec.merge_cli(&CliOverrides {
            speculative: Some(false),
            ..CliOverrides::default()
        });
        let baseline = Engine::new(no_spec).unwrap();
        let without_out = baseline.generate("hello", params).unwrap();
        assert_eq!(
            with_out.token_ids, without_out.token_ids,
            "Target Model must decide tokens; Draft Provider only proposes"
        );
    }

    fn is_special_token(piece: &str) -> bool {
        piece.starts_with('<') && piece.ends_with('>')
    }

    /// Public-API oracle: a prefix is JSON-legal if it is valid JSON or a
    /// truncated value (object/array/string/number/literal) with no junk.
    fn json_prefix_legal(s: &str) -> bool {
        if s.is_empty() {
            return true;
        }
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        let mut chars = s.chars().peekable();
        let mut started = false;
        while let Some(c) = chars.next() {
            if in_string {
                if escape {
                    escape = false;
                    continue;
                }
                match c {
                    '\\' => escape = true,
                    '"' => in_string = false,
                    ch if ch.is_control() => return false,
                    _ => {}
                }
                continue;
            }
            if c.is_whitespace() {
                continue;
            }
            started = true;
            match c {
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                '"' => in_string = true,
                ':' | ',' => {
                    if depth == 0 {
                        return false;
                    }
                }
                't' => {
                    let rest: String = std::iter::once('t').chain(chars.clone()).take(4).collect();
                    if !"true".starts_with(&rest) && rest != "true"[..rest.len().min(4)] {
                        return false;
                    }
                    for _ in 0..rest.len().saturating_sub(1).min(3) {
                        chars.next();
                    }
                }
                'f' => {
                    let rest: String = std::iter::once('f').chain(chars.clone()).take(5).collect();
                    if !"false".starts_with(&rest) {
                        return false;
                    }
                    for _ in 0..rest.len().saturating_sub(1).min(4) {
                        chars.next();
                    }
                }
                'n' => {
                    let rest: String = std::iter::once('n').chain(chars.clone()).take(4).collect();
                    if !"null".starts_with(&rest) {
                        return false;
                    }
                    for _ in 0..rest.len().saturating_sub(1).min(3) {
                        chars.next();
                    }
                }
                '-' | '0'..='9' => {
                    while matches!(chars.peek(), Some('0'..='9' | '.' | 'e' | 'E' | '+' | '-')) {
                        chars.next();
                    }
                }
                _ => return false,
            }
        }
        started || s.chars().all(char::is_whitespace)
    }

    fn write_tiny_target_gguf(path: &std::path::Path) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
        let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight".into(), vec![vocab as u64, dim as u64], vec![0.01; vocab * dim]),
            ("blk.0.attn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
            ("blk.0.attn_q.weight".into(), vec![dim as u64, dim as u64], vec![0.01; dim * dim]),
            ("blk.0.attn_k.weight".into(), vec![2, dim as u64], vec![0.01; 2 * dim]),
            ("blk.0.attn_v.weight".into(), vec![2, dim as u64], vec![0.01; 2 * dim]),
            ("blk.0.attn_output.weight".into(), vec![dim as u64, dim as u64], vec![0.01; dim * dim]),
            ("blk.0.ffn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
            ("blk.0.ffn_gate.weight".into(), vec![ff as u64, dim as u64], vec![0.01; ff * dim]),
            ("blk.0.ffn_up.weight".into(), vec![ff as u64, dim as u64], vec![0.01; ff * dim]),
            ("blk.0.ffn_down.weight".into(), vec![dim as u64, ff as u64], vec![0.01; dim * ff]),
            ("output_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
        ];
        let views: Vec<GgufWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data,
            })
            .collect();
        write_gguf(path, &meta, &views).unwrap();
    }

    fn write_tiny_quant_gguf(path: &std::path::Path) {
        use allpaca_gguf::{
            quantize_q4_0, quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
            GgufWriteTensor,
        };
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

        let embd = vec![0.01f32; vocab * dim];
        let q = vec![0.01f32; dim * dim];
        let k = vec![0.01f32; 2 * dim];
        let v = vec![0.01f32; 2 * dim];
        let o = vec![0.01f32; dim * dim];
        let gate = vec![0.01f32; ff * dim];
        let up = vec![0.01f32; ff * dim];
        let down = vec![0.01f32; dim * ff];
        let embd_q = quantize_q8_0(&embd);
        let q_q = quantize_q4_0(&q);
        let k_q = quantize_q4_0(&k);
        let v_q = quantize_q4_0(&v);
        let o_q = quantize_q4_0(&o);
        let gate_q = quantize_q4_0(&gate);
        let up_q = quantize_q4_0(&up);
        let down_q = quantize_q4_0(&down);
        let attn_norm = vec![1.0f32; dim];
        let ffn_norm = vec![1.0f32; dim];
        let output_norm = vec![1.0f32; dim];

        let f32_tensors = [
            GgufWriteTensor {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &attn_norm,
            },
            GgufWriteTensor {
                name: "blk.0.ffn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &ffn_norm,
            },
            GgufWriteTensor {
                name: "output_norm.weight".into(),
                dims: vec![dim as u64],
                data: &output_norm,
            },
        ];
        let quant_tensors = [
            GgufWriteQuantTensor {
                name: "token_embd.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type: GgmlType::Q8_0,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &down_q,
            },
        ];
        write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).unwrap();
    }

    #[test]
    fn engine_fatal_without_model_or_valid_config() {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        let engine = Engine::new(cfg).unwrap();
        let err = engine
            .generate("hello", GenerationParams::default())
            .unwrap_err();
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);

        let mut bad = Config::compiled_defaults();
        bad.kv.page_tokens = 0;
        let err = match Engine::new(bad) {
            Ok(_) => panic!("invalid config must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(matches!(err, AllpacaError::Fatal(FatalError::InvalidConfig(_))));

        let junk = std::env::temp_dir().join("allpaca_corrupt.gguf");
        std::fs::write(&junk, b"not a gguf").unwrap();
        let mut corrupt = Config::compiled_defaults();
        corrupt.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        corrupt.model.path = Some(junk.clone());
        let err = match Engine::new(corrupt) {
            Ok(_) => panic!("corrupt GGUF must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let _ = std::fs::remove_file(junk);
    }

    #[test]
    fn engine_fatal_on_missing_gguf() {
        let mut missing = Config::compiled_defaults();
        missing.merge_cli(&CliOverrides {
            cpu_only: true,
            model: Some(std::env::temp_dir().join("allpaca_missing_target.gguf")),
            ..CliOverrides::default()
        });
        assert_eq!(
            missing.model.path.as_deref(),
            Some(std::path::Path::new(
                &std::env::temp_dir().join("allpaca_missing_target.gguf")
            ))
        );
        let err = match Engine::new(missing) {
            Ok(_) => panic!("missing GGUF must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
    }

    #[test]
    fn engine_fatal_on_missing_or_invalid_safetensors() {
        let mut missing = Config::compiled_defaults();
        missing.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        missing.model.path = Some(std::env::temp_dir().join("allpaca_missing.safetensors"));
        let err = match Engine::new(missing) {
            Ok(_) => panic!("missing SafeTensors must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "a .safetensors path must not be treated as GGUF: {err}"
        );

        let junk = std::env::temp_dir().join("allpaca_corrupt.safetensors");
        std::fs::write(&junk, b"not a safetensors file").unwrap();
        let mut corrupt = Config::compiled_defaults();
        corrupt.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        corrupt.model.path = Some(junk.clone());
        let err = match Engine::new(corrupt) {
            Ok(_) => panic!("invalid SafeTensors must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "invalid SafeTensors must not surface as corrupt GGUF: {err}"
        );
        let _ = std::fs::remove_file(junk);
    }

    #[test]
    fn engine_fatal_on_missing_or_invalid_onnx() {
        let mut missing = Config::compiled_defaults();
        missing.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        missing.model.path = Some(std::env::temp_dir().join("allpaca_missing.onnx"));
        let err = match Engine::new(missing) {
            Ok(_) => panic!("missing ONNX must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "a .onnx path must not be treated as GGUF: {err}"
        );

        let junk = std::env::temp_dir().join("allpaca_corrupt.onnx");
        std::fs::write(&junk, b"not an onnx file").unwrap();
        let mut corrupt = Config::compiled_defaults();
        corrupt.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        corrupt.model.path = Some(junk.clone());
        let err = match Engine::new(corrupt) {
            Ok(_) => panic!("invalid ONNX must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "invalid ONNX must not surface as corrupt GGUF: {err}"
        );
        let _ = std::fs::remove_file(junk);
    }

    fn write_tiny_safetensors_checkpoint(dir: &std::path::Path) -> std::path::PathBuf {
        use allpaca_safetensors::{write_safetensors, SafetensorsDtype, SafetensorsWriteTensor};

        std::fs::create_dir_all(dir).unwrap();
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let kv_dim = 2usize;
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures":["LlamaForCausalLM"],"vocab_size":{vocab},"hidden_size":{dim},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":1,"intermediate_size":{ff},"rms_norm_eps":1e-5,"rope_theta":10000.0}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();

        let embd = vec![0.01f32; vocab * dim];
        let ones = vec![1.0f32; dim];
        let q = vec![0.01f32; dim * dim];
        let k = vec![0.01f32; kv_dim * dim];
        let v = vec![0.01f32; kv_dim * dim];
        let o = vec![0.01f32; dim * dim];
        let gate = vec![0.01f32; ff * dim];
        let up = vec![0.01f32; ff * dim];
        let down = vec![0.01f32; dim * ff];
        let path = dir.join("model.safetensors");
        let owned = [
            ("token_embd.weight", vec![vocab as u64, dim as u64], &embd, SafetensorsDtype::F32),
            ("blk.0.attn_norm.weight", vec![dim as u64], &ones, SafetensorsDtype::F32),
            ("blk.0.attn_q.weight", vec![dim as u64, dim as u64], &q, SafetensorsDtype::F32),
            ("blk.0.attn_k.weight", vec![kv_dim as u64, dim as u64], &k, SafetensorsDtype::F32),
            ("blk.0.attn_v.weight", vec![kv_dim as u64, dim as u64], &v, SafetensorsDtype::F32),
            ("blk.0.attn_output.weight", vec![dim as u64, dim as u64], &o, SafetensorsDtype::F32),
            ("blk.0.ffn_norm.weight", vec![dim as u64], &ones, SafetensorsDtype::F32),
            ("blk.0.ffn_gate.weight", vec![ff as u64, dim as u64], &gate, SafetensorsDtype::F32),
            ("blk.0.ffn_up.weight", vec![ff as u64, dim as u64], &up, SafetensorsDtype::F32),
            ("blk.0.ffn_down.weight", vec![dim as u64, ff as u64], &down, SafetensorsDtype::F32),
            ("output_norm.weight", vec![dim as u64], &ones, SafetensorsDtype::F32),
        ];
        let views: Vec<SafetensorsWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data, dtype)| SafetensorsWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                dtype: *dtype,
                data,
            })
            .collect();
        write_safetensors(&path, &views).unwrap();
        path
    }

    #[test]
    fn engine_generate_from_tiny_safetensors() {
        let dir = std::env::temp_dir().join("allpaca_tiny_safetensors_ckpt");
        let path = write_tiny_safetensors_checkpoint(&dir);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("SafeTensors checkpoint must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on SafeTensors weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn write_tiny_onnx_checkpoint(dir: &std::path::Path) -> std::path::PathBuf {
        use allpaca_onnx::{write_onnx_weights, OnnxWriteTensor};

        std::fs::create_dir_all(dir).unwrap();
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let kv_dim = 2usize;
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures":["LlamaForCausalLM"],"vocab_size":{vocab},"hidden_size":{dim},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":1,"intermediate_size":{ff},"rms_norm_eps":1e-5,"rope_theta":10000.0}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();
        let embd = vec![0.01f32; vocab * dim];
        let ones = vec![1.0f32; dim];
        let q = vec![0.01f32; dim * dim];
        let k = vec![0.01f32; kv_dim * dim];
        let v = vec![0.01f32; kv_dim * dim];
        let o = vec![0.01f32; dim * dim];
        let gate = vec![0.01f32; ff * dim];
        let up = vec![0.01f32; ff * dim];
        let down = vec![0.01f32; dim * ff];
        let path = dir.join("model.onnx");
        let owned = [
            ("token_embd.weight", vec![vocab as i64, dim as i64], &embd),
            ("blk.0.attn_norm.weight", vec![dim as i64], &ones),
            ("blk.0.attn_q.weight", vec![dim as i64, dim as i64], &q),
            ("blk.0.attn_k.weight", vec![kv_dim as i64, dim as i64], &k),
            ("blk.0.attn_v.weight", vec![kv_dim as i64, dim as i64], &v),
            ("blk.0.attn_output.weight", vec![dim as i64, dim as i64], &o),
            ("blk.0.ffn_norm.weight", vec![dim as i64], &ones),
            ("blk.0.ffn_gate.weight", vec![ff as i64, dim as i64], &gate),
            ("blk.0.ffn_up.weight", vec![ff as i64, dim as i64], &up),
            ("blk.0.ffn_down.weight", vec![dim as i64, ff as i64], &down),
            ("output_norm.weight", vec![dim as i64], &ones),
        ];
        let views: Vec<OnnxWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| OnnxWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                data,
            })
            .collect();
        write_onnx_weights(&path, "tiny-llama", &views, &[]).unwrap();
        path
    }

    #[test]
    fn engine_generate_from_tiny_onnx() {
        let dir = std::env::temp_dir().join("allpaca_tiny_onnx_ckpt");
        let path = write_tiny_onnx_checkpoint(&dir);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("ONNX weight-export must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on ONNX initializer weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_fatal_on_unsupported_onnx_graph() {
        use allpaca_onnx::{write_onnx_weights, OnnxWriteNode, OnnxWriteTensor};

        let dir = std::env::temp_dir().join("allpaca_onnx_bad_op");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":4,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":4,"intermediate_size":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();
        let w = [1.0f32, 2.0];
        let path = dir.join("model.onnx");
        write_onnx_weights(
            &path,
            "conv-net",
            &[OnnxWriteTensor {
                name: "w".into(),
                dims: vec![2],
                data: &w,
            }],
            &[OnnxWriteNode {
                op_type: "Conv".into(),
                name: "conv0".into(),
            }],
        )
        .unwrap();
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("unsupported ONNX graph must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let msg = err.to_string();
        assert!(
            msg.contains("Conv") && !msg.to_lowercase().contains("panic"),
            "Fatal error must name the unsupported op: {msg}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_fatal_on_unsupported_safetensors_dtype() {
        let dir = std::env::temp_dir().join("allpaca_st_bad_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":4,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":4,"intermediate_size":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();
        let header = r#"{"token_embd.weight":{"dtype":"I32","shape":[4,8],"data_offsets":[0,128]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 128]);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("unsupported SafeTensors dtype must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let msg = err.to_string();
        assert!(
            msg.contains("I32") && msg.to_lowercase().contains("unsupported"),
            "Fatal error must name the unsupported dtype: {msg}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_generate_from_q4_q8_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_q4q8.gguf");
        write_tiny_quant_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("Q4_0/Q8_0 GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on dequantized Q4/Q8 weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_speculative_generate_from_tiny_written_draft_gguf() {
        let target = std::env::temp_dir().join("allpaca_target_for_draft.gguf");
        let draft = std::env::temp_dir().join("allpaca_tiny_draft.gguf");
        write_tiny_target_gguf(&target);
        write_tiny_target_gguf(&draft);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(target.clone());
        cfg.model.draft_path = Some(draft.clone());
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("tiny target+draft GGUF must load");
        assert_eq!(engine.stats().degraded_events, 0);
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("speculative generate with a written draft GGUF must run");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(draft);
    }

    #[test]
    fn engine_degraded_when_draft_gguf_missing() {
        let target = std::env::temp_dir().join("allpaca_target_tiny.gguf");
        write_tiny_target_gguf(&target);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(target.clone());
        cfg.model.draft_path = Some(std::env::temp_dir().join("allpaca_missing_draft.gguf"));
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("target GGUF loads; missing draft is Degraded");
        assert!(engine.stats().degraded_events > 0);
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("Degraded draft failure must continue on the Target Model");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(target);
    }

    #[test]
    fn engine_generate_from_gguf_still_works_alongside_safetensors() {
        let path = std::env::temp_dir().join("allpaca_gguf_regression.gguf");
        write_tiny_target_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("GGUF path must still load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("GGUF generate must still work");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn json_mode_emits_only_legal_tokens() {
        let engine = test_engine();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 12,
                    temperature: 0.0,
                    json_mode: true,
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let tok = EngineTokenizer::synthetic(64).unwrap();
        let mut prefix = String::new();
        for id in &out.token_ids {
            let piece = tok.decode_one(*id);
            if is_special_token(&piece) {
                continue;
            }
            prefix.push_str(&piece);
            assert!(
                json_prefix_legal(&prefix),
                "Constraint Mask allowed illegal JSON after {prefix:?} (+{piece:?})"
            );
        }
        assert!(
            prefix.starts_with('{')
                || prefix.starts_with('[')
                || prefix.starts_with('"')
                || prefix.starts_with('-')
                || prefix.starts_with(|c: char| c.is_ascii_digit())
                || prefix.starts_with("true")
                || prefix.starts_with("false")
                || prefix.starts_with("null")
                || prefix.is_empty(),
            "JSON mode must start with a JSON value, got {prefix:?}"
        );
    }
}
