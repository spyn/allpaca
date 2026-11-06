use std::collections::HashMap;

use allpaca_hip::HipWorkspace;
use allpaca_types::{AllpacaError, GenerationParams};

use crate::model::{DenseKv, ResidentModel, TinyModel};
use crate::sampling::{argmax, rng_from_params, sample, sample_from_distribution, softmax_candidates};
use crate::telemetry::Metrics;

pub trait DraftProvider {
    fn name(&self) -> &'static str;
    fn observe(&mut self, tokens: &[u32]);
    fn propose(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>, AllpacaError>;
}

/// CPU trie / n-gram counts over recent context.
pub struct NGramProvider {
    n: usize,
    counts: HashMap<Vec<u32>, HashMap<u32, u32>>,
}

impl NGramProvider {
    pub fn new(n: usize) -> Self {
        Self {
            n: n.max(2),
            counts: HashMap::new(),
        }
    }

    fn best_next(&self, ctx: &[u32]) -> Option<u32> {
        self.top_next(ctx, 1).into_iter().next().map(|(tok, _)| tok)
    }

    pub(crate) fn top_next(&self, ctx: &[u32], k: usize) -> Vec<(u32, f32)> {
        if k == 0 || ctx.is_empty() {
            return Vec::new();
        }
        let max_ctx = self.n - 1;
        let start = ctx.len().saturating_sub(max_ctx);
        for prefix_len in (1..=ctx.len() - start).rev() {
            let key = ctx[ctx.len() - prefix_len..].to_vec();
            if let Some(nexts) = self.counts.get(&key) {
                let total: f32 = nexts.values().map(|c| *c as f32).sum();
                let mut ranked: Vec<(u32, f32)> = nexts
                    .iter()
                    .map(|(tok, c)| {
                        let p = if total > 0.0 { *c as f32 / total } else { 0.0 };
                        (*tok, p.ln())
                    })
                    .collect();
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                ranked.truncate(k);
                return ranked;
            }
        }
        Vec::new()
    }
}

impl DraftProvider for NGramProvider {
    fn name(&self) -> &'static str {
        "ngram"
    }

    fn observe(&mut self, tokens: &[u32]) {
        if tokens.len() < 2 {
            return;
        }
        for end in 1..tokens.len() {
            let start = end.saturating_sub(self.n - 1);
            let ctx = tokens[start..end].to_vec();
            let next = tokens[end];
            *self.counts.entry(ctx).or_default().entry(next).or_default() += 1;
        }
    }

    fn propose(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Vec<u32>, AllpacaError> {
        let mut out = Vec::new();
        let mut window = ctx.to_vec();
        for _ in 0..max_tokens {
            let Some(tok) = self.best_next(&window) else {
                break;
            };
            out.push(tok);
            window.push(tok);
        }
        Ok(out)
    }
}

/// Secondary model draft path. HIP GEMM/attention run on the resident draft GPU.
///
/// MTP/NextN self-speculation loads trailing `blk[n_layers..]` via
/// `TinyModel::from_gguf_mtp` when `DraftProviderKind::Mtp` is set. Target
/// `last_hidden` seeds `h_nextn` through process(); packed-verify ahead
/// proposes on that warmed process KV (not an empty DenseKv clone).
pub struct DraftModelProvider {
    model: TinyModel,
    /// P14: when set (and T > 0) drafts are sampled from the model's own
    /// softmaxed distribution with this RNG seed, and that distribution is kept
    /// in `last_q`. Leviathan is only valid if the draft was drawn from `q`.
    q_seed: Option<u64>,
    last_q: Vec<Vec<(u32, f32)>>,
}

/// Candidates kept in a sampled draft distribution. The draft is drawn from
/// the truncated, renormalised list, so it is still exactly `q`.
const DRAFT_Q_TOP_K: usize = 64;

impl DraftModelProvider {
    pub fn new(model: TinyModel) -> Self {
        Self {
            model,
            q_seed: None,
            last_q: Vec::new(),
        }
    }

    /// Sample drafts from the draft model's own distribution (seeded by
    /// `seed`) instead of the fixed-seed `top_k: 1` path. Opt-in (P14).
    pub fn with_sampled_q(mut self, seed: u64) -> Self {
        self.q_seed = Some(seed);
        self
    }

    /// Per-token draft distributions of the last propose, aligned with its
    /// tokens (a forced token gets a one-hot). Empty unless `with_sampled_q`
    /// was set and the temperature was above greedy.
    pub fn take_q(&mut self) -> Vec<Vec<(u32, f32)>> {
        std::mem::take(&mut self.last_q)
    }

    pub fn propose_on(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        temperature: f32,
        cache: &mut DenseKv,
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<u32>, AllpacaError> {
        self.propose_inner(ctx, None, max_tokens, temperature, cache, hip, resident, metrics)
    }

    /// Draft after `ctx` with the next token already known (`forced`, e.g. a
    /// rejected verify's correction that the target has not run yet). Returns
    /// `[forced, d1, ..]`, `max_tokens` long; only `d1..` are drafted.
    #[allow(clippy::too_many_arguments)]
    pub fn propose_after(
        &mut self,
        ctx: &[u32],
        forced: u32,
        max_tokens: usize,
        temperature: f32,
        cache: &mut DenseKv,
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<u32>, AllpacaError> {
        self.propose_inner(ctx, Some(forced), max_tokens, temperature, cache, hip, resident, metrics)
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_inner(
        &mut self,
        ctx: &[u32],
        forced: Option<u32>,
        max_tokens: usize,
        temperature: f32,
        cache: &mut DenseKv,
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<u32>, AllpacaError> {
        if ctx.is_empty() {
            return Ok(Vec::new());
        }
        let kv_dim = self.model.config.n_kv_heads * self.model.config.head_dim();
        let is_mtp = self.model.mtp_nextn.is_some()
            || self.model.config.gguf_layer_base > 0
            || resident.is_some_and(|r| r.has_mtp_nextn());
        let vocab = self.model.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if is_mtp {
            // llama.cpp draft-mtp: when process() has warmed draft KV to ctx,
            // first draft step is (id_last, pending_h) at RoPE pos n_past=len.
            // If KV is cold/out-of-sync, fall back to a single last-token seed.
            let n_past = ctx.len();
            let id_last = ctx[n_past - 1];
            if cache.seq_matches(ctx) && forced.is_some() {
                // The seed step's logits would only predict `forced`.
                self.model
                    .process_dense_on(id_last, n_past, cache, hip, resident, metrics)?;
                cache.record(id_last);
            } else if cache.seq_matches(ctx) {
                self.model.forward_dense_on(
                    id_last,
                    n_past,
                    cache,
                    &mut logits,
                    hip,
                    resident,
                    metrics,
                )?;
                cache.record(id_last);
            } else {
                cache.truncate(0, kv_dim);
                self.model.forward_dense_on(
                    id_last,
                    n_past.saturating_sub(1),
                    cache,
                    &mut logits,
                    hip,
                    resident,
                    metrics,
                )?;
                cache.record(id_last);
            }
        } else {
            let common = cache.prefix_len(ctx);
            let start = if common == ctx.len() {
                cache.truncate(common.saturating_sub(1), kv_dim);
                common.saturating_sub(1)
            } else {
                cache.truncate(common, kv_dim);
                common
            };
            for (pos, tok) in ctx.iter().enumerate().skip(start) {
                self.model.forward_dense_on(
                    *tok,
                    pos,
                    cache,
                    &mut logits,
                    hip,
                    resident,
                    metrics,
                )?;
                cache.record(*tok);
            }
        }
        let mut out = Vec::new();
        let mut window = ctx.to_vec();
        let params = GenerationParams {
            temperature,
            top_k: 1,
            top_p: 1.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let mut rng = rng_from_params(&params, Some(1));
        let sampled_q = self.q_seed.filter(|_| temperature > 1e-5);
        let mut q_rng = rng_from_params(
            &GenerationParams {
                seed: sampled_q,
                ..GenerationParams::default()
            },
            Some(1),
        );
        let q_params = GenerationParams {
            temperature,
            top_k: DRAFT_Q_TOP_K,
            top_p: 1.0,
            ..GenerationParams::default()
        };
        self.last_q.clear();
        for i in 0..max_tokens {
            let next = if let (0, Some(tok)) = (i, forced) {
                if sampled_q.is_some() {
                    self.last_q.push(vec![(tok, 1.0)]);
                }
                tok
            } else if sampled_q.is_some() {
                let q = softmax_candidates(&mut logits, None, &q_params);
                let tok = sample_from_distribution(&q, &mut q_rng);
                self.last_q.push(q);
                tok
            } else if temperature <= 1e-5 {
                argmax(&logits)
            } else {
                sample(&mut logits, None, &params, &mut rng)
            };
            out.push(next);
            window.push(next);
            if i + 1 == max_tokens {
                // Nothing samples after the last draft: extend the KV only.
                self.model
                    .process_dense_on(next, window.len() - 1, cache, hip, resident, metrics)?;
            } else {
                self.model.forward_dense_on(
                    next,
                    window.len() - 1,
                    cache,
                    &mut logits,
                    hip,
                    resident,
                    metrics,
                )?;
            }
            cache.record(next);
        }
        if let Some(metrics) = metrics {
            metrics
                .draft_tokens_drafted
                .fetch_add((out.len() - usize::from(forced.is_some())) as u64, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(out)
    }
}

/// The qwen4exp (Flash-Next) NextN head as a drafter, with llama.cpp
/// `draft-mtp` pairing: [`process`](Self::process) runs token `k` with the
/// target's hidden streams for `k - 1` at position `k`, so its logits predict
/// token `k + 1`. Once the target has forwarded the whole context, the last
/// `process` step's logits give the first draft; each later draft feeds the
/// one before at the next position with the head's own streams.
pub struct Qwen4MtpProvider {
    model: TinyModel,
    memory: crate::qwen4exp::Qwen4Memory,
    /// Logits of the last `process` step: the prediction for the token after
    /// the processed context.
    last_logits: Vec<f32>,
}

impl Qwen4MtpProvider {
    pub fn new(model: TinyModel) -> Self {
        Self {
            model,
            memory: crate::qwen4exp::Qwen4Memory::default(),
            last_logits: Vec::new(),
        }
    }

    pub fn process(&mut self, token: u32, pos: usize, hidden: &[f32], metrics: &Metrics) -> Result<(), AllpacaError> {
        if pos == 0 && self.memory.tokens() != 0 {
            self.memory = crate::qwen4exp::Qwen4Memory::default();
        }
        if self.memory.tokens() != pos {
            return Ok(());
        }
        let dim = self.model.config.dim;
        let hidden_width = self.model.q4x.as_ref().map_or(dim, |q4| q4.spec.hc * dim);
        self.memory.nextn_seed = Some(if hidden.len() == hidden_width {
            hidden.to_vec()
        } else if pos == 0 {
            // Nothing precedes the first token: llama.cpp pairs it with zeros.
            vec![0.0; hidden_width]
        } else {
            // A hidden of the wrong width would mislead every later draft.
            return Ok(());
        });
        self.last_logits.resize(self.model.config.vocab_size, 0.0);
        crate::qwen4exp::forward_qwen4_token(&self.model, token, pos, &mut self.last_logits, &mut self.memory)?;
        metrics.draft_gpu_forwards.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn propose_on(&mut self, ctx: &[u32], max_tokens: usize, temperature: f32, metrics: &Metrics) -> Result<Vec<u32>, AllpacaError> {
        self.propose_inner(ctx, None, max_tokens, temperature, metrics)
    }

    /// Drafts after a correction `forced` that the target has not run yet:
    /// `[forced, d1, ..]`. `target_hidden` is the target's hidden streams for
    /// the last token of `ctx`.
    pub fn propose_after(
        &mut self,
        ctx: &[u32],
        forced: u32,
        target_hidden: &[f32],
        max_tokens: usize,
        temperature: f32,
        metrics: &Metrics,
    ) -> Result<Vec<u32>, AllpacaError> {
        self.propose_inner(ctx, Some((forced, target_hidden)), max_tokens, temperature, metrics)
    }

    fn propose_inner(
        &mut self,
        ctx: &[u32],
        forced: Option<(u32, &[f32])>,
        max_tokens: usize,
        temperature: f32,
        metrics: &Metrics,
    ) -> Result<Vec<u32>, AllpacaError> {
        let vocab = self.model.config.vocab_size;
        if ctx.is_empty() || max_tokens == 0 || self.memory.tokens() != ctx.len() || self.last_logits.len() != vocab {
            return Ok(Vec::new());
        }
        let params = GenerationParams {
            temperature,
            top_k: 1,
            top_p: 1.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let mut rng = rng_from_params(&params, Some(1));
        let mut memory = self.memory.clone();
        let mut out = Vec::with_capacity(max_tokens);
        let mut logits = vec![0.0f32; vocab];
        // The position of the next token the head runs.
        let mut pos = ctx.len();
        let step = |token: u32, pos: usize, logits: &mut [f32], memory: &mut crate::qwen4exp::Qwen4Memory| {
            metrics.draft_gpu_forwards.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::qwen4exp::forward_qwen4_token(&self.model, token, pos, logits, memory)
        };
        match forced {
            Some((tok, target_hidden)) => {
                // The head's streams from `process` stand in only when the
                // target's are missing or of another width.
                if memory.nextn_seed.as_ref().is_some_and(|s| s.len() == target_hidden.len()) {
                    memory.nextn_seed = Some(target_hidden.to_vec());
                }
                out.push(tok);
                step(tok, pos, &mut logits, &mut memory)?;
                pos += 1;
            }
            None => logits.copy_from_slice(&self.last_logits),
        }
        while out.len() < max_tokens {
            let next = if temperature <= 1e-5 {
                argmax(&logits)
            } else {
                sample(&mut logits, None, &params, &mut rng)
            };
            out.push(next);
            if out.len() < max_tokens {
                step(next, pos, &mut logits, &mut memory)?;
                pos += 1;
            }
        }
        metrics.draft_tokens_drafted.fetch_add(
            (out.len() - usize::from(forced.is_some())) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(out)
    }
}

impl DraftProvider for DraftModelProvider {
    fn name(&self) -> &'static str {
        "draft_model"
    }

    fn observe(&mut self, _tokens: &[u32]) {}

    fn propose(
        &mut self,
        ctx: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Vec<u32>, AllpacaError> {
        let mut cache = DenseKv::default();
        self.propose_on(ctx, max_tokens, temperature, &mut cache, None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelConfig, TinyModel};

    #[test]
    fn ngram_repeats_seen_continuation() {
        let mut p = NGramProvider::new(3);
        p.observe(&[1, 2, 3, 1, 2, 3]);
        let draft = p.propose(&[1, 2], 2, 0.0).unwrap();
        assert_eq!(draft.first().copied(), Some(3));
        assert!(!draft.is_empty());
    }

    #[test]
    fn draft_model_proposes_tokens() {
        let model = TinyModel::synthetic(ModelConfig::draft_tiny(), 1, "draft");
        let mut p = DraftModelProvider::new(model);
        let draft = p.propose(&[1, 2], 3, 0.0).unwrap();
        assert_eq!(draft.len(), 3);
        assert!(draft.iter().all(|t| *t < 64));
    }

    #[test]
    fn draft_model_reuses_kv_prefix_instead_of_replaying_context() {
        let model = TinyModel::synthetic(ModelConfig::draft_tiny(), 1, "draft");
        let mut p = DraftModelProvider::new(model);
        let mut cache = DenseKv::default();
        let first = p
            .propose_on(&[1, 2], 3, 0.0, &mut cache, None, None, None)
            .unwrap();
        assert_eq!(first.len(), 3);
        let after_first = cache.forwards;
        assert_eq!(
            after_first, 5,
            "first propose forwards the 2-token ctx then 3 drafts"
        );
        let ctx2 = vec![1, 2, first[0]];
        let second = p
            .propose_on(&ctx2, 3, 0.0, &mut cache, None, None, None)
            .unwrap();
        assert_eq!(second.len(), 3);
        let added = cache.forwards - after_first;
        assert_eq!(
            added, 4,
            "prefix hit refreshes the last ctx token then runs 3 drafts; replay would be {}; added={added}",
            ctx2.len() + 3
        );
    }

    /// A prompt-cache restore of the draft KV continues the same drafts a
    /// cache that never left memory would, and does not prefill the prefix again.
    #[test]
    fn restored_draft_cache_continues_without_replaying_the_prefix() {
        let cfg = ModelConfig::draft_tiny();
        let mut live = DraftModelProvider::new(TinyModel::synthetic(cfg.clone(), 1, "draft"));
        let mut cache = DenseKv::default();
        let first = live.propose_on(&[1, 2], 3, 0.0, &mut cache, None, None, None).unwrap();
        let bytes = cache.to_cache_bytes(&[0.5, 0.25]).unwrap();
        let (mut restored, seed) = DenseKv::from_cache_bytes(&bytes).expect("draft cache");
        assert_eq!(seed, vec![0.5, 0.25]);

        let ctx2 = vec![1, 2, first[0]];
        let live_before = cache.forwards;
        let from_live = live
            .propose_on(&ctx2, 3, 0.0, &mut cache, None, None, None)
            .unwrap();
        let mut resumed = DraftModelProvider::new(TinyModel::synthetic(cfg, 1, "draft"));
        let restored_before = restored.forwards;
        let from_restored = resumed
            .propose_on(&ctx2, 3, 0.0, &mut restored, None, None, None)
            .unwrap();
        assert_eq!(from_restored, from_live);
        assert_eq!(restored.forwards - restored_before, cache.forwards - live_before);
    }

    #[test]
    fn a_damaged_draft_cache_is_refused_without_allocating_for_it() {
        let cfg = ModelConfig::draft_tiny();
        let mut live = DraftModelProvider::new(TinyModel::synthetic(cfg, 1, "draft"));
        let mut cache = DenseKv::default();
        live.propose_on(&[1, 2], 3, 0.0, &mut cache, None, None, None).unwrap();
        let bytes = cache.to_cache_bytes(&[]).unwrap();
        assert!(DenseKv::from_cache_bytes(&bytes).is_some());
        assert!(DenseKv::from_cache_bytes(&bytes[..bytes.len() - 1]).is_none());
        let mut huge = bytes.clone();
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(DenseKv::from_cache_bytes(&huge).is_none());
        let huge_layers = [0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255];
        assert!(DenseKv::from_cache_bytes(&huge_layers).is_none());
    }
}
