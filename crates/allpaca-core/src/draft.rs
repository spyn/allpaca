use std::collections::HashMap;

use allpaca_hip::HipWorkspace;
use allpaca_types::{AllpacaError, GenerationParams};

use crate::model::{DenseKv, ResidentModel, TinyModel};
use crate::sampling::{argmax, rng_from_params, sample};
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
}

impl DraftModelProvider {
    pub fn new(model: TinyModel) -> Self {
        Self { model }
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
        for i in 0..max_tokens {
            let next = if let (0, Some(tok)) = (i, forced) {
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

pub struct Qwen4MtpProvider {
    model: TinyModel,
    memory: crate::qwen4exp::Qwen4Memory,
}

impl Qwen4MtpProvider {
    pub fn new(model: TinyModel) -> Self {
        Self {
            model,
            memory: crate::qwen4exp::Qwen4Memory::default(),
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
        self.memory.nextn_seed = Some(if pos == 0 && hidden.len() == dim && hidden_width != dim {
            vec![0.0; hidden_width]
        } else {
            hidden.to_vec()
        });
        let mut logits = vec![0.0f32; self.model.config.vocab_size];
        crate::qwen4exp::forward_qwen4_token(&self.model, token, pos, &mut logits, &mut self.memory)?;
        metrics.draft_gpu_forwards.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn propose_on(&mut self, ctx: &[u32], max_tokens: usize, temperature: f32, metrics: &Metrics) -> Result<Vec<u32>, AllpacaError> {
        self.propose_inner(ctx, None, max_tokens, temperature, metrics)
    }

    pub fn propose_after(&mut self, ctx: &[u32], forced: u32, max_tokens: usize, temperature: f32, metrics: &Metrics) -> Result<Vec<u32>, AllpacaError> {
        self.propose_inner(ctx, Some(forced), max_tokens, temperature, metrics)
    }

    fn propose_inner(&mut self, ctx: &[u32], forced: Option<u32>, max_tokens: usize, temperature: f32, metrics: &Metrics) -> Result<Vec<u32>, AllpacaError> {
        if ctx.is_empty() || self.memory.tokens() != ctx.len() {
            return Ok(Vec::new());
        }
        let mut memory = self.memory.clone();
        let mut out = Vec::with_capacity(max_tokens);
        let mut window = ctx.to_vec();
        let params = GenerationParams {
            temperature,
            top_k: 1,
            top_p: 1.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let mut rng = rng_from_params(&params, Some(1));
        for i in 0..max_tokens {
            let token = match (i, forced) {
                (0, Some(tok)) => tok,
                (0, None) => ctx[ctx.len() - 1],
                _ => out[out.len() - 1],
            };
            let mut logits = vec![0.0f32; self.model.config.vocab_size];
            crate::qwen4exp::forward_qwen4_token(&self.model, token, window.len(), &mut logits, &mut memory)?;
            metrics.draft_gpu_forwards.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let next = if let (0, Some(tok)) = (i, forced) {
                tok
            } else if temperature <= 1e-5 {
                argmax(&logits)
            } else {
                sample(&mut logits, None, &params, &mut rng)
            };
            out.push(next);
            window.push(next);
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
}
