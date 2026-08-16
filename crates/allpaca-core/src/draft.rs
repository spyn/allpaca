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
                        let p = if total > 0.0 {
                            *c as f32 / total
                        } else {
                            0.0
                        };
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
            *self
                .counts
                .entry(ctx)
                .or_default()
                .entry(next)
                .or_default() += 1;
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
        if ctx.is_empty() {
            return Ok(Vec::new());
        }
        cache.reset();
        let vocab = self.model.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        for (pos, tok) in ctx.iter().enumerate() {
            self.model.forward_dense_on(
                *tok,
                pos,
                cache,
                &mut logits,
                hip,
                resident,
                metrics,
            )?;
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
        for _ in 0..max_tokens {
            let next = if temperature <= 1e-5 {
                argmax(&logits)
            } else {
                sample(&mut logits, None, &params, &mut rng)
            };
            out.push(next);
            window.push(next);
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
}
