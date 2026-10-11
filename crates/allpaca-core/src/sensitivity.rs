//! Measured sensitivity: how far the model's next-token distribution moves
//! when one weight matrix is replaced by a quantised version of itself.
//!
//! A matrix's own error says how well it was quantised, not how much the
//! model cares. [`Probe`] reads calibration texts once with the original
//! weights, then again with one matrix swapped, and reports the mean
//! Kullback-Leibler divergence from the original predictions. That number is
//! comparable across matrices, which is what a bit budget needs.
//!
//! It runs on the CPU forward with every weight decoded to F32 in RAM, and
//! covers the token embedding and each block's dense attention and
//! feed-forward matrices.

use std::path::Path;
use std::sync::Arc;

use allpaca_gguf::{extract_tokenizer, GgufFile};
use allpaca_types::{AllpacaError, FatalError};

use crate::anchors::RecurrentState;
use crate::kv_tier::KvTierManager;
use crate::model::{alloc_pages_for_seq, TinyModel};
use crate::telemetry::Metrics;
use crate::tokenizer::EngineTokenizer;

const PAGE_TOKENS: u32 = 128;

pub struct Probe {
    model: TinyModel,
    /// The output head is the embedding table (no `output.weight`).
    tied_head: bool,
    texts: Vec<Vec<u32>>,
    /// Log-probabilities of the original weights: `positions * vocab`.
    reference: Vec<f32>,
}

/// Where a probed tensor lives in the loaded model.
#[derive(Clone, Copy)]
enum Slot {
    Embed,
    Head,
    /// Layer and which of q, k, v, output, gate, up, down.
    Layer(usize, usize),
}

fn bad(msg: impl Into<String>) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg.into()))
}

impl Probe {
    /// Load `model` for the CPU and read `texts` (at most `max_tokens` tokens
    /// of each) with its original weights.
    pub fn open(model: &Path, texts: &[&str], max_tokens: usize) -> Result<Self, AllpacaError> {
        let (loaded, tok, tied_head) = load_host(model)?;
        let mut ids = Vec::new();
        for text in texts {
            let mut t = tok.encode(text)?;
            t.truncate(max_tokens);
            // A single token has no context to be sensitive to.
            if t.len() >= 2 {
                ids.push(t);
            }
        }
        if ids.is_empty() {
            return Err(bad("sensitivity probe: no calibration text of two or more tokens"));
        }
        let mut probe = Self { model: loaded, tied_head, texts: ids, reference: Vec::new() };
        probe.reference = log_probs(&probe.model, &probe.texts)?;
        Ok(probe)
    }

    /// Token positions the divergence is averaged over.
    pub fn positions(&self) -> usize {
        self.texts.iter().map(Vec::len).sum()
    }

    /// Whether [`Self::divergence`] can swap GGUF tensor `name`.
    pub fn covers(&self, name: &str) -> bool {
        self.len_of(name).is_some()
    }

    /// Mean divergence of the predictions, in nats per token, with tensor
    /// `name` holding `weights` instead of its own. The model is left as it was.
    pub fn divergence(&mut self, name: &str, weights: &[f32]) -> Result<f64, AllpacaError> {
        let len = self.len_of(name).ok_or_else(|| bad(format!("sensitivity probe: no matrix {name}")))?;
        if weights.len() != len {
            return Err(bad(format!("sensitivity probe: {name} holds {len} weights, got {}", weights.len())));
        }
        let original = self.put(name, weights.to_vec());
        let run = log_probs(&self.model, &self.texts);
        self.put(name, original);
        Ok(mean_divergence(&self.reference, &run?, self.model.config.vocab_size))
    }

    fn locate(&self, name: &str) -> Option<Slot> {
        if name == "token_embd.weight" {
            return Some(Slot::Embed);
        }
        if name == "output.weight" {
            return (!self.tied_head).then_some(Slot::Head);
        }
        let mut parts = name.split('.');
        let (blk, index, kind, tail) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
        if blk != "blk" || tail != "weight" || parts.next().is_some() {
            return None;
        }
        let index: usize = index.parse().ok()?;
        let cfg = &self.model.config;
        let local = (0..cfg.n_layers).find(|l| cfg.gguf_blk(*l) == index)?;
        let kind = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
            .iter()
            .position(|k| *k == kind)?;
        Some(Slot::Layer(local, kind))
    }

    fn slot(&mut self, at: Slot) -> &mut Vec<f32> {
        match at {
            Slot::Embed => &mut self.model.embed,
            Slot::Head => &mut self.model.lm_head,
            Slot::Layer(local, kind) => {
                let layer = &mut self.model.layers[local];
                match kind {
                    0 => &mut layer.wq,
                    1 => &mut layer.wk,
                    2 => &mut layer.wv,
                    3 => &mut layer.wo,
                    4 => &mut layer.w_gate,
                    5 => &mut layer.w_up,
                    _ => &mut layer.w_down,
                }
            }
        }
    }

    fn len_of(&self, name: &str) -> Option<usize> {
        let len = match self.locate(name)? {
            Slot::Embed => self.model.embed.len(),
            Slot::Head => self.model.lm_head.len(),
            Slot::Layer(local, kind) => {
                let layer = self.model.layers.get(local)?;
                [&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w_gate, &layer.w_up, &layer.w_down][kind].len()
            }
        };
        (len > 0).then_some(len)
    }

    /// Put `weights` in tensor `name`'s place and return what was there.
    fn put(&mut self, name: &str, weights: Vec<f32>) -> Vec<f32> {
        if name == "token_embd.weight" && self.tied_head {
            self.model.lm_head = weights.clone();
        }
        match self.locate(name) {
            Some(at) => std::mem::replace(self.slot(at), weights),
            None => weights,
        }
    }
}

/// A model with every weight in F32 for the CPU forward, its tokenizer,
/// and whether its output head is the embedding table.
fn load_host(path: &Path) -> Result<(TinyModel, EngineTokenizer, bool), AllpacaError> {
    let file = Arc::new(GgufFile::open(path)?);
    let tok = EngineTokenizer::from_gguf(extract_tokenizer(&file)?);
    let mut model = TinyModel::from_gguf_host(&file)?;
    if !model.host_weights_ready() {
        return Err(bad("sensitivity probe: this architecture has no CPU forward on F32 weights"));
    }
    let tied_head = file.tensor("output.weight").is_none();
    model.bind_gguf(file);
    Ok((model, tok, tied_head))
}

/// Mean over positions of `KL(reference || now)`, both log-probabilities.
fn mean_divergence(reference: &[f32], now: &[f32], vocab: usize) -> f64 {
    let total: f64 = reference
        .chunks(vocab)
        .zip(now.chunks(vocab))
        .map(|(p, q)| p.iter().zip(q).map(|(lp, lq)| f64::from(lp.exp()) * f64::from(lp - lq)).sum::<f64>())
        .sum();
    (total / (reference.len() / vocab).max(1) as f64).max(0.0)
}

/// Log-probabilities of every position of `texts`: `positions * vocab`.
fn log_probs(model: &TinyModel, texts: &[Vec<u32>]) -> Result<Vec<f32>, AllpacaError> {
    let cfg = &model.config;
    let vocab = cfg.vocab_size;
    let metrics = Metrics::default();
    let mut out = Vec::with_capacity(texts.iter().map(Vec::len).sum::<usize>() * vocab);
    let mut logits = vec![0.0f32; vocab];
    for tokens in texts {
        let mut kv = KvTierManager::new(1 << 30, 1 << 30, 0.5);
        let pages = alloc_pages_for_seq(&mut kv, cfg, PAGE_TOKENS, tokens.len(), &metrics)?;
        let mut state = RecurrentState::zeros(cfg.n_layers, cfg.dim);
        for (pos, &token) in tokens.iter().enumerate() {
            model.forward_token(
                token,
                pos,
                &pages,
                &mut kv,
                &metrics,
                &mut logits,
                None,
                None,
                false,
                Some(&mut state),
                None,
                None,
                None,
            )?;
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let norm = max + logits.iter().map(|l| f64::from(l - max).exp()).sum::<f64>().ln() as f32;
            out.extend(logits.iter().map(|l| l - norm));
        }
    }
    Ok(out)
}

/// Mean divergence (nats per token) of the predictions of the model at
/// `other` from those of the model at `reference` on `texts`, both on the
/// CPU: how much of the reference a requantised file kept. The texts are
/// tokenised with the reference's tokenizer.
pub fn divergence_between(
    reference: &Path,
    other: &Path,
    texts: &[&str],
    max_tokens: usize,
) -> Result<f64, AllpacaError> {
    let Probe { model, texts, reference, .. } = Probe::open(reference, texts, max_tokens)?;
    let vocab = model.config.vocab_size;
    drop(model);
    let (other, _, _) = load_host(other)?;
    if other.config.vocab_size != vocab {
        return Err(bad(format!(
            "divergence: vocabularies differ ({vocab} and {})",
            other.config.vocab_size
        )));
    }
    Ok(mean_divergence(&reference, &log_probs(&other, &texts)?, vocab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calib::tests::write_tiny_calib_gguf;

    /// The probe reports nothing for a matrix's own weights, more the
    /// further the replacement is from them, and leaves the model unchanged.
    #[test]
    fn divergence_grows_with_the_damage_and_the_model_is_restored() {
        let path = std::env::temp_dir().join(format!("allpaca_probe-{}.gguf", std::process::id()));
        write_tiny_calib_gguf(&path, 64, 128);
        let file = GgufFile::open(&path).unwrap();
        let mut probe = Probe::open(&path, &["hello world again hello world", "again hello", "hello"], 16).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(probe.positions() >= 7, "the one-token text is left out: {}", probe.positions());

        for name in [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert!(probe.covers(name), "{name}");
            let w = file.tensor_f32(name).unwrap();
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: its own weights");
            let slightly: Vec<f32> = w.iter().enumerate().map(|(i, v)| v * if i % 2 == 0 { 1.05 } else { 0.95 }).collect();
            let zeros = vec![0.0f32; w.len()];
            let (small, large) = (probe.divergence(name, &slightly).unwrap(), probe.divergence(name, &zeros).unwrap());
            assert!(small > 0.0, "{name}: a 5% change moves the output ({small})");
            assert!(large > small, "{name}: zeroed {large} vs 5% {small}");
            assert!(probe.divergence(name, &w).unwrap() < 1e-9, "{name}: restored after probing");
        }

        assert!(!probe.covers("blk.0.attn_norm.weight"));
        assert!(!probe.covers("blk.3.attn_q.weight"));
        assert!(!probe.covers("output.weight"), "this file ties the head to the embedding");
        assert!(probe.divergence("blk.0.attn_norm.weight", &[1.0; 64]).is_err());
        assert!(probe.divergence("blk.0.attn_q.weight", &[0.0; 7]).is_err());
    }

    /// A file is no distance from itself, and a 2-bit requantisation of it
    /// is further from it than an 8-bit one.
    #[test]
    fn divergence_between_files_grows_as_bits_drop() {
        use crate::requant::{requantise, Options};
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let path = |tag: &str| dir.join(format!("allpaca_kld_{tag}-{pid}.gguf"));
        let (src, eight, two) = (path("src"), path("q8"), path("q2"));
        write_tiny_calib_gguf(&src, 64, 128);
        let gsq = crate::gsq::GsqConfig { steps: 20, ..Default::default() };
        requantise(&src, &eight, &Options { avg_bits: 8.5, gsq: gsq.clone(), ..Options::default() }).unwrap();
        requantise(&src, &two, &Options { avg_bits: 2.25, gsq, ..Options::default() }).unwrap();
        let texts = ["hello world again hello world", "again again hello"];
        let same = divergence_between(&src, &src, &texts, 32).unwrap();
        let d8 = divergence_between(&src, &eight, &texts, 32).unwrap();
        let d2 = divergence_between(&src, &two, &texts, 32).unwrap();
        for p in [src, eight, two] {
            let _ = std::fs::remove_file(p);
        }
        assert!(same < 1e-9, "{same}");
        assert!(d8 > 0.0 && d8 < d2, "8 bits {d8}, 2 bits {d2}");
    }
}
