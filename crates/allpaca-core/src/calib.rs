//! Calibration statistics: the second moments of what each weight matrix is
//! multiplied with while the model reads a text.
//!
//! A quantiser that only sees the weights minimises `|W - Q|^2`. What the model
//! feels is `|(W - Q) x|^2` over the inputs `x` it really gets, which is
//! `tr((W - Q) H (W - Q)^T)` with `H = E[x x^T]`. [`capture`] measures `H` for
//! every dense attention and feed-forward matrix on the CPU forward.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use allpaca_types::{AllpacaError, Config, FatalError};

use crate::engine::Engine;

/// A point of the forward whose vector is the input of one or more matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Site {
    /// Normed residual: `attn_q`, `attn_k`, `attn_v`.
    AttnIn,
    /// Attention result: `attn_output`.
    AttnOut,
    /// Normed residual: `ffn_gate`, `ffn_up`.
    FfnIn,
    /// Gated activations: `ffn_down`.
    FfnAct,
}

impl Site {
    fn of_tensor(kind: &str) -> Option<Self> {
        Some(match kind {
            "attn_q" | "attn_k" | "attn_v" => Self::AttnIn,
            "attn_output" => Self::AttnOut,
            "ffn_gate" | "ffn_up" => Self::FfnIn,
            "ffn_down" => Self::FfnAct,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Default)]
struct Moments {
    cols: usize,
    count: u64,
    /// Upper triangle of the sum of `x x^T`, row-major (`j >= i`).
    sum: Vec<f32>,
}

impl Moments {
    fn add(&mut self, x: &[f32]) {
        let n = x.len();
        if self.sum.is_empty() {
            self.cols = n;
            self.sum = vec![0.0; n * (n + 1) / 2];
        }
        if n != self.cols {
            return;
        }
        let mut at = 0;
        for i in 0..n {
            let xi = x[i];
            let row = &mut self.sum[at..at + n - i];
            for (s, xj) in row.iter_mut().zip(&x[i..]) {
                *s += xi * xj;
            }
            at += n - i;
        }
        self.count += 1;
    }
}

/// `E[x x^T]` of one matrix's input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputGram {
    pub cols: usize,
    /// Input vectors averaged.
    pub count: u64,
    /// `cols * cols`, symmetric.
    pub mean: Vec<f32>,
}

impl InputGram {
    /// The Gram with `damping` times its mean diagonal added to the
    /// diagonal. A short calibration text leaves directions it never
    /// excited; this keeps errors there from counting as free.
    pub fn damped(&self, damping: f32) -> Vec<f32> {
        let n = self.cols;
        let add = damping * (0..n).map(|i| self.mean[i * n + i]).sum::<f32>() / n.max(1) as f32;
        let mut h = self.mean.clone();
        for i in 0..n {
            h[i * n + i] += add;
        }
        h
    }
}

/// Input second moments per block and site, from a calibration run.
#[derive(Debug, Clone, Default)]
pub struct InputStats {
    sites: HashMap<(usize, Site), Moments>,
}

impl InputStats {
    /// The Gram of the input of GGUF tensor `name` (`blk.<i>.<kind>.weight`),
    /// when the run measured it.
    pub fn gram(&self, name: &str) -> Option<InputGram> {
        let mut parts = name.split('.');
        let (blk, index, kind, tail) = (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
        if blk != "blk" || tail != "weight" || parts.next().is_some() {
            return None;
        }
        let m = self.sites.get(&(index.parse().ok()?, Site::of_tensor(kind)?))?;
        if m.count == 0 {
            return None;
        }
        let n = m.cols;
        let scale = 1.0 / m.count as f32;
        let mut mean = vec![0.0f32; n * n];
        let mut at = 0;
        for i in 0..n {
            for j in i..n {
                let v = m.sum[at] * scale;
                mean[i * n + j] = v;
                mean[j * n + i] = v;
                at += 1;
            }
        }
        Some(InputGram { cols: n, count: m.count, mean })
    }

    /// Number of measured (block, site) pairs.
    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }
}

/// The model's recording point: off except during [`capture`].
#[derive(Debug, Default)]
pub(crate) struct Tap {
    on: AtomicBool,
    stats: Mutex<InputStats>,
}

impl Tap {
    #[inline]
    pub(crate) fn observe(&self, blk: usize, site: Site, x: &[f32]) {
        if !self.on.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.sites.entry((blk, site)).or_default().add(x);
        }
    }

    pub(crate) fn start(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            *stats = InputStats::default();
        }
        self.on.store(true, Ordering::Relaxed);
    }

    pub(crate) fn finish(&self) -> InputStats {
        self.on.store(false, Ordering::Relaxed);
        self.stats.lock().map(|mut s| std::mem::take(&mut *s)).unwrap_or_default()
    }
}

/// Run `texts` through the model at `model` on the CPU and return the input
/// second moments of its dense attention and feed-forward matrices. Every
/// weight is decoded to F32 in RAM for the run.
pub fn capture(model: &Path, texts: &[&str]) -> Result<InputStats, AllpacaError> {
    if texts.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "calibration needs at least one text".into(),
        )));
    }
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = false;
    cfg.model.path = Some(model.to_path_buf());
    cfg.engine.cpu_only = true;
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg.weights.host_f32 = true;
    let engine = Engine::new(cfg)?;
    engine.capture_input_stats(texts)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};

    /// One llama block with pseudo-random weights and unit norm gains.
    pub(crate) fn write_tiny_calib_gguf(path: &Path, dim: usize, ff: usize) {
        write_tiny_calib_gguf_with(path, dim, ff, 0.3);
    }

    /// [`write_tiny_calib_gguf`] with the attention output projection's
    /// weights drawn from `-attn_out..attn_out`: near zero, attention
    /// adds almost nothing to the residual.
    pub(crate) fn write_tiny_calib_gguf_with(path: &Path, dim: usize, ff: usize, attn_out: f32) {
        write_tiny_calib_gguf_shape(path, dim, ff, attn_out, 1, false);
    }

    /// [`write_tiny_calib_gguf_with`] with `blocks` blocks and, with
    /// `own_head`, an `output.weight` separate from the embedding table.
    pub(crate) fn write_tiny_calib_gguf_shape(
        path: &Path,
        dim: usize,
        ff: usize,
        attn_out: f32,
        blocks: usize,
        own_head: bool,
    ) {
        let vocab = 6usize;
        let mut meta = HashMap::new();
        let mut put = |k: &str, v: GgufValue| {
            meta.insert(k.to_string(), v);
        };
        put("general.architecture", GgufValue::String("llama".into()));
        put("llama.vocab_size", GgufValue::Uint32(vocab as u32));
        put("llama.embedding_length", GgufValue::Uint32(dim as u32));
        put("llama.block_count", GgufValue::Uint32(blocks as u32));
        put("llama.attention.head_count", GgufValue::Uint32(4));
        put("llama.attention.head_count_kv", GgufValue::Uint32(4));
        put("llama.feed_forward_length", GgufValue::Uint32(ff as u32));
        put("tokenizer.ggml.model", GgufValue::String("wordlevel".into()));
        put(
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["<unk>", "<s>", "</s>", "hello", "world", "again"]
                    .iter()
                    .map(|t| GgufValue::String((*t).into()))
                    .collect(),
            ),
        );
        put("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1));
        put("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2));
        let mut seed = 0x9e37_79b9u32;
        let mut rand = |n: usize, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
                })
                .collect()
        };
        // Dimensions in ggml order: the row length (the input width) first.
        let (d, f, v) = (dim as u64, ff as u64, vocab as u64);
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> =
            vec![("token_embd.weight".into(), vec![d, v], rand(vocab * dim, 1.0))];
        for b in 0..blocks {
            let mut add = |part: &str, dims: Vec<u64>, data: Vec<f32>| {
                owned.push((format!("blk.{b}.{part}.weight"), dims, data));
            };
            add("attn_norm", vec![d], vec![1.0; dim]);
            add("attn_q", vec![d, d], rand(dim * dim, 0.3));
            add("attn_k", vec![d, d], rand(dim * dim, 0.3));
            add("attn_v", vec![d, d], rand(dim * dim, 0.3));
            add("attn_output", vec![d, d], rand(dim * dim, attn_out));
            add("ffn_norm", vec![d], vec![1.0; dim]);
            add("ffn_gate", vec![d, f], rand(ff * dim, 0.3));
            add("ffn_up", vec![d, f], rand(ff * dim, 0.3));
            add("ffn_down", vec![f, d], rand(dim * ff, 0.3));
        }
        owned.push(("output_norm.weight".into(), vec![d], vec![1.0; dim]));
        if own_head {
            owned.push(("output.weight".into(), vec![d, v], rand(vocab * dim, 1.0)));
        }
        let views: Vec<GgufWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor { name: name.clone(), dims: dims.clone(), data })
            .collect();
        write_gguf(path, &meta, &views).unwrap();
    }

    fn trace(g: &InputGram) -> f32 {
        (0..g.cols).map(|i| g.mean[i * g.cols + i]).sum()
    }

    /// Every dense matrix of a block gets the second moments of its own
    /// input: matrices fed the same vector share them, and a normed input
    /// with unit gains has mean square 1 per element.
    #[test]
    fn capture_measures_the_input_of_every_dense_matrix() {
        let path = std::env::temp_dir().join(format!("allpaca_calib-{}.gguf", std::process::id()));
        write_tiny_calib_gguf(&path, 8, 16);
        let stats = capture(&path, &["hello world again hello", "world hello"]).unwrap();
        let _ = std::fs::remove_file(&path);

        let gram = |kind: &str| stats.gram(&format!("blk.0.{kind}.weight")).unwrap_or_else(|| panic!("no {kind}"));
        let q = gram("attn_q");
        assert_eq!(q.cols, 8);
        assert!(q.count >= 6, "one input per prompt token, got {}", q.count);
        assert_eq!(q, gram("attn_k"));
        assert_eq!(q, gram("attn_v"));
        assert_eq!(gram("ffn_gate"), gram("ffn_up"));
        assert_eq!(gram("attn_output").cols, 8);
        assert_eq!(gram("ffn_down").cols, 16);
        for kind in ["attn_q", "attn_output", "ffn_gate", "ffn_down"] {
            let g = gram(kind);
            assert_eq!(g.count, q.count, "{kind}");
            assert_eq!(g.mean.len(), g.cols * g.cols);
            for i in 0..g.cols {
                assert!(g.mean[i * g.cols + i] >= 0.0, "{kind} diagonal");
                for j in 0..i {
                    assert_eq!(g.mean[i * g.cols + j], g.mean[j * g.cols + i], "{kind} symmetry");
                }
            }
        }
        // RMS norm with unit gains: |x|^2 = dim for every input.
        assert!((trace(&q) - 8.0).abs() < 1e-2, "attention input trace {}", trace(&q));
        assert!((trace(&gram("ffn_gate")) - 8.0).abs() < 1e-2);
        assert!(trace(&gram("attn_output")) > 0.0);
        assert!(trace(&gram("ffn_down")) > 0.0);
        // The inputs differ from token to token, so this is not one outer product.
        assert!(q.mean[0] * q.mean[9] - q.mean[1] * q.mean[8] > 1e-6);

        assert!(stats.gram("blk.0.attn_norm.weight").is_none());
        assert!(stats.gram("token_embd.weight").is_none());
        assert!(stats.gram("blk.7.attn_q.weight").is_none());
    }
}
