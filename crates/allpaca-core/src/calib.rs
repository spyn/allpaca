//! Calibration statistics: the second moments of what each weight matrix is
//! multiplied with while the model reads a text.
//!
//! A quantiser that only sees the weights minimises `|W - Q|^2`. What the model
//! feels is `|(W - Q) x|^2` over the inputs `x` it really gets, which is
//! `tr((W - Q) H (W - Q)^T)` with `H = E[x x^T]`. [`capture`] measures `H` for
//! the attention, delta-net and feed-forward matrices on the CPU forward,
//! routed and shared experts included.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use allpaca_types::{AllpacaError, FatalError};


/// A point of the forward whose vector is the input of one or more matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Site {
    /// Normed residual: `attn_q`, `attn_k`, `attn_v`, and the delta-net
    /// projections `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`.
    AttnIn,
    /// Attention result: `attn_output`; the delta-net's gated heads: `ssm_out`.
    AttnOut,
    /// Normed residual: `ffn_gate`, `ffn_up`, the router `ffn_gate_inp`.
    FfnIn,
    /// Gated activations: `ffn_down`.
    FfnAct,
    /// Normed residual once per routed expert, weighted by the square of
    /// its routing weight: the stacked `ffn_gate_exps`, `ffn_up_exps`.
    ExpIn,
    /// A routed expert's activations, weighted the same: `ffn_down_exps`.
    ExpAct,
    /// Normed residual weighted by the square of the shared expert's gate:
    /// `ffn_gate_shexp`, `ffn_up_shexp`.
    ShexpIn,
    /// The shared expert's activations, weighted the same: `ffn_down_shexp`.
    ShexpAct,
}

impl Site {
    fn of_tensor(kind: &str) -> Option<Self> {
        Some(match kind {
            "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "attn_gate" | "ssm_alpha" | "ssm_beta" => Self::AttnIn,
            "attn_output" | "ssm_out" => Self::AttnOut,
            "ffn_gate" | "ffn_up" | "ffn_gate_inp" => Self::FfnIn,
            "ffn_down" => Self::FfnAct,
            "ffn_gate_exps" | "ffn_up_exps" => Self::ExpIn,
            "ffn_down_exps" => Self::ExpAct,
            "ffn_gate_shexp" | "ffn_up_shexp" => Self::ShexpIn,
            "ffn_down_shexp" => Self::ShexpAct,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Default)]
struct Moments {
    cols: usize,
    count: u64,
    /// Sum of the inputs' weights: the mean's denominator.
    weight: f64,
    /// Upper triangle of the weighted sum of `x x^T`, row-major (`j >= i`).
    sum: Vec<f32>,
}

impl Moments {
    fn add(&mut self, x: &[f32], weight: f32) {
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
            let xi = weight * x[i];
            let row = &mut self.sum[at..at + n - i];
            for (s, xj) in row.iter_mut().zip(&x[i..]) {
                *s += xi * xj;
            }
            at += n - i;
        }
        self.count += 1;
        self.weight += weight as f64;
    }

    /// [`Self::add`] of each `cols`-wide row of `xs` with its weight, on all
    /// cores: each sum still takes the rows in order, so it is the same.
    fn add_rows(&mut self, xs: &[f32], cols: usize, weights: &[f32]) {
        if cols == 0 || xs.is_empty() {
            return;
        }
        if self.sum.is_empty() {
            self.cols = cols;
            self.sum = vec![0.0; cols * (cols + 1) / 2];
        }
        if cols != self.cols {
            return;
        }
        let n = cols;
        let rows = |sum: &mut [f32], first: usize| {
            let mut at = 0;
            for i in first.. {
                if at >= sum.len() {
                    break;
                }
                let row = &mut sum[at..at + n - i];
                for (x, w) in xs.chunks(n).zip(weights) {
                    let xi = w * x[i];
                    for (s, xj) in row.iter_mut().zip(&x[i..]) {
                        *s += xi * xj;
                    }
                }
                at += n - i;
            }
        };
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        if threads < 2 || xs.len() / n * self.sum.len() < (1 << 22) {
            rows(&mut self.sum, 0);
        } else {
            // Upper-triangle rows split into runs of about equal area.
            let share = self.sum.len().div_ceil(threads);
            let mut runs = Vec::new();
            let (mut rest, mut first) = (&mut self.sum[..], 0);
            while first < n {
                let (mut len, mut i) = (0, first);
                while i < n && (len < share || len == 0) {
                    len += n - i;
                    i += 1;
                }
                let (run, tail) = rest.split_at_mut(len);
                runs.push((run, first));
                rest = tail;
                first = i;
            }
            std::thread::scope(|s| {
                for (run, first) in runs {
                    let rows = &rows;
                    s.spawn(move || rows(run, first));
                }
            });
        }
        self.count += weights.len().min(xs.len() / n) as u64;
        self.weight += weights.iter().take(xs.len() / n).map(|w| f64::from(*w)).sum::<f64>();
    }
}

/// `E[x x^T]` of one matrix's input.
#[derive(Debug, Clone, PartialEq)]
pub struct InputGram {
    pub cols: usize,
    /// Input vectors averaged (each routed expert's input counts once).
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
        if m.count == 0 || m.weight <= 0.0 {
            return None;
        }
        let n = m.cols;
        let scale = (1.0 / m.weight) as f32;
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
    /// While set: the vectors of one block's sites, in the order seen.
    keeping: AtomicBool,
    kept: Mutex<(usize, HashMap<Site, Vec<f32>>)>,
}

impl Tap {
    /// Whether anything is recorded: a site whose vector costs work of
    /// its own (an expert's activations) is skipped otherwise.
    #[inline]
    pub(crate) fn wants(&self) -> bool {
        self.on.load(Ordering::Relaxed) || self.keeping.load(Ordering::Relaxed)
    }

    #[inline]
    pub(crate) fn observe(&self, blk: usize, site: Site, x: &[f32]) {
        self.observe_weighted(blk, site, x, 1.0);
    }

    /// [`Self::observe`] of an input whose share of the mean is `weight`.
    #[inline]
    pub(crate) fn observe_weighted(&self, blk: usize, site: Site, x: &[f32], weight: f32) {
        if self.keeping.load(Ordering::Relaxed) {
            if let Ok(mut kept) = self.kept.lock() {
                if kept.0 == blk {
                    kept.1.entry(site).or_default().extend_from_slice(x);
                }
            }
        }
        if !self.on.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.sites.entry((blk, site)).or_default().add(x, weight);
        }
    }

    /// [`Self::observe_weighted`] of each `cols`-wide row of `xs`, row `i`
    /// with `weights[i]` (all 1 when `None`).
    pub(crate) fn observe_rows(&self, blk: usize, site: Site, xs: &[f32], cols: usize, weights: Option<&[f32]>) {
        if self.keeping.load(Ordering::Relaxed) {
            if let Ok(mut kept) = self.kept.lock() {
                if kept.0 == blk {
                    kept.1.entry(site).or_default().extend_from_slice(xs);
                }
            }
        }
        if !self.on.load(Ordering::Relaxed) || cols == 0 {
            return;
        }
        let ones;
        let weights = match weights {
            Some(w) => w,
            None => {
                ones = vec![1.0f32; xs.len() / cols];
                &ones
            }
        };
        if let Ok(mut stats) = self.stats.lock() {
            stats.sites.entry((blk, site)).or_default().add_rows(xs, cols, weights);
        }
    }

    /// Keep every vector block `blk`'s sites see from now on.
    pub(crate) fn keep(&self, blk: usize) {
        if let Ok(mut kept) = self.kept.lock() {
            *kept = (blk, HashMap::new());
        }
        self.keeping.store(true, Ordering::Relaxed);
    }

    /// Stop keeping and hand over what was kept, per site.
    pub(crate) fn take_kept(&self) -> HashMap<Site, Vec<f32>> {
        self.keeping.store(false, Ordering::Relaxed);
        self.kept.lock().map(|mut k| std::mem::take(&mut k.1)).unwrap_or_default()
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
/// second moments of its dense attention and feed-forward matrices. The
/// weights are decoded to F32 in RAM for the run, except routed experts,
/// which stay packed ([`crate::host_experts`]).
pub fn capture(model: &Path, texts: &[&str]) -> Result<InputStats, AllpacaError> {
    if texts.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "calibration needs at least one text".into(),
        )));
    }
    let (model, tok, _) = crate::sensitivity::load_host(model)?;
    model.calib.start();
    let mut run = Ok(());
    for text in texts {
        let read = tok.encode(text).and_then(|ids| match model.q4x.is_some() {
            // Its host forward block by block, where the sites are.
            true => {
                let mut logits = vec![0.0f32; ids.len() * model.config.vocab_size];
                let mut mems = [crate::qwen4exp::Qwen4Memory::traced(None)];
                crate::qwen4exp::forward_qwen4_host_rows(&model, &[&ids], &mut mems, &mut logits)
            }
            false => crate::sensitivity::text_log_probs(&model, &ids, None).map(drop),
        });
        if let Err(e) = read {
            run = Err(e);
            break;
        }
    }
    let stats = model.calib.finish();
    run.map(|()| stats)
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
        write_tiny_calib_gguf_moe(path, dim, ff, attn_out, blocks, own_head, 0);
    }

    /// [`write_tiny_calib_gguf_shape`] whose feed-forward is `experts`
    /// routed experts (two used per token, stacked `ffn_*_exps` tensors)
    /// plus a shared expert (`ffn_*_shexp`); 0 keeps the dense one.
    pub(crate) fn write_tiny_calib_gguf_moe(
        path: &Path,
        dim: usize,
        ff: usize,
        attn_out: f32,
        blocks: usize,
        own_head: bool,
        experts: usize,
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
        if experts > 0 {
            put("llama.expert_count", GgufValue::Uint32(experts as u32));
            put("llama.expert_used_count", GgufValue::Uint32(2));
        }
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
            if experts == 0 {
                add("ffn_gate", vec![d, f], rand(ff * dim, 0.3));
                add("ffn_up", vec![d, f], rand(ff * dim, 0.3));
                add("ffn_down", vec![f, d], rand(dim * ff, 0.3));
                continue;
            }
            let n = experts as u64;
            add("ffn_gate_inp", vec![d, n], rand(experts * dim, 1.0));
            add("ffn_gate_exps", vec![d, f, n], rand(experts * ff * dim, 0.3));
            add("ffn_up_exps", vec![d, f, n], rand(experts * ff * dim, 0.3));
            add("ffn_down_exps", vec![f, d, n], rand(experts * dim * ff, 0.3));
            add("ffn_gate_shexp", vec![d, f], rand(ff * dim, 0.3));
            add("ffn_up_shexp", vec![d, f], rand(ff * dim, 0.3));
            add("ffn_down_shexp", vec![f, d], rand(dim * ff, 0.3));
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

    /// Two Qwen3.5-MoE (`qwen35moe`) blocks of gated delta-net attention,
    /// each with `experts` routed experts (two used), a shared expert and
    /// its sigmoid gate: the layout of Qwen3.5/3.6 35B-A3B, tiny.
    pub(crate) fn write_tiny_qwen35moe_gguf(path: &Path, experts: usize) {
        write_tiny_qwen35moe_gguf_with(path, experts, false);
    }

    /// [`write_tiny_qwen35moe_gguf`] whose second block, with
    /// `attention_block`, is gated full attention instead (query and gate
    /// interleaved per head, q/k norms, IMRoPE).
    pub(crate) fn write_tiny_qwen35moe_gguf_with(path: &Path, experts: usize, attention_block: bool) {
        write_tiny_qwen35moe_gguf_sized(path, experts, attention_block, 16, 8, 24);
    }

    /// [`write_tiny_qwen35moe_gguf_with`] with residual width `dim`,
    /// delta-net head width `head` and expert width `ff`.
    pub(crate) fn write_tiny_qwen35moe_gguf_sized(
        path: &Path,
        experts: usize,
        attention_block: bool,
        dim: usize,
        head: usize,
        ff: usize,
    ) {
        let (vocab, n_v, n_k, conv_k) = (6usize, 2usize, 2usize, 2usize);
        let inner = n_v * head;
        let qkv_dim = n_k * head * 2 + inner;
        let mut meta = HashMap::new();
        let mut put = |k: &str, v: GgufValue| {
            meta.insert(k.to_string(), v);
        };
        let a = "qwen35moe";
        put("general.architecture", GgufValue::String(a.into()));
        put(&format!("{a}.vocab_size"), GgufValue::Uint32(vocab as u32));
        put(&format!("{a}.embedding_length"), GgufValue::Uint32(dim as u32));
        put(&format!("{a}.block_count"), GgufValue::Uint32(2));
        put(&format!("{a}.attention.head_count"), GgufValue::Uint32(2));
        put(&format!("{a}.attention.head_count_kv"), GgufValue::Uint32(1));
        put(&format!("{a}.expert_count"), GgufValue::Uint32(experts as u32));
        put(&format!("{a}.expert_used_count"), GgufValue::Uint32(2));
        put(&format!("{a}.expert_feed_forward_length"), GgufValue::Uint32(ff as u32));
        put(&format!("{a}.expert_shared_feed_forward_length"), GgufValue::Uint32(ff as u32));
        put(&format!("{a}.ssm.state_size"), GgufValue::Uint32(head as u32));
        put(&format!("{a}.ssm.inner_size"), GgufValue::Uint32(inner as u32));
        put(&format!("{a}.ssm.time_step_rank"), GgufValue::Uint32(n_v as u32));
        put(&format!("{a}.ssm.group_count"), GgufValue::Uint32(n_k as u32));
        put(&format!("{a}.ssm.conv_kernel"), GgufValue::Uint32(conv_k as u32));
        if attention_block {
            put(&format!("{a}.rope.dimension_sections"), GgufValue::Array([2i32, 2, 0, 0].map(GgufValue::Int32).to_vec()));
        }
        put("tokenizer.ggml.model", GgufValue::String("wordlevel".into()));
        put(
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["<unk>", "<s>", "</s>", "hello", "world", "again"].iter().map(|t| GgufValue::String((*t).into())).collect(),
            ),
        );
        put("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1));
        put("tokenizer.ggml.eos_token_id", GgufValue::Uint32(2));
        let mut seed = 0x51ed_270bu32;
        let mut rand = |n: usize, amp: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
                })
                .collect()
        };
        let (d, f, n) = (dim as u64, ff as u64, experts as u64);
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![("token_embd.weight".into(), vec![d, vocab as u64], rand(vocab * dim, 1.0))];
        for b in 0..2 {
            let mut add = |part: &str, dims: Vec<u64>, data: Vec<f32>| owned.push((format!("blk.{b}.{part}"), dims, data));
            add("attn_norm.weight", vec![d], vec![1.0; dim]);
            if attention_block && b == 1 {
                // Two heads of dim / 2, one key/value head.
                let hd = dim / 2;
                add("attn_q.weight", vec![d, 4 * hd as u64], rand(4 * hd * dim, 0.3));
                add("attn_k.weight", vec![d, hd as u64], rand(hd * dim, 0.3));
                add("attn_v.weight", vec![d, hd as u64], rand(hd * dim, 0.3));
                add("attn_q_norm.weight", vec![hd as u64], vec![1.0; hd]);
                add("attn_k_norm.weight", vec![hd as u64], vec![1.0; hd]);
                add("attn_output.weight", vec![2 * hd as u64, d], rand(dim * 2 * hd, 0.3));
            } else {
                add("attn_qkv.weight", vec![d, qkv_dim as u64], rand(qkv_dim * dim, 0.3));
                add("attn_gate.weight", vec![d, inner as u64], rand(inner * dim, 0.3));
                add("ssm_alpha.weight", vec![d, n_v as u64], rand(n_v * dim, 0.3));
                add("ssm_beta.weight", vec![d, n_v as u64], rand(n_v * dim, 0.3));
                add("ssm_out.weight", vec![inner as u64, d], rand(dim * inner, 0.3));
                add("ssm_a", vec![n_v as u64], vec![-0.5; n_v]);
                add("ssm_dt.bias", vec![n_v as u64], vec![0.0; n_v]);
                add("ssm_norm.weight", vec![head as u64], vec![1.0; head]);
                add("ssm_conv1d.weight", vec![conv_k as u64, qkv_dim as u64], rand(conv_k * qkv_dim, 0.3));
            }
            add("post_attention_norm.weight", vec![d], vec![1.0; dim]);
            add("ffn_gate_inp.weight", vec![d, n], rand(experts * dim, 1.0));
            add("ffn_gate_exps.weight", vec![d, f, n], rand(experts * ff * dim, 0.3));
            add("ffn_up_exps.weight", vec![d, f, n], rand(experts * ff * dim, 0.3));
            add("ffn_down_exps.weight", vec![f, d, n], rand(experts * dim * ff, 0.3));
            add("ffn_gate_inp_shexp.weight", vec![d], rand(dim, 0.3));
            add("ffn_gate_shexp.weight", vec![d, f], rand(ff * dim, 0.3));
            add("ffn_up_shexp.weight", vec![d, f], rand(ff * dim, 0.3));
            add("ffn_down_shexp.weight", vec![f, d], rand(dim * ff, 0.3));
        }
        owned.push(("output_norm.weight".into(), vec![d], vec![1.0; dim]));
        owned.push(("output.weight".into(), vec![d, vocab as u64], rand(vocab * dim, 1.0)));
        let views: Vec<GgufWriteTensor<'_>> =
            owned.iter().map(|(name, dims, data)| GgufWriteTensor { name: name.clone(), dims: dims.clone(), data }).collect();
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

    fn assert_gram_shape(g: &InputGram, what: &str) {
        assert_eq!(g.mean.len(), g.cols * g.cols, "{what}");
        for i in 0..g.cols {
            assert!(g.mean[i * g.cols + i] >= 0.0, "{what} diagonal");
            for j in 0..i {
                assert_eq!(g.mean[i * g.cols + j], g.mean[j * g.cols + i], "{what} symmetry");
            }
        }
    }

    /// Qwen3.5 blocks: the delta-net projections read the normed residual
    /// and `ssm_out` the gated head outputs; the gated full-attention
    /// block's projections read the normed residual and its output
    /// projection the gated attention.
    #[test]
    fn capture_measures_the_delta_net_and_gated_attention_inputs() {
        let path = std::env::temp_dir().join(format!("allpaca_calib_q35-{}.gguf", std::process::id()));
        write_tiny_qwen35moe_gguf_with(&path, 4, true);
        let stats = capture(&path, &["hello world again hello", "world hello"]).unwrap();
        let _ = std::fs::remove_file(&path);
        let gram = |name: &str| stats.gram(&format!("{name}.weight")).unwrap_or_else(|| panic!("no {name}"));

        let qkv = gram("blk.0.attn_qkv");
        assert_eq!(qkv.cols, 16);
        assert!(qkv.count >= 6, "one input per prompt token, got {}", qkv.count);
        for kind in ["attn_gate", "ssm_alpha", "ssm_beta"] {
            assert_eq!(gram(&format!("blk.0.{kind}")), qkv, "{kind}");
        }
        assert!((trace(&qkv) - 16.0).abs() < 1e-2, "delta-net input trace {}", trace(&qkv));
        let out = gram("blk.0.ssm_out");
        assert_eq!((out.cols, out.count), (16, qkv.count));
        assert!(trace(&out) > 0.0);

        let q = gram("blk.1.attn_q");
        assert_eq!((q.cols, q.count), (16, qkv.count));
        assert_eq!(gram("blk.1.attn_k"), q);
        assert_eq!(gram("blk.1.attn_v"), q);
        assert!((trace(&q) - 16.0).abs() < 1e-2, "attention input trace {}", trace(&q));
        let o = gram("blk.1.attn_output");
        assert_eq!((o.cols, o.count), (16, qkv.count));
        assert!(trace(&o) > 0.0);
        for g in [&qkv, &out, &q, &o] {
            assert_gram_shape(g, "qwen35");
        }
    }

    /// Routed experts: the router and the stacked gate/up read the normed
    /// residual, the stacked down projection the activations of every
    /// routed expert (one input per token and routed expert), and the
    /// shared expert its own.
    #[test]
    fn capture_measures_the_inputs_of_routed_and_shared_experts() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (llama, q35) = (dir.join(format!("allpaca_calib_moe-{pid}.gguf")), dir.join(format!("allpaca_calib_q35moe-{pid}.gguf")));
        write_tiny_calib_gguf_moe(&llama, 8, 16, 0.3, 1, false, 4);
        write_tiny_qwen35moe_gguf(&q35, 4);
        for (path, dim, ff) in [(&llama, 8usize, 16usize), (&q35, 16, 24)] {
            let stats = capture(path, &["hello world again hello", "world hello"]).unwrap();
            let _ = std::fs::remove_file(path);
            let gram = |kind: &str| stats.gram(&format!("blk.0.{kind}.weight")).unwrap_or_else(|| panic!("{path:?}: no {kind}"));
            let input = gram("ffn_gate_inp");
            assert_eq!(input.cols, dim);
            assert!(input.count >= 6, "one input per prompt token, got {}", input.count);
            assert!((trace(&input) - dim as f32).abs() < 1e-2, "router input trace {}", trace(&input));

            let exps = gram("ffn_gate_exps");
            assert_eq!(gram("ffn_up_exps"), exps);
            assert_eq!((exps.cols, exps.count), (dim, 2 * input.count), "two experts used per token");
            // Normed inputs, however they are weighted.
            assert!((trace(&exps) - dim as f32).abs() < 1e-2, "expert input trace {}", trace(&exps));
            let down = gram("ffn_down_exps");
            assert_eq!((down.cols, down.count), (ff, 2 * input.count));
            assert!(trace(&down) > 0.0);

            let sh = gram("ffn_gate_shexp");
            assert_eq!(gram("ffn_up_shexp"), sh);
            assert_eq!((sh.cols, sh.count), (dim, input.count));
            let sh_down = gram("ffn_down_shexp");
            assert_eq!((sh_down.cols, sh_down.count), (ff, input.count));
            assert!(trace(&sh_down) > 0.0);
            assert_ne!(sh_down, down, "the shared expert's activations are its own");
            for g in [&input, &exps, &down, &sh, &sh_down] {
                assert_gram_shape(g, "moe");
            }
        }
    }

    /// Rows added together, on several threads, sum to the same bits as
    /// one at a time.
    #[test]
    fn rows_added_together_equal_one_at_a_time() {
        let (cols, rows) = (300usize, 200usize);
        let xs: Vec<f32> = (0..cols * rows).map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect();
        let weights: Vec<f32> = (0..rows).map(|r| 0.1 + (r % 7) as f32 * 0.3).collect();
        let (mut one, mut all) = (Moments::default(), Moments::default());
        for (x, w) in xs.chunks(cols).zip(&weights) {
            one.add(x, *w);
        }
        all.add_rows(&xs[..cols * 3], cols, &weights[..3]);
        all.add_rows(&xs[cols * 3..], cols, &weights[3..]);
        assert_eq!((one.cols, one.count), (all.cols, all.count));
        assert_eq!(one.weight, all.weight);
        assert!(one.sum.iter().zip(&all.sum).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// A qwen4exp model's host forward measures the input of every matrix
    /// a site covers, at that matrix's width, without changing its output.
    #[test]
    fn capture_measures_the_qwen4exp_sites() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (plain, packed) = (dir.join(format!("allpaca_calib_q4x-{pid}.gguf")), dir.join(format!("allpaca_calib_q4xp-{pid}.gguf")));
        crate::qwen4exp::write_tiny_qwen4exp_gguf(&plain);
        crate::qwen4exp::write_tiny_qwen4exp_packed_gguf(&packed);
        let texts = ["hello world again hello", "world hello"];
        for path in [&plain, &packed] {
            let stats = capture(path, &texts).unwrap();
            let file = allpaca_gguf::GgufFile::open(path).unwrap();
            let (model, tok, _) = crate::sensitivity::load_host(path).unwrap();
            let ids = tok.encode(texts[0]).unwrap();
            let tokens = (ids.len() + tok.encode(texts[1]).unwrap().len()) as u64;
            let n_active = model.config.n_active as u64;
            let mut seen = 0;
            for t in &file.tensors {
                let kind = t.name.split('.').nth(2).unwrap_or("");
                if !t.name.starts_with("blk.") || Site::of_tensor(kind).is_none() {
                    continue;
                }
                let g = stats.gram(&t.name).unwrap_or_else(|| panic!("{path:?}: no input for {}", t.name));
                assert_eq!(g.cols as u64, t.dims[0], "{}", t.name);
                let per_token = if kind.ends_with("_exps") { n_active } else { 1 };
                assert_eq!(g.count, tokens * per_token, "{}", t.name);
                assert!(trace(&g) > 0.0, "{}", t.name);
                assert_gram_shape(&g, &t.name);
                seen += 1;
            }
            assert!(seen >= 30, "{path:?}: {seen} matrices");

            // The forward reads the same with the tap on.
            let vocab = model.config.vocab_size;
            let run = |on: bool| {
                let mut logits = vec![0.0f32; ids.len() * vocab];
                let mut mems = [crate::qwen4exp::Qwen4Memory::traced(None)];
                if on {
                    model.calib.start();
                }
                crate::qwen4exp::forward_qwen4_host_rows(&model, &[&ids], &mut mems, &mut logits).unwrap();
                let _ = model.calib.finish();
                logits
            };
            assert!(run(false).iter().zip(&run(true)).all(|(a, b)| a.to_bits() == b.to_bits()), "{path:?}");
            let _ = std::fs::remove_file(path);
        }
    }
}
