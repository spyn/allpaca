//! `qwen4exp` (Qwen3.8-Flash-Next): the Qwen3-Next trunk (Gated DeltaNet
//! layers with a full-attention layer every few, routed plus shared experts)
//! on a wide residual of `hc` parallel streams mixed by low-rank
//! hyper-connections in place of layer norms, with n-gram hash embeddings
//! (PLE) added at one layer. Follows llama.cpp's `src/models/qwen4exp.cpp`.
//!
//! Streams are stored stream-major: `x[s * dim + d]`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use allpaca_cpu::{attention, matmul_nt, rmsnorm, rope_imrope_neox};
use allpaca_gguf::{dequant_packed, GgufFile, GgufValue};
use allpaca_types::{AllpacaError, FatalError};

use crate::deltanet::{
    conv1d_step, decay_from_alpha, deinterleave_q_gate, expand_k_heads, gated_delta_step_eps,
    rmsnorm_heads, sigmoid, silu, GdnDims,
};
use crate::model::{topk_from_logits, ModelConfig, TinyModel};
use crate::qsa::{QsaKeys, QsaShape};
use crate::qwen4_gpu::{Qwen4DeviceHost, Qwen4DeviceState, Qwen4Gpu};

fn load_err(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

pub fn file_is_qwen4exp(file: &GgufFile) -> bool {
    file.architecture() == Some("qwen4exp")
}

/// N-gram hash embeddings: which rows of the shared table a token gathers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PleSpec {
    /// Layers that add the embedding (one in practice).
    pub layers: Vec<usize>,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub eos_token: i64,
    /// Width of one table row.
    pub head_dim: usize,
    /// One multiplier per position in the n-gram, newest first.
    pub multipliers: Vec<u64>,
    /// First row and row count of each head's slice of the table.
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
}

impl PleSpec {
    pub fn n_heads(&self) -> usize {
        self.ngram_size.saturating_sub(1) * self.heads_per_ngram
    }

    /// Positions of conv history kept: the conv is dilated by the n-gram size.
    pub fn conv_history(&self) -> usize {
        self.conv_kernel.saturating_sub(1) * self.ngram_size
    }

    /// Table rows for `token`. `prev` holds the tokens before it, newest
    /// first; a negative or missing one is before the sequence start. An
    /// end-of-sequence token in the window hides everything at or before it
    /// (the token's own EOS does not cut its context).
    pub fn rows(&self, token: i64, prev: &[i64]) -> Vec<u64> {
        let n = self.ngram_size;
        let mut ctx = vec![self.eos_token; n];
        ctx[0] = token;
        for s in 1..n {
            match prev.get(s - 1) {
                Some(&t) if t >= 0 && t != self.eos_token => ctx[s] = t,
                _ => break,
            }
        }
        let mut rows = Vec::with_capacity(self.n_heads());
        for order in 2..=n {
            let mixed = (0..order).fold(0u64, |acc, j| {
                acc ^ (ctx[j] as u64).wrapping_mul(self.multipliers[j])
            });
            let base = (order - 2) * self.heads_per_ngram;
            for h in base..base + self.heads_per_ngram {
                rows.push(mixed % self.head_vocab_sizes[h] + self.head_offsets[h]);
            }
        }
        rows
    }
}

/// What the file says about the architecture beyond the generic config.
#[derive(Debug, Clone, Default)]
pub struct Qwen4Spec {
    /// Parallel residual streams.
    pub hc: usize,
    /// A full-attention layer every this many layers; the rest are Gated DeltaNet.
    pub full_attention_interval: usize,
    pub indexer_top_k: usize,
    /// Indexer query heads and key width.
    pub indexer_heads: usize,
    pub indexer_dim: usize,
    /// Per layer: cells per indexer block on a full-attention layer (0 = dense).
    pub compress_ratios: Vec<u32>,
    pub ple: Option<PleSpec>,
    /// Gated DeltaNet shape.
    pub gdn: Option<GdnDims>,
}

fn meta_u64_array(file: &GgufFile, key: &str) -> Vec<u64> {
    match file.metadata.get(key) {
        Some(GgufValue::Array(items)) => items
            .iter()
            .map(|v| match v {
                GgufValue::Uint64(x) => *x,
                GgufValue::Int64(x) => *x as u64,
                other => u64::from(other.as_u32().unwrap_or(0)),
            })
            .collect(),
        _ => Vec::new(),
    }
}

impl Qwen4Spec {
    pub fn from_gguf(file: &GgufFile, n_layers: usize) -> Result<Self, AllpacaError> {
        let u = |key: &str| file.meta_u32(&format!("qwen4exp.{key}")).map(|v| v as usize);
        let need = |key: &str| u(key).filter(|v| *v > 0).ok_or_else(|| load_err(format!("qwen4exp.{key} missing or zero")));
        let hc = need("hyper_connection.count")?;
        if hc <= 1 {
            return Err(load_err(format!("qwen4exp.hyper_connection.count must be above one, got {hc}")));
        }
        let ple_layers = meta_u64_array(file, "qwen4exp.ple.layers");
        let ple = if ple_layers.is_empty() {
            None
        } else {
            let spec = PleSpec {
                layers: ple_layers.iter().map(|l| *l as usize).collect(),
                ngram_size: need("ple.ngram_size")?,
                heads_per_ngram: need("ple.heads_per_ngram")?,
                conv_kernel: need("ple.conv_kernel")?,
                eos_token: u("ple.eos_token_id").unwrap_or(0) as i64,
                head_dim: need("embedding_length_per_layer_input")?,
                multipliers: meta_u64_array(file, "qwen4exp.ple.layer_multipliers"),
                head_offsets: meta_u64_array(file, "qwen4exp.ple.head_offsets"),
                head_vocab_sizes: meta_u64_array(file, "qwen4exp.ple.head_vocab_sizes"),
            };
            if spec.layers.len() != 1 || spec.layers[0] >= n_layers {
                return Err(load_err(format!("qwen4exp supports one PLE layer in range, got {:?}", spec.layers)));
            }
            if spec.ngram_size < 2
                || spec.multipliers.len() < spec.ngram_size
                || spec.head_offsets.len() < spec.n_heads()
                || spec.head_vocab_sizes.len() < spec.n_heads()
                || spec.head_vocab_sizes[..spec.n_heads()].contains(&0)
            {
                return Err(load_err("qwen4exp PLE hash tables are shorter than its head count".into()));
            }
            Some(spec)
        };
        Ok(Self {
            hc,
            full_attention_interval: u("full_attention_interval").unwrap_or(4).max(1),
            indexer_top_k: u("attention.indexer.top_k").unwrap_or(0),
            indexer_heads: u("attention.indexer.head_count").unwrap_or(0),
            indexer_dim: u("attention.indexer.key_length").unwrap_or(0),
            compress_ratios: meta_u64_array(file, "qwen4exp.attention.compress_ratios")
                .into_iter()
                .map(|r| r as u32)
                .collect(),
            ple,
            gdn: GdnDims::from_meta(
                need("ssm.state_size")?,
                need("ssm.inner_size")?,
                need("ssm.time_step_rank")?,
                need("ssm.group_count")?,
                need("ssm.conv_kernel")?,
            ),
        })
    }

    /// Layer `layer` is a Gated DeltaNet (linear attention) layer.
    pub fn is_recurrent(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attention_interval != 0
    }
}

/// Weights of one hyper-connection mixer.
pub struct HcWeights<'a> {
    /// `hc * dim` gains, stream-major.
    pub norm: &'a [f32],
    /// `[low_rank, hc * dim]`.
    pub down: &'a [f32],
    /// `[hc * dim, low_rank]`.
    pub up: &'a [f32],
    /// `[hc, hc * dim]`; absent on the output head.
    pub inject: Option<&'a [f32]>,
}

/// RMS-normalise each stream of `x` and scale by `gains`.
fn grouped_norm(x: &[f32], gains: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for ((src, dst), gain) in x.chunks(dim).zip(out.chunks_mut(dim)).zip(gains.chunks(dim)) {
        let ms = src.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((d, s), g) in dst.iter_mut().zip(src).zip(gain) {
            *d = s * inv * g;
        }
    }
    out
}

/// Collapse the `hc` streams of one token into the block input: each stream
/// is normed, gated by a low-rank function of all streams, and the gated
/// streams are averaged. Also returns the `hc` injection logits that
/// [`hc_combine`] turns into per-stream weights for the block's output.
pub fn hc_mix(
    x: &[f32],
    w: &HcWeights<'_>,
    hc: usize,
    dim: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let hc_dim = hc * dim;
    let low_rank = w.down.len() / hc_dim.max(1);
    let xn = grouped_norm(x, w.norm, dim, eps);
    let mut lo = vec![0.0f32; low_rank];
    matmul_nt(&xn, w.down, &mut lo, 1, low_rank, hc_dim)?;
    for v in &mut lo {
        *v = silu(*v / hc as f32);
    }
    let mut gate = vec![0.0f32; hc_dim];
    matmul_nt(&lo, w.up, &mut gate, 1, hc_dim, low_rank)?;
    let mixed = gated_mean(&xn, &gate, hc, dim);
    let mut inject = vec![0.0f32; if w.inject.is_some() { hc } else { 0 }];
    if let Some(w_inject) = w.inject {
        matmul_nt(&xn, w_inject, &mut inject, 1, hc, hc_dim)?;
    }
    Ok((mixed, inject))
}

/// Mean over the streams of the normed stream times the sigmoid of its gate.
fn gated_mean(xn: &[f32], gate: &[f32], hc: usize, dim: usize) -> Vec<f32> {
    let mut mixed = vec![0.0f32; dim];
    for (xs, gs) in xn.chunks(dim).zip(gate.chunks(dim)) {
        for ((m, x), g) in mixed.iter_mut().zip(xs).zip(gs) {
            *m += x * sigmoid(*g);
        }
    }
    for m in &mut mixed {
        *m /= hc as f32;
    }
    mixed
}

/// Add a block's output to every stream, stream `s` weighted by
/// `2 * sigmoid(inject[s] / hc)`: a zero logit is a plain residual add.
pub fn hc_combine(streams: &mut [f32], block_out: &[f32], inject: &[f32], dim: usize) {
    let hc = inject.len();
    for (stream, logit) in streams.chunks_mut(dim).zip(inject) {
        let w = 2.0 * sigmoid(logit / hc as f32);
        for (s, b) in stream.iter_mut().zip(block_out) {
            *s += b * w;
        }
    }
}

/// Weights at which a tensor decoded on every use is split over the cores
/// (every size in tests, so that the split is checked against the file).
const DECODE_THREADED: usize = if cfg!(test) { 0 } else { 1 << 20 };

/// The model's weights, read from the file as they are needed.
pub struct Qwen4Weights {
    pub spec: Qwen4Spec,
    file: Option<Arc<GgufFile>>,
    shared: Mutex<Option<Arc<GgufFile>>>,
    /// Dense tensors decoded to F32 on first use.
    dense: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// Whether [`Self::w`] keeps what it decodes (off: decoded on every use,
    /// for a run that cannot hold every dense matrix in F32).
    keep_dense: std::sync::atomic::AtomicBool,
    /// Tensors read from these weights instead of the file, dense or not
    /// (the sensitivity probe's swaps). Host forward only.
    swapped: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// Matrices and experts on the GPU, once attached.
    pub(crate) gpu: Mutex<Option<Qwen4Gpu>>,
}

impl std::fmt::Debug for Qwen4Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen4Weights").field("spec", &self.spec).finish_non_exhaustive()
    }
}

impl Qwen4Weights {
    pub fn from_gguf(file: &GgufFile, n_layers: usize) -> Result<Self, AllpacaError> {
        let spec = Qwen4Spec::from_gguf(file, n_layers)?;
        if spec.gdn.is_none() {
            return Err(load_err("qwen4exp.ssm.* sizes do not describe a Gated DeltaNet".into()));
        }
        Ok(Self {
            spec,
            file: None,
            shared: Mutex::new(None),
            dense: Mutex::new(HashMap::new()),
            keep_dense: std::sync::atomic::AtomicBool::new(true),
            swapped: Mutex::new(HashMap::new()),
            gpu: Mutex::new(None),
        })
    }

    /// Read tensor `name` from `weights` instead of the file (`None`: from
    /// the file again), and return what it was read from before. The host
    /// forward only: weights already on a GPU are not changed.
    pub(crate) fn swap(&self, name: &str, weights: Option<Arc<Vec<f32>>>) -> Result<Option<Arc<Vec<f32>>>, AllpacaError> {
        let mut swapped = self.swapped.lock().map_err(|_| load_err("qwen4exp swapped weights poisoned".into()))?;
        Ok(match weights {
            Some(w) => swapped.insert(name.to_string(), w),
            None => swapped.remove(name),
        })
    }

    /// Tensor `name` as the host forward reads it now, in F32.
    pub(crate) fn current(&self, name: &str) -> Result<Arc<Vec<f32>>, AllpacaError> {
        match self.swapped_now(name)? {
            Some(w) => Ok(w),
            None => Ok(Arc::new(self.tensor_source(name)?.tensor_f32(name)?)),
        }
    }

    fn swapped_now(&self, name: &str) -> Result<Option<Arc<Vec<f32>>>, AllpacaError> {
        let swapped = self.swapped.lock().map_err(|_| load_err("qwen4exp swapped weights poisoned".into()))?;
        Ok(swapped.get(name).cloned())
    }

    pub fn attach_gguf(&mut self, file: Arc<GgufFile>) {
        self.file = Some(file);
    }

    pub fn attach_shared_gguf(&self, file: Arc<GgufFile>) {
        if let Ok(mut shared) = self.shared.lock() {
            *shared = Some(file);
        }
    }

    fn file(&self) -> Result<&GgufFile, AllpacaError> {
        self.file.as_deref().ok_or_else(|| load_err("qwen4exp weights have no GGUF handle".into()))
    }

    fn tensor_source(&self, name: &str) -> Result<Arc<GgufFile>, AllpacaError> {
        let file = self.file.as_ref().ok_or_else(|| load_err("qwen4exp weights have no GGUF handle".into()))?;
        if file.tensor(name).is_some() {
            return Ok(file.clone());
        }
        if let Ok(shared) = self.shared.lock() {
            if let Some(shared) = shared.as_ref().filter(|f| f.tensor(name).is_some()) {
                return Ok(shared.clone());
            }
        }
        Ok(file.clone())
    }

    pub(crate) fn gguf(&self) -> Result<&GgufFile, AllpacaError> {
        self.file()
    }

    fn has(&self, name: &str) -> bool {
        self.file.as_ref().is_some_and(|f| f.tensor(name).is_some())
            || self.shared.lock().is_ok_and(|f| f.as_ref().is_some_and(|f| f.tensor(name).is_some()))
    }

    /// Whether dense tensors stay decoded once read (the default). Off, the
    /// ones kept so far are dropped and each use decodes its tensor again,
    /// over all cores.
    pub(crate) fn keep_dense(&self, keep: bool) -> Result<(), AllpacaError> {
        self.keep_dense.store(keep, std::sync::atomic::Ordering::Relaxed);
        if !keep {
            self.dense.lock().map_err(|_| load_err("qwen4exp weight cache poisoned".into()))?.clear();
        }
        Ok(())
    }

    /// A whole tensor as F32, decoded once.
    fn w(&self, name: &str) -> Result<Arc<Vec<f32>>, AllpacaError> {
        if let Some(w) = self.swapped_now(name)? {
            return Ok(w);
        }
        if !self.keep_dense.load(std::sync::atomic::Ordering::Relaxed) {
            return self.decode_threaded(name).map(Arc::new);
        }
        let mut cache = self.dense.lock().map_err(|_| load_err("qwen4exp weight cache poisoned".into()))?;
        if let Some(w) = cache.get(name) {
            return Ok(w.clone());
        }
        let w = Arc::new(self.tensor_source(name)?.tensor_f32(name)?);
        cache.insert(name.to_string(), w.clone());
        Ok(w)
    }

    /// Tensor `name` from the file in F32, its rows split over the cores
    /// (the same values as [`GgufFile::tensor_f32`]).
    fn decode_threaded(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let file = self.tensor_source(name)?;
        let info = file.tensor(name).ok_or_else(|| load_err(format!("missing tensor {name}")))?;
        let n = info.n_elements() as usize;
        let row_len = info.dims.first().copied().unwrap_or(1).max(1) as usize;
        let n_rows = n / row_len;
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(n_rows);
        if threads < 2 || n < DECODE_THREADED {
            return file.tensor_f32(name);
        }
        let mut out = vec![0.0f32; n];
        let per = n_rows.div_ceil(threads);
        std::thread::scope(|s| {
            let jobs: Vec<_> = out
                .chunks_mut(per * row_len)
                .enumerate()
                .map(|(i, chunk)| s.spawn(move || self.rows_into(name, i * per, chunk, row_len)))
                .collect();
            jobs.into_iter().try_for_each(|j| j.join().unwrap_or_else(|_| Err(load_err("qwen4exp decode thread panicked".into()))))
        })?;
        Ok(out)
    }

    /// `n_rows` rows of `row_len` weights starting at row `first`, decoded
    /// from the packed tensor (rows must be whole quantisation blocks).
    fn rows(&self, name: &str, first: usize, n_rows: usize, row_len: usize) -> Result<Vec<f32>, AllpacaError> {
        if let Some(w) = self.swapped_now(name)? {
            return w
                .get(first * row_len..(first + n_rows) * row_len)
                .map(<[f32]>::to_vec)
                .ok_or_else(|| load_err(format!("{name}: rows {first}+{n_rows} of {row_len} past its end")));
        }
        let mut out = vec![0.0f32; n_rows * row_len];
        self.rows_into(name, first, &mut out, row_len)?;
        Ok(out)
    }

    /// The file's rows from `first` decoded into `out` (whole rows of `row_len`).
    fn rows_into(&self, name: &str, first: usize, out: &mut [f32], row_len: usize) -> Result<(), AllpacaError> {
        let n_rows = out.len() / row_len.max(1);
        let file = self.tensor_source(name)?;
        let info = file.tensor(name).ok_or_else(|| load_err(format!("missing tensor {name}")))?;
        let total = info.n_elements() as usize;
        let (lo, n) = (first * row_len, n_rows * row_len);
        // Bytes of a run of weights of this type; runs add up only on block boundaries.
        let bytes_of = |weights: usize| {
            allpaca_gguf::TensorInfo {
                dims: vec![weights as u64],
                ..info.clone()
            }
            .nbytes()
        };
        let (skip, take) = (bytes_of(lo), bytes_of(n));
        if lo + n > total || skip + take != bytes_of(lo + n) {
            return Err(load_err(format!("{name}: rows {first}+{n_rows} of {row_len} do not fall on blocks")));
        }
        let bytes = &file.tensor_bytes(name)?[skip..skip + take];
        dequant_packed(info.ggml_type, bytes, n, out)
    }
}

impl Qwen4Weights {
    /// The blocks of `n_rows` rows of `row_len` weights starting at row
    /// `first` of a `Q2_0` tensor; `None` for another encoding or rows that
    /// are not whole blocks.
    fn q2_0_rows(&self, name: &str, first: usize, n_rows: usize, row_len: usize) -> Result<Option<&[u8]>, AllpacaError> {
        if self.swapped_now(name)?.is_some() {
            return Ok(None);
        }
        let file = self.file()?;
        let info = file.tensor(name).ok_or_else(|| load_err(format!("missing tensor {name}")))?;
        if info.ggml_type != allpaca_gguf::GgmlType::Q2_0 || row_len % allpaca_cpu::Q2_0_BLOCK_ELEMS != 0 {
            return Ok(None);
        }
        let row_bytes = row_len / allpaca_cpu::Q2_0_BLOCK_ELEMS * allpaca_cpu::Q2_0_BLOCK_BYTES;
        let bytes = file.tensor_bytes(name)?;
        Ok(bytes.get(first * row_bytes..(first + n_rows) * row_bytes))
    }
}

/// SwiGLU of one expert read from its `Q2_0` blocks.
/// A `Q2_0` expert's SwiGLU on the input and hidden row rounded to 8 bits
/// per block, as the expert pool, the GPU kernels and the worker run it.
fn swiglu_q2_0(x: &[f32], gate: &[u8], up: &[u8], down: &[u8], ff: usize, dim: usize) -> Result<Vec<f32>, AllpacaError> {
    let (mut h, mut u, mut y) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; dim]);
    let input = allpaca_cpu::Q2_0Input::new(x)?;
    allpaca_cpu::gemv_q2_0_i8(gate, &input, &mut h)?;
    allpaca_cpu::gemv_q2_0_i8(up, &input, &mut u)?;
    for (h, u) in h.iter_mut().zip(&u) {
        *h = silu(*h) * u;
    }
    allpaca_cpu::gemv_q2_0_i8(down, &allpaca_cpu::Q2_0Input::new(&h)?, &mut y)?;
    Ok(y)
}

/// One sequence's state at one layer.
#[derive(Debug, Clone, Default)]
struct LayerState {
    gdn_state: Vec<f32>,
    gdn_conv: Vec<f32>,
    /// Keys and values of a full-attention layer, one row per token.
    k: Vec<f32>,
    v: Vec<f32>,
    /// Its indexer keys, when its attention can turn sparse.
    qsa: QsaKeys,
}

/// One sequence's state: the recurrent layers' states, the attention
/// layers' keys and values, the tokens so far (the n-gram hash looks back)
/// and the n-gram layer's conv history.
#[derive(Debug, Clone, Default)]
pub struct Qwen4Memory {
    tokens: Vec<i64>,
    layers: Vec<LayerState>,
    ple_history: Vec<f32>,
    pub(crate) nextn_seed: Option<Vec<f32>>,
    pub(crate) row_hiddens: Option<Vec<f32>>,
    /// When set, every routed (layer, expert) pair is appended, in order.
    pub routes: Option<Vec<(u16, u16)>>,
    /// When set, experts marked `true` (index `blk * n_experts + expert`)
    /// are never picked; the router's weights renormalise over the rest.
    pub expert_mask: Option<Arc<Vec<bool>>>,
    /// When set, per `blk * n_experts + expert`: times picked, the sum of its
    /// routing weights, and the sum of weight x the L2 norm of its output
    /// (REAP saliency). Host forward only.
    pub saliency: Option<Vec<[f64; 3]>>,
    /// When set, seconds spent per part of the forward are added up.
    pub times: Option<Qwen4Times>,
    /// What the host forward does at the top of each block, the streams
    /// standing for the hidden state (the sensitivity probe's prefix reuse).
    pub(crate) trace: Option<crate::sensitivity::LayerTrace>,
    /// The recurrent states, keys and values when the forward runs on the GPU.
    device: Option<Qwen4DeviceState>,
    /// The same read back to the host (a prompt-cache copy), put on the
    /// cards by the next GPU forward.
    device_host: Option<Qwen4DeviceHost>,
    /// This sequence's state on the worker that hosts blocks, when one does.
    remote: Option<crate::qwen4_remote::RemoteSeq>,
    /// A prompt-cache copy's state on that worker, kept there
    /// ([`Self::for_cache`]); [`Self::restore_remote`] makes it a sequence.
    remote_saved: Option<Arc<crate::qwen4_remote::RemoteSaved>>,
}

/// Seconds spent in each part of the forward.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen4Times {
    pub embed_ple: f32,
    pub mix: f32,
    /// On the GPU these two run up to the router's scores coming back, so
    /// they hold the mixers and the router as well.
    pub gdn: f32,
    pub attention: f32,
    /// Feed-forward of blocks whose routed experts are on a GPU, and of the rest.
    pub ffn_gpu: f32,
    pub ffn_host: f32,
    /// Feed-forward of blocks whose routed experts are on an RPC worker.
    pub ffn_rpc: f32,
    /// Router and shared expert up to the scores coming back (GPU forward
    /// with `ALLPACA_Q4X_SYNC_TIMES`).
    pub route: f32,
    pub combine: f32,
    pub head: f32,
}

impl Qwen4Memory {
    /// A fresh sequence whose host forward follows `trace`.
    pub(crate) fn traced(trace: Option<crate::sensitivity::LayerTrace>) -> Self {
        Self { trace, ..Self::default() }
    }

    pub fn tokens(&self) -> usize {
        self.tokens.len()
    }

    /// The tokens this memory has read, in order.
    pub(crate) fn token_ids(&self) -> &[i64] {
        &self.tokens
    }

    /// A copy for the prompt cache that holds nothing on the cards: the
    /// device state is read back to the host, and the state on a worker
    /// that hosts blocks is kept there ([`crate::qwen4_remote::RemoteSaved`]).
    /// `None` when that worker has no room for it.
    pub(crate) fn for_cache(&self, model: &TinyModel) -> Result<Option<Self>, AllpacaError> {
        let remote_saved = match (&self.remote, &self.remote_saved) {
            (Some(seq), _) => {
                let host = match model.q4x.as_ref() {
                    Some(q4) => q4.gpu.lock().map_err(|_| load_err("qwen4exp GPU lock".into()))?.as_ref().and_then(|g| g.hosted()).cloned(),
                    None => None,
                };
                let Some(host) = host else {
                    return Ok(None);
                };
                match host.save(seq, self.tokens.len())? {
                    Some(saved) => Some(Arc::new(saved)),
                    None => return Ok(None),
                }
            }
            (None, saved) => saved.clone(),
        };
        let (device_host, ple_history) = match &self.device {
            Some(device) => {
                if let Some(q4) = model.q4x.as_ref() {
                    if let Some(gpu) = q4.gpu.lock().map_err(|_| load_err("qwen4exp GPU lock".into()))?.as_ref() {
                        gpu.sync_all()?;
                    }
                }
                // The n-gram block's history is on the card once it ran there.
                let ple = device.ple_history()?.unwrap_or_else(|| self.ple_history.clone());
                (Some(device.to_host(self.tokens.len())?), ple)
            }
            None => (self.device_host.clone(), self.ple_history.clone()),
        };
        Ok(Some(Self {
            tokens: self.tokens.clone(),
            layers: self.layers.clone(),
            ple_history,
            nextn_seed: None,
            row_hiddens: None,
            routes: None,
            expert_mask: None,
            saliency: None,
            times: None,
            trace: None,
            device: None,
            device_host,
            remote: None,
            remote_saved,
        }))
    }

    /// Whether a prompt-cache copy can go to a file: not when part of it is
    /// kept on a worker.
    pub(crate) fn persistable(&self) -> bool {
        self.remote.is_none() && self.remote_saved.is_none()
    }

    /// Make a restored prompt-cache copy's state on the worker a sequence
    /// of its own. `false` when the worker no longer has it: the copy must
    /// not be used. `true` when there was nothing to restore.
    pub(crate) fn restore_remote(&mut self) -> Result<bool, AllpacaError> {
        let Some(saved) = self.remote_saved.take() else {
            return Ok(true);
        };
        let seq = saved.host().load(&saved)?;
        let restored = seq.is_some();
        self.remote = seq;
        Ok(restored)
    }

    /// Bytes held on the host.
    pub(crate) fn host_bytes(&self) -> usize {
        let layers: usize = self
            .layers
            .iter()
            .map(|l| (l.gdn_state.len() + l.gdn_conv.len() + l.k.len() + l.v.len() + l.qsa.floats()) * 4)
            .sum();
        self.tokens.len() * 8
            + layers
            + self.ple_history.len() * 4
            + self.device_host.as_ref().map_or(0, Qwen4DeviceHost::host_bytes)
    }

    /// Append to a prompt-cache checkpoint file (what [`Self::for_cache`]
    /// keeps).
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        use crate::kv_codec::write_f32s;
        out.extend_from_slice(&(self.tokens.len() as u64).to_le_bytes());
        for t in &self.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            for v in [&l.gdn_state, &l.gdn_conv, &l.k, &l.v] {
                write_f32s(out, v);
            }
            l.qsa.write_to(out);
        }
        write_f32s(out, &self.ple_history);
        // A copy with part of it on a worker cannot be read back from a file.
        out.push(if !self.persistable() { 2 } else { self.device_host.is_some() as u8 });
        if let Some(host) = &self.device_host {
            host.write_to(out);
        }
    }

    /// What [`Self::write_to`] wrote.
    pub(crate) fn read_from(r: &mut crate::kv_codec::Reader<'_>) -> Option<Self> {
        let n = usize::try_from(r.u64()?).ok()?;
        let tokens = (0..n)
            .map(|_| r.u64().map(|t| t as i64))
            .collect::<Option<Vec<_>>>()?;
        let n_layers = r.u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers.min(1024));
        for _ in 0..n_layers {
            layers.push(LayerState {
                gdn_state: r.f32s()?,
                gdn_conv: r.f32s()?,
                k: r.f32s()?,
                v: r.f32s()?,
                qsa: crate::qsa::QsaKeys::read_from(r)?,
            });
        }
        let ple_history = r.f32s()?;
        let device_host = match r.u8()? {
            0 => None,
            1 => Some(Qwen4DeviceHost::read_from(r)?),
            _ => return None,
        };
        Some(Self {
            tokens,
            layers,
            ple_history,
            device_host,
            ..Self::default()
        })
    }
}

fn gemv(x: &[f32], w: &[f32], n: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; n];
    matmul_nt(x, w, &mut out, 1, n, x.len())?;
    Ok(out)
}

fn mix(
    q4: &Qwen4Weights,
    prefix: &str,
    inject: bool,
    streams: &[f32],
    dim: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let norm = q4.w(&format!("{prefix}_norm.weight"))?;
    let down = q4.w(&format!("{prefix}_down.weight"))?;
    let up = q4.w(&format!("{prefix}_up.weight"))?;
    let inj = if inject {
        Some(q4.w(&format!("{prefix}_inject.weight"))?)
    } else {
        None
    };
    let w = HcWeights {
        norm: &norm,
        down: &down,
        up: &up,
        inject: inj.as_ref().map(|w| w.as_slice()),
    };
    hc_mix(streams, &w, q4.spec.hc, dim, eps)
}

/// Gated DeltaNet layer for one token: the Qwen3-Next step with a sigmoid
/// (not SiLU) output gate.
fn gdn_layer(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    state: &mut LayerState,
) -> Result<Vec<f32>, AllpacaError> {
    let dims = q4.spec.gdn.ok_or_else(|| load_err("qwen4exp has no Gated DeltaNet shape".into()))?;
    let name = |s: &str| format!("blk.{blk}.{s}");
    let qkv = gemv(x, &q4.w(&name("attn_qkv.weight"))?, dims.qkv_dim)?;
    let z = gemv(x, &q4.w(&name("attn_gate.weight"))?, dims.value_dim)?;
    let alpha = gemv(x, &q4.w(&name("ssm_alpha.weight"))?, dims.n_v_heads)?;
    let mut beta = gemv(x, &q4.w(&name("ssm_beta.weight"))?, dims.n_v_heads)?;
    beta.iter_mut().for_each(|b| *b = sigmoid(*b));
    if state.gdn_state.is_empty() {
        state.gdn_state = vec![0.0; dims.state_floats()];
        state.gdn_conv = vec![0.0; dims.conv_state_floats()];
    }
    let mut conv = vec![0.0f32; dims.qkv_dim];
    conv1d_step(&qkv, &q4.w(&name("ssm_conv1d.weight"))?, &mut state.gdn_conv, &mut conv, dims.conv_kernel);
    let (q_raw, rest) = conv.split_at(dims.key_dim);
    let (k_raw, v) = rest.split_at(dims.key_dim);
    let mut q = vec![0.0f32; dims.n_v_heads * dims.head_k];
    let mut k = vec![0.0f32; dims.n_v_heads * dims.head_k];
    expand_k_heads(q_raw, &mut q, dims.n_k_heads, dims.n_v_heads, dims.head_k);
    expand_k_heads(k_raw, &mut k, dims.n_k_heads, dims.n_v_heads, dims.head_k);
    let mut decay = vec![0.0f32; dims.n_v_heads];
    decay_from_alpha(&alpha, &q4.w(&name("ssm_dt.bias"))?, &q4.w(&name("ssm_a"))?, &mut decay);
    let mut y = vec![0.0f32; dims.value_dim];
    gated_delta_step_eps(
        &mut q,
        &mut k,
        v,
        &decay,
        &beta,
        &mut state.gdn_state,
        &mut y,
        dims.n_v_heads,
        dims.head_v,
        dims.head_k,
        cfg.eps,
    );
    rmsnorm_heads(&mut y, &q4.w(&name("ssm_norm.weight"))?, dims.head_v, cfg.eps);
    for (y, z) in y.iter_mut().zip(&z) {
        *y *= sigmoid(*z);
    }
    gemv(&y, &q4.w(&name("ssm_out.weight"))?, cfg.dim)
}

/// The indexer of full-attention block `blk`, when the file has one: its
/// attention turns sparse past `top_k` cells ([`crate::qsa`]).
pub(crate) fn qsa_shape(q4: &Qwen4Weights, cfg: &ModelConfig, blk: usize) -> Result<Option<QsaShape>, AllpacaError> {
    let spec = &q4.spec;
    let ratio = spec.compress_ratios.get(blk).copied().unwrap_or(0) as usize;
    if ratio == 0 || !q4.has(&format!("blk.{blk}.indexer.k_proj.weight")) {
        return Ok(None);
    }
    let (heads, dim, top_k) = (spec.indexer_heads, spec.indexer_dim, spec.indexer_top_k);
    let n_rot = if cfg.rope_dims > 0 { cfg.rope_dims } else { cfg.head_dim() };
    if ratio < 2 || top_k == 0 || top_k % ratio != 0 || heads == 0 || dim == 0 || dim % 2 != 0 || n_rot > dim {
        return Err(load_err(format!(
            "qwen4exp block {blk}: indexer of {heads} heads of {dim} with budget {top_k} cannot pool {ratio} cells"
        )));
    }
    Ok(Some(QsaShape {
        ratio,
        top_k,
        heads,
        dim,
        n_rot,
        rope_theta: cfg.rope_theta,
        rope_sections: cfg.rope_sections,
        eps: cfg.eps,
    }))
}

/// Full-attention layer for one token: gated queries, normed and rotated
/// keys, attention over every earlier token of the sequence.
fn attn_layer(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    pos: usize,
    state: &mut LayerState,
) -> Result<Vec<f32>, AllpacaError> {
    let (head_dim, n_heads, n_kv) = (cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads.max(1));
    let name = |s: &str| format!("blk.{blk}.{s}");
    let q_full = gemv(x, &q4.w(&name("attn_q.weight"))?, n_heads * head_dim * 2)?;
    let mut q = vec![0.0f32; n_heads * head_dim];
    let mut gate = vec![0.0f32; n_heads * head_dim];
    deinterleave_q_gate(&q_full, &mut q, &mut gate, head_dim);
    rmsnorm_heads(&mut q, &q4.w(&name("attn_q_norm.weight"))?, head_dim, cfg.eps);
    let mut k = gemv(x, &q4.w(&name("attn_k.weight"))?, n_kv * head_dim)?;
    rmsnorm_heads(&mut k, &q4.w(&name("attn_k_norm.weight"))?, head_dim, cfg.eps);
    let v = gemv(x, &q4.w(&name("attn_v.weight"))?, n_kv * head_dim)?;
    let n_rot = if cfg.rope_dims > 0 { cfg.rope_dims } else { head_dim };
    for head in q.chunks_mut(head_dim).chain(k.chunks_mut(head_dim)) {
        rope_imrope_neox(head, pos, cfg.rope_theta, n_rot, cfg.rope_sections)?;
    }
    if state.k.len() != pos * n_kv * head_dim {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "qwen4exp layer {blk} holds {} keys, token is at position {pos}",
            state.k.len() / (n_kv * head_dim)
        ))));
    }
    state.k.extend_from_slice(&k);
    state.v.extend_from_slice(&v);
    // The cells the indexer keeps, once there are more than its budget.
    let mut cells = None;
    if let Some(shape) = qsa_shape(q4, cfg, blk)? {
        let raw = gemv(x, &q4.w(&name("indexer.k_proj.weight"))?, shape.dim)?;
        state.qsa.push(&shape, &raw, &q4.w(&name("indexer.k_norm.weight"))?)?;
        if shape.is_sparse(pos) {
            let mut iq = gemv(x, &q4.w(&name("indexer.q_proj.weight"))?, shape.heads * shape.dim)?;
            cells = state.qsa.select(&shape, pos, &mut iq, &q4.w(&name("indexer.q_norm.weight"))?)?;
        }
    }
    let mut out = vec![0.0f32; n_heads * head_dim];
    match cells {
        None => attention(&q, &state.k, &state.v, &mut out, 1, pos + 1, n_heads, n_kv, head_dim, true)?,
        Some(cells) => {
            let row = n_kv * head_dim;
            let gather = |all: &[f32]| -> Vec<f32> {
                cells.iter().flat_map(|c| all[*c as usize * row..(*c as usize + 1) * row].iter().copied()).collect()
            };
            let (keys, values) = (gather(&state.k), gather(&state.v));
            attention(&q, &keys, &values, &mut out, 1, cells.len(), n_heads, n_kv, head_dim, true)?;
        }
    }
    for (o, g) in out.iter_mut().zip(&gate) {
        *o *= sigmoid(*g);
    }
    gemv(&out, &q4.w(&name("attn_output.weight"))?, cfg.dim)
}

fn swiglu(x: &[f32], gate: &[f32], up: &[f32], down: &[f32], ff: usize, dim: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut h = gemv(x, gate, ff)?;
    let u = gemv(x, up, ff)?;
    for (h, u) in h.iter_mut().zip(&u) {
        *h = silu(*h) * u;
    }
    gemv(&h, down, dim)
}

type ExpertJob = Box<dyn FnOnce() + Send>;

/// Threads that run the experts a GPU forward finds off its cards. Each has
/// its own queue, so handing out a block's experts wakes them all at once.
struct ExpertPool {
    workers: Mutex<Vec<std::sync::mpsc::Sender<ExpertJob>>>,
}

fn expert_pool() -> &'static ExpertPool {
    static POOL: std::sync::OnceLock<ExpertPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let mut workers = Vec::new();
        for _ in 0..threads {
            let (jobs, queue) = std::sync::mpsc::channel::<ExpertJob>();
            let spawned = std::thread::Builder::new().name("q4x-expert".into()).spawn(move || {
                for job in queue {
                    job();
                }
            });
            if spawned.is_ok() {
                workers.push(jobs);
            }
        }
        ExpertPool {
            workers: Mutex::new(workers),
        }
    })
}

/// [`routed_experts`] for several tokens at once on the worker pool, with
/// the `Q2_0` products on 8-bit activations. `xs` holds a row of `dim` per
/// token, `picks` each token's routed experts; the sums come back a row per
/// token. An expert picked by several tokens is one job that reads its
/// weights once for all of them ([`allpaca_cpu::gemm_q2_0_i8`]); each
/// token's result is the same as on its own. A calibration `tap` gets each
/// routed expert's input and activations, weighted by its routing weight
/// squared.
fn routed_experts_pooled(
    q4: &Arc<Qwen4Weights>,
    cfg: &ModelConfig,
    blk: usize,
    xs: &[f32],
    picks: &[Vec<(usize, f32)>],
    tap: Option<&crate::calib::Tap>,
) -> Result<Vec<f32>, AllpacaError> {
    let tap = tap.filter(|t| t.wants());
    let acts = tap.is_some();
    let (dim, ff) = (cfg.dim, cfg.ff_dim.max(1));
    let packed = dim % allpaca_cpu::Q2_0_BLOCK_ELEMS == 0;
    let rows: Vec<Arc<Vec<f32>>> = xs.chunks(dim).map(|x| Arc::new(x.to_vec())).collect();
    let inputs: Vec<Option<Arc<allpaca_cpu::Q2_0Input>>> = rows
        .iter()
        .map(|x| packed.then(|| allpaca_cpu::Q2_0Input::new(x).map(Arc::new)).transpose())
        .collect::<Result<_, _>>()?;
    // Each expert with the (slot, token) pairs that picked it, in first-pick order.
    let mut by_expert: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    let mut slot_row = Vec::new();
    for (row, picks) in picks.iter().enumerate() {
        for (e, _) in picks {
            let slot = slot_row.len();
            slot_row.push(row);
            match by_expert.iter_mut().find(|(x, _)| x == e) {
                Some((_, users)) => users.push((slot, row)),
                None => by_expert.push((*e, vec![(slot, row)])),
            }
        }
    }
    let (done, results) = std::sync::mpsc::channel();
    {
        let workers = expert_pool().workers.lock().map_err(|_| load_err("qwen4exp expert pool poisoned".into()))?;
        let jobs = by_expert.len();
        for (j, (e, users)) in by_expert.into_iter().enumerate() {
            let (q4, done) = (q4.clone(), done.clone());
            let xs: Vec<Arc<Vec<f32>>> = users.iter().map(|(_, r)| rows[*r].clone()).collect();
            let ins: Option<Vec<Arc<allpaca_cpu::Q2_0Input>>> = users.iter().map(|(_, r)| inputs[*r].clone()).collect();
            let job = move || {
                let name = |s: &str| format!("blk.{blk}.{s}");
                let ys = (|| -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), AllpacaError> {
                    if let (Some(ins), Some(gate), Some(up), Some(down)) = (
                        ins.as_ref(),
                        q4.q2_0_rows(&name("ffn_gate_exps.weight"), e * ff, ff, dim)?,
                        q4.q2_0_rows(&name("ffn_up_exps.weight"), e * ff, ff, dim)?,
                        q4.q2_0_rows(&name("ffn_down_exps.weight"), e * dim, dim, ff)?,
                    ) {
                        let n = ins.len();
                        let refs: Vec<&allpaca_cpu::Q2_0Input> = ins.iter().map(|x| x.as_ref()).collect();
                        let (mut h, mut u, mut y) = (vec![0.0f32; n * ff], vec![0.0f32; n * ff], vec![0.0f32; n * dim]);
                        allpaca_cpu::gemm_q2_0_i8(gate, &refs, &mut h)?;
                        allpaca_cpu::gemm_q2_0_i8(up, &refs, &mut u)?;
                        for (h, u) in h.iter_mut().zip(&u) {
                            *h = silu(*h) * u;
                        }
                        let hs: Vec<allpaca_cpu::Q2_0Input> =
                            h.chunks(ff).map(allpaca_cpu::Q2_0Input::new).collect::<Result<_, _>>()?;
                        let refs: Vec<&allpaca_cpu::Q2_0Input> = hs.iter().collect();
                        allpaca_cpu::gemm_q2_0_i8(down, &refs, &mut y)?;
                        let kept = if acts { h.chunks(ff).map(<[f32]>::to_vec).collect() } else { Vec::new() };
                        return Ok((y.chunks(dim).map(<[f32]>::to_vec).collect(), kept));
                    }
                    let gate = q4.rows(&name("ffn_gate_exps.weight"), e * ff, ff, dim)?;
                    let up = q4.rows(&name("ffn_up_exps.weight"), e * ff, ff, dim)?;
                    let down = q4.rows(&name("ffn_down_exps.weight"), e * dim, dim, ff)?;
                    if !acts {
                        let ys = xs.iter().map(|x| swiglu(x, &gate, &up, &down, ff, dim)).collect::<Result<_, _>>()?;
                        return Ok((ys, Vec::new()));
                    }
                    // swiglu with the activations kept.
                    let (mut ys, mut kept) = (Vec::new(), Vec::new());
                    for x in &xs {
                        let mut h = gemv(x, &gate, ff)?;
                        let u = gemv(x, &up, ff)?;
                        for (h, u) in h.iter_mut().zip(&u) {
                            *h = silu(*h) * u;
                        }
                        ys.push(gemv(&h, &down, dim)?);
                        kept.push(h);
                    }
                    Ok((ys, kept))
                })();
                let _ = done.send((users, ys));
            };
            // The last expert runs here: this thread would only wait.
            if j + 1 == jobs || workers.is_empty() {
                job();
            } else {
                let worker = &workers[j % workers.len()];
                worker.send(Box::new(job)).map_err(|_| load_err("qwen4exp expert pool stopped".into()))?;
            }
        }
    }
    drop(done);
    let mut ys: Vec<Option<Vec<f32>>> = vec![None; slot_row.len()];
    let mut hs: Vec<Vec<f32>> = vec![Vec::new(); if acts { slot_row.len() } else { 0 }];
    for (users, y) in results {
        let (y, h) = y?;
        for ((slot, _), y) in users.iter().zip(y) {
            ys[*slot] = Some(y);
        }
        for ((slot, _), h) in users.iter().zip(h) {
            hs[*slot] = h;
        }
    }
    if let Some(tap) = tap {
        // In slot order: tokens in order, each token's picks in routing order.
        let w2: Vec<f32> = picks.iter().flatten().map(|(_, w)| w * w).collect();
        let ins: Vec<f32> = slot_row.iter().flat_map(|r| xs[r * dim..(r + 1) * dim].iter().copied()).collect();
        let experts: Vec<usize> = picks.iter().flatten().map(|(e, _)| *e).collect();
        tap.observe_experts(blk, &ins, dim, &hs.concat(), ff, &experts, &w2);
    }
    // Each token's sum keeps its routing order.
    let weights = picks.iter().flatten().map(|(_, w)| *w);
    let mut out = vec![0.0f32; picks.len() * dim];
    for ((row, weight), y) in slot_row.iter().zip(weights).zip(ys) {
        let y = y.ok_or_else(|| load_err("qwen4exp expert worker stopped".into()))?;
        for (o, y) in out[row * dim..(row + 1) * dim].iter_mut().zip(&y) {
            *o += weight * y;
        }
    }
    Ok(out)
}

/// The weighted sum of the routed experts `picks` of block `blk` on `x`,
/// read from the file.
fn routed_experts(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    picks: &[(usize, f32)],
) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; cfg.dim];
    // The sum keeps the routing order.
    for ((_, weight), y) in picks.iter().zip(expert_outputs(q4, cfg, blk, x, picks)?) {
        for (o, y) in out.iter_mut().zip(&y) {
            *o += weight * y;
        }
    }
    Ok(out)
}

/// Each picked expert's output, in pick order (one thread per expert).
fn expert_outputs(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    picks: &[(usize, f32)],
) -> Result<Vec<Vec<f32>>, AllpacaError> {
    let (dim, ff) = (cfg.dim, cfg.ff_dim.max(1));
    let name = |s: &str| format!("blk.{blk}.{s}");
    let expert = |e: usize| -> Result<Vec<f32>, AllpacaError> {
        if let (Some(gate), Some(up), Some(down)) = (
            q4.q2_0_rows(&name("ffn_gate_exps.weight"), e * ff, ff, dim)?,
            q4.q2_0_rows(&name("ffn_up_exps.weight"), e * ff, ff, dim)?,
            q4.q2_0_rows(&name("ffn_down_exps.weight"), e * dim, dim, ff)?,
        ) {
            return swiglu_q2_0(x, gate, up, down, ff, dim);
        }
        let gate = q4.rows(&name("ffn_gate_exps.weight"), e * ff, ff, dim)?;
        let up = q4.rows(&name("ffn_up_exps.weight"), e * ff, ff, dim)?;
        let down = q4.rows(&name("ffn_down_exps.weight"), e * dim, dim, ff)?;
        swiglu(x, &gate, &up, &down, ff, dim)
    };
    let ys: Vec<Result<Vec<f32>, AllpacaError>> = std::thread::scope(|scope| {
        let expert = &expert;
        let jobs: Vec<_> = picks.iter().map(|(e, _)| scope.spawn(move || expert(*e))).collect();
        jobs.into_iter()
            .map(|j| j.join().unwrap_or_else(|_| Err(load_err("qwen4exp expert thread panicked".into()))))
            .collect()
    });
    ys.into_iter().collect()
}

/// Block `blk`'s part of an expert mask ([`Qwen4Memory::expert_mask`]).
fn block_mask(mask: Option<&Arc<Vec<bool>>>, blk: usize, n_experts: usize) -> Option<&[bool]> {
    mask.and_then(|m| m.get(blk * n_experts..(blk + 1) * n_experts))
}

/// Routed experts (softmax over all, the top few renormalised) plus the
/// shared expert behind its own sigmoid gate.
fn ffn_layer(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    mem: &mut Qwen4Memory,
) -> Result<Vec<f32>, AllpacaError> {
    let dim = cfg.dim;
    let name = |s: &str| format!("blk.{blk}.{s}");
    let router = q4.w(&name("ffn_gate_inp.weight"))?;
    let mut scores = gemv(x, &router, cfg.n_experts)?;
    let picks = topk_from_logits(&mut scores, cfg.n_active, cfg, block_mask(mem.expert_mask.as_ref(), blk, cfg.n_experts), &[]);
    if let Some(routes) = mem.routes.as_mut() {
        routes.extend(picks.iter().map(|(e, _)| (blk as u16, *e as u16)));
    }
    let mut out = match mem.saliency.as_mut() {
        Some(sal) => {
            let ys = expert_outputs(q4, cfg, blk, x, &picks)?;
            let mut out = vec![0.0f32; dim];
            for ((e, weight), y) in picks.iter().zip(&ys) {
                let norm = y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>().sqrt();
                if let Some(slot) = sal.get_mut(blk * cfg.n_experts + e) {
                    slot[0] += 1.0;
                    slot[1] += f64::from(*weight);
                    slot[2] += f64::from(*weight) * norm;
                }
                for (o, y) in out.iter_mut().zip(y) {
                    *o += weight * y;
                }
            }
            out
        }
        None => routed_experts(q4, cfg, blk, x, &picks)?,
    };
    if q4.has(&name("ffn_up_shexp.weight")) {
        let gate = q4.w(&name("ffn_gate_shexp.weight"))?;
        let up = q4.w(&name("ffn_up_shexp.weight"))?;
        let down = q4.w(&name("ffn_down_shexp.weight"))?;
        let y = swiglu(x, &gate, &up, &down, up.len() / dim, dim)?;
        let gate_inp = q4.w(&name("ffn_gate_inp_shexp.weight"))?;
        let open = sigmoid(x.iter().zip(gate_inp.iter()).map(|(a, b)| a * b).sum());
        for (o, y) in out.iter_mut().zip(&y) {
            *o += open * y;
        }
    }
    Ok(out)
}

/// The n-gram layer's update of the streams for one token: a value gathered
/// from the hash table, gated per stream by how well its key matches the
/// stream, plus a dilated causal conv over the gated values of earlier tokens.
/// `history` holds [`PleSpec::conv_history`] earlier inputs of the conv,
/// oldest first, and is advanced. `norms` are the key, query and conv gains.
#[allow(clippy::too_many_arguments)]
pub fn ple_update(
    streams: &mut [f32],
    emb: &[f32],
    key_w: &[f32],
    value_w: &[f32],
    norms: [&[f32]; 3],
    conv_w: &[f32],
    history: &mut Vec<f32>,
    ple: &PleSpec,
    dim: usize,
    eps: f32,
) -> Result<(), AllpacaError> {
    let key = gemv(emb, key_w, streams.len())?;
    let value = gemv(emb, value_w, dim)?;
    ple_apply(streams, &key, &value, norms, conv_w, history, ple, dim, eps)
}

/// [`ple_update`] from the embedding's key and value projections.
#[allow(clippy::too_many_arguments)]
fn ple_apply(
    streams: &mut [f32],
    key: &[f32],
    value: &[f32],
    norms: [&[f32]; 3],
    conv_w: &[f32],
    history: &mut Vec<f32>,
    ple: &PleSpec,
    dim: usize,
    eps: f32,
) -> Result<(), AllpacaError> {
    let hc_dim = streams.len();
    let key = grouped_norm(key, norms[0], dim, eps);
    let query = grouped_norm(streams, norms[1], dim, eps);
    let mut gated = vec![0.0f32; hc_dim];
    for ((dst, k), q) in gated.chunks_mut(dim).zip(key.chunks(dim)).zip(query.chunks(dim)) {
        let s = k.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() / (dim as f32).sqrt();
        // Signed square root before the sigmoid.
        let signed = if s == 0.0 { 0.0 } else { s.signum() * s.abs().max(1e-6).sqrt() };
        let open = sigmoid(signed);
        for (d, v) in dst.iter_mut().zip(value) {
            *d = v * open;
        }
    }
    let x = grouped_norm(&gated, norms[2], dim, eps);
    let (kern, hist) = (ple.conv_kernel, ple.conv_history());
    if history.len() != hist * hc_dim {
        *history = vec![0.0; hist * hc_dim];
    }
    for c in 0..hc_dim {
        // Tap k reads (kern - 1 - k) * ngram_size positions back; the last is the token itself.
        let mut acc = conv_w[kern - 1 + c * kern] * x[c];
        for k in 0..kern - 1 {
            let back = (kern - 1 - k) * ple.ngram_size;
            acc += conv_w[k + c * kern] * history[(hist - back) * hc_dim + c];
        }
        streams[c] += gated[c] + silu(acc);
    }
    if hist > 0 {
        history.copy_within(hc_dim.., 0);
        history[(hist - 1) * hc_dim..].copy_from_slice(&x);
    }
    Ok(())
}

/// The n-gram rows of `token` after `earlier`, gathered from the table.
fn ple_embedding(q4: &Qwen4Weights, ple: &PleSpec, token: u32, earlier: &[i64]) -> Result<Vec<f32>, AllpacaError> {
    let prev: Vec<i64> = earlier.iter().rev().take(ple.ngram_size - 1).copied().collect();
    let mut emb = Vec::with_capacity(ple.n_heads() * ple.head_dim);
    for row in ple.rows(i64::from(token), &prev) {
        emb.extend(q4.rows("per_layer_token_embd.weight", row as usize, 1, ple.head_dim)?);
    }
    Ok(emb)
}

/// The NextN block's starting streams for `token` after the hidden streams
/// `hidden` (`hc * dim`, as the trunk leaves them before its output mixer):
/// per stream `s`, `eh_proj([enorm(e) ; hnorm_s(h_s)])`, each stream normed
/// on its own (llama.cpp `graph_mtp`). The result is `hc * dim` wide.
fn nextn_seed(q4: &Qwen4Weights, cfg: &ModelConfig, token: u32, hidden: &[f32]) -> Result<Vec<f32>, AllpacaError> {
    let (dim, hc) = (cfg.dim, q4.spec.hc);
    let blk = cfg.gguf_blk(0);
    let hnorm = q4.w(&format!("blk.{blk}.nextn.hnorm.weight"))?;
    if hidden.len() != hc * dim || hnorm.len() != hc * dim {
        return Err(load_err("NextN hidden state has the wrong width".into()));
    }
    let tok = q4.rows("token_embd.weight", token as usize, 1, dim)?;
    let mut e_norm = vec![0.0f32; dim];
    rmsnorm(&tok, &q4.w(&format!("blk.{blk}.nextn.enorm.weight"))?, cfg.eps, &mut e_norm)?;
    // One `[e_norm ; h_norm_s]` row per stream.
    let mut concat = vec![0.0f32; hc * 2 * dim];
    for (s, row) in concat.chunks_mut(2 * dim).enumerate() {
        let (e_row, h_row) = row.split_at_mut(dim);
        e_row.copy_from_slice(&e_norm);
        let stream = s * dim..(s + 1) * dim;
        rmsnorm(&hidden[stream.clone()], &hnorm[stream], cfg.eps, h_row)?;
    }
    let mut seed = vec![0.0f32; hc * dim];
    matmul_nt(&concat, &q4.w(&format!("blk.{blk}.nextn.eh_proj.weight"))?, &mut seed, hc, dim, 2 * dim)?;
    Ok(seed)
}

/// Forward one token at `pos`; `logits` gets the next-token logits.
pub(crate) fn forward_qwen4_token(
    model: &TinyModel,
    token: u32,
    pos: usize,
    logits: &mut [f32],
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    forward_qwen4_tokens(model, &[token], pos, logits, None, mem)
}

/// Forward `tokens` from `start_pos`: on the GPU several per pass, on the
/// host one at a time. `logits` ends with the last token's; `row_logits`,
/// when given, gets every token's logits, one row of the vocabulary size
/// each (a draft verify reads them all).
fn forward_qwen4_tokens_impl(
    model: &TinyModel,
    tokens: &[u32],
    start_pos: usize,
    logits: &mut [f32],
    mut row_logits: Option<&mut [f32]>,
    mut out_row_hiddens: Option<&mut Vec<f32>>,
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    let q4 = model.q4x.as_ref().ok_or_else(|| load_err("missing qwen4exp weights".into()))?;
    let cfg = &model.config;
    let vocab = cfg.vocab_size;
    if row_logits.as_ref().is_some_and(|rows| rows.len() < tokens.len() * vocab) || logits.len() < vocab {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "qwen4exp logits buffer is smaller than the rows asked for".into(),
        )));
    }
    if mem.tokens.len() != start_pos {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "qwen4exp sequence memory holds {} tokens, forward is at position {start_pos}",
            mem.tokens.len()
        ))));
    }
    if !mem.restore_remote()? {
        return Err(load_err("qwen4exp: the worker no longer keeps this sequence's state".into()));
    }
    let gpu = q4.gpu.lock().map_err(|_| load_err("qwen4exp gpu lock poisoned".into()))?;
    if let Some(gpu) = gpu.as_ref() {
        let rpc = model.rpc.as_deref();
        if let Some(host) = gpu.hosted().filter(|_| row_logits.is_none() && tokens.len() > crate::qwen4_gpu::MAX_ROWS) {
            let host = host.clone();
            return forward_pipelined(q4, gpu, rpc, &host, cfg, tokens, start_pos, logits, out_row_hiddens, mem);
        }
        for (pass, chunk) in tokens.chunks(crate::qwen4_gpu::MAX_ROWS).enumerate() {
            let first = pass * crate::qwen4_gpu::MAX_ROWS;
            let rows = row_logits.as_deref_mut().map(|rows| &mut rows[first * vocab..(first + chunk.len()) * vocab]);
            forward_device(q4, gpu, rpc, cfg, chunk, start_pos + first, &mut logits[..vocab], rows, mem, None)?;
            if let Some(out) = out_row_hiddens.as_deref_mut() {
                if let Some(captured) = mem.row_hiddens.take() {
                    out.extend(captured);
                }
            }
        }
        return Ok(());
    }
    for (i, &token) in tokens.iter().enumerate() {
        forward_host(q4, cfg, token, start_pos + i, logits, mem)?;
        mem.tokens.push(i64::from(token));
        if let Some(rows) = row_logits.as_deref_mut() {
            rows[i * vocab..(i + 1) * vocab].copy_from_slice(&logits[..vocab]);
        }
        if let Some(out) = out_row_hiddens.as_deref_mut() {
            if let Some(captured) = mem.row_hiddens.take() {
                out.extend(captured);
            }
        }
    }
    Ok(())
}

pub(crate) fn forward_qwen4_tokens(
    model: &TinyModel,
    tokens: &[u32],
    start_pos: usize,
    logits: &mut [f32],
    row_logits: Option<&mut [f32]>,
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    forward_qwen4_tokens_impl(model, tokens, start_pos, logits, row_logits, None, mem)
}

pub(crate) fn forward_qwen4_tokens_with_hiddens(
    model: &TinyModel,
    tokens: &[u32],
    start_pos: usize,
    logits: &mut [f32],
    row_logits: Option<&mut [f32]>,
    out_row_hiddens: &mut Vec<f32>,
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    forward_qwen4_tokens_impl(model, tokens, start_pos, logits, row_logits, Some(out_row_hiddens), mem)
}

/// A prompt through a worker that hosts blocks, in passes that overlap:
/// while the worker runs one pass's hosted blocks, this machine runs the
/// next pass's blocks up to them. Each block still sees the passes in
/// order, so the result is that of one pass after the other. A pass goes
/// to the worker only once the one before is back, so a single pass is in
/// flight whatever its size. `logits` gets the last token's;
/// `out_row_hiddens`, when given, every token's hidden streams in order.
#[allow(clippy::too_many_arguments)]
fn forward_pipelined(
    q4: &Arc<Qwen4Weights>,
    gpu: &Qwen4Gpu,
    rpc: Option<&crate::rpc_runtime::RpcRuntime>,
    host: &Arc<crate::qwen4_remote::RemoteBlocks>,
    cfg: &ModelConfig,
    tokens: &[u32],
    start_pos: usize,
    logits: &mut [f32],
    mut out_row_hiddens: Option<&mut Vec<f32>>,
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    let (vocab, rows) = (cfg.vocab_size, crate::qwen4_gpu::MAX_ROWS);
    host.pipeline(|pipe| {
        // The pass whose hosted blocks the worker has, with its tokens and position.
        let mut in_flight: Option<(&[u32], usize, DevicePass)> = None;
        let mut finish = |pipe: &mut crate::qwen4_remote::Pipe<'_>, mem: &mut Qwen4Memory, logits: &mut [f32], done: Option<(&[u32], usize, DevicePass)>| {
            let Some((chunk, pos, mut pass)) = done else {
                return Ok(());
            };
            // What this machine waits for the worker shows as `ffn_rpc`.
            let clock = std::time::Instant::now();
            pipe.recv(&mut pass.streams)?;
            if let Some(times) = mem.times.as_mut() {
                times.ffn_rpc += clock.elapsed().as_secs_f32();
            }
            if forward_device(q4, gpu, rpc, cfg, chunk, pos, &mut logits[..vocab], None, mem, Some(Some(pass)))?.is_some() {
                return Err(load_err("qwen4exp: a second run of hosted blocks in one pass".into()));
            }
            if let (Some(out), Some(captured)) = (out_row_hiddens.as_deref_mut(), mem.row_hiddens.take()) {
                out.extend(captured);
            }
            Ok(())
        };
        for (i, chunk) in tokens.chunks(rows).enumerate() {
            let pos = start_pos + i * rows;
            let pass = forward_device(q4, gpu, rpc, cfg, chunk, pos, &mut logits[..vocab], None, mem, Some(None))?
                .ok_or_else(|| load_err("qwen4exp: a pipelined pass did not reach the hosted blocks".into()))?;
            finish(pipe, mem, logits, in_flight.take())?;
            let seq = mem.remote.as_ref().ok_or_else(|| load_err("qwen4exp: no sequence on the worker".into()))?;
            pipe.send(seq, pos, chunk.len(), &pass.streams)?;
            in_flight = Some((chunk, pos, pass));
        }
        finish(pipe, mem, logits, in_flight.take())
    })
}

/// One forward over the rows of several sequences (the engine's batched
/// decode tick): `tokens` holds each sequence's rows in `segs` order and
/// `row_logits` gets a row of logits per token. Each sequence's rows equal
/// its own forward. With a worker hosting blocks the sequences are
/// pipelined: the worker runs one's hosted blocks while this machine runs
/// the next one's local blocks.
pub(crate) fn forward_qwen4_batch(
    model: &TinyModel,
    tokens: &[u32],
    segs: &mut [crate::model::SeqRows<'_>],
    row_logits: &mut [f32],
) -> Result<(), AllpacaError> {
    use crate::qwen4_gpu::MAX_ROWS;
    let q4 = model.q4x.as_ref().ok_or_else(|| load_err("missing qwen4exp weights".into()))?;
    let (cfg, vocab) = (&model.config, model.config.vocab_size);
    let no_state = || load_err("qwen4exp needs the sequence's state".into());
    let mut first_row = Vec::with_capacity(segs.len());
    let mut row = 0usize;
    for seg in segs.iter() {
        first_row.push(row);
        row += seg.n_rows;
    }
    if row != tokens.len() || row_logits.len() < row * vocab {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "qwen4exp batch: rows, tokens and logits do not line up".into(),
        )));
    }
    let mut last = vec![0.0f32; vocab];
    let guard = q4.gpu.lock().map_err(|_| load_err("qwen4exp gpu lock poisoned".into()))?;
    let stream_bytes = tokens.len() * q4.spec.hc * cfg.dim * 4;
    let host = guard
        .as_ref()
        .and_then(|gpu| gpu.hosted())
        .filter(|_| segs.len() > 1 && stream_bytes <= crate::qwen4_remote::PIPE_BYTES)
        .filter(|_| segs.iter().all(|s| (1..=MAX_ROWS).contains(&s.n_rows)))
        .cloned();
    let (Some(host), Some(gpu)) = (host, guard.as_ref()) else {
        drop(guard);
        for (seg, &at) in segs.iter_mut().zip(&first_row) {
            let mem = &mut seg.recurrent.as_deref_mut().ok_or_else(no_state)?.q4x;
            let rows = &mut row_logits[at * vocab..(at + seg.n_rows) * vocab];
            forward_qwen4_tokens(model, &tokens[at..at + seg.n_rows], seg.start_pos, &mut last, Some(rows), mem)?;
        }
        return Ok(());
    };
    let rpc = model.rpc.as_deref();
    host.pipeline(|pipe| {
        // Every sequence up to the hosted blocks, handing each to the worker
        // as soon as it gets there.
        let mut passes = Vec::with_capacity(segs.len());
        for (seg, &at) in segs.iter_mut().zip(&first_row) {
            let mem = &mut seg.recurrent.as_deref_mut().ok_or_else(no_state)?.q4x;
            if mem.tokens.len() != seg.start_pos {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "qwen4exp sequence memory holds {} tokens, forward is at position {}",
                    mem.tokens.len(),
                    seg.start_pos
                ))));
            }
            let (toks, rows) = (&tokens[at..at + seg.n_rows], &mut row_logits[at * vocab..(at + seg.n_rows) * vocab]);
            let pass = forward_device(q4, gpu, rpc, cfg, toks, seg.start_pos, &mut last, Some(rows), mem, Some(None))?;
            if let (Some(pass), Some(remote)) = (&pass, mem.remote.as_ref()) {
                pipe.send(remote, seg.start_pos, seg.n_rows, &pass.streams)?;
            }
            passes.push(pass);
        }
        // Then each one's result, in the order sent, and its blocks after.
        for ((seg, &at), pass) in segs.iter_mut().zip(&first_row).zip(passes) {
            let Some(mut pass) = pass else {
                continue;
            };
            pipe.recv(&mut pass.streams)?;
            let mem = &mut seg.recurrent.as_deref_mut().ok_or_else(no_state)?.q4x;
            let (toks, rows) = (&tokens[at..at + seg.n_rows], &mut row_logits[at * vocab..(at + seg.n_rows) * vocab]);
            if forward_device(q4, gpu, rpc, cfg, toks, seg.start_pos, &mut last, Some(rows), mem, Some(Some(pass)))?.is_some() {
                return Err(load_err("qwen4exp: a second run of hosted blocks in one pass".into()));
            }
        }
        Ok(())
    })
}

/// A GPU pass that stopped where the worker's hosted blocks begin.
struct DevicePass {
    /// The streams there; the worker's result goes in their place.
    streams: Vec<f32>,
    /// Where the pass goes on: the layer after the hosted blocks and its card.
    resume: usize,
    card: usize,
}

fn forward_host(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    token: u32,
    pos: usize,
    logits: &mut [f32],
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    let (dim, hc, eps) = (cfg.dim, q4.spec.hc, cfg.eps);
    mem.layers.resize_with(cfg.n_layers, LayerState::default);
    let mut times = mem.times.unwrap_or_default();
    let mut clock = std::time::Instant::now();
    let mut lap = |slot: &mut f32| {
        *slot += clock.elapsed().as_secs_f32();
        clock = std::time::Instant::now();
    };
    let mut streams = match mem.nextn_seed.as_deref() {
        Some(hidden) => nextn_seed(q4, cfg, token, hidden)?,
        // The trunk starts from hc copies of the embedding.
        None => q4.rows("token_embd.weight", token as usize, 1, dim)?.repeat(hc),
    };
    for layer in 0..cfg.n_layers {
        if mem.trace.as_mut().is_some_and(|t| t.enter(pos, layer, cfg.n_layers, &mut streams)) {
            continue;
        }
        let blk = cfg.gguf_blk(layer);
        if let Some(ple) = q4.spec.ple.as_ref().filter(|p| p.layers.contains(&blk)) {
            let emb = ple_embedding(q4, ple, token, &mem.tokens)?;
            let name = |s: &str| format!("blk.{blk}.ple_{s}.weight");
            let (nk, nq, nc) = (q4.w(&name("norm_key"))?, q4.w(&name("norm_query"))?, q4.w(&name("norm_conv"))?);
            ple_update(
                &mut streams,
                &emb,
                &q4.w(&name("key"))?,
                &q4.w(&name("value"))?,
                [&nk, &nq, &nc],
                &q4.w(&name("conv1d"))?,
                &mut mem.ple_history,
                ple,
                dim,
                eps,
            )?;
        }
        lap(&mut times.embed_ple);
        let (x, inject) = mix(q4, &format!("blk.{blk}.hc_attn"), true, &streams, dim, eps)?;
        lap(&mut times.mix);
        let out = if q4.spec.is_recurrent(layer) {
            let out = gdn_layer(q4, cfg, blk, &x, &mut mem.layers[layer])?;
            lap(&mut times.gdn);
            out
        } else {
            let out = attn_layer(q4, cfg, blk, &x, pos, &mut mem.layers[layer])?;
            lap(&mut times.attention);
            out
        };
        hc_combine(&mut streams, &out, &inject, dim);
        lap(&mut times.combine);
        let (x, inject) = mix(q4, &format!("blk.{blk}.hc_ffn"), true, &streams, dim, eps)?;
        lap(&mut times.mix);
        let out = ffn_layer(q4, cfg, blk, &x, mem)?;
        lap(&mut times.ffn_host);
        hc_combine(&mut streams, &out, &inject, dim);
        lap(&mut times.combine);
    }
    if mem.trace.as_mut().is_some_and(|t| t.enter(pos, cfg.n_layers, cfg.n_layers, &mut streams)) {
        return Ok(());
    }
    // The output mixer stands in for the final norm.
    mem.row_hiddens = Some(streams.clone());
    if mem.nextn_seed.is_some() {
        mem.nextn_seed = mem.row_hiddens.clone();
    }
    let mixer = if mem.nextn_seed.is_some() {
        format!("blk.{}.nextn.hc_head", cfg.gguf_blk(0))
    } else {
        "output_hc".into()
    };
    let (x, _) = mix(q4, &mixer, false, &streams, dim, eps)?;
    let head = if q4.has("output.weight") { "output.weight" } else { "token_embd.weight" };
    matmul_nt(&x, &q4.w(head)?, logits, 1, cfg.vocab_size, dim)?;
    lap(&mut times.head);
    if mem.times.is_some() {
        mem.times = Some(times);
    }
    Ok(())
}

/// One row of a block-major host pass: its sequence, position and token.
#[derive(Clone, Copy)]
struct HostRow {
    seq: usize,
    pos: usize,
    token: u32,
}

/// `x` (`rows` rows) times the transpose of `w` (`n` rows).
fn gemv_rows(x: &[f32], w: &[f32], rows: usize, n: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; rows * n];
    matmul_nt(x, w, &mut out, rows, n, x.len() / rows.max(1))?;
    Ok(out)
}

/// [`mix`] of `rows` tokens' streams at once: the mixed rows and the
/// injection logits, a row per token.
fn mix_rows(
    q4: &Qwen4Weights,
    prefix: &str,
    inject: bool,
    streams: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let hc = q4.spec.hc;
    let hc_dim = hc * dim;
    let norm = q4.w(&format!("{prefix}_norm.weight"))?;
    let down = q4.w(&format!("{prefix}_down.weight"))?;
    let up = q4.w(&format!("{prefix}_up.weight"))?;
    let low_rank = down.len() / hc_dim.max(1);
    let xn: Vec<f32> = streams.chunks(hc_dim).flat_map(|x| grouped_norm(x, &norm, dim, eps)).collect();
    let mut lo = gemv_rows(&xn, &down, rows, low_rank)?;
    for v in &mut lo {
        *v = silu(*v / hc as f32);
    }
    let gate = gemv_rows(&lo, &up, rows, hc_dim)?;
    let mixed = xn.chunks(hc_dim).zip(gate.chunks(hc_dim)).flat_map(|(x, g)| gated_mean(x, g, hc, dim)).collect();
    let inject = match inject {
        true => gemv_rows(&xn, &q4.w(&format!("{prefix}_inject.weight"))?, rows, hc)?,
        false => Vec::new(),
    };
    Ok((mixed, inject))
}

/// [`gdn_layer`] for `rows` (one input row each), the projections shared.
fn gdn_rows(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    layer: usize,
    x: &[f32],
    rows: &[HostRow],
    mems: &mut [Qwen4Memory],
    tap: &crate::calib::Tap,
) -> Result<Vec<f32>, AllpacaError> {
    let dims = q4.spec.gdn.ok_or_else(|| load_err("qwen4exp has no Gated DeltaNet shape".into()))?;
    let name = |s: &str| format!("blk.{blk}.{s}");
    let n = rows.len();
    tap.observe_rows(blk, crate::calib::Site::AttnIn, x, cfg.dim, None);
    let qkv = gemv_rows(x, &q4.w(&name("attn_qkv.weight"))?, n, dims.qkv_dim)?;
    let z = gemv_rows(x, &q4.w(&name("attn_gate.weight"))?, n, dims.value_dim)?;
    let alpha = gemv_rows(x, &q4.w(&name("ssm_alpha.weight"))?, n, dims.n_v_heads)?;
    let mut beta = gemv_rows(x, &q4.w(&name("ssm_beta.weight"))?, n, dims.n_v_heads)?;
    beta.iter_mut().for_each(|b| *b = sigmoid(*b));
    let (conv_w, dt, a, norm) =
        (q4.w(&name("ssm_conv1d.weight"))?, q4.w(&name("ssm_dt.bias"))?, q4.w(&name("ssm_a"))?, q4.w(&name("ssm_norm.weight"))?);
    let mut ys = vec![0.0f32; n * dims.value_dim];
    for (i, row) in rows.iter().enumerate() {
        let state = &mut mems[row.seq].layers[layer];
        if state.gdn_state.is_empty() {
            state.gdn_state = vec![0.0; dims.state_floats()];
            state.gdn_conv = vec![0.0; dims.conv_state_floats()];
        }
        let mut conv = vec![0.0f32; dims.qkv_dim];
        conv1d_step(&qkv[i * dims.qkv_dim..(i + 1) * dims.qkv_dim], &conv_w, &mut state.gdn_conv, &mut conv, dims.conv_kernel);
        let (q_raw, rest) = conv.split_at(dims.key_dim);
        let (k_raw, v) = rest.split_at(dims.key_dim);
        let mut q = vec![0.0f32; dims.n_v_heads * dims.head_k];
        let mut k = vec![0.0f32; dims.n_v_heads * dims.head_k];
        expand_k_heads(q_raw, &mut q, dims.n_k_heads, dims.n_v_heads, dims.head_k);
        expand_k_heads(k_raw, &mut k, dims.n_k_heads, dims.n_v_heads, dims.head_k);
        let heads = i * dims.n_v_heads..(i + 1) * dims.n_v_heads;
        let mut decay = vec![0.0f32; dims.n_v_heads];
        decay_from_alpha(&alpha[heads.clone()], &dt, &a, &mut decay);
        let y = &mut ys[i * dims.value_dim..(i + 1) * dims.value_dim];
        gated_delta_step_eps(
            &mut q,
            &mut k,
            v,
            &decay,
            &beta[heads],
            &mut state.gdn_state,
            y,
            dims.n_v_heads,
            dims.head_v,
            dims.head_k,
            cfg.eps,
        );
        rmsnorm_heads(y, &norm, dims.head_v, cfg.eps);
        for (y, z) in y.iter_mut().zip(&z[i * dims.value_dim..]) {
            *y *= sigmoid(*z);
        }
    }
    tap.observe_rows(blk, crate::calib::Site::AttnOut, &ys, dims.value_dim, None);
    gemv_rows(&ys, &q4.w(&name("ssm_out.weight"))?, n, cfg.dim)
}

/// [`attn_layer`] for `rows` (one input row each), the projections shared.
/// A sequence's rows attend in order, each to the keys up to its own.
fn attn_rows(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    layer: usize,
    x: &[f32],
    rows: &[HostRow],
    mems: &mut [Qwen4Memory],
    tap: &crate::calib::Tap,
) -> Result<Vec<f32>, AllpacaError> {
    let (head_dim, n_heads, n_kv) = (cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads.max(1));
    let name = |s: &str| format!("blk.{blk}.{s}");
    let n = rows.len();
    let (q_width, kv_width) = (n_heads * head_dim, n_kv * head_dim);
    tap.observe_rows(blk, crate::calib::Site::AttnIn, x, cfg.dim, None);
    let q_full = gemv_rows(x, &q4.w(&name("attn_q.weight"))?, n, q_width * 2)?;
    let k_all = gemv_rows(x, &q4.w(&name("attn_k.weight"))?, n, kv_width)?;
    let v_all = gemv_rows(x, &q4.w(&name("attn_v.weight"))?, n, kv_width)?;
    let (q_norm, k_norm) = (q4.w(&name("attn_q_norm.weight"))?, q4.w(&name("attn_k_norm.weight"))?);
    let n_rot = if cfg.rope_dims > 0 { cfg.rope_dims } else { head_dim };
    let qsa = qsa_shape(q4, cfg, blk)?;
    // The indexer's keys for every row, its queries for the rows past its budget.
    let mut index_keys = Vec::new();
    let mut index_queries = HashMap::new();
    if let Some(shape) = qsa.as_ref() {
        index_keys = gemv_rows(x, &q4.w(&name("indexer.k_proj.weight"))?, n, shape.dim)?;
        let sparse: Vec<usize> = (0..n).filter(|i| shape.is_sparse(rows[*i].pos)).collect();
        if !sparse.is_empty() {
            let xs: Vec<f32> = sparse.iter().flat_map(|i| x[i * cfg.dim..(i + 1) * cfg.dim].iter().copied()).collect();
            let width = shape.heads * shape.dim;
            let iq = gemv_rows(&xs, &q4.w(&name("indexer.q_proj.weight"))?, sparse.len(), width)?;
            for (j, i) in sparse.iter().enumerate() {
                index_queries.insert(*i, iq[j * width..(j + 1) * width].to_vec());
            }
        }
    }
    let mut outs = vec![0.0f32; n * q_width];
    for (i, row) in rows.iter().enumerate() {
        let mut q = vec![0.0f32; q_width];
        let mut gate = vec![0.0f32; q_width];
        deinterleave_q_gate(&q_full[i * q_width * 2..(i + 1) * q_width * 2], &mut q, &mut gate, head_dim);
        rmsnorm_heads(&mut q, &q_norm, head_dim, cfg.eps);
        let mut k = k_all[i * kv_width..(i + 1) * kv_width].to_vec();
        rmsnorm_heads(&mut k, &k_norm, head_dim, cfg.eps);
        for head in q.chunks_mut(head_dim).chain(k.chunks_mut(head_dim)) {
            rope_imrope_neox(head, row.pos, cfg.rope_theta, n_rot, cfg.rope_sections)?;
        }
        let state = &mut mems[row.seq].layers[layer];
        if state.k.len() != row.pos * kv_width {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "qwen4exp layer {blk} holds {} keys, token is at position {}",
                state.k.len() / kv_width,
                row.pos
            ))));
        }
        state.k.extend_from_slice(&k);
        state.v.extend_from_slice(&v_all[i * kv_width..(i + 1) * kv_width]);
        let mut cells = None;
        if let Some(shape) = qsa.as_ref() {
            state.qsa.push(shape, &index_keys[i * shape.dim..(i + 1) * shape.dim], &q4.w(&name("indexer.k_norm.weight"))?)?;
            if let Some(mut iq) = index_queries.remove(&i) {
                cells = state.qsa.select(shape, row.pos, &mut iq, &q4.w(&name("indexer.q_norm.weight"))?)?;
            }
        }
        let out = &mut outs[i * q_width..(i + 1) * q_width];
        match cells {
            None => attention(&q, &state.k, &state.v, out, 1, row.pos + 1, n_heads, n_kv, head_dim, true)?,
            Some(cells) => {
                let gather = |all: &[f32]| -> Vec<f32> {
                    cells.iter().flat_map(|c| all[*c as usize * kv_width..(*c as usize + 1) * kv_width].iter().copied()).collect()
                };
                let (keys, values) = (gather(&state.k), gather(&state.v));
                attention(&q, &keys, &values, out, 1, cells.len(), n_heads, n_kv, head_dim, true)?;
            }
        }
        for (o, g) in out.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
    }
    tap.observe_rows(blk, crate::calib::Site::AttnOut, &outs, q_width, None);
    gemv_rows(&outs, &q4.w(&name("attn_output.weight"))?, n, cfg.dim)
}

/// [`ffn_layer`] for `rows` (one input row each): the router and shared
/// expert shared, each routed expert run once for the rows that picked it.
fn ffn_rows(
    q4: &Arc<Qwen4Weights>,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    rows: &[HostRow],
    mems: &mut [Qwen4Memory],
    tap: &crate::calib::Tap,
) -> Result<Vec<f32>, AllpacaError> {
    let (dim, n) = (cfg.dim, rows.len());
    if rows.iter().any(|r| mems[r.seq].saliency.is_some()) {
        // The saliency needs each expert's own output: one row at a time.
        let mut out = Vec::with_capacity(n * dim);
        for (row, x) in rows.iter().zip(x.chunks(dim)) {
            out.extend(ffn_layer(q4, cfg, blk, x, &mut mems[row.seq])?);
        }
        return Ok(out);
    }
    let name = |s: &str| format!("blk.{blk}.{s}");
    tap.observe_rows(blk, crate::calib::Site::FfnIn, x, dim, None);
    let mut scores = gemv_rows(x, &q4.w(&name("ffn_gate_inp.weight"))?, n, cfg.n_experts)?;
    let mut picks = Vec::with_capacity(n);
    for (row, scores) in rows.iter().zip(scores.chunks_mut(cfg.n_experts)) {
        let mem = &mut mems[row.seq];
        let row_picks = topk_from_logits(scores, cfg.n_active, cfg, block_mask(mem.expert_mask.as_ref(), blk, cfg.n_experts), &[]);
        if let Some(routes) = mem.routes.as_mut() {
            routes.extend(row_picks.iter().map(|(e, _)| (blk as u16, *e as u16)));
        }
        picks.push(row_picks);
    }
    let mut out = routed_experts_pooled(q4, cfg, blk, x, &picks, Some(tap))?;
    if q4.has(&name("ffn_up_shexp.weight")) {
        let gate = q4.w(&name("ffn_gate_shexp.weight"))?;
        let up = q4.w(&name("ffn_up_shexp.weight"))?;
        let down = q4.w(&name("ffn_down_shexp.weight"))?;
        let ff = up.len() / dim;
        let mut h = gemv_rows(x, &gate, n, ff)?;
        let u = gemv_rows(x, &up, n, ff)?;
        for (h, u) in h.iter_mut().zip(&u) {
            *h = silu(*h) * u;
        }
        let y = gemv_rows(&h, &down, n, dim)?;
        let gate_inp = q4.w(&name("ffn_gate_inp_shexp.weight"))?;
        let opens: Vec<f32> = x.chunks(dim).map(|x| sigmoid(x.iter().zip(gate_inp.iter()).map(|(a, b)| a * b).sum())).collect();
        if tap.wants() {
            let w2: Vec<f32> = opens.iter().map(|o| o * o).collect();
            tap.observe_rows(blk, crate::calib::Site::ShexpIn, x, dim, Some(&w2));
            tap.observe_rows(blk, crate::calib::Site::ShexpAct, &h, ff, Some(&w2));
        }
        for ((o, y), open) in out.chunks_mut(dim).zip(y.chunks(dim)).zip(&opens) {
            for (o, y) in o.iter_mut().zip(y) {
                *o += open * y;
            }
        }
    }
    Ok(out)
}

/// The host forward over several sequences at once, block by block: each
/// matrix is read once per block for all their tokens, and each routed
/// expert once for all the tokens that picked it. `seqs[i]` carries on
/// from `mems[i]`; `row_logits` gets a row of logits per token, in `seqs`
/// order (a row whose trace skips the head is left as it was). A
/// sequence's tokens still meet its recurrent states, keys and n-gram
/// conv in order, so every row equals [`forward_qwen4_tokens`]'s on the
/// host, bit for bit.
pub(crate) fn forward_qwen4_host_rows(
    model: &TinyModel,
    seqs: &[&[u32]],
    mems: &mut [Qwen4Memory],
    row_logits: &mut [f32],
) -> Result<(), AllpacaError> {
    let q4 = model.q4x.as_ref().ok_or_else(|| load_err("missing qwen4exp weights".into()))?;
    let cfg = &model.config;
    let (dim, hc, eps, vocab) = (cfg.dim, q4.spec.hc, cfg.eps, cfg.vocab_size);
    let hc_dim = hc * dim;
    let mut rows = Vec::new();
    for (seq, (tokens, mem)) in seqs.iter().zip(mems.iter()).enumerate() {
        let start = mem.tokens.len();
        rows.extend(tokens.iter().enumerate().map(|(i, &token)| HostRow { seq, pos: start + i, token }));
    }
    if seqs.len() != mems.len() || row_logits.len() < rows.len() * vocab {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "qwen4exp host rows: sequences, memories and logits do not line up".into(),
        )));
    }
    if q4.spec.ple.as_ref().is_some_and(|p| p.layers.len() > 1) {
        return Err(load_err("qwen4exp host rows: more than one n-gram layer".into()));
    }
    // Each sequence's tokens, earlier and new: the n-gram hash looks back.
    let all: Vec<Vec<i64>> =
        seqs.iter().zip(mems.iter()).map(|(t, m)| m.tokens.iter().copied().chain(t.iter().map(|t| i64::from(*t))).collect()).collect();
    for mem in mems.iter_mut() {
        mem.layers.resize_with(cfg.n_layers, LayerState::default);
    }
    let mut streams = Vec::with_capacity(rows.len() * hc_dim);
    for row in &rows {
        streams.extend(q4.rows("token_embd.weight", row.token as usize, 1, dim)?.repeat(hc));
    }
    // The rows whose trace lets block `layer` run (the head at the block count).
    let enter = |layer: usize, streams: &mut [f32], mems: &mut [Qwen4Memory]| -> Vec<usize> {
        (0..rows.len())
            .filter(|i| {
                let (row, x) = (rows[*i], &mut streams[i * hc_dim..(i + 1) * hc_dim]);
                !mems[row.seq].trace.as_mut().is_some_and(|t| t.enter(row.pos, layer, cfg.n_layers, x))
            })
            .collect()
    };
    for layer in 0..cfg.n_layers {
        let active = enter(layer, &mut streams, mems);
        if active.is_empty() {
            continue;
        }
        let blk = cfg.gguf_blk(layer);
        let act: Vec<HostRow> = active.iter().map(|i| rows[*i]).collect();
        let n = act.len();
        let mut s: Vec<f32> = active.iter().flat_map(|i| streams[i * hc_dim..(i + 1) * hc_dim].iter().copied()).collect();
        if let Some(ple) = q4.spec.ple.as_ref().filter(|p| p.layers.contains(&blk)) {
            let mut embs = Vec::new();
            for row in &act {
                embs.extend(ple_embedding(q4, ple, row.token, &all[row.seq][..row.pos])?);
            }
            let name = |s: &str| format!("blk.{blk}.ple_{s}.weight");
            let key = gemv_rows(&embs, &q4.w(&name("key"))?, n, hc_dim)?;
            let value = gemv_rows(&embs, &q4.w(&name("value"))?, n, dim)?;
            let (nk, nq, nc, conv) = (q4.w(&name("norm_key"))?, q4.w(&name("norm_query"))?, q4.w(&name("norm_conv"))?, q4.w(&name("conv1d"))?);
            for (i, row) in act.iter().enumerate() {
                ple_apply(
                    &mut s[i * hc_dim..(i + 1) * hc_dim],
                    &key[i * hc_dim..(i + 1) * hc_dim],
                    &value[i * dim..(i + 1) * dim],
                    [&nk, &nq, &nc],
                    &conv,
                    &mut mems[row.seq].ple_history,
                    ple,
                    dim,
                    eps,
                )?;
            }
        }
        let (x, inject) = mix_rows(q4, &format!("blk.{blk}.hc_attn"), true, &s, n, dim, eps)?;
        let out = match q4.spec.is_recurrent(layer) {
            true => gdn_rows(q4, cfg, blk, layer, &x, &act, mems, &model.calib)?,
            false => attn_rows(q4, cfg, blk, layer, &x, &act, mems, &model.calib)?,
        };
        for ((s, out), inject) in s.chunks_mut(hc_dim).zip(out.chunks(dim)).zip(inject.chunks(hc)) {
            hc_combine(s, out, inject, dim);
        }
        let (x, inject) = mix_rows(q4, &format!("blk.{blk}.hc_ffn"), true, &s, n, dim, eps)?;
        let out = ffn_rows(q4, cfg, blk, &x, &act, mems, &model.calib)?;
        for ((s, out), inject) in s.chunks_mut(hc_dim).zip(out.chunks(dim)).zip(inject.chunks(hc)) {
            hc_combine(s, out, inject, dim);
        }
        for (i, s) in active.iter().zip(s.chunks(hc_dim)) {
            streams[i * hc_dim..(i + 1) * hc_dim].copy_from_slice(s);
        }
    }
    let active = enter(cfg.n_layers, &mut streams, mems);
    if !active.is_empty() {
        let s: Vec<f32> = active.iter().flat_map(|i| streams[i * hc_dim..(i + 1) * hc_dim].iter().copied()).collect();
        let (x, _) = mix_rows(q4, "output_hc", false, &s, active.len(), dim, eps)?;
        let head = if q4.has("output.weight") { "output.weight" } else { "token_embd.weight" };
        let logits = gemv_rows(&x, &q4.w(head)?, active.len(), vocab)?;
        for (i, l) in active.iter().zip(logits.chunks(vocab)) {
            row_logits[i * vocab..(i + 1) * vocab].copy_from_slice(l);
        }
    }
    for (mem, tokens) in mems.iter_mut().zip(seqs) {
        mem.tokens.extend(tokens.iter().map(|t| i64::from(*t)));
    }
    Ok(())
}

/// FNV-1a over the little-endian bits of a pass's residual streams.
fn layer_trace_hash(streams: &[f32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in streams {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Append one line when `ALLPACA_Q4X_LAYER_TRACE` names a file. Off unless that
/// variable is set, so a normal forward does not sync or write.
fn layer_trace_append(path: &str, line: &str) -> Result<(), AllpacaError> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|err| load_err(format!("qwen4exp layer trace: {err}")))?;
    writeln!(file, "{line}").map_err(|err| load_err(format!("qwen4exp layer trace: {err}")))
}

static LOGIT_DUMP_REQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write one last-row logit vector as little-endian f32 plus a JSON sidecar
/// when `ALLPACA_Q4X_LOGIT_DUMP` names a directory. Off unless that variable
/// is set, so a normal forward does not write. `req` is the sequence counter
/// so three chats on one server do not overwrite each other.
fn dump_logits_row(
    dir: &str,
    logits: &[f32],
    pos: usize,
    rows: usize,
    req: u64,
) -> Result<(), AllpacaError> {
    std::fs::create_dir_all(dir).map_err(|err| load_err(format!("qwen4exp logit dump: {err}")))?;
    let stem = format!("req{req}_pos{pos}_rows{rows}");
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for value in logits {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(std::path::Path::new(dir).join(format!("{stem}.f32")), &bytes)
        .map_err(|err| load_err(format!("qwen4exp logit dump: {err}")))?;
    let (mut top1, mut best) = (0usize, f32::NEG_INFINITY);
    for (index, value) in logits.iter().enumerate() {
        if *value > best {
            best = *value;
            top1 = index;
        }
    }
    let meta = format!(
        "{{\"req\":{req},\"pos\":{pos},\"rows\":{rows},\"vocab\":{},\"top1\":{top1},\"top1_logit\":{best}}}\n",
        logits.len()
    );
    std::fs::write(std::path::Path::new(dir).join(format!("{stem}.json")), meta)
        .map_err(|err| load_err(format!("qwen4exp logit dump: {err}")))
}

#[cfg(test)]
#[test]
fn layer_trace_hash_matches_fnv_of_the_float_bits() {
    // 1.0 is 0x3f800000 and -0.5 is 0xbf000000, little-endian.
    assert_eq!(layer_trace_hash(&[1.0, -0.5]), 0x0979_d8ee_2da2_0b75);
}

#[cfg(test)]
#[test]
fn logit_dump_round_trips_little_endian_floats_and_records_the_top_token() {
    let dir = std::env::temp_dir().join(format!(
        "q20-logit-dump-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let logits = [0.1f32, 3.5, -2.0, 1.25];
    dump_logits_row(dir.to_str().unwrap(), &logits, 16, 3, 7).unwrap();
    let bytes = std::fs::read(dir.join("req7_pos16_rows3.f32")).unwrap();
    let mut got = [0.0f32; 4];
    for (slot, chunk) in got.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    assert_eq!(got, logits);
    let meta = std::fs::read_to_string(dir.join("req7_pos16_rows3.json")).unwrap();
    assert!(meta.contains("\"top1\":1"), "{meta}");
    assert!(meta.contains("\"vocab\":4"), "{meta}");
    assert!(meta.contains("\"req\":7"), "{meta}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// One GPU pass over `tokens` (at most [`crate::qwen4_gpu::MAX_ROWS`])
/// from `pos`: the streams of all of them stay on the card of the block
/// they are in. They come to the host where the cards change and for the
/// n-gram layer; the router's scores come back once per block. `logits`
/// (one row) gets the last token's, `row_logits` every token's.
///
/// `split` is for a caller that talks to the worker hosting blocks itself:
/// `Some(None)` stops where the hosted blocks begin and returns the pass
/// there, `Some(Some(pass))` goes on from it once the worker's result is
/// in `pass.streams`. `None` runs the hosted blocks in place.
#[allow(clippy::too_many_arguments)]
fn forward_device(
    q4: &Arc<Qwen4Weights>,
    gpu: &Qwen4Gpu,
    rpc: Option<&crate::rpc_runtime::RpcRuntime>,
    cfg: &ModelConfig,
    tokens: &[u32],
    pos: usize,
    logits: &mut [f32],
    row_logits: Option<&mut [f32]>,
    mem: &mut Qwen4Memory,
    split: Option<Option<DevicePass>>,
) -> Result<Option<DevicePass>, AllpacaError> {
    let (dim, hc, eps, rows) = (cfg.dim, q4.spec.hc, cfg.eps, tokens.len());
    let (vocab, n_experts) = (cfg.vocab_size, cfg.n_experts);
    let dims = q4.spec.gdn.ok_or_else(|| load_err("qwen4exp has no Gated DeltaNet shape".into()))?;
    if mem.device.is_none() {
        if let Some(host) = mem.device_host.take() {
            mem.device = Some(Qwen4DeviceState::from_host(&host)?);
        }
    }
    let device = mem.device.get_or_insert_with(Qwen4DeviceState::default);
    // A token's logits must not depend on how many share its pass.
    let _exact = allpaca_hip::ExactRows::enable();
    let mut times = mem.times.unwrap_or_default();
    // Kernels are queued: a part's time only shows where the host waits for
    // the card, unless `ALLPACA_Q4X_SYNC_TIMES` makes every part wait.
    let wait = mem.times.is_some() && std::env::var_os("ALLPACA_Q4X_SYNC_TIMES").is_some();
    let mut clock = std::time::Instant::now();
    let mut lap = |card: usize, slot: &mut f32| -> Result<(), AllpacaError> {
        if wait {
            gpu.sync(card)?;
        }
        *slot += clock.elapsed().as_secs_f32();
        clock = std::time::Instant::now();
        Ok(())
    };
    let stop_at_hosted = split.is_some();
    // A pass that goes on after the worker had its tokens added already.
    let resumed = matches!(split, Some(Some(_)));
    let (mut streams, mut card, mut resume) = match split.flatten() {
        Some(pass) => (pass.streams, pass.card, pass.resume),
        None => {
            let mut streams = Vec::with_capacity(rows * hc * dim);
            for &token in tokens {
                match mem.nextn_seed.as_deref() {
                    Some(hidden) => streams.extend(nextn_seed(q4, cfg, token, hidden)?),
                    None => {
                        let embed = q4.rows("token_embd.weight", token as usize, 1, dim)?;
                        for _ in 0..hc {
                            streams.extend_from_slice(&embed);
                        }
                    }
                }
            }
            (streams, gpu.card_of(cfg.gguf_blk(0)), 0)
        }
    };
    gpu.set_streams(card, &streams)?;
    // The n-gram block runs on its card unless `ALLPACA_Q4X_HOST_PLE`; its
    // table rows go there with the streams.
    let host_ple = std::env::var_os("ALLPACA_Q4X_HOST_PLE").is_some();
    let trace_path = std::env::var("ALLPACA_Q4X_LAYER_TRACE").ok().filter(|path| !path.is_empty());
    let trace_file = trace_path.clone();
    let logit_dump = std::env::var("ALLPACA_Q4X_LOGIT_DUMP")
        .ok()
        .filter(|path| !path.is_empty());
    if pos == 0 && logit_dump.is_some() {
        LOGIT_DUMP_REQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let trace = |blk: usize, layer: usize, span: usize, streams: &mut [f32]| -> Result<(), AllpacaError> {
        let Some(path) = trace_path.as_deref() else {
            return Ok(());
        };
        let hash = layer_trace_hash(streams);
        layer_trace_append(
            path,
            &format!("pos={pos} rows={rows} layer={layer} blk={blk} span={span} hash={hash:016x}"),
        )
    };
    if let Some(ple) = q4.spec.ple.as_ref().filter(|_| !host_ple) {
        let blk = ple.layers[0];
        if (resume..cfg.n_layers).any(|layer| cfg.gguf_blk(layer) == blk) {
            let mut earlier = mem.tokens.clone();
            let mut emb = Vec::with_capacity(rows * ple.n_heads() * ple.head_dim);
            for &token in tokens {
                emb.extend(ple_embedding(q4, ple, token, &earlier)?);
                earlier.push(i64::from(token));
            }
            gpu.set_ple_rows(gpu.card_of(blk), &emb)?;
        }
    }
    let mut scores = vec![0.0f32; rows * n_experts];
    let mut x = Vec::new();
    // Blocks with their experts on a card pick them there, so nothing of
    // the block comes back to the host. `ALLPACA_Q4X_HOST_ROUTING` and the
    // route recorder keep the picking on the host.
    let card_scale = crate::model::plain_softmax_routing(cfg)
        .filter(|_| {
            mem.routes.is_none() && mem.expert_mask.is_none() && std::env::var_os("ALLPACA_Q4X_HOST_ROUTING").is_none()
        });
    for layer in 0..cfg.n_layers {
        if layer < resume {
            continue;
        }
        let blk = cfg.gguf_blk(layer);
        // The blocks a worker hosts run there in one round trip.
        if let Some(host) = gpu.hosted().filter(|h| h.blocks().first() == Some(&blk)) {
            if mem.expert_mask.is_some() {
                return Err(load_err("qwen4exp: an expert mask does not reach blocks a worker hosts".into()));
            }
            gpu.get_streams(card, &mut streams)?;
            resume = layer + host.blocks().len();
            card = if resume < cfg.n_layers { gpu.card_of(cfg.gguf_blk(resume)) } else { 0 };
            let seq = mem.remote.get_or_insert_with(|| host.new_seq());
            if stop_at_hosted {
                if mem.times.is_some() {
                    mem.times = Some(times);
                }
                // The next pass may start before this one ends: its n-gram
                // embeddings must see these tokens.
                mem.tokens.extend(tokens.iter().map(|t| i64::from(*t)));
                return Ok(Some(DevicePass { streams, resume, card }));
            }
            let span = host.blocks().len();
            host.run(seq, pos, rows, &mut streams)?;
            trace(blk, layer, span, &mut streams)?;
            gpu.set_streams(card, &streams)?;
            lap(card, &mut times.ffn_rpc)?;
            continue;
        }
        let home = gpu.card_of(blk);
        let ple = q4.spec.ple.as_ref().filter(|p| p.layers.contains(&blk));
        if home != card || (ple.is_some() && host_ple) {
            gpu.get_streams(card, &mut streams)?;
            if let Some(ple) = ple.filter(|_| host_ple) {
                let name = |s: &str| format!("blk.{blk}.ple_{s}.weight");
                let (nk, nq, nc) = (q4.w(&name("norm_key"))?, q4.w(&name("norm_query"))?, q4.w(&name("norm_conv"))?);
                let conv = q4.w(&name("conv1d"))?;
                // Each token's n-gram reaches back over the ones before it in the pass.
                let mut earlier = mem.tokens.clone();
                for (&token, row) in tokens.iter().zip(streams.chunks_mut(hc * dim)) {
                    let emb = ple_embedding(q4, ple, token, &earlier)?;
                    let mut kv = gpu.products(&[name("key"), name("value")], &emb)?;
                    let value = kv.swap_remove(1);
                    ple_apply(row, &kv[0], &value, [&nk, &nq, &nc], &conv, &mut mem.ple_history, ple, dim, eps)?;
                    earlier.push(i64::from(token));
                }
            }
            gpu.set_streams(home, &streams)?;
            card = home;
        }
        if let Some(ple) = ple.filter(|_| !host_ple) {
            gpu.ple(blk, rows, ple.conv_kernel, ple.ngram_size, device, &mem.ple_history)?;
        }
        lap(card, &mut times.embed_ple)?;
        gpu.mix(card, &format!("blk.{blk}.hc_attn"), true, rows)?;
        if wait {
            lap(card, &mut times.mix)?;
        }
        let recurrent = q4.spec.is_recurrent(layer);
        if recurrent {
            gpu.gdn(blk, &dims, device, rows)?;
        } else {
            let shape = crate::qwen4_gpu::AttnShape::of(cfg, qsa_shape(q4, cfg, blk)?);
            gpu.attention(blk, &shape, pos, device, rows)?;
        }
        if wait {
            lap(card, if recurrent { &mut times.gdn } else { &mut times.attention })?;
        }
        gpu.combine(card, rows)?;
        gpu.mix(card, &format!("blk.{blk}.hc_ffn"), true, rows)?;
        if wait {
            lap(card, &mut times.mix)?;
        }
        if let Some(scale) = card_scale.filter(|_| gpu.routes_on_card(blk, cfg.n_active)) {
            gpu.ffn_route(blk, rows, None, &mut x)?;
            if wait {
                lap(card, &mut times.route)?;
            }
            gpu.ffn_finish_routed(blk, rows, cfg.n_active, scale)?;
            lap(card, &mut times.ffn_gpu)?;
            gpu.combine(card, rows)?;
            if trace_file.is_some() {
                gpu.get_streams(card, &mut streams)?;
                trace(blk, layer, 1, &mut streams)?;
            }
            continue;
        }
        gpu.ffn_route(blk, rows, Some(&mut scores), &mut x)?;
        lap(card, if wait { &mut times.route } else if recurrent { &mut times.gdn } else { &mut times.attention })?;
        let picks: Vec<Vec<(usize, f32)>> = scores
            .chunks_mut(n_experts.max(1))
            .map(|row| topk_from_logits(row, cfg.n_active, cfg, block_mask(mem.expert_mask.as_ref(), blk, n_experts), &[]))
            .collect();
        if let Some(routes) = mem.routes.as_mut() {
            routes.extend(picks.iter().flatten().map(|(e, _)| (blk as u16, *e as u16)));
        }
        if gpu.has_experts(blk) {
            gpu.ffn_finish(blk, &picks, None)?;
            lap(card, &mut times.ffn_gpu)?;
        } else if let Some(rpc) = rpc.filter(|_| gpu.remote_experts(blk)) {
            let sums = rpc
                .moe_ffn(layer, &x, &picks)?
                .ok_or_else(|| load_err(format!("qwen4exp: block {blk}'s experts are not on a worker")))?;
            gpu.ffn_finish(blk, &picks, Some(&sums))?;
            lap(card, &mut times.ffn_rpc)?;
        } else {
            let sums = routed_experts_pooled(q4, cfg, blk, &x, &picks, None)?;
            gpu.ffn_finish(blk, &picks, Some(&sums))?;
            lap(card, &mut times.ffn_host)?;
        }
        gpu.combine(card, rows)?;
        if trace_file.is_some() {
            gpu.get_streams(card, &mut streams)?;
            trace(blk, layer, 1, &mut streams)?;
        }
    }
    if card != 0 {
        gpu.get_streams(card, &mut streams)?;
        gpu.set_streams(0, &streams)?;
    }
    // The output mixer stands in for the final norm.
    mem.row_hiddens = Some(gpu.hidden_rows(rows)?);
    if mem.nextn_seed.is_some() {
        mem.nextn_seed = mem.row_hiddens.clone();
    }
    let mixer = if mem.nextn_seed.is_some() {
        format!("blk.{}.nextn.hc_head", cfg.gguf_blk(0))
    } else {
        "output_hc".into()
    };
    gpu.mix(0, &mixer, false, rows)?;
    let head = if q4.has("output.weight") { "output.weight" } else { "token_embd.weight" };
    let q4_draft = mem.nextn_seed.is_some() && gpu.has_q4_0_draft_head();
    match row_logits {
        Some(all) => {
            if q4_draft {
                gpu.head_q4_0(rows, &mut all[..rows * vocab])?;
            } else {
                gpu.head(head, rows, &mut all[..rows * vocab])?;
            }
            logits[..vocab].copy_from_slice(&all[(rows - 1) * vocab..rows * vocab]);
        }
        None if q4_draft => gpu.head_q4_0(rows, &mut logits[..vocab])?,
        None => gpu.head(head, rows, &mut logits[..vocab])?,
    }
    lap(0, &mut times.head)?;
    if let Some(path) = trace_file.as_deref() {
        let (mut best_i, mut best) = (0usize, f32::NEG_INFINITY);
        for (index, value) in logits[..vocab].iter().enumerate() {
            if *value > best {
                best = *value;
                best_i = index;
            }
        }
        layer_trace_append(path, &format!("pos={pos} rows={rows} top1={best_i}"))?;
    }
    if let Some(dir) = logit_dump.as_deref() {
        dump_logits_row(
            dir,
            &logits[..vocab],
            pos,
            rows,
            LOGIT_DUMP_REQ.load(std::sync::atomic::Ordering::Relaxed),
        )?;
    }
    if mem.times.is_some() {
        mem.times = Some(times);
    }
    if !resumed {
        mem.tokens.extend(tokens.iter().map(|t| i64::from(*t)));
    }
    Ok(None)
}

/// Tiny `qwen4exp` GGUF for tests: four layers (three Gated DeltaNet, one
/// full attention), two residual streams, n-gram embeddings at layer 1,
/// routed and shared experts. Weights are seeded noise, all F32.
#[cfg(test)]
pub fn write_tiny_qwen4exp_gguf(path: &std::path::Path) {
    write_tiny(path, 16, 8, false, None);
}

/// The tiny model with an indexer on its full-attention layer: pools of
/// two cells, of which the attention keeps `top_k / 2` and the tail.
#[cfg(test)]
pub fn write_tiny_qwen4exp_sparse_gguf(path: &std::path::Path, top_k: usize) {
    write_tiny(path, 16, 8, false, Some(top_k));
}

/// The tiny model at width 64 with packed tensors: Q2_0 routed experts and
/// two Q2_0 matrices, two BF16 mixer matrices and an F16 output head.
#[cfg(test)]
pub fn write_tiny_qwen4exp_packed_gguf(path: &std::path::Path) {
    write_tiny(path, 64, 64, true, None);
}

/// The packed tiny model with an indexer that keeps `top_k` cells.
#[cfg(test)]
pub fn write_tiny_qwen4exp_packed_sparse_gguf(path: &std::path::Path, top_k: usize) {
    write_tiny(path, 64, 64, true, Some(top_k));
}

/// The encoding a tensor of the packed fixture is stored in.
#[cfg(test)]
fn packed_type(name: &str) -> Option<allpaca_gguf::GgmlType> {
    use allpaca_gguf::GgmlType;
    let ends = |suffixes: &[&str]| suffixes.iter().any(|s| name.ends_with(s));
    if ends(&["_exps.weight", "attn_gate.weight", "ffn_gate_shexp.weight"]) {
        Some(GgmlType::Q2_0)
    } else if ends(&["hc_attn_down.weight", "hc_ffn_up.weight"]) {
        Some(GgmlType::BF16)
    } else if name == "output.weight" {
        Some(GgmlType::F16)
    } else {
        None
    }
}

#[cfg(test)]
fn write_tiny(path: &std::path::Path, dim: usize, ff: usize, packed: bool, indexer: Option<usize>) {
    write_tiny_ple(path, dim, ff, packed, indexer, 4);
}

/// The tiny model with n-gram table rows `ple_dim` wide (32 fits IQ4_NL blocks).
#[cfg(test)]
pub fn write_tiny_qwen4exp_wide_ple_gguf(path: &std::path::Path) {
    write_tiny_ple(path, 16, 8, false, None, 32);
}

#[cfg(test)]
fn write_tiny_ple(path: &std::path::Path, dim: usize, ff: usize, packed: bool, indexer: Option<usize>, ple_dim: usize) {
    use allpaca_gguf::{write_gguf_quant, GgmlType, GgufWriteQuantTensor, GgufWriteTensor};

    let (hc, lr, vocab, n_layers) = (2usize, 4usize, 8usize, 4usize);
    let (n_heads, head_dim, n_rot) = (2usize, 8usize, 4usize);
    let (state, n_k, n_v, conv_kernel) = (4usize, 1usize, 2usize, 2usize);
    let (key_dim, value_dim) = (state * n_k, state * n_v);
    let qkv_dim = key_dim * 2 + value_dim;
    let n_exp = 4usize;
    let (ngram, heads_per, ple_kernel) = (3usize, 2usize, 2usize);
    let ple_heads = (ngram - 1) * heads_per;
    let ple_vocab = 7usize;
    let hc_dim = hc * dim;

    let mut meta = HashMap::new();
    let u = |v: usize| GgufValue::Uint32(v as u32);
    let u64s = |v: Vec<u64>| GgufValue::Array(v.into_iter().map(GgufValue::Uint64).collect());
    for (key, value) in [
        ("block_count", u(n_layers)),
        ("context_length", u(256)),
        ("embedding_length", u(dim)),
        ("attention.head_count", u(n_heads)),
        ("attention.head_count_kv", u(1)),
        ("attention.key_length", u(head_dim)),
        ("attention.value_length", u(head_dim)),
        ("attention.layer_norm_rms_epsilon", GgufValue::Float32(1e-6)),
        ("rope.dimension_count", u(n_rot)),
        ("rope.freq_base", GgufValue::Float32(10000.0)),
        ("expert_count", u(n_exp)),
        ("expert_used_count", u(2)),
        ("expert_feed_forward_length", u(ff)),
        ("expert_shared_feed_forward_length", u(ff)),
        ("ssm.conv_kernel", u(conv_kernel)),
        ("ssm.state_size", u(state)),
        ("ssm.group_count", u(n_k)),
        ("ssm.time_step_rank", u(n_v)),
        ("ssm.inner_size", u(value_dim)),
        ("full_attention_interval", u(4)),
        ("hyper_connection.count", u(hc)),
        ("hyper_connection.low_rank", u(lr)),
        ("ple.layers", GgufValue::Array(vec![u(1)])),
        ("ple.ngram_size", u(ngram)),
        ("ple.heads_per_ngram", u(heads_per)),
        ("ple.conv_kernel", u(ple_kernel)),
        ("ple.eos_token_id", u(2)),
        ("embedding_length_per_layer_input", u(ple_dim)),
        ("ple.layer_multipliers", u64s(vec![23703573157769, 20109073645365, 8052911324071])),
        ("ple.head_offsets", u64s((0..ple_heads).map(|h| (h * ple_vocab) as u64).collect())),
        ("ple.head_vocab_sizes", u64s(vec![ple_vocab as u64; ple_heads])),
    ] {
        meta.insert(format!("qwen4exp.{key}"), value);
    }
    let (idx_heads, idx_dim, idx_ratio) = (2usize, 4usize, 2usize);
    if let Some(top_k) = indexer {
        meta.insert("qwen4exp.attention.indexer.top_k".into(), u(top_k));
        meta.insert("qwen4exp.attention.indexer.head_count".into(), u(idx_heads));
        meta.insert("qwen4exp.attention.indexer.key_length".into(), u(idx_dim));
        let ratios = (0..n_layers).map(|l| if (l + 1) % 4 == 0 { idx_ratio as u64 } else { 0 }).collect();
        meta.insert("qwen4exp.attention.compress_ratios".into(), u64s(ratios));
    }
    meta.insert("general.architecture".into(), GgufValue::String("qwen4exp".into()));
    meta.insert("general.name".into(), GgufValue::String("tiny-qwen4exp".into()));
    meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(
            ["<unk>", "<s>", "</s>", "hello", "a", "b", "c", "d"]
                .into_iter()
                .map(|t| GgufValue::String(t.into()))
                .collect(),
        ),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let mut seed = 0x1234_5678u32;
    let mut noise = |n: usize, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    };
    // Dims are listed outermost first here and stored innermost first, as
    // GGUF files have them.
    let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, dims: &[usize], data: Vec<f32>| {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "{name}");
        owned.push((name, dims.iter().rev().map(|d| *d as u64).collect(), data));
    };
    add("token_embd.weight".into(), &[vocab, dim], noise(vocab * dim, 1.0));
    add("output.weight".into(), &[vocab, dim], noise(vocab * dim, 0.5));
    add(
        "per_layer_token_embd.weight".into(),
        &[ple_heads * ple_vocab, ple_dim],
        noise(ple_heads * ple_vocab * ple_dim, 1.0),
    );
    add("output_hc_norm.weight".into(), &[hc_dim], vec![1.0; hc_dim]);
    add("output_hc_down.weight".into(), &[lr, hc_dim], noise(lr * hc_dim, 0.3));
    add("output_hc_up.weight".into(), &[hc_dim, lr], noise(hc_dim * lr, 0.3));
    add("blk.0.nextn.enorm.weight".into(), &[dim], vec![1.0; dim]);
    add("blk.0.nextn.hnorm.weight".into(), &[hc_dim], vec![1.0; hc_dim]);
    add("blk.0.nextn.eh_proj.weight".into(), &[dim, 2 * dim], noise(dim * 2 * dim, 0.3));
    add("blk.0.nextn.hc_head_norm.weight".into(), &[hc_dim], vec![1.0; hc_dim]);
    add("blk.0.nextn.hc_head_down.weight".into(), &[lr, hc_dim], noise(lr * hc_dim, 0.3));
    add("blk.0.nextn.hc_head_up.weight".into(), &[hc_dim, lr], noise(hc_dim * lr, 0.3));
    for blk in 0..n_layers {
        let mut t = |name: &str, dims: &[usize], data: Vec<f32>| add(format!("blk.{blk}.{name}"), dims, data);
        for kind in ["attn", "ffn"] {
            t(&format!("hc_{kind}_norm.weight"), &[hc_dim], vec![1.0; hc_dim]);
            t(&format!("hc_{kind}_down.weight"), &[lr, hc_dim], noise(lr * hc_dim, 0.3));
            t(&format!("hc_{kind}_up.weight"), &[hc_dim, lr], noise(hc_dim * lr, 0.3));
            t(&format!("hc_{kind}_inject.weight"), &[hc, hc_dim], noise(hc * hc_dim, 0.3));
        }
        if (blk + 1) % 4 == 0 {
            t("attn_q.weight", &[n_heads * head_dim * 2, dim], noise(n_heads * head_dim * 2 * dim, 0.3));
            t("attn_k.weight", &[head_dim, dim], noise(head_dim * dim, 0.3));
            t("attn_v.weight", &[head_dim, dim], noise(head_dim * dim, 0.3));
            t("attn_output.weight", &[dim, n_heads * head_dim], noise(dim * n_heads * head_dim, 0.3));
            t("attn_q_norm.weight", &[head_dim], vec![1.0; head_dim]);
            t("attn_k_norm.weight", &[head_dim], vec![1.0; head_dim]);
        } else {
            t("attn_qkv.weight", &[qkv_dim, dim], noise(qkv_dim * dim, 0.3));
            t("attn_gate.weight", &[value_dim, dim], noise(value_dim * dim, 0.3));
            t("ssm_conv1d.weight", &[qkv_dim, conv_kernel], noise(qkv_dim * conv_kernel, 0.5));
            t("ssm_dt.bias", &[n_v], noise(n_v, 0.5));
            t("ssm_a", &[n_v], vec![-0.5; n_v]);
            t("ssm_beta.weight", &[n_v, dim], noise(n_v * dim, 0.3));
            t("ssm_alpha.weight", &[n_v, dim], noise(n_v * dim, 0.3));
            t("ssm_norm.weight", &[state], vec![1.0; state]);
            t("ssm_out.weight", &[dim, value_dim], noise(dim * value_dim, 0.3));
        }
        if blk == 1 {
            t("ple_key.weight", &[hc_dim, ple_heads * ple_dim], noise(hc_dim * ple_heads * ple_dim, 0.3));
            t("ple_value.weight", &[dim, ple_heads * ple_dim], noise(dim * ple_heads * ple_dim, 0.3));
            for norm in ["key", "query", "conv"] {
                t(&format!("ple_norm_{norm}.weight"), &[hc_dim], vec![1.0; hc_dim]);
            }
            t("ple_conv1d.weight", &[hc_dim, ple_kernel], noise(hc_dim * ple_kernel, 0.5));
        }
        t("ffn_gate_inp.weight", &[n_exp, dim], noise(n_exp * dim, 0.5));
        t("ffn_gate_exps.weight", &[n_exp, ff, dim], noise(n_exp * ff * dim, 0.3));
        t("ffn_up_exps.weight", &[n_exp, ff, dim], noise(n_exp * ff * dim, 0.3));
        t("ffn_down_exps.weight", &[n_exp, dim, ff], noise(n_exp * dim * ff, 0.3));
        t("ffn_gate_inp_shexp.weight", &[dim], noise(dim, 0.5));
        t("ffn_gate_shexp.weight", &[ff, dim], noise(ff * dim, 0.3));
        t("ffn_up_shexp.weight", &[ff, dim], noise(ff * dim, 0.3));
        t("ffn_down_shexp.weight", &[dim, ff], noise(dim * ff, 0.3));
    }
    // Last, so that the other weights are the same with and without.
    if indexer.is_some() {
        for blk in (0..n_layers).filter(|b| (b + 1) % 4 == 0) {
            let mut t = |name: &str, dims: &[usize], data: Vec<f32>| add(format!("blk.{blk}.indexer.{name}"), dims, data);
            t("q_proj.weight", &[idx_heads * idx_dim, dim], noise(idx_heads * idx_dim * dim, 0.5));
            t("k_proj.weight", &[idx_dim, dim], noise(idx_dim * dim, 0.5));
            t("q_norm.weight", &[idx_dim], vec![1.0; idx_dim]);
            t("k_norm.weight", &[idx_dim], vec![1.0; idx_dim]);
        }
    }
    let encoded: Vec<Option<(GgmlType, Vec<u8>)>> = owned
        .iter()
        .map(|(name, _, data)| {
            let ty = packed_type(name).filter(|_| packed)?;
            let bytes = match ty {
                GgmlType::Q2_0 => allpaca_cpu::quantize_q2_0(data),
                GgmlType::BF16 => data.iter().flat_map(|v| allpaca_cpu::f32_to_bf16(*v).to_le_bytes()).collect(),
                _ => data.iter().flat_map(|v| allpaca_cpu::f32_to_f16(*v).to_le_bytes()).collect(),
            };
            Some((ty, bytes))
        })
        .collect();
    let (mut plain, mut quant) = (Vec::new(), Vec::new());
    for ((name, dims, data), encoded) in owned.iter().zip(&encoded) {
        match encoded {
            Some((ty, bytes)) => quant.push(GgufWriteQuantTensor {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: *ty,
                bytes,
            }),
            None => plain.push(GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data,
            }),
        }
    }
    write_gguf_quant(path, &meta, &plain, &quant).expect("write tiny qwen4exp");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ple() -> PleSpec {
        PleSpec {
            layers: vec![1],
            ngram_size: 3,
            heads_per_ngram: 2,
            conv_kernel: 4,
            eos_token: 9,
            head_dim: 4,
            multipliers: vec![7, 11, 13],
            head_offsets: vec![0, 100, 200, 300],
            head_vocab_sizes: vec![97, 89, 83, 79],
        }
    }

    #[test]
    fn ple_rows_hash_the_bigram_and_trigram_ending_at_the_token() {
        let spec = ple();
        assert_eq!(spec.n_heads(), 4);
        assert_eq!(spec.conv_history(), 9);
        // Bigram heads hash (token, previous); trigram heads add the one before.
        let bi = (5u64 * 7) ^ (4 * 11);
        let tri = bi ^ (3 * 13);
        assert_eq!(
            spec.rows(5, &[4, 3]),
            vec![bi % 97, bi % 89 + 100, tri % 83 + 200, tri % 79 + 300]
        );
        // Before the sequence start the missing tokens read as EOS.
        let bi0 = (5u64 * 7) ^ (9 * 11);
        let tri0 = bi0 ^ (9 * 13);
        assert_eq!(spec.rows(5, &[]), vec![bi0 % 97, bi0 % 89 + 100, tri0 % 83 + 200, tri0 % 79 + 300]);
        // An EOS in the window hides what came before it...
        let tri_cut = (5u64 * 7) ^ (9 * 11) ^ (9 * 13);
        assert_eq!(spec.rows(5, &[9, 3])[2], tri_cut % 83 + 200);
        // ...but a token that is itself EOS keeps its context.
        let bi_eos = (9u64 * 7) ^ (4 * 11);
        assert_eq!(spec.rows(9, &[4, 3])[0], bi_eos % 97);
    }

    #[test]
    fn hc_mix_with_zero_mixer_weights_is_half_the_mean_of_the_normed_streams() {
        let (hc, dim, lr) = (2usize, 4usize, 3usize);
        // Stream 0 has RMS 2, stream 1 RMS 1; gains 1 and 3.
        let x = [2.0f32, -2.0, 2.0, -2.0, 1.0, 1.0, -1.0, 1.0];
        let norm = [1.0f32, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0];
        let (down, up, inject) = (vec![0.0f32; lr * hc * dim], vec![0.0f32; hc * dim * lr], vec![0.0f32; hc * hc * dim]);
        let w = HcWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: Some(&inject),
        };
        let (mixed, logits) = hc_mix(&x, &w, hc, dim, 0.0).unwrap();
        // Normed streams: (1,-1,1,-1) and (3,3,-3,3); gate sigmoid(0) = 0.5; mean over 2.
        assert_eq!(mixed, vec![1.0, 0.5, -0.5, 0.5]);
        assert_eq!(logits, vec![0.0, 0.0]);
        // The output head has no injection.
        let head = HcWeights { inject: None, ..w };
        assert!(hc_mix(&x, &head, hc, dim, 0.0).unwrap().1.is_empty());
    }

    #[test]
    fn hc_mix_gates_each_stream_through_the_low_rank_path() {
        let (hc, dim) = (2usize, 2usize);
        let x = [1.0f32, 1.0, 1.0, -1.0];
        let norm = [1.0f32; 4];
        // Rank 1: lo = silu(sum of stream 0 / hc) = silu(1); the gate opens
        // dim 0 of stream 1 by 10 * lo and leaves the rest at zero.
        let down = [1.0f32, 1.0, 0.0, 0.0];
        let up = [0.0f32, 0.0, 10.0, 0.0];
        let inject = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0];
        let w = HcWeights {
            norm: &norm,
            down: &down,
            up: &up,
            inject: Some(&inject),
        };
        let (mixed, logits) = hc_mix(&x, &w, hc, dim, 0.0).unwrap();
        let open = sigmoid(10.0 * silu(1.0));
        assert!((mixed[0] - (0.5 + open) / 2.0).abs() < 1e-6);
        assert!((mixed[1] - (0.5 - 0.5) / 2.0).abs() < 1e-6);
        // Injection logits read the normed streams: x[0] and 2 * x[3].
        assert_eq!(logits, vec![1.0, -2.0]);
    }

    #[test]
    fn hc_combine_adds_the_block_output_weighted_per_stream() {
        let mut streams = [1.0f32, 2.0, 3.0, 4.0];
        // A zero logit is a plain residual add; a large one doubles it.
        hc_combine(&mut streams, &[10.0, 20.0], &[0.0, 200.0], 2);
        assert_eq!(streams, [11.0, 22.0, 23.0, 44.0]);
    }

    fn tiny_model() -> TinyModel {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_qwen4exp-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        write_tiny_qwen4exp_gguf(&path);
        let file = Arc::new(GgufFile::open(&path).expect("open tiny qwen4exp"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let _ = std::fs::remove_file(&path);
        model
    }

    /// The tiny model whose attention keeps `top_k` cells through its indexer.
    fn tiny_sparse_model(top_k: usize) -> TinyModel {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_qwen4exp_sparse-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        write_tiny_qwen4exp_sparse_gguf(&path, top_k);
        let file = Arc::new(GgufFile::open(&path).expect("open sparse tiny qwen4exp"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let _ = std::fs::remove_file(&path);
        model
    }

    #[cfg(feature = "hip")]
    fn tiny_packed_sparse_model(top_k: usize) -> TinyModel {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_qwen4exp_packed_sparse-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        write_tiny_qwen4exp_packed_sparse_gguf(&path, top_k);
        let file = Arc::new(GgufFile::open(&path).expect("open packed sparse tiny qwen4exp"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let _ = std::fs::remove_file(&path);
        model
    }

    /// Tokens of the tiny vocabulary without the end-of-sequence token.
    fn long_tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| [3u32, 4, 5, 6, 7, 4, 6][(i * i + i / 3) % 7]).collect()
    }

    #[cfg(feature = "hip")]
    fn tiny_packed_model() -> TinyModel {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_qwen4exp_packed-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        write_tiny_qwen4exp_packed_gguf(&path);
        let file = Arc::new(GgufFile::open(&path).expect("open packed tiny qwen4exp"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let _ = std::fs::remove_file(&path);
        model
    }

    /// With its matrices and routed experts on the GPU, the forward gives
    /// the host forward's logits: for the F32 model (experts stay on the
    /// host) and for the packed one (Q2_0 experts in GPU slots).
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_on_the_gpu_matches_the_host_forward() {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let tokens = [1u32, 3, 4, 5, 3, 6, 7, 4, 4, 5];
        type Build = fn() -> TinyModel;
        for (what, build, expert_layers) in [("f32", tiny_model as Build, 0usize), ("packed", tiny_packed_model as Build, 4)] {
            let (host, gpu) = (build(), build());
            crate::qwen4_gpu::attach(&gpu, &hip, None, None, None).expect("attach");
            {
                let attached = gpu.q4x.as_ref().unwrap().gpu.lock().unwrap();
                let attached = attached.as_ref().expect("gpu attached");
                assert_eq!(attached.expert_layers()[0].len(), expert_layers, "{what}: expert layers on the card");
                assert!(attached.has("blk.0.hc_attn_down.weight") && attached.has("output.weight"), "{what}");
            }
            let (want, got) = (run(&host, &tokens), run(&gpu, &tokens));
            for (pos, (want, got)) in want.iter().zip(&got).enumerate() {
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(scale > 1e-3, "{what}: logits at {pos} are all zero");
                assert!(
                    max_diff(want, got) <= 2e-3 * scale.max(1.0),
                    "{what}: logits at {pos} differ by {} (scale {scale})",
                    max_diff(want, got)
                );
            }
        }
    }

    /// On the GPU a run of tokens goes through in passes of several: every
    /// token's logits must be what one token at a time gives, across a pass
    /// boundary and with a run that starts mid-sequence.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_gpu_passes_of_several_tokens_match_single_tokens() {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let tokens: Vec<u32> = (0..23u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();
        let model = tiny_packed_model();
        crate::qwen4_gpu::attach(&model, &hip, None, None, None).expect("attach");
        let single = run(&model, &tokens);
        let vocab = model.config.vocab_size;
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; vocab];
        forward_qwen4_tokens(&model, &tokens[..3], 0, &mut logits, None, &mut mem).expect("first run");
        assert_eq!(logits, single[2], "last logits of the first run");
        // Twenty tokens: one full pass and a partial one.
        let mut rows = vec![f32::NAN; 20 * vocab];
        forward_qwen4_tokens(&model, &tokens[3..], 3, &mut logits, Some(&mut rows), &mut mem).expect("second run");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, single[3 + i], "row {i}");
        }
        assert_eq!(logits, rows[19 * vocab..]);
        assert_eq!(mem.tokens(), 23);

        // Two blocks' experts on the card, two on the host workers (8-bit
        // activations): passes of several still match single tokens, and
        // the whole stays near the all-on-card logits.
        let mixed = tiny_packed_model();
        crate::qwen4_gpu::attach(&mixed, &hip, None, Some(&[2]), None).expect("attach");
        assert_eq!(mixed.q4x.as_ref().unwrap().gpu.lock().unwrap().as_ref().unwrap().expert_layers()[0], [0, 1]);
        let mixed_single = run(&mixed, &tokens);
        let mut mem = Qwen4Memory::default();
        let mut rows = vec![f32::NAN; 23 * vocab];
        forward_qwen4_tokens(&mixed, &tokens, 0, &mut logits, Some(&mut rows), &mut mem).expect("mixed run");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, mixed_single[i], "mixed row {i}");
            let scale = single[i].iter().fold(1.0f32, |m, v| m.max(v.abs()));
            assert!(max_diff(row, &single[i]) <= 0.05 * scale, "mixed row {i} is {} from the card's", max_diff(row, &single[i]));
        }
    }

    /// Off-card host experts (`gemm_q2_0_i8`) must match the same experts
    /// on the card when both paths pick with the host router. The 5% bound
    /// in `qwen4exp_gpu_passes_of_several_tokens_match_single_tokens` still
    /// includes GPU `expf` routing on the all-card side; a `routes`
    /// recorder keeps picking on the host, so only the Q2_0 products differ.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_host_routed_experts_off_the_card_match_the_same_picks_on_the_card() {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let tokens: Vec<u32> = (0..23u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();

        let on_card = tiny_packed_model();
        crate::qwen4_gpu::attach(&on_card, &hip, None, None, None).expect("attach");
        let mixed = tiny_packed_model();
        crate::qwen4_gpu::attach(&mixed, &hip, None, Some(&[2]), None).expect("attach");
        assert_eq!(mixed.q4x.as_ref().unwrap().gpu.lock().unwrap().as_ref().unwrap().expert_layers()[0], [0, 1]);

        let run_host_picks = |model: &TinyModel| {
            let mut mem = Qwen4Memory { routes: Some(Vec::new()), ..Default::default() };
            let mut logits = vec![0.0f32; model.config.vocab_size];
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &t)| {
                    forward_qwen4_token(model, t, pos, &mut logits, &mut mem).expect("forward");
                    logits.clone()
                })
                .collect::<Vec<_>>()
        };
        let want = run_host_picks(&on_card);
        let got = run_host_picks(&mixed);
        let top = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
        for (pos, (want, got)) in want.iter().zip(&got).enumerate() {
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(scale > 1e-3, "logits at {pos} are all zero");
            let diff = max_diff(want, got);
            assert!(
                diff <= 2e-3 * scale.max(1.0),
                "host-routed off-card experts at {pos} differ by {diff} (scale {scale})"
            );
            assert_eq!(top(want), top(got), "top-1 at {pos}");
        }
    }

    /// Past the indexer's budget the GPU forward keeps the cells the host
    /// forward keeps: its logits stay the host's, a pass of several tokens
    /// (some before the budget, some past it) gives what one token at a
    /// time gives, and a copied sequence goes on from its own keys.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_sparse_attention_on_the_gpu_matches_the_host_forward() {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let tokens = long_tokens(40);
        type Build = fn(usize) -> TinyModel;
        for (what, build) in [("f32", tiny_sparse_model as Build), ("packed", tiny_packed_sparse_model as Build)] {
            let (host, gpu, dense) = (build(8), build(8), build(64));
            crate::qwen4_gpu::attach(&gpu, &hip, None, None, None).expect("attach");
            crate::qwen4_gpu::attach(&dense, &hip, None, None, None).expect("attach");
            let (want, got, all) = (run(&host, &tokens), run(&gpu, &tokens), run(&dense, &tokens));
            assert_eq!(got[..9], all[..9], "{what}: every cell is kept up to rank 8");
            for (pos, (want, got)) in want.iter().zip(&got).enumerate() {
                let scale = want.iter().fold(1.0f32, |m, v| m.max(v.abs()));
                assert!(
                    max_diff(want, got) <= 2e-3 * scale,
                    "{what}: logits at {pos} differ by {} (scale {scale})",
                    max_diff(want, got)
                );
                if pos >= 9 {
                    assert!(max_diff(got, &all[pos]) > 1e-6, "{what}: rank {pos} must drop cells");
                }
            }
            let vocab = gpu.config.vocab_size;
            let (mut mem, mut logits) = (Qwen4Memory::default(), vec![0.0f32; vocab]);
            let mut rows = vec![f32::NAN; 37 * vocab];
            forward_qwen4_tokens(&gpu, &tokens[..3], 0, &mut logits, None, &mut mem).expect("prompt");
            forward_qwen4_tokens(&gpu, &tokens[3..], 3, &mut logits, Some(&mut rows), &mut mem).expect("rows");
            for (i, row) in rows.chunks(vocab).enumerate() {
                assert_eq!(row, got[3 + i], "{what}: row {i}");
            }
            // A copy taken past the budget continues like the original.
            let mut mem = Qwen4Memory::default();
            forward_qwen4_tokens(&gpu, &tokens[..20], 0, &mut logits, None, &mut mem).expect("first half");
            let mut fork = mem.clone();
            forward_qwen4_tokens(&gpu, &[3, 3, 3, 3], 20, &mut logits, None, &mut mem).expect("other tokens");
            for (pos, &t) in tokens.iter().enumerate().skip(20) {
                forward_qwen4_token(&gpu, t, pos, &mut logits, &mut fork).expect("fork");
                assert_eq!(logits, got[pos], "{what}: fork at {pos}");
            }
        }
    }

    /// A worker that hosts a full-attention block keeps its indexer keys
    /// and selects there: the logits are those of every block on the card.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_sparse_attention_runs_on_the_worker_that_hosts_the_block() {
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
            serve_connection_ext(stream, backend, None, vec![crate::qwen4_remote::worker_extension(None)]).unwrap();
        });
        let tokens = long_tokens(40);
        let local = tiny_packed_sparse_model(8);
        crate::qwen4_gpu::attach(&local, &hip, None, None, None).expect("attach");
        let on_card = run(&local, &tokens);

        let mut remote = tiny_packed_sparse_model(8);
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2]), None).expect("attach");
        assert_eq!(remote.q4x.as_ref().unwrap().gpu.lock().unwrap().as_ref().unwrap().hosted_blocks(), [2, 3]);
        let vocab = remote.config.vocab_size;
        let (mut mem, mut logits) = (Qwen4Memory::default(), vec![0.0f32; vocab]);
        let mut rows = vec![f32::NAN; tokens.len() * vocab];
        forward_qwen4_tokens(&remote, &tokens, 0, &mut logits, Some(&mut rows), &mut mem).expect("rows");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, on_card[i], "row {i}");
        }
        drop((mem, remote, rt));
        worker.join().unwrap();
    }

    /// Blocks whose experts are not on a card go to an RPC worker when one
    /// is connected: passes of several tokens still match single tokens,
    /// and the logits stay near the all-on-card ones (the worker gets the
    /// rows rounded to 8 bits).
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_experts_off_the_cards_run_on_an_rpc_worker() {
        use allpaca_rpc::{serve_connection, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"))).unwrap();
        });
        let tokens: Vec<u32> = (0..23u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();
        let local = tiny_packed_model();
        crate::qwen4_gpu::attach(&local, &hip, None, None, None).expect("attach");
        let on_card = run(&local, &tokens);

        let mut remote = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2]), None).expect("attach");
        {
            let gpu = remote.q4x.as_ref().unwrap().gpu.lock().unwrap();
            let gpu = gpu.as_ref().unwrap();
            assert_eq!(gpu.expert_layers()[0], [0, 1]);
            assert_eq!(gpu.remote_expert_layers(), [2, 3]);
        }
        let single = run(&remote, &tokens);
        let calls = rt.roundtrips.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(calls, 2 * tokens.len() as u64, "one call per remote block and token");
        let vocab = remote.config.vocab_size;
        let (mut mem, mut logits) = (Qwen4Memory::default(), vec![0.0f32; vocab]);
        let mut rows = vec![f32::NAN; tokens.len() * vocab];
        forward_qwen4_tokens(&remote, &tokens, 0, &mut logits, Some(&mut rows), &mut mem).expect("rows");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, single[i], "row {i}");
            let scale = on_card[i].iter().fold(1.0f32, |m, v| m.max(v.abs()));
            assert!(max_diff(row, &on_card[i]) <= 0.05 * scale, "row {i} is {} from the card's", max_diff(row, &on_card[i]));
        }
        // Two passes (16 + 7 tokens), a call per remote block each.
        assert_eq!(rt.roundtrips.load(std::sync::atomic::Ordering::Relaxed), calls + 4);
        drop((remote, rt));
        worker.join().unwrap();
    }

    /// With a worker that hosts blocks, the blocks past the cards run whole
    /// on it: one round trip a pass, the same logits as with every block on
    /// the card (the worker runs the same kernels), a pass of several
    /// tokens equal to one at a time, and a copied sequence going its own way.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_blocks_past_the_cards_run_whole_on_an_rpc_worker() {
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
            serve_connection_ext(stream, backend, None, vec![crate::qwen4_remote::worker_extension(None)]).unwrap();
        });
        let tokens: Vec<u32> = (0..23u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();
        let local = tiny_packed_model();
        crate::qwen4_gpu::attach(&local, &hip, None, None, None).expect("attach");
        let on_card = run(&local, &tokens);

        let mut remote = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2]), None).expect("attach");
        {
            let gpu = remote.q4x.as_ref().unwrap().gpu.lock().unwrap();
            let gpu = gpu.as_ref().unwrap();
            assert_eq!(gpu.expert_layers()[0], [0, 1]);
            assert_eq!(gpu.hosted_blocks(), [2, 3], "a Gated DeltaNet and a full-attention block");
            assert!(gpu.remote_expert_layers().is_empty());
            assert_eq!(gpu.hosted().unwrap().cache_bytes.load(std::sync::atomic::Ordering::Relaxed), 0);
        }
        let trips = || rt.roundtrips.load(std::sync::atomic::Ordering::Relaxed);
        let single = run(&remote, &tokens);
        assert_eq!(trips(), tokens.len() as u64, "one round trip per token");
        for (i, row) in single.iter().enumerate() {
            assert_eq!(row, &on_card[i], "token {i}");
        }
        let vocab = remote.config.vocab_size;
        let (mut mem, mut logits) = (Qwen4Memory::default(), vec![0.0f32; vocab]);
        let mut rows = vec![f32::NAN; tokens.len() * vocab];
        forward_qwen4_tokens(&remote, &tokens, 0, &mut logits, Some(&mut rows), &mut mem).expect("rows");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, single[i], "row {i}");
        }
        assert_eq!(trips(), tokens.len() as u64 + 2, "two passes (16 + 7 tokens)");

        // A copy keeps the state it was taken at while the original moves on.
        drop(mem);
        let mut mem = Qwen4Memory::default();
        for (pos, &t) in tokens[..10].iter().enumerate() {
            forward_qwen4_token(&remote, t, pos, &mut logits, &mut mem).expect("forward");
        }
        let mut fork = mem.clone();
        forward_qwen4_token(&remote, 3 + (tokens[10] - 2) % 5, 10, &mut logits, &mut mem).expect("other token");
        assert_ne!(logits, single[10], "the other token gives other logits");
        forward_qwen4_token(&remote, tokens[10], 10, &mut logits, &mut fork).expect("fork");
        assert_eq!(logits, single[10], "the copy continues from token 10");
        drop((mem, fork, remote, rt));
        worker.join().unwrap();
    }

    /// A prompt read without row logits through a worker that hosts blocks
    /// goes in overlapping passes: the worker runs one pass's blocks while
    /// the card runs the next one's. The logits and the state it leaves
    /// are those of one token at a time, n-gram embeddings included.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_prompt_passes_overlap_on_the_worker_and_match_one_at_a_time() {
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
            serve_connection_ext(stream, backend, None, vec![crate::qwen4_remote::worker_extension(None)]).unwrap();
        });
        let tokens: Vec<u32> = (0..45u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();
        let mut remote = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2]), None).expect("attach");
        let single = run(&remote, &tokens);
        let trips = || rt.roundtrips.load(std::sync::atomic::Ordering::Relaxed);
        let before = trips();
        let vocab = remote.config.vocab_size;
        let passes = (tokens.len() - 1).div_ceil(crate::qwen4_gpu::MAX_ROWS) as u64;
        assert!(passes >= 3, "{passes} passes");
        let (mut mem, mut logits) = (Qwen4Memory::default(), vec![0.0f32; vocab]);
        let last = tokens.len() - 1;
        forward_qwen4_tokens(&remote, &tokens[..last], 0, &mut logits, None, &mut mem).expect("prompt");
        assert_eq!(logits, single[last - 1], "the prompt's last logits");
        assert_eq!(trips() - before, passes, "one round trip a pass");
        assert_eq!(mem.tokens(), last);
        forward_qwen4_token(&remote, tokens[last], last, &mut logits, &mut mem).expect("next");
        assert_eq!(logits, single[last], "the token after the prompt");
        drop((mem, remote, rt));
        worker.join().unwrap();
    }

    /// The NextN drafter pairs each prompt token with the target's hidden
    /// streams for the token before, so a prompt read in overlapping passes
    /// through a worker that hosts blocks still hands back every row's
    /// hidden streams, the same as one token at a time.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_prompt_passes_on_the_worker_keep_every_rows_hidden_streams() {
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
            serve_connection_ext(stream, backend, None, vec![crate::qwen4_remote::worker_extension(None)]).unwrap();
        });
        let tokens: Vec<u32> = (0..40u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();
        let mut remote = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2]), None).expect("attach");
        assert!(tokens.len() > crate::qwen4_gpu::MAX_ROWS, "the prompt goes in overlapping passes");
        let hc_dim = remote.q4x.as_ref().unwrap().spec.hc * remote.config.dim;
        let vocab = remote.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];

        let mut one_at_a_time = Vec::new();
        let mut mem = Qwen4Memory::default();
        for (pos, &t) in tokens.iter().enumerate() {
            forward_qwen4_tokens_with_hiddens(&remote, &[t], pos, &mut logits, None, &mut one_at_a_time, &mut mem).expect("token");
        }
        assert_eq!(one_at_a_time.len(), tokens.len() * hc_dim);
        // A memory with a sequence on the worker keeps its connection open.
        drop(mem);

        let (mut passes, mut mem) = (Vec::new(), Qwen4Memory::default());
        forward_qwen4_tokens_with_hiddens(&remote, &tokens, 0, &mut logits, None, &mut passes, &mut mem).expect("prompt");
        assert_eq!(passes.len(), one_at_a_time.len(), "a hidden row per prompt token");
        assert_eq!(passes, one_at_a_time);
        drop((mem, remote, rt));
        worker.join().unwrap();
    }

    /// Several sequences in one batched forward, pipelined through the
    /// worker that hosts blocks, each get the logits of their own forward:
    /// one token each, and a span of several rows next to a single one.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_sequences_batched_through_the_worker_match_their_own_runs() {
        use crate::anchors::RecurrentState;
        use crate::model::SeqRows;
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
            serve_connection_ext(stream, backend, None, vec![crate::qwen4_remote::worker_extension(None)]).unwrap();
        });
        let mut model = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        model.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&model, &hip, None, Some(&[2]), None).expect("attach");
        assert!(model.supports_forward_batch(&hip), "hosted blocks make the batched forward worth it");
        let vocab = model.config.vocab_size;
        let seqs: [Vec<u32>; 3] = [
            (0..8u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect(),
            (0..8u32).map(|i| 3 + (i * 3 + 1) % 5).collect(),
            (0..8u32).map(|i| 3 + (i * 2 + i / 2) % 5).collect(),
        ];
        let solo: Vec<Vec<Vec<f32>>> = seqs.iter().map(|tokens| run(&model, tokens)).collect();
        let trips = || rt.roundtrips.load(std::sync::atomic::Ordering::Relaxed);
        let before = trips();
        let mut states: Vec<RecurrentState> = (0..3).map(|_| RecurrentState::zeros(model.config.n_layers, model.config.dim)).collect();
        // Five ticks of one token for each sequence.
        for step in 0..5 {
            let tokens: Vec<u32> = seqs.iter().map(|s| s[step]).collect();
            let mut rows = vec![f32::NAN; 3 * vocab];
            let mut segs: Vec<SeqRows<'_>> = states
                .iter_mut()
                .map(|state| SeqRows { start_pos: step, n_rows: 1, page_ids: &[], request_id: 0, recurrent: Some(state) })
                .collect();
            forward_qwen4_batch(&model, &tokens, &mut segs, &mut rows).expect("batch");
            for (s, row) in rows.chunks(vocab).enumerate() {
                assert_eq!(row, solo[s][step], "sequence {s} token {step}");
            }
        }
        assert_eq!(trips() - before, 15, "one round trip per sequence and tick");
        // A tick with a three-row span for sequence 0 and one row for sequence 1.
        let tokens: Vec<u32> = seqs[0][5..8].iter().chain(&seqs[1][5..6]).copied().collect();
        let mut rows = vec![f32::NAN; 4 * vocab];
        {
            let (a, rest) = states.split_at_mut(1);
            let mut segs = vec![
                SeqRows { start_pos: 5, n_rows: 3, page_ids: &[], request_id: 0, recurrent: Some(&mut a[0]) },
                SeqRows { start_pos: 5, n_rows: 1, page_ids: &[], request_id: 1, recurrent: Some(&mut rest[0]) },
            ];
            forward_qwen4_batch(&model, &tokens, &mut segs, &mut rows).expect("span batch");
        }
        for i in 0..3 {
            assert_eq!(&rows[i * vocab..(i + 1) * vocab], solo[0][5 + i], "span row {i}");
        }
        assert_eq!(&rows[3 * vocab..], solo[1][5], "the single row next to the span");
        drop((states, model, rt));
        worker.join().unwrap();
    }

    /// A worker with a weight cache is sent a model's weights once: the
    /// second coordinator to host the same blocks uploads next to nothing
    /// and gets the same logits.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_hosted_weights_come_from_the_workers_cache_the_second_time() {
        use crate::qwen4_remote::{worker_extension, WeightCache};
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        use std::sync::atomic::Ordering;
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let dir = std::env::temp_dir().join(format!("allpaca-weight-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = WeightCache::new(&dir, 1 << 30);
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = std::thread::spawn(move || {
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
                serve_connection_ext(stream, backend, None, vec![worker_extension(Some(cache.clone()))]).unwrap();
            }
        });
        let tokens: Vec<u32> = (0..9u32).map(|i| 3 + (i * 7 + i / 3) % 5).collect();
        let mut loads = Vec::new();
        for _ in 0..2 {
            let mut model = tiny_packed_model();
            let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr.clone()], None).expect("connect"));
            model.rpc = Some(rt.clone());
            crate::qwen4_gpu::attach(&model, &hip, None, Some(&[2]), None).expect("attach");
            let cached = {
                let gpu = model.q4x.as_ref().unwrap().gpu.lock().unwrap();
                gpu.as_ref().unwrap().hosted().expect("hosted blocks").cache_bytes.load(Ordering::Relaxed)
            };
            loads.push((rt.h2d_bytes.load(Ordering::Relaxed), cached, run(&model, &tokens)));
        }
        worker.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let (first, second) = (&loads[0], &loads[1]);
        // Weights with the same bytes (the tiny model's norm gains) go once even the first time.
        assert!(first.0 > 10 * first.1, "the first load uploads: {} sent, {} from the cache", first.0, first.1);
        assert_eq!(second.1, first.0 + first.1, "the second load takes every weight from the cache");
        assert_eq!(second.0, 0, "and uploads none");
        assert_eq!(second.2, first.2, "same logits");
    }

    /// The cache gives back what it was given, drops a damaged file, and
    /// makes room by deleting the weight used longest ago.
    #[test]
    fn weight_cache_checks_what_it_reads_and_stays_under_its_limit() {
        use crate::qwen4_remote::{weight_key, WeightCache};
        let dir = std::env::temp_dir().join(format!("allpaca-weight-cache-unit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cache = WeightCache::new(&dir, 250);
        let weight = |seed: u8| -> Vec<u8> { (0..100u8).map(|i| i.wrapping_mul(7).wrapping_add(seed)).collect() };
        let (a, b, c) = (weight(1), weight(2), weight(3));
        assert_ne!(weight_key(&a), weight_key(&b));
        assert_ne!(weight_key(&a[..99]), weight_key(&a), "the length is part of the key");
        assert_eq!(cache.get(weight_key(&a), 100), None, "nothing yet");
        cache.put(weight_key(&a), &a);
        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.put(weight_key(&b), &b);
        assert_eq!(cache.get(weight_key(&a), 100).as_deref(), Some(&a[..]));
        assert_eq!(cache.get(weight_key(&a), 99), None, "another length is another weight");
        // `a` was just read, so `b` is the one used longest ago.
        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.put(weight_key(&c), &c);
        assert_eq!(cache.get(weight_key(&b), 100), None, "the limit of 250 bytes holds two");
        assert!(cache.get(weight_key(&a), 100).is_some() && cache.get(weight_key(&c), 100).is_some());
        // A file whose bytes changed is not handed out.
        let file = std::fs::read_dir(&dir).unwrap().flatten().find(|e| e.file_name().to_string_lossy().starts_with(&format!("{:016x}", weight_key(&c)))).unwrap();
        let mut bytes = std::fs::read(file.path()).unwrap();
        bytes[50] ^= 1;
        std::fs::write(file.path(), bytes).unwrap();
        assert_eq!(cache.get(weight_key(&c), 100), None);
        assert!(!file.path().exists(), "and is deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Logits after each token of `tokens`.
    /// The memory kept for the prompt cache (read back to the host, then
    /// through its file encoding) carries on exactly as the live memory does.
    fn carries_on_from_the_cache_copy(model: &TinyModel, what: &str) {
        let tokens = long_tokens(14);
        let vocab = model.config.vocab_size;
        let mut live = Qwen4Memory::default();
        let mut logits = vec![0.0f32; vocab];
        forward_qwen4_tokens(model, &tokens[..9], 0, &mut logits, None, &mut live).expect("prefix");
        let kept = live.for_cache(model).expect("read back").expect("cacheable");
        assert!(kept.host_bytes() > 0, "{what}");
        let mut bytes = Vec::new();
        kept.write_to(&mut bytes);
        let mut copy = Qwen4Memory::read_from(&mut crate::kv_codec::Reader(&bytes)).expect("decode");
        assert_eq!(copy.tokens(), 9, "{what}");
        let (mut want, mut got) = (vec![0.0f32; 5 * vocab], vec![0.0f32; 5 * vocab]);
        forward_qwen4_tokens(model, &tokens[9..], 9, &mut logits, Some(&mut want), &mut live).expect("live");
        forward_qwen4_tokens(model, &tokens[9..], 9, &mut logits, Some(&mut got), &mut copy).expect("copy");
        assert!(want.iter().any(|v| *v != 0.0), "{what}");
        assert_eq!(got, want, "{what}");
    }

    #[test]
    fn qwen4exp_memory_kept_for_the_prompt_cache_carries_on_like_the_live_one() {
        carries_on_from_the_cache_copy(&tiny_model(), "host");
        carries_on_from_the_cache_copy(&tiny_sparse_model(2), "host, sparse");
    }

    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_memory_kept_for_the_prompt_cache_carries_on_on_the_gpu() {
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        for (what, model) in [("packed", tiny_packed_model()), ("packed, sparse", tiny_packed_sparse_model(2))] {
            crate::qwen4_gpu::attach(&model, &hip, None, None, None).expect("attach");
            carries_on_from_the_cache_copy(&model, what);
        }
    }

    /// A sequence whose last blocks a worker hosts is kept for the prompt
    /// cache with those blocks' state saved on the worker: a copy restored
    /// from it carries on exactly as the live sequence, as often as it is
    /// restored. It stays out of files (the worker's part is not in them).
    /// A worker with no room for it makes the sequence uncacheable, and a
    /// copy whose saved part the worker lost says so when restored.
    #[cfg(feature = "hip")]
    #[test]
    fn qwen4exp_memory_with_hosted_blocks_is_kept_with_their_state_on_the_worker() {
        use allpaca_rpc::{serve_connection_ext, HipWorker, WorkerBackend};
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 16) else {
            return;
        };
        let hip = Arc::new(hip);
        let start = |saved_bytes: usize| {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap().to_string();
            let worker = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
                let ext = crate::qwen4_remote::worker_extension_saving(None, saved_bytes);
                serve_connection_ext(stream, backend, None, vec![ext]).unwrap();
            });
            let mut model = tiny_packed_model();
            let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
            model.rpc = Some(rt);
            crate::qwen4_gpu::attach(&model, &hip, None, Some(&[2]), None).expect("attach");
            assert_eq!(model.q4x.as_ref().unwrap().gpu.lock().unwrap().as_ref().unwrap().hosted_blocks(), [2, 3]);
            (model, worker)
        };
        let tokens = long_tokens(14);
        let (model, worker) = start(64 << 20);
        let vocab = model.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        let mut live = Qwen4Memory::default();
        forward_qwen4_tokens(&model, &tokens[..9], 0, &mut logits, None, &mut live).expect("prefix");
        let kept = live.for_cache(&model).expect("save").expect("cacheable with the worker's part saved");
        assert!(!kept.persistable(), "the worker's part is not in a file");
        let mut bytes = Vec::new();
        kept.write_to(&mut bytes);
        assert!(Qwen4Memory::read_from(&mut crate::kv_codec::Reader(&bytes)).is_none(), "and a file of it is refused");
        let mut want = vec![0.0f32; 5 * vocab];
        forward_qwen4_tokens(&model, &tokens[9..], 9, &mut logits, Some(&mut want), &mut live).expect("live");
        for round in 0..2 {
            let mut copy = kept.clone();
            assert!(copy.restore_remote().expect("restore"), "round {round}");
            let mut got = vec![0.0f32; 5 * vocab];
            forward_qwen4_tokens(&model, &tokens[9..], 9, &mut logits, Some(&mut got), &mut copy).expect("copy");
            assert_eq!(got, want, "round {round}");
        }
        drop((live, kept, model));
        worker.join().unwrap();

        // No room on the worker: not cacheable.
        let (model, worker) = start(1);
        let mut live = Qwen4Memory::default();
        forward_qwen4_tokens(&model, &tokens[..9], 0, &mut logits, None, &mut live).expect("prefix");
        assert!(live.for_cache(&model).expect("save").is_none());
        drop((live, model));
        worker.join().unwrap();
    }

    fn run(model: &TinyModel, tokens: &[u32]) -> Vec<Vec<f32>> {
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; model.config.vocab_size];
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| {
                forward_qwen4_token(model, t, pos, &mut logits, &mut mem).expect("forward");
                logits.clone()
            })
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn tiny_qwen4exp_loads_its_shape_from_the_file() {
        let model = tiny_model();
        let q4 = model.q4x.as_ref().expect("qwen4exp weights");
        assert_eq!((q4.spec.hc, q4.spec.full_attention_interval), (2, 4));
        assert_eq!(q4.spec.ple.as_ref().map(|p| (p.layers.clone(), p.n_heads())), Some((vec![1], 4)));
        // Layers 0..2 are Gated DeltaNet, layer 3 is full attention.
        assert_eq!((0..4).map(|l| q4.spec.is_recurrent(l)).collect::<Vec<_>>(), [true, true, true, false]);
        assert!(model.config.recurrent, "the engine must give the sequence a state");
        assert_eq!((model.config.n_layers, model.config.n_experts, model.config.n_active), (4, 4, 2));
    }

    #[test]
    fn tiny_qwen4exp_forward_depends_on_the_whole_sequence() {
        let model = tiny_model();
        let a = run(&model, &[3, 4, 5, 6, 7, 3]);
        assert!(a.iter().flatten().all(|v| v.is_finite()));
        // The same tokens give the same logits again.
        let again = run(&model, &[3, 4, 5, 6, 7, 3]);
        assert_eq!(a, again);
        // A token's logits depend on what came before it (recurrent state,
        // attention and the n-gram hash all look back)...
        assert!(max_diff(&a[5], &a[0]) > 1e-4, "token 3 at position 0 and 5 must differ");
        // ...including a change five tokens back, which only the recurrent
        // layers and attention can carry (the n-gram window is three tokens).
        let b = run(&model, &[4, 4, 5, 6, 7, 3]);
        assert!(max_diff(&a[5], &b[5]) > 1e-5, "{}", max_diff(&a[5], &b[5]));
        // A different last token changes the result even with the same history.
        let c = run(&model, &[3, 4, 5, 6, 7, 4]);
        assert!(max_diff(&a[5], &c[5]) > 1e-4);
        assert_eq!(a[..5], c[..5]);
    }

    #[test]
    fn tiny_qwen4exp_nextn_seed_uses_every_hidden_stream() {
        let model = tiny_model();
        let dim = model.config.dim;
        let hidden: Vec<f32> = (0..model.q4x.as_ref().unwrap().spec.hc * dim)
            .map(|i| (i + 1) as f32 / dim as f32)
            .collect();
        let seeded = |streams: &[f32]| {
            let mut mem = Qwen4Memory { nextn_seed: Some(streams.to_vec()), ..Default::default() };
            let mut logits = vec![0.0f32; model.config.vocab_size];
            forward_qwen4_token(&model, 3, 0, &mut logits, &mut mem).expect("NextN forward");
            assert_eq!(mem.tokens(), 1);
            logits
        };
        let first = seeded(&hidden);
        let mut swapped = hidden.clone();
        swapped.copy_within(dim.., 0);
        swapped[dim..].copy_from_slice(&hidden[..hidden.len() - dim]);
        let second = seeded(&swapped);
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(max_diff(&first, &second) > 1e-5, "the NextN seed must mix all hidden streams");
    }

    /// llama.cpp's qwen4exp MTP graph: the NextN block starts from one
    /// residual per stream, `eh_proj([enorm(e) ; hnorm(h_s)])`, with each
    /// stream of the hidden RMS-normed on its own. No mixer folds the
    /// streams together first.
    #[test]
    fn tiny_qwen4exp_nextn_block_starts_from_a_residual_per_stream() {
        let model = tiny_model();
        let q4 = model.q4x.as_ref().unwrap();
        let (dim, hc, eps) = (model.config.dim, q4.spec.hc, model.config.eps);
        // Streams of very different sizes: one norm over all of them is not
        // one norm per stream.
        let hidden: Vec<f32> = (0..hc * dim)
            .map(|i| ((i % 7) as f32 - 3.0) * (1 + 4 * (i / dim)) as f32 * 0.1)
            .collect();
        let mut mem = Qwen4Memory {
            nextn_seed: Some(hidden.clone()),
            trace: Some(crate::sensitivity::LayerTrace::Record { saved: Vec::new(), ran: 0 }),
            ..Default::default()
        };
        let mut logits = vec![0.0f32; model.config.vocab_size];
        forward_qwen4_token(&model, 3, 0, &mut logits, &mut mem).expect("NextN forward");
        let Some(crate::sensitivity::LayerTrace::Record { saved, .. }) = mem.trace else {
            panic!("the trace records block inputs");
        };

        let tok = q4.rows("token_embd.weight", 3, 1, dim).unwrap();
        let (enorm, hnorm) = (q4.w("blk.0.nextn.enorm.weight").unwrap(), q4.w("blk.0.nextn.hnorm.weight").unwrap());
        let eh_proj = q4.w("blk.0.nextn.eh_proj.weight").unwrap();
        let mut e_norm = vec![0.0f32; dim];
        rmsnorm(&tok, &enorm, eps, &mut e_norm).unwrap();
        let mut want = Vec::with_capacity(hc * dim);
        for s in 0..hc {
            let mut h_norm = vec![0.0f32; dim];
            rmsnorm(&hidden[s * dim..(s + 1) * dim], &hnorm[s * dim..(s + 1) * dim], eps, &mut h_norm).unwrap();
            let concat = [e_norm.as_slice(), h_norm.as_slice()].concat();
            let mut out = vec![0.0f32; dim];
            matmul_nt(&concat, &eh_proj, &mut out, 1, dim, 2 * dim).unwrap();
            want.extend(out);
        }
        let got = &saved[..hc * dim];
        assert!(max_diff(got, &want) < 1e-5, "NextN block input differs by {}", max_diff(got, &want));
    }

    /// The hidden streams the target hands the NextN head for position `k`.
    fn target_hiddens(hc_dim: usize, n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|k| (0..hc_dim).map(|i| ((i * 7 + k * 13) % 11) as f32 * 0.2 - 1.0).collect())
            .collect()
    }

    /// The head's greedy chain from `mem`: each draft goes back in at the
    /// next position with the head's own streams as its hidden.
    fn head_chain(model: &TinyModel, mem: &mut Qwen4Memory, first: u32, pos: usize, n: usize) -> Vec<u32> {
        let mut out = vec![first];
        let mut logits = vec![0.0f32; model.config.vocab_size];
        while out.len() < n {
            let at = pos + out.len() - 1;
            forward_qwen4_token(model, out[out.len() - 1], at, &mut logits, mem).expect("head step");
            out.push(crate::sampling::argmax(&logits));
        }
        out
    }

    /// llama.cpp `draft-mtp`: `process` pairs token `k` with the target's
    /// hidden for `k - 1` at position `k`, and its logits predict token
    /// `k + 1`. With the context forwarded through the target, the first
    /// draft is that prediction for the last context token; each draft after
    /// it feeds the one before at the next position.
    #[test]
    fn tiny_qwen4exp_mtp_drafts_follow_the_last_processed_pair() {
        let model = tiny_model();
        let hc_dim = model.q4x.as_ref().unwrap().spec.hc * model.config.dim;
        let ctx = [3u32, 4, 5, 6, 7, 3, 4];
        let hiddens = target_hiddens(hc_dim, ctx.len());
        let metrics = crate::telemetry::Metrics::default();
        let mut provider = crate::draft::Qwen4MtpProvider::new(model.clone());
        let mut want = Qwen4Memory::default();
        let mut logits = vec![0.0f32; model.config.vocab_size];
        for (k, &tok) in ctx.iter().enumerate() {
            let h_prev = if k == 0 { vec![0.0; hc_dim] } else { hiddens[k - 1].clone() };
            provider.process(tok, k, &h_prev, &metrics).expect("process");
            want.nextn_seed = Some(h_prev);
            forward_qwen4_token(&model, tok, k, &mut logits, &mut want).expect("reference process");
        }
        let first = crate::sampling::argmax(&logits);
        let expected = head_chain(&model, &mut want, first, ctx.len(), 3);

        let drafts = provider.propose_on(&ctx, 3, 0.0, &metrics).expect("propose");
        assert_eq!(drafts, expected);
    }

    /// After a rejected draft the target's correction is known but not yet
    /// forwarded. Like llama.cpp's first draft step `(id_last, pending_h)`,
    /// the head runs the correction with the target's hidden streams for the
    /// last context token, at the position after the context.
    #[test]
    fn tiny_qwen4exp_mtp_correction_drafts_from_the_targets_last_hidden() {
        let model = tiny_model();
        let hc_dim = model.q4x.as_ref().unwrap().spec.hc * model.config.dim;
        let ctx = [3u32, 4, 5, 6, 7, 3, 4];
        let hiddens = target_hiddens(hc_dim, ctx.len());
        let metrics = crate::telemetry::Metrics::default();
        let mut provider = crate::draft::Qwen4MtpProvider::new(model.clone());
        let mut want = Qwen4Memory::default();
        let mut logits = vec![0.0f32; model.config.vocab_size];
        for (k, &tok) in ctx.iter().enumerate() {
            let h_prev = if k == 0 { vec![0.0; hc_dim] } else { hiddens[k - 1].clone() };
            provider.process(tok, k, &h_prev, &metrics).expect("process");
            want.nextn_seed = Some(h_prev);
            forward_qwen4_token(&model, tok, k, &mut logits, &mut want).expect("reference process");
        }
        let last_hidden = &hiddens[ctx.len() - 1];
        // Every token as the correction: an 8-token vocabulary often agrees
        // on one by chance.
        for correction in 0..model.config.vocab_size as u32 {
            let mut after = want.clone();
            after.nextn_seed = Some(last_hidden.clone());
            forward_qwen4_token(&model, correction, ctx.len(), &mut logits, &mut after).expect("correction step");
            let mut expected = vec![correction];
            expected.extend(head_chain(&model, &mut after, crate::sampling::argmax(&logits), ctx.len() + 1, 3));

            let drafts = provider
                .propose_after(&ctx, correction, last_hidden, 4, 0.0, &metrics)
                .expect("propose after a correction");
            assert_eq!(drafts, expected, "correction {correction}");
        }
    }

    #[test]
    fn tiny_qwen4exp_mtp_sidecar_loads_its_full_attention_block() {
        let path = std::env::temp_dir().join(format!("allpaca_tiny_qwen4exp_mtp-{}.gguf", std::process::id()));
        write_tiny_qwen4exp_gguf(&path);
        let file = GgufFile::open(&path).expect("open tiny qwen4exp");
        let model = TinyModel::from_gguf_mtp(&file).expect("MTP sidecar");
        let _ = std::fs::remove_file(&path);
        assert_eq!(model.config.n_layers, 1);
        assert!(!model.q4x.as_ref().unwrap().spec.is_recurrent(0));
    }

    /// A run of tokens leaves each token's logits in its row: a draft
    /// verify compares every row with the token drafted after it.
    #[test]
    fn tiny_qwen4exp_token_runs_keep_every_rows_logits() {
        let model = tiny_model();
        let tokens = [3u32, 4, 5, 6, 7, 3];
        let single = run(&model, &tokens);
        let vocab = model.config.vocab_size;
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; vocab];
        let mut rows = vec![f32::NAN; 4 * vocab];
        forward_qwen4_tokens(&model, &tokens[..2], 0, &mut logits, None, &mut mem).expect("prompt");
        assert_eq!(logits, single[1]);
        forward_qwen4_tokens(&model, &tokens[2..], 2, &mut logits, Some(&mut rows), &mut mem).expect("verify");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, single[2 + i], "row {i}");
        }
        assert_eq!(logits, single[5]);
        // Rows that do not fit are refused before anything runs.
        let mut short = vec![0.0f32; vocab];
        assert!(forward_qwen4_tokens(&model, &[4, 5], 6, &mut logits, Some(&mut short), &mut mem).is_err());
        assert_eq!(mem.tokens(), 6);
    }

    /// Past its indexer's budget the attention keeps the best pools and
    /// the tail: the sequence goes on, and from the first token that sees
    /// more pools than it may keep its logits differ from the same
    /// weights with a budget that covers everything.
    #[test]
    fn tiny_qwen4exp_attends_sparsely_past_its_indexer_budget() {
        let tokens = long_tokens(14);
        // Two pools of two cells are kept: rank 5 is the first to see three.
        let sparse = run(&tiny_sparse_model(4), &tokens);
        let dense = run(&tiny_sparse_model(64), &tokens);
        assert!(sparse.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(sparse[..5], dense[..5], "every cell is kept up to rank 4");
        for rank in 5..tokens.len() {
            assert!(max_diff(&sparse[rank], &dense[rank]) > 1e-6, "rank {rank} must drop cells");
        }
        // Without pools to drop the indexer changes nothing.
        assert_eq!(dense, run(&tiny_model(), &tokens));
        // A run of tokens gives each of them the logits it gets alone.
        let model = tiny_sparse_model(4);
        let vocab = model.config.vocab_size;
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; vocab];
        let mut rows = vec![f32::NAN; 10 * vocab];
        forward_qwen4_tokens(&model, &tokens[..4], 0, &mut logits, None, &mut mem).expect("prompt");
        forward_qwen4_tokens(&model, &tokens[4..], 4, &mut logits, Some(&mut rows), &mut mem).expect("rows");
        for (i, row) in rows.chunks(vocab).enumerate() {
            assert_eq!(row, sparse[4 + i], "row {i}");
        }
    }

    /// Several sequences through the block-major host forward come out as
    /// each does on its own, bit for bit: dense and sparse attention, F32
    /// and Q2_0 experts, an expert mask on one of them, and a second call
    /// that carries on from the first.
    #[test]
    fn host_rows_of_several_sequences_match_their_own_runs() {
        let path = std::env::temp_dir().join(format!("allpaca_tiny_qwen4exp_rows_packed-{}.gguf", std::process::id()));
        write_tiny_qwen4exp_packed_gguf(&path);
        let file = Arc::new(GgufFile::open(&path).expect("open packed tiny qwen4exp"));
        let mut packed = TinyModel::from_gguf(&file).expect("model");
        packed.bind_gguf(file);
        let _ = std::fs::remove_file(&path);
        let (n_blocks, n_experts) = (tiny_model().config.n_layers, tiny_model().config.n_experts);
        let mut mask = vec![false; n_blocks * n_experts];
        mask[n_experts + 1] = true;
        let mask = Arc::new(mask);
        let texts = [long_tokens(11), long_tokens(14)[3..].to_vec(), vec![5u32, 3]];
        for (what, model) in [("dense", tiny_model()), ("sparse", tiny_sparse_model(4)), ("packed", packed)] {
            let vocab = model.config.vocab_size;
            let fresh = |i: usize| Qwen4Memory { expert_mask: (i == 1).then(|| mask.clone()), ..Default::default() };
            // Each on its own, a token at a time.
            let alone: Vec<Vec<f32>> = texts
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let mut mem = fresh(i);
                    let (mut logits, mut rows) = (vec![0.0f32; vocab], vec![0.0f32; t.len() * vocab]);
                    forward_qwen4_tokens(&model, t, 0, &mut logits, Some(&mut rows), &mut mem).expect("alone");
                    rows
                })
                .collect();
            // Together, the first four tokens of each, then the rest.
            let mut mems: Vec<Qwen4Memory> = (0..texts.len()).map(fresh).collect();
            let mut together: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
            for part in [0..4usize, 4..usize::MAX] {
                let seqs: Vec<&[u32]> = texts.iter().map(|t| &t[part.start.min(t.len())..part.end.min(t.len())]).collect();
                let mut rows = vec![f32::NAN; seqs.iter().map(|s| s.len()).sum::<usize>() * vocab];
                forward_qwen4_host_rows(&model, &seqs, &mut mems, &mut rows).expect("together");
                let mut at = 0;
                for (out, s) in together.iter_mut().zip(&seqs) {
                    out.extend_from_slice(&rows[at * vocab..(at + s.len()) * vocab]);
                    at += s.len();
                }
            }
            for (i, (a, b)) in alone.iter().zip(&together).enumerate() {
                assert!(a.iter().all(|v| v.is_finite()), "{what} text {i}");
                assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{what} text {i}: {}", max_diff(a, b));
            }
            assert_ne!(alone[1][alone[1].len() - vocab..], {
                let mut mem = Qwen4Memory::default();
                let mut logits = vec![0.0f32; vocab];
                forward_qwen4_tokens(&model, &texts[1], 0, &mut logits, None, &mut mem).expect("unmasked");
                logits
            }[..], "{what}: the mask changes the output");
        }
    }

    /// Dense tensors decoded on every use, over the cores, read as the
    /// file's own decode does, and as the cached ones: the same forward.
    #[test]
    fn dense_tensors_decoded_on_use_match_the_cached_ones() {
        let path = std::env::temp_dir().join(format!("allpaca_tiny_qwen4exp_decode-{}.gguf", std::process::id()));
        write_tiny_qwen4exp_packed_gguf(&path);
        let file = Arc::new(GgufFile::open(&path).expect("open packed tiny qwen4exp"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file.clone());
        let _ = std::fs::remove_file(&path);
        let q4 = model.q4x.as_ref().unwrap();
        for t in &file.tensors {
            let want = file.tensor_f32(&t.name).unwrap();
            let got = q4.decode_threaded(&t.name).unwrap();
            assert!(want.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()) && want.len() == got.len(), "{}", t.name);
        }
        let tokens = long_tokens(9);
        let cached = run(&model, &tokens);
        q4.keep_dense(false).unwrap();
        assert!(q4.dense.lock().unwrap().is_empty(), "the kept tensors are dropped");
        assert_eq!(run(&model, &tokens), cached);
        assert!(q4.dense.lock().unwrap().is_empty(), "nothing is kept while off");
    }

    #[test]
    fn tiny_qwen4exp_refuses_a_position_its_memory_is_not_at() {
        let model = tiny_model();
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; model.config.vocab_size];
        forward_qwen4_token(&model, 3, 0, &mut logits, &mut mem).unwrap();
        assert_eq!(mem.tokens(), 1);
        let err = forward_qwen4_token(&model, 4, 5, &mut logits, &mut mem).expect_err("skipped positions");
        assert!(err.to_string().contains("sequence memory"), "{err}");
    }

    #[test]
    fn ple_update_adds_a_gated_value_and_a_dilated_conv_of_earlier_ones() {
        // One stream of two dims, n-gram size 2 (so the conv looks 2 tokens back), kernel 2.
        let ple = PleSpec {
            ngram_size: 2,
            conv_kernel: 2,
            ..PleSpec::default()
        };
        let (key_w, value_w) = ([1.0f32, 0.0], [1.0f32, 3.0]);
        let ones = [1.0f32, 1.0];
        // conv taps per channel: [two back, current].
        let conv_w = [10.0f32, 0.0, 10.0, 0.0];
        let mut history = Vec::new();
        let step = |emb: f32, history: &mut Vec<f32>| {
            let mut streams = [1.0f32, 0.0];
            ple_update(&mut streams, &[emb], &key_w, &value_w, [&ones, &ones, &ones], &conv_w, history, &ple, 2, 0.0).unwrap();
            streams
        };
        // Key (emb, 0) normed to (sqrt 2, 0); query (1, 0) normed to (sqrt 2, 0):
        // s = 2 / sqrt 2, gate = sigmoid(sqrt(s)); value = emb * (1, 3).
        let gate = sigmoid((2.0f32 / 2f32.sqrt()).sqrt());
        let first = step(1.0, &mut history);
        // No history yet: the conv adds silu(0) = 0.
        assert!((first[0] - (1.0 + gate)).abs() < 1e-5 && (first[1] - 3.0 * gate).abs() < 1e-5, "{first:?}");
        let second = step(1.0, &mut history);
        assert!((second[0] - first[0]).abs() < 1e-6, "one token back is not a tap");
        // Third token: the tap two back reads the first token's conv input,
        // the gated value normed to unit RMS: (1, 3) / sqrt 5.
        let third = step(1.0, &mut history);
        let tap = 10.0 * (1.0 / 5f32.sqrt());
        assert!((third[0] - (1.0 + gate + silu(tap))).abs() < 1e-4, "{third:?}");
        assert!((third[1] - (3.0 * gate + silu(3.0 * tap))).abs() < 1e-4, "{third:?}");
    }

    /// A masked expert is never picked, the rest still are, and the saliency
    /// recorder counts exactly the recorded routes.
    #[test]
    fn tiny_qwen4exp_expert_mask_keeps_experts_out() {
        let model = tiny_model();
        let n_experts = model.config.n_experts;
        let n_blocks = model.config.n_layers;
        let tokens = [3u32, 4, 5, 6, 7, 3, 5, 4];
        let mut logits = vec![0.0f32; model.config.vocab_size];
        let read = |mem: &mut Qwen4Memory, logits: &mut Vec<f32>| -> Vec<Vec<f32>> {
            tokens
                .iter()
                .enumerate()
                .map(|(pos, &t)| {
                    forward_qwen4_token(&model, t, pos, logits, mem).expect("forward");
                    logits.clone()
                })
                .collect()
        };
        let mut plain = Qwen4Memory {
            routes: Some(Vec::new()),
            saliency: Some(vec![[0.0; 3]; n_blocks * n_experts]),
            ..Default::default()
        };
        let a = read(&mut plain, &mut logits);
        let routes = plain.routes.take().unwrap();
        let sal = plain.saliency.take().unwrap();
        for (i, slot) in sal.iter().enumerate() {
            let picked = routes.iter().filter(|(l, e)| usize::from(*l) * n_experts + usize::from(*e) == i).count();
            assert_eq!(slot[0], picked as f64, "slot {i}");
            assert!(slot[1] >= 0.0 && slot[2] >= 0.0);
            assert_eq!(slot[1] > 0.0, picked > 0);
        }
        // Mask the most used expert of every block.
        let mut mask = vec![false; n_blocks * n_experts];
        for blk in 0..n_blocks {
            let hot = (0..n_experts).max_by(|x, y| sal[blk * n_experts + x][0].total_cmp(&sal[blk * n_experts + y][0]));
            mask[blk * n_experts + hot.unwrap()] = true;
        }
        let mut masked = Qwen4Memory {
            routes: Some(Vec::new()),
            expert_mask: Some(Arc::new(mask.clone())),
            ..Default::default()
        };
        let b = read(&mut masked, &mut logits);
        let routes = masked.routes.take().unwrap();
        assert_eq!(routes.len(), tokens.len() * n_blocks * model.config.n_active);
        assert!(routes.iter().all(|(l, e)| !mask[usize::from(*l) * n_experts + usize::from(*e)]));
        assert!(a.iter().zip(&b).any(|(x, y)| max_diff(x, y) > 1e-6), "masking the hottest experts changed nothing");
    }

    /// `ALLPACA_Q4X_GPU` = number of cards to put matrices and experts on;
    /// `ALLPACA_Q4X_RPC` = worker (`host:port`, token in `ALLPACA_RPC_TOKEN`)
    /// for the experts off the cards. Neither: the host forward.
    fn attach_from_env(model: &mut TinyModel) {
        #[cfg(feature = "hip")]
        if let Some(cards) = std::env::var("ALLPACA_Q4X_GPU").ok().and_then(|v| v.parse::<i32>().ok()) {
            let load = std::time::Instant::now();
            let vocab = model.config.vocab_size;
            let first = Arc::new(allpaca_hip::HipWorkspace::new(0, 1 << 20, vocab).expect("card 0"));
            let second = (cards > 1).then(|| Arc::new(allpaca_hip::HipWorkspace::new(1, 1 << 20, vocab).expect("card 1")));
            if let Ok(endpoint) = std::env::var("ALLPACA_Q4X_RPC") {
                let token = std::env::var("ALLPACA_RPC_TOKEN").ok();
                let rt = crate::rpc_runtime::RpcRuntime::connect(&[endpoint], token.as_deref()).expect("rpc worker");
                model.rpc = Some(Arc::new(rt));
            }
            crate::qwen4_gpu::attach(model, &first, second.as_ref(), None, None).expect("attach");
            let gpu = model.q4x.as_ref().unwrap().gpu.lock().unwrap();
            eprintln!(
                "Q4X gpu attached in {:.1}s, expert layers per card {:?}",
                load.elapsed().as_secs_f32(),
                gpu.as_ref().unwrap().expert_layers().iter().map(Vec::len).collect::<Vec<_>>()
            );
        }
        #[cfg(not(feature = "hip"))]
        let _ = model;
    }

    /// Opt-in: what pruning experts costs on a real qwen4exp file.
    /// `ALLPACA_QWEN4EXP` is the first shard, `ALLPACA_Q4X_TOKENS` the tokens
    /// read per paragraph (blank-line separated). Without the scores file
    /// `ALLPACA_Q4X_PRUNE_SCORES`, the host forward reads `ALLPACA_Q4X_TEXTS`
    /// and writes it (`blk expert picks sum_weight sum_weight_x_norm` per
    /// line). With it, every paragraph of `ALLPACA_Q4X_HELDOUT` is read
    /// whole and with each block's least salient fraction of experts masked
    /// (`ALLPACA_Q4X_PRUNE`, e.g. `0.125,0.25,0.5`), by REAP saliency (mean
    /// weight x output norm over its picks) and by pick count, and the mean
    /// KL to the unmasked reading and top-1 agreement are printed.
    #[test]
    fn real_qwen4exp_expert_pruning_when_env_set() {
        let (Ok(path), Ok(scores_path)) = (std::env::var("ALLPACA_QWEN4EXP"), std::env::var("ALLPACA_Q4X_PRUNE_SCORES"))
        else {
            return;
        };
        let n: usize = std::env::var("ALLPACA_Q4X_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(192);
        let file = Arc::new(GgufFile::open(std::path::Path::new(&path)).expect("open"));
        let tok = crate::tokenizer::EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).expect("tokenizer"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let (n_experts, n_blocks, vocab) = (model.config.n_experts, model.config.n_layers, model.config.vocab_size);
        let paragraphs = |var: &str| -> Vec<Vec<u32>> {
            let text = std::fs::read_to_string(std::env::var(var).expect(var)).expect("texts").replace("\r\n", "\n");
            text.split("\n\n")
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(|t| tok.encode_plain(t).expect("encode").into_iter().take(n).collect())
                .collect()
        };
        let scores_file = std::path::Path::new(&scores_path);
        if !scores_file.exists() {
            let mut sal = vec![[0.0f64; 3]; n_blocks * n_experts];
            let mut logits = vec![0.0f32; vocab];
            for (pi, ids) in paragraphs("ALLPACA_Q4X_TEXTS").iter().enumerate() {
                let t = std::time::Instant::now();
                let mut mem = Qwen4Memory { saliency: Some(std::mem::take(&mut sal)), ..Default::default() };
                for (pos, token) in ids.iter().enumerate() {
                    forward_qwen4_token(&model, *token, pos, &mut logits, &mut mem).expect("forward");
                }
                sal = mem.saliency.take().unwrap();
                eprintln!("Q4X prune score: paragraph {pi}, {} tokens, {:.0}s", ids.len(), t.elapsed().as_secs_f32());
            }
            let mut lines = String::new();
            for (i, [count, w, wn]) in sal.iter().enumerate() {
                lines += &format!("{}\t{}\t{count}\t{w}\t{wn}\n", i / n_experts, i % n_experts);
            }
            std::fs::write(scores_file, lines).expect("write scores");
            return;
        }
        let mut sal = vec![[0.0f64; 3]; n_blocks * n_experts];
        for line in std::fs::read_to_string(scores_file).expect("scores").lines() {
            let f: Vec<f64> = line.split('\t').map(|v| v.parse().expect("number")).collect();
            sal[f[0] as usize * n_experts + f[1] as usize] = [f[2], f[3], f[4]];
        }
        let fractions: Vec<f64> = std::env::var("ALLPACA_Q4X_PRUNE")
            .unwrap_or_else(|_| "0.125,0.25,0.5".into())
            .split(',')
            .map(|v| v.parse().expect("fraction"))
            .collect();
        // Each block's `fraction` of experts with the lowest `key` (ties: lower index first).
        let mask_by = |fraction: f64, key: &dyn Fn(&[f64; 3]) -> f64| -> Arc<Vec<bool>> {
            let mut mask = vec![false; n_blocks * n_experts];
            let cut = ((fraction * n_experts as f64).round() as usize).min(n_experts);
            for blk in 0..n_blocks {
                let row = &sal[blk * n_experts..(blk + 1) * n_experts];
                let mut order: Vec<usize> = (0..n_experts).collect();
                order.sort_by(|a, b| key(&row[*a]).total_cmp(&key(&row[*b])).then(a.cmp(b)));
                for e in &order[..cut] {
                    mask[blk * n_experts + e] = true;
                }
            }
            Arc::new(mask)
        };
        let reap = |s: &[f64; 3]| if s[0] > 0.0 { s[2] / s[0] } else { 0.0 };
        let freq = |s: &[f64; 3]| s[0];
        let mut runs: Vec<(String, Arc<Vec<bool>>)> = Vec::new();
        for f in &fractions {
            runs.push((format!("reap {f}"), mask_by(*f, &reap)));
            runs.push((format!("freq {f}"), mask_by(*f, &freq)));
        }
        attach_from_env(&mut model);
        let log_softmax = |row: &[f32]| -> Vec<f64> {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = row.iter().map(|v| (*v as f64 - max).exp()).sum();
            row.iter().map(|v| *v as f64 - max - sum.ln()).collect()
        };
        let top = |v: &[f64]| (0..v.len()).max_by(|x, y| v[*x].total_cmp(&v[*y])).unwrap_or(0);
        let mut logits = vec![0.0f32; vocab];
        let mut totals = vec![(0.0f64, 0usize, 0usize); runs.len()];
        for (pi, ids) in paragraphs("ALLPACA_Q4X_HELDOUT").iter().enumerate() {
            let mut reference = vec![0.0f32; ids.len() * vocab];
            forward_qwen4_tokens(&model, ids, 0, &mut logits, Some(&mut reference), &mut Qwen4Memory::default())
                .expect("reference");
            let reference: Vec<Vec<f64>> = reference.chunks(vocab).map(log_softmax).collect();
            let mut line = format!("Q4X prune paragraph {pi} ({} tokens):", ids.len());
            for ((name, mask), total) in runs.iter().zip(totals.iter_mut()) {
                let mut rows = vec![0.0f32; ids.len() * vocab];
                let mut mem = Qwen4Memory { expert_mask: Some(mask.clone()), ..Default::default() };
                forward_qwen4_tokens(&model, ids, 0, &mut logits, Some(&mut rows), &mut mem).expect("masked");
                let mut kl = 0.0f64;
                for (p, q) in reference.iter().zip(rows.chunks(vocab).map(log_softmax)) {
                    kl += p.iter().zip(&q).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
                    total.2 += usize::from(top(p) == top(&q));
                }
                total.0 += kl;
                total.1 += ids.len();
                line += &format!(" | {name} {:.4}", kl / ids.len() as f64);
            }
            eprintln!("{line}");
        }
        for ((name, mask), (kl, count, agree)) in runs.iter().zip(&totals) {
            let pruned = mask.iter().filter(|m| **m).count();
            eprintln!(
                "Q4X prune {name}: {pruned} experts masked, KL {:.4} nats/token, top-1 {:.1}% over {count} tokens",
                kl / *count as f64,
                100.0 * *agree as f64 / *count as f64
            );
        }
    }

    /// Opt-in: greedy continuation of a prompt on a real qwen4exp file, to
    /// compare with llama.cpp. `ALLPACA_QWEN4EXP` is the first shard's path,
    /// `ALLPACA_QWEN4EXP_PROMPT` the raw prompt, `ALLPACA_QWEN4EXP_N` the
    /// number of tokens to generate.
    #[test]
    fn real_qwen4exp_greedy_continuation_when_env_set() {
        let Ok(path) = std::env::var("ALLPACA_QWEN4EXP") else {
            return;
        };
        let prompt = std::env::var("ALLPACA_QWEN4EXP_PROMPT").unwrap_or_else(|_| "The capital of France is".into());
        let n: usize = std::env::var("ALLPACA_QWEN4EXP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        let file = Arc::new(GgufFile::open(std::path::Path::new(&path)).expect("open"));
        let tok = crate::tokenizer::EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).expect("tokenizer"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        attach_from_env(&mut model);
        let mut ids = tok.encode_plain(&prompt).expect("encode");
        eprintln!("Q4X prompt tokens {ids:?}");
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; model.config.vocab_size];
        let start = std::time::Instant::now();
        let mut generating = start;
        mem.times = Some(Qwen4Times::default());
        // `ALLPACA_Q4X_ROWS_CHECK`: the prompt in one run against one token at a time.
        if std::env::var_os("ALLPACA_Q4X_ROWS_CHECK").is_some() {
            let vocab = model.config.vocab_size;
            let mut rows = vec![0.0f32; ids.len() * vocab];
            forward_qwen4_tokens(&model, &ids, 0, &mut logits, Some(&mut rows), &mut Qwen4Memory::default()).expect("run");
            let mut one = Qwen4Memory::default();
            for (i, id) in ids.iter().enumerate() {
                forward_qwen4_token(&model, *id, i, &mut logits, &mut one).expect("token");
                let diff = max_diff(&logits, &rows[i * vocab..(i + 1) * vocab]);
                if diff != 0.0 || i + 1 == ids.len() {
                    eprintln!("Q4X rows check: token {i} differs by {diff}");
                }
            }
        }
        // All of the prompt but its last token in one run.
        let mut pos = ids.len() - 1;
        forward_qwen4_tokens(&model, &ids[..pos], 0, &mut logits, None, &mut mem).expect("prompt");
        if pos > 0 {
            let secs = start.elapsed().as_secs_f32();
            eprintln!("Q4X prompt: {pos} tokens in {secs:.2}s, {:.1} tok/s", pos as f32 / secs);
            eprintln!("Q4X prompt seconds: {:?}", mem.times);
        }
        let mut out = Vec::new();
        while out.len() < n {
            let token = ids[pos];
            let t = std::time::Instant::now();
            forward_qwen4_token(&model, token, pos, &mut logits, &mut mem).expect("forward");
            pos += 1;
            if pos == ids.len() {
                // Only the top five are needed; sorting the whole vocabulary
                // took as long as the forward on a slow CPU.
                let mut order: Vec<usize> = (0..logits.len()).collect();
                let by_logit = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
                order.select_nth_unstable_by(4, by_logit);
                order[..5].sort_by(by_logit);
                let top: Vec<(usize, f32, String)> =
                    order[..5].iter().map(|i| (*i, logits[*i], tok.decode_one(*i as u32))).collect();
                eprintln!("Q4X pos {pos} ({:.3}s) top {top:?}", t.elapsed().as_secs_f32());
                if out.is_empty() {
                    generating = std::time::Instant::now();
                    mem.times = Some(Qwen4Times::default());
                }
                ids.push(order[0] as u32);
                out.push(order[0] as u32);
            }
        }
        eprintln!(
            "Q4X generated {out:?} = {:?} in {:.1}s, {:.2} tok/s after the prompt",
            tok.decode(&out).unwrap_or_default(),
            start.elapsed().as_secs_f32(),
            out.len().saturating_sub(1) as f32 / generating.elapsed().as_secs_f32().max(1e-6)
        );
        eprintln!("Q4X seconds over {} tokens after the prompt: {:?}", out.len().saturating_sub(1), mem.times);
        for (kind, calls, seconds) in crate::qwen4_gpu::profile_report() {
            eprintln!("Q4X matrix {kind}: {calls} calls, {:.1} us each, {:.3} s", seconds * 1e6 / calls as f64, seconds);
        }
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// Opt-in: the experts a real qwen4exp file routes to while reading
    /// texts. `ALLPACA_QWEN4EXP` is the first shard, `ALLPACA_Q4X_TEXTS` a
    /// file of paragraphs (blank-line separated), `ALLPACA_Q4X_TOKENS` the
    /// tokens read per paragraph, `ALLPACA_Q4X_ROUTES` the output: one line
    /// per paragraph of `layer:expert` pairs.
    #[test]
    fn real_qwen4exp_routes_when_env_set() {
        let (Ok(path), Ok(texts), Ok(out)) = (
            std::env::var("ALLPACA_QWEN4EXP"),
            std::env::var("ALLPACA_Q4X_TEXTS"),
            std::env::var("ALLPACA_Q4X_ROUTES"),
        ) else {
            return;
        };
        let n: usize = std::env::var("ALLPACA_Q4X_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
        let file = Arc::new(GgufFile::open(std::path::Path::new(&path)).expect("open"));
        let tok = crate::tokenizer::EngineTokenizer::from_gguf(allpaca_gguf::extract_tokenizer(&file).expect("tokenizer"));
        let mut model = TinyModel::from_gguf(&file).expect("model");
        model.bind_gguf(file);
        let texts = std::fs::read_to_string(texts).expect("texts");
        let mut logits = vec![0.0f32; model.config.vocab_size];
        let mut lines = String::new();
        for (ti, text) in texts.split("\n\n").map(str::trim).filter(|t| !t.is_empty()).enumerate() {
            let ids = tok.encode_plain(text).expect("encode");
            let mut mem = Qwen4Memory { routes: Some(Vec::new()), ..Default::default() };
            let t = std::time::Instant::now();
            for (pos, token) in ids.iter().take(n).enumerate() {
                forward_qwen4_token(&model, *token, pos, &mut logits, &mut mem).expect("forward");
            }
            let routes = mem.routes.take().unwrap_or_default();
            eprintln!("Q4X text {ti}: {} tokens, {} routes, {:.1}s", ids.len().min(n), routes.len(), t.elapsed().as_secs_f32());
            let line: Vec<String> = routes.iter().map(|(l, e)| format!("{l}:{e}")).collect();
            lines += &line.join(" ");
            lines.push('\n');
            std::fs::write(&out, &lines).expect("write routes");
        }
    }
}
