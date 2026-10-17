//! `qwen4exp` (Qwen3.8-Flash-Next): the Qwen3-Next trunk (Gated DeltaNet
//! layers with a full-attention layer every few, routed plus shared experts)
//! on a wide residual of `hc` parallel streams mixed by low-rank
//! hyper-connections in place of layer norms, with n-gram hash embeddings
//! (PLE) added at one layer. Follows llama.cpp's `src/models/qwen4exp.cpp`.
//!
//! Streams are stored stream-major: `x[s * dim + d]`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use allpaca_cpu::{attention, matmul_nt, rope_imrope_neox};
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

/// The model's weights, read from the file as they are needed.
pub struct Qwen4Weights {
    pub spec: Qwen4Spec,
    file: Option<Arc<GgufFile>>,
    /// Dense tensors decoded to F32 on first use.
    dense: Mutex<HashMap<String, Arc<Vec<f32>>>>,
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
            dense: Mutex::new(HashMap::new()),
            gpu: Mutex::new(None),
        })
    }

    pub fn attach_gguf(&mut self, file: Arc<GgufFile>) {
        self.file = Some(file);
    }

    fn file(&self) -> Result<&GgufFile, AllpacaError> {
        self.file.as_deref().ok_or_else(|| load_err("qwen4exp weights have no GGUF handle".into()))
    }

    pub(crate) fn gguf(&self) -> Result<&GgufFile, AllpacaError> {
        self.file()
    }

    fn has(&self, name: &str) -> bool {
        self.file.as_ref().is_some_and(|f| f.tensor(name).is_some())
    }

    /// A whole tensor as F32, decoded once.
    fn w(&self, name: &str) -> Result<Arc<Vec<f32>>, AllpacaError> {
        let mut cache = self.dense.lock().map_err(|_| load_err("qwen4exp weight cache poisoned".into()))?;
        if let Some(w) = cache.get(name) {
            return Ok(w.clone());
        }
        let w = Arc::new(self.file()?.tensor_f32(name)?);
        cache.insert(name.to_string(), w.clone());
        Ok(w)
    }

    /// `n_rows` rows of `row_len` weights starting at row `first`, decoded
    /// from the packed tensor (rows must be whole quantisation blocks).
    fn rows(&self, name: &str, first: usize, n_rows: usize, row_len: usize) -> Result<Vec<f32>, AllpacaError> {
        let file = self.file()?;
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
        let mut out = vec![0.0f32; n];
        dequant_packed(info.ggml_type, bytes, n, &mut out)?;
        Ok(out)
    }
}

impl Qwen4Weights {
    /// The blocks of `n_rows` rows of `row_len` weights starting at row
    /// `first` of a `Q2_0` tensor; `None` for another encoding or rows that
    /// are not whole blocks.
    fn q2_0_rows(&self, name: &str, first: usize, n_rows: usize, row_len: usize) -> Result<Option<&[u8]>, AllpacaError> {
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
fn swiglu_q2_0(x: &[f32], gate: &[u8], up: &[u8], down: &[u8], ff: usize, dim: usize) -> Result<Vec<f32>, AllpacaError> {
    let (mut h, mut u, mut y) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; dim]);
    allpaca_cpu::gemv_q2_0(gate, x, &mut h)?;
    allpaca_cpu::gemv_q2_0(up, x, &mut u)?;
    for (h, u) in h.iter_mut().zip(&u) {
        *h = silu(*h) * u;
    }
    allpaca_cpu::gemv_q2_0(down, &h, &mut y)?;
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
    /// When set, every routed (layer, expert) pair is appended, in order.
    pub routes: Option<Vec<(u16, u16)>>,
    /// When set, seconds spent per part of the forward are added up.
    pub times: Option<Qwen4Times>,
    /// The recurrent states, keys and values when the forward runs on the GPU.
    device: Option<Qwen4DeviceState>,
    /// The same read back to the host (a prompt-cache copy), put on the
    /// cards by the next GPU forward.
    device_host: Option<Qwen4DeviceHost>,
    /// This sequence's state on the worker that hosts blocks, when one does.
    remote: Option<crate::qwen4_remote::RemoteSeq>,
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
    pub fn tokens(&self) -> usize {
        self.tokens.len()
    }

    /// A copy for the prompt cache that holds nothing on the cards: the
    /// device state is read back to the host. `None` when part of the
    /// sequence lives on a worker that hosts blocks.
    pub(crate) fn for_cache(&self, model: &TinyModel) -> Result<Option<Self>, AllpacaError> {
        if self.remote.is_some() {
            return Ok(None);
        }
        let device_host = match &self.device {
            Some(device) => {
                if let Some(q4) = model.q4x.as_ref() {
                    if let Some(gpu) = q4.gpu.lock().map_err(|_| load_err("qwen4exp GPU lock".into()))?.as_ref() {
                        gpu.sync_all()?;
                    }
                }
                Some(device.to_host(self.tokens.len())?)
            }
            None => self.device_host.clone(),
        };
        Ok(Some(Self {
            tokens: self.tokens.clone(),
            layers: self.layers.clone(),
            ple_history: self.ple_history.clone(),
            routes: None,
            times: None,
            device: None,
            device_host,
            remote: None,
        }))
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
        out.push(self.device_host.is_some() as u8);
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
/// token's result is the same as on its own.
fn routed_experts_pooled(
    q4: &Arc<Qwen4Weights>,
    cfg: &ModelConfig,
    blk: usize,
    xs: &[f32],
    picks: &[Vec<(usize, f32)>],
) -> Result<Vec<f32>, AllpacaError> {
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
                let ys = (|| -> Result<Vec<Vec<f32>>, AllpacaError> {
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
                        return Ok(y.chunks(dim).map(<[f32]>::to_vec).collect());
                    }
                    let gate = q4.rows(&name("ffn_gate_exps.weight"), e * ff, ff, dim)?;
                    let up = q4.rows(&name("ffn_up_exps.weight"), e * ff, ff, dim)?;
                    let down = q4.rows(&name("ffn_down_exps.weight"), e * dim, dim, ff)?;
                    xs.iter().map(|x| swiglu(x, &gate, &up, &down, ff, dim)).collect()
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
    for (users, y) in results {
        for ((slot, _), y) in users.into_iter().zip(y?) {
            ys[slot] = Some(y);
        }
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
    let (dim, ff) = (cfg.dim, cfg.ff_dim.max(1));
    let name = |s: &str| format!("blk.{blk}.{s}");
    let mut out = vec![0.0f32; dim];
    // One thread per routed expert; the sum keeps the routing order.
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
    for ((_, weight), y) in picks.iter().zip(ys) {
        for (o, y) in out.iter_mut().zip(&y?) {
            *o += weight * y;
        }
    }
    Ok(out)
}

/// Routed experts (softmax over all, the top few renormalised) plus the
/// shared expert behind its own sigmoid gate.
fn ffn_layer(
    q4: &Qwen4Weights,
    cfg: &ModelConfig,
    blk: usize,
    x: &[f32],
    routes: Option<&mut Vec<(u16, u16)>>,
) -> Result<Vec<f32>, AllpacaError> {
    let dim = cfg.dim;
    let name = |s: &str| format!("blk.{blk}.{s}");
    let router = q4.w(&name("ffn_gate_inp.weight"))?;
    let mut scores = gemv(x, &router, cfg.n_experts)?;
    let picks = topk_from_logits(&mut scores, cfg.n_active, cfg, None, &[]);
    if let Some(routes) = routes {
        routes.extend(picks.iter().map(|(e, _)| (blk as u16, *e as u16)));
    }
    let mut out = routed_experts(q4, cfg, blk, x, &picks)?;
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
pub(crate) fn forward_qwen4_tokens(
    model: &TinyModel,
    tokens: &[u32],
    start_pos: usize,
    logits: &mut [f32],
    mut row_logits: Option<&mut [f32]>,
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
    let gpu = q4.gpu.lock().map_err(|_| load_err("qwen4exp gpu lock poisoned".into()))?;
    if let Some(gpu) = gpu.as_ref() {
        let rpc = model.rpc.as_deref();
        if let Some(host) = gpu.hosted().filter(|_| row_logits.is_none() && tokens.len() > crate::qwen4_gpu::MAX_ROWS) {
            let host = host.clone();
            return forward_pipelined(q4, gpu, rpc, &host, cfg, tokens, start_pos, logits, mem);
        }
        for (pass, chunk) in tokens.chunks(crate::qwen4_gpu::MAX_ROWS).enumerate() {
            let first = pass * crate::qwen4_gpu::MAX_ROWS;
            let rows = row_logits.as_deref_mut().map(|rows| &mut rows[first * vocab..(first + chunk.len()) * vocab]);
            forward_device(q4, gpu, rpc, cfg, chunk, start_pos + first, &mut logits[..vocab], rows, mem, None)?;
        }
        return Ok(());
    }
    for (i, &token) in tokens.iter().enumerate() {
        forward_host(q4, cfg, token, start_pos + i, logits, mem)?;
        mem.tokens.push(i64::from(token));
        if let Some(rows) = row_logits.as_deref_mut() {
            rows[i * vocab..(i + 1) * vocab].copy_from_slice(&logits[..vocab]);
        }
    }
    Ok(())
}

/// A prompt through a worker that hosts blocks, in passes that overlap:
/// while the worker runs one pass's hosted blocks, this machine runs the
/// next pass's blocks up to them. Each block still sees the passes in
/// order, so the result is that of one pass after the other. A pass goes
/// to the worker only once the one before is back, so a single pass is in
/// flight whatever its size. `logits` gets the last token's.
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
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    let (vocab, rows) = (cfg.vocab_size, crate::qwen4_gpu::MAX_ROWS);
    host.pipeline(|pipe| {
        // The pass whose hosted blocks the worker has, with its tokens and position.
        let mut in_flight: Option<(&[u32], usize, DevicePass)> = None;
        let finish = |pipe: &mut crate::qwen4_remote::Pipe<'_>, mem: &mut Qwen4Memory, logits: &mut [f32], done: Option<(&[u32], usize, DevicePass)>| {
            let Some((chunk, pos, mut pass)) = done else {
                return Ok(());
            };
            // What this machine waits for the worker shows as `ffn_rpc`.
            let clock = std::time::Instant::now();
            pipe.recv(&mut pass.streams)?;
            if let Some(times) = mem.times.as_mut() {
                times.ffn_rpc += clock.elapsed().as_secs_f32();
            }
            match forward_device(q4, gpu, rpc, cfg, chunk, pos, &mut logits[..vocab], None, mem, Some(Some(pass)))? {
                Some(_) => Err(load_err("qwen4exp: a second run of hosted blocks in one pass".into())),
                None => Ok(()),
            }
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
    let embed = q4.rows("token_embd.weight", token as usize, 1, dim)?;
    let mut streams = embed.repeat(hc);
    for layer in 0..cfg.n_layers {
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
        let out = if q4.spec.is_recurrent(blk) {
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
        let out = ffn_layer(q4, cfg, blk, &x, mem.routes.as_mut())?;
        lap(&mut times.ffn_host);
        hc_combine(&mut streams, &out, &inject, dim);
        lap(&mut times.combine);
    }
    // The output mixer stands in for the final norm.
    let (x, _) = mix(q4, "output_hc", false, &streams, dim, eps)?;
    let head = if q4.has("output.weight") { "output.weight" } else { "token_embd.weight" };
    matmul_nt(&x, &q4.w(head)?, logits, 1, cfg.vocab_size, dim)?;
    lap(&mut times.head);
    if mem.times.is_some() {
        mem.times = Some(times);
    }
    Ok(())
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
                let embed = q4.rows("token_embd.weight", token as usize, 1, dim)?;
                for _ in 0..hc {
                    streams.extend_from_slice(&embed);
                }
            }
            (streams, gpu.card_of(cfg.gguf_blk(0)), 0)
        }
    };
    gpu.set_streams(card, &streams)?;
    let mut scores = vec![0.0f32; rows * n_experts];
    let mut x = Vec::new();
    // Blocks with their experts on a card pick them there, so nothing of
    // the block comes back to the host. `ALLPACA_Q4X_HOST_ROUTING` and the
    // route recorder keep the picking on the host.
    let card_scale = crate::model::plain_softmax_routing(cfg)
        .filter(|_| mem.routes.is_none() && std::env::var_os("ALLPACA_Q4X_HOST_ROUTING").is_none());
    for layer in 0..cfg.n_layers {
        if layer < resume {
            continue;
        }
        let blk = cfg.gguf_blk(layer);
        // The blocks a worker hosts run there in one round trip.
        if let Some(host) = gpu.hosted().filter(|h| h.blocks().first() == Some(&blk)) {
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
            host.run(seq, pos, rows, &mut streams)?;
            gpu.set_streams(card, &streams)?;
            lap(card, &mut times.ffn_rpc)?;
            continue;
        }
        let home = gpu.card_of(blk);
        let ple = q4.spec.ple.as_ref().filter(|p| p.layers.contains(&blk));
        if home != card || ple.is_some() {
            gpu.get_streams(card, &mut streams)?;
            if let Some(ple) = ple {
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
        lap(card, &mut times.embed_ple)?;
        gpu.mix(card, &format!("blk.{blk}.hc_attn"), true, rows)?;
        if wait {
            lap(card, &mut times.mix)?;
        }
        let recurrent = q4.spec.is_recurrent(blk);
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
            continue;
        }
        gpu.ffn_route(blk, rows, Some(&mut scores), &mut x)?;
        lap(card, if wait { &mut times.route } else if recurrent { &mut times.gdn } else { &mut times.attention })?;
        let picks: Vec<Vec<(usize, f32)>> = scores
            .chunks_mut(n_experts.max(1))
            .map(|row| topk_from_logits(row, cfg.n_active, cfg, None, &[]))
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
            let sums = routed_experts_pooled(q4, cfg, blk, &x, &picks)?;
            gpu.ffn_finish(blk, &picks, Some(&sums))?;
            lap(card, &mut times.ffn_host)?;
        }
        gpu.combine(card, rows)?;
    }
    if card != 0 {
        gpu.get_streams(card, &mut streams)?;
        gpu.set_streams(0, &streams)?;
    }
    // The output mixer stands in for the final norm.
    gpu.mix(0, "output_hc", false, rows)?;
    let head = if q4.has("output.weight") { "output.weight" } else { "token_embd.weight" };
    match row_logits {
        Some(all) => {
            gpu.head(head, rows, &mut all[..rows * vocab])?;
            logits[..vocab].copy_from_slice(&all[(rows - 1) * vocab..rows * vocab]);
        }
        None => gpu.head(head, rows, &mut logits[..vocab])?,
    }
    lap(0, &mut times.head)?;
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
pub fn write_tiny_qwen4exp_gguf(path: &std::path::Path) {
    write_tiny(path, 16, 8, false, None);
}

/// The tiny model with an indexer on its full-attention layer: pools of
/// two cells, of which the attention keeps `top_k / 2` and the tail.
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

fn write_tiny(path: &std::path::Path, dim: usize, ff: usize, packed: bool, indexer: Option<usize>) {
    use allpaca_gguf::{write_gguf_quant, GgmlType, GgufWriteQuantTensor, GgufWriteTensor};

    let (hc, lr, vocab, n_layers) = (2usize, 4usize, 8usize, 4usize);
    let (n_heads, head_dim, n_rot) = (2usize, 8usize, 4usize);
    let (state, n_k, n_v, conv_kernel) = (4usize, 1usize, 2usize, 2usize);
    let (key_dim, value_dim) = (state * n_k, state * n_v);
    let qkv_dim = key_dim * 2 + value_dim;
    let n_exp = 4usize;
    let (ngram, heads_per, ple_dim, ple_kernel) = (3usize, 2usize, 4usize, 2usize);
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
            crate::qwen4_gpu::attach(&gpu, &hip, None, None).expect("attach");
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
        crate::qwen4_gpu::attach(&model, &hip, None, None).expect("attach");
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
        crate::qwen4_gpu::attach(&mixed, &hip, None, Some(&[2])).expect("attach");
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
            crate::qwen4_gpu::attach(&gpu, &hip, None, None).expect("attach");
            crate::qwen4_gpu::attach(&dense, &hip, None, None).expect("attach");
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
        crate::qwen4_gpu::attach(&local, &hip, None, None).expect("attach");
        let on_card = run(&local, &tokens);

        let mut remote = tiny_packed_sparse_model(8);
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2])).expect("attach");
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
        crate::qwen4_gpu::attach(&local, &hip, None, None).expect("attach");
        let on_card = run(&local, &tokens);

        let mut remote = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2])).expect("attach");
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
        crate::qwen4_gpu::attach(&local, &hip, None, None).expect("attach");
        let on_card = run(&local, &tokens);

        let mut remote = tiny_packed_model();
        let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[addr], None).expect("connect"));
        remote.rpc = Some(rt.clone());
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2])).expect("attach");
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
        crate::qwen4_gpu::attach(&remote, &hip, None, Some(&[2])).expect("attach");
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
        crate::qwen4_gpu::attach(&model, &hip, None, Some(&[2])).expect("attach");
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
            crate::qwen4_gpu::attach(&model, &hip, None, Some(&[2])).expect("attach");
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
            crate::qwen4_gpu::attach(&model, &hip, None, None).expect("attach");
            carries_on_from_the_cache_copy(&model, what);
        }
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
        let mut step = |emb: f32, history: &mut Vec<f32>| {
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
        // `ALLPACA_Q4X_GPU` = number of cards to put matrices and experts on.
        #[cfg(feature = "hip")]
        if let Some(cards) = std::env::var("ALLPACA_Q4X_GPU").ok().and_then(|v| v.parse::<i32>().ok()) {
            let load = std::time::Instant::now();
            let vocab = model.config.vocab_size;
            let first = Arc::new(allpaca_hip::HipWorkspace::new(0, 1 << 20, vocab).expect("card 0"));
            let second = (cards > 1).then(|| Arc::new(allpaca_hip::HipWorkspace::new(1, 1 << 20, vocab).expect("card 1")));
            // `ALLPACA_Q4X_RPC` = worker (`host:port`, token in `ALLPACA_RPC_TOKEN`) for the experts off the cards.
            if let Ok(endpoint) = std::env::var("ALLPACA_Q4X_RPC") {
                let token = std::env::var("ALLPACA_RPC_TOKEN").ok();
                let rt = crate::rpc_runtime::RpcRuntime::connect(&[endpoint], token.as_deref()).expect("rpc worker");
                model.rpc = Some(Arc::new(rt));
            }
            crate::qwen4_gpu::attach(&model, &first, second.as_ref(), None).expect("attach");
            let gpu = model.q4x.as_ref().unwrap().gpu.lock().unwrap();
            eprintln!(
                "Q4X gpu attached in {:.1}s, expert layers per card {:?}",
                load.elapsed().as_secs_f32(),
                gpu.as_ref().unwrap().expert_layers().iter().map(Vec::len).collect::<Vec<_>>()
            );
        }
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
                let mut order: Vec<usize> = (0..logits.len()).collect();
                order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
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
