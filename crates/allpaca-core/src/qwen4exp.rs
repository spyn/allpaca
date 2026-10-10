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
use crate::model::{topk_route, ModelConfig, TinyModel};

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
    let mut mixed = vec![0.0f32; dim];
    for (xs, gs) in xn.chunks(dim).zip(gate.chunks(dim)) {
        for ((m, x), g) in mixed.iter_mut().zip(xs).zip(gs) {
            *m += x * sigmoid(*g);
        }
    }
    for m in &mut mixed {
        *m /= hc as f32;
    }
    let mut inject = vec![0.0f32; if w.inject.is_some() { hc } else { 0 }];
    if let Some(w_inject) = w.inject {
        matmul_nt(&xn, w_inject, &mut inject, 1, hc, hc_dim)?;
    }
    Ok((mixed, inject))
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
        })
    }

    pub fn attach_gguf(&mut self, file: Arc<GgufFile>) {
        self.file = Some(file);
    }

    fn file(&self) -> Result<&GgufFile, AllpacaError> {
        self.file.as_deref().ok_or_else(|| load_err("qwen4exp weights have no GGUF handle".into()))
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

/// One sequence's state at one layer.
#[derive(Debug, Clone, Default)]
struct LayerState {
    gdn_state: Vec<f32>,
    gdn_conv: Vec<f32>,
    /// Keys and values of a full-attention layer, one row per token.
    k: Vec<f32>,
    v: Vec<f32>,
}

/// One sequence's state: the recurrent layers' states, the attention
/// layers' keys and values, the tokens so far (the n-gram hash looks back)
/// and the n-gram layer's conv history.
#[derive(Debug, Clone, Default)]
pub struct Qwen4Memory {
    tokens: Vec<i64>,
    layers: Vec<LayerState>,
    ple_history: Vec<f32>,
}

impl Qwen4Memory {
    pub fn tokens(&self) -> usize {
        self.tokens.len()
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
    // The indexer keeps every block while no more than `top_k` cells exist.
    let ratio = q4.spec.compress_ratios.get(blk).copied().unwrap_or(0) as usize;
    if ratio > 0 && q4.spec.indexer_top_k > 0 && pos >= q4.spec.indexer_top_k {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "qwen4exp sparse attention past {} tokens is not implemented",
            q4.spec.indexer_top_k
        ))));
    }
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
    let mut out = vec![0.0f32; n_heads * head_dim];
    attention(&q, &state.k, &state.v, &mut out, 1, pos + 1, n_heads, n_kv, head_dim, true)?;
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

/// Routed experts (softmax over all, the top few renormalised) plus the
/// shared expert behind its own sigmoid gate.
fn ffn_layer(q4: &Qwen4Weights, cfg: &ModelConfig, blk: usize, x: &[f32]) -> Result<Vec<f32>, AllpacaError> {
    let (dim, ff) = (cfg.dim, cfg.ff_dim.max(1));
    let name = |s: &str| format!("blk.{blk}.{s}");
    let router = q4.w(&name("ffn_gate_inp.weight"))?;
    let picks = topk_route(x, &router, cfg.n_experts, cfg.n_active, cfg, None, &[]);
    let mut out = vec![0.0f32; dim];
    for (e, weight) in picks {
        let gate = q4.rows(&name("ffn_gate_exps.weight"), e * ff, ff, dim)?;
        let up = q4.rows(&name("ffn_up_exps.weight"), e * ff, ff, dim)?;
        let down = q4.rows(&name("ffn_down_exps.weight"), e * dim, dim, ff)?;
        let y = swiglu(x, &gate, &up, &down, ff, dim)?;
        for (o, y) in out.iter_mut().zip(&y) {
            *o += weight * y;
        }
    }
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
    let hc_dim = streams.len();
    let key = grouped_norm(&gemv(emb, key_w, hc_dim)?, norms[0], dim, eps);
    let query = grouped_norm(streams, norms[1], dim, eps);
    let value = gemv(emb, value_w, dim)?;
    let mut gated = vec![0.0f32; hc_dim];
    for ((dst, k), q) in gated.chunks_mut(dim).zip(key.chunks(dim)).zip(query.chunks(dim)) {
        let s = k.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() / (dim as f32).sqrt();
        // Signed square root before the sigmoid.
        let signed = if s == 0.0 { 0.0 } else { s.signum() * s.abs().max(1e-6).sqrt() };
        let open = sigmoid(signed);
        for (d, v) in dst.iter_mut().zip(&value) {
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

/// Forward one token at `pos`; `logits` gets the next-token logits.
pub(crate) fn forward_qwen4_token(
    model: &TinyModel,
    token: u32,
    pos: usize,
    logits: &mut [f32],
    mem: &mut Qwen4Memory,
) -> Result<(), AllpacaError> {
    let q4 = model.q4x.as_ref().ok_or_else(|| load_err("missing qwen4exp weights".into()))?;
    let cfg = &model.config;
    let (dim, hc, eps) = (cfg.dim, q4.spec.hc, cfg.eps);
    if mem.tokens.len() != pos {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "qwen4exp sequence memory holds {} tokens, forward is at position {pos}",
            mem.tokens.len()
        ))));
    }
    mem.layers.resize_with(cfg.n_layers, LayerState::default);
    let embed = q4.rows("token_embd.weight", token as usize, 1, dim)?;
    let mut streams = embed.repeat(hc);
    for layer in 0..cfg.n_layers {
        let blk = cfg.gguf_blk(layer);
        if let Some(ple) = q4.spec.ple.as_ref().filter(|p| p.layers.contains(&blk)) {
            let prev: Vec<i64> = mem.tokens.iter().rev().take(ple.ngram_size - 1).copied().collect();
            let mut emb = Vec::with_capacity(ple.n_heads() * ple.head_dim);
            for row in ple.rows(i64::from(token), &prev) {
                emb.extend(q4.rows("per_layer_token_embd.weight", row as usize, 1, ple.head_dim)?);
            }
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
        let (x, inject) = mix(q4, &format!("blk.{blk}.hc_attn"), true, &streams, dim, eps)?;
        let out = if q4.spec.is_recurrent(blk) {
            gdn_layer(q4, cfg, blk, &x, &mut mem.layers[layer])?
        } else {
            attn_layer(q4, cfg, blk, &x, pos, &mut mem.layers[layer])?
        };
        hc_combine(&mut streams, &out, &inject, dim);
        let (x, inject) = mix(q4, &format!("blk.{blk}.hc_ffn"), true, &streams, dim, eps)?;
        let out = ffn_layer(q4, cfg, blk, &x)?;
        hc_combine(&mut streams, &out, &inject, dim);
    }
    // The output mixer stands in for the final norm.
    let (x, _) = mix(q4, "output_hc", false, &streams, dim, eps)?;
    let head = if q4.has("output.weight") { "output.weight" } else { "token_embd.weight" };
    matmul_nt(&x, &q4.w(head)?, logits, 1, cfg.vocab_size, dim)?;
    mem.tokens.push(i64::from(token));
    Ok(())
}

/// Tiny `qwen4exp` GGUF for tests: four layers (three Gated DeltaNet, one
/// full attention), two residual streams, n-gram embeddings at layer 1,
/// routed and shared experts. Weights are seeded noise, all F32.
pub fn write_tiny_qwen4exp_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufWriteTensor};

    let (dim, hc, lr, vocab, n_layers) = (16usize, 2usize, 4usize, 8usize, 4usize);
    let (n_heads, head_dim, n_rot) = (2usize, 8usize, 4usize);
    let (state, n_k, n_v, conv_kernel) = (4usize, 1usize, 2usize, 2usize);
    let (key_dim, value_dim) = (state * n_k, state * n_v);
    let qkv_dim = key_dim * 2 + value_dim;
    let (n_exp, ff) = (4usize, 8usize);
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
    // Dims are listed outermost first, as the writer expects.
    let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, dims: &[usize], data: Vec<f32>| {
        assert_eq!(dims.iter().product::<usize>(), data.len(), "{name}");
        owned.push((name, dims.iter().map(|d| *d as u64).collect(), data));
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
    let views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    write_gguf(path, &meta, &views).expect("write tiny qwen4exp");
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

    /// Logits after each token of `tokens`.
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
        let mut ids = tok.encode_plain(&prompt).expect("encode");
        eprintln!("Q4X prompt tokens {ids:?}");
        let mut mem = Qwen4Memory::default();
        let mut logits = vec![0.0f32; model.config.vocab_size];
        let start = std::time::Instant::now();
        let mut pos = 0;
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
                eprintln!("Q4X pos {pos} ({:.1}s) top {top:?}", t.elapsed().as_secs_f32());
                ids.push(order[0] as u32);
                out.push(order[0] as u32);
            }
        }
        eprintln!(
            "Q4X generated {out:?} = {:?} in {:.1}s",
            tok.decode(&out).unwrap_or_default(),
            start.elapsed().as_secs_f32()
        );
        assert!(logits.iter().all(|v| v.is_finite()));
    }
}
