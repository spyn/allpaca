//! The Qwen3.5 / Qwen3.6 (`qwen35`, `qwen35moe`) blocks on the CPU, one
//! token at a time, from host F32 weights: the gated delta-net step of the
//! linear-attention blocks and the gated full attention of every fourth
//! block. The GPU forward has its own; this one serves CPU readings
//! (calibration and the sensitivity probe). It is the qwen4exp host step
//! ([`crate::qwen4exp`]) with Qwen3.5's SiLU output gate and tiled key heads.

use allpaca_cpu::{attention, matmul_nt, rope_imrope_neox};
use allpaca_types::{AllpacaError, FatalError};

use crate::deltanet::{
    conv1d_step, decay_from_alpha, deinterleave_q_gate, expand_k_heads_as, gated_delta_step_eps, rmsnorm_heads, sigmoid,
    silu,
};
use crate::model::{GdnHostWeights, LayerWeights, ModelConfig};

fn gemv(x: &[f32], w: &[f32], n: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; n];
    matmul_nt(x, w, &mut out, 1, n, x.len())?;
    Ok(out)
}

/// The gated delta-net step for one token: `x` is the block's normed
/// input; `state` and `conv` the layer's recurrent state and conv history
/// (sized on first use). Returns what the block adds to the residual.
pub(crate) fn gdn_step(
    gdn: &GdnHostWeights,
    cfg: &ModelConfig,
    x: &[f32],
    state: &mut Vec<f32>,
    conv_history: &mut Vec<f32>,
) -> Result<Vec<f32>, AllpacaError> {
    let dims = &gdn.dims;
    let qkv = gemv(x, &gdn.w_qkv, dims.qkv_dim)?;
    let z = gemv(x, &gdn.w_z, dims.value_dim)?;
    let alpha = gemv(x, &gdn.w_alpha, dims.n_v_heads)?;
    let mut beta = gemv(x, &gdn.w_beta, dims.n_v_heads)?;
    beta.iter_mut().for_each(|b| *b = sigmoid(*b));
    if state.len() != dims.state_floats() {
        *state = vec![0.0; dims.state_floats()];
        *conv_history = vec![0.0; dims.conv_state_floats()];
    }
    let mut conv = vec![0.0f32; dims.qkv_dim];
    conv1d_step(&qkv, &gdn.conv1d, conv_history, &mut conv, dims.conv_kernel);
    let (q_raw, rest) = conv.split_at(dims.key_dim);
    let (k_raw, v) = rest.split_at(dims.key_dim);
    let mut q = vec![0.0f32; dims.n_v_heads * dims.head_k];
    let mut k = vec![0.0f32; dims.n_v_heads * dims.head_k];
    expand_k_heads_as(q_raw, &mut q, dims.n_k_heads, dims.n_v_heads, dims.head_k, dims.k_repeat);
    expand_k_heads_as(k_raw, &mut k, dims.n_k_heads, dims.n_v_heads, dims.head_k, dims.k_repeat);
    let mut decay = vec![0.0f32; dims.n_v_heads];
    decay_from_alpha(&alpha, &gdn.dt_bias, &gdn.a, &mut decay);
    let mut y = vec![0.0f32; dims.value_dim];
    gated_delta_step_eps(
        &mut q,
        &mut k,
        v,
        &decay,
        &beta,
        state,
        &mut y,
        dims.n_v_heads,
        dims.head_v,
        dims.head_k,
        cfg.eps,
    );
    rmsnorm_heads(&mut y, &gdn.norm, dims.head_v, cfg.eps);
    for (y, z) in y.iter_mut().zip(&z) {
        *y *= silu(*z);
    }
    gemv(&y, &gdn.w_out, cfg.dim)
}

/// Whether `layer` is a Qwen3.5 full-attention block: its query projection
/// carries a gate beside each head's query.
pub(crate) fn is_gated_attention(cfg: &ModelConfig, layer: &LayerWeights) -> bool {
    let q_full = 2 * cfg.n_heads * cfg.head_dim() * cfg.dim;
    cfg.uses_imrope() && layer.wq.len() == q_full && layer.q_norm.len() == cfg.head_dim()
}

/// Gated full attention for the token at `pos`: `x` is the block's normed
/// input; `keys` and `values` the layer's cache, which this extends.
pub(crate) fn gated_attention_step(
    layer: &LayerWeights,
    cfg: &ModelConfig,
    x: &[f32],
    pos: usize,
    keys: &mut Vec<f32>,
    values: &mut Vec<f32>,
) -> Result<Vec<f32>, AllpacaError> {
    let (head_dim, n_heads, n_kv) = (cfg.head_dim(), cfg.n_heads, cfg.n_kv_heads.max(1));
    let q_full = gemv(x, &layer.wq, n_heads * head_dim * 2)?;
    let mut q = vec![0.0f32; n_heads * head_dim];
    let mut gate = vec![0.0f32; n_heads * head_dim];
    deinterleave_q_gate(&q_full, &mut q, &mut gate, head_dim);
    rmsnorm_heads(&mut q, &layer.q_norm, head_dim, cfg.eps);
    let mut k = gemv(x, &layer.wk, n_kv * head_dim)?;
    rmsnorm_heads(&mut k, &layer.k_norm, head_dim, cfg.eps);
    let v = gemv(x, &layer.wv, n_kv * head_dim)?;
    let n_rot = if cfg.rope_dims > 0 { cfg.rope_dims } else { head_dim };
    for head in q.chunks_mut(head_dim).chain(k.chunks_mut(head_dim)) {
        rope_imrope_neox(head, pos, cfg.rope_theta, n_rot, cfg.rope_sections)?;
    }
    if keys.len() != pos * n_kv * head_dim {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "qwen35 attention holds {} keys, token is at position {pos}",
            keys.len() / (n_kv * head_dim)
        ))));
    }
    keys.extend_from_slice(&k);
    values.extend_from_slice(&v);
    let mut out = vec![0.0f32; n_heads * head_dim];
    attention(&q, keys, values, &mut out, 1, pos + 1, n_heads, n_kv, head_dim, true)?;
    for (o, g) in out.iter_mut().zip(&gate) {
        *o *= sigmoid(*g);
    }
    gemv(&out, &layer.wo, cfg.dim)
}
