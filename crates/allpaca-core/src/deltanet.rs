//! Host Gated DeltaNet (Qwen3.5 linear-attention) — single-token recurrent step.
//!
//! Layout matches llama.cpp / TensorSharp for `qwen35`:
//! `state ∈ R^{head_v × head_k}` per V-head, `q,k ∈ R^{head_k}`, `v ∈ R^{head_v}`.

/// Softplus: ln(1 + e^x), stable for large x.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Infer GDN dims from GGUF meta + projection widths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDims {
    /// How V heads share K heads when `n_v_heads > n_k_heads`.
    pub k_repeat: KHeadRepeat,
    pub n_v_heads: usize,
    pub n_k_heads: usize,
    pub head_v: usize,
    pub head_k: usize,
    pub conv_kernel: usize,
    pub qkv_dim: usize,
    pub value_dim: usize,
    pub key_dim: usize,
}

impl GdnDims {
    pub fn from_meta(
        state_size: usize,
        inner_size: usize,
        time_step_rank: usize,
        group_count: usize,
        conv_kernel: usize,
    ) -> Option<Self> {
        let head_v = state_size.max(1);
        let head_k = state_size.max(1);
        let n_v_heads = time_step_rank.max(1);
        let n_k_heads = group_count.max(1);
        if inner_size != head_v.saturating_mul(n_v_heads) {
            return None;
        }
        let key_dim = head_k.saturating_mul(n_k_heads);
        let value_dim = inner_size;
        let qkv_dim = key_dim.saturating_mul(2).saturating_add(value_dim);
        Some(Self {
            k_repeat: KHeadRepeat::Tile,
            n_v_heads,
            n_k_heads,
            head_v,
            head_k,
            conv_kernel: conv_kernel.max(1),
            qkv_dim,
            value_dim,
            key_dim,
        })
    }

    pub fn state_floats(&self) -> usize {
        self.n_v_heads
            .saturating_mul(self.head_v)
            .saturating_mul(self.head_k)
    }

    pub fn conv_state_floats(&self) -> usize {
        self.conv_kernel
            .saturating_sub(1)
            .saturating_mul(self.qkv_dim)
    }
}

/// Depthwise causal conv1d for one token.
///
/// `kernel` is `[conv_kernel * qkv_dim]` with ggml layout `ne0=kernel` contiguous:
/// `kernel[i + c * conv_kernel]`.
/// `conv_state` is `[(conv_kernel-1) * qkv_dim]` history (oldest first per channel).
pub fn conv1d_step(
    x: &[f32],
    kernel: &[f32],
    conv_state: &mut [f32],
    out: &mut [f32],
    conv_kernel: usize,
) {
    let qkv_dim = x.len();
    debug_assert_eq!(out.len(), qkv_dim);
    debug_assert_eq!(kernel.len(), conv_kernel.saturating_mul(qkv_dim));
    let hist = conv_kernel.saturating_sub(1);
    debug_assert_eq!(conv_state.len(), hist.saturating_mul(qkv_dim));
    for c in 0..qkv_dim {
        let mut acc = 0.0f32;
        for i in 0..hist {
            let w = kernel[i + c * conv_kernel];
            acc += w * conv_state[i * qkv_dim + c];
        }
        acc += kernel[(conv_kernel - 1) + c * conv_kernel] * x[c];
        out[c] = silu(acc);
    }
    // Shift history: drop oldest, append x.
    if hist > 0 {
        for i in 0..hist.saturating_sub(1) {
            let dst = i * qkv_dim;
            let src = (i + 1) * qkv_dim;
            conv_state.copy_within(src..src + qkv_dim, dst);
        }
        let last = (hist - 1) * qkv_dim;
        conv_state[last..last + qkv_dim].copy_from_slice(x);
    }
}

fn l2_normalize(v: &mut [f32], eps: f32) {
    let mut ss = 0.0f32;
    for &x in v.iter() {
        ss += x * x;
    }
    // ggml_l2_norm: x / sqrt(sumsq + eps)
    let inv = 1.0 / (ss + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// One-token gated delta rule over all V-heads (llama.cpp AR path).
///
/// After L2-normalizing q/k, q is scaled by `1/sqrt(head_k)` (see
/// `build_delta_net_autoregressive`). State is `[n_v][head_v][head_k]`
/// (FLA `STATE_V_FIRST`): `kv = S@k`, `S += outer(δ, k)`, `y = S@q`.
pub fn gated_delta_step(
    q_k: &mut [f32],
    k_k: &mut [f32],
    v: &[f32],
    decay: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_v_heads: usize,
    head_v: usize,
    head_k: usize,
) {
    gated_delta_step_eps(
        q_k, k_k, v, decay, beta, state, out, n_v_heads, head_v, head_k, 1e-6,
    );
}

/// Like [`gated_delta_step`] with explicit L2-norm epsilon (llama `f_norm_rms_eps`).
pub fn gated_delta_step_eps(
    q_k: &mut [f32],
    k_k: &mut [f32],
    v: &[f32],
    decay: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    n_v_heads: usize,
    head_v: usize,
    head_k: usize,
    eps: f32,
) {
    debug_assert_eq!(q_k.len(), n_v_heads.saturating_mul(head_k));
    debug_assert_eq!(k_k.len(), n_v_heads.saturating_mul(head_k));
    debug_assert_eq!(v.len(), n_v_heads.saturating_mul(head_v));
    debug_assert_eq!(decay.len(), n_v_heads);
    debug_assert_eq!(beta.len(), n_v_heads);
    debug_assert_eq!(
        state.len(),
        n_v_heads.saturating_mul(head_v).saturating_mul(head_k)
    );
    debug_assert_eq!(out.len(), n_v_heads.saturating_mul(head_v));

    let q_scale = 1.0 / (head_k as f32).sqrt();

    for h in 0..n_v_heads {
        let q = &mut q_k[h * head_k..(h + 1) * head_k];
        let k = &mut k_k[h * head_k..(h + 1) * head_k];
        l2_normalize(q, eps);
        l2_normalize(k, eps);
        for x in q.iter_mut() {
            *x *= q_scale;
        }
        let v_h = &v[h * head_v..(h + 1) * head_v];
        let g = decay[h];
        let b = beta[h];
        let s_off = h * head_v * head_k;
        let state_h = &mut state[s_off..s_off + head_v * head_k];
        let out_h = &mut out[h * head_v..(h + 1) * head_v];

        // kv_mem = state @ k, state *= g
        let mut kv_mem = vec![0.0f32; head_v];
        for i in 0..head_v {
            let row = &mut state_h[i * head_k..(i + 1) * head_k];
            let mut acc = 0.0f32;
            for j in 0..head_k {
                row[j] *= g;
                acc += row[j] * k[j];
            }
            kv_mem[i] = acc;
        }
        // state += β * (v - kv_mem) ⊗ kᵀ ; y = state @ q
        for i in 0..head_v {
            let delta = b * (v_h[i] - kv_mem[i]);
            let row = &mut state_h[i * head_k..(i + 1) * head_k];
            let mut y = 0.0f32;
            for j in 0..head_k {
                row[j] += delta * k[j];
                y += row[j] * q[j];
            }
            out_h[i] = y;
        }
    }
}

/// Deinterleave llama.cpp gated-attention Q projection:
/// `[head0_q|head0_gate|head1_q|head1_gate|…]` → separate Q and gate buffers.
pub fn deinterleave_q_gate(q_full: &[f32], q: &mut [f32], gate: &mut [f32], head_dim: usize) {
    debug_assert_eq!(q_full.len(), q.len().saturating_mul(2));
    debug_assert_eq!(q.len(), gate.len());
    debug_assert_eq!(q.len() % head_dim, 0);
    let n_heads = q.len() / head_dim;
    for h in 0..n_heads {
        let src = h * 2 * head_dim;
        let dst = h * head_dim;
        q[dst..dst + head_dim].copy_from_slice(&q_full[src..src + head_dim]);
        gate[dst..dst + head_dim].copy_from_slice(&q_full[src + head_dim..src + 2 * head_dim]);
    }
}

/// Per-head RMSNorm used by Qwen3.5 full-attention Q/K.
pub fn rmsnorm_heads(x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
    debug_assert!(!weight.is_empty());
    debug_assert_eq!(weight.len(), head_dim);
    debug_assert_eq!(x.len() % head_dim, 0);
    let n_heads = x.len() / head_dim;
    for h in 0..n_heads {
        let row = &mut x[h * head_dim..(h + 1) * head_dim];
        let mut ss = 0.0f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            row[i] = row[i] * inv * weight[i];
        }
    }
}

/// `out = silu(z) * rmsnorm(y; weight)` with `y,z,out` length `n`, `weight` length `head_v`
/// applied per head chunk.
pub fn rmsnorm_silu_gate(
    y: &[f32],
    z: &[f32],
    weight: &[f32],
    out: &mut [f32],
    head_v: usize,
    eps: f32,
) {
    debug_assert_eq!(y.len(), z.len());
    debug_assert_eq!(y.len(), out.len());
    debug_assert_eq!(weight.len(), head_v);
    debug_assert_eq!(y.len() % head_v, 0);
    let n_heads = y.len() / head_v;
    for h in 0..n_heads {
        let y_h = &y[h * head_v..(h + 1) * head_v];
        let z_h = &z[h * head_v..(h + 1) * head_v];
        let out_h = &mut out[h * head_v..(h + 1) * head_v];
        let mut ss = 0.0f32;
        for &v in y_h {
            ss += v * v;
        }
        let inv = 1.0 / (ss / head_v as f32 + eps).sqrt();
        for i in 0..head_v {
            out_h[i] = silu(z_h[i]) * (y_h[i] * inv * weight[i]);
        }
    }
}

/// How a Gated DeltaNet's V heads share K heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KHeadRepeat {
    /// Qwen3.5 (`ggml_repeat`): the K-head block tiles, `dst[vh] = src[vh % n_k]`.
    #[default]
    Tile,
    /// Qwen3-Next (repeat-interleave): each K head repeats for its run of V
    /// heads, `dst[vh] = src[vh / (n_v / n_k)]`.
    Interleave,
}

/// Repeat K-head vectors to V-heads (`n_v % n_k == 0`) by tiling; see
/// [`KHeadRepeat::Tile`].
pub fn expand_k_heads(src: &[f32], dst: &mut [f32], n_k: usize, n_v: usize, head_k: usize) {
    expand_k_heads_as(src, dst, n_k, n_v, head_k, KHeadRepeat::Tile);
}

/// Repeat K-head vectors to V-heads (`n_v % n_k == 0`) in the given layout.
pub fn expand_k_heads_as(
    src: &[f32],
    dst: &mut [f32],
    n_k: usize,
    n_v: usize,
    head_k: usize,
    repeat: KHeadRepeat,
) {
    debug_assert_eq!(src.len(), n_k.saturating_mul(head_k));
    debug_assert_eq!(dst.len(), n_v.saturating_mul(head_k));
    debug_assert_eq!(n_v % n_k, 0);
    let per_k = (n_v / n_k.max(1)).max(1);
    for vh in 0..n_v {
        let kh = match repeat {
            KHeadRepeat::Tile => vh % n_k,
            KHeadRepeat::Interleave => vh / per_k,
        };
        let s = &src[kh * head_k..(kh + 1) * head_k];
        dst[vh * head_k..(vh + 1) * head_k].copy_from_slice(s);
    }
}

/// Build per-head decay `exp(softplus(α + dt) * a)` (llama.cpp linear-attn gate).
pub fn decay_from_alpha(alpha: &[f32], dt_bias: &[f32], a: &[f32], out: &mut [f32]) {
    debug_assert_eq!(alpha.len(), out.len());
    debug_assert_eq!(dt_bias.len(), out.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..out.len() {
        out[i] = (softplus(alpha[i] + dt_bias[i]) * a[i]).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_dims_match_qwen38_meta() {
        let d = GdnDims::from_meta(128, 6144, 48, 16, 4).expect("dims");
        assert_eq!(d.qkv_dim, 10240);
        assert_eq!(d.value_dim, 6144);
        assert_eq!(d.key_dim, 2048);
        assert_eq!(d.state_floats(), 48 * 128 * 128);
        assert_eq!(d.conv_state_floats(), 3 * 10240);
    }

    #[test]
    fn gated_delta_step_writes_and_reads_tiny_state() {
        // 1 head, head_v=2, head_k=2 — hand-checked with llama.cpp q_scale=1/√2.
        let mut q = [1.0f32, 0.0];
        let mut k = [0.0f32, 1.0];
        let v = [2.0f32, -1.0];
        let decay = [0.5f32];
        let beta = [1.0f32]; // full write
        let mut state = [0.0f32; 4]; // 2x2
        let mut out = [0.0f32; 2];
        gated_delta_step(
            &mut q, &mut k, &v, &decay, &beta, &mut state, &mut out, 1, 2, 2,
        );

        // After L2: q=[1,0] then *1/√2; k=[0,1].
        // state starts 0 → kv_mem=0 → δ=v → state rows: [0, 2], [0, -1]
        // y = state @ q = q_scale * col0 = 0.
        let qs = 1.0f32 / 2.0f32.sqrt();
        assert!((q[0] - qs).abs() < 1e-5);
        assert!((state[1] - 2.0).abs() < 1e-5);
        assert!((state[3] - -1.0).abs() < 1e-5);
        assert!(out[0].abs() < 1e-5);
        assert!(out[1].abs() < 1e-5);

        // Second step with q=[0,1] reads the written column, scaled.
        let mut q2 = [0.0f32, 1.0];
        let mut k2 = [0.0f32, 1.0];
        let v2 = [0.0f32, 0.0];
        let decay2 = [1.0f32]; // keep
        let beta2 = [0.0f32]; // no write
        let mut out2 = [0.0f32; 2];
        gated_delta_step(
            &mut q2, &mut k2, &v2, &decay2, &beta2, &mut state, &mut out2, 1, 2, 2,
        );
        assert!((out2[0] - 2.0 * qs).abs() < 1e-4, "out0={}", out2[0]);
        assert!((out2[1] - -1.0 * qs).abs() < 1e-4, "out1={}", out2[1]);
    }

    #[test]
    fn deinterleave_q_gate_matches_llama_stride() {
        // 2 heads × head_dim=2: [q0,q0,g0,g0, q1,q1,g1,g1] wait — per head [q|gate]
        let full = [1.0f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let mut q = [0.0f32; 4];
        let mut gate = [0.0f32; 4];
        deinterleave_q_gate(&full, &mut q, &mut gate, 2);
        assert_eq!(q, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gate, [10.0, 20.0, 30.0, 40.0]);
    }

    /// Reference numbers from llama.cpp `build_delta_net_autoregressive` algebra
    /// with STATE_V_FIRST layout (equivalent transpose of ggml [K,V] storage).
    #[test]
    fn gated_delta_matches_llama_ar_reference() {
        let mut q = [3.0f32, 4.0]; // L2 → [0.6, 0.8], then *1/√2
        let mut k = [0.0f32, 5.0]; // L2 → [0, 1]
        let v = [1.0f32, 2.0];
        let decay = [0.25f32];
        let beta = [0.5f32];
        let mut state = [0.0f32; 4];
        let mut out = [0.0f32; 2];
        gated_delta_step(
            &mut q, &mut k, &v, &decay, &beta, &mut state, &mut out, 1, 2, 2,
        );

        let qs = 1.0f32 / 2.0f32.sqrt();
        // state0 *= 0.25 → 0; δ = 0.5*(v-0) = [0.5, 1.0]
        // state += outer(δ, k=[0,1]) → rows [0, 0.5], [0, 1.0]
        assert!((state[0] - 0.0).abs() < 1e-5);
        assert!((state[1] - 0.5).abs() < 1e-5);
        assert!((state[2] - 0.0).abs() < 1e-5);
        assert!((state[3] - 1.0).abs() < 1e-5);
        // y = state @ q_scaled; q after L2+scale = [0.6, 0.8]*qs
        let q0 = 0.6 * qs;
        let q1 = 0.8 * qs;
        let y0 = state[0] * q0 + state[1] * q1;
        let y1 = state[2] * q0 + state[3] * q1;
        assert!((out[0] - y0).abs() < 1e-5, "out0={} want={y0}", out[0]);
        assert!((out[1] - y1).abs() < 1e-5, "out1={} want={y1}", out[1]);
    }

    #[test]
    fn conv1d_step_shifts_history() {
        let k = 2usize;
        let _dim = 2usize;
        // kernel: for c0 weights [1, 0], c1 weights [0, 1]
        let kernel = [1.0f32, 0.0, 0.0, 1.0]; // i + c*k
        let mut state = [3.0f32, 4.0]; // hist=1
        let x = [5.0f32, 6.0];
        let mut out = [0.0f32; 2];
        conv1d_step(&x, &kernel, &mut state, &mut out, k);
        // c0: 1*3 + 0*5 = 3 → silu(3)
        // c1: 0*4 + 1*6 = 6 → silu(6)
        assert!((out[0] - silu(3.0)).abs() < 1e-5);
        assert!((out[1] - silu(6.0)).abs() < 1e-5);
        assert_eq!(state, x);
    }

    #[test]
    fn expand_k_heads_tiles_like_ggml_repeat() {
        let src = [1.0f32, 2.0, 3.0, 4.0]; // 2 k-heads × 2
        let mut dst = [0.0f32; 8]; // 4 v-heads
        expand_k_heads(&src, &mut dst, 2, 4, 2);
        // ggml_repeat tiles the whole block: [h0,h1, h0,h1], not [h0,h0, h1,h1].
        assert_eq!(dst, [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn expand_k_heads_interleaves_for_qwen3next() {
        // Qwen3-Next repeats each K head for its V heads: [h0,h0, h1,h1].
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 8];
        expand_k_heads_as(&src, &mut dst, 2, 4, 2, KHeadRepeat::Interleave);
        assert_eq!(dst, [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }
}
