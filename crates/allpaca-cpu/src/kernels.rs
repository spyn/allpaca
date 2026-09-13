use allpaca_types::{AllpacaError, FatalError};

/// Stateless CPU kernel backend. AVX2 is used when the host supports it.
#[derive(Debug, Default, Clone)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn gemm(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), AllpacaError> {
        gemm(a, b, c, m, n, k)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        n_q: usize,
        n_kv: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        attention(
            q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
        )
    }

    pub fn copy_async(&self, dst: &mut [u8], src: &[u8]) -> Result<(), AllpacaError> {
        copy_async(dst, src)
    }
}

/// C = A (m×k) * B (k×n). Matches the Kernel ABI `allpaca_gemm`.
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    matmul(a, b, c, m, n, k)
}

pub fn matmul(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "matmul shape mismatch m={m} n={n} k={k} a={} b={} c={}",
            a.len(),
            b.len(),
            c.len()
        ))));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                matmul_avx2(a, b, c, m, n, k);
            }
            return Ok(());
        }
    }
    matmul_scalar(a, b, c, m, n, k);
    Ok(())
}

fn matmul_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// y = x @ w_t  where x is [rows, k], w_t is [n, k] stored row-major (typical lm_head).
pub fn matmul_nt(
    x: &[f32],
    w_t: &[f32],
    y: &mut [f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    if x.len() < rows * k || w_t.len() < n * k || y.len() < rows * n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "matmul_nt shape mismatch".into(),
        )));
    }
    for r in 0..rows {
        for j in 0..n {
            let mut acc = 0.0f32;
            let x_row = &x[r * k..(r + 1) * k];
            let w_row = &w_t[j * k..(j + 1) * k];
            for p in 0..k {
                acc += x_row[p] * w_row[p];
            }
            y[r * n + j] = acc;
        }
    }
    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    for i in 0..m {
        let mut j = 0;
        while j + 8 <= n {
            let mut acc = _mm256_setzero_ps();
            for p in 0..k {
                let av = _mm256_set1_ps(a[i * k + p]);
                let bv = _mm256_loadu_ps(b.as_ptr().add(p * n + j));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(av, bv));
            }
            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
            j += 8;
        }
        while j < n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
            j += 1;
        }
    }
}

/// Causal multi-head / GQA attention. q:[n_q, n_heads, d] k/v:[n_kv, n_kv_heads, d].
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<(), AllpacaError> {
    if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "invalid attention head configuration".into(),
        )));
    }
    let q_need = n_q * n_heads * head_dim;
    let kv_need = n_kv * n_kv_heads * head_dim;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || out.len() < q_need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "attention buffer size mismatch".into(),
        )));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let reps = n_heads / n_kv_heads;
    let mut scores = vec![0.0f32; n_kv];
    for qi in 0..n_q {
        for h in 0..n_heads {
            let kv_h = h / reps;
            let q_off = (qi * n_heads + h) * head_dim;
            let qh = &q[q_off..q_off + head_dim];
            for t in 0..n_kv {
                if causal && t > qi + (n_kv.saturating_sub(n_q)) {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (t * n_kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += qh[d] * k[k_off + d];
                }
                scores[t] = dot * scale;
            }
            softmax_in_place(&mut scores);
            let o_off = (qi * n_heads + h) * head_dim;
            for d in 0..head_dim {
                out[o_off + d] = 0.0;
            }
            for t in 0..n_kv {
                let w = scores[t];
                if w == 0.0 {
                    continue;
                }
                let v_off = (t * n_kv_heads + kv_h) * head_dim;
                for d in 0..head_dim {
                    out[o_off + d] += w * v[v_off + d];
                }
            }
        }
    }
    Ok(())
}

/// Attention with an explicit [n_q, n_kv] ancestry/causal mask (`true` = attend).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attention_with_mask(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    mask: &[bool],
) -> Result<(), AllpacaError> {
    if n_heads == 0 || n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "invalid attention head configuration".into(),
        )));
    }
    let q_need = n_q * n_heads * head_dim;
    let kv_need = n_kv * n_kv_heads * head_dim;
    if q.len() < q_need
        || k.len() < kv_need
        || v.len() < kv_need
        || out.len() < q_need
        || mask.len() < n_q * n_kv
    {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "attention mask buffer size mismatch".into(),
        )));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let reps = n_heads / n_kv_heads;
    let mut scores = vec![0.0f32; n_kv];
    for qi in 0..n_q {
        for h in 0..n_heads {
            let kv_h = h / reps;
            let q_off = (qi * n_heads + h) * head_dim;
            let qh = &q[q_off..q_off + head_dim];
            for t in 0..n_kv {
                if !mask[qi * n_kv + t] {
                    scores[t] = f32::NEG_INFINITY;
                    continue;
                }
                let k_off = (t * n_kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += qh[d] * k[k_off + d];
                }
                scores[t] = dot * scale;
            }
            softmax_in_place(&mut scores);
            let o_off = (qi * n_heads + h) * head_dim;
            for d in 0..head_dim {
                out[o_off + d] = 0.0;
            }
            for t in 0..n_kv {
                let w = scores[t];
                if w == 0.0 {
                    continue;
                }
                let v_off = (t * n_kv_heads + kv_h) * head_dim;
                for d in 0..head_dim {
                    out[o_off + d] += w * v[v_off + d];
                }
            }
        }
    }
    Ok(())
}

pub fn copy_async(dst: &mut [u8], src: &[u8]) -> Result<(), AllpacaError> {
    if dst.len() < src.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "copy_async destination too small".into(),
        )));
    }
    dst[..src.len()].copy_from_slice(src);
    Ok(())
}

pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<(), AllpacaError> {
    if x.len() != weight.len() || out.len() != x.len() || x.is_empty() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "rmsnorm length mismatch".into(),
        )));
    }
    let mut ss = 0.0f32;
    for v in x {
        ss += *v * *v;
    }
    let inv = (ss / x.len() as f32 + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
    Ok(())
}

/// In-place RoPE on a single head vector (interleaved even/odd pairs).
pub fn rope(x: &mut [f32], pos: usize, theta: f32) -> Result<(), AllpacaError> {
    if x.len() % 2 != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "rope head_dim must be even".into(),
        )));
    }
    let half = x.len() / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(i as f32 / half as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * cos - b * sin;
        x[2 * i + 1] = a * sin + b * cos;
    }
    Ok(())
}

/// Qwen3.5 / llama.cpp `GGML_ROPE_TYPE_IMROPE`: NeoX pairing on the first `n_rot`
/// dims only. Text tokens use positions `(pos, pos, pos, 0)`.
///
/// When `sections` sum to 0, falls back to plain NeoX partial RoPE (same angle
/// stream for every pair). `n_rot == 0` means rotate the full head.
pub fn rope_imrope_neox(
    x: &mut [f32],
    pos: usize,
    theta: f32,
    n_rot: usize,
    sections: [i32; 4],
) -> Result<(), AllpacaError> {
    let head_dim = x.len();
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "imrope head_dim must be even and non-empty".into(),
        )));
    }
    let n_rot = if n_rot == 0 { head_dim } else { n_rot };
    if n_rot > head_dim || n_rot % 2 != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "imrope n_rot must be even and <= head_dim".into(),
        )));
    }
    let half = n_rot / 2;
    let theta_scale = theta.powf(-2.0 / n_rot as f32);
    let sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    let use_imrope = sect_dims > 0 && sections[0] > 0 && sections[1] > 0;

    let mut theta_t = pos as f32;
    let mut theta_h = pos as f32;
    let mut theta_w = pos as f32;
    let mut theta_e = 0.0f32;

    for i in 0..half {
        let angle = if use_imrope {
            let sector = i as i32 % sect_dims;
            let t = if sector % 3 == 1 && sector < 3 * sections[1] {
                theta_h
            } else if sector % 3 == 2 && sector < 3 * sections[2] {
                theta_w
            } else if sector % 3 == 0 && sector < 3 * sections[0] {
                theta_t
            } else {
                theta_e
            };
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
            t
        } else {
            let t = theta_t;
            theta_t *= theta_scale;
            t
        };
        let (sin, cos) = angle.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin;
        x[i + half] = a * sin + b * cos;
    }
    Ok(())
}

/// Apply [`rope_imrope_neox`] to packed multi-head Q or K `[n_heads * head_dim]`.
pub fn rope_imrope_neox_heads(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    n_rot: usize,
    sections: [i32; 4],
) -> Result<(), AllpacaError> {
    if n_heads == 0 || head_dim == 0 || x.len() != n_heads.saturating_mul(head_dim) {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "imrope heads length mismatch".into(),
        )));
    }
    for h in 0..n_heads {
        let row = &mut x[h * head_dim..(h + 1) * head_dim];
        rope_imrope_neox(row, pos, theta, n_rot, sections)?;
    }
    Ok(())
}

pub fn softmax(x: &mut [f32]) {
    softmax_in_place(x);
}

fn softmax_in_place(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for v in x.iter() {
        if *v > max {
            max = *v;
        }
    }
    if !max.is_finite() {
        let n = x.len() as f32;
        for v in x.iter_mut() {
            *v = 1.0 / n;
        }
        return;
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        let n = x.len() as f32;
        for v in x.iter_mut() {
            *v = 1.0 / n;
        }
        return;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Legacy Q4_1: f16 d + f16 m + qs[16] (32 nibbles). `x = d*q + m`.
pub const Q4_1_BLOCK_ELEMS: usize = 32;
pub const Q4_1_BLOCK_BYTES: usize = 20;
/// Legacy Q5_0: f16 d + qh u32 + qs[16]. `x = d*(q-16)` with 5-bit q.
pub const Q5_0_BLOCK_ELEMS: usize = 32;
pub const Q5_0_BLOCK_BYTES: usize = 22;
/// Legacy Q5_1: f16 d + f16 m + qh u32 + qs[16]. `x = d*q + m`.
pub const Q5_1_BLOCK_ELEMS: usize = 32;
pub const Q5_1_BLOCK_BYTES: usize = 24;
/// ggml_type_id=100 ROCmFP4 dual-scale: 18 bytes / 32 elems (4.5 bpw).
/// Block layout is `qs[16]` then `e[2]`: nibble `j` low → element `j` (scale
/// `e[0]`), nibble `j` high → element `j + 16` (scale `e[1]`).
pub const ROCMFP4_BLOCK_ELEMS: usize = 32;
pub const ROCMFP4_BLOCK_BYTES: usize = 18;
/// Signed E2M1-derived codebook at half-scale, top magnitude retuned 12 → 10.
pub const ROCMFP4_CODEBOOK: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10];
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// Legacy Q8_1: f16 d + f16 s + qs[32]. `x = d*q` (s is block sum, unused here).
pub const Q8_1_BLOCK_ELEMS: usize = 32;
pub const Q8_1_BLOCK_BYTES: usize = 36;
/// GGUF TQ1_0 ternary super-block: qs[48] + qh[4] + f16 d.
pub const TQ1_0_BLOCK_ELEMS: usize = 256;
pub const TQ1_0_BLOCK_BYTES: usize = 54;
/// GGUF TQ2_0 ternary super-block: qs[64] + f16 d.
pub const TQ2_0_BLOCK_ELEMS: usize = 256;
pub const TQ2_0_BLOCK_BYTES: usize = 66;
/// GGUF Q2_K super-block (256 elems).
pub const Q2_K_BLOCK_ELEMS: usize = 256;
pub const Q2_K_BLOCK_BYTES: usize = 84;
/// GGUF Q3_K super-block (256 elems).
pub const Q3_K_BLOCK_ELEMS: usize = 256;
pub const Q3_K_BLOCK_BYTES: usize = 110;
/// GGUF Q4_K / Q4_K_S / Q4_K_M super-block (256 elems).
pub const Q4_K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// GGUF Q5_K super-block (256 elems): d + dmin + scales[12] + qh[32] + qs[128].
pub const Q5_K_BLOCK_ELEMS: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// GGUF Q6_K super-block (256 elems): ql[128] + qh[64] + scales[16] + d.
pub const Q6_K_BLOCK_ELEMS: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// GGUF IQ4_NL (ggml_type_id=20): f16 d + 16 packed nibbles / 32 elems.
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;
pub const IQ4_NL_BLOCK_BYTES: usize = 18;
/// GGUF IQ4_XS (ggml_type_id=23): f16 d + u16 scales_h + 4 scale nibbles + 128 qs / 256 elems.
pub const IQ4_XS_BLOCK_ELEMS: usize = 256;
pub const IQ4_XS_BLOCK_BYTES: usize = 136;
/// ggml `kvalues_iq4nl` (shared by IQ4_NL and IQ4_XS).
pub const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

/// IEEE-754 binary16 → f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = u32::from(h & 0x3ff);
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((u32::from(exp) + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | ((mant >> 13) as u16);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        return sign;
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(Q8_0_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * Q8_0_BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * Q8_0_BLOCK_ELEMS;
        let end = (start + Q8_0_BLOCK_ELEMS).min(data.len());
        let mut amax = 0.0f32;
        for v in &data[start..end] {
            amax = amax.max(v.abs());
        }
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        let off = b * Q8_0_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for i in 0..Q8_0_BLOCK_ELEMS {
            let src = start + i;
            let q = if src < data.len() {
                (data[src] * inv).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            out[off + 2 + i] = q as u8;
        }
    }
    out
}

pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(Q4_0_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * Q4_0_BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * Q4_0_BLOCK_ELEMS;
        let end = (start + Q4_0_BLOCK_ELEMS).min(data.len());
        let mut amax = 0.0f32;
        for v in &data[start..end] {
            amax = amax.max(v.abs());
        }
        let d = if amax == 0.0 { 0.0 } else { amax / 8.0 };
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        let off = b * Q4_0_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let q_lo = quant_nibble(data, start + j, inv);
            let q_hi = quant_nibble(data, start + j + 16, inv);
            out[off + 2 + j] = q_lo | (q_hi << 4);
        }
    }
    out
}

/// Quantize to GGUF Q4_K super-blocks (tests / tiny written models).
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(Q4_K_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * Q4_K_BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * Q4_K_BLOCK_ELEMS;
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            let g0 = start + j * 32;
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for l in 0..32 {
                let v = data.get(g0 + l).copied().unwrap_or(0.0);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            if min_v == f32::MAX {
                min_v = 0.0;
                max_v = 0.0;
            }
            scales[j] = (max_v - min_v) / 15.0;
            mins[j] = -min_v;
        }
        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = mins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_min = mins.iter().copied().fold(f32::INFINITY, f32::min);
        let extremum = if max_min.abs() >= min_min.abs() {
            max_min
        } else {
            min_min
        };
        let d = if max_scale > 0.0 {
            max_scale / 63.0
        } else {
            0.0
        };
        let dmin = if extremum.abs() > 0.0 {
            extremum / 63.0
        } else {
            0.0
        };
        let mut sc_bytes = [0u8; 12];
        for j in 0..8 {
            let ls = if d > 0.0 {
                ((scales[j] / d).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            let lm = if dmin.abs() > 0.0 {
                ((mins[j] / dmin).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            if j < 4 {
                sc_bytes[j] = ls;
                sc_bytes[j + 4] = lm;
            } else {
                sc_bytes[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                sc_bytes[j - 4] |= (ls >> 4) << 6;
                sc_bytes[j] |= (lm >> 4) << 6;
            }
        }
        let off = b * Q4_K_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[off + 2..off + 4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out[off + 4..off + 16].copy_from_slice(&sc_bytes);
        let mut qs = [0u8; 256];
        for j in 0..8 {
            let (sc, m) = q4_k_scale_min(j, &sc_bytes);
            let dj = d * f32::from(sc);
            let mj = dmin * f32::from(m);
            let g0 = start + j * 32;
            for l in 0..32 {
                let x = data.get(g0 + l).copied().unwrap_or(0.0);
                let q = if dj.abs() > 0.0 {
                    ((x + mj) / dj).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                qs[j * 32 + l] = q;
            }
        }
        let q_out = &mut out[off + 16..off + 144];
        for grp in 0..4 {
            let base = grp * 64;
            for l in 0..32 {
                q_out[grp * 32 + l] = qs[base + l] | (qs[base + 32 + l] << 4);
            }
        }
    }
    out
}

/// Quantize to GGUF Q5_K super-blocks (tests / tiny written models).
pub fn quantize_q5_k(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(Q5_K_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * Q5_K_BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * Q5_K_BLOCK_ELEMS;
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];
        for j in 0..8 {
            let g0 = start + j * 32;
            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for l in 0..32 {
                let v = data.get(g0 + l).copied().unwrap_or(0.0);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            if min_v == f32::MAX {
                min_v = 0.0;
                max_v = 0.0;
            }
            scales[j] = (max_v - min_v) / 31.0;
            mins[j] = -min_v;
        }
        let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = mins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_min = mins.iter().copied().fold(f32::INFINITY, f32::min);
        let extremum = if max_min.abs() >= min_min.abs() {
            max_min
        } else {
            min_min
        };
        let d = if max_scale > 0.0 {
            max_scale / 63.0
        } else {
            0.0
        };
        let dmin = if extremum.abs() > 0.0 {
            extremum / 63.0
        } else {
            0.0
        };
        let mut sc_bytes = [0u8; 12];
        for j in 0..8 {
            let ls = if d > 0.0 {
                ((scales[j] / d).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            let lm = if dmin.abs() > 0.0 {
                ((mins[j] / dmin).round() as i32).clamp(0, 63) as u8
            } else {
                0
            };
            if j < 4 {
                sc_bytes[j] = ls;
                sc_bytes[j + 4] = lm;
            } else {
                sc_bytes[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                sc_bytes[j - 4] |= (ls >> 4) << 6;
                sc_bytes[j] |= (lm >> 4) << 6;
            }
        }
        let off = b * Q5_K_BLOCK_BYTES;
        out[off..off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[off + 2..off + 4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out[off + 4..off + 16].copy_from_slice(&sc_bytes);
        let mut qs = [0u8; 256];
        for j in 0..8 {
            let (sc, m) = q4_k_scale_min(j, &sc_bytes);
            let dj = d * f32::from(sc);
            let mj = dmin * f32::from(m);
            let g0 = start + j * 32;
            for l in 0..32 {
                let x = data.get(g0 + l).copied().unwrap_or(0.0);
                let q = if dj.abs() > 0.0 {
                    ((x + mj) / dj).round().clamp(0.0, 31.0) as u8
                } else {
                    0
                };
                qs[j * 32 + l] = q;
            }
        }
        let packed = &mut out[off + 16..off + 176];
        let (qh, q_out) = packed.split_at_mut(32);
        for grp in 0..4 {
            let base = grp * 64;
            let u1 = 1u8 << (2 * grp);
            let u2 = 2u8 << (2 * grp);
            for l in 0..32 {
                let lo = qs[base + l];
                let hi = qs[base + 32 + l];
                q_out[grp * 32 + l] = (lo & 0x0f) | ((hi & 0x0f) << 4);
                if lo & 0x10 != 0 {
                    qh[l] |= u1;
                }
                if hi & 0x10 != 0 {
                    qh[l] |= u2;
                }
            }
        }
    }
    out
}

/// Quantize to GGUF Q6_K super-blocks (tests / tiny written models).
pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(Q6_K_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * Q6_K_BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * Q6_K_BLOCK_ELEMS;
        let mut levels = [0i8; Q6_K_BLOCK_ELEMS];
        let mut scales = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        for ib in 0..16 {
            let g0 = start + ib * 16;
            let mut amax = 0.0f32;
            for l in 0..16 {
                amax = amax.max(data.get(g0 + l).copied().unwrap_or(0.0).abs());
            }
            scales[ib] = amax / 32.0;
            max_scale = max_scale.max(scales[ib].abs());
        }
        let d = if max_scale > 0.0 {
            max_scale / 127.0
        } else {
            0.0
        };
        let off = b * Q6_K_BLOCK_BYTES;
        out[off + 208..off + 210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for ib in 0..16 {
            let sc = if d > 0.0 {
                ((scales[ib] / d).round() as i32).clamp(-128, 127) as i8
            } else {
                0
            };
            out[off + 192 + ib] = sc as u8;
            let dj = d * f32::from(sc);
            let g0 = start + ib * 16;
            for l in 0..16 {
                let q = if dj.abs() > 0.0 {
                    (data.get(g0 + l).copied().unwrap_or(0.0) / dj)
                        .round()
                        .clamp(-32.0, 31.0) as i8
                } else {
                    0
                };
                levels[ib * 16 + l] = q;
            }
        }
        for half in 0..2 {
            let j = half * 128;
            for l in 0..32 {
                let q1 = (i16::from(levels[j + l]) + 32) as u8;
                let q2 = (i16::from(levels[j + l + 32]) + 32) as u8;
                let q3 = (i16::from(levels[j + l + 64]) + 32) as u8;
                let q4 = (i16::from(levels[j + l + 96]) + 32) as u8;
                out[off + half * 64 + l] = (q1 & 0x0f) | ((q3 & 0x0f) << 4);
                out[off + half * 64 + 32 + l] = (q2 & 0x0f) | ((q4 & 0x0f) << 4);
                out[off + 128 + half * 32 + l] =
                    (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
    }
    out
}

fn quant_nibble(data: &[f32], idx: usize, inv: f32) -> u8 {
    if idx >= data.len() {
        return 8;
    }
    ((data[idx] * inv).round() + 8.0).clamp(0.0, 15.0) as u8
}

/// Dequantize GGUF Q4_0 blocks: `f16 d` + packed nibbles → `d * (q - 8)`.
pub fn dequant_q4_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q4_0 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q4_0_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q4_0_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q4_0 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q4_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        for j in 0..16 {
            let packed = blocks[off + 2 + j];
            let i0 = b * Q4_0_BLOCK_ELEMS + j;
            let i1 = i0 + 16;
            if i0 < n {
                out[i0] = d * ((packed & 0x0f) as i8 as f32 - 8.0);
            }
            if i1 < n {
                out[i1] = d * ((packed >> 4) as i8 as f32 - 8.0);
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF Q4_1: `x = d*q + m` with unsigned 4-bit q.
pub fn dequant_q4_1(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q4_1 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q4_1_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q4_1_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q4_1 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q4_1_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([blocks[off + 2], blocks[off + 3]]));
        for j in 0..16 {
            let packed = blocks[off + 4 + j];
            let i0 = b * Q4_1_BLOCK_ELEMS + j;
            let i1 = i0 + 16;
            if i0 < n {
                out[i0] = d * f32::from(packed & 0x0f) + m;
            }
            if i1 < n {
                out[i1] = d * f32::from(packed >> 4) + m;
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF Q5_0: 5-bit q via qs nibbles + qh bits → `d*(q-16)`.
pub fn dequant_q5_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q5_0 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q5_0_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q5_0_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q5_0 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q5_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let qh = u32::from_le_bytes([
            blocks[off + 2],
            blocks[off + 3],
            blocks[off + 4],
            blocks[off + 5],
        ]);
        for j in 0..16 {
            let packed = blocks[off + 6 + j];
            let i0 = b * Q5_0_BLOCK_ELEMS + j;
            let i1 = i0 + 16;
            let q0 = (packed & 0x0f) as u32 | (((qh >> j) & 1) << 4);
            let q1 = (packed >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4);
            if i0 < n {
                out[i0] = d * (q0 as f32 - 16.0);
            }
            if i1 < n {
                out[i1] = d * (q1 as f32 - 16.0);
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF Q5_1: 5-bit q → `d*q + m`.
pub fn dequant_q5_1(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q5_1 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q5_1_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q5_1_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q5_1 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q5_1_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let m = f16_to_f32(u16::from_le_bytes([blocks[off + 2], blocks[off + 3]]));
        let qh = u32::from_le_bytes([
            blocks[off + 4],
            blocks[off + 5],
            blocks[off + 6],
            blocks[off + 7],
        ]);
        for j in 0..16 {
            let packed = blocks[off + 8 + j];
            let i0 = b * Q5_1_BLOCK_ELEMS + j;
            let i1 = i0 + 16;
            let q0 = (packed & 0x0f) as u32 | (((qh >> j) & 1) << 4);
            let q1 = (packed >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4);
            if i0 < n {
                out[i0] = d * q0 as f32 + m;
            }
            if i1 < n {
                out[i1] = d * q1 as f32 + m;
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF Q8_1: `x = d * q` (int8 qs); block sum `s` unused for F32 materialize.
pub fn dequant_q8_1(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q8_1 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q8_1_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q8_1_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q8_1 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q8_1_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        for j in 0..Q8_1_BLOCK_ELEMS {
            let i = b * Q8_1_BLOCK_ELEMS + j;
            if i < n {
                out[i] = d * (blocks[off + 4 + j] as i8 as f32);
            }
        }
    }
    Ok(())
}

const TQ1_0_POW3: [u8; 5] = [1, 3, 9, 27, 81];

#[inline]
fn decode_tq1_0_trit(byte: u8, digit: usize) -> i16 {
    let q = byte.wrapping_mul(TQ1_0_POW3[digit]);
    (((u16::from(q) * 3) >> 8) as i16) - 1
}

/// Dequantize GGUF TQ1_0 blocks using llama.cpp's fixed-point base-3 decode.
pub fn dequant_tq1_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_tq1_0 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(TQ1_0_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(TQ1_0_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_tq1_0 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * TQ1_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off + 52], blocks[off + 53]]));
        let base = b * TQ1_0_BLOCK_ELEMS;
        let mut dst = base;

        for digit in 0..5 {
            for m in 0..32 {
                if dst < n {
                    out[dst] = d * f32::from(decode_tq1_0_trit(blocks[off + m], digit));
                }
                dst += 1;
            }
        }
        for digit in 0..5 {
            for m in 0..16 {
                if dst < n {
                    out[dst] = d * f32::from(decode_tq1_0_trit(blocks[off + 32 + m], digit));
                }
                dst += 1;
            }
        }
        for digit in 0..4 {
            for m in 0..4 {
                if dst < n {
                    out[dst] = d * f32::from(decode_tq1_0_trit(blocks[off + 48 + m], digit));
                }
                dst += 1;
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF TQ2_0 blocks: four 2-bit trits per byte, then f16 d.
pub fn dequant_tq2_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_tq2_0 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(TQ2_0_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(TQ2_0_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_tq2_0 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * TQ2_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off + 64], blocks[off + 65]]));
        let base = b * TQ2_0_BLOCK_ELEMS;
        for group in 0..2 {
            for shift in 0..4 {
                for m in 0..32 {
                    let i = base + group * 128 + shift * 32 + m;
                    if i < n {
                        let trit = (blocks[off + group * 32 + m] >> (shift * 2)) & 3;
                        out[i] = d * (f32::from(trit) - 1.0);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Quantize to GGUF TQ2_0 super-blocks (tests / tiny written BitNet models).
pub fn quantize_tq2_0(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(TQ2_0_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * TQ2_0_BLOCK_BYTES];
    for b in 0..n_blocks {
        let start = b * TQ2_0_BLOCK_ELEMS;
        let end = (start + TQ2_0_BLOCK_ELEMS).min(data.len());
        let mut amax = 0.0f32;
        for v in &data[start..end] {
            amax = amax.max(v.abs());
        }
        let d = amax;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        let off = b * TQ2_0_BLOCK_BYTES;
        let mut trits = [0i8; TQ2_0_BLOCK_ELEMS];
        for i in 0..TQ2_0_BLOCK_ELEMS {
            let src = start + i;
            let q = if src < data.len() {
                (data[src] * inv).round().clamp(-1.0, 1.0) as i8
            } else {
                0
            };
            trits[i] = q;
        }
        for group in 0..2 {
            for m in 0..32 {
                let mut packed = 0u8;
                for digit in 0..4 {
                    let trit = trits[group * 128 + digit * 32 + m] + 1;
                    packed |= (trit as u8) << (digit * 2);
                }
                out[off + group * 32 + m] = packed;
            }
        }
        out[off + 64..off + 66].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

/// Dequantize GGUF Q2_K (llama.cpp `dequantize_row_q2_K`).
pub fn dequant_q2_k(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q2_k output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q2_K_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q2_K_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q2_k truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q2_K_BLOCK_BYTES;
        let scales = &blocks[off..off + 16];
        let mut q = &blocks[off + 16..off + 80];
        let d = f16_to_f32(u16::from_le_bytes([blocks[off + 80], blocks[off + 81]]));
        let min = f16_to_f32(u16::from_le_bytes([blocks[off + 82], blocks[off + 83]]));
        let mut y = b * Q2_K_BLOCK_ELEMS;
        let mut is = 0usize;
        for _ in 0..2 {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    if y + l < n {
                        out[y + l] = dl * f32::from((q[l] >> shift) & 3) - ml;
                    }
                }
                y += 16;
                let sc = scales[is];
                is += 1;
                let dl = d * f32::from(sc & 0x0f);
                let ml = min * f32::from(sc >> 4);
                for l in 0..16 {
                    if y + l < n {
                        out[y + l] = dl * f32::from((q[l + 16] >> shift) & 3) - ml;
                    }
                }
                y += 16;
                shift += 2;
            }
            q = &q[32..];
        }
    }
    Ok(())
}

/// Dequantize GGUF Q3_K (llama.cpp `dequantize_row_q3_K`).
pub fn dequant_q3_k(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q3_k output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q3_K_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q3_K_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q3_k truncated blocks".into(),
        )));
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    for b in 0..n_blocks {
        let off = b * Q3_K_BLOCK_BYTES;
        let hm = &blocks[off..off + 32];
        let mut q = &blocks[off + 32..off + 96];
        let scales_raw = &blocks[off + 96..off + 108];
        let d_all = f16_to_f32(u16::from_le_bytes([blocks[off + 108], blocks[off + 109]]));
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes(scales_raw[0..4].try_into().unwrap());
        aux[1] = u32::from_le_bytes(scales_raw[4..8].try_into().unwrap());
        aux[2] = u32::from_le_bytes(scales_raw[8..12].try_into().unwrap());
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let mut scale_bytes = [0i8; 16];
        for (i, word) in aux.iter().enumerate() {
            let le = word.to_le_bytes();
            for (j, b) in le.into_iter().enumerate() {
                scale_bytes[i * 4 + j] = b as i8;
            }
        }
        let mut y = b * Q3_K_BLOCK_ELEMS;
        let mut is = 0usize;
        let mut m: u8 = 1;
        for _ in 0..2 {
            let mut shift = 0u32;
            for _ in 0..4 {
                let dl = d_all * f32::from(i16::from(scale_bytes[is]) - 32);
                is += 1;
                for l in 0..16 {
                    if y + l < n {
                        let qv = (q[l] >> shift) & 3;
                        let hi: u8 = if hm[l] & m != 0 { 0 } else { 4 };
                        out[y + l] = dl * (f32::from(qv) - f32::from(hi));
                    }
                }
                y += 16;
                let dl = d_all * f32::from(i16::from(scale_bytes[is]) - 32);
                is += 1;
                for l in 0..16 {
                    if y + l < n {
                        let qv = (q[l + 16] >> shift) & 3;
                        let hi: u8 = if hm[l + 16] & m != 0 { 0 } else { 4 };
                        out[y + l] = dl * (f32::from(qv) - f32::from(hi));
                    }
                }
                y += 16;
                shift += 2;
                m = m.wrapping_shl(1);
            }
            q = &q[32..];
        }
    }
    Ok(())
}

fn q4_k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize GGUF Q4_K super-blocks: `y = d*sc*q - dmin*m` (q in 0..15).
pub fn dequant_q4_k(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q4_k output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q4_K_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q4_K_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q4_k truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q4_K_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blocks[off + 2], blocks[off + 3]]));
        let scales = &blocks[off + 4..off + 16];
        let qs = &blocks[off + 16..off + 144];
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut y_base = b * Q4_K_BLOCK_ELEMS;
        for _ in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);
            for l in 0..32 {
                let packed = qs[q_off + l];
                let i0 = y_base + l;
                let i1 = y_base + 32 + l;
                if i0 < n {
                    out[i0] = d1 * f32::from(packed & 0x0f) - min1;
                }
                if i1 < n {
                    out[i1] = d2 * f32::from(packed >> 4) - min2;
                }
            }
            q_off += 32;
            y_base += 64;
            is += 2;
        }
    }
    Ok(())
}

/// Decode a ROCmFP4 unsigned-E4M3 scale byte to the half-scale multiplier the
/// codebook is expressed in. `0x7f`/`0xff` are NaN in this encoding, not scales.
pub fn rocmfp4_scale(e: u8) -> f32 {
    if e == 0 || e == 0x7f || e == 0xff {
        return 0.0;
    }
    let exp = u32::from((e >> 3) & 0x0f);
    let man = u32::from(e & 0x07);
    if exp == 0 {
        return man as f32 * (1.0 / 1024.0);
    }
    f32::from_bits(((exp + 119) << 23) | (man << 20))
}

/// A scale byte is valid when it is finite and unsigned (`<= 0x7e`).
pub fn rocmfp4_scale_is_valid(e: u8) -> bool {
    e <= 0x7e
}

/// Dequantize ROCmFP4 (ggml_type_id=100) blocks: `y = codebook[q] * scale(e)`.
pub fn dequant_rocmfp4(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_rocmfp4 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(ROCMFP4_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(ROCMFP4_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_rocmfp4 truncated blocks".into(),
        )));
    }
    let half = ROCMFP4_BLOCK_ELEMS / 2;
    for b in 0..n_blocks {
        let off = b * ROCMFP4_BLOCK_BYTES;
        let qs = &blocks[off..off + half];
        let d0 = rocmfp4_scale(blocks[off + half]);
        let d1 = rocmfp4_scale(blocks[off + half + 1]);
        let base = b * ROCMFP4_BLOCK_ELEMS;
        for (j, &packed) in qs.iter().enumerate() {
            let i0 = base + j;
            let i1 = i0 + half;
            if i0 < n {
                out[i0] = f32::from(ROCMFP4_CODEBOOK[usize::from(packed & 0x0f)]) * d0;
            }
            if i1 < n {
                out[i1] = f32::from(ROCMFP4_CODEBOOK[usize::from(packed >> 4)]) * d1;
            }
        }
    }
    Ok(())
}

/// Nearest codebook index for `x` under `1 / scale_half`. Ties keep the lower
/// magnitude, matching the reference encoder.
fn rocmfp4_best_index(x: f32, inv_scale_half: f32) -> u8 {
    if !x.is_finite() {
        return 0;
    }
    let a = (x * inv_scale_half).abs();
    if a <= 0.5 {
        return 0;
    }
    let neg = x < 0.0;
    let pos = if a <= 1.5 {
        1
    } else if a <= 2.5 {
        2
    } else if a <= 3.5 {
        3
    } else if a <= 5.0 {
        4
    } else if a <= 7.0 {
        5
    } else if a <= 9.0 {
        6
    } else {
        7
    };
    if neg {
        pos + 8
    } else {
        pos
    }
}

/// Exhaustive finite-E4M3 scale search minimizing reconstruction MSE over one
/// 16-element half block. Ties keep the lower scale byte.
fn rocmfp4_choose_scale(values: &[f32]) -> u8 {
    let mut max_abs = 0.0f32;
    for v in values {
        max_abs = max_abs.max(v.abs());
    }
    if !(max_abs > 0.0) || !max_abs.is_finite() {
        return 0;
    }
    let mut best_e = 0u8;
    let mut best_err = f32::MAX;
    for e in 1u8..=126 {
        let scale_half = rocmfp4_scale(e);
        if !(scale_half > 0.0) {
            continue;
        }
        let inv = 1.0 / scale_half;
        let mut err = 0.0f32;
        for &x in values {
            let q = rocmfp4_best_index(x, inv);
            let y = f32::from(ROCMFP4_CODEBOOK[usize::from(q)]) * scale_half;
            let d = x - y;
            err += d * d;
            if err > best_err {
                break;
            }
        }
        if err < best_err {
            best_err = err;
            best_e = e;
        }
    }
    best_e
}

/// Quantize to ROCmFP4 blocks (tests / tiny written models).
pub fn quantize_rocmfp4(data: &[f32]) -> Vec<u8> {
    let half = ROCMFP4_BLOCK_ELEMS / 2;
    let n_blocks = data.len().div_ceil(ROCMFP4_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * ROCMFP4_BLOCK_BYTES];
    let mut lo = [0.0f32; 16];
    let mut hi = [0.0f32; 16];
    for b in 0..n_blocks {
        let start = b * ROCMFP4_BLOCK_ELEMS;
        for j in 0..half {
            lo[j] = data.get(start + j).copied().unwrap_or(0.0);
            hi[j] = data.get(start + half + j).copied().unwrap_or(0.0);
        }
        let e0 = rocmfp4_choose_scale(&lo);
        let e1 = rocmfp4_choose_scale(&hi);
        let inv0 = inv_half_scale(e0);
        let inv1 = inv_half_scale(e1);
        let off = b * ROCMFP4_BLOCK_BYTES;
        for j in 0..half {
            let q0 = rocmfp4_best_index(lo[j], inv0);
            let q1 = rocmfp4_best_index(hi[j], inv1);
            out[off + j] = q0 | (q1 << 4);
        }
        out[off + half] = e0;
        out[off + half + 1] = e1;
    }
    out
}

fn inv_half_scale(e: u8) -> f32 {
    let s = rocmfp4_scale(e);
    if s > 0.0 {
        1.0 / s
    } else {
        0.0
    }
}

/// Dequantize GGUF Q5_K super-blocks: `y = d*sc*q - dmin*m` (q in 0..31).
pub fn dequant_q5_k(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q5_k output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q5_K_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q5_K_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q5_k truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q5_K_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blocks[off + 2], blocks[off + 3]]));
        let scales = &blocks[off + 4..off + 16];
        let qh = &blocks[off + 16..off + 48];
        let qs = &blocks[off + 48..off + 176];
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut y_base = b * Q5_K_BLOCK_ELEMS;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for _ in 0..4 {
            let (sc1, m1) = q4_k_scale_min(is, scales);
            let (sc2, m2) = q4_k_scale_min(is + 1, scales);
            let d1 = d * f32::from(sc1);
            let min1 = dmin * f32::from(m1);
            let d2 = d * f32::from(sc2);
            let min2 = dmin * f32::from(m2);
            for l in 0..32 {
                let packed = qs[q_off + l];
                let i0 = y_base + l;
                let i1 = y_base + 32 + l;
                if i0 < n {
                    let q = f32::from(packed & 0x0f) + if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                    out[i0] = d1 * q - min1;
                }
                if i1 < n {
                    let q = f32::from(packed >> 4) + if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                    out[i1] = d2 * q - min2;
                }
            }
            q_off += 32;
            y_base += 64;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(())
}

/// Dequantize GGUF Q6_K super-blocks: `y = d * scale * (q - 32)` (q in 0..63).
pub fn dequant_q6_k(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q6_k output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q6_K_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q6_K_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q6_k truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q6_K_BLOCK_BYTES;
        let ql = &blocks[off..off + 128];
        let qh = &blocks[off + 128..off + 192];
        let scales = &blocks[off + 192..off + 208];
        let d = f16_to_f32(u16::from_le_bytes([blocks[off + 208], blocks[off + 209]]));
        let mut y_base = b * Q6_K_BLOCK_ELEMS;
        for half in 0..2 {
            let ql_h = &ql[half * 64..half * 64 + 64];
            let qh_h = &qh[half * 32..half * 32 + 32];
            let sc_h = &scales[half * 8..half * 8 + 8];
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from(ql_h[l] & 0x0f | ((qh_h[l] >> 0) & 3) << 4) - 32;
                let q2 = i32::from(ql_h[l + 32] & 0x0f | ((qh_h[l] >> 2) & 3) << 4) - 32;
                let q3 = i32::from((ql_h[l] >> 4) | ((qh_h[l] >> 4) & 3) << 4) - 32;
                let q4 = i32::from((ql_h[l + 32] >> 4) | ((qh_h[l] >> 6) & 3) << 4) - 32;
                let i0 = y_base + l;
                let i1 = y_base + 32 + l;
                let i2 = y_base + 64 + l;
                let i3 = y_base + 96 + l;
                if i0 < n {
                    out[i0] = d * (sc_h[is] as i8 as f32) * q1 as f32;
                }
                if i1 < n {
                    out[i1] = d * (sc_h[is + 2] as i8 as f32) * q2 as f32;
                }
                if i2 < n {
                    out[i2] = d * (sc_h[is + 4] as i8 as f32) * q3 as f32;
                }
                if i3 < n {
                    out[i3] = d * (sc_h[is + 6] as i8 as f32) * q4 as f32;
                }
            }
            y_base += 128;
        }
    }
    Ok(())
}

/// Dequantize GGUF IQ4_NL (type 20): `y = d * kvalues_iq4nl[nibble]`.
pub fn dequant_iq4_nl(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_iq4_nl output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(IQ4_NL_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(IQ4_NL_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_iq4_nl truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * IQ4_NL_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let qs = &blocks[off + 2..off + IQ4_NL_BLOCK_BYTES];
        let base = b * IQ4_NL_BLOCK_ELEMS;
        for j in 0..16 {
            let packed = qs[j];
            let i0 = base + j;
            let i1 = base + j + 16;
            if i0 < n {
                out[i0] = d * KVALUES_IQ4NL[usize::from(packed & 0x0f)];
            }
            if i1 < n {
                out[i1] = d * KVALUES_IQ4NL[usize::from(packed >> 4)];
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF IQ4_XS (type 23): `y = d * (ls - 32) * kvalues_iq4nl[nibble]`.
pub fn dequant_iq4_xs(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_iq4_xs output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(IQ4_XS_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(IQ4_XS_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_iq4_xs truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * IQ4_XS_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        let scales_h = u16::from_le_bytes([blocks[off + 2], blocks[off + 3]]);
        let scales_l = &blocks[off + 4..off + 8];
        let qs = &blocks[off + 8..off + IQ4_XS_BLOCK_BYTES];
        let super_base = b * IQ4_XS_BLOCK_ELEMS;
        for ib in 0..8 {
            let nibble = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f;
            let hi = u8::try_from((scales_h >> (2 * ib)) & 3).unwrap_or(0);
            let ls = i16::from(nibble) | (i16::from(hi) << 4);
            let dl = d * f32::from(ls - 32);
            let q_off = ib * 16;
            let y_base = super_base + ib * 32;
            for j in 0..16 {
                let packed = qs[q_off + j];
                let i0 = y_base + j;
                let i1 = y_base + j + 16;
                if i0 < n {
                    out[i0] = dl * KVALUES_IQ4NL[usize::from(packed & 0x0f)];
                }
                if i1 < n {
                    out[i1] = dl * KVALUES_IQ4NL[usize::from(packed >> 4)];
                }
            }
        }
    }
    Ok(())
}

/// Dequantize GGUF Q8_0 blocks: `f16 d` + `i8[32]` → `d * q`.
pub fn dequant_q8_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q8_0 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(Q8_0_BLOCK_ELEMS);
    let need = n_blocks.saturating_mul(Q8_0_BLOCK_BYTES);
    if blocks.len() < need {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_q8_0 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let off = b * Q8_0_BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([blocks[off], blocks[off + 1]]));
        for i in 0..Q8_0_BLOCK_ELEMS {
            let idx = b * Q8_0_BLOCK_ELEMS + i;
            if idx >= n {
                break;
            }
            out[idx] = d * (blocks[off + 2 + i] as i8 as f32);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut c = [0.0; 4];
        matmul(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rmsnorm_unit_weight() {
        let x = [3.0, 4.0];
        let w = [1.0, 1.0];
        let mut out = [0.0; 2];
        rmsnorm(&x, &w, 1e-6, &mut out).unwrap();
        let rms = (25.0f32 / 2.0 + 1e-6).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rope_preserves_norm() {
        let mut x = [1.0, 0.0, 0.0, 1.0];
        let before: f32 = x.iter().map(|v| v * v).sum();
        rope(&mut x, 3, 10_000.0).unwrap();
        let after: f32 = x.iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn imrope_neox_partial_preserves_tail_and_norm() {
        // head_dim=8, n_rot=4 → rotate NeoX pairs (0,2) and (1,3); leave 4..7.
        let mut x = [1.0, 0.5, 0.0, 0.25, 9.0, 8.0, 7.0, 6.0];
        let before: f32 = x[..4].iter().map(|v| v * v).sum();
        rope_imrope_neox(&mut x, 2, 10_000.0, 4, [0, 0, 0, 0]).unwrap();
        let after: f32 = x[..4].iter().map(|v| v * v).sum();
        assert!((before - after).abs() < 1e-5);
        assert_eq!(&x[4..], &[9.0, 8.0, 7.0, 6.0]);
        // pair0: angle = 2
        let (s0, c0) = 2.0f32.sin_cos();
        assert!((x[0] - (1.0 * c0 - 0.0 * s0)).abs() < 1e-5);
        assert!((x[2] - (1.0 * s0 + 0.0 * c0)).abs() < 1e-5);
    }

    #[test]
    fn imrope_text_equal_pos_matches_plain_neox_for_qwen35_sections() {
        // Qwen3.5 sections [11,11,10,0]: every sector maps to t/h/w, so text
        // positions (pos,pos,pos,0) match plain NeoX partial RoPE.
        let sections = [11, 11, 10, 0];
        let mut a = vec![0.0f32; 256];
        for (i, v) in a.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }
        let mut b = a.clone();
        rope_imrope_neox(&mut a, 5, 10_000_000.0, 64, [0, 0, 0, 0]).unwrap();
        rope_imrope_neox(&mut b, 5, 10_000_000.0, 64, sections).unwrap();
        for i in 0..256 {
            assert!((a[i] - b[i]).abs() < 1e-4, "i={i} a={} b={}", a[i], b[i]);
        }
    }

    #[test]
    fn attention_single_query() {
        let q = [1.0, 0.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [2.0, 0.0, 0.0, 3.0];
        let mut out = [0.0; 2];
        attention(&q, &k, &v, &mut out, 1, 2, 1, 1, 2, false).unwrap();
        let sum = out[0] + out[1];
        assert!(sum.is_finite());
    }

    /// Scale bytes decode as unsigned finite E4M3 at half-scale, so codebook
    /// level 10 reconstructs 5.0 raw units at scale byte 0x38 (1.0).
    #[test]
    fn rocmfp4_scale_decodes_ue4m3_half() {
        assert_eq!(rocmfp4_scale(0), 0.0);
        assert_eq!(rocmfp4_scale(0x7f), 0.0);
        assert_eq!(rocmfp4_scale(0xff), 0.0);
        assert!(!rocmfp4_scale_is_valid(0x7f));
        assert!(rocmfp4_scale_is_valid(0x7e));
        assert!((rocmfp4_scale(0x40) - 1.0).abs() < 1e-6);
        assert!((rocmfp4_scale(0x38) - 0.5).abs() < 1e-6);
        assert!((rocmfp4_scale(0x44) - 1.5).abs() < 1e-6);
        // exp == 0 is the subnormal ramp: man / 1024.
        assert!((rocmfp4_scale(1) - 1.0 / 1024.0).abs() < 1e-9);
    }

    /// One hand-built block must decode with `qs` first and the two scale
    /// bytes last, low nibble → lane j, high nibble → lane j+16.
    #[test]
    fn rocmfp4_block_layout_is_qs_then_two_scales() {
        let mut block = vec![0u8; ROCMFP4_BLOCK_BYTES];
        // lane 0 → code 7 (+10), lane 16 → code 15 (-10).
        block[0] = 0x07 | (0x0f << 4);
        block[16] = 0x40; // scale 1.0 for lanes 0..15
        block[17] = 0x38; // scale 0.5 for lanes 16..31
        let mut out = vec![0.0f32; ROCMFP4_BLOCK_ELEMS];
        dequant_rocmfp4(&block, ROCMFP4_BLOCK_ELEMS, &mut out).unwrap();
        assert!((out[0] - 10.0).abs() < 1e-6, "lane 0 = {}", out[0]);
        assert!((out[16] + 5.0).abs() < 1e-6, "lane 16 = {}", out[16]);
        assert_eq!(out[1], 0.0);
    }

    /// Values that sit exactly on codebook levels must round-trip exactly.
    #[test]
    fn rocmfp4_round_trips_exact_codebook_levels() {
        let mut data = vec![0.0f32; ROCMFP4_BLOCK_ELEMS];
        for (i, slot) in data.iter_mut().enumerate() {
            let level = f32::from(ROCMFP4_CODEBOOK[i % 16]);
            *slot = level;
        }
        let packed = quantize_rocmfp4(&data);
        assert_eq!(packed.len(), ROCMFP4_BLOCK_BYTES);
        let mut out = vec![0.0f32; ROCMFP4_BLOCK_ELEMS];
        dequant_rocmfp4(&packed, ROCMFP4_BLOCK_ELEMS, &mut out).unwrap();
        for (i, (&want, &got)) in data.iter().zip(out.iter()).enumerate() {
            assert!((want - got).abs() < 1e-4, "lane {i}: want {want} got {got}");
        }
    }

    /// A smooth ramp must survive the scale search within 4-bit resolution.
    #[test]
    fn rocmfp4_round_trip_ramp_within_quant_error() {
        let n = ROCMFP4_BLOCK_ELEMS * 4;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.031).collect();
        let packed = quantize_rocmfp4(&data);
        let mut out = vec![0.0f32; n];
        dequant_rocmfp4(&packed, n, &mut out).unwrap();
        let max_abs = data.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for (i, (&want, &got)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (want - got).abs() <= max_abs * 0.2,
                "lane {i}: want {want} got {got}"
            );
        }
    }

    #[test]
    fn rocmfp4_rejects_truncated_blocks() {
        let short = vec![0u8; ROCMFP4_BLOCK_BYTES - 1];
        let mut out = vec![0.0f32; ROCMFP4_BLOCK_ELEMS];
        assert!(dequant_rocmfp4(&short, ROCMFP4_BLOCK_ELEMS, &mut out).is_err());
    }

    fn encode_iq4_nl_ones() -> Vec<u8> {
        let mut payload = vec![0u8; IQ4_NL_BLOCK_BYTES];
        payload[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        for slot in payload.iter_mut().skip(2) {
            *slot = 0x88;
        }
        payload
    }

    #[test]
    fn iq4_nl_dequants_codebook_nibble_8_to_one() {
        let packed = encode_iq4_nl_ones();
        let mut out = vec![0.0f32; IQ4_NL_BLOCK_ELEMS];
        dequant_iq4_nl(&packed, IQ4_NL_BLOCK_ELEMS, &mut out).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-4, "lane {i}: {v}");
        }
    }

    #[test]
    fn iq4_nl_rejects_truncated_blocks() {
        let short = vec![0u8; IQ4_NL_BLOCK_BYTES - 1];
        let mut out = vec![0.0f32; IQ4_NL_BLOCK_ELEMS];
        assert!(dequant_iq4_nl(&short, IQ4_NL_BLOCK_ELEMS, &mut out).is_err());
    }

    fn encode_iq4_xs_first_group_ones() -> Vec<u8> {
        let mut payload = vec![0u8; IQ4_XS_BLOCK_BYTES];
        payload[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // Every group ls=32 (dl=0); ib=0 uses ls=33 (dl=1).
        payload[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes());
        payload[4] = 0x01;
        for slot in payload.iter_mut().skip(8).take(16) {
            *slot = 0x88;
        }
        payload
    }

    #[test]
    fn iq4_xs_dequants_first_group_ones_and_zeros_the_rest() {
        let packed = encode_iq4_xs_first_group_ones();
        let mut out = vec![0.0f32; IQ4_XS_BLOCK_ELEMS];
        dequant_iq4_xs(&packed, IQ4_XS_BLOCK_ELEMS, &mut out).unwrap();
        for (i, v) in out.iter().enumerate().take(32) {
            assert!((v - 1.0).abs() < 1e-4, "lane {i}: {v}");
        }
        for (i, v) in out.iter().enumerate().skip(32) {
            assert!(v.abs() < 1e-5, "tail {i}: {v}");
        }
    }

    #[test]
    fn iq4_xs_rejects_truncated_blocks() {
        let short = vec![0u8; IQ4_XS_BLOCK_BYTES - 1];
        let mut out = vec![0.0f32; IQ4_XS_BLOCK_ELEMS];
        assert!(dequant_iq4_xs(&short, IQ4_XS_BLOCK_ELEMS, &mut out).is_err());
    }
}
