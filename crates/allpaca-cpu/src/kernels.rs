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
    if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
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
    if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
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
    if !x.len().is_multiple_of(2) {
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
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;
/// GGUF Q4_K / Q4_K_S / Q4_K_M super-block (256 elems).
pub const Q4_K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;

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
    fn attention_single_query() {
        let q = [1.0, 0.0];
        let k = [1.0, 0.0, 0.0, 1.0];
        let v = [2.0, 0.0, 0.0, 3.0];
        let mut out = [0.0; 2];
        attention(&q, &k, &v, &mut out, 1, 2, 1, 1, 2, false).unwrap();
        let sum = out[0] + out[1];
        assert!(sum.is_finite());
    }
}
