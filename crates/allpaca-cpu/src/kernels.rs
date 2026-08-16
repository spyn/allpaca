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
