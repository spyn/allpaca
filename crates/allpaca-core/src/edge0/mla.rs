//! Ling / DeepSeek-style Multi-head Latent Attention (MLA) host tensors.

/// MLA shape parameters for edge0-8b (Ling `bailing_hybrid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MlaDims {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
}

impl MlaDims {
    /// Sensible defaults when config omits explicit MLA ranks (scaled mini Ling).
    pub fn from_hidden(hidden: usize, n_heads: usize) -> Self {
        Self {
            hidden_size: hidden,
            num_heads: n_heads.max(1),
            q_lora_rank: (hidden / 8).max(16),
            kv_lora_rank: (hidden / 8).max(16),
            qk_nope_head_dim: 64,
            qk_rope_head_dim: 32,
            v_head_dim: 64,
        }
    }

    pub fn qk_head_dim(self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// Host MLA weights for one layer (dequantized / f32 working copy).
#[derive(Debug, Clone, Default)]
pub struct MlaHostWeights {
    pub dims: MlaDims,
    /// WDQ: `[q_lora_rank, hidden]`
    pub wdq: Vec<f32>,
    /// WUQ: `[n_heads * qk_head_dim, q_lora_rank]`
    pub wuq: Vec<f32>,
    /// WDKV: `[kv_lora_rank + qk_rope_head_dim, hidden]` compressed KV absorb
    pub wdkv: Vec<f32>,
    /// WUKV: `[n_heads * (qk_nope + v), kv_lora_rank]`
    pub wukv: Vec<f32>,
    pub wo: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
}

impl MlaHostWeights {
    pub fn synthetic(dims: MlaDims, fill: f32) -> Self {
        let qk = dims.qk_head_dim();
        let n = dims.num_heads;
        Self {
            dims,
            wdq: vec![fill; dims.q_lora_rank * dims.hidden_size],
            wuq: vec![fill; n * qk * dims.q_lora_rank],
            wdkv: vec![fill; (dims.kv_lora_rank + dims.qk_rope_head_dim) * dims.hidden_size],
            wukv: vec![fill; n * (dims.qk_nope_head_dim + dims.v_head_dim) * dims.kv_lora_rank],
            wo: vec![fill; dims.hidden_size * n * dims.v_head_dim],
            q_norm: vec![1.0; dims.q_lora_rank],
            kv_norm: vec![1.0; dims.kv_lora_rank],
        }
    }
}

/// Compressed MLA decode step (CPU reference).
///
/// Latent cache stores `c_kv` of length `kv_lora_rank` and rope k of `qk_rope_head_dim`
/// per position; this reference expands on the fly for correctness tests.
pub fn mla_attend_cpu(
    x: &[f32],
    w: &MlaHostWeights,
    cache_ckv: &mut Vec<f32>,
    cache_krope: &mut Vec<f32>,
    rope_theta: f32,
    eps: f32,
    out: &mut [f32],
) -> Result<(), allpaca_types::AllpacaError> {
    use allpaca_types::{AllpacaError, FatalError};
    let d = w.dims;
    if x.len() < d.hidden_size || out.len() < d.hidden_size {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "mla hidden size mismatch".into(),
        )));
    }
    let n_past = if d.kv_lora_rank == 0 {
        0
    } else {
        cache_ckv.len() / d.kv_lora_rank
    };
    // q_lora = rmsnorm(x @ WDQ^T)
    let mut q_lora = vec![0.0f32; d.q_lora_rank];
    for r in 0..d.q_lora_rank {
        let row = &w.wdq[r * d.hidden_size..(r + 1) * d.hidden_size];
        let mut s = 0.0f32;
        for i in 0..d.hidden_size {
            s += x[i] * row[i];
        }
        q_lora[r] = s;
    }
    rmsnorm_inplace(&mut q_lora, &w.q_norm, eps);

    let qk = d.qk_head_dim();
    let mut q = vec![0.0f32; d.num_heads * qk];
    for h in 0..d.num_heads {
        for j in 0..qk {
            let row = &w.wuq[(h * qk + j) * d.q_lora_rank..(h * qk + j + 1) * d.q_lora_rank];
            let mut s = 0.0f32;
            for r in 0..d.q_lora_rank {
                s += q_lora[r] * row[r];
            }
            q[h * qk + j] = s;
        }
    }
    // Apply RoPE to rope half of Q for current position.
    for h in 0..d.num_heads {
        let base = h * qk + d.qk_nope_head_dim;
        rope_inplace(&mut q[base..base + d.qk_rope_head_dim], n_past, rope_theta);
    }

    // Current compressed KV
    let kv_in = d.kv_lora_rank + d.qk_rope_head_dim;
    let mut kv_raw = vec![0.0f32; kv_in];
    for r in 0..kv_in {
        let row = &w.wdkv[r * d.hidden_size..(r + 1) * d.hidden_size];
        let mut s = 0.0f32;
        for i in 0..d.hidden_size {
            s += x[i] * row[i];
        }
        kv_raw[r] = s;
    }
    let mut c_kv = kv_raw[..d.kv_lora_rank].to_vec();
    rmsnorm_inplace(&mut c_kv, &w.kv_norm, eps);
    let mut k_rope = kv_raw[d.kv_lora_rank..].to_vec();
    rope_inplace(&mut k_rope, n_past, rope_theta);

    let seq = n_past + 1;
    let mut ctx = vec![0.0f32; d.num_heads * d.v_head_dim];
    let scale = 1.0 / (qk as f32).sqrt();
    for h in 0..d.num_heads {
        let qh = &q[h * qk..(h + 1) * qk];
        let mut scores = vec![0.0f32; seq];
        for t in 0..n_past {
            let c = &cache_ckv[t * d.kv_lora_rank..(t + 1) * d.kv_lora_rank];
            let kr = &cache_krope[t * d.qk_rope_head_dim..(t + 1) * d.qk_rope_head_dim];
            let (k, v) = expand_kv(c, kr, w, h);
            let mut s = 0.0f32;
            for i in 0..qk {
                s += qh[i] * k[i];
            }
            scores[t] = s * scale;
            let _ = v;
        }
        let (k_cur, v_cur) = expand_kv(&c_kv, &k_rope, w, h);
        let mut s = 0.0f32;
        for i in 0..qk {
            s += qh[i] * k_cur[i];
        }
        scores[n_past] = s * scale;
        softmax_inplace(&mut scores);
        let mut acc = vec![0.0f32; d.v_head_dim];
        for t in 0..n_past {
            let c = &cache_ckv[t * d.kv_lora_rank..(t + 1) * d.kv_lora_rank];
            let kr = &cache_krope[t * d.qk_rope_head_dim..(t + 1) * d.qk_rope_head_dim];
            let (_, v) = expand_kv(c, kr, w, h);
            for i in 0..d.v_head_dim {
                acc[i] += scores[t] * v[i];
            }
        }
        for i in 0..d.v_head_dim {
            acc[i] += scores[n_past] * v_cur[i];
        }
        ctx[h * d.v_head_dim..(h + 1) * d.v_head_dim].copy_from_slice(&acc);
    }

    out[..d.hidden_size].fill(0.0);
    let v_all = d.num_heads * d.v_head_dim;
    for o in 0..d.hidden_size {
        let row = &w.wo[o * v_all..(o + 1) * v_all];
        let mut s = 0.0f32;
        for i in 0..v_all {
            s += ctx[i] * row[i];
        }
        out[o] = s;
    }
    cache_ckv.extend_from_slice(&c_kv);
    cache_krope.extend_from_slice(&k_rope);
    Ok(())
}

fn expand_kv(
    c_kv: &[f32],
    k_rope: &[f32],
    w: &MlaHostWeights,
    head: usize,
) -> (Vec<f32>, Vec<f32>) {
    let d = w.dims;
    let row_w = d.qk_nope_head_dim + d.v_head_dim;
    let base = head * row_w * d.kv_lora_rank;
    let mut nope = vec![0.0f32; d.qk_nope_head_dim];
    let mut v = vec![0.0f32; d.v_head_dim];
    for j in 0..d.qk_nope_head_dim {
        let row = &w.wukv[base + j * d.kv_lora_rank..base + (j + 1) * d.kv_lora_rank];
        let mut s = 0.0f32;
        for r in 0..d.kv_lora_rank {
            s += c_kv[r] * row[r];
        }
        nope[j] = s;
    }
    for j in 0..d.v_head_dim {
        let row = &w.wukv
            [base + (d.qk_nope_head_dim + j) * d.kv_lora_rank
                ..base + (d.qk_nope_head_dim + j + 1) * d.kv_lora_rank];
        let mut s = 0.0f32;
        for r in 0..d.kv_lora_rank {
            s += c_kv[r] * row[r];
        }
        v[j] = s;
    }
    let mut k = Vec::with_capacity(d.qk_head_dim());
    k.extend_from_slice(&nope);
    k.extend_from_slice(k_rope);
    (k, v)
}

fn rmsnorm_inplace(x: &mut [f32], w: &[f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        let scale = w.get(i).copied().unwrap_or(1.0);
        *v *= inv * scale;
    }
}

fn rope_inplace(x: &mut [f32], pos: usize, theta: f32) {
    let half = x.len() / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / x.len() as f32);
        let angle = pos as f32 * freq;
        let (c, s) = (angle.cos(), angle.sin());
        let a = x[i];
        let b = x[i + half];
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-9);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mla_cpu_runs_empty_cache() {
        let dims = MlaDims::from_hidden(32, 2);
        let w = MlaHostWeights::synthetic(dims, 0.01);
        let x = vec![0.1f32; 32];
        let mut out = vec![0.0f32; 32];
        let mut ckv = Vec::new();
        let mut krope = Vec::new();
        mla_attend_cpu(&x, &w, &mut ckv, &mut krope, 10_000.0, 1e-5, &mut out).unwrap();
        assert!(out.iter().any(|v| *v != 0.0));
        assert!(!ckv.is_empty());
        let ckv_len = ckv.len();
        mla_attend_cpu(&x, &w, &mut ckv, &mut krope, 10_000.0, 1e-5, &mut out).unwrap();
        assert_eq!(ckv.len(), ckv_len * 2);
    }
}
