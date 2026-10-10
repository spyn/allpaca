//! GSQ (Gumbel-Softmax Quantization, arXiv 2604.18556) onto ggml `Q2_0`.
//!
//! Every weight chooses one of the grid values -2, -1, 0, +1 times its
//! group's scale (a group is a `Q2_0` block of 64 weights; the scale's sign
//! is free). The choice is relaxed to a softmax over per-weight logits with
//! Gumbel noise and a temperature that anneals to near zero, and the logits
//! and scales are trained with Adam to minimise the layer's output error
//! `sum over rows of (w - q) H (w - q)^T`, where `H` is the Gram matrix of
//! the layer's inputs. At the end each weight takes its most likely value.
//!
//! Follows IST-DASLab/GSQ (`src/quantization/gumbel_quantizer_2bit.py`).
//! The reference starts from a GPTQ solution and trains against the block's
//! output on calibration data; this starts from the best scale per group
//! under rounding and trains against one layer's Gram matrix.

use allpaca_cpu::{f32_to_f16, Q2_0_BLOCK_BYTES, Q2_0_BLOCK_ELEMS};
use allpaca_types::{AllpacaError, FatalError};

/// The grid, in units of the group's scale.
const VALUES: [f32; 4] = [-2.0, -1.0, 0.0, 1.0];

/// Training settings.
#[derive(Debug, Clone)]
pub struct GsqConfig {
    pub steps: usize,
    pub learning_rate: f32,
    /// Softmax temperature at the first and last step (annealed geometrically).
    pub temperature: (f32, f32),
    /// How sharply the initial logits favour the nearest grid value.
    pub init_strength: f32,
    pub seed: u64,
}

impl Default for GsqConfig {
    fn default() -> Self {
        Self {
            steps: 300,
            learning_rate: 0.05,
            temperature: (1.0, 0.05),
            init_strength: 4.0,
            seed: 0,
        }
    }
}

/// The inputs' second moments, which weight the error.
pub enum Gram<'a> {
    /// Inputs taken as uncorrelated with unit variance: plain squared error.
    Identity,
    /// `cols * cols`, symmetric: `H[i * cols + j] = E[x_i x_j]`.
    Full(&'a [f32]),
}

/// A quantised matrix: grid index per weight and scale per group.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantised {
    pub rows: usize,
    pub cols: usize,
    /// Index into the grid (-2, -1, 0, +1) per weight, row-major.
    pub index: Vec<u8>,
    /// Scale per group of 64 weights, row-major.
    pub scales: Vec<f32>,
}

impl Quantised {
    fn groups_per_row(&self) -> usize {
        self.cols / Q2_0_BLOCK_ELEMS
    }

    /// The weights this stands for.
    pub fn weights(&self) -> Vec<f32> {
        let gpr = self.groups_per_row();
        self.index
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let (r, c) = (i / self.cols, i % self.cols);
                VALUES[usize::from(k)] * self.scales[r * gpr + c / Q2_0_BLOCK_ELEMS]
            })
            .collect()
    }

    /// ggml `Q2_0` blocks. Its codes stand for -1, 0, +1, +2 times `d`,
    /// the mirror of the grid here, so `d` is the negated scale. The scale
    /// is stored as a half, so decoded weights are within its rounding.
    pub fn to_q2_0(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.scales.len() * Q2_0_BLOCK_BYTES);
        for (codes, scale) in self.index.chunks(Q2_0_BLOCK_ELEMS).zip(&self.scales) {
            out.extend_from_slice(&f32_to_f16(-scale).to_le_bytes());
            for quad in codes.chunks(4) {
                // Grid index k is value k - 2; the Q2_0 code for value v is 1 - v.
                out.push(quad.iter().enumerate().fold(0u8, |b, (j, &k)| b | ((3 - k) << (2 * j))));
            }
        }
        out
    }
}

/// Output error of `q` against `w`: `sum over rows of (w - q) H (w - q)^T`.
pub fn output_error(w: &[f32], q: &[f32], cols: usize, gram: &Gram<'_>) -> f64 {
    let mut total = 0.0f64;
    let mut hd = vec![0.0f32; cols];
    for (wr, qr) in w.chunks(cols).zip(q.chunks(cols)) {
        let d: Vec<f32> = wr.iter().zip(qr).map(|(a, b)| a - b).collect();
        times_gram(&d, gram, &mut hd);
        total += d.iter().zip(&hd).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>();
    }
    total
}

fn times_gram(d: &[f32], gram: &Gram<'_>, out: &mut [f32]) {
    match gram {
        Gram::Identity => out.copy_from_slice(d),
        Gram::Full(h) => {
            for (o, row) in out.iter_mut().zip(h.chunks(d.len())) {
                *o = row.iter().zip(d).map(|(a, b)| a * b).sum();
            }
        }
    }
}

fn nearest(x: f32) -> u8 {
    (x.round().clamp(-2.0, 1.0) + 2.0) as u8
}

/// Rounding to the grid with, per group, the scale (of either sign) that
/// gives the least squared error: where the search starts.
fn round_with_best_scales(w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut index = vec![0u8; w.len()];
    let mut scales = Vec::with_capacity(w.len() / Q2_0_BLOCK_ELEMS);
    for (src, dst) in w.chunks(Q2_0_BLOCK_ELEMS).zip(index.chunks_mut(Q2_0_BLOCK_ELEMS)) {
        let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let mut best = (f32::INFINITY, 0.0f32);
        if amax > 0.0 {
            for step in 4..=40 {
                for sign in [1.0f32, -1.0] {
                    let s = sign * amax * step as f32 / 40.0;
                    let err: f32 = src.iter().map(|v| (v - VALUES[usize::from(nearest(v / s))] * s).powi(2)).sum();
                    if err < best.0 {
                        best = (err, s);
                    }
                }
            }
        }
        let s = best.1;
        for (d, v) in dst.iter_mut().zip(src) {
            *d = if s == 0.0 { 2 } else { nearest(v / s) };
        }
        scales.push(s);
    }
    (index, scales)
}

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    fn gumbel(&mut self) -> f32 {
        -(-self.uniform().ln()).ln()
    }
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
        }
    }

    fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32, t: i32) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        for i in 0..params.len() {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad[i] * grad[i];
            params[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}

/// Quantise `w` (`rows * cols`, row-major; `cols` a multiple of 64) to the
/// 2-bit grid. The result is never worse, by [`output_error`], than rounding
/// with the best scale per group.
pub fn quantise_2bit(
    w: &[f32],
    rows: usize,
    cols: usize,
    gram: &Gram<'_>,
    cfg: &GsqConfig,
) -> Result<Quantised, AllpacaError> {
    let bad = |msg: &str| AllpacaError::Fatal(FatalError::InvalidConfig(format!("GSQ: {msg}")));
    if w.len() != rows * cols || cols == 0 || cols % Q2_0_BLOCK_ELEMS != 0 {
        return Err(bad("weights must be rows * cols with cols a multiple of 64"));
    }
    if let Gram::Full(h) = gram {
        if h.len() != cols * cols {
            return Err(bad("Gram matrix must be cols * cols"));
        }
    }
    let gpr = cols / Q2_0_BLOCK_ELEMS;
    let group_of = |i: usize| (i / cols) * gpr + (i % cols) / Q2_0_BLOCK_ELEMS;
    let (index0, scales0) = round_with_best_scales(w);
    let start = Quantised {
        rows,
        cols,
        index: index0,
        scales: scales0,
    };
    let start_err = output_error(w, &start.weights(), cols, gram);

    // Logits favour each weight's nearest grid values at the starting scale.
    let n = w.len();
    let mut logits = vec![0.0f32; n * 4];
    for i in 0..n {
        let s = start.scales[group_of(i)];
        let x = if s == 0.0 { 0.0 } else { w[i] / s };
        let raw: Vec<f32> = VALUES.iter().map(|v| -0.5 * (x - v).powi(2)).collect();
        let mean = raw.iter().sum::<f32>() / 4.0;
        for k in 0..4 {
            logits[i * 4 + k] = (raw[k] - mean) * cfg.init_strength;
        }
    }
    let mut scales = start.scales.clone();
    let mut rng = Rng(cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let (mut opt_logits, mut opt_scales) = (Adam::new(n * 4), Adam::new(scales.len()));
    let mut probs = vec![0.0f32; n * 4];
    let mut soft = vec![0.0f32; n];
    let mut grad_out = vec![0.0f32; n];
    let mut grad_logits = vec![0.0f32; n * 4];
    let mut grad_scales = vec![0.0f32; scales.len()];
    let mut diff = vec![0.0f32; cols];
    // Scales move in proportion to their size.
    let scale_unit = scales.iter().map(|s| s.abs()).sum::<f32>() / scales.len().max(1) as f32;
    for step in 0..cfg.steps {
        let frac = step as f32 / cfg.steps.max(2).saturating_sub(1) as f32;
        let tau = cfg.temperature.0 * (cfg.temperature.1 / cfg.temperature.0).powf(frac);
        for i in 0..n {
            let z: Vec<f32> = (0..4).map(|k| (logits[i * 4 + k] + rng.gumbel()) / tau).collect();
            let max = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = z.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            soft[i] = 0.0;
            for k in 0..4 {
                probs[i * 4 + k] = exps[k] / sum;
                soft[i] += probs[i * 4 + k] * VALUES[k];
            }
        }
        // d(error)/d(output) = 2 (q - w) H per row.
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                diff[c] = soft[i] * scales[group_of(i)] - w[i];
            }
            times_gram(&diff, gram, &mut grad_out[r * cols..(r + 1) * cols]);
        }
        grad_scales.fill(0.0);
        for i in 0..n {
            let g = 2.0 * grad_out[i];
            let s = scales[group_of(i)];
            grad_scales[group_of(i)] += g * soft[i];
            let gq: Vec<f32> = VALUES.iter().map(|v| g * s * v).collect();
            let dot: f32 = (0..4).map(|k| gq[k] * probs[i * 4 + k]).sum();
            for k in 0..4 {
                grad_logits[i * 4 + k] = probs[i * 4 + k] * (gq[k] - dot) / tau;
            }
        }
        let t = (step + 1) as i32;
        opt_logits.step(&mut logits, &grad_logits, cfg.learning_rate, t);
        opt_scales.step(&mut scales, &grad_scales, cfg.learning_rate * 0.1 * scale_unit, t);
    }
    let index: Vec<u8> = logits
        .chunks(4)
        .map(|l| (0..4u8).max_by(|a, b| l[usize::from(*a)].total_cmp(&l[usize::from(*b)])).unwrap_or(2))
        .collect();
    let trained = Quantised {
        rows,
        cols,
        index,
        scales,
    };
    if output_error(w, &trained.weights(), cols, gram) < start_err {
        Ok(trained)
    } else {
        Ok(start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed | 1);
        (0..n).map(|_| rng.uniform() + rng.uniform() + rng.uniform() - 1.5).collect()
    }

    /// Gram matrix of inputs that share a few common factors, so columns correlate.
    fn correlated_gram(cols: usize, seed: u64) -> Vec<f32> {
        let factors = 8;
        let load = noise(cols * factors, seed);
        let mut h = vec![0.0f32; cols * cols];
        for i in 0..cols {
            for j in 0..cols {
                h[i * cols + j] = (0..factors).map(|f| load[i * factors + f] * load[j * factors + f]).sum::<f32>()
                    + if i == j { 0.05 } else { 0.0 };
            }
        }
        h
    }

    #[test]
    fn q2_0_blocks_decode_to_the_quantised_weights() {
        let q = Quantised {
            rows: 1,
            cols: 64,
            index: (0..64).map(|i| (i % 4) as u8).collect(),
            scales: vec![-0.5],
        };
        let want = q.weights();
        assert_eq!(&want[..4], &[1.0, 0.5, 0.0, -0.5]);
        let blocks = q.to_q2_0();
        assert_eq!(blocks.len(), Q2_0_BLOCK_BYTES);
        let mut got = vec![0.0f32; 64];
        allpaca_cpu::dequant_q2_0(&blocks, 64, &mut got).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn plain_error_is_no_worse_than_ggmls_rounding() {
        let (rows, cols) = (6usize, 128usize);
        let w = noise(rows * cols, 11);
        let q = quantise_2bit(&w, rows, cols, &Gram::Identity, &GsqConfig::default()).unwrap();
        assert!(q.index.iter().all(|k| *k < 4));
        let ours = output_error(&w, &q.weights(), cols, &Gram::Identity);
        let mut ggml = vec![0.0f32; w.len()];
        allpaca_cpu::dequant_q2_0(&allpaca_cpu::quantize_q2_0(&w), w.len(), &mut ggml).unwrap();
        let theirs = output_error(&w, &ggml, cols, &Gram::Identity);
        // ggml's scale is the largest magnitude, which leaves most weights at zero.
        assert!(ours < 0.6 * theirs, "GSQ {ours} vs ggml rounding {theirs}");
    }

    #[test]
    fn training_against_the_inputs_gram_lowers_the_output_error() {
        let (rows, cols) = (4usize, 64usize);
        let w = noise(rows * cols, 5);
        let h = correlated_gram(cols, 9);
        let gram = Gram::Full(&h);
        let (index, scales) = round_with_best_scales(&w);
        let rounded = Quantised { rows, cols, index, scales };
        let before = output_error(&w, &rounded.weights(), cols, &gram);
        let cfg = GsqConfig::default();
        let q = quantise_2bit(&w, rows, cols, &gram, &cfg).unwrap();
        let after = output_error(&w, &q.weights(), cols, &gram);
        // Correlated inputs let errors on different weights cancel, which
        // rounding each weight on its own cannot use.
        assert!(after < 0.7 * before, "rounded {before} -> trained {after}");
        // Same seed, same result; the packed blocks carry it (scales to half precision).
        assert_eq!(q, quantise_2bit(&w, rows, cols, &gram, &cfg).unwrap());
        let mut decoded = vec![0.0f32; w.len()];
        allpaca_cpu::dequant_q2_0(&q.to_q2_0(), w.len(), &mut decoded).unwrap();
        let packed = output_error(&w, &decoded, cols, &gram);
        assert!((packed - after).abs() < 0.02 * after.max(1e-6), "{packed} vs {after}");
    }

    #[test]
    fn shapes_that_are_not_whole_blocks_are_refused() {
        assert!(quantise_2bit(&[0.0; 60], 1, 60, &Gram::Identity, &GsqConfig::default()).is_err());
        assert!(quantise_2bit(&[0.0; 64], 1, 64, &Gram::Full(&[0.0; 9]), &GsqConfig::default()).is_err());
    }
}
