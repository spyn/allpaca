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
pub(crate) const VALUES: [f32; 4] = [-2.0, -1.0, 0.0, 1.0];

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
    /// Matrices with more weights than this keep their start untrained:
    /// a step costs a few microseconds a weight on the CPU.
    pub max_trained: usize,
    /// Each group's rounding scale times this, rounded again. The scale
    /// with the least weight error clips the largest weights, which the
    /// model needs: on Qwen3.8-Flash-Next's Q2_0 experts 1.15 cut KL to
    /// the Q8_0 from 0.110 to 0.097 (1.07 gave 0.0995, 1.25 0.110).
    pub stretch: f32,
}

impl Default for GsqConfig {
    fn default() -> Self {
        Self {
            steps: 300,
            learning_rate: 0.05,
            temperature: (1.0, 0.05),
            init_strength: 4.0,
            seed: 0,
            max_trained: 1 << 22,
            stretch: 1.0,
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

/// Weights handled at a time where a whole matrix would take gigabytes
/// (the search keeps about 23 floats per weight).
const CHUNK_WEIGHTS: usize = 1 << 24;

/// Output error of `q` against `w`: `sum over rows of (w - q) H (w - q)^T`.
pub fn output_error(w: &[f32], q: &[f32], cols: usize, gram: &Gram<'_>) -> f64 {
    let mut total = 0.0f64;
    for (w, q) in w.chunks(chunk_rows(cols) * cols).zip(q.chunks(chunk_rows(cols) * cols)) {
        add_output_error(&mut total, w, |i| q[i], cols, gram);
    }
    total
}

/// [`output_error`] against zero: the output's own size.
pub fn output_energy(w: &[f32], cols: usize, gram: &Gram<'_>) -> f64 {
    let mut total = 0.0f64;
    for w in w.chunks(chunk_rows(cols) * cols) {
        add_output_error(&mut total, w, |_| 0.0, cols, gram);
    }
    total
}

/// [`output_error`] of a quantised matrix, decoded a chunk of rows at a time.
pub fn quantised_error(w: &[f32], q: &Quantised, gram: &Gram<'_>) -> f64 {
    let (cols, gpr) = (q.cols, q.groups_per_row());
    let mut total = 0.0f64;
    let rows = chunk_rows(cols);
    for (c, w) in w.chunks(rows * cols).enumerate() {
        let (first, groups) = (c * rows * cols, c * rows * gpr);
        let value = |i: usize| {
            let at = first + i;
            VALUES[usize::from(q.index[at])] * q.scales[groups + (i / cols) * gpr + (i % cols) / Q2_0_BLOCK_ELEMS]
        };
        add_output_error(&mut total, w, value, cols, gram);
    }
    total
}

fn chunk_rows(cols: usize) -> usize {
    (CHUNK_WEIGHTS / cols.max(1)).max(1)
}

/// Add whole rows' error to `total`, element by element in order.
fn add_output_error(total: &mut f64, w: &[f32], q: impl Fn(usize) -> f32, cols: usize, gram: &Gram<'_>) {
    let d: Vec<f32> = w.iter().enumerate().map(|(i, a)| a - q(i)).collect();
    let Gram::Full(_) = gram else {
        *total = d.iter().fold(*total, |t, v| t + f64::from(*v) * f64::from(*v));
        return;
    };
    let mut hd = vec![0.0f32; d.len()];
    rows_times_gram(&d, cols, gram, &mut hd);
    *total = d.iter().zip(&hd).fold(*total, |t, (a, b)| t + f64::from(*a) * f64::from(*b));
}

/// `out = d H` for every row of `d` (rows of `cols`), on all cores when the
/// product is large enough to be worth the threads.
fn rows_times_gram(d: &[f32], cols: usize, gram: &Gram<'_>, out: &mut [f32]) {
    let Gram::Full(h) = gram else {
        out.copy_from_slice(d);
        return;
    };
    let rows = d.len() / cols;
    let run = |d: &[f32], out: &mut [f32]| {
        for (dr, or) in d.chunks(cols).zip(out.chunks_mut(cols)) {
            for (o, hrow) in or.iter_mut().zip(h.chunks(cols)) {
                *o = allpaca_cpu::dot(hrow, dr);
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows);
    if threads < 2 || d.len() * cols < (1 << 24) {
        run(d, out);
        return;
    }
    let per = rows.div_ceil(threads) * cols;
    std::thread::scope(|s| {
        for (dc, oc) in d.chunks(per).zip(out.chunks_mut(per)) {
            s.spawn(|| run(dc, oc));
        }
    });
}

pub(crate) fn nearest(x: f32) -> u8 {
    (x.round().clamp(-2.0, 1.0) + 2.0) as u8
}

/// Threads for a pass over `n` values, each on its own (the result does not
/// depend on the split).
fn pass_threads(n: usize) -> usize {
    if n < (1 << 16) {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |t| t.get())
    }
}

/// Rounding to the grid with, per group, the scale (of either sign) that
/// gives the least squared error: where the search starts. Groups are
/// spread over the cores.
fn round_with_best_scales(w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut index = vec![0u8; w.len()];
    let mut scales = vec![0.0f32; w.len().div_ceil(Q2_0_BLOCK_ELEMS)];
    let groups = scales.len();
    let per = groups.div_ceil(pass_threads(w.len())).max(1);
    std::thread::scope(|s| {
        let parts = w.chunks(per * Q2_0_BLOCK_ELEMS).zip(index.chunks_mut(per * Q2_0_BLOCK_ELEMS)).zip(scales.chunks_mut(per));
        for ((w, index), scales) in parts {
            s.spawn(move || round_groups(w, index, scales));
        }
    });
    (index, scales)
}

/// Each group's scale times `stretch`, and its weights rounded to it.
fn stretch_groups(w: &[f32], index: &mut [u8], scales: &mut [f32], stretch: f32) {
    let per = scales.len().div_ceil(pass_threads(w.len())).max(1);
    std::thread::scope(|s| {
        let parts = w.chunks(per * Q2_0_BLOCK_ELEMS).zip(index.chunks_mut(per * Q2_0_BLOCK_ELEMS)).zip(scales.chunks_mut(per));
        for ((w, index), scales) in parts {
            s.spawn(move || {
                for ((src, dst), scale) in w.chunks(Q2_0_BLOCK_ELEMS).zip(index.chunks_mut(Q2_0_BLOCK_ELEMS)).zip(scales.iter_mut()) {
                    *scale *= stretch;
                    for (d, v) in dst.iter_mut().zip(src) {
                        *d = if *scale == 0.0 { 2 } else { nearest(v / *scale) };
                    }
                }
            });
        }
    });
}

pub(crate) fn round_groups(w: &[f32], index: &mut [u8], scales: &mut [f32]) {
    let groups = w.chunks(Q2_0_BLOCK_ELEMS).zip(index.chunks_mut(Q2_0_BLOCK_ELEMS)).zip(scales.iter_mut());
    for ((src, dst), scale) in groups {
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
        *scale = s;
    }
}

struct Rng(u64);

impl Rng {
    /// The stream of one row at one step: rows are sampled on separate
    /// threads, and the result must not depend on how they are split.
    fn for_row(seed: u64, step: usize, row: usize) -> Self {
        let mut x = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((step as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
            .wrapping_add((row as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
        x ^= x >> 31;
        x = x.wrapping_mul(0xD6E8_FEB8_6659_FD93);
        x ^= x >> 32;
        Self(x | 1)
    }

    /// Strictly inside (0, 1). With 24 bits the top draw, `2^24 - 0.5`,
    /// rounds to `2^24` in f32: a 1.0 makes the Gumbel noise infinite.
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 41) as f32 + 0.5) / (1u64 << 23) as f32
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

    /// One step for every parameter, spread over the cores.
    fn step(&mut self, params: &mut [f32], grad: &[f32], lr: f32, t: i32) {
        let (b1, b2) = (0.9f32, 0.999f32);
        let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
        let run = |m: &mut [f32], v: &mut [f32], params: &mut [f32], grad: &[f32]| {
            for i in 0..params.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
                v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
                params[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8);
            }
        };
        let per = params.len().div_ceil(pass_threads(params.len())).max(1);
        std::thread::scope(|s| {
            let parts = self.m.chunks_mut(per).zip(self.v.chunks_mut(per)).zip(params.chunks_mut(per)).zip(grad.chunks(per));
            for (((m, v), p), g) in parts {
                let run = &run;
                s.spawn(move || run(m, v, p, g));
            }
        });
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
    quantise_2bit_by(w, rows, cols, gram, cfg, chunk_rows(cols))
}

/// [`quantise_2bit`] training `per` rows at a time.
fn quantise_2bit_by(
    w: &[f32],
    rows: usize,
    cols: usize,
    gram: &Gram<'_>,
    cfg: &GsqConfig,
    per: usize,
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
    let (mut index0, mut scales0) = round_with_best_scales(w);
    if cfg.stretch != 1.0 {
        stretch_groups(w, &mut index0, &mut scales0, cfg.stretch);
    }
    let mut start = Quantised {
        rows,
        cols,
        index: index0,
        scales: scales0,
    };
    let mut start_error = quantised_error(w, &start, gram);
    // Against measured inputs GPTQ starts closer, as in the reference.
    if let Gram::Full(h) = gram {
        let gptq = crate::gptq::gptq_2bit(w, rows, cols, h)?;
        let e = quantised_error(w, &gptq, gram);
        if e < start_error {
            (start, start_error) = (gptq, e);
        }
    }
    if cfg.steps == 0 || w.len() > cfg.max_trained {
        return Ok(start);
    }
    // Scales move in proportion to their size.
    let scale_unit = start.scales.iter().map(|s| s.abs()).sum::<f32>() / start.scales.len().max(1) as f32;
    // Rows train on their own (each has its own noise), so a large matrix
    // goes a chunk of rows at a time: the same result in less memory.
    let mut index = Vec::with_capacity(w.len());
    let mut scales = Vec::with_capacity(start.scales.len());
    for first in (0..rows).step_by(per.max(1)) {
        let n_rows = per.min(rows - first);
        let (i, s) = train(
            &w[first * cols..(first + n_rows) * cols],
            first,
            cols,
            &start.index[first * cols..(first + n_rows) * cols],
            &start.scales[first * gpr..(first + n_rows) * gpr],
            scale_unit,
            gram,
            cfg,
        );
        index.extend(i);
        scales.extend(s);
    }
    let trained = Quantised {
        rows,
        cols,
        index,
        scales,
    };
    if quantised_error(w, &trained, gram) < start_error {
        Ok(trained)
    } else {
        Ok(start)
    }
}

/// The search on whole rows of a matrix, the first being row `row0`, from
/// `start_index` and `start_scales`: each weight's grid index and the
/// trained scales.
#[allow(clippy::too_many_arguments)]
fn train(
    w: &[f32],
    row0: usize,
    cols: usize,
    start_index: &[u8],
    start_scales: &[f32],
    scale_unit: f32,
    gram: &Gram<'_>,
    cfg: &GsqConfig,
) -> (Vec<u8>, Vec<f32>) {
    let rows = w.len() / cols;
    let gpr = cols / Q2_0_BLOCK_ELEMS;
    // Logits favour each weight's start value, by how far the weight is
    // from the grid values at the starting scale (its nearest's place
    // taken by the start's choice: the same when the start is rounding).
    let n = w.len();
    let mut logits = vec![0.0f32; n * 4];
    // Whole groups per thread (a group's weights are consecutive).
    let per_groups = n.div_ceil(Q2_0_BLOCK_ELEMS).div_ceil(pass_threads(n)).max(1);
    let per_weights = per_groups * Q2_0_BLOCK_ELEMS;
    std::thread::scope(|scope| {
        let parts = w
            .chunks(per_weights)
            .zip(logits.chunks_mut(per_weights * 4))
            .zip(start_scales.chunks(per_groups))
            .zip(start_index.chunks(per_weights));
        for (((w, logits), scales), index) in parts {
            scope.spawn(move || {
                for (i, x) in w.iter().enumerate() {
                    let s = scales[i / Q2_0_BLOCK_ELEMS];
                    let x = if s == 0.0 { 0.0 } else { x / s };
                    let x = x - VALUES[usize::from(nearest(x))] + VALUES[usize::from(index[i])];
                    let raw = VALUES.map(|v| -0.5 * (x - v).powi(2));
                    let mean = raw.iter().sum::<f32>() / 4.0;
                    for k in 0..4 {
                        logits[i * 4 + k] = (raw[k] - mean) * cfg.init_strength;
                    }
                }
            });
        }
    });
    let mut scales = start_scales.to_vec();
    // Rows are independent in everything but the Gram product: whole rows
    // per thread once the matrix is large enough to pay for the threads.
    let threads = if n < (1 << 16) { 1 } else { std::thread::available_parallelism().map_or(1, |t| t.get()).min(rows) };
    let per = rows.div_ceil(threads);
    let (mut opt_logits, mut opt_scales) = (Adam::new(n * 4), Adam::new(scales.len()));
    let mut probs = vec![0.0f32; n * 4];
    let mut soft = vec![0.0f32; n];
    let mut grad_out = vec![0.0f32; n];
    let mut grad_logits = vec![0.0f32; n * 4];
    let mut grad_scales = vec![0.0f32; scales.len()];
    let mut diff = vec![0.0f32; n];
    for step in 0..cfg.steps {
        let frac = step as f32 / cfg.steps.max(2).saturating_sub(1) as f32;
        let tau = cfg.temperature.0 * (cfg.temperature.1 / cfg.temperature.0).powf(frac);
        // Relaxed sample of every weight: softmax((logits + Gumbel noise) / tau).
        let sample = |first_row: usize, logits: &[f32], probs: &mut [f32], soft: &mut [f32]| {
            for (r, soft_row) in soft.chunks_mut(cols).enumerate() {
                let mut rng = Rng::for_row(cfg.seed, step, row0 + first_row + r);
                for (c, s) in soft_row.iter_mut().enumerate() {
                    let at = (r * cols + c) * 4;
                    let mut z = [0.0f32; 4];
                    let mut max = f32::NEG_INFINITY;
                    for k in 0..4 {
                        z[k] = (logits[at + k] + rng.gumbel()) / tau;
                        max = max.max(z[k]);
                    }
                    let mut sum = 0.0f32;
                    for v in &mut z {
                        *v = (*v - max).exp();
                        sum += *v;
                    }
                    *s = 0.0;
                    for k in 0..4 {
                        probs[at + k] = z[k] / sum;
                        *s += probs[at + k] * VALUES[k];
                    }
                }
            }
        };
        if threads < 2 {
            sample(0, &logits, &mut probs, &mut soft);
        } else {
            std::thread::scope(|s| {
                let sample = &sample;
                let chunks = logits.chunks(per * cols * 4).zip(probs.chunks_mut(per * cols * 4)).zip(soft.chunks_mut(per * cols));
                for (i, ((l, p), so)) in chunks.enumerate() {
                    s.spawn(move || sample(i * per, l, p, so));
                }
            });
        }
        // d(error)/d(output) = 2 (q - w) H per row.
        std::thread::scope(|scope| {
            let parts =
                diff.chunks_mut(per_weights).zip(soft.chunks(per_weights)).zip(w.chunks(per_weights)).zip(scales.chunks(per_groups));
            for (((diff, soft), w), scales) in parts {
                scope.spawn(move || {
                    for (i, d) in diff.iter_mut().enumerate() {
                        *d = soft[i] * scales[i / Q2_0_BLOCK_ELEMS] - w[i];
                    }
                });
            }
        });
        rows_times_gram(&diff, cols, gram, &mut grad_out);
        grad_scales.fill(0.0);
        // Chain rule through the scale and the softmax; a chunk's groups are its own.
        let backward = |scales: &[f32],
                        soft: &[f32],
                        probs: &[f32],
                        grad_out: &[f32],
                        grad_logits: &mut [f32],
                        grad_scales: &mut [f32]| {
            for i in 0..soft.len() {
                let group = (i / cols) * gpr + (i % cols) / Q2_0_BLOCK_ELEMS;
                let g = 2.0 * grad_out[i];
                let s = scales[group];
                grad_scales[group] += g * soft[i];
                let mut gq = [0.0f32; 4];
                let mut dot = 0.0f32;
                for k in 0..4 {
                    gq[k] = g * s * VALUES[k];
                    dot += gq[k] * probs[i * 4 + k];
                }
                for k in 0..4 {
                    grad_logits[i * 4 + k] = probs[i * 4 + k] * (gq[k] - dot) / tau;
                }
            }
        };
        if threads < 2 {
            backward(&scales, &soft, &probs, &grad_out, &mut grad_logits, &mut grad_scales);
        } else {
            std::thread::scope(|s| {
                let backward = &backward;
                let chunks = scales
                    .chunks(per * gpr)
                    .zip(soft.chunks(per * cols))
                    .zip(probs.chunks(per * cols * 4))
                    .zip(grad_out.chunks(per * cols))
                    .zip(grad_logits.chunks_mut(per * cols * 4))
                    .zip(grad_scales.chunks_mut(per * gpr));
                for (((((sc, so), p), go), gl), gs) in chunks {
                    s.spawn(move || backward(sc, so, p, go, gl, gs));
                }
            });
        }
        let t = (step + 1) as i32;
        opt_logits.step(&mut logits, &grad_logits, cfg.learning_rate, t);
        opt_scales.step(&mut scales, &grad_scales, cfg.learning_rate * 0.1 * scale_unit, t);
    }
    let index: Vec<u8> = logits
        .chunks(4)
        .map(|l| (0..4u8).max_by(|a, b| l[usize::from(*a)].total_cmp(&l[usize::from(*b)])).unwrap_or(2))
        .collect();
    (index, scales)
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

    /// Rows trained a few at a time give the matrix trained whole, and the
    /// error summed a chunk at a time is the whole one's.
    #[test]
    fn training_rows_in_chunks_gives_the_same_result() {
        let (rows, cols) = (7usize, 128usize);
        let w = noise(rows * cols, 21);
        let h = correlated_gram(cols, 4);
        let gram = Gram::Full(&h);
        let cfg = GsqConfig { steps: 40, ..GsqConfig::default() };
        let whole = quantise_2bit_by(&w, rows, cols, &gram, &cfg, rows).unwrap();
        for per in [1, 3, 5] {
            assert_eq!(quantise_2bit_by(&w, rows, cols, &gram, &cfg, per).unwrap(), whole, "{per} rows at a time");
        }
        let full: f64 = {
            let q = whole.weights();
            let d: Vec<f32> = w.iter().zip(&q).map(|(a, b)| a - b).collect();
            let mut hd = vec![0.0f32; d.len()];
            rows_times_gram(&d, cols, &gram, &mut hd);
            d.iter().zip(&hd).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum()
        };
        assert_eq!(quantised_error(&w, &whole, &gram), full);
        assert_eq!(output_error(&w, &whole.weights(), cols, &gram), full);
        assert_eq!(output_energy(&w, cols, &Gram::Identity), output_error(&w, &vec![0.0; w.len()], cols, &Gram::Identity));
    }

    /// Real matrices draw tens of millions of samples a step: none may be
    /// infinite (a uniform draw of 1.0 once made every training NaN).
    #[test]
    fn a_stretched_start_clips_less_and_is_kept() {
        let (rows, cols) = (8, 256);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919 % 997) as f32 / 997.0 - 0.5) * if i % 61 == 0 { 6.0 } else { 1.0 }).collect();
        let at = |stretch: f32| {
            quantise_2bit(&w, rows, cols, &Gram::Identity, &GsqConfig { steps: 0, stretch, ..GsqConfig::default() }).unwrap()
        };
        let (plain, wide) = (at(1.0), at(1.15));
        for (p, s) in plain.scales.iter().zip(&wide.scales) {
            assert!((s - p * 1.15).abs() <= 1e-6 * p.abs().max(1.0), "{p} -> {s}");
        }
        // More weight error (rounding's scale minimises it), and the largest weights kept closer.
        let err = |q: &Quantised| w.iter().zip(q.weights()).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
        assert!(err(&wide) > err(&plain));
        let outliers = |q: &Quantised| w.iter().zip(q.weights()).step_by(61).map(|(a, b)| (a - b).powi(2)).sum::<f32>();
        assert!(outliers(&wide) < outliers(&plain), "{} vs {}", outliers(&wide), outliers(&plain));
    }

    #[test]
    fn gumbel_noise_stays_finite_over_many_draws() {
        let mut rng = Rng::for_row(0, 0, 0);
        for _ in 0..1 << 25 {
            assert!(rng.gumbel().is_finite());
        }
    }

    #[test]
    fn shapes_that_are_not_whole_blocks_are_refused() {
        assert!(quantise_2bit(&[0.0; 60], 1, 60, &Gram::Identity, &GsqConfig::default()).is_err());
        assert!(quantise_2bit(&[0.0; 64], 1, 64, &Gram::Full(&[0.0; 9]), &GsqConfig::default()).is_err());
    }
}

#[cfg(test)]
mod timing {
    use super::*;

    /// Opt-in: where the search's time goes on a large matrix
    /// (`ALLPACA_GSQ_TIMING=1`).
    #[test]
    fn gsq_phase_timing_when_env_set() {
        if std::env::var("ALLPACA_GSQ_TIMING").is_err() {
            return;
        }
        let (rows, cols) = (16384usize, 2048usize);
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
            })
            .collect();
        let t = std::time::Instant::now();
        let start = round_with_best_scales(&w);
        eprintln!("round_with_best_scales {:.2} s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let mut adam = Adam::new(w.len() * 4);
        let mut p = vec![0.0f32; w.len() * 4];
        let g = vec![0.01f32; w.len() * 4];
        adam.step(&mut p, &g, 0.05, 1);
        eprintln!("one Adam step {:.2} s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let q = quantise_2bit(&w, rows, cols, &Gram::Identity, &GsqConfig { steps: 8, ..GsqConfig::default() }).unwrap();
        eprintln!("quantise_2bit (8 steps) {:.2} s for {} weights", t.elapsed().as_secs_f32(), w.len());
        let t = std::time::Instant::now();
        let _ = quantised_error(&w, &q, &Gram::Identity);
        eprintln!("quantised_error {:.2} s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let bytes = allpaca_cpu::encode_q4_k(&w);
        let mut d = vec![0.0f32; w.len()];
        eprintln!("encode_q4_k {:.2} s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        allpaca_gguf::dequant_packed(allpaca_gguf::GgmlType::Q4_K, &bytes, w.len(), &mut d).unwrap();
        eprintln!("dequant q4_k {:.2} s", t.elapsed().as_secs_f32());
        drop(start);
    }
}
