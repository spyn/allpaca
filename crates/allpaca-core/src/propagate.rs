//! Propagated calibration: quantise each block for the original model's
//! outputs on the inputs the quantised model really has.
//!
//! Quantised alone, a matrix is fitted to inputs that only the original
//! model sees. In the quantised model every earlier matrix has already
//! moved those inputs, and the errors add up through the blocks. Here the
//! blocks are done in order, and within a block the matrices in the order
//! the data flows (queries, keys and values; the attention output; the
//! feed-forward's gate and up; its down). For a matrix `W` with original
//! input `x` and current input `x~`, the weights that give the original
//! output on the current input, without leaving `W` further than the
//! calibration text asks for, are
//!
//! ```text
//!   W* = argmin  E|W x - Q x~|^2 + ridge |Q - W|^2
//!      = W + (W E^T) (X~ X~^T + ridge T I)^-1 X~,      E = X - X~
//! ```
//!
//! with the `T` calibration positions as the rows of `X` and `X~`. `W*` is
//! then quantised like any matrix (for its output on `x~`), put in the
//! model, and the next matrix sees what it does.
//!
//! No gradient is taken through the block: this is the error-propagation
//! half of block-output training.

use std::collections::HashMap;
use std::path::Path;

use allpaca_cpu::matmul_nt;
use allpaca_types::{AllpacaError, FatalError};

use crate::calib::Site;
use crate::gsq::{Gram, GsqConfig};
use crate::requant::Target;
use crate::sensitivity::Probe;

fn bad(msg: impl Into<String>) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg.into()))
}

/// The calibration text for [`crate::requant::Options::propagate`].
#[derive(Debug, Clone)]
pub struct Settings {
    pub texts: Vec<String>,
    /// Tokens read of each text.
    pub max_tokens: usize,
    /// How strongly a matrix is held to its original weights, as a share
    /// of the mean squared length of its inputs. Small values fit the
    /// calibration text exactly and nothing else.
    pub ridge: f32,
}

/// `out = a @ b^T` (`a` is `[m, k]`, `b` is `[n, k]`), rows of `a` spread
/// over the cores.
fn product_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut out = vec![0.0f32; m * n];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(m).max(1);
    if threads == 1 || m * n * k < (1 << 22) {
        matmul_nt(a, b, &mut out, m, n, k)?;
        return Ok(out);
    }
    let per = m.div_ceil(threads);
    let mut failed = None;
    std::thread::scope(|scope| {
        let jobs: Vec<_> = a
            .chunks(per * k)
            .zip(out.chunks_mut(per * n))
            .map(|(a, out)| scope.spawn(move || matmul_nt(a, b, out, a.len() / k, n, k)))
            .collect();
        for job in jobs {
            if let Err(e) = job.join().expect("product thread") {
                failed = Some(e);
            }
        }
    });
    failed.map_or(Ok(out), Err)
}

fn transposed(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len()];
    for (r, row) in a.chunks(cols).enumerate() {
        for (c, v) in row.iter().enumerate() {
            out[c * rows + r] = *v;
        }
    }
    out
}

/// One input site of one block: what the original model fed it and what
/// the quantised model feeds it now, ready to refit the matrices that
/// read it.
pub struct Refit {
    cols: usize,
    positions: usize,
    /// `X - X~`, `[positions, cols]`.
    gap: Vec<f32>,
    /// `X~^T`, `[cols, positions]`.
    now_t: Vec<f32>,
    /// Lower Cholesky factor of `X~ X~^T + ridge * positions * I`.
    factor: Vec<f64>,
}

impl Refit {
    /// `original` and `now` hold one input of `cols` floats per position.
    pub fn new(original: &[f32], now: &[f32], cols: usize, ridge: f32) -> Result<Self, AllpacaError> {
        if cols == 0 || original.len() != now.len() || now.len() % cols != 0 || now.is_empty() {
            return Err(bad(format!("refit: {} and {} input floats for width {cols}", original.len(), now.len())));
        }
        let t = now.len() / cols;
        let gap: Vec<f32> = original.iter().zip(now).map(|(a, b)| a - b).collect();
        let kernel = product_nt(now, now, t, t, cols)?;
        // The ridge in the units of the kernel: a share of its mean diagonal.
        let mean: f64 = (0..t).map(|i| f64::from(kernel[i * t + i])).sum::<f64>() / t as f64;
        let add = f64::from(ridge.max(1e-6)) * mean.max(1e-30);
        let mut factor = vec![0.0f64; t * t];
        for i in 0..t {
            for j in 0..=i {
                let mut sum = f64::from(kernel[i * t + j]) + if i == j { add } else { 0.0 };
                for p in 0..j {
                    sum -= factor[i * t + p] * factor[j * t + p];
                }
                factor[i * t + j] = if i == j {
                    if sum <= 0.0 {
                        return Err(bad("refit: the inputs' kernel is not positive definite"));
                    }
                    sum.sqrt()
                } else {
                    sum / factor[j * t + j]
                };
            }
        }
        Ok(Self {
            cols,
            positions: t,
            gap,
            now_t: transposed(now, t, cols),
            factor,
        })
    }

    /// The weights that give `w`'s output on the original inputs when fed
    /// the current ones (`w` is `[rows, cols]`).
    pub fn target(&self, w: &[f32], rows: usize) -> Result<Vec<f32>, AllpacaError> {
        let (t, cols) = (self.positions, self.cols);
        if w.len() != rows * cols {
            return Err(bad(format!("refit: {} weights for {rows} rows of {cols}", w.len())));
        }
        // What each row's output is short of, per position: `[rows, positions]`.
        let mut short = product_nt(w, &self.gap, rows, t, cols)?;
        if short.iter().all(|v| *v == 0.0) {
            return Ok(w.to_vec());
        }
        // Solve the kernel system for every row.
        let solve = |rows: &mut [f32]| {
            let mut y = vec![0.0f64; t];
            for row in rows.chunks_mut(t) {
                for i in 0..t {
                    let mut sum = f64::from(row[i]);
                    for (l, y) in self.factor[i * t..i * t + i].iter().zip(&y) {
                        sum -= l * y;
                    }
                    y[i] = sum / self.factor[i * t + i];
                }
                for i in (0..t).rev() {
                    let mut sum = y[i];
                    for j in i + 1..t {
                        sum -= self.factor[j * t + i] * y[j];
                    }
                    y[i] = sum / self.factor[i * t + i];
                }
                for (out, y) in row.iter_mut().zip(&y) {
                    *out = *y as f32;
                }
            }
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(rows).max(1);
        let per = rows.div_ceil(threads) * t;
        std::thread::scope(|scope| {
            for part in short.chunks_mut(per) {
                scope.spawn(|| solve(part));
            }
        });
        let mut out = product_nt(&short, &self.now_t, rows, cols, t)?;
        for (o, w) in out.iter_mut().zip(w) {
            *o += w;
        }
        Ok(out)
    }

    /// `E[x~ x~^T]` with `damping` times its mean diagonal on the diagonal.
    fn gram(&self, damping: f32) -> Result<Vec<f32>, AllpacaError> {
        let (t, cols) = (self.positions, self.cols);
        let mut h = product_nt(&self.now_t, &self.now_t, cols, cols, t)?;
        let scale = 1.0 / t as f32;
        h.iter_mut().for_each(|v| *v *= scale);
        let add = damping * (0..cols).map(|i| h[i * cols + i]).sum::<f32>() / cols as f32;
        for i in 0..cols {
            h[i * cols + i] += add;
        }
        Ok(h)
    }
}

/// The matrices of a block by the input they read, in the order the data
/// flows through the block: attention or the delta-net, then the dense
/// feed-forward or the shared expert. Routed experts are not here (their
/// stack's recorded inputs are not split by expert), nor is the router (the
/// probe cannot swap it): both are quantised alone.
const STAGES: [(Site, &[&str]); 6] = [
    (Site::AttnIn, &["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"]),
    (Site::AttnOut, &["attn_output", "ssm_out"]),
    (Site::FfnIn, &["ffn_gate", "ffn_up"]),
    (Site::FfnAct, &["ffn_down"]),
    (Site::ShexpIn, &["ffn_gate_shexp", "ffn_up_shexp"]),
    (Site::ShexpAct, &["ffn_down_shexp"]),
];

/// Quantise the block matrices of the model at `input` in order, each for
/// the original outputs on the inputs the matrices before it leave it.
/// `chosen` gives a tensor's type and shape (rows, row length), or `None`
/// for one that keeps its weights. Returns the quantised bytes by tensor
/// name. The token embedding is quantised first when `chosen` names it
/// (its bytes are not returned: it has no input to refit to).
pub(crate) fn quantise_in_order(
    input: &Path,
    settings: &Settings,
    chosen: &dyn Fn(&str) -> Option<(Target, usize, usize)>,
    gsq: &GsqConfig,
    damping: f32,
    progress: Option<fn(usize, usize, &str)>,
) -> Result<HashMap<String, Vec<u8>>, AllpacaError> {
    let texts: Vec<&str> = settings.texts.iter().map(String::as_str).collect();
    let mut probe = Probe::open(input, &texts, settings.max_tokens)?;
    let blocks = probe.blocks();
    let decode = |target: Target, bytes: &[u8], n: usize| -> Result<Vec<f32>, AllpacaError> {
        let mut out = vec![0.0f32; n];
        allpaca_gguf::dequant_packed(target.ggml(), bytes, n, &mut out)?;
        Ok(out)
    };
    // What the first block gets: the embedding as the file will hold it.
    let mut now = probe.original_inputs(0);
    if let Some((target, rows, cols)) = chosen("token_embd.weight") {
        let w = probe.weights("token_embd.weight").ok_or_else(|| bad("refit: no token embedding"))?.to_vec();
        let bytes = target.quantise(&w, rows, cols, &Gram::Identity, gsq)?;
        probe.replace("token_embd.weight", decode(target, &bytes, w.len())?);
        now = probe.first_inputs()?;
    }
    let mut out = HashMap::new();
    for layer in 0..blocks {
        let blk = probe.gguf_block(layer);
        let original = probe.block_sites(layer, &probe.original_inputs(layer))?.1;
        for (site, kinds) in STAGES {
            let todo: Vec<(String, Target, usize, usize)> = kinds
                .iter()
                .map(|kind| format!("blk.{blk}.{kind}.weight"))
                .filter_map(|name| chosen(&name).map(|(target, rows, cols)| (name, target, rows, cols)))
                .collect();
            if todo.is_empty() {
                continue;
            }
            let current = probe.block_sites(layer, &now)?.1;
            let (x, x_now) = match (original.get(&site), current.get(&site)) {
                (Some(x), Some(x_now)) => (x, x_now),
                _ => return Err(bad(format!("refit: block {blk} has no input recorded for {site:?}"))),
            };
            let mut refit = None;
            for (name, target, rows, cols) in todo {
                let w = probe.weights(&name).ok_or_else(|| bad(format!("refit: no matrix {name}")))?.to_vec();
                if w.len() != rows * cols || x.len() % cols != 0 || x.len() != x_now.len() {
                    return Err(bad(format!("refit: {name} is not {rows} rows of {cols} on its recorded input")));
                }
                if refit.is_none() {
                    refit = Some(Refit::new(x, x_now, cols, settings.ridge)?);
                }
                let fit = refit.as_ref().expect("built above");
                let wanted = fit.target(&w, rows)?;
                // Only the 2-bit search uses the inputs' second moments.
                let h = if target == Target::Q2_0 { Some(fit.gram(damping)?) } else { None };
                let gram = h.as_deref().map_or(Gram::Identity, Gram::Full);
                let bytes = target.quantise(&wanted, rows, cols, &gram, gsq)?;
                probe.replace(&name, decode(target, &bytes, w.len())?);
                out.insert(name.clone(), bytes);
                if let Some(progress) = progress {
                    progress(layer + 1, blocks, &name);
                }
            }
        }
        now = probe.block_sites(layer, &now)?.0;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    }

    /// `sum |x w^T - x_now q^T|^2` over positions.
    fn output_gap(w: &[f32], q: &[f32], x: &[f32], x_now: &[f32], rows: usize, cols: usize) -> f64 {
        let t = x.len() / cols;
        let (a, b) = (product_nt(x, w, t, rows, cols).unwrap(), product_nt(x_now, q, t, rows, cols).unwrap());
        a.iter().zip(&b).map(|(a, b)| f64::from(a - b).powi(2)).sum()
    }

    #[test]
    fn the_refitted_weights_give_the_original_outputs_on_the_moved_inputs() {
        let (rows, cols, t) = (5, 16, 9);
        let w = noise(rows * cols, 1, 1.0);
        let x = noise(t * cols, 2, 1.0);
        let moved: Vec<f32> = x.iter().zip(noise(t * cols, 3, 0.2)).map(|(a, b)| a + b).collect();
        let before = output_gap(&w, &w, &x, &moved, rows, cols);
        // A light hold: with fewer positions than inputs the fit is nearly exact.
        let fit = Refit::new(&x, &moved, cols, 1e-4).unwrap();
        let target = fit.target(&w, rows).unwrap();
        let after = output_gap(&w, &target, &x, &moved, rows, cols);
        assert!(after < 1e-4 * before, "{after} of {before}");
        // A heavy hold keeps the weights close and closes less of the gap.
        let held = Refit::new(&x, &moved, cols, 10.0).unwrap().target(&w, rows).unwrap();
        let dist = |q: &[f32]| w.iter().zip(q).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>();
        assert!(dist(&held) < 0.05 * dist(&target), "{} vs {}", dist(&held), dist(&target));
        let held_gap = output_gap(&w, &held, &x, &moved, rows, cols);
        assert!(after < held_gap && held_gap < before, "{after} < {held_gap} < {before}");
    }

    #[test]
    fn unmoved_inputs_leave_the_weights_as_they_are() {
        let (rows, cols, t) = (3, 8, 5);
        let w = noise(rows * cols, 4, 1.0);
        let x = noise(t * cols, 5, 1.0);
        assert_eq!(Refit::new(&x, &x, cols, 0.1).unwrap().target(&w, rows).unwrap(), w);
        assert!(Refit::new(&x, &x[..cols], cols, 0.1).is_err());
        assert!(Refit::new(&x, &x, cols, 0.1).unwrap().target(&w[..7], rows).is_err());
    }

    /// The refit moves the weights only along the inputs it saw: what the
    /// matrix does to an input at right angles to all of them is unchanged.
    #[test]
    fn directions_the_text_never_showed_keep_their_weights() {
        let (rows, cols) = (2, 4);
        let w = noise(rows * cols, 6, 1.0);
        // Two positions, both in the plane of the first two coordinates.
        let x = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let moved = [1.2, 0.1, 0.0, 0.0, -0.1, 0.9, 0.0, 0.0];
        let target = Refit::new(&x, &moved, cols, 1e-4).unwrap().target(&w, rows).unwrap();
        for r in 0..rows {
            assert_eq!(target[r * cols + 2..r * cols + 4], w[r * cols + 2..r * cols + 4], "row {r}");
            assert!((target[r * cols] - w[r * cols]).abs() > 1e-3, "row {r} moved in the plane");
        }
    }
}
