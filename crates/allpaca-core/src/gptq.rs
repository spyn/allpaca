//! GPTQ (arXiv 2210.17323) onto ggml `Q2_0`: one pass over the columns of
//! each row, rounding a weight at a time and moving its error onto the
//! columns still to come, weighted by the inverse of the inputs' Gram.
//! A group's scale is chosen, as in [`crate::gsq`]'s start, from the
//! group's weights as they stand when the pass reaches it.
//!
//! Rows are independent: tiles of rows run on all cores, and within a tile
//! the error moves within a group first and onto the later columns a group
//! at a time (the reference's lazy batch updates).

use allpaca_cpu::Q2_0_BLOCK_ELEMS;
use allpaca_types::{AllpacaError, FatalError};

use crate::gsq::{nearest, round_groups, Quantised, VALUES};

/// Rows a thread carries through the pass together.
const TILE_ROWS: usize = 64;

fn bad(msg: &str) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(format!("GPTQ: {msg}")))
}

/// Lower Cholesky factor of `a` (`n * n`, symmetric positive definite) in
/// place; the upper triangle is left as it was. False if not positive.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let row_j = &head[j * n..j * n + j];
        for row_i in tail.chunks_mut(n) {
            let s = row_i[..j].iter().zip(row_j).map(|(a, b)| a * b).sum::<f64>();
            row_i[j] = (row_i[j] - s) / d;
        }
    }
    true
}

/// The upper triangular `U` with `H^-1 = U^T U` (row-major `n * n`, zero
/// below the diagonal), the factor GPTQ's pass runs on.
pub(crate) fn inverse_factor(h: &[f32], n: usize) -> Result<Vec<f32>, AllpacaError> {
    let mut a: Vec<f64> = h.iter().map(|v| f64::from(*v)).collect();
    // A column no input ever reached carries no error: give it a unit.
    for i in 0..n {
        if a[i * n + i] <= 0.0 {
            a[i * n + i] = 1.0;
        }
    }
    if !cholesky(&mut a, n) {
        return Err(bad("Gram matrix is not positive definite"));
    }
    // L^-1, lower, a row at a time: row i is (e_i - sum over k < i of
    // L[i][k] row k) / L[i][i].
    let mut inv = vec![0.0f64; n * n];
    let mut acc = vec![0.0f64; n];
    for i in 0..n {
        acc[..i].fill(0.0);
        for k in 0..i {
            let l = a[i * n + k];
            for (s, v) in acc[..=k].iter_mut().zip(&inv[k * n..k * n + k + 1]) {
                *s += l * v;
            }
        }
        let d = a[i * n + i];
        for c in 0..i {
            inv[i * n + c] = -acc[c] / d;
        }
        inv[i * n + i] = 1.0 / d;
    }
    // H^-1 = L^-T L^-1: entry (i, j), j >= i, is the dot product of columns
    // i and j of L^-1 from row j down: rows of its transpose.
    let mut t = vec![0.0f64; n * n];
    for i in 0..n {
        for k in i..n {
            t[i * n + k] = inv[k * n + i];
        }
    }
    let mut hinv = vec![0.0f64; n * n];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(n.max(1));
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        for (c, rows) in hinv.chunks_mut(per * n).enumerate() {
            let t = &t;
            s.spawn(move || {
                for (r, row) in rows.chunks_mut(n).enumerate() {
                    let i = c * per + r;
                    for j in i..n {
                        row[j] = t[i * n + j..(i + 1) * n].iter().zip(&t[j * n + j..(j + 1) * n]).map(|(a, b)| a * b).sum();
                    }
                }
            });
        }
    });
    for i in 0..n {
        for j in 0..i {
            hinv[i * n + j] = hinv[j * n + i];
        }
    }
    if !cholesky(&mut hinv, n) {
        return Err(bad("inverse Gram matrix is not positive definite"));
    }
    let mut u = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            u[j * n + i] = hinv[i * n + j] as f32;
        }
    }
    Ok(u)
}

/// Quantise `w` (`rows * cols`, row-major; `cols` a multiple of 64) to the
/// 2-bit grid for the least output error on inputs with Gram `h`
/// (`cols * cols`, damped).
pub fn gptq_2bit(w: &[f32], rows: usize, cols: usize, h: &[f32]) -> Result<Quantised, AllpacaError> {
    if w.len() != rows * cols || cols == 0 || cols % Q2_0_BLOCK_ELEMS != 0 {
        return Err(bad("weights must be rows * cols with cols a multiple of 64"));
    }
    if h.len() != cols * cols {
        return Err(bad("Gram matrix must be cols * cols"));
    }
    let u = inverse_factor(h, cols)?;
    let (n, g) = (cols, Q2_0_BLOCK_ELEMS);
    let gpr = cols / g;
    // Per group, the factor's rows of that group for every later column,
    // a row of 64 per column: the error of a group reaches column c as a
    // dot product with row c.
    let later: Vec<Vec<f32>> = (0..gpr)
        .map(|b| {
            let first = (b + 1) * g;
            let u = &u;
            (first..n).flat_map(|c| (b * g..first).map(move |j| u[j * n + c])).collect()
        })
        .collect();
    let mut index = vec![0u8; w.len()];
    let mut scales = vec![0.0f32; rows * gpr];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = rows.div_ceil(threads).div_ceil(TILE_ROWS).max(1) * TILE_ROWS;
    std::thread::scope(|s| {
        let parts = w.chunks(per * cols).zip(index.chunks_mut(per * cols)).zip(scales.chunks_mut(per * gpr));
        for ((w, index), scales) in parts {
            let (u, later) = (&u, &later);
            s.spawn(move || {
                let tiles = w.chunks(TILE_ROWS * cols).zip(index.chunks_mut(TILE_ROWS * cols)).zip(scales.chunks_mut(TILE_ROWS * gpr));
                for ((w, index), scales) in tiles {
                    tile(w, index, scales, cols, u, later);
                }
            });
        }
    });
    Ok(Quantised { rows, cols, index, scales })
}

/// The pass over a tile of whole rows.
fn tile(w: &[f32], index: &mut [u8], scales: &mut [f32], cols: usize, u: &[f32], later: &[Vec<f32>]) {
    let (n, g) = (cols, Q2_0_BLOCK_ELEMS);
    let (rows, gpr) = (w.len() / n, n / g);
    let mut w = w.to_vec();
    let mut err = vec![0.0f32; rows * g];
    let mut scratch = [0u8; Q2_0_BLOCK_ELEMS];
    for b in 0..gpr {
        let first = b * g;
        for r in 0..rows {
            round_groups(&w[r * n + first..r * n + first + g], &mut scratch, &mut scales[r * gpr + b..r * gpr + b + 1]);
        }
        for j in first..first + g {
            let ujj = u[j * n + j];
            let urow = &u[j * n + j + 1..j * n + first + g];
            for r in 0..rows {
                let s = scales[r * gpr + b];
                let row = &mut w[r * n..(r + 1) * n];
                let x = row[j];
                let qi = if s == 0.0 { 2 } else { nearest(x / s) };
                index[r * n + j] = qi;
                let e = (x - VALUES[usize::from(qi)] * s) / ujj;
                err[r * g + j - first] = e;
                for (v, uk) in row[j + 1..first + g].iter_mut().zip(urow) {
                    *v -= e * uk;
                }
            }
        }
        let after = &later[b];
        for r in 0..rows {
            let e = &err[r * g..(r + 1) * g];
            for (v, uc) in w[r * n + first + g..(r + 1) * n].iter_mut().zip(after.chunks(g)) {
                *v -= allpaca_cpu::dot(e, uc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsq::{output_error, quantise_2bit, Gram, GsqConfig};

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
            })
            .collect()
    }

    /// Inputs with a few strong, correlated directions, damped like the
    /// requantiser's.
    fn correlated_gram(cols: usize, samples: usize, seed: u64) -> Vec<f32> {
        let basis = noise(4 * cols, seed);
        let coef = noise(samples * 4, seed + 1);
        let small = noise(samples * cols, seed + 2);
        let mut h = vec![0.0f32; cols * cols];
        for s in 0..samples {
            let x: Vec<f32> = (0..cols)
                .map(|i| (0..4).map(|k| coef[s * 4 + k] * basis[k * cols + i] * (3.0 + k as f32)).sum::<f32>() + 0.1 * small[s * cols + i])
                .collect();
            for i in 0..cols {
                for j in 0..cols {
                    h[i * cols + j] += x[i] * x[j] / samples as f32;
                }
            }
        }
        let add = 0.01 * (0..cols).map(|i| h[i * cols + i]).sum::<f32>() / cols as f32;
        for i in 0..cols {
            h[i * cols + i] += add;
        }
        h
    }

    #[test]
    fn inverse_factor_squares_to_the_inverse() {
        let n = 128;
        let h = correlated_gram(n, 400, 3);
        let u = inverse_factor(&h, n).unwrap();
        // U^T U H = I.
        let mut hinv = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                hinv[i * n + j] = (0..n).map(|k| f64::from(u[k * n + i]) * f64::from(u[k * n + j])).sum();
            }
        }
        for i in 0..n {
            assert_eq!(u[i * n..i * n + i].iter().filter(|v| **v != 0.0).count(), 0, "upper triangular");
            for j in 0..n {
                let v: f64 = (0..n).map(|k| hinv[i * n + k] * f64::from(h[k * n + j])).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-3, "({i}, {j}): {v}");
            }
        }
    }

    /// On correlated inputs GPTQ's output error is well under rounding's,
    /// and the result decodes to the grid with its own scales.
    #[test]
    fn gptq_beats_rounding_on_correlated_inputs() {
        let (rows, cols) = (150, 256);
        let w = noise(rows * cols, 7);
        let h = correlated_gram(cols, 600, 11);
        let full = Gram::Full(&h);
        let rounded = quantise_2bit(&w, rows, cols, &Gram::Identity, &GsqConfig { steps: 0, ..GsqConfig::default() }).unwrap();
        let q = gptq_2bit(&w, rows, cols, &h).unwrap();
        assert_eq!((q.rows, q.cols, q.index.len(), q.scales.len()), (rows, cols, rows * cols, rows * cols / 64));
        assert!(q.index.iter().all(|i| *i < 4));
        let (e_round, e_gptq) = (output_error(&w, &rounded.weights(), cols, &full), output_error(&w, &q.weights(), cols, &full));
        assert!(e_gptq < 0.7 * e_round, "rounding {e_round} vs GPTQ {e_gptq}");
        // Uncorrelated unit inputs leave nothing to move: plain rounding.
        let eye: Vec<f32> = (0..cols * cols).map(|i| if i % (cols + 1) == 0 { 1.0 } else { 0.0 }).collect();
        assert_eq!(gptq_2bit(&w, rows, cols, &eye).unwrap(), rounded);
    }

    #[test]
    fn tiles_and_threads_do_not_change_the_result() {
        let (rows, cols) = (TILE_ROWS * 3 + 5, 128);
        let w = noise(rows * cols, 21);
        let h = correlated_gram(cols, 300, 23);
        let all = gptq_2bit(&w, rows, cols, &h).unwrap();
        for r in [0, 1, TILE_ROWS, rows - 1] {
            let one = gptq_2bit(&w[r * cols..(r + 1) * cols], 1, cols, &h).unwrap();
            assert_eq!(one.index, all.index[r * cols..(r + 1) * cols], "row {r}");
            assert_eq!(one.scales, all.scales[r * 2..(r + 1) * 2], "row {r}");
        }
    }
}
