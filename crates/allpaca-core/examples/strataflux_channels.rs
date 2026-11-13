//! Board Q48: does a small high-precision channel residual pay on our quants?
//!
//! StrataFlux (arXiv 2611.08363) keeps a low-bit matrix on the fast device and
//! adds a higher-precision residual for a few output channels. This simulates
//! the best case on the CPU: a Q4_K matrix whose top channels (output rows,
//! ranked by the largest magnitude in the row) are replaced by their Q8_0
//! values, measured as the relative L2 distance of the GEMV result from the
//! all-Q8_0 result. That ranking is the board task's criterion.
//!
//! The paper itself refines input channels (columns), chosen per input by
//! `|x_i|` (its eq. 1-2, after DecDEC): `y = W4 x + sum over S(x) of
//! x_i (W8 - W4)[:, i]`. That selection is printed too, as extra evidence;
//! it is not the task's go/no-go.
//!
//! `cargo run --release -p allpaca-core --example strataflux_channels`

use allpaca_cpu::{dequant_q4_k, dequant_q8_0, matmul_nt, quantize_q4_k, quantize_q8_0};

const N_OUT: usize = 4096;
const N_IN: usize = 4096;
const N_VEC: usize = 32;
const SEED: u64 = 0x5eed_2611_0836_3001;

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        // xorshift64*; top 24 bits to [0, 1).
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

fn gemv(x: &[f32], w: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; N_OUT];
    matmul_nt(x, w, &mut y, 1, N_OUT, N_IN).expect("gemv");
    y
}

fn l2(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum::<f64>().sqrt()
}

/// One matrix: the Q4-only error and the error with `fractions` of the
/// channels taken from Q8_0, each as the mean over the vectors of
/// `L2(y - y_q8) / L2(y_q8)`.
fn run(label: &str, w: &[f32], xs: &[Vec<f32>], fractions: &[f64]) {
    let n = N_OUT * N_IN;
    let (mut w4, mut w8) = (vec![0.0f32; n], vec![0.0f32; n]);
    dequant_q4_k(&quantize_q4_k(w), n, &mut w4).expect("q4_k");
    dequant_q8_0(&quantize_q8_0(w), n, &mut w8).expect("q8_0");
    // Rank output channels by the largest magnitude in the Q8 row.
    let mut order: Vec<usize> = (0..N_OUT).collect();
    let peak = |r: usize| w8[r * N_IN..(r + 1) * N_IN].iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let peaks: Vec<f32> = (0..N_OUT).map(peak).collect();
    order.sort_by(|a, b| peaks[*b].total_cmp(&peaks[*a]));
    let y8: Vec<Vec<f32>> = xs.iter().map(|x| gemv(x, &w8)).collect();
    let y4: Vec<Vec<f32>> = xs.iter().map(|x| gemv(x, &w4)).collect();
    let zero = vec![0.0f32; N_OUT];
    let mean_err = |keep: usize| -> f64 {
        let kept = &order[..keep];
        y4.iter()
            .zip(&y8)
            .map(|(a, b)| {
                // A kept channel is the Q8 row itself, so its output is y_q8's.
                let mut mix = a.clone();
                for &r in kept {
                    mix[r] = b[r];
                }
                l2(&mix, b) / l2(b, &zero)
            })
            .sum::<f64>()
            / y8.len() as f64
    };
    let q4 = mean_err(0);
    println!("{label}: Q4_K only, relative L2 vs Q8_0 = {q4:.6}");
    for f in fractions {
        let keep = (N_OUT as f64 * f).round() as usize;
        let e = mean_err(keep);
        println!(
            "{label}: {:>4.0}% of channels at Q8_0 ({keep} rows): {e:.6} = {:.3} x Q4-only, removes {:.1}%",
            f * 100.0,
            e / q4,
            (1.0 - e / q4) * 100.0
        );
    }
    let five = mean_err((N_OUT as f64 * 0.05).round() as usize);
    println!(
        "{label}: verdict at 5%: {} (go needs <= 0.60 x Q4-only, got {:.3})",
        if five <= 0.60 * q4 { "GO" } else { "NO-GO" },
        five / q4
    );
    // Extra: the paper's selection, input columns with the largest |x_i|.
    let resid: Vec<f32> = w8.iter().zip(&w4).map(|(a, b)| a - b).collect();
    for f in fractions {
        let keep = (N_IN as f64 * f).round() as usize;
        let e = xs
            .iter()
            .zip(y4.iter().zip(&y8))
            .map(|(x, (a, b))| {
                let mut idx: Vec<usize> = (0..N_IN).collect();
                idx.sort_by(|i, j| x[*j].abs().total_cmp(&x[*i].abs()));
                let mut xs_sel = vec![0.0f32; N_IN];
                for &i in &idx[..keep] {
                    xs_sel[i] = x[i];
                }
                let add = gemv(&xs_sel, &resid);
                let mix: Vec<f32> = a.iter().zip(&add).map(|(p, q)| p + q).collect();
                l2(&mix, b) / l2(b, &zero)
            })
            .sum::<f64>()
            / xs.len() as f64;
        println!(
            "{label}: [paper's selection] {:>4.0}% of input columns by |x_i| ({keep}): {e:.6} = {:.3} x Q4-only, removes {:.1}%",
            f * 100.0,
            e / q4,
            (1.0 - e / q4) * 100.0
        );
    }
}

fn main() {
    let mut rng = Rng(SEED);
    println!("seed {SEED:#x}, {N_OUT} outputs x {N_IN} inputs, {N_VEC} input vectors");
    let xs: Vec<Vec<f32>> = (0..N_VEC).map(|_| (0..N_IN).map(|_| rng.normal()).collect()).collect();
    let fractions = [0.01, 0.05, 0.10];

    // The task's case: every weight from one normal distribution.
    let w: Vec<f32> = (0..N_OUT * N_IN).map(|_| rng.normal() * 0.02).collect();
    run("iid normal", &w, &xs, &fractions);

    // Extra, not the task's criterion: rows with very different scales
    // (log-normal, sigma 1), where a few channels carry most of the output.
    let mut w = w;
    for row in w.chunks_mut(N_IN) {
        let s = rng.normal().exp();
        for v in row {
            *v *= s;
        }
    }
    run("row scales log-normal", &w, &xs, &fractions);

    // Extra: heavy-tailed activations (each input channel scaled by a
    // log-normal factor, sigma 1) on the iid matrix: the case the paper's
    // |x_i| selection is built for.
    let w: Vec<f32> = {
        let mut r = Rng(SEED ^ 1);
        (0..N_OUT * N_IN).map(|_| r.normal() * 0.02).collect()
    };
    let scales: Vec<f32> = (0..N_IN).map(|_| rng.normal().exp()).collect();
    let xs: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().zip(&scales).map(|(v, s)| v * s).collect()).collect();
    run("iid normal, heavy-tailed inputs", &w, &xs, &fractions);
}
