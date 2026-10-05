//! MoE experts computed on the CPU straight from packed MXFP4 (e8m0 scale +
//! 32 four-bit codes per 17-byte block), with no F32 copy of the weights: an
//! expert that is not in VRAM is read once from the mapped file, at memory
//! speed. Activations stay F32, so the result differs from a dequantise-then-
//! multiply reference only by the order of the sums.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

use crate::kernels::{e8m0_to_fp32_half, silu, KVALUES_MXFP4, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};

/// One routed expert and the rows it runs on.
pub struct Mxfp4Expert<'a> {
    /// `ff` rows of `dim` codes.
    pub gate: &'a [u8],
    pub up: &'a [u8],
    /// `dim` rows of `ff` codes.
    pub down: &'a [u8],
    /// `m` input rows of `dim`.
    pub x: &'a [f32],
    pub m: usize,
}

/// Dot product of one MXFP4 row of `nblk` blocks with `x`.
fn dot_scalar(row: &[u8], nblk: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..nblk {
        let blk = &row[b * MXFP4_BLOCK_BYTES..(b + 1) * MXFP4_BLOCK_BYTES];
        let xs = &x[b * MXFP4_BLOCK_ELEMS..(b + 1) * MXFP4_BLOCK_ELEMS];
        let mut s = 0.0f32;
        for j in 0..MXFP4_BLOCK_ELEMS / 2 {
            let q = blk[1 + j];
            s += f32::from(KVALUES_MXFP4[usize::from(q & 0x0f)]) * xs[j];
            s += f32::from(KVALUES_MXFP4[usize::from(q >> 4)]) * xs[j + MXFP4_BLOCK_ELEMS / 2];
        }
        acc += s * e8m0_to_fp32_half(blk[0]);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(row: *const u8, nblk: usize, x: *const f32, scales: &[f32; 256]) -> f32 {
    use std::arch::x86_64::*;
    let lut = _mm_loadu_si128(KVALUES_MXFP4.as_ptr().cast());
    let nib = _mm_set1_epi8(0x0f);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    for b in 0..nblk {
        let p = row.add(b * MXFP4_BLOCK_BYTES);
        let q = _mm_loadu_si128(p.add(1).cast());
        // Low nibbles are elements 0..16, high nibbles 16..32.
        let lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, nib));
        let hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), nib));
        let xb = x.add(b * MXFP4_BLOCK_ELEMS);
        let k0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        let k1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        let k2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        let k3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        let a = _mm256_fmadd_ps(k1, _mm256_loadu_ps(xb.add(8)), _mm256_mul_ps(k0, _mm256_loadu_ps(xb)));
        let c = _mm256_fmadd_ps(
            k3,
            _mm256_loadu_ps(xb.add(24)),
            _mm256_mul_ps(k2, _mm256_loadu_ps(xb.add(16))),
        );
        let d = _mm256_set1_ps(scales[usize::from(*p)]);
        if b & 1 == 0 {
            acc0 = _mm256_fmadd_ps(_mm256_add_ps(a, c), d, acc0);
        } else {
            acc1 = _mm256_fmadd_ps(_mm256_add_ps(a, c), d, acc1);
        }
    }
    let acc = _mm256_add_ps(acc0, acc1);
    let lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    _mm_cvtss_f32(_mm_add_ss(s, _mm_shuffle_ps(s, s, 1)))
}

fn scale_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (e, v) in t.iter_mut().enumerate() {
            *v = e8m0_to_fp32_half(e as u8);
        }
        t
    })
}

#[derive(Clone, Copy)]
struct Dot {
    avx2: bool,
    scales: &'static [f32; 256],
}

impl Dot {
    fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Self {
            avx2,
            scales: scale_table(),
        }
    }

    #[inline]
    fn run(self, row: &[u8], nblk: usize, x: &[f32]) -> f32 {
        debug_assert!(row.len() >= nblk * MXFP4_BLOCK_BYTES && x.len() >= nblk * MXFP4_BLOCK_ELEMS);
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: lengths checked above; the features were detected.
            return unsafe { dot_avx2(row.as_ptr(), nblk, x.as_ptr(), self.scales) };
        }
        dot_scalar(row, nblk, x)
    }
}

/// A `*mut f32` the workers write disjoint ranges of.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// Rows per unit of work handed to a thread.
const CHUNK: usize = 32;

/// SwiGLU of every expert on its rows: `down · (silu(gate · x) * up · x)`.
/// Returns one `m * dim` vector per expert. `dim` and `ff` must be multiples
/// of 32. Rows are shared out over `threads` threads; `threads <= 1` runs on
/// the caller.
pub fn mxfp4_swiglu_experts(experts: &[Mxfp4Expert<'_>], dim: usize, ff: usize, threads: usize) -> Vec<Vec<f32>> {
    let (kb, fb) = (dim / MXFP4_BLOCK_ELEMS, ff / MXFP4_BLOCK_ELEMS);
    let row_in = kb * MXFP4_BLOCK_BYTES;
    let row_down = fb * MXFP4_BLOCK_BYTES;
    assert!(dim % MXFP4_BLOCK_ELEMS == 0 && ff % MXFP4_BLOCK_ELEMS == 0, "MXFP4 expert shape");
    for e in experts {
        assert!(
            e.gate.len() >= ff * row_in
                && e.up.len() >= ff * row_in
                && e.down.len() >= dim * row_down
                && e.x.len() >= e.m * dim,
            "MXFP4 expert slice too short"
        );
    }
    let mut hidden: Vec<Vec<f32>> = experts.iter().map(|e| vec![0.0f32; e.m * ff]).collect();
    let mut out: Vec<Vec<f32>> = experts.iter().map(|e| vec![0.0f32; e.m * dim]).collect();
    if experts.is_empty() {
        return out;
    }
    let hid: Vec<Out> = hidden.iter_mut().map(|v| Out(v.as_mut_ptr())).collect();
    let dst: Vec<Out> = out.iter_mut().map(|v| Out(v.as_mut_ptr())).collect();
    let dot = Dot::new();
    let n = experts.len();
    let (next_up, next_down) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let threads = threads.clamp(1, (n * ff).div_ceil(CHUNK));
    let barrier = Barrier::new(threads);
    let work = || {
        // gate and up rows -> hidden
        loop {
            let at = next_up.fetch_add(CHUNK, Ordering::Relaxed);
            if at >= n * ff {
                break;
            }
            for r in at..(at + CHUNK).min(n * ff) {
                let (e, row) = (r / ff, r % ff);
                let ex = &experts[e];
                let g = &ex.gate[row * row_in..(row + 1) * row_in];
                let u = &ex.up[row * row_in..(row + 1) * row_in];
                for i in 0..ex.m {
                    let x = &ex.x[i * dim..(i + 1) * dim];
                    let h = silu(dot.run(g, kb, x)) * dot.run(u, kb, x);
                    // SAFETY: each (expert, row, i) is written by one thread.
                    unsafe { *hid[e].0.add(i * ff + row) = h };
                }
            }
        }
        barrier.wait();
        // down rows -> out
        loop {
            let at = next_down.fetch_add(CHUNK, Ordering::Relaxed);
            if at >= n * dim {
                break;
            }
            for r in at..(at + CHUNK).min(n * dim) {
                let (e, row) = (r / dim, r % dim);
                let ex = &experts[e];
                let d = &ex.down[row * row_down..(row + 1) * row_down];
                for i in 0..ex.m {
                    // SAFETY: hidden is complete (barrier) and read-only here.
                    let h = unsafe { std::slice::from_raw_parts(hid[e].0.add(i * ff).cast_const(), ff) };
                    unsafe { *dst[e].0.add(i * dim + row) = dot.run(d, fb, h) };
                }
            }
        }
    };
    if threads == 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 1..threads {
                s.spawn(work);
            }
            work();
        });
    }
    drop(hidden);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{dequant_mxfp4, quantize_mxfp4};

    fn rand(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
            })
            .collect()
    }

    #[test]
    fn packed_expert_matches_dequantised_reference() {
        let (dim, ff) = (96usize, 64usize);
        let mut packed = Vec::new();
        let mut reference = Vec::new();
        for (e, m) in [(0u64, 1usize), (1, 3)] {
            let gate = quantize_mxfp4(&rand(ff * dim, 11 + e, 0.4));
            let up = quantize_mxfp4(&rand(ff * dim, 23 + e, 0.4));
            let down = quantize_mxfp4(&rand(dim * ff, 37 + e, 0.4));
            let x = rand(m * dim, 51 + e, 2.0);
            // Reference: dequantise, then plain F32 SwiGLU.
            let deq = |b: &[u8], n: usize| {
                let mut o = vec![0.0f32; n];
                dequant_mxfp4(b, n, &mut o).unwrap();
                o
            };
            let (g, u, d) = (deq(&gate, ff * dim), deq(&up, ff * dim), deq(&down, dim * ff));
            let mut y = vec![0.0f32; m * dim];
            for i in 0..m {
                let xi = &x[i * dim..(i + 1) * dim];
                let h: Vec<f32> = (0..ff)
                    .map(|r| {
                        let dot = |w: &[f32]| w[r * dim..(r + 1) * dim].iter().zip(xi).map(|(a, b)| a * b).sum::<f32>();
                        silu(dot(&g)) * dot(&u)
                    })
                    .collect();
                for j in 0..dim {
                    y[i * dim + j] = d[j * ff..(j + 1) * ff].iter().zip(&h).map(|(a, b)| a * b).sum();
                }
            }
            reference.push(y);
            packed.push((gate, up, down, x, m));
        }
        let experts: Vec<Mxfp4Expert<'_>> = packed
            .iter()
            .map(|(gate, up, down, x, m)| Mxfp4Expert { gate, up, down, x, m: *m })
            .collect();
        for threads in [1usize, 4] {
            let got = mxfp4_swiglu_experts(&experts, dim, ff, threads);
            for (y, want) in got.iter().zip(&reference) {
                assert_eq!(y.len(), want.len());
                let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
                for (a, b) in y.iter().zip(want) {
                    assert!((a - b).abs() <= 1e-4 * scale, "threads={threads} got {a} want {b}");
                }
            }
        }
        // The scalar path agrees with the vector one.
        let row = &packed[0].0[..dim / 32 * MXFP4_BLOCK_BYTES];
        let x = &packed[0].3[..dim];
        let (a, b) = (dot_scalar(row, dim / 32, x), Dot::new().run(row, dim / 32, x));
        assert!((a - b).abs() <= 1e-4 * a.abs().max(1.0), "{a} vs {b}");
    }
}
