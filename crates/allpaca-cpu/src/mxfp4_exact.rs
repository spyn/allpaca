//! MXFP4 experts on the CPU that give the same bits as the GPU path (P21).
//!
//! The ordinary CPU expert in `mxfp4_expert.rs` is fast but sums in its own order, so a
//! DeepSeek-V4 expert computed on the CPU differs in the last bits from the same expert
//! computed on a GPU. Which experts run where depends on the GPU cache size, so output
//! then changed from start to start. This module reproduces the HIP kernels'
//! arithmetic exactly: the same fused multiply-adds (`mul_add`) in the same places and the
//! same order as the kernels, plain adds elsewhere, and `det_silu_mul` for the activation.
//!
//! The GPU picks its kernel by the number of rows `m` an expert runs on:
//! * `m == 1`: the wide kernel. 32 lanes per output; lane `L` sums blocks `L, L + 32, ...`
//!   (each block as `scale * block_dot`), then the lanes are combined by an xor butterfly
//!   (16, 8, 4, 2, 1).
//! * `m > 1`: the tiled multi-row kernel. One running sum per output, over the blocks in
//!   order, element `j` then element `j + 16` of each block, with the scale folded into
//!   the code first.
//!
//! The kernels are in `hip/src/kernels.cpp`: `gemv_nt_mxfp4_wide_kernel`,
//! `mxfp4_swiglu_gate_up_wide_kernel`, `gemm_nt_mxfp4_tiled_m_kernel`,
//! `mxfp4_swiglu_gate_up_tiled_m_kernel`. Change one and change this file with it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Barrier;

use crate::kernels::{
    det_silu_mul, e8m0_to_fp32_half, KVALUES_MXFP4, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS,
};
use crate::mxfp4_expert::Mxfp4Expert;

/// Value of a 4-bit code as the wide kernel computes it: magnitudes 0, 1, 2, 3, 4, 6, 8, 12,
/// bit 3 is the sign.
#[inline]
fn code_wide(n: u32) -> f32 {
    let mag = ((0x0C08_0604_0302_0100u64 >> ((n & 7) * 8)) & 0xFF) as f32;
    if n & 8 != 0 {
        -mag
    } else {
        mag
    }
}

/// Unscaled dot product of one 17-byte block with 32 inputs, in the wide kernel's order.
#[inline]
fn block_dot(blk: &[u8], x: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for j in 0..8 {
        let b0 = u32::from(blk[1 + j]);
        let b1 = u32::from(blk[9 + j]);
        s += x[j].mul_add(code_wide(b0 & 15), x[16 + j] * code_wide(b0 >> 4));
        s += x[8 + j].mul_add(code_wide(b1 & 15), x[24 + j] * code_wide(b1 >> 4));
    }
    s
}

/// `m == 1` dot of one row of `nblk` blocks with `x` (the GPU wide kernel).
fn dot_wide(row: &[u8], nblk: usize, x: &[f32]) -> f32 {
    let mut lane = [0.0f32; 32];
    for ib in 0..nblk {
        let blk = &row[ib * MXFP4_BLOCK_BYTES..(ib + 1) * MXFP4_BLOCK_BYTES];
        let bd = block_dot(blk, &x[ib * MXFP4_BLOCK_ELEMS..]);
        lane[ib % 32] = e8m0_to_fp32_half(blk[0]).mul_add(bd, lane[ib % 32]);
    }
    let mut off = 16;
    while off > 0 {
        let mut next = [0.0f32; 32];
        for i in 0..32 {
            next[i] = lane[i] + lane[i ^ off];
        }
        lane = next;
        off >>= 1;
    }
    lane[0]
}

/// `m > 1` dot of one row of `nblk` blocks with `x` (the GPU tiled multi-row kernel).
fn dot_tiled(row: &[u8], nblk: usize, x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for ib in 0..nblk {
        let blk = &row[ib * MXFP4_BLOCK_BYTES..(ib + 1) * MXFP4_BLOCK_BYTES];
        let d = e8m0_to_fp32_half(blk[0]);
        let xb = &x[ib * MXFP4_BLOCK_ELEMS..];
        for j in 0..16 {
            let p = blk[1 + j];
            let c0 = f32::from(KVALUES_MXFP4[usize::from(p & 0x0f)]) * d;
            let c1 = f32::from(KVALUES_MXFP4[usize::from(p >> 4)]) * d;
            acc = xb[j].mul_add(c0, acc);
            acc = xb[16 + j].mul_add(c1, acc);
        }
    }
    acc
}

#[inline]
fn dot(m: usize, row: &[u8], nblk: usize, x: &[f32]) -> f32 {
    if m == 1 {
        dot_wide(row, nblk, x)
    } else {
        dot_tiled(row, nblk, x)
    }
}

/// `silu(gate) * up` with the DeepSeek-V4 clamp, as the GPU `mxfp4_swiglu` computes it.
#[inline]
fn swiglu(mut g: f32, mut u: f32, limit: f32) -> f32 {
    if limit > 0.0 {
        g = g.min(limit);
        u = u.max(-limit).min(limit);
    }
    det_silu_mul(g, u)
}

#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

const CHUNK: usize = 16;

/// Same contract as `mxfp4_swiglu_experts`, but bit-identical to the GPU path.
/// `dim` and `ff` must be multiples of 32. The GPU kernel choice depends on each
/// expert's `m`, so `m` must be the number of rows the GPU would have run it on.
pub fn mxfp4_swiglu_experts_exact(
    experts: &[Mxfp4Expert<'_>],
    dim: usize,
    ff: usize,
    limit: f32,
    threads: usize,
) -> Vec<Vec<f32>> {
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
    let n = experts.len();
    let (next_up, next_down) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let threads = threads.clamp(1, (n * ff).div_ceil(CHUNK));
    let barrier = Barrier::new(threads);
    let work = || {
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
                    let h = swiglu(dot(ex.m, g, kb, x), dot(ex.m, u, kb, x), limit);
                    // SAFETY: each (expert, row, i) is written by one thread.
                    unsafe { *hid[e].0.add(i * ff + row) = h };
                }
            }
        }
        barrier.wait();
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
                    unsafe { *dst[e].0.add(i * dim + row) = dot(ex.m, d, fb, h) };
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

/// `c[r][col] = dot(a[r], b[col])` for `m` rows of `k` floats and `n` packed rows, in the
/// order of the GPU `gemm_nt_mxfp4` kernel for this `m`. Single-threaded; for tests.
pub fn mxfp4_gemm_exact(a: &[f32], b: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
    let nblk = k / MXFP4_BLOCK_ELEMS;
    let row_bytes = nblk * MXFP4_BLOCK_BYTES;
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            c[r * n + col] = dot(m, &b[col * row_bytes..(col + 1) * row_bytes], nblk, &a[r * k..(r + 1) * k]);
        }
    }
    c
}

/// Fused gate/up step as the GPU `mxfp4_swiglu_gate_up` computes it. Single-threaded; for tests.
pub fn mxfp4_gate_up_exact(
    a: &[f32],
    bg: &[u8],
    bu: &[u8],
    m: usize,
    n: usize,
    k: usize,
    limit: f32,
) -> Vec<f32> {
    let nblk = k / MXFP4_BLOCK_ELEMS;
    let row_bytes = nblk * MXFP4_BLOCK_BYTES;
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        let x = &a[r * k..(r + 1) * k];
        for col in 0..n {
            let g = dot(m, &bg[col * row_bytes..(col + 1) * row_bytes], nblk, x);
            let u = dot(m, &bu[col * row_bytes..(col + 1) * row_bytes], nblk, x);
            c[r * n + col] = swiglu(g, u, limit);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::quantize_mxfp4;
    use crate::mxfp4_expert::mxfp4_swiglu_experts;

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

    /// The exact path agrees with the fast CPU kernel up to summation order, for every
    /// row-count class, and does not depend on the thread count.
    #[test]
    fn exact_path_matches_the_fast_kernel_and_is_thread_independent() {
        let (dim, ff) = (96usize, 64usize);
        let gate = quantize_mxfp4(&rand(ff * dim, 3, 0.5));
        let up = quantize_mxfp4(&rand(ff * dim, 5, 0.5));
        let down = quantize_mxfp4(&rand(dim * ff, 7, 0.5));
        for m in [1usize, 2, 9] {
            let x = rand(m * dim, 11 + m as u64, 4.0);
            let expert = || Mxfp4Expert { gate: &gate, up: &up, down: &down, x: &x, m };
            let fast = mxfp4_swiglu_experts(&[expert()], dim, ff, 10.0, 1).remove(0);
            let one = mxfp4_swiglu_experts_exact(&[expert()], dim, ff, 10.0, 1).remove(0);
            let four = mxfp4_swiglu_experts_exact(&[expert()], dim, ff, 10.0, 4).remove(0);
            assert_eq!(
                one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                four.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "m={m}: thread count changed the bits"
            );
            let scale = fast.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
            for (a, b) in one.iter().zip(&fast) {
                assert!((a - b).abs() <= 1e-3 * scale, "m={m}: exact {a} fast {b}");
            }
        }
    }
}
