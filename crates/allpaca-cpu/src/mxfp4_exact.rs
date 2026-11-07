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

/// Output columns handed to a thread at a time.
const UNIT: usize = 16;

/// Whether the AVX2 kernels can run here. `ALLPACA_CPU_EXACT_SCALAR=1` forces the scalar
/// reference (they give the same bits; the scalar one is only slower).
fn use_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
                && !std::env::var("ALLPACA_CPU_EXACT_SCALAR").is_ok_and(|v| v == "1")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `out[i * (c1 - c0) + (c - c0)]` = dot of row `i` of `x` (`m` rows of `k`) with weight row
/// `c` (`k / 32` blocks), for the columns `c0..c1`, in the GPU's order for this `m`.
fn gemm_cols(m: usize, w: &[u8], k: usize, x: &[f32], c0: usize, c1: usize, out: &mut [f32], simd: bool) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: `simd` is only true when AVX2 and FMA were detected.
        unsafe { avx2::gemm_cols(m, w, k, x, c0, c1, out) };
        return;
    }
    let _ = simd;
    gemm_cols_scalar(m, w, k, x, c0, c1, out);
}

fn gemm_cols_scalar(m: usize, w: &[u8], k: usize, x: &[f32], c0: usize, c1: usize, out: &mut [f32]) {
    let nblk = k / MXFP4_BLOCK_ELEMS;
    let rb = nblk * MXFP4_BLOCK_BYTES;
    let width = c1 - c0;
    for c in c0..c1 {
        let row = &w[c * rb..(c + 1) * rb];
        for i in 0..m {
            out[i * width + (c - c0)] = dot(m, row, nblk, &x[i * k..(i + 1) * k]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! The same arithmetic as the scalar `dot_wide` / `dot_tiled`, eight chains at a time.
    //! Every lane runs the scalar sequence of `fma`, `mul` and `add` on its own data, so the
    //! bits are the scalar ones.
    use std::arch::x86_64::*;

    use super::{dot_tiled, e8m0_to_fp32_half, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};

    fn scales() -> &'static [f32; 256] {
        static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
        T.get_or_init(|| {
            let mut t = [0.0f32; 256];
            for (e, v) in t.iter_mut().enumerate() {
                *v = e8m0_to_fp32_half(e as u8);
            }
            t
        })
    }

    /// Code value for the low 4 bits of every lane of `n`: magnitude from the 8-entry table,
    /// bit 3 flips the sign (so code 8 is -0.0, as in the GPU wide kernel).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn code(n: __m256i, mags: __m256) -> __m256 {
        let mag = _mm256_permutevar8x32_ps(mags, n);
        let sign = _mm256_slli_epi32(_mm256_and_si256(n, _mm256_set1_epi32(8)), 28);
        _mm256_xor_ps(mag, _mm256_castsi256_ps(sign))
    }

    /// Nibble `half` (0 low, 1 high) of byte `b` (0..4) of every dword lane.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn nibble(dw: __m256i, b: i32, half: i32) -> __m256i {
        let sh = _mm_cvtsi32_si128(8 * b + 4 * half);
        _mm256_and_si256(_mm256_srl_epi32(dw, sh), _mm256_set1_epi32(15))
    }

    /// Four dword gathers covering the 16 code bytes (bytes 1..17) of the blocks at `offs`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gather_codes(base: *const u8, offs: __m256i) -> [__m256i; 4] {
        [
            _mm256_i32gather_epi32::<1>(base.add(1).cast(), offs),
            _mm256_i32gather_epi32::<1>(base.add(5).cast(), offs),
            _mm256_i32gather_epi32::<1>(base.add(9).cast(), offs),
            _mm256_i32gather_epi32::<1>(base.add(13).cast(), offs),
        ]
    }

    /// Scale (`e8m0 / 2`) of the blocks at `offs`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gather_scales(base: *const u8, offs: __m256i) -> __m256 {
        let e = _mm256_and_si256(_mm256_i32gather_epi32::<1>(base.cast(), offs), _mm256_set1_epi32(0xFF));
        _mm256_i32gather_ps::<4>(scales().as_ptr(), e)
    }

    /// `x` with each group of 8 blocks transposed: `xt[(g * 32 + p) * 8 + l] = x[(g * 8 + l) * 32 + p]`.
    fn transpose(x: &[f32], groups: usize) -> Vec<f32> {
        let mut xt = vec![0.0f32; groups * 256];
        for g in 0..groups {
            for l in 0..8 {
                for p in 0..32 {
                    xt[(g * 32 + p) * 8 + l] = x[(g * 8 + l) * 32 + p];
                }
            }
        }
        xt
    }

    /// `m == 1` dot of the row at `row` (`nblk` blocks) with `x`, bit-identical to `dot_wide`.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_wide(row: *const u8, nblk: usize, x: &[f32], xt: &[f32]) -> f32 {
        let groups = nblk / 8;
        let mags = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0);
        let offs = _mm256_setr_epi32(0, 17, 34, 51, 68, 85, 102, 119);
        let mut acc = [_mm256_setzero_ps(); 4];
        for g in 0..groups {
            let base = row.add(g * 8 * MXFP4_BLOCK_BYTES);
            let d = gather_scales(base, offs);
            let dw = gather_codes(base, offs);
            let xg = xt.as_ptr().add(g * 256);
            let mut s = _mm256_setzero_ps();
            for j in 0..8 {
                let (da, db) = if j < 4 { (dw[0], dw[2]) } else { (dw[1], dw[3]) };
                let b = (j % 4) as i32;
                let ca = code(nibble(da, b, 0), mags);
                let cb = code(nibble(da, b, 1), mags);
                let cc = code(nibble(db, b, 0), mags);
                let cd = code(nibble(db, b, 1), mags);
                let x0 = _mm256_loadu_ps(xg.add(j * 8));
                let x8 = _mm256_loadu_ps(xg.add((8 + j) * 8));
                let x16 = _mm256_loadu_ps(xg.add((16 + j) * 8));
                let x24 = _mm256_loadu_ps(xg.add((24 + j) * 8));
                s = _mm256_add_ps(s, _mm256_fmadd_ps(x0, ca, _mm256_mul_ps(x16, cb)));
                s = _mm256_add_ps(s, _mm256_fmadd_ps(x8, cc, _mm256_mul_ps(x24, cd)));
            }
            let a = &mut acc[g % 4];
            *a = _mm256_fmadd_ps(d, s, *a);
        }
        let mut lane = [0.0f32; 32];
        for q in 0..4 {
            _mm256_storeu_ps(lane.as_mut_ptr().add(q * 8), acc[q]);
        }
        // Blocks that do not fill a group of 8 go on the same lanes, one at a time.
        for ib in groups * 8..nblk {
            let blk = std::slice::from_raw_parts(row.add(ib * MXFP4_BLOCK_BYTES), MXFP4_BLOCK_BYTES);
            let bd = super::block_dot(blk, &x[ib * MXFP4_BLOCK_ELEMS..]);
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

    /// `m > 1`: eight output columns (`rb` bytes apart) for up to four rows of `x` (`k` apart),
    /// bit-identical to `dot_tiled` per column and row.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tiled8(w: *const u8, rb: usize, nblk: usize, x: &[f32], k: usize, rows: usize) -> [__m256; 4] {
        let mags = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0);
        let r = rb as i32;
        let offs = _mm256_setr_epi32(0, r, 2 * r, 3 * r, 4 * r, 5 * r, 6 * r, 7 * r);
        let mut acc = [_mm256_setzero_ps(); 4];
        for ib in 0..nblk {
            let base = w.add(ib * MXFP4_BLOCK_BYTES);
            let d = gather_scales(base, offs);
            let dw = gather_codes(base, offs);
            for j in 0..16 {
                let (src, b) = (dw[j / 4], (j % 4) as i32);
                let c0 = _mm256_mul_ps(code_signed_zero_plus(nibble(src, b, 0), mags), d);
                let c1 = _mm256_mul_ps(code_signed_zero_plus(nibble(src, b, 1), mags), d);
                for ri in 0..rows {
                    let xr = x.as_ptr().add(ri * k + ib * MXFP4_BLOCK_ELEMS);
                    acc[ri] = _mm256_fmadd_ps(_mm256_set1_ps(*xr.add(j)), c0, acc[ri]);
                    acc[ri] = _mm256_fmadd_ps(_mm256_set1_ps(*xr.add(16 + j)), c1, acc[ri]);
                }
            }
        }
        acc
    }

    /// The tiled kernel's code table is `int8` (-0 is +0): same as `code` but code 8 is +0.0.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn code_signed_zero_plus(n: __m256i, mags: __m256) -> __m256 {
        let mag = _mm256_permutevar8x32_ps(mags, n);
        let neg = _mm256_cmpeq_epi32(_mm256_and_si256(n, _mm256_set1_epi32(8)), _mm256_set1_epi32(8));
        let nz = _mm256_cmp_ps::<_CMP_NEQ_OQ>(mag, _mm256_setzero_ps());
        let flip = _mm256_and_si256(_mm256_and_si256(neg, _mm256_castps_si256(nz)), _mm256_set1_epi32(i32::MIN));
        _mm256_xor_ps(mag, _mm256_castsi256_ps(flip))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_cols(m: usize, w: &[u8], k: usize, x: &[f32], c0: usize, c1: usize, out: &mut [f32]) {
        let nblk = k / MXFP4_BLOCK_ELEMS;
        let rb = nblk * MXFP4_BLOCK_BYTES;
        let width = c1 - c0;
        if m == 1 {
            let groups = nblk / 8;
            let xt = transpose(&x[..k], groups);
            for c in c0..c1 {
                out[c - c0] = dot_wide(w.as_ptr().add(c * rb), nblk, &x[..k], &xt);
            }
            return;
        }
        let mut c = c0;
        while c + 8 <= c1 {
            let mut r0 = 0;
            while r0 < m {
                let rows = (m - r0).min(4);
                let acc = tiled8(w.as_ptr().add(c * rb), rb, nblk, &x[r0 * k..], k, rows);
                for ri in 0..rows {
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), acc[ri]);
                    for l in 0..8 {
                        out[(r0 + ri) * width + (c - c0) + l] = tmp[l];
                    }
                }
                r0 += rows;
            }
            c += 8;
        }
        while c < c1 {
            let row = &w[c * rb..(c + 1) * rb];
            for i in 0..m {
                out[i * width + (c - c0)] = dot_tiled(row, nblk, &x[i * k..(i + 1) * k]);
            }
            c += 1;
        }
    }
}

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
    let simd = use_simd();
    let units_up = ff.div_ceil(UNIT);
    let units_down = dim.div_ceil(UNIT);
    let (next_up, next_down) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let threads = threads.clamp(1, (n * units_up).max(1));
    let barrier = Barrier::new(threads);
    let work = || {
        let (mut tg, mut tu, mut td) = (Vec::new(), Vec::new(), Vec::new());
        loop {
            let at = next_up.fetch_add(1, Ordering::Relaxed);
            if at >= n * units_up {
                break;
            }
            let (e, u) = (at / units_up, at % units_up);
            let ex = &experts[e];
            let (c0, c1) = (u * UNIT, ((u + 1) * UNIT).min(ff));
            let width = c1 - c0;
            tg.resize(ex.m * width, 0.0f32);
            tu.resize(ex.m * width, 0.0f32);
            gemm_cols(ex.m, ex.gate, dim, ex.x, c0, c1, &mut tg, simd);
            gemm_cols(ex.m, ex.up, dim, ex.x, c0, c1, &mut tu, simd);
            for i in 0..ex.m {
                for c in 0..width {
                    let h = swiglu(tg[i * width + c], tu[i * width + c], limit);
                    // SAFETY: each (expert, row, column) is written by one thread.
                    unsafe { *hid[e].0.add(i * ff + c0 + c) = h };
                }
            }
        }
        barrier.wait();
        loop {
            let at = next_down.fetch_add(1, Ordering::Relaxed);
            if at >= n * units_down {
                break;
            }
            let (e, u) = (at / units_down, at % units_down);
            let ex = &experts[e];
            let (c0, c1) = (u * UNIT, ((u + 1) * UNIT).min(dim));
            let width = c1 - c0;
            td.resize(ex.m * width, 0.0f32);
            // SAFETY: hidden is complete (barrier) and read-only here.
            let h = unsafe { std::slice::from_raw_parts(hid[e].0.cast_const(), ex.m * ff) };
            gemm_cols(ex.m, ex.down, ff, h, c0, c1, &mut td, simd);
            for i in 0..ex.m {
                for c in 0..width {
                    unsafe { *dst[e].0.add(i * dim + c0 + c) = td[i * width + c] };
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

    /// The AVX2 columns give the scalar reference's bits, for every row-count class and for
    /// block counts that do and do not fill whole groups of 8.
    #[test]
    fn simd_columns_are_bit_identical_to_the_scalar_reference() {
        if !use_simd() {
            return;
        }
        for (k, n) in [(4096usize, 40usize), (352, 37), (96, 21), (256, 16)] {
            let w = quantize_mxfp4(&rand(n * k, 21 + k as u64, 0.5));
            let rb = k / 32 * MXFP4_BLOCK_BYTES;
            assert_eq!(w.len(), n * rb);
            for m in [1usize, 2, 3, 5, 9] {
                let x = rand(m * k, 31 + m as u64, 6.0);
                for (c0, c1) in [(0usize, n), (3, n.min(19)), (n - 5, n)] {
                    let width = c1 - c0;
                    let mut a = vec![0.0f32; m * width];
                    let mut b = vec![0.0f32; m * width];
                    gemm_cols(m, &w, k, &x, c0, c1, &mut a, true);
                    gemm_cols_scalar(m, &w, k, &x, c0, c1, &mut b);
                    for (i, (p, q)) in a.iter().zip(&b).enumerate() {
                        assert_eq!(p.to_bits(), q.to_bits(), "k={k} n={n} m={m} cols {c0}..{c1} idx={i}: simd {p} scalar {q}");
                    }
                }
            }
        }
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
