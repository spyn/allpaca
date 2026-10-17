//! ggml `Q2_0` (type 42) and `Q1_0` (type 41): a half-float scale per block
//! and 2 bits or 1 bit per weight.
//!
//! `Q2_0`: 64 weights in 18 bytes. Code `q` (weight `j` in bits `2*(j%4)` of
//! byte `j/4`) is `(q - 1) * d`: the grid -1, 0, +1, +2. The scale may be
//! negative (GSQ learns its sign), which mirrors the grid.
//!
//! `Q1_0`: 128 weights in 18 bytes; bit `j%8` of byte `j/8` set is `+d`,
//! clear is `-d`.

use allpaca_types::{AllpacaError, FatalError};

use crate::kernels::{f16_to_f32, f32_to_f16};

pub const Q2_0_BLOCK_ELEMS: usize = 64;
pub const Q2_0_BLOCK_BYTES: usize = 18;
pub const Q1_0_BLOCK_ELEMS: usize = 128;
pub const Q1_0_BLOCK_BYTES: usize = 18;

fn check(what: &str, blocks: &[u8], n: usize, out: &[f32], elems: usize, bytes: usize) -> Result<(), AllpacaError> {
    if out.len() < n || blocks.len() < n.div_ceil(elems) * bytes {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "{what} buffer too small"
        ))));
    }
    Ok(())
}

fn block_scale(block: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
}

/// Decode `n` weights from `Q2_0` blocks.
pub fn dequant_q2_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("dequant_q2_0", blocks, n, out, Q2_0_BLOCK_ELEMS, Q2_0_BLOCK_BYTES)?;
    for (block, dst) in blocks.chunks_exact(Q2_0_BLOCK_BYTES).zip(out[..n].chunks_mut(Q2_0_BLOCK_ELEMS)) {
        let d = block_scale(block);
        for (j, v) in dst.iter_mut().enumerate() {
            let q = (block[2 + j / 4] >> ((j % 4) * 2)) & 3;
            *v = (f32::from(q) - 1.0) * d;
        }
    }
    Ok(())
}

/// The four weights of one code byte, in units of the block scale.
static Q2_0_BYTE_WEIGHTS: [[f32; 4]; 256] = {
    let mut table = [[0.0f32; 4]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut j = 0;
        while j < 4 {
            table[byte][j] = ((byte >> (2 * j)) & 3) as f32 - 1.0;
            j += 1;
        }
        byte += 1;
    }
    table
};

fn q2_0_row_dot(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for (block, xb) in row.chunks_exact(Q2_0_BLOCK_BYTES).zip(x.chunks_exact(Q2_0_BLOCK_ELEMS)) {
        let mut s = 0.0f32;
        for (byte, x4) in block[2..].iter().zip(xb.chunks_exact(4)) {
            let w = &Q2_0_BYTE_WEIGHTS[usize::from(*byte)];
            s += w[0] * x4[0] + w[1] * x4[1] + w[2] * x4[2] + w[3] * x4[3];
        }
        total += block_scale(block) * s;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q2_0_row_dot_avx2(row: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut total = _mm256_setzero_ps();
    for (block, xb) in row.chunks_exact(Q2_0_BLOCK_BYTES).zip(x.chunks_exact(Q2_0_BLOCK_ELEMS)) {
        let mut acc = _mm256_setzero_ps();
        for j in 0..8 {
            let lo = _mm_loadu_ps(Q2_0_BYTE_WEIGHTS[usize::from(block[2 + 2 * j])].as_ptr());
            let hi = _mm_loadu_ps(Q2_0_BYTE_WEIGHTS[usize::from(block[3 + 2 * j])].as_ptr());
            acc = _mm256_fmadd_ps(_mm256_set_m128(hi, lo), _mm256_loadu_ps(xb.as_ptr().add(8 * j)), acc);
        }
        total = _mm256_fmadd_ps(acc, _mm256_set1_ps(block_scale(block)), total);
    }
    let quad = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)))
}

/// `out[r] = w[r] . x` for a row-major `Q2_0` matrix of `out.len()` rows of
/// `x.len()` weights, read from its blocks; rows must be whole blocks.
pub fn gemv_q2_0(blocks: &[u8], x: &[f32], out: &mut [f32]) -> Result<(), AllpacaError> {
    let row_bytes = x.len() / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES;
    if x.is_empty() || x.len() % Q2_0_BLOCK_ELEMS != 0 || blocks.len() < out.len() * row_bytes {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "gemv_q2_0 rows are not whole blocks of the matrix".into(),
        )));
    }
    #[cfg(target_arch = "x86_64")]
    let wide = std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");
    for (o, row) in out.iter_mut().zip(blocks.chunks_exact(row_bytes)) {
        #[cfg(target_arch = "x86_64")]
        if wide {
            // SAFETY: AVX2 and FMA were detected above.
            *o = unsafe { q2_0_row_dot_avx2(row, x) };
            continue;
        }
        *o = q2_0_row_dot(row, x);
    }
    Ok(())
}

/// A row of activations rounded to 8 bits per 64-value block, for
/// [`gemv_q2_0_i8`]: one input serves every matrix it multiplies.
#[derive(Debug, Clone)]
pub struct Q2_0Input {
    codes: Vec<i8>,
    /// Per block: the step of its codes, and the sum of its codes.
    step: Vec<f32>,
    sum: Vec<f32>,
}

impl Q2_0Input {
    /// `x` must be whole 64-value blocks.
    pub fn new(x: &[f32]) -> Result<Self, AllpacaError> {
        if x.len() % Q2_0_BLOCK_ELEMS != 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "Q2_0 input is not whole blocks".into(),
            )));
        }
        let n_blocks = x.len() / Q2_0_BLOCK_ELEMS;
        let (mut codes, mut step, mut sum) = (vec![0i8; x.len()], vec![0.0f32; n_blocks], vec![0.0f32; n_blocks]);
        for (b, (src, dst)) in x.chunks_exact(Q2_0_BLOCK_ELEMS).zip(codes.chunks_exact_mut(Q2_0_BLOCK_ELEMS)).enumerate() {
            let max = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            if max > 0.0 && max.is_finite() {
                let inv = 127.0 / max;
                let mut total = 0i32;
                for (c, v) in dst.iter_mut().zip(src) {
                    *c = (v * inv).round() as i8;
                    total += i32::from(*c);
                }
                step[b] = max / 127.0;
                sum[b] = total as f32;
            }
        }
        Ok(Self { codes, step, sum })
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

fn q2_0_row_dot_i8(row: &[u8], input: &Q2_0Input) -> f32 {
    let mut total = 0.0f32;
    for (b, (block, xb)) in row.chunks_exact(Q2_0_BLOCK_BYTES).zip(input.codes.chunks_exact(Q2_0_BLOCK_ELEMS)).enumerate() {
        let mut s = 0i32;
        for (j, x) in xb.iter().enumerate() {
            s += i32::from((block[2 + j / 4] >> ((j % 4) * 2)) & 3) * i32::from(*x);
        }
        total += block_scale(block) * input.step[b] * (s as f32 - input.sum[b]);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q2_0_row_dot_i8_avx2(row: &[u8], input: &Q2_0Input) -> f32 {
    use std::arch::x86_64::*;
    let three = _mm_set1_epi8(3);
    let ones = _mm256_set1_epi16(1);
    let mut total = _mm256_setzero_ps();
    let mut offset = 0.0f32;
    for (b, block) in row.chunks_exact(Q2_0_BLOCK_BYTES).enumerate() {
        // Sixteen code bytes to sixty-four codes in weight order.
        let c = _mm_loadu_si128(block.as_ptr().add(2).cast());
        let f0 = _mm_and_si128(c, three);
        let f1 = _mm_and_si128(_mm_srli_epi16(c, 2), three);
        let f2 = _mm_and_si128(_mm_srli_epi16(c, 4), three);
        let f3 = _mm_and_si128(_mm_srli_epi16(c, 6), three);
        let (lo01, hi01) = (_mm_unpacklo_epi8(f0, f1), _mm_unpackhi_epi8(f0, f1));
        let (lo23, hi23) = (_mm_unpacklo_epi8(f2, f3), _mm_unpackhi_epi8(f2, f3));
        let first = _mm256_set_m128i(_mm_unpackhi_epi16(lo01, lo23), _mm_unpacklo_epi16(lo01, lo23));
        let second = _mm256_set_m128i(_mm_unpackhi_epi16(hi01, hi23), _mm_unpacklo_epi16(hi01, hi23));
        let x = input.codes.as_ptr().add(b * Q2_0_BLOCK_ELEMS);
        // Codes are 0..3 and activations within 127: pair sums fit 16 bits.
        let pairs = _mm256_add_epi16(
            _mm256_maddubs_epi16(first, _mm256_loadu_si256(x.cast())),
            _mm256_maddubs_epi16(second, _mm256_loadu_si256(x.add(32).cast())),
        );
        let scale = block_scale(block) * input.step[b];
        total = _mm256_add_ps(
            total,
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones)), _mm256_set1_ps(scale)),
        );
        offset += scale * input.sum[b];
    }
    let quad = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
    let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
    _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1))) - offset
}

/// [`gemv_q2_0`] on activations rounded to 8 bits per block: several times
/// faster, within a fraction of a percent of the output's scale.
pub fn gemv_q2_0_i8(blocks: &[u8], input: &Q2_0Input, out: &mut [f32]) -> Result<(), AllpacaError> {
    let row_bytes = input.len() / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES;
    if input.is_empty() || blocks.len() < out.len() * row_bytes {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "gemv_q2_0_i8 rows are not whole blocks of the matrix".into(),
        )));
    }
    #[cfg(target_arch = "x86_64")]
    let wide = std::arch::is_x86_feature_detected!("avx2");
    for (o, row) in out.iter_mut().zip(blocks.chunks_exact(row_bytes)) {
        #[cfg(target_arch = "x86_64")]
        if wide {
            // SAFETY: AVX2 was detected above.
            *o = unsafe { q2_0_row_dot_i8_avx2(row, input) };
            continue;
        }
        *o = q2_0_row_dot_i8(row, input);
    }
    Ok(())
}

/// [`gemv_q2_0_i8`] of several inputs (same length) through one read of
/// the matrix: `out` holds a row of the matrix's rows per input, each equal
/// bit for bit to that input's own product.
pub fn gemm_q2_0_i8(blocks: &[u8], inputs: &[&Q2_0Input], out: &mut [f32]) -> Result<(), AllpacaError> {
    let Some(first) = inputs.first() else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig("gemm_q2_0_i8 has no inputs".into())));
    };
    let n = out.len() / inputs.len();
    let row_bytes = first.len() / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES;
    if first.is_empty()
        || out.len() != n * inputs.len()
        || inputs.iter().any(|x| x.len() != first.len())
        || blocks.len() < n * row_bytes
    {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "gemm_q2_0_i8 inputs, rows and matrix do not line up".into(),
        )));
    }
    #[cfg(target_arch = "x86_64")]
    let wide = std::arch::is_x86_feature_detected!("avx2");
    let mut dots = [0.0f32; Q2_0_GEMM_INPUTS];
    for (r, row) in blocks.chunks_exact(row_bytes).take(n).enumerate() {
        for (group, chunk) in inputs.chunks(Q2_0_GEMM_INPUTS).enumerate() {
            let dots = &mut dots[..chunk.len()];
            #[cfg(target_arch = "x86_64")]
            if wide {
                // SAFETY: AVX2 was detected above.
                unsafe { q2_0_rows_dot_i8_avx2(row, chunk, dots) };
            } else {
                dots.iter_mut().zip(chunk).for_each(|(d, x)| *d = q2_0_row_dot_i8(row, x));
            }
            #[cfg(not(target_arch = "x86_64"))]
            dots.iter_mut().zip(chunk).for_each(|(d, x)| *d = q2_0_row_dot_i8(row, x));
            for (i, d) in dots.iter().enumerate() {
                out[(group * Q2_0_GEMM_INPUTS + i) * n + r] = *d;
            }
        }
    }
    Ok(())
}

/// Inputs [`gemm_q2_0_i8`] takes per decode of a block.
const Q2_0_GEMM_INPUTS: usize = 8;

/// [`q2_0_row_dot_i8_avx2`] of up to [`Q2_0_GEMM_INPUTS`] inputs, each
/// block's codes unpacked once; each dot's sums run in the same order.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q2_0_rows_dot_i8_avx2(row: &[u8], inputs: &[&Q2_0Input], dots: &mut [f32]) {
    use std::arch::x86_64::*;
    let three = _mm_set1_epi8(3);
    let ones = _mm256_set1_epi16(1);
    let mut totals = [_mm256_setzero_ps(); Q2_0_GEMM_INPUTS];
    let mut offsets = [0.0f32; Q2_0_GEMM_INPUTS];
    for (b, block) in row.chunks_exact(Q2_0_BLOCK_BYTES).enumerate() {
        let c = _mm_loadu_si128(block.as_ptr().add(2).cast());
        let f0 = _mm_and_si128(c, three);
        let f1 = _mm_and_si128(_mm_srli_epi16(c, 2), three);
        let f2 = _mm_and_si128(_mm_srli_epi16(c, 4), three);
        let f3 = _mm_and_si128(_mm_srli_epi16(c, 6), three);
        let (lo01, hi01) = (_mm_unpacklo_epi8(f0, f1), _mm_unpackhi_epi8(f0, f1));
        let (lo23, hi23) = (_mm_unpacklo_epi8(f2, f3), _mm_unpackhi_epi8(f2, f3));
        let first = _mm256_set_m128i(_mm_unpackhi_epi16(lo01, lo23), _mm_unpacklo_epi16(lo01, lo23));
        let second = _mm256_set_m128i(_mm_unpackhi_epi16(hi01, hi23), _mm_unpacklo_epi16(hi01, hi23));
        let d = block_scale(block);
        for (i, input) in inputs.iter().enumerate() {
            let x = input.codes.as_ptr().add(b * Q2_0_BLOCK_ELEMS);
            let pairs = _mm256_add_epi16(
                _mm256_maddubs_epi16(first, _mm256_loadu_si256(x.cast())),
                _mm256_maddubs_epi16(second, _mm256_loadu_si256(x.add(32).cast())),
            );
            let scale = d * input.step[b];
            totals[i] = _mm256_add_ps(
                totals[i],
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(pairs, ones)), _mm256_set1_ps(scale)),
            );
            offsets[i] += scale * input.sum[b];
        }
    }
    for (i, dot) in dots.iter_mut().enumerate() {
        let total = totals[i];
        let quad = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        let pair = _mm_add_ps(quad, _mm_movehl_ps(quad, quad));
        *dot = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1))) - offsets[i];
    }
}

/// ggml's reference `Q2_0` quantiser: the scale is the block's largest
/// magnitude, each weight rounds to the nearest of -1, 0, +1, +2 scales.
pub fn quantize_q2_0(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len().div_ceil(Q2_0_BLOCK_ELEMS) * Q2_0_BLOCK_BYTES];
    for (src, block) in data.chunks(Q2_0_BLOCK_ELEMS).zip(out.chunks_exact_mut(Q2_0_BLOCK_BYTES)) {
        let d = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        // Missing weights of a short last block are zero: code 1.
        block[2..].fill(0x55);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for (j, w) in src.iter().enumerate() {
            let q = ((w * inv).round() as i32 + 1).clamp(0, 3) as u8;
            let shift = (j % 4) * 2;
            block[2 + j / 4] = (block[2 + j / 4] & !(3 << shift)) | (q << shift);
        }
    }
    out
}

/// Decode `n` weights from `Q1_0` blocks.
pub fn dequant_q1_0(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("dequant_q1_0", blocks, n, out, Q1_0_BLOCK_ELEMS, Q1_0_BLOCK_BYTES)?;
    for (block, dst) in blocks.chunks_exact(Q1_0_BLOCK_BYTES).zip(out[..n].chunks_mut(Q1_0_BLOCK_ELEMS)) {
        let d = block_scale(block);
        for (j, v) in dst.iter_mut().enumerate() {
            *v = if (block[2 + j / 8] >> (j % 8)) & 1 == 1 { d } else { -d };
        }
    }
    Ok(())
}

/// ggml's reference `Q1_0` quantiser: the scale is the block's mean
/// magnitude, each weight keeps its sign.
pub fn quantize_q1_0(data: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; data.len().div_ceil(Q1_0_BLOCK_ELEMS) * Q1_0_BLOCK_BYTES];
    for (src, block) in data.chunks(Q1_0_BLOCK_ELEMS).zip(out.chunks_exact_mut(Q1_0_BLOCK_BYTES)) {
        let d = src.iter().map(|v| v.abs()).sum::<f32>() / Q1_0_BLOCK_ELEMS as f32;
        block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (j, w) in src.iter().enumerate() {
            if *w >= 0.0 {
                block[2 + j / 8] |= 1 << (j % 8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q2_0_codes_are_minus_one_to_plus_two_scales() {
        // Scale 0.5 (half 0x3800); codes 0, 1, 2, 3 in the first byte, then
        // code 3 for weight 4 and code 0 for weight 63.
        let mut block = [0x55u8; 18];
        block[..2].copy_from_slice(&[0x00, 0x38]);
        block[2] = 0b11_10_01_00;
        block[3] = 0b01_01_01_11;
        block[17] = 0b00_01_01_01;
        let mut out = [9.0f32; 64];
        dequant_q2_0(&block, 64, &mut out).unwrap();
        assert_eq!(&out[..5], &[-0.5, 0.0, 0.5, 1.0, 1.0]);
        assert!(out[5..63].iter().all(|v| *v == 0.0));
        assert_eq!(out[63], -0.5);
        // A negative scale mirrors the grid: +1, 0, -1, -2.
        block[..2].copy_from_slice(&[0x00, 0xb8]);
        dequant_q2_0(&block, 64, &mut out).unwrap();
        assert_eq!(&out[..4], &[0.5, 0.0, -0.5, -1.0]);
        assert!(dequant_q2_0(&block[..17], 64, &mut out).is_err());
    }

    #[test]
    fn q2_0_gemv_on_the_blocks_matches_dequant_then_dot() {
        let (rows, cols) = (5usize, 192usize);
        let mut blocks = vec![0u8; rows * cols / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES];
        let mut seed = 0x2545_f491u32;
        for (bi, block) in blocks.chunks_exact_mut(Q2_0_BLOCK_BYTES).enumerate() {
            for byte in block[2..].iter_mut() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (seed >> 24) as u8;
            }
            let d = (0.1 + 0.05 * bi as f32) * if bi % 2 == 0 { 1.0 } else { -1.0 };
            block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }
        let x: Vec<f32> = (0..cols).map(|i| ((i * 29 % 67) as f32 / 33.0 - 1.0) * 0.7).collect();
        let mut w = vec![0.0f32; rows * cols];
        dequant_q2_0(&blocks, rows * cols, &mut w).unwrap();
        let mut got = vec![0.0f32; rows];
        gemv_q2_0(&blocks, &x, &mut got).unwrap();
        for (r, got) in got.iter().enumerate() {
            let want: f32 = w[r * cols..(r + 1) * cols].iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "row {r}: {got} vs {want}");
            let scalar = q2_0_row_dot(&blocks[r * 54..(r + 1) * 54], &x);
            assert!((scalar - want).abs() <= 1e-4 * want.abs().max(1.0), "scalar row {r}: {scalar} vs {want}");
        }
        assert!(got.iter().any(|v| v.abs() > 0.1));
        // Rows that are not whole blocks, or more rows than the blocks hold.
        assert!(gemv_q2_0(&blocks, &x[..100], &mut got).is_err());
        assert!(gemv_q2_0(&blocks, &x, &mut [0.0; 6]).is_err());
    }

    #[test]
    fn q2_0_gemv_on_8_bit_activations_stays_close_to_the_float_one() {
        let (rows, cols) = (7usize, 640usize);
        let mut blocks = vec![0u8; rows * cols / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES];
        let mut seed = 0x1f3d_5b79u32;
        for (bi, block) in blocks.chunks_exact_mut(Q2_0_BLOCK_BYTES).enumerate() {
            for byte in block[2..].iter_mut() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (seed >> 24) as u8;
            }
            let d = (0.02 + 0.01 * (bi % 9) as f32) * if bi % 4 == 0 { -1.0 } else { 1.0 };
            block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }
        // Blocks of very different sizes, one all zero.
        let x: Vec<f32> = (0..cols)
            .map(|i| match i / 64 {
                3 => 0.0,
                b => ((i * 31 % 89) as f32 / 44.0 - 1.0) * [0.01, 1.0, 30.0][b % 3],
            })
            .collect();
        let mut exact = vec![0.0f32; rows];
        gemv_q2_0(&blocks, &x, &mut exact).unwrap();
        let input = Q2_0Input::new(&x).unwrap();
        let mut got = vec![0.0f32; rows];
        gemv_q2_0_i8(&blocks, &input, &mut got).unwrap();
        let scale = exact.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1.0);
        let row_bytes = cols / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES;
        for r in 0..rows {
            assert!((got[r] - exact[r]).abs() <= 0.01 * scale, "row {r}: {} vs {}", got[r], exact[r]);
            let scalar = q2_0_row_dot_i8(&blocks[r * row_bytes..(r + 1) * row_bytes], &input);
            assert!((scalar - got[r]).abs() <= 1e-4 * scale, "scalar row {r}: {scalar} vs {}", got[r]);
        }
        assert!(Q2_0Input::new(&x[..100]).is_err());
        assert!(gemv_q2_0_i8(&blocks, &input, &mut [0.0; 8]).is_err());
    }

    /// Several inputs through one read of the matrix give each input's
    /// [`gemv_q2_0_i8`] bit for bit, for any number of inputs.
    #[test]
    fn q2_0_i8_products_of_several_inputs_equal_one_at_a_time() {
        let (rows, cols) = (7usize, 256usize);
        let mut blocks = vec![0u8; rows * cols / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES];
        let mut seed = 0x1234_5677u32;
        for (bi, block) in blocks.chunks_exact_mut(Q2_0_BLOCK_BYTES).enumerate() {
            for byte in block[2..].iter_mut() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (seed >> 24) as u8;
            }
            let d = (0.03 + 0.01 * (bi % 5) as f32) * if bi % 3 == 0 { -1.0 } else { 1.0 };
            block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }
        for n_inputs in [1usize, 3, 8, 11, 16] {
            let inputs: Vec<Q2_0Input> = (0..n_inputs)
                .map(|i| {
                    let x: Vec<f32> = (0..cols).map(|j| ((j * 13 + i * 7) % 41) as f32 / 20.0 - 1.0 + i as f32 * 0.1).collect();
                    Q2_0Input::new(&x).unwrap()
                })
                .collect();
            let refs: Vec<&Q2_0Input> = inputs.iter().collect();
            let mut got = vec![0.0f32; n_inputs * rows];
            gemm_q2_0_i8(&blocks, &refs, &mut got).unwrap();
            for (i, input) in inputs.iter().enumerate() {
                let mut want = vec![0.0f32; rows];
                gemv_q2_0_i8(&blocks, input, &mut want).unwrap();
                let row = &got[i * rows..(i + 1) * rows];
                assert!(row.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "{n_inputs} inputs, input {i}: {row:?} vs {want:?}");
            }
        }
        assert!(gemm_q2_0_i8(&blocks, &[], &mut []).is_err());
    }

    /// Opt-in: seconds per expert-sized SwiGLU (640 x 2560 twice, 2560 x 640).
    #[test]
    #[ignore]
    fn bench_gemv_q2_0() {
        let (dim, ff) = (2560usize, 640usize);
        let bytes = 3 * ff * dim / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES;
        let mut seed = 7u32;
        // 64 experts so the data does not sit in cache.
        let pool: Vec<u8> = (0..64 * bytes)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect();
        let x: Vec<f32> = (0..dim).map(|i| (i % 17) as f32 * 0.01).collect();
        let (mut h, mut u, mut y) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; dim]);
        let start = std::time::Instant::now();
        for e in 0..64 {
            let m = &pool[e * bytes..(e + 1) * bytes];
            gemv_q2_0(&m[..bytes / 3], &x, &mut h).unwrap();
            gemv_q2_0(&m[bytes / 3..2 * bytes / 3], &x, &mut u).unwrap();
            gemv_q2_0(&m[2 * bytes / 3..], &h, &mut y).unwrap();
        }
        eprintln!("gemv_q2_0: {:.3} ms per expert, one thread", start.elapsed().as_secs_f64() * 1e3 / 64.0);
        let start = std::time::Instant::now();
        for e in 0..64 {
            let m = &pool[e * bytes..(e + 1) * bytes];
            let input = Q2_0Input::new(&x).unwrap();
            gemv_q2_0_i8(&m[..bytes / 3], &input, &mut h).unwrap();
            gemv_q2_0_i8(&m[bytes / 3..2 * bytes / 3], &input, &mut u).unwrap();
            gemv_q2_0_i8(&m[2 * bytes / 3..], &Q2_0Input::new(&h).unwrap(), &mut y).unwrap();
        }
        eprintln!("gemv_q2_0_i8: {:.3} ms per expert, one thread", start.elapsed().as_secs_f64() * 1e3 / 64.0);
        let start = std::time::Instant::now();
        std::thread::scope(|s| {
            for _ in 0..10 {
                s.spawn(|| std::hint::black_box(0));
            }
        });
        eprintln!("spawning and joining 10 threads: {:.3} ms", start.elapsed().as_secs_f64() * 1e3);
    }

    #[test]
    fn q2_0_reference_quantiser_rounds_to_the_grid() {
        // Largest magnitude 2.0 is the scale: the grid is -2, 0, 2, 4.
        let mut w = vec![0.0f32; 70];
        w[..6].copy_from_slice(&[2.0, -2.0, 0.9, 1.1, -0.9, -1.1]);
        w[64] = -3.0;
        w[65] = 1.0;
        let blocks = quantize_q2_0(&w);
        assert_eq!(blocks.len(), 2 * Q2_0_BLOCK_BYTES);
        let mut out = vec![9.0f32; 70];
        dequant_q2_0(&blocks, 70, &mut out).unwrap();
        assert_eq!(&out[..6], &[2.0, -2.0, 0.0, 2.0, 0.0, -2.0]);
        // The second block has its own scale (3.0); 1.0 rounds to zero.
        assert_eq!(&out[64..66], &[-3.0, 0.0]);
        assert!(out[6..64].iter().chain(&out[66..]).all(|v| *v == 0.0));
    }

    #[test]
    fn q1_0_keeps_signs_at_the_mean_magnitude() {
        let mut w = vec![1.0f32; 128];
        w[1] = -3.0;
        w[127] = -1.0;
        let blocks = quantize_q1_0(&w);
        assert_eq!(blocks.len(), Q1_0_BLOCK_BYTES);
        let mut out = vec![0.0f32; 128];
        dequant_q1_0(&blocks, 128, &mut out).unwrap();
        let d = 130.0 / 128.0;
        assert_eq!(out[0], d);
        assert_eq!(out[1], -d);
        assert_eq!(out[127], -d);
        assert!(out[2..127].iter().all(|v| *v == d));
    }
}
