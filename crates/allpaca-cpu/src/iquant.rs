//! ggml's low-bit "i-quants" with lattice codebooks: `IQ2_XXS` (16),
//! `IQ2_XS` (17), `IQ3_XXS` (18), `IQ1_S` (19), `IQ2_S` (22) and `IQ1_M` (29).
//!
//! Every type packs 256 weights per block. Groups of 8 (or 4) weights are
//! one point of a fixed grid; signs and sub-block scales come from the rest
//! of the block. Each decoder follows llama.cpp's `dequantize_row_*`
//! (`ggml/src/ggml-quants.c`) operation for operation, so the f32 results
//! are bit-identical to it (checked against `iq_golden.txt`).

use allpaca_types::{AllpacaError, FatalError};

use crate::iq_grids::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS};
use crate::kernels::f16_to_f32;

/// Weights per block, the same for every type in this module.
pub const IQ_BLOCK_ELEMS: usize = 256;
pub const IQ2_XXS_BLOCK_BYTES: usize = 66;
pub const IQ2_XS_BLOCK_BYTES: usize = 74;
pub const IQ2_S_BLOCK_BYTES: usize = 82;
pub const IQ3_XXS_BLOCK_BYTES: usize = 98;
pub const IQ1_S_BLOCK_BYTES: usize = 50;
pub const IQ1_M_BLOCK_BYTES: usize = 56;

/// Fail unless `n` is whole blocks and both buffers hold them.
fn check(name: &str, blocks: &[u8], block_bytes: usize, n: usize, out: &[f32]) -> Result<(), AllpacaError> {
    if n % IQ_BLOCK_ELEMS != 0 || out.len() < n || blocks.len() < n / IQ_BLOCK_ELEMS * block_bytes {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "dequant_{name}: {n} weights are not whole blocks, or a buffer is too small"
        ))));
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of `IQ2_XXS`: 2.0625 bits per weight.
///
/// # Errors
/// When `n` is not a multiple of 256 or a buffer is too small.
pub fn dequant_iq2_xxs(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("iq2_xxs", blocks, IQ2_XXS_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(IQ2_XXS_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(IQ_BLOCK_ELEMS)) {
        let d = half(block, 0);
        // Per 32 weights, 8 bytes: four grid indices, then a word holding
        // four 7-bit sign indices and a 4-bit scale on top.
        for (group, y) in block[2..].chunks_exact(8).zip(y.chunks_exact_mut(32)) {
            let aux = u32::from_le_bytes([group[4], group[5], group[6], group[7]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
            for (l, y) in y.chunks_exact_mut(8).enumerate() {
                let grid = IQ2XXS_GRID[usize::from(group[l])].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                put_signed(y, db, &grid, signs);
            }
        }
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of `IQ2_XS`: 2.3125 bits per weight.
///
/// # Errors
/// When `n` is not a multiple of 256 or a buffer is too small.
pub fn dequant_iq2_xs(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("iq2_xs", blocks, IQ2_XS_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(IQ2_XS_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(IQ_BLOCK_ELEMS)) {
        let d = half(block, 0);
        let (qs, scales) = (&block[2..66], &block[66..74]);
        // Per 32 weights: four 16-bit words (9-bit grid index, 7-bit sign
        // index) and one byte with a 4-bit scale for each half.
        for ((words, &sc), y) in qs.chunks_exact(8).zip(scales).zip(y.chunks_exact_mut(32)) {
            let db = [d * (0.5 + f32::from(sc & 0xf)) * 0.25, d * (0.5 + f32::from(sc >> 4)) * 0.25];
            for (l, (w, y)) in words.chunks_exact(2).zip(y.chunks_exact_mut(8)).enumerate() {
                let q = u16::from_le_bytes([w[0], w[1]]);
                let grid = IQ2XS_GRID[usize::from(q & 511)].to_le_bytes();
                put_signed(y, db[l / 2], &grid, KSIGNS_IQ2XS[usize::from(q >> 9)]);
            }
        }
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of `IQ2_S`: 2.5625 bits per weight.
///
/// # Errors
/// When `n` is not a multiple of 256 or a buffer is too small.
pub fn dequant_iq2_s(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("iq2_s", blocks, IQ2_S_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(IQ2_S_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(IQ_BLOCK_ELEMS)) {
        let d = half(block, 0);
        let (qs, signs, qh, scales) = (&block[2..34], &block[34..66], &block[66..74], &block[74..82]);
        // Per 32 weights: four low index bytes, their two high bits each in
        // one `qh` byte, four sign bytes, and a 4-bit scale for each half.
        for (ib, y) in y.chunks_exact_mut(32).enumerate() {
            let sc = scales[ib];
            let db = [d * (0.5 + f32::from(sc & 0xf)) * 0.25, d * (0.5 + f32::from(sc >> 4)) * 0.25];
            let high = u16::from(qh[ib]);
            for (l, y) in y.chunks_exact_mut(8).enumerate() {
                let index = u16::from(qs[4 * ib + l]) | ((high << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[usize::from(index)].to_le_bytes();
                put_signed(y, db[l / 2], &grid, signs[4 * ib + l]);
            }
        }
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of `IQ3_XXS`: 3.0625 bits per weight.
///
/// # Errors
/// When `n` is not a multiple of 256 or a buffer is too small.
pub fn dequant_iq3_xxs(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("iq3_xxs", blocks, IQ3_XXS_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(IQ3_XXS_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(IQ_BLOCK_ELEMS)) {
        let d = half(block, 0);
        let (qs, scales_and_signs) = (&block[2..66], &block[66..98]);
        // Per 32 weights: eight grid indices of 4 weights each, and a word
        // holding four 7-bit sign indices and a 4-bit scale on top.
        for ((idx, aux), y) in qs.chunks_exact(8).zip(scales_and_signs.chunks_exact(4)).zip(y.chunks_exact_mut(32)) {
            let aux = u32::from_le_bytes([aux[0], aux[1], aux[2], aux[3]]);
            let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
            for (l, (pair, y)) in idx.chunks_exact(2).zip(y.chunks_exact_mut(8)).enumerate() {
                let mut grid = [0u8; 8];
                grid[..4].copy_from_slice(&IQ3XXS_GRID[usize::from(pair[0])].to_le_bytes());
                grid[4..].copy_from_slice(&IQ3XXS_GRID[usize::from(pair[1])].to_le_bytes());
                put_signed(y, db, &grid, KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize]);
            }
        }
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of `IQ1_S`: 1.5625 bits per weight.
///
/// # Errors
/// When `n` is not a multiple of 256 or a buffer is too small.
pub fn dequant_iq1_s(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("iq1_s", blocks, IQ1_S_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(IQ1_S_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(IQ_BLOCK_ELEMS)) {
        let d = half(block, 0);
        let (qs, qh) = (&block[2..34], &block[34..50]);
        // Per 32 weights: four low index bytes and one word with their three
        // high bits each, a 3-bit scale and the sign of the shift.
        for ((idx, h), y) in qs.chunks_exact(4).zip(qh.chunks_exact(2)).zip(y.chunks_exact_mut(32)) {
            let h = u16::from_le_bytes([h[0], h[1]]);
            let dl = d * f32::from(2 * ((h >> 12) & 7) + 1);
            let delta = if h & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            for (l, y) in y.chunks_exact_mut(8).enumerate() {
                let index = u16::from(idx[l]) | (((h >> (3 * l)) & 7) << 8);
                put_shifted(y, dl, IQ1S_GRID[usize::from(index)], delta);
            }
        }
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of `IQ1_M`: 1.75 bits per weight.
///
/// # Errors
/// When `n` is not a multiple of 256 or a buffer is too small.
pub fn dequant_iq1_m(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("iq1_m", blocks, IQ1_M_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(IQ1_M_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(IQ_BLOCK_ELEMS)) {
        let (qs, qh) = (&block[..32], &block[32..48]);
        let sc: [u16; 4] = std::array::from_fn(|i| u16::from_le_bytes([block[48 + 2 * i], block[49 + 2 * i]]));
        // The block's half scale is spread over the top nibbles of the four
        // scale words.
        let d = f16_to_f32((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        // Per 32 weights: four low index bytes, two `qh` bytes with the high
        // index bits and shift signs of each 8, and 3-bit scales for each 16.
        for (ib, ((idx, h), y)) in qs.chunks_exact(4).zip(qh.chunks_exact(2)).zip(y.chunks_exact_mut(32)).enumerate() {
            let shift = 6 * (ib % 2);
            let dl1 = d * f32::from(2 * ((sc[ib / 2] >> shift) & 7) + 1);
            let dl2 = d * f32::from(2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1);
            let (h0, h1) = (u16::from(h[0]), u16::from(h[1]));
            let index = [
                u16::from(idx[0]) | ((h0 << 8) & 0x700),
                u16::from(idx[1]) | ((h0 << 4) & 0x700),
                u16::from(idx[2]) | ((h1 << 8) & 0x700),
                u16::from(idx[3]) | ((h1 << 4) & 0x700),
            ];
            let delta = |bits: u16| if bits != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let deltas = [delta(h0 & 0x08), delta(h0 & 0x80), delta(h1 & 0x08), delta(h1 & 0x80)];
            for (l, y) in y.chunks_exact_mut(8).enumerate() {
                let dl = if l < 2 { dl1 } else { dl2 };
                put_shifted(y, dl, IQ1S_GRID[usize::from(index[l])], deltas[l]);
            }
        }
    }
    Ok(())
}

/// The little-endian half at `at` as f32 (exact, as `GGML_FP16_TO_FP32`).
fn half(block: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([block[at], block[at + 1]]))
}

/// `IQ1S_DELTA` / `IQ1M_DELTA` in ggml-common.h: the grid shift of the IQ1 types.
const IQ1S_DELTA: f32 = 0.125;

/// `y[j] = scale * (grid[j] + delta)` over the eight signed bytes of an
/// `iq1s_grid` entry, as llama.cpp's IQ1 decoders round it.
#[inline]
fn put_shifted(y: &mut [f32], scale: f32, grid: u64, delta: f32) {
    for (y, g) in y.iter_mut().zip(grid.to_le_bytes()) {
        *y = scale * (f32::from(g as i8) + delta);
    }
}

/// `y[j] = scale * grid[j]`, negated where bit `j` of `signs` is set: the
/// same rounding as llama.cpp's `db * grid[j] * (sign ? -1.f : 1.f)`.
#[inline]
fn put_signed(y: &mut [f32], scale: f32, grid: &[u8], signs: u8) {
    for (j, (y, &g)) in y.iter_mut().zip(grid).enumerate() {
        let v = scale * f32::from(g);
        *y = if signs & (1 << j) != 0 { -v } else { v };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// Blocks and llama.cpp's dequantised f32 bits for `name` from
    /// `iq_golden.txt`: random blocks with a finite scale, decoded by
    /// llama.cpp's own `dequantize_row_*` (`iq_golden_gen.c`, built against
    /// ggml-org/llama.cpp 3e7344670adf63ce28527a4d42f2d71eca27c41e).
    fn golden(name: &str) -> (Vec<u8>, Vec<u32>) {
        let line = include_str!("iq_golden.txt")
            .lines()
            .find(|l| l.split(' ').next() == Some(name))
            .unwrap_or_else(|| panic!("no golden blocks for {name}"));
        let mut parts = line.split(' ').skip(1);
        let blocks = unhex(parts.next().unwrap());
        let bits = unhex(parts.next().unwrap())
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (blocks, bits)
    }

    /// Decode the golden blocks of `name` and require the exact f32 bits
    /// llama.cpp produced.
    fn assert_matches_llama(name: &str, dequant: fn(&[u8], usize, &mut [f32]) -> Result<(), AllpacaError>) {
        let (blocks, want) = golden(name);
        let mut got = vec![0.0f32; want.len()];
        dequant(&blocks, want.len(), &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), *w, "{name} weight {i}: {g} vs llama.cpp {}", f32::from_bits(*w));
        }
    }

    #[test]
    fn iq2_xxs_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq2_xxs", dequant_iq2_xxs);
    }

    #[test]
    fn iq2_xs_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq2_xs", dequant_iq2_xs);
    }

    #[test]
    fn iq2_s_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq2_s", dequant_iq2_s);
    }

    #[test]
    fn iq3_xxs_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq3_xxs", dequant_iq3_xxs);
    }

    #[test]
    fn iq1_s_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq1_s", dequant_iq1_s);
    }

    #[test]
    fn iq1_m_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq1_m", dequant_iq1_m);
    }

    /// The existing `IQ3_S` decoder (iq3s.rs) against the same reference.
    #[test]
    fn iq3_s_matches_llama_cpp_bit_for_bit() {
        assert_matches_llama("iq3_s", crate::dequant_iq3_s);
    }

    #[test]
    fn partial_blocks_and_short_buffers_are_refused() {
        let blocks = vec![0u8; IQ2_XXS_BLOCK_BYTES];
        let mut out = vec![0.0f32; IQ_BLOCK_ELEMS];
        assert!(dequant_iq2_xxs(&blocks, 128, &mut out).is_err());
        assert!(dequant_iq2_xxs(&blocks[..10], IQ_BLOCK_ELEMS, &mut out).is_err());
        assert!(dequant_iq1_m(&[0u8; IQ1_M_BLOCK_BYTES], IQ_BLOCK_ELEMS, &mut out[..100]).is_err());
    }
}
