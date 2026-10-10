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
