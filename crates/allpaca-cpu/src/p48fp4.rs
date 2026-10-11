//! Paired-4:8 sparse FP4 weights (MoESQ, `paired48_sparse` "pair-bitmask").
//!
//! Along a row, every 8 weights are 4 adjacent pairs and exactly 2 pairs are
//! kept; the other 4 weights are zero. A kept pair is one byte of two E2M1
//! codes (the lower index in the low nibble), and 32 weights share one
//! scale. A block is 32 weights in 12 bytes:
//!
//! ```text
//! [0..8)   the 2 kept bytes of each of the 4 chunks of 8 weights, in order
//! [8..10)  one mask nibble per chunk (bit i = pair i kept, exactly 2 set);
//!          the low nibble of a byte is the lower chunk
//! [10..12) scale, IEEE half, little endian
//! ```
//!
//! The reference stores the parts as separate tensors (`weight_sparse_packed`,
//! `weight_sparse_mask`, an E4M3 `weight_scale` per 32 weights) and divides
//! by one `weight_global_scale` per tensor. [`p48fp4_interleave`] makes
//! blocks of them with both scales folded into the half (rounded to its 11
//! bits), so a kernel needs nothing but the block.

use allpaca_types::{AllpacaError, FatalError};

use crate::kernels::{f16_to_f32, f32_to_f16};

pub const P48FP4_BLOCK_ELEMS: usize = 32;
pub const P48FP4_BLOCK_BYTES: usize = 12;

/// Magnitude of an E2M1 code's low three bits; bit 3 is the sign.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn invalid(msg: &str) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg.into()))
}

fn e2m1(code: u8) -> f32 {
    let mag = E2M1[usize::from(code & 7)];
    if code & 8 != 0 {
        -mag
    } else {
        mag
    }
}

/// FP8 E4M3 (`float8_e4m3fn`) to f32, the reference's per-32 scale.
pub fn e4m3_to_f32(byte: u8) -> f32 {
    let (exp, man) = (i32::from((byte >> 3) & 0x0f), f32::from(byte & 7));
    let mag = if exp == 0 {
        man * 2f32.powi(-9)
    } else if exp == 15 && byte & 7 == 7 {
        f32::NAN
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    if byte & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

/// The two pairs a mask nibble keeps, lowest first. `None` unless exactly
/// two bits are set.
fn kept_pairs(nibble: u8) -> Option<(usize, usize)> {
    if nibble.count_ones() != 2 {
        return None;
    }
    Some((nibble.trailing_zeros() as usize, 3 - (nibble << 4).leading_zeros() as usize))
}

/// Decode `n` weights from paired-4:8 FP4 blocks.
pub fn dequant_p48fp4(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    let n_blocks = n.div_ceil(P48FP4_BLOCK_ELEMS);
    if out.len() < n || blocks.len() < n_blocks * P48FP4_BLOCK_BYTES {
        return Err(invalid("dequant_p48fp4 buffer too small"));
    }
    out[..n].fill(0.0);
    for (b, block) in blocks.chunks_exact(P48FP4_BLOCK_BYTES).take(n_blocks).enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[10], block[11]]));
        for chunk in 0..4 {
            let nibble = (block[8 + chunk / 2] >> (4 * (chunk % 2))) & 0x0f;
            let (first, second) =
                kept_pairs(nibble).ok_or_else(|| invalid("paired-4:8 mask without exactly two pairs"))?;
            let base = b * P48FP4_BLOCK_ELEMS + chunk * 8;
            for (pair, byte) in [(first, block[chunk * 2]), (second, block[chunk * 2 + 1])] {
                for (i, code) in [byte & 0x0f, byte >> 4].into_iter().enumerate() {
                    if let Some(slot) = out[..n].get_mut(base + pair * 2 + i) {
                        *slot = e2m1(code) * scale;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Blocks from the reference's tensors of one or more rows: `values` holds
/// `K/4` bytes per row, `mask` `K/16` and `scales` `K/32` E4M3 bytes, for a
/// row length `K` that is a multiple of 32. A weight is its code times its
/// E4M3 scale divided by `global_scale`.
pub fn p48fp4_interleave(
    values: &[u8],
    mask: &[u8],
    scales: &[u8],
    global_scale: f32,
) -> Result<Vec<u8>, AllpacaError> {
    let n_blocks = scales.len();
    if values.len() != n_blocks * 8 || mask.len() != n_blocks * 2 {
        return Err(invalid("paired-4:8 parts do not describe the same weights"));
    }
    let mut out = Vec::with_capacity(n_blocks * P48FP4_BLOCK_BYTES);
    for b in 0..n_blocks {
        out.extend_from_slice(&values[b * 8..(b + 1) * 8]);
        out.extend_from_slice(&mask[b * 2..(b + 1) * 2]);
        out.extend_from_slice(&f32_to_f16(e4m3_to_f32(scales[b]) / global_scale).to_le_bytes());
    }
    Ok(out)
}

/// Prune to paired 4:8 (keep the two pairs of each 8 weights with the largest
/// `|a| + |b|`) and quantise, as the reference's `prune_paired_4of8` and
/// `quantize_weight` do. For tests and fixtures: real checkpoints choose
/// their pairs by optimisation.
pub fn quantize_p48fp4(data: &[f32]) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(P48FP4_BLOCK_ELEMS);
    let mut out = vec![0u8; n_blocks * P48FP4_BLOCK_BYTES];
    let at = |i: usize| data.get(i).copied().unwrap_or(0.0);
    for (b, block) in out.chunks_exact_mut(P48FP4_BLOCK_BYTES).enumerate() {
        let start = b * P48FP4_BLOCK_ELEMS;
        let mut keep = [(0usize, 0usize); 4];
        let mut amax = 0.0f32;
        for (chunk, slot) in keep.iter_mut().enumerate() {
            let base = start + chunk * 8;
            let mut order = [0usize, 1, 2, 3];
            let mag = |p: usize| at(base + p * 2).abs() + at(base + p * 2 + 1).abs();
            // Stable: equal pairs keep the lower index, as a descending argsort does.
            order.sort_by(|a, b| mag(*b).total_cmp(&mag(*a)));
            *slot = (order[0].min(order[1]), order[0].max(order[1]));
            for p in [slot.0, slot.1] {
                amax = amax.max(at(base + p * 2).abs()).max(at(base + p * 2 + 1).abs());
            }
        }
        block[10..12].copy_from_slice(&f32_to_f16(amax / 6.0).to_le_bytes());
        let scale = f16_to_f32(u16::from_le_bytes([block[10], block[11]]));
        let code = |v: f32| -> u8 {
            let x = if scale > 0.0 { v / scale } else { 0.0 };
            let a = x.abs();
            let mut c = 0u8;
            for (i, thr) in [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0].into_iter().enumerate() {
                if a > thr {
                    c = i as u8 + 1;
                }
            }
            if x < 0.0 && c != 0 {
                c | 8
            } else {
                c
            }
        };
        for (chunk, (first, second)) in keep.into_iter().enumerate() {
            let base = start + chunk * 8;
            for (k, pair) in [first, second].into_iter().enumerate() {
                block[chunk * 2 + k] = code(at(base + pair * 2)) | (code(at(base + pair * 2 + 1)) << 4);
            }
            block[8 + chunk / 2] |= ((1u8 << first) | (1u8 << second)) << (4 * (chunk % 2));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_decodes_kept_pairs_at_their_positions() {
        // Scale 2.0 (half 0x4000). Chunk 0 keeps pairs 0 and 2, chunk 1 pairs 1 and 3,
        // chunk 2 pairs 0 and 1, chunk 3 pairs 2 and 3.
        let block: [u8; 12] = [
            0x21, 0xb7, // chunk 0: (0.5, 1.0) at pair 0, (6.0, -1.5) at pair 2
            0x05, 0x90, // chunk 1: (3.0, 0) at pair 1, (0, -0.5) at pair 3
            0x11, 0x22, // chunk 2: (0.5, 0.5), (1.0, 1.0)
            0xff, 0x00, // chunk 3: (-6, -6) at pair 2, (0, 0) at pair 3
            0b1010_0101, // chunks 0 and 1
            0b1100_0011, // chunks 2 and 3
            0x00, 0x40,
        ];
        let mut out = [9.0f32; 32];
        dequant_p48fp4(&block, 32, &mut out).unwrap();
        let want: [f32; 32] = [
            1.0, 2.0, 0.0, 0.0, 12.0, -3.0, 0.0, 0.0, //
            0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, -1.0, //
            1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, -12.0, -12.0, 0.0, 0.0,
        ];
        assert_eq!(out, want);
        // A mask that does not keep exactly two pairs is refused.
        let mut bad = block;
        bad[8] = 0b1010_0111;
        assert!(dequant_p48fp4(&bad, 32, &mut out).is_err());
    }

    #[test]
    fn e4m3_scale_bytes_decode_as_fp8() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x3c), 1.5);
        assert_eq!(e4m3_to_f32(0x7e), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 1.0 / 512.0);
        assert_eq!(e4m3_to_f32(0x00), 0.0);
    }

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn quantised_weights_are_paired_4_of_8_and_keep_the_largest_pairs() {
        let w = noise(3 * 64, 7);
        let blocks = quantize_p48fp4(&w);
        assert_eq!(blocks.len(), w.len() / 32 * P48FP4_BLOCK_BYTES);
        let mut got = vec![0.0f32; w.len()];
        dequant_p48fp4(&blocks, w.len(), &mut got).unwrap();
        for (c, (src, dst)) in w.chunks(8).zip(got.chunks(8)).enumerate() {
            let mag = |p: usize| src[p * 2].abs() + src[p * 2 + 1].abs();
            let mut order = [0usize, 1, 2, 3];
            order.sort_by(|a, b| mag(*b).total_cmp(&mag(*a)));
            for p in 0..4 {
                let kept = p == order[0] || p == order[1];
                if !kept {
                    assert_eq!(&dst[p * 2..p * 2 + 2], &[0.0, 0.0], "chunk {c} pair {p}");
                }
            }
        }
        // Kept weights are within half a grid step (at most 1 of 6 parts of
        // the block's largest weight) of the original.
        for (b, (src, dst)) in w.chunks(32).zip(got.chunks(32)).enumerate() {
            let amax = dst.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for (s, d) in src.iter().zip(dst) {
                if *d != 0.0 {
                    assert!((s - d).abs() <= amax / 6.0 + 1e-6, "block {b}: {s} vs {d}");
                }
            }
        }
        // Weights already on the grid come back exactly.
        let again = quantize_p48fp4(&got);
        let mut twice = vec![0.0f32; w.len()];
        dequant_p48fp4(&again, w.len(), &mut twice).unwrap();
        assert_eq!(got, twice);
    }

    #[test]
    fn interleave_folds_both_reference_scales_into_the_block() {
        // One block: every chunk keeps pairs 0 and 3 with codes (1.0, -0.5)
        // and (6.0, 0); E4M3 scale 0x40 = 2.0, global scale 8.
        let values = [0x92u8, 0x07].repeat(4);
        let blocks = p48fp4_interleave(&values, &[0x99, 0x99], &[0x40], 8.0).unwrap();
        assert_eq!(blocks.len(), P48FP4_BLOCK_BYTES);
        let mut out = [0.0f32; 32];
        dequant_p48fp4(&blocks, 32, &mut out).unwrap();
        for chunk in out.chunks(8) {
            assert_eq!(chunk, [0.25, -0.125, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0]);
        }
        assert!(p48fp4_interleave(&values[1..], &[0x99, 0x99], &[0x40], 8.0).is_err());
    }
}
