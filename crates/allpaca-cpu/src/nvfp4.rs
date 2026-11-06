//! ggml NVFP4 (type 40): 64 weights per 36-byte block, four UE4M3 scales (one
//! per 16-weight sub-block) then 32 bytes of packed E2M1 nibbles. Inside a
//! sub-block the low nibbles are weights 0..8 and the high nibbles 8..16.
//! A tensor can also carry an F32 `.scale` tensor; that is applied by the
//! caller (see `GgufFile::tensor_f32`), never here.

use crate::kernels::{f16_to_f32, f32_to_f16, rocmfp4_scale, KVALUES_MXFP4};
use allpaca_types::{AllpacaError, FatalError};

pub const NVFP4_BLOCK_ELEMS: usize = 64;
pub const NVFP4_BLOCK_BYTES: usize = 36;
const SUB_ELEMS: usize = 16;

/// Dequantize `n` weights of NVFP4 blocks, matching ggml's `dequantize_row_nvfp4`.
pub fn dequant_nvfp4(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_nvfp4 output too small".into(),
        )));
    }
    let n_blocks = n.div_ceil(NVFP4_BLOCK_ELEMS);
    if blocks.len() < n_blocks.saturating_mul(NVFP4_BLOCK_BYTES) {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "dequant_nvfp4 truncated blocks".into(),
        )));
    }
    for b in 0..n_blocks {
        let blk = &blocks[b * NVFP4_BLOCK_BYTES..(b + 1) * NVFP4_BLOCK_BYTES];
        let (scales, qs) = blk.split_at(4);
        for (s, &sb) in scales.iter().enumerate() {
            let d = rocmfp4_scale(sb);
            let base = b * NVFP4_BLOCK_ELEMS + s * SUB_ELEMS;
            for j in 0..SUB_ELEMS / 2 {
                let packed = qs[s * (SUB_ELEMS / 2) + j];
                let (i0, i1) = (base + j, base + j + SUB_ELEMS / 2);
                if i0 < n {
                    out[i0] = f32::from(KVALUES_MXFP4[usize::from(packed & 0x0f)]) * d;
                }
                if i1 < n {
                    out[i1] = f32::from(KVALUES_MXFP4[usize::from(packed >> 4)]) * d;
                }
            }
        }
    }
    Ok(())
}

/// Device layout for NVFP4 GEMV: 32 weights per 20-byte block, `qs[16]` (low
/// nibbles = weights 0..15, high = 16..31, the type-100 shape) then two f16
/// scales (low half, high half). The UE4M3 scale and the tensor's `.scale`
/// are multiplied into the f16 scale here, so the kernel needs neither.
pub const NVFP4_DEV_BLOCK_ELEMS: usize = 32;
pub const NVFP4_DEV_BLOCK_BYTES: usize = 20;

/// Re-pack `n` NVFP4 weights (`n` a whole number of 64-weight blocks) into the
/// device layout, folding `tensor_scale` into the scales.
pub fn repack_nvfp4_dev(blocks: &[u8], n: usize, tensor_scale: f32) -> Result<Vec<u8>, AllpacaError> {
    if n % NVFP4_BLOCK_ELEMS != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "repack_nvfp4_dev needs whole 64-weight blocks".into(),
        )));
    }
    let n_blocks = n / NVFP4_BLOCK_ELEMS;
    if blocks.len() < n_blocks * NVFP4_BLOCK_BYTES {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "repack_nvfp4_dev truncated blocks".into(),
        )));
    }
    let mut out = Vec::with_capacity(n_blocks * 2 * NVFP4_DEV_BLOCK_BYTES);
    for b in 0..n_blocks {
        let blk = &blocks[b * NVFP4_BLOCK_BYTES..(b + 1) * NVFP4_BLOCK_BYTES];
        let (scales, qs) = blk.split_at(4);
        // Two device blocks per NVFP4 block: sub-blocks (0,1) and (2,3).
        for pair in 0..2 {
            let mut dev = [0u8; NVFP4_DEV_BLOCK_BYTES];
            for half in 0..2 {
                let s = pair * 2 + half;
                for j in 0..SUB_ELEMS / 2 {
                    let packed = qs[s * (SUB_ELEMS / 2) + j];
                    // weight j (low) and weight j+8 (high) of this sub-block.
                    for (w, code) in [(j, packed & 0x0f), (j + SUB_ELEMS / 2, packed >> 4)] {
                        dev[w] |= code << (4 * half);
                    }
                }
                let d = f32_to_f16(rocmfp4_scale(scales[s]) * tensor_scale);
                dev[16 + 2 * half..18 + 2 * half].copy_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&dev);
        }
    }
    Ok(out)
}

/// Reference decode of the device layout (what the GEMV kernel computes with).
pub fn dequant_nvfp4_dev(blocks: &[u8], n: usize, out: &mut [f32]) {
    for (b, blk) in blocks.chunks_exact(NVFP4_DEV_BLOCK_BYTES).enumerate() {
        let d = [
            f16_to_f32(u16::from_le_bytes([blk[16], blk[17]])),
            f16_to_f32(u16::from_le_bytes([blk[18], blk[19]])),
        ];
        for w in 0..NVFP4_DEV_BLOCK_ELEMS {
            let i = b * NVFP4_DEV_BLOCK_ELEMS + w;
            if i >= n {
                return;
            }
            let packed = blk[w % 16];
            let code = if w < 16 { packed & 0x0f } else { packed >> 4 };
            out[i] = f32::from(KVALUES_MXFP4[usize::from(code)]) * d[w / 16];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sub_blocks_have_their_own_scale_and_nibble_halves() {
        let mut blk = [0u8; NVFP4_BLOCK_BYTES];
        blk[0] = 0x40; // sub-block 0: scale 1.0
        blk[1] = 0x38; // sub-block 1: scale 0.5
        blk[2] = 0x44; // sub-block 2: scale 1.5
        blk[3] = 0x7f; // sub-block 3: NaN code, decodes to 0
        // Sub-block 0, byte 0: weight 0 = code 7 (12), weight 8 = code 15 (-12).
        blk[4] = 0x07 | (0x0f << 4);
        // Sub-block 1, byte 0 (qs index 8): weight 16 = code 5 (6), weight 24 = code 2 (2).
        blk[4 + 8] = 0x05 | (0x02 << 4);
        // Sub-block 2, byte 3 (qs index 19): weight 32+3 = code 4 (4).
        blk[4 + 19] = 0x04;
        // Sub-block 3 holds code 7 everywhere, but its scale is NaN.
        for q in blk[4 + 24..].iter_mut() {
            *q = 0x77;
        }
        let mut out = vec![1.0f32; NVFP4_BLOCK_ELEMS];
        dequant_nvfp4(&blk, NVFP4_BLOCK_ELEMS, &mut out).unwrap();
        assert_eq!(out[0], 12.0);
        assert_eq!(out[8], -12.0);
        assert_eq!(out[16], 3.0);
        assert_eq!(out[24], 1.0);
        assert_eq!(out[35], 6.0);
        assert!(out[48..].iter().all(|&v| v == 0.0));
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn a_short_tail_and_a_short_buffer() {
        let blk = [0u8; NVFP4_BLOCK_BYTES];
        let mut out = vec![0.0f32; 64];
        assert!(dequant_nvfp4(&blk[..35], 64, &mut out).is_err());
        assert!(dequant_nvfp4(&blk, 64, &mut out[..63]).is_err());
        // 70 weights need two blocks.
        let mut out = vec![0.0f32; 70];
        assert!(dequant_nvfp4(&blk, 70, &mut out).is_err());
    }
    #[test]
    fn dev_repack_decodes_like_the_nvfp4_blocks_times_the_tensor_scale() {
        // Two 64-weight blocks with varied codes and scales.
        let mut blocks = vec![0u8; 2 * NVFP4_BLOCK_BYTES];
        for (i, b) in blocks.iter_mut().enumerate() {
            *b = (i * 37 + 11) as u8;
        }
        for blk in blocks.chunks_exact_mut(NVFP4_BLOCK_BYTES) {
            // Plausible finite scales (0x30..0x50) in the four scale bytes.
            for (i, s) in blk[..4].iter_mut().enumerate() {
                *s = 0x30 + (i as u8) * 5;
            }
        }
        let n = 128;
        let mut want = vec![0.0f32; n];
        dequant_nvfp4(&blocks, n, &mut want).unwrap();
        let ts = 3.1e-4f32;
        let dev = repack_nvfp4_dev(&blocks, n, ts).unwrap();
        assert_eq!(dev.len(), n / NVFP4_DEV_BLOCK_ELEMS * NVFP4_DEV_BLOCK_BYTES);
        let mut got = vec![0.0f32; n];
        dequant_nvfp4_dev(&dev, n, &mut got);
        for i in 0..n {
            let w = want[i] * ts;
            assert!((got[i] - w).abs() <= w.abs() * 1.0e-3 + 1.0e-12, "idx {i}: {} vs {w}", got[i]);
        }
        assert!(repack_nvfp4_dev(&blocks, 100, 1.0).is_err());
    }
}
