//! ggml NVFP4 (type 40): 64 weights per 36-byte block, four UE4M3 scales (one
//! per 16-weight sub-block) then 32 bytes of packed E2M1 nibbles. Inside a
//! sub-block the low nibbles are weights 0..8 and the high nibbles 8..16.
//! A tensor can also carry an F32 `.scale` tensor; that is applied by the
//! caller (see `GgufFile::tensor_f32`), never here.

use crate::kernels::{rocmfp4_scale, KVALUES_MXFP4};
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
}
