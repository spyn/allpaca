//! ROCmFPX fork weight types (ROCmFPX/ROCmFPX `ggml/rocmfpx/rocmfpx.c`):
//! 32 weights per block, small integer codes times UE4M3 scale bytes.
//!
//! | id  | fork name       | block bytes | codes                                   | scales     |
//! |-----|-----------------|-------------|-----------------------------------------|------------|
//! | 102 | `Q6_0_ROCMFPX`  | 26          | 6-bit sign-magnitude, -0 means -32       | one per 16 |
//! | 103 | `Q8_0_ROCMFPX`  | 33          | int8                                     | one        |
//! | 104 | `Q3_0_ROCMFPX`  | 14          | 3-bit: 0, ±1, ±2, ±4                     | one per 16 |
//! | 109 | `Q5_0_ROCMFPX`  | 22          | 5-bit sign-magnitude, -0 means -16       | one per 16 |
//! | 110 | `Q7_0_ROCMFPX`  | 30          | 7-bit sign-magnitude, -0 means -64       | one per 16 |
//! | 111 | `Q2_0_ROCMFPX`  | 10          | 2-bit S40 codebook {-4, -1, +1, +4}      | one per 16 |
//!
//! Scale bytes come after the codes. A scale byte above 0x7e (0x7f or a set
//! sign bit) decodes as zero, as in the fork. Type 107 is not here: the fork
//! wrote two different layouts under that id and refuses it.
//!
//! Each decoder computes `code as f32 * scale` like the fork's
//! `rocmfpx_dequantize_row_*`, so results are bit-identical to it (checked
//! against `rocmfpx_golden.txt`).

use allpaca_types::{AllpacaError, FatalError};

/// Weights per block, the same for every ROCmFPX type.
pub const ROCMFPX_BLOCK_ELEMS: usize = 32;
/// Type 103 (`Q8_0_ROCMFPX`): 32 int8 codes, then one scale byte.
pub const ROCMFPX_Q8_BLOCK_BYTES: usize = 33;
/// Type 102 (`Q6_0_ROCMFPX`): 24 code bytes, then two scale bytes.
pub const ROCMFPX_Q6_BLOCK_BYTES: usize = 26;
/// Type 104 (`Q3_0_ROCMFPX`): 12 code bytes, then two scale bytes.
pub const ROCMFPX_Q3_BLOCK_BYTES: usize = 14;
/// Type 109 (`Q5_0_ROCMFPX`): 20 code bytes, then two scale bytes.
pub const ROCMFPX_Q5_BLOCK_BYTES: usize = 22;
/// Type 110 (`Q7_0_ROCMFPX`): 28 code bytes, then two scale bytes.
pub const ROCMFPX_Q7_BLOCK_BYTES: usize = 30;
/// Type 111 (`Q2_0_ROCMFPX`): 8 code bytes, then two scale bytes.
pub const ROCMFPX_Q2_BLOCK_BYTES: usize = 10;

/// The fork's UE4M3 scale: `mant * 2^-10` for exponent 0, otherwise
/// `(8 + mant) * 2^(exp - 11)`; bytes above 0x7e are zero.
pub fn rocmfpx_scale(e: u8) -> f32 {
    if e > 0x7e {
        return 0.0;
    }
    let (exp, man) = (u32::from(e >> 3), u32::from(e & 7));
    if exp == 0 {
        return man as f32 * (1.0 / 1024.0);
    }
    f32::from_bits(((exp + 119) << 23) | (man << 20))
}

/// Fail unless `n` is whole blocks and both buffers hold them.
fn check(name: &str, blocks: &[u8], block_bytes: usize, n: usize, out: &[f32]) -> Result<(), AllpacaError> {
    if n % ROCMFPX_BLOCK_ELEMS != 0 || out.len() < n || blocks.len() < n / ROCMFPX_BLOCK_ELEMS * block_bytes {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "dequant_{name}: {n} weights are not whole blocks, or a buffer is too small"
        ))));
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of ROCmFPX type 103
/// (`Q8_0_ROCMFPX`, 8.25 bits per weight).
///
/// # Errors
/// When `n` is not a multiple of 32 or a buffer is too small.
pub fn dequant_rocmfpx_q8(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    check("rocmfpx_q8", blocks, ROCMFPX_Q8_BLOCK_BYTES, n, out)?;
    for (block, y) in blocks.chunks_exact(ROCMFPX_Q8_BLOCK_BYTES).zip(out[..n].chunks_exact_mut(ROCMFPX_BLOCK_ELEMS)) {
        let scale = rocmfpx_scale(block[32]);
        for (y, &q) in y.iter_mut().zip(&block[..32]) {
            *y = f32::from(q as i8) * scale;
        }
    }
    Ok(())
}

/// Dequantize `n` weights (whole blocks) of ROCmFPX type 102
/// (`Q6_0_ROCMFPX`, 6.5 bits per weight).
///
/// # Errors
/// When `n` is not a multiple of 32 or a buffer is too small.
pub fn dequant_rocmfpx_q6(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    dequant_two_scale("rocmfpx_q6", blocks, n, out, 6, sign_magnitude::<6>)
}

/// Dequantize `n` weights (whole blocks) of ROCmFPX type 104
/// (`Q3_0_ROCMFPX`, 3.5 bits per weight).
///
/// # Errors
/// When `n` is not a multiple of 32 or a buffer is too small.
pub fn dequant_rocmfpx_q3(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    dequant_two_scale("rocmfpx_q3", blocks, n, out, 3, |code| {
        let mag = [0, 1, 2, 4][usize::from(code & 3)];
        if code & 4 != 0 { -mag } else { mag }
    })
}

/// Dequantize `n` weights (whole blocks) of ROCmFPX type 109
/// (`Q5_0_ROCMFPX`, 5.5 bits per weight).
///
/// # Errors
/// When `n` is not a multiple of 32 or a buffer is too small.
pub fn dequant_rocmfpx_q5(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    dequant_two_scale("rocmfpx_q5", blocks, n, out, 5, sign_magnitude::<5>)
}

/// Dequantize `n` weights (whole blocks) of ROCmFPX type 110
/// (`Q7_0_ROCMFPX`, 7.5 bits per weight).
///
/// # Errors
/// When `n` is not a multiple of 32 or a buffer is too small.
pub fn dequant_rocmfpx_q7(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    dequant_two_scale("rocmfpx_q7", blocks, n, out, 7, sign_magnitude::<7>)
}

/// Dequantize `n` weights (whole blocks) of ROCmFPX type 111
/// (`Q2_0_ROCMFPX`, 2.5 bits per weight).
///
/// # Errors
/// When `n` is not a multiple of 32 or a buffer is too small.
pub fn dequant_rocmfpx_q2(blocks: &[u8], n: usize, out: &mut [f32]) -> Result<(), AllpacaError> {
    // The fork's frozen S40 code order.
    dequant_two_scale("rocmfpx_q2", blocks, n, out, 2, |code| [-4, -1, 1, 4][usize::from(code & 3)])
}

/// A `BITS`-wide sign-magnitude code as the fork reads it for types 102,
/// 109 and 110: a set sign bit with magnitude 0 is the most negative value.
fn sign_magnitude<const BITS: u32>(code: u8) -> i32 {
    let sign = 1i32 << (BITS - 1);
    let mag = i32::from(code) & (sign - 1);
    if i32::from(code) & sign == 0 {
        mag
    } else if mag == 0 {
        -sign
    } else {
        -mag
    }
}

/// Shared decoder of the two-scale types: `32 * bits / 8` bytes of codes as
/// one little-endian bit stream (code `i` at bit `i * bits`), then one scale
/// byte for weights 0..16 and one for 16..32.
fn dequant_two_scale(
    name: &str,
    blocks: &[u8],
    n: usize,
    out: &mut [f32],
    bits: usize,
    decode: fn(u8) -> i32,
) -> Result<(), AllpacaError> {
    let payload = ROCMFPX_BLOCK_ELEMS * bits / 8;
    check(name, blocks, payload + 2, n, out)?;
    let mask = (1u16 << bits) - 1;
    for (block, y) in blocks.chunks_exact(payload + 2).zip(out[..n].chunks_exact_mut(ROCMFPX_BLOCK_ELEMS)) {
        let scales = [rocmfpx_scale(block[payload]), rocmfpx_scale(block[payload + 1])];
        for (i, y) in y.iter_mut().enumerate() {
            let (byte, shift) = (i * bits / 8, i * bits % 8);
            let mut word = u16::from(block[byte]);
            if shift + bits > 8 {
                word |= u16::from(block[byte + 1]) << 8;
            }
            let code = ((word >> shift) & mask) as u8;
            *y = decode(code) as f32 * scales[i / 16];
        }
    }
    Ok(())
}

/// Re-pack type 103 blocks as ggml `Q8_0` blocks (half scale, then the same
/// 32 int8 codes). Lossless: every UE4M3 scale is exact in f16, so
/// `dequant_q8_0` of the result gives the same f32 bits as
/// [`dequant_rocmfpx_q8`].
///
/// # Errors
/// When `blocks` is not whole 33-byte blocks.
pub fn repack_rocmfpx_q8_as_q8_0(blocks: &[u8]) -> Result<Vec<u8>, AllpacaError> {
    if blocks.len() % ROCMFPX_Q8_BLOCK_BYTES != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
            "repack_rocmfpx_q8_as_q8_0: {} bytes are not whole {ROCMFPX_Q8_BLOCK_BYTES}-byte blocks",
            blocks.len()
        ))));
    }
    let mut out = Vec::with_capacity(blocks.len() / ROCMFPX_Q8_BLOCK_BYTES * crate::Q8_0_BLOCK_BYTES);
    for block in blocks.chunks_exact(ROCMFPX_Q8_BLOCK_BYTES) {
        out.extend_from_slice(&crate::f32_to_f16(rocmfpx_scale(block[32])).to_le_bytes());
        out.extend_from_slice(&block[..32]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    /// Blocks and the fork's dequantised f32 bits for type `id` from
    /// `rocmfpx_golden.txt`: random codes and scale bytes (some invalid),
    /// decoded by the fork's own `rocmfpx_dequantize_row_*`
    /// (`rocmfpx_golden_gen.c`, built against ROCmFPX/ROCmFPX 120141f).
    fn golden(id: u32) -> (Vec<u8>, Vec<u32>) {
        let line = include_str!("rocmfpx_golden.txt")
            .lines()
            .find(|l| l.split(' ').next() == Some(id.to_string().as_str()))
            .unwrap_or_else(|| panic!("no golden blocks for type {id}"));
        let mut parts = line.split(' ').skip(1);
        let blocks = unhex(parts.next().unwrap());
        let bits = unhex(parts.next().unwrap())
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (blocks, bits)
    }

    /// Decode the golden blocks of type `id` and require the fork's exact
    /// f32 bits.
    fn assert_matches_fork(id: u32, dequant: fn(&[u8], usize, &mut [f32]) -> Result<(), AllpacaError>) {
        let (blocks, want) = golden(id);
        let mut got = vec![0.0f32; want.len()];
        dequant(&blocks, want.len(), &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), *w, "type {id} weight {i}: {g} vs ROCmFPX {}", f32::from_bits(*w));
        }
    }

    #[test]
    fn q8_rocmfpx_103_matches_the_fork_bit_for_bit() {
        assert_matches_fork(103, dequant_rocmfpx_q8);
    }

    #[test]
    fn q6_rocmfpx_102_matches_the_fork_bit_for_bit() {
        assert_matches_fork(102, dequant_rocmfpx_q6);
    }

    #[test]
    fn q3_rocmfpx_104_matches_the_fork_bit_for_bit() {
        assert_matches_fork(104, dequant_rocmfpx_q3);
    }

    #[test]
    fn q5_rocmfpx_109_matches_the_fork_bit_for_bit() {
        assert_matches_fork(109, dequant_rocmfpx_q5);
    }

    #[test]
    fn q7_rocmfpx_110_matches_the_fork_bit_for_bit() {
        assert_matches_fork(110, dequant_rocmfpx_q7);
    }

    #[test]
    fn q2_rocmfpx_111_matches_the_fork_bit_for_bit() {
        assert_matches_fork(111, dequant_rocmfpx_q2);
    }

    /// Every value any ROCmFPX type can decode to (int8-range code times
    /// any scale byte) survives f32 -> f16 -> f32 bit for bit, so weights
    /// staged as F16 on a GPU equal the CPU dequant exactly.
    #[test]
    fn every_rocmfpx_weight_value_is_exact_in_f16() {
        for e in 0..=255u8 {
            let scale = rocmfpx_scale(e);
            for code in -128i32..=127 {
                let v = code as f32 * scale;
                let back = crate::f16_to_f32(crate::f32_to_f16(v));
                assert_eq!(back.to_bits(), v.to_bits(), "code {code} scale byte {e:#04x}");
            }
        }
    }

    #[test]
    fn q8_rocmfpx_repacks_to_q8_0_without_changing_a_bit() {
        let (blocks, want) = golden(103);
        let q8 = repack_rocmfpx_q8_as_q8_0(&blocks).unwrap();
        assert_eq!(q8.len(), blocks.len() / ROCMFPX_Q8_BLOCK_BYTES * crate::Q8_0_BLOCK_BYTES);
        let mut got = vec![0.0f32; want.len()];
        crate::dequant_q8_0(&q8, want.len(), &mut got).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), *w, "weight {i}");
        }
        assert!(repack_rocmfpx_q8_as_q8_0(&blocks[..40]).is_err());
    }
}
