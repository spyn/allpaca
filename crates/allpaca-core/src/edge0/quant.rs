//! Affine 4-bit (group 64) expert weight dequant for Edge0 checkpoints.

/// Packed affine-4 tensor as stored in Edge0 safetensors.
#[derive(Debug, Clone)]
pub struct Affine4Tensor {
    /// Packed nibbles: two 4-bit values per byte (low nibble first).
    pub qweight: Vec<u8>,
    /// Per-group scale.
    pub scales: Vec<f32>,
    /// Per-group zero-point (same layout as scales).
    pub zeros: Vec<f32>,
    pub group_size: usize,
    /// Logical float element count after dequant.
    pub n_elements: usize,
}

/// Dequantize affine-4 packed weights to f32.
///
/// `out[i] = scales[g] * (q[i] as f32 - zeros[g])` with `g = i / group_size`.
pub fn dequant_affine4(t: &Affine4Tensor) -> Vec<f32> {
    let gsz = t.group_size.max(1);
    let mut out = vec![0.0f32; t.n_elements];
    for i in 0..t.n_elements {
        let byte = t.qweight.get(i / 2).copied().unwrap_or(0);
        let nibble = if i % 2 == 0 {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        let g = i / gsz;
        let scale = t.scales.get(g).copied().unwrap_or(1.0);
        let zero = t.zeros.get(g).copied().unwrap_or(0.0);
        out[i] = scale * (nibble as f32 - zero);
    }
    out
}

/// Pack f32 weights into affine-4 for tests / synthetic Edge0 experts.
pub fn quant_affine4(data: &[f32], group_size: usize) -> Affine4Tensor {
    let gsz = group_size.max(1);
    let n = data.len();
    let n_groups = n.div_ceil(gsz);
    let mut scales = vec![1.0f32; n_groups];
    let mut zeros = vec![0.0f32; n_groups];
    let mut qweight = vec![0u8; n.div_ceil(2)];
    for g in 0..n_groups {
        let start = g * gsz;
        let end = (start + gsz).min(n);
        let slice = &data[start..end];
        let min = slice.iter().copied().fold(f32::INFINITY, f32::min);
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = ((max - min) / 15.0).max(1e-8);
        let zero = (min / scale).round().clamp(0.0, 15.0);
        scales[g] = scale;
        zeros[g] = zero;
        for (j, &v) in slice.iter().enumerate() {
            let i = start + j;
            let q = ((v / scale) + zero).round().clamp(0.0, 15.0) as u8;
            let byte_i = i / 2;
            if i % 2 == 0 {
                qweight[byte_i] = (qweight[byte_i] & 0xf0) | (q & 0x0f);
            } else {
                qweight[byte_i] = (qweight[byte_i] & 0x0f) | ((q & 0x0f) << 4);
            }
        }
    }
    Affine4Tensor {
        qweight,
        scales,
        zeros,
        group_size: gsz,
        n_elements: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_affine4_near_original() {
        let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let packed = quant_affine4(&data, 64);
        let got = dequant_affine4(&packed);
        assert_eq!(got.len(), data.len());
        let mut max_err = 0.0f32;
        for (a, b) in data.iter().zip(got.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.55, "max_err={max_err}");
    }
}
