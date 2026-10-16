//! K-quant encoders that search for each group's scale, as llama.cpp's
//! reference quantisers do (`quantize_row_q{3,4,5,6}_K_ref`): the same
//! blocks for the same weights. The quick encoders in [`crate::kernels`]
//! (`quantize_q4_k` and the like) take each group's range as it is and are
//! for tests and tiny models.

use crate::kernels::{f16_to_f32, Q3_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};

/// Weights per block of every K-quant.
const BLOCK: usize = 256;
/// A group whose largest weight is below this is all zeros.
const GROUP_MAX_EPS: f32 = 1e-15;

/// `v` as a half float, rounded to nearest with halves to even
/// ([`crate::kernels::f32_to_f16`] drops the bits that do not fit).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e < -10 {
        return sign;
    }
    // Subnormal halves keep the leading one and lose `1 - e` more bits.
    let (full, shift, base) = if e <= 0 { (mant | 0x80_0000, (14 - e) as u32, 0) } else { (mant, 13, (e as u32) << 10) };
    let mut out = base + (full >> shift);
    let (rest, half) = (full & ((1 << shift) - 1), 1 << (shift - 1));
    if rest > half || (rest == half && out & 1 == 1) {
        // A carry out of the mantissa lands in the exponent, as it should.
        out += 1;
    }
    sign | out as u16
}

/// Round to nearest, halves to even, as the reference does it.
fn nearest_int(value: f32) -> i32 {
    let bits = (value + 12_582_912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// Signed levels `-nmax..nmax` (stored plus `nmax`) and the scale that
/// fits them best with each weight counted by its square; a few scales
/// around the one that maps the largest weight to `-nmax` are tried.
fn signed_levels(nmax: i32, x: &[f32], levels: &mut [i8]) -> f32 {
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }
    let level = |iscale: f32, v: f32| nearest_int(iscale * v).clamp(-nmax, nmax - 1);
    let sums = |iscale: f32| {
        let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
        for &v in x {
            let l = level(iscale, v) as f32;
            let w = v * v;
            sumlx += w * v * l;
            suml2 += w * l * l;
        }
        (sumlx, suml2)
    };
    let iscale = -(nmax as f32) / max;
    for (l, &v) in levels.iter_mut().zip(x) {
        *l = (level(iscale, v) + nmax) as i8;
    }
    let (sumlx, suml2) = sums(iscale);
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for step in (-9i32..=9).filter(|s| *s != 0) {
        let iscale = -(nmax as f32 + 0.1 * step as f32) / max;
        let (sumlx, suml2) = sums(iscale);
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for (l, &v) in levels.iter_mut().zip(x) {
                *l = (level(iscale, v) + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// The 3-bit variant: levels `-4..4` (stored plus 4), refined one weight
/// at a time against the scale the others imply.
fn signed_levels_refined(x: &[f32], levels: &mut [i8]) -> f32 {
    const NMAX: i32 = 4;
    let (mut max, mut amax) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }
    let iscale = -(NMAX as f32) / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for (l, &v) in levels.iter_mut().zip(x) {
        let level = nearest_int(iscale * v).clamp(-NMAX, NMAX - 1);
        *l = level as i8;
        let w = v * v;
        sumlx += w * v * level as f32;
        suml2 += w * level as f32 * level as f32;
    }
    for _ in 0..5 {
        let mut changed = 0;
        for (l, &v) in levels.iter_mut().zip(x) {
            let w = v * v;
            let old = f32::from(*l);
            let mut slx = sumlx - w * v * old;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * old * old;
                let new = nearest_int(v * sl2 / slx).clamp(-NMAX, NMAX - 1);
                if new != i32::from(*l) {
                    slx += w * v * new as f32;
                    sl2 += w * new as f32 * new as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        *l = new as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        changed += 1;
                    }
                }
            }
        }
        if changed == 0 {
            break;
        }
    }
    levels.iter_mut().for_each(|l| *l += NMAX as i8);
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Levels `0..=nmax`, a scale and a non-negative offset (returned as
/// `(scale, -min)`) for 32 weights: of `steps + 1` scales from
/// `(rmin + nmax) / range` up in steps of `rdelta / range`, the one whose
/// least-squares scale and offset leave the least weighted error.
fn offset_levels(nmax: i32, x: &[f32], weights: &[f32], levels: &mut [u8], rmin: f32, rdelta: f32, steps: i32) -> (f32, f32) {
    let (mut min, mut max) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for (&v, &w) in x.iter().zip(weights).skip(1) {
        min = min.min(v);
        max = max.max(v);
        sum_w += w;
        sum_x += w * v;
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        levels.fill(0);
        return (0.0, -min);
    }
    let iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for ((l, &v), &w) in levels.iter_mut().zip(x).zip(weights) {
        *l = nearest_int(iscale * (v - min)).clamp(0, nmax) as u8;
        let diff = scale * f32::from(*l) + min - v;
        best_error += w * (diff * diff);
    }
    let mut trial = [0u8; 32];
    for step in 0..=steps {
        // The offset moves with every better fit, and the range with it.
        let iscale = (rmin + rdelta * step as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for ((t, &v), &w) in trial.iter_mut().zip(x).zip(weights) {
            let l = nearest_int(iscale * (v - min)).clamp(0, nmax);
            *t = l as u8;
            let l = l as f32;
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * v;
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut error = 0.0f32;
            for ((&t, &v), &w) in trial.iter().zip(x).zip(weights) {
                let diff = this_scale * f32::from(t) + this_min - v;
                error += w * (diff * diff);
            }
            if error < best_error {
                levels[..x.len()].copy_from_slice(&trial[..x.len()]);
                best_error = error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// The 6-bit scale and offset of group `j` in a Q4_K or Q5_K block.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Scales, offsets and levels `0..=nmax` of a Q4_K or Q5_K block; `head`
/// gets `d`, `dmin` and the 12 packed scale bytes.
fn offset_block(x: &[f32], nmax: i32, rmin: f32, steps: i32, head: &mut [u8], levels: &mut [u8; BLOCK]) {
    let (mut scales, mut mins) = ([0.0f32; 8], [0.0f32; 8]);
    let (mut max_scale, mut max_min) = (0.0f32, 0.0f32);
    for j in 0..8 {
        let group = &x[32 * j..32 * j + 32];
        let sum_x2: f32 = group.iter().fold(0.0, |s, v| s + v * v);
        let av_x = (sum_x2 / 32.0).sqrt();
        let mut weights = [0.0f32; 32];
        for (w, v) in weights.iter_mut().zip(group) {
            *w = av_x + v.abs();
        }
        (scales[j], mins[j]) = offset_levels(nmax, group, &weights, &mut levels[32 * j..32 * j + 32], rmin, 0.1, steps);
        max_scale = max_scale.max(scales[j]);
        max_min = max_min.max(mins[j]);
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let packed = &mut head[4..16];
    packed.fill(0);
    for j in 0..8 {
        let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
        let lm = (nearest_int(inv_min * mins[j]) as u8).min(63);
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
    let (d, dmin) = (f32_to_f16(max_scale / 63.0), f32_to_f16(max_min / 63.0));
    head[0..2].copy_from_slice(&d.to_le_bytes());
    head[2..4].copy_from_slice(&dmin.to_le_bytes());
    // The levels again, for the scales and offsets as they were stored.
    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &head[4..16]);
        let scale = f16_to_f32(d) * f32::from(sc);
        if scale == 0.0 {
            continue;
        }
        let offset = f16_to_f32(dmin) * f32::from(m);
        for (l, &v) in levels[32 * j..32 * j + 32].iter_mut().zip(&x[32 * j..]) {
            *l = nearest_int((v + offset) / scale).clamp(0, nmax) as u8;
        }
    }
}

fn q4_k_block(x: &[f32], out: &mut [u8]) {
    let mut levels = [0u8; BLOCK];
    let (head, qs) = out.split_at_mut(16);
    offset_block(x, 15, -1.0, 20, head, &mut levels);
    for (half, q) in qs.chunks_mut(32).enumerate() {
        for (l, q) in q.iter_mut().enumerate() {
            *q = levels[64 * half + l] | (levels[64 * half + l + 32] << 4);
        }
    }
}

fn q5_k_block(x: &[f32], out: &mut [u8]) {
    let mut levels = [0u8; BLOCK];
    let (head, rest) = out.split_at_mut(16);
    offset_block(x, 31, -0.5, 15, head, &mut levels);
    let (qh, qs) = rest.split_at_mut(32);
    qh.fill(0);
    for (half, q) in qs.chunks_mut(32).enumerate() {
        for (j, q) in q.iter_mut().enumerate() {
            let (mut l1, mut l2) = (levels[64 * half + j], levels[64 * half + j + 32]);
            if l1 > 15 {
                l1 -= 16;
                qh[j] |= 1 << (2 * half);
            }
            if l2 > 15 {
                l2 -= 16;
                qh[j] |= 2 << (2 * half);
            }
            *q = l1 | (l2 << 4);
        }
    }
}

fn q6_k_block(x: &[f32], out: &mut [u8]) {
    let mut levels = [0i8; BLOCK];
    let mut scales = [0.0f32; 16];
    let (mut max_scale, mut max_abs) = (0.0f32, 0.0f32);
    for (j, scale) in scales.iter_mut().enumerate() {
        *scale = signed_levels(32, &x[16 * j..16 * j + 16], &mut levels[16 * j..16 * j + 16]);
        if scale.abs() > max_abs {
            max_abs = scale.abs();
            max_scale = *scale;
        }
    }
    out.fill(0);
    if max_abs < GROUP_MAX_EPS {
        return;
    }
    let iscale = -128.0 / max_scale;
    let d = f32_to_f16(1.0 / iscale);
    let (ql, rest) = out.split_at_mut(128);
    let (qh, tail) = rest.split_at_mut(64);
    for (j, scale) in scales.iter().enumerate() {
        let stored = nearest_int(iscale * scale).min(127) as i8;
        tail[j] = stored as u8;
        let step = f16_to_f32(d) * f32::from(stored);
        if step == 0.0 {
            continue;
        }
        for (l, &v) in levels[16 * j..16 * j + 16].iter_mut().zip(&x[16 * j..]) {
            *l = (nearest_int(v / step).clamp(-32, 31) + 32) as i8;
        }
    }
    tail[16..18].copy_from_slice(&d.to_le_bytes());
    let levels = levels.map(|l| l as u8);
    for part in 0..2 {
        let (at, ql, qh) = (128 * part, &mut ql[64 * part..64 * part + 64], &mut qh[32 * part..32 * part + 32]);
        for l in 0..32 {
            let (q1, q2, q3, q4) = (levels[at + l], levels[at + l + 32], levels[at + l + 64], levels[at + l + 96]);
            ql[l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            ql[l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            qh[l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
}

fn q3_k_block(x: &[f32], out: &mut [u8]) {
    let mut levels = [0i8; BLOCK];
    let mut scales = [0.0f32; 16];
    let (mut max_scale, mut amax) = (0.0f32, 0.0f32);
    for (j, scale) in scales.iter_mut().enumerate() {
        *scale = signed_levels_refined(&x[16 * j..16 * j + 16], &mut levels[16 * j..16 * j + 16]);
        if scale.abs() > amax {
            amax = scale.abs();
            max_scale = *scale;
        }
    }
    out.fill(0);
    let (hmask, rest) = out.split_at_mut(32);
    let (qs, tail) = rest.split_at_mut(64);
    let (packed, d_bytes) = tail.split_at_mut(12);
    let mut d = f32_to_f16(0.0);
    if max_scale != 0.0 {
        let iscale = -32.0 / max_scale;
        for (j, scale) in scales.iter().enumerate() {
            let l = (nearest_int(iscale * scale).clamp(-32, 31) + 32) as u8;
            if j < 8 {
                packed[j] = l & 0xF;
            } else {
                packed[j - 8] |= (l & 0xF) << 4;
            }
            packed[j % 4 + 8] |= (l >> 4) << (2 * (j / 4));
        }
        d = f32_to_f16(1.0 / iscale);
    }
    d_bytes.copy_from_slice(&d.to_le_bytes());
    // The levels again, for the scales as they were stored.
    for j in 0..16 {
        let low = if j < 8 { packed[j] & 0xF } else { packed[j - 8] >> 4 };
        let stored = i32::from(low | (((packed[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
        let step = f16_to_f32(d) * stored as f32;
        if step == 0.0 {
            continue;
        }
        for (l, &v) in levels[16 * j..16 * j + 16].iter_mut().zip(&x[16 * j..]) {
            *l = (nearest_int(v / step).clamp(-4, 3) + 4) as i8;
        }
    }
    // The high bit of weight j goes to bit j / 32 of mask byte j % 32.
    for (j, l) in levels.iter_mut().enumerate() {
        if *l > 3 {
            hmask[j % 32] |= 1 << (j / 32);
            *l -= 4;
        }
    }
    let levels = levels.map(|l| l as u8);
    for part in 0..2 {
        let at = 128 * part;
        for l in 0..32 {
            qs[32 * part + l] =
                levels[at + l] | (levels[at + l + 32] << 2) | (levels[at + l + 64] << 4) | (levels[at + l + 96] << 6);
        }
    }
}

/// Encode `data` block by block on every core; a last block that is not
/// full is padded with zeros.
fn encode(data: &[f32], block_bytes: usize, block: fn(&[f32], &mut [u8])) -> Vec<u8> {
    let n_blocks = data.len().div_ceil(BLOCK);
    let mut out = vec![0u8; n_blocks * block_bytes];
    let run = |first: usize, out: &mut [u8]| {
        for (i, bytes) in out.chunks_mut(block_bytes).enumerate() {
            let at = (first + i) * BLOCK;
            match data.get(at..at + BLOCK) {
                Some(x) => block(x, bytes),
                None => {
                    let mut padded = [0.0f32; BLOCK];
                    padded[..data.len() - at].copy_from_slice(&data[at..]);
                    block(&padded, bytes);
                }
            }
        }
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(n_blocks / 64).max(1);
    if threads == 1 {
        run(0, &mut out);
        return out;
    }
    let per = n_blocks.div_ceil(threads);
    std::thread::scope(|scope| {
        for (i, part) in out.chunks_mut(per * block_bytes).enumerate() {
            let run = &run;
            scope.spawn(move || run(i * per, part));
        }
    });
    out
}

/// `Q3_K` blocks of `data` (3.4375 bits per weight).
pub fn encode_q3_k(data: &[f32]) -> Vec<u8> {
    encode(data, Q3_K_BLOCK_BYTES, q3_k_block)
}

/// `Q4_K` blocks of `data` (4.5 bits per weight).
pub fn encode_q4_k(data: &[f32]) -> Vec<u8> {
    encode(data, Q4_K_BLOCK_BYTES, q4_k_block)
}

/// `Q5_K` blocks of `data` (5.5 bits per weight).
pub fn encode_q5_k(data: &[f32]) -> Vec<u8> {
    encode(data, Q5_K_BLOCK_BYTES, q5_k_block)
}

/// `Q6_K` blocks of `data` (6.5625 bits per weight).
pub fn encode_q6_k(data: &[f32]) -> Vec<u8> {
    encode(data, Q6_K_BLOCK_BYTES, q6_k_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 512 seeded weights whose 32-weight groups differ in size, with one
    /// group of zeros and half a group of one value.
    fn weights() -> Vec<f32> {
        let mut s = 0x2545_F491u32;
        let mut x: Vec<f32> = (0..512)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let t = (s >> 8) as f32 / 16_777_216.0 - 0.5;
                t * (0.5 * (1 + (i / 32) % 5) as f32)
            })
            .collect();
        x[11 * 32..12 * 32].fill(0.0);
        x[14 * 32..14 * 32 + 16].fill(0.375);
        x
    }

    /// What llama.cpp's reference encoders give for [`weights`], as hex,
    /// one `name bytes` line per type (built from `ggml-quants.c` without
    /// fused multiply-adds).
    fn golden(name: &str) -> Vec<u8> {
        let line = include_str!("kquant_golden.txt")
            .lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(' '))
            .unwrap_or_else(|| panic!("no golden blocks for {name}"));
        (0..line.len() / 2).map(|i| u8::from_str_radix(&line[2 * i..2 * i + 2], 16).unwrap()).collect()
    }

    fn first_difference(got: &[u8], want: &[u8]) -> Option<usize> {
        assert_eq!(got.len(), want.len());
        got.iter().zip(want).position(|(a, b)| a != b)
    }

    #[test]
    fn half_floats_round_to_nearest_even() {
        // 1 + 2^-11 is halfway between two halves: it goes to the even one.
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0 + 1.0 / 65536.0), 0x3c01);
        // Rounding up the largest mantissa carries into the exponent.
        assert_eq!(f32_to_f16(2.0 - 1.0 / 4096.0), 0x4000);
        // Subnormals: 2^-24 is the smallest; half of it is a tie to zero.
        assert_eq!(f32_to_f16(2.0f32.powi(-24)), 1);
        assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0);
        assert_eq!(f32_to_f16(-1.5 * 2.0f32.powi(-24)), 0x8002);
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        for v in [0.0f32, -0.0, 1.0, -2.5, 0.1, 65504.0] {
            assert!((f16_to_f32(f32_to_f16(v)) - v).abs() <= v.abs() / 2048.0, "{v}");
        }
    }

    #[test]
    fn blocks_are_the_reference_encoders_blocks() {
        let x = weights();
        type Encoder = fn(&[f32]) -> Vec<u8>;
        for (name, encode) in [
            ("q3_k", encode_q3_k as Encoder),
            ("q4_k", encode_q4_k as Encoder),
            ("q5_k", encode_q5_k as Encoder),
            ("q6_k", encode_q6_k as Encoder),
        ] {
            let (got, want) = (encode(&x), golden(name));
            if let Some(at) = first_difference(&got, &want) {
                let n = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                panic!("{name}: {n} bytes differ, first at {at}: got {:02x?}, want {:02x?}", &got[at..at + 8], &want[at..at + 8]);
            }
        }
    }

    /// The search pays: decoded with the crate's own decoders, the blocks
    /// are closer to the weights than the quick encoders' blocks.
    #[test]
    fn searched_blocks_are_closer_than_the_quick_encoders() {
        let x = weights();
        let error = |decoded: &[f32]| x.iter().zip(decoded).map(|(a, b)| f64::from(a - b).powi(2)).sum::<f64>();
        let mut y = vec![0.0f32; x.len()];
        crate::kernels::dequant_q4_k(&encode_q4_k(&x), x.len(), &mut y).unwrap();
        let searched = error(&y);
        crate::kernels::dequant_q4_k(&crate::kernels::quantize_q4_k(&x), x.len(), &mut y).unwrap();
        assert!(searched < 0.9 * error(&y), "{searched} vs {}", error(&y));
        crate::kernels::dequant_q3_k(&encode_q3_k(&x), x.len(), &mut y).unwrap();
        let energy: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
        assert!(error(&y) < 0.02 * energy, "3 bits: {} of {energy}", error(&y));
    }

    #[test]
    fn many_blocks_on_several_threads_match_one_at_a_time() {
        let x: Vec<f32> = weights().iter().cycle().take(200 * 256 + 100).enumerate().map(|(i, v)| v * (1 + i % 3) as f32).collect();
        let all = encode_q5_k(&x);
        assert_eq!(all.len(), 201 * Q5_K_BLOCK_BYTES);
        for b in [0usize, 77, 199] {
            let alone = encode_q5_k(&x[b * 256..(b + 1) * 256]);
            assert_eq!(&all[b * Q5_K_BLOCK_BYTES..(b + 1) * Q5_K_BLOCK_BYTES], alone.as_slice(), "block {b}");
        }
        // The last, short block is the block of its weights followed by zeros.
        let mut padded = x[200 * 256..].to_vec();
        padded.resize(256, 0.0);
        assert_eq!(&all[200 * Q5_K_BLOCK_BYTES..], encode_q5_k(&padded).as_slice());
    }
}
