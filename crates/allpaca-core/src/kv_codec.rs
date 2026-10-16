//! How prompt-cache checkpoints hold their keys and values.
//!
//! `F32` keeps them exactly, so a cache hit continues as the full prefill
//! would. `F16`, `Q8` and `Q4` trade that for size: Q8 and Q4 quantise each
//! row in blocks of [`BLOCK`] values, Q8 symmetric with one scale per block,
//! Q4 with a scale and a minimum per block.

use allpaca_types::KvCodec;

use crate::anchors::KvRow;

/// Values per quantisation block.
pub(crate) const BLOCK: usize = 32;

/// Rows of keys and values, each row `k` then `v`, `width` values apiece.
#[derive(Debug, Clone)]
pub(crate) struct EncodedRows {
    codec: KvCodec,
    width: usize,
    rows: usize,
    bytes: Vec<u8>,
}

impl EncodedRows {
    pub(crate) fn encode(codec: KvCodec, rows: &[KvRow]) -> Self {
        let width = rows.first().map_or(0, |r| r.k.len());
        let mut bytes = Vec::with_capacity(rows.len() * 2 * vector_bytes(codec, width));
        for row in rows {
            debug_assert!(row.k.len() == width && row.v.len() == width);
            encode_vector(codec, &row.k, &mut bytes);
            encode_vector(codec, &row.v, &mut bytes);
        }
        Self {
            codec,
            width,
            rows: rows.len(),
            bytes,
        }
    }

    pub(crate) fn decode(&self) -> Vec<KvRow> {
        let n = vector_bytes(self.codec, self.width);
        (0..self.rows)
            .map(|r| {
                let row = &self.bytes[r * 2 * n..(r + 1) * 2 * n];
                KvRow {
                    k: decode_vector(self.codec, self.width, &row[..n]),
                    v: decode_vector(self.codec, self.width, &row[n..]),
                }
            })
            .collect()
    }

    /// The first `rows` rows.
    pub(crate) fn prefix(&self, rows: usize) -> Self {
        let rows = rows.min(self.rows);
        let n = 2 * vector_bytes(self.codec, self.width);
        Self {
            codec: self.codec,
            width: self.width,
            rows,
            bytes: self.bytes[..rows * n].to_vec(),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Encoded size.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Append to a checkpoint file.
    pub(crate) fn write_to(&self, out: &mut Vec<u8>) {
        out.push(match self.codec {
            KvCodec::F32 => 0,
            KvCodec::F16 => 1,
            KvCodec::Q8 => 2,
            KvCodec::Q4 => 3,
        });
        out.extend_from_slice(&(self.width as u32).to_le_bytes());
        out.extend_from_slice(&(self.rows as u32).to_le_bytes());
        out.extend_from_slice(&(self.bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.bytes);
    }

    /// Read what [`Self::write_to`] wrote; `None` when it does not add up.
    pub(crate) fn read_from(r: &mut Reader<'_>) -> Option<Self> {
        let codec = match r.u8()? {
            0 => KvCodec::F32,
            1 => KvCodec::F16,
            2 => KvCodec::Q8,
            3 => KvCodec::Q4,
            _ => return None,
        };
        let width = r.u32()? as usize;
        let rows = r.u32()? as usize;
        let n = usize::try_from(r.u64()?).ok()?;
        if n != rows.checked_mul(2 * vector_bytes(codec, width))? {
            return None;
        }
        Some(Self {
            codec,
            width,
            rows,
            bytes: r.take(n)?.to_vec(),
        })
    }
}

/// Append a length-prefixed run of floats.
pub(crate) fn write_f32s(out: &mut Vec<u8>, values: &[f32]) {
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// [`write_f32s`] for something that may be absent.
pub(crate) fn write_opt_f32s(out: &mut Vec<u8>, values: Option<&[f32]>) {
    out.push(values.is_some() as u8);
    if let Some(values) = values {
        write_f32s(out, values);
    }
}

/// Little-endian reads from a byte slice.
pub(crate) struct Reader<'a>(pub(crate) &'a [u8]);

impl<'a> Reader<'a> {
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.0.len() {
            return None;
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(head)
    }

    pub(crate) fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|b| b[0])
    }

    pub(crate) fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub(crate) fn u64(&mut self) -> Option<u64> {
        self.take(8).map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }

    /// What [`write_f32s`] wrote.
    pub(crate) fn f32s(&mut self) -> Option<Vec<f32>> {
        let n = usize::try_from(self.u64()?).ok()?;
        let raw = self.take(n.checked_mul(4)?)?;
        Some(
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
    }

    /// What [`write_opt_f32s`] wrote.
    pub(crate) fn opt_f32s(&mut self) -> Option<Option<Vec<f32>>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(self.f32s()?)),
            _ => None,
        }
    }
}

/// Encoded size of one vector of `width` values.
fn vector_bytes(codec: KvCodec, width: usize) -> usize {
    let blocks = width.div_ceil(BLOCK);
    match codec {
        KvCodec::F32 => width * 4,
        KvCodec::F16 => width * 2,
        KvCodec::Q8 => blocks * 2 + width,
        KvCodec::Q4 => blocks * 4 + (0..blocks).map(|b| block_len(width, b).div_ceil(2)).sum::<usize>(),
    }
}

fn block_len(width: usize, block: usize) -> usize {
    (width - block * BLOCK).min(BLOCK)
}

fn encode_vector(codec: KvCodec, x: &[f32], out: &mut Vec<u8>) {
    match codec {
        KvCodec::F32 => x.iter().for_each(|v| out.extend_from_slice(&v.to_le_bytes())),
        KvCodec::F16 => x
            .iter()
            .for_each(|&v| out.extend_from_slice(&f32_to_f16(v).to_le_bytes())),
        KvCodec::Q8 => {
            for block in x.chunks(BLOCK) {
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = f16_to_f32(f32_to_f16(amax / 127.0));
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                out.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
                out.extend(
                    block
                        .iter()
                        .map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8 as u8),
                );
            }
        }
        KvCodec::Q4 => {
            for block in x.chunks(BLOCK) {
                let lo = block.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let min = f16_to_f32(f32_to_f16(lo));
                let scale = f16_to_f32(f32_to_f16((hi - min).max(0.0) / 15.0));
                let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                out.extend_from_slice(&f32_to_f16(scale).to_le_bytes());
                out.extend_from_slice(&f32_to_f16(min).to_le_bytes());
                let q = |v: f32| ((v - min) * inv).round().clamp(0.0, 15.0) as u8;
                for pair in block.chunks(2) {
                    out.push(q(pair[0]) | pair.get(1).map_or(0, |&v| q(v) << 4));
                }
            }
        }
    }
}

fn decode_vector(codec: KvCodec, width: usize, bytes: &[u8]) -> Vec<f32> {
    let half = |b: &[u8]| f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    let mut x = Vec::with_capacity(width);
    match codec {
        KvCodec::F32 => x.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        ),
        KvCodec::F16 => x.extend(bytes.chunks_exact(2).map(half)),
        KvCodec::Q8 => {
            let mut at = 0;
            for b in 0..width.div_ceil(BLOCK) {
                let n = block_len(width, b);
                let scale = half(&bytes[at..]);
                x.extend(bytes[at + 2..at + 2 + n].iter().map(|&q| q as i8 as f32 * scale));
                at += 2 + n;
            }
        }
        KvCodec::Q4 => {
            let mut at = 0;
            for b in 0..width.div_ceil(BLOCK) {
                let n = block_len(width, b);
                let scale = half(&bytes[at..]);
                let min = half(&bytes[at + 2..]);
                let packed = &bytes[at + 4..at + 4 + n.div_ceil(2)];
                x.extend((0..n).map(|i| {
                    let q = (packed[i / 2] >> (4 * (i % 2))) & 0x0f;
                    min + q as f32 * scale
                }));
                at += 4 + n.div_ceil(2);
            }
        }
    }
    x
}

/// IEEE half from f32, rounding to nearest even.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let up = rest > mid || (rest == mid && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let up = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    sign | (half + up as u32) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, width: usize) -> Vec<KvRow> {
        (0..n)
            .map(|r| KvRow {
                k: (0..width)
                    .map(|i| ((r * 31 + i * 7) % 97) as f32 / 13.0 - 3.5)
                    .collect(),
                v: (0..width)
                    .map(|i| ((r * 17 + i * 5) % 89) as f32 / 29.0 - 1.25)
                    .collect(),
            })
            .collect()
    }

    fn max_error(a: &[KvRow], b: &[KvRow]) -> f32 {
        a.iter()
            .zip(b)
            .flat_map(|(x, y)| {
                x.k.iter()
                    .zip(&y.k)
                    .chain(x.v.iter().zip(&y.v))
                    .map(|(p, q)| (p - q).abs())
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn f32_checkpoints_restore_exactly() {
        let src = rows(5, 70);
        let enc = EncodedRows::encode(KvCodec::F32, &src);
        assert_eq!(enc.rows(), 5);
        assert_eq!(enc.decode(), src);
    }

    #[test]
    fn smaller_codecs_stay_close_and_shrink() {
        // 70 values per row: two whole blocks and a short one.
        let src = rows(9, 70);
        let f32_bytes = EncodedRows::encode(KvCodec::F32, &src).bytes();
        let mut last = f32_bytes;
        for (codec, tolerance) in [
            (KvCodec::F16, 0.004),
            (KvCodec::Q8, 0.04),
            (KvCodec::Q4, 0.6),
        ] {
            let enc = EncodedRows::encode(codec, &src);
            let back = enc.decode();
            assert_eq!(back.len(), src.len(), "{codec:?}");
            assert_eq!(back[0].k.len(), 70, "{codec:?}");
            let err = max_error(&src, &back);
            assert!(err <= tolerance, "{codec:?} error {err}");
            assert!(enc.bytes() < last, "{codec:?} {} bytes", enc.bytes());
            last = enc.bytes();
        }
        assert!(last * 5 < f32_bytes, "Q4 {last} vs F32 {f32_bytes}");
    }

    #[test]
    fn zeros_and_flat_blocks_quantise_to_themselves() {
        let zeros = vec![KvRow {
            k: vec![0.0; 40],
            v: vec![0.0; 40],
        }];
        for codec in [KvCodec::Q8, KvCodec::Q4] {
            assert_eq!(EncodedRows::encode(codec, &zeros).decode(), zeros, "{codec:?}");
        }
        let flat = vec![KvRow {
            k: vec![2.5; 40],
            v: vec![-1.0; 40],
        }];
        assert_eq!(EncodedRows::encode(KvCodec::Q4, &flat).decode(), flat);
    }

    #[test]
    fn layers_without_attention_keep_their_row_count() {
        let src = vec![KvRow { k: vec![], v: vec![] }; 3];
        for codec in [KvCodec::F32, KvCodec::Q4] {
            let enc = EncodedRows::encode(codec, &src);
            assert_eq!(enc.decode(), src, "{codec:?}");
            assert_eq!(enc.prefix(2).decode().len(), 2, "{codec:?}");
        }
    }

    #[test]
    fn prefixes_slice_without_reencoding() {
        let src = rows(6, 33);
        for codec in [KvCodec::F32, KvCodec::F16, KvCodec::Q8, KvCodec::Q4] {
            let enc = EncodedRows::encode(codec, &src);
            let head = enc.prefix(4);
            assert_eq!(head.rows(), 4);
            assert_eq!(head.decode(), enc.decode()[..4].to_vec(), "{codec:?}");
        }
    }
}
