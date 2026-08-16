use allpaca_types::{AllpacaError, FatalError};

pub(crate) struct ProtoCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoCursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn read_varint(&mut self) -> Result<u64, AllpacaError> {
        let mut result = 0u64;
        let mut shift = 0;
        for _ in 0..10 {
            if self.pos >= self.data.len() {
                return Err(onnx_err("truncated protobuf varint"));
            }
            let b = self.data[self.pos];
            self.pos += 1;
            result |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err(onnx_err("protobuf varint overflow"))
    }

    pub fn read_tag(&mut self) -> Result<(u32, u32), AllpacaError> {
        let v = self.read_varint()?;
        Ok(((v >> 3) as u32, (v & 7) as u32))
    }

    pub fn read_bytes(&mut self) -> Result<&'a [u8], AllpacaError> {
        let len = self.read_varint()? as usize;
        if self.pos.saturating_add(len) > self.data.len() {
            return Err(onnx_err("truncated protobuf bytes"));
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.data[start..self.pos])
    }

    pub fn read_f32_le(&mut self) -> Result<f32, AllpacaError> {
        if self.remaining() < 4 {
            return Err(onnx_err("truncated protobuf float"));
        }
        let bits = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(f32::from_bits(bits))
    }

    pub fn skip_field(&mut self, wire: u32) -> Result<(), AllpacaError> {
        match wire {
            0 => {
                self.read_varint()?;
            }
            1 => {
                if self.remaining() < 8 {
                    return Err(onnx_err("truncated protobuf fixed64"));
                }
                self.pos += 8;
            }
            2 => {
                self.read_bytes()?;
            }
            5 => {
                if self.remaining() < 4 {
                    return Err(onnx_err("truncated protobuf fixed32"));
                }
                self.pos += 4;
            }
            other => return Err(onnx_err(&format!("unsupported protobuf wire type {other}"))),
        }
        Ok(())
    }
}

pub(crate) fn onnx_err(msg: &str) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(format!("corrupt ONNX: {msg}")))
}

pub(crate) fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

pub(crate) fn write_tag(out: &mut Vec<u8>, field: u32, wire: u32) {
    write_varint(out, u64::from((field << 3) | wire));
}

pub(crate) fn write_len_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    write_tag(out, field, 2);
    write_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

pub(crate) fn write_string(out: &mut Vec<u8>, field: u32, s: &str) {
    write_len_bytes(out, field, s.as_bytes());
}

pub(crate) fn write_int64(out: &mut Vec<u8>, field: u32, v: i64) {
    write_tag(out, field, 0);
    write_varint(out, v as u64);
}
