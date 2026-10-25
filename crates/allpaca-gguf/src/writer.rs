use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};

use crate::parser::{GgmlType, GgufValue, GGUF_MAGIC};

pub struct GgufWriteTensor<'a> {
    pub name: String,
    pub dims: Vec<u64>,
    pub data: &'a [f32],
}

/// Minimal GGUF writer used by tests and the synthetic-export path.
pub fn write_gguf(
    path: &Path,
    metadata: &HashMap<String, GgufValue>,
    tensors: &[GgufWriteTensor<'_>],
) -> Result<(), AllpacaError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    let mut keys: Vec<&String> = metadata.keys().collect();
    keys.sort();
    for key in keys {
        write_string(&mut buf, key);
        write_value(&mut buf, &metadata[key])?;
    }

    let mut offset = 0u64;
    for tensor in tensors {
        write_string(&mut buf, &tensor.name);
        buf.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for d in &tensor.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += (tensor.data.len() * 4) as u64;
    }

    let align = 32usize;
    while buf.len() % align != 0 {
        buf.push(0);
    }
    for tensor in tensors {
        for v in tensor.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    let mut file = std::fs::File::create(path).map_err(FatalError::from)?;
    file.write_all(&buf).map_err(FatalError::from)?;
    Ok(())
}

pub struct GgufWriteQuantTensor<'a> {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub bytes: &'a [u8],
}

pub fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    allpaca_cpu::quantize_q8_0(data)
}

pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    allpaca_cpu::quantize_q4_0(data)
}

pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    allpaca_cpu::quantize_q4_k(data)
}

pub fn quantize_q5_k(data: &[f32]) -> Vec<u8> {
    allpaca_cpu::quantize_q5_k(data)
}

pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    allpaca_cpu::quantize_q6_k(data)
}

pub fn quantize_tq2_0(data: &[f32]) -> Vec<u8> {
    allpaca_cpu::quantize_tq2_0(data)
}

pub fn quantize_f16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for v in data {
        out.extend_from_slice(&allpaca_cpu::f32_to_f16(*v).to_le_bytes());
    }
    out
}

pub fn quantize_bf16(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for v in data {
        out.extend_from_slice(&allpaca_cpu::f32_to_bf16(*v).to_le_bytes());
    }
    out
}

/// Write a GGUF that mixes F32 tensors with quantized blocks (tests / tiny models).
pub fn write_gguf_quant(
    path: &Path,
    metadata: &HashMap<String, GgufValue>,
    f32_tensors: &[GgufWriteTensor<'_>],
    quant_tensors: &[GgufWriteQuantTensor<'_>],
) -> Result<(), AllpacaError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC);
    buf.extend_from_slice(&3u32.to_le_bytes());
    let n_tensors = (f32_tensors.len() + quant_tensors.len()) as u64;
    buf.extend_from_slice(&n_tensors.to_le_bytes());
    buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    let mut keys: Vec<&String> = metadata.keys().collect();
    keys.sort();
    for key in keys {
        write_string(&mut buf, key);
        write_value(&mut buf, &metadata[key])?;
    }

    let mut offset = 0u64;
    for tensor in f32_tensors {
        write_string(&mut buf, &tensor.name);
        buf.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for d in &tensor.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&(GgmlType::F32 as u32).to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += (tensor.data.len() * 4) as u64;
    }
    for tensor in quant_tensors {
        write_string(&mut buf, &tensor.name);
        buf.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for d in &tensor.dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&(tensor.ggml_type as u32).to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        offset += tensor.bytes.len() as u64;
    }

    let align = 32usize;
    while buf.len() % align != 0 {
        buf.push(0);
    }
    for tensor in f32_tensors {
        for v in tensor.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    for tensor in quant_tensors {
        buf.extend_from_slice(tensor.bytes);
    }

    let mut file = std::fs::File::create(path).map_err(FatalError::from)?;
    file.write_all(&buf).map_err(FatalError::from)?;
    Ok(())
}

/// One tensor of a file written by [`GgufStreamWriter`].
pub struct GgufStreamTensor {
    pub name: String,
    /// Innermost dimension (the row length) first: ggml order, as written
    /// to the file and as [`crate::GgufFile`] reads them back.
    pub dims: Vec<u64>,
    pub ggml_type_id: u32,
    pub nbytes: usize,
}

/// Writes a GGUF one tensor at a time, for files too large to assemble in
/// memory: the header (metadata and every tensor's place) first, then each
/// tensor's bytes in the order they were declared.
pub struct GgufStreamWriter {
    file: std::io::BufWriter<std::fs::File>,
    sizes: Vec<usize>,
    next: usize,
}

impl GgufStreamWriter {
    const ALIGN: usize = 32;

    pub fn create(
        path: &Path,
        metadata: &HashMap<String, GgufValue>,
        tensors: &[GgufStreamTensor],
    ) -> Result<Self, AllpacaError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        let mut keys: Vec<&String> = metadata.keys().collect();
        keys.sort();
        for key in keys {
            write_string(&mut buf, key);
            write_value(&mut buf, &metadata[key])?;
        }
        let mut offset = 0u64;
        for tensor in tensors {
            write_string(&mut buf, &tensor.name);
            buf.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for d in &tensor.dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&tensor.ggml_type_id.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += tensor.nbytes.next_multiple_of(Self::ALIGN) as u64;
        }
        buf.resize(buf.len().next_multiple_of(Self::ALIGN), 0);
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).map_err(FatalError::from)?);
        file.write_all(&buf).map_err(FatalError::from)?;
        Ok(Self {
            file,
            sizes: tensors.iter().map(|t| t.nbytes).collect(),
            next: 0,
        })
    }

    /// The next declared tensor's bytes.
    pub fn write_tensor(&mut self, bytes: &[u8]) -> Result<(), AllpacaError> {
        match self.sizes.get(self.next) {
            Some(size) if *size == bytes.len() => {}
            other => {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "GGUF tensor {} takes {other:?} bytes, got {}",
                    self.next,
                    bytes.len()
                ))))
            }
        }
        self.file.write_all(bytes).map_err(FatalError::from)?;
        let pad = bytes.len().next_multiple_of(Self::ALIGN) - bytes.len();
        self.file.write_all(&[0u8; Self::ALIGN][..pad]).map_err(FatalError::from)?;
        self.next += 1;
        Ok(())
    }

    /// The next declared tensor's bytes in parts, for a tensor too large to
    /// hold at once: together they must be its size.
    pub fn write_tensor_parts(
        &mut self,
        parts: impl IntoIterator<Item = Result<Vec<u8>, AllpacaError>>,
    ) -> Result<(), AllpacaError> {
        let size = *self.sizes.get(self.next).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig(format!("GGUF has no tensor {} declared", self.next)))
        })?;
        let mut written = 0usize;
        for part in parts {
            let part = part?;
            written += part.len();
            if written > size {
                break;
            }
            self.file.write_all(&part).map_err(FatalError::from)?;
        }
        if written != size {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "GGUF tensor {} takes {size} bytes, got {written}",
                self.next
            ))));
        }
        let pad = size.next_multiple_of(Self::ALIGN) - size;
        self.file.write_all(&[0u8; Self::ALIGN][..pad]).map_err(FatalError::from)?;
        self.next += 1;
        Ok(())
    }

    /// Flush; every declared tensor must have been written.
    pub fn finish(mut self) -> Result<(), AllpacaError> {
        if self.next != self.sizes.len() {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "GGUF has {} tensors declared, {} written",
                self.sizes.len(),
                self.next
            ))));
        }
        self.file.flush().map_err(FatalError::from)?;
        Ok(())
    }
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn write_value(buf: &mut Vec<u8>, value: &GgufValue) -> Result<(), AllpacaError> {
    match value {
        GgufValue::Uint8(v) => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.push(*v);
        }
        GgufValue::Int8(v) => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.push(*v as u8);
        }
        GgufValue::Uint16(v) => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Int16(v) => {
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Uint32(v) => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Int32(v) => {
            buf.extend_from_slice(&5u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Float32(v) => {
            buf.extend_from_slice(&6u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Bool(v) => {
            buf.extend_from_slice(&7u32.to_le_bytes());
            buf.push(u8::from(*v));
        }
        GgufValue::String(v) => {
            buf.extend_from_slice(&8u32.to_le_bytes());
            write_string(buf, v);
        }
        GgufValue::Array(items) => {
            buf.extend_from_slice(&9u32.to_le_bytes());
            let elem_ty = array_elem_type(items)?;
            buf.extend_from_slice(&elem_ty.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_array_elem(buf, item)?;
            }
        }
        GgufValue::Uint64(v) => {
            buf.extend_from_slice(&10u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Int64(v) => {
            buf.extend_from_slice(&11u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        GgufValue::Float64(v) => {
            buf.extend_from_slice(&12u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(())
}

fn array_elem_type(items: &[GgufValue]) -> Result<u32, AllpacaError> {
    Ok(match items.first() {
        Some(GgufValue::String(_)) => 8,
        Some(GgufValue::Float32(_)) => 6,
        Some(GgufValue::Uint32(_)) => 4,
        Some(GgufValue::Int32(_)) => 5,
        Some(GgufValue::Uint64(_)) => 10,
        Some(GgufValue::Int64(_)) => 11,
        None => 8,
        Some(other) => {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "unsupported GGUF array element {other:?}"
            ))))
        }
    })
}

fn write_array_elem(buf: &mut Vec<u8>, value: &GgufValue) -> Result<(), AllpacaError> {
    match value {
        GgufValue::String(v) => write_string(buf, v),
        GgufValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Int32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Int64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        other => {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "unsupported GGUF array element {other:?}"
            ))))
        }
    }
    Ok(())
}
