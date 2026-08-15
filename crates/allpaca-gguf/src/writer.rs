use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};

use crate::parser::{GgufValue, GGUF_MAGIC};

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
        other => {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "unsupported GGUF array element {other:?}"
            ))))
        }
    }
    Ok(())
}
