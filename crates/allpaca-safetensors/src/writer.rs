use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};
use serde_json::{json, Map, Value};

use crate::parser::{f32_to_bf16, f32_to_f16, SafetensorsDtype};

pub struct SafetensorsWriteTensor<'a> {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: SafetensorsDtype,
    pub data: &'a [f32],
}

pub fn write_safetensors(
    path: &Path,
    tensors: &[SafetensorsWriteTensor<'_>],
) -> Result<(), AllpacaError> {
    let mut payloads = Vec::with_capacity(tensors.len());
    let mut offset = 0u64;
    let mut header = Map::new();
    for t in tensors {
        let n: usize = t.dims.iter().map(|d| *d as usize).product();
        if t.data.len() != n {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors write {}: expected {n} elems, got {}",
                t.name,
                t.data.len()
            ))));
        }
        let bytes = match t.dtype {
            SafetensorsDtype::F32 => t
                .data
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            SafetensorsDtype::F16 => t
                .data
                .iter()
                .flat_map(|v| f32_to_f16(*v).to_le_bytes())
                .collect::<Vec<u8>>(),
            SafetensorsDtype::BF16 => t
                .data
                .iter()
                .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
                .collect::<Vec<u8>>(),
        };
        let end = offset + bytes.len() as u64;
        header.insert(
            t.name.clone(),
            json!({
                "dtype": t.dtype.as_str(),
                "shape": t.dims,
                "data_offsets": [offset, end],
            }),
        );
        payloads.push(bytes);
        offset = end;
    }
    let header_json = Value::Object(header);
    let header_bytes = serde_json::to_vec(&header_json).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "SafeTensors write header: {e}"
        )))
    })?;
    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset as usize);
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for p in payloads {
        out.extend_from_slice(&p);
    }
    std::fs::write(path, out)?;
    Ok(())
}
