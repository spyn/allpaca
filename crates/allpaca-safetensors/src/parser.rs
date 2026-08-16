use std::collections::HashMap;
use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};
use serde::Deserialize;

/// Reject unbounded header allocations from a corrupt file.
pub const SAFETENSORS_MAX_HEADER: u64 = 100 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetensorsDtype {
    F32,
    F16,
}

impl SafetensorsDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
        }
    }

    pub fn from_header(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            _ => None,
        }
    }

    pub fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
}

#[derive(Debug, Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

pub struct SafetensorsFile {
    bytes: Vec<u8>,
    pub tensors: HashMap<String, TensorInfo>,
    pub metadata: HashMap<String, String>,
    pub data_offset: usize,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self, AllpacaError> {
        let bytes = std::fs::read(path).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "SafeTensors {}: {e}",
                path.display()
            )))
        })?;
        Self::parse(bytes)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<Self, AllpacaError> {
        if bytes.len() < 8 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "corrupt SafeTensors: file shorter than header size".into(),
            )));
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes"));
        if header_len == 0 || header_len > SAFETENSORS_MAX_HEADER {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "corrupt SafeTensors: invalid header size {header_len}"
            ))));
        }
        let header_end = 8usize.saturating_add(header_len as usize);
        if header_end > bytes.len() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "corrupt SafeTensors: header overruns file".into(),
            )));
        }
        let header_src = std::str::from_utf8(&bytes[8..header_end]).map_err(|_| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "corrupt SafeTensors: header is not UTF-8 JSON".into(),
            ))
        })?;
        let raw: HashMap<String, serde_json::Value> = serde_json::from_str(header_src).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "corrupt SafeTensors: header JSON: {e}"
            )))
        })?;

        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                if let Some(map) = value.as_object() {
                    for (k, v) in map {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let entry: HeaderEntry = serde_json::from_value(value).map_err(|e| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "corrupt SafeTensors: tensor {name}: {e}"
                )))
            })?;
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dtype: entry.dtype,
                    shape: entry.shape,
                    data_offsets: entry.data_offsets,
                },
            );
        }

        Ok(Self {
            bytes,
            tensors,
            metadata,
            data_offset: header_end,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        let dtype = SafetensorsDtype::from_header(&info.dtype).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "unsupported SafeTensors dtype {} for {name}",
                info.dtype
            )))
        })?;
        let n: usize = info.shape.iter().product();
        let (begin, end) = (info.data_offsets[0] as usize, info.data_offsets[1] as usize);
        let start = self.data_offset.saturating_add(begin);
        let stop = self.data_offset.saturating_add(end);
        if stop > self.bytes.len() || start > stop {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "corrupt SafeTensors: tensor {name} data out of range"
            ))));
        }
        let payload = &self.bytes[start..stop];
        match dtype {
            SafetensorsDtype::F32 => {
                if payload.len() != n.saturating_mul(4) {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "corrupt SafeTensors: tensor {name} F32 size mismatch"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = i * 4;
                    *slot = f32::from_le_bytes([
                        payload[off],
                        payload[off + 1],
                        payload[off + 2],
                        payload[off + 3],
                    ]);
                }
                Ok(out)
            }
            SafetensorsDtype::F16 => {
                if payload.len() != n.saturating_mul(2) {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "corrupt SafeTensors: tensor {name} F16 size mismatch"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([payload[off], payload[off + 1]]);
                    *slot = f16_to_f32(bits);
                }
                Ok(out)
            }
        }
    }
}

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x3ff;
    let bits = if exp == 0 {
        if frac == 0 {
            (sign as u32) << 31
        } else {
            let mut e = -14i32;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | ((m as u32) << 13)
        }
    } else if exp == 31 {
        ((sign as u32) << 31) | (0xff << 23) | ((frac as u32) << 13)
    } else {
        ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((frac as u32) << 13)
    };
    f32::from_bits(bits)
}

pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) as u16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | ((frac >> 13) as u16);
    }
    let exp16 = exp - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let m = (frac | 0x800_000) >> (1 - exp16);
        return sign | ((m >> 13) as u16);
    }
    sign | ((exp16 as u16) << 10) | ((frac >> 13) as u16)
}
