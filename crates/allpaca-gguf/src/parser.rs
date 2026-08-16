use std::collections::HashMap;
use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};
use memmap2::Mmap;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufType {
    fn from_u32(v: u32) -> Result<Self, AllpacaError> {
        Ok(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            other => {
                return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                    "unknown GGUF type {other}"
                ))))
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => u32::try_from(*v).ok(),
            Self::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int32(v) if *v >= 0 => Some(*v as u64),
            Self::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.as_str()?.to_string());
                }
                Some(out)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Self::Float32(v) => out.push(*v),
                        Self::Float64(v) => out.push(*v as f32),
                        _ => return None,
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Other = 255,
}

impl GgmlType {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            _ => Self::Other,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub ggml_type_id: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize,
    pub alignment: u32,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, AllpacaError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!("mmap failed: {e}")))
        })?;
        Self::parse(mmap)
    }

    pub fn parse(mmap: Mmap) -> Result<Self, AllpacaError> {
        let mut cur = Cursor::new(&mmap);
        let magic = cur.read_bytes(4)?;
        if magic != GGUF_MAGIC {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(
                "magic is not GGUF".into(),
            )));
        }
        let version = cur.read_u32()?;
        if version < 2 {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "unsupported GGUF version {version}"
            ))));
        }
        let tensor_count = cur.read_u64()?;
        let kv_count = cur.read_u64()?;

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = cur.read_string()?;
            let ty = GgufType::from_u32(cur.read_u32()?)?;
            let value = cur.read_value(ty)?;
            metadata.insert(key, value);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name = cur.read_string()?;
            let n_dims = cur.read_u32()? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cur.read_u64()?);
            }
            let ggml_type_id = cur.read_u32()?;
            let offset = cur.read_u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type: GgmlType::from_u32(ggml_type_id),
                ggml_type_id,
                offset,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u32)
            .unwrap_or(32);
        let tensor_data_offset = align_up(cur.pos, alignment as usize);

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            tensor_data_offset,
            alignment,
        })
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(GgufValue::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// F32 tensor bytes. Other dtypes return Fatal — dequant is a follow-up.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        if info.ggml_type != GgmlType::F32 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "tensor {name} is {:?} (only F32 materialization in MVP; TODO dequant)",
                info.ggml_type
            ))));
        }
        let n = info.n_elements() as usize;
        let start = self.tensor_data_offset + info.offset as usize;
        let bytes = n.saturating_mul(4);
        if start.saturating_add(bytes) > self.mmap.len() {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "tensor {name} out of range"
            ))));
        }
        let mut out = vec![0.0f32; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let off = start + i * 4;
            *slot = f32::from_le_bytes([
                self.mmap[off],
                self.mmap[off + 1],
                self.mmap[off + 2],
                self.mmap[off + 3],
            ]);
        }
        Ok(out)
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(GgufValue::as_u32)
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], AllpacaError> {
        if self.remaining() < n {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(
                "unexpected EOF".into(),
            )));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, AllpacaError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, AllpacaError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, AllpacaError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, AllpacaError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn read_i8(&mut self) -> Result<i8, AllpacaError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16, AllpacaError> {
        Ok(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> Result<i32, AllpacaError> {
        Ok(self.read_u32()? as i32)
    }

    fn read_i64(&mut self) -> Result<i64, AllpacaError> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, AllpacaError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, AllpacaError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_string(&mut self) -> Result<String, AllpacaError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidUtf8))
    }

    fn read_value(&mut self, ty: GgufType) -> Result<GgufValue, AllpacaError> {
        Ok(match ty {
            GgufType::Uint8 => GgufValue::Uint8(self.read_u8()?),
            GgufType::Int8 => GgufValue::Int8(self.read_i8()?),
            GgufType::Uint16 => GgufValue::Uint16(self.read_u16()?),
            GgufType::Int16 => GgufValue::Int16(self.read_i16()?),
            GgufType::Uint32 => GgufValue::Uint32(self.read_u32()?),
            GgufType::Int32 => GgufValue::Int32(self.read_i32()?),
            GgufType::Float32 => GgufValue::Float32(self.read_f32()?),
            GgufType::Bool => GgufValue::Bool(self.read_u8()? != 0),
            GgufType::String => GgufValue::String(self.read_string()?),
            GgufType::Uint64 => GgufValue::Uint64(self.read_u64()?),
            GgufType::Int64 => GgufValue::Int64(self.read_i64()?),
            GgufType::Float64 => GgufValue::Float64(self.read_f64()?),
            GgufType::Array => {
                let elem = GgufType::from_u32(self.read_u32()?)?;
                let count = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.read_value(elem)?);
                }
                GgufValue::Array(items)
            }
        })
    }
}

fn align_up(value: usize, align: usize) -> usize {
    if align == 0 {
        return value;
    }
    value.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_gguf, GgufWriteTensor};
    use std::collections::HashMap;

    fn sample_meta() -> HashMap<String, GgufValue> {
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta
    }

    fn write_typed_gguf(path: &std::path::Path, ggml_type_id: u32, nbytes: usize) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let name = "quant.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&ggml_type_id.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend(std::iter::repeat_n(0u8, nbytes));
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn roundtrip_header_and_f32_tensor() {
        let dir = std::env::temp_dir();
        let path = dir.join("allpaca_gguf_roundtrip.gguf");
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        write_gguf(
            &path,
            &sample_meta(),
            &[GgufWriteTensor {
                name: "token_embd.weight".into(),
                dims: vec![2, 2],
                data: &data,
            }],
        )
        .unwrap();

        let file = GgufFile::open(&path).unwrap();
        assert_eq!(file.version, 3);
        assert_eq!(file.architecture(), Some("llama"));
        assert_eq!(file.tensors.len(), 1);
        let info = file.tensor("token_embd.weight").expect("tensor info");
        assert_eq!(info.dims, vec![2, 2]);
        assert_eq!(info.ggml_type, GgmlType::F32);
        let loaded = file.tensor_f32("token_embd.weight").unwrap();
        assert_eq!(loaded, data);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q4_and_q8_tensor_load_is_fatal() {
        let dir = std::env::temp_dir();
        for (name, type_id, expected) in [
            ("allpaca_gguf_q4.gguf", 2u32, GgmlType::Q4_0),
            ("allpaca_gguf_q8.gguf", 8u32, GgmlType::Q8_0),
        ] {
            let path = dir.join(name);
            write_typed_gguf(&path, type_id, 32);
            let file = GgufFile::open(&path).unwrap();
            let info = file.tensor("quant.weight").expect("header lists tensor");
            assert_eq!(info.ggml_type, expected);
            let err = file.tensor_f32("quant.weight").unwrap_err();
            assert!(
                err.is_fatal(),
                "{expected:?} load must be Fatal, got {err:?}"
            );
            let _ = std::fs::remove_file(path);
        }
    }
}
