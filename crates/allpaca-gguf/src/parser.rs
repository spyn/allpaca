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

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            Self::Uint32(v) => Some(*v as f32),
            Self::Uint64(v) => Some(*v as f32),
            Self::Int32(v) => Some(*v as f32),
            Self::Int64(v) => Some(*v as f32),
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
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    /// AMD ROCmFP4 dual-scale (ggml_type_id=100): host bytes loadable; F32 dequant not yet.
    RocmFp4 = 100,
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
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            100 => Self::RocmFp4,
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

    pub fn nbytes(&self) -> usize {
        let n = self.n_elements() as usize;
        match self.ggml_type {
            GgmlType::F32 => n.saturating_mul(4),
            GgmlType::F16 => n.saturating_mul(2),
            GgmlType::Q4_0 => n
                .div_ceil(allpaca_cpu::Q4_0_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q4_0_BLOCK_BYTES),
            GgmlType::Q8_0 => n
                .div_ceil(allpaca_cpu::Q8_0_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q8_0_BLOCK_BYTES),
            GgmlType::Q4_K => n
                .div_ceil(allpaca_cpu::Q4_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q4_K_BLOCK_BYTES),
            GgmlType::Q5_K => n
                .div_ceil(allpaca_cpu::Q5_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q5_K_BLOCK_BYTES),
            GgmlType::Q6_K => n
                .div_ceil(allpaca_cpu::Q6_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q6_K_BLOCK_BYTES),
            GgmlType::RocmFp4 => n
                .div_ceil(allpaca_cpu::ROCMFP4_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::ROCMFP4_BLOCK_BYTES),
            _ => n.saturating_mul(4),
        }
    }
}

/// Per-block role in a hybrid GGUF (SSM / attention / MoE FFN).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerKind {
    Ssm,
    Attention,
    DenseFfn,
    /// Layer has MoE expert tensors (may also be Attention or Ssm for the sequence block).
    Moe,
}

/// One layer in a hybrid map: sequence block kind plus whether MoE FFN is present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridLayerInfo {
    pub index: usize,
    pub block: HybridLayerKind,
    pub moe: bool,
}

/// Header-only GGUF probe: architecture, layer mix, and quant type ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufCompatReport {
    pub architecture: String,
    pub n_layers: usize,
    pub ssm_layers: usize,
    pub attention_layers: usize,
    pub moe_layers: usize,
    /// Per-layer map (`attn` / `ssm` / dense); `moe` is a separate flag per layer.
    pub layers: Vec<HybridLayerInfo>,
    pub ggml_type_ids: Vec<u32>,
    /// Named Fatal message when this file cannot be loaded for generate.
    pub reject_reason: Option<String>,
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

    /// Probe arch / layer mix / ggml type ids without loading weights.
    /// `reject_reason` is set when Allpaca cannot run the file (hybrid SSM, unknown quant, …).
    pub fn compat_report(&self) -> GgufCompatReport {
        let architecture = self
            .architecture()
            .unwrap_or("unknown")
            .to_string();
        let n_layers = self
            .meta_u32(&format!("{architecture}.block_count"))
            .or_else(|| self.meta_u32("llama.block_count"))
            .unwrap_or_else(|| {
                self.tensors
                    .iter()
                    .filter_map(|t| {
                        let mut parts = t.name.split('.');
                        if parts.next() != Some("blk") {
                            return None;
                        }
                        parts.next()?.parse::<u32>().ok()
                    })
                    .max()
                    .map(|i| i + 1)
                    .unwrap_or(0)
            }) as usize;
        let mut ssm_layers = 0usize;
        let mut attention_layers = 0usize;
        let mut moe_layers = 0usize;
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let ssm = self.tensor(&format!("blk.{i}.ssm_in.weight")).is_some()
                || self.tensor(&format!("blk.{i}.ssm_conv1d.weight")).is_some();
            let attn = self.tensor(&format!("blk.{i}.attn_q.weight")).is_some();
            let moe = self.tensor(&format!("blk.{i}.ffn_gate_exps.weight")).is_some()
                || self.tensor(&format!("blk.{i}.ffn_up_exps.weight")).is_some()
                || self.tensor(&format!("blk.{i}.ffn_gate.0.weight")).is_some();
            if ssm {
                ssm_layers += 1;
            }
            if attn {
                attention_layers += 1;
            }
            if moe {
                moe_layers += 1;
            }
            let block = if ssm {
                HybridLayerKind::Ssm
            } else if attn {
                HybridLayerKind::Attention
            } else {
                HybridLayerKind::DenseFfn
            };
            layers.push(HybridLayerInfo {
                index: i,
                block,
                moe,
            });
        }
        let mut ggml_type_ids: Vec<u32> = self.tensors.iter().map(|t| t.ggml_type_id).collect();
        ggml_type_ids.sort_unstable();
        ggml_type_ids.dedup();

        const SUPPORTED_GGML: &[u32] = &[
            GgmlType::F32 as u32,
            GgmlType::F16 as u32,
            GgmlType::Q4_0 as u32,
            GgmlType::Q8_0 as u32,
            GgmlType::Q4_K as u32,
            GgmlType::Q5_K as u32,
            GgmlType::Q6_K as u32,
        ];
        let unsupported_types: Vec<u32> = ggml_type_ids
            .iter()
            .copied()
            .filter(|id| !SUPPORTED_GGML.contains(id))
            .collect();

        let mut reasons = Vec::new();
        // SSM layers are runnable via the CPU selective-SSM stub when weights are
        // F32/F16/Q* (type 100 ROCmFP4 still blocks below).
        if unsupported_types.contains(&100) {
            reasons.push(
                "ROCmFP4 (ggml_type_id=100): host packed buffer OK; F32 dequant / HIP GEMV not implemented (roadmap: CPU dequant → fused GEMV)"
                    .into(),
            );
        }
        let other_unsupported: Vec<u32> = unsupported_types
            .iter()
            .copied()
            .filter(|id| *id != 100)
            .collect();
        if !other_unsupported.is_empty() {
            reasons.push(format!(
                "unsupported ggml_type_id={other_unsupported:?} (need F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K)"
            ));
        }
        let llama_family = matches!(
            architecture.as_str(),
            "llama" | "mistral" | "mixtral" | "qwen2" | "qwen2moe" | "phi3" | "minicpm"
        );
        if !llama_family && ssm_layers == 0 && attention_layers == 0 && n_layers > 0 {
            reasons.push(format!(
                "architecture '{architecture}' is outside the Llama-layout loader"
            ));
        }
        let reject_reason = if reasons.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported GGUF architecture '{architecture}': {}; layer mix attn={attention_layers} ssm={ssm_layers} moe={moe_layers}; ggml_type_ids={ggml_type_ids:?}",
                reasons.join("; ")
            ))
        };
        GgufCompatReport {
            architecture,
            n_layers,
            ssm_layers,
            attention_layers,
            moe_layers,
            layers,
            ggml_type_ids,
            reject_reason,
        }
    }

    /// Materialize a tensor as F32. F16 / Q4_0 / Q8_0 / Q4_K / Q5_K / Q6_K convert on CPU; other quants are Fatal.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        let n = info.n_elements() as usize;
        let start = self.tensor_data_offset + info.offset as usize;
        match info.ggml_type {
            GgmlType::F32 => {
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
            GgmlType::F16 => {
                let bytes = n.saturating_mul(2);
                if start.saturating_add(bytes) > self.mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} F16 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 2;
                    let h = u16::from_le_bytes([self.mmap[off], self.mmap[off + 1]]);
                    *slot = allpaca_cpu::f16_to_f32(h);
                }
                Ok(out)
            }
            GgmlType::Q4_0 => {
                let nbytes = n
                    .div_ceil(allpaca_cpu::Q4_0_BLOCK_ELEMS)
                    .saturating_mul(allpaca_cpu::Q4_0_BLOCK_BYTES);
                if start.saturating_add(nbytes) > self.mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q4_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_0(&self.mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q8_0 => {
                let nbytes = n
                    .div_ceil(allpaca_cpu::Q8_0_BLOCK_ELEMS)
                    .saturating_mul(allpaca_cpu::Q8_0_BLOCK_BYTES);
                if start.saturating_add(nbytes) > self.mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q8_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q8_0(&self.mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q4_K => {
                let nbytes = n
                    .div_ceil(allpaca_cpu::Q4_K_BLOCK_ELEMS)
                    .saturating_mul(allpaca_cpu::Q4_K_BLOCK_BYTES);
                if start.saturating_add(nbytes) > self.mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q4_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_k(&self.mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q5_K => {
                let nbytes = n
                    .div_ceil(allpaca_cpu::Q5_K_BLOCK_ELEMS)
                    .saturating_mul(allpaca_cpu::Q5_K_BLOCK_BYTES);
                if start.saturating_add(nbytes) > self.mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q5_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q5_k(&self.mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q6_K => {
                let nbytes = n
                    .div_ceil(allpaca_cpu::Q6_K_BLOCK_ELEMS)
                    .saturating_mul(allpaca_cpu::Q6_K_BLOCK_BYTES);
                if start.saturating_add(nbytes) > self.mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q6_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q6_k(&self.mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::RocmFp4 => Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "tensor {name} is ROCmFP4 (ggml_type_id=100): packed host bytes via tensor_bytes; F32 dequant not implemented"
            )))),
            other => Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "tensor {name} is {other:?} (unsupported GGUF quant)"
            )))),
        }
    }

    /// Copy ROCmFP4 (ggml_type_id=100) packed blocks into a host buffer (no dequant).
    pub fn tensor_rocmfp4_host(&self, name: &str) -> Result<Vec<u8>, AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        if info.ggml_type != GgmlType::RocmFp4 && info.ggml_type_id != 100 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "tensor {name} is {:?}/id={} not ROCmFP4(100)",
                info.ggml_type, info.ggml_type_id
            ))));
        }
        Ok(self.tensor_bytes(name)?.to_vec())
    }

    /// Raw on-disk tensor bytes (quant blocks or IEEE floats).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        let start = self.tensor_data_offset + info.offset as usize;
        let nbytes = info.nbytes();
        if start.saturating_add(nbytes) > self.mmap.len() {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "tensor {name} out of range"
            ))));
        }
        Ok(&self.mmap[start..start + nbytes])
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(GgufValue::as_u32)
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(GgufValue::as_f32)
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

    fn write_typed_gguf(path: &std::path::Path, ggml_type_id: u32, payload: &[u8], n_elems: u64) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        let name = "quant.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name.as_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&n_elems.to_le_bytes());
        buf.extend_from_slice(&ggml_type_id.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(payload);
        std::fs::write(path, buf).unwrap();
    }

    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 255 {
            return sign | 0x7c00 | ((mant >> 13) as u16);
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return sign | 0x7c00;
        }
        if new_exp <= 0 {
            return sign;
        }
        sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
    }

    fn encode_q8_0(values: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(values.len(), 32);
        let mut out = Vec::with_capacity(34);
        out.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        out.extend(values.iter().map(|v| *v as u8));
        out
    }

    fn encode_q4_0(qs_minus_8: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(qs_minus_8.len(), 32);
        let mut out = Vec::with_capacity(18);
        out.extend_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        for i in 0..16 {
            let lo = (qs_minus_8[i] + 8) as u8 & 0x0f;
            let hi = (qs_minus_8[i + 16] + 8) as u8 & 0x0f;
            out.push(lo | (hi << 4));
        }
        out
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
    fn meta_f32_reads_rope_freq_base() {
        let path = std::env::temp_dir().join("allpaca_gguf_rope.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "llama.rope.freq_base".into(),
            GgufValue::Float32(500_000.0),
        );
        meta.insert(
            "llama.attention.layer_norm_rms_epsilon".into(),
            GgufValue::Float32(1e-5),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(file.meta_f32("llama.rope.freq_base"), Some(500_000.0));
        assert!(
            file.meta_f32("llama.attention.layer_norm_rms_epsilon")
                .is_some_and(|e| (e - 1e-5).abs() < 1e-8)
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn f16_tensor_load_materializes_to_f32() {
        let path = std::env::temp_dir().join("allpaca_gguf_f16.gguf");
        let values = [1.0f32, -2.0, 0.5, 3.25];
        let mut payload = Vec::with_capacity(values.len() * 2);
        for v in values {
            payload.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
        }
        write_typed_gguf(&path, 1, &payload, values.len() as u64);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::F16);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("F16 must materialize to F32");
        assert_eq!(loaded.len(), values.len());
        for (i, (got, exp)) in loaded.iter().zip(values).enumerate() {
            assert!(
                (*got - exp).abs() < 1e-3,
                "F16 mismatch at {i}: {got} vs {exp}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_0_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q8.gguf");
        let qs: Vec<i8> = (0..32).map(|i| i as i8).collect();
        let payload = encode_q8_0(&qs, 1.0);
        write_typed_gguf(&path, 8, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q8_0);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("Q8_0 must materialize to F32");
        assert_eq!(loaded.len(), 32);
        for (i, v) in loaded.iter().enumerate() {
            assert!(
                (*v - i as f32).abs() < 1e-2,
                "Q8_0 dequant mismatch at {i}: {v}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q4_0_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q4.gguf");
        let qs: Vec<i8> = (0..32).map(|i| (i % 9) as i8 - 4).collect();
        let payload = encode_q4_0(&qs, 0.5);
        write_typed_gguf(&path, 2, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q4_0);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("Q4_0 must materialize to F32");
        assert_eq!(loaded.len(), 32);
        for (i, v) in loaded.iter().enumerate() {
            let expect = 0.5 * qs[i] as f32;
            assert!(
                (*v - expect).abs() < 1e-2,
                "Q4_0 dequant mismatch at {i}: {v} vs {expect}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    fn encode_q4_k_identity_block() -> Vec<u8> {
        // One Q4_K super-block (256 elems, 144 bytes): d=1, dmin=0, first
        // group scale=1, qs low nibbles 0..15 repeating so y[i] = i % 16.
        let mut payload = vec![0u8; 144];
        payload[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        payload[4] = 1;
        for i in 0..32 {
            payload[16 + i] = (i as u8) & 0x0f;
        }
        payload
    }

    #[test]
    fn q4_k_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q4k.gguf");
        let payload = encode_q4_k_identity_block();
        write_typed_gguf(&path, 12, &payload, 256);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q4_K);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("Q4_K must materialize to F32");
        assert_eq!(loaded.len(), 256);
        for i in 0..32 {
            assert!(
                (loaded[i] - (i % 16) as f32).abs() < 1e-2,
                "Q4_K dequant mismatch at {i}: {}",
                loaded[i]
            );
        }
        for i in 32..256 {
            assert!(
                loaded[i].abs() < 1e-3,
                "Q4_K tail should be zero at {i}: {}",
                loaded[i]
            );
        }
        let _ = std::fs::remove_file(path);
    }

    fn encode_q6_k_identity_block() -> Vec<u8> {
        // One Q6_K super-block (256 elems, 210 bytes): d=1, scales[0]=1,
        // first 16 quants 0..15 so y[i] = i.
        let mut payload = vec![0u8; 210];
        for l in 0..16 {
            let stored = (l as u8) + 32;
            payload[l] = stored & 0x0f;
            payload[128 + l] = stored >> 4;
        }
        payload[192] = 1;
        payload[208..210].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        payload
    }

    #[test]
    fn q6_k_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q6k.gguf");
        let payload = encode_q6_k_identity_block();
        write_typed_gguf(&path, 14, &payload, 256);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q6_K);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("Q6_K must materialize to F32");
        assert_eq!(loaded.len(), 256);
        for i in 0..16 {
            assert!(
                (loaded[i] - i as f32).abs() < 1e-2,
                "Q6_K dequant mismatch at {i}: {}",
                loaded[i]
            );
        }
        for i in 16..256 {
            assert!(
                loaded[i].abs() < 1e-3,
                "Q6_K tail should be zero at {i}: {}",
                loaded[i]
            );
        }
        let _ = std::fs::remove_file(path);
    }

    fn encode_q5_k_identity_block() -> Vec<u8> {
        // One Q5_K super-block (256 elems, 176 bytes): d=1, dmin=0, first
        // group scale=1, qs low nibbles 0..15 so y[i] = i % 16.
        let mut payload = vec![0u8; 176];
        payload[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        payload[4] = 1;
        for i in 0..32 {
            payload[48 + i] = (i as u8) & 0x0f;
        }
        payload
    }

    #[test]
    fn q5_k_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q5k.gguf");
        let payload = encode_q5_k_identity_block();
        write_typed_gguf(&path, 13, &payload, 256);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q5_K);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("Q5_K must materialize to F32");
        assert_eq!(loaded.len(), 256);
        for i in 0..32 {
            assert!(
                (loaded[i] - (i % 16) as f32).abs() < 1e-2,
                "Q5_K dequant mismatch at {i}: {}",
                loaded[i]
            );
        }
        for i in 32..256 {
            assert!(
                loaded[i].abs() < 1e-3,
                "Q5_K tail should be zero at {i}: {}",
                loaded[i]
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rocmfp4_type_100_loads_host_bytes_and_refuses_f32_dequant() {
        let path = std::env::temp_dir().join("allpaca_gguf_rocmfp4.gguf");
        // One ROCmFP4 block: 32 elems → 18 bytes.
        let mut payload = vec![0u8; allpaca_cpu::ROCMFP4_BLOCK_BYTES];
        payload[0] = 0xA5;
        payload[17] = 0x5A;
        write_typed_gguf(&path, 100, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::RocmFp4);
        assert_eq!(info.ggml_type_id, 100);
        assert_eq!(info.nbytes(), allpaca_cpu::ROCMFP4_BLOCK_BYTES);
        let host = file.tensor_rocmfp4_host("quant.weight").unwrap();
        assert_eq!(host, payload);
        let err = file.tensor_f32("quant.weight").unwrap_err();
        assert!(err.is_fatal(), "ROCmFP4 F32 path must be Fatal, got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("ROCmFP4") && msg.contains("100") && msg.contains("dequant"),
            "named refuse must cite ROCmFP4/100/dequant; got {msg}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unsupported_quant_tensor_load_is_fatal() {
        let path = std::env::temp_dir().join("allpaca_gguf_q5.gguf");
        write_typed_gguf(&path, 6, &[0u8; 32], 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q5_0);
        let err = file.tensor_f32("quant.weight").unwrap_err();
        assert!(
            err.is_fatal(),
            "unsupported GGUF quant must be Fatal, got {err:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compat_report_allows_f32_hybrid_ssm_stub() {
        let path = std::env::temp_dir().join("allpaca_gguf_hybrid_probe.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("nemotron_h_moe".into()),
        );
        meta.insert(
            "nemotron_h_moe.block_count".into(),
            GgufValue::Uint32(1),
        );
        let ssm = vec![0.0f32; 4];
        write_gguf(
            &path,
            &meta,
            &[GgufWriteTensor {
                name: "blk.0.ssm_in.weight".into(),
                dims: vec![4, 1],
                data: &ssm,
            }],
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        let report = file.compat_report();
        assert_eq!(report.architecture, "nemotron_h_moe");
        assert_eq!(report.n_layers, 1);
        assert_eq!(report.ssm_layers, 1);
        assert!(
            report.reject_reason.is_none(),
            "F32 hybrid SSM must be stub-runnable; got {:?}",
            report.reject_reason
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn nemotron_rocmfp4_gguf_probe_fatals_with_named_reason() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf");
        if !path.exists() {
            eprintln!(
                "skip nemotron_rocmfp4_gguf_probe: missing {}; see compat_report_allows_f32_hybrid_ssm_stub",
                path.display()
            );
            return;
        }
        let file = GgufFile::open(&path).expect("mmap Nemotron header");
        let report = file.compat_report();
        assert_eq!(report.architecture, "nemotron_h_moe");
        assert_eq!(report.n_layers, 52);
        assert!(report.ssm_layers > 0, "expected SSM layers; {report:?}");
        assert!(
            report.attention_layers > 0,
            "expected attention layers; {report:?}"
        );
        assert!(report.moe_layers > 0, "expected MoE layers; {report:?}");
        assert!(
            report.ggml_type_ids.contains(&100),
            "ROCmFP4 uses ggml_type_id=100; ids={:?}",
            report.ggml_type_ids
        );
        let reason = report
            .reject_reason
            .expect("Nemotron ROCmFP4 must reject by name");
        assert!(
            reason.contains("nemotron_h_moe")
                && reason.contains("100")
                && reason.contains("ROCmFP4"),
            "Fatal reason must name arch, type 100, and ROCmFP4; got {reason}"
        );
        assert_eq!(report.layers.len(), 52);
        assert!(
            report
                .layers
                .iter()
                .any(|l| l.block == HybridLayerKind::Ssm),
            "layer map must tag SSM blocks; {:?}",
            report.layers.iter().take(8).collect::<Vec<_>>()
        );
        assert!(
            report
                .layers
                .iter()
                .any(|l| l.block == HybridLayerKind::Attention),
            "layer map must tag attention blocks"
        );
        assert!(
            report.layers.iter().any(|l| l.moe),
            "layer map must flag MoE FFN layers"
        );
    }
}
