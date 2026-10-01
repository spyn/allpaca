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

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            Self::Uint32(v) => Some(*v != 0),
            Self::Uint64(v) => Some(*v != 0),
            Self::Int32(v) => Some(*v != 0),
            _ => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.as_u32()?);
                }
                Some(out)
            }
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

    pub fn as_i32_array(&self) -> Option<Vec<i32>> {
        match self {
            Self::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Self::Int32(v) => out.push(*v),
                        Self::Int64(v) => out.push(i32::try_from(*v).ok()?),
                        Self::Uint32(v) => out.push(i32::try_from(*v).ok()?),
                        Self::Uint64(v) => out.push(i32::try_from(*v).ok()?),
                        Self::Int8(v) => out.push(*v as i32),
                        Self::Uint8(v) => out.push(*v as i32),
                        Self::Int16(v) => out.push(*v as i32),
                        Self::Uint16(v) => out.push(*v as i32),
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
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    /// Unsloth / ggml IQ4_NL (ggml_type_id=20).
    IQ4_NL = 20,
    /// Unsloth / ggml IQ4_XS (ggml_type_id=23).
    IQ4_XS = 23,
    /// ggml I32 (tid2eid tables).
    I32 = 26,
    /// IEEE bfloat16 (ggml_type_id=30).
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    /// ggml MXFP4 (E8M0 + nibbles, ggml_type_id=39).
    MXFP4 = 39,
    /// AMD ROCmFP4 dual-scale (ggml_type_id=100).
    RocmFp4 = 100,
    Other = 255,
}

/// Dequantize a packed slice of `ty` (whole blocks, `n` elements) into `out`.
///
/// Row-granular counterpart of [`GgufFile::tensor_f32`]: callers pass one row's
/// bytes (e.g. a single `token_embd` row) instead of the whole tensor.
pub fn dequant_packed(
    ty: GgmlType,
    bytes: &[u8],
    n: usize,
    out: &mut [f32],
) -> Result<(), AllpacaError> {
    if out.len() < n {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "dequant {ty:?}: out holds {} of {n}",
            out.len()
        ))));
    }
    let out = &mut out[..n];
    match ty {
        GgmlType::F32 => {
            for (slot, c) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *slot = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            Ok(())
        }
        GgmlType::F16 => {
            for (slot, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = allpaca_cpu::f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
            Ok(())
        }
        GgmlType::BF16 => {
            for (slot, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = allpaca_cpu::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
            Ok(())
        }
        GgmlType::Q4_0 => allpaca_cpu::dequant_q4_0(bytes, n, out),
        GgmlType::Q4_1 => allpaca_cpu::dequant_q4_1(bytes, n, out),
        GgmlType::Q5_0 => allpaca_cpu::dequant_q5_0(bytes, n, out),
        GgmlType::Q5_1 => allpaca_cpu::dequant_q5_1(bytes, n, out),
        GgmlType::Q8_0 => allpaca_cpu::dequant_q8_0(bytes, n, out),
        GgmlType::Q8_1 => allpaca_cpu::dequant_q8_1(bytes, n, out),
        GgmlType::Q2_K => allpaca_cpu::dequant_q2_k(bytes, n, out),
        GgmlType::Q3_K => allpaca_cpu::dequant_q3_k(bytes, n, out),
        GgmlType::Q4_K => allpaca_cpu::dequant_q4_k(bytes, n, out),
        GgmlType::Q5_K => allpaca_cpu::dequant_q5_k(bytes, n, out),
        GgmlType::Q6_K => allpaca_cpu::dequant_q6_k(bytes, n, out),
        GgmlType::IQ4_NL => allpaca_cpu::dequant_iq4_nl(bytes, n, out),
        GgmlType::IQ4_XS => allpaca_cpu::dequant_iq4_xs(bytes, n, out),
        GgmlType::TQ1_0 => allpaca_cpu::dequant_tq1_0(bytes, n, out),
        GgmlType::TQ2_0 => allpaca_cpu::dequant_tq2_0(bytes, n, out),
        GgmlType::MXFP4 => allpaca_cpu::dequant_mxfp4(bytes, n, out),
        GgmlType::RocmFp4 => allpaca_cpu::dequant_rocmfp4(bytes, n, out),
        GgmlType::I32 | GgmlType::Other => Err(AllpacaError::Fatal(FatalError::ModelLoad(
            format!("dequant {ty:?} unsupported"),
        ))),
    }
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
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            20 => Self::IQ4_NL,
            23 => Self::IQ4_XS,
            26 => Self::I32,
            30 => Self::BF16,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            39 => Self::MXFP4,
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
    /// Index into `GgufFile::shards` for split GGUF.
    pub shard: usize,
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
            GgmlType::Q4_1 => n
                .div_ceil(allpaca_cpu::Q4_1_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q4_1_BLOCK_BYTES),
            GgmlType::Q5_0 => n
                .div_ceil(allpaca_cpu::Q5_0_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q5_0_BLOCK_BYTES),
            GgmlType::Q5_1 => n
                .div_ceil(allpaca_cpu::Q5_1_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q5_1_BLOCK_BYTES),
            GgmlType::Q8_0 => n
                .div_ceil(allpaca_cpu::Q8_0_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q8_0_BLOCK_BYTES),
            GgmlType::Q8_1 => n
                .div_ceil(allpaca_cpu::Q8_1_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q8_1_BLOCK_BYTES),
            GgmlType::Q2_K => n
                .div_ceil(allpaca_cpu::Q2_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q2_K_BLOCK_BYTES),
            GgmlType::Q3_K => n
                .div_ceil(allpaca_cpu::Q3_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q3_K_BLOCK_BYTES),
            GgmlType::Q4_K => n
                .div_ceil(allpaca_cpu::Q4_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q4_K_BLOCK_BYTES),
            GgmlType::Q5_K => n
                .div_ceil(allpaca_cpu::Q5_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q5_K_BLOCK_BYTES),
            GgmlType::Q6_K => n
                .div_ceil(allpaca_cpu::Q6_K_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::Q6_K_BLOCK_BYTES),
            GgmlType::IQ4_NL => n
                .div_ceil(allpaca_cpu::IQ4_NL_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::IQ4_NL_BLOCK_BYTES),
            GgmlType::IQ4_XS => n
                .div_ceil(allpaca_cpu::IQ4_XS_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::IQ4_XS_BLOCK_BYTES),
            GgmlType::I32 => n.saturating_mul(4),
            GgmlType::BF16 => n.saturating_mul(2),
            GgmlType::MXFP4 => n
                .div_ceil(allpaca_cpu::MXFP4_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::MXFP4_BLOCK_BYTES),
            GgmlType::TQ1_0 => n
                .div_ceil(allpaca_cpu::TQ1_0_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::TQ1_0_BLOCK_BYTES),
            GgmlType::TQ2_0 => n
                .div_ceil(allpaca_cpu::TQ2_0_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::TQ2_0_BLOCK_BYTES),
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

struct GgufShard {
    mmap: Mmap,
    tensor_data_offset: usize,
}

pub struct GgufFile {
    shards: Vec<GgufShard>,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub tensor_data_offset: usize,
    pub alignment: u32,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self, AllpacaError> {
        if path.is_dir() {
            return Self::open_dir(path);
        }
        if let Some(siblings) = split_sibling_paths(path)? {
            return Self::open_shards(&siblings);
        }
        Self::open_one(path)
    }

    fn open_one(path: &Path) -> Result<Self, AllpacaError> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!("mmap failed: {e}")))
        })?;
        Self::parse(mmap)
    }

    fn open_dir(dir: &Path) -> Result<Self, AllpacaError> {
        if let Some(first) = first_split_shard_in_dir(dir) {
            return Self::open(&first);
        }
        // Non-split GGUF in this directory (or one nested quant folder).
        let mut found = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for ent in entries.flatten() {
                let path = ent.path();
                if path.is_dir() {
                    if let Some(first) = first_split_shard_in_dir(&path) {
                        return Self::open(&first);
                    }
                    if let Ok(inner) = std::fs::read_dir(&path) {
                        for e in inner.flatten() {
                            let p = e.path();
                            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                            if p.extension().and_then(|x| x.to_str()) == Some("gguf")
                                && !is_dspark_filename(name)
                            {
                                found.push(p);
                            }
                        }
                    }
                    continue;
                }
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if path.extension().and_then(|e| e.to_str()) == Some("gguf")
                    && !is_dspark_filename(name)
                {
                    found.push(path);
                }
            }
        }
        found.sort();
        found
            .into_iter()
            .next()
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "no GGUF shards in {}",
                    dir.display()
                )))
            })
            .and_then(|p| Self::open(&p))
    }

    fn open_shards(paths: &[std::path::PathBuf]) -> Result<Self, AllpacaError> {
        if paths.is_empty() {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(
                "split GGUF has no shards".into(),
            )));
        }
        if paths.len() == 1 {
            return Self::open_one(&paths[0]);
        }
        let mut parts = Vec::with_capacity(paths.len());
        for p in paths {
            parts.push(Self::open_one(p)?);
        }
        Self::merge_shards(parts)
    }

    fn merge_shards(parts: Vec<Self>) -> Result<Self, AllpacaError> {
        let meta_idx = parts
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.metadata.len())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let version = parts[meta_idx].version;
        let alignment = parts[meta_idx].alignment;
        let metadata = parts[meta_idx].metadata.clone();
        let tensor_data_offset = parts[meta_idx].tensor_data_offset;
        let mut tensors = Vec::new();
        let mut shards = Vec::with_capacity(parts.len());
        for (i, mut part) in parts.into_iter().enumerate() {
            for mut t in part.tensors.drain(..) {
                t.shard = i;
                tensors.push(t);
            }
            shards.extend(part.shards.drain(..));
        }
        Ok(Self {
            shards,
            version,
            metadata,
            tensors,
            tensor_data_offset,
            alignment,
        })
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
                shard: 0,
            });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u32)
            .unwrap_or(32);
        let tensor_data_offset = align_up(cur.pos, alignment as usize);

        Ok(Self {
            shards: vec![GgufShard {
                mmap,
                tensor_data_offset,
            }],
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
        let architecture = self.architecture().unwrap_or("unknown").to_string();
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
                || self.tensor(&format!("blk.{i}.ssm_conv1d.weight")).is_some()
                || self.tensor(&format!("blk.{i}.ssm_out.weight")).is_some();
            let attn = self.tensor(&format!("blk.{i}.attn_q.weight")).is_some()
                || self.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some()
                || self.tensor(&format!("blk.{i}.attn_q_a.weight")).is_some();
            let moe = self
                .tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
                .is_some()
                || self
                    .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                    .is_some()
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
            } else if moe {
                // MoE-only block (Nemotron-H): the FFN must route experts,
                // which `apply_ffn` skips for DenseFfn blocks.
                HybridLayerKind::Moe
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
            GgmlType::Q4_1 as u32,
            GgmlType::Q5_0 as u32,
            GgmlType::Q5_1 as u32,
            GgmlType::Q8_0 as u32,
            GgmlType::Q8_1 as u32,
            GgmlType::Q2_K as u32,
            GgmlType::Q3_K as u32,
            GgmlType::Q4_K as u32,
            GgmlType::Q5_K as u32,
            GgmlType::Q6_K as u32,
            GgmlType::IQ4_NL as u32,
            GgmlType::IQ4_XS as u32,
            GgmlType::I32 as u32,
            GgmlType::BF16 as u32,
            GgmlType::MXFP4 as u32,
            GgmlType::TQ1_0 as u32,
            GgmlType::TQ2_0 as u32,
            GgmlType::RocmFp4 as u32,
        ];
        let unsupported_types: Vec<u32> = ggml_type_ids
            .iter()
            .copied()
            .filter(|id| !SUPPORTED_GGML.contains(id))
            .collect();

        let mut reasons = Vec::new();
        if !unsupported_types.is_empty() {
            reasons.push(format!(
                "unsupported ggml_type_id={unsupported_types:?} (need F32/F16/BF16/I32/MXFP4/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/TQ1_0/TQ2_0/ROCmFP4)"
            ));
        }
        let llama_family = matches!(
            architecture.as_str(),
            "llama"
                | "mistral"
                | "mixtral"
                | "qwen2"
                | "qwen2moe"
                | "qwen3"
                | "qwen3moe"
                | "qwen35"
                | "qwen3next"
                | "phi3"
                | "minicpm"
                | "nemotron_h_moe"
                | "spark2_5"
                | "gemma2"
                | "gemma3"
                | "deepseek2"
                | "deepseek4"
                | "deepseek_v4"
                | "dflash"
                | "deepseek_v4_flash_dspark_draft"
        );
        if !llama_family && ssm_layers == 0 && attention_layers == 0 && n_layers > 0 {
            reasons.push(format!(
                "architecture '{architecture}' is outside the Llama-layout loader"
            ));
        }
        if architecture == "nemotron_h_moe" && ssm_layers > 0 {
            for suffix in [
                "ssm.inner_size",
                "ssm.state_size",
                "ssm.time_step_rank",
                "ssm.group_count",
                "ssm.conv_kernel",
            ] {
                let key = format!("{architecture}.{suffix}");
                if self.meta_u32(&key).is_none() {
                    reasons.push(format!("missing Mamba-2 metadata {key}"));
                }
            }
            for layer in layers.iter().filter(|l| l.block == HybridLayerKind::Ssm) {
                for suffix in [
                    "ssm_in.weight",
                    "ssm_conv1d.weight",
                    "ssm_dt.bias",
                    "ssm_a",
                    "ssm_d",
                    "ssm_norm.weight",
                    "ssm_out.weight",
                ] {
                    let name = format!("blk.{}.{suffix}", layer.index);
                    if self.tensor(&name).is_none() {
                        reasons.push(format!("missing Mamba-2 tensor {name}"));
                    }
                }
            }
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

    /// Materialize a tensor as F32. Supported quants convert on CPU; unknown types are Fatal.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        let n = info.n_elements() as usize;
        let shard = self.shards.get(info.shard).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "tensor {name} shard {} missing",
                info.shard
            )))
        })?;
        let start = shard.tensor_data_offset + info.offset as usize;
        let mmap = &shard.mmap;
        match info.ggml_type {
            GgmlType::F32 => {
                let bytes = n.saturating_mul(4);
                if start.saturating_add(bytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 4;
                    *slot = f32::from_le_bytes([
                        mmap[off],
                        mmap[off + 1],
                        mmap[off + 2],
                        mmap[off + 3],
                    ]);
                }
                Ok(out)
            }
            GgmlType::F16 => {
                let bytes = n.saturating_mul(2);
                if start.saturating_add(bytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} F16 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 2;
                    let h = u16::from_le_bytes([mmap[off], mmap[off + 1]]);
                    *slot = allpaca_cpu::f16_to_f32(h);
                }
                Ok(out)
            }
            GgmlType::Q4_0 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q4_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_0(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q4_1 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q4_1 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_1(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q5_0 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q5_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q5_0(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q5_1 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q5_1 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q5_1(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q8_0 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q8_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q8_0(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q8_1 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q8_1 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q8_1(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q2_K => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q2_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q2_k(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q3_K => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q3_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q3_k(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q4_K => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q4_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_k(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q5_K => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q5_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q5_k(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::Q6_K => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} Q6_K out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_q6_k(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::IQ4_NL => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} IQ4_NL out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_iq4_nl(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::IQ4_XS => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} IQ4_XS out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_iq4_xs(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::TQ1_0 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} TQ1_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_tq1_0(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::TQ2_0 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} TQ2_0 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_tq2_0(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::BF16 => {
                let nbytes = n.saturating_mul(2);
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} BF16 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 2;
                    let h = u16::from_le_bytes([mmap[off], mmap[off + 1]]);
                    *slot = allpaca_cpu::bf16_to_f32(h);
                }
                Ok(out)
            }
            GgmlType::MXFP4 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} MXFP4 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_mxfp4(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
            GgmlType::I32 => {
                let ints = self.tensor_i32(name)?;
                Ok(ints.into_iter().map(|v| v as f32).collect())
            }
            GgmlType::RocmFp4 => {
                let nbytes = info.nbytes();
                if start.saturating_add(nbytes) > mmap.len() {
                    return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                        "tensor {name} ROCmFP4 out of range"
                    ))));
                }
                let mut out = vec![0.0f32; n];
                allpaca_cpu::dequant_rocmfp4(&mmap[start..start + nbytes], n, &mut out)?;
                Ok(out)
            }
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
        self.tensor_bytes_of(info)
    }

    pub fn tensor_bytes_of(&self, info: &TensorInfo) -> Result<&[u8], AllpacaError> {
        let shard = self.shards.get(info.shard).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "tensor {} shard {} missing",
                info.name, info.shard
            )))
        })?;
        let start = shard.tensor_data_offset + info.offset as usize;
        let nbytes = info.nbytes();
        if start.saturating_add(nbytes) > shard.mmap.len() {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "tensor {} out of range",
                info.name
            ))));
        }
        Ok(&shard.mmap[start..start + nbytes])
    }

    pub fn tensor_i32(&self, name: &str) -> Result<Vec<i32>, AllpacaError> {
        let info = self.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        if info.ggml_type != GgmlType::I32 && info.ggml_type_id != 26 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "tensor {name} is {:?}/id={} not I32",
                info.ggml_type, info.ggml_type_id
            ))));
        }
        let bytes = self.tensor_bytes_of(info)?;
        let n = info.n_elements() as usize;
        if bytes.len() < n.saturating_mul(4) {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "tensor {name} I32 truncated"
            ))));
        }
        let mut out = vec![0i32; n];
        for (i, slot) in out.iter_mut().enumerate() {
            let off = i * 4;
            *slot = i32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        Ok(out)
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

    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(GgufValue::as_bool)
    }
}

/// `stem-00001-of-00005.gguf` → (`stem`, 1, 5).
pub fn parse_split_filename(name: &str) -> Option<(String, u32, u32)> {
    let name = name
        .strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))?;
    let (left, total_s) = name.rsplit_once("-of-")?;
    let (stem, idx_s) = left.rsplit_once('-')?;
    let idx: u32 = idx_s.parse().ok()?;
    let total: u32 = total_s.parse().ok()?;
    if total == 0 || idx == 0 || idx > total {
        return None;
    }
    Some((stem.to_string(), idx, total))
}

pub fn is_dspark_filename(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.contains("dspark") || l.contains("dflash")
}

pub fn is_split_gguf_name(name: &str) -> bool {
    parse_split_filename(name).is_some()
}

fn padded_split_name(stem: &str, idx: u32, total: u32) -> String {
    format!("{stem}-{idx:05}-of-{total:05}.gguf")
}

/// Sibling shards for a `*-NNNNN-of-MMMMM.gguf` path. `None` if not a split name.
pub fn split_sibling_paths(path: &Path) -> Result<Option<Vec<std::path::PathBuf>>, AllpacaError> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let Some((stem, _, total)) = parse_split_filename(name) else {
        return Ok(None);
    };
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::with_capacity(total as usize);
    for i in 1..=total {
        let padded = dir.join(padded_split_name(&stem, i, total));
        let loose = dir.join(format!("{stem}-{i}-of-{total}.gguf"));
        if padded.is_file() {
            out.push(padded);
        } else if loose.is_file() {
            out.push(loose);
        } else {
            return Err(AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "split GGUF missing shard {i}/{total} next to {}",
                path.display()
            ))));
        }
    }
    Ok(Some(out))
}

fn first_split_shard_in_dir(dir: &Path) -> Option<std::path::PathBuf> {
    let mut found = Vec::new();
    collect_split_shards(dir, 1, &mut found);
    found.sort_by(|a, b| {
        let an = a.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let bn = b.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let ar = if an.contains("00001-of") { 0 } else { 1 };
        let br = if bn.contains("00001-of") { 0 } else { 1 };
        ar.cmp(&br).then_with(|| an.cmp(bn))
    });
    found.into_iter().next()
}

fn collect_split_shards(dir: &Path, nested: u32, found: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if path.is_dir() && nested > 0 {
            collect_split_shards(&path, nested - 1, found);
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        if is_dspark_filename(name) {
            continue;
        }
        if parse_split_filename(name).is_some() {
            found.push(path);
        }
    }
}

/// DSpark sidecar next to a target GGUF, at the snapshot root, or under `dspark/`.
pub fn discover_dspark_sidecar(target: &Path) -> Option<std::path::PathBuf> {
    let mut dirs = Vec::new();
    if target.is_file() {
        if let Some(dir) = target.parent() {
            dirs.push(dir.to_path_buf());
            if let Some(up) = dir.parent() {
                dirs.push(up.to_path_buf());
            }
        }
    } else if target.is_dir() {
        dirs.push(target.to_path_buf());
        if let Some(up) = target.parent() {
            dirs.push(up.to_path_buf());
        }
    }
    let mut hits = Vec::new();
    for dir in dirs {
        collect_dspark_files(&dir, 1, &mut hits);
    }
    hits.sort_by(|a, b| dspark_hit_rank(a).cmp(&dspark_hit_rank(b)));
    hits.into_iter().next()
}

fn dspark_hit_rank(path: &Path) -> (u8, String) {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let pref = if name.contains("q8_0") {
        0
    } else if name.contains("bf16") {
        2
    } else {
        1
    };
    (pref, name)
}

fn collect_dspark_files(dir: &Path, nested: u32, hits: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if path.is_dir() && nested > 0 {
            collect_dspark_files(&path, nested - 1, hits);
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") && is_dspark_filename(name) {
            hits.push(path);
        }
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
        String::from_utf8(bytes.to_vec()).map_err(|_| AllpacaError::Fatal(FatalError::InvalidUtf8))
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
    use crate::writer::{write_gguf, write_gguf_quant, GgufWriteQuantTensor, GgufWriteTensor};
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

    fn encode_tq2_0(values: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(values.len(), 256);
        let mut out = vec![0u8; allpaca_cpu::TQ2_0_BLOCK_BYTES];
        for group in 0..2 {
            for m in 0..32 {
                let mut packed = 0u8;
                for digit in 0..4 {
                    let trit = values[group * 128 + digit * 32 + m] + 1;
                    assert!((0..=2).contains(&trit));
                    packed |= (trit as u8) << (digit * 2);
                }
                out[group * 32 + m] = packed;
            }
        }
        out[64..66].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        out
    }

    fn encode_tq1_0(values: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(values.len(), 256);
        let mut out = vec![0u8; allpaca_cpu::TQ1_0_BLOCK_BYTES];
        let encode_digits = |digits: &[i8]| {
            let mut q = 0u16;
            for &value in digits {
                assert!((-1..=1).contains(&value));
                q = q * 3 + (value + 1) as u16;
            }
            ((q * 256 + 242) / 243) as u8
        };

        for m in 0..32 {
            let digits: [i8; 5] = std::array::from_fn(|digit| values[digit * 32 + m]);
            out[m] = encode_digits(&digits);
        }
        for m in 0..16 {
            let digits: [i8; 5] = std::array::from_fn(|digit| values[160 + digit * 16 + m]);
            out[32 + m] = encode_digits(&digits);
        }
        for m in 0..4 {
            let mut digits = [0i8; 5];
            for digit in 0..4 {
                digits[digit] = values[240 + digit * 4 + m];
            }
            digits[4] = -1;
            out[48 + m] = encode_digits(&digits);
        }
        out[52..54].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
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
        meta.insert("llama.rope.freq_base".into(), GgufValue::Float32(500_000.0));
        meta.insert(
            "llama.attention.layer_norm_rms_epsilon".into(),
            GgufValue::Float32(1e-5),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(file.meta_f32("llama.rope.freq_base"), Some(500_000.0));
        assert!(file
            .meta_f32("llama.attention.layer_norm_rms_epsilon")
            .is_some_and(|e| (e - 1e-5).abs() < 1e-8));
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

    #[test]
    fn tq2_0_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_tq2_0.gguf");
        let values: Vec<i8> = (0..256).map(|i| (i % 3) as i8 - 1).collect();
        let scale = 0.5f32;
        let payload = encode_tq2_0(&values, scale);
        write_typed_gguf(&path, 35, &payload, 256);

        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::TQ2_0);
        assert_eq!(info.nbytes(), allpaca_cpu::TQ2_0_BLOCK_BYTES);
        assert!(file.compat_report().reject_reason.is_none());
        let loaded = file.tensor_f32("quant.weight").expect("TQ2_0");
        for (i, (&value, &got)) in values.iter().zip(loaded.iter()).enumerate() {
            let want = scale * f32::from(value);
            assert!((got - want).abs() < 1e-3, "TQ2_0 {i}: {got} vs {want}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn tq1_0_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_tq1_0.gguf");
        let values: Vec<i8> = (0..256).map(|i| ((i * 7) % 3) as i8 - 1).collect();
        let scale = 0.5f32;
        let payload = encode_tq1_0(&values, scale);
        write_typed_gguf(&path, 34, &payload, 256);

        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::TQ1_0);
        assert_eq!(info.nbytes(), allpaca_cpu::TQ1_0_BLOCK_BYTES);
        assert!(file.compat_report().reject_reason.is_none());
        let loaded = file.tensor_f32("quant.weight").expect("TQ1_0");
        for (i, (&value, &got)) in values.iter().zip(loaded.iter()).enumerate() {
            let want = scale * f32::from(value);
            assert!((got - want).abs() < 1e-3, "TQ1_0 {i}: {got} vs {want}");
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
    fn iq4_nl_tensor_load_dequants_ones() {
        let path = std::env::temp_dir().join("allpaca_gguf_iq4nl.gguf");
        let mut payload = vec![0u8; allpaca_cpu::IQ4_NL_BLOCK_BYTES];
        payload[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        for slot in payload.iter_mut().skip(2) {
            *slot = 0x88;
        }
        write_typed_gguf(&path, 20, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::IQ4_NL);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("IQ4_NL must materialize to F32");
        assert_eq!(loaded.len(), 32);
        assert!(loaded.iter().all(|v| (v - 1.0).abs() < 1e-4));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn iq4_xs_tensor_load_dequants_first_group() {
        let path = std::env::temp_dir().join("allpaca_gguf_iq4xs.gguf");
        let mut payload = vec![0u8; allpaca_cpu::IQ4_XS_BLOCK_BYTES];
        payload[0..2].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        payload[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes());
        payload[4] = 0x01;
        for slot in payload.iter_mut().skip(8).take(16) {
            *slot = 0x88;
        }
        write_typed_gguf(&path, 23, &payload, 256);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::IQ4_XS);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("IQ4_XS must materialize to F32");
        assert_eq!(loaded.len(), 256);
        assert!(loaded[..32].iter().all(|v| (v - 1.0).abs() < 1e-4));
        assert!(loaded[32..].iter().all(|v| v.abs() < 1e-5));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rocmfp4_type_100_loads_host_bytes_and_dequants_to_f32() {
        let path = std::env::temp_dir().join("allpaca_gguf_rocmfp4.gguf");
        // One ROCmFP4 block: 32 elems → 18 bytes.
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let payload = allpaca_cpu::quantize_rocmfp4(&values);
        assert_eq!(payload.len(), allpaca_cpu::ROCMFP4_BLOCK_BYTES);
        write_typed_gguf(&path, 100, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::RocmFp4);
        assert_eq!(info.ggml_type_id, 100);
        assert_eq!(info.nbytes(), allpaca_cpu::ROCMFP4_BLOCK_BYTES);
        let host = file.tensor_rocmfp4_host("quant.weight").unwrap();
        assert_eq!(host, payload);
        let loaded = file.tensor_f32("quant.weight").unwrap();
        assert_eq!(loaded.len(), 32);
        for (i, (&want, &got)) in values.iter().zip(loaded.iter()).enumerate() {
            assert!(
                (want - got).abs() <= 0.8,
                "ROCmFP4 dequant lane {i}: want {want} got {got}"
            );
        }
    }

    #[test]
    fn q5_0_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q5_0.gguf");
        // 5-bit codes 0..31 → dequant d*(q-16); pick q = 16+k for k in -4..4 repeating.
        let mut qs = [0u8; 16];
        let mut qh = 0u32;
        let mut expect = [0.0f32; 32];
        let d = 0.5f32;
        for i in 0..32 {
            let q = ((i % 9) + 12) as u8; // 12..20 → (q-16) in -4..4
            expect[i] = d * (q as f32 - 16.0);
            if i < 16 {
                qs[i] = q & 0x0f;
                if q & 0x10 != 0 {
                    qh |= 1u32 << i;
                }
            } else {
                let j = i - 16;
                qs[j] |= (q & 0x0f) << 4;
                if q & 0x10 != 0 {
                    qh |= 1u32 << i;
                }
            }
        }
        let mut payload = Vec::with_capacity(22);
        payload.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        payload.extend_from_slice(&qh.to_le_bytes());
        payload.extend_from_slice(&qs);
        write_typed_gguf(&path, 6, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Q5_0);
        let loaded = file
            .tensor_f32("quant.weight")
            .expect("Q5_0 must materialize to F32");
        assert_eq!(loaded.len(), 32);
        for (i, (&want, &got)) in expect.iter().zip(loaded.iter()).enumerate() {
            assert!(
                (want - got).abs() < 1e-2,
                "Q5_0 dequant mismatch at {i}: {got} vs {want}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q4_1_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q4_1.gguf");
        let d = 0.25f32;
        let m = -1.0f32;
        let mut qs = [0u8; 16];
        let mut expect = [0.0f32; 32];
        for i in 0..32 {
            let q = (i % 16) as u8;
            expect[i] = d * q as f32 + m;
            if i < 16 {
                qs[i] = q;
            } else {
                qs[i - 16] |= q << 4;
            }
        }
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        payload.extend_from_slice(&f32_to_f16_bits(m).to_le_bytes());
        payload.extend_from_slice(&qs);
        write_typed_gguf(&path, 3, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            file.tensor("quant.weight").unwrap().ggml_type,
            GgmlType::Q4_1
        );
        let loaded = file.tensor_f32("quant.weight").expect("Q4_1");
        for (i, (&want, &got)) in expect.iter().zip(loaded.iter()).enumerate() {
            assert!((want - got).abs() < 1e-2, "Q4_1 {i}: {got} vs {want}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q8_1_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q8_1.gguf");
        let d = 0.5f32;
        let values: Vec<i8> = (0..32).map(|i| i - 16).collect();
        let mut payload = Vec::with_capacity(36);
        payload.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        payload.extend_from_slice(&f32_to_f16_bits(0.0).to_le_bytes()); // unused sum
        payload.extend(values.iter().map(|v| *v as u8));
        write_typed_gguf(&path, 9, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            file.tensor("quant.weight").unwrap().ggml_type,
            GgmlType::Q8_1
        );
        let loaded = file.tensor_f32("quant.weight").expect("Q8_1");
        for (i, v) in loaded.iter().enumerate() {
            let expect = d * values[i] as f32;
            assert!((*v - expect).abs() < 1e-2, "Q8_1 {i}: {v} vs {expect}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q5_1_tensor_load_dequants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q5_1.gguf");
        let d = 0.5f32;
        let m = -2.0f32;
        let mut qs = [0u8; 16];
        let mut qh = 0u32;
        let mut expect = [0.0f32; 32];
        for i in 0..32 {
            let q = (i % 17) as u8; // 0..16
            expect[i] = d * q as f32 + m;
            if i < 16 {
                qs[i] = q & 0x0f;
                if q & 0x10 != 0 {
                    qh |= 1u32 << i;
                }
            } else {
                let j = i - 16;
                qs[j] |= (q & 0x0f) << 4;
                if q & 0x10 != 0 {
                    qh |= 1u32 << i;
                }
            }
        }
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        payload.extend_from_slice(&f32_to_f16_bits(m).to_le_bytes());
        payload.extend_from_slice(&qh.to_le_bytes());
        payload.extend_from_slice(&qs);
        write_typed_gguf(&path, 7, &payload, 32);
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            file.tensor("quant.weight").unwrap().ggml_type,
            GgmlType::Q5_1
        );
        let loaded = file.tensor_f32("quant.weight").expect("Q5_1");
        for (i, (&want, &got)) in expect.iter().zip(loaded.iter()).enumerate() {
            assert!((want - got).abs() < 1e-2, "Q5_1 {i}: {got} vs {want}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q2_k_tensor_load_dequants_zero_block() {
        let path = std::env::temp_dir().join("allpaca_gguf_q2_k.gguf");
        // d=1, dmin=0, scales=1, qs=0 → all zeros.
        let mut payload = vec![0u8; 84];
        for s in &mut payload[0..16] {
            *s = 0x01;
        }
        payload[80..82].copy_from_slice(&f32_to_f16_bits(1.0).to_le_bytes());
        write_typed_gguf(&path, 10, &payload, 256);
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            file.tensor("quant.weight").unwrap().ggml_type,
            GgmlType::Q2_K
        );
        let loaded = file.tensor_f32("quant.weight").expect("Q2_K");
        assert_eq!(loaded.len(), 256);
        assert!(loaded.iter().all(|v| v.abs() < 1e-5), "expected zeros");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn q3_k_tensor_load_dequants_zero_quants() {
        let path = std::env::temp_dir().join("allpaca_gguf_q3_k.gguf");
        // hmask all-bits set (no -4), qs=0, scales remapped to 32 (dl=0), d=1 → zeros.
        let mut payload = vec![0u8; 110];
        payload[0..32].fill(0xff);
        // scales bytes that decode to 32 after remapping: pack 0x20 in low nibbles.
        // Easiest: set raw scales so remapped scale_bytes are all 32.
        // aux after remap uses low 4 bits of first 8 bytes + packed bits from bytes 8..12.
        // Force all scale_bytes=32 by writing known pattern from llama.cpp identity.
        for i in 0..8 {
            payload[96 + i] = 0x20; // low nibble 0, high will be remapped
        }
        // After remap with tmp=0, aux[0]/aux[1] keep low nibbles → 0x20 & 0x0f = 0, not 32.
        // Use scale value 32 = 0x20 in the remapped i8 slots via aux packing:
        // Put 0x20 in each byte of aux[0] and aux[1] after remap: set low nibbles to 0x0 and
        // high from tmp. Simpler path: d=0 → all zeros regardless of scales.
        payload[108..110].copy_from_slice(&f32_to_f16_bits(0.0).to_le_bytes());
        write_typed_gguf(&path, 11, &payload, 256);
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            file.tensor("quant.weight").unwrap().ggml_type,
            GgmlType::Q3_K
        );
        let loaded = file.tensor_f32("quant.weight").expect("Q3_K");
        assert_eq!(loaded.len(), 256);
        assert!(
            loaded.iter().all(|v| v.abs() < 1e-5),
            "d=0 must yield zeros"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn unsupported_quant_tensor_load_is_fatal() {
        let path = std::env::temp_dir().join("allpaca_gguf_unknown_quant.gguf");
        write_typed_gguf(&path, 99, &[0u8; 32], 32);
        let file = GgufFile::open(&path).unwrap();
        let info = file.tensor("quant.weight").expect("header lists tensor");
        assert_eq!(info.ggml_type, GgmlType::Other);
        let err = file.tensor_f32("quant.weight").unwrap_err();
        assert!(
            err.is_fatal(),
            "unsupported GGUF quant must be Fatal, got {err:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compat_report_rejects_incomplete_nemotron_mamba2_layout() {
        let path = std::env::temp_dir().join("allpaca_gguf_hybrid_probe.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("nemotron_h_moe".into()),
        );
        meta.insert("nemotron_h_moe.block_count".into(), GgufValue::Uint32(1));
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
        let reject = report
            .reject_reason
            .expect("incomplete Mamba-2 must reject");
        assert!(reject.contains("missing Mamba-2"), "{reject}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn nemotron_rocmfp4_gguf_probe_dequants_type_100() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf");
        if !path.exists() {
            eprintln!(
                "skip nemotron_rocmfp4_gguf_probe: missing {}; synthetic full Mamba-2 fixture remains covered",
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
        assert!(
            report.reject_reason.is_none(),
            "ROCmFP4 must no longer reject; got {:?}",
            report.reject_reason
        );
        let arch = report.architecture.as_str();
        eprintln!(
            "Nemotron Mamba-2 meta inner={:?} state={:?} heads={:?} groups={:?} conv={:?} experts={:?} used={:?} expert_ff={:?} ff={:?}",
            file.meta_u32(&format!("{arch}.ssm.inner_size")),
            file.meta_u32(&format!("{arch}.ssm.state_size")),
            file.meta_u32(&format!("{arch}.ssm.time_step_rank")),
            file.meta_u32(&format!("{arch}.ssm.group_count")),
            file.meta_u32(&format!("{arch}.ssm.conv_kernel")),
            file.meta_u32(&format!("{arch}.expert_count")),
            file.meta_u32(&format!("{arch}.expert_used_count")),
            file.meta_u32(&format!("{arch}.expert_feed_forward_length")),
            file.meta_u32(&format!("{arch}.feed_forward_length")),
        );
        eprintln!(
            "Nemotron grouped/shared experts group={:?} group_used={:?} shared={:?} shared_ff={:?} w_norm={:?} w_scale={:?}",
            file.meta_u32(&format!("{arch}.expert_group_count")),
            file.meta_u32(&format!("{arch}.expert_group_used_count")),
            file.meta_u32(&format!("{arch}.expert_shared_count")),
            file.meta_u32(&format!("{arch}.expert_shared_feed_forward_length")),
            file.meta_bool(&format!("{arch}.expert_weights_norm")),
            file.meta_f32(&format!("{arch}.expert_weights_scale")),
        );
        for suffix in [
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_inp.weight",
            "ffn_exp_probs_b.bias",
        ] {
            if let Some(t) = file.tensor(&format!("blk.1.{suffix}")) {
                eprintln!(
                    "Nemotron blk.1.{suffix} type={:?} dims={:?}",
                    t.ggml_type, t.dims
                );
            }
        }

        // Smallest type-100 tensor must dequant to finite, non-degenerate weights.
        let smallest = file
            .tensors
            .iter()
            .filter(|t| t.ggml_type_id == 100)
            .min_by_key(|t| t.n_elements())
            .expect("Nemotron carries ROCmFP4 tensors")
            .clone();
        let values = file
            .tensor_f32(&smallest.name)
            .expect("ROCmFP4 tensor must dequant");
        assert_eq!(values.len(), smallest.n_elements() as usize);
        assert!(
            values.iter().all(|v| v.is_finite()),
            "ROCmFP4 dequant produced non-finite values in {}",
            smallest.name
        );
        let max_abs = values.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            max_abs > 0.0 && max_abs < 100.0,
            "ROCmFP4 {} max|w|={max_abs} is outside a plausible weight range",
            smallest.name
        );
        let nonzero = values.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero * 4 > values.len(),
            "ROCmFP4 {} decoded mostly zeros ({nonzero}/{}), layout is likely wrong",
            smallest.name,
            values.len()
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

    #[test]
    fn probe_local_qwen38_layouts() {
        for rel in [
            "../../models/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF/Qwen3.8-27B-Q4_0_ROCMFP4_COHERENT.gguf",
            "../../models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q5_K_XL.gguf",
        ] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
            if !path.exists() {
                eprintln!("skip probe_local_qwen38_layouts: missing {}", path.display());
                continue;
            }
            let file = GgufFile::open(&path).expect("mmap qwen");
            let report = file.compat_report();
            let arch = report.architecture.as_str();
            eprintln!(
                "QWEN layout: path={} arch={} layers={} ssm={} attn={} types={:?} reject={:?} heads={:?}/{:?} dim={:?} ff={:?}",
                path.display(),
                report.architecture,
                report.n_layers,
                report.ssm_layers,
                report.attention_layers,
                report.ggml_type_ids,
                report.reject_reason,
                file.meta_u32(&format!("{arch}.attention.head_count")),
                file.meta_u32(&format!("{arch}.attention.head_count_kv")),
                file.meta_u32(&format!("{arch}.embedding_length")),
                file.meta_u32(&format!("{arch}.feed_forward_length")),
            );
            for i in [0usize, 1, 3, 4] {
                let mut names: Vec<_> = file
                    .tensors
                    .iter()
                    .filter(|t| t.name.starts_with(&format!("blk.{i}.")))
                    .map(|t| {
                        format!(
                            "{}:{:?}:{:?}",
                            t.name.trim_start_matches(&format!("blk.{i}.")),
                            t.ggml_type,
                            t.dims
                        )
                    })
                    .collect();
                names.sort();
                eprintln!("  blk.{i} kind={:?}: {}", report.layers.get(i).map(|l| l.block), names.join(" | "));
            }
            for key in [
                "ssm.conv_kernel",
                "ssm.state_size",
                "ssm.inner_size",
                "ssm.time_step_rank",
                "ssm.group_count",
                "full_attention_interval",
                "rope.dimension_count",
                "attention.key_length",
                "attention.value_length",
                "context_length",
            ] {
                let full = format!("{arch}.{key}");
                if let Some(v) = file.meta_u32(&full) {
                    eprintln!("  meta {full}={v}");
                }
            }
            if let Some(v) = file.meta_f32(&format!("{arch}.rope.freq_base")) {
                eprintln!("  meta {arch}.rope.freq_base={v}");
            }
            if let Some(v) = file.meta_f32(&format!("{arch}.attention.layer_norm_rms_eps")) {
                eprintln!("  meta {arch}.attention.layer_norm_rms_eps={v}");
            }
            if let Some(v) = file.meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon")) {
                eprintln!("  meta {arch}.attention.layer_norm_rms_epsilon={v}");
            }
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.contains("UD-Q5_K_XL"))
            {
                assert!(
                    report.reject_reason.is_none(),
                    "UD-Q5_K_XL IQ4 extras must load; got {:?}",
                    report.reject_reason
                );
                assert!(report.ggml_type_ids.contains(&20));
                assert!(report.ggml_type_ids.contains(&23));
                let iq_nl = file
                    .tensors
                    .iter()
                    .find(|t| t.ggml_type == GgmlType::IQ4_NL)
                    .expect("type 20 tensor");
                let packed = file.tensor_bytes(&iq_nl.name).expect("IQ4_NL bytes");
                let mut head = vec![0.0f32; allpaca_cpu::IQ4_NL_BLOCK_ELEMS];
                allpaca_cpu::dequant_iq4_nl(packed, head.len(), &mut head)
                    .expect("dequant first IQ4_NL block");
                assert!(
                    head.iter().any(|v| v.abs() > 1e-6),
                    "IQ4_NL {} first block was all zeros",
                    iq_nl.name
                );
            }
            for name in ["token_embd.weight", "output.weight", "output_norm.weight"] {
                if let Some(t) = file.tensor(name) {
                    eprintln!(
                        "  tensor {name}: {:?}/{} dims={:?}",
                        t.ggml_type, t.ggml_type_id, t.dims
                    );
                }
            }
            if let Ok(a) = file.tensor_f32("blk.0.ssm_a") {
                eprintln!(
                    "  ssm_a[0..4]={:?} all_neg={}",
                    &a[..4.min(a.len())],
                    a.iter().all(|x| *x < 0.0)
                );
            }
            let mut rope_keys: Vec<_> = file
                .metadata
                .iter()
                .filter(|(k, _)| k.contains("rope") || k.contains("mrope"))
                .collect();
            rope_keys.sort_by(|a, b| a.0.cmp(b.0));
            for (k, v) in rope_keys {
                eprintln!("  rope-key {k}={v:?}");
            }
            for id in [20u32, 23] {
                let hit: Vec<_> = file
                    .tensors
                    .iter()
                    .filter(|t| t.ggml_type_id == id)
                    .take(6)
                    .map(|t| t.name.as_str())
                    .collect();
                let n = file.tensors.iter().filter(|t| t.ggml_type_id == id).count();
                if n > 0 {
                    eprintln!("  type {id} count={n} sample={hit:?}");
                }
            }
        }
    }

    #[test]
    fn compat_report_recognizes_spark2_5_architecture() {
        let path = std::env::temp_dir().join("allpaca_spark2_5_arch.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("spark2_5".into()),
        );
        meta.insert("spark2_5.block_count".into(), GgufValue::Uint32(4));
        meta.insert("spark2_5.embedding_length".into(), GgufValue::Uint32(32));
        meta.insert(
            "spark2_5.attention.head_count".into(),
            GgufValue::Uint32(4),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let report = file.compat_report();
        assert_eq!(report.architecture, "spark2_5");
        if let Some(reason) = &report.reject_reason {
            assert!(
                !reason.contains("outside the Llama-layout loader"),
                "{reason}"
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn compat_report_recognizes_qwen3_and_qwen3moe() {
        for arch in ["qwen3", "qwen3moe"] {
            let path = std::env::temp_dir().join(format!("allpaca_{arch}_arch.gguf"));
            let mut meta = HashMap::new();
            meta.insert(
                "general.architecture".into(),
                GgufValue::String(arch.into()),
            );
            meta.insert(format!("{arch}.block_count"), GgufValue::Uint32(4));
            meta.insert(format!("{arch}.embedding_length"), GgufValue::Uint32(32));
            meta.insert(
                format!("{arch}.attention.head_count"),
                GgufValue::Uint32(4),
            );
            write_gguf(&path, &meta, &[]).unwrap();
            let file = GgufFile::open(&path).unwrap();
            let report = file.compat_report();
            assert_eq!(report.architecture, arch);
            if let Some(reason) = &report.reject_reason {
                assert!(
                    !reason.contains("outside the Llama-layout loader"),
                    "{arch}: {reason}"
                );
            }
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn compat_report_recognizes_gemma_and_deepseek2() {
        for arch in ["gemma2", "gemma3", "deepseek2"] {
            let path = std::env::temp_dir().join(format!("allpaca_{arch}_arch.gguf"));
            let mut meta = HashMap::new();
            meta.insert(
                "general.architecture".into(),
                GgufValue::String(arch.into()),
            );
            meta.insert(format!("{arch}.block_count"), GgufValue::Uint32(4));
            meta.insert(format!("{arch}.embedding_length"), GgufValue::Uint32(32));
            meta.insert(
                format!("{arch}.attention.head_count"),
                GgufValue::Uint32(4),
            );
            write_gguf(&path, &meta, &[]).unwrap();
            let file = GgufFile::open(&path).unwrap();
            let report = file.compat_report();
            assert_eq!(report.architecture, arch);
            if let Some(reason) = &report.reject_reason {
                assert!(
                    !reason.contains("outside the Llama-layout loader"),
                    "{arch}: {reason}"
                );
            }
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn parse_split_filename_five_digit() {
        let (stem, idx, total) = parse_split_filename(
            "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00003-of-00005.gguf",
        )
        .expect("split name");
        assert_eq!(stem, "DeepSeek-V4-Flash-0731-UD-Q8_K_XL");
        assert_eq!(idx, 3);
        assert_eq!(total, 5);
        assert!(is_dspark_filename("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"));
        assert!(!is_dspark_filename(
            "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"
        ));
    }

    #[test]
    fn split_gguf_merges_metadata_and_tensors() {
        let dir = std::env::temp_dir().join("allpaca_split_gguf");
        let _ = std::fs::create_dir_all(&dir);
        let a = dir.join("tiny-00001-of-00002.gguf");
        let b = dir.join("tiny-00002-of-00002.gguf");
        let mut meta_a = HashMap::new();
        meta_a.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta_a.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta_a.insert("split.count".into(), GgufValue::Uint32(2));
        let emb = vec![1.0f32, 2.0, 3.0, 4.0];
        write_gguf(
            &a,
            &meta_a,
            &[GgufWriteTensor {
                name: "token_embd.weight".into(),
                dims: vec![2, 2],
                data: &emb,
            }],
        )
        .unwrap();
        let mut meta_b = HashMap::new();
        meta_b.insert("split.count".into(), GgufValue::Uint32(2));
        meta_b.insert("split.no".into(), GgufValue::Uint32(1));
        let out = vec![5.0f32, 6.0];
        write_gguf(
            &b,
            &meta_b,
            &[GgufWriteTensor {
                name: "output.weight".into(),
                dims: vec![2, 1],
                data: &out,
            }],
        )
        .unwrap();
        let file = GgufFile::open(&a).expect("open split");
        assert_eq!(file.architecture(), Some("llama"));
        assert!(file.tensor("token_embd.weight").is_some());
        assert!(file.tensor("output.weight").is_some());
        assert_eq!(file.tensor_f32("token_embd.weight").unwrap(), emb);
        assert_eq!(file.tensor_f32("output.weight").unwrap(), out);
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn bf16_and_mxfp4_tensor_f32() {
        let path = std::env::temp_dir().join("allpaca_gguf_bf16_mxfp4.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("deepseek4".into()),
        );
        meta.insert("deepseek4.block_count".into(), GgufValue::Uint32(1));
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x0000, 0xbf80, 0x4000] {
            bf16.extend_from_slice(&bits.to_le_bytes());
        }
        let mut mx = vec![0u8; allpaca_cpu::MXFP4_BLOCK_BYTES];
        mx[0] = 127;
        mx[1] = 0x01;
        write_gguf_quant(
            &path,
            &meta,
            &[],
            &[
                GgufWriteQuantTensor {
                    name: "bf.weight".into(),
                    dims: vec![4],
                    ggml_type: GgmlType::BF16,
                    bytes: &bf16,
                },
                GgufWriteQuantTensor {
                    name: "mx.weight".into(),
                    dims: vec![allpaca_cpu::MXFP4_BLOCK_ELEMS as u64],
                    ggml_type: GgmlType::MXFP4,
                    bytes: &mx,
                },
            ],
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        let bf = file.tensor_f32("bf.weight").unwrap();
        assert!((bf[0] - 1.0).abs() < 1e-5);
        assert_eq!(bf[1], 0.0);
        assert!((bf[2] + 1.0).abs() < 1e-5);
        assert!((bf[3] - 2.0).abs() < 1e-5);
        let mxv = file.tensor_f32("mx.weight").unwrap();
        assert!((mxv[0] - 0.5).abs() < 1e-5);
        let report = file.compat_report();
        assert_eq!(report.architecture, "deepseek4");
        assert!(
            report.reject_reason.is_none(),
            "deepseek4 BF16/MXFP4 must load; {:?}",
            report.reject_reason
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn discover_dspark_prefers_q8_in_nested_folder() {
        let dir = std::env::temp_dir().join(format!(
            "allpaca_dspark_discover-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let quant = dir.join("UD-Q8_K_XL");
        let nested = dir.join("dspark");
        let _ = std::fs::create_dir_all(&quant);
        let _ = std::fs::create_dir_all(&nested);
        let target = quant.join("DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf");
        let q8 = dir.join("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf");
        let bf = nested.join("dspark-DeepSeek-V4-Flash-0731-BF16.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("dflash".into()),
        );
        let w = vec![1.0f32];
        let tensor = [GgufWriteTensor {
            name: "markov_w1.weight".into(),
            dims: vec![1],
            data: &w,
        }];
        write_gguf(&q8, &meta, &tensor).unwrap();
        write_gguf(&bf, &meta, &tensor).unwrap();
        write_gguf(&target, &meta, &tensor).unwrap();
        let hit = discover_dspark_sidecar(&target).expect("sidecar");
        assert_eq!(hit.file_name().and_then(|s| s.to_str()), Some("dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
