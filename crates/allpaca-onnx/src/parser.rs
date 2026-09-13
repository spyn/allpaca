use std::collections::HashMap;
use std::path::Path;

use allpaca_types::{AllpacaError, FatalError};

use crate::proto::{onnx_err, ProtoCursor};

pub const TENSOR_FLOAT: i32 = 1;
pub const TENSOR_FLOAT16: i32 = 10;

#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub data_type: i32,
    pub data: Vec<f32>,
}

pub struct OnnxFile {
    pub ir_version: i64,
    pub producer_name: String,
    pub graph_name: String,
    pub nodes: Vec<OnnxNode>,
    pub initializers: HashMap<String, OnnxTensor>,
}

#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
}

impl OnnxFile {
    pub fn open(path: &Path) -> Result<Self, AllpacaError> {
        let bytes = std::fs::read(path).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "ONNX {}: {e}",
                path.display()
            )))
        })?;
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, AllpacaError> {
        if bytes.is_empty() {
            return Err(onnx_err("empty file"));
        }
        let mut cur = ProtoCursor::new(bytes);
        let mut ir_version = None;
        let mut producer_name = String::new();
        let mut graph = None;
        while cur.remaining() > 0 {
            let (field, wire) = cur.read_tag()?;
            match (field, wire) {
                (1, 0) => ir_version = Some(cur.read_varint()? as i64),
                (2, 2) => {
                    producer_name = String::from_utf8_lossy(cur.read_bytes()?).into_owned();
                }
                (7, 2) => {
                    graph = Some(parse_graph(cur.read_bytes()?)?);
                }
                (_, w) => cur.skip_field(w)?,
            }
        }
        let ir_version = ir_version.ok_or_else(|| onnx_err("missing ir_version"))?;
        let ParsedGraph {
            name: graph_name,
            nodes,
            initializers,
        } = graph.unwrap_or_default();
        Ok(Self {
            ir_version,
            producer_name,
            graph_name,
            nodes,
            initializers,
        })
    }

    /// Weight-export graphs only: no compute ops (or only Identity/Constant).
    pub fn ensure_weight_export(&self) -> Result<(), AllpacaError> {
        const ALLOWED: &[&str] = &["Identity", "Constant"];
        for node in &self.nodes {
            if !ALLOWED.iter().any(|op| *op == node.op_type) {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "unsupported ONNX op {} (weight-export graphs only; not a general ONNX runtime)",
                    node.op_type
                ))));
            }
        }
        Ok(())
    }

    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, AllpacaError> {
        let t = self.initializers.get(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "ONNX missing initializer {name}"
            )))
        })?;
        Ok(t.data.clone())
    }
}

#[derive(Default)]
struct ParsedGraph {
    name: String,
    nodes: Vec<OnnxNode>,
    initializers: HashMap<String, OnnxTensor>,
}

fn parse_graph(bytes: &[u8]) -> Result<ParsedGraph, AllpacaError> {
    let mut cur = ProtoCursor::new(bytes);
    let mut name = String::new();
    let mut nodes = Vec::new();
    let mut initializers = HashMap::new();
    while cur.remaining() > 0 {
        let (field, wire) = cur.read_tag()?;
        match (field, wire) {
            (1, 2) => nodes.push(parse_node(cur.read_bytes()?)?),
            (2, 2) => name = String::from_utf8_lossy(cur.read_bytes()?).into_owned(),
            (5, 2) => {
                let t = parse_tensor(cur.read_bytes()?)?;
                initializers.insert(t.name.clone(), t);
            }
            (_, w) => cur.skip_field(w)?,
        }
    }
    Ok(ParsedGraph {
        name,
        nodes,
        initializers,
    })
}

fn parse_node(bytes: &[u8]) -> Result<OnnxNode, AllpacaError> {
    let mut cur = ProtoCursor::new(bytes);
    let mut name = String::new();
    let mut op_type = String::new();
    while cur.remaining() > 0 {
        let (field, wire) = cur.read_tag()?;
        match (field, wire) {
            (3, 2) => name = String::from_utf8_lossy(cur.read_bytes()?).into_owned(),
            (4, 2) => op_type = String::from_utf8_lossy(cur.read_bytes()?).into_owned(),
            (_, w) => cur.skip_field(w)?,
        }
    }
    Ok(OnnxNode { op_type, name })
}

fn parse_tensor(bytes: &[u8]) -> Result<OnnxTensor, AllpacaError> {
    let mut cur = ProtoCursor::new(bytes);
    let mut dims = Vec::new();
    let mut data_type = 0i32;
    let mut name = String::new();
    let mut raw = Vec::new();
    let mut float_data = Vec::new();
    while cur.remaining() > 0 {
        let (field, wire) = cur.read_tag()?;
        match (field, wire) {
            (1, 0) => dims.push(cur.read_varint()? as i64),
            (2, 0) => data_type = cur.read_varint()? as i32,
            (4, 5) => float_data.push(cur.read_f32_le()?),
            (4, 2) => {
                let packed = cur.read_bytes()?;
                for chunk in packed.chunks_exact(4) {
                    float_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (8, 2) => name = String::from_utf8_lossy(cur.read_bytes()?).into_owned(),
            (9, 2) => raw = cur.read_bytes()?.to_vec(),
            (_, w) => cur.skip_field(w)?,
        }
    }
    let data = if !float_data.is_empty() {
        float_data
    } else {
        decode_raw(data_type, &raw, &name)?
    };
    Ok(OnnxTensor {
        name,
        dims,
        data_type,
        data,
    })
}

fn decode_raw(data_type: i32, raw: &[u8], name: &str) -> Result<Vec<f32>, AllpacaError> {
    match data_type {
        TENSOR_FLOAT => {
            if raw.len() % 4 != 0 {
                return Err(onnx_err(&format!("tensor {name} F32 raw size")));
            }
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }
        TENSOR_FLOAT16 => {
            if raw.len() % 2 != 0 {
                return Err(onnx_err(&format!("tensor {name} F16 raw size")));
            }
            Ok(raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect())
        }
        other => Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "unsupported ONNX dtype {other} for {name}"
        )))),
    }
}

fn f16_to_f32(h: u16) -> f32 {
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
