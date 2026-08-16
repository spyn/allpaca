use std::path::Path;

use allpaca_types::AllpacaError;

use crate::parser::TENSOR_FLOAT;
use crate::proto::{write_int64, write_len_bytes, write_string};

pub struct OnnxWriteTensor<'a> {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: &'a [f32],
}

pub struct OnnxWriteNode {
    pub op_type: String,
    pub name: String,
}

pub fn write_onnx_weights(
    path: &Path,
    graph_name: &str,
    tensors: &[OnnxWriteTensor<'_>],
    nodes: &[OnnxWriteNode],
) -> Result<(), AllpacaError> {
    let mut graph = Vec::new();
    for n in nodes {
        let mut node = Vec::new();
        write_string(&mut node, 3, &n.name);
        write_string(&mut node, 4, &n.op_type);
        write_len_bytes(&mut graph, 1, &node);
    }
    write_string(&mut graph, 2, graph_name);
    for t in tensors {
        write_len_bytes(&mut graph, 5, &encode_tensor(t));
    }

    let mut model = Vec::new();
    write_int64(&mut model, 1, 8);
    write_string(&mut model, 2, "allpaca");
    write_len_bytes(&mut model, 7, &graph);
    let mut opset = Vec::new();
    write_int64(&mut opset, 2, 18);
    write_len_bytes(&mut model, 8, &opset);
    std::fs::write(path, model)?;
    Ok(())
}

fn encode_tensor(t: &OnnxWriteTensor<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for d in &t.dims {
        write_int64(&mut out, 1, *d);
    }
    write_int64(&mut out, 2, i64::from(TENSOR_FLOAT));
    write_string(&mut out, 8, &t.name);
    let mut raw = Vec::with_capacity(t.data.len() * 4);
    for v in t.data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    write_len_bytes(&mut out, 9, &raw);
    out
}
