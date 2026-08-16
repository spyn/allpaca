//! ONNX Model Format: detect files and import initializer weights.
//!
//! This is not a general ONNX Runtime. Unsupported graphs are Fatal.

mod parser;
mod proto;
mod writer;

pub use parser::{OnnxFile, OnnxNode, OnnxTensor, TENSOR_FLOAT, TENSOR_FLOAT16};
pub use writer::{write_onnx_weights, OnnxWriteNode, OnnxWriteTensor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_float_initializers_from_written_onnx() {
        let path = std::env::temp_dir().join("allpaca_tiny_onnx_init.onnx");
        let data = [1.0f32, 2.0, 3.0, 4.0];
        write_onnx_weights(
            &path,
            "tiny",
            &[OnnxWriteTensor {
                name: "w".into(),
                dims: vec![2, 2],
                data: &data,
            }],
            &[],
        )
        .unwrap();
        let file = OnnxFile::open(&path).unwrap();
        assert_eq!(file.tensor_f32("w").unwrap(), data);
        let _ = std::fs::remove_file(path);
    }
}
