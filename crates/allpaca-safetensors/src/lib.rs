//! SafeTensors weight loading and HuggingFace sidecar config/tokenizer.

mod parser;
mod sidecar;
mod writer;

pub use parser::{SafetensorsDtype, SafetensorsFile, TensorInfo, SAFETENSORS_MAX_HEADER};
pub use sidecar::{
    load_hf_config, load_vocab_sidecar, resolve_sidecar, HfConfig, VocabSidecar,
};
pub use writer::{write_safetensors, SafetensorsWriteTensor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_f32_and_f16_tensors_from_written_file() {
        let path = std::env::temp_dir().join("allpaca_tiny_f32_f16.safetensors");
        let f32_data = [1.0f32, 2.0, 3.0, 4.0];
        let f16_src = [0.5f32, -1.0];
        write_safetensors(
            &path,
            &[
                SafetensorsWriteTensor {
                    name: "w_f32".into(),
                    dims: vec![2, 2],
                    dtype: SafetensorsDtype::F32,
                    data: &f32_data,
                },
                SafetensorsWriteTensor {
                    name: "w_f16".into(),
                    dims: vec![2],
                    dtype: SafetensorsDtype::F16,
                    data: &f16_src,
                },
            ],
        )
        .unwrap();
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(file.tensor_f32("w_f32").unwrap(), f32_data);
        let got = file.tensor_f32("w_f16").unwrap();
        assert_eq!(got.len(), 2);
        assert!((got[0] - 0.5).abs() < 1e-3);
        assert!((got[1] + 1.0).abs() < 1e-3);
        let _ = std::fs::remove_file(path);
    }
}
