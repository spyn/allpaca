//! SafeTensors weight loading and HuggingFace sidecar config/tokenizer.

mod bytes;
mod catalog;
mod names;
mod parser;
mod sidecar;
mod writer;

pub use catalog::{
    discover_checkpoint, discover_gguf_path, discover_weight_path, CheckpointKind,
    DiscoveredCheckpoint, SafetensorsCatalog,
};
pub use names::{aliases, WeightKey};
pub use parser::{SafetensorsDtype, SafetensorsFile, TensorInfo, SAFETENSORS_MAX_HEADER};
pub use sidecar::{
    load_hf_config, load_tokenizer_sidecar, load_vocab_sidecar, resolve_sidecar,
    EffectiveTextConfig, HfConfig, HfQuantOverride, HfRopeParameters, HfTextConfig,
    SidecarTokenizer, VocabSidecar,
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

    #[test]
    fn load_bf16_tensor_from_written_file() {
        let path = std::env::temp_dir().join("allpaca_tiny_bf16.safetensors");
        let src = [1.0f32, -2.5, 0.125];
        write_safetensors(
            &path,
            &[SafetensorsWriteTensor {
                name: "w_bf16".into(),
                dims: vec![3],
                dtype: SafetensorsDtype::BF16,
                data: &src,
            }],
        )
        .unwrap();
        let file = SafetensorsFile::open(&path).unwrap();
        let got = file.tensor_f32("w_bf16").unwrap();
        assert_eq!(got.len(), 3);
        for (a, b) in got.iter().zip(src.iter()) {
            assert!((a - b).abs() < 1e-2, "{a} vs {b}");
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn catalog_loads_sharded_index() {
        let dir = std::env::temp_dir().join("allpaca_st_shards");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let a = [1.0f32, 2.0];
        let b = [3.0f32, 4.0];
        write_safetensors(
            &dir.join("model-00001-of-00002.safetensors"),
            &[SafetensorsWriteTensor {
                name: "model.embed_tokens.weight".into(),
                dims: vec![2],
                dtype: SafetensorsDtype::F32,
                data: &a,
            }],
        )
        .unwrap();
        write_safetensors(
            &dir.join("model-00002-of-00002.safetensors"),
            &[SafetensorsWriteTensor {
                name: "lm_head.weight".into(),
                dims: vec![2],
                dtype: SafetensorsDtype::BF16,
                data: &b,
            }],
        )
        .unwrap();
        let index = serde_json::json!({
            "metadata": {"total_size": 16},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
        let cat = SafetensorsCatalog::open(&dir).unwrap();
        assert_eq!(cat.tensor_f32_key(WeightKey::Embed).unwrap(), a);
        let head = cat.tensor_f32_key(WeightKey::LmHead).unwrap();
        assert!((head[0] - 3.0).abs() < 1e-2);
        assert!((head[1] - 4.0).abs() < 1e-2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn discover_checkpoint_prefers_index_in_dir() {
        let dir = std::env::temp_dir().join("allpaca_st_discover");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_safetensors(
            &dir.join("model.safetensors"),
            &[SafetensorsWriteTensor {
                name: "w".into(),
                dims: vec![1],
                dtype: SafetensorsDtype::F32,
                data: &[1.0],
            }],
        )
        .unwrap();
        let found = discover_checkpoint(&dir, None).unwrap();
        assert_eq!(found.kind, CheckpointKind::Safetensors);
        assert!(found.weights.ends_with("model.safetensors"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn sidecar_detects_hf_tokenizer_json() {
        let dir = std::env::temp_dir().join("allpaca_tok_detect");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"version":"1.0","model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"a"}}"#,
        )
        .unwrap();
        match load_tokenizer_sidecar(&dir.join("tokenizer.json")).unwrap() {
            SidecarTokenizer::HuggingFace { tokenizer_json, .. } => {
                assert!(tokenizer_json.ends_with("tokenizer.json"));
            }
            SidecarTokenizer::Vocab(_) => {
                panic!("HF tokenizer.json must not parse as vocab sidecar")
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }
}
