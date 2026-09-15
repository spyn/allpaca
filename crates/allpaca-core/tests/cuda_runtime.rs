//! Public Engine behaviors for the CUDA decode path.
//! GPU-gated tests skip when `device_count() == 0` or runtime is not linked.

#![cfg(feature = "cuda")]

use allpaca_core::Engine;
use allpaca_types::{Config, GenerationParams};

fn cuda_synthetic() -> Config {
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = true;
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "cuda".into();
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg
}

fn greedy(max_tokens: usize) -> GenerationParams {
    GenerationParams {
        max_tokens,
        temperature: 0.0,
        seed: Some(7),
        ..GenerationParams::default()
    }
}

fn cuda_device_available() -> bool {
    allpaca_cuda::runtime_linked() && allpaca_cuda::device_count() > 0
}

#[test]
fn cuda_synthetic_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "embed/layer weights should upload at load"
    );
    assert!(
        before.weight_bytes_on_device > 0,
        "weights should be resident after load"
    );
    let h2d_after_load = before.activation_h2d_bytes;

    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());

    let stats = engine.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    // Weight H2D is load-time only; decode must not re-upload the full pack each token.
    assert_eq!(
        stats.weight_h2d_bytes, before.weight_h2d_bytes,
        "weight_h2d must not grow during generate"
    );
    let activation_delta = stats.activation_h2d_bytes.saturating_sub(h2d_after_load);
    assert!(
        activation_delta < before.weight_h2d_bytes,
        "per-generate activation H2D ({activation_delta}) must stay below weight pack ({})",
        before.weight_h2d_bytes
    );
}

#[test]
fn cuda_synthetic_moe_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false; // load via load_synthetic below
    let engine = Engine::new(cfg).expect("engine");
    engine
        .load_synthetic(allpaca_core::ModelConfig::tiny_moe())
        .expect("load tiny_moe");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "MoE experts must upload to CUDA"
    );
    let out = engine.generate("hi", greedy(2)).expect("moe generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE forwards, got gpu={} target={} (CPU fallback?)",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_hybrid_attention_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    let engine = Engine::new(cfg).expect("engine");
    engine
        .load_synthetic(allpaca_core::ModelConfig::tiny_hybrid())
        .expect("load tiny_hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(before.weight_bytes_on_device > 0);
    let out = engine.generate("hi", greedy(2)).expect("hybrid generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA hybrid forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_absent_falls_back_to_cpu_cleanly() {
    if cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "cpu");
    assert_eq!(stats.gpu_forwards, 0);
    assert_eq!(stats.weight_h2d_bytes, 0);
}

fn write_tiny_llama_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let w = 0.01f32;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert(
        "llama.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
    meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
    meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
    meta.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert(
        "tokenizer.ggml.model".into(),
        GgufValue::String("wordlevel".into()),
    );
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(vec![
            GgufValue::String("<unk>".into()),
            GgufValue::String("<s>".into()),
            GgufValue::String("</s>".into()),
            GgufValue::String("hello".into()),
        ]),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
    let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
        (
            "token_embd.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ),
        (
            "blk.0.attn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.attn_q.weight".into(),
            vec![dim as u64, dim as u64],
            vec![w; dim * dim],
        ),
        (
            "blk.0.attn_k.weight".into(),
            vec![2, dim as u64],
            vec![w; 2 * dim],
        ),
        (
            "blk.0.attn_v.weight".into(),
            vec![2, dim as u64],
            vec![w; 2 * dim],
        ),
        (
            "blk.0.attn_output.weight".into(),
            vec![dim as u64, dim as u64],
            vec![w; dim * dim],
        ),
        (
            "blk.0.ffn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.ffn_gate.weight".into(),
            vec![ff as u64, dim as u64],
            vec![w; ff * dim],
        ),
        (
            "blk.0.ffn_up.weight".into(),
            vec![ff as u64, dim as u64],
            vec![w; ff * dim],
        ),
        (
            "blk.0.ffn_down.weight".into(),
            vec![dim as u64, ff as u64],
            vec![w; dim * ff],
        ),
        ("output_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
    ];
    let views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    write_gguf(path, &meta, &views).expect("write tiny gguf");
}

fn cuda_gguf_config(path: &std::path::Path) -> Config {
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.to_path_buf());
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "cuda".into();
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg
}

/// Tiny host-materialized GGUF must upload to CUDA and decode on device.
#[test]
fn cuda_tiny_gguf_uploads_and_generates_when_device_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca-cuda-tiny-{}.gguf",
        std::process::id()
    ));
    write_tiny_llama_gguf(&path);
    struct Remove(std::path::PathBuf);
    impl Drop for Remove {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Remove(path.clone());
    let engine = Engine::new(cuda_gguf_config(&path)).expect("engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "GGUF weights should be CUDA-resident after load"
    );
    assert!(before.weight_h2d_bytes > 0);
    let out = engine.generate("hello", greedy(2)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA forwards on tiny GGUF"
    );
}

/// Gated real GGUF (1B/3B/7B). Skips unless `ALLPACA_7B_GGUF` / `ALLPACA_GGUF`
/// or a discovered local `models/Llama-3.2-*-Instruct-*.gguf` exists.
/// Large GGUFs skip host F32 materialize; CUDA-only must still upload+generate.
#[test]
fn cuda_generate_real_gguf_when_path_set() {
    if !cuda_device_available() {
        return;
    }
    let Some(path) = gated_cuda_gguf() else {
        eprintln!("skip cuda_generate_real_gguf_when_path_set: set ALLPACA_7B_GGUF or place models/Llama-3.2-*-Instruct-*.gguf");
        return;
    };
    let engine = Engine::new(cuda_gguf_config(&path)).expect("engine load GGUF on CUDA");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "expected CUDA weight upload for {}",
        path.display()
    );
    // Packed Q8 must be far below F32 blow-up (~4× element bytes).
    if path
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|n| n.contains("Q8_0"))
    {
        let file_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        assert!(
            before.weight_bytes_on_device < file_len.saturating_mul(3),
            "Q8 CUDA upload should stay near packed size, not F32: weights={} file={}",
            before.weight_bytes_on_device,
            file_len
        );
    }
    let out = engine.generate("hi", greedy(1)).expect("generate");
    assert_eq!(out.token_ids.len(), 1);
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA forwards"
    );
}

fn gated_cuda_gguf() -> Option<std::path::PathBuf> {
    for key in ["ALLPACA_7B_GGUF", "ALLPACA_GGUF", "ALLPACA_CUDA_GGUF"] {
        if let Ok(raw) = std::env::var(key) {
            let path = std::path::PathBuf::from(raw);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let local = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
    for name in [
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Llama-3.2-1B-Instruct-F16.gguf",
    ] {
        let p = local.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}
