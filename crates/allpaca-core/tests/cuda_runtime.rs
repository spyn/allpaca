//! Public Engine behaviors for the CUDA decode path.
//! GPU-gated tests skip when `device_count() == 0` or runtime is not linked.

#![cfg(feature = "cuda")]

use allpaca_core::Engine;
use allpaca_types::{Config, GenerationParams};
use std::path::{Path, PathBuf};

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

fn cuda_gguf(path: PathBuf) -> Config {
    let mut cfg = Config::compiled_defaults();
    cfg.model.path = Some(path);
    cfg.model.synthetic = false;
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "cuda".into();
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 8 * 1024;
    cfg.kv.dram_bytes = 64 * 1024;
    cfg
}

fn tiny_llama_meta() -> std::collections::HashMap<String, allpaca_gguf::GgufValue> {
    use allpaca_gguf::GgufValue;
    use std::collections::HashMap;

    let dim = 8u32;
    let vocab = 4u32;
    let ff = 16u32;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim));
    meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
    meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
    meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
    meta.insert("llama.feed_forward_length".into(), GgufValue::Uint32(ff));
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
    meta
}

/// Tiny llama-family F32 GGUF (dim=8, vocab=4, 1 layer). Host-materializes so
/// `CudaResidentModel::upload` can run without packed CUDA quant GEMM.
fn write_tiny_f32_gguf(path: &Path) {
    use allpaca_gguf::{write_gguf, GgufWriteTensor};

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let meta = tiny_llama_meta();
    let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
        (
            "token_embd.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![0.01; vocab * dim],
        ),
        (
            "blk.0.attn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.attn_q.weight".into(),
            vec![dim as u64, dim as u64],
            vec![0.01; dim * dim],
        ),
        (
            "blk.0.attn_k.weight".into(),
            vec![2, dim as u64],
            vec![0.01; 2 * dim],
        ),
        (
            "blk.0.attn_v.weight".into(),
            vec![2, dim as u64],
            vec![0.01; 2 * dim],
        ),
        (
            "blk.0.attn_output.weight".into(),
            vec![dim as u64, dim as u64],
            vec![0.01; dim * dim],
        ),
        (
            "blk.0.ffn_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
        (
            "blk.0.ffn_gate.weight".into(),
            vec![ff as u64, dim as u64],
            vec![0.01; ff * dim],
        ),
        (
            "blk.0.ffn_up.weight".into(),
            vec![ff as u64, dim as u64],
            vec![0.01; ff * dim],
        ),
        (
            "blk.0.ffn_down.weight".into(),
            vec![dim as u64, ff as u64],
            vec![0.01; dim * ff],
        ),
        (
            "output_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ),
    ];
    let views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    write_gguf(path, &meta, &views).expect("write tiny F32 GGUF");
}

/// Tiny packed GGUF (Q4_K / F16). Host dequants under `HOST_F32_LIMIT`, then CUDA F32 upload.
fn write_tiny_quant_gguf(
    path: &Path,
    quantize: fn(&[f32]) -> Vec<u8>,
    ggml_type: allpaca_gguf::GgmlType,
) {
    use allpaca_gguf::{write_gguf_quant, GgufWriteQuantTensor, GgufWriteTensor};

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let meta = tiny_llama_meta();
    let embd_q = quantize(&vec![0.01f32; vocab * dim]);
    let q_q = quantize(&vec![0.01f32; dim * dim]);
    let k_q = quantize(&vec![0.01f32; 2 * dim]);
    let v_q = quantize(&vec![0.01f32; 2 * dim]);
    let o_q = quantize(&vec![0.01f32; dim * dim]);
    let gate_q = quantize(&vec![0.01f32; ff * dim]);
    let up_q = quantize(&vec![0.01f32; ff * dim]);
    let down_q = quantize(&vec![0.01f32; dim * ff]);
    let attn_norm = vec![1.0f32; dim];
    let ffn_norm = vec![1.0f32; dim];
    let output_norm = vec![1.0f32; dim];
    write_gguf_quant(
        path,
        &meta,
        &[
            GgufWriteTensor {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &attn_norm,
            },
            GgufWriteTensor {
                name: "blk.0.ffn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &ffn_norm,
            },
            GgufWriteTensor {
                name: "output_norm.weight".into(),
                dims: vec![dim as u64],
                data: &output_norm,
            },
        ],
        &[
            GgufWriteQuantTensor {
                name: "token_embd.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type,
                bytes: &down_q,
            },
        ],
    )
    .expect("write tiny packed GGUF");
}

fn write_tiny_q4k_gguf(path: &Path) {
    write_tiny_quant_gguf(
        path,
        allpaca_gguf::quantize_q4_k,
        allpaca_gguf::GgmlType::Q4_K,
    );
}

fn write_tiny_f16_gguf(path: &Path) {
    write_tiny_quant_gguf(
        path,
        allpaca_gguf::quantize_f16,
        allpaca_gguf::GgmlType::F16,
    );
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

#[test]
fn cuda_synthetic_moe_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    engine
        .load_synthetic(allpaca_core::ModelConfig::tiny_moe())
        .expect("tiny moe");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "MoE expert weights should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    assert_eq!(
        stats.weight_h2d_bytes, before.weight_h2d_bytes,
        "weight_h2d must not grow during MoE generate"
    );
}

#[test]
fn cuda_synthetic_moe_matches_cpu_when_present() {
    if !cuda_device_available() {
        return;
    }
    let mut cpu_cfg = cuda_synthetic();
    cpu_cfg.engine.cpu_only = true;
    cpu_cfg.engine.gpu_backend = "cpu".into();
    let cpu = Engine::new(cpu_cfg).expect("cpu engine");
    cpu.load_synthetic(allpaca_core::ModelConfig::tiny_moe())
        .expect("cpu moe");
    let want = cpu.generate("hello", greedy(4)).expect("cpu generate");

    let gpu = Engine::new(cuda_synthetic()).expect("cuda engine");
    gpu.load_synthetic(allpaca_core::ModelConfig::tiny_moe())
        .expect("cuda moe");
    let got = gpu.generate("hello", greedy(4)).expect("cuda generate");
    assert_eq!(
        got.token_ids, want.token_ids,
        "CUDA MoE greedy tokens must match CPU"
    );
}

#[test]
fn cuda_synthetic_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    engine
        .load_synthetic(allpaca_core::ModelConfig::tiny_hybrid())
        .expect("tiny hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "hybrid TinyModel should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
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
fn cuda_tiny_f32_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_f32.gguf");
    write_tiny_f32_gguf(&path);
    let engine = match Engine::new(cuda_gguf(path.clone())) {
        Ok(e) if e.stats().backend == "cuda" => e,
        Ok(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("tiny F32 GGUF must load on CUDA: {e}");
        }
    };
    let before = engine.stats();
    assert!(
        before.weight_bytes_on_device > 0,
        "F32 GGUF weights should upload; stats={before:?}"
    );
    let h2d_after_load = before.weight_h2d_bytes;
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny F32 CUDA generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GGUF forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    assert_eq!(
        stats.weight_h2d_bytes, h2d_after_load,
        "weight_h2d must not grow during GGUF generate"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cuda_tiny_f32_gguf_matches_cpu_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_f32_match.gguf");
    write_tiny_f32_gguf(&path);
    let mut cpu_cfg = cuda_gguf(path.clone());
    cpu_cfg.engine.cpu_only = true;
    cpu_cfg.engine.gpu_backend = "cpu".into();
    let cpu = Engine::new(cpu_cfg).expect("cpu gguf engine");
    let want = cpu.generate("hello", greedy(4)).expect("cpu generate");
    let gpu = match Engine::new(cuda_gguf(path.clone())) {
        Ok(e) if e.stats().backend == "cuda" => e,
        Ok(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("tiny F32 GGUF must load on CUDA: {e}");
        }
    };
    let got = gpu.generate("hello", greedy(4)).expect("cuda generate");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        got.token_ids, want.token_ids,
        "CUDA F32 GGUF greedy tokens must match CPU"
    );
}

#[test]
fn cuda_tiny_f32_gguf_absent_falls_back_to_cpu() {
    if cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_f32_cpu.gguf");
    write_tiny_f32_gguf(&path);
    let engine = Engine::new(cuda_gguf(path.clone())).expect("engine");
    let out = engine.generate("hello", greedy(4)).expect("generate");
    let _ = std::fs::remove_file(&path);
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "cpu");
    assert_eq!(stats.gpu_forwards, 0);
    assert_eq!(stats.weight_h2d_bytes, 0);
}

#[test]
fn cuda_tiny_q4k_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_q4k.gguf");
    write_tiny_q4k_gguf(&path);
    let engine = match Engine::new(cuda_gguf(path.clone())) {
        Ok(e) if e.stats().backend == "cuda" => e,
        Ok(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("tiny Q4_K GGUF must load on CUDA: {e}");
        }
    };
    let before = engine.stats();
    assert!(
        before.weight_bytes_on_device > 0,
        "Q4_K GGUF should host-dequant and upload; stats={before:?}"
    );
    let h2d_after_load = before.weight_h2d_bytes;
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny Q4_K CUDA generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA Q4_K forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    assert_eq!(
        stats.weight_h2d_bytes, h2d_after_load,
        "weight_h2d must not grow during Q4_K generate"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cuda_tiny_q4k_gguf_matches_cpu_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_q4k_match.gguf");
    write_tiny_q4k_gguf(&path);
    let mut cpu_cfg = cuda_gguf(path.clone());
    cpu_cfg.engine.cpu_only = true;
    cpu_cfg.engine.gpu_backend = "cpu".into();
    let cpu = Engine::new(cpu_cfg).expect("cpu q4k engine");
    let want = cpu.generate("hello", greedy(4)).expect("cpu generate");
    let gpu = match Engine::new(cuda_gguf(path.clone())) {
        Ok(e) if e.stats().backend == "cuda" => e,
        Ok(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("tiny Q4_K GGUF must load on CUDA: {e}");
        }
    };
    let got = gpu.generate("hello", greedy(4)).expect("cuda generate");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        got.token_ids, want.token_ids,
        "CUDA Q4_K GGUF greedy tokens must match CPU"
    );
}

#[test]
fn cuda_tiny_f16_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_f16.gguf");
    write_tiny_f16_gguf(&path);
    let engine = match Engine::new(cuda_gguf(path.clone())) {
        Ok(e) if e.stats().backend == "cuda" => e,
        Ok(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("tiny F16 GGUF must load on CUDA: {e}");
        }
    };
    let before = engine.stats();
    assert!(
        before.weight_bytes_on_device > 0,
        "F16 GGUF should host-dequant and upload; stats={before:?}"
    );
    let h2d_after_load = before.weight_h2d_bytes;
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny F16 CUDA generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA F16 forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    assert_eq!(
        stats.weight_h2d_bytes, h2d_after_load,
        "weight_h2d must not grow during F16 generate"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cuda_tiny_f16_gguf_matches_cpu_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_f16_match.gguf");
    write_tiny_f16_gguf(&path);
    let mut cpu_cfg = cuda_gguf(path.clone());
    cpu_cfg.engine.cpu_only = true;
    cpu_cfg.engine.gpu_backend = "cpu".into();
    let cpu = Engine::new(cpu_cfg).expect("cpu f16 engine");
    let want = cpu.generate("hello", greedy(4)).expect("cpu generate");
    let gpu = match Engine::new(cuda_gguf(path.clone())) {
        Ok(e) if e.stats().backend == "cuda" => e,
        Ok(_) => {
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("tiny F16 GGUF must load on CUDA: {e}");
        }
    };
    let got = gpu.generate("hello", greedy(4)).expect("cuda generate");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        got.token_ids, want.token_ids,
        "CUDA F16 GGUF greedy tokens must match CPU"
    );
}

#[test]
fn cuda_tiny_f16_gguf_absent_falls_back_to_cpu() {
    if cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_tiny_f16_cpu.gguf");
    write_tiny_f16_gguf(&path);
    let engine = Engine::new(cuda_gguf(path.clone())).expect("engine");
    let out = engine.generate("hello", greedy(4)).expect("generate");
    let _ = std::fs::remove_file(&path);
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "cpu");
    assert_eq!(stats.gpu_forwards, 0);
    assert_eq!(stats.weight_h2d_bytes, 0);
}
