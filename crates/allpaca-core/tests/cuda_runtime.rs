//! Public Engine behaviors for the CUDA decode path.
//! GPU-gated tests skip when `device_count() == 0` or runtime is not linked.

#![cfg(feature = "cuda")]

use allpaca_core::Engine;
use allpaca_types::{Config, GenerationParams, StepResult};

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
fn cuda_live_request_kv_pages_are_device_resident() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let mut handle = engine.begin_request("hello", greedy(4)).expect("begin");
    assert!(matches!(
        engine.step(&mut handle).expect("step"),
        StepResult::Token(_) | StepResult::Finished(_) | StepResult::Prefill { .. }
    ));
    let stats = engine.stats();
    assert!(
        stats.device_memory_bytes > 0,
        "CUDA workspace must report device memory; got {}",
        stats.device_memory_bytes
    );
    assert!(
        stats.kv_bytes_on_device > 0,
        "live Request KV pages must be cudaMalloc, not host Vecs; kv_bytes_on_device={}",
        stats.kv_bytes_on_device
    );
    drop(handle);
}

#[test]
fn cuda_device_attention_spans_multiple_kv_pages() {
    if !cuda_device_available() {
        return;
    }
    let mut cfg = cuda_synthetic();
    cfg.kv.page_tokens = 2;
    let engine = Engine::new(cfg).expect("engine");
    let mut handle = engine.begin_request("hello", greedy(8)).expect("begin");
    for _ in 0..8 {
        match engine.step(&mut handle).expect("step") {
            StepResult::Finished(_) => break,
            _ => {}
        }
    }
    let stats = engine.stats();
    assert!(
        stats.kv_bytes_on_device > 0,
        "multi-page CUDA KV must stay on device; stats={stats:?}"
    );
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "multi-page decode must use CUDA attention; stats={stats:?}"
    );
    drop(handle);
}

#[test]
fn cuda_qkv_fusion_keeps_qkv_on_device() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let after_load = engine.stats().activation_h2d_bytes;
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    let forwards = stats.gpu_forwards.max(stats.target_gpu_forwards);
    assert!(
        forwards > 0,
        "expected CUDA forwards, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    let cfg = allpaca_core::ModelConfig::tiny();
    let dim = cfg.dim as u64;
    let n_layers = cfg.n_layers as u64;
    let activation_delta = stats.activation_h2d_bytes.saturating_sub(after_load);
    // Host QKV H2Ds GEMM A (×3), Q for attention, and K/V slot writes.
    // Device fusion uploads residual once per attn layer; FFN still host-A.
    let ceiling = forwards * n_layers * dim * 4 * 8;
    assert!(
        activation_delta < ceiling,
        "QKV must stay on device after GEMM; activation H2D {activation_delta} >= ceiling {ceiling} (forwards={forwards})"
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
fn cuda_synthetic_ssm_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    let engine = Engine::new(cfg).expect("engine");
    let seed = 7u64;
    engine
        .load_tiny_model(
            allpaca_core::TinyModel::synthetic_ssm_hybrid(seed, "synthetic-ssm-hybrid"),
            "synthetic-ssm-hybrid",
        )
        .expect("load ssm hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "SSM hybrid attention/FFN weights must upload"
    );
    let out = engine
        .generate("hi", greedy(2))
        .expect("ssm hybrid generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA SSM hybrid forwards, got gpu={} target={} (CPU fallback?)",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_gdn_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    let engine = Engine::new(cfg).expect("engine");
    engine
        .load_tiny_model(
            allpaca_core::TinyModel::synthetic_gdn_hybrid(7, "synthetic-gdn-hybrid"),
            "synthetic-gdn-hybrid",
        )
        .expect("load gdn hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "GDN hybrid attention/FFN weights must upload"
    );
    let out = engine
        .generate("hi", greedy(2))
        .expect("gdn hybrid generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GDN hybrid forwards, got gpu={} target={} (CPU fallback?)",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_gdn_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_gdn.gguf");
    write_tiny_qwen_gdn_gguf(&path);
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.clone());
    cfg.kv.page_tokens = 8;
    let engine = Engine::new(cfg).expect("GDN GGUF engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "GDN GGUF weights must upload to CUDA"
    );
    let out = engine.generate("hi", greedy(2)).expect("gdn gguf generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GDN GGUF forwards; stats={stats:?}"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_gdn_q8_gguf_skips_host_f32_projections() {
    let path = std::env::temp_dir().join("allpaca_cuda_gdn_q8_host.gguf");
    write_tiny_qwen_gdn_q8_gguf(&path);
    let file = allpaca_gguf::GgufFile::open(&path).expect("open gdn q8");
    let model = allpaca_core::TinyModel::from_gguf(&file).expect("gdn q8 model");
    assert!(
        model.layers.iter().any(|l| l
            .gdn
            .as_ref()
            .is_some_and(|g| g.w_qkv.is_empty() && g.w_out.is_empty())),
        "Q8 GDN projections must stay off host F32"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_upload_gguf_gdn_q8_device_projections_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_gdn_q8_upload.gguf");
    write_tiny_qwen_gdn_q8_gguf(&path);
    let file = allpaca_gguf::GgufFile::open(&path).expect("open gdn q8");
    let cfg = allpaca_core::ModelConfig::from_gguf(&file).expect("cfg");
    let resident = allpaca_core::CudaResidentModel::upload_gguf(&file, &cfg, 0)
        .expect("CUDA GDN Q8 upload_gguf");
    assert!(
        resident.has_gdn_device_projections(),
        "Q8 GDN projections must live on device"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_gdn_q8_gguf_generate_without_host_f32_projections() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_gdn_q8_gen.gguf");
    write_tiny_qwen_gdn_q8_gguf(&path);
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.clone());
    cfg.kv.page_tokens = 8;
    let engine = Engine::new(cfg).expect("GDN Q8 GGUF engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "Q8 GDN GGUF weights must upload to CUDA"
    );
    let out = engine
        .generate("hi", greedy(2))
        .expect("gdn q8 gguf generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GDN Q8 GGUF forwards; stats={stats:?}"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_upload_gguf_gdn_q8_projections_stay_packed_when_n_exceeds_dequant() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_gdn_q8_packed.gguf");
    write_qwen_gdn_q8_packed_gguf(&path);
    let file = allpaca_gguf::GgufFile::open(&path).expect("open packed gdn q8");
    let qkv = file.tensor("blk.0.attn_qkv.weight").expect("qkv");
    assert!(
        qkv.n_elements() > 65_536,
        "fixture qkv must exceed F32 dequant threshold, got {}",
        qkv.n_elements()
    );
    let cfg = allpaca_core::ModelConfig::from_gguf(&file).expect("cfg");
    let resident = allpaca_core::CudaResidentModel::upload_gguf(&file, &cfg, 0)
        .expect("CUDA packed GDN Q8 upload_gguf");
    assert!(
        resident.has_q8_gdn_projections(),
        "large Q8 GDN projections must stay packed, not F32-dequant"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_gdn_q8_packed_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_gdn_q8_packed_gen.gguf");
    write_qwen_gdn_q8_packed_gguf(&path);
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.clone());
    cfg.kv.page_tokens = 8;
    let engine = Engine::new(cfg).expect("packed GDN Q8 engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "packed Q8 GDN weights must upload to CUDA"
    );
    let out = engine
        .generate("hi", greedy(2))
        .expect("packed gdn q8 generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA packed GDN Q8 forwards; stats={stats:?}"
    );
    let _ = std::fs::remove_file(path);
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
fn cuda_moe_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_moe.gguf");
    write_tiny_moe_gguf(&path);
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.clone());
    cfg.kv.page_tokens = 8;
    let engine = Engine::new(cfg).expect("MoE GGUF engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_bytes_on_device > 0,
        "MoE GGUF weights must upload to CUDA"
    );
    let out = engine
        .generate("hello", greedy(2))
        .expect("moe gguf generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE GGUF forwards; stats={stats:?}"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_upload_gguf_moe_packs_stacked_experts_when_device_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_moe_upload.gguf");
    write_tiny_moe_gguf(&path);
    let file = allpaca_gguf::GgufFile::open(&path).expect("open moe gguf");
    let cfg = allpaca_core::ModelConfig::from_gguf(&file).expect("cfg");
    assert!(cfg.n_experts >= 4, "fixture must be MoE");
    let resident =
        allpaca_core::CudaResidentModel::upload_gguf(&file, &cfg, 0).expect("CUDA MoE upload_gguf");
    assert!(
        resident.bytes() > 0 && resident.packed_h2d_bytes() > 0,
        "stacked experts must H2D"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_upload_gguf_moe_q8_experts_stay_packed_when_device_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_moe_q8.gguf");
    write_tiny_moe_q8_gguf(&path);
    let file = allpaca_gguf::GgufFile::open(&path).expect("open moe q8");
    let cfg = allpaca_core::ModelConfig::from_gguf(&file).expect("cfg");
    assert!(cfg.n_experts >= 4);
    let resident = allpaca_core::CudaResidentModel::upload_gguf(&file, &cfg, 0)
        .expect("CUDA MoE Q8 upload_gguf");
    assert!(
        resident.has_q8_moe_experts(),
        "stacked Q8 experts must stay packed, not F32-dequant"
    );
    assert!(resident.bytes() > 0);
    let _ = std::fs::remove_file(path);
}

#[test]
fn cuda_moe_q8_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join("allpaca_cuda_moe_q8_gen.gguf");
    write_tiny_moe_q8_gguf(&path);
    let mut cfg = cuda_synthetic();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.clone());
    cfg.kv.page_tokens = 8;
    let engine = Engine::new(cfg).expect("MoE Q8 engine");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(before.weight_bytes_on_device > 0);
    let out = engine
        .generate("hello", greedy(2))
        .expect("moe q8 generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE Q8 forwards; stats={stats:?}"
    );
    let _ = std::fs::remove_file(path);
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
    write_gguf(path, &meta, &views).expect("write tiny gguf");
}

fn write_tiny_moe_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let n_exp = 4usize;
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
    meta.insert("llama.expert_count".into(), GgufValue::Uint32(n_exp as u32));
    meta.insert("llama.expert_used_count".into(), GgufValue::Uint32(2));
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
    let w = 0.01f32;
    let mut gate_exps = Vec::new();
    let mut up_exps = Vec::new();
    let mut down_exps = Vec::new();
    for _ in 0..n_exp {
        gate_exps.extend(std::iter::repeat(w).take(ff * dim));
        up_exps.extend(std::iter::repeat(w).take(ff * dim));
        down_exps.extend(std::iter::repeat(w).take(dim * ff));
    }
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
            "blk.0.ffn_gate_inp.weight".into(),
            vec![n_exp as u64, dim as u64],
            vec![w; n_exp * dim],
        ),
        (
            "blk.0.ffn_gate_exps.weight".into(),
            vec![n_exp as u64, ff as u64, dim as u64],
            gate_exps,
        ),
        (
            "blk.0.ffn_up_exps.weight".into(),
            vec![n_exp as u64, ff as u64, dim as u64],
            up_exps,
        ),
        (
            "blk.0.ffn_down_exps.weight".into(),
            vec![n_exp as u64, dim as u64, ff as u64],
            down_exps,
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
    write_gguf(path, &meta, &views).expect("write tiny moe gguf");
}

/// Tiny Qwen35 Gated DeltaNet GGUF (2 GDN layers, no dense attn Q).
fn write_tiny_qwen_gdn_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 16usize;
    let vocab = 4usize;
    let n_layers = 2usize;
    let n_v = 2usize;
    let n_k = 2usize;
    let head = 8usize;
    let inner = n_v * head;
    let key_dim = n_k * head;
    let qkv_dim = key_dim * 2 + inner;
    let conv_k = 2usize;
    let w = 0.02f32;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("qwen35".into()),
    );
    meta.insert("qwen35.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert(
        "qwen35.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert(
        "qwen35.block_count".into(),
        GgufValue::Uint32(n_layers as u32),
    );
    meta.insert("qwen35.attention.head_count".into(), GgufValue::Uint32(2));
    meta.insert(
        "qwen35.attention.head_count_kv".into(),
        GgufValue::Uint32(1),
    );
    meta.insert(
        "qwen35.feed_forward_length".into(),
        GgufValue::Uint32((dim * 2) as u32),
    );
    meta.insert(
        "qwen35.ssm.state_size".into(),
        GgufValue::Uint32(head as u32),
    );
    meta.insert(
        "qwen35.ssm.inner_size".into(),
        GgufValue::Uint32(inner as u32),
    );
    meta.insert(
        "qwen35.ssm.time_step_rank".into(),
        GgufValue::Uint32(n_v as u32),
    );
    meta.insert(
        "qwen35.ssm.group_count".into(),
        GgufValue::Uint32(n_k as u32),
    );
    meta.insert(
        "qwen35.ssm.conv_kernel".into(),
        GgufValue::Uint32(conv_k as u32),
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
            GgufValue::String("hi".into()),
        ]),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![(
        "token_embd.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![w; vocab * dim],
    )];
    for i in 0..n_layers {
        owned.push((
            format!("blk.{i}.attn_norm.weight"),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        owned.push((
            format!("blk.{i}.attn_qkv.weight"),
            vec![qkv_dim as u64, dim as u64],
            vec![w; qkv_dim * dim],
        ));
        owned.push((
            format!("blk.{i}.attn_gate.weight"),
            vec![inner as u64, dim as u64],
            vec![w; inner * dim],
        ));
        owned.push((
            format!("blk.{i}.ssm_alpha.weight"),
            vec![n_v as u64, dim as u64],
            vec![w; n_v * dim],
        ));
        owned.push((
            format!("blk.{i}.ssm_beta.weight"),
            vec![n_v as u64, dim as u64],
            vec![w; n_v * dim],
        ));
        owned.push((
            format!("blk.{i}.ssm_out.weight"),
            vec![dim as u64, inner as u64],
            vec![w; dim * inner],
        ));
        owned.push((format!("blk.{i}.ssm_a"), vec![n_v as u64], vec![-0.5; n_v]));
        owned.push((
            format!("blk.{i}.ssm_dt.bias"),
            vec![n_v as u64],
            vec![0.0; n_v],
        ));
        owned.push((
            format!("blk.{i}.ssm_norm.weight"),
            vec![head as u64],
            vec![1.0; head],
        ));
        owned.push((
            format!("blk.{i}.ssm_conv1d.weight"),
            vec![(conv_k * qkv_dim) as u64],
            vec![w; conv_k * qkv_dim],
        ));
    }
    owned.push((
        "output_norm.weight".into(),
        vec![dim as u64],
        vec![1.0; dim],
    ));
    owned.push((
        "output.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![w; vocab * dim],
    ));
    let views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    write_gguf(path, &meta, &views).expect("write tiny qwen gdn gguf");
}

/// Tiny Qwen35 GDN with Q8_0 projections (host F32 fill skipped).
fn write_tiny_qwen_gdn_q8_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor,
    };
    use std::collections::HashMap;

    let dim = 16usize;
    let vocab = 4usize;
    let n_layers = 2usize;
    let n_v = 2usize;
    let n_k = 2usize;
    let head = 8usize;
    let inner = n_v * head;
    let key_dim = n_k * head;
    let qkv_dim = key_dim * 2 + inner;
    let conv_k = 2usize;
    let w = 0.02f32;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("qwen35".into()),
    );
    meta.insert("qwen35.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert(
        "qwen35.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert(
        "qwen35.block_count".into(),
        GgufValue::Uint32(n_layers as u32),
    );
    meta.insert("qwen35.attention.head_count".into(), GgufValue::Uint32(2));
    meta.insert(
        "qwen35.attention.head_count_kv".into(),
        GgufValue::Uint32(1),
    );
    meta.insert(
        "qwen35.feed_forward_length".into(),
        GgufValue::Uint32((dim * 2) as u32),
    );
    meta.insert(
        "qwen35.ssm.state_size".into(),
        GgufValue::Uint32(head as u32),
    );
    meta.insert(
        "qwen35.ssm.inner_size".into(),
        GgufValue::Uint32(inner as u32),
    );
    meta.insert(
        "qwen35.ssm.time_step_rank".into(),
        GgufValue::Uint32(n_v as u32),
    );
    meta.insert(
        "qwen35.ssm.group_count".into(),
        GgufValue::Uint32(n_k as u32),
    );
    meta.insert(
        "qwen35.ssm.conv_kernel".into(),
        GgufValue::Uint32(conv_k as u32),
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
            GgufValue::String("hi".into()),
        ]),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let mut f32_owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![(
        "token_embd.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![w; vocab * dim],
    )];
    let mut q_owned: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    for i in 0..n_layers {
        f32_owned.push((
            format!("blk.{i}.attn_norm.weight"),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        q_owned.push((
            format!("blk.{i}.attn_qkv.weight"),
            vec![qkv_dim as u64, dim as u64],
            quantize_q8_0(&vec![w; qkv_dim * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.attn_gate.weight"),
            vec![inner as u64, dim as u64],
            quantize_q8_0(&vec![w; inner * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.ssm_alpha.weight"),
            vec![n_v as u64, dim as u64],
            quantize_q8_0(&vec![w; n_v * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.ssm_beta.weight"),
            vec![n_v as u64, dim as u64],
            quantize_q8_0(&vec![w; n_v * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.ssm_out.weight"),
            vec![dim as u64, inner as u64],
            quantize_q8_0(&vec![w; dim * inner]),
        ));
        f32_owned.push((format!("blk.{i}.ssm_a"), vec![n_v as u64], vec![-0.5; n_v]));
        f32_owned.push((
            format!("blk.{i}.ssm_dt.bias"),
            vec![n_v as u64],
            vec![0.0; n_v],
        ));
        f32_owned.push((
            format!("blk.{i}.ssm_norm.weight"),
            vec![head as u64],
            vec![1.0; head],
        ));
        f32_owned.push((
            format!("blk.{i}.ssm_conv1d.weight"),
            vec![(conv_k * qkv_dim) as u64],
            vec![w; conv_k * qkv_dim],
        ));
    }
    f32_owned.push((
        "output_norm.weight".into(),
        vec![dim as u64],
        vec![1.0; dim],
    ));
    f32_owned.push((
        "output.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![w; vocab * dim],
    ));
    let f32_tensors: Vec<GgufWriteTensor<'_>> = f32_owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    let quant_tensors: Vec<GgufWriteQuantTensor<'_>> = q_owned
        .iter()
        .map(|(name, dims, bytes)| GgufWriteQuantTensor {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: GgmlType::Q8_0,
            bytes,
        })
        .collect();
    write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).expect("write tiny qwen gdn q8");
}

/// Qwen35 GDN with Q8_0 projections large enough to stay packed (n>65536).
/// GGUF dims are innermost-first so GEMV K is `dims[0]` (must be multiple of 32).
fn write_qwen_gdn_q8_packed_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor,
    };
    use std::collections::HashMap;

    let dim = 288usize;
    let vocab = 4usize;
    let n_layers = 1usize;
    let n_v = 4usize;
    let n_k = 4usize;
    let head = 64usize;
    let inner = n_v * head;
    let key_dim = n_k * head;
    let qkv_dim = key_dim * 2 + inner;
    let conv_k = 2usize;
    let w = 0.02f32;
    debug_assert!(qkv_dim * dim > 65_536);
    debug_assert_eq!(dim % 32, 0);
    debug_assert_eq!(inner % 32, 0);

    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("qwen35".into()),
    );
    meta.insert("qwen35.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert(
        "qwen35.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert(
        "qwen35.block_count".into(),
        GgufValue::Uint32(n_layers as u32),
    );
    meta.insert("qwen35.attention.head_count".into(), GgufValue::Uint32(4));
    meta.insert(
        "qwen35.attention.head_count_kv".into(),
        GgufValue::Uint32(1),
    );
    meta.insert(
        "qwen35.feed_forward_length".into(),
        GgufValue::Uint32((dim * 2) as u32),
    );
    meta.insert(
        "qwen35.ssm.state_size".into(),
        GgufValue::Uint32(head as u32),
    );
    meta.insert(
        "qwen35.ssm.inner_size".into(),
        GgufValue::Uint32(inner as u32),
    );
    meta.insert(
        "qwen35.ssm.time_step_rank".into(),
        GgufValue::Uint32(n_v as u32),
    );
    meta.insert(
        "qwen35.ssm.group_count".into(),
        GgufValue::Uint32(n_k as u32),
    );
    meta.insert(
        "qwen35.ssm.conv_kernel".into(),
        GgufValue::Uint32(conv_k as u32),
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
            GgufValue::String("hi".into()),
        ]),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let mut f32_owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![(
        "token_embd.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![w; vocab * dim],
    )];
    let mut q_owned: Vec<(String, Vec<u64>, Vec<u8>)> = Vec::new();
    for i in 0..n_layers {
        f32_owned.push((
            format!("blk.{i}.attn_norm.weight"),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        q_owned.push((
            format!("blk.{i}.attn_qkv.weight"),
            vec![dim as u64, qkv_dim as u64],
            quantize_q8_0(&vec![w; qkv_dim * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.attn_gate.weight"),
            vec![dim as u64, inner as u64],
            quantize_q8_0(&vec![w; inner * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.ssm_alpha.weight"),
            vec![dim as u64, n_v as u64],
            quantize_q8_0(&vec![w; n_v * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.ssm_beta.weight"),
            vec![dim as u64, n_v as u64],
            quantize_q8_0(&vec![w; n_v * dim]),
        ));
        q_owned.push((
            format!("blk.{i}.ssm_out.weight"),
            vec![inner as u64, dim as u64],
            quantize_q8_0(&vec![w; dim * inner]),
        ));
        f32_owned.push((format!("blk.{i}.ssm_a"), vec![n_v as u64], vec![-0.5; n_v]));
        f32_owned.push((
            format!("blk.{i}.ssm_dt.bias"),
            vec![n_v as u64],
            vec![0.0; n_v],
        ));
        f32_owned.push((
            format!("blk.{i}.ssm_norm.weight"),
            vec![head as u64],
            vec![1.0; head],
        ));
        f32_owned.push((
            format!("blk.{i}.ssm_conv1d.weight"),
            vec![(conv_k * qkv_dim) as u64],
            vec![w; conv_k * qkv_dim],
        ));
    }
    f32_owned.push((
        "output_norm.weight".into(),
        vec![dim as u64],
        vec![1.0; dim],
    ));
    f32_owned.push((
        "output.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![w; vocab * dim],
    ));
    let f32_tensors: Vec<GgufWriteTensor<'_>> = f32_owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data,
        })
        .collect();
    let quant_tensors: Vec<GgufWriteQuantTensor<'_>> = q_owned
        .iter()
        .map(|(name, dims, bytes)| GgufWriteQuantTensor {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: GgmlType::Q8_0,
            bytes,
        })
        .collect();
    write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).expect("write packed qwen gdn q8");
}

/// Mixtral-style MoE with Q8_0 stacked experts (K multiple of 32 for GEMV stage).
fn write_tiny_moe_q8_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor,
    };
    use std::collections::HashMap;

    let dim = 32usize;
    let vocab = 4usize;
    let ff = 64usize;
    let n_exp = 4usize;
    let n_kv = 2usize;
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
    meta.insert(
        "llama.attention.head_count_kv".into(),
        GgufValue::Uint32(n_kv as u32),
    );
    meta.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert("llama.expert_count".into(), GgufValue::Uint32(n_exp as u32));
    meta.insert("llama.expert_used_count".into(), GgufValue::Uint32(2));
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

    let w = 0.01f32;
    let embd = vec![w; vocab * dim];
    let attn_norm = vec![1.0f32; dim];
    let ffn_norm = vec![1.0f32; dim];
    let output_norm = vec![1.0f32; dim];
    let q = vec![w; dim * dim];
    let k = vec![w; n_kv * (dim / 4) * dim];
    let v = vec![w; n_kv * (dim / 4) * dim];
    let o = vec![w; dim * dim];
    let router = vec![w; n_exp * dim];
    let mut gate_exps = Vec::new();
    let mut up_exps = Vec::new();
    let mut down_exps = Vec::new();
    for _ in 0..n_exp {
        gate_exps.extend(std::iter::repeat(w).take(ff * dim));
        up_exps.extend(std::iter::repeat(w).take(ff * dim));
        down_exps.extend(std::iter::repeat(w).take(dim * ff));
    }
    let gate_q = quantize_q8_0(&gate_exps);
    let up_q = quantize_q8_0(&up_exps);
    let down_q = quantize_q8_0(&down_exps);

    let f32_tensors = [
        GgufWriteTensor {
            name: "token_embd.weight".into(),
            dims: vec![vocab as u64, dim as u64],
            data: &embd,
        },
        GgufWriteTensor {
            name: "blk.0.attn_norm.weight".into(),
            dims: vec![dim as u64],
            data: &attn_norm,
        },
        GgufWriteTensor {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![dim as u64, dim as u64],
            data: &q,
        },
        GgufWriteTensor {
            name: "blk.0.attn_k.weight".into(),
            dims: vec![(n_kv * (dim / 4)) as u64, dim as u64],
            data: &k,
        },
        GgufWriteTensor {
            name: "blk.0.attn_v.weight".into(),
            dims: vec![(n_kv * (dim / 4)) as u64, dim as u64],
            data: &v,
        },
        GgufWriteTensor {
            name: "blk.0.attn_output.weight".into(),
            dims: vec![dim as u64, dim as u64],
            data: &o,
        },
        GgufWriteTensor {
            name: "blk.0.ffn_norm.weight".into(),
            dims: vec![dim as u64],
            data: &ffn_norm,
        },
        GgufWriteTensor {
            name: "blk.0.ffn_gate_inp.weight".into(),
            dims: vec![n_exp as u64, dim as u64],
            data: &router,
        },
        GgufWriteTensor {
            name: "output_norm.weight".into(),
            dims: vec![dim as u64],
            data: &output_norm,
        },
    ];
    let quant_tensors = [
        GgufWriteQuantTensor {
            name: "blk.0.ffn_gate_exps.weight".into(),
            // GGUF: innermost dim first → K=dim for GEMV rows n_exp*ff.
            dims: vec![dim as u64, ff as u64, n_exp as u64],
            ggml_type: GgmlType::Q8_0,
            bytes: &gate_q,
        },
        GgufWriteQuantTensor {
            name: "blk.0.ffn_up_exps.weight".into(),
            dims: vec![dim as u64, ff as u64, n_exp as u64],
            ggml_type: GgmlType::Q8_0,
            bytes: &up_q,
        },
        GgufWriteQuantTensor {
            name: "blk.0.ffn_down_exps.weight".into(),
            dims: vec![ff as u64, dim as u64, n_exp as u64],
            ggml_type: GgmlType::Q8_0,
            bytes: &down_q,
        },
    ];
    write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).expect("write moe q8");
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
    let path = std::env::temp_dir().join(format!("allpaca-cuda-tiny-{}.gguf", std::process::id()));
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
