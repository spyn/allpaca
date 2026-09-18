//! Public Engine behaviors for the CUDA decode path.
//! GPU-gated tests skip when `device_count() == 0` or runtime is not linked.

#![cfg(feature = "cuda")]

use allpaca_core::{Engine, ModelConfig};
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
fn cuda_mobile_tiered_when_device_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let stats = engine.stats();
    assert!(
        stats.pipeline_mode.contains("MobileTiered"),
        "single CUDA GPU should be MobileTiered, got {}",
        stats.pipeline_mode
    );
    assert_eq!(stats.backend, "cuda");
    assert!(
        engine.kv_vram_budget_bytes() > 4 * 1024 * 1024,
        "expected GPU-scaled kv budget, got {}",
        engine.kv_vram_budget_bytes()
    );
}

#[test]
fn cuda_smoke_decode_keeps_device_resident() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let load = engine.stats();
    let out = engine.generate("hi", greedy(8)).expect("generate");
    assert_eq!(out.token_ids.len(), 8);
    let stats = engine.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(stats.weight_bytes_on_device >= load.weight_bytes_on_device);
    assert!(
        stats.gpu_forwards + stats.target_gpu_forwards >= 8,
        "expected ≥8 CUDA forwards for 8 tokens, got gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    assert!(
        stats.elementwise_gpu_ops > 0,
        "RoPE/SiLU/residual should run on device, elementwise_gpu_ops={}",
        stats.elementwise_gpu_ops
    );
}

#[test]
fn cuda_stats_sample_vram_when_device_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let stats = engine.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(
        stats.gpu_vram_total_bytes > 0,
        "sample_gpu should report board total, got {}",
        stats.gpu_vram_total_bytes
    );
    assert!(
        stats.gpu_vram_used_bytes <= stats.gpu_vram_total_bytes,
        "used={} total={}",
        stats.gpu_vram_used_bytes,
        stats.gpu_vram_total_bytes
    );
    let (arch, l2) = allpaca_cuda::device_sm_arch(0).expect("sm arch");
    assert!(arch.starts_with("sm_"), "arch={arch}");
    assert!(l2 > 0, "l2={l2}");
}

#[test]
fn cuda_stats_gpu_util_after_generate_when_device_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_synthetic()).expect("engine");
    let _ = engine.generate("hi", greedy(4)).expect("generate");
    let stats = engine.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(
        stats.gpu_samples > 0,
        "sampler should have run, samples={}",
        stats.gpu_samples
    );
    // Host still owns much of the CUDA decode loop; util is GPU/(GPU+host).
    assert!(
        stats.gpu_util > 0.0 && stats.gpu_util <= 1.0,
        "expected positive GPU util after decode profile, got {}",
        stats.gpu_util
    );
}

fn cuda_moe_cfg() -> Config {
    let mut cfg = cuda_synthetic();
    // Engine::new auto-loads dense tiny when synthetic; load MoE explicitly.
    cfg.model.synthetic = false;
    cfg
}

#[test]
fn cuda_synthetic_moe_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic(ModelConfig::tiny_moe())
        .expect("load moe");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "MoE router+experts should upload at load; stats={before:?}"
    );
    assert!(
        before.weight_bytes_on_device > 0,
        "MoE weights should be resident; stats={before:?}"
    );
    let h2d_after_load = before.activation_h2d_bytes;

    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());

    let stats = engine.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    assert_eq!(
        stats.weight_h2d_bytes, before.weight_h2d_bytes,
        "weight_h2d must not grow during MoE generate"
    );
    let activation_delta = stats.activation_h2d_bytes.saturating_sub(h2d_after_load);
    assert!(
        activation_delta < before.weight_h2d_bytes,
        "per-generate activation H2D ({activation_delta}) must stay below weight pack ({})",
        before.weight_h2d_bytes
    );
}

#[test]
fn cuda_synthetic_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic(ModelConfig::tiny_hybrid())
        .expect("load hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "attention-hybrid (recurrent flag) should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA hybrid forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_ssm_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_ssm_hybrid(7))
        .expect("load ssm hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "SSM hybrid should upload attention/FFN pack; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA SSM-hybrid forwards (host mamba2_step), got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_gdn_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_gdn_hybrid(11))
        .expect("load gdn hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "GDN hybrid should upload attention pack; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GDN-hybrid forwards (host gated_delta_step), got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_mla_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_mla_hybrid(13))
        .expect("load mla hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "MLA hybrid should upload FFN/embed pack; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MLA-hybrid forwards (host mla_attend_cpu), got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}
