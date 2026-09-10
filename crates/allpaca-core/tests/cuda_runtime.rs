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
    let activation_delta = stats
        .activation_h2d_bytes
        .saturating_sub(h2d_after_load);
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
