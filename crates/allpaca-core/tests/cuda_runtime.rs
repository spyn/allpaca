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

#[test]
fn cuda_coordinator_rpc_split_keeps_local_layers_on_device() {
    if !cuda_device_available() {
        return;
    }
    use std::net::TcpListener;
    use std::thread;

    use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let worker = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
    });

    let mut cfg = cuda_synthetic();
    cfg.rpc.endpoints = vec![addr.to_string()];
    cfg.rpc.max_seq = 64;
    cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
    let engine = Engine::new(cfg).expect("engine");
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.rpc_roundtrips > 0,
        "expected remote layer launches, got {}",
        stats.rpc_roundtrips
    );
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "CUDA coordinator should keep local layers on device, gpu={} target={}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
    drop(engine);
    worker.join().unwrap();
}

#[test]
fn cuda_rpc_activations_stay_device_resident_across_local_layers() {
    if !cuda_device_available() {
        return;
    }
    use std::net::TcpListener;
    use std::thread;

    use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let worker = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
    });

    let mut cfg = cuda_synthetic();
    cfg.rpc.endpoints = vec![addr.to_string()];
    cfg.rpc.max_seq = 64;
    cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
    let engine = Engine::new(cfg).expect("engine");
    let _ = engine.generate("hello", greedy(4)).expect("generate");
    let stats = engine.stats();
    assert!(
        stats.rpc_roundtrips > 0 && stats.rpc_activation_bytes > 0,
        "expected RPC activation bounce at remote boundaries; trips={} bytes={}",
        stats.rpc_roundtrips,
        stats.rpc_activation_bytes
    );
    // Local layers keep x on CUDA; coordinator activation H2D should stay near the
    // remote-boundary bounce (plus a small embed/logits allowance), not per-GEMM.
    let bound = stats.rpc_activation_bytes.saturating_mul(4).max(8 * 1024);
    assert!(
        stats.activation_h2d_bytes <= bound,
        "local-layer activations must stay device-resident; h2d={} rpc_act={} bound={}",
        stats.activation_h2d_bytes,
        stats.rpc_activation_bytes,
        bound
    );
    drop(engine);
    worker.join().unwrap();
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
        "expected CUDA SSM-hybrid forwards (resident in/out GEMMs + device conv/ssd), got gpu={} target={}; stats={stats:?}",
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
        "GDN hybrid should upload projections + pack; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GDN-hybrid forwards (resident projection GEMMs), got gpu={} target={}; stats={stats:?}",
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

#[test]
fn cuda_synthetic_moe_shared_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_moe_shared(17))
        .expect("load moe shared");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "MoE+shared should upload experts+shared; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE+shared forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_dense_ffn_hybrid_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_dense_ffn_hybrid(19))
        .expect("load dense-ffn hybrid");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "DenseFfn hybrid should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA DenseFfn-hybrid forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_moe_shared_relu_sqr_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_moe_shared_relu_sqr(23))
        .expect("load moe shared relu2");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "MoE+shared ReLU² should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA MoE+shared ReLU² forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_softcap_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_softcap(29))
        .expect("load softcap");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "softcap dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA softcap forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_swa_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_swa(37))
        .expect("load swa");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "SWA dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(6)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA SWA forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_headwise_gate_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_headwise_gate(41))
        .expect("load headwise gate");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "headwise-gate dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA headwise-gate forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_gemma_norms_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_gemma_norms(43))
        .expect("load gemma norms");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "gemma-norms dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA gemma-norms forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_gelu_ffn_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_gelu_ffn(47))
        .expect("load gelu ffn");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "gelu-ffn dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA gelu-ffn forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_qk_norms_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_qk_norms(53))
        .expect("load qk norms");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "qk-norms dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA qk-norms forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_gated_attn_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_gated_attn(59))
        .expect("load gated attn");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "gated-attn dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA gated-attn forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_attn_sinks_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_attn_sinks(61))
        .expect("load attn sinks");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "attn-sinks dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA attn-sinks forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_partial_rope_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_partial_rope(67))
        .expect("load partial rope");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "partial-rope dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA partial-rope forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_synthetic_imrope_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let engine = Engine::new(cuda_moe_cfg()).expect("engine");
    engine
        .load_synthetic_model(allpaca_core::TinyModel::synthetic_imrope(71))
        .expect("load imrope");
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "imrope dense should upload; stats={before:?}"
    );
    let out = engine.generate("hello", greedy(4)).expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA imrope forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

fn write_tiny_qwen_gdn_gguf(path: &std::path::Path, w: f32) {
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
    write_gguf(path, &meta, &views).unwrap();
}

#[test]
fn cuda_tiny_gdn_gguf_generate_uses_device_when_present() {
    if !cuda_device_available() {
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_cuda_tiny_gdn-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_tiny_qwen_gdn_gguf(&path, 0.02);
    let mut cfg = cuda_synthetic();
    cfg.model.path = Some(path.clone());
    cfg.model.synthetic = false;
    let engine = match Engine::new(cfg) {
        Ok(e) => e,
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            panic!("Engine::new GDN GGUF: {e}");
        }
    };
    let before = engine.stats();
    assert_eq!(before.backend, "cuda");
    assert!(
        before.weight_h2d_bytes > 0,
        "GDN GGUF should upload CUDA-resident pack; stats={before:?}"
    );
    let out = engine.generate("hi", greedy(4));
    let _ = std::fs::remove_file(&path);
    let out = out.expect("generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.gpu_forwards > 0 || stats.target_gpu_forwards > 0,
        "expected CUDA GDN GGUF forwards, got gpu={} target={}; stats={stats:?}",
        stats.gpu_forwards,
        stats.target_gpu_forwards
    );
}

#[test]
fn cuda_batched_prefill_one_forward_per_chunk() {
    if !cuda_device_available() {
        return;
    }
    let mut cfg = cuda_synthetic();
    cfg.prefill.chunk_tokens = 16;
    cfg.kv.page_tokens = 64;
    cfg.kv.vram_bytes = 64 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("engine");
    let prompt = "hello world. ".repeat(40);
    let out = engine.generate(&prompt, greedy(2)).expect("generate");
    assert!(
        out.prompt_tokens >= 16,
        "need a multi-token prompt, got {}",
        out.prompt_tokens
    );
    let stats = engine.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(
        stats.gpu_forwards < out.prompt_tokens as u64,
        "batched prefill must not run one forward per prompt token; gpu_forwards={} prompt_tokens={}",
        stats.gpu_forwards,
        out.prompt_tokens
    );
    assert!(!out.token_ids.is_empty());
}

#[test]
fn cuda_moe_batched_prefill_matches_per_token_and_batches() {
    if !cuda_device_available() {
        return;
    }
    let prompt = "hello world. ".repeat(40);
    let params = greedy(4);

    let mut per_cfg = cuda_synthetic();
    per_cfg.model.synthetic = false;
    per_cfg.prefill.chunk_tokens = 1;
    per_cfg.kv.page_tokens = 64;
    per_cfg.kv.vram_bytes = 64 * 1024 * 1024;
    let per = Engine::new(per_cfg).expect("per-token engine");
    per.load_synthetic(allpaca_core::ModelConfig::tiny_moe())
        .expect("load tiny_moe");
    let per_out = per
        .generate(&prompt, params.clone())
        .expect("per-token generate");
    assert!(
        per_out.prompt_tokens >= 16,
        "need a multi-token prompt, got {}",
        per_out.prompt_tokens
    );

    let mut batch_cfg = cuda_synthetic();
    batch_cfg.model.synthetic = false;
    batch_cfg.prefill.chunk_tokens = 16;
    batch_cfg.kv.page_tokens = 64;
    batch_cfg.kv.vram_bytes = 64 * 1024 * 1024;
    let batch = Engine::new(batch_cfg).expect("batch engine");
    batch
        .load_synthetic(allpaca_core::ModelConfig::tiny_moe())
        .expect("load tiny_moe");
    let batch_out = batch.generate(&prompt, params).expect("batch generate");
    let stats = batch.stats();
    assert_eq!(stats.backend, "cuda");
    assert!(
        stats.gpu_forwards < batch_out.prompt_tokens as u64,
        "MoE batched prefill must not run one forward per prompt token; gpu_forwards={} prompt_tokens={}",
        stats.gpu_forwards,
        batch_out.prompt_tokens
    );
    assert_eq!(
        batch_out.token_ids, per_out.token_ids,
        "MoE chunked prefill must match per-token CUDA decode"
    );
}

#[test]
fn cuda_dense_batched_prefill_matches_per_token() {
    if !cuda_device_available() {
        return;
    }
    let prompt = "hello world. ".repeat(40);
    let params = greedy(4);
    let mut per_cfg = cuda_synthetic();
    per_cfg.prefill.chunk_tokens = 1;
    per_cfg.kv.page_tokens = 64;
    per_cfg.kv.vram_bytes = 64 * 1024 * 1024;
    let per = Engine::new(per_cfg).expect("per");
    let per_out = per.generate(&prompt, params.clone()).expect("per");
    assert!(per_out.prompt_tokens >= 16);
    let mut batch_cfg = cuda_synthetic();
    batch_cfg.prefill.chunk_tokens = 16;
    batch_cfg.kv.page_tokens = 64;
    batch_cfg.kv.vram_bytes = 64 * 1024 * 1024;
    let batch = Engine::new(batch_cfg).expect("batch");
    let batch_out = batch.generate(&prompt, params).expect("batch");
    let stats = batch.stats();
    assert!(
        stats.gpu_forwards < batch_out.prompt_tokens as u64,
        "dense batched prefill must batch; gpu_forwards={} prompt_tokens={}",
        stats.gpu_forwards,
        batch_out.prompt_tokens
    );
    assert_eq!(
        batch_out.token_ids, per_out.token_ids,
        "dense chunked prefill must match per-token CUDA decode"
    );
}
