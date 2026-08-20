//! Public Engine / DeviceInventory / Config behaviors for the HIP path.
//! GPU-gated tests skip when `hipGetDeviceCount() == 0`.

use allpaca_core::Engine;
#[cfg(feature = "hip")]
use allpaca_core::ModelConfig;
use allpaca_types::{Config, GenerationParams};
#[cfg(feature = "hip")]
use allpaca_types::{DeviceKind, DeviceSelector};

fn cpu_only_synthetic() -> Config {
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = true;
    cfg.engine.cpu_only = true;
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg
}

#[cfg(feature = "hip")]
fn hip_auto_synthetic() -> Config {
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = true;
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "auto".into();
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

#[test]
fn cpu_only_generate_reports_cpu_backend() {
    let engine = Engine::new(cpu_only_synthetic()).unwrap();
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "cpu");
    assert_eq!(stats.gpu_forwards, 0);
    assert_eq!(stats.target_gpu_forwards, 0);
    assert_eq!(stats.draft_gpu_forwards, 0);
    assert_eq!(stats.weight_bytes_migrated, 0);
    assert_eq!(stats.weight_bytes_on_device, 0);
    assert_eq!(stats.weight_h2d_bytes, 0);
    assert_eq!(stats.kv_bytes_on_device, 0);
    assert_eq!(stats.draft_kv_bytes_on_device, 0);
    assert_eq!(stats.elementwise_gpu_ops, 0);
    assert_eq!(stats.residual_gpu_ops, 0);
    assert_eq!(stats.activation_h2d_bytes, 0);
    assert_eq!(stats.activation_d2h_bytes, 0);
}

#[test]
fn cpu_only_decode_profile_buckets_are_zero() {
    let engine = Engine::new(cpu_only_synthetic()).unwrap();
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.q8_gemm_us, 0.0);
    assert_eq!(stats.attention_us, 0.0);
    assert_eq!(stats.lm_head_us, 0.0);
    assert_eq!(stats.other_gpu_us, 0.0);
    assert_eq!(stats.host_us, 0.0);
}

#[test]
fn zero_or_one_gpu_does_not_claim_p2p() {
    let engine = Engine::new(cpu_only_synthetic()).unwrap();
    let stats = engine.stats();
    assert!(!stats.p2p_capable, "cpu_only must not claim P2P capability");
    assert!(!stats.p2p_enabled, "cpu_only must not claim P2P enabled");
    assert_eq!(stats.transfer_path, "HostStaging");
    assert_eq!(stats.p2p_status, "p2p_unavailable");
    let inv = engine.inventory();
    assert!(
        inv.devices.iter().all(|d| d.peers.is_empty()),
        "0–1 GPU inventory must record no P2P peers"
    );
    assert!(!inv.can_p2p(0, 1));
    assert!(!inv.p2p_capable);
    assert!(!inv.p2p_enabled);
}

#[cfg(feature = "hip")]
fn hip_gpus(engine: &Engine) -> Vec<&allpaca_types::DeviceInfo> {
    engine
        .inventory()
        .devices
        .iter()
        .filter(|d| d.kind == DeviceKind::HipGpu)
        .collect()
}

#[cfg(feature = "hip")]
#[test]
fn inventory_records_can_p2p_after_enable() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    let gpus = hip_gpus(&engine);
    if gpus.is_empty() {
        return;
    }
    let inv = engine.inventory();
    let stats = engine.stats();
    for a in &gpus {
        for b in &gpus {
            if a.id == b.id {
                assert!(
                    !inv.can_p2p(a.id, b.id),
                    "a device is not a P2P peer of itself"
                );
                continue;
            }
            assert_eq!(
                inv.can_p2p(a.id, b.id),
                a.peers.contains(&b.id),
                "inventory can_p2p must match recorded peers after enable"
            );
        }
    }
    let any_pair = gpus.iter().any(|d| !d.peers.is_empty());
    assert_eq!(stats.p2p_capable, any_pair);
    assert_eq!(inv.p2p_capable, any_pair);
    if gpus.len() < 2 {
        assert!(!stats.p2p_capable);
        assert!(!stats.p2p_enabled);
        assert!(gpus.iter().all(|d| d.peers.is_empty()));
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_inventory_enumerates_stable_ids() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    let gpus = hip_gpus(&engine);
    if gpus.is_empty() {
        return;
    }
    assert!(gpus.len() >= 1);
    let ids: Vec<i32> = gpus.iter().map(|d| d.id).collect();
    assert!(ids.iter().all(|id| *id >= 0));
    assert!(gpus.iter().all(|d| d.memory_bytes > 0 && !d.name.is_empty()));
    let again = Engine::new(hip_auto_synthetic()).unwrap();
    let ids2: Vec<i32> = hip_gpus(&again).iter().map(|d| d.id).collect();
    assert_eq!(ids, ids2, "HIP device ids must be stable across Engine boots");
}

#[cfg(feature = "hip")]
#[test]
fn transfer_path_is_measured_not_assumed_p2p() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let stats = engine.stats();
    eprintln!(
        "transfer_path={} p2p_capable={} p2p_enabled={} p2p_status={} p2p_gbps={:.3} host_staging_gbps={:.3} p2p_us={:.1} host_us={:.1} target_device={} draft_device={}",
        stats.transfer_path,
        stats.p2p_capable,
        stats.p2p_enabled,
        stats.p2p_status,
        stats.p2p_gbps,
        stats.host_staging_gbps,
        stats.p2p_us,
        stats.host_staging_us,
        stats.target_device,
        stats.draft_device
    );
    for d in hip_gpus(&engine) {
        eprintln!(
            "gpu id={} name={} mem_gb={:.2} bw_gbps={:.1} peers={:?}",
            d.id,
            d.name,
            d.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            d.bandwidth_gbps,
            d.peers
        );
    }
    assert!(
        stats.transfer_path == "HostStaging" || stats.transfer_path == "P2p",
        "unexpected transfer_path {}",
        stats.transfer_path
    );
    if hip_gpus(&engine).len() >= 2 {
        assert!(
            engine.inventory().transfer_benchmark.is_some(),
            "two GPUs must record a P2P-vs-host benchmark; P2P is never assumed"
        );
        assert!(
            stats.host_staging_gbps > 0.0 || stats.host_staging_us > 0.0,
            "two-GPU boot must measure host staging"
        );
        if stats.p2p_capable && stats.p2p_enabled && stats.p2p_gbps > 0.0 {
            assert_eq!(stats.p2p_status, "measured");
            assert!(stats.p2p_us > 0.0);
        } else {
            assert_eq!(
                stats.p2p_status, "p2p_unavailable",
                "failed enable+copy must not pretend P2P was measured"
            );
            assert_eq!(stats.p2p_gbps, 0.0);
            assert_eq!(stats.transfer_path, "HostStaging");
        }
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_load_residents_weights_on_device() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let stats = engine.stats();
    assert!(
        stats.weight_bytes_on_device > 0,
        "HIP load must upload weights once; weight_bytes_on_device={}",
        stats.weight_bytes_on_device
    );
    assert!(
        stats.device_memory_bytes >= stats.weight_bytes_on_device,
        "device_memory_bytes must include resident weights, not just 8MiB scratch; stats={stats:?}"
    );
    assert_eq!(
        stats.weight_h2d_bytes, stats.weight_bytes_on_device,
        "initial H2D must be the resident weight blob; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn decode_does_not_h2d_weights() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let after_load = engine.stats();
    assert!(after_load.weight_h2d_bytes > 0);
    let h2d = after_load.weight_h2d_bytes;
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let after = engine.stats();
    assert_eq!(
        after.weight_h2d_bytes, h2d,
        "decode GEMM must use device weight pointers; H2D grew {} -> {}; stats={after:?}",
        h2d, after.weight_h2d_bytes
    );
}

#[cfg(feature = "hip")]
#[test]
fn generate_on_hip_uses_real_device_memory() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let mut handle = engine.begin_request("hello", greedy(4)).unwrap();
    assert!(matches!(
        engine.step(&mut handle).unwrap(),
        allpaca_types::StepResult::Token(_) | allpaca_types::StepResult::Finished(_)
    ));
    let stats = engine.stats();
    assert!(
        stats.device_memory_bytes > 0,
        "hipMalloc workspace must report device memory used > 0, got {}",
        stats.device_memory_bytes
    );
    assert!(
        stats.kv_bytes_on_device > 0,
        "live Request KV pages must be hipMalloc, not host Vecs labeled VRAM; kv_bytes_on_device={}",
        stats.kv_bytes_on_device
    );
    drop(handle);
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
}

/// Synthetic `ModelConfig::tiny` vocab. Each target forward D2H's one logit row.
#[cfg(feature = "hip")]
const SYNTHETIC_VOCAB: u64 = 64;

#[cfg(feature = "hip")]
#[test]
fn hip_embed_gather_stays_on_device() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let after_load = engine.stats();
    assert!(after_load.weight_bytes_on_device > 0);
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let after = engine.stats();
    let embed_row = 32u64 * 4;
    let embed_h2d = after.gpu_forwards.saturating_mul(embed_row);
    assert!(
        after.activation_h2d_bytes.saturating_sub(after_load.activation_h2d_bytes) < embed_h2d
            || after.activation_h2d_bytes == after_load.activation_h2d_bytes,
        "embed gather must use the resident weight row, not H2D each token; load={} after={} forwards={} stats={after:?}",
        after_load.activation_h2d_bytes,
        after.activation_h2d_bytes,
        after.gpu_forwards
    );
    assert_eq!(after.weight_h2d_bytes, after_load.weight_h2d_bytes);
}

#[cfg(feature = "hip")]
#[test]
fn decode_activation_traffic_is_logits_only() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let after_load = engine.stats();
    assert_eq!(after_load.activation_h2d_bytes, 0);
    assert_eq!(after_load.activation_d2h_bytes, 0);
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let after = engine.stats();
    assert_eq!(
        after.activation_h2d_bytes, after_load.activation_h2d_bytes,
        "decode hidden/QKV must stay on device; activation_h2d grew {} -> {}; stats={after:?}",
        after_load.activation_h2d_bytes, after.activation_h2d_bytes
    );
    let logits_d2h = after.gpu_forwards.saturating_mul(SYNTHETIC_VOCAB * 4);
    assert_eq!(
        after.activation_d2h_bytes, logits_d2h,
        "D2H must be logits only (vocab*4*forwards={logits_d2h}); stats={after:?}"
    );
    let hidden_scale = after.gpu_forwards.saturating_mul(2 * 32 * 4 * 8);
    assert!(
        after.activation_d2h_bytes < hidden_scale || hidden_scale == 0,
        "D2H must not be per-layer hidden traffic; d2h={} hidden_scale={hidden_scale} stats={after:?}",
        after.activation_d2h_bytes
    );
    assert_eq!(after.weight_h2d_bytes, after_load.weight_h2d_bytes);
}

#[cfg(feature = "hip")]
#[test]
fn attention_reads_resident_device_kv() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let mut handle = engine.begin_request("hello", greedy(4)).unwrap();
    assert!(matches!(
        engine.step(&mut handle).unwrap(),
        allpaca_types::StepResult::Token(_) | allpaca_types::StepResult::Finished(_)
    ));
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.kv_bytes_on_device > 0,
        "attention must read hipMalloc KV pages; kv_bytes_on_device={}",
        stats.kv_bytes_on_device
    );
    assert!(stats.gpu_forwards > 0);
    assert_eq!(
        stats.activation_h2d_bytes, 0,
        "resident KV attention must not H2D QKV; stats={stats:?}"
    );
    drop(handle);
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
}

#[cfg(feature = "hip")]
#[test]
fn residual_add_runs_on_hip() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.residual_gpu_ops > 0,
        "residual add must launch a HIP kernel; residual_gpu_ops={}",
        stats.residual_gpu_ops
    );
    assert!(stats.elementwise_gpu_ops > 0);
}

/// profile-blas 8B Q8_0 `other_gpu` EMA was ~24 ms. Fused hidden-state ops must shrink it.
#[cfg(feature = "hip")]
const PROFILE_BLAS_OTHER_GPU_US: f64 = 24_000.0;
/// int-dot branch 8B Q8_0 `host_us` EMA was ~32 ms (vocab clone / ring / events).
#[cfg(feature = "hip")]
const INTDOT_HOST_US: f64 = 32_000.0;

#[cfg(feature = "hip")]
#[test]
fn hip_other_gpu_ema_shrinks_and_greedy_matches_cpu() {
    let cpu = Engine::new(cpu_only_synthetic()).unwrap();
    let hip = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&hip).is_empty() {
        return;
    }
    let cpu_out = cpu.generate("hello", greedy(4)).unwrap();
    let hip_out = hip.generate("hello", greedy(4)).unwrap();
    assert_eq!(
        hip_out.token_ids, cpu_out.token_ids,
        "fused other_gpu must keep greedy HIP==CPU; hip={:?} cpu={:?}",
        hip_out.token_ids, cpu_out.token_ids
    );
    let stats = hip.stats();
    assert!(
        stats.other_gpu_us > 0.0,
        "other_gpu_us must stay on the public stats API; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_greedy_generate_matches_cpu_on_synthetic() {
    let cpu = Engine::new(cpu_only_synthetic()).unwrap();
    let hip = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&hip).is_empty() {
        return;
    }
    let cpu_out = cpu.generate("hello", greedy(4)).unwrap();
    let hip_out = hip.generate("hello", greedy(4)).unwrap();
    assert!(!hip_out.token_ids.is_empty());
    assert_eq!(
        hip_out.token_ids, cpu_out.token_ids,
        "HIP greedy tokens must match CPU; hip={:?} cpu={:?}",
        hip_out.token_ids, cpu_out.token_ids
    );
    assert_eq!(hip.stats().backend, "hip");
}

fn long_hello_prompt() -> String {
    "hello ".repeat(16)
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_long_prompt_prefills_without_per_token_forwards() {
    let cpu = Engine::new(cpu_only_synthetic()).unwrap();
    let hip = Engine::new(hip_auto_synthetic()).unwrap();
    let prompt = long_hello_prompt();
    let cpu_out = cpu.generate(&prompt, greedy(4)).unwrap();
    assert!(
        cpu_out.prompt_tokens >= 16,
        "need a prompt of at least 16 tokens, got {}",
        cpu_out.prompt_tokens
    );
    assert!(!cpu_out.token_ids.is_empty());
    if hip_gpus(&hip).is_empty() {
        return;
    }
    let hip_out = hip.generate(&prompt, greedy(4)).unwrap();
    assert!(!hip_out.token_ids.is_empty());
    let stats = hip.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.gpu_forwards < hip_out.prompt_tokens as u64,
        "a prompt longer than 8 tokens must not run N decode-sized GEMMs; gpu_forwards={} prompt_tokens={} stats={stats:?}",
        stats.gpu_forwards,
        hip_out.prompt_tokens
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_greedy_long_prompt_matches_cpu_on_synthetic() {
    let cpu = Engine::new(cpu_only_synthetic()).unwrap();
    let hip = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&hip).is_empty() {
        return;
    }
    let prompt = long_hello_prompt();
    let cpu_out = cpu.generate(&prompt, greedy(4)).unwrap();
    let hip_out = hip.generate(&prompt, greedy(4)).unwrap();
    assert_eq!(
        hip_out.token_ids, cpu_out.token_ids,
        "batched HIP prefill must match per-token CPU greedy; hip={:?} cpu={:?}",
        hip_out.token_ids, cpu_out.token_ids
    );
}

/// Unconstrained HIP generate must not pay a vocab-sized host tax (string clone /
/// logits ring) — `host_us` stays small vs a 32k vocab.
#[cfg(feature = "hip")]
#[test]
fn hip_unconstrained_generate_host_us_not_vocab_bound() {
    let cpu = Engine::new(cpu_only_synthetic()).unwrap();
    let hip = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&hip).is_empty() {
        return;
    }
    let mut cfg = ModelConfig::tiny();
    cfg.vocab_size = 32_768;
    cpu.load_synthetic(cfg.clone()).unwrap();
    hip.load_synthetic(cfg).unwrap();
    let cpu_out = cpu.generate("hello", greedy(4)).unwrap();
    let hip_out = hip.generate("hello", greedy(4)).unwrap();
    assert_eq!(
        hip_out.token_ids, cpu_out.token_ids,
        "HIP greedy tokens must match CPU at large vocab; hip={:?} cpu={:?}",
        hip_out.token_ids, cpu_out.token_ids
    );
    let stats = hip.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.host_us > 0.0,
        "host_us must stay on the public stats API; stats={stats:?}"
    );
    assert!(
        stats.host_us < 8_000.0,
        "unconstrained host_us {:.1}us must not clone a 32k vocab each token; stats={stats:?}",
        stats.host_us
    );
}

/// Public Engine contract for this branch: HIP generate still works and
/// `/stats` decode profile fields stay present after a greedy generate.
#[cfg(feature = "hip")]
#[test]
fn hip_generate_keeps_decode_profile_fields() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(
        !out.token_ids.is_empty(),
        "HIP generate must still emit tokens"
    );
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert!(stats.gpu_forwards > 0);
    let _ = (
        stats.q8_gemm_us,
        stats.attention_us,
        stats.lm_head_us,
        stats.other_gpu_us,
        stats.host_us,
    );
    assert!(
        stats.attention_us > 0.0
            && stats.lm_head_us > 0.0
            && stats.other_gpu_us > 0.0
            && stats.host_us > 0.0,
        "decode profile fields must stay present on HIP generate; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_reports_decode_profile_buckets() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.q8_gemm_us, 0.0,
        "synthetic F32 decode has no Q8 GEMM; q8_gemm_us={}",
        stats.q8_gemm_us
    );
    assert!(
        stats.attention_us > 0.0,
        "HIP attention must land in attention_us; stats={stats:?}"
    );
    assert!(
        stats.lm_head_us > 0.0,
        "HIP lm_head GEMV must land in lm_head_us; stats={stats:?}"
    );
    assert!(
        stats.other_gpu_us > 0.0,
        "layer GEMM / elementwise must land in other_gpu_us; stats={stats:?}"
    );
    assert!(
        stats.host_us > 0.0,
        "logits D2H / host bookkeeping must land in host_us; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_decode_profile_buckets_sum_toward_itl() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let out = engine.generate("hello", greedy(4)).unwrap();
    let elapsed_us = started.elapsed().as_secs_f64() * 1e6;
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    let sum = stats.q8_gemm_us
        + stats.attention_us
        + stats.lm_head_us
        + stats.other_gpu_us
        + stats.host_us;
    assert!(
        sum > 0.0,
        "decode profile buckets must be non-zero on HIP; stats={stats:?}"
    );
    let itl_us = stats.itl_ms_ema * 1_000.0;
    if itl_us >= 500.0 {
        assert!(
            sum > itl_us / 10.0 && sum < itl_us * 10.0,
            "profile sum {sum:.1}us should be order-of-magnitude of ITL {itl_us:.1}us; stats={stats:?}"
        );
    } else {
        let n = out.token_ids.len().max(1) as f64;
        let per_tok = elapsed_us / n;
        assert!(
            sum > per_tok / 20.0 && sum < per_tok * 20.0,
            "profile sum {sum:.1}us vs wall/token {per_tok:.1}us; stats={stats:?}"
        );
    }
}

#[cfg(feature = "hip")]
#[test]
fn generate_runs_hip_gemm_and_attention() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    assert!(out.token_ids.iter().all(|_| true));
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.gpu_forwards > 0,
        "target decode must run HIP kernels, gpu_forwards={}",
        stats.gpu_forwards
    );
    assert!(
        stats.elementwise_gpu_ops > 0,
        "HIP forward must run RoPE/RMSNorm/SiLU, elementwise_gpu_ops={}",
        stats.elementwise_gpu_ops
    );
}

#[cfg(feature = "hip")]
#[test]
fn asymmetric_placement_uses_distinct_devices_when_two_gpus() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    let gpus = hip_gpus(&engine);
    if gpus.is_empty() {
        return;
    }
    let inv = engine.inventory();
    if gpus.len() >= 2 {
        assert_ne!(
            inv.target_device, inv.draft_device,
            "target and draft must sit on distinct HIP devices"
        );
        assert!(inv.target_device >= 0 && inv.draft_device >= 0);
    } else {
        assert!(
            inv.target_device >= 0,
            "single GPU still runs generate on HIP"
        );
        assert!(
            inv.draft_device < 0,
            "one GPU must fall back to CPU draft (no second resident device)"
        );
        let out = engine.generate("hello", greedy(2)).unwrap();
        assert!(!out.token_ids.is_empty());
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_tree_packed_verify_uses_device_mask_and_matches_linear() {
    let mut linear = hip_auto_synthetic();
    linear.speculative.enabled = true;
    linear.features.ngram_speculation = Some(true);
    linear.features.draft_model_speculation = Some(true);
    linear.features.tree_speculation = Some(false);
    linear.speculative.tree.enabled = false;
    let mut tree = linear.clone();
    tree.speculative.tree.enabled = true;
    tree.speculative.tree.max_nodes = 16;
    tree.features.tree_speculation = Some(true);
    let linear_e = Engine::new(linear).unwrap();
    let tree_e = Engine::new(tree).unwrap();
    if hip_gpus(&tree_e).is_empty() {
        return;
    }
    let params = greedy(8);
    let base = linear_e.generate("hello", params.clone()).unwrap();
    let forest = tree_e.generate("hello", params).unwrap();
    assert_eq!(
        forest.token_ids, base.token_ids,
        "packed tree verify must match linear greedy; tree={:?} linear={:?}",
        forest.token_ids, base.token_ids
    );
    let stats = tree_e.stats();
    assert!(
        stats.tree_nodes > 0,
        "tree Feature Flag must build a Token Forest; stats={stats:?}"
    );
    assert!(
        stats.tree_gpu_verifies > 0,
        "ancestry mask must run on the target GPU; stats={stats:?}"
    );
    assert!(stats.tree_path_valid);
    assert_eq!(linear_e.stats().tree_gpu_verifies, 0);
}

/// Linear draft verify must run one Target Model forward over the proposed span,
/// not one decode per draft token.
#[cfg(feature = "hip")]
#[test]
fn hip_speculative_linear_verify_packs_draft_span() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_speculative_linear_verify_packs_draft_span: no HIP device");
        return;
    }
    let target = std::env::temp_dir().join("allpaca_hip_pack_verify_tgt.gguf");
    let draft = std::env::temp_dir().join("allpaca_hip_pack_verify_dft.gguf");
    write_packed_f16_gguf(&target);
    write_packed_f16_gguf(&draft);
    let mut cfg = hip_7b_config(target.clone());
    cfg.model.draft_path = Some(draft.clone());
    cfg.speculative.enabled = true;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    cfg.features.tree_speculation = Some(false);
    cfg.speculative.min_depth = 2;
    cfg.speculative.max_depth = 4;
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 64 * 1024 * 1024;
    cfg.kv.dram_bytes = 64 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("packed F16 target+draft must load");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&target);
        let _ = std::fs::remove_file(&draft);
        eprintln!("skip hip_speculative_linear_verify_packs_draft_span: no HIP device");
        return;
    }
    let out = engine.generate("hello", greedy(8)).expect("packed linear verify generate");
    assert!(!out.token_ids.is_empty());
    let n = out.token_ids.len() as u64;
    let stats = engine.stats();
    assert!(
        stats.linear_packed_verifies > 0,
        "identical F16 target+draft must pack the linear span; gen={n} stats={stats:?}"
    );
    assert!(
        stats.target_gpu_forwards < n.saturating_add(stats.prefill_chunks),
        "packed verify must do fewer target forwards than generated tokens plus prefill; gen={n} stats={stats:?}"
    );
    let _ = std::fs::remove_file(target);
    let _ = std::fs::remove_file(draft);
}

#[cfg(feature = "hip")]
fn hip_speculative_synthetic() -> Config {
    let mut cfg = hip_auto_synthetic();
    cfg.speculative.enabled = true;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    cfg.features.tree_speculation = Some(false);
    cfg
}

#[cfg(feature = "hip")]
#[test]
fn speculative_generate_keeps_draft_kv_on_draft_device() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    if hip_gpus(&engine).len() < 2 {
        return;
    }
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.draft_kv_bytes_on_device > 0,
        "draft DenseKv must be hipMalloc on the draft device; stats={stats:?}"
    );
    assert!(
        stats.kv_bytes_on_device >= stats.draft_kv_bytes_on_device,
        "kv_bytes_on_device must include draft pages; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn speculative_generate_draft_decode_does_not_h2d_activations() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    if hip_gpus(&engine).len() < 2 {
        return;
    }
    let after_load = engine.stats();
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let after = engine.stats();
    assert_eq!(
        after.activation_h2d_bytes, after_load.activation_h2d_bytes,
        "draft decode must stay on device; activation_h2d grew {} -> {}; stats={after:?}",
        after_load.activation_h2d_bytes, after.activation_h2d_bytes
    );
    assert!(
        after.draft_tokens_copied > 0,
        "draft token ids may still host-stage; stats={after:?}"
    );
    assert_eq!(after.weight_bytes_migrated, 0);
    assert!(after.draft_gpu_forwards > 0 && after.target_gpu_forwards > 0);
}

#[test]
fn cpu_only_speculative_draft_kv_stays_on_host() {
    let mut cfg = cpu_only_synthetic();
    cfg.speculative.enabled = true;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    let engine = Engine::new(cfg).unwrap();
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "cpu");
    assert_eq!(stats.draft_kv_bytes_on_device, 0);
    assert_eq!(stats.kv_bytes_on_device, 0);
    assert_eq!(stats.activation_h2d_bytes, 0);
}

#[cfg(feature = "hip")]
#[test]
fn one_gpu_speculative_draft_kv_stays_on_host() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    let gpus = hip_gpus(&engine);
    if gpus.len() != 1 {
        return;
    }
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.draft_device < 0,
        "one GPU must fall back to host draft; stats={stats:?}"
    );
    assert_eq!(stats.draft_kv_bytes_on_device, 0);
}

#[cfg(feature = "hip")]
#[test]
fn speculative_generate_runs_draft_and_target_on_distinct_gpus() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    let gpus = hip_gpus(&engine);
    if gpus.len() < 2 {
        return;
    }
    let inv = engine.inventory();
    assert_ne!(
        inv.target_device, inv.draft_device,
        "target stays on the fast GPU; draft stays on the slow GPU"
    );
    assert!(inv.target_device >= 0 && inv.draft_device >= 0);
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.target_device, inv.target_device);
    assert_eq!(stats.draft_device, inv.draft_device);
    assert!(
        stats.target_gpu_forwards > 0,
        "target verify must run HIP on the target device; stats={stats:?}"
    );
    assert!(
        stats.draft_gpu_forwards > 0,
        "draft propose must run HIP on the draft device; stats={stats:?}"
    );
    assert!(
        stats.gpu_forwards >= stats.target_gpu_forwards + stats.draft_gpu_forwards,
        "gpu_forwards must count both resident models; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn speculative_synthetic_draft_forwards_stay_linear_in_generated_tokens() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    if hip_gpus(&engine).len() < 2 {
        return;
    }
    let gen = 24usize;
    let out = engine.generate("hello", greedy(gen)).unwrap();
    assert!(!out.token_ids.is_empty());
    let n = out.token_ids.len() as u64;
    let stats = engine.stats();
    // Prefix reuse: ~prompt + steps*(1 refresh + depth). Replay would be ~O(n²).
    assert!(
        stats.draft_gpu_forwards > 0 && stats.draft_gpu_forwards < n.saturating_mul(16).max(32),
        "draft GPU forwards must stay linear in generated tokens; gen={n} stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn speculative_generate_host_stages_tokens_not_weights() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    if hip_gpus(&engine).len() < 2 {
        return;
    }
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(
        stats.weight_bytes_migrated, 0,
        "resident speculation must not copy weights between GPUs; stats={stats:?}"
    );
    assert!(
        stats.draft_tokens_copied > 0,
        "draft token ids must cross via host staging; stats={stats:?}"
    );
    assert!(
        stats.host_staging_bytes > 0,
        "host-staged draft traffic must be recorded; stats={stats:?}"
    );
    const TOKEN_SCALE_BOUND: u64 = 64 * 1024;
    assert!(
        stats.host_staging_bytes < TOKEN_SCALE_BOUND,
        "host staging must stay token/logit scale, not weight scale; bytes={} stats={stats:?}",
        stats.host_staging_bytes
    );
}

#[cfg(feature = "hip")]
#[test]
fn host_staging_transfer_path_still_uses_both_gpus() {
    let engine = Engine::new(hip_speculative_synthetic()).unwrap();
    if hip_gpus(&engine).len() < 2 {
        return;
    }
    let stats_boot = engine.stats();
    assert!(
        stats_boot.transfer_path == "HostStaging" || stats_boot.transfer_path == "P2p",
        "unexpected transfer_path {}",
        stats_boot.transfer_path
    );
    let out = engine.generate("hello", greedy(8)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_ne!(stats.target_device, stats.draft_device);
    assert!(stats.target_gpu_forwards > 0 && stats.draft_gpu_forwards > 0);
    if stats.transfer_path == "HostStaging" {
        assert!(
            !stats.p2p_capable || !stats.p2p_enabled || stats.p2p_gbps == 0.0,
            "HostStaging is valid when P2P is unavailable or slower; stats={stats:?}"
        );
    }
}

#[cfg(feature = "hip")]
fn write_tiny_q4k_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_q4_k, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor,
    };
    use std::collections::HashMap;

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
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
    let embd_q = quantize_q4_k(&vec![0.01f32; vocab * dim]);
    let q_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
    let k_q = quantize_q4_k(&vec![0.01f32; 2 * dim]);
    let v_q = quantize_q4_k(&vec![0.01f32; 2 * dim]);
    let o_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
    let gate_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
    let up_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
    let down_q = quantize_q4_k(&vec![0.01f32; dim * ff]);
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
                ggml_type: GgmlType::Q4_K,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &down_q,
            },
        ],
    )
    .unwrap();
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_from_tiny_q4k_gguf() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_from_tiny_q4k_gguf: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_hip_tiny_q4k.gguf");
    write_tiny_q4k_gguf(&path);
    let mut cfg = hip_7b_config(path.clone());
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 16 * 1024 * 1024;
    cfg.kv.dram_bytes = 16 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("tiny Q4_K GGUF must load on HIP");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_generate_from_tiny_q4k_gguf: no HIP device");
        return;
    }
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny Q4_K HIP generate must produce tokens");
    assert!(!out.token_ids.is_empty());
    assert_eq!(engine.stats().backend, "hip");
    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "hip")]
fn write_tiny_uniform_kquant_gguf(
    path: &std::path::Path,
    ggml_type: allpaca_gguf::GgmlType,
    quantize: fn(&[f32]) -> Vec<u8>,
) {
    use allpaca_gguf::{write_gguf_quant, GgufValue, GgufWriteQuantTensor, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
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
    .unwrap();
}

/// F16 GGUF with GEMM tensors larger than the n≤65536 F32 materialize cutoff.
#[cfg(feature = "hip")]
fn write_packed_f16_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf_quant, GgufValue, GgufWriteQuantTensor, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 288usize;
    let vocab = 4usize;
    let n_heads = 8usize;
    let n_kv_heads = 2usize;
    let kv_dim = dim / n_heads * n_kv_heads;
    let ff = 768usize;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
    meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
    meta.insert(
        "llama.attention.head_count".into(),
        GgufValue::Uint32(n_heads as u32),
    );
    meta.insert(
        "llama.attention.head_count_kv".into(),
        GgufValue::Uint32(n_kv_heads as u32),
    );
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
    let quantize = allpaca_gguf::quantize_f16;
    let ggml_type = allpaca_gguf::GgmlType::F16;
    let embd_q = quantize(&vec![0.01f32; vocab * dim]);
    let q_q = quantize(&vec![0.01f32; dim * dim]);
    let k_q = quantize(&vec![0.01f32; kv_dim * dim]);
    let v_q = quantize(&vec![0.01f32; kv_dim * dim]);
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
                dims: vec![kv_dim as u64, dim as u64],
                ggml_type,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![kv_dim as u64, dim as u64],
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
    .unwrap();
}

#[cfg(feature = "hip")]
fn write_tiny_f32_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
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
        ("token_embd.weight".into(), vec![vocab as u64, dim as u64], vec![0.01; vocab * dim]),
        ("blk.0.attn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
        ("blk.0.attn_q.weight".into(), vec![dim as u64, dim as u64], vec![0.01; dim * dim]),
        ("blk.0.attn_k.weight".into(), vec![2, dim as u64], vec![0.01; 2 * dim]),
        ("blk.0.attn_v.weight".into(), vec![2, dim as u64], vec![0.01; 2 * dim]),
        ("blk.0.attn_output.weight".into(), vec![dim as u64, dim as u64], vec![0.01; dim * dim]),
        ("blk.0.ffn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
        ("blk.0.ffn_gate.weight".into(), vec![ff as u64, dim as u64], vec![0.01; ff * dim]),
        ("blk.0.ffn_up.weight".into(), vec![ff as u64, dim as u64], vec![0.01; ff * dim]),
        ("blk.0.ffn_down.weight".into(), vec![dim as u64, ff as u64], vec![0.01; dim * ff]),
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
    write_gguf(path, &meta, &views).unwrap();
}

#[cfg(feature = "hip")]
fn write_tiny_q4k_q6k_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_q4_k, quantize_q6_k, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
        GgufWriteTensor,
    };
    use std::collections::HashMap;

    let dim = 8usize;
    let vocab = 4usize;
    let ff = 16usize;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
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
    let embd_q = quantize_q4_k(&vec![0.01f32; vocab * dim]);
    let q_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
    let k_q = quantize_q4_k(&vec![0.01f32; 2 * dim]);
    let v_q = quantize_q6_k(&vec![0.01f32; 2 * dim]);
    let o_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
    let gate_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
    let up_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
    let down_q = quantize_q6_k(&vec![0.01f32; dim * ff]);
    let output_q = quantize_q6_k(&vec![0.01f32; vocab * dim]);
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
                ggml_type: GgmlType::Q4_K,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q6_K,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::Q6_K,
                bytes: &down_q,
            },
            GgufWriteQuantTensor {
                name: "output.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type: GgmlType::Q6_K,
                bytes: &output_q,
            },
        ],
    )
    .unwrap();
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_from_tiny_q6k_gguf() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_from_tiny_q6k_gguf: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_hip_tiny_q6k.gguf");
    write_tiny_uniform_kquant_gguf(&path, allpaca_gguf::GgmlType::Q6_K, allpaca_gguf::quantize_q6_k);
    let mut cfg = hip_7b_config(path.clone());
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 16 * 1024 * 1024;
    cfg.kv.dram_bytes = 16 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("tiny Q6_K GGUF must load on HIP");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_generate_from_tiny_q6k_gguf: no HIP device");
        return;
    }
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny Q6_K HIP generate must produce tokens");
    assert!(!out.token_ids.is_empty());
    assert_eq!(engine.stats().backend, "hip");
    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_from_tiny_f16_gguf() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_from_tiny_f16_gguf: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_hip_tiny_f16.gguf");
    write_tiny_uniform_kquant_gguf(&path, allpaca_gguf::GgmlType::F16, allpaca_gguf::quantize_f16);
    let mut cfg = hip_7b_config(path.clone());
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 16 * 1024 * 1024;
    cfg.kv.dram_bytes = 16 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("tiny F16 GGUF must load on HIP");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_generate_from_tiny_f16_gguf: no HIP device");
        return;
    }
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny F16 HIP generate must produce tokens");
    assert!(!out.token_ids.is_empty());
    assert_eq!(engine.stats().backend, "hip");
    let _ = std::fs::remove_file(path);
}

/// Large F16 GEMM tensors stay packed (not F32) and still generate on HIP.
#[cfg(feature = "hip")]
#[test]
fn hip_packed_f16_gguf_stays_half_width_and_generates() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_packed_f16_gguf_stays_half_width_and_generates: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_hip_packed_f16.gguf");
    write_packed_f16_gguf(&path);
    let mut cfg = hip_7b_config(path.clone());
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 64 * 1024 * 1024;
    cfg.kv.dram_bytes = 64 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("packed F16 GGUF must load on HIP");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_packed_f16_gguf_stays_half_width_and_generates: no HIP device");
        return;
    }
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.weight_bytes_on_device > 0 && stats.weight_bytes_on_device < 2_400_000,
        "packed F16 must stay well under F32 materialize (~3.5MiB); stats={stats:?}"
    );
    let out = engine
        .generate("hello", greedy(4))
        .expect("packed F16 HIP generate must produce tokens");
    assert!(!out.token_ids.is_empty());
    let _ = std::fs::remove_file(path);
}

/// Packed F16 as the Draft Model must use F16 GEMV on the slow GPU. Target
/// decode already dispatched by WeightKind; the draft dense path used to run
/// F32 GEMV over half-width weights and hang on a real F16 draft (3B lm_head).
#[cfg(feature = "hip")]
#[test]
fn hip_packed_f16_draft_speculates_on_second_gpu() {
    if allpaca_hip::device_count() < 2 {
        eprintln!("skip hip_packed_f16_draft_speculates_on_second_gpu: need 2 HIP devices");
        return;
    }
    let target = std::env::temp_dir().join("allpaca_hip_packed_f16_spec_tgt.gguf");
    let draft = std::env::temp_dir().join("allpaca_hip_packed_f16_spec_dft.gguf");
    write_packed_f16_gguf(&target);
    write_packed_f16_gguf(&draft);
    let mut cfg = hip_7b_config(target.clone());
    cfg.model.draft_path = Some(draft.clone());
    cfg.speculative.enabled = true;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    cfg.features.tree_speculation = Some(false);
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 64 * 1024 * 1024;
    cfg.kv.dram_bytes = 64 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("packed F16 target+draft must load");
    if hip_gpus(&engine).len() < 2 {
        let _ = std::fs::remove_file(&target);
        let _ = std::fs::remove_file(&draft);
        eprintln!("skip hip_packed_f16_draft_speculates_on_second_gpu: need 2 HIP devices");
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let engine_t = engine.clone();
    std::thread::spawn(move || {
        let _ = tx.send(engine_t.generate("hello", greedy(2)));
    });
    let out = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("packed F16 draft generate hung — draft path must dispatch F16 GEMV")
        .expect("packed F16 draft generate must run");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.draft_gpu_forwards > 0,
        "draft GPU must run packed F16 forwards; stats={stats:?}"
    );
    let _ = std::fs::remove_file(target);
    let _ = std::fs::remove_file(draft);
}

#[cfg(feature = "hip")]
fn tiny_written_cfg(path: std::path::PathBuf, cpu_only: bool) -> Config {
    let mut cfg = hip_7b_config(path);
    cfg.engine.cpu_only = cpu_only;
    if cpu_only {
        cfg.engine.gpu_backend = "cpu".into();
    }
    cfg.kv.page_tokens = 32;
    cfg.kv.vram_bytes = 16 * 1024 * 1024;
    cfg.kv.dram_bytes = 16 * 1024 * 1024;
    cfg
}

#[cfg(feature = "hip")]
#[test]
fn hip_greedy_long_prompt_matches_cpu_on_tiny_written_ggufs() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_greedy_long_prompt_matches_cpu_on_tiny_written_ggufs: no HIP device");
        return;
    }
    let prompt = long_hello_prompt();
    let cases: [(&str, Box<dyn Fn(&std::path::Path)>); 3] = [
        ("f32", Box::new(write_tiny_f32_gguf)),
        (
            "f16",
            Box::new(|p| {
                write_tiny_uniform_kquant_gguf(p, allpaca_gguf::GgmlType::F16, allpaca_gguf::quantize_f16)
            }),
        ),
        (
            "q8",
            Box::new(|p| {
                write_tiny_uniform_kquant_gguf(
                    p,
                    allpaca_gguf::GgmlType::Q8_0,
                    allpaca_gguf::quantize_q8_0,
                )
            }),
        ),
    ];
    for (label, write) in cases {
        let path = std::env::temp_dir().join(format!("allpaca_hip_tiny_prefill_{label}.gguf"));
        write(&path);
        let cpu = Engine::new(tiny_written_cfg(path.clone(), true)).expect("tiny CPU load");
        let hip = Engine::new(tiny_written_cfg(path.clone(), false)).expect("tiny HIP load");
        if hip_gpus(&hip).is_empty() {
            let _ = std::fs::remove_file(&path);
            eprintln!("skip {label}: no HIP device");
            continue;
        }
        let cpu_out = cpu
            .generate(&prompt, greedy(4))
            .unwrap_or_else(|e| panic!("{label} CPU generate: {e}"));
        let hip_out = hip
            .generate(&prompt, greedy(4))
            .unwrap_or_else(|e| panic!("{label} HIP generate: {e}"));
        assert!(
            cpu_out.prompt_tokens >= 16,
            "{label} prompt_tokens={}",
            cpu_out.prompt_tokens
        );
        assert_eq!(
            hip_out.token_ids, cpu_out.token_ids,
            "{label} batched HIP greedy must match CPU; hip={:?} cpu={:?}",
            hip_out.token_ids, cpu_out.token_ids
        );
        let stats = hip.stats();
        assert_eq!(stats.backend, "hip");
        assert!(
            stats.gpu_forwards < hip_out.prompt_tokens as u64,
            "{label} must batch prefill; gpu_forwards={} prompt_tokens={}",
            stats.gpu_forwards,
            hip_out.prompt_tokens
        );
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_from_tiny_q4k_q6k_gguf() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_from_tiny_q4k_q6k_gguf: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_hip_tiny_q4k_q6k.gguf");
    write_tiny_q4k_q6k_gguf(&path);
    let mut cfg = hip_7b_config(path.clone());
    cfg.kv.page_tokens = 8;
    cfg.kv.vram_bytes = 16 * 1024 * 1024;
    cfg.kv.dram_bytes = 16 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("tiny mixed Q4_K+Q6_K GGUF must load on HIP");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_generate_from_tiny_q4k_q6k_gguf: no HIP device");
        return;
    }
    let out = engine
        .generate("hello", greedy(4))
        .expect("tiny mixed Q4_K+Q6_K HIP generate must produce tokens");
    assert!(!out.token_ids.is_empty());
    assert_eq!(engine.stats().backend, "hip");
    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "hip")]
fn large_gguf_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Gated 7B (or Llama-3 8B-class) GGUF. Skips unless `ALLPACA_7B_GGUF` /
/// `ALLPACA_GGUF` or a discovered local file exists. Never downloads weights.
#[cfg(feature = "hip")]
fn gated_7b_gguf() -> Option<std::path::PathBuf> {
    for key in ["ALLPACA_7B_GGUF", "ALLPACA_GGUF"] {
        if let Ok(raw) = std::env::var(key) {
            let path = std::path::PathBuf::from(raw);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    const DISCOVERED: &[&str] = &[
        r"E:\Models\huggingface\models--bartowski--Llama-3-Groq-8B-Tool-Use-GGUF\snapshots\2d49903869805faad16daab2450fcda3487685b8\Llama-3-Groq-8B-Tool-Use-Q8_0.gguf",
    ];
    DISCOVERED
        .iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.is_file())
}

/// Gated Q4_K GGUF. Uses `ALLPACA_Q4K_GGUF`, or `ALLPACA_7B_GGUF` only when that
/// file is actually Q4_K (the Groq Q8_0 bench path is unchanged).
#[cfg(feature = "hip")]
fn gated_q4k_gguf() -> Option<std::path::PathBuf> {
    if let Ok(raw) = std::env::var("ALLPACA_Q4K_GGUF") {
        let path = std::path::PathBuf::from(raw);
        if path.is_file() {
            return Some(path);
        }
    }
    gated_7b_gguf().filter(|p| gguf_is_supported_q4k(p))
}

/// Gated mixed Q4_K + Q6_K GGUF (typical Q4_K_M). Skips 27B/31B-class files.
#[cfg(feature = "hip")]
fn gated_mixed_kquant_gguf() -> Option<std::path::PathBuf> {
    if let Ok(raw) = std::env::var("ALLPACA_Q4KM_GGUF") {
        let path = std::path::PathBuf::from(raw);
        if path.is_file() && gguf_is_mixed_kquant(&path) && !gguf_too_large_for_gated(&path) {
            return Some(path);
        }
    }
    gated_q4k_gguf().filter(|p| gguf_is_mixed_kquant(p) && !gguf_too_large_for_gated(p))
}

#[cfg(feature = "hip")]
fn gguf_too_large_for_gated(path: &std::path::Path) -> bool {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_uppercase();
    if name.contains("27B") || name.contains("31B") || name.contains("35B") {
        return true;
    }
    path.metadata()
        .map(|m| m.len() > 12 * 1024 * 1024 * 1024)
        .unwrap_or(true)
}

#[cfg(feature = "hip")]
fn gguf_is_mixed_kquant(path: &std::path::Path) -> bool {
    let Ok(file) = allpaca_gguf::GgufFile::open(path) else {
        return false;
    };
    let mut saw_q4k = false;
    let mut saw_q6k = false;
    for t in &file.tensors {
        match t.ggml_type {
            allpaca_gguf::GgmlType::Q4_K => saw_q4k = true,
            allpaca_gguf::GgmlType::Q6_K => saw_q6k = true,
            allpaca_gguf::GgmlType::F32
            | allpaca_gguf::GgmlType::F16
            | allpaca_gguf::GgmlType::Q4_0
            | allpaca_gguf::GgmlType::Q8_0
            | allpaca_gguf::GgmlType::Q5_K => {}
            _ => return false,
        }
    }
    saw_q4k && saw_q6k
}

#[cfg(feature = "hip")]
fn gguf_is_supported_q4k(path: &std::path::Path) -> bool {
    let Ok(file) = allpaca_gguf::GgufFile::open(path) else {
        return false;
    };
    let mut saw_q4k = false;
    for t in &file.tensors {
        match t.ggml_type {
            allpaca_gguf::GgmlType::Q4_K => saw_q4k = true,
            allpaca_gguf::GgmlType::F32
            | allpaca_gguf::GgmlType::F16
            | allpaca_gguf::GgmlType::Q4_0
            | allpaca_gguf::GgmlType::Q8_0
            | allpaca_gguf::GgmlType::Q5_K
            | allpaca_gguf::GgmlType::Q6_K => {}
            _ => return false,
        }
    }
    saw_q4k
}

#[cfg(feature = "hip")]
fn gated_draft_gguf() -> Option<std::path::PathBuf> {
    if let Ok(raw) = std::env::var("ALLPACA_DRAFT_GGUF") {
        let path = std::path::PathBuf::from(raw);
        if path.is_file() && looks_like_gguf(&path) {
            return Some(path);
        }
    }
    if let Some(p) = local_llama32_draft_gguf() {
        return Some(p);
    }
    const DISCOVERED: &[&str] = &[
        r"E:\Models\huggingface\models--bartowski--Llama-3.2-1B-Instruct-GGUF",
        r"E:\Models\huggingface\models--bartowski--Llama-3.2-3B-Instruct-GGUF",
        r"E:\Models\huggingface\models--ggml-org--Qwen2.5-Coder-0.5B-Q8_0-GGUF",
    ];
    for root in DISCOVERED {
        if let Some(p) = find_gguf_under(std::path::Path::new(root)) {
            return Some(p);
        }
    }
    None
}

#[cfg(feature = "hip")]
fn looks_like_gguf(path: &std::path::Path) -> bool {
    if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
        return path.is_file();
    }
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut f, &mut magic).is_ok() && magic == allpaca_gguf::GGUF_MAGIC
}

/// HF hub cache may store the file as `snapshots/<rev>/*.gguf` or as a hash under `blobs/`.
#[cfg(feature = "hip")]
fn find_gguf_under(root: &std::path::Path) -> Option<std::path::PathBuf> {
    if root.is_file() {
        return looks_like_gguf(root).then(|| root.to_path_buf());
    }
    if !root.is_dir() {
        return None;
    }
    let snapshots = root.join("snapshots");
    if snapshots.is_dir() {
        if let Ok(commits) = std::fs::read_dir(&snapshots) {
            for commit in commits.flatten() {
                let dir = commit.path();
                if !dir.is_dir() {
                    continue;
                }
                if let Ok(files) = std::fs::read_dir(&dir) {
                    for f in files.flatten() {
                        let p = f.path();
                        if p.extension().and_then(|e| e.to_str()) == Some("gguf") && p.is_file() {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    let blobs = root.join("blobs");
    if blobs.is_dir() {
        if let Ok(files) = std::fs::read_dir(&blobs) {
            for f in files.flatten() {
                let p = f.path();
                if p.is_file() && looks_like_gguf(&p) {
                    return Some(p);
                }
            }
        }
    }
    if let Ok(files) = std::fs::read_dir(root) {
        for f in files.flatten() {
            let p = f.path();
            if p.extension().and_then(|e| e.to_str()) == Some("gguf") && p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(feature = "hip")]
#[test]
fn discovers_qwen_coder_05b_gguf_in_hf_cache_when_present() {
    let root = std::path::Path::new(
        r"E:\Models\huggingface\models--ggml-org--Qwen2.5-Coder-0.5B-Q8_0-GGUF",
    );
    if !root.exists() {
        eprintln!("skip discovers_qwen_coder_05b_gguf_in_hf_cache_when_present: cache dir missing");
        return;
    }
    let found = find_gguf_under(root).expect("Qwen 0.5B HF cache must contain a GGUF blob or snapshots/*.gguf");
    assert!(found.is_file(), "discovered path must be a file: {}", found.display());
    eprintln!("discovered draft GGUF {}", found.display());
}

#[cfg(feature = "hip")]
fn local_models_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models")
}

/// Repo `models/` Llama-3.2 drafts beat the Qwen HF blob (same-family vocab).
#[cfg(feature = "hip")]
fn local_llama32_draft_gguf() -> Option<std::path::PathBuf> {
    let dir = local_models_dir();
    const NAMES: &[&str] = &[
        "Llama-3.2-1B-Instruct-f16.gguf",
        "Llama-3.2-3B-Instruct-f16.gguf",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
    ];
    for name in NAMES {
        let p = dir.join(name);
        if p.is_file() && looks_like_gguf(&p) {
            return Some(p);
        }
    }
    find_gguf_under(&dir).filter(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains("Llama-3.2"))
    })
}

#[cfg(feature = "hip")]
#[test]
fn gated_draft_prefers_local_llama32_3b_f16_over_qwen() {
    let p = local_models_dir().join("Llama-3.2-3B-Instruct-f16.gguf");
    if !p.is_file() {
        eprintln!(
            "skip gated_draft_prefers_local_llama32_3b_f16_over_qwen: missing {}",
            p.display()
        );
        return;
    }
    let found = gated_draft_gguf().expect("local Llama-3.2-3B F16 must be discoverable");
    assert_eq!(
        found.canonicalize().unwrap_or(found.clone()),
        p.canonicalize().unwrap_or(p.clone()),
        "same-family Llama draft must win over Qwen; found={}",
        found.display()
    );
}

/// Local Llama-3.2-3B F16 must stay packed (~6–8GiB), not F32 (~12GiB).
#[cfg(feature = "hip")]
#[test]
fn hip_local_llama32_f16_stays_packed_on_load() {
    let _guard = large_gguf_lock();
    let p = local_models_dir().join("Llama-3.2-3B-Instruct-f16.gguf");
    if !p.is_file() {
        eprintln!(
            "skip hip_local_llama32_f16_stays_packed_on_load: missing {}",
            p.display()
        );
        return;
    }
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_local_llama32_f16_stays_packed_on_load: no HIP device");
        return;
    }
    let mut cfg = hip_7b_config(p);
    cfg.kv.vram_bytes = 512 * 1024 * 1024;
    cfg.kv.dram_bytes = 512 * 1024 * 1024;
    let engine = Engine::new(cfg).expect("Llama-3.2-3B F16 must load on HIP");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_local_llama32_f16_stays_packed_on_load: no HIP device");
        return;
    }
    let stats = engine.stats();
    eprintln!(
        "llama32-3b-f16 packed weight_bytes_on_device={}",
        stats.weight_bytes_on_device
    );
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.weight_bytes_on_device > 5_000_000_000
            && stats.weight_bytes_on_device < 6_500_000_000,
        "F16 GEMM, lm_head, and token_embd must stay packed (F32 embed is ~6.7GiB); stats={stats:?}"
    );
    let out = engine
        .generate("hello", greedy(2))
        .expect("Llama-3.2-3B packed F16 must decode");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    eprintln!(
        "llama32-3b-f16 gen={:?} itl_ms_ema={:.2} lm_head_us={:.0} other_gpu_us={:.0}",
        out.text, stats.itl_ms_ema, stats.lm_head_us, stats.other_gpu_us
    );
}

/// Soak the slow GPU as Target Model. Skips unless `ALLPACA_OVERNIGHT=1`.
#[cfg(feature = "hip")]
#[test]
fn hip_overnight_llama32_f16_on_second_gpu_when_env_set() {
    if std::env::var("ALLPACA_OVERNIGHT").ok().as_deref() != Some("1") {
        eprintln!("skip hip_overnight_llama32_f16_on_second_gpu_when_env_set: set ALLPACA_OVERNIGHT=1");
        return;
    }
    let _guard = large_gguf_lock();
    let p = local_models_dir().join("Llama-3.2-3B-Instruct-f16.gguf");
    if !p.is_file() {
        eprintln!(
            "skip hip_overnight_llama32_f16_on_second_gpu_when_env_set: missing {}",
            p.display()
        );
        return;
    }
    if allpaca_hip::device_count() < 2 {
        eprintln!("skip hip_overnight_llama32_f16_on_second_gpu_when_env_set: need 2 HIP devices");
        return;
    }
    let soak_9060 = std::env::var("ALLPACA_OVERNIGHT_9060").ok().as_deref() == Some("1");
    let tokens = std::env::var("ALLPACA_OVERNIGHT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let rounds = std::env::var("ALLPACA_OVERNIGHT_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    let Some(target) = gated_7b_gguf() else {
        eprintln!("overnight skip speculative: no 8B GGUF");
        return;
    };
    let Some(draft) = gated_draft_gguf() else {
        eprintln!("overnight skip speculative: no draft GGUF");
        return;
    };
    let spec_tokens = std::env::var("ALLPACA_OVERNIGHT_SPEC_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let spec_rounds = std::env::var("ALLPACA_OVERNIGHT_SPEC_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut spec_cfg = hip_7b_config(target.clone());
    spec_cfg.model.draft_path = Some(draft.clone());
    spec_cfg.speculative.enabled = true;
    spec_cfg.features.draft_model_speculation = Some(true);
    spec_cfg.features.ngram_speculation = Some(false);
    spec_cfg.features.tree_speculation = Some(false);
    spec_cfg.kv.vram_bytes = 512 * 1024 * 1024;
    spec_cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("overnight spec (clean boot) Engine::new starting spec_tokens={spec_tokens} spec_rounds={spec_rounds}");
    let spec = Engine::new(spec_cfg).expect("overnight 8B+3B load");
    let stats = spec.stats();
    eprintln!(
        "overnight spec load target_device={} draft_device={} weight_bytes={} transfer_path={} p2p_status={}",
        stats.target_device,
        stats.draft_device,
        stats.weight_bytes_on_device,
        stats.transfer_path,
        stats.p2p_status
    );
    for round in 0..spec_rounds {
        eprintln!(
            "overnight spec round={}/{} generate starting",
            round + 1,
            spec_rounds
        );
        let before = spec.stats();
        let t0 = std::time::Instant::now();
        let out = generate_or_timeout(
            &spec,
            "hello",
            spec_tokens,
            std::time::Duration::from_secs(180),
        );
        let stats = spec.stats();
        let secs = t0.elapsed().as_secs_f64();
        let tps = out.token_ids.len() as f64 / secs.max(1e-6);
        let draft_delta = stats.draft_gpu_forwards.saturating_sub(before.draft_gpu_forwards);
        eprintln!(
            "overnight spec round={}/{} gen={} tps={:.2} itl_ms={:.2} acceptance={:.3} target_fwd={} draft_fwd={} draft_fwd_delta={} host_staging_bytes={} weight_bytes_migrated={} text_prefix={:?}",
            round + 1,
            spec_rounds,
            out.token_ids.len(),
            tps,
            stats.itl_ms_ema,
            stats.acceptance_rate,
            stats.target_gpu_forwards,
            stats.draft_gpu_forwards,
            draft_delta,
            stats.host_staging_bytes,
            stats.weight_bytes_migrated,
            out.text.chars().take(40).collect::<String>()
        );
        assert!(
            draft_delta > 0 && draft_delta < (spec_tokens as u64).saturating_mul(16).max(64),
            "overnight spec round {} must reuse draft KV; gen={} draft_fwd_delta={} stats={stats:?}",
            round + 1,
            out.token_ids.len(),
            draft_delta
        );
    }
    drop(spec);

    if !soak_9060 {
        eprintln!("overnight skip 9060 soak: set ALLPACA_OVERNIGHT_9060=1");
        return;
    }
    let mut cfg = hip_7b_config(p);
    cfg.engine.target_device = DeviceSelector::Hip(1);
    cfg.engine.draft_device = DeviceSelector::Cpu;
    cfg.kv.vram_bytes = 512 * 1024 * 1024;
    cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("overnight 9060 Engine::new starting tokens={tokens} rounds={rounds}");
    let engine = Engine::new(cfg).expect("Llama-3.2-3B F16 must load on HIP:1");
    let stats = engine.stats();
    eprintln!(
        "overnight 9060 load backend={} target_device={} draft_device={} weight_bytes={} transfer_path={} p2p_status={}",
        stats.backend,
        stats.target_device,
        stats.draft_device,
        stats.weight_bytes_on_device,
        stats.transfer_path,
        stats.p2p_status
    );
    assert_eq!(stats.target_device, 1, "overnight soak must run on the second HIP device");
    for round in 0..rounds {
        let t0 = std::time::Instant::now();
        let out = engine
            .generate("hello", greedy(tokens))
            .expect("overnight 9060 generate");
        let stats = engine.stats();
        let secs = t0.elapsed().as_secs_f64();
        let tps = out.token_ids.len() as f64 / secs.max(1e-6);
        eprintln!(
            "overnight 9060 round={}/{} gen={} tps={:.2} itl_ms={:.2} lm_head_us={:.0} other_gpu_us={:.0} q8_gemm_us={:.0} attention_us={:.0} text_prefix={:?}",
            round + 1,
            rounds,
            out.token_ids.len(),
            tps,
            stats.itl_ms_ema,
            stats.lm_head_us,
            stats.other_gpu_us,
            stats.q8_gemm_us,
            stats.attention_us,
            out.text.chars().take(40).collect::<String>()
        );
    }
    drop(engine);

    let Some(target) = gated_7b_gguf() else {
        eprintln!("overnight skip speculative: no 8B GGUF");
        return;
    };
    let Some(draft) = gated_draft_gguf() else {
        eprintln!("overnight skip speculative: no draft GGUF");
        return;
    };
    let spec_tokens = std::env::var("ALLPACA_OVERNIGHT_SPEC_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let spec_rounds = std::env::var("ALLPACA_OVERNIGHT_SPEC_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let mut spec_cfg = hip_7b_config(target);
    spec_cfg.model.draft_path = Some(draft);
    spec_cfg.speculative.enabled = true;
    spec_cfg.features.draft_model_speculation = Some(true);
    spec_cfg.features.ngram_speculation = Some(false);
    spec_cfg.features.tree_speculation = Some(false);
    spec_cfg.kv.vram_bytes = 512 * 1024 * 1024;
    spec_cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("overnight spec (post-9060) Engine::new starting spec_tokens={spec_tokens} spec_rounds={spec_rounds}");
    let spec = Engine::new(spec_cfg).expect("overnight 8B+3B load");
    let stats = spec.stats();
    eprintln!(
        "overnight spec load target_device={} draft_device={} weight_bytes={} transfer_path={} p2p_status={} host_staging_bytes={}",
        stats.target_device,
        stats.draft_device,
        stats.weight_bytes_on_device,
        stats.transfer_path,
        stats.p2p_status,
        stats.host_staging_bytes
    );
    for round in 0..spec_rounds {
        eprintln!(
            "overnight spec round={}/{} generate starting",
            round + 1,
            spec_rounds
        );
        let before = spec.stats();
        let t0 = std::time::Instant::now();
        let out = generate_or_timeout(
            &spec,
            "hello",
            spec_tokens,
            std::time::Duration::from_secs(180),
        );
        let stats = spec.stats();
        let secs = t0.elapsed().as_secs_f64();
        let tps = out.token_ids.len() as f64 / secs.max(1e-6);
        let draft_delta = stats.draft_gpu_forwards.saturating_sub(before.draft_gpu_forwards);
        eprintln!(
            "overnight spec round={}/{} gen={} tps={:.2} itl_ms={:.2} acceptance={:.3} target_fwd={} draft_fwd={} draft_fwd_delta={} host_staging_bytes={} weight_bytes_migrated={} text_prefix={:?}",
            round + 1,
            spec_rounds,
            out.token_ids.len(),
            tps,
            stats.itl_ms_ema,
            stats.acceptance_rate,
            stats.target_gpu_forwards,
            stats.draft_gpu_forwards,
            draft_delta,
            stats.host_staging_bytes,
            stats.weight_bytes_migrated,
            out.text.chars().take(40).collect::<String>()
        );
        assert!(
            draft_delta > 0 && draft_delta < (spec_tokens as u64).saturating_mul(16).max(64),
            "post-9060 spec round {} must reuse draft KV; gen={} draft_fwd_delta={} stats={stats:?}",
            round + 1,
            out.token_ids.len(),
            draft_delta
        );
    }
}

/// 8B-only greedy on the Target Model GPU (RX 9070 / HIP:0). Dual-GPU spec waits.
/// Requires `ALLPACA_OVERNIGHT=1` and `ALLPACA_OVERNIGHT_9070=1`. Skips when more
/// than one HIP device is visible so boot P2P probe cannot touch the 9060;
/// run with `HIP_VISIBLE_DEVICES=0`.
#[cfg(feature = "hip")]
#[test]
fn hip_overnight_8b_greedy_on_target_gpu_when_env_set() {
    if std::env::var("ALLPACA_OVERNIGHT").ok().as_deref() != Some("1") {
        eprintln!("skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: set ALLPACA_OVERNIGHT=1");
        return;
    }
    if std::env::var("ALLPACA_OVERNIGHT_9070").ok().as_deref() != Some("1") {
        eprintln!("skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: set ALLPACA_OVERNIGHT_9070=1");
        return;
    }
    let _guard = large_gguf_lock();
    let Some(path) = gated_7b_gguf() else {
        eprintln!("skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: no 8B GGUF");
        return;
    };
    let n_dev = allpaca_hip::device_count();
    if n_dev == 0 {
        eprintln!("skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: no HIP device");
        return;
    }
    if n_dev > 1 {
        eprintln!(
            "skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: {n_dev} HIP devices visible; set HIP_VISIBLE_DEVICES=0 so the 9060 is not P2P-probed"
        );
        return;
    }
    let tokens = std::env::var("ALLPACA_OVERNIGHT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let rounds = std::env::var("ALLPACA_OVERNIGHT_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let mut cfg = hip_7b_config(path);
    cfg.engine.target_device = DeviceSelector::Hip(0);
    cfg.engine.draft_device = DeviceSelector::Cpu;
    cfg.speculative.enabled = false;
    cfg.features.draft_model_speculation = Some(false);
    cfg.kv.vram_bytes = 512 * 1024 * 1024;
    cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("overnight 9070 Engine::new starting tokens={tokens} rounds={rounds}");
    let engine = Engine::new(cfg).expect("8B GGUF must load on HIP:0");
    let stats = engine.stats();
    eprintln!(
        "overnight 9070 load backend={} target_device={} draft_device={} weight_bytes={} gpu_count={}",
        stats.backend,
        stats.target_device,
        stats.draft_device,
        stats.weight_bytes_on_device,
        hip_gpus(&engine).len()
    );
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.target_device, 0,
        "9070 soak must run on the only visible HIP device"
    );
    assert_eq!(stats.draft_device, -1, "9070 soak must not place a Draft Model");
    let timeout = std::time::Duration::from_secs((tokens as u64).saturating_mul(3).max(180));
    for round in 0..rounds {
        eprintln!(
            "overnight 9070 round={}/{} generate starting",
            round + 1,
            rounds
        );
        let before = engine.stats();
        let t0 = std::time::Instant::now();
        let out = generate_or_timeout(&engine, "hello", tokens, timeout);
        let stats = engine.stats();
        let secs = t0.elapsed().as_secs_f64();
        let tps = out.token_ids.len() as f64 / secs.max(1e-6);
        let target_delta = stats
            .target_gpu_forwards
            .saturating_sub(before.target_gpu_forwards);
        eprintln!(
            "overnight 9070 round={}/{} gen={} tps={:.2} itl_ms={:.2} target_fwd={} target_fwd_delta={} draft_fwd={} q8_gemm_us={:.0} attention_us={:.0} lm_head_us={:.0} other_gpu_us={:.0} host_us={:.0} text_prefix={:?}",
            round + 1,
            rounds,
            out.token_ids.len(),
            tps,
            stats.itl_ms_ema,
            stats.target_gpu_forwards,
            target_delta,
            stats.draft_gpu_forwards,
            stats.q8_gemm_us,
            stats.attention_us,
            stats.lm_head_us,
            stats.other_gpu_us,
            stats.host_us,
            out.text.chars().take(40).collect::<String>()
        );
        assert!(
            !out.token_ids.is_empty(),
            "overnight 9070 round {} must emit tokens",
            round + 1
        );
        assert_eq!(
            stats.draft_gpu_forwards, 0,
            "9070-only soak must not run a Draft Model; stats={stats:?}"
        );
        assert!(
            target_delta > 0,
            "overnight 9070 round {} must forward the Target Model; stats={stats:?}",
            round + 1
        );
    }
}

#[cfg(feature = "hip")]
fn hip_7b_config(path: std::path::PathBuf) -> Config {
    let mut cfg = Config::compiled_defaults();
    cfg.model.path = Some(path);
    cfg.model.synthetic = false;
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "auto".into();
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg.kv.vram_bytes = 256 * 1024 * 1024;
    cfg.kv.dram_bytes = 256 * 1024 * 1024;
    cfg.kv.page_tokens = 128;
    cfg
}

#[cfg(feature = "hip")]
#[test]
fn hip_loads_real_7b_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_7b_gguf() else {
        eprintln!("skip hip_loads_real_7b_gguf_when_path_set: set ALLPACA_7B_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_loads_real_7b_gguf_when_path_set: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path)).expect("7B GGUF load must not be Fatal");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_loads_real_7b_gguf_when_path_set: no HIP device");
        return;
    }
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.weight_bytes_on_device > 0,
        "7B HIP load must upload weights; stats={stats:?}"
    );
    assert_eq!(
        stats.weight_h2d_bytes, stats.weight_bytes_on_device,
        "load-only H2D must match resident weights; stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_real_7b_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_7b_gguf() else {
        eprintln!("skip hip_generate_real_7b_gguf_when_path_set: set ALLPACA_7B_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_real_7b_gguf_when_path_set: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path)).expect("7B GGUF load must not be Fatal");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_generate_real_7b_gguf_when_path_set: no HIP device");
        return;
    }
    let after_load = engine.stats();
    let h2d = after_load.weight_h2d_bytes;
    const N: usize = 8;
    let started = std::time::Instant::now();
    let out = engine
        .generate("hello", greedy(N))
        .expect("7B HIP generate must not panic or OOM");
    let elapsed = started.elapsed();
    assert!(
        out.token_ids.len() >= N || out.finish_reason != allpaca_types::FinishReason::Length,
        "expected at least {N} tokens or an early stop, got {} ids finish={:?}",
        out.token_ids.len(),
        out.finish_reason
    );
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.weight_h2d_bytes, h2d,
        "decode must not H2D weights; load={h2d} after={} stats={stats:?}",
        stats.weight_h2d_bytes
    );
    let toks = out.token_ids.len().max(1) as f64;
    let tps = toks / elapsed.as_secs_f64().max(1e-6);
    let profile_sum = stats.q8_gemm_us
        + stats.attention_us
        + stats.lm_head_us
        + stats.other_gpu_us
        + stats.host_us;
    assert!(
        stats.q8_gemm_us > 0.0,
        "Q8_0 7B/8B decode must time Q8 GEMV; q8_gemm_us={}",
        stats.q8_gemm_us
    );
    assert!(
        stats.attention_us > 0.0 && stats.lm_head_us > 0.0,
        "7B/8B decode must time attention and lm_head; stats={stats:?}"
    );
    assert!(
        stats.other_gpu_us > 0.0 && stats.other_gpu_us < PROFILE_BLAS_OTHER_GPU_US * 0.85,
        "other_gpu_us {:.1} must shrink vs profile-blas {:.1}; stats={stats:?}",
        stats.other_gpu_us, PROFILE_BLAS_OTHER_GPU_US
    );
    let itl_us = stats.itl_ms_ema * 1_000.0;
    assert!(
        itl_us > 10_000.0,
        "8B decode ITL should be tens of ms; itl_us={itl_us:.1} stats={stats:?}"
    );
    assert!(
        profile_sum > itl_us * 0.8 && profile_sum < itl_us * 1.25,
        "decode profile sum {profile_sum:.1}us must explain ITL {itl_us:.1}us; stats={stats:?}"
    );
    assert!(
        stats.host_us < INTDOT_HOST_US * 0.65,
        "host_us {:.1} must drop vs int-dot ~32ms; stats={stats:?}",
        stats.host_us
    );
    eprintln!(
        "7B HIP generate: tokens={} backend={} devices={} ttft_ms={} itl_ms={:.2} tokens/s={:.2} elapsed_ms={} text={:?}",
        out.token_ids.len(),
        stats.backend,
        hip_gpus(&engine).len(),
        stats.ttft_ms_last,
        stats.itl_ms_ema,
        tps,
        elapsed.as_millis(),
        out.text
    );
    eprintln!(
        "7B HIP decode profile us: q8_gemm={:.1} attention={:.1} lm_head={:.1} other_gpu={:.1} host={:.1} sum={:.1} itl_us={:.1}",
        stats.q8_gemm_us,
        stats.attention_us,
        stats.lm_head_us,
        stats.other_gpu_us,
        stats.host_us,
        profile_sum,
        stats.itl_ms_ema * 1000.0
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_long_prompt_real_7b_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_7b_gguf() else {
        eprintln!("skip hip_generate_long_prompt_real_7b_gguf_when_path_set: set ALLPACA_7B_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_long_prompt_real_7b_gguf_when_path_set: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path)).expect("7B GGUF load must not be Fatal");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_generate_long_prompt_real_7b_gguf_when_path_set: no HIP device");
        return;
    }
    let prompt = "hello world. ".repeat(80);
    const N: usize = 4;
    let started = std::time::Instant::now();
    let out = engine
        .generate(&prompt, greedy(N))
        .expect("7B HIP long-prompt generate must not panic or OOM");
    let elapsed = started.elapsed();
    assert!(!out.token_ids.is_empty());
    assert!(
        out.prompt_tokens >= 64,
        "long prompt must encode at least 64 tokens, got {}",
        out.prompt_tokens
    );
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert!(
        stats.ttft_ms_last > 0,
        "TTFT must be recorded; stats={stats:?}"
    );
    assert!(
        stats.gpu_forwards < out.prompt_tokens as u64,
        "batched prefill must not run one forward per prompt token; gpu_forwards={} prompt_tokens={} stats={stats:?}",
        stats.gpu_forwards,
        out.prompt_tokens
    );
    let toks = out.token_ids.len().max(1) as f64;
    let tps = toks / elapsed.as_secs_f64().max(1e-6);
    eprintln!(
        "7B HIP long prompt: prompt_tokens={} gen={} backend={} ttft_ms={} itl_ms={:.2} tokens/s={:.2} elapsed_ms={} gpu_forwards={} text={:?}",
        out.prompt_tokens,
        out.token_ids.len(),
        stats.backend,
        stats.ttft_ms_last,
        stats.itl_ms_ema,
        tps,
        elapsed.as_millis(),
        stats.gpu_forwards,
        out.text
    );
    eprintln!(
        "7B HIP long-prompt profile us: q8_gemm={:.1} attention={:.1} lm_head={:.1} other_gpu={:.1} host={:.1}",
        stats.q8_gemm_us,
        stats.attention_us,
        stats.lm_head_us,
        stats.other_gpu_us,
        stats.host_us
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_speculative_7b_with_draft_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(target) = gated_7b_gguf() else {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: set ALLPACA_7B_GGUF");
        return;
    };
    let Some(draft) = gated_draft_gguf() else {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: set ALLPACA_DRAFT_GGUF or place a small GGUF in the discovered HF cache dirs");
        return;
    };
    eprintln!("speculative 7B target path={}", target.display());
    eprintln!("speculative 7B draft path={}", draft.display());
    if allpaca_hip::device_count() < 2 {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: need 2 HIP devices");
        return;
    }
    let mut cfg = hip_7b_config(target);
    cfg.model.draft_path = Some(draft);
    cfg.speculative.enabled = true;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    cfg.features.tree_speculation = Some(false);
    let t0 = std::time::Instant::now();
    eprintln!("speculative 7B Engine::new starting");
    let engine = Engine::new(cfg).expect("7B+draft GGUF load must not be Fatal");
    eprintln!(
        "speculative 7B Engine::new done in {:.1}s weight_bytes={}",
        t0.elapsed().as_secs_f64(),
        engine.stats().weight_bytes_on_device
    );
    if hip_gpus(&engine).len() < 2 {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: need 2 HIP devices");
        return;
    }
    eprintln!("speculative 7B generate starting");
    let t1 = std::time::Instant::now();
    let out = engine
        .generate("hello", greedy(8))
        .expect("speculative 7B generate must run");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.target_gpu_forwards > 0 && stats.draft_gpu_forwards > 0,
        "both forward counters must increment; stats={stats:?}"
    );
    eprintln!(
        "speculative 7B: gen={} acceptance={:.3} target_fwd={} draft_fwd={} itl_ms={:.2} generate_s={:.1} text={:?} transfer_path={} p2p_status={} p2p_gbps={:.3} host_staging_bytes={} weight_bytes_migrated={} target_device={} draft_device={}",
        out.token_ids.len(),
        stats.acceptance_rate,
        stats.target_gpu_forwards,
        stats.draft_gpu_forwards,
        stats.itl_ms_ema,
        t1.elapsed().as_secs_f64(),
        out.text,
        stats.transfer_path,
        stats.p2p_status,
        stats.p2p_gbps,
        stats.host_staging_bytes,
        stats.weight_bytes_migrated,
        stats.target_device,
        stats.draft_device
    );
}

#[cfg(feature = "hip")]
fn generate_or_timeout(
    engine: &Engine,
    prompt: &str,
    n: usize,
    timeout: std::time::Duration,
) -> allpaca_types::GenerationOutput {
    let (tx, rx) = std::sync::mpsc::channel();
    let engine = engine.clone();
    let prompt = prompt.to_string();
    std::thread::spawn(move || {
        let _ = tx.send(engine.generate(&prompt, greedy(n)));
    });
    rx.recv_timeout(timeout)
        .unwrap_or_else(|_| panic!("generate({n}) hung after {timeout:?}"))
        .expect("generate must run")
}

/// 64-token speculative decode must finish on a clean boot (the overnight soak
/// hung here before draft KV prefix reuse).
#[cfg(feature = "hip")]
#[test]
fn hip_speculative_7b_64_tokens_finishes_before_timeout() {
    let _guard = large_gguf_lock();
    let Some(target) = gated_7b_gguf() else {
        eprintln!("skip hip_speculative_7b_64_tokens_finishes_before_timeout: set ALLPACA_7B_GGUF");
        return;
    };
    let Some(draft) = gated_draft_gguf() else {
        eprintln!("skip hip_speculative_7b_64_tokens_finishes_before_timeout: no draft GGUF");
        return;
    };
    if allpaca_hip::device_count() < 2 {
        eprintln!("skip hip_speculative_7b_64_tokens_finishes_before_timeout: need 2 HIP devices");
        return;
    }
    let mut cfg = hip_7b_config(target);
    cfg.model.draft_path = Some(draft);
    cfg.speculative.enabled = true;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    cfg.features.tree_speculation = Some(false);
    cfg.kv.vram_bytes = 512 * 1024 * 1024;
    cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("speculative 7B-64 Engine::new starting");
    let engine = Engine::new(cfg).expect("7B+draft GGUF load must not be Fatal");
    if hip_gpus(&engine).len() < 2 {
        eprintln!("skip hip_speculative_7b_64_tokens_finishes_before_timeout: need 2 HIP devices");
        return;
    }
    let t0 = std::time::Instant::now();
    let out = generate_or_timeout(
        &engine,
        "hello",
        64,
        std::time::Duration::from_secs(180),
    );
    assert!(!out.token_ids.is_empty());
    let n = out.token_ids.len() as u64;
    let stats = engine.stats();
    let secs = t0.elapsed().as_secs_f64();
    let tps = n as f64 / secs.max(1e-6);
    eprintln!(
        "speculative 7B-64: gen={n} tps={:.2} itl_ms={:.2} acceptance={:.3} target_fwd={} draft_fwd={} generate_s={:.1} text_prefix={:?}",
        tps,
        stats.itl_ms_ema,
        stats.acceptance_rate,
        stats.target_gpu_forwards,
        stats.draft_gpu_forwards,
        secs,
        out.text.chars().take(48).collect::<String>()
    );
    assert!(
        stats.draft_gpu_forwards > 0 && stats.draft_gpu_forwards < n.saturating_mul(16).max(64),
        "64-token spec must reuse draft KV; gen={n} stats={stats:?}"
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_real_q4k_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_q4k_gguf() else {
        eprintln!("skip hip_generate_real_q4k_gguf_when_path_set: set ALLPACA_Q4K_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_real_q4k_gguf_when_path_set: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path)).expect("Q4_K GGUF load must not be Fatal");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_generate_real_q4k_gguf_when_path_set: no HIP device");
        return;
    }
    let after_load = engine.stats();
    let h2d = after_load.weight_h2d_bytes;
    let out = engine
        .generate("hello", greedy(4))
        .expect("Q4_K HIP generate must not panic or OOM");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.weight_h2d_bytes, h2d,
        "decode must not H2D weights; load={h2d} after={}",
        stats.weight_h2d_bytes
    );
    eprintln!(
        "Q4_K HIP generate: tokens={} backend={} text={:?}",
        out.token_ids.len(),
        stats.backend,
        out.text
    );
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_real_mixed_kquant_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_mixed_kquant_gguf() else {
        eprintln!(
            "skip hip_generate_real_mixed_kquant_gguf_when_path_set: set ALLPACA_Q4KM_GGUF to a reasonably sized Q4_K_M (skip 27B/31B)"
        );
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_real_mixed_kquant_gguf_when_path_set: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path.clone())).expect("mixed K-quant GGUF load must not be Fatal");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_generate_real_mixed_kquant_gguf_when_path_set: no HIP device");
        return;
    }
    let out = engine
        .generate("hello", greedy(4))
        .expect("mixed Q4_K+Q6_K HIP generate must not panic or OOM");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    eprintln!(
        "mixed K-quant HIP generate: path={} tokens={} backend={} text={:?}",
        path.display(),
        out.token_ids.len(),
        stats.backend,
        out.text
    );
}
