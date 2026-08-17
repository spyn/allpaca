//! Public Engine / DeviceInventory / Config behaviors for the HIP path.
//! GPU-gated tests skip when `hipGetDeviceCount() == 0`.

use allpaca_core::Engine;
#[cfg(feature = "hip")]
use allpaca_core::ModelConfig;
use allpaca_types::{Config, GenerationParams};
#[cfg(feature = "hip")]
use allpaca_types::DeviceKind;

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
        if path.is_file() {
            return Some(path);
        }
    }
    const DISCOVERED: &[&str] = &[
        r"E:\Models\huggingface\models--bartowski--Llama-3.2-1B-Instruct-GGUF",
        r"E:\Models\huggingface\models--bartowski--Llama-3.2-3B-Instruct-GGUF",
    ];
    for root in DISCOVERED {
        let root = std::path::Path::new(root);
        if !root.exists() {
            continue;
        }
        if let Ok(walk) = std::fs::read_dir(root) {
            for entry in walk.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("gguf") && p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    None
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
fn hip_speculative_7b_with_draft_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(target) = gated_7b_gguf() else {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: set ALLPACA_7B_GGUF");
        return;
    };
    let Some(draft) = gated_draft_gguf() else {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: set ALLPACA_DRAFT_GGUF");
        return;
    };
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
    let engine = Engine::new(cfg).expect("7B+draft GGUF load must not be Fatal");
    if hip_gpus(&engine).len() < 2 {
        eprintln!("skip hip_speculative_7b_with_draft_gguf_when_path_set: need 2 HIP devices");
        return;
    }
    let out = engine
        .generate("hello", greedy(8))
        .expect("speculative 7B generate must run");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.target_gpu_forwards > 0 && stats.draft_gpu_forwards > 0,
        "both forward counters must increment; stats={stats:?}"
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
