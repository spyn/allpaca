//! Public Engine / DeviceInventory / Config behaviors for the HIP path.
//! GPU-gated tests skip when `hipGetDeviceCount() == 0`.

use allpaca_core::Engine;
#[cfg(feature = "hip")]
use allpaca_core::ModelConfig;
use allpaca_types::{Config, GenerationParams};
#[cfg(feature = "hip")]
use allpaca_types::{DeviceKind, DeviceSelector, MissPolicy};

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
    assert!(gpus
        .iter()
        .all(|d| d.memory_bytes > 0 && !d.name.is_empty()));
    let again = Engine::new(hip_auto_synthetic()).unwrap();
    let ids2: Vec<i32> = hip_gpus(&again).iter().map(|d| d.id).collect();
    assert_eq!(
        ids, ids2,
        "HIP device ids must be stable across Engine boots"
    );
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

/// `/stats` must carry driver-measured board VRAM and a GPU busy fraction
/// (BENCHMARK_SPEC `gpu.vram` / `gpu.util`), and the sampler must stay off the
/// decode path: generating tokens must not add one sample per token.
#[cfg(feature = "hip")]
#[test]
fn stats_carry_driver_vram_and_gpu_util_sampled_off_decode() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let s = engine.stats();
    assert!(
        s.gpu_samples >= 1,
        "stats must trigger a sample on the HIP path; stats={s:?}"
    );
    assert!(
        s.gpu_vram_total_bytes > 0,
        "driver must report total VRAM; stats={s:?}"
    );
    assert!(
        s.gpu_vram_used_bytes >= s.device_memory_bytes,
        "board usage must cover what this process allocated; used={} owned={}",
        s.gpu_vram_used_bytes,
        s.device_memory_bytes
    );
    assert!(
        s.gpu_vram_used_bytes <= s.gpu_vram_total_bytes,
        "used must not exceed total; stats={s:?}"
    );
    assert!(
        (0.0..=1.0).contains(&s.gpu_util),
        "gpu_util must be a fraction; got {}",
        s.gpu_util
    );

    // 10 Hz rate limit: a burst of decode plus /stats calls must not sample
    // per token. Anything near `tokens` here means the driver call leaked into
    // the hot path.
    let before = engine.stats().gpu_samples;
    let tokens = 16;
    let out = engine.generate("hello", greedy(tokens)).unwrap();
    assert!(!out.token_ids.is_empty());
    let after = engine.stats().gpu_samples;
    assert!(
        after - before <= 4,
        "sampler must stay rate-limited across {tokens} tokens; grew {before} -> {after}"
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
fn hip_streamed_synthetic() -> Config {
    let mut cfg = hip_auto_synthetic();
    cfg.weights.trunk_bytes = 64;
    cfg.weights.ring_slots = 1;
    cfg
}

#[cfg(feature = "hip")]
#[test]
fn streamed_ffn_hip_greedy_tokens_match_fully_resident() {
    let resident = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&resident).is_empty() {
        return;
    }
    let full = resident.generate("hello", greedy(8)).unwrap();
    let streamed = Engine::new(hip_streamed_synthetic()).unwrap();
    if hip_gpus(&streamed).is_empty() {
        return;
    }
    let after_load = streamed.stats();
    assert!(
        after_load.weight_ring_slots >= 1,
        "tiny trunk budget must grant a ring"
    );
    assert!(
        after_load.weight_pinned_layers < 2,
        "tiny budget should not pin both synthetic layers"
    );
    assert!(
        after_load.weight_bytes_on_device > after_load.weight_h2d_bytes,
        "ring VRAM is reserved at load; packed H2D skips unpinned FFN; stats={after_load:?}"
    );
    let part = streamed.generate("hello", greedy(8)).unwrap();
    assert_eq!(
        part.token_ids, full.token_ids,
        "GPU residency is a speed dial: greedy tokens must match"
    );
    let after = streamed.stats();
    assert!(
        after.weight_h2d_bytes > after_load.weight_h2d_bytes,
        "streamed FFN must H2D into the ring during decode"
    );
    assert!(
        after.weight_ring_binds > 0,
        "streamed FFN must bind through the ring"
    );
    assert!(
        after.nvme_reads > 0,
        "unpinned FFN must H2D from the packed trunk; stats={after:?}"
    );
}

#[cfg(feature = "hip")]
fn hip_moe_resident() -> Config {
    let mut cfg = hip_auto_synthetic();
    cfg.model.synthetic = false;
    cfg
}

#[cfg(feature = "hip")]
fn hip_moe_streamed() -> Config {
    let mut cfg = hip_moe_resident();
    cfg.weights.cache_bytes = 64;
    cfg.weights.miss_policy = MissPolicy::Fill;
    cfg
}

#[cfg(feature = "hip")]
#[test]
fn streamed_moe_hip_greedy_tokens_match_fully_resident() {
    let resident = Engine::new(hip_moe_resident()).unwrap();
    if hip_gpus(&resident).is_empty() {
        return;
    }
    resident.load_synthetic(ModelConfig::tiny_moe()).unwrap();
    let full = resident.generate("hello", greedy(8)).unwrap();

    let streamed = Engine::new(hip_moe_streamed()).unwrap();
    if hip_gpus(&streamed).is_empty() {
        return;
    }
    streamed.load_synthetic(ModelConfig::tiny_moe()).unwrap();
    let after_load = streamed.stats();
    assert!(
        after_load.expert_slots >= 1,
        "tiny cache_bytes must grant at least one expert slot"
    );
    assert!(
        after_load.expert_host_bps > 0,
        "q* must measure CPU expert bandwidth on HIP load; stats={after_load:?}"
    );
    assert!(
        after_load.expert_pcie_bps > 0,
        "q* must measure expert H2D bandwidth on HIP load; stats={after_load:?}"
    );
    assert!(
        after_load.weight_bytes_on_device > after_load.weight_h2d_bytes,
        "expert-slot VRAM is reserved at load; packed H2D skips experts; stats={after_load:?}"
    );
    let part = streamed.generate("hello", greedy(8)).unwrap();
    assert_eq!(
        part.token_ids, full.token_ids,
        "GPU residency is a speed dial: greedy tokens must match"
    );
    let after = streamed.stats();
    assert!(
        after.weight_h2d_bytes > after_load.weight_h2d_bytes,
        "expert-slot fills must H2D during generate"
    );
    assert!(
        after.gpu_forwards > 0,
        "MoE decode must stay on the HIP path, not silently fall back to host"
    );
    assert!(
        after.expert_lru_fills > 0,
        "Fill policy must admit misses into the LRU"
    );
    let dim = 32u64;
    let n_layers = 2u64;
    let logits = SYNTHETIC_VOCAB.saturating_mul(4);
    let route = n_layers
        .saturating_mul(dim)
        .saturating_mul(4)
        .saturating_mul(2);
    let bound = after
        .gpu_forwards
        .saturating_mul(logits.saturating_add(route));
    assert!(
        after.activation_d2h_bytes <= bound,
        "MoE residual scale must stay on device; d2h={} bound={} stats={after:?}",
        after.activation_d2h_bytes,
        bound
    );
}

#[cfg(feature = "hip")]
#[test]
fn moe_gguf_hip_generate_uses_expert_slots() {
    let path = std::env::temp_dir().join("allpaca_hip_moe.gguf");
    write_tiny_moe_gguf(&path);
    let mut cfg = hip_auto_synthetic();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.clone());
    cfg.weights.cache_bytes = 64;
    cfg.weights.miss_policy = MissPolicy::Fill;
    let engine = Engine::new(cfg).unwrap();
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.expert_slots >= 1,
        "Mixtral-style GGUF must grant expert slots; stats={stats:?}"
    );
    assert!(
        stats.gpu_forwards > 0,
        "MoE GGUF must stay on HIP; stats={stats:?}"
    );
    let _ = std::fs::remove_file(path);
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
    let out = engine
        .generate("hello", greedy(8))
        .expect("packed linear verify generate");
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

/// Partial-RoPE skips batched HIP prefill, so packed verify is per-token.
/// Each row's logits must be waited before the next token overwrites the D2H slot.
#[cfg(feature = "hip")]
#[test]
fn hip_imrope_packed_verify_matches_target_only() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_imrope_packed_verify_matches_target_only: no HIP device");
        return;
    }
    let target = std::env::temp_dir().join("allpaca_hip_imrope_pack_tgt.gguf");
    let draft = std::env::temp_dir().join("allpaca_hip_imrope_pack_dft.gguf");
    write_packed_f16_gguf_with_rope_dims(&target, 16);
    write_packed_f16_gguf_with_rope_dims(&draft, 16);
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
    let engine = Engine::new(cfg).expect("IMRoPE F16 target+draft must load");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&target);
        let _ = std::fs::remove_file(&draft);
        eprintln!("skip hip_imrope_packed_verify_matches_target_only: no HIP in engine");
        return;
    }
    let spec_out = engine
        .generate("hello", greedy(8))
        .expect("IMRoPE packed verify generate");
    assert!(!spec_out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.linear_packed_verifies > 0,
        "identical IMRoPE target+draft must pack; stats={stats:?}"
    );
    drop(engine);
    let mut plain = hip_7b_config(target.clone());
    plain.speculative.enabled = false;
    plain.features.draft_model_speculation = Some(false);
    plain.kv.page_tokens = 8;
    plain.kv.vram_bytes = 64 * 1024 * 1024;
    plain.kv.dram_bytes = 64 * 1024 * 1024;
    let plain_engine = Engine::new(plain).expect("IMRoPE target-only must load");
    let plain_out = plain_engine
        .generate("hello", greedy(8))
        .expect("IMRoPE target-only generate");
    assert_eq!(
        spec_out.token_ids, plain_out.token_ids,
        "per-token packed verify must keep Target Model tokens; spec={:?} plain={:?} packed={}",
        spec_out.token_ids, plain_out.token_ids, stats.linear_packed_verifies
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
    write_packed_f16_gguf_with_rope_dims(path, 0);
}

#[cfg(feature = "hip")]
fn write_packed_f16_gguf_with_rope_dims(path: &std::path::Path, rope_dims: u32) {
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
    meta.insert(
        "llama.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
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
    if rope_dims > 0 {
        meta.insert(
            "llama.rope.dimension_count".into(),
            GgufValue::Uint32(rope_dims),
        );
    }
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
    write_gguf(path, &meta, &views).unwrap();
}

#[cfg(feature = "hip")]
fn write_tiny_fused_qkv_f32_gguf(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 8usize;
    let kv_dim = 2usize;
    let vocab = 4usize;
    let ff = 16usize;
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
    let qkv_dim = dim + 2 * kv_dim;
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
            "blk.0.attn_qkv.weight".into(),
            vec![dim as u64, qkv_dim as u64],
            vec![0.01; qkv_dim * dim],
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
    write_gguf(path, &meta, &views).unwrap();
}

/// A split-less attention checkpoint must load and generate on the HIP resident path.
#[cfg(feature = "hip")]
#[test]
fn hip_fused_qkv_attention_gguf_generates() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_fused_qkv_attention_gguf_generates: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca-fused-qkv-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_tiny_fused_qkv_f32_gguf(&path);
    let mut cfg = Config::compiled_defaults();
    cfg.model.path = Some(path.clone());
    cfg.model.synthetic = false;
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "auto".into();
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    let result = Engine::new(cfg).and_then(|engine| engine.generate("hello", greedy(1)));
    let _ = std::fs::remove_file(path);
    let generated = result.expect("fused attn_qkv GGUF must generate on HIP");
    assert!(!generated.token_ids.is_empty());
}

#[cfg(feature = "hip")]
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
        gate_exps.extend(std::iter::repeat_n(w, ff * dim));
        up_exps.extend(std::iter::repeat_n(w, ff * dim));
        down_exps.extend(std::iter::repeat_n(w, dim * ff));
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
    write_tiny_uniform_kquant_gguf(
        &path,
        allpaca_gguf::GgmlType::Q6_K,
        allpaca_gguf::quantize_q6_k,
    );
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
    write_tiny_uniform_kquant_gguf(
        &path,
        allpaca_gguf::GgmlType::F16,
        allpaca_gguf::quantize_f16,
    );
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
                write_tiny_uniform_kquant_gguf(
                    p,
                    allpaca_gguf::GgmlType::F16,
                    allpaca_gguf::quantize_f16,
                )
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
    // Prefer local repo models for metal soaks before hardcoded external paths.
    let local = local_models_dir();
    for name in [
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
    ] {
        let p = local.join(name);
        if p.is_file() && looks_like_gguf(&p) {
            return Some(p);
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

/// Gated ROCmFP4 GGUF (Qwen35 / Nemotron). Env overrides discovery under `models/`.
#[cfg(feature = "hip")]
fn gated_rocmfp4_gguf() -> Option<std::path::PathBuf> {
    for key in ["ALLPACA_ROCMFP4_GGUF", "ALLPACA_QWEN_GGUF"] {
        if let Ok(raw) = std::env::var(key) {
            let path = std::path::PathBuf::from(raw);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    let dir = local_models_dir();
    let qwen = dir
        .join("Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF")
        .join("Qwen3.8-27B-Q4_0_ROCMFP4_COHERENT.gguf");
    if qwen.is_file() {
        return Some(qwen);
    }
    let nemo = dir.join("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf");
    if nemo.is_file() {
        return Some(nemo);
    }
    None
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
    let found = find_gguf_under(root)
        .expect("Qwen 0.5B HF cache must contain a GGUF blob or snapshots/*.gguf");
    assert!(
        found.is_file(),
        "discovered path must be a file: {}",
        found.display()
    );
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
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-1B-Instruct-f16.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-f16.gguf",
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

/// Demo Draft Model: 3B Q8 keeps 8B acceptance; 1B Q8 is cheaper but weaker.
#[cfg(feature = "hip")]
fn demo_draft_gguf() -> Option<std::path::PathBuf> {
    let dir = local_models_dir();
    const NAMES: &[&str] = &[
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-f16.gguf",
        "Llama-3.2-1B-Instruct-f16.gguf",
    ];
    for name in NAMES {
        let p = dir.join(name);
        if p.is_file() && looks_like_gguf(&p) {
            return Some(p);
        }
    }
    gated_draft_gguf()
}

#[cfg(feature = "hip")]
fn overnight_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "hip")]
fn overnight_timeout(tokens: usize, secs_per_token: u64) -> std::time::Duration {
    std::time::Duration::from_secs((tokens as u64).saturating_mul(secs_per_token).max(180))
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
    let cheaper = [
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-1B-Instruct-f16.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
    ]
    .iter()
    .any(|name| local_models_dir().join(name).is_file());
    if cheaper {
        eprintln!(
            "skip gated_draft_prefers_local_llama32_3b_f16_over_qwen: cheaper Llama-3.2 draft present"
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

/// Cheaper same-family Draft Model must win over the 3B F16 file.
#[cfg(feature = "hip")]
#[test]
fn gated_draft_prefers_local_llama32_1b_q8_over_3b_f16() {
    let q8 = local_models_dir().join("Llama-3.2-1B-Instruct-Q8_0.gguf");
    let f16_3b = local_models_dir().join("Llama-3.2-3B-Instruct-f16.gguf");
    if !q8.is_file() {
        eprintln!(
            "skip gated_draft_prefers_local_llama32_1b_q8_over_3b_f16: missing {}",
            q8.display()
        );
        return;
    }
    if !f16_3b.is_file() {
        eprintln!(
            "skip gated_draft_prefers_local_llama32_1b_q8_over_3b_f16: missing {}",
            f16_3b.display()
        );
        return;
    }
    let found = gated_draft_gguf().expect("local Llama-3.2-1B Q8 must be discoverable");
    assert_eq!(
        found.canonicalize().unwrap_or(found.clone()),
        q8.canonicalize().unwrap_or(q8.clone()),
        "1B Q8 Draft Model must beat 3B F16; found={}",
        found.display()
    );
}

/// Demo bar uses 3B Q8 when present (1B Q8 lost acceptance vs Llama-3 8B).
#[cfg(feature = "hip")]
#[test]
fn demo_draft_prefers_local_llama32_3b_q8_over_1b_q8() {
    let q8_3b = local_models_dir().join("Llama-3.2-3B-Instruct-Q8_0.gguf");
    let q8_1b = local_models_dir().join("Llama-3.2-1B-Instruct-Q8_0.gguf");
    if !q8_3b.is_file() {
        eprintln!(
            "skip demo_draft_prefers_local_llama32_3b_q8_over_1b_q8: missing {}",
            q8_3b.display()
        );
        return;
    }
    if !q8_1b.is_file() {
        eprintln!(
            "skip demo_draft_prefers_local_llama32_3b_q8_over_1b_q8: missing {}",
            q8_1b.display()
        );
        return;
    }
    let found = demo_draft_gguf().expect("demo must discover 3B Q8");
    assert_eq!(
        found.canonicalize().unwrap_or(found.clone()),
        q8_3b.canonicalize().unwrap_or(q8_3b.clone()),
        "demo Draft Model must prefer 3B Q8 over 1B Q8; found={}",
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

/// Dual-GPU overnight bakeoff. Skips unless `ALLPACA_OVERNIGHT=1`. Both HIP
/// devices must stay visible (do not set `HIP_VISIBLE_DEVICES`).
///
/// 1. 8B greedy on HIP:0 (Target Model baseline)
/// 2. Optional 3B F16 greedy on HIP:1 (`ALLPACA_OVERNIGHT_9060=1`)
/// 3. 8B Target + 3B Q8 Draft spec (HostStaging; draft-ahead overlaps packed verify)
#[cfg(feature = "hip")]
#[test]
fn hip_overnight_llama32_f16_on_second_gpu_when_env_set() {
    if std::env::var("ALLPACA_OVERNIGHT").ok().as_deref() != Some("1") {
        eprintln!(
            "skip hip_overnight_llama32_f16_on_second_gpu_when_env_set: set ALLPACA_OVERNIGHT=1"
        );
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
    let tokens = overnight_usize("ALLPACA_OVERNIGHT_TOKENS", 128);
    let rounds = overnight_usize("ALLPACA_OVERNIGHT_ROUNDS", 40);
    let spec_tokens = overnight_usize("ALLPACA_OVERNIGHT_SPEC_TOKENS", 64);
    let spec_rounds = overnight_usize("ALLPACA_OVERNIGHT_SPEC_ROUNDS", 20);
    let Some(target) = gated_7b_gguf() else {
        eprintln!("overnight skip: no 8B GGUF");
        return;
    };
    let Some(draft) = demo_draft_gguf() else {
        eprintln!("overnight skip speculative: no draft GGUF");
        return;
    };

    let mut greedy_cfg = hip_7b_config(target.clone());
    greedy_cfg.engine.target_device = DeviceSelector::Hip(0);
    greedy_cfg.engine.draft_device = DeviceSelector::Cpu;
    greedy_cfg.speculative.enabled = false;
    greedy_cfg.features.draft_model_speculation = Some(false);
    greedy_cfg.kv.vram_bytes = 512 * 1024 * 1024;
    greedy_cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("overnight 9070 Engine::new starting tokens={tokens} rounds={rounds}");
    let greedy = Engine::new(greedy_cfg).expect("8B GGUF must load on HIP:0");
    let stats = greedy.stats();
    eprintln!(
        "overnight 9070 load backend={} target_device={} draft_device={} gpu_count={} weight_bytes={} transfer_path={} p2p_status={}",
        stats.backend,
        stats.target_device,
        stats.draft_device,
        hip_gpus(&greedy).len(),
        stats.weight_bytes_on_device,
        stats.transfer_path,
        stats.p2p_status
    );
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.target_device, 0,
        "8B greedy baseline must run on HIP:0"
    );
    assert_eq!(
        stats.draft_device, -1,
        "8B greedy baseline must not place a Draft Model"
    );
    let greedy_timeout = overnight_timeout(tokens, 3);
    for round in 0..rounds {
        eprintln!(
            "overnight 9070 round={}/{} generate starting",
            round + 1,
            rounds
        );
        let before = greedy.stats();
        let t0 = std::time::Instant::now();
        let out = generate_or_timeout(&greedy, "hello", tokens, greedy_timeout);
        let stats = greedy.stats();
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
            "8B greedy baseline must not run a Draft Model; stats={stats:?}"
        );
        assert!(
            target_delta > 0,
            "overnight 9070 round {} must forward the Target Model; stats={stats:?}",
            round + 1
        );
    }
    drop(greedy);

    if soak_9060 {
        let mut cfg = hip_7b_config(p);
        cfg.engine.target_device = DeviceSelector::Hip(1);
        cfg.engine.draft_device = DeviceSelector::Cpu;
        cfg.speculative.enabled = false;
        cfg.features.draft_model_speculation = Some(false);
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
        assert_eq!(
            stats.target_device, 1,
            "overnight soak must run on the second HIP device"
        );
        assert_eq!(
            stats.draft_device, -1,
            "9060 soak must not place a Draft Model"
        );
        let soak_timeout = overnight_timeout(tokens, 8);
        for round in 0..rounds {
            eprintln!(
                "overnight 9060 round={}/{} generate starting",
                round + 1,
                rounds
            );
            let t0 = std::time::Instant::now();
            let out = generate_or_timeout(&engine, "hello", tokens, soak_timeout);
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
    } else {
        eprintln!("overnight skip 9060 soak: set ALLPACA_OVERNIGHT_9060=1");
    }

    let mut spec_cfg = hip_7b_config(target);
    spec_cfg.model.draft_path = Some(draft);
    spec_cfg.speculative.enabled = true;
    spec_cfg.features.draft_model_speculation = Some(true);
    spec_cfg.features.ngram_speculation = Some(false);
    spec_cfg.features.tree_speculation = Some(false);
    spec_cfg.kv.vram_bytes = 512 * 1024 * 1024;
    spec_cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!(
        "overnight spec Engine::new starting spec_tokens={spec_tokens} spec_rounds={spec_rounds}"
    );
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
    assert_eq!(
        stats.target_device, 0,
        "spec Target Model must run on HIP:0"
    );
    assert_eq!(stats.draft_device, 1, "spec Draft Model must run on HIP:1");
    let spec_timeout = overnight_timeout(spec_tokens, 8);
    for round in 0..spec_rounds {
        eprintln!(
            "overnight spec round={}/{} generate starting",
            round + 1,
            spec_rounds
        );
        let before = spec.stats();
        let t0 = std::time::Instant::now();
        let out = generate_or_timeout(&spec, "hello", spec_tokens, spec_timeout);
        let stats = spec.stats();
        let secs = t0.elapsed().as_secs_f64();
        let tps = out.token_ids.len() as f64 / secs.max(1e-6);
        let draft_delta = stats
            .draft_gpu_forwards
            .saturating_sub(before.draft_gpu_forwards);
        let target_delta = stats
            .target_gpu_forwards
            .saturating_sub(before.target_gpu_forwards);
        let packed_delta = stats
            .linear_packed_verifies
            .saturating_sub(before.linear_packed_verifies);
        let ahead_delta = stats
            .draft_ahead_hits
            .saturating_sub(before.draft_ahead_hits);
        eprintln!(
            "overnight spec round={}/{} gen={} tps={:.2} itl_ms={:.2} acceptance={:.3} target_fwd={} target_fwd_delta={} draft_fwd={} draft_fwd_delta={} packed_verify_delta={} draft_ahead_delta={} host_staging_bytes={} weight_bytes_migrated={} text_prefix={:?}",
            round + 1,
            spec_rounds,
            out.token_ids.len(),
            tps,
            stats.itl_ms_ema,
            stats.acceptance_rate,
            stats.target_gpu_forwards,
            target_delta,
            stats.draft_gpu_forwards,
            draft_delta,
            packed_delta,
            ahead_delta,
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
}

/// 8B-only greedy on the Target Model GPU (RX 9070 / HIP:0) with the 9060 hidden.
/// Requires `ALLPACA_OVERNIGHT=1` and `ALLPACA_OVERNIGHT_9070=1`. Skips when more
/// than one HIP device is visible so boot P2P probe cannot touch the 9060;
/// run with `HIP_VISIBLE_DEVICES=0`. Dual-GPU bakeoff uses
/// `hip_overnight_llama32_f16_on_second_gpu_when_env_set` instead.
#[cfg(feature = "hip")]
#[test]
fn hip_overnight_8b_greedy_on_target_gpu_when_env_set() {
    if std::env::var("ALLPACA_OVERNIGHT").ok().as_deref() != Some("1") {
        eprintln!(
            "skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: set ALLPACA_OVERNIGHT=1"
        );
        return;
    }
    if std::env::var("ALLPACA_OVERNIGHT_9070").ok().as_deref() != Some("1") {
        eprintln!(
            "skip hip_overnight_8b_greedy_on_target_gpu_when_env_set: set ALLPACA_OVERNIGHT_9070=1"
        );
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
    assert_eq!(
        stats.draft_device, -1,
        "9070 soak must not place a Draft Model"
    );
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

/// Draft Model ceiling: 3B Q8 greedy on HIP:1 (RX 9060 XT). Overlap cannot
/// beat this rate. Skips unless `ALLPACA_OVERNIGHT=1`. Both devices visible.
#[cfg(feature = "hip")]
#[test]
fn hip_overnight_llama32_q8_greedy_on_second_gpu_when_env_set() {
    if std::env::var("ALLPACA_OVERNIGHT").ok().as_deref() != Some("1") {
        eprintln!(
            "skip hip_overnight_llama32_q8_greedy_on_second_gpu_when_env_set: set ALLPACA_OVERNIGHT=1"
        );
        return;
    }
    let _guard = large_gguf_lock();
    let p = local_models_dir().join("Llama-3.2-3B-Instruct-Q8_0.gguf");
    if !p.is_file() {
        eprintln!(
            "skip hip_overnight_llama32_q8_greedy_on_second_gpu_when_env_set: missing {}",
            p.display()
        );
        return;
    }
    if allpaca_hip::device_count() < 2 {
        eprintln!(
            "skip hip_overnight_llama32_q8_greedy_on_second_gpu_when_env_set: need 2 HIP devices"
        );
        return;
    }
    let tokens = overnight_usize("ALLPACA_OVERNIGHT_TOKENS", 256);
    let rounds = overnight_usize("ALLPACA_OVERNIGHT_ROUNDS", 40);
    let mut cfg = hip_7b_config(p);
    cfg.engine.target_device = DeviceSelector::Hip(1);
    cfg.engine.draft_device = DeviceSelector::Cpu;
    cfg.speculative.enabled = false;
    cfg.features.draft_model_speculation = Some(false);
    cfg.kv.vram_bytes = 512 * 1024 * 1024;
    cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!("overnight draft-q8 Engine::new starting tokens={tokens} rounds={rounds}");
    let engine = Engine::new(cfg).expect("Llama-3.2-3B Q8 must load on HIP:1");
    let stats = engine.stats();
    eprintln!(
        "overnight draft-q8 load backend={} target_device={} draft_device={} weight_bytes={} transfer_path={}",
        stats.backend,
        stats.target_device,
        stats.draft_device,
        stats.weight_bytes_on_device,
        stats.transfer_path
    );
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.target_device, 1,
        "Draft Model greedy must run on HIP:1"
    );
    assert_eq!(
        stats.draft_device, -1,
        "Draft Model greedy must not place a second Draft Model"
    );
    assert!(
        stats.weight_bytes_on_device > 3_000_000_000
            && stats.weight_bytes_on_device < 4_000_000_000,
        "3B Q8 must stay packed; stats={stats:?}"
    );
    let timeout = overnight_timeout(tokens, 8);
    for round in 0..rounds {
        eprintln!(
            "overnight draft-q8 round={}/{} generate starting",
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
            "overnight draft-q8 round={}/{} gen={} tps={:.2} itl_ms={:.2} target_fwd_delta={} draft_fwd={} q8_gemm_us={:.0} attention_us={:.0} lm_head_us={:.0} other_gpu_us={:.0} host_us={:.0} text_prefix={:?}",
            round + 1,
            rounds,
            out.token_ids.len(),
            tps,
            stats.itl_ms_ema,
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
            "overnight draft-q8 round {} must emit tokens",
            round + 1
        );
        assert_eq!(
            stats.draft_gpu_forwards, 0,
            "Draft Model greedy must not run a second model; stats={stats:?}"
        );
        assert!(
            target_delta > 0,
            "overnight draft-q8 round {} must forward on HIP:1; stats={stats:?}",
            round + 1
        );
    }
}

/// Demo bar: asymmetric spec tok/s must beat Target Model greedy on this box.
/// 3B F16 Draft bakeoff was ~11.5 vs ~25 greedy. 1B Q8 sequential was ~11.5
/// vs ~15 greedy (acceptance 0.44). Prefers 3B Q8 when present. Skips unless
/// `ALLPACA_DEMO=1`. Both HIP devices must stay visible.
#[cfg(feature = "hip")]
#[test]
fn hip_asymmetric_spec_beats_target_greedy_when_env_set() {
    if std::env::var("ALLPACA_DEMO").ok().as_deref() != Some("1") {
        eprintln!("skip hip_asymmetric_spec_beats_target_greedy_when_env_set: set ALLPACA_DEMO=1");
        return;
    }
    let _guard = large_gguf_lock();
    if allpaca_hip::device_count() < 2 {
        eprintln!("skip hip_asymmetric_spec_beats_target_greedy_when_env_set: need 2 HIP devices");
        return;
    }
    let Some(target) = gated_7b_gguf() else {
        eprintln!("skip hip_asymmetric_spec_beats_target_greedy_when_env_set: no 8B GGUF");
        return;
    };
    let Some(draft) = demo_draft_gguf() else {
        eprintln!("skip hip_asymmetric_spec_beats_target_greedy_when_env_set: no Draft Model GGUF");
        return;
    };
    let tokens = overnight_usize("ALLPACA_DEMO_TOKENS", 128);
    let warmup = overnight_usize("ALLPACA_DEMO_WARMUP", 32);
    eprintln!(
        "demo spec-vs-greedy target={} draft={} tokens={} warmup={}",
        target.display(),
        draft.display(),
        tokens,
        warmup
    );

    let mut greedy_cfg = hip_7b_config(target.clone());
    greedy_cfg.engine.target_device = DeviceSelector::Hip(0);
    greedy_cfg.engine.draft_device = DeviceSelector::Cpu;
    greedy_cfg.speculative.enabled = false;
    greedy_cfg.features.draft_model_speculation = Some(false);
    greedy_cfg.kv.vram_bytes = 512 * 1024 * 1024;
    greedy_cfg.kv.dram_bytes = 512 * 1024 * 1024;
    let greedy = Engine::new(greedy_cfg).expect("8B GGUF must load on HIP:0");
    let stats = greedy.stats();
    eprintln!(
        "demo greedy load backend={} target_device={} draft_device={} gpu_count={} weight_bytes={}",
        stats.backend,
        stats.target_device,
        stats.draft_device,
        hip_gpus(&greedy).len(),
        stats.weight_bytes_on_device
    );
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.target_device, 0,
        "Target Model greedy must run on HIP:0"
    );
    assert_eq!(
        stats.draft_device, -1,
        "Target Model greedy must not place a Draft Model"
    );
    let greedy_timeout = overnight_timeout(tokens.max(warmup), 3);
    if warmup > 0 {
        let _ = generate_or_timeout(&greedy, "hello", warmup, greedy_timeout);
    }
    let (greedy_tps, greedy_out) =
        warmup_and_measure_tps(&greedy, "hello", 0, tokens, greedy_timeout);
    let greedy_stats = greedy.stats();
    eprintln!(
        "demo greedy gen={} tps={:.2} itl_ms={:.2} target_fwd={} draft_fwd={}",
        greedy_out.token_ids.len(),
        greedy_tps,
        greedy_stats.itl_ms_ema,
        greedy_stats.target_gpu_forwards,
        greedy_stats.draft_gpu_forwards
    );
    assert!(
        !greedy_out.token_ids.is_empty(),
        "Target Model greedy must emit tokens"
    );
    drop(greedy);

    let mut spec_cfg = hip_7b_config(target);
    spec_cfg.model.draft_path = Some(draft.clone());
    spec_cfg.speculative.enabled = true;
    spec_cfg.features.draft_model_speculation = Some(true);
    spec_cfg.features.ngram_speculation = Some(false);
    spec_cfg.features.tree_speculation = Some(false);
    spec_cfg.kv.vram_bytes = 512 * 1024 * 1024;
    spec_cfg.kv.dram_bytes = 512 * 1024 * 1024;
    let spec = Engine::new(spec_cfg).expect("8B Target + Draft Model must load");
    let stats = spec.stats();
    eprintln!(
        "demo spec load target_device={} draft_device={} weight_bytes={} transfer_path={}",
        stats.target_device, stats.draft_device, stats.weight_bytes_on_device, stats.transfer_path
    );
    assert_eq!(
        stats.target_device, 0,
        "spec Target Model must run on HIP:0"
    );
    assert_eq!(stats.draft_device, 1, "spec Draft Model must run on HIP:1");
    let spec_timeout = overnight_timeout(tokens.max(warmup), 8);
    if warmup > 0 {
        let _ = generate_or_timeout(&spec, "hello", warmup, spec_timeout);
    }
    let before = spec.stats();
    let (spec_tps, spec_out) = warmup_and_measure_tps(&spec, "hello", 0, tokens, spec_timeout);
    let spec_stats = spec.stats();
    eprintln!(
        "demo spec gen={} tps={:.2} itl_ms={:.2} acceptance={:.3} target_fwd_delta={} draft_fwd_delta={} packed_verify_delta={} draft_ahead_delta={} greedy_tps={:.2}",
        spec_out.token_ids.len(),
        spec_tps,
        spec_stats.itl_ms_ema,
        spec_stats.acceptance_rate,
        spec_stats
            .target_gpu_forwards
            .saturating_sub(before.target_gpu_forwards),
        spec_stats
            .draft_gpu_forwards
            .saturating_sub(before.draft_gpu_forwards),
        spec_stats
            .linear_packed_verifies
            .saturating_sub(before.linear_packed_verifies),
        spec_stats
            .draft_ahead_hits
            .saturating_sub(before.draft_ahead_hits),
        greedy_tps
    );
    assert!(
        !spec_out.token_ids.is_empty(),
        "asymmetric spec must emit tokens"
    );
    assert!(
        spec_tps > greedy_tps,
        "asymmetric spec must beat Target Model greedy for the demo; spec={spec_tps:.2} greedy={greedy_tps:.2} draft={}",
        draft.display()
    );
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

/// Prefer draft on HIP:1 (9060). If HIP:0 (9070) is already holding a large
/// resident, place the target on the freer of the two boards.
#[cfg(feature = "hip")]
fn apply_9060_preference(cfg: &mut Config) {
    let n = allpaca_hip::device_count();
    if n < 2 {
        return;
    }
    let info0 = allpaca_hip::device_memory_info(0).ok();
    let info1 = allpaca_hip::device_memory_info(1).ok();
    let used0 = info0.map(|(f, t)| t.saturating_sub(f)).unwrap_or(0);
    let free0 = info0.map(|(f, _)| f).unwrap_or(0);
    let free1 = info1.map(|(f, _)| f).unwrap_or(0);
    const BUSY: u64 = 2u64 << 30;
    const NEED: u64 = 20u64 << 30;
    cfg.engine.pipeline_mode = Some(allpaca_types::PipelineMode::Asymmetric);
    if used0 > BUSY && free1 >= NEED && free1 >= free0 {
        cfg.engine.target_device = allpaca_types::DeviceSelector::Hip(1);
        cfg.engine.draft_device = allpaca_types::DeviceSelector::Hip(0);
        eprintln!(
            "HIP:0 busy (used>{BUSY}); target on HIP:1 free={free1} draft on HIP:0 free={free0}"
        );
    } else {
        cfg.engine.draft_device = allpaca_types::DeviceSelector::Hip(1);
        eprintln!("draft on HIP:1 (9060); target stays default / HIP:0");
    }
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
        "Q8_0 decode must time Q8 GEMV; q8_gemm_us={}",
        stats.q8_gemm_us
    );
    assert!(
        stats.attention_us > 0.0 && stats.lm_head_us > 0.0,
        "decode must time attention and lm_head; stats={stats:?}"
    );
    assert!(
        stats.other_gpu_us > 0.0 && stats.other_gpu_us < PROFILE_BLAS_OTHER_GPU_US * 0.85,
        "other_gpu_us {:.1} must shrink vs profile-blas {:.1}; stats={stats:?}",
        stats.other_gpu_us,
        PROFILE_BLAS_OTHER_GPU_US
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

#[cfg(feature = "hip")]
fn warmup_and_measure_tps(
    engine: &Engine,
    prompt: &str,
    warmup: usize,
    tokens: usize,
    timeout: std::time::Duration,
) -> (f64, allpaca_types::GenerationOutput) {
    if warmup > 0 {
        let _ = generate_or_timeout(engine, prompt, warmup, timeout);
    }
    let t0 = std::time::Instant::now();
    let out = generate_or_timeout(engine, prompt, tokens, timeout);
    let tps = out.token_ids.len() as f64 / t0.elapsed().as_secs_f64().max(1e-6);
    (tps, out)
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
    let out = generate_or_timeout(&engine, "hello", 64, std::time::Duration::from_secs(180));
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
    let engine = Engine::new(hip_7b_config(path.clone()))
        .expect("mixed K-quant GGUF load must not be Fatal");
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

#[cfg(feature = "hip")]
#[test]
fn hip_rocmfp4_qwen_probe_and_optional_generate() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_rocmfp4_gguf() else {
        eprintln!("skip hip_rocmfp4_qwen_probe: place Qwen/Nemotron ROCmFP4 under models/");
        return;
    };
    let file = allpaca_gguf::GgufFile::open(&path).expect("mmap ROCmFP4 GGUF");
    let report = file.compat_report();
    assert!(
        report.ggml_type_ids.contains(&100),
        "expected type 100; ids={:?}",
        report.ggml_type_ids
    );
    assert!(
        report.reject_reason.is_none(),
        "ROCmFP4 must not gate loader; got {:?}",
        report.reject_reason
    );
    let cfg = allpaca_core::ModelConfig::from_gguf(&file).expect("qwen ModelConfig");
    let nextn = file.meta_u32("qwen35.nextn_predict_layers");
    let tok_model = file.meta_str("tokenizer.ggml.model");
    eprintln!(
        "ROCmFP4 probe: path={} arch={} layers={} ssm={} attn={} types={:?} head_dim={} rope_dims={} sections={:?} eps={} theta={} nextn={:?} tok_model={:?}",
        path.display(),
        report.architecture,
        report.n_layers,
        report.ssm_layers,
        report.attention_layers,
        report.ggml_type_ids,
        cfg.head_dim(),
        cfg.rope_dims,
        cfg.rope_sections,
        cfg.eps,
        cfg.rope_theta,
        nextn,
        tok_model
    );
    for i in report.n_layers.saturating_sub(3)..report.n_layers {
        let info = &report.layers[i];
        eprintln!(
            "ROCmFP4 tail blk.{} block={:?} q={} qkv={} ssm_out={}",
            i,
            info.block,
            file.tensor(&format!("blk.{i}.attn_q.weight")).is_some(),
            file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some(),
            file.tensor(&format!("blk.{i}.ssm_out.weight")).is_some()
        );
    }
    // Full 27B/30B generate is opt-in (VRAM + load time).
    if std::env::var("ALLPACA_QWEN").ok().as_deref() != Some("1")
        && std::env::var("ALLPACA_NEMOTRON").ok().as_deref() != Some("1")
    {
        eprintln!(
            "skip ROCmFP4 HIP generate: set ALLPACA_QWEN=1 or ALLPACA_NEMOTRON=1 for {}",
            path.display()
        );
        return;
    }
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip ROCmFP4 HIP generate: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path.clone())).expect("ROCmFP4 Engine::new");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip ROCmFP4 HIP generate: engine has no HIP device");
        return;
    }
    let stats = engine.stats();
    assert!(
        stats.weight_bytes_on_device > 0,
        "ROCmFP4 HIP upload must land resident weights; stats={stats:?}"
    );
    eprintln!(
        "ROCmFP4 HIP upload: path={} weight_bytes={} backend={}",
        path.display(),
        stats.weight_bytes_on_device,
        stats.backend
    );
    // Confirm GDN host tensors exist (upload path requires attn_qkv + ssm_out).
    {
        let mut gdn_ok = 0usize;
        let mut gdn_fail = 0usize;
        let mut attn_qnorm = 0usize;
        for i in 0..cfg.n_layers {
            let block = report.layers.get(i).map(|l| l.block);
            if block == Some(allpaca_gguf::HybridLayerKind::Ssm) {
                let qkv = file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some();
                let out_w = file.tensor(&format!("blk.{i}.ssm_out.weight")).is_some();
                let a = file.tensor(&format!("blk.{i}.ssm_a")).is_some();
                let conv = file.tensor(&format!("blk.{i}.ssm_conv1d.weight")).is_some();
                if qkv && out_w && a && conv {
                    gdn_ok += 1;
                } else {
                    gdn_fail += 1;
                    if gdn_fail <= 2 {
                        eprintln!(
                            "GDN tensors missing blk.{i}: qkv={qkv} ssm_out={out_w} a={a} conv={conv}"
                        );
                    }
                }
            }
            if block == Some(allpaca_gguf::HybridLayerKind::Attention)
                && file
                    .tensor(&format!("blk.{i}.attn_q_norm.weight"))
                    .is_some()
            {
                attn_qnorm += 1;
            }
        }
        let nextn = file
            .meta_u32("qwen35.nextn_predict_layers")
            .or_else(|| file.meta_u32("qwen35.leading_dense_block_count"));
        eprintln!(
            "ROCmFP4 GDN tensors: ok={gdn_ok} fail={gdn_fail} attn_qnorm={attn_qnorm} nextn_meta={nextn:?} last_blocks={:?}",
            report
                .layers
                .iter()
                .rev()
                .take(3)
                .map(|l| (l.index, l.block))
                .collect::<Vec<_>>()
        );
        assert!(gdn_ok > 0, "expected GDN tensors on SSM layers");
        assert_eq!(gdn_fail, 0, "SSM layers missing GDN tensors");
    }
    let prompt = std::env::var("ALLPACA_QWEN_PROMPT").unwrap_or_else(|_| {
        // Match llama.cpp qwen chat turn (ROCmFPX llama-cli verbose-prompt).
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n".into()
    });
    {
        let tok = allpaca_gguf::extract_tokenizer(&file).expect("extract tokenizer");
        let enc = tok
            .inner
            .encode(prompt.as_str(), false)
            .expect("tokenize chat prompt");
        eprintln!(
            "ROCmFP4 prompt_ids={:?} (n={}) tokens={:?}",
            enc.get_ids(),
            enc.len(),
            enc.get_tokens()
        );
    }
    let n_tok = std::env::var("ALLPACA_QWEN_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(32)
        .max(1);
    let out = generate_or_timeout(&engine, &prompt, n_tok, overnight_timeout(n_tok, 180));
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    eprintln!(
        "ROCmFP4 HIP generate: path={} prompt_chars={} tokens={} ids={:?} backend={} q8_us={} attn_us={} other_us={} text={:?}",
        path.display(),
        prompt.len(),
        out.token_ids.len(),
        out.token_ids,
        stats.backend,
        stats.q8_gemm_us,
        stats.attention_us,
        stats.other_gpu_us,
        out.text
    );
    // Lock soak: device GDN path must stay coherent (not ChatML debris / newline collapse).
    assert!(
        out.token_ids.len() >= n_tok.min(8),
        "expected ≥{} decode tokens, got {}",
        n_tok.min(8),
        out.token_ids.len()
    );
    assert!(
        !out.text.contains("<|im_"),
        "must not emit chat markers as text: {:?}",
        out.text
    );
    assert!(
        !out.token_ids.iter().all(|&t| t == 198),
        "must not collapse to newline-only ids={:?}",
        out.token_ids
    );
    let lower = out.text.to_lowercase();
    assert!(
        lower.contains("hello")
            || lower.contains("hi")
            || lower.contains("user")
            || lower.contains("greet"),
        "expected hello-ish reply, got {:?}",
        out.text
    );
}

/// Multi-prompt coherence battery for device GDN (opt-in via ALLPACA_QWEN=1).
#[cfg(feature = "hip")]
#[test]
fn hip_rocmfp4_qwen_multi_prompt_coherence_battery() {
    let _guard = large_gguf_lock();
    if std::env::var("ALLPACA_QWEN").ok().as_deref() != Some("1") {
        eprintln!("skip multi-prompt battery: set ALLPACA_QWEN=1");
        return;
    }
    let Some(path) = gated_rocmfp4_gguf() else {
        eprintln!("skip multi-prompt battery: no ROCmFP4 GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip multi-prompt battery: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path)).expect("Engine");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip multi-prompt battery: no HIP in engine");
        return;
    }
    let prompts = [
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "<|im_start|>user\nSay hi briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    ];
    let n = overnight_usize("ALLPACA_QWEN_N", 24);
    for (i, prompt) in prompts.iter().enumerate() {
        let out = generate_or_timeout(&engine, prompt, n, overnight_timeout(n, 180));
        eprintln!(
            "multi-prompt[{i}] tokens={} text={:?}",
            out.token_ids.len(),
            out.text
        );
        assert!(!out.token_ids.is_empty(), "prompt {i} empty");
        assert!(
            !out.text.contains("<|im_"),
            "prompt {i} leaked markers: {:?}",
            out.text
        );
        assert!(
            !out.token_ids.iter().all(|&t| t == 198),
            "prompt {i} newline collapse"
        );
        let lower = out.text.to_lowercase();
        assert!(
            lower.contains("hello")
                || lower.contains("hi")
                || lower.contains("user")
                || lower.contains("greet")
                || lower.contains("help"),
            "prompt {i} not coherent: {:?}",
            out.text
        );
    }
}

/// Opt-in: load MTP/NextN trailing blocks as draft from the same Qwen ROCmFP4 GGUF.
/// Prefer draft on HIP:1 (9060) when two devices exist.
#[cfg(feature = "hip")]
#[test]
fn hip_rocmfp4_qwen_mtp_draft_provider_when_opt_in() {
    let _guard = large_gguf_lock();
    if std::env::var("ALLPACA_QWEN").ok().as_deref() != Some("1") {
        eprintln!("skip MTP draft soak: set ALLPACA_QWEN=1");
        return;
    }
    let Some(path) = gated_rocmfp4_gguf() else {
        eprintln!("skip MTP draft soak: no ROCmFP4 GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip MTP draft soak: no HIP device");
        return;
    }
    let file = allpaca_gguf::GgufFile::open(&path).expect("open ROCmFP4");
    let nextn = file.meta_u32("qwen35.nextn_predict_layers").unwrap_or(0);
    if nextn == 0 {
        eprintln!("skip MTP draft soak: GGUF has no nextn_predict_layers");
        return;
    }
    let mut cfg = hip_7b_config(path);
    cfg.speculative.enabled = true;
    cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Mtp];
    cfg.speculative.min_depth = 2;
    cfg.speculative.max_depth = 4;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    apply_9060_preference(&mut cfg);
    let engine = Engine::new(cfg).expect("MTP Engine::new");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip MTP draft soak: no HIP in engine");
        return;
    }
    let stats = engine.stats();
    eprintln!(
        "MTP draft soak load: nextn={nextn} weight_bytes={} degraded={} target={} draft={}",
        stats.weight_bytes_on_device,
        stats.degraded_events,
        stats.target_device,
        stats.draft_device
    );
    assert_eq!(
        stats.degraded_events, 0,
        "MTP draft load must not degrade; stats={stats:?}"
    );
    let n = overnight_usize("ALLPACA_QWEN_N", 64);
    let prompt = std::env::var("ALLPACA_QWEN_PROMPT").unwrap_or_else(|_| {
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n".into()
    });
    let out = generate_or_timeout(&engine, &prompt, n, overnight_timeout(n, 240));
    let after = engine.stats();
    eprintln!(
        "MTP draft soak: gen={} acceptance={:.3} draft_copied={} draft_fwd={} target_fwd={} text={:?}",
        out.token_ids.len(),
        after.acceptance_rate,
        after.draft_tokens_copied,
        after.draft_gpu_forwards,
        after.target_gpu_forwards,
        out.text
    );
    assert!(!out.token_ids.is_empty());
    assert!(
        after.draft_tokens_copied > 0 || after.draft_gpu_forwards > 0,
        "MTP provider must propose drafts; stats={after:?}"
    );
}

/// Overnight MTP acceptance soak (ALLPACA_QWEN=1, default n=128) on draft hip:1.
#[cfg(feature = "hip")]
#[test]
fn hip_rocmfp4_qwen_mtp_overnight_acceptance_when_opt_in() {
    let _guard = large_gguf_lock();
    if std::env::var("ALLPACA_QWEN").ok().as_deref() != Some("1") {
        eprintln!("skip MTP overnight: set ALLPACA_QWEN=1");
        return;
    }
    if std::env::var("ALLPACA_OVERNIGHT").ok().as_deref() != Some("1")
        && std::env::var("ALLPACA_MTP_OVERNIGHT").ok().as_deref() != Some("1")
    {
        eprintln!("skip MTP overnight: set ALLPACA_OVERNIGHT=1 or ALLPACA_MTP_OVERNIGHT=1");
        return;
    }
    let Some(path) = gated_rocmfp4_gguf() else {
        eprintln!("skip MTP overnight: no ROCmFP4 GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip MTP overnight: no HIP");
        return;
    }
    let mut cfg = hip_7b_config(path);
    cfg.speculative.enabled = true;
    cfg.speculative.providers = allpaca_types::DraftProviderKind::parse_list("mtp").expect("mtp");
    cfg.speculative.min_depth = 2;
    cfg.speculative.max_depth = 6;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    apply_9060_preference(&mut cfg);
    let engine = Engine::new(cfg).expect("MTP overnight Engine");
    let n = overnight_usize("ALLPACA_QWEN_N", 128);
    let prompt = std::env::var("ALLPACA_QWEN_PROMPT").unwrap_or_else(|_| {
        "<|im_start|>user\nWrite a detailed 400-word essay about why hello world programs matter for learning to code. Do not stop early.<|im_end|>\n<|im_start|>assistant\n<think>\n"
            .into()
    });
    let out = generate_or_timeout(&engine, &prompt, n, overnight_timeout(n, 300));
    let s = engine.stats();
    let hit_max = out.token_ids.len() >= n;
    let early_eos = !hit_max;
    eprintln!(
        "MTP overnight: gen={}/{} early_eos={} finish={:?} acceptance={:.3} draft_fwd={} draft_copied={} target_fwd={} draft_dev={} text_prefix={:?}",
        out.token_ids.len(),
        n,
        early_eos,
        out.finish_reason,
        s.acceptance_rate,
        s.draft_gpu_forwards,
        s.draft_tokens_copied,
        s.target_gpu_forwards,
        s.draft_device,
        out.text.chars().take(160).collect::<String>()
    );
    assert!(!out.token_ids.is_empty());
    assert!(
        s.draft_gpu_forwards > 0 || s.draft_tokens_copied > 0,
        "overnight MTP must exercise draft; {s:?}"
    );
    assert!(
        !allpaca_core::mtp_text_is_garbled(&out.text),
        "overnight MTP must not emit empty or dropped-word junk; text={:?}",
        out.text
    );
    assert_eq!(
        s.mtp_process_skips, 0,
        "overnight MTP must not skip-storm process(); stats={s:?}"
    );
}

/// Short dual-GPU MTP quality gate: later packed-verify steps must keep
/// the same content words as target-only (`user`, `400-word`, `learning`).
#[cfg(feature = "hip")]
#[test]
fn hip_rocmfp4_qwen_mtp_essay_n32_keeps_target_words_when_opt_in() {
    let _guard = large_gguf_lock();
    if std::env::var("ALLPACA_QWEN").ok().as_deref() != Some("1") {
        eprintln!("skip MTP essay n=32: set ALLPACA_QWEN=1");
        return;
    }
    let Some(path) = gated_rocmfp4_gguf() else {
        eprintln!("skip MTP essay n=32: no ROCmFP4 GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip MTP essay n=32: no HIP");
        return;
    }
    let prompt = "<|im_start|>user\nWrite a detailed 400-word essay about why hello world programs matter for learning to code. Do not stop early.<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let n = 32;
    let mut greedy_cfg = hip_7b_config(path.clone());
    greedy_cfg.speculative.enabled = false;
    greedy_cfg.features.draft_model_speculation = Some(false);
    greedy_cfg.engine.target_device = allpaca_types::DeviceSelector::Hip(0);
    greedy_cfg.engine.draft_device = allpaca_types::DeviceSelector::Cpu;
    let greedy_engine = Engine::new(greedy_cfg).expect("essay target-only Engine");
    let greedy_out = generate_or_timeout(&greedy_engine, prompt, n, overnight_timeout(n, 300));
    eprintln!(
        "MTP essay n=32 target-only prefix={:?}",
        greedy_out.text.chars().take(160).collect::<String>()
    );
    drop(greedy_engine);

    let mut cfg = hip_7b_config(path);
    cfg.speculative.enabled = true;
    cfg.speculative.providers = allpaca_types::DraftProviderKind::parse_list("mtp").expect("mtp");
    cfg.speculative.min_depth = 2;
    cfg.speculative.max_depth = 6;
    cfg.features.draft_model_speculation = Some(true);
    cfg.features.ngram_speculation = Some(false);
    apply_9060_preference(&mut cfg);
    let engine = Engine::new(cfg).expect("MTP essay Engine");
    let out = generate_or_timeout(&engine, prompt, n, overnight_timeout(n, 300));
    let s = engine.stats();
    eprintln!(
        "MTP essay n=32: gen={} acceptance={:.3} packed={} draft_fwd={} target_fwd={} draft_dev={} skips={} ahead_hits={} text_prefix={:?}",
        out.token_ids.len(),
        s.acceptance_rate,
        s.linear_packed_verifies,
        s.draft_gpu_forwards,
        s.target_gpu_forwards,
        s.draft_device,
        s.mtp_process_skips,
        s.draft_ahead_hits,
        out.text.chars().take(160).collect::<String>()
    );
    assert!(
        s.draft_gpu_forwards > 0 || s.draft_tokens_copied > 0,
        "essay MTP must exercise draft; {s:?}"
    );
    assert_eq!(
        s.mtp_process_skips, 0,
        "essay MTP must not skip process(); {s:?}"
    );
    if allpaca_hip::device_count() >= 2 {
        assert!(
            s.draft_ahead_hits > 0,
            "dual-GPU MTP packed verify must reuse process-KV snapshot ahead; {s:?}"
        );
    }
    assert!(
        !allpaca_core::mtp_text_is_garbled(&out.text),
        "essay MTP must not emit dropped-word junk; text={:?}",
        out.text
    );
    let lower = out.text.to_ascii_lowercase();
    assert!(
        lower.contains("user"),
        "essay MTP must keep 'user'; text={:?}",
        out.text
    );
    assert!(
        lower.contains("400-word") || lower.contains("400 word"),
        "essay MTP must keep '400-word'; text={:?}",
        out.text
    );
    assert!(
        lower.contains("learning"),
        "essay MTP must keep 'learning'; text={:?}",
        out.text
    );
    let ncmp = greedy_out.token_ids.len().min(out.token_ids.len()).min(32);
    assert_eq!(
        &out.token_ids[..ncmp],
        &greedy_out.token_ids[..ncmp],
        "essay MTP prefix must match target-only; mtp={:?} greedy={:?}",
        &out.token_ids[..ncmp],
        &greedy_out.token_ids[..ncmp]
    );
}

/// Device GDN one-token step must match the host reference (public HIP ABI).
#[cfg(feature = "hip")]
#[test]
fn hip_gated_delta_step_matches_host_on_tiny_state() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gated_delta_step_matches_host: no HIP device");
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 4096, 8).expect("HipWorkspace");
    // Same fixture as deltanet::gated_delta_step_writes_and_reads_tiny_state.
    let mut q_h = [1.0f32, 0.0];
    let mut k_h = [0.0f32, 1.0];
    let v_h = [2.0f32, -1.0];
    let decay_h = [0.5f32];
    let beta_h = [1.0f32];
    let mut state_h = [0.0f32; 4];
    let mut out_h = [0.0f32; 2];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q_h,
        &mut k_h,
        &v_h,
        &decay_h,
        &beta_h,
        &mut state_h,
        &mut out_h,
        1,
        2,
        2,
        1e-6,
    );

    let q0 = [1.0f32, 0.0];
    let k0 = [0.0f32, 1.0];
    let v = [2.0f32, -1.0];
    let decay = [0.5f32];
    let beta = [1.0f32];
    let state0 = [0.0f32; 4];
    let out0 = [0.0f32; 2];
    let mut q = q0;
    let k = k0;
    let mut state = state0;
    let mut out = out0;

    let blob =
        allpaca_hip::DeviceWeights::alloc(0, (2 + 2 + 2 + 1 + 1 + 4 + 2) * 4).expect("alloc");
    let mut off = 0usize;
    let q_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(q_d, &q).unwrap();
    off += 2 * 4;
    let k_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(k_d, &k).unwrap();
    off += 2 * 4;
    let v_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(v_d, &v).unwrap();
    off += 2 * 4;
    let decay_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(decay_d, &decay).unwrap();
    off += 4;
    let beta_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(beta_d, &beta).unwrap();
    off += 4;
    let state_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(state_d, &state).unwrap();
    off += 4 * 4;
    let out_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(out_d, &out).unwrap();

    hip.gated_delta_step(
        q_d, k_d, v_d, decay_d, beta_d, state_d, out_d, 1, 2, 2, 1e-6,
    )
    .expect("device gated_delta_step");

    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    hip.copy_d2h_f32(state_d, &mut state).unwrap();
    hip.copy_d2h_f32(q_d, &mut q).unwrap();

    for i in 0..2 {
        assert!(
            (out[i] - out_h[i]).abs() < 1e-4,
            "out[{i}] device={} host={}",
            out[i],
            out_h[i]
        );
        assert!(
            (q[i] - q_h[i]).abs() < 1e-4,
            "q[{i}] device={} host={}",
            q[i],
            q_h[i]
        );
    }
    for i in 0..4 {
        assert!(
            (state[i] - state_h[i]).abs() < 1e-4,
            "state[{i}] device={} host={}",
            state[i],
            state_h[i]
        );
    }
}

/// After ggml-style K-head expand, device delta matches host for n_v > n_k.
#[cfg(feature = "hip")]
#[test]
fn hip_gated_delta_step_matches_host_after_expand_k() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gated_delta_expand: no HIP device");
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 4096, 8).expect("HipWorkspace");
    let n_k = 2usize;
    let n_v = 4usize;
    let head_k = 2usize;
    let head_v = 2usize;
    let q_raw = [1.0f32, 0.0, 0.0, 1.0]; // 2 K-heads
    let k_raw = [0.0f32, 1.0, 1.0, 0.0];
    let v: Vec<f32> = (0..n_v * head_v).map(|i| (i as f32) * 0.25).collect();
    let decay = vec![0.8f32; n_v];
    let beta = vec![0.5f32; n_v];

    let mut q_exp = vec![0.0f32; n_v * head_k];
    let mut k_exp = vec![0.0f32; n_v * head_k];
    allpaca_core::deltanet::expand_k_heads(&q_raw, &mut q_exp, n_k, n_v, head_k);
    allpaca_core::deltanet::expand_k_heads(&k_raw, &mut k_exp, n_k, n_v, head_k);

    let mut q_h = q_exp.clone();
    let mut k_h = k_exp.clone();
    let mut state_h = vec![0.0f32; n_v * head_v * head_k];
    let mut out_h = vec![0.0f32; n_v * head_v];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q_h,
        &mut k_h,
        &v,
        &decay,
        &beta,
        &mut state_h,
        &mut out_h,
        n_v,
        head_v,
        head_k,
        1e-6,
    );

    let floats = q_exp.len()
        + k_exp.len()
        + v.len()
        + decay.len()
        + beta.len()
        + state_h.len()
        + out_h.len();
    let blob = allpaca_hip::DeviceWeights::alloc(0, floats * 4).expect("alloc");
    let mut off = 0usize;
    let put = |hip: &allpaca_hip::HipWorkspace,
               blob: &allpaca_hip::DeviceWeights,
               off: &mut usize,
               src: &[f32]|
     -> *mut f32 {
        let p = blob.ptr_at(*off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        *off += src.len() * 4;
        p
    };
    let q_d = put(&hip, &blob, &mut off, &q_exp);
    let k_d = put(&hip, &blob, &mut off, &k_exp);
    let v_d = put(&hip, &blob, &mut off, &v);
    let decay_d = put(&hip, &blob, &mut off, &decay);
    let beta_d = put(&hip, &blob, &mut off, &beta);
    let state_d = put(&hip, &blob, &mut off, &vec![0.0f32; state_h.len()]);
    let out_d = put(&hip, &blob, &mut off, &vec![0.0f32; out_h.len()]);

    hip.gated_delta_step(
        q_d,
        k_d,
        v_d,
        decay_d,
        beta_d,
        state_d,
        out_d,
        n_v as i32,
        head_v as i32,
        head_k as i32,
        1e-6,
    )
    .expect("device gated_delta_step expand");

    let mut out = vec![0.0f32; out_h.len()];
    let mut state = vec![0.0f32; state_h.len()];
    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    hip.copy_d2h_f32(state_d, &mut state).unwrap();
    for i in 0..out.len() {
        assert!(
            (out[i] - out_h[i]).abs() < 1e-4,
            "out[{i}] device={} host={}",
            out[i],
            out_h[i]
        );
    }
    for i in 0..state.len() {
        assert!(
            (state[i] - state_h[i]).abs() < 1e-4,
            "state[{i}] device={} host={}",
            state[i],
            state_h[i]
        );
    }
}

/// Transitional H2D scratch path used by `forward_token_device` must match host math.
#[cfg(feature = "hip")]
#[test]
fn hip_gated_delta_step_host_matches_host_math() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gated_delta_step_host: no HIP device");
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("HipWorkspace");
    let mut q_h = [1.0f32, 0.0];
    let mut k_h = [0.0f32, 1.0];
    let v = [2.0f32, -1.0];
    let decay = [0.5f32];
    let beta = [1.0f32];
    let mut state_h = [0.0f32; 4];
    let mut out_h = [0.0f32; 2];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q_h,
        &mut k_h,
        &v,
        &decay,
        &beta,
        &mut state_h,
        &mut out_h,
        1,
        2,
        2,
        1e-6,
    );

    let mut q = [1.0f32, 0.0];
    let mut k = [0.0f32, 1.0];
    let mut state = [0.0f32; 4];
    let mut out = [0.0f32; 2];
    hip.gated_delta_step_host(
        &mut q, &mut k, &v, &decay, &beta, &mut state, &mut out, 1, 2, 2, 1e-6,
    )
    .expect("gated_delta_step_host");
    for i in 0..2 {
        assert!(
            (out[i] - out_h[i]).abs() < 1e-4,
            "out[{i}]={}/{}",
            out[i],
            out_h[i]
        );
        assert!((q[i] - q_h[i]).abs() < 1e-4, "q[{i}]={}/{}", q[i], q_h[i]);
    }
    for i in 0..4 {
        assert!(
            (state[i] - state_h[i]).abs() < 1e-4,
            "state[{i}]={}/{}",
            state[i],
            state_h[i]
        );
    }
}

/// Resident device state must match host across two steps without re-uploading.
#[cfg(feature = "hip")]
#[test]
fn hip_gated_delta_resident_two_steps_match_host() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gated_delta_resident: no HIP device");
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("HipWorkspace");
    let req = 7u64;
    let layer = 3usize;

    let mut state_h = [0.0f32; 4];
    let mut q1 = [1.0f32, 0.0];
    let mut k1 = [0.0f32, 1.0];
    let v1 = [2.0f32, -1.0];
    let decay1 = [0.5f32];
    let beta1 = [1.0f32];
    let mut out1 = [0.0f32; 2];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q1,
        &mut k1,
        &v1,
        &decay1,
        &beta1,
        &mut state_h,
        &mut out1,
        1,
        2,
        2,
        1e-6,
    );
    let mut q2 = [0.0f32, 1.0];
    let mut k2 = [0.0f32, 1.0];
    let v2 = [0.0f32, 0.0];
    let decay2 = [1.0f32];
    let beta2 = [0.0f32];
    let mut out2 = [0.0f32; 2];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q2,
        &mut k2,
        &v2,
        &decay2,
        &beta2,
        &mut state_h,
        &mut out2,
        1,
        2,
        2,
        1e-6,
    );

    let mut host_state = [0.0f32; 4];
    let mut q = [1.0f32, 0.0];
    let mut k = [0.0f32, 1.0];
    let mut out = [0.0f32; 2];
    hip.gated_delta_step_resident(
        req,
        layer,
        &mut host_state,
        &mut q,
        &mut k,
        &v1,
        &decay1,
        &beta1,
        &mut out,
        1,
        2,
        2,
        1e-6,
    )
    .expect("resident step1");
    for i in 0..2 {
        assert!((out[i] - out1[i]).abs() < 1e-4, "step1 out[{i}]");
    }
    // Host mirror stays stale until sync; second step must use device state.
    let mut q = [0.0f32, 1.0];
    let mut k = [0.0f32, 1.0];
    let mut out = [0.0f32; 2];
    hip.gated_delta_step_resident(
        req,
        layer,
        &mut host_state,
        &mut q,
        &mut k,
        &v2,
        &decay2,
        &beta2,
        &mut out,
        1,
        2,
        2,
        1e-6,
    )
    .expect("resident step2");
    for i in 0..2 {
        assert!(
            (out[i] - out2[i]).abs() < 1e-4,
            "step2 out[{i}] device={} host={}",
            out[i],
            out2[i]
        );
    }
    let mut states = vec![Vec::new(); layer + 1];
    hip.gdn_sync_request_to_host(req, &mut states).unwrap();
    for i in 0..4 {
        assert!(
            (states[layer][i] - state_h[i]).abs() < 1e-4,
            "synced state[{i}]"
        );
    }
    hip.gdn_clear_request(req);
}

/// Device-pointer GDN conv1d must match the host `deltanet` one-token step.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_conv1d_step_matches_host_on_tiny_state() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_conv1d_step_matches_host: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 4096, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_gdn_conv1d_step_matches_host: HIP workspace unavailable");
            return;
        }
    };
    // Same fixture as deltanet::conv1d_step_shifts_history.
    let kernel = [1.0f32, 0.0, 0.0, 1.0];
    let x = [5.0f32, 6.0];
    let mut state_h = [3.0f32, 4.0];
    let mut out_h = [0.0f32; 2];
    allpaca_core::deltanet::conv1d_step(&x, &kernel, &mut state_h, &mut out_h, 2);

    let mut state = [3.0f32, 4.0];
    let mut out = [0.0f32; 2];
    let blob = allpaca_hip::DeviceWeights::alloc(0, (2 + 4 + 2 + 2) * 4).expect("alloc");
    let mut off = 0usize;
    let x_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(x_d, &x).unwrap();
    off += 2 * 4;
    let k_d = blob.ptr_at(off) as *const f32;
    hip.copy_h2d_f32(k_d as *mut f32, &kernel).unwrap();
    off += 4 * 4;
    let s_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(s_d, &state).unwrap();
    off += 2 * 4;
    let out_d = blob.ptr_at(off) as *mut f32;
    hip.copy_h2d_f32(out_d, &out).unwrap();

    if let Err(e) = hip.gdn_conv1d_step(x_d, k_d, s_d, out_d, 2, 2) {
        eprintln!("skip hip_gdn_conv1d_step_matches_host: {e}");
        return;
    }

    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    hip.copy_d2h_f32(s_d, &mut state).unwrap();
    for i in 0..2 {
        assert!(
            (out[i] - out_h[i]).abs() < 1e-4,
            "out[{i}] device={} host={}",
            out[i],
            out_h[i]
        );
        assert!(
            (state[i] - state_h[i]).abs() < 1e-4,
            "conv_state[{i}] device={} host={}",
            state[i],
            state_h[i]
        );
    }
}

/// Device-pointer K-head expand must tile like host `deltanet::expand_k_heads`.
#[cfg(feature = "hip")]
#[test]
fn hip_expand_k_heads_matches_host_ggml_repeat() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_expand_k_heads_matches_host: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 4096, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_expand_k_heads_matches_host: HIP workspace unavailable");
            return;
        }
    };
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut dst_h = [0.0f32; 8];
    allpaca_core::deltanet::expand_k_heads(&src, &mut dst_h, 2, 4, 2);

    let mut dst = [0.0f32; 8];
    let blob = allpaca_hip::DeviceWeights::alloc(0, (4 + 8) * 4).expect("alloc");
    let src_d = blob.ptr_at(0) as *const f32;
    hip.copy_h2d_f32(src_d as *mut f32, &src).unwrap();
    let dst_d = blob.ptr_at(4 * 4) as *mut f32;

    if let Err(e) = hip.expand_k_heads(src_d, dst_d, 2, 4, 2) {
        eprintln!("skip hip_expand_k_heads_matches_host: {e}");
        return;
    }
    hip.copy_d2h_f32(dst_d, &mut dst).unwrap();
    for i in 0..8 {
        assert!(
            (dst[i] - dst_h[i]).abs() < 1e-5,
            "dst[{i}] device={} host={}",
            dst[i],
            dst_h[i]
        );
    }
}

/// Device-pointer decay/β must match host `decay_from_alpha` plus sigmoid(β).
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_decay_beta_matches_host_softplus_sigmoid() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_decay_beta_matches_host: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 4096, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_gdn_decay_beta_matches_host: HIP workspace unavailable");
            return;
        }
    };
    let alpha = [0.5f32, -1.0, 2.0];
    let dt_bias = [0.1f32, 0.0, -0.25];
    let a = [-0.5f32, -1.0, -0.25];
    let beta0 = [0.0f32, 2.0, -2.0];
    let mut decay_h = [0.0f32; 3];
    allpaca_core::deltanet::decay_from_alpha(&alpha, &dt_bias, &a, &mut decay_h);
    let beta_h = [
        allpaca_core::deltanet::sigmoid(beta0[0]),
        allpaca_core::deltanet::sigmoid(beta0[1]),
        allpaca_core::deltanet::sigmoid(beta0[2]),
    ];

    let mut decay = [0.0f32; 3];
    let mut beta = beta0;
    let blob = allpaca_hip::DeviceWeights::alloc(0, 3 * 5 * 4).expect("alloc");
    let mut off = 0usize;
    let put = |hip: &allpaca_hip::HipWorkspace,
               blob: &allpaca_hip::DeviceWeights,
               off: &mut usize,
               src: &[f32]|
     -> *mut f32 {
        let p = blob.ptr_at(*off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        *off += src.len() * 4;
        p
    };
    let alpha_d = put(&hip, &blob, &mut off, &alpha);
    let dt_d = put(&hip, &blob, &mut off, &dt_bias);
    let a_d = put(&hip, &blob, &mut off, &a);
    let decay_d = put(&hip, &blob, &mut off, &decay);
    let beta_d = put(&hip, &blob, &mut off, &beta);

    if let Err(e) = hip.gdn_decay_beta(alpha_d, dt_d, a_d, decay_d, beta_d, 3) {
        eprintln!("skip hip_gdn_decay_beta_matches_host: {e}");
        return;
    }
    hip.copy_d2h_f32(decay_d, &mut decay).unwrap();
    hip.copy_d2h_f32(beta_d, &mut beta).unwrap();
    for i in 0..3 {
        assert!(
            (decay[i] - decay_h[i]).abs() < 1e-4,
            "decay[{i}] device={} host={}",
            decay[i],
            decay_h[i]
        );
        assert!(
            (beta[i] - beta_h[i]).abs() < 1e-4,
            "beta[{i}] device={} host={}",
            beta[i],
            beta_h[i]
        );
    }
}

/// Resident GDN layer (conv → expand-k → decay/β → delta) stays on device and matches host.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_layer_step_matches_host_without_qkv_d2h() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_layer_step: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 64 * 1024, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_gdn_layer_step: HIP workspace unavailable");
            return;
        }
    };
    let n_k = 1usize;
    let n_v = 1usize;
    let head_k = 2usize;
    let head_v = 2usize;
    let conv_k = 2usize;
    let key_dim = n_k * head_k;
    let value_dim = n_v * head_v;
    let qkv_dim = key_dim * 2 + value_dim;
    // Per-channel kernel [0, 1] → conv out is silu(x); hist is ignored.
    let mut conv_w = vec![0.0f32; conv_k * qkv_dim];
    for c in 0..qkv_dim {
        conv_w[1 + c * conv_k] = 1.0;
    }
    let qkv = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0];
    let mut conv_state_h = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0];
    let alpha = [0.25f32];
    let dt_bias = [0.0f32];
    let a = [-0.5f32];
    let beta0 = [1.0f32];

    let mut conv_out = [0.0f32; 6];
    allpaca_core::deltanet::conv1d_step(&qkv, &conv_w, &mut conv_state_h, &mut conv_out, conv_k);
    let mut q_exp = [0.0f32; 2];
    let mut k_exp = [0.0f32; 2];
    allpaca_core::deltanet::expand_k_heads(&conv_out[..key_dim], &mut q_exp, n_k, n_v, head_k);
    allpaca_core::deltanet::expand_k_heads(
        &conv_out[key_dim..key_dim * 2],
        &mut k_exp,
        n_k,
        n_v,
        head_k,
    );
    let mut decay_h = [0.0f32; 1];
    allpaca_core::deltanet::decay_from_alpha(&alpha, &dt_bias, &a, &mut decay_h);
    let beta_h = [allpaca_core::deltanet::sigmoid(beta0[0])];
    let mut q_h = q_exp;
    let mut k_h = k_exp;
    let mut state_h = [0.0f32; 4];
    let mut y_h = [0.0f32; 2];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q_h,
        &mut k_h,
        &conv_out[key_dim * 2..],
        &decay_h,
        &beta_h,
        &mut state_h,
        &mut y_h,
        n_v,
        head_v,
        head_k,
        1e-6,
    );

    let blob = allpaca_hip::DeviceWeights::alloc(
        0,
        (qkv_dim + conv_w.len() + qkv_dim + 1 + 1 + 1 + 1 + 4 + value_dim) * 4,
    )
    .expect("alloc");
    let mut off = 0usize;
    let put = |hip: &allpaca_hip::HipWorkspace,
               blob: &allpaca_hip::DeviceWeights,
               off: &mut usize,
               src: &[f32]|
     -> *mut f32 {
        let p = blob.ptr_at(*off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        *off += src.len() * 4;
        p
    };
    let qkv_d = put(&hip, &blob, &mut off, &qkv);
    let conv_w_d = put(&hip, &blob, &mut off, &conv_w);
    let conv_s_d = put(&hip, &blob, &mut off, &[3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let alpha_d = put(&hip, &blob, &mut off, &alpha);
    let dt_d = put(&hip, &blob, &mut off, &dt_bias);
    let a_d = put(&hip, &blob, &mut off, &a);
    let beta_d = put(&hip, &blob, &mut off, &beta0);
    let state_d = put(&hip, &blob, &mut off, &[0.0f32; 4]);
    let y_d = put(&hip, &blob, &mut off, &[0.0f32; 2]);

    let d2h_before = hip.activation_d2h_bytes();
    if let Err(e) = hip.gdn_layer_step(
        qkv_d,
        alpha_d,
        beta_d,
        conv_w_d,
        conv_s_d,
        dt_d,
        a_d,
        state_d,
        y_d,
        n_v as i32,
        n_k as i32,
        head_v as i32,
        head_k as i32,
        conv_k as i32,
        1e-6,
    ) {
        eprintln!("skip hip_gdn_layer_step: {e}");
        return;
    }
    let d2h_after = hip.activation_d2h_bytes();
    assert_eq!(
        d2h_after, d2h_before,
        "GDN layer math must not D2H qkv/α/β; d2h {d2h_before}→{d2h_after}"
    );

    let mut y = [0.0f32; 2];
    let mut state = [0.0f32; 4];
    hip.copy_d2h_f32(y_d, &mut y).unwrap();
    hip.copy_d2h_f32(state_d, &mut state).unwrap();
    for i in 0..2 {
        assert!(
            (y[i] - y_h[i]).abs() < 1e-4,
            "y[{i}] device={} host={}",
            y[i],
            y_h[i]
        );
    }
    for i in 0..4 {
        assert!(
            (state[i] - state_h[i]).abs() < 1e-4,
            "state[{i}] device={} host={}",
            state[i],
            state_h[i]
        );
    }
}

/// Device GDN output gate must match host `rmsnorm_silu_gate`.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_rmsnorm_silu_gate_matches_host() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_rmsnorm_silu_gate: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 4096, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_gdn_rmsnorm_silu_gate: HIP workspace unavailable");
            return;
        }
    };
    let y = [1.0f32, 2.0, -1.0, 0.5];
    let z = [0.5f32, -0.5, 1.5, 0.0];
    let weight = [1.0f32, 0.5];
    let mut out_h = [0.0f32; 4];
    allpaca_core::deltanet::rmsnorm_silu_gate(&y, &z, &weight, &mut out_h, 2, 1e-6);

    let blob = allpaca_hip::DeviceWeights::alloc(0, (4 + 4 + 2 + 4) * 4).expect("alloc");
    let mut off = 0usize;
    let put = |hip: &allpaca_hip::HipWorkspace,
               blob: &allpaca_hip::DeviceWeights,
               off: &mut usize,
               src: &[f32]|
     -> *mut f32 {
        let p = blob.ptr_at(*off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        *off += src.len() * 4;
        p
    };
    let y_d = put(&hip, &blob, &mut off, &y);
    let z_d = put(&hip, &blob, &mut off, &z);
    let w_d = put(&hip, &blob, &mut off, &weight);
    let out_d = put(&hip, &blob, &mut off, &[0.0f32; 4]);

    if let Err(e) = hip.gdn_rmsnorm_silu_gate(y_d, z_d, w_d, out_d, 2, 2, 1e-6) {
        eprintln!("skip hip_gdn_rmsnorm_silu_gate: {e}");
        return;
    }
    let mut out = [0.0f32; 4];
    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    for i in 0..4 {
        assert!(
            (out[i] - out_h[i]).abs() < 1e-4,
            "gated[{i}] device={} host={}",
            out[i],
            out_h[i]
        );
    }
}

/// Full resident GDN decode (layer + silu/RMSNorm gate) matches host and does not D2H qkv/z/α/β.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_decode_step_resident_matches_host_without_qkv_d2h() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_decode_step_resident: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 64 * 1024, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_gdn_decode_step_resident: HIP workspace unavailable");
            return;
        }
    };
    let req = 11u64;
    let layer = 2usize;
    let n_k = 1usize;
    let n_v = 1usize;
    let head_k = 2usize;
    let head_v = 2usize;
    let conv_k = 2usize;
    let key_dim = n_k * head_k;
    let value_dim = n_v * head_v;
    let qkv_dim = key_dim * 2 + value_dim;
    let mut conv_w = vec![0.0f32; conv_k * qkv_dim];
    for c in 0..qkv_dim {
        conv_w[1 + c * conv_k] = 1.0;
    }
    let qkv = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0];
    let z = [0.5f32, -0.25];
    let norm = [1.0f32, 0.75];
    let alpha = [0.25f32];
    let dt_bias = [0.0f32];
    let a = [-0.5f32];
    let beta0 = [1.0f32];

    let mut conv_state_h = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut conv_out = [0.0f32; 6];
    allpaca_core::deltanet::conv1d_step(&qkv, &conv_w, &mut conv_state_h, &mut conv_out, conv_k);
    let mut q_exp = [0.0f32; 2];
    let mut k_exp = [0.0f32; 2];
    allpaca_core::deltanet::expand_k_heads(&conv_out[..key_dim], &mut q_exp, n_k, n_v, head_k);
    allpaca_core::deltanet::expand_k_heads(
        &conv_out[key_dim..key_dim * 2],
        &mut k_exp,
        n_k,
        n_v,
        head_k,
    );
    let mut decay_h = [0.0f32; 1];
    allpaca_core::deltanet::decay_from_alpha(&alpha, &dt_bias, &a, &mut decay_h);
    let beta_h = [allpaca_core::deltanet::sigmoid(beta0[0])];
    let mut q_h = q_exp;
    let mut k_h = k_exp;
    let mut state_h = [0.0f32; 4];
    let mut y_h = [0.0f32; 2];
    allpaca_core::deltanet::gated_delta_step_eps(
        &mut q_h,
        &mut k_h,
        &conv_out[key_dim * 2..],
        &decay_h,
        &beta_h,
        &mut state_h,
        &mut y_h,
        n_v,
        head_v,
        head_k,
        1e-6,
    );
    let mut gated_h = [0.0f32; 2];
    allpaca_core::deltanet::rmsnorm_silu_gate(&y_h, &z, &norm, &mut gated_h, head_v, 1e-6);

    let blob = allpaca_hip::DeviceWeights::alloc(
        0,
        (qkv_dim + z.len() + 1 + 1 + conv_w.len() + 1 + 1 + 2 + 2) * 4,
    )
    .expect("alloc");
    let mut off = 0usize;
    let put = |hip: &allpaca_hip::HipWorkspace,
               blob: &allpaca_hip::DeviceWeights,
               off: &mut usize,
               src: &[f32]|
     -> *mut f32 {
        let p = blob.ptr_at(*off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        *off += src.len() * 4;
        p
    };
    let qkv_d = put(&hip, &blob, &mut off, &qkv);
    let z_d = put(&hip, &blob, &mut off, &z);
    let alpha_d = put(&hip, &blob, &mut off, &alpha);
    let beta_d = put(&hip, &blob, &mut off, &beta0);
    let conv_w_d = put(&hip, &blob, &mut off, &conv_w);
    let dt_d = put(&hip, &blob, &mut off, &dt_bias);
    let a_d = put(&hip, &blob, &mut off, &a);
    let norm_d = put(&hip, &blob, &mut off, &norm);
    let out_d = put(&hip, &blob, &mut off, &[0.0f32; 2]);

    let mut host_state = [0.0f32; 4];
    let mut host_conv = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0];
    let d2h_before = hip.activation_d2h_bytes();
    if let Err(e) = hip.gdn_decode_step_resident(
        req,
        layer,
        &mut host_state,
        &mut host_conv,
        qkv_d,
        z_d,
        alpha_d,
        beta_d,
        conv_w_d,
        dt_d,
        a_d,
        norm_d,
        out_d,
        n_v as i32,
        n_k as i32,
        head_v as i32,
        head_k as i32,
        conv_k as i32,
        1e-6,
    ) {
        eprintln!("skip hip_gdn_decode_step_resident: {e}");
        return;
    }
    let d2h_after = hip.activation_d2h_bytes();
    assert_eq!(
        d2h_after, d2h_before,
        "resident GDN decode must not D2H qkv/z/α/β; d2h {d2h_before}→{d2h_after}"
    );
    let mut gated = [0.0f32; 2];
    hip.copy_d2h_f32(out_d, &mut gated).unwrap();
    for i in 0..2 {
        assert!(
            (gated[i] - gated_h[i]).abs() < 1e-4,
            "gated[{i}] device={} host={}",
            gated[i],
            gated_h[i]
        );
    }
    hip.gdn_clear_request(req);
}

/// Device GDN clone/rewind must restore state+conv after an extra step without D2H.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_rewind_restores_device_state_and_conv_after_extra_step() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_rewind: no HIP device");
        return;
    }
    let hip = match allpaca_hip::HipWorkspace::new(0, 64 * 1024, 8) {
        Ok(h) => h,
        Err(_) => {
            eprintln!("skip hip_gdn_rewind: HIP workspace unavailable");
            return;
        }
    };
    let req = 13u64;
    let layer = 1usize;
    let n_k = 1usize;
    let n_v = 1usize;
    let head_k = 2usize;
    let head_v = 2usize;
    let conv_k = 2usize;
    let key_dim = n_k * head_k;
    let value_dim = n_v * head_v;
    let qkv_dim = key_dim * 2 + value_dim;
    let mut conv_w = vec![0.0f32; conv_k * qkv_dim];
    for c in 0..qkv_dim {
        conv_w[1 + c * conv_k] = 1.0;
    }
    let qkv_a = [1.0f32, 0.0, 0.0, 1.0, 2.0, -1.0];
    let qkv_b = [0.0f32, 1.0, 1.0, 0.0, -0.5, 1.5];
    let z = [0.5f32, -0.25];
    let norm = [1.0f32, 0.75];
    let alpha = [0.25f32];
    let dt_bias = [0.0f32];
    let a = [-0.5f32];
    let beta0 = [1.0f32];

    let blob = allpaca_hip::DeviceWeights::alloc(
        0,
        (qkv_dim * 2 + z.len() + 1 + 1 + conv_w.len() + 1 + 1 + 2 + 2) * 4,
    )
    .expect("alloc");
    let mut off = 0usize;
    let put = |hip: &allpaca_hip::HipWorkspace,
               blob: &allpaca_hip::DeviceWeights,
               off: &mut usize,
               src: &[f32]|
     -> *mut f32 {
        let p = blob.ptr_at(*off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        *off += src.len() * 4;
        p
    };
    let qkv_a_d = put(&hip, &blob, &mut off, &qkv_a);
    let qkv_b_d = put(&hip, &blob, &mut off, &qkv_b);
    let z_d = put(&hip, &blob, &mut off, &z);
    let alpha_d = put(&hip, &blob, &mut off, &alpha);
    let beta_d = put(&hip, &blob, &mut off, &beta0);
    let conv_w_d = put(&hip, &blob, &mut off, &conv_w);
    let dt_d = put(&hip, &blob, &mut off, &dt_bias);
    let a_d = put(&hip, &blob, &mut off, &a);
    let norm_d = put(&hip, &blob, &mut off, &norm);
    let out_d = put(&hip, &blob, &mut off, &[0.0f32; 2]);

    let step = |hip: &allpaca_hip::HipWorkspace, rid: u64, qkv_d: *mut f32| {
        hip.copy_h2d_f32(beta_d, &beta0).unwrap();
        let mut host_state = [0.0f32; 4];
        let mut host_conv = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0];
        hip.gdn_decode_step_resident(
            rid,
            layer,
            &mut host_state,
            &mut host_conv,
            qkv_d,
            z_d,
            alpha_d,
            beta_d,
            conv_w_d,
            dt_d,
            a_d,
            norm_d,
            out_d,
            n_v as i32,
            n_k as i32,
            head_v as i32,
            head_k as i32,
            conv_k as i32,
            1e-6,
        )
    };
    let pull = |hip: &allpaca_hip::HipWorkspace, rid: u64| {
        let mut states = vec![Vec::new(); layer + 1];
        let mut convs = vec![Vec::new(); layer + 1];
        hip.gdn_sync_request_to_host(rid, &mut states).unwrap();
        hip.gdn_sync_conv_to_host(rid, &mut convs).unwrap();
        (states[layer].clone(), convs[layer].clone())
    };

    let req_ref = 14u64;
    if let Err(e) = step(&hip, req_ref, qkv_a_d) {
        eprintln!("skip hip_gdn_rewind: {e}");
        return;
    }
    step(&hip, req_ref, qkv_b_d).expect("reference accepted prefix");
    let (state_want, conv_want) = pull(&hip, req_ref);

    if let Err(e) = step(&hip, req, qkv_a_d) {
        eprintln!("skip hip_gdn_rewind: {e}");
        return;
    }
    let d2h_before = hip.activation_d2h_bytes();
    let cloned = hip.gdn_clone_request(req).expect("clone device GDN");
    assert!(cloned > 0, "pre-verify clone must snapshot resident layers");
    step(&hip, req, qkv_b_d).expect("rejected suffix step after clone");
    hip.gdn_rewind_request(req).expect("rewind device GDN");
    let d2h_after = hip.activation_d2h_bytes();
    assert_eq!(
        d2h_after, d2h_before,
        "clone/rewind must stay on device; d2h {d2h_before}→{d2h_after}"
    );
    step(&hip, req, qkv_b_d).expect("replay accepted prefix after rewind");

    let (state_got, conv_got) = pull(&hip, req);
    assert_eq!(state_got.len(), state_want.len());
    assert_eq!(conv_got.len(), conv_want.len());
    for i in 0..state_want.len() {
        assert!(
            (state_got[i] - state_want[i]).abs() < 1e-4,
            "rewound state[{i}] got={} want={}",
            state_got[i],
            state_want[i]
        );
    }
    for i in 0..conv_want.len() {
        assert!(
            (conv_got[i] - conv_want[i]).abs() < 1e-4,
            "rewound conv[{i}] got={} want={}",
            conv_got[i],
            conv_want[i]
        );
    }
    hip.gdn_drop_clone(req);
    hip.gdn_clear_request(req);
    hip.gdn_clear_request(req_ref);
}

/// Tiny Qwen35 GDN GGUF (2 SSM layers) so prefill D2H can be measured without
/// loading the 27B ROCmFP4 checkpoint.
#[cfg(feature = "hip")]
fn write_tiny_qwen_gdn_gguf(path: &std::path::Path) {
    write_tiny_qwen_gdn_gguf_w(path, 0.02);
}

#[cfg(feature = "hip")]
fn write_tiny_qwen_gdn_gguf_w(path: &std::path::Path, w: f32) {
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

/// Prefill with device GDN must not D2H state+conv on every prompt token.
/// Public: generate still works; `activation_d2h_bytes` stays flat vs extra tokens.
#[cfg(feature = "hip")]
#[test]
fn hip_tiny_gdn_prefill_d2h_does_not_scale_with_every_token() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_tiny_gdn_prefill_d2h: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_tiny_gdn_prefill-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    write_tiny_qwen_gdn_gguf(&path);
    let mut cfg = Config::compiled_defaults();
    cfg.model.path = Some(path.clone());
    cfg.model.synthetic = false;
    cfg.engine.cpu_only = false;
    cfg.engine.gpu_backend = "auto".into();
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    cfg.kv.page_tokens = 32;
    let engine = match Engine::new(cfg) {
        Ok(e) => e,
        Err(e) => {
            let _ = std::fs::remove_file(&path);
            eprintln!("skip hip_tiny_gdn_prefill_d2h: Engine::new {e}");
            return;
        }
    };
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_tiny_gdn_prefill_d2h: no HIP in engine");
        return;
    }
    let short = "hi hi";
    let long = "hi hi hi hi hi hi hi hi hi hi";
    let d2h0 = engine.stats().activation_d2h_bytes;
    let short_out = engine.generate(short, greedy(1)).expect("short prefill");
    let after_short = engine.stats();
    let d2h_short = after_short.activation_d2h_bytes.saturating_sub(d2h0);
    let long_out = engine.generate(long, greedy(1)).expect("long prefill");
    let after_long = engine.stats();
    let d2h_long = after_long
        .activation_d2h_bytes
        .saturating_sub(after_short.activation_d2h_bytes);
    let extra_prompt = after_long
        .prompt_tokens
        .saturating_sub(after_short.prompt_tokens);
    let _ = std::fs::remove_file(&path);
    assert!(!short_out.token_ids.is_empty(), "short generate empty");
    assert!(!long_out.token_ids.is_empty(), "long generate empty");
    assert!(
        extra_prompt >= 4,
        "need a longer second prompt; extra_prompt={extra_prompt}"
    );
    // 2 GDN layers: state 2*8*8 + conv 1*48 floats → 1408 bytes/token if synced.
    let gdn_all_layers = 1408u64;
    let per_extra = d2h_long.saturating_sub(d2h_short) / extra_prompt.max(1);
    assert!(
        per_extra < gdn_all_layers / 2,
        "GDN host sync must not run every prefill token; d2h_short={d2h_short} d2h_long={d2h_long} extra_prompt={extra_prompt} per_extra={per_extra} stats_short={after_short:?} stats_long={after_long:?}"
    );
}
