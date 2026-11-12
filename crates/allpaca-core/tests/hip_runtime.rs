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

/// gemma4 keeps its K/V in its sequence state, which the prompt cache does
/// not store: on the CPU path (where prefill captures checkpoints) a repeat
/// prompt must never hit, and must give the same tokens.
#[test]
fn gemma4_never_hits_the_prompt_cache() {
    let path = std::env::temp_dir().join(format!("allpaca_g4_cpu_cache_{}.gguf", std::process::id()));
    allpaca_core::write_tiny_gemma4_gguf(&path, 2);
    let mut cfg = Config::compiled_defaults();
    cfg.model.path = Some(path);
    cfg.model.synthetic = false;
    cfg.engine.cpu_only = true;
    cfg.engine.determinism_seed = Some(7);
    cfg.speculative.enabled = false;
    let engine = Engine::new(cfg).expect("tiny gemma4 must load on the CPU");
    let prompt = "t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12";
    let first = engine.generate(prompt, greedy(8)).unwrap().token_ids;
    let again = engine.generate(prompt, greedy(8)).unwrap().token_ids;
    let longer = engine.generate(&format!("{prompt} t13 t14"), greedy(8)).unwrap().token_ids;
    // A one-token prompt takes the per-token prefill, which captures checkpoints.
    let short = engine.generate("t3", greedy(4)).unwrap().token_ids;
    let after = engine.generate("t3 t4 t5", greedy(4)).unwrap().token_ids;
    assert_eq!((short.len(), after.len()), (4, 4));
    assert_eq!(first, again);
    assert_eq!(longer.len(), 8);
    let stats = engine.stats();
    assert_eq!((stats.radix_hits, stats.anchor_hits), (0, 0), "gemma4 must not resume from the prompt cache: {stats:?}");
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
        if stats.p2p_gbps > 0.0 {
            assert_eq!(stats.p2p_status, "measured");
            assert!(stats.p2p_us > 0.0);
            assert!(
                stats.p2p_capable,
                "a working hipMemcpyPeer must mark P2P capable (EnablePeerAccess may still fail on gfx906)"
            );
        } else {
            assert_eq!(
                stats.p2p_status, "p2p_unavailable",
                "failed copy must not pretend P2P was measured"
            );
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
    // `device_memory_bytes` is process-wide and spans both boards, and the
    // driver reading is up to one sample interval old: other tests in this
    // binary allocate meanwhile, so allow the sampler to catch up.
    let covered = |s: &allpaca_types::StatsSnapshot| {
        s.gpu_vram_used_bytes + s.draft_gpu_vram_used_bytes >= s.device_memory_bytes
    };
    let mut seen = s.clone();
    for _ in 0..40 {
        if covered(&seen) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
        seen = engine.stats();
    }
    assert!(
        covered(&seen),
        "board usage must cover what this process allocated; used={}+{} owned={}",
        seen.gpu_vram_used_bytes,
        seen.draft_gpu_vram_used_bytes,
        seen.device_memory_bytes
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
    // Two sampled forwards (1 and 17): one tiny-kernel sample can read 0
    // when the GPU is shared with parallel tests.
    let out = engine.generate("hello", greedy(20)).unwrap();
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
    // Wall/token includes load and prefill: only a fallback when no ITL was
    // measured.
    if itl_us > 0.0 {
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

/// Two-layer F16 llama with varied weights for the gfx12 WMMA tests:
/// head_dim 64 (a WMMA attention shape), GQA 4:1, F16 weights,
/// and 40 words so a prompt gives a wide prefill.
#[cfg(feature = "hip")]
fn write_varied_f16_llama(path: &std::path::Path) {
    use allpaca_gguf::{write_gguf_quant, GgufValue, GgufWriteQuantTensor, GgufWriteTensor};
    use std::collections::HashMap;

    let (dim, n_heads, n_kv_heads, ff, vocab, layers) = (256usize, 4usize, 1usize, 512usize, 40usize, 2usize);
    let kv_dim = dim / n_heads * n_kv_heads;
    let mut seed = 0x5eed_u64;
    let mut values = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * scale
            })
            .collect()
    };
    let mut meta = HashMap::new();
    meta.insert("general.architecture".into(), GgufValue::String("llama".into()));
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
    meta.insert("llama.block_count".into(), GgufValue::Uint32(layers as u32));
    meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(n_heads as u32));
    meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(n_kv_heads as u32));
    meta.insert("llama.feed_forward_length".into(), GgufValue::Uint32(ff as u32));
    meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
    let mut tokens = vec![
        GgufValue::String("<unk>".into()),
        GgufValue::String("<s>".into()),
        GgufValue::String("</s>".into()),
    ];
    tokens.extend((3..vocab).map(|i| GgufValue::String(format!("w{i}"))));
    meta.insert("tokenizer.ggml.tokens".into(), GgufValue::Array(tokens));
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
    let f16 = allpaca_gguf::GgmlType::F16;
    let w = 2.0 / (dim as f32).sqrt();
    let mut quant: Vec<(String, Vec<u64>, Vec<u8>)> = vec![(
        "token_embd.weight".into(),
        vec![vocab as u64, dim as u64],
        allpaca_gguf::quantize_f16(&values(vocab * dim, 2.0)),
    )];
    for l in 0..layers {
        for (name, rows, cols) in [
            ("attn_q", dim, dim),
            ("attn_k", kv_dim, dim),
            ("attn_v", kv_dim, dim),
            ("attn_output", dim, dim),
            ("ffn_gate", ff, dim),
            ("ffn_up", ff, dim),
            ("ffn_down", dim, ff),
        ] {
            quant.push((
                format!("blk.{l}.{name}.weight"),
                vec![rows as u64, cols as u64],
                allpaca_gguf::quantize_f16(&values(rows * cols, w)),
            ));
        }
    }
    let ones = vec![1.0f32; dim];
    let mut plain: Vec<GgufWriteTensor> = vec![GgufWriteTensor {
        name: "output_norm.weight".into(),
        dims: vec![dim as u64],
        data: &ones,
    }];
    for l in 0..layers {
        for norm in ["attn_norm", "ffn_norm"] {
            plain.push(GgufWriteTensor {
                name: format!("blk.{l}.{norm}.weight"),
                dims: vec![dim as u64],
                data: &ones,
            });
        }
    }
    let quant_refs: Vec<GgufWriteQuantTensor> = quant
        .iter()
        .map(|(name, dims, bytes)| GgufWriteQuantTensor {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: f16,
            bytes,
        })
        .collect();
    write_gguf_quant(path, &meta, &plain, &quant_refs).unwrap();
}

/// gfx12 WMMA tests read the process-wide WMMA launch counter; run them one
/// at a time so one test's launches don't land in another's window.
#[cfg(feature = "hip")]
static WMMA_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "hip")]
fn wmma_prompt() -> String {
    (0..40).map(|i| format!("w{}", 3 + (i * 7) % 37)).collect::<Vec<_>>().join(" ")
}

/// Engine config for the WMMA tests: the varied F16 llama, optionally with
/// itself as the draft model (packed linear verify spans of several rows).
#[cfg(feature = "hip")]
fn wmma_engine_config(dir: &std::path::Path, spec: bool, row_variant: bool) -> Config {
    let target = dir.join("wmma_target.gguf");
    let draft = dir.join("wmma_draft.gguf");
    if !target.exists() {
        write_varied_f16_llama(&target);
        write_varied_f16_llama(&draft);
    }
    let mut cfg = hip_7b_config(target);
    cfg.kv.page_tokens = 16;
    cfg.kv.vram_bytes = 64 * 1024 * 1024;
    cfg.kv.dram_bytes = 64 * 1024 * 1024;
    cfg.experimental.row_variant_kernels = row_variant;
    if spec {
        cfg.model.draft_path = Some(draft);
        cfg.speculative.enabled = true;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 4;
    }
    cfg
}

#[cfg(feature = "hip")]
fn wmma_test_dir(tag: &str) -> std::path::PathBuf {
    // Two folders below temp: sidecar discovery reads the model's folder and
    // the files of the one above (see hip-test-flakes).
    let dir = std::env::temp_dir()
        .join(format!("allpaca_wmma_{tag}_{}", std::process::id()))
        .join("model");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs `generate` on a fresh engine; returns (tokens, WMMA launches, stats).
#[cfg(feature = "hip")]
fn wmma_generate(cfg: Config, max_tokens: usize) -> Option<(Vec<u32>, u64, allpaca_types::StatsSnapshot)> {
    let engine = Engine::new(cfg).expect("varied F16 llama must load");
    if hip_gpus(&engine).is_empty() {
        return None;
    }
    let before = allpaca_hip::wmma_launches();
    let out = engine.generate(&wmma_prompt(), greedy(max_tokens)).expect("generate");
    let launches = allpaca_hip::wmma_launches() - before;
    Some((out.token_ids, launches, engine.stats()))
}

#[cfg(feature = "hip")]
fn any_gfx12() -> bool {
    (0..allpaca_hip::device_count()).any(allpaca_hip::device_is_gfx12)
}

/// With `experimental.row_variant_kernels` off (the default) no WMMA kernel
/// runs at all, in prefill, decode or packed verify, and speculative output
/// equals plain greedy output.
#[cfg(feature = "hip")]
#[test]
fn hip_gfx12_wmma_off_by_default_never_launches_and_spec_matches_plain() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip: no HIP device");
        return;
    }
    let _lock = WMMA_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = wmma_test_dir("off");
    assert!(!Config::compiled_defaults().experimental.row_variant_kernels);
    let Some((plain, plain_wmma, _)) = wmma_generate(wmma_engine_config(&dir, false, false), 24) else {
        return;
    };
    let (spec, spec_wmma, stats) = wmma_generate(wmma_engine_config(&dir, true, false), 24).unwrap();
    assert!(stats.linear_packed_verifies > 0, "the draft must pack verify spans; stats={stats:?}");
    assert_eq!(plain_wmma, 0, "WMMA ran with the option off (plain)");
    assert_eq!(spec_wmma, 0, "WMMA ran with the option off (spec)");
    assert_eq!(spec, plain, "spec != no-spec with the option off");
    let _ = std::fs::remove_dir_all(dir.parent().unwrap());
}

/// With the option on, WMMA runs in prefill (on gfx12) and nowhere else:
/// a 24-token speculative generate launches exactly as many WMMA kernels as
/// a 1-token one with the same prompt (decode and verify add none), and
/// speculative output still equals plain output with the option on.
#[cfg(feature = "hip")]
#[test]
fn hip_gfx12_wmma_on_runs_in_prefill_only() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip: no HIP device");
        return;
    }
    let _lock = WMMA_COUNTER_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = wmma_test_dir("on");
    let Some((plain, _, _)) = wmma_generate(wmma_engine_config(&dir, false, true), 24) else {
        return;
    };
    let (_, prefill_only, _) = wmma_generate(wmma_engine_config(&dir, true, true), 1).unwrap();
    let (spec, spec_wmma, stats) = wmma_generate(wmma_engine_config(&dir, true, true), 24).unwrap();
    assert!(stats.linear_packed_verifies > 0, "the draft must pack verify spans; stats={stats:?}");
    if any_gfx12() {
        assert!(prefill_only > 0, "the option is on but prefill launched no WMMA");
    }
    assert_eq!(spec_wmma, prefill_only, "decode or verify launched WMMA kernels");
    assert_eq!(spec, plain, "spec != no-spec with the option on");
    let _ = std::fs::remove_dir_all(dir.parent().unwrap());
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
fn write_tiny_f32_two_layer_gguf(path: &std::path::Path) {
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
    meta.insert("llama.block_count".into(), GgufValue::Uint32(2));
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
    let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![(
        "token_embd.weight".into(),
        vec![vocab as u64, dim as u64],
        vec![0.01; vocab * dim],
    )];
    for blk in 0..2u32 {
        let p = format!("blk.{blk}");
        owned.extend([
            (
                format!("{p}.attn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                format!("{p}.attn_q.weight"),
                vec![dim as u64, dim as u64],
                vec![0.01; dim * dim],
            ),
            (
                format!("{p}.attn_k.weight"),
                vec![2, dim as u64],
                vec![0.01; 2 * dim],
            ),
            (
                format!("{p}.attn_v.weight"),
                vec![2, dim as u64],
                vec![0.01; 2 * dim],
            ),
            (
                format!("{p}.attn_output.weight"),
                vec![dim as u64, dim as u64],
                vec![0.01; dim * dim],
            ),
            (
                format!("{p}.ffn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                format!("{p}.ffn_gate.weight"),
                vec![ff as u64, dim as u64],
                vec![0.01; ff * dim],
            ),
            (
                format!("{p}.ffn_up.weight"),
                vec![ff as u64, dim as u64],
                vec![0.01; ff * dim],
            ),
            (
                format!("{p}.ffn_down.weight"),
                vec![dim as u64, ff as u64],
                vec![0.01; dim * ff],
            ),
        ]);
    }
    owned.push((
        "output_norm.weight".into(),
        vec![dim as u64],
        vec![1.0; dim],
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

/// Operator expert controls on a real MoE load, CPU and HIP: recording
/// counts routing hits, a disabled expert gets none, top-k must stay
/// satisfiable, and clearing the mask restores the original tokens.
#[cfg(feature = "hip")]
#[test]
fn expert_controls_disable_experts_and_count_usage() {
    let path = std::env::temp_dir().join("allpaca_expert_controls_moe.gguf");
    write_tiny_moe_gguf(&path);
    for cpu_only in [true, false] {
        let mut cfg = if cpu_only { cpu_only_synthetic() } else { hip_auto_synthetic() };
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        let engine = Engine::new(cfg).unwrap();
        if !cpu_only && hip_gpus(&engine).is_empty() {
            continue;
        }
        let ctl = engine.expert_controls().expect("model loaded");
        let shape = ctl.shape();
        assert_eq!((shape.n_layers, shape.n_experts, shape.n_active), (1, 4, 2));
        ctl.set_recording(true);
        let base = engine.generate("hello", greedy(6)).unwrap().token_ids;
        let usage = engine.expert_usage().unwrap();
        assert_eq!(usage.len(), 4);
        assert!(usage.iter().sum::<u64>() > 0, "cpu_only={cpu_only}: no routing hits counted");
        let hot = (0..4).max_by_key(|&e| usage[e]).unwrap();

        assert!(ctl.set_disabled(&[vec![0, 1, 2]]).is_err(), "must keep top-k enabled");
        ctl.set_disabled(&[vec![hot as u32]]).unwrap();
        ctl.reset_usage();
        engine.generate("hello", greedy(6)).unwrap();
        let usage = engine.expert_usage().unwrap();
        assert_eq!(usage[hot], 0, "cpu_only={cpu_only}: disabled expert {hot} was routed: {usage:?}");
        assert!(usage.iter().sum::<u64>() > 0);

        ctl.set_disabled(&[]).unwrap();
        let again = engine.generate("hello", greedy(6)).unwrap().token_ids;
        assert_eq!(again, base, "cpu_only={cpu_only}: clearing the mask must restore the output");
    }
    let _ = std::fs::remove_file(path);
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

/// Tiny Mixtral-style MoE whose routed experts are MoESQ paired-4:8 FP4
/// (tensor type 200); everything else F32. Weights are seeded noise so tokens
/// and experts differ.
#[cfg(feature = "hip")]
fn write_tiny_p48_moe_gguf(path: &std::path::Path, dim: usize, ff: usize) {
    use allpaca_gguf::{write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor};
    use std::collections::HashMap;

    let (vocab, n_exp) = (4usize, 4usize);
    let mut meta = HashMap::new();
    meta.insert("general.architecture".into(), GgufValue::String("llama".into()));
    for (key, value) in [
        ("vocab_size", vocab),
        ("embedding_length", dim),
        ("block_count", 1),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 1),
        ("feed_forward_length", ff),
        ("expert_count", n_exp),
        ("expert_used_count", 2),
    ] {
        meta.insert(format!("llama.{key}"), GgufValue::Uint32(value as u32));
    }
    meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(
            ["<unk>", "<s>", "</s>", "hello"].into_iter().map(|t| GgufValue::String(t.into())).collect(),
        ),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
    let mut seed = 0x2545_f491u32;
    let mut noise = |n: usize, amp: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amp
            })
            .collect()
    };
    let head = dim / 4;
    let owned: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
        ("token_embd.weight", vec![vocab as u64, dim as u64], noise(vocab * dim, 1.0)),
        ("blk.0.attn_norm.weight", vec![dim as u64], vec![1.0; dim]),
        ("blk.0.attn_q.weight", vec![dim as u64, dim as u64], noise(dim * dim, 0.3)),
        ("blk.0.attn_k.weight", vec![head as u64, dim as u64], noise(head * dim, 0.3)),
        ("blk.0.attn_v.weight", vec![head as u64, dim as u64], noise(head * dim, 0.3)),
        ("blk.0.attn_output.weight", vec![dim as u64, dim as u64], noise(dim * dim, 0.3)),
        ("blk.0.ffn_norm.weight", vec![dim as u64], vec![1.0; dim]),
        ("blk.0.ffn_gate_inp.weight", vec![n_exp as u64, dim as u64], noise(n_exp * dim, 0.5)),
        ("output_norm.weight", vec![dim as u64], vec![1.0; dim]),
        ("output.weight", vec![vocab as u64, dim as u64], noise(vocab * dim, 0.5)),
    ];
    let experts: Vec<(&str, Vec<u64>, Vec<u8>)> = vec![
        (
            "blk.0.ffn_gate_exps.weight",
            vec![n_exp as u64, ff as u64, dim as u64],
            allpaca_cpu::quantize_p48fp4(&noise(n_exp * ff * dim, 0.4)),
        ),
        (
            "blk.0.ffn_up_exps.weight",
            vec![n_exp as u64, ff as u64, dim as u64],
            allpaca_cpu::quantize_p48fp4(&noise(n_exp * ff * dim, 0.4)),
        ),
        (
            "blk.0.ffn_down_exps.weight",
            vec![n_exp as u64, dim as u64, ff as u64],
            allpaca_cpu::quantize_p48fp4(&noise(n_exp * dim * ff, 0.4)),
        ),
    ];
    let f32_views: Vec<GgufWriteTensor<'_>> = owned
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: (*name).into(),
            dims: dims.clone(),
            data,
        })
        .collect();
    let quant_views: Vec<GgufWriteQuantTensor<'_>> = experts
        .iter()
        .map(|(name, dims, bytes)| GgufWriteQuantTensor {
            name: (*name).into(),
            dims: dims.clone(),
            ggml_type: GgmlType::P48Fp4,
            bytes,
        })
        .collect();
    write_gguf_quant(path, &meta, &f32_views, &quant_views).unwrap();
}

/// At widths in whole 256-value blocks the experts go through the fused
/// expert-parallel kernels (int8 activations): same tokens as the CPU.
#[cfg(feature = "hip")]
#[test]
fn hip_p48fp4_experts_fused_match_cpu() {
    let path = std::env::temp_dir().join(format!("allpaca_hip_p48_moe_fused-{}.gguf", std::process::id()));
    write_tiny_p48_moe_gguf(&path, 256, 256);
    let with_model = |mut cfg: Config| {
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg
    };
    let cpu = Engine::new(with_model(cpu_only_synthetic())).unwrap();
    let hip = Engine::new(with_model(hip_auto_synthetic())).unwrap();
    if hip_gpus(&hip).is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    for prompt in ["hello", "hello hello hello hello hello hello"] {
        let cpu_out = cpu.generate(prompt, greedy(6)).unwrap();
        let hip_out = hip.generate(prompt, greedy(6)).unwrap();
        assert!(!hip_out.token_ids.is_empty());
        assert_eq!(hip_out.token_ids, cpu_out.token_ids, "prompt {prompt:?}");
    }
    assert!(hip.stats().gpu_forwards > 0, "must run on HIP: {:?}", hip.stats());
    let _ = std::fs::remove_file(path);
}

/// MoESQ experts run from their sparse blocks on the GPU and give the tokens
/// the CPU gets from the same weights decoded to F32.
#[cfg(feature = "hip")]
#[test]
fn hip_p48fp4_experts_match_cpu() {
    let path = std::env::temp_dir().join(format!("allpaca_hip_p48_moe-{}.gguf", std::process::id()));
    write_tiny_p48_moe_gguf(&path, 32, 32);
    let with_model = |mut cfg: Config| {
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg
    };
    let cpu = Engine::new(with_model(cpu_only_synthetic())).unwrap();
    let hip = Engine::new(with_model(hip_auto_synthetic())).unwrap();
    if hip_gpus(&hip).is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    for prompt in ["hello", "hello hello hello hello hello hello"] {
        let cpu_out = cpu.generate(prompt, greedy(6)).unwrap();
        let hip_out = hip.generate(prompt, greedy(6)).unwrap();
        assert!(!hip_out.token_ids.is_empty());
        assert_eq!(hip_out.token_ids, cpu_out.token_ids, "prompt {prompt:?}");
    }
    assert!(hip.stats().gpu_forwards > 0, "must run on HIP: {:?}", hip.stats());
    let _ = std::fs::remove_file(path);
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

/// Layer 0 on HIP:0, layer 1 on HIP:1. Logits must land on the target workspace
/// Engine waits on — a peer-GPU lm_head used to emit `!` tokens.
#[cfg(feature = "hip")]
#[test]
fn hip_local_gpu_pool_two_layer_matches_cpu() {
    if allpaca_hip::device_count() < 2 {
        eprintln!("skip hip_local_gpu_pool_two_layer_matches_cpu: need 2 HIP devices");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_hip_tiny_two_layer_pool.gguf");
    write_tiny_f32_two_layer_gguf(&path);
    let cpu = Engine::new(tiny_written_cfg(path.clone(), true)).expect("two-layer CPU load");
    // The byte-aware planner keeps a model this small on the target GPU; an
    // even split forces one layer onto the peer so the hand-off is exercised.
    let mut hip_cfg = tiny_written_cfg(path.clone(), false);
    hip_cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
    let hip = Engine::new(hip_cfg).expect("two-layer HIP load");
    if hip_gpus(&hip).len() < 2 {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_local_gpu_pool_two_layer_matches_cpu: inventory has <2 HIP GPUs");
        return;
    }
    let cpu_out = cpu
        .generate("hello", greedy(4))
        .unwrap_or_else(|e| panic!("two-layer CPU generate: {e}"));
    let hip_out = hip
        .generate("hello", greedy(4))
        .unwrap_or_else(|e| panic!("two-layer HIP generate: {e}"));
    assert_eq!(
        hip_out.token_ids, cpu_out.token_ids,
        "local GPU pool must match CPU; hip={:?} cpu={:?} text={:?}",
        hip_out.token_ids, cpu_out.token_ids, hip_out.text
    );
    let stats = hip.stats();
    assert!(
        stats.draft_gpu_vram_used_bytes > 0 && stats.draft_gpu_vram_total_bytes > 0,
        "second local GPU must report VRAM after a pool split; stats={stats:?}"
    );
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

/// Shared ROCmFP4 quality prompt (Qwen, Nemotron, …). Override with `ALLPACA_QWEN_PROMPT`.
#[cfg(feature = "hip")]
const ROCMFP4_FACT_PROMPT: &str =
    "<|im_start|>user\nWhat is the capitol of Western Australia<|im_end|>\n<|im_start|>assistant\n";

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

/// Optional 9060 soak GGUF: Q8 for better ITL unless `ALLPACA_OVERNIGHT_9060_Q8=0`.
#[cfg(feature = "hip")]
fn overnight_9060_prefer_q8() -> bool {
    std::env::var("ALLPACA_OVERNIGHT_9060_Q8").ok().as_deref() != Some("0")
}

#[cfg(feature = "hip")]
fn overnight_9060_is_q8(path: &std::path::Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.contains("Q8"))
}

/// Resolve 9060 soak weights: prefer Llama-3.2 Q8_0, else 3B F16.
#[cfg(feature = "hip")]
fn overnight_9060_gguf_with(prefer_q8: bool) -> Option<std::path::PathBuf> {
    let dir = local_models_dir();
    if prefer_q8 {
        for name in [
            "Llama-3.2-3B-Instruct-Q8_0.gguf",
            "Llama-3.2-1B-Instruct-Q8_0.gguf",
        ] {
            let p = dir.join(name);
            if p.is_file() && looks_like_gguf(&p) {
                return Some(p);
            }
        }
    }
    let f16 = dir.join("Llama-3.2-3B-Instruct-f16.gguf");
    if f16.is_file() && looks_like_gguf(&f16) {
        return Some(f16);
    }
    None
}

#[cfg(feature = "hip")]
fn overnight_9060_gguf() -> Option<std::path::PathBuf> {
    overnight_9060_gguf_with(overnight_9060_prefer_q8())
}

/// 9060 soak prefers 3B Q8 over F16 when both are present.
#[cfg(feature = "hip")]
#[test]
fn overnight_9060_gguf_prefers_q8_over_f16_when_available() {
    let q8 = local_models_dir().join("Llama-3.2-3B-Instruct-Q8_0.gguf");
    let f16 = local_models_dir().join("Llama-3.2-3B-Instruct-f16.gguf");
    if !q8.is_file() {
        eprintln!(
            "skip overnight_9060_gguf_prefers_q8_over_f16_when_available: missing {}",
            q8.display()
        );
        return;
    }
    if !f16.is_file() {
        eprintln!(
            "skip overnight_9060_gguf_prefers_q8_over_f16_when_available: missing {}",
            f16.display()
        );
        return;
    }
    let found = overnight_9060_gguf_with(true).expect("Q8 must win when prefer_q8");
    assert_eq!(
        found.canonicalize().unwrap_or(found.clone()),
        q8.canonicalize().unwrap_or(q8.clone()),
        "9060 soak must prefer 3B Q8 over F16; found={}",
        found.display()
    );
    assert!(overnight_9060_is_q8(&found));
}

/// `ALLPACA_OVERNIGHT_9060_Q8=0` (prefer_q8=false) keeps the F16 soak path.
#[cfg(feature = "hip")]
#[test]
fn overnight_9060_gguf_falls_back_to_f16_when_q8_disabled() {
    let f16 = local_models_dir().join("Llama-3.2-3B-Instruct-f16.gguf");
    if !f16.is_file() {
        eprintln!(
            "skip overnight_9060_gguf_falls_back_to_f16_when_q8_disabled: missing {}",
            f16.display()
        );
        return;
    }
    let found = overnight_9060_gguf_with(false).expect("F16 fallback must resolve");
    assert_eq!(
        found.canonicalize().unwrap_or(found.clone()),
        f16.canonicalize().unwrap_or(f16.clone()),
        "prefer_q8=false must select 3B F16; found={}",
        found.display()
    );
    assert!(!overnight_9060_is_q8(&found));
}

/// Short Q8 smoke on HIP (prefers HIP:1): asserts `q8_gemm_us > 0` after a few
/// tokens. Prefers 1B Q8 when present (faster load). Skips without a local
/// Llama-3.2 Q8 GGUF or HIP device. Not overnight.
#[cfg(feature = "hip")]
#[test]
fn hip_llama32_q8_greedy_reports_q8_gemm_when_available() {
    let _guard = large_gguf_lock();
    let dir = local_models_dir();
    let path = [
        "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "Llama-3.2-3B-Instruct-Q8_0.gguf",
    ]
    .iter()
    .map(|name| dir.join(name))
    .find(|p| p.is_file() && looks_like_gguf(p));
    let Some(path) = path else {
        eprintln!(
            "skip hip_llama32_q8_greedy_reports_q8_gemm_when_available: no Llama-3.2 Q8 under models/"
        );
        return;
    };
    let n_dev = allpaca_hip::device_count();
    if n_dev == 0 {
        eprintln!("skip hip_llama32_q8_greedy_reports_q8_gemm_when_available: no HIP device");
        return;
    }
    let device = if n_dev >= 2 { 1 } else { 0 };
    let mut cfg = hip_7b_config(path.clone());
    cfg.engine.target_device = DeviceSelector::Hip(device);
    cfg.engine.draft_device = DeviceSelector::Cpu;
    cfg.speculative.enabled = false;
    cfg.features.draft_model_speculation = Some(false);
    cfg.kv.vram_bytes = 512 * 1024 * 1024;
    cfg.kv.dram_bytes = 512 * 1024 * 1024;
    eprintln!(
        "q8 smoke Engine::new path={} device={}",
        path.display(),
        device
    );
    let engine = Engine::new(cfg).expect("Llama-3.2 Q8 must load on HIP");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_llama32_q8_greedy_reports_q8_gemm_when_available: no HIP device");
        return;
    }
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.target_device, device as i32,
        "Q8 smoke must place Target Model on HIP:{device}"
    );
    const N: usize = 4;
    let out = engine
        .generate("hello", greedy(N))
        .expect("Llama-3.2 Q8 HIP generate must produce tokens");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    eprintln!(
        "q8 smoke gen={} q8_gemm_us={:.0} attention_us={:.0} lm_head_us={:.0} itl_ms={:.2}",
        out.token_ids.len(),
        stats.q8_gemm_us,
        stats.attention_us,
        stats.lm_head_us,
        stats.itl_ms_ema
    );
    assert!(
        stats.q8_gemm_us > 0.0,
        "Q8 greedy must time Q8 GEMV; q8_gemm_us={} stats={stats:?}",
        stats.q8_gemm_us
    );
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
/// 2. Optional 9060 soak on HIP:1 (`ALLPACA_OVERNIGHT_9060=1`): prefers
///    Llama-3.2 Q8_0 (3B then 1B) for better ITL; F16 only if Q8 missing or
///    `ALLPACA_OVERNIGHT_9060_Q8=0`
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
        if let Some(soak_path) = overnight_9060_gguf() {
            let soak_is_q8 = overnight_9060_is_q8(&soak_path);
            let mut cfg = hip_7b_config(soak_path.clone());
            cfg.engine.target_device = DeviceSelector::Hip(1);
            cfg.engine.draft_device = DeviceSelector::Cpu;
            cfg.speculative.enabled = false;
            cfg.features.draft_model_speculation = Some(false);
            cfg.kv.vram_bytes = 512 * 1024 * 1024;
            cfg.kv.dram_bytes = 512 * 1024 * 1024;
            eprintln!(
                "overnight 9060 Engine::new starting path={} q8={} tokens={tokens} rounds={rounds}",
                soak_path.display(),
                soak_is_q8
            );
            let engine = Engine::new(cfg).expect("9060 soak GGUF must load on HIP:1");
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
                assert!(
                    !out.token_ids.is_empty(),
                    "overnight 9060 round {} must emit tokens",
                    round + 1
                );
                if soak_is_q8 {
                    assert!(
                        stats.q8_gemm_us > 0.0,
                        "9060 Q8 soak must time Q8 GEMV; q8_gemm_us={} stats={stats:?}",
                        stats.q8_gemm_us
                    );
                }
            }
            drop(engine);
        } else {
            eprintln!("overnight skip 9060 soak: no Llama-3.2 Q8/F16 GGUF under models/");
        }
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
        assert!(
            stats.q8_gemm_us > 0.0,
            "overnight draft-q8 must time Q8 GEMV; q8_gemm_us={} stats={stats:?}",
            stats.q8_gemm_us
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
    // A real multi-GB GGUF reads all its weights per token: not sub-ms.
    assert!(
        itl_us > 1_000.0,
        "real GGUF decode ITL should be milliseconds; itl_us={itl_us:.1} stats={stats:?}"
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

/// Gated BitNet / ternary GGUF. Uses `ALLPACA_TQ_GGUF` or `ALLPACA_BITNET_GGUF`.
fn gated_tq_gguf() -> Option<std::path::PathBuf> {
    for key in ["ALLPACA_TQ_GGUF", "ALLPACA_BITNET_GGUF"] {
        if let Ok(raw) = std::env::var(key) {
            let p = std::path::PathBuf::from(raw.trim());
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn write_tiny_tq2_gguf(path: &std::path::Path) {
    use allpaca_gguf::{
        quantize_tq2_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
        GgufWriteTensor,
    };
    use std::collections::HashMap;

    // dim*ff > 65536 so FFN stays packed TQ2_0 and exercises fused HIP GEMV.
    let dim = 256usize;
    let vocab = 4usize;
    let ff = 512usize;
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
    let embd_q = quantize_tq2_0(&vec![0.01f32; vocab * dim]);
    let q_q = quantize_tq2_0(&vec![0.01f32; dim * dim]);
    let k_q = quantize_tq2_0(&vec![0.01f32; (dim / 4) * dim]);
    let v_q = quantize_tq2_0(&vec![0.01f32; (dim / 4) * dim]);
    let o_q = quantize_tq2_0(&vec![0.01f32; dim * dim]);
    let gate_q = quantize_tq2_0(&vec![0.01f32; ff * dim]);
    let up_q = quantize_tq2_0(&vec![0.01f32; ff * dim]);
    let down_q = quantize_tq2_0(&vec![0.01f32; dim * ff]);
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
                ggml_type: GgmlType::TQ2_0,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![(dim / 4) as u64, dim as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![(dim / 4) as u64, dim as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::TQ2_0,
                bytes: &down_q,
            },
        ],
    )
    .expect("write tiny TQ2_0 GGUF");
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_tiny_tq2_gguf_uses_fused_path() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_tiny_tq2_gguf_uses_fused_path: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join("allpaca_tiny_tq2.gguf");
    write_tiny_tq2_gguf(&path);
    let engine = Engine::new(hip_7b_config(path.clone())).expect("TQ2_0 tiny load");
    if hip_gpus(&engine).is_empty() {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_generate_tiny_tq2_gguf_uses_fused_path: no HIP device");
        return;
    }
    let after_load = engine.stats();
    let h2d = after_load.weight_h2d_bytes;
    assert!(
        after_load.weight_bytes_on_device > 0,
        "TQ2 FFN should stay packed on device"
    );
    let out = engine
        .generate("hello", greedy(4))
        .expect("TQ2_0 HIP generate");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.weight_h2d_bytes, h2d,
        "decode must not H2D weights; load={h2d} after={}",
        stats.weight_h2d_bytes
    );
    eprintln!(
        "TQ2_0 tiny HIP generate: tokens={} text={:?}",
        out.token_ids.len(),
        out.text
    );
    let _ = std::fs::remove_file(path);
}

#[cfg(feature = "hip")]
#[test]
fn hip_generate_real_tq_gguf_when_path_set() {
    let _guard = large_gguf_lock();
    let Some(path) = gated_tq_gguf() else {
        eprintln!("skip hip_generate_real_tq_gguf_when_path_set: set ALLPACA_TQ_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_generate_real_tq_gguf_when_path_set: no HIP device");
        return;
    }
    let engine = Engine::new(hip_7b_config(path.clone())).expect("TQ GGUF load must not be Fatal");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_generate_real_tq_gguf_when_path_set: no HIP device");
        return;
    }
    let after_load = engine.stats();
    let h2d = after_load.weight_h2d_bytes;
    let out = engine
        .generate("hello", greedy(4))
        .expect("TQ HIP generate must not panic or OOM");
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert_eq!(stats.backend, "hip");
    assert_eq!(
        stats.weight_h2d_bytes, h2d,
        "decode must not H2D weights; load={h2d} after={}",
        stats.weight_h2d_bytes
    );
    eprintln!(
        "TQ HIP generate: path={} tokens={} backend={} text={:?}",
        path.display(),
        out.token_ids.len(),
        stats.backend,
        out.text
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
    eprintln!(
        "ROCmFP4 moe: experts={} active={} groups={} group_used={} shared={} shared_ff={} w_norm={} w_scale={} sigmoid={} relu_sqr={}",
        cfg.n_experts,
        cfg.n_active,
        cfg.n_expert_groups,
        cfg.n_group_used,
        cfg.n_shared_experts,
        cfg.shared_ff_dim,
        cfg.expert_weights_norm,
        cfg.expert_weights_scale,
        cfg.moe_sigmoid_gating,
        cfg.moe_relu_sqr
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
    // Qwen35 GDN uses attn_qkv + ssm_out; Nemotron Lightning uses ssm_in + ssm_out.
    {
        let mut gdn_ok = 0usize;
        let mut gdn_fail = 0usize;
        let mut mamba_ok = 0usize;
        let mut mamba_fail = 0usize;
        let mut attn_qnorm = 0usize;
        for i in 0..cfg.n_layers {
            let block = report.layers.get(i).map(|l| l.block);
            if block == Some(allpaca_gguf::HybridLayerKind::Ssm) {
                let qkv = file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some();
                let ssm_in = file.tensor(&format!("blk.{i}.ssm_in.weight")).is_some();
                let out_w = file.tensor(&format!("blk.{i}.ssm_out.weight")).is_some();
                let a = file.tensor(&format!("blk.{i}.ssm_a")).is_some();
                let conv = file.tensor(&format!("blk.{i}.ssm_conv1d.weight")).is_some();
                if qkv && out_w && a && conv {
                    gdn_ok += 1;
                } else if qkv || report.architecture == "qwen35" {
                    gdn_fail += 1;
                    if gdn_fail <= 2 {
                        eprintln!(
                            "GDN tensors missing blk.{i}: qkv={qkv} ssm_out={out_w} a={a} conv={conv}"
                        );
                    }
                }
                if ssm_in && out_w && a && conv {
                    mamba_ok += 1;
                } else if ssm_in || report.architecture == "nemotron_h_moe" {
                    mamba_fail += 1;
                    if mamba_fail <= 2 {
                        eprintln!(
                            "Mamba-2 tensors missing blk.{i}: ssm_in={ssm_in} ssm_out={out_w} a={a} conv={conv}"
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
            "ROCmFP4 hybrid tensors: arch={} gdn_ok={gdn_ok} gdn_fail={gdn_fail} mamba_ok={mamba_ok} mamba_fail={mamba_fail} attn_qnorm={attn_qnorm} nextn_meta={nextn:?} last_blocks={:?}",
            report.architecture,
            report
                .layers
                .iter()
                .rev()
                .take(3)
                .map(|l| (l.index, l.block))
                .collect::<Vec<_>>()
        );
        if report.architecture == "qwen35" {
            assert!(gdn_ok > 0, "expected GDN tensors on Qwen SSM layers");
            assert_eq!(gdn_fail, 0, "Qwen SSM layers missing GDN tensors");
        } else if report.architecture == "nemotron_h_moe" {
            assert!(
                mamba_ok > 0,
                "expected Mamba-2 tensors on Nemotron SSM layers"
            );
            assert_eq!(mamba_fail, 0, "Nemotron SSM layers missing Mamba-2 tensors");
        }
    }
    let prompt =
        std::env::var("ALLPACA_QWEN_PROMPT").unwrap_or_else(|_| ROCMFP4_FACT_PROMPT.into());
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
    let gen_started = std::time::Instant::now();
    let out = generate_or_timeout(&engine, &prompt, n_tok, overnight_timeout(n_tok, 180));
    let generate_elapsed = gen_started.elapsed();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    let toks = out.token_ids.len().max(1) as f64;
    let tps = toks / generate_elapsed.as_secs_f64().max(1e-6);
    eprintln!(
        "ROCmFP4 HIP generate: path={} prompt_tokens={} gen={} completion_tokens={} ttft_ms={} itl_ms={:.2} tokens/s={:.2} generate_ms={} gpu_forwards={} prefill_chunks={} tokens_generated={} backend={} q8_us={} attn_us={} other_us={} ids={:?} text={:?}",
        path.display(),
        out.prompt_tokens,
        out.token_ids.len(),
        out.completion_tokens,
        stats.ttft_ms_last,
        stats.itl_ms_ema,
        tps,
        generate_elapsed.as_millis(),
        stats.gpu_forwards,
        stats.prefill_chunks,
        stats.tokens_generated,
        stats.backend,
        stats.q8_gemm_us,
        stats.attention_us,
        stats.other_gpu_us,
        out.token_ids,
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
    let unique = out
        .token_ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        unique >= 4,
        "must not role-loop; unique={unique} ids={:?} text={:?}",
        out.token_ids,
        out.text
    );
    assert!(
        lower.contains("perth"),
        "must name Perth for Western Australia, got {:?}",
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

/// Warp-per-(head, v-row) kernel: head_k=32 fills a wave, head_v>1 is parallel.
#[cfg(feature = "hip")]
#[test]
fn hip_gated_delta_step_matches_host_warp_k32() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gated_delta_warp_k32: no HIP device");
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 4096, 8).expect("HipWorkspace");
    let n_v = 4usize;
    let head_k = 32usize;
    let head_v = 16usize;
    let mut q = vec![0.0f32; n_v * head_k];
    let mut k = vec![0.0f32; n_v * head_k];
    for i in 0..q.len() {
        q[i] = (i as f32).mul_add(0.01, 0.1);
        k[i] = (i as f32).mul_add(-0.02, 0.3);
    }
    let v: Vec<f32> = (0..n_v * head_v)
        .map(|i| (i as f32) * 0.05)
        .collect();
    let decay = vec![0.7f32; n_v];
    let beta = vec![0.4f32; n_v];
    let state0: Vec<f32> = (0..n_v * head_v * head_k)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let mut q_h = q.clone();
    let mut k_h = k.clone();
    let mut state_h = state0.clone();
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

    let floats = q.len() + k.len() + v.len() + decay.len() + beta.len() + state0.len() + out_h.len();
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
    let q_d = put(&hip, &blob, &mut off, &q);
    let k_d = put(&hip, &blob, &mut off, &k);
    let v_d = put(&hip, &blob, &mut off, &v);
    let decay_d = put(&hip, &blob, &mut off, &decay);
    let beta_d = put(&hip, &blob, &mut off, &beta);
    let state_d = put(&hip, &blob, &mut off, &state0);
    let out_d = put(&hip, &blob, &mut off, &out_h);

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
    .expect("device gated_delta_step warp k32");

    let mut out = vec![0.0f32; out_h.len()];
    let mut state = vec![0.0f32; state_h.len()];
    let mut q_dev = vec![0.0f32; q_h.len()];
    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    hip.copy_d2h_f32(state_d, &mut state).unwrap();
    hip.copy_d2h_f32(q_d, &mut q_dev).unwrap();
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
    for i in 0..q_dev.len() {
        assert!(
            (q_dev[i] - q_h[i]).abs() < 1e-4,
            "q[{i}] device={} host={}",
            q_dev[i],
            q_h[i]
        );
    }
}

/// Qwen3.5-27B GDN is head_k=128, head_v=128, 48 V-heads. Exercise the K loop.
#[cfg(feature = "hip")]
#[test]
fn hip_gated_delta_step_matches_host_warp_k128() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gated_delta_warp_k128: no HIP device");
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 4096, 8).expect("HipWorkspace");
    let n_v = 2usize;
    let head_k = 128usize;
    let head_v = 32usize;
    let mut q = vec![0.0f32; n_v * head_k];
    let mut k = vec![0.0f32; n_v * head_k];
    for i in 0..q.len() {
        q[i] = (i as f32).mul_add(0.01, 0.1);
        k[i] = (i as f32).mul_add(-0.02, 0.3);
    }
    let v: Vec<f32> = (0..n_v * head_v)
        .map(|i| (i as f32) * 0.05)
        .collect();
    let decay = vec![0.7f32; n_v];
    let beta = vec![0.4f32; n_v];
    let state0: Vec<f32> = (0..n_v * head_v * head_k)
        .map(|i| (i as f32) * 0.001)
        .collect();

    let mut q_h = q.clone();
    let mut k_h = k.clone();
    let mut state_h = state0.clone();
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

    let floats = q.len() + k.len() + v.len() + decay.len() + beta.len() + state0.len() + out_h.len();
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
    let q_d = put(&hip, &blob, &mut off, &q);
    let k_d = put(&hip, &blob, &mut off, &k);
    let v_d = put(&hip, &blob, &mut off, &v);
    let decay_d = put(&hip, &blob, &mut off, &decay);
    let beta_d = put(&hip, &blob, &mut off, &beta);
    let state_d = put(&hip, &blob, &mut off, &state0);
    let out_d = put(&hip, &blob, &mut off, &out_h);

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
    .expect("device gated_delta_step warp k128");

    let mut out = vec![0.0f32; out_h.len()];
    let mut state = vec![0.0f32; state_h.len()];
    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    hip.copy_d2h_f32(state_d, &mut state).unwrap();
    for i in 0..out.len() {
        assert!(
            (out[i] - out_h[i]).abs() < 2e-4,
            "out[{i}] device={} host={}",
            out[i],
            out_h[i]
        );
    }
    for i in 0..state.len() {
        assert!(
            (state[i] - state_h[i]).abs() < 2e-4,
            "state[{i}] device={} host={}",
            state[i],
            state_h[i]
        );
    }
    let digest = |v: &[f32]| {
        v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        })
    };
    assert_eq!(digest(&out), 0xa8ca08248c574d69, "delta output bits");
    assert_eq!(digest(&state), 0x8b64af77fdd441ae, "delta state bits");
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

    if let Err(e) = hip.expand_k_heads(src_d, dst_d, 2, 4, 2, false) {
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
    // Qwen3-Next interleave: [h0,h0, h1,h1].
    let mut dst_i = [0.0f32; 8];
    allpaca_core::deltanet::expand_k_heads_as(
        &src,
        &mut dst_i,
        2,
        4,
        2,
        allpaca_core::deltanet::KHeadRepeat::Interleave,
    );
    hip.expand_k_heads(src_d, dst_d, 2, 4, 2, true).unwrap();
    hip.copy_d2h_f32(dst_d, &mut dst).unwrap();
    assert_eq!(dst, dst_i, "device interleave must match host");
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
        false,
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
        false,
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
    let digest = gated.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
    });
    assert_eq!(digest, 590683133665046509, "cross-card GDN digest");
    hip.gdn_clear_request(req);
}

#[cfg(feature = "hip")]
#[test]
fn hip_gdn_decode_step_real_shape_has_stable_cross_card_bits() {
    if allpaca_hip::device_count() == 0 {
        return;
    }
    let hip = allpaca_hip::HipWorkspace::new(0, 8 << 20, 8).unwrap();
    let (n_k, n_v, head_k, head_v, conv_k) = (16usize, 32usize, 128usize, 128usize, 4usize);
    let key_dim = n_k * head_k;
    let qkv_dim = 2 * key_dim + n_v * head_v;
    let value_dim = n_v * head_v;
    let vector = |n: usize, seed: usize| {
        (0..n)
            .map(|i| ((i * 17 + seed * 31) % 257) as f32 / 128.0 - 1.0)
            .collect::<Vec<f32>>()
    };
    let qkv = vector(qkv_dim, 1);
    let z = vector(value_dim, 2);
    let alpha = vector(n_v, 3);
    let beta = vector(n_v, 4);
    let conv_w = vector(conv_k * qkv_dim, 5);
    let dt = vector(n_v, 6);
    let a = vector(n_v, 7);
    let norm = vector(head_v, 8);
    let mut host_state = vector(n_v * head_v * head_k, 9);
    let mut host_conv = vector((conv_k - 1) * qkv_dim, 10);
    let parts = [
        &qkv, &z, &alpha, &beta, &conv_w, &dt, &a, &norm,
    ];
    let total = parts.iter().map(|v| v.len()).sum::<usize>() + value_dim;
    let blob = allpaca_hip::DeviceWeights::alloc(0, total * 4).unwrap();
    let mut off = 0usize;
    let mut put = |src: &[f32]| {
        let p = blob.ptr_at(off) as *mut f32;
        hip.copy_h2d_f32(p, src).unwrap();
        off += src.len() * 4;
        p
    };
    let qkv_d = put(&qkv);
    let z_d = put(&z);
    let alpha_d = put(&alpha);
    let beta_d = put(&beta);
    let conv_w_d = put(&conv_w);
    let dt_d = put(&dt);
    let a_d = put(&a);
    let norm_d = put(&norm);
    let out_d = put(&vec![0.0f32; value_dim]);
    hip.gdn_decode_step_resident(
        41,
        41,
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
        true,
    )
    .unwrap();
    let mut out = vec![0.0f32; value_dim];
    hip.copy_d2h_f32(out_d, &mut out).unwrap();
    let digest = out.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
    });
    assert_eq!(digest, 18417993370823597083, "real-shape cross-card GDN digest");
    hip.gdn_clear_request(41);
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
            false,
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

/// Tiny Qwen3.8-shaped hybrid: GDN blocks with gated full attention every 4th
/// layer (`[Q|gate]`, Q/K norm, partial sectioned RoPE), an FFN on every
/// block, and seeded pseudo-random weights so greedy tokens are not all ties.
#[cfg(feature = "hip")]
fn write_tiny_qwen_hybrid_gguf(path: &std::path::Path) {
    write_tiny_qwen_hybrid_gguf_opts(path, false);
}

/// `mtp` appends a NextN block (`blk.{n_layers}`, gated attention + FFN +
/// `nextn.eh_proj` / norms) like Qwen3.8's in-file MTP head.
#[cfg(feature = "hip")]
fn write_tiny_qwen_hybrid_gguf_opts(path: &std::path::Path, mtp: bool) {
    use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
    use std::collections::HashMap;

    let dim = 32usize;
    let vocab = 16usize;
    let n_layers = 8usize;
    let ff = 64usize;
    // n_heads * head_dim > dim, as in Qwen3.8 (24 * 256 > 5120): that is how
    // the loader tells a gated `[Q|gate]` projection from a plain one.
    let (n_heads, n_kv, head_dim) = (4usize, 1usize, 16usize);
    let (n_v, n_k, state) = (2usize, 2usize, 8usize);
    let inner = n_v * state;
    let qkv_dim = n_k * state * 2 + inner;
    let conv_k = 4usize;
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rand = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let u32v = |v: usize| GgufValue::Uint32(v as u32);
    let mut meta = HashMap::new();
    meta.insert("general.architecture".into(), GgufValue::String("qwen35".into()));
    for (k, v) in [
        ("vocab_size", vocab),
        ("embedding_length", dim),
        ("block_count", n_layers + usize::from(mtp)),
        ("nextn_predict_layers", usize::from(mtp)),
        ("feed_forward_length", ff),
        ("attention.head_count", n_heads),
        ("attention.head_count_kv", n_kv),
        ("attention.key_length", head_dim),
        ("attention.value_length", head_dim),
        ("rope.dimension_count", head_dim / 2),
        ("full_attention_interval", 4),
        ("ssm.state_size", state),
        ("ssm.inner_size", inner),
        ("ssm.time_step_rank", n_v),
        ("ssm.group_count", n_k),
        ("ssm.conv_kernel", conv_k),
    ] {
        meta.insert(format!("qwen35.{k}"), u32v(v));
    }
    meta.insert(
        "qwen35.rope.dimension_sections".into(),
        GgufValue::Array([2i32, 1, 1, 0].map(GgufValue::Int32).to_vec()),
    );
    meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(
            (0..vocab)
                .map(|i| {
                    GgufValue::String(match i {
                        0 => "<unk>".into(),
                        1 => "<s>".into(),
                        2 => "</s>".into(),
                        _ => format!("t{i}"),
                    })
                })
                .collect(),
        ),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), u32v(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), u32v(2));

    let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let mut push = |name: String, dims: Vec<usize>, data: Vec<f32>| {
        owned.push((name, dims.into_iter().map(|d| d as u64).collect(), data));
    };
    let embed = rand(vocab * dim, 1.0);
    push("token_embd.weight".into(), vec![dim, vocab], embed.clone());
    for i in 0..n_layers + usize::from(mtp) {
        let b = |t: &str| format!("blk.{i}.{t}.weight");
        push(b("attn_norm"), vec![dim], vec![1.0; dim]);
        if i == n_layers {
            push(b("nextn.eh_proj"), vec![2 * dim, dim], rand(dim * 2 * dim, 0.3));
            push(b("nextn.enorm"), vec![dim], vec![1.0; dim]);
            push(b("nextn.hnorm"), vec![dim], vec![1.0; dim]);
            push(b("nextn.shared_head_norm"), vec![dim], vec![1.0; dim]);
        }
        if i % 4 == 3 || i == n_layers {
            push(b("attn_q"), vec![dim, 2 * n_heads * head_dim], rand(2 * n_heads * head_dim * dim, 0.3));
            push(b("attn_k"), vec![dim, n_kv * head_dim], rand(n_kv * head_dim * dim, 0.3));
            push(b("attn_v"), vec![dim, n_kv * head_dim], rand(n_kv * head_dim * dim, 0.3));
            push(b("attn_output"), vec![n_heads * head_dim, dim], rand(dim * n_heads * head_dim, 0.3));
            push(b("attn_q_norm"), vec![head_dim], vec![1.0; head_dim]);
            push(b("attn_k_norm"), vec![head_dim], vec![1.0; head_dim]);
        } else {
            push(b("attn_qkv"), vec![dim, qkv_dim], rand(qkv_dim * dim, 0.3));
            push(b("attn_gate"), vec![dim, inner], rand(inner * dim, 0.3));
            push(b("ssm_alpha"), vec![dim, n_v], rand(n_v * dim, 0.3));
            push(b("ssm_beta"), vec![dim, n_v], rand(n_v * dim, 0.3));
            push(b("ssm_out"), vec![inner, dim], rand(dim * inner, 0.3));
            push(format!("blk.{i}.ssm_a"), vec![n_v], vec![-0.5; n_v]);
            push(format!("blk.{i}.ssm_dt.bias"), vec![n_v], vec![0.0; n_v]);
            push(b("ssm_norm"), vec![state], vec![1.0; state]);
            push(b("ssm_conv1d"), vec![conv_k * qkv_dim], rand(conv_k * qkv_dim, 0.5));
        }
        push(b("post_attention_norm"), vec![dim], vec![1.0; dim]);
        push(b("ffn_gate"), vec![dim, ff], rand(ff * dim, 0.3));
        push(b("ffn_up"), vec![dim, ff], rand(ff * dim, 0.3));
        push(b("ffn_down"), vec![ff, dim], rand(dim * ff, 0.3));
    }
    push("output_norm.weight".into(), vec![dim], vec![1.0; dim]);
    // Unembed row t(i+1) = embed(t i) over the cycle t3 -> t4 -> t5 -> t6 -> t3,
    // so greedy loops and n-gram drafts get accepted; the blocks still perturb
    // the residual enough that a wrong verify forward shows up as a token diff.
    let mut output = rand(vocab * dim, 0.3);
    for (from, to) in [(3usize, 4usize), (4, 5), (5, 6), (6, 3)] {
        output[to * dim..(to + 1) * dim].copy_from_slice(&embed[from * dim..(from + 1) * dim]);
    }
    push("output.weight".into(), vec![dim, vocab], output);
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

/// Greedy generate on the tiny hybrid; returns ids plus the target forwards and
/// draft counters it cost. `None` when the engine finds no HIP GPU.
/// `prefill_chunk = 1` keeps prefill on the per-token decode path.
#[cfg(feature = "hip")]
fn tiny_hybrid_generate(
    path: &std::path::Path,
    prefill_chunk: usize,
    speculative: bool,
    tensor_split: Option<Vec<f64>>,
) -> Option<(Vec<u32>, u64, u64, u64)> {
    let mut cfg = tiny_written_cfg(path.to_path_buf(), false);
    cfg.prefill.chunk_tokens = prefill_chunk;
    cfg.speculative.enabled = speculative;
    cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Ngram];
    cfg.speculative.max_depth = 4;
    cfg.features.ngram_speculation = Some(speculative);
    cfg.rpc.tensor_split = tensor_split;
    let engine = Engine::new(cfg).expect("tiny hybrid Engine::new");
    if hip_gpus(&engine).is_empty() {
        return None;
    }
    let before = engine.stats();
    let out = engine
        .generate("t3 t4 t5 t6 t3 t4 t5 t6 t3 t4 t5 t6", greedy(40))
        .expect("tiny hybrid generate");
    let after = engine.stats();
    Some((
        out.token_ids,
        after.target_gpu_forwards - before.target_gpu_forwards,
        after.draft_proposed - before.draft_proposed,
        after.draft_accepted - before.draft_accepted,
    ))
}

/// Qwen3.8-shaped hybrid (GDN + gated IMRoPE attention) on HIP, one GPU and
/// split across two: greedy output matches per-token prefill + decode (the
/// host path has no GDN, so that is the reference), and speculative
/// verify packs each drafted span into one target forward, so accepted
/// drafts cost fewer target forwards than plain decode.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_hybrid_speculative_verify_batches_and_matches_greedy() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_hybrid_speculative_verify: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_tiny_qwen_hybrid-{}.gguf",
        std::process::id()
    ));
    write_tiny_qwen_hybrid_gguf(&path);
    let Some((reference, ..)) = tiny_hybrid_generate(&path, 1, false, None) else {
        let _ = std::fs::remove_file(&path);
        eprintln!("skip hip_gdn_hybrid_speculative_verify: no HIP in engine");
        return;
    };
    let mut splits = vec![None];
    if allpaca_hip::device_count() >= 2 {
        splits.push(Some(vec![1.0, 1.0]));
    }
    for split in splits {
        let (plain, plain_fwd, _, _) =
            tiny_hybrid_generate(&path, 512, false, split.clone()).expect("plain engine");
        let (spec, spec_fwd, proposed, accepted) =
            tiny_hybrid_generate(&path, 512, true, split.clone()).expect("spec engine");
        eprintln!(
            "tiny hybrid split={split:?}: target forwards plain={plain_fwd} spec={spec_fwd}, drafts accepted {accepted}/{proposed}"
        );
        assert_eq!(plain, reference, "batched prefill must match per-token (split {split:?})");
        assert_eq!(spec, reference, "speculative greedy must match per-token (split {split:?})");
        assert!(
            accepted > 0,
            "fixture must get drafts accepted to exercise verify; proposed={proposed}"
        );
        assert!(
            spec_fwd < plain_fwd,
            "accepted drafts must save target forwards (split {split:?}): spec={spec_fwd} plain={plain_fwd} accepted={accepted}/{proposed}"
        );
    }
    let _ = std::fs::remove_file(&path);
}

/// Greedy MTP generate on the tiny hybrid with its in-file NextN head.
/// Returns ids, target forwards, drafts accepted / proposed.
#[cfg(feature = "hip")]
fn tiny_hybrid_mtp_generate(
    path: &std::path::Path,
    prefill_chunk: usize,
    tensor_split: Option<Vec<f64>>,
) -> Option<(Vec<u32>, u64, u64, u64)> {
    let mut cfg = tiny_written_cfg(path.to_path_buf(), false);
    cfg.prefill.chunk_tokens = prefill_chunk;
    cfg.speculative.enabled = true;
    cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Mtp];
    cfg.speculative.max_depth = 3;
    cfg.rpc.tensor_split = tensor_split;
    let engine = Engine::new(cfg).expect("tiny hybrid MTP Engine::new");
    if hip_gpus(&engine).is_empty() {
        return None;
    }
    let before = engine.stats();
    let out = engine
        .generate("t3 t4 t5 t6 t3 t4 t5 t6 t3 t4 t5 t6", greedy(24))
        .expect("tiny hybrid MTP generate");
    let after = engine.stats();
    Some((
        out.token_ids,
        after.target_gpu_forwards - before.target_gpu_forwards,
        after.draft_accepted - before.draft_accepted,
        after.draft_proposed - before.draft_proposed,
    ))
}

/// Fixed soft tokens on a 2x2 grid (M-RoPE placement, like the Qwen3-VL mmproj).
struct GridVision {
    dim: usize,
}

impl allpaca_core::EncoderPlacement for GridVision {
    fn device(&self) -> Option<i32> {
        None
    }
}

impl allpaca_core::VisionEncoder for GridVision {
    fn encode(
        &self,
        images: &[allpaca_core::ImageInput],
    ) -> Result<allpaca_core::VisionTokens, allpaca_types::AllpacaError> {
        let n = 4 * images.len();
        Ok(allpaca_core::VisionTokens {
            embeddings: (0..n * self.dim).map(|i| ((i * 29) % 17) as f32 * 0.05 - 0.4).collect(),
            n_tokens: n,
            device: None,
            grids: vec![(2, 2); images.len()],
        })
    }
}

#[cfg(feature = "hip")]
fn tiny_hybrid_image_generate(path: &std::path::Path, prefill_chunk: usize) -> Option<(Vec<u32>, u64)> {
    let mut cfg = tiny_written_cfg(path.to_path_buf(), false);
    cfg.prefill.chunk_tokens = prefill_chunk;
    cfg.speculative.enabled = false;
    let engine = Engine::new(cfg).expect("tiny hybrid Engine::new");
    if hip_gpus(&engine).is_empty() {
        return None;
    }
    engine
        .set_vision_encoder(std::sync::Arc::new(GridVision { dim: 32 }))
        .expect("vision encoder");
    let before = engine.stats();
    let image = allpaca_core::ImageInput::new(vec![0u8; 4], "image/png");
    let out = engine
        .generate_multimodal("t3 t4 t5 t6 t3 t4 t5 t6", &[image], greedy(12))
        .expect("tiny hybrid image generate");
    let after = engine.stats();
    Some((out.token_ids, after.target_gpu_forwards - before.target_gpu_forwards))
}

/// Image soft tokens prefill in the same batched forward as the text around
/// them, and match one-token-at-a-time prefill.
#[cfg(feature = "hip")]
#[test]
fn hip_image_prompt_prefills_in_one_forward() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_image_prompt_prefills_in_one_forward: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!("allpaca_tiny_qwen_image-{}.gguf", std::process::id()));
    write_tiny_qwen_hybrid_gguf_opts(&path, false);
    let Some((reference, ref_fwd)) = tiny_hybrid_image_generate(&path, 1) else {
        let _ = std::fs::remove_file(&path);
        return;
    };
    let (ids, fwd) = tiny_hybrid_image_generate(&path, 512).expect("batched");
    let _ = std::fs::remove_file(&path);
    assert_eq!(ids, reference, "batched image prefill must match per-token");
    // 4 image + 8 text prompt tokens: one chunk forward instead of 12.
    assert_eq!(fwd + 11, ref_fwd, "image prompt must prefill in one target forward");
}

/// With an MTP draft the prompt still prefills as one target forward per
/// chunk; the draft's process KV is built from the chunk's row hiddens, so
/// drafts, acceptance and greedy output match one-token-at-a-time prefill.
#[cfg(feature = "hip")]
#[test]
fn hip_mtp_prefill_batches_and_keeps_drafts() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_mtp_prefill_batches: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_tiny_qwen_hybrid_mtp-{}.gguf",
        std::process::id()
    ));
    write_tiny_qwen_hybrid_gguf_opts(&path, true);
    let mut splits = vec![None];
    if allpaca_hip::device_count() >= 2 {
        splits.push(Some(vec![1.0, 1.0]));
    }
    for split in splits {
        let Some((reference, ref_fwd, ref_acc, ref_prop)) =
            tiny_hybrid_mtp_generate(&path, 1, split.clone())
        else {
            eprintln!("skip hip_mtp_prefill_batches: no HIP in engine");
            break;
        };
        let (ids, fwd, acc, prop) =
            tiny_hybrid_mtp_generate(&path, 512, split.clone()).expect("batched MTP engine");
        eprintln!(
            "tiny MTP split={split:?}: forwards per-token={ref_fwd} batched={fwd}, accepted {ref_acc}/{ref_prop} vs {acc}/{prop}"
        );
        assert!(ref_prop > 0, "fixture must exercise MTP drafts");
        assert_eq!(ids, reference, "batched MTP prefill must match per-token (split {split:?})");
        assert_eq!((acc, prop), (ref_acc, ref_prop), "draft KV must match per-token process");
        // 12 prompt tokens: one chunk forward instead of 12.
        assert_eq!(fwd + 11, ref_fwd, "prompt must prefill in one target forward");
    }
    let _ = std::fs::remove_file(&path);
}

/// A later turn keeps the drafter the earlier one warmed: no MTP step is
/// skipped, and the drafts and text equal a cold run's.
#[cfg(feature = "hip")]
#[test]
fn hip_resumed_mtp_drafts_like_a_cold_run() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_resumed_mtp_drafts_like_a_cold_run: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_tiny_qwen_hybrid_mtp_resume-{}.gguf",
        std::process::id()
    ));
    write_tiny_qwen_hybrid_gguf_opts(&path, true);
    let engine = || {
        let mut cfg = tiny_written_cfg(path.clone(), false);
        cfg.prefill.chunk_tokens = 512;
        cfg.speculative.enabled = true;
        cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Mtp];
        cfg.speculative.max_depth = 3;
        cfg.prompt_cache.codec = allpaca_types::KvCodec::F32;
        Engine::new(cfg).expect("tiny hybrid MTP Engine::new")
    };
    let first = "t3 t4 t5 t6 t3 t4 t5 t6";
    let second = "t3 t4 t5 t6 t3 t4 t5 t6 t3 t4 t5 t6 t3 t4 t5 t6";

    let cold = engine();
    if hip_gpus(&cold).is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let want = cold.generate(second, greedy(24)).expect("cold");
    let cold_stats = cold.stats();
    drop(cold);

    let warm = engine();
    warm.generate(first, greedy(2)).expect("first turn");
    let before = warm.stats();
    let got = warm.generate(second, greedy(24)).expect("second turn");
    let after = warm.stats();
    let _ = std::fs::remove_file(&path);

    assert_eq!(after.radix_hits, 1, "{after:?}");
    assert_eq!(got.token_ids, want.token_ids);
    assert_eq!(after.mtp_process_skips, before.mtp_process_skips, "the drafter must not fall behind");
    assert!(cold_stats.draft_proposed > 0, "fixture must exercise MTP drafts");
    assert_eq!(
        (
            after.draft_accepted - before.draft_accepted,
            after.draft_proposed - before.draft_proposed
        ),
        (cold_stats.draft_accepted, cold_stats.draft_proposed),
        "restored draft K/V must draft like the cold run's"
    );
}

/// The next turn of a chat keeps the live drafter: it does not read the
/// earlier turn again, and the text stays a cold run's.
#[cfg(feature = "hip")]
#[test]
fn hip_next_turn_keeps_the_live_drafter() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_next_turn_keeps_the_live_drafter: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_tiny_qwen_hybrid_mtp_turn-{}.gguf",
        std::process::id()
    ));
    write_tiny_qwen_hybrid_gguf_opts(&path, true);
    let engine = || {
        let mut cfg = tiny_written_cfg(path.clone(), false);
        cfg.prefill.chunk_tokens = 512;
        cfg.speculative.enabled = true;
        cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Mtp];
        cfg.speculative.max_depth = 3;
        cfg.prompt_cache.codec = allpaca_types::KvCodec::F32;
        Engine::new(cfg).expect("tiny hybrid MTP Engine::new")
    };
    let warm = engine();
    if hip_gpus(&warm).is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let first = "t3 t4 t5 t6 t3 t4 t5 t6";
    let reply = warm.generate(first, greedy(16)).expect("first turn");
    let turn: Vec<String> = reply.token_ids.iter().map(|id| format!("t{id}")).collect();
    let next = format!("{first} {} t3 t4", turn.join(" "));
    let before = warm.stats();
    let got = warm.generate(&next, greedy(8)).expect("second turn");
    let after = warm.stats();

    let cold = engine();
    let want = cold.generate(&next, greedy(8)).expect("cold");
    let cold_stats = cold.stats();
    let _ = std::fs::remove_file(&path);

    assert_eq!(after.radix_hits, 1, "{after:?}");
    assert_eq!(got.token_ids, want.token_ids);
    let forwards = after.draft_gpu_forwards - before.draft_gpu_forwards;
    assert!(
        forwards + 8 < cold_stats.draft_gpu_forwards,
        "resumed {forwards} vs cold {}",
        cold_stats.draft_gpu_forwards
    );
}

/// MTP process steps (pairing an accepted token with the target hidden) only
/// write the draft KV; they must not run the vocab-wide draft lm_head, and
/// greedy output must still match plain decode.
#[cfg(feature = "hip")]
#[test]
fn hip_mtp_process_skips_draft_lm_head() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_mtp_process_skips_draft_lm_head: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_tiny_qwen_hybrid_mtp_lm-{}.gguf",
        std::process::id()
    ));
    write_tiny_qwen_hybrid_gguf_opts(&path, true);
    let mut plain_cfg = tiny_written_cfg(path.clone(), false);
    plain_cfg.speculative.enabled = false;
    let plain = Engine::new(plain_cfg).expect("plain engine");
    if hip_gpus(&plain).is_empty() {
        let _ = std::fs::remove_file(&path);
        return;
    }
    let prompt = "t3 t4 t5 t6 t3 t4 t5 t6 t3 t4 t5 t6";
    let want = plain.generate(prompt, greedy(24)).expect("plain").token_ids;
    let mut cfg = tiny_written_cfg(path.clone(), false);
    cfg.speculative.enabled = true;
    cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Mtp];
    cfg.speculative.max_depth = 3;
    let engine = Engine::new(cfg).expect("MTP engine");
    let got = engine.generate(prompt, greedy(24)).expect("MTP").token_ids;
    let stats = engine.stats();
    let _ = std::fs::remove_file(&path);
    assert_eq!(got, want, "MTP greedy must match plain greedy");
    assert!(
        stats.draft_logits_skipped > 0,
        "process steps must skip the draft lm_head; draft_gpu_forwards={}",
        stats.draft_gpu_forwards
    );
    // A span of d drafts needs d vocab projections: the forward after the
    // last draft token only extends the draft KV.
    let lm_head_passes = stats.draft_gpu_forwards - stats.draft_logits_skipped;
    assert!(stats.draft_tokens_drafted >= stats.draft_proposed);
    assert_eq!(
        lm_head_passes, stats.draft_tokens_drafted,
        "one draft lm_head pass per drafted token"
    );
    // Draft-ahead only runs after a full accept, so at most the final step's
    // ahead span goes unverified.
    assert!(
        stats.draft_tokens_drafted <= stats.draft_proposed + 3,
        "drafted {} but proposed {}",
        stats.draft_tokens_drafted,
        stats.draft_proposed
    );
    // A rejected verify's correction token rides in the next packed span
    // (the draft predicts that slot) instead of its own target forward.
    assert!(
        stats.deferred_corrections_packed > 0,
        "corrections must pack into the next verify; accepted {}/{}",
        stats.draft_accepted,
        stats.draft_proposed
    );
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

#[cfg(feature = "hip")]
#[test]
fn hip_vision_encoder_install_records_weight_residency() {
    use allpaca_core::PatchVisionEncoder;
    use std::sync::Arc;

    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_vision_encoder_install_records_weight_residency: no HIP device");
        return;
    }
    let engine = match Engine::new(hip_auto_synthetic()) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skip hip_vision_encoder_install_records_weight_residency: Engine::new {e}");
            return;
        }
    };
    let gpus = hip_gpus(&engine);
    if gpus.is_empty() {
        eprintln!("skip hip_vision_encoder_install_records_weight_residency: no HIP in engine");
        return;
    }
    let dev = gpus[0].id;
    let dim = ModelConfig::tiny().dim;
    let enc = PatchVisionEncoder::synthetic(Some(dev), dim, 42).expect("synthetic vision");
    engine
        .set_vision_encoder(Arc::new(enc))
        .expect("install vision");
    let stats = engine.stats();
    assert!(
        stats.vision_weight_bytes_on_device > 0,
        "expected resident vision weights; stats={stats:?}"
    );
    // Soft-embed staging metric stays separate until a multimodal request.
    assert_eq!(stats.vision_bytes_on_device, 0);
}

#[cfg(feature = "hip")]
fn tiny_dsv4_generate(
    path: &std::path::Path,
    cpu_only: bool,
) -> (Vec<u32>, allpaca_types::StatsSnapshot) {
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path.to_path_buf());
    cfg.engine.cpu_only = cpu_only;
    cfg.engine.gpu_backend = "auto".into();
    cfg.speculative.enabled = false;
    let engine = Engine::new(cfg).expect("tiny DeepSeek-V4 must load");
    let out = engine
        .generate("hi", greedy(4))
        .expect("tiny DeepSeek-V4 must generate");
    (out.token_ids, engine.stats())
}

#[cfg(feature = "hip")]
#[test]
fn tiny_deepseek4_hip_f32_stays_cpu() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip tiny_deepseek4_hip_f32_stays_cpu: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_hip_tiny_dsv4_f32-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    allpaca_core::write_tiny_deepseek4_gguf(&path);
    let (cpu_ids, _) = tiny_dsv4_generate(&path, true);
    let (hip_ids, hip_stats) = tiny_dsv4_generate(&path, false);
    let _ = std::fs::remove_file(&path);
    assert!(!cpu_ids.is_empty());
    assert_eq!(hip_ids, cpu_ids, "F32 fixture must not use HIP GEMV");
    assert_eq!(hip_stats.gpu_forwards, 0, "stats={hip_stats:?}");
}

#[cfg(feature = "hip")]
#[test]
fn tiny_deepseek4_hip_matches_cpu() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip tiny_deepseek4_hip_matches_cpu: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_hip_tiny_dsv4_packed-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    allpaca_core::write_tiny_deepseek4_packed_gguf(&path);
    let (cpu_ids, _) = tiny_dsv4_generate(&path, true);
    let (hip_ids, hip_stats) = tiny_dsv4_generate(&path, false);
    let _ = std::fs::remove_file(&path);
    assert!(!cpu_ids.is_empty());
    assert_eq!(
        hip_ids, cpu_ids,
        "packed BF16/MXFP4 HIP must match CPU; hip={hip_ids:?} cpu={cpu_ids:?} stats={hip_stats:?}"
    );
    if hip_stats.backend == "hip" {
        assert!(
            hip_stats.gpu_forwards > 0,
            "packed HIP generate should record gpu_forwards; stats={hip_stats:?}"
        );
    }
}

/// Dense tensors stored as Q8_0 (Unsloth UD-Q4_K_XL) run on the GPU from
/// staged int8 blocks, through the same path as BF16.
#[cfg(feature = "hip")]
#[test]
fn tiny_deepseek4_q8_dense_hip_matches_cpu() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip tiny_deepseek4_q8_dense_hip_matches_cpu: no HIP device");
        return;
    }
    let path = std::env::temp_dir().join(format!(
        "allpaca_hip_tiny_dsv4_q8-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    allpaca_core::write_tiny_deepseek4_q8_gguf(&path);
    let (cpu_ids, _) = tiny_dsv4_generate(&path, true);
    let (hip_ids, hip_stats) = tiny_dsv4_generate(&path, false);
    let _ = std::fs::remove_file(&path);
    assert!(!cpu_ids.is_empty());
    assert_eq!(
        hip_ids, cpu_ids,
        "Q8_0 dense HIP must match CPU; hip={hip_ids:?} cpu={cpu_ids:?} stats={hip_stats:?}"
    );
    if hip_stats.backend == "hip" {
        assert!(
            hip_stats.gpu_forwards > 0,
            "Q8_0 dense HIP generate should record gpu_forwards; stats={hip_stats:?}"
        );
    }
}

/// Decode `prompts` concurrently through `poll` (all admitted at once) and
/// return each request's token ids in prompt order.
#[cfg(feature = "hip")]
fn poll_concurrently(engine: &Engine, prompts: &[&str], params: &GenerationParams) -> Vec<Vec<u32>> {
    use allpaca_types::StepResult;
    let handles: Vec<_> = prompts
        .iter()
        .map(|p| engine.begin_request(p, params.clone()).expect("begin_request"))
        .collect();
    let ids: Vec<u64> = handles.iter().map(|h| h.id()).collect();
    let mut out = vec![Vec::new(); ids.len()];
    let mut done = vec![false; ids.len()];
    for _ in 0..10_000 {
        let Some(ev) = engine.poll().expect("poll") else {
            break;
        };
        let i = ids.iter().position(|&id| id == ev.request_id).expect("known request");
        match ev.result {
            StepResult::Token(tok) => {
                out[i].push(tok.token_id);
                done[i] |= tok.finish_reason.is_some();
            }
            StepResult::Finished(_) => done[i] = true,
            StepResult::Prefill { .. } => {}
        }
        if done.iter().all(|&d| d) {
            break;
        }
    }
    assert!(done.iter().all(|&d| d), "every request must finish; got {out:?}");
    out
}

/// gemma4 (tiny, on the GPU): n-gram speculation equals plain greedy, and
/// requests run together (batched and not) equal their solo runs.
#[cfg(feature = "hip")]
#[test]
fn hip_gemma4_spec_and_concurrent_match_solo_greedy() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gemma4_spec_and_concurrent: no HIP");
        return;
    }
    let path = std::env::temp_dir().join(format!("allpaca_g4_engine_{}.gguf", std::process::id()));
    allpaca_core::write_tiny_gemma4_gguf(&path, 2);
    let prompts = [
        "t1 t2 t3 t4 t1 t2 t3 t4 t1 t2 t3",
        "t5 t6 t7 t5 t6 t7 t5 t6",
        "t9 t8 t9 t8 t9 t8 t9",
    ];
    let params = greedy(40);
    let engine_for = |ngram: bool, batch_decode: bool| {
        let mut cfg = hip_7b_config(path.clone());
        cfg.engine.max_active_requests = 3;
        cfg.engine.batch_decode = batch_decode;
        cfg.prompt_cache.ram_bytes = 1;
        if ngram {
            cfg.speculative.enabled = true;
            cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Ngram];
        }
        Engine::new(cfg).expect("tiny gemma4 must load")
    };
    let plain = engine_for(false, false);
    let solo: Vec<Vec<u32>> = prompts.iter().map(|p| poll_concurrently(&plain, &[p], &params).remove(0)).collect();
    for (ngram, batch_decode) in [(true, false), (false, true), (true, true)] {
        let engine = engine_for(ngram, batch_decode);
        for (i, p) in prompts.iter().enumerate() {
            let one = poll_concurrently(&engine, &[p], &params).remove(0);
            assert_eq!(one, solo[i], "request {i} solo (ngram={ngram} batch={batch_decode})");
        }
        let together = poll_concurrently(&engine, &prompts, &params);
        assert_eq!(together, solo, "requests together (ngram={ngram} batch={batch_decode})");
        let stats = engine.stats();
        if ngram {
            assert!(stats.linear_packed_verifies > 0, "n-gram spans must be verified: {stats:?}");
        }
        if batch_decode {
            assert!(stats.batched_decode_steps > 0, "batched decode must run: {stats:?}");
        }
    }
}

/// Requests decoded side by side must produce exactly what each produces
/// alone: per-request state (KV pages, logits, sampling) never leaks across.
#[cfg(feature = "hip")]
#[test]
fn hip_concurrent_requests_match_solo_greedy() {
    let _guard = large_gguf_lock();
    let path = local_models_dir()
        .join("Qwen2.5-Coder-0.5B")
        .join("qwen2.5-coder-0.5b-instruct-q8_0.gguf");
    if !path.is_file() || allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_concurrent_requests_match_solo_greedy: no Qwen2.5-Coder-0.5B or HIP");
        return;
    }
    // (batched decode, n-gram speculation, experimental.row_variant_kernels):
    // n-gram verify spans ride in the batched forward too. With the option on,
    // prefill may take the gfx12 WMMA attention, but batched decode and verify
    // must still match solo.
    for (batch_decode, ngram, row_variant) in [
        (true, false, false),
        (false, false, false),
        (true, true, false),
        (true, false, true),
        (true, true, true),
    ] {
        let mut cfg = hip_7b_config(path.clone());
        cfg.experimental.row_variant_kernels = row_variant;
        cfg.engine.max_active_requests = 3;
        cfg.engine.batch_decode = batch_decode;
        // Small pages: sequences span several, as long chats do.
        cfg.kv.page_tokens = 16;
        if ngram {
            cfg.speculative.enabled = true;
            cfg.speculative.providers = vec![allpaca_types::DraftProviderKind::Ngram];
        }
        concurrent_matches_solo(cfg, batch_decode, ngram);
    }
}

#[cfg(feature = "hip")]
fn concurrent_matches_solo(cfg: Config, batch_decode: bool, ngram: bool) {
    let engine = Engine::new(cfg).expect("Qwen2.5-Coder-0.5B must load");
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_concurrent_requests_match_solo_greedy: no HIP device");
        return;
    }
    // A shared preamble, as a chat template gives every request.
    let prompts = [
        "You are a helpful assistant. Answer briefly.
User: def fibonacci(n):",
        "You are a helpful assistant. Answer briefly.
User: The three largest cities in France are",
        "You are a helpful assistant. Answer briefly.
User: SELECT name, email FROM users WHERE",
    ];
    let params = greedy(24);
    let solo: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| poll_concurrently(&engine, &[p], &params).remove(0))
        .collect();
    let together = poll_concurrently(&engine, &prompts, &params);
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(together[i], solo[i], "request {i} ({p:?}) diverged when run concurrently (batch_decode={batch_decode} ngram={ngram})");
    }
    // One thread per request, as the HTTP server drives them.
    let threaded: Vec<Vec<u32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|p| {
                let (engine, params) = (engine.clone(), params.clone());
                scope.spawn(move || engine.generate(p, params).expect("generate").token_ids)
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("thread")).collect()
    });
    for (i, p) in prompts.iter().enumerate() {
        assert_eq!(threaded[i], solo[i], "request {i} ({p:?}) diverged on its own thread (batch_decode={batch_decode} ngram={ngram})");
    }
    let stats = engine.stats();
    assert_eq!(
        stats.batched_decode_steps > 0 && stats.batched_decode_rows > stats.batched_decode_steps,
        batch_decode,
        "batched forwards must run exactly when batch_decode is on; stats={stats:?}"
    );
    if ngram {
        assert!(
            stats.linear_packed_verifies > 0,
            "n-gram pass must verify drafted spans; stats={stats:?}"
        );
    }
    eprintln!(
        "batched decode (ngram={ngram}): {} forwards, {} rows, {} packed verifies",
        stats.batched_decode_steps, stats.batched_decode_rows, stats.linear_packed_verifies
    );
}

/// The fused GDN decode step (3 launches) gives the same bits as the unfused
/// sequence (conv, shift, 2x expand, decay/beta, L2 norm, delta step) at
/// Qwen3-Next's shape, for both K-head mappings, over several steps.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_fused_step_matches_unfused_bit_for_bit() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_fused_step: no HIP device");
        return;
    }
    let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 8) else {
        eprintln!("skip hip_gdn_fused_step: HIP workspace unavailable");
        return;
    };
    let (n_k, n_v, hk, hv, ck) = (16usize, 32usize, 128usize, 128usize, 4usize);
    let qkv_dim = 2 * n_k * hk + n_v * hv;
    let mut seed = 0x9e37_79b9u32;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
            })
            .collect()
    };
    let conv_w = rnd(ck * qkv_dim, 1.0);
    let dt_bias = rnd(n_v, 1.0);
    let a: Vec<f32> = rnd(n_v, 1.0).iter().map(|x| -x.abs() - 0.1).collect();
    let conv_state0 = rnd((ck - 1) * qkv_dim, 1.0);
    let state0 = rnd(n_v * hv * hk, 0.2);
    let steps: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> =
        (0..3).map(|_| (rnd(qkv_dim, 2.0), rnd(n_v, 2.0), rnd(n_v, 2.0))).collect();

    for interleave in [true, false] {
        // One set of device buffers per path; same inputs, stepped in lockstep.
        let run = |fused: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let floats = qkv_dim * 2 + conv_w.len() + conv_state0.len() + n_v * 4 + state0.len() + n_v * hv;
            let blob = allpaca_hip::DeviceWeights::alloc(0, floats * 4).expect("alloc");
            let mut off = 0usize;
            let mut put = |src: &[f32]| -> *mut f32 {
                let p = blob.ptr_at(off) as *mut f32;
                hip.copy_h2d_f32(p, src).unwrap();
                off += src.len() * 4;
                p
            };
            let conv_w_d = put(&conv_w);
            let conv_s_d = put(&conv_state0);
            let dt_d = put(&dt_bias);
            let a_d = put(&a);
            let state_d = put(&state0);
            let qkv_d = put(&vec![0.0; qkv_dim]);
            let alpha_d = put(&vec![0.0; n_v]);
            let beta_d = put(&vec![0.0; n_v]);
            let y_d = put(&vec![0.0; n_v * hv]);
            let mut y = vec![0.0f32; n_v * hv];
            let mut beta = vec![0.0f32; n_v];
            for (qkv, alpha, beta0) in &steps {
                hip.copy_h2d_f32(qkv_d, qkv).unwrap();
                hip.copy_h2d_f32(alpha_d, alpha).unwrap();
                hip.copy_h2d_f32(beta_d, beta0).unwrap();
                let args = (
                    qkv_d as *const f32, alpha_d as *const f32, beta_d, conv_w_d as *const f32,
                    conv_s_d, dt_d as *const f32, a_d as *const f32, state_d, y_d,
                );
                let r = if fused {
                    hip.gdn_layer_step(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8,
                        n_v as i32, n_k as i32, hv as i32, hk as i32, ck as i32, 1e-6, interleave)
                } else {
                    hip.gdn_layer_step_unfused(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8,
                        n_v as i32, n_k as i32, hv as i32, hk as i32, ck as i32, 1e-6, interleave)
                };
                r.expect("gdn step");
            }
            hip.copy_d2h_f32(y_d, &mut y).unwrap();
            hip.copy_d2h_f32(beta_d, &mut beta).unwrap();
            let mut state = vec![0.0f32; state0.len()];
            hip.copy_d2h_f32(state_d, &mut state).unwrap();
            let mut conv_s = vec![0.0f32; conv_state0.len()];
            hip.copy_d2h_f32(conv_s_d, &mut conv_s).unwrap();
            (y, state, conv_s, beta)
        };
        let fused = run(true);
        let unfused = run(false);
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(fused.0.iter().any(|v| *v != 0.0), "fused step produced no output");
        assert!(same(&fused.0, &unfused.0), "y differs (interleave={interleave})");
        assert!(same(&fused.1, &unfused.1), "recurrent state differs (interleave={interleave})");
        assert!(same(&fused.2, &unfused.2), "conv state differs (interleave={interleave})");
        assert!(same(&fused.3, &unfused.3), "beta differs (interleave={interleave})");
    }
}

/// The fused GDN output (per-head RMSNorm + SiLU gate) gives the same bits as
/// copies + rmsnorm_heads + silu_mul, including decode's `out == z` aliasing.
#[cfg(feature = "hip")]
#[test]
fn hip_gdn_norm_gate_fused_matches_unfused_bit_for_bit() {
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_gdn_norm_gate_fused: no HIP device");
        return;
    }
    let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 1 << 20, 8) else {
        eprintln!("skip hip_gdn_norm_gate_fused: HIP workspace unavailable");
        return;
    };
    let (n_heads, hv) = (32usize, 128usize);
    let n = n_heads * hv;
    let mut seed = 0x1234_5677u32;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed >> 8) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
            })
            .collect()
    };
    let (y, z, w) = (rnd(n), rnd(n), rnd(hv));
    let run = |fused: bool, alias_z: bool| -> Vec<f32> {
        let blob = allpaca_hip::DeviceWeights::alloc(0, (3 * n + hv) * 4).expect("alloc");
        let (y_d, z_d, w_d, o_d) = (
            blob.ptr_at(0) as *mut f32,
            blob.ptr_at(n * 4) as *mut f32,
            blob.ptr_at(2 * n * 4) as *mut f32,
            blob.ptr_at((2 * n + hv) * 4) as *mut f32,
        );
        hip.copy_h2d_f32(y_d, &y).unwrap();
        hip.copy_h2d_f32(z_d, &z).unwrap();
        hip.copy_h2d_f32(w_d, &w).unwrap();
        let out = if alias_z { z_d } else { o_d };
        let r = if fused {
            hip.gdn_rmsnorm_silu_gate(y_d, z_d, w_d, out, n_heads as i32, hv as i32, 1e-6)
        } else {
            hip.gdn_rmsnorm_silu_gate_unfused(y_d, z_d, w_d, out, n_heads as i32, hv as i32, 1e-6)
        };
        r.expect("norm gate");
        let mut o = vec![0.0f32; n];
        hip.copy_d2h_f32(out, &mut o).unwrap();
        o
    };
    for alias_z in [false, true] {
        let (f, u) = (run(true, alias_z), run(false, alias_z));
        assert!(f.iter().any(|v| *v != 0.0), "no output");
        assert!(
            f.iter().zip(&u).all(|(a, b)| a.to_bits() == b.to_bits()),
            "fused norm+gate differs (alias_z={alias_z})"
        );
    }
}

/// Wall time of one GDN decode step (delta step + norm/gate) at Qwen3-Next's
/// shape, fused vs unfused, launch -> sync. Run with
/// `cargo test --release -p allpaca-core --features hip --test hip_runtime bench_gdn_step_fused -- --ignored --nocapture`.
#[cfg(feature = "hip")]
#[test]
#[ignore]
fn bench_gdn_step_fused_vs_unfused() {
    if allpaca_hip::device_count() == 0 {
        return;
    }
    let dev = std::env::var("BENCH_DEV").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let Ok(hip) = allpaca_hip::HipWorkspace::new(dev, 1 << 20, 8) else { return };
    let (n_k, n_v, hk, hv, ck) = (16usize, 32usize, 128usize, 128usize, 4usize);
    let qkv_dim = 2 * n_k * hk + n_v * hv;
    let floats = qkv_dim * (1 + ck + (ck - 1)) + n_v * 4 + n_v * hv * hk + n_v * hv * 2 + hv;
    let blob = allpaca_hip::DeviceWeights::alloc(dev, floats * 4).expect("alloc");
    let mut off = 0usize;
    let mut take = |n: usize, v: f32| -> *mut f32 {
        let p = blob.ptr_at(off) as *mut f32;
        hip.copy_h2d_f32(p, &vec![v; n]).unwrap();
        off += n * 4;
        p
    };
    let qkv = take(qkv_dim, 0.1);
    let conv_w = take(ck * qkv_dim, 0.2);
    let conv_s = take((ck - 1) * qkv_dim, 0.0);
    let alpha = take(n_v, 0.1);
    let beta = take(n_v, 0.5);
    let dt = take(n_v, 0.0);
    let a = take(n_v, -0.5);
    let state = take(n_v * hv * hk, 0.0);
    let y = take(n_v * hv, 0.0);
    let z = take(n_v * hv, 0.3);
    let norm = take(hv, 1.0);
    let step = |fused: bool| {
        if fused {
            hip.gdn_layer_step(qkv, alpha, beta, conv_w, conv_s, dt, a, state, y, n_v as i32, n_k as i32, hv as i32, hk as i32, ck as i32, 1e-6, true).unwrap();
            hip.gdn_rmsnorm_silu_gate(y, z, norm, z, n_v as i32, hv as i32, 1e-6).unwrap();
        } else {
            hip.gdn_layer_step_unfused(qkv, alpha, beta, conv_w, conv_s, dt, a, state, y, n_v as i32, n_k as i32, hv as i32, hk as i32, ck as i32, 1e-6, true).unwrap();
            hip.gdn_rmsnorm_silu_gate_unfused(y, z, norm, z, n_v as i32, hv as i32, 1e-6).unwrap();
        }
    };
    for gap_us in [0u64, 500] {
        for fused in [false, true, false, true] {
            let mut tot = 0.0;
            for _ in 0..200 {
                let until = std::time::Instant::now() + std::time::Duration::from_micros(gap_us);
                while std::time::Instant::now() < until {}
                let t = std::time::Instant::now();
                step(fused);
                allpaca_hip::stream_synchronize().unwrap();
                tot += t.elapsed().as_secs_f64() * 1e6;
            }
            eprintln!("GDNSTEP dev{dev} gap {gap_us}us {}: {:.1} us", if fused { "fused  " } else { "unfused" }, tot / 200.0);
        }
    }
}

/// Nemotron-H attention has no rotary positions (its Mamba layers carry
/// order). A `rope_none` model must decode the same on HIP as on CPU, dense
/// and hybrid, and must not decode like the rotating model.
#[cfg(feature = "hip")]
#[test]
fn hip_rope_none_matches_cpu_and_differs_from_rope() {
    let run = |cfg: Config, model: ModelConfig| {
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(model).unwrap();
        let out = engine
            .generate("hello world hello hello world", greedy(12))
            .unwrap();
        (out.token_ids, hip_gpus(&engine).len())
    };
    for (name, base) in [("dense", ModelConfig::tiny()), ("hybrid", ModelConfig::tiny_hybrid())] {
        let mut none = base.clone();
        none.rope_none = true;
        let (cpu, _) = run(cpu_only_synthetic(), none.clone());
        let (hip, gpus) = run(hip_auto_synthetic(), none);
        if gpus == 0 {
            eprintln!("skip hip_rope_none_matches_cpu_and_differs_from_rope: no HIP device");
            return;
        }
        assert_eq!(hip, cpu, "{name}: HIP must skip the rotation like CPU");
        let (rotated, _) = run(cpu_only_synthetic(), base);
        assert_ne!(cpu, rotated, "{name}: rope_none must change the decode");
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_prompt_cache_hit_decodes_like_a_full_prefill() {
    let cached = "hello world hello world hello true";
    // A follow-up turn resumes any model; a mid-prompt edit only resumes
    // models without recurrent state.
    let cases = [
        ("dense extended", ModelConfig::tiny(), "hello world hello world hello true world hello", 7),
        ("hybrid extended", ModelConfig::tiny_hybrid(), "hello world hello world hello true world hello", 7),
        ("dense edited", ModelConfig::tiny(), "hello world hello world hello there world", 6),
    ];
    for (name, model, prompt, skipped) in cases {
        let fresh = Engine::new(hip_auto_synthetic()).unwrap();
        fresh.load_synthetic(model.clone()).unwrap();
        if hip_gpus(&fresh).is_empty() {
            eprintln!("skip hip_prompt_cache_hit_decodes_like_a_full_prefill: no HIP device");
            return;
        }
        let full = fresh.generate(prompt, greedy(12)).unwrap();

        let mut cfg = hip_auto_synthetic();
        cfg.prompt_cache.codec = allpaca_types::KvCodec::F32;
        let warm = Engine::new(cfg).unwrap();
        warm.load_synthetic(model).unwrap();
        warm.generate(cached, greedy(4)).unwrap();
        let hit = warm.generate(prompt, greedy(12)).unwrap();
        let stats = warm.stats();
        assert_eq!(stats.radix_hits, 1, "{name}: the cached prefix must hit; {stats:?}");
        assert_eq!(stats.radix_skip_tokens, skipped, "{name}");
        assert_eq!(hit.token_ids, full.token_ids, "{name}: a hit must decode as a full prefill");
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_dense_chat_restores_a_mid_edit_without_splitting_prefill() {
    let model = ModelConfig::tiny();
    let fresh = Engine::new(hip_auto_synthetic()).unwrap();
    fresh.load_synthetic(model.clone()).unwrap();
    if hip_gpus(&fresh).is_empty() {
        eprintln!("skip hip_dense_chat_restores_a_mid_edit_without_splitting_prefill: no HIP device");
        return;
    }
    let edited = fresh
        .format_chat_with_checkpoints(&[
            ("user", "hello"),
            ("assistant", "world"),
            ("user", "true"),
        ])
        .unwrap();
    let cold = fresh.generate(&edited.text, greedy(12)).unwrap();

    let mut cfg = hip_auto_synthetic();
    cfg.prompt_cache.codec = allpaca_types::KvCodec::F32;
    let warm = Engine::new(cfg).unwrap();
    warm.load_synthetic(model).unwrap();
    let cached = warm
        .format_chat_with_checkpoints(&[
            ("user", "hello"),
            ("assistant", "world"),
            ("user", "there"),
        ])
        .unwrap();
    warm.generate_multimodal_with_checkpoints(&cached, &[], greedy(4))
        .unwrap();
    let first = warm.stats();
    assert_eq!(first.prefill_chunks, 1, "{first:?}");
    let hit = warm
        .generate_multimodal_with_checkpoints(&edited, &[], greedy(12))
        .unwrap();

    let stats = warm.stats();
    assert_eq!(stats.radix_hits, 1, "the shared prefix must hit; {stats:?}");
    assert!(
        stats.radix_skip_tokens >= edited.checkpoint_offsets[1] as u64,
        "the whole second message must be skipped; {stats:?}"
    );
    assert_eq!(hit.token_ids, cold.token_ids);
}

/// An extra prefill stop changes a hybrid's text, so a chat does not add any.
#[cfg(feature = "hip")]
#[test]
fn hip_hybrid_chat_prefills_in_one_chunk() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    engine.load_synthetic(ModelConfig::tiny_hybrid()).unwrap();
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_hybrid_chat_prefills_in_one_chunk: no HIP device");
        return;
    }
    let prompt = engine.format_chat_with_checkpoints(&[("user", "hello"); 4]).unwrap();
    engine
        .generate_multimodal_with_checkpoints(&prompt, &[], greedy(1))
        .unwrap();
    let stats = engine.stats();
    assert_eq!(stats.prefill_chunks, 1, "{stats:?}");
}

/// Real model: a prompt that extends a cached one decodes exactly as a full
/// prefill. `ALLPACA_CACHE_GGUF` = model path (a hybrid such as Qwen3.5
/// checks the recurrent state too).
#[cfg(feature = "hip")]
#[test]
fn hip_prompt_cache_hit_matches_full_prefill_on_a_real_model() {
    let _guard = large_gguf_lock();
    let Some(path) = std::env::var_os("ALLPACA_CACHE_GGUF").map(std::path::PathBuf::from) else {
        eprintln!("skip hip_prompt_cache_hit_matches_full_prefill_on_a_real_model: set ALLPACA_CACHE_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_prompt_cache_hit_matches_full_prefill_on_a_real_model: no HIP device");
        return;
    }
    let turn = "The river carried silt from the mountains to the plain, and every \
                spring the farmers waited for the flood to leave its dark soil behind. ";
    // Ends on a token boundary: a trailing space would merge into the next word.
    let cached = turn.repeat(12).trim_end().to_string();
    let prompt = format!("{cached} Question: what did the farmers wait for each spring? Answer:");
    // `ALLPACA_CACHE_CODEC` = f16 / q8 / q4 reports how far a lossy hit
    // follows the full prefill instead of requiring the same tokens.
    let codec = match std::env::var("ALLPACA_CACHE_CODEC").as_deref() {
        Ok("f16") => allpaca_types::KvCodec::F16,
        Ok("q8") => allpaca_types::KvCodec::Q8,
        Ok("q4") => allpaca_types::KvCodec::Q4,
        _ => allpaca_types::KvCodec::F32,
    };
    let full = {
        let engine = Engine::new(hip_7b_config(path.clone())).unwrap();
        engine.generate(&prompt, greedy(24)).unwrap()
    };
    let mut cfg = hip_7b_config(path);
    cfg.prompt_cache.codec = codec;
    let engine = Engine::new(cfg).unwrap();
    engine.generate(&cached, greedy(2)).unwrap();
    let hit = engine.generate(&prompt, greedy(24)).unwrap();
    let stats = engine.stats();
    let same = hit
        .token_ids
        .iter()
        .zip(&full.token_ids)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "prompt cache {codec:?}: hits={} skipped={} bytes={} same first {same} of {} tokens",
        stats.radix_hits,
        stats.radix_skip_tokens,
        stats.radix_bytes,
        full.token_ids.len()
    );
    assert_eq!(stats.radix_hits, 1, "{stats:?}");
    assert!(stats.radix_skip_tokens > 200, "{stats:?}");
    if codec == allpaca_types::KvCodec::F32 {
        assert_eq!(hit.token_ids, full.token_ids, "a hit must decode as a full prefill");
    }
}

#[cfg(feature = "hip")]
#[test]
fn hip_tapped_logits_are_what_greedy_decode_picked_from() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    engine.load_synthetic(ModelConfig::tiny()).unwrap();
    if hip_gpus(&engine).is_empty() {
        eprintln!("skip hip_tapped_logits_are_what_greedy_decode_picked_from: no HIP device");
        return;
    }
    engine.tap_logits(true).unwrap();
    let out = engine.generate("hello world hello", greedy(6)).unwrap();
    let rows = engine.take_tapped_logits().unwrap();
    assert_eq!(rows.len(), out.token_ids.len(), "one row per decoded token");
    for (row, &token) in rows.iter().zip(&out.token_ids) {
        let best = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
        assert_eq!(best as u32, token);
    }
    assert!(engine.take_tapped_logits().unwrap().is_empty(), "taking drains the tap");
}

/// KL(p || q) of two logit rows, in nats.
fn logits_kl(p: &[f32], q: &[f32]) -> f64 {
    let log_softmax = |row: &[f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
        let log_sum = sum.ln() + max;
        row.iter().map(|&x| x as f64 - log_sum).collect::<Vec<_>>()
    };
    let (lp, lq) = (log_softmax(p), log_softmax(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum::<f64>().max(0.0)
}

/// Real model: how far each checkpoint codec moves the next-token
/// distribution of a cache hit from a full prefill (KL per decode step,
/// counted while both runs picked the same tokens). f32 must pick the same
/// tokens; its KL is the float-drift floor.
/// `ALLPACA_CACHE_GGUF` = model path, `ALLPACA_CACHE_KL=1` to run.
#[cfg(feature = "hip")]
#[test]
fn hip_prompt_cache_codec_kl_on_a_real_model() {
    let _guard = large_gguf_lock();
    let Some(path) = std::env::var_os("ALLPACA_CACHE_GGUF").map(std::path::PathBuf::from) else {
        eprintln!("skip hip_prompt_cache_codec_kl_on_a_real_model: set ALLPACA_CACHE_GGUF");
        return;
    };
    if std::env::var("ALLPACA_CACHE_KL").ok().as_deref() != Some("1") {
        eprintln!("skip hip_prompt_cache_codec_kl_on_a_real_model: set ALLPACA_CACHE_KL=1");
        return;
    }
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_prompt_cache_codec_kl_on_a_real_model: no HIP device");
        return;
    }
    let steps = 48;
    let cases = [
        (
            "The river carried silt from the mountains to the plain, and every spring the farmers waited for the flood to leave its dark soil behind. ",
            " Question: what did the farmers wait for each spring? Answer:",
        ),
        (
            "fn parse(input: &str) -> Result<Vec<u32>, String> { input.split(',').map(|s| s.trim().parse::<u32>().map_err(|e| e.to_string())).collect() } ",
            " // Explain what parse returns for an empty string:",
        ),
        (
            "Meeting notes: the team agreed to move the release to Thursday, Priya owns the migration, and the dashboard stays read-only until the audit closes. ",
            " Summary of the decisions in one sentence:",
        ),
    ];
    let mut prompts = Vec::new();
    for (turn, question) in cases {
        let cached = turn.repeat(8).trim_end().to_string();
        let prompt = format!("{cached}{question}");
        let engine = Engine::new(hip_7b_config(path.clone())).unwrap();
        engine.tap_logits(true).unwrap();
        let full = engine.generate(&prompt, greedy(steps)).unwrap();
        let rows = engine.take_tapped_logits().unwrap();
        prompts.push((cached, prompt, full.token_ids, rows));
    }
    for codec in [
        allpaca_types::KvCodec::F32,
        allpaca_types::KvCodec::F16,
        allpaca_types::KvCodec::Q8,
        allpaca_types::KvCodec::Q4,
    ] {
        let (mut sum, mut max, mut n, mut agreed, mut total, mut bytes) = (0.0f64, 0.0f64, 0usize, 0usize, 0usize, 0u64);
        for (cached, prompt, full_tokens, full_rows) in &prompts {
            let mut cfg = hip_7b_config(path.clone());
            cfg.prompt_cache.codec = codec;
            let engine = Engine::new(cfg).unwrap();
            engine.generate(cached, greedy(1)).unwrap();
            engine.tap_logits(true).unwrap();
            let hit = engine.generate(prompt, greedy(steps)).unwrap();
            let rows = engine.take_tapped_logits().unwrap();
            let stats = engine.stats();
            assert_eq!(stats.radix_hits, 1, "{codec:?}: {stats:?}");
            bytes += stats.radix_bytes as u64;
            let same = hit.token_ids.iter().zip(full_tokens).take_while(|(a, b)| a == b).count();
            agreed += same;
            total += full_tokens.len();
            if let (true, Some(row)) = (same < full_tokens.len(), full_rows.get(same)) {
                let mut sorted = row.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                eprintln!(
                    "  {codec:?} diverged at step {same}: full prefill's top-2 logit margin {:.4}",
                    sorted[0] - sorted[1]
                );
            }
            // Row i was sampled after the same i tokens in both runs.
            for (p, q) in full_rows.iter().zip(&rows).take((same + 1).min(rows.len())) {
                let kl = logits_kl(p, q);
                sum += kl;
                max = max.max(kl);
                n += 1;
            }
        }
        let mean = sum / n.max(1) as f64;
        eprintln!(
            "prompt cache KL {codec:?}: mean {mean:.6} max {max:.6} over {n} steps; same tokens {agreed}/{total}; cache RAM {bytes} bytes"
        );
        // f32 stores the rows as they were; what is left is float drift from
        // prefilling the suffix alone, the floor the other codecs sit on. It
        // can flip a near-tie, so the tokens are only reported.
        if codec == allpaca_types::KvCodec::F32 {
            assert!(max < 0.05, "f32 drift beyond float noise: {max}");
        }
    }
}

/// Real model: checkpoints a first engine wrote to the disk tier serve a
/// second engine, which decodes exactly as a full prefill.
#[cfg(feature = "hip")]
#[test]
fn hip_prompt_cache_from_disk_matches_full_prefill_on_a_real_model() {
    let _guard = large_gguf_lock();
    let Some(path) = std::env::var_os("ALLPACA_CACHE_GGUF").map(std::path::PathBuf::from) else {
        eprintln!("skip hip_prompt_cache_from_disk_matches_full_prefill_on_a_real_model: set ALLPACA_CACHE_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_prompt_cache_from_disk_matches_full_prefill_on_a_real_model: no HIP device");
        return;
    }
    let dir = std::env::temp_dir()
        .join("allpaca_prompt_cache_tests")
        .join(format!("real_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let turn = "The river carried silt from the mountains to the plain, and every spring the farmers waited for the flood to leave its dark soil behind. ";
    let cached = turn.repeat(12).trim_end().to_string();
    let prompt = format!("{cached} Question: what did the farmers wait for each spring? Answer:");
    let with_disk = || {
        let mut cfg = hip_7b_config(path.clone());
        cfg.prompt_cache.codec = allpaca_types::KvCodec::F32;
        cfg.prompt_cache.disk_dir = Some(dir.clone());
        cfg
    };
    let full = Engine::new(hip_7b_config(path.clone())).unwrap().generate(&prompt, greedy(24)).unwrap();
    Engine::new(with_disk()).unwrap().generate(&cached, greedy(2)).unwrap();
    let engine = Engine::new(with_disk()).unwrap();
    let started = std::time::Instant::now();
    let hit = engine.generate(&prompt, greedy(24)).unwrap();
    let stats = engine.stats();
    eprintln!(
        "prompt cache from disk: hits={} skipped={} in {:?}",
        stats.radix_hits,
        stats.radix_skip_tokens,
        started.elapsed()
    );
    assert_eq!(stats.radix_hits, 1, "{stats:?}");
    assert_eq!(hit.token_ids, full.token_ids, "a hit from disk must decode as a full prefill");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Real hybrid model: an edit inside a long cached prompt resumes from the
/// last state checkpoint before it and decodes exactly as a full prefill.
#[cfg(feature = "hip")]
#[test]
fn hip_prompt_cache_resumes_a_hybrid_mid_prompt_on_a_real_model() {
    let _guard = large_gguf_lock();
    let Some(path) = std::env::var_os("ALLPACA_CACHE_HYBRID_GGUF").map(std::path::PathBuf::from) else {
        eprintln!("skip hip_prompt_cache_resumes_a_hybrid_mid_prompt_on_a_real_model: set ALLPACA_CACHE_HYBRID_GGUF");
        return;
    };
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_prompt_cache_resumes_a_hybrid_mid_prompt_on_a_real_model: no HIP device");
        return;
    }
    let turn = "The river carried silt from the mountains to the plain, and every spring the farmers waited for the flood to leave its dark soil behind. ";
    let cached = format!("{}The harvest was good that year.", turn.repeat(40));
    let edited = format!("{}The harvest failed that year. What happened to the harvest?", turn.repeat(40));
    let config = || {
        let mut cfg = hip_7b_config(path.clone());
        cfg.prompt_cache.codec = allpaca_types::KvCodec::F32;
        cfg.prompt_cache.boundary_tokens = 256;
        cfg.prefill.chunk_tokens = 256;
        cfg
    };
    let full = Engine::new(config()).unwrap().generate(&edited, greedy(24)).unwrap();
    let engine = Engine::new(config()).unwrap();
    engine.generate(&cached, greedy(2)).unwrap();
    let hit = engine.generate(&edited, greedy(24)).unwrap();
    let stats = engine.stats();
    eprintln!(
        "prompt cache mid-prompt: hits={} skipped={} of {} prompt tokens",
        stats.radix_hits,
        stats.radix_skip_tokens,
        stats.prompt_tokens
    );
    assert_eq!(stats.radix_hits, 1, "{stats:?}");
    assert!(stats.radix_skip_tokens >= 1024, "{stats:?}");
    assert_eq!(stats.radix_skip_tokens % 256, 0, "resumes at a boundary; {stats:?}");
    assert_eq!(hit.token_ids, full.token_ids, "a mid-prompt hit must decode as a full prefill");
}

/// P13: a tiny llama whose matrices are random valid blocks of type `ty`
/// (`block_bytes` per `block_elems` weights). The FFN matrices (131072
/// weights) take the device staging path; scale fields are set so the
/// layer, not the small embedding, decides the logits.
#[cfg(feature = "hip")]
fn write_tiny_typed_llama_gguf(path: &std::path::Path, ty: allpaca_gguf::GgmlType, block_bytes: usize, block_elems: usize) {
    use allpaca_gguf::{write_gguf_quant, GgmlType as T, GgufValue, GgufWriteQuantTensor, GgufWriteTensor};
    use std::collections::HashMap;

    let (dim, ff, vocab) = (256usize, 512usize, 16usize);
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (ty as u64);
    let mut blocks = |n: usize| -> Vec<u8> {
        let mut out: Vec<u8> = (0..n / block_elems * block_bytes)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 24) as u8
            })
            .collect();
        for (b, block) in out.chunks_exact_mut(block_bytes).enumerate() {
            match ty {
                // Half scale 2^-7 in the top nibbles of the four scale words.
                T::IQ1_M => {
                    for (k, nibble) in [0u8, 0, 0, 2].into_iter().enumerate() {
                        block[49 + 2 * k] = (block[49 + 2 * k] & 0x0f) | (nibble << 4);
                    }
                }
                T::IQ2_XXS | T::IQ2_XS | T::IQ2_S | T::IQ3_XXS | T::IQ1_S => {
                    block[..2].copy_from_slice(&0x2000u16.to_le_bytes());
                }
                // UE4M3 scale bytes 0x20..0x27: (8 + m) * 2^-7.
                T::RocmFpxQ8 => block[block_bytes - 1] = 0x20 | (b % 8) as u8,
                _ => {
                    block[block_bytes - 2] = 0x20 | (b % 8) as u8;
                    block[block_bytes - 1] = 0x20 | ((b + 3) % 8) as u8;
                }
            }
        }
        out
    };
    let (q, k, v, o) = (blocks(dim * dim), blocks(dim / 4 * dim), blocks(dim / 4 * dim), blocks(dim * dim));
    let (gate, up, down) = (blocks(ff * dim), blocks(ff * dim), blocks(dim * ff));

    let mut meta = HashMap::new();
    meta.insert("general.architecture".into(), GgufValue::String("llama".into()));
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
    meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
    meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
    meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
    meta.insert("llama.feed_forward_length".into(), GgufValue::Uint32(ff as u32));
    meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
    let mut tokens = vec!["<unk>".to_string(), "<s>".into(), "</s>".into(), "hello".into()];
    tokens.extend((4..vocab).map(|i| format!("t{i}")));
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
    let embd: Vec<f32> = (0..vocab * dim).map(|i| ((i * 37 % 101) as f32 - 50.0) / 4096.0).collect();
    let ones = vec![1.0f32; dim];
    let quant = |name: &str, dims: [usize; 2]| (name.to_string(), vec![dims[0] as u64, dims[1] as u64]);
    let named = [
        (quant("blk.0.attn_q.weight", [dim, dim]), &q),
        (quant("blk.0.attn_k.weight", [dim, dim / 4]), &k),
        (quant("blk.0.attn_v.weight", [dim, dim / 4]), &v),
        (quant("blk.0.attn_output.weight", [dim, dim]), &o),
        (quant("blk.0.ffn_gate.weight", [dim, ff]), &gate),
        (quant("blk.0.ffn_up.weight", [dim, ff]), &up),
        (quant("blk.0.ffn_down.weight", [ff, dim]), &down),
    ];
    let quant_tensors: Vec<GgufWriteQuantTensor> = named
        .iter()
        .map(|((name, dims), bytes)| GgufWriteQuantTensor { name: name.clone(), dims: dims.clone(), ggml_type: ty, bytes })
        .collect();
    write_gguf_quant(
        path,
        &meta,
        &[
            GgufWriteTensor { name: "token_embd.weight".into(), dims: vec![dim as u64, vocab as u64], data: &embd },
            GgufWriteTensor { name: "blk.0.attn_norm.weight".into(), dims: vec![dim as u64], data: &ones },
            GgufWriteTensor { name: "blk.0.ffn_norm.weight".into(), dims: vec![dim as u64], data: &ones },
            GgufWriteTensor { name: "output_norm.weight".into(), dims: vec![dim as u64], data: &ones },
        ],
        &quant_tensors,
    )
    .expect("write tiny typed GGUF");
}

/// P13: every new GGUF type loads on the HIP path, runs its FFN on the GPU
/// (IQ types as the Q8_0 re-pack, ROCmFPX 103 as lossless Q8_0, the other
/// ROCmFPX types as lossless F16) and gives the CPU path's greedy tokens.
#[cfg(feature = "hip")]
#[test]
fn hip_new_gguf_types_match_cpu_greedy() {
    use allpaca_gguf::GgmlType as T;
    if allpaca_hip::device_count() == 0 {
        eprintln!("skip hip_new_gguf_types_match_cpu_greedy: no HIP device");
        return;
    }
    let types = [
        (T::IQ2_XXS, allpaca_cpu::IQ2_XXS_BLOCK_BYTES, 256),
        (T::IQ2_XS, allpaca_cpu::IQ2_XS_BLOCK_BYTES, 256),
        (T::IQ2_S, allpaca_cpu::IQ2_S_BLOCK_BYTES, 256),
        (T::IQ3_XXS, allpaca_cpu::IQ3_XXS_BLOCK_BYTES, 256),
        (T::IQ3_S, allpaca_cpu::IQ3_S_BLOCK_BYTES, 256),
        (T::IQ4_XS, allpaca_cpu::IQ4_XS_BLOCK_BYTES, 256),
        (T::IQ1_S, allpaca_cpu::IQ1_S_BLOCK_BYTES, 256),
        (T::IQ1_M, allpaca_cpu::IQ1_M_BLOCK_BYTES, 256),
        (T::RocmFpxQ6, allpaca_cpu::ROCMFPX_Q6_BLOCK_BYTES, 32),
        (T::RocmFpxQ8, allpaca_cpu::ROCMFPX_Q8_BLOCK_BYTES, 32),
        (T::RocmFpxQ3, allpaca_cpu::ROCMFPX_Q3_BLOCK_BYTES, 32),
        (T::RocmFpxQ5, allpaca_cpu::ROCMFPX_Q5_BLOCK_BYTES, 32),
        (T::RocmFpxQ7, allpaca_cpu::ROCMFPX_Q7_BLOCK_BYTES, 32),
        (T::RocmFpxQ2, allpaca_cpu::ROCMFPX_Q2_BLOCK_BYTES, 32),
    ];
    for (ty, block_bytes, block_elems) in types {
        let path = std::env::temp_dir().join(format!("allpaca_p13_tiny_{}_{}.gguf", ty as u32, std::process::id()));
        write_tiny_typed_llama_gguf(&path, ty, block_bytes, block_elems);
        let run = |cpu_only: bool| {
            let mut cfg = hip_7b_config(path.clone());
            cfg.engine.cpu_only = cpu_only;
            let engine = Engine::new(cfg).unwrap_or_else(|e| panic!("{ty:?} load (cpu_only={cpu_only}): {e}"));
            let out = engine.generate("hello", greedy(6)).unwrap_or_else(|e| panic!("{ty:?} generate: {e}"));
            (out.token_ids, engine.stats())
        };
        let (cpu_ids, _) = run(true);
        let (hip_ids, stats) = run(false);
        let _ = std::fs::remove_file(&path);
        eprintln!("{ty:?}: cpu {cpu_ids:?} hip {hip_ids:?} gpu_forwards {}", stats.gpu_forwards);
        assert!(!cpu_ids.is_empty(), "{ty:?}");
        assert_eq!(stats.backend, "hip", "{ty:?}");
        assert!(stats.gpu_forwards > 0, "{ty:?} must run on the GPU; stats={stats:?}");
        assert_eq!(hip_ids, cpu_ids, "{ty:?}: HIP greedy tokens must equal CPU");
    }
}
