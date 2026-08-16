//! Public Engine / DeviceInventory / Config behaviors for the HIP path.
//! GPU-gated tests skip when `hipGetDeviceCount() == 0`.

use allpaca_core::Engine;
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
fn generate_on_hip_uses_real_device_memory() {
    let engine = Engine::new(hip_auto_synthetic()).unwrap();
    if hip_gpus(&engine).is_empty() {
        return;
    }
    let out = engine.generate("hello", greedy(4)).unwrap();
    assert!(!out.token_ids.is_empty());
    let stats = engine.stats();
    assert!(
        stats.device_memory_bytes > 0,
        "hipMalloc workspace must report device memory used > 0, got {}",
        stats.device_memory_bytes
    );
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
        let out = engine.generate("hello", greedy(2)).unwrap();
        assert!(!out.token_ids.is_empty());
    }
}
