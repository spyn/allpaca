use allpaca_types::{
    BandwidthBenchmark, Config, DeviceInfo, DeviceKind, DeviceSelector, PipelineMode, TransferPath,
    RPC_DEVICE_ID_BASE,
};

use crate::scheduler::Scheduler;

#[derive(Debug, Clone)]
pub struct DeviceInventory {
    pub devices: Vec<DeviceInfo>,
    pub pipeline_mode: PipelineMode,
    pub transfer_path: TransferPath,
    pub transfer_benchmark: Option<BandwidthBenchmark>,
    pub p2p_capable: bool,
    pub p2p_enabled: bool,
    pub target_device: i32,
    pub draft_device: i32,
    pub vision_device: Option<i32>,
}

impl DeviceInventory {
    /// Per-pair P2P after the boot enable + re-query, not a raw pre-enable probe.
    pub fn can_p2p(&self, src: i32, dst: i32) -> bool {
        self.devices
            .iter()
            .find(|d| d.id == src)
            .map(|d| d.peers.contains(&dst))
            .unwrap_or(false)
    }

    pub fn discover(config: &Config) -> Self {
        let mut devices = vec![DeviceInfo::cpu()];
        if !config.engine.cpu_only {
            devices.extend(allpaca_hip::enumerate_devices());
            #[cfg(feature = "cuda")]
            {
                devices.extend(allpaca_cuda::enumerate_devices());
            }
            #[cfg(feature = "vulkan")]
            {
                devices.extend(allpaca_vulkan::enumerate_devices());
            }
        }
        for (i, endpoint) in config.rpc.endpoints.iter().enumerate() {
            let id = RPC_DEVICE_ID_BASE + i as i32;
            devices.push(DeviceInfo::rpc(id, endpoint, 0, DeviceKind::Cpu));
        }
        let pipeline_mode = config
            .engine
            .pipeline_mode
            .unwrap_or_else(|| detect_pipeline_mode(&devices));
        let mut flags = allpaca_types::FeatureFlags::from_mode(pipeline_mode);
        flags.overlay(&config.features);
        let scheduler = Scheduler::new(pipeline_mode, flags, &config.speculative);
        let vision_device = resolve_vision_device(config, &devices);
        let placement = scheduler.placement(&devices, vision_device);
        let (target_device, draft_device) = resolve_devices(config, &placement, &devices);
        let (transfer_path, transfer_benchmark) =
            crate::transfer::select_transfer_path_with_bench(&devices, config.engine.cpu_only);
        let (p2p_capable, p2p_enabled) = peer_flags(&devices, config.engine.cpu_only);
        tracing::info!(
            ?pipeline_mode,
            ?transfer_path,
            p2p_capable,
            p2p_enabled,
            target_device,
            draft_device,
            ?vision_device,
            devices = devices.len(),
            "device inventory ready"
        );
        Self {
            devices,
            pipeline_mode,
            transfer_path,
            transfer_benchmark,
            p2p_capable,
            p2p_enabled,
            target_device,
            draft_device,
            vision_device,
        }
    }

    pub fn placement_vision_device(&self) -> Option<i32> {
        self.vision_device
    }
}

pub fn detect_pipeline_mode(devices: &[DeviceInfo]) -> PipelineMode {
    let gpus = local_gpus(devices);
    match gpus.len() {
        0 | 1 => PipelineMode::MobileTiered,
        _ => {
            let b0 = gpus[0].bandwidth_gbps;
            let b1 = gpus[1].bandwidth_gbps;
            if b0 > 0.0 && b1 > 0.0 {
                let hi = b0.max(b1);
                let lo = b0.min(b1).max(1e-6);
                if hi / lo >= 1.3 {
                    PipelineMode::Asymmetric
                } else {
                    PipelineMode::Symmetric
                }
            } else {
                PipelineMode::Asymmetric
            }
        }
    }
}

/// When `kv.vram_bytes` is still the tiny compiled default, scale from device VRAM.
/// Leaves ≥50% for weights/workspace and caps KV at 4 GiB (fits RTX 3060 12 GB,
/// Ada/Blackwell laptop MobileTiered profiles).
pub fn suggested_kv_vram_bytes(total_vram: u64) -> usize {
    let floor = allpaca_types::KvSection::default().vram_bytes;
    if total_vram == 0 {
        return floor;
    }
    let half = (total_vram / 2) as usize;
    let cap = 4usize << 30;
    half.min(cap).max(floor)
}

fn local_gpus(devices: &[DeviceInfo]) -> Vec<&DeviceInfo> {
    devices
        .iter()
        .filter(|d| d.is_gpu() && !d.is_rpc())
        .collect()
}

fn peer_flags(devices: &[DeviceInfo], cpu_only: bool) -> (bool, bool) {
    if cpu_only {
        return (false, false);
    }
    let gpus: Vec<i32> = devices
        .iter()
        .filter(|d| d.kind == DeviceKind::HipGpu && !d.is_rpc())
        .map(|d| d.id)
        .collect();
    if gpus.len() < 2 {
        return (false, false);
    }
    let ab = allpaca_hip::probe_peer(gpus[0], gpus[1]);
    let ba = allpaca_hip::probe_peer(gpus[1], gpus[0]);
    (ab.capable && ba.capable, ab.enabled && ba.enabled)
}

fn resolve_devices(
    config: &Config,
    placement: &allpaca_types::Placement,
    devices: &[DeviceInfo],
) -> (i32, i32) {
    let target = match config.engine.target_device {
        DeviceSelector::Auto => placement.target_device,
        DeviceSelector::Cpu => -1,
        DeviceSelector::Hip(id) => id,
        // Engine execute path is still HIP; cuda:N / vulkan:N select inventory ids.
        DeviceSelector::Cuda(id) | DeviceSelector::Vulkan(id) => id,
    };
    let draft = match config.engine.draft_device {
        DeviceSelector::Auto => placement.draft_device,
        DeviceSelector::Cpu => -1,
        DeviceSelector::Hip(id) => id,
        DeviceSelector::Cuda(id) | DeviceSelector::Vulkan(id) => id,
    };
    let target = if devices.iter().any(|d| d.id == target) {
        target
    } else {
        -1
    };
    let draft = if devices.iter().any(|d| d.id == draft) {
        draft
    } else {
        -1
    };
    (target, draft)
}

fn resolve_vision_device(config: &Config, devices: &[DeviceInfo]) -> Option<i32> {
    let sel = config.multimodal.vision_device?;
    let id = match sel {
        DeviceSelector::Auto => return None,
        DeviceSelector::Cpu => -1,
        DeviceSelector::Hip(id) | DeviceSelector::Cuda(id) | DeviceSelector::Vulkan(id) => id,
    };
    if devices.iter().any(|d| d.id == id) {
        Some(id)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_is_mobile_tiered() {
        let devices = vec![DeviceInfo::cpu()];
        assert_eq!(detect_pipeline_mode(&devices), PipelineMode::MobileTiered);
    }

    #[test]
    fn single_cuda_3060_is_mobile_tiered() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo {
                id: 0,
                kind: DeviceKind::CudaGpu,
                name: "NVIDIA GeForce RTX 3060".into(),
                memory_bytes: 12 << 30,
                bandwidth_gbps: 360.0,
                peers: vec![],
            },
        ];
        assert_eq!(detect_pipeline_mode(&devices), PipelineMode::MobileTiered);
    }

    #[test]
    fn two_cuda_gpus_unequal_bandwidth_are_asymmetric() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo {
                id: 0,
                kind: DeviceKind::CudaGpu,
                name: "fast".into(),
                memory_bytes: 12 << 30,
                bandwidth_gbps: 600.0,
                peers: vec![],
            },
            DeviceInfo {
                id: 1,
                kind: DeviceKind::CudaGpu,
                name: "laptop".into(),
                memory_bytes: 8 << 30,
                bandwidth_gbps: 300.0,
                peers: vec![],
            },
        ];
        assert_eq!(detect_pipeline_mode(&devices), PipelineMode::Asymmetric);
    }

    #[test]
    fn suggested_kv_vram_for_12gb_leaves_half_capped_at_4gib() {
        let twelve = 12u64 << 30;
        assert_eq!(suggested_kv_vram_bytes(twelve), 4usize << 30);
        assert_eq!(suggested_kv_vram_bytes(0), 4 * 1024 * 1024);
        assert_eq!(suggested_kv_vram_bytes(1 << 30), 512 * 1024 * 1024);
        assert_eq!(suggested_kv_vram_bytes(6 << 30), 3usize << 30);
        assert_eq!(suggested_kv_vram_bytes(8 << 30), 4usize << 30);
    }

    #[test]
    fn two_unequal_gpus_are_asymmetric() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo {
                id: 0,
                kind: DeviceKind::HipGpu,
                name: "fast".into(),
                memory_bytes: 16 << 30,
                bandwidth_gbps: 640.0,
                peers: vec![1],
            },
            DeviceInfo {
                id: 1,
                kind: DeviceKind::HipGpu,
                name: "slow".into(),
                memory_bytes: 16 << 30,
                bandwidth_gbps: 320.0,
                peers: vec![0],
            },
        ];
        assert_eq!(detect_pipeline_mode(&devices), PipelineMode::Asymmetric);
    }

    #[test]
    fn placement_uses_configured_vision_device() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo {
                id: 0,
                kind: DeviceKind::HipGpu,
                name: "fast".into(),
                memory_bytes: 16 << 30,
                bandwidth_gbps: 640.0,
                peers: vec![1],
            },
            DeviceInfo {
                id: 1,
                kind: DeviceKind::HipGpu,
                name: "slow".into(),
                memory_bytes: 16 << 30,
                bandwidth_gbps: 320.0,
                peers: vec![0],
            },
        ];
        let mut cfg = Config::compiled_defaults();
        cfg.multimodal.vision_device = Some(DeviceSelector::Hip(1));
        let vision = resolve_vision_device(&cfg, &devices);
        assert_eq!(vision, Some(1));
        let flags = allpaca_types::FeatureFlags::from_mode(PipelineMode::Asymmetric);
        let scheduler = Scheduler::new(PipelineMode::Asymmetric, flags, &cfg.speculative);
        let placement = scheduler.placement(&devices, vision);
        assert_eq!(placement.vision_device, Some(1));
        assert_eq!(placement.target_device, 0);
        assert_eq!(placement.draft_device, 1);
    }

    #[test]
    fn rpc_endpoints_append_devices() {
        let mut cfg = Config::compiled_defaults();
        cfg.engine.cpu_only = true;
        cfg.rpc.endpoints = vec!["127.0.0.1:50052".into()];
        let inv = DeviceInventory::discover(&cfg);
        assert!(inv.devices.iter().any(|d| d.is_rpc()));
        assert_eq!(
            inv.devices
                .iter()
                .find(|d| d.is_rpc())
                .unwrap()
                .rpc_endpoint(),
            Some("127.0.0.1:50052")
        );
        let mut named = DeviceInfo::rpc(1001, "10.0.0.2:50052", 1, DeviceKind::HipGpu);
        named.name = "rpc[10.0.0.2:50052#1]".into();
        assert_eq!(named.rpc_endpoint(), Some("10.0.0.2:50052"));
    }
}
