use allpaca_types::{
    BandwidthBenchmark, Config, DeviceInfo, DeviceKind, DeviceSelector, PipelineMode, TransferPath,
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
        }
        let pipeline_mode = config
            .engine
            .pipeline_mode
            .unwrap_or_else(|| detect_pipeline_mode(&devices));
        let mut flags = allpaca_types::FeatureFlags::from_mode(pipeline_mode);
        flags.overlay(&config.features);
        let scheduler = Scheduler::new(pipeline_mode, flags, &config.speculative);
        let placement = scheduler.placement(&devices);
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
        }
    }
}

pub fn detect_pipeline_mode(devices: &[DeviceInfo]) -> PipelineMode {
    let gpus: Vec<&DeviceInfo> = devices.iter().filter(|d| d.kind == DeviceKind::HipGpu).collect();
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

fn peer_flags(devices: &[DeviceInfo], cpu_only: bool) -> (bool, bool) {
    if cpu_only {
        return (false, false);
    }
    let gpus: Vec<i32> = devices
        .iter()
        .filter(|d| d.kind == DeviceKind::HipGpu)
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
        // Engine execute path is still HIP; cuda:N selects a CUDA id for inventory only.
        DeviceSelector::Cuda(id) => id,
    };
    let draft = match config.engine.draft_device {
        DeviceSelector::Auto => placement.draft_device,
        DeviceSelector::Cpu => -1,
        DeviceSelector::Hip(id) => id,
        DeviceSelector::Cuda(id) => id,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_only_is_mobile_tiered() {
        let devices = vec![DeviceInfo::cpu()];
        assert_eq!(detect_pipeline_mode(&devices), PipelineMode::MobileTiered);
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
}
