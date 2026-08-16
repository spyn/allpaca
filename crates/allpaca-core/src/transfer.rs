use allpaca_types::{BandwidthBenchmark, DeviceKind, DeviceInfo, TransferPath};

/// Benchmark P2P vs host staging when two HIP devices exist. Host staging is the fallback.
pub fn select_transfer_path(devices: &[DeviceInfo], cpu_only: bool) -> TransferPath {
    select_transfer_path_with_bench(devices, cpu_only).0
}

pub fn select_transfer_path_with_bench(
    devices: &[DeviceInfo],
    cpu_only: bool,
) -> (TransferPath, Option<BandwidthBenchmark>) {
    if cpu_only {
        tracing::info!("transfer path: HostStaging (cpu_only; P2P not assumed)");
        return (TransferPath::HostStaging, None);
    }
    let gpus: Vec<&DeviceInfo> = devices.iter().filter(|d| d.kind == DeviceKind::HipGpu).collect();
    if gpus.len() < 2 {
        tracing::info!("transfer path: HostStaging (fewer than two GPUs; P2P not assumed)");
        return (TransferPath::HostStaging, None);
    }
    let src = gpus[0].id;
    let dst = gpus[1].id;
    match allpaca_hip::benchmark_transfer(src, dst, 4 * 1024 * 1024) {
        Ok(bench) => {
            let path = if !allpaca_hip::can_p2p(src, dst) {
                tracing::warn!(src, dst, "P2P capability false; using host staging");
                TransferPath::HostStaging
            } else {
                bench.faster_path()
            };
            tracing::info!(?path, ?bench, "selected transfer path from boot benchmark");
            (path, Some(bench))
        }
        Err(e) => {
            tracing::warn!(error = %e, "transfer benchmark degraded; host staging fallback");
            (TransferPath::HostStaging, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_inventory_uses_host_staging() {
        assert_eq!(
            select_transfer_path(&[DeviceInfo::cpu()], true),
            TransferPath::HostStaging
        );
    }
}
