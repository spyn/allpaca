use std::sync::Mutex;

use serde::Serialize;
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Serialize)]
pub struct HostSnapshot {
    pub cpu_usage: f32,
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
}

pub struct HostSampler {
    sys: Mutex<System>,
}

impl HostSampler {
    pub fn new() -> Self {
        let sys = System::new_with_specifics(
            RefreshKind::nothing()
                .with_cpu(sysinfo::CpuRefreshKind::nothing().with_cpu_usage())
                .with_memory(MemoryRefreshKind::nothing().with_ram()),
        );
        Self {
            sys: Mutex::new(sys),
        }
    }

    pub fn snapshot(&self) -> HostSnapshot {
        let Ok(mut sys) = self.sys.lock() else {
            return HostSnapshot {
                cpu_usage: 0.0,
                ram_used_bytes: 0,
                ram_total_bytes: 0,
            };
        };
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        HostSnapshot {
            cpu_usage: sys.global_cpu_usage(),
            ram_used_bytes: sys.used_memory(),
            ram_total_bytes: sys.total_memory(),
        }
    }
}
