use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    HipGpu,
    CudaGpu,
    VulkanGpu,
    /// Retained for serde compat; Coral/NPU backends are dropped from scope.
    EdgeTpu,
    Nvme,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub id: i32,
    pub kind: DeviceKind,
    pub name: String,
    pub memory_bytes: u64,
    pub bandwidth_gbps: f64,
    pub peers: Vec<i32>,
}

impl DeviceInfo {
    pub fn cpu() -> Self {
        Self {
            id: -1,
            kind: DeviceKind::Cpu,
            name: "cpu".to_string(),
            memory_bytes: 0,
            bandwidth_gbps: 50.0,
            peers: Vec::new(),
        }
    }

    pub fn is_gpu(&self) -> bool {
        matches!(
            self.kind,
            DeviceKind::HipGpu | DeviceKind::CudaGpu | DeviceKind::VulkanGpu
        )
    }

    /// Remote worker advertised as `rpc[host:port]`.
    pub fn rpc(id: i32, endpoint: &str, memory_bytes: u64, kind: DeviceKind) -> Self {
        Self {
            id,
            kind,
            name: format!("rpc[{endpoint}]"),
            memory_bytes,
            bandwidth_gbps: 1.0,
            peers: Vec::new(),
        }
    }

    pub fn is_rpc(&self) -> bool {
        self.name.starts_with("rpc[")
    }

    pub fn rpc_endpoint(&self) -> Option<&str> {
        let inner = self.name.strip_prefix("rpc[")?.strip_suffix(']')?;
        Some(match inner.split_once('#') {
            Some((endpoint, _)) => endpoint,
            None => inner,
        })
    }
}

/// Device ids for RPC workers start here so they do not collide with HIP ordinals.
pub const RPC_DEVICE_ID_BASE: i32 = 1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Cpu,
    Hip(i32),
    Cuda(i32),
    Vulkan(i32),
}

impl Serialize for DeviceSelector {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let owned;
        let s = match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Hip(id) => {
                owned = format!("hip:{id}");
                owned.as_str()
            }
            Self::Cuda(id) => {
                owned = format!("cuda:{id}");
                owned.as_str()
            }
            Self::Vulkan(id) => {
                owned = format!("vulkan:{id}");
                owned.as_str()
            }
        };
        serializer.serialize_str(s)
    }
}

impl<'de> Deserialize<'de> for DeviceSelector {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        Self::parse(&raw).map_err(serde::de::Error::custom)
    }
}

impl DeviceSelector {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if trimmed.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if let Some(rest) = trimmed
            .strip_prefix("hip:")
            .or_else(|| trimmed.strip_prefix("HIP:"))
        {
            let id = rest
                .parse::<i32>()
                .map_err(|_| format!("invalid HIP device id in '{raw}'"))?;
            return Ok(Self::Hip(id));
        }
        if let Some(rest) = trimmed
            .strip_prefix("cuda:")
            .or_else(|| trimmed.strip_prefix("CUDA:"))
        {
            let id = rest
                .parse::<i32>()
                .map_err(|_| format!("invalid CUDA device id in '{raw}'"))?;
            return Ok(Self::Cuda(id));
        }
        if let Some(rest) = trimmed
            .strip_prefix("vulkan:")
            .or_else(|| trimmed.strip_prefix("VULKAN:"))
        {
            let id = rest
                .parse::<i32>()
                .map_err(|_| format!("invalid Vulkan device id in '{raw}'"))?;
            return Ok(Self::Vulkan(id));
        }
        Err(format!(
            "unknown device selector '{raw}' (expected cpu, hip:N, cuda:N, vulkan:N, or auto)"
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TaskCost {
    pub compute_us: f64,
    pub h2d_us: f64,
    pub d2h_us: f64,
    pub p2p_us: f64,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

impl TaskCost {
    pub fn predicted_latency_us(&self) -> f64 {
        self.compute_us + self.h2d_us + self.d2h_us + self.p2p_us
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPath {
    P2p,
    HostStaging,
}

/// Prefer P2P when enable+copy succeeded and measured P2P bandwidth is not
/// more than 10% below host-staging bandwidth.
pub const P2P_MAX_SLOWER_FRACTION: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandwidthBenchmark {
    pub bytes: u64,
    pub p2p_gbps: f64,
    pub host_staging_gbps: f64,
    #[serde(default)]
    pub p2p_us: f64,
    #[serde(default)]
    pub host_staging_us: f64,
    #[serde(default)]
    pub p2p_copy_ok: bool,
}

impl BandwidthBenchmark {
    pub fn faster_path(&self) -> TransferPath {
        self.selected_path()
    }

    /// `P2p` only when the peer copy succeeded and is not slower than host
    /// staging by more than [`P2P_MAX_SLOWER_FRACTION`].
    pub fn selected_path(&self) -> TransferPath {
        if !self.p2p_copy_ok || self.p2p_gbps <= 0.0 {
            return TransferPath::HostStaging;
        }
        if self.host_staging_gbps <= 0.0 {
            return TransferPath::P2p;
        }
        let floor = self.host_staging_gbps * (1.0 - P2P_MAX_SLOWER_FRACTION);
        if self.p2p_gbps + f64::EPSILON >= floor {
            TransferPath::P2p
        } else {
            TransferPath::HostStaging
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub target_device: i32,
    pub draft_device: i32,
    pub vision_device: Option<i32>,
    pub warm_kv: KvTierKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvTierKind {
    Vram,
    Dram,
    Nvme,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p2p_selected_only_when_copy_ok_and_not_much_slower() {
        let ready = BandwidthBenchmark {
            bytes: 4 << 20,
            p2p_gbps: 12.0,
            host_staging_gbps: 10.0,
            p2p_us: 10.0,
            host_staging_us: 12.0,
            p2p_copy_ok: true,
        };
        assert_eq!(ready.selected_path(), TransferPath::P2p);

        let within_margin = BandwidthBenchmark {
            p2p_gbps: 9.2,
            host_staging_gbps: 10.0,
            p2p_copy_ok: true,
            ..ready
        };
        assert_eq!(within_margin.selected_path(), TransferPath::P2p);

        let much_slower = BandwidthBenchmark {
            p2p_gbps: 5.0,
            host_staging_gbps: 10.0,
            p2p_copy_ok: true,
            ..ready
        };
        assert_eq!(much_slower.selected_path(), TransferPath::HostStaging);

        let copy_failed = BandwidthBenchmark {
            p2p_gbps: 40.0,
            p2p_copy_ok: false,
            ..ready
        };
        assert_eq!(copy_failed.selected_path(), TransferPath::HostStaging);
    }

    #[test]
    fn parse_device_selector() {
        assert_eq!(DeviceSelector::parse("cpu").unwrap(), DeviceSelector::Cpu);
        assert_eq!(
            DeviceSelector::parse("hip:1").unwrap(),
            DeviceSelector::Hip(1)
        );
        assert_eq!(
            DeviceSelector::parse("cuda:0").unwrap(),
            DeviceSelector::Cuda(0)
        );
        assert_eq!(
            DeviceSelector::parse("vulkan:1").unwrap(),
            DeviceSelector::Vulkan(1)
        );
        assert_eq!(DeviceSelector::parse("auto").unwrap(), DeviceSelector::Auto);
        assert!(DeviceSelector::parse("tpu:0").is_err());
    }
}
