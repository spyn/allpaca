use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    HipGpu,
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
        matches!(self.kind, DeviceKind::HipGpu)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceSelector {
    #[default]
    Auto,
    Cpu,
    Hip(i32),
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
        Err(format!(
            "unknown device selector '{raw}' (expected cpu, hip:N, or auto)"
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandwidthBenchmark {
    pub bytes: u64,
    pub p2p_gbps: f64,
    pub host_staging_gbps: f64,
}

impl BandwidthBenchmark {
    pub fn faster_path(&self) -> TransferPath {
        if self.p2p_gbps > self.host_staging_gbps {
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
    fn parse_device_selector() {
        assert_eq!(DeviceSelector::parse("cpu").unwrap(), DeviceSelector::Cpu);
        assert_eq!(
            DeviceSelector::parse("hip:1").unwrap(),
            DeviceSelector::Hip(1)
        );
        assert_eq!(
            DeviceSelector::parse("auto").unwrap(),
            DeviceSelector::Auto
        );
        assert!(DeviceSelector::parse("cuda:0").is_err());
    }
}
