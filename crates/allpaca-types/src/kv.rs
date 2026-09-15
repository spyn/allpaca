use serde::{Deserialize, Serialize};

pub type PageId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KvResidency {
    Vram,
    PinnedDram,
    Nvme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceBufferKind {
    CpuVram,
    CpuPinnedDram,
    Hip,
    Cuda,
    Nvme,
}

/// Location of a KV page's bytes. HIP pointers stay opaque to the control plane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceBuffer {
    pub kind: DeviceBufferKind,
    pub device_id: i32,
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvPage {
    pub id: PageId,
    pub layer: u32,
    pub page_index: u32,
    pub filled: u32,
    pub capacity_tokens: u32,
    pub tier: KvResidency,
    pub refcount: u32,
    pub last_access: u64,
    pub predicted_reuse: f32,
    pub location: DeviceBuffer,
}

impl KvPage {
    /// `priority = predicted_reuse * recency_weight * layer_weight - transfer_cost`
    pub fn priority(&self, now: u64, recency_weight: f32, transfer_cost: f32) -> f32 {
        let recency = 1.0 / (1.0 + now.saturating_sub(self.last_access) as f32);
        let layer_weight = 1.0 / (1.0 + self.layer as f32 * 0.05);
        self.predicted_reuse * recency * recency_weight * layer_weight - transfer_cost
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvHandle {
    pub request_id: u64,
    pub epoch: u64,
    pub token_count: usize,
    pub pages: Vec<PageId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recent_page_scores_higher() {
        let loc = DeviceBuffer {
            kind: DeviceBufferKind::CpuVram,
            device_id: -1,
            offset: 0,
            len: 64,
        };
        let hot = KvPage {
            id: 1,
            layer: 0,
            page_index: 0,
            filled: 8,
            capacity_tokens: 16,
            tier: KvResidency::Vram,
            refcount: 1,
            last_access: 100,
            predicted_reuse: 1.0,
            location: loc.clone(),
        };
        let cold = KvPage {
            last_access: 1,
            id: 2,
            ..hot.clone()
        };
        assert!(hot.priority(100, 1.0, 0.0) > cold.priority(100, 1.0, 0.0));
    }
}
