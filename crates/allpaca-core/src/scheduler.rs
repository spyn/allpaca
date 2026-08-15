use allpaca_types::{
    DeviceInfo, DeviceKind, FeatureFlags, Placement, PipelineMode, SpeculativeSection, TaskCost,
    KvTierKind,
};

/// Heuristic scheduler. Modes only seed [`FeatureFlags`]; flags stay composable.
pub struct Scheduler {
    pub mode: PipelineMode,
    pub flags: FeatureFlags,
    pub draft_depth: usize,
    min_depth: usize,
    max_depth: usize,
    accepted: u64,
    proposed: u64,
}

impl Scheduler {
    pub fn new(mode: PipelineMode, flags: FeatureFlags, spec: &SpeculativeSection) -> Self {
        Self {
            mode,
            flags,
            draft_depth: spec.min_depth.max(1),
            min_depth: spec.min_depth.max(1),
            max_depth: spec.max_depth.max(1),
            accepted: 0,
            proposed: 0,
        }
    }

    pub fn score(&self, cost: &TaskCost, sla_weight: f64, queue_penalty: f64, memory_penalty: f64, reuse_bonus: f64) -> f64 {
        sla_weight * cost.predicted_latency_us()
            + cost.h2d_us
            + cost.d2h_us
            + cost.p2p_us
            + queue_penalty
            + memory_penalty
            - reuse_bonus
    }

    pub fn record_acceptance(&mut self, accepted: u64, proposed: u64) {
        self.accepted += accepted;
        self.proposed += proposed;
        if !self.flags.adaptive_draft_depth {
            return;
        }
        self.draft_depth = next_draft_depth(
            accepted,
            proposed,
            self.draft_depth,
            self.min_depth,
            self.max_depth,
        );
    }

    pub fn placement(&self, devices: &[DeviceInfo]) -> Placement {
        let mut gpus: Vec<&DeviceInfo> = devices.iter().filter(|d| d.kind == DeviceKind::HipGpu).collect();
        gpus.sort_by(|a, b| {
            b.bandwidth_gbps
                .partial_cmp(&a.bandwidth_gbps)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.memory_bytes.cmp(&a.memory_bytes))
        });
        let cpu = devices
            .iter()
            .find(|d| d.kind == DeviceKind::Cpu)
            .map(|d| d.id)
            .unwrap_or(-1);
        let target = gpus.first().map(|d| d.id).unwrap_or(cpu);
        let draft = if self.flags.asymmetric_drafting {
            gpus.get(1).map(|d| d.id).unwrap_or(cpu)
        } else {
            gpus.first().map(|d| d.id).unwrap_or(cpu)
        };
        Placement {
            target_device: target,
            draft_device: draft,
            vision_device: None,
            warm_kv: KvTierKind::Dram,
        }
    }
}

/// Acceptance-aware adaptive speculation (CODE_EXAMPLES example 6).
pub fn next_draft_depth(accepted: u64, proposed: u64, depth: usize, lo: usize, hi: usize) -> usize {
    if proposed == 0 {
        return lo;
    }
    let rate = accepted as f64 / proposed as f64;
    if rate > 0.80 {
        (depth + 2).min(hi)
    } else if rate < 0.45 {
        depth.saturating_sub(2).max(lo)
    } else {
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_depth_grows_on_high_accept() {
        assert_eq!(next_draft_depth(9, 10, 4, 2, 12), 6);
        assert_eq!(next_draft_depth(1, 10, 4, 2, 12), 2);
        assert_eq!(next_draft_depth(0, 0, 4, 2, 12), 2);
    }
}
