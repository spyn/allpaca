use allpaca_types::{
    DeviceInfo, DeviceKind, FeatureFlags, KvTierKind, PipelineMode, Placement, PollSchedule,
    SpeculativeSection, TaskCost,
};
use std::time::Instant;

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

    pub fn score(
        &self,
        cost: &TaskCost,
        sla_weight: f64,
        queue_penalty: f64,
        memory_penalty: f64,
        reuse_bonus: f64,
    ) -> f64 {
        sla_weight * cost.predicted_latency_us()
            + cost.h2d_us
            + cost.d2h_us
            + cost.p2p_us
            + queue_penalty
            + memory_penalty
            - reuse_bonus
    }

    pub fn record_acceptance(&mut self, accepted: u64, proposed: u64) {
        self.record_acceptance_with_pressure(accepted, proposed, 0);
    }

    /// Acceptance-aware depth, optionally shrunk under concurrent Request pressure
    /// (GEM research / US-02: target queue pressure).
    pub fn record_acceptance_with_pressure(
        &mut self,
        accepted: u64,
        proposed: u64,
        active_requests: usize,
    ) {
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
        if active_requests > 1 {
            self.draft_depth = self
                .draft_depth
                .saturating_sub(active_requests.saturating_sub(1))
                .max(self.min_depth);
        }
    }

    pub fn placement(&self, devices: &[DeviceInfo], vision_device: Option<i32>) -> Placement {
        let mut gpus: Vec<&DeviceInfo> = devices
            .iter()
            .filter(|d| d.kind == DeviceKind::HipGpu)
            .collect();
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
        // Resident draft: second GPU when present. P2P is not required.
        // One GPU / CPU-only falls back to CPU draft (id == cpu).
        let draft = gpus.get(1).map(|d| d.id).unwrap_or(cpu);
        Placement {
            target_device: target,
            draft_device: draft,
            vision_device,
            warm_kv: KvTierKind::Dram,
        }
    }

    /// Continuous batching: among runnable Requests, pick the longest-waiting.
    /// Ties break round-robin after `after_id`, then lowest id.
    pub fn pick_runnable(
        &self,
        candidates: &[(u64, Instant, bool)],
        after_id: Option<u64>,
        schedule: PollSchedule,
        consecutive_prefill: usize,
        max_consecutive_prefill: usize,
    ) -> Option<u64> {
        pick_runnable_phased(
            candidates,
            after_id,
            schedule,
            consecutive_prefill,
            max_consecutive_prefill,
        )
    }
}

/// Among runnable Requests, pick the one waiting longest since last progress.
/// When `schedule` is [`PollSchedule::DecodePriority`], decode/emitted Requests
/// win over prefilling peers regardless of wait time.
/// When `schedule` is [`PollSchedule::PrefillBudget`] and the consecutive prefill
/// streak has hit the cap, force a decode pick if any decode peer is runnable.
pub fn pick_runnable_phased(
    candidates: &[(u64, Instant, bool)],
    after_id: Option<u64>,
    schedule: PollSchedule,
    consecutive_prefill: usize,
    max_consecutive_prefill: usize,
) -> Option<u64> {
    let force_decode = matches!(schedule, PollSchedule::PrefillBudget)
        && max_consecutive_prefill > 0
        && consecutive_prefill >= max_consecutive_prefill
        && candidates.iter().any(|(_, _, prefill)| !prefill);
    let effective = if force_decode {
        PollSchedule::DecodePriority
    } else if matches!(schedule, PollSchedule::PrefillBudget) {
        PollSchedule::Fair
    } else {
        schedule
    };
    let narrowed: Vec<(u64, Instant)> = match effective {
        PollSchedule::Fair | PollSchedule::PrefillBudget => {
            candidates.iter().map(|(id, t, _)| (*id, *t)).collect()
        }
        PollSchedule::DecodePriority => {
            let decode: Vec<(u64, Instant)> = candidates
                .iter()
                .filter(|(_, _, prefill)| !prefill)
                .map(|(id, t, _)| (*id, *t))
                .collect();
            if decode.is_empty() {
                candidates.iter().map(|(id, t, _)| (*id, *t)).collect()
            } else {
                decode
            }
        }
    };
    pick_runnable(&narrowed, after_id)
}

/// Among runnable Requests, pick the one waiting longest since last progress.
/// Ties break by round-robin after `after_id` (exclusive), then by lowest id.
pub fn pick_runnable(candidates: &[(u64, Instant)], after_id: Option<u64>) -> Option<u64> {
    if candidates.is_empty() {
        return None;
    }
    let min_t = candidates.iter().map(|(_, t)| *t).min()?;
    let mut tied: Vec<u64> = candidates
        .iter()
        .filter(|(_, t)| *t == min_t)
        .map(|(id, _)| *id)
        .collect();
    tied.sort_unstable();
    if let Some(after) = after_id {
        if let Some(&id) = tied.iter().find(|&&id| id > after) {
            return Some(id);
        }
    }
    tied.first().copied()
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

    #[test]
    fn pick_runnable_prefers_oldest_wait_then_round_robin() {
        let t0 = Instant::now();
        let t1 = t0 + std::time::Duration::from_millis(1);
        let candidates = [(1u64, t0), (2u64, t0), (3u64, t1)];
        assert_eq!(pick_runnable(&candidates, None), Some(1));
        assert_eq!(pick_runnable(&candidates, Some(1)), Some(2));
        assert_eq!(
            pick_runnable(&candidates, Some(2)),
            Some(1),
            "wrap round-robin among equal waiters"
        );
    }

    #[test]
    fn decode_priority_prefers_decode_over_older_prefill() {
        let t0 = Instant::now();
        let t1 = t0 + std::time::Duration::from_millis(5);
        let phased = [(1u64, t0, true), (2u64, t1, false)];
        assert_eq!(
            pick_runnable_phased(&phased, None, PollSchedule::Fair, 0, 0),
            Some(1),
            "fair picks oldest waiter even when prefilling"
        );
        assert_eq!(
            pick_runnable_phased(&phased, None, PollSchedule::DecodePriority, 0, 0),
            Some(2),
            "decode priority must pick decoding Request over older prefill"
        );
    }

    #[test]
    fn placement_carries_vision_device() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo {
                id: 0,
                kind: DeviceKind::HipGpu,
                name: "a".into(),
                memory_bytes: 8 << 30,
                bandwidth_gbps: 400.0,
                peers: vec![1],
            },
            DeviceInfo {
                id: 1,
                kind: DeviceKind::HipGpu,
                name: "b".into(),
                memory_bytes: 8 << 30,
                bandwidth_gbps: 200.0,
                peers: vec![0],
            },
        ];
        let flags = FeatureFlags::from_mode(PipelineMode::Asymmetric);
        let spec = SpeculativeSection::default();
        let s = Scheduler::new(PipelineMode::Asymmetric, flags, &spec);
        let p = s.placement(&devices, Some(1));
        assert_eq!(p.vision_device, Some(1));
        assert_eq!(p.target_device, 0);
    }

    #[test]
    fn prefill_budget_forces_decode_after_streak() {
        let t0 = Instant::now();
        let t1 = t0 + std::time::Duration::from_millis(5);
        let phased = [(1u64, t0, true), (2u64, t1, false)];
        assert_eq!(
            pick_runnable_phased(&phased, None, PollSchedule::PrefillBudget, 0, 2),
            Some(1),
            "under budget, fair pick still wins"
        );
        assert_eq!(
            pick_runnable_phased(&phased, None, PollSchedule::PrefillBudget, 2, 2),
            Some(2),
            "at budget cap, decode must win"
        );
    }
}
