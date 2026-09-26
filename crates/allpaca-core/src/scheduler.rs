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
    /// Decayed counts per draft position k (index k - 1): spans that reached
    /// the k-th draft, and those where it was accepted.
    pos_hits: Vec<f64>,
    pos_trials: Vec<f64>,
    /// Consecutive verifies whose estimate asked for a deeper / shallower span.
    want_deeper: u32,
    want_shallower: u32,
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
            pos_hits: Vec::new(),
            pos_trials: Vec::new(),
            want_deeper: 0,
            want_shallower: 0,
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
        if !self.flags.adaptive_draft_depth || proposed == 0 {
            return;
        }
        // A span stops at its first rejected draft: drafts 1..=accepted
        // landed, and draft accepted + 1 (if proposed) was tried and missed.
        let proposed = proposed as usize;
        let accepted = (accepted as usize).min(proposed);
        if self.pos_trials.len() < proposed {
            self.pos_trials.resize(proposed, 0.0);
            self.pos_hits.resize(proposed, 0.0);
        }
        for (hits, trials) in self.pos_hits.iter_mut().zip(self.pos_trials.iter_mut()) {
            *hits *= DEPTH_DECAY;
            *trials *= DEPTH_DECAY;
        }
        for k in 0..proposed.min(accepted + 1) {
            self.pos_trials[k] += 1.0;
            if k < accepted {
                self.pos_hits[k] += 1.0;
            }
        }
        // Each position's prior is the previous position's estimate, so an
        // untried next draft is assumed to land as often as the last one.
        let mut prior = DEPTH_PRIOR_P;
        let q: Vec<f64> = (0..self.max_depth)
            .map(|k| {
                let h = self.pos_hits.get(k).copied().unwrap_or(0.0);
                let t = self.pos_trials.get(k).copied().unwrap_or(0.0);
                prior = (h + prior * DEPTH_PRIOR_TRIALS) / (t + DEPTH_PRIOR_TRIALS);
                prior
            })
            .collect();
        let want = next_draft_depth(&q, self.draft_depth, self.min_depth, self.max_depth);
        // Act only when the estimate has asked for the same move several
        // verifies in a row, so sampling noise does not move depth.
        let (deeper, shallower) = (want > self.draft_depth, want < self.draft_depth);
        self.want_deeper = if deeper { self.want_deeper + 1 } else { 0 };
        self.want_shallower = if shallower { self.want_shallower + 1 } else { 0 };
        if self.want_deeper >= DEPTH_PATIENCE || self.want_shallower >= DEPTH_PATIENCE {
            self.draft_depth = want;
            self.want_deeper = 0;
            self.want_shallower = 0;
        }
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
            .filter(|d| d.is_gpu() && !d.is_rpc())
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
/// Weight of the previous acceptance counts at each verify (~20-step memory).
const DEPTH_DECAY: f64 = 0.95;
/// Draft 1 starts as if it had seen `DEPTH_PRIOR_TRIALS` drafts at this rate;
/// each later position starts from the one before it.
const DEPTH_PRIOR_P: f64 = 0.6;
const DEPTH_PRIOR_TRIALS: f64 = 12.0;
/// Verifies in a row that must ask for the same depth change before it happens.
const DEPTH_PATIENCE: u32 = 4;
/// Draft one token deeper once that token would land with this probability.
const DEPTH_GROW_P: f64 = 0.33;
/// Drop the last draft once its chance of landing falls below this.
const DEPTH_SHRINK_P: f64 = 0.25;

/// Moves draft depth at most one step. `q[k - 1]` is the estimated chance
/// the k-th draft lands given the earlier ones did, so a span reaches draft
/// k with probability q[0] * .. * q[k - 1]. Grow while the next draft would
/// still be reached often enough, shrink once the last one rarely is; the
/// gap between the thresholds keeps depth steady under sampling noise. The
/// old rule stepped by 2 on each span's raw accept rate and swung 2..8. On
/// Qwen3.8-27B MTP (later drafts land less often) this settles at 3, the
/// measured best.
pub fn next_draft_depth(q: &[f64], depth: usize, lo: usize, hi: usize) -> usize {
    let depth = depth.clamp(lo, hi);
    let reach = |k: usize| q.iter().take(k).product::<f64>();
    if depth < hi && reach(depth + 1) >= DEPTH_GROW_P {
        depth + 1
    } else if depth > lo && reach(depth) < DEPTH_SHRINK_P {
        depth - 1
    } else {
        depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adaptive(min_depth: usize, max_depth: usize) -> Scheduler {
        let mut flags = FeatureFlags::from_mode(PipelineMode::Asymmetric);
        flags.adaptive_draft_depth = true;
        let spec = SpeculativeSection {
            min_depth,
            max_depth,
            ..SpeculativeSection::default()
        };
        Scheduler::new(PipelineMode::Asymmetric, flags, &spec)
    }

    /// Verify one span at the current depth where each draft lands with
    /// probability `p` (deterministic LCG), stopping at the first miss.
    fn verify_step(s: &mut Scheduler, p: f64, rng: &mut u64) {
        verify_step_by_position(s, &[p], rng);
    }

    /// Like [`verify_step`], with the k-th draft landing with probability
    /// `p[k - 1]` given the earlier ones did (the last entry repeats).
    fn verify_step_by_position(s: &mut Scheduler, p: &[f64], rng: &mut u64) {
        let depth = s.draft_depth as u64;
        let mut accepted = 0;
        while accepted < depth {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pk = p[(accepted as usize).min(p.len() - 1)];
            if ((*rng >> 11) as f64 / (1u64 << 53) as f64) >= pk {
                break;
            }
            accepted += 1;
        }
        s.record_acceptance(accepted, depth);
    }

    #[test]
    fn adaptive_depth_weighs_later_drafts_by_their_own_acceptance() {
        // Measured on Qwen3.8-27B MTP: later drafts land less often, so the
        // chance of reaching draft 4 is ~0.19 even though the pooled per-draft
        // rate is ~0.7. Depth must settle at 3, not drift to 4.
        let profile = [0.70, 0.76, 0.64, 0.56];
        let mut s = adaptive(2, 8);
        let mut rng = 5u64;
        for _ in 0..100 {
            verify_step_by_position(&mut s, &profile, &mut rng);
        }
        let mut depths = Vec::new();
        for _ in 0..200 {
            verify_step_by_position(&mut s, &profile, &mut rng);
            depths.push(s.draft_depth);
        }
        let at3 = depths.iter().filter(|&&d| d == 3).count();
        let changes = depths.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(at3 >= 130, "depth 3 on {at3}/200 steps: {depths:?}");
        assert!(changes <= 6, "{changes} depth changes in 200 steps: {depths:?}");
    }

    #[test]
    fn adaptive_depth_recovers_quickly_after_an_easy_start() {
        // A reply that opens by restating the prompt drafts perfectly for a
        // while; once acceptance drops to normal, depth must come back down
        // within a few spans instead of lingering deep.
        let mut s = adaptive(2, 8);
        for _ in 0..10 {
            let d = s.draft_depth as u64;
            s.record_acceptance(d, d);
        }
        let easy = s.draft_depth;
        let profile = [0.70, 0.76, 0.64, 0.56];
        let mut rng = 9u64;
        for _ in 0..15 {
            verify_step_by_position(&mut s, &profile, &mut rng);
        }
        assert!(s.draft_depth <= 4, "depth {} 15 spans after the easy start (was {easy})", s.draft_depth);
    }

    #[test]
    fn adaptive_depth_settles_where_the_next_draft_is_still_likely() {
        // Qwen3.8-27B MTP: per-draft acceptance ~0.70, measured best depth 3.
        let mut s = adaptive(2, 8);
        let mut rng = 7u64;
        for _ in 0..100 {
            verify_step(&mut s, 0.70, &mut rng);
        }
        let mut depths = Vec::new();
        for _ in 0..200 {
            verify_step(&mut s, 0.70, &mut rng);
            depths.push(s.draft_depth);
        }
        let changes = depths.windows(2).filter(|w| w[0] != w[1]).count();
        let mean = depths.iter().sum::<usize>() as f64 / depths.len() as f64;
        assert!((2.5..=3.5).contains(&mean), "mean depth {mean}, depths {depths:?}");
        assert!(changes <= 20, "{changes} depth changes in 200 steps: {depths:?}");
    }

    #[test]
    fn adaptive_depth_holds_through_one_rejected_span() {
        let mut s = adaptive(2, 8);
        let mut rng = 11u64;
        for _ in 0..100 {
            verify_step(&mut s, 0.74, &mut rng);
        }
        let settled = s.draft_depth;
        s.record_acceptance(0, settled as u64);
        assert_eq!(s.draft_depth, settled, "one rejected span must not move depth");
    }

    #[test]
    fn adaptive_depth_follows_sustained_acceptance() {
        let mut s = adaptive(2, 8);
        let mut rng = 3u64;
        for _ in 0..60 {
            verify_step(&mut s, 0.97, &mut rng);
        }
        assert_eq!(s.draft_depth, 8, "sustained full accepts reach max_depth");
        for _ in 0..60 {
            verify_step(&mut s, 0.3, &mut rng);
        }
        assert_eq!(s.draft_depth, 2, "sustained rejects fall to min_depth");
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
    fn placement_targets_single_cuda_gpu() {
        let devices = vec![
            DeviceInfo::cpu(),
            DeviceInfo {
                id: 0,
                kind: DeviceKind::CudaGpu,
                name: "RTX 3060".into(),
                memory_bytes: 12 << 30,
                bandwidth_gbps: 360.0,
                peers: vec![],
            },
        ];
        let flags = FeatureFlags::from_mode(PipelineMode::MobileTiered);
        let spec = SpeculativeSection::default();
        let s = Scheduler::new(PipelineMode::MobileTiered, flags, &spec);
        let p = s.placement(&devices, None);
        assert_eq!(p.target_device, 0);
        assert_eq!(p.draft_device, -1);
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
