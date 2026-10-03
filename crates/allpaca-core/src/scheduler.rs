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
    /// Wall time of a speculative step (drafting + verify) by rows verified
    /// (index rows - 1): EMA in microseconds and sample count.
    step_us: Vec<f64>,
    step_n: Vec<u32>,
    /// Steps handed out by [`Self::step_depth`] (probe cadence).
    step_calls: u64,
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
            step_us: Vec::new(),
            step_n: Vec::new(),
            step_calls: 0,
        }
    }

    /// Depth for the next speculative step: the adaptive depth, except that
    /// every [`PROBE_EVERY`]th step tries a neighbouring depth whose cost has
    /// not been measured, so [`Self::record_acceptance`] can weigh it.
    pub fn step_depth(&mut self) -> usize {
        self.step_calls = self.step_calls.wrapping_add(1);
        let depth = self.draft_depth;
        if !self.flags.adaptive_draft_depth || self.step_calls % PROBE_EVERY != 0 {
            return depth;
        }
        let unmeasured = |d: usize| self.step_n.get(d).copied().unwrap_or(0) < STEP_MIN_SAMPLES;
        [depth + 1, depth.wrapping_sub(1)]
            .into_iter()
            .find(|&d| (self.min_depth..=self.max_depth).contains(&d) && unmeasured(d))
            .unwrap_or(depth)
    }

    /// Wall time of one speculative step that verified `rows` rows (drafts
    /// + 1), drafting included.
    pub fn record_step_time(&mut self, rows: usize, us: f64) {
        if rows == 0 || !us.is_finite() || us <= 0.0 {
            return;
        }
        if self.step_us.len() < rows {
            self.step_us.resize(rows, 0.0);
            self.step_n.resize(rows, 0);
        }
        let i = rows - 1;
        self.step_us[i] = if self.step_n[i] == 0 { us } else { 0.8 * self.step_us[i] + 0.2 * us };
        self.step_n[i] = self.step_n[i].saturating_add(1);
    }

    /// Measured (or linearly interpolated) step time for `rows` rows.
    fn step_cost(&self, rows: usize) -> Option<f64> {
        let known: Vec<(f64, f64)> = self
            .step_us
            .iter()
            .zip(&self.step_n)
            .enumerate()
            .filter(|(_, (_, &n))| n >= STEP_MIN_SAMPLES)
            .map(|(i, (&us, _))| ((i + 1) as f64, us))
            .collect();
        if let Some(&(_, us)) = known.iter().find(|(r, _)| *r == rows as f64) {
            return Some(us);
        }
        if known.len() < 2 {
            return None;
        }
        // Least-squares line through the measured row counts.
        let n = known.len() as f64;
        let (sx, sy) = known.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
        let (mx, my) = (sx / n, sy / n);
        let (sxy, sxx) = known
            .iter()
            .fold((0.0, 0.0), |(a, b), (x, y)| (a + (x - mx) * (y - my), b + (x - mx) * (x - mx)));
        let slope = if sxx > 0.0 { (sxy / sxx).max(0.0) } else { 0.0 };
        Some((my + slope * (rows as f64 - mx)).max(1.0))
    }

    /// DSpark / MiMo-style depth: of the current depth and its neighbours,
    /// the one with the most expected tokens per microsecond, where a span of
    /// d drafts yields 1 + reach(1) + .. + reach(d) tokens and costs the
    /// measured step time for d + 1 rows. `None` while those costs are
    /// unknown.
    fn throughput_depth(&self, q: &[f64]) -> Option<usize> {
        let tokens = |d: usize| 1.0 + (1..=d).map(|k| q.iter().take(k).product::<f64>()).sum::<f64>();
        let lo = self.draft_depth.saturating_sub(1).max(self.min_depth);
        let hi = (self.draft_depth + 1).min(self.max_depth);
        let mut best = (self.draft_depth, f64::MIN);
        for d in lo..=hi {
            let rate = tokens(d) / self.step_cost(d + 1)?;
            // Ties keep the current depth.
            if rate > best.1 * (1.0 + 1e-9) || (d == self.draft_depth && rate >= best.1) {
                best = (d, rate);
            }
        }
        Some(best.0)
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
        let want = self
            .throughput_depth(&q)
            .unwrap_or_else(|| next_draft_depth(&q, self.draft_depth, self.min_depth, self.max_depth));
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
/// Timed steps a row count needs before its cost is trusted.
const STEP_MIN_SAMPLES: u32 = 3;
/// Every this many steps, try a neighbouring depth whose cost is unknown.
const PROBE_EVERY: u64 = 16;
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

    /// One timed span: depth from `step_depth` (it may probe a neighbour),
    /// per-position acceptance `p`, wall time `cost(rows)` for drafts + verify.
    fn timed_step(s: &mut Scheduler, p: f64, cost: &dyn Fn(usize) -> f64, rng: &mut u64) {
        let depth = s.step_depth() as u64;
        let mut accepted = 0;
        while accepted < depth {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if ((*rng >> 11) as f64 / (1u64 << 53) as f64) >= p {
                break;
            }
            accepted += 1;
        }
        s.record_step_time(depth as usize + 1, cost(depth as usize + 1));
        s.record_acceptance(accepted, depth);
    }

    fn settled_depth(p: f64, cost: &dyn Fn(usize) -> f64) -> f64 {
        let mut s = adaptive(2, 8);
        let mut rng = 21u64;
        for _ in 0..300 {
            timed_step(&mut s, p, cost, &mut rng);
        }
        let mut sum = 0usize;
        for _ in 0..200 {
            timed_step(&mut s, p, cost, &mut rng);
            sum += s.draft_depth;
        }
        sum as f64 / 200.0
    }

    /// DSpark / MiMo-style: depth maximises expected tokens per second. When
    /// every verified row costs as much as a whole step (RPC verify with no
    /// batching win), extra drafts never pay: depth falls to the minimum,
    /// where the reach thresholds alone would hold it at 3.
    #[test]
    fn throughput_depth_stays_shallow_when_each_row_costs_a_step() {
        let mean = settled_depth(0.70, &|rows| rows as f64 * 1000.0);
        assert!(mean <= 2.3, "mean depth {mean}");
    }

    /// When verify rows are nearly free (a bandwidth-bound GEMV reads the
    /// weights once for all rows), drafting deeper pays even for drafts that
    /// usually miss.
    #[test]
    fn throughput_depth_goes_deep_when_rows_are_nearly_free() {
        let mean = settled_depth(0.70, &|rows| 1000.0 + 15.0 * rows as f64);
        assert!(mean >= 5.5, "mean depth {mean}");
    }

    /// With timings for a single row count the cost of other depths is
    /// unknown: the scheduler tries a neighbouring depth now and then,
    /// otherwise it keeps its depth.
    #[test]
    fn throughput_depth_probes_unmeasured_neighbours() {
        let mut s = adaptive(2, 8);
        s.draft_depth = 4;
        for _ in 0..20 {
            s.record_step_time(5, 1000.0);
        }
        let depths: Vec<usize> = (0..64).map(|_| s.step_depth()).collect();
        let probes = depths.iter().filter(|&&d| d != 4).count();
        assert!(probes >= 1, "never probed: {depths:?}");
        assert!(probes <= 8, "probed {probes} of 64 steps: {depths:?}");
        assert!(depths.iter().all(|&d| (3..=5).contains(&d)));
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
