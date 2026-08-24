//! Weight working set: dense pin-prefix + ring, and the MoE `q*` miss split.
//!
//! Not KV. Cyclic dense layers must not use LRU (hit rate is zero on a full walk).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use allpaca_types::{AllpacaError, FatalError};

use crate::model::LayerWeights;
use crate::telemetry::Metrics;

/// Alignment for ring slots (O_DIRECT-style). Prefix layers keep exact sizes.
pub const DIRECT_ALIGN: usize = 4096;

pub fn align_up(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    n.div_ceil(align) * align
}

/// How many of `misses` unique experts to fill over PCIe; the rest run on CPU.
/// Always fills at least one miss so the cache keeps warming.
pub fn qstar_fill_count(misses: usize, b_pcie: f64, b_host: f64) -> usize {
    if misses == 0 {
        return 0;
    }
    if !b_host.is_finite() || b_host <= 0.0 {
        return misses;
    }
    let ratio = if b_pcie.is_finite() && b_pcie > 0.0 {
        (b_pcie / b_host).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let q = ((misses as f64) * ratio).round() as usize;
    q.clamp(1, misses)
}

/// Hits an LRU cache of `n_slots` records over `cycles` walks of `0..n_layers`.
/// A cyclic scan with `n_slots < n_layers` has zero hits after the first fill.
#[allow(dead_code)]
pub fn lru_hits_on_cyclic_scan(n_layers: usize, n_slots: usize, cycles: usize) -> u64 {
    if n_layers == 0 || n_slots == 0 || cycles == 0 {
        return 0;
    }
    let mut slots: Vec<Option<usize>> = vec![None; n_slots];
    let mut used_at: Vec<u64> = vec![0; n_slots];
    let mut clock = 0u64;
    let mut hits = 0u64;
    for _ in 0..cycles {
        for layer in 0..n_layers {
            clock += 1;
            if let Some(i) = slots.iter().position(|&s| s == Some(layer)) {
                hits += 1;
                used_at[i] = clock;
                continue;
            }
            let victim = slots
                .iter()
                .position(|s| s.is_none())
                .unwrap_or_else(|| {
                    used_at
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, t)| *t)
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                });
            slots[victim] = Some(layer);
            used_at[victim] = clock;
        }
    }
    hits
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseTrunkPlan {
    pub n_layers: usize,
    pub pinned_prefix: usize,
    pub ring_slots: usize,
    pub slot_bytes: usize,
}

impl DenseTrunkPlan {
    /// `budget_bytes == 0` means unlimited: pin every layer, no ring.
    pub fn plan(layer_bytes: &[usize], budget_bytes: usize, requested_ring_slots: usize) -> Self {
        let n_layers = layer_bytes.len();
        if n_layers == 0 || budget_bytes == 0 {
            return Self {
                n_layers,
                pinned_prefix: n_layers,
                ring_slots: 0,
                slot_bytes: 0,
            };
        }
        let max_run = layer_bytes.iter().copied().max().unwrap_or(0);
        let mut slot = align_up(max_run.max(1), DIRECT_ALIGN);
        let want_ring = requested_ring_slots.clamp(1, 2);
        let mut npin = 0usize;
        let mut nring = want_ring;
        for _ in 0..8 {
            let ring_cost = slot.saturating_mul(nring);
            let avail = budget_bytes.saturating_sub(ring_cost);
            let mut n = 0usize;
            let mut used = 0usize;
            while n < n_layers && used.saturating_add(layer_bytes[n]) <= avail {
                used += layer_bytes[n];
                n += 1;
            }
            let mut need = 0usize;
            for &b in &layer_bytes[n..] {
                need = need.max(b);
            }
            if need == 0 {
                // Everything pinned: no ring.
                npin = n;
                nring = 0;
                slot = DIRECT_ALIGN;
                break;
            }
            let next_slot = align_up(need, DIRECT_ALIGN);
            let mut next_ring = want_ring;
            let prefix = used;
            if prefix.saturating_add(next_slot.saturating_mul(2)) > budget_bytes {
                next_ring = 1;
            }
            if n == npin && next_slot == slot && next_ring == nring {
                break;
            }
            npin = n;
            slot = next_slot;
            nring = next_ring;
        }
        if npin < n_layers && nring == 0 {
            nring = 1;
        }
        Self {
            n_layers,
            pinned_prefix: npin,
            ring_slots: nring,
            slot_bytes: if nring == 0 { 0 } else { slot },
        }
    }

    pub fn streams(&self) -> bool {
        self.ring_slots > 0 && self.pinned_prefix < self.n_layers
    }

    pub fn prefix_hits_per_cycle(&self) -> usize {
        self.pinned_prefix
    }
}

#[derive(Debug, Clone, Default)]
pub struct LayerFfn {
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl LayerFfn {
    pub fn take_from(layer: &mut LayerWeights) -> Self {
        Self {
            ffn_norm: std::mem::take(&mut layer.ffn_norm),
            w_gate: std::mem::take(&mut layer.w_gate),
            w_up: std::mem::take(&mut layer.w_up),
            w_down: std::mem::take(&mut layer.w_down),
        }
    }

    pub fn bytes(&self) -> usize {
        (self.ffn_norm.len() + self.w_gate.len() + self.w_up.len() + self.w_down.len()) * 4
    }

    pub fn pack_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes());
        for part in [&self.ffn_norm, &self.w_gate, &self.w_up, &self.w_down] {
            for v in part {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }
}

impl LayerWeights {
    pub fn ffn_bytes(&self) -> usize {
        (self.ffn_norm.len() + self.w_gate.len() + self.w_up.len() + self.w_down.len()) * 4
    }
}

struct RingState {
    slots: Vec<LayerFfn>,
    bound: Vec<Option<usize>>,
    cursor: usize,
}

pub struct BoundFfn<'a> {
    guard: std::sync::MutexGuard<'a, RingState>,
    slot: usize,
}

impl BoundFfn<'_> {
    pub fn ffn_norm(&self) -> &[f32] {
        &self.guard.slots[self.slot].ffn_norm
    }
    pub fn w_gate(&self) -> &[f32] {
        &self.guard.slots[self.slot].w_gate
    }
    pub fn w_up(&self) -> &[f32] {
        &self.guard.slots[self.slot].w_up
    }
    pub fn w_down(&self) -> &[f32] {
        &self.guard.slots[self.slot].w_down
    }
}

pub enum FfnRefs<'a> {
    Resident(&'a LayerWeights),
    Bound(BoundFfn<'a>),
}

impl FfnRefs<'_> {
    pub fn ffn_norm(&self) -> &[f32] {
        match self {
            Self::Resident(l) => &l.ffn_norm,
            Self::Bound(b) => b.ffn_norm(),
        }
    }
    pub fn w_gate(&self) -> &[f32] {
        match self {
            Self::Resident(l) => &l.w_gate,
            Self::Bound(b) => b.w_gate(),
        }
    }
    pub fn w_up(&self) -> &[f32] {
        match self {
            Self::Resident(l) => &l.w_up,
            Self::Bound(b) => b.w_up(),
        }
    }
    pub fn w_down(&self) -> &[f32] {
        match self {
            Self::Resident(l) => &l.w_down,
            Self::Bound(b) => b.w_down(),
        }
    }
}

pub struct WeightWorkingSet {
    pub plan: DenseTrunkPlan,
    pool: Vec<LayerFfn>,
    ring: Mutex<RingState>,
    binds: AtomicU64,
    hits: AtomicU64,
}

impl std::fmt::Debug for WeightWorkingSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightWorkingSet")
            .field("plan", &self.plan)
            .finish()
    }
}

impl WeightWorkingSet {
    pub fn detach_unpinned(model_layers: &mut [LayerWeights], plan: DenseTrunkPlan) -> Arc<Self> {
        let n = model_layers.len();
        let mut pool = Vec::with_capacity(n);
        for (i, layer) in model_layers.iter_mut().enumerate() {
            if i < plan.pinned_prefix {
                pool.push(LayerFfn::default());
            } else {
                pool.push(LayerFfn::take_from(layer));
            }
        }
        let nring = plan.ring_slots.max(1);
        let ring = RingState {
            slots: (0..nring).map(|_| LayerFfn::default()).collect(),
            bound: vec![None; nring],
            cursor: 0,
        };
        Arc::new(Self {
            plan,
            pool,
            ring: Mutex::new(ring),
            binds: AtomicU64::new(0),
            hits: AtomicU64::new(0),
        })
    }

    pub fn is_pinned(&self, layer: usize) -> bool {
        layer < self.plan.pinned_prefix
    }

    pub fn streams(&self) -> bool {
        self.plan.streams()
    }

    pub fn ring_binds(&self) -> u64 {
        self.binds.load(Ordering::Relaxed)
    }

    pub fn ring_hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn note_bind(&self, hit: bool) {
        self.binds.fetch_add(1, Ordering::Relaxed);
        if hit {
            self.hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn host_ffn(&self, layer: usize) -> Option<&LayerFfn> {
        self.pool.get(layer).filter(|f| !f.w_gate.is_empty())
    }

    pub fn bind(&self, layer: usize) -> Result<BoundFfn<'_>, AllpacaError> {
        let mut ring = self.ring.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "weight working-set mutex poisoned".into(),
            ))
        })?;
        self.binds.fetch_add(1, Ordering::Relaxed);
        if let Some(slot) = ring.bound.iter().position(|&b| b == Some(layer)) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(BoundFfn { guard: ring, slot });
        }
        let n = ring.slots.len();
        let slot = ring
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                let s = ring.cursor % n;
                ring.cursor = ring.cursor.wrapping_add(1);
                s
            });
        let src = self.pool.get(layer).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "FFN layer {layer} missing from weight working set"
            )))
        })?;
        ring.slots[slot] = src.clone();
        ring.bound[slot] = Some(layer);
        Ok(BoundFfn { guard: ring, slot })
    }

    pub fn record_plan(metrics: &Metrics, plan: &DenseTrunkPlan) {
        metrics
            .weight_pinned_layers
            .store(plan.pinned_prefix as u64, Ordering::Relaxed);
        metrics
            .weight_ring_slots
            .store(plan.ring_slots as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_budget_pins_every_layer() {
        let plan = DenseTrunkPlan::plan(&[100, 200, 300], 0, 2);
        assert_eq!(plan.pinned_prefix, 3);
        assert_eq!(plan.ring_slots, 0);
        assert!(!plan.streams());
    }

    #[test]
    fn small_budget_pins_none_and_keeps_one_ring_slot() {
        let layers = [1000usize, 2000, 3000];
        let plan = DenseTrunkPlan::plan(&layers, 100, 2);
        assert_eq!(plan.pinned_prefix, 0);
        assert_eq!(plan.ring_slots, 1);
        assert_eq!(plan.slot_bytes, align_up(3000, DIRECT_ALIGN));
        assert!(plan.streams());
    }

    #[test]
    fn prefix_plan_hits_where_lru_misses_a_cyclic_walk() {
        let layers = vec![1_000_000usize; 93];
        let slot = align_up(1_000_000, DIRECT_ALIGN);
        let budget = 90 * 1_000_000 + slot;
        let plan = DenseTrunkPlan::plan(&layers, budget, 1);
        assert_eq!(plan.pinned_prefix, 90);
        assert_eq!(plan.ring_slots, 1);
        assert_eq!(plan.prefix_hits_per_cycle(), 90);
        assert_eq!(
            lru_hits_on_cyclic_scan(93, 90, 2),
            0,
            "LRU of 90 slots over a 93-layer cycle has hit rate 0"
        );
    }

    #[test]
    fn second_ring_slot_only_when_it_fits() {
        let layers = vec![DIRECT_ALIGN; 4];
        let one = DIRECT_ALIGN;
        let pin0_two_slots = one * 2;
        let plan = DenseTrunkPlan::plan(&layers, pin0_two_slots, 2);
        assert_eq!(plan.pinned_prefix, 0);
        assert_eq!(plan.ring_slots, 2);
        let plan_tight = DenseTrunkPlan::plan(&layers, one + 10, 2);
        assert_eq!(plan_tight.ring_slots, 1);
    }

    #[test]
    fn qstar_fills_all_when_host_cannot_outrun_pcie() {
        assert_eq!(qstar_fill_count(8, 50.0, 50.0), 8);
        assert_eq!(qstar_fill_count(8, 52.7, 77.3), 5);
        assert_eq!(qstar_fill_count(8, 11.8, 47.5), 2);
        assert_eq!(qstar_fill_count(4, 1.0, 100.0), 1, "always keep warming");
        assert_eq!(qstar_fill_count(0, 50.0, 80.0), 0);
    }
}
