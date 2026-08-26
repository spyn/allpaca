//! Weight working set: dense pin-prefix + ring, and the MoE `q*` miss split.
//!
//! Not KV. Cyclic dense layers must not use LRU (hit rate is zero on a full walk).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use allpaca_types::{AllpacaError, FatalError, MissPolicy};

use crate::model::LayerWeights;
use crate::packed::{temp_pack_path, PackedStore};
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
        0.0
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

    pub fn unpack_bytes(
        bytes: &[u8],
        n_norm: usize,
        n_gate: usize,
        n_up: usize,
        n_down: usize,
    ) -> Result<Self, AllpacaError> {
        let need = n_norm
            .saturating_add(n_gate)
            .saturating_add(n_up)
            .saturating_add(n_down)
            .saturating_mul(4);
        if bytes.len() < need {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "packed expert truncated: have {} want {need}",
                bytes.len()
            ))));
        }
        let mut off = 0usize;
        let mut take = |n: usize| -> Result<Vec<f32>, AllpacaError> {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                let chunk: [u8; 4] = bytes[off..off + 4].try_into().map_err(|_| {
                    AllpacaError::Fatal(FatalError::ModelLoad("packed expert slice".into()))
                })?;
                v.push(f32::from_le_bytes(chunk));
                off += 4;
            }
            Ok(v)
        };
        Ok(Self {
            ffn_norm: take(n_norm)?,
            w_gate: take(n_gate)?,
            w_up: take(n_up)?,
            w_down: take(n_down)?,
        })
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
    store: Option<PackedStore>,
    ring: Mutex<RingState>,
    binds: AtomicU64,
    hits: AtomicU64,
}

impl std::fmt::Debug for WeightWorkingSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightWorkingSet")
            .field("plan", &self.plan)
            .field("packed", &self.store.is_some())
            .finish()
    }
}

impl WeightWorkingSet {
    pub fn detach_unpinned(
        model_layers: &mut [LayerWeights],
        plan: DenseTrunkPlan,
    ) -> Result<Arc<Self>, AllpacaError> {
        let n = model_layers.len();
        let mut pool = Vec::with_capacity(n);
        let mut packed = Vec::new();
        for (i, layer) in model_layers.iter_mut().enumerate() {
            if i < plan.pinned_prefix {
                pool.push(LayerFfn::default());
            } else {
                let ffn = LayerFfn::take_from(layer);
                packed.push((
                    ExpertId {
                        layer: i as u32,
                        expert: 0,
                    },
                    ffn,
                ));
                pool.push(LayerFfn::default());
            }
        }
        let store = if packed.is_empty() {
            None
        } else {
            Some(PackedStore::pack(
                &temp_pack_path(),
                packed.iter().map(|(id, ffn)| (*id, ffn)),
            )?)
        };
        let nring = plan.ring_slots.max(1);
        let ring = RingState {
            slots: (0..nring).map(|_| LayerFfn::default()).collect(),
            bound: vec![None; nring],
            cursor: 0,
        };
        Ok(Arc::new(Self {
            plan,
            pool,
            store,
            ring: Mutex::new(ring),
            binds: AtomicU64::new(0),
            hits: AtomicU64::new(0),
        }))
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

    pub fn nvme_reads(&self) -> u64 {
        self.store.as_ref().map(|s| s.reads()).unwrap_or(0)
    }

    pub fn nvme_bytes(&self) -> u64 {
        self.store.as_ref().map(|s| s.bytes_read()).unwrap_or(0)
    }

    pub fn nvme_direct(&self) -> bool {
        self.store.as_ref().map(|s| s.is_direct()).unwrap_or(false)
    }

    pub fn load_layer(&self, layer: usize) -> Result<LayerFfn, AllpacaError> {
        if let Ok(ring) = self.ring.lock() {
            if let Some(slot) = ring.bound.iter().position(|&b| b == Some(layer)) {
                return Ok(ring.slots[slot].clone());
            }
        }
        if let Some(store) = &self.store {
            return store.read_expert(ExpertId {
                layer: layer as u32,
                expert: 0,
            });
        }
        self.pool.get(layer).cloned().filter(|f| !f.w_gate.is_empty()).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "FFN layer {layer} missing from weight working set"
            )))
        })
    }

    fn fill_from_source(&self, layer: usize) -> Result<LayerFfn, AllpacaError> {
        if let Some(store) = &self.store {
            return store.read_expert(ExpertId {
                layer: layer as u32,
                expert: 0,
            });
        }
        self.pool.get(layer).cloned().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "FFN layer {layer} missing from weight working set"
            )))
        })
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
        ring.slots[slot] = self.fill_from_source(layer)?;
        ring.bound[slot] = Some(layer);
        if n >= 2 {
            let next = layer + 1;
            if next < self.plan.n_layers && !self.is_pinned(next) {
                let other = (slot + 1) % n;
                if ring.bound[other] != Some(next) {
                    if let Ok(ffn) = self.fill_from_source(next) {
                        ring.slots[other] = ffn;
                        ring.bound[other] = Some(next);
                    }
                }
            }
        }
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

/// Complete `(layer, expert)` identity. FTW bank index is `layer * n_experts + expert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertId {
    pub fn bank(self, n_experts: u32) -> u32 {
        self.layer.saturating_mul(n_experts).saturating_add(self.expert)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissPlan {
    pub hits: Vec<ExpertId>,
    pub fill: Vec<ExpertId>,
    pub cpu: Vec<ExpertId>,
}

/// How many LRU slots `cache_bytes` buys. `cache_bytes == 0` means no GPU/hot cache.
pub fn expert_lru_slots(expert_bytes: usize, cache_bytes: usize) -> usize {
    if cache_bytes == 0 || expert_bytes == 0 {
        0
    } else {
        (cache_bytes / expert_bytes).max(1)
    }
}

/// How many DRAM slots `host_bytes` buys. `host_bytes == 0` means unlimited RAM (no NVMe).
pub fn host_pool_slots(expert_bytes: usize, host_bytes: usize) -> usize {
    expert_lru_slots(expert_bytes, host_bytes)
}

fn unique_in_order(ids: &[ExpertId]) -> Vec<ExpertId> {
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        if !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

struct ExpertLruInner {
    bound: Vec<Option<ExpertId>>,
    used_at: Vec<u64>,
    slots: Vec<LayerFfn>,
    clock: u64,
}

/// Host pool is source of truth. These slots are the GPU (or hot DRAM) LRU.
pub struct ExpertLru {
    n_slots: usize,
    policy: MissPolicy,
    b_pcie: f64,
    b_host: f64,
    inner: Mutex<ExpertLruInner>,
    hits: AtomicU64,
    misses: AtomicU64,
    fills: AtomicU64,
    cpu_runs: AtomicU64,
}

impl std::fmt::Debug for ExpertLru {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertLru")
            .field("n_slots", &self.n_slots)
            .field("policy", &self.policy)
            .finish()
    }
}

impl ExpertLru {
    pub fn new(n_slots: usize, policy: MissPolicy) -> Arc<Self> {
        Self::with_bandwidth(n_slots, policy, 0.0, 0.0)
    }

    pub fn with_bandwidth(
        n_slots: usize,
        policy: MissPolicy,
        b_pcie: f64,
        b_host: f64,
    ) -> Arc<Self> {
        let n = n_slots.max(1);
        Arc::new(Self {
            n_slots: n,
            policy,
            b_pcie,
            b_host,
            inner: Mutex::new(ExpertLruInner {
                bound: vec![None; n],
                used_at: vec![0; n],
                slots: vec![LayerFfn::default(); n],
                clock: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            fills: AtomicU64::new(0),
            cpu_runs: AtomicU64::new(0),
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
    pub fn fills(&self) -> u64 {
        self.fills.load(Ordering::Relaxed)
    }
    pub fn cpu_runs(&self) -> u64 {
        self.cpu_runs.load(Ordering::Relaxed)
    }

    pub fn record_slots(metrics: &Metrics, n_slots: usize) {
        metrics
            .expert_slots
            .store(n_slots as u64, Ordering::Relaxed);
    }

    pub fn plan(&self, needed: &[ExpertId]) -> MissPlan {
        let unique = unique_in_order(needed);
        let inner = self.inner.lock().ok();
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        if let Some(inner) = inner.as_ref() {
            for &id in &unique {
                if inner.bound.iter().any(|&b| b == Some(id)) {
                    hits.push(id);
                } else {
                    misses.push(id);
                }
            }
        } else {
            misses = unique;
        }
        let q = match self.policy {
            MissPolicy::Fill => misses.len(),
            MissPolicy::Cpu => 0,
            MissPolicy::Qstar => qstar_fill_count(misses.len(), self.b_pcie, self.b_host),
        };
        let fill = misses.iter().copied().take(q).collect();
        let cpu = misses.iter().copied().skip(q).collect();
        MissPlan { hits, fill, cpu }
    }

    pub fn admit(&self, id: ExpertId, host: &LayerFfn) -> Result<(), AllpacaError> {
        let mut inner = self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "expert LRU mutex poisoned".into(),
            ))
        })?;
        inner.clock = inner.clock.saturating_add(1);
        if let Some(slot) = inner.bound.iter().position(|&b| b == Some(id)) {
            inner.used_at[slot] = inner.clock;
            return Ok(());
        }
        let victim = inner
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                inner
                    .used_at
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
        inner.slots[victim] = host.clone();
        inner.bound[victim] = Some(id);
        inner.used_at[victim] = inner.clock;
        Ok(())
    }

    pub fn materialize(&self, id: ExpertId, pool: &LayerFfn) -> Result<LayerFfn, AllpacaError> {
        let mut inner = self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "expert LRU mutex poisoned".into(),
            ))
        })?;
        inner.clock = inner.clock.saturating_add(1);
        if let Some(slot) = inner.bound.iter().position(|&b| b == Some(id)) {
            inner.used_at[slot] = inner.clock;
            return Ok(inner.slots[slot].clone());
        }
        Ok(pool.clone())
    }

    pub fn note_plan(&self, plan: &MissPlan) {
        self.hits
            .fetch_add(plan.hits.len() as u64, Ordering::Relaxed);
        let miss_n = (plan.fill.len() + plan.cpu.len()) as u64;
        self.misses.fetch_add(miss_n, Ordering::Relaxed);
        self.fills
            .fetch_add(plan.fill.len() as u64, Ordering::Relaxed);
        self.cpu_runs
            .fetch_add(plan.cpu.len() as u64, Ordering::Relaxed);
    }

    /// True when `n_layers` full expert sets fit in the LRU (prefill double-buffer).
    pub fn holds_layers(&self, n_experts: usize, n_layers: usize) -> bool {
        n_experts > 0 && n_layers > 0 && self.n_slots >= n_experts.saturating_mul(n_layers)
    }

    /// Admit every expert of `layer`. Prefill uses this instead of on-demand `q*`.
    pub fn admit_layer(&self, layer: u32, experts: &[LayerFfn]) -> Result<(), AllpacaError> {
        let mut fills = 0u64;
        let mut hits = 0u64;
        for (e, host) in experts.iter().enumerate() {
            let id = ExpertId {
                layer,
                expert: e as u32,
            };
            let already = self.plan(&[id]).hits.len() == 1;
            self.admit(id, host)?;
            if already {
                hits += 1;
            } else {
                fills += 1;
            }
        }
        self.hits.fetch_add(hits, Ordering::Relaxed);
        self.misses.fetch_add(fills, Ordering::Relaxed);
        self.fills.fetch_add(fills, Ordering::Relaxed);
        Ok(())
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

    fn ffn_layer(gate: f32) -> crate::model::LayerWeights {
        crate::model::LayerWeights {
            attn_norm: vec![],
            wq: vec![],
            wk: vec![],
            wv: vec![],
            wo: vec![],
            ffn_norm: vec![1.0],
            w_gate: vec![gate],
            w_up: vec![gate + 1.0],
            w_down: vec![gate + 2.0],
            w_router: vec![],
            experts: vec![],
        }
    }

    #[test]
    fn unpinned_layers_page_from_packed_trunk() {
        let mut layers = vec![ffn_layer(1.0), ffn_layer(10.0), ffn_layer(20.0)];
        let plan = DenseTrunkPlan {
            n_layers: 3,
            pinned_prefix: 0,
            ring_slots: 1,
            slot_bytes: DIRECT_ALIGN,
        };
        let ws = WeightWorkingSet::detach_unpinned(&mut layers, plan).unwrap();
        assert!(ws.host_ffn(0).is_none(), "unpinned FFN must leave RAM");
        let a = ws.bind(0).unwrap();
        assert_eq!(a.w_gate(), &[1.0]);
        drop(a);
        let b = ws.bind(1).unwrap();
        assert_eq!(b.w_down(), &[12.0]);
        drop(b);
        assert!(
            ws.nvme_reads() >= 2,
            "one ring slot over two layers must miss to the packed trunk"
        );
    }

    #[test]
    fn spare_ring_slot_prefetches_next_layer() {
        let mut layers = vec![ffn_layer(1.0), ffn_layer(10.0), ffn_layer(20.0)];
        let plan = DenseTrunkPlan {
            n_layers: 3,
            pinned_prefix: 0,
            ring_slots: 2,
            slot_bytes: DIRECT_ALIGN,
        };
        let ws = WeightWorkingSet::detach_unpinned(&mut layers, plan).unwrap();
        let _ = ws.bind(0).unwrap();
        let reads_after_bind0 = ws.nvme_reads();
        let b = ws.bind(1).unwrap();
        assert_eq!(b.w_gate(), &[10.0]);
        assert!(
            ws.ring_hits() >= 1,
            "prefetch of L+1 into the spare slot must hit on the next bind"
        );
        assert_eq!(
            ws.nvme_reads(),
            reads_after_bind0,
            "hit must not re-read the packed trunk"
        );
    }

    #[test]
    fn qstar_fills_all_when_host_cannot_outrun_pcie() {
        assert_eq!(qstar_fill_count(8, 50.0, 50.0), 8);
        assert_eq!(qstar_fill_count(8, 52.7, 77.3), 5);
        assert_eq!(qstar_fill_count(8, 11.8, 47.5), 2);
        assert_eq!(qstar_fill_count(4, 1.0, 100.0), 1, "always keep warming");
        assert_eq!(qstar_fill_count(8, 0.0, 50.0), 1, "unmeasured PCIe fills one, rest CPU");
        assert_eq!(qstar_fill_count(0, 50.0, 80.0), 0);
    }

    #[test]
    fn expert_lru_slots_zero_budget_means_no_cache() {
        assert_eq!(expert_lru_slots(24_704, 0), 0);
        assert_eq!(expert_lru_slots(24_704, 64), 1);
        assert_eq!(expert_lru_slots(100, 250), 2);
    }

    #[test]
    fn expert_lru_hits_repeating_ids_not_a_cyclic_layer_walk() {
        let lru = ExpertLru::new(2, MissPolicy::Fill);
        let dummy = LayerFfn::default();
        let a = ExpertId { layer: 0, expert: 1 };
        let b = ExpertId { layer: 0, expert: 3 };
        for id in [a, b, a, b, a] {
            let plan = lru.plan(&[id]);
            for f in &plan.fill {
                lru.admit(*f, &dummy).unwrap();
            }
            lru.note_plan(&plan);
        }
        assert!(lru.hits() >= 3, "short-range expert reuse must hit LRU");
        assert!(lru.misses() >= 2);
    }

    #[test]
    fn qstar_splits_unique_misses_fill_then_cpu() {
        let lru = ExpertLru::with_bandwidth(1, MissPolicy::Qstar, 1.0, 100.0);
        let dummy = LayerFfn::default();
        let misses = [
            ExpertId { layer: 0, expert: 0 },
            ExpertId { layer: 0, expert: 1 },
            ExpertId { layer: 0, expert: 2 },
            ExpertId { layer: 0, expert: 3 },
        ];
        let plan = lru.plan(&misses);
        assert_eq!(plan.hits.len(), 0);
        assert_eq!(plan.fill.len(), 1, "q* always fills at least one");
        assert_eq!(plan.cpu.len(), 3);
        lru.admit(plan.fill[0], &dummy).unwrap();
        let again = lru.plan(&misses);
        assert_eq!(again.hits, vec![plan.fill[0]]);
    }

    #[test]
    fn cpu_miss_policy_never_admits() {
        let lru = ExpertLru::new(2, MissPolicy::Cpu);
        let plan = lru.plan(&[ExpertId { layer: 1, expert: 2 }]);
        assert!(plan.fill.is_empty());
        assert_eq!(plan.cpu.len(), 1);
    }

    #[test]
    fn prefill_admits_a_full_layer_when_slots_fit() {
        let lru = ExpertLru::new(4, MissPolicy::Fill);
        assert!(lru.holds_layers(4, 1));
        assert!(!lru.holds_layers(4, 2));
        let experts = vec![LayerFfn::default(); 4];
        lru.admit_layer(0, &experts).unwrap();
        let needed: Vec<_> = (0..4)
            .map(|e| ExpertId {
                layer: 0,
                expert: e,
            })
            .collect();
        let plan = lru.plan(&needed);
        assert_eq!(plan.hits.len(), 4);
        assert!(plan.fill.is_empty());
        assert!(lru.fills() >= 4);
    }

    #[test]
    fn double_buffer_keeps_current_and_next_layer() {
        let lru = ExpertLru::new(8, MissPolicy::Fill);
        assert!(lru.holds_layers(4, 2));
        let experts = vec![LayerFfn::default(); 4];
        lru.admit_layer(0, &experts).unwrap();
        lru.admit_layer(1, &experts).unwrap();
        assert_eq!(
            lru.plan(&[ExpertId {
                layer: 0,
                expert: 2
            }])
            .hits
            .len(),
            1
        );
        assert_eq!(
            lru.plan(&[ExpertId {
                layer: 1,
                expert: 3
            }])
            .hits
            .len(),
            1
        );
    }
}
