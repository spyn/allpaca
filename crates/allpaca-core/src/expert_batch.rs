//! Same-tick decode expert union for bulk H2D (GEM slice 2).

use std::collections::HashSet;
use std::sync::Mutex;

use crate::weight_set::ExpertId;

/// Per-poll-tick union of routed experts across micro-batch partners.
#[derive(Default)]
pub struct MicroBatchExpertUnion {
    inner: Mutex<UnionInner>,
}

#[derive(Default)]
struct UnionInner {
    tick: u64,
    /// First partner's routed ids for the current MoE layer (set before fill).
    pending: HashSet<ExpertId>,
    /// Active dual-step tick; zero when idle.
    dual_tick: u64,
}

impl MicroBatchExpertUnion {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_dual_step(&self, tick: u64) {
        if let Ok(mut g) = self.inner.lock() {
            g.tick = tick;
            g.dual_tick = tick;
            g.pending.clear();
        }
    }

    pub fn end_dual_step(&self) {
        if let Ok(mut g) = self.inner.lock() {
            g.dual_tick = 0;
            g.pending.clear();
        }
    }

    pub fn dual_active(&self) -> bool {
        self.inner.lock().map(|g| g.dual_tick != 0).unwrap_or(false)
    }

    /// Record routed experts from the first partner at `layer`; returns None until
    /// the second partner calls `union_with` for the same tick.
    pub fn stage_routed(&self, layer: u32, ids: &[ExpertId]) {
        if let Ok(mut g) = self.inner.lock() {
            if g.dual_tick == 0 {
                return;
            }
            g.pending.clear();
            for id in ids {
                if id.layer == layer {
                    g.pending.insert(*id);
                }
            }
        }
    }

    /// Union `ids` with staged partner ids when dual-step is active.
    pub fn union_with(&self, layer: u32, ids: &[ExpertId]) -> Vec<ExpertId> {
        let mut out: Vec<ExpertId> = ids.iter().copied().filter(|id| id.layer == layer).collect();
        if let Ok(mut g) = self.inner.lock() {
            if g.dual_tick != 0 {
                for id in g.pending.drain() {
                    if id.layer == layer && !out.contains(&id) {
                        out.push(id);
                    }
                }
            }
        }
        out.sort_by_key(|id| (id.layer, id.expert));
        out.dedup();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_merges_partner_routed_experts() {
        let u = MicroBatchExpertUnion::new();
        u.begin_dual_step(1);
        u.stage_routed(
            0,
            &[
                ExpertId {
                    layer: 0,
                    expert: 1,
                },
                ExpertId {
                    layer: 0,
                    expert: 3,
                },
            ],
        );
        let merged = u.union_with(
            0,
            &[
                ExpertId {
                    layer: 0,
                    expert: 2,
                },
                ExpertId {
                    layer: 0,
                    expert: 3,
                },
            ],
        );
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&ExpertId {
            layer: 0,
            expert: 1
        }));
        u.end_dual_step();
        assert!(!u.dual_active());
    }
}
