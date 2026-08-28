//! Cross-request layer-staggered micro-batching (GEM).
//!
//! Under concurrency, FreeToken-style co-execution binds CPU and GPU to the same
//! layer of the same request. Better utilization: GPU runs request A at layer
//! `l+1` while CPU finishes cache-missed experts of request B at layer `l`.
//!
//! Vertical: `plan_micro_batch` on multi-request `poll`, same-tick partner
//! `step` with parked `PollEvent` observability (prefer-next is fallback only).

use std::sync::Mutex;

/// Request progress phase for stagger planning (mirrors engine poll phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaggerPhase {
    Prefill,
    Decode,
}

/// One request's position in a staggered step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaggerSlot {
    pub request_id: u64,
    pub layer: usize,
    pub on_gpu: bool,
}

/// Pair of concurrent work items that may run without a shared barrier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroBatchPlan {
    pub gpu: Option<StaggerSlot>,
    pub cpu: Option<StaggerSlot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveLayerHint {
    pub request_id: u64,
    pub phase: StaggerPhase,
    pub layer: usize,
    /// True when this request has pending CPU expert work at `layer`.
    pub cpu_expert_pending: bool,
}

/// Active cross-request layer stagger for the current dual poll tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaggerActive {
    pub gpu_request: u64,
    pub gpu_layer: usize,
    pub cpu_request: u64,
    pub cpu_layer: usize,
}

/// Per-poll layer stagger context shared with MoE forward.
#[derive(Default)]
pub struct LayerStagger {
    inner: Mutex<Option<StaggerActive>>,
    /// Snapshot of the CPU partner's MoE router input at `cpu_layer`.
    partner_norm: Mutex<Vec<f32>>,
}

impl LayerStagger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin(
        &self,
        gpu_request: u64,
        gpu_layer: usize,
        cpu_request: u64,
        cpu_layer: usize,
        partner_norm: &[f32],
    ) {
        if let Ok(mut g) = self.inner.lock() {
            *g = Some(StaggerActive {
                gpu_request,
                gpu_layer,
                cpu_request,
                cpu_layer,
            });
        }
        if let Ok(mut n) = self.partner_norm.lock() {
            *n = partner_norm.to_vec();
        }
    }

    pub fn end(&self) {
        if let Ok(mut g) = self.inner.lock() {
            *g = None;
        }
        if let Ok(mut n) = self.partner_norm.lock() {
            n.clear();
        }
    }

    pub fn active(&self) -> Option<StaggerActive> {
        self.inner.lock().ok().and_then(|g| *g)
    }

    pub fn partner_norm(&self) -> Vec<f32> {
        self.partner_norm
            .lock()
            .map(|n| n.clone())
            .unwrap_or_default()
    }
}

/// Pick a GPU slot and a CPU slot from distinct requests when possible.
/// CPU partner is pinned one layer behind the GPU slot when both are decoding.
pub fn plan_micro_batch(active: &[ActiveLayerHint]) -> MicroBatchPlan {
    let gpu = active
        .iter()
        .find(|h| matches!(h.phase, StaggerPhase::Decode | StaggerPhase::Prefill))
        .map(|h| StaggerSlot {
            request_id: h.request_id,
            layer: h.layer,
            on_gpu: true,
        });
    let cpu = active
        .iter()
        .filter(|h| h.cpu_expert_pending)
        .find(|h| gpu.is_none_or(|g| g.request_id != h.request_id))
        .map(|h| {
            let layer = gpu
                .filter(|_g| matches!(h.phase, StaggerPhase::Decode))
                .map(|g| g.layer.saturating_sub(1))
                .unwrap_or(h.layer);
            StaggerSlot {
                request_id: h.request_id,
                layer,
                on_gpu: false,
            }
        })
        .or_else(|| {
            gpu.and_then(|g| {
                active
                    .iter()
                    .find(|h| {
                        matches!(h.phase, StaggerPhase::Decode) && h.request_id != g.request_id
                    })
                    .map(|h| StaggerSlot {
                        request_id: h.request_id,
                        layer: g.layer.saturating_sub(1),
                        on_gpu: false,
                    })
            })
        });
    MicroBatchPlan { gpu, cpu }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stagger_pairs_gpu_and_cpu_across_requests() {
        let hints = [
            ActiveLayerHint {
                request_id: 1,
                phase: StaggerPhase::Decode,
                layer: 4,
                cpu_expert_pending: false,
            },
            ActiveLayerHint {
                request_id: 2,
                phase: StaggerPhase::Decode,
                layer: 3,
                cpu_expert_pending: true,
            },
        ];
        let plan = plan_micro_batch(&hints);
        assert_eq!(plan.gpu.map(|g| g.request_id), Some(1));
        assert_eq!(plan.cpu.map(|c| c.request_id), Some(2));
        assert_eq!(plan.cpu.map(|c| c.layer), Some(3), "cpu one layer behind gpu");
        assert_ne!(
            plan.gpu.map(|g| g.request_id),
            plan.cpu.map(|c| c.request_id)
        );
    }

    #[test]
    fn stagger_pairs_two_decode_requests_without_cpu_expert_pending() {
        let hints = [
            ActiveLayerHint {
                request_id: 1,
                phase: StaggerPhase::Decode,
                layer: 2,
                cpu_expert_pending: false,
            },
            ActiveLayerHint {
                request_id: 2,
                phase: StaggerPhase::Decode,
                layer: 1,
                cpu_expert_pending: false,
            },
        ];
        let plan = plan_micro_batch(&hints);
        assert_eq!(plan.gpu.map(|g| g.request_id), Some(1));
        assert_eq!(plan.cpu.map(|c| c.request_id), Some(2));
        assert_eq!(plan.cpu.map(|c| c.layer), Some(1));
    }

    #[test]
    fn single_request_does_not_force_cpu_pair() {
        let hints = [ActiveLayerHint {
            request_id: 7,
            phase: StaggerPhase::Prefill,
            layer: 0,
            cpu_expert_pending: true,
        }];
        let plan = plan_micro_batch(&hints);
        assert_eq!(plan.gpu.map(|g| g.request_id), Some(7));
        assert!(plan.cpu.is_none(), "same-request CPU pair is not staggered");
    }

    #[test]
    fn layer_stagger_tracks_active_tick() {
        let s = LayerStagger::new();
        s.begin(1, 5, 2, 4, &[1.0, 2.0]);
        let a = s.active().unwrap();
        assert_eq!(a.gpu_request, 1);
        assert_eq!(a.cpu_layer, 4);
        s.end();
        assert!(s.active().is_none());
    }
}
