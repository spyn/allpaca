//! Cross-request layer-staggered micro-batching (GEM).
//!
//! Under concurrency, FreeToken-style co-execution binds CPU and GPU to the same
//! layer of the same request. Better utilization: GPU runs request A at layer
//! `l+1` while CPU finishes cache-missed experts of request B at layer `l`.
//!
//! Vertical: `plan_micro_batch` runs on each multi-request `poll`, and MoE
//! forwards run CPU expert MLPs in parallel with the GPU expert loop.

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

/// Pick a GPU slot and a CPU slot from distinct requests when possible.
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
        .map(|h| StaggerSlot {
            request_id: h.request_id,
            layer: h.layer,
            on_gpu: false,
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
        assert_ne!(
            plan.gpu.map(|g| g.request_id),
            plan.cpu.map(|c| c.request_id)
        );
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
}
