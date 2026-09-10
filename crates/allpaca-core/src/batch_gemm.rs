//! Cross-request packed GEMM planning + soft execute seam.
//!
//! When 2+ decode-ready Requests share the same model shape, [`plan_batch_gemm`]
//! builds a [`BatchGemmPlan`]. `Engine::poll` soft-executes by co-stepping the
//! other plan members in the same tick (parking their `StepResult`s). True fused
//! multi-request HIP kernels remain deferred.

/// Model dims that must match for a packed cross-request GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelShapeKey {
    pub n_layers: usize,
    pub dim: usize,
    pub vocab_size: usize,
}

/// Prefill vs decode phase for batch eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchGemmPhase {
    Prefill,
    Decode,
}

/// One runnable Request considered for a packed GEMM plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchGemmCandidate {
    pub request_id: u64,
    pub phase: BatchGemmPhase,
    pub shape: ModelShapeKey,
    /// Tokens this request would contribute to M (decode = 1).
    pub m_tokens: usize,
}

/// Planned packed GEMM across Requests (soft co-step in poll; fused HIP deferred).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchGemmPlan {
    pub request_ids: Vec<u64>,
    pub m_tokens: usize,
    pub reason: &'static str,
}

/// Plan members other than `primary` (soft co-step peers).
pub fn batch_gemm_other_ids(plan: &BatchGemmPlan, primary: u64) -> Vec<u64> {
    plan.request_ids
        .iter()
        .copied()
        .filter(|&id| id != primary)
        .collect()
}

/// Plan a packed GEMM when 2+ decode-ready Requests share one model shape.
///
/// Ineligible: fewer than two decode peers, prefill/decode mix in the decode
/// cohort (prefill is filtered out), or mismatched shapes among decode peers.
pub fn plan_batch_gemm(candidates: &[BatchGemmCandidate]) -> Option<BatchGemmPlan> {
    let decode: Vec<&BatchGemmCandidate> = candidates
        .iter()
        .filter(|c| matches!(c.phase, BatchGemmPhase::Decode) && c.m_tokens > 0)
        .collect();
    if decode.len() < 2 {
        return None;
    }
    let shape = decode[0].shape;
    if decode.iter().any(|c| c.shape != shape) {
        return None;
    }
    let request_ids: Vec<u64> = decode.iter().map(|c| c.request_id).collect();
    let m_tokens: usize = decode.iter().map(|c| c.m_tokens).sum();
    Some(BatchGemmPlan {
        request_ids,
        m_tokens,
        reason: "same_shape_decode_cohort",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_a() -> ModelShapeKey {
        ModelShapeKey {
            n_layers: 2,
            dim: 32,
            vocab_size: 64,
        }
    }

    fn shape_b() -> ModelShapeKey {
        ModelShapeKey {
            n_layers: 4,
            dim: 64,
            vocab_size: 128,
        }
    }

    #[test]
    fn plans_two_decode_same_shape() {
        let cands = [
            BatchGemmCandidate {
                request_id: 1,
                phase: BatchGemmPhase::Decode,
                shape: shape_a(),
                m_tokens: 1,
            },
            BatchGemmCandidate {
                request_id: 2,
                phase: BatchGemmPhase::Decode,
                shape: shape_a(),
                m_tokens: 1,
            },
        ];
        let plan = plan_batch_gemm(&cands).expect("eligible");
        assert_eq!(plan.request_ids, vec![1, 2]);
        assert_eq!(plan.m_tokens, 2);
        assert_eq!(plan.reason, "same_shape_decode_cohort");
    }

    #[test]
    fn rejects_different_shapes() {
        let cands = [
            BatchGemmCandidate {
                request_id: 1,
                phase: BatchGemmPhase::Decode,
                shape: shape_a(),
                m_tokens: 1,
            },
            BatchGemmCandidate {
                request_id: 2,
                phase: BatchGemmPhase::Decode,
                shape: shape_b(),
                m_tokens: 1,
            },
        ];
        assert!(plan_batch_gemm(&cands).is_none());
    }

    #[test]
    fn rejects_prefill_decode_mix_without_two_decode() {
        let cands = [
            BatchGemmCandidate {
                request_id: 1,
                phase: BatchGemmPhase::Prefill,
                shape: shape_a(),
                m_tokens: 8,
            },
            BatchGemmCandidate {
                request_id: 2,
                phase: BatchGemmPhase::Decode,
                shape: shape_a(),
                m_tokens: 1,
            },
        ];
        assert!(
            plan_batch_gemm(&cands).is_none(),
            "prefill peers are filtered; one decode is not enough"
        );
    }

    #[test]
    fn ignores_prefill_when_two_decode_eligible() {
        let cands = [
            BatchGemmCandidate {
                request_id: 1,
                phase: BatchGemmPhase::Prefill,
                shape: shape_a(),
                m_tokens: 8,
            },
            BatchGemmCandidate {
                request_id: 2,
                phase: BatchGemmPhase::Decode,
                shape: shape_a(),
                m_tokens: 1,
            },
            BatchGemmCandidate {
                request_id: 3,
                phase: BatchGemmPhase::Decode,
                shape: shape_a(),
                m_tokens: 1,
            },
        ];
        let plan = plan_batch_gemm(&cands).expect("two decode peers");
        assert_eq!(plan.request_ids, vec![2, 3]);
        assert_eq!(plan.m_tokens, 2);
    }

    #[test]
    fn single_decode_ineligible() {
        let cands = [BatchGemmCandidate {
            request_id: 7,
            phase: BatchGemmPhase::Decode,
            shape: shape_a(),
            m_tokens: 1,
        }];
        assert!(plan_batch_gemm(&cands).is_none());
    }

    #[test]
    fn other_ids_exclude_primary() {
        let plan = BatchGemmPlan {
            request_ids: vec![10, 20, 30],
            m_tokens: 3,
            reason: "same_shape_decode_cohort",
        };
        assert_eq!(batch_gemm_other_ids(&plan, 20), vec![10, 30]);
        assert_eq!(batch_gemm_other_ids(&plan, 99), vec![10, 20, 30]);
    }
}
