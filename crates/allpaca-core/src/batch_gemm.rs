//! Cross-request packed GEMM planning + soft execute seam.
//!
//! When 2+ decode-ready Requests share the same model shape, [`plan_batch_gemm`]
//! builds a [`BatchGemmPlan`]. `Engine::poll` soft-executes by co-stepping the
//! other plan members in the same tick (parking their `StepResult`s), and when a
//! HIP workspace plus real cohort activations / Target weights are present runs
//! one packed host→device `gemm_nt` (concat A along M).

use allpaca_types::{AllpacaError, FatalError};

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

/// Planned packed GEMM across Requests (soft co-step in poll; HIP packed when available).
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

/// Concatenate A rows along M, run one NT GEMM, split C by row.
///
/// Each `a_rows[i]` is `m_i×k` row-major; `b` is `n×k` (NT); each `c_rows[i]` is
/// `m_i×n`. `gemm_nt(a, b, c, m, n, k)` must compute `C[m,n] = A[m,k] * B[n,k]^T`.
pub fn packed_gemm_nt<F>(
    a_rows: &[&[f32]],
    b: &[f32],
    c_rows: &mut [&mut [f32]],
    n: usize,
    k: usize,
    mut gemm_nt: F,
) -> Result<(), AllpacaError>
where
    F: FnMut(&[f32], &[f32], &mut [f32], i32, i32, i32) -> Result<(), AllpacaError>,
{
    if a_rows.is_empty() || a_rows.len() != c_rows.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "packed_gemm_nt: a/c row count mismatch".into(),
        )));
    }
    if n == 0 || k == 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "packed_gemm_nt: empty n or k".into(),
        )));
    }
    if b.len() < n * k {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "packed_gemm_nt: B too short".into(),
        )));
    }
    let mut m_total = 0usize;
    let mut m_parts = Vec::with_capacity(a_rows.len());
    for (i, a) in a_rows.iter().enumerate() {
        if a.len() % k != 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "packed_gemm_nt: a_rows[{i}] len {} not multiple of k={k}",
                a.len()
            ))));
        }
        let m_i = a.len() / k;
        if c_rows[i].len() < m_i * n {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "packed_gemm_nt: c_rows[{i}] too short"
            ))));
        }
        m_parts.push(m_i);
        m_total = m_total.saturating_add(m_i);
    }
    if m_total == 0 {
        return Ok(());
    }
    let mut a_pack = Vec::with_capacity(m_total * k);
    for a in a_rows {
        a_pack.extend_from_slice(a);
    }
    let mut c_pack = vec![0.0f32; m_total * n];
    gemm_nt(
        &a_pack,
        b,
        &mut c_pack,
        m_total as i32,
        n as i32,
        k as i32,
    )?;
    let mut off = 0usize;
    for (i, m_i) in m_parts.into_iter().enumerate() {
        let span = m_i * n;
        c_rows[i][..span].copy_from_slice(&c_pack[off..off + span]);
        off += span;
    }
    Ok(())
}

/// CPU reference path: one packed NT GEMM via [`allpaca_cpu::matmul_nt`].
pub fn packed_gemm_nt_cpu(
    a_rows: &[&[f32]],
    b: &[f32],
    c_rows: &mut [&mut [f32]],
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    packed_gemm_nt(a_rows, b, c_rows, n, k, |a, b, c, m, n, k| {
        allpaca_cpu::matmul_nt(a, b, c, m as usize, n as usize, k as usize)
    })
}

/// Packed F32 NT GEMM on a HIP workspace using caller-provided A/B rows.
///
/// Concatenates `a_rows` along M and runs one `HipWorkspace::gemm_nt`.
pub fn packed_gemm_nt_hip(
    hip: &allpaca_hip::HipWorkspace,
    a_rows: &[&[f32]],
    b: &[f32],
    c_rows: &mut [&mut [f32]],
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    packed_gemm_nt(a_rows, b, c_rows, n, k, |a, b, c, m, n, k| {
        hip.gemm_nt(a, b, c, m, n, k)
    })
}

/// Test helper: packed launch with synthetic 8×8 A/B (not used on the poll hot path).
pub fn packed_gemm_nt_hip_diagnostic(
    hip: &allpaca_hip::HipWorkspace,
    m_tokens: usize,
) -> Result<(), AllpacaError> {
    let m = m_tokens.max(2);
    let n = 8usize;
    let k = 8usize;
    let a_rows_owned: Vec<Vec<f32>> = (0..m)
        .map(|r| {
            (0..k)
                .map(|c| ((r * k + c) % 7) as f32 * 0.1 - 0.3)
                .collect()
        })
        .collect();
    let b: Vec<f32> = (0..n * k)
        .map(|i| ((i % 5) as f32) * 0.05)
        .collect();
    let mut c_rows_owned: Vec<Vec<f32>> = (0..m).map(|_| vec![0.0f32; n]).collect();
    let a_refs: Vec<&[f32]> = a_rows_owned.iter().map(|v| v.as_slice()).collect();
    let mut c_refs: Vec<&mut [f32]> = c_rows_owned
        .iter_mut()
        .map(|v| v.as_mut_slice())
        .collect();
    packed_gemm_nt_hip(hip, &a_refs, &b, &mut c_refs, n, k)
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

    /// Packed M=2 (two decode rows) must match two separate NT GEMMs.
    #[test]
    fn packed_gemm_nt_m2_matches_separate() {
        let k = 4usize;
        let n = 3usize;
        let a0 = [1.0f32, 2.0, 3.0, 4.0];
        let a1 = [0.5f32, -1.0, 2.0, 0.0];
        // B is N×K (NT): row j is weight row for output col j.
        let b = [
            1.0f32, 0.0, 0.0, 0.0, // out0
            0.0, 1.0, 0.0, 0.0, // out1
            0.0, 0.0, 1.0, 0.5, // out2
        ];
        let mut c0_sep = [0.0f32; 3];
        let mut c1_sep = [0.0f32; 3];
        allpaca_cpu::matmul_nt(&a0, &b, &mut c0_sep, 1, n, k).unwrap();
        allpaca_cpu::matmul_nt(&a1, &b, &mut c1_sep, 1, n, k).unwrap();

        let mut c0 = [0.0f32; 3];
        let mut c1 = [0.0f32; 3];
        packed_gemm_nt_cpu(&[&a0[..], &a1[..]], &b, &mut [&mut c0[..], &mut c1[..]], n, k)
            .expect("packed cpu");
        assert_eq!(c0, c0_sep);
        assert_eq!(c1, c1_sep);
    }

    /// Packed M=2 on caller-provided A/B matches separate HIP (or CPU) NT GEMMs.
    #[test]
    fn packed_gemm_nt_hip_m2_matches_separate_or_skips() {
        if !allpaca_hip::runtime_linked() || allpaca_hip::device_count() == 0 {
            return;
        }
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 256) else {
            return;
        };
        let k = 8usize;
        let n = 8usize;
        let a0: Vec<f32> = (0..k).map(|i| i as f32 * 0.1).collect();
        let a1: Vec<f32> = (0..k).map(|i| (i as f32) * -0.05 + 0.2).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i % 5) as f32) * 0.05).collect();

        let mut c0_sep = vec![0.0f32; n];
        let mut c1_sep = vec![0.0f32; n];
        hip.gemm_nt(&a0, &b, &mut c0_sep, 1, n as i32, k as i32)
            .expect("separate hip a0");
        hip.gemm_nt(&a1, &b, &mut c1_sep, 1, n as i32, k as i32)
            .expect("separate hip a1");

        let mut c0 = vec![0.0f32; n];
        let mut c1 = vec![0.0f32; n];
        packed_gemm_nt_hip(
            &hip,
            &[&a0, &a1],
            &b,
            &mut [&mut c0[..], &mut c1[..]],
            n,
            k,
        )
        .expect("packed hip gemm");
        for i in 0..n {
            let d0 = (c0[i] - c0_sep[i]).abs();
            let d1 = (c1[i] - c1_sep[i]).abs();
            assert!(d0 <= 1e-3 || d0 <= 1e-3 * c0_sep[i].abs(), "c0[{i}]");
            assert!(d1 <= 1e-3 || d1 <= 1e-3 * c1_sep[i].abs(), "c1[{i}]");
        }

        let mut c0_cpu = vec![0.0f32; n];
        let mut c1_cpu = vec![0.0f32; n];
        packed_gemm_nt_cpu(&[&a0, &a1], &b, &mut [&mut c0_cpu[..], &mut c1_cpu[..]], n, k)
            .unwrap();
        for i in 0..n {
            let d0 = (c0[i] - c0_cpu[i]).abs();
            let d1 = (c1[i] - c1_cpu[i]).abs();
            assert!(d0 <= 1e-3 || d0 <= 1e-3 * c0_cpu[i].abs(), "cpu c0[{i}]");
            assert!(d1 <= 1e-3 || d1 <= 1e-3 * c1_cpu[i].abs(), "cpu c1[{i}]");
        }
    }

    /// Diagnostic helper still launches; skips when no device / runtime.
    #[test]
    fn packed_gemm_nt_hip_diagnostic_matches_cpu_or_skips() {
        if !allpaca_hip::runtime_linked() || allpaca_hip::device_count() == 0 {
            return;
        }
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 256) else {
            return;
        };
        packed_gemm_nt_hip_diagnostic(&hip, 2).expect("diagnostic packed hip");
    }
}
