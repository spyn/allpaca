//! Cross-request packed GEMM planning + soft execute seam.
//!
//! When 2+ decode-ready Requests share the same model shape, [`plan_batch_gemm`]
//! builds a [`BatchGemmPlan`]. `Engine::poll` soft-executes by co-stepping the
//! other plan members in the same tick (parking their `StepResult`s), and when a
//! HIP workspace plus real cohort activations / Target weights are present runs
//! packed host→device `gemm_nt` (concat A along M). Packed **lm_head** outputs
//! are written into each Request's `pending_logits` for the sample path; packed
//! FFN ([`apply_packed_ffn`] / [`apply_packed_ffn_hip`]) runs gate/up → silu×up
//! → **w_down** (HIP poll path uses the HIP variant).

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

/// One NT weight matrix that shares the decode activation (`k == hidden dim`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackedGemmWeight<'a> {
    pub name: &'static str,
    pub b: &'a [f32],
    pub n: usize,
    pub k: usize,
}

/// Attn GEMMs whose B is `n×dim` (share last_hidden as A). Q/K/V/O only.
pub fn decode_layer_attn_shared_a_weights<'a>(
    dim: usize,
    wq: &'a [f32],
    wk: &'a [f32],
    wv: &'a [f32],
    wo: &'a [f32],
) -> Vec<PackedGemmWeight<'a>> {
    if dim == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(4);
    let push = |out: &mut Vec<PackedGemmWeight<'a>>, name: &'static str, b: &'a [f32]| {
        if b.len() < dim || b.len() % dim != 0 {
            return;
        }
        let n = b.len() / dim;
        out.push(PackedGemmWeight { name, b, n, k: dim });
    };
    push(&mut out, "wq", wq);
    push(&mut out, "wk", wk);
    push(&mut out, "wv", wv);
    push(&mut out, "wo", wo);
    out
}

/// Decode-layer shared-A GEMMs: attn Q/K/V/O plus SwiGLU gate/up.
///
/// Prefer [`decode_layer_attn_shared_a_weights`] + [`apply_packed_ffn_hip`] on the
/// poll path so `w_down` is included via the FFN helper.
pub fn decode_layer_shared_a_weights<'a>(
    dim: usize,
    wq: &'a [f32],
    wk: &'a [f32],
    wv: &'a [f32],
    wo: &'a [f32],
    w_gate: &'a [f32],
    w_up: &'a [f32],
) -> Vec<PackedGemmWeight<'a>> {
    let mut out = decode_layer_attn_shared_a_weights(dim, wq, wk, wv, wo);
    if dim == 0 {
        return out;
    }
    let push = |out: &mut Vec<PackedGemmWeight<'a>>, name: &'static str, b: &'a [f32]| {
        if b.len() < dim || b.len() % dim != 0 {
            return;
        }
        let n = b.len() / dim;
        out.push(PackedGemmWeight { name, b, n, k: dim });
    };
    push(&mut out, "w_gate", w_gate);
    push(&mut out, "w_up", w_up);
    out
}

/// FFN weights: gate/up share activation A (`k=dim`); `w_down` uses post-silu A (`k=ff_dim`).
#[derive(Debug, Clone, PartialEq)]
pub struct PackedFfnWeights<'a> {
    pub w_gate: Option<PackedGemmWeight<'a>>,
    pub w_up: Option<PackedGemmWeight<'a>>,
    pub w_down: Option<PackedGemmWeight<'a>>,
    pub ff_dim: usize,
}

/// List SwiGLU FFN weights for packed apply (gate/up shared-A, then down).
pub fn packed_ffn_shared_a<'a>(
    dim: usize,
    w_gate: &'a [f32],
    w_up: &'a [f32],
    w_down: &'a [f32],
) -> PackedFfnWeights<'a> {
    if dim == 0 {
        return PackedFfnWeights {
            w_gate: None,
            w_up: None,
            w_down: None,
            ff_dim: 0,
        };
    }
    let gate = if !w_gate.is_empty() && w_gate.len() % dim == 0 {
        let n = w_gate.len() / dim;
        Some(PackedGemmWeight {
            name: "w_gate",
            b: w_gate,
            n,
            k: dim,
        })
    } else {
        None
    };
    let up = if !w_up.is_empty() && w_up.len() % dim == 0 {
        let n = w_up.len() / dim;
        Some(PackedGemmWeight {
            name: "w_up",
            b: w_up,
            n,
            k: dim,
        })
    } else {
        None
    };
    let ff_dim = gate.map(|g| g.n).or_else(|| up.map(|u| u.n)).unwrap_or(0);
    let down = if ff_dim > 0 && !w_down.is_empty() && w_down.len() % ff_dim == 0 {
        let n = w_down.len() / ff_dim;
        Some(PackedGemmWeight {
            name: "w_down",
            b: w_down,
            n,
            k: ff_dim,
        })
    } else {
        None
    };
    PackedFfnWeights {
        w_gate: gate,
        w_up: up,
        w_down: down,
        ff_dim,
    }
}

/// Packed FFN: gate/up GEMM → `silu(gate)*up` per row → packed `w_down` into `out_rows`.
///
/// Each `a_rows[i]` / `out_rows[i]` is `m_i×dim`. Uses CPU NT GEMM.
pub fn apply_packed_ffn(
    a_rows: &[&[f32]],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    dim: usize,
    out_rows: &mut [&mut [f32]],
) -> Result<(), AllpacaError> {
    let ffn = packed_ffn_shared_a(dim, w_gate, w_up, w_down);
    let Some(gate_w) = ffn.w_gate else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn: missing w_gate".into(),
        )));
    };
    let Some(up_w) = ffn.w_up else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn: missing w_up".into(),
        )));
    };
    let Some(down_w) = ffn.w_down else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn: missing w_down".into(),
        )));
    };
    if gate_w.n != up_w.n || down_w.n != dim {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn: dim/ff_dim mismatch".into(),
        )));
    }
    let ff_dim = ffn.ff_dim;
    if a_rows.len() != out_rows.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn: a/out row count mismatch".into(),
        )));
    }
    let mut gate_rows: Vec<Vec<f32>> = a_rows
        .iter()
        .map(|a| {
            let m = a.len() / dim;
            vec![0.0f32; m * ff_dim]
        })
        .collect();
    let mut up_rows: Vec<Vec<f32>> = a_rows
        .iter()
        .map(|a| {
            let m = a.len() / dim;
            vec![0.0f32; m * ff_dim]
        })
        .collect();
    {
        let mut g_refs: Vec<&mut [f32]> = gate_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
        packed_gemm_nt_cpu(a_rows, gate_w.b, &mut g_refs, ff_dim, dim)?;
    }
    {
        let mut u_refs: Vec<&mut [f32]> = up_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
        packed_gemm_nt_cpu(a_rows, up_w.b, &mut u_refs, ff_dim, dim)?;
    }
    let mut act_rows: Vec<Vec<f32>> = Vec::with_capacity(a_rows.len());
    for (g, u) in gate_rows.iter().zip(up_rows.iter()) {
        let mut act = vec![0.0f32; g.len()];
        for i in 0..g.len() {
            act[i] = allpaca_cpu::silu(g[i]) * u[i];
        }
        act_rows.push(act);
    }
    let act_refs: Vec<&[f32]> = act_rows.iter().map(|v| v.as_slice()).collect();
    packed_gemm_nt_cpu(&act_refs, down_w.b, out_rows, dim, ff_dim)
}

/// Packed FFN on HIP: gate/up `gemm_nt` → `silu_mul` → `w_down` `gemm_nt`.
///
/// Returns the number of HIP GEMM kernels launched (3 on full success).
pub fn apply_packed_ffn_hip(
    hip: &allpaca_hip::HipWorkspace,
    a_rows: &[&[f32]],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    dim: usize,
    out_rows: &mut [&mut [f32]],
) -> Result<u64, AllpacaError> {
    let ffn = packed_ffn_shared_a(dim, w_gate, w_up, w_down);
    let Some(gate_w) = ffn.w_gate else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn_hip: missing w_gate".into(),
        )));
    };
    let Some(up_w) = ffn.w_up else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn_hip: missing w_up".into(),
        )));
    };
    let Some(down_w) = ffn.w_down else {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn_hip: missing w_down".into(),
        )));
    };
    if gate_w.n != up_w.n || down_w.n != dim {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn_hip: dim/ff_dim mismatch".into(),
        )));
    }
    let ff_dim = ffn.ff_dim;
    if a_rows.len() != out_rows.len() {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_ffn_hip: a/out row count mismatch".into(),
        )));
    }
    let mut gate_rows: Vec<Vec<f32>> = a_rows
        .iter()
        .map(|a| {
            let m = a.len() / dim;
            vec![0.0f32; m * ff_dim]
        })
        .collect();
    let mut up_rows: Vec<Vec<f32>> = a_rows
        .iter()
        .map(|a| {
            let m = a.len() / dim;
            vec![0.0f32; m * ff_dim]
        })
        .collect();
    {
        let mut g_refs: Vec<&mut [f32]> = gate_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
        packed_gemm_nt_hip(hip, a_rows, gate_w.b, &mut g_refs, ff_dim, dim)?;
    }
    {
        let mut u_refs: Vec<&mut [f32]> = up_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
        packed_gemm_nt_hip(hip, a_rows, up_w.b, &mut u_refs, ff_dim, dim)?;
    }
    for (g, u) in gate_rows.iter_mut().zip(up_rows.iter()) {
        hip.silu_mul(g, u)?;
    }
    let act_refs: Vec<&[f32]> = gate_rows.iter().map(|v| v.as_slice()).collect();
    packed_gemm_nt_hip(hip, &act_refs, down_w.b, out_rows, dim, ff_dim)?;
    Ok(3)
}

/// Packed lm_head: shared-A NT GEMM of hidden rows × `lm_head` (`n=vocab`, `k=dim`).
pub fn apply_packed_lm_head(
    a_rows: &[&[f32]],
    lm_head: &[f32],
    dim: usize,
    c_rows: &mut [&mut [f32]],
) -> Result<(), AllpacaError> {
    if dim == 0 || lm_head.is_empty() || lm_head.len() % dim != 0 {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "apply_packed_lm_head: bad lm_head shape".into(),
        )));
    }
    let n = lm_head.len() / dim;
    packed_gemm_nt_cpu(a_rows, lm_head, c_rows, n, dim)
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

    #[test]
    fn decode_layer_shared_a_lists_qkv_and_ffn_gate_up() {
        let dim = 4usize;
        let wq = vec![1.0f32; dim * dim];
        let wk = vec![2.0f32; 2 * dim];
        let wv = vec![3.0f32; 2 * dim];
        let wo = vec![4.0f32; dim * dim];
        let w_gate = vec![5.0f32; 8 * dim];
        let w_up = vec![6.0f32; 8 * dim];
        let attn = decode_layer_attn_shared_a_weights(dim, &wq, &wk, &wv, &wo);
        assert_eq!(
            attn.iter().map(|x| x.name).collect::<Vec<_>>(),
            vec!["wq", "wk", "wv", "wo"]
        );
        let w = decode_layer_shared_a_weights(dim, &wq, &wk, &wv, &wo, &w_gate, &w_up);
        let names: Vec<&str> = w.iter().map(|x| x.name).collect();
        assert_eq!(names, vec!["wq", "wk", "wv", "wo", "w_gate", "w_up"]);
        assert_eq!(w[1].n, 2);
        assert_eq!(w[4].n, 8);
        assert!(w.iter().all(|x| x.k == dim));
        assert!(decode_layer_shared_a_weights(0, &wq, &wk, &wv, &wo, &w_gate, &w_up).is_empty());
    }

    #[test]
    fn packed_ffn_lists_gate_up_and_down() {
        let dim = 4usize;
        let ff = 8usize;
        let w_gate = vec![1.0f32; ff * dim];
        let w_up = vec![2.0f32; ff * dim];
        let w_down = vec![3.0f32; dim * ff];
        let ffn = packed_ffn_shared_a(dim, &w_gate, &w_up, &w_down);
        assert_eq!(ffn.ff_dim, ff);
        assert_eq!(ffn.w_gate.unwrap().n, ff);
        assert_eq!(ffn.w_up.unwrap().n, ff);
        let down = ffn.w_down.unwrap();
        assert_eq!(down.n, dim);
        assert_eq!(down.k, ff);
    }

    /// Packed SwiGLU (gate/up → silu*up → down) matches per-row separate path.
    #[test]
    fn packed_ffn_gate_up_down_matches_separate() {
        let dim = 4usize;
        let ff = 6usize;
        let a0 = [0.1f32, -0.2, 0.3, 0.0];
        let a1 = [0.5f32, 0.25, -0.1, 0.05];
        let w_gate: Vec<f32> = (0..ff * dim)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let w_up: Vec<f32> = (0..ff * dim)
            .map(|i| ((i % 5) as f32) * 0.05)
            .collect();
        let w_down: Vec<f32> = (0..dim * ff)
            .map(|i| ((i % 3) as f32) * 0.2 - 0.1)
            .collect();

        let sep = |a: &[f32]| {
            let mut gate = vec![0.0f32; ff];
            let mut up = vec![0.0f32; ff];
            let mut out = vec![0.0f32; dim];
            allpaca_cpu::matmul_nt(a, &w_gate, &mut gate, 1, ff, dim).unwrap();
            allpaca_cpu::matmul_nt(a, &w_up, &mut up, 1, ff, dim).unwrap();
            for i in 0..ff {
                gate[i] = allpaca_cpu::silu(gate[i]) * up[i];
            }
            allpaca_cpu::matmul_nt(&gate, &w_down, &mut out, 1, dim, ff).unwrap();
            out
        };
        let c0_sep = sep(&a0);
        let c1_sep = sep(&a1);

        let mut c0 = [0.0f32; 4];
        let mut c1 = [0.0f32; 4];
        apply_packed_ffn(
            &[&a0[..], &a1[..]],
            &w_gate,
            &w_up,
            &w_down,
            dim,
            &mut [&mut c0[..], &mut c1[..]],
        )
        .expect("packed ffn");
        for i in 0..dim {
            assert!((c0[i] - c0_sep[i]).abs() < 1e-5, "c0[{i}]");
            assert!((c1[i] - c1_sep[i]).abs() < 1e-5, "c1[{i}]");
        }
    }

    #[test]
    fn packed_ffn_hip_matches_cpu_or_skips() {
        if !allpaca_hip::runtime_linked() || allpaca_hip::device_count() == 0 {
            return;
        }
        let Ok(hip) = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 256) else {
            return;
        };
        let dim = 4usize;
        let ff = 6usize;
        let a0 = [0.1f32, -0.2, 0.3, 0.0];
        let a1 = [0.5f32, 0.25, -0.1, 0.05];
        let w_gate: Vec<f32> = (0..ff * dim)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let w_up: Vec<f32> = (0..ff * dim)
            .map(|i| ((i % 5) as f32) * 0.05)
            .collect();
        let w_down: Vec<f32> = (0..dim * ff)
            .map(|i| ((i % 3) as f32) * 0.2 - 0.1)
            .collect();
        let mut c0_cpu = [0.0f32; 4];
        let mut c1_cpu = [0.0f32; 4];
        apply_packed_ffn(
            &[&a0[..], &a1[..]],
            &w_gate,
            &w_up,
            &w_down,
            dim,
            &mut [&mut c0_cpu[..], &mut c1_cpu[..]],
        )
        .expect("cpu ffn");
        let mut c0 = [0.0f32; 4];
        let mut c1 = [0.0f32; 4];
        let n = apply_packed_ffn_hip(
            &hip,
            &[&a0[..], &a1[..]],
            &w_gate,
            &w_up,
            &w_down,
            dim,
            &mut [&mut c0[..], &mut c1[..]],
        )
        .expect("hip ffn");
        assert_eq!(n, 3);
        for i in 0..dim {
            let d0 = (c0[i] - c0_cpu[i]).abs();
            let d1 = (c1[i] - c1_cpu[i]).abs();
            assert!(d0 <= 1e-3 || d0 <= 1e-3 * c0_cpu[i].abs(), "c0[{i}]");
            assert!(d1 <= 1e-3 || d1 <= 1e-3 * c1_cpu[i].abs(), "c1[{i}]");
        }
    }

    #[test]
    fn packed_lm_head_matches_separate() {
        let dim = 4usize;
        let vocab = 5usize;
        let a0 = [0.2f32, -0.1, 0.4, 0.0];
        let a1 = [0.0f32, 0.3, -0.2, 0.1];
        let lm: Vec<f32> = (0..vocab * dim)
            .map(|i| ((i % 6) as f32) * 0.1 - 0.2)
            .collect();
        let mut c0_sep = [0.0f32; 5];
        let mut c1_sep = [0.0f32; 5];
        allpaca_cpu::matmul_nt(&a0, &lm, &mut c0_sep, 1, vocab, dim).unwrap();
        allpaca_cpu::matmul_nt(&a1, &lm, &mut c1_sep, 1, vocab, dim).unwrap();
        let mut c0 = [0.0f32; 5];
        let mut c1 = [0.0f32; 5];
        apply_packed_lm_head(
            &[&a0[..], &a1[..]],
            &lm,
            dim,
            &mut [&mut c0[..], &mut c1[..]],
        )
        .expect("packed lm_head");
        assert_eq!(c0, c0_sep);
        assert_eq!(c1, c1_sep);
    }
}
