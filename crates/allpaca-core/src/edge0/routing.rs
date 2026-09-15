//! Bit-identical Edge0 routing math (softmax top-k + sigmoid group).

use super::spec::RouterKind;

/// Softmax → top-k → optional renormalize. Returns `(expert_idx, weight)`.
pub fn select_from_logits(logits: &[f32], top_k: usize, norm: bool) -> Vec<(usize, f32)> {
    let n = logits.len();
    if n == 0 || top_k == 0 {
        return Vec::new();
    }
    let k = top_k.min(n);
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut gates: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
    let sum: f32 = gates.iter().sum::<f32>().max(1e-9);
    for g in &mut gates {
        *g /= sum;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        gates[b]
            .partial_cmp(&gates[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    let mut scores: Vec<f32> = idx.iter().map(|&i| gates[i]).collect();
    if norm {
        let s: f32 = scores.iter().sum::<f32>().max(1e-9);
        for w in &mut scores {
            *w /= s;
        }
    }
    idx.into_iter().zip(scores).collect()
}

/// Sigmoid + group-limited top-k (DeepSeek-V3 / Bailing).
///
/// Selection uses `sigmoid(logits) + bias`; weights use raw sigmoid (bias excluded),
/// optionally normalized, then scaled by `routed_scaling`.
pub fn group_select_from_logits(
    logits: &[f32],
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling: f32,
    norm: bool,
    expert_bias: Option<&[f32]>,
) -> Vec<(usize, f32)> {
    let n = logits.len();
    if n == 0 || top_k == 0 || n_group == 0 {
        return Vec::new();
    }
    let k = top_k.min(n);
    let per_group = n / n_group;
    if per_group == 0 || n % n_group != 0 {
        // Fall back to plain top-k on sigmoid scores when groups do not divide.
        let scores: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        idx.truncate(k);
        let mut w: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
        if norm {
            let s: f32 = w.iter().sum::<f32>() + 1e-20;
            for x in &mut w {
                *x /= s;
            }
        }
        for x in &mut w {
            *x *= routed_scaling;
        }
        return idx.into_iter().zip(w).collect();
    }

    let scores: Vec<f32> = logits.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    let mut select = scores.clone();
    if let Some(bias) = expert_bias {
        for (i, s) in select.iter_mut().enumerate() {
            if i < bias.len() {
                *s += bias[i];
            }
        }
    }

    let k_drop = n_group.saturating_sub(topk_group);
    if k_drop > 0 {
        let mut group_scores = vec![0.0f32; n_group];
        for g in 0..n_group {
            let base = g * per_group;
            let mut top2 = [f32::NEG_INFINITY; 2];
            for e in 0..per_group {
                let v = select[base + e];
                if v > top2[0] {
                    top2[1] = top2[0];
                    top2[0] = v;
                } else if v > top2[1] {
                    top2[1] = v;
                }
            }
            group_scores[g] = top2[0] + top2[1].max(0.0);
        }
        let mut gidx: Vec<usize> = (0..n_group).collect();
        gidx.sort_by(|&a, &b| {
            group_scores[a]
                .partial_cmp(&group_scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        for &g in gidx.iter().take(k_drop) {
            let base = g * per_group;
            for e in 0..per_group {
                select[base + e] = f32::NEG_INFINITY;
            }
        }
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        select[b]
            .partial_cmp(&select[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(k);
    let mut w: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
    if norm {
        let s: f32 = w.iter().sum::<f32>() + 1e-20;
        for x in &mut w {
            *x /= s;
        }
    }
    for x in &mut w {
        *x *= routed_scaling;
    }
    idx.into_iter().zip(w).collect()
}

/// Dispatch by [`RouterKind`].
pub fn route_logits(
    kind: RouterKind,
    logits: &[f32],
    top_k: usize,
    norm: bool,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    routed_scaling: Option<f32>,
    expert_bias: Option<&[f32]>,
) -> Vec<(usize, f32)> {
    match kind {
        RouterKind::SoftmaxTopk => select_from_logits(logits, top_k, norm),
        RouterKind::SigmoidGroup => group_select_from_logits(
            logits,
            top_k,
            n_group.unwrap_or(1),
            topk_group.unwrap_or(1),
            routed_scaling.unwrap_or(1.0),
            norm,
            expert_bias,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_topk_renorms_and_picks_largest() {
        let logits = [0.0f32, 2.0, 1.0, -1.0];
        let got = select_from_logits(&logits, 2, true);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, 1);
        assert_eq!(got[1].0, 2);
        let sum: f32 = got.iter().map(|(_, w)| *w).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_group_scales_weights() {
        // 8 experts, 2 groups of 4; keep both groups → plain top-2 on sigmoid.
        let logits = [0.0f32, 3.0, 1.0, -2.0, 0.5, 2.0, -1.0, 0.1];
        let got = group_select_from_logits(&logits, 2, 2, 2, 2.5, true, None);
        assert_eq!(got.len(), 2);
        let sum: f32 = got.iter().map(|(_, w)| *w).sum();
        // After norm, sum of weights ≈ routed_scaling.
        assert!((sum - 2.5).abs() < 1e-4, "sum={sum}");
    }
}
