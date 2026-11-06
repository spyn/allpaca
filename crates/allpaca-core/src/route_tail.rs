//! Strata route-tail skip: drop missed experts that every token in the
//! verify window ranks at or below a threshold, without renormalising
//! remaining weights.

/// Runtime policy for [`filter_picks`]. Off by default; rank 0 is best.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    pub enabled: bool,
    /// Rank at or below which a missed expert is skipped (Strata default 7).
    pub rank_threshold: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            enabled: false,
            rank_threshold: 7,
        }
    }
}

impl Policy {
    /// `rank_threshold` is clamped to at least 1: rank 0 is the top pick, and
    /// dropping it could leave a token with no routed expert at all.
    pub fn from_experimental(enabled: bool, rank_threshold: u32) -> Self {
        Self {
            enabled,
            rank_threshold: rank_threshold.max(1),
        }
    }
}

/// Drop missed experts whose rank is at or below the threshold on every token
/// of the window. Remaining weights are left as-is (no renormalise). Resident
/// experts are never dropped, and the top pick of a token never is.
///
/// The threshold is capped at `k - 1`, where `k` is the longest pick list in
/// the window, so a model that picks fewer experts than the threshold (DSV4
/// picks 6, the default threshold is 7) still has a tail to skip: its
/// last-ranked experts. `is_missed` may be stateful, so it is asked only about
/// tail candidates (every token ranks the expert at or below the threshold),
/// at most once per expert, in first-seen order.
///
/// Off (`policy.enabled == false`) is identity. The window can span rows of
/// different requests, so with the policy on a row's picks depend on the rest
/// of the batch; that is part of this technique being lossy.
pub fn filter_picks(
    picks_per_token: &[Vec<(usize, f32)>],
    mut is_missed: impl FnMut(usize) -> bool,
    policy: Policy,
) -> Vec<Vec<(usize, f32)>> {
    if !policy.enabled {
        return picks_per_token.to_vec();
    }
    let k = picks_per_token.iter().map(Vec::len).max().unwrap_or(0);
    if k < 2 {
        return picks_per_token.to_vec();
    }
    let threshold = policy.rank_threshold.max(1).min(k as u32 - 1);
    let n = picks_per_token.len();
    let mut ranks: std::collections::HashMap<usize, Vec<Option<u32>>> =
        std::collections::HashMap::new();
    let mut first_seen: Vec<usize> = Vec::new();
    for (t, picks) in picks_per_token.iter().enumerate() {
        for (i, &(e, _)) in picks.iter().enumerate() {
            ranks
                .entry(e)
                .or_insert_with(|| {
                    first_seen.push(e);
                    vec![None; n]
                })[t] = Some(i as u32);
        }
    }
    let skip: std::collections::HashSet<usize> = first_seen
        .into_iter()
        .filter(|e| skip_missed_expert(&ranks[e], threshold) && is_missed(*e))
        .collect();
    if skip.is_empty() {
        return picks_per_token.to_vec();
    }
    picks_per_token
        .iter()
        .map(|picks| picks.iter().copied().filter(|(e, _)| !skip.contains(e)).collect())
        .collect()
}

/// Rank 0 is the highest-weighted selected expert for that token.
/// `None` means the token did not select the expert (worse than any rank).
pub fn skip_missed_expert(ranks: &[Option<u32>], rank_threshold: u32) -> bool {
    !ranks.is_empty()
        && ranks.iter().all(|r| match r {
            Some(rank) => *rank >= rank_threshold,
            None => true,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skips_when_every_token_ranks_the_missed_expert_at_or_below_threshold() {
        let ranks = [Some(7), Some(8), Some(9)];
        assert!(skip_missed_expert(&ranks, 7));
    }

    #[test]
    fn keeps_when_any_token_ranks_the_expert_better_than_threshold() {
        let ranks = [Some(8), Some(3), Some(9)];
        assert!(!skip_missed_expert(&ranks, 7));
    }

    #[test]
    fn treats_unselected_as_worse_than_threshold() {
        let ranks = [Some(8), None, Some(7)];
        assert!(skip_missed_expert(&ranks, 7));
        let ranks = [Some(8), None, Some(2)];
        assert!(!skip_missed_expert(&ranks, 7));
    }

    #[test]
    fn empty_window_does_not_skip() {
        assert!(!skip_missed_expert(&[], 7));
    }

    fn window() -> Vec<Vec<(usize, f32)>> {
        vec![
            vec![
                (10, 0.40),
                (11, 0.20),
                (12, 0.10),
                (13, 0.08),
                (14, 0.07),
                (15, 0.06),
                (16, 0.05),
                (17, 0.04),
            ],
            vec![
                (10, 0.35),
                (11, 0.20),
                (12, 0.12),
                (13, 0.09),
                (14, 0.08),
                (15, 0.07),
                (18, 0.05),
                (17, 0.04),
            ],
        ]
    }

    fn on(rank: u32) -> Policy {
        Policy {
            enabled: true,
            rank_threshold: rank,
        }
    }

    #[test]
    fn drops_missed_tail_experts_and_keeps_original_weights() {
        let missed = |e: usize| e == 17 || e == 18;
        let out = filter_picks(&window(), missed, on(7));
        assert_eq!(
            out[0],
            vec![
                (10, 0.40),
                (11, 0.20),
                (12, 0.10),
                (13, 0.08),
                (14, 0.07),
                (15, 0.06),
                (16, 0.05),
            ]
        );
        assert_eq!(
            out[1],
            vec![
                (10, 0.35),
                (11, 0.20),
                (12, 0.12),
                (13, 0.09),
                (14, 0.08),
                (15, 0.07),
                (18, 0.05),
            ]
        );
        let w0: f32 = out[0].iter().map(|(_, w)| *w).sum();
        assert!((w0 - 0.96).abs() < 1e-6, "no renormalise, got {w0}");
    }

    #[test]
    fn flag_off_is_identity() {
        let missed = |e: usize| e == 17 || e == 18;
        let picks = window();
        let out = filter_picks(&picks, missed, Policy::default());
        assert_eq!(out, picks);
    }

    #[test]
    fn keeps_resident_tail_experts() {
        let missed = |e: usize| e == 18;
        let out = filter_picks(&window(), missed, on(7));
        assert!(out[0].iter().any(|(e, _)| *e == 17));
        assert!(out[1].iter().any(|(e, _)| *e == 17));
        assert!(out[1].iter().any(|(e, _)| *e == 18));
    }
    #[test]
    fn rank_zero_is_clamped_so_the_top_pick_is_never_dropped() {
        assert_eq!(Policy::from_experimental(true, 0).rank_threshold, 1);
        // Every pick missed, every window row the same: rank 0 survives.
        let picks = window();
        let out = filter_picks(&picks, |_| true, Policy::from_experimental(true, 0));
        for (o, p) in out.iter().zip(&picks) {
            assert_eq!(o[0], p[0], "top pick kept");
            assert!(o.len() >= 1);
        }
        // Even an all-miss window keeps something to sum.
        assert!(out.iter().all(|o| !o.is_empty()));
    }

    #[test]
    fn the_miss_test_only_runs_for_tail_candidates() {
        // Experts ranked better than the threshold on some token never reach
        // the (stateful) miss test.
        let mut asked = Vec::new();
        let _ = filter_picks(
            &window(),
            |e| {
                asked.push(e);
                true
            },
            on(7),
        );
        // 10..=16 sit above rank 7 on both tokens, or are ranked well on one;
        // only 17 and 18 are tail candidates, 18 only on token 1.
        assert!(asked.iter().all(|e| *e == 17 || *e == 18), "asked {asked:?}");
        let mut seen = asked.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), asked.len(), "each expert asked at most once");
    }

    #[test]
    fn the_rank_is_capped_below_the_pick_count() {
        // DSV4-Flash picks 6 experts, so ranks run 0..=5 and the default
        // threshold 7 would never skip. The window's pick count caps it.
        let w: Vec<Vec<(usize, f32)>> = vec![
            vec![(1, 0.3), (2, 0.2), (3, 0.2), (4, 0.1), (5, 0.1), (6, 0.1)],
            vec![(1, 0.3), (2, 0.2), (3, 0.2), (4, 0.1), (5, 0.1), (7, 0.1)],
        ];
        let out = filter_picks(&w, |e| e == 6 || e == 7, on(7));
        assert_eq!(out[0].len(), 5);
        assert_eq!(out[1].len(), 5);
        // Resident last-ranked experts stay.
        let out = filter_picks(&w, |_| false, on(7));
        assert_eq!(out, w);
        // A threshold already below the cap is used as given.
        let out = filter_picks(&w, |e| e == 5 || e == 6 || e == 7, on(4));
        assert_eq!(out[0].iter().map(|(e, _)| *e).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_single_pick_per_token_is_never_dropped() {
        let w = vec![vec![(3usize, 1.0f32)], vec![(4, 1.0)]];
        assert_eq!(filter_picks(&w, |_| true, on(1)), w);
    }
}
