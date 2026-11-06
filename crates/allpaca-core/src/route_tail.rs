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
    pub fn from_experimental(enabled: bool, rank_threshold: u32) -> Self {
        Self {
            enabled,
            rank_threshold,
        }
    }
}

/// Drop missed experts whose rank is at or below `policy.rank_threshold` on
/// every token. Remaining weights are left as-is (no renormalise). Resident
/// experts are never dropped. Off (`policy.enabled == false`) is identity.
pub fn filter_picks(
    picks_per_token: &[Vec<(usize, f32)>],
    mut is_missed: impl FnMut(usize) -> bool,
    policy: Policy,
) -> Vec<Vec<(usize, f32)>> {
    if !policy.enabled {
        return picks_per_token.to_vec();
    }
    let mut skip: Vec<usize> = Vec::new();
    for picks in picks_per_token {
        for &(e, _) in picks {
            if skip.contains(&e) || !is_missed(e) {
                continue;
            }
            let ranks: Vec<Option<u32>> = picks_per_token
                .iter()
                .map(|p| p.iter().position(|(id, _)| *id == e).map(|i| i as u32))
                .collect();
            if skip_missed_expert(&ranks, policy.rank_threshold) {
                skip.push(e);
            }
        }
    }
    if skip.is_empty() {
        return picks_per_token.to_vec();
    }
    picks_per_token
        .iter()
        .map(|picks| {
            picks
                .iter()
                .copied()
                .filter(|(e, _)| !skip.contains(e))
                .collect()
        })
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
}
