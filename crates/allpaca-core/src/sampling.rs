use allpaca_types::{GenerationParams, TokenMask};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn rng_from_params(params: &GenerationParams, fallback_seed: Option<u64>) -> StdRng {
    let seed = params.seed.or(fallback_seed).unwrap_or(0xA11_0ACA);
    StdRng::seed_from_u64(seed)
}

/// Softmaxed `(id, probability)` list for `logits` under `params`.
/// Greedy (`temperature <= 1e-5`) is a one-hot on argmax. Mutates `logits`
/// the same way [`sample`] does (mask, then temperature).
pub fn softmax_candidates(
    logits: &mut [f32],
    mask: Option<&TokenMask>,
    params: &GenerationParams,
) -> Vec<(u32, f32)> {
    softmax_inner(logits, mask, params).0
}

/// [`softmax_candidates`] plus whether a plain draw consumes the RNG. Main's
/// `sample` returned without touching the RNG when every logit was
/// non-finite or no candidate survived temperature; keep those streams.
fn softmax_inner(
    logits: &mut [f32],
    mask: Option<&TokenMask>,
    params: &GenerationParams,
) -> (Vec<(u32, f32)>, bool) {
    if let Some(mask) = mask {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !mask.allowed(i) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    if logits.iter().all(|l| !l.is_finite()) {
        return (vec![(0, 1.0)], false);
    }
    if params.temperature <= 1e-5 {
        return (vec![(argmax(logits), 1.0)], false);
    }
    apply_temperature(logits, params.temperature);
    let mut candidates = collect_finite(logits);
    if candidates.is_empty() {
        return (vec![(argmax(logits), 1.0)], false);
    }
    apply_top_k(&mut candidates, params.top_k);
    apply_top_p(&mut candidates, params.top_p);
    softmax_pairs(&mut candidates);
    (candidates, true)
}

/// Draw from a softmaxed candidate list with the same unit-interval walk
/// [`sample`] has always used. Greedy skips the RNG.
fn draw_softmaxed(candidates: &[(u32, f32)], temperature: f32, rng: &mut StdRng) -> u32 {
    if temperature <= 1e-5 {
        return candidates.first().map(|(id, _)| *id).unwrap_or(0);
    }
    let mut r: f32 = rng.gen();
    for (id, p) in candidates {
        r -= *p;
        if r <= 0.0 {
            return *id;
        }
    }
    candidates.last().map(|(id, _)| *id).unwrap_or(0)
}

pub fn sample(
    logits: &mut [f32],
    mask: Option<&TokenMask>,
    params: &GenerationParams,
    rng: &mut StdRng,
) -> u32 {
    trace_top2(logits);
    dump_logits(logits);
    let (candidates, draws) = softmax_inner(logits, mask, params);
    if !draws {
        return candidates.first().map(|(id, _)| *id).unwrap_or(0);
    }
    draw_softmaxed(&candidates, params.temperature, rng)
}

/// `ALLPACA_LOGIT_TRACE=1`: print the two largest logits of every sampled
/// step to stderr (`logit_trace id1 l1 id2 l2`), to see whether a greedy
/// divergence sits on a near tie (P21). Off by default.
fn trace_top2(logits: &[f32]) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("ALLPACA_LOGIT_TRACE").is_ok_and(|v| v == "1")) {
        return;
    }
    let (mut a, mut b) = ((u32::MAX, f32::NEG_INFINITY), (u32::MAX, f32::NEG_INFINITY));
    for (i, &l) in logits.iter().enumerate() {
        if l > a.1 {
            b = a;
            a = (i as u32, l);
        } else if l > b.1 {
            b = (i as u32, l);
        }
    }
    eprintln!("logit_trace {} {:.6} {} {:.6}", a.0, a.1, b.0, b.1);
}

/// `ALLPACA_LOGITS_DUMP=<file>`: append the logits of every sampled step to
/// the file as little-endian f32 (a vocabulary per step), to compare two
/// runs' distributions over the same tokens (KL of an experimental option
/// against the exact run). Off by default.
fn dump_logits(logits: &[f32]) {
    use std::io::Write;
    static FILE: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let file = FILE.get_or_init(|| {
        let path = std::env::var_os("ALLPACA_LOGITS_DUMP")?;
        match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(std::sync::Mutex::new(file)),
            Err(e) => {
                tracing::warn!("ALLPACA_LOGITS_DUMP: cannot open {}: {e}", std::path::Path::new(&path).display());
                None
            }
        }
    });
    if let Some(file) = file {
        let bytes: Vec<u8> = logits.iter().flat_map(|l| l.to_le_bytes()).collect();
        if let Err(e) = file.lock().unwrap_or_else(|e| e.into_inner()).write_all(&bytes) {
            tracing::warn!("ALLPACA_LOGITS_DUMP: {e}");
        }
    }
}

/// Draw the next token. When `q` is a non-empty draft distribution and
/// temperature is above greedy, use Leviathan rejection against `draft_id`.
/// Otherwise sample as usual and report whether the result equals `draft_id`.
pub fn sample_with_draft(
    logits: &mut [f32],
    mask: Option<&TokenMask>,
    params: &GenerationParams,
    rng: &mut StdRng,
    draft_id: Option<u32>,
    q: Option<&[(u32, f32)]>,
) -> (u32, bool) {
    if params.temperature > 1e-5 {
        if let (Some(draft_id), Some(q)) = (draft_id, q) {
            if !q.is_empty() {
                let p = softmax_candidates(logits, mask, params);
                return rejection_sample(&p, q, draft_id, rng);
            }
        }
    }
    let tok = sample(logits, mask, params, rng);
    (tok, draft_id == Some(tok))
}

/// Probability of `id` in a softmaxed candidate list. Missing ids are 0.
fn prob_of(dist: &[(u32, f32)], id: u32) -> f32 {
    dist.iter()
        .find(|(i, _)| *i == id)
        .map(|(_, p)| *p)
        .unwrap_or(0.0)
}

/// Draw one id from a softmaxed `(id, probability)` list. This is how a
/// drafter samples from the `q` it then reports for Leviathan verify.
pub fn sample_from_distribution(dist: &[(u32, f32)], rng: &mut StdRng) -> u32 {
    sample_weighted(dist, rng)
}

fn sample_weighted(dist: &[(u32, f32)], rng: &mut StdRng) -> u32 {
    let sum: f32 = dist.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 || !sum.is_finite() {
        return dist.first().map(|(id, _)| *id).unwrap_or(0);
    }
    let mut r: f32 = rng.gen::<f32>() * sum;
    for (id, p) in dist {
        r -= *p;
        if r <= 0.0 {
            return *id;
        }
    }
    dist.last().map(|(id, _)| *id).unwrap_or(0)
}

/// Accept `draft_id` with probability `min(1, p/q)`, else sample from
/// `norm(max(0, p − q))`. `p` and `q` are softmaxed `(id, probability)` lists.
/// Returns `(token, accepted)`.
///
/// Leviathan et al. / llama.cpp #27694. Greedy one-hot `p` and `q` accept
/// iff they name the same token, so T = 0 stays bit-identical to exact match.
pub fn rejection_sample(
    p: &[(u32, f32)],
    q: &[(u32, f32)],
    draft_id: u32,
    rng: &mut StdRng,
) -> (u32, bool) {
    let p_x = prob_of(p, draft_id);
    let q_x = prob_of(q, draft_id);
    if p_x > 0.0 && (q_x <= 0.0 || p_x >= q_x || rng.gen::<f32>() < p_x / q_x) {
        return (draft_id, true);
    }
    let mut residual = Vec::with_capacity(p.len());
    for &(id, p_i) in p {
        let r = p_i - prob_of(q, id);
        if r > 0.0 {
            residual.push((id, r));
        }
    }
    if residual.is_empty() {
        // p == q up to rounding and the draft was rejected: draw from p.
        if p.is_empty() {
            return (draft_id, false);
        }
        return (sample_weighted(p, rng), false);
    }
    (sample_weighted(&residual, rng), false)
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, v) in logits.iter().enumerate() {
        if *v > best {
            best = *v;
            best_i = i;
        }
    }
    best_i as u32
}

fn apply_temperature(logits: &mut [f32], temperature: f32) {
    let t = temperature.max(1e-5);
    for l in logits.iter_mut() {
        if l.is_finite() {
            *l /= t;
        }
    }
}

fn collect_finite(logits: &[f32]) -> Vec<(u32, f32)> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i as u32, *v))
        .collect()
}

/// Descending logit, ties by token id: the order a stable full sort gives.
fn by_logit_desc(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

/// Moves the `n` largest candidates to the front, sorted. A full sort of a
/// 248k-token vocabulary cost ~10 ms per sampled token; this is linear.
fn sort_top(candidates: &mut [(u32, f32)], n: usize) {
    let n = n.min(candidates.len());
    if n == 0 {
        return;
    }
    if n < candidates.len() {
        candidates.select_nth_unstable_by(n - 1, by_logit_desc);
    }
    candidates[..n].sort_unstable_by(by_logit_desc);
}

fn apply_top_k(candidates: &mut Vec<(u32, f32)>, top_k: usize) {
    if top_k == 0 || candidates.len() <= top_k {
        return;
    }
    sort_top(candidates, top_k);
    candidates.truncate(top_k);
}

/// Keeps the smallest sorted prefix whose probability reaches `top_p`
/// (probabilities over all candidates). The nucleus is usually a few tokens,
/// so only a growing head of the list is sorted.
fn apply_top_p(candidates: &mut Vec<(u32, f32)>, top_p: f32) {
    if !(0.0..1.0).contains(&top_p) || candidates.len() <= 1 {
        return;
    }
    let len = candidates.len();
    let max = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = candidates.iter().map(|(_, v)| (*v - max).exp()).sum();
    if sum == 0.0 || !sum.is_finite() {
        // Degenerate weights count as uniform.
        sort_top(candidates, len);
        let keep = ((top_p * len as f32).ceil() as usize).clamp(1, len);
        candidates.truncate(keep);
        return;
    }
    let mut head = 64.min(len);
    loop {
        sort_top(candidates, head);
        let mut acc = 0.0f32;
        for (i, (_, v)) in candidates[..head].iter().enumerate() {
            acc += (*v - max).exp() / sum;
            if acc >= top_p {
                candidates.truncate(i + 1);
                return;
            }
        }
        if head == len {
            return;
        }
        head = (head * 8).min(len);
    }
}

fn softmax_pairs(candidates: &mut [(u32, f32)]) {
    let max = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in candidates.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum == 0.0 {
        let n = candidates.len() as f32;
        for (_, v) in candidates.iter_mut() {
            *v = 1.0 / n;
        }
        return;
    }
    for (_, v) in candidates.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejection_accepts_greedy_match_without_rng() {
        let p = [(1u32, 1.0f32)];
        let q = [(1u32, 1.0f32)];
        let mut rng = StdRng::seed_from_u64(0);
        let (tok, accepted) = rejection_sample(&p, &q, 1, &mut rng);
        assert_eq!(tok, 1);
        assert!(accepted);
    }

    #[test]
    fn rejection_greedy_mismatch_resamples_target_argmax() {
        let p = [(3u32, 1.0f32)];
        let q = [(1u32, 1.0f32)];
        let mut rng = StdRng::seed_from_u64(0);
        let (tok, accepted) = rejection_sample(&p, &q, 1, &mut rng);
        assert_eq!(tok, 3);
        assert!(!accepted);
    }

    #[test]
    fn rejection_always_accepts_when_p_covers_q() {
        let p = [(0u32, 0.8), (1u32, 0.2)];
        let q = [(0u32, 0.5), (1u32, 0.5)];
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (tok, accepted) = rejection_sample(&p, &q, 0, &mut rng);
            assert_eq!(tok, 0);
            assert!(accepted, "seed {seed}");
        }
    }

    #[test]
    fn rejection_never_accepts_zero_p_draft() {
        let p = [(1u32, 1.0f32)];
        let q = [(0u32, 1.0f32)];
        for seed in 0..32u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (tok, accepted) = rejection_sample(&p, &q, 0, &mut rng);
            assert!(!accepted, "seed {seed}");
            assert_eq!(tok, 1);
        }
    }

    #[test]
    fn rejection_token_frequencies_match_target_p() {
        // p = [0.7, 0.3], greedy-draft q = one-hot on token 0.
        // Accept with p(0)=0.7; residual is token 1 with mass 0.3.
        let p = [(0u32, 0.7), (1u32, 0.3)];
        let q = [(0u32, 1.0)];
        let n = 20_000u32;
        let mut counts = [0u32; 2];
        let mut rng = StdRng::seed_from_u64(0xA11_0ACA);
        for _ in 0..n {
            let (tok, _) = rejection_sample(&p, &q, 0, &mut rng);
            counts[tok as usize] += 1;
        }
        let f0 = counts[0] as f32 / n as f32;
        let f1 = counts[1] as f32 / n as f32;
        assert!((f0 - 0.7).abs() < 0.02, "f0={f0}");
        assert!((f1 - 0.3).abs() < 0.02, "f1={f1}");
    }

    #[test]
    fn sample_with_draft_greedy_matches_exact_match() {
        let mut logits = [0.1f32, 3.0, 0.2];
        let params = GenerationParams {
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let q = [(1u32, 1.0f32)];
        let mut rng = StdRng::seed_from_u64(0);
        let (tok, accepted) =
            sample_with_draft(&mut logits, None, &params, &mut rng, Some(1), Some(&q));
        assert_eq!(tok, 1);
        assert!(accepted);
        let mut logits = [0.1f32, 3.0, 0.2];
        let (tok, accepted) =
            sample_with_draft(&mut logits, None, &params, &mut rng, Some(0), Some(&q));
        assert_eq!(tok, 1);
        assert!(!accepted);
    }

    #[test]
    fn sample_with_draft_without_q_equals_sample() {
        let logits = [0.2f32, 1.5, 0.4, 0.1];
        let params = GenerationParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            ..GenerationParams::default()
        };
        for seed in 0..32u64 {
            let mut a = logits;
            let mut b = logits;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            let from_sample = sample(&mut a, None, &params, &mut rng_a);
            let (from_draft, accepted) =
                sample_with_draft(&mut b, None, &params, &mut rng_b, Some(1), None);
            assert_eq!(from_sample, from_draft, "seed {seed}");
            assert_eq!(accepted, from_draft == 1, "seed {seed}");
        }
    }

    #[test]
    fn softmax_candidates_greedy_is_one_hot_argmax() {
        let mut logits = [0.1, 3.0, 0.2];
        let params = GenerationParams {
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let p = softmax_candidates(&mut logits, None, &params);
        assert_eq!(p, vec![(1, 1.0)]);
    }

    #[test]
    fn softmax_candidates_then_weighted_draw_matches_sample() {
        let logits = [0.2f32, 1.5, 0.4, 0.1];
        let params = GenerationParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            ..GenerationParams::default()
        };
        for seed in 0..32u64 {
            let mut a = logits;
            let mut b = logits;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            let from_sample = sample(&mut a, None, &params, &mut rng_a);
            let p = softmax_candidates(&mut b, None, &params);
            let from_p = draw_softmaxed(&p, params.temperature, &mut rng_b);
            assert_eq!(from_sample, from_p, "seed {seed}");
        }
    }

    /// Main never touched the RNG when no candidate survived at T > 0; the
    /// softmax refactor must not shift the stream in those edge cases.
    #[test]
    fn sample_leaves_rng_untouched_when_every_logit_is_non_finite() {
        let params = GenerationParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            ..GenerationParams::default()
        };
        let mut logits = [f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY];
        let mut rng = StdRng::seed_from_u64(5);
        let mut fresh = StdRng::seed_from_u64(5);
        assert_eq!(sample(&mut logits, None, &params, &mut rng), 0);
        assert_eq!(rng.gen::<u64>(), fresh.gen::<u64>());
    }

    #[test]
    fn sample_leaves_rng_untouched_when_temperature_overflows_every_logit() {
        let params = GenerationParams {
            temperature: 0.5,
            top_k: 0,
            top_p: 1.0,
            ..GenerationParams::default()
        };
        let mut logits = [3.0e38f32, 2.0e38];
        let mut rng = StdRng::seed_from_u64(5);
        let mut fresh = StdRng::seed_from_u64(5);
        assert_eq!(sample(&mut logits, None, &params, &mut rng), 0);
        assert_eq!(rng.gen::<u64>(), fresh.gen::<u64>());
    }

    #[test]
    fn sample_still_draws_once_for_a_single_candidate() {
        let params = GenerationParams {
            temperature: 0.8,
            top_k: 1,
            top_p: 1.0,
            ..GenerationParams::default()
        };
        let mut logits = [0.1f32, 3.0, 0.2];
        let mut rng = StdRng::seed_from_u64(5);
        let mut fresh = StdRng::seed_from_u64(5);
        assert_eq!(sample(&mut logits, None, &params, &mut rng), 1);
        let _: f32 = fresh.gen();
        assert_eq!(rng.gen::<u64>(), fresh.gen::<u64>());
    }

    #[test]
    fn rejection_with_an_all_zero_residual_draws_from_p() {
        // q covers p exactly and the draft id has p_x < q_x only through
        // rounding: force the empty-residual path with p == q and a draft
        // token p gives zero mass.
        let p = vec![(1u32, 0.5f32), (2, 0.5)];
        let q = vec![(1u32, 0.5f32), (2, 0.5)];
        let mut seen = std::collections::HashSet::new();
        for seed in 0..64u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (tok, accepted) = rejection_sample(&p, &q, 7, &mut rng);
            assert!(!accepted);
            seen.insert(tok);
        }
        assert_eq!(seen, [1u32, 2].into_iter().collect(), "must draw from p, not p.first()");
    }

    #[test]
    fn greedy_picks_max() {
        let mut logits = [0.1, 3.0, 0.2];
        let mut rng = StdRng::seed_from_u64(1);
        let params = GenerationParams {
            temperature: 0.0,
            ..GenerationParams::default()
        };
        assert_eq!(sample(&mut logits, None, &params, &mut rng), 1);
    }

    #[test]
    fn mask_hides_winner() {
        let mut logits = [0.1, 3.0, 2.0];
        let mut mask = TokenMask::empty(3);
        mask.set(2, true);
        let mut rng = StdRng::seed_from_u64(1);
        let params = GenerationParams {
            temperature: 0.0,
            ..GenerationParams::default()
        };
        assert_eq!(sample(&mut logits, Some(&mask), &params, &mut rng), 2);
    }
    /// The partial sort keeps what the full sort kept, in the same order.
    #[test]
    fn top_k_and_top_p_match_a_full_sort() {
        let mut seed = 0x9E37_79B9u64;
        let logits: Vec<f32> = (0..5000)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Coarse values so ties occur; a few clear winners.
                ((seed >> 40) % 97) as f32 * 0.05 + if i % 1250 == 7 { 6.0 } else { 0.0 }
            })
            .collect();
        let full = |top_k: usize, top_p: f32| -> Vec<(u32, f32)> {
            let mut c = collect_finite(&logits);
            c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if top_k > 0 {
                c.truncate(top_k);
            }
            let mut probs = c.clone();
            softmax_pairs(&mut probs);
            let mut acc = 0.0f32;
            let keep = probs.iter().position(|(_, p)| {
                acc += *p;
                acc >= top_p
            });
            c.truncate(keep.map_or(c.len(), |i| i + 1));
            c
        };
        for (top_k, top_p) in [(0, 0.95), (40, 0.95), (0, 0.5), (3, 0.999), (0, 0.9999)] {
            let mut c = collect_finite(&logits);
            apply_top_k(&mut c, top_k);
            apply_top_p(&mut c, top_p);
            assert_eq!(c, full(top_k, top_p), "top_k={top_k} top_p={top_p}");
        }
    }
}
