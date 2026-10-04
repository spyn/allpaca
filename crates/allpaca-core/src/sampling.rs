use allpaca_types::{GenerationParams, TokenMask};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn rng_from_params(params: &GenerationParams, fallback_seed: Option<u64>) -> StdRng {
    let seed = params.seed.or(fallback_seed).unwrap_or(0xA11_0ACA);
    StdRng::seed_from_u64(seed)
}

pub fn sample(
    logits: &mut [f32],
    mask: Option<&TokenMask>,
    params: &GenerationParams,
    rng: &mut StdRng,
) -> u32 {
    if let Some(mask) = mask {
        for (i, logit) in logits.iter_mut().enumerate() {
            if !mask.allowed(i) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    if logits.iter().all(|l| !l.is_finite()) {
        return 0;
    }
    if params.temperature <= 1e-5 {
        return argmax(logits);
    }
    apply_temperature(logits, params.temperature);
    let mut candidates = collect_finite(logits);
    if candidates.is_empty() {
        return argmax(logits);
    }
    apply_top_k(&mut candidates, params.top_k);
    apply_top_p(&mut candidates, params.top_p);
    softmax_pairs(&mut candidates);
    let mut r: f32 = rng.gen();
    for (id, p) in &candidates {
        r -= *p;
        if r <= 0.0 {
            return *id;
        }
    }
    candidates.last().map(|(id, _)| *id).unwrap_or(0)
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
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
