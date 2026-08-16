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

fn apply_top_k(candidates: &mut Vec<(u32, f32)>, top_k: usize) {
    if top_k == 0 || candidates.len() <= top_k {
        return;
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k);
}

fn apply_top_p(candidates: &mut Vec<(u32, f32)>, top_p: f32) {
    if !(0.0..1.0).contains(&top_p) || candidates.len() <= 1 {
        return;
    }
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut tmp = candidates.clone();
    softmax_pairs(&mut tmp);
    let mut acc = 0.0f32;
    let mut keep = 0;
    for (_, p) in &tmp {
        acc += *p;
        keep += 1;
        if acc >= top_p {
            break;
        }
    }
    candidates.truncate(keep.max(1));
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
}
