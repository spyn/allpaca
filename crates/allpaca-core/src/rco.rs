//! RCO (Riemannian Constrained Optimization, arXiv 2605.00649): choose one
//! option per group under a total cost budget, such as a bit width per
//! tensor that averages to a target, or which experts of a layer to keep.
//!
//! Each group has logits over the options. A step samples a discrete
//! assignment that meets the budget exactly (Gumbel noise, then a
//! multiple-choice knapsack), asks the caller for the loss gradient of that
//! assignment per option (a straight-through estimate), removes the
//! gradient's component along the budget surface's normal, takes an Adam
//! step and shifts the logits back onto the surface. The final assignment
//! is the knapsack optimum of the log-probabilities.
//!
//! Follows IST-DASLab/RCO (`src/manifold.py`, `src/search/quant.py`). The
//! reference backpropagates a calibration loss through the model to get the
//! gradient; here the caller supplies it.

use allpaca_types::{AllpacaError, FatalError};

fn invalid(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg))
}

/// What is being chosen: `groups` groups, each taking one of `costs.len()`
/// options. Option `k` costs `costs[k]` per unit of group weight; the
/// weighted mean cost over the groups must equal `budget`.
#[derive(Debug, Clone)]
pub struct Problem {
    /// Cost of each option (e.g. bits per weight).
    pub costs: Vec<f64>,
    /// Weight of each group (e.g. its share of the parameters); any positive scale.
    pub weights: Vec<f64>,
    /// Target weighted mean cost.
    pub budget: f64,
}

impl Problem {
    fn groups(&self) -> usize {
        self.weights.len()
    }

    fn options(&self) -> usize {
        self.costs.len()
    }

    fn fractions(&self) -> Vec<f64> {
        let total: f64 = self.weights.iter().sum();
        self.weights.iter().map(|w| w / total).collect()
    }

    /// Weighted mean cost of a discrete assignment.
    pub fn cost_of(&self, assignment: &[usize]) -> f64 {
        self.fractions().iter().zip(assignment).map(|(f, k)| f * self.costs[*k]).sum()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Expected weighted mean cost of `logits` (`groups * options`, group-major).
pub fn expected_cost(problem: &Problem, logits: &[f64]) -> f64 {
    let k = problem.options();
    problem
        .fractions()
        .iter()
        .zip(logits.chunks(k))
        .map(|(f, row)| f * softmax(row).iter().zip(&problem.costs).map(|(p, c)| p * c).sum::<f64>())
        .sum()
}

/// Normal of the budget surface at `logits`: the gradient of
/// [`expected_cost`], `w_i * p_ik * (c_k - E_i[c])`.
pub fn budget_normal(problem: &Problem, logits: &[f64]) -> Vec<f64> {
    let k = problem.options();
    let mut out = Vec::with_capacity(logits.len());
    for (f, row) in problem.fractions().iter().zip(logits.chunks(k)) {
        let p = softmax(row);
        let mean: f64 = p.iter().zip(&problem.costs).map(|(p, c)| p * c).sum();
        out.extend(p.iter().zip(&problem.costs).map(|(p, c)| f * p * (c - mean)));
    }
    out
}

/// Remove from `v` its component along the budget normal at `logits`, so a
/// step along `v` keeps the expected cost to first order.
pub fn project_tangent(problem: &Problem, logits: &[f64], v: &mut [f64]) {
    let n = budget_normal(problem, logits);
    let nn: f64 = n.iter().map(|x| x * x).sum::<f64>() + 1e-12;
    let coeff = v.iter().zip(&n).map(|(a, b)| a * b).sum::<f64>() / nn;
    for (v, n) in v.iter_mut().zip(&n) {
        *v -= coeff * n;
    }
}

/// Shift every group's logits by `t * costs` with the `t` that puts the
/// expected cost on the budget (found by bisection: the cost is monotone in
/// `t`). Returns the cost reached.
pub fn retract(problem: &Problem, logits: &mut [f64]) -> f64 {
    let k = problem.options();
    let at = |logits: &[f64], t: f64| {
        let shifted: Vec<f64> = logits.iter().enumerate().map(|(i, l)| l + t * problem.costs[i % k]).collect();
        expected_cost(problem, &shifted)
    };
    let (mut lo, mut hi) = (-1.0f64, 1.0f64);
    for _ in 0..60 {
        if at(logits, lo) > problem.budget {
            lo *= 2.0;
        } else if at(logits, hi) < problem.budget {
            hi *= 2.0;
        } else {
            break;
        }
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if at(logits, mid) > problem.budget {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    for (i, l) in logits.iter_mut().enumerate() {
        *l += t * problem.costs[i % k];
    }
    expected_cost(problem, logits)
}

/// The assignment with the highest total `values` (`groups * options`)
/// whose weighted mean cost is the budget, by dynamic programming over the
/// cost in steps of `budget / resolution`. When rounding makes the exact
/// budget unreachable, the nearest reachable cost is used.
pub fn knapsack(problem: &Problem, values: &[f64], resolution: usize) -> Result<Vec<usize>, AllpacaError> {
    let (g, k) = (problem.groups(), problem.options());
    if g == 0 || k == 0 || values.len() != g * k || resolution == 0 {
        return Err(invalid("RCO knapsack: values do not match groups * options".into()));
    }
    let unit = problem.budget / resolution as f64;
    let fractions = problem.fractions();
    let cost = |i: usize, o: usize| (fractions[i] * problem.costs[o] / unit).round() as usize;
    // A little slack above the budget, for rounding.
    let cap = resolution + g + 1;
    let mut best = vec![f64::NEG_INFINITY; cap + 1];
    best[0] = 0.0;
    let mut choice = vec![u8::MAX; g * (cap + 1)];
    for i in 0..g {
        let mut next = vec![f64::NEG_INFINITY; cap + 1];
        for o in 0..k {
            let c = cost(i, o);
            for j in c..=cap {
                let cand = best[j - c] + values[i * k + o];
                if cand > next[j] {
                    next[j] = cand;
                    choice[i * (cap + 1) + j] = o as u8;
                }
            }
        }
        best = next;
    }
    let mut at = (0..=cap)
        .filter(|j| best[*j] > f64::NEG_INFINITY)
        .min_by_key(|j| j.abs_diff(resolution))
        .ok_or_else(|| invalid("RCO knapsack: no assignment fits".into()))?;
    let mut out = vec![0usize; g];
    for i in (0..g).rev() {
        let o = choice[i * (cap + 1) + at] as usize;
        out[i] = o;
        at -= cost(i, o);
    }
    Ok(out)
}

/// Search settings.
#[derive(Debug, Clone)]
pub struct Search {
    pub steps: usize,
    pub learning_rate: f64,
    /// Temperature of the relaxed sample at the first and last step (annealed linearly).
    pub temperature: (f64, f64),
    /// Knapsack cost steps per budget.
    pub resolution: usize,
    pub seed: u64,
}

impl Default for Search {
    fn default() -> Self {
        Self {
            steps: 200,
            learning_rate: 0.1,
            temperature: (1.0, 0.05),
            resolution: 2000,
            seed: 0,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gumbel(&mut self) -> f64 {
        -(-self.uniform().ln()).ln()
    }
}

/// Run the search from `logits` (`groups * options`; they are retracted to
/// the budget first). `gradient(assignment, out)` writes the loss gradient
/// with respect to choosing each option of each group, evaluated at the
/// discrete `assignment`: lower is better. Returns the final assignment.
///
/// When the loss is a plain sum of one term per group and option, the
/// gradient never changes and [`knapsack`] on the negated terms gives the
/// exact optimum at once: use that. The search is for losses where groups
/// interact.
pub fn optimize(
    problem: &Problem,
    logits: &mut [f64],
    search: &Search,
    gradient: &mut dyn FnMut(&[usize], &mut [f64]) -> Result<(), AllpacaError>,
) -> Result<Vec<usize>, AllpacaError> {
    let (g, k) = (problem.groups(), problem.options());
    if logits.len() != g * k {
        return Err(invalid("RCO: logits do not match groups * options".into()));
    }
    let (lo, hi) = problem.costs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| (lo.min(*c), hi.max(*c)));
    if !(lo < problem.budget && problem.budget < hi) {
        return Err(invalid(format!(
            "RCO: budget {} is not strictly between the cheapest and dearest option ({lo}, {hi})",
            problem.budget
        )));
    }
    retract(problem, logits);
    let mut rng = Rng(search.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let (mut m, mut v) = (vec![0.0f64; g * k], vec![0.0f64; g * k]);
    let (b1, b2) = (0.9f64, 0.999f64);
    let mut grad = vec![0.0f64; g * k];
    for step in 0..search.steps {
        let frac = step as f64 / search.steps.max(2).saturating_sub(1) as f64;
        let tau = search.temperature.0 + (search.temperature.1 - search.temperature.0) * frac;
        // One Gumbel draw serves both the discrete sample and its relaxation.
        let noisy: Vec<f64> = logits.iter().map(|l| l + rng.gumbel()).collect();
        let assignment = knapsack(problem, &noisy, search.resolution)?;
        grad.fill(0.0);
        gradient(&assignment, &mut grad)?;
        // Straight through the relaxed sample softmax(noisy / tau):
        // d/d logit_k = p_k * (g_k - sum_j p_j g_j) / tau.
        for (row, g) in noisy.chunks(k).zip(grad.chunks_mut(k)) {
            let scaled: Vec<f64> = row.iter().map(|v| v / tau).collect();
            let p = softmax(&scaled);
            let mean: f64 = p.iter().zip(g.iter()).map(|(p, g)| p * g).sum();
            for (g, p) in g.iter_mut().zip(&p) {
                *g = p * (*g - mean) / tau;
            }
        }
        project_tangent(problem, logits, &mut grad);
        let t = (step + 1) as i32;
        for i in 0..g * k {
            m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
            v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
            let (mh, vh) = (m[i] / (1.0 - b1.powi(t)), v[i] / (1.0 - b2.powi(t)));
            logits[i] -= search.learning_rate * mh / (vh.sqrt() + 1e-8);
        }
        retract(problem, logits);
        // The tangent plane has turned: keep the momentum in it.
        project_tangent(problem, logits, &mut m);
    }
    let log_probs: Vec<f64> = logits.chunks(k).flat_map(|row| softmax(row).into_iter().map(|p| p.max(1e-10).ln())).collect();
    knapsack(problem, &log_probs, search.resolution)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn problem() -> Problem {
        Problem {
            costs: vec![2.0, 3.0, 4.0],
            weights: vec![1.0, 1.0, 2.0, 4.0],
            budget: 3.0,
        }
    }

    /// Every assignment whose cost is within `tol` of the budget.
    fn feasible(p: &Problem, tol: f64) -> Vec<Vec<usize>> {
        let (g, k) = (p.groups(), p.options());
        (0..k.pow(g as u32))
            .map(|mut code| {
                (0..g)
                    .map(|_| {
                        let o = code % k;
                        code /= k;
                        o
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|a| (p.cost_of(a) - p.budget).abs() < tol)
            .collect()
    }

    #[test]
    fn knapsack_finds_the_best_assignment_on_the_budget() {
        let p = problem();
        let values = [0.9, 0.2, 0.1, 0.1, 0.3, 0.8, 0.5, 0.1, 0.4, 0.2, 0.6, 0.3];
        let total = |a: &[usize]| a.iter().enumerate().map(|(i, o)| values[i * 3 + o]).sum::<f64>();
        let got = knapsack(&p, &values, 800).unwrap();
        assert!((p.cost_of(&got) - 3.0).abs() < 1e-9, "cost {}", p.cost_of(&got));
        let best = feasible(&p, 1e-9).into_iter().map(|a| total(&a)).fold(f64::NEG_INFINITY, f64::max);
        assert!((total(&got) - best).abs() < 1e-12, "{got:?}: {} vs {best}", total(&got));
    }

    #[test]
    fn retraction_puts_the_expected_cost_on_the_budget() {
        let p = problem();
        let mut logits = vec![0.3, -1.0, 2.0, 0.0, 0.0, 0.0, -2.0, 1.0, 0.5, 4.0, -1.0, -3.0];
        assert!((expected_cost(&p, &logits) - 3.0).abs() > 0.05);
        let reached = retract(&p, &mut logits);
        assert!((reached - 3.0).abs() < 1e-9, "{reached}");
        assert!((expected_cost(&p, &logits) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn projection_keeps_the_cost_to_first_order() {
        let p = problem();
        let mut logits = vec![0.3, -1.0, 2.0, 0.0, 0.0, 0.0, -2.0, 1.0, 0.5, 4.0, -1.0, -3.0];
        retract(&p, &mut logits);
        // The normal is the gradient of the expected cost.
        let n = budget_normal(&p, &logits);
        for i in [0usize, 5, 10] {
            let mut bumped = logits.clone();
            bumped[i] += 1e-6;
            let slope = (expected_cost(&p, &bumped) - expected_cost(&p, &logits)) / 1e-6;
            assert!((slope - n[i]).abs() < 1e-5, "normal[{i}] {} vs slope {slope}", n[i]);
        }
        let mut step = vec![1.0, -0.5, 0.2, 0.7, 0.0, -0.3, 0.4, 0.4, -0.9, 0.1, 0.6, -0.2];
        project_tangent(&p, &logits, &mut step);
        assert!(step.iter().zip(&n).map(|(a, b)| a * b).sum::<f64>().abs() < 1e-9);
        let moved: Vec<f64> = logits.iter().zip(&step).map(|(l, s)| l + 1e-3 * s).collect();
        assert!((expected_cost(&p, &moved) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn search_comes_close_to_the_cheapest_loss_on_the_budget() {
        // Many groups of differing size and sensitivity; within a group each
        // extra bit helps less than the one before. The search sees only
        // gradients. (With a handful of lumpy groups the relaxation it
        // descends is loose and it can settle on a worse assignment.)
        let mut rng = Rng(7);
        let groups = 48;
        let p = Problem {
            costs: vec![2.0, 3.0, 4.0],
            weights: (0..groups).map(|_| 1.0 + rng.uniform()).collect(),
            budget: 3.0,
        };
        let loss: Vec<f64> = (0..groups)
            .flat_map(|_| {
                let base = 0.2 + 3.0 * rng.uniform();
                [base, 0.3 * base, 0.05 * base]
            })
            .collect();
        let total = |a: &[usize]| a.iter().enumerate().map(|(i, o)| loss[i * 3 + o]).sum::<f64>();
        let gain: Vec<f64> = loss.iter().map(|l| -l).collect();
        let best = knapsack(&p, &gain, 2000).unwrap();
        let mut logits = vec![0.0f64; groups * 3];
        let start = knapsack(&p, &logits, 2000).unwrap();
        let mut calls = 0;
        let got = optimize(&p, &mut logits, &Search::default(), &mut |assignment, grad| {
            calls += 1;
            assert!((p.cost_of(assignment) - 3.0).abs() < 0.02, "sampled assignments stay on the budget");
            grad.copy_from_slice(&loss);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 200);
        assert!((p.cost_of(&got) - 3.0).abs() < 0.02, "{}", p.cost_of(&got));
        assert!(total(&got) < 0.8 * total(&start), "search {} vs start {}", total(&got), total(&start));
        assert!(total(&got) < 1.05 * total(&best), "search {} vs exact optimum {}", total(&got), total(&best));
        // The logits stayed on the budget surface throughout.
        assert!((expected_cost(&p, &logits) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_budget_outside_the_options_is_refused() {
        let mut p = problem();
        p.budget = 4.5;
        let mut logits = vec![0.0f64; 12];
        assert!(optimize(&p, &mut logits, &Search::default(), &mut |_, _| Ok(())).is_err());
    }
}
