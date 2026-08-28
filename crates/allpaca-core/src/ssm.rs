//! CPU Mamba-2 / selective-SSM stub for hybrid schedules (not production kernels).

#[derive(Debug, Clone)]
pub struct SsmLayerWeights {
    pub d_state: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d_skip: Vec<f32>,
    pub dt_bias: f32,
}

/// Softplus: ln(1 + e^x), stable for moderate x.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// One-token selective SSM over `d_model` channels sharing `d_state`.
///
/// For each channel `i`:
/// `A_bar[j] = exp(dt * a[j])`, `h[i,j] = A_bar[j]*h[i,j] + dt*b[j]*x[i]`,
/// `y[i] = Σ_j c[j]*h[i,j] + d_skip[i]*x[i]`.
///
/// `h` is row-major `[d_model * d_state]`. Tiny-dim reference path only.
pub fn selective_step(
    x: &[f32],
    h: &mut [f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    dt_raw: f32,
    d_skip: &[f32],
    y: &mut [f32],
) {
    let d_model = x.len();
    let d_state = a.len();
    debug_assert_eq!(b.len(), d_state);
    debug_assert_eq!(c.len(), d_state);
    debug_assert_eq!(d_skip.len(), d_model);
    debug_assert_eq!(h.len(), d_model.saturating_mul(d_state));
    debug_assert_eq!(y.len(), d_model);

    let dt = softplus(dt_raw);
    let mut a_bar = vec![0.0f32; d_state];
    for j in 0..d_state {
        a_bar[j] = (dt * a[j]).exp();
    }
    for i in 0..d_model {
        let mut acc = 0.0f32;
        let base = i * d_state;
        for j in 0..d_state {
            let hj = a_bar[j] * h[base + j] + dt * b[j] * x[i];
            h[base + j] = hj;
            acc += c[j] * hj;
        }
        y[i] = acc + d_skip[i] * x[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_step_matches_fixed_tiny_reference() {
        // d_model=2, d_state=2 — hand-computed for dt_raw=0 → softplus(0)=ln2.
        let x = [1.0f32, -0.5];
        let a = [-1.0f32, -2.0];
        let b = [0.5f32, 1.0];
        let c = [1.0f32, -1.0];
        let d_skip = [0.25f32, 0.5];
        let mut h = [0.0f32; 4];
        let mut y = [0.0f32; 2];
        selective_step(&x, &mut h, &a, &b, &c, 0.0, &d_skip, &mut y);

        let dt = (2.0f32).ln();
        let a0 = (dt * -1.0).exp();
        let a1 = (dt * -2.0).exp();
        let h00 = a0 * 0.0 + dt * 0.5 * 1.0;
        let h01 = a1 * 0.0 + dt * 1.0 * 1.0;
        let h10 = a0 * 0.0 + dt * 0.5 * -0.5;
        let h11 = a1 * 0.0 + dt * 1.0 * -0.5;
        let y0 = 1.0 * h00 + -1.0 * h01 + 0.25 * 1.0;
        let y1 = 1.0 * h10 + -1.0 * h11 + 0.5 * -0.5;

        for (got, want) in h.iter().zip([h00, h01, h10, h11]) {
            assert!(
                (got - want).abs() < 1e-5,
                "h mismatch got={got} want={want}"
            );
        }
        assert!((y[0] - y0).abs() < 1e-5, "y0 got={} want={y0}", y[0]);
        assert!((y[1] - y1).abs() < 1e-5, "y1 got={} want={y1}", y[1]);

        // Second step must advance state (not reset).
        let mut y2 = [0.0f32; 2];
        selective_step(&x, &mut h, &a, &b, &c, 0.0, &d_skip, &mut y2);
        assert!(
            (y2[0] - y[0]).abs() > 1e-4 || (h[0] - h00).abs() > 1e-4,
            "state must carry across tokens"
        );
    }
}
