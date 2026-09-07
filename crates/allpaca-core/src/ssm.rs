//! CPU reference for the one-token Mamba-2 SSD block used by Nemotron-H.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mamba2Dims {
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub n_heads: usize,
    pub n_groups: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub conv_dim: usize,
    pub in_proj_dim: usize,
}

impl Mamba2Dims {
    pub fn new(
        d_model: usize,
        d_inner: usize,
        d_state: usize,
        n_heads: usize,
        n_groups: usize,
        conv_kernel: usize,
    ) -> Option<Self> {
        if d_model == 0
            || d_inner == 0
            || d_state == 0
            || n_heads == 0
            || n_groups == 0
            || conv_kernel == 0
            || d_inner % n_heads != 0
            || n_heads % n_groups != 0
            || d_inner % n_groups != 0
        {
            return None;
        }
        let head_dim = d_inner / n_heads;
        let conv_dim = d_inner.checked_add(2usize.checked_mul(n_groups.checked_mul(d_state)?)?)?;
        let in_proj_dim = d_inner.checked_add(conv_dim)?.checked_add(n_heads)?;
        Some(Self {
            d_model,
            d_inner,
            d_state,
            n_heads,
            n_groups,
            head_dim,
            conv_kernel,
            conv_dim,
            in_proj_dim,
        })
    }

    pub fn state_floats(self) -> usize {
        self.d_inner.saturating_mul(self.d_state)
    }

    pub fn conv_state_floats(self) -> usize {
        self.conv_kernel.saturating_sub(1).saturating_mul(self.conv_dim)
    }
}

#[derive(Debug, Clone)]
pub struct SsmLayerWeights {
    pub dims: Mamba2Dims,
    /// GGUF row-major logical matrix `[in_proj_dim, d_model]`.
    pub w_in: Vec<f32>,
    /// GGUF row-major logical matrix `[d_model, d_inner]`.
    pub w_out: Vec<f32>,
    pub a: Vec<f32>,
    pub d_skip: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Depthwise conv weights, GGUF layout `[channel][kernel]`.
    pub conv1d: Vec<f32>,
    pub conv_bias: Vec<f32>,
    /// Grouped RMSNorm weights, flattened `[group][d_inner / n_groups]`.
    pub norm: Vec<f32>,
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

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn matvec(weight: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weight.len(), rows.saturating_mul(cols));
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &weight[r * cols..(r + 1) * cols];
        out[r] = row.iter().zip(x).map(|(w, v)| w * v).sum();
    }
}

pub fn validate_weights(w: &SsmLayerWeights) -> Result<(), String> {
    let d = w.dims;
    for (name, got, want) in [
        ("ssm_in.weight", w.w_in.len(), d.in_proj_dim.saturating_mul(d.d_model)),
        ("ssm_out.weight", w.w_out.len(), d.d_model.saturating_mul(d.d_inner)),
        ("ssm_a", w.a.len(), d.n_heads),
        ("ssm_d", w.d_skip.len(), d.n_heads),
        ("ssm_dt.bias", w.dt_bias.len(), d.n_heads),
        ("ssm_conv1d.weight", w.conv1d.len(), d.conv_dim.saturating_mul(d.conv_kernel)),
        ("ssm_norm.weight", w.norm.len(), d.d_inner),
    ] {
        if got != want {
            return Err(format!("{name} elements={got}, expected={want} for {d:?}"));
        }
    }
    if !w.conv_bias.is_empty() && w.conv_bias.len() != d.conv_dim {
        return Err(format!(
            "ssm_conv1d.bias elements={}, expected={} or absent",
            w.conv_bias.len(),
            d.conv_dim
        ));
    }
    Ok(())
}

/// Production-faithful autoregressive Mamba-2 block for one token.
///
/// Projection order follows llama.cpp / the GGUF contract:
/// `z[d_inner] | x[d_inner] | B[n_groups*d_state] |
/// C[n_groups*d_state] | dt[n_heads]`.
pub fn mamba2_step(
    input: &[f32],
    state: &mut [f32],
    conv_state: &mut [f32],
    w: &SsmLayerWeights,
    eps: f32,
    out: &mut [f32],
) -> Result<(), String> {
    validate_weights(w)?;
    let d = w.dims;
    if input.len() != d.d_model
        || out.len() != d.d_model
        || state.len() != d.state_floats()
        || conv_state.len() != d.conv_state_floats()
    {
        return Err(format!(
            "Mamba-2 activation/state mismatch input={} out={} state={} conv={} dims={d:?}",
            input.len(),
            out.len(),
            state.len(),
            conv_state.len()
        ));
    }

    let mut proj = vec![0.0f32; d.in_proj_dim];
    matvec(&w.w_in, d.in_proj_dim, d.d_model, input, &mut proj);
    let z = &proj[..d.d_inner];
    let xbc_start = d.d_inner;
    let dt_start = xbc_start + d.conv_dim;
    let xbc_raw = &proj[xbc_start..dt_start];
    let dt_raw = &proj[dt_start..];

    let hist = d.conv_kernel - 1;
    let mut xbc = vec![0.0f32; d.conv_dim];
    for channel in 0..d.conv_dim {
        let mut acc = w.conv_bias.get(channel).copied().unwrap_or(0.0);
        for k in 0..hist {
            acc += w.conv1d[k + channel * d.conv_kernel]
                * conv_state[k * d.conv_dim + channel];
        }
        acc += w.conv1d[hist + channel * d.conv_kernel] * xbc_raw[channel];
        xbc[channel] = silu(acc);
    }
    if hist > 0 {
        for k in 0..hist.saturating_sub(1) {
            let src = (k + 1) * d.conv_dim;
            conv_state.copy_within(src..src + d.conv_dim, k * d.conv_dim);
        }
        let last = (hist - 1) * d.conv_dim;
        conv_state[last..last + d.conv_dim].copy_from_slice(xbc_raw);
    }

    let x = &xbc[..d.d_inner];
    let b_start = d.d_inner;
    let c_start = b_start + d.n_groups * d.d_state;
    let b = &xbc[b_start..c_start];
    let c = &xbc[c_start..];
    let heads_per_group = d.n_heads / d.n_groups;
    let mut y = vec![0.0f32; d.d_inner];
    for head in 0..d.n_heads {
        let group = head / heads_per_group;
        let dt = softplus(dt_raw[head] + w.dt_bias[head]);
        for lane in 0..d.head_dim {
            let channel = head * d.head_dim + lane;
            let state_row = &mut state[channel * d.d_state..(channel + 1) * d.d_state];
            let mut sum = 0.0f32;
            for n in 0..d.d_state {
                let a_bar = (dt * w.a[head]).exp();
                let value = a_bar * state_row[n]
                    + dt * b[group * d.d_state + n] * x[channel];
                state_row[n] = value;
                sum += c[group * d.d_state + n] * value;
            }
            y[channel] = sum + w.d_skip[head] * x[channel];
            y[channel] *= silu(z[channel]);
        }
    }

    let group_width = d.d_inner / d.n_groups;
    for group in 0..d.n_groups {
        let start = group * group_width;
        let row = &mut y[start..start + group_width];
        let inv = 1.0
            / (row.iter().map(|v| v * v).sum::<f32>() / group_width as f32 + eps).sqrt();
        for i in 0..group_width {
            row[i] *= inv * w.norm[start + i];
        }
    }
    matvec(&w.w_out, d.d_model, d.d_inner, &y, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_step_uses_projection_conv_selectivity_and_state() {
        let dims = Mamba2Dims::new(2, 2, 1, 1, 1, 2).unwrap();
        let mut w_in = vec![0.0; dims.in_proj_dim * dims.d_model];
        // z=x=input, B=0.5*x0, C=x1, dt=x0.
        for (row, col, value) in [
            (0, 0, 1.0), (1, 1, 1.0), (2, 0, 1.0), (3, 1, 1.0),
            (4, 0, 0.5), (5, 1, 1.0), (6, 0, 1.0),
        ] {
            w_in[row * dims.d_model + col] = value;
        }
        let weights = SsmLayerWeights {
            dims,
            w_in,
            w_out: vec![1.0, 0.0, 0.0, 1.0],
            a: vec![-1.0],
            d_skip: vec![0.25],
            dt_bias: vec![0.0],
            // Current sample only; the second tap makes history observable next step.
            conv1d: vec![0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
            conv_bias: vec![0.0; 4],
            norm: vec![1.0; 2],
        };
        let mut state = vec![0.0; dims.state_floats()];
        let mut conv = vec![0.0; dims.conv_state_floats()];
        let mut first = vec![0.0; 2];
        mamba2_step(&[1.0, -0.5], &mut state, &mut conv, &weights, 1e-5, &mut first).unwrap();
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(state.iter().any(|v| *v != 0.0), "SSD state must advance");
        assert_eq!(conv, vec![1.0, -0.5, 0.5, -0.5]);

        let before = state.clone();
        let mut second = vec![0.0; 2];
        mamba2_step(&[0.25, 0.75], &mut state, &mut conv, &weights, 1e-5, &mut second).unwrap();
        assert_ne!(state, before, "input-dependent dt/B/C must alter state");
        assert_ne!(first, second, "conv history and selectivity must alter output");
    }

    #[test]
    fn invalid_layout_fails_truthfully() {
        assert!(Mamba2Dims::new(8, 6, 2, 4, 2, 4).is_none());
    }
}
