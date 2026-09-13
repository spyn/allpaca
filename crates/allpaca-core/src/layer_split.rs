use allpaca_types::{AllpacaError, FatalError};

/// Per-layer owner index into the split device list (local target first, then RPC).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPlan {
    pub owners: Vec<usize>,
}

impl LayerPlan {
    pub fn all_local(n_layers: usize) -> Self {
        Self {
            owners: vec![0; n_layers],
        }
    }

    pub fn owner(&self, layer: usize) -> usize {
        self.owners.get(layer).copied().unwrap_or(0)
    }

    pub fn has_remote(&self) -> bool {
        self.owners.iter().any(|&o| o != 0)
    }

    /// Contiguous runs of `(owner, start, end)` (end exclusive).
    pub fn runs(&self) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        if self.owners.is_empty() {
            return out;
        }
        let mut owner = self.owners[0];
        let mut start = 0usize;
        for (i, &o) in self.owners.iter().enumerate().skip(1) {
            if o != owner {
                out.push((owner, start, i));
                owner = o;
                start = i;
            }
        }
        out.push((owner, start, self.owners.len()));
        out
    }
}

/// Assign contiguous layer ranges by relative weights (VRAM or `--tensor-split`).
pub fn plan_layer_owners(
    n_layers: usize,
    device_bytes: &[u64],
    tensor_split: Option<&[f64]>,
) -> Result<LayerPlan, AllpacaError> {
    if n_layers == 0 {
        return Ok(LayerPlan { owners: Vec::new() });
    }
    if device_bytes.is_empty() {
        return Ok(LayerPlan::all_local(n_layers));
    }
    let n_dev = device_bytes.len();
    let weights: Vec<f64> = if let Some(split) = tensor_split {
        if split.len() != n_dev {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "tensor_split has {} entries but {} split devices",
                split.len(),
                n_dev
            ))));
        }
        split.to_vec()
    } else {
        device_bytes.iter().map(|b| (*b as f64).max(1.0)).collect()
    };
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Ok(LayerPlan::all_local(n_layers));
    }
    let mut owners = vec![0usize; n_layers];
    let mut acc = 0.0f64;
    let mut cuts = Vec::with_capacity(n_dev);
    for w in &weights {
        acc += *w;
        cuts.push((acc / total) * n_layers as f64);
    }
    for (layer, slot) in owners.iter_mut().enumerate() {
        let mut owner = n_dev - 1;
        for (j, cut) in cuts.iter().enumerate() {
            if (layer as f64) < *cut {
                owner = j;
                break;
            }
        }
        *slot = owner;
    }
    Ok(LayerPlan { owners })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_device_all_local() {
        let plan = plan_layer_owners(4, &[8 << 30], None).unwrap();
        assert_eq!(plan.owners, vec![0, 0, 0, 0]);
        assert!(!plan.has_remote());
    }

    #[test]
    fn even_two_devices() {
        let plan = plan_layer_owners(2, &[8 << 30, 8 << 30], None).unwrap();
        assert_eq!(plan.owners, vec![0, 1]);
        assert!(plan.has_remote());
        assert_eq!(plan.runs(), vec![(0, 0, 1), (1, 1, 2)]);
    }

    #[test]
    fn tensor_split_overrides_memory() {
        let plan = plan_layer_owners(4, &[1, 100], Some(&[3.0, 1.0])).unwrap();
        assert_eq!(plan.owners, vec![0, 0, 0, 1]);
    }

    #[test]
    fn split_len_mismatch_is_fatal() {
        assert!(plan_layer_owners(2, &[1, 1], Some(&[1.0])).is_err());
    }
}
