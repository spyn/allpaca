//! Operator controls over a MoE model's routed experts: a per-layer disabled
//! set applied at routing time, and per-(layer, expert) routing-hit counts.
//!
//! Disabling works on the router, not the weights: a disabled expert's logit
//! is excluded before top-k on every path (host `topk_from_logits`, the HIP
//! router kernel), so it is never picked and its weight mass goes to the
//! remaining experts through the usual top-k renormalization. Decode, verify
//! and batched forwards share the same routing calls, so they keep agreeing.
//! With nothing disabled every path computes exactly what it did before.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use allpaca_types::{AllpacaError, FatalError};

/// Routed-expert shape the controls were sized for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertShape {
    pub n_layers: usize,
    pub n_experts: usize,
    /// Experts picked per token (top-k).
    pub n_active: usize,
}

#[derive(Default)]
struct Masks {
    shape: ExpertShape,
    /// `layers[l]` is `None` when layer `l` has nothing disabled.
    layers: Vec<Option<Arc<Vec<bool>>>>,
}

/// Disabled experts and usage counters for one loaded model.
#[derive(Default)]
pub struct ExpertControls {
    masks: RwLock<Masks>,
    /// Bumped on every mask change; device mask copies key on it.
    generation: AtomicU64,
    /// Count routing hits (host paths here, device paths in a device buffer).
    recording: AtomicBool,
    host_counts: RwLock<Vec<AtomicU64>>,
    /// Bumped by `reset_usage`; device counters clear themselves when they see
    /// a new epoch.
    usage_epoch: AtomicU64,
}

impl ExpertControls {
    /// Size for a MoE model. Clears any previous mask and counts.
    pub fn configure(&self, shape: ExpertShape) {
        if let Ok(mut m) = self.masks.write() {
            m.shape = shape;
            m.layers = vec![None; shape.n_layers];
        }
        if let Ok(mut c) = self.host_counts.write() {
            *c = (0..shape.n_layers * shape.n_experts)
                .map(|_| AtomicU64::new(0))
                .collect();
        }
        self.generation.fetch_add(1, Ordering::Relaxed);
        self.usage_epoch.fetch_add(1, Ordering::Relaxed);
    }

    pub fn shape(&self) -> ExpertShape {
        self.masks.read().map(|m| m.shape).unwrap_or_default()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Replace the whole disabled set: `disabled[l]` lists layer `l`'s
    /// disabled expert indices. Every layer must keep at least `n_active`
    /// experts enabled, or top-k would have to pick a disabled one.
    pub fn set_disabled(&self, disabled: &[Vec<u32>]) -> Result<(), AllpacaError> {
        let mut m = self
            .masks
            .write()
            .map_err(|_| invalid("expert mask lock".into()))?;
        let shape = m.shape;
        if shape.n_experts == 0 {
            if disabled.iter().all(Vec::is_empty) {
                return Ok(());
            }
            return Err(invalid("model has no routed experts".into()));
        }
        if disabled.len() > shape.n_layers {
            return Err(invalid(format!(
                "disabled experts list {} layers, model has {}",
                disabled.len(),
                shape.n_layers
            )));
        }
        let mut layers = vec![None; shape.n_layers];
        for (l, ids) in disabled.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }
            let mut mask = vec![false; shape.n_experts];
            for &e in ids {
                let e = e as usize;
                if e >= shape.n_experts {
                    return Err(invalid(format!(
                        "layer {l}: expert {e} out of range (n_experts {})",
                        shape.n_experts
                    )));
                }
                mask[e] = true;
            }
            let off = mask.iter().filter(|d| **d).count();
            if shape.n_experts - off < shape.n_active {
                return Err(invalid(format!(
                    "layer {l}: {off} of {} experts disabled leaves fewer than the {} each token routes to",
                    shape.n_experts, shape.n_active
                )));
            }
            layers[l] = Some(Arc::new(mask));
        }
        m.layers = layers;
        self.generation.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Disabled set per layer, as indices.
    pub fn disabled(&self) -> Vec<Vec<u32>> {
        let Ok(m) = self.masks.read() else {
            return Vec::new();
        };
        m.layers
            .iter()
            .map(|l| {
                l.as_ref().map_or_else(Vec::new, |mask| {
                    mask.iter()
                        .enumerate()
                        .filter(|(_, d)| **d)
                        .map(|(i, _)| i as u32)
                        .collect()
                })
            })
            .collect()
    }

    /// Layer `l`'s mask (`true` = disabled), or `None` when nothing is.
    pub fn layer_mask(&self, layer: usize) -> Option<Arc<Vec<bool>>> {
        let m = self.masks.read().ok()?;
        m.layers.get(layer).cloned().flatten()
    }

    pub fn set_recording(&self, on: bool) {
        self.recording.store(on, Ordering::Relaxed);
    }

    pub fn recording(&self) -> bool {
        self.recording.load(Ordering::Relaxed)
    }

    /// Count one host-routed token's picks for `layer`.
    pub fn record(&self, layer: usize, routed: &[(usize, f32)]) {
        if !self.recording() {
            return;
        }
        let Ok(c) = self.host_counts.read() else {
            return;
        };
        let n_exp = self.shape().n_experts;
        for &(e, _) in routed {
            if let Some(slot) = c.get(layer * n_exp + e) {
                slot.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Host-path counts, `n_layers * n_experts`, layer-major.
    pub fn host_usage(&self) -> Vec<u64> {
        self.host_counts
            .read()
            .map(|c| c.iter().map(|a| a.load(Ordering::Relaxed)).collect())
            .unwrap_or_default()
    }

    pub fn usage_epoch(&self) -> u64 {
        self.usage_epoch.load(Ordering::Relaxed)
    }

    pub fn reset_usage(&self) {
        if let Ok(c) = self.host_counts.read() {
            for a in c.iter() {
                a.store(0, Ordering::Relaxed);
            }
        }
        self.usage_epoch.fetch_add(1, Ordering::Relaxed);
    }
}

impl std::fmt::Debug for ExpertControls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertControls")
            .field("shape", &self.shape())
            .field("generation", &self.generation())
            .field("recording", &self.recording())
            .finish()
    }
}

fn invalid(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::InvalidConfig(msg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controls() -> ExpertControls {
        let c = ExpertControls::default();
        c.configure(ExpertShape {
            n_layers: 2,
            n_experts: 4,
            n_active: 2,
        });
        c
    }

    #[test]
    fn disabled_round_trips_and_bumps_generation() {
        let c = controls();
        let g = c.generation();
        c.set_disabled(&[vec![], vec![3, 1]]).unwrap();
        assert!(c.generation() > g);
        assert!(c.layer_mask(0).is_none());
        assert_eq!(*c.layer_mask(1).unwrap(), vec![false, true, false, true]);
        assert_eq!(c.disabled(), vec![vec![], vec![1, 3]]);
    }

    #[test]
    fn must_leave_top_k_enabled() {
        let c = controls();
        assert!(c.set_disabled(&[vec![0, 1, 2]]).is_err());
        assert!(c.set_disabled(&[vec![9]]).is_err());
        assert!(c.set_disabled(&[vec![], vec![], vec![0]]).is_err());
        c.set_disabled(&[vec![0, 1]]).unwrap();
    }

    #[test]
    fn records_only_while_recording() {
        let c = controls();
        c.record(1, &[(2, 0.5)]);
        assert!(c.host_usage().iter().all(|&n| n == 0));
        c.set_recording(true);
        c.record(1, &[(2, 0.5), (0, 0.5)]);
        let u = c.host_usage();
        assert_eq!(u[4 + 2], 1);
        assert_eq!(u[4], 1);
        c.reset_usage();
        assert!(c.host_usage().iter().all(|&n| n == 0));
    }
}
