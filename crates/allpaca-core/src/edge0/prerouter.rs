//! Prerouter heads + cross-token stager (Edge0 double-shift semantics).

use std::path::Path;

use allpaca_safetensors::SafetensorsFile;
use allpaca_types::{AllpacaError, FatalError};

use super::routing::route_logits;
use super::spec::{PrerouterSpec, RouterKind};
use crate::weight_set::ExpertId;

/// One owner's fc1/fc2 (+ optional linear residual init).
#[derive(Debug, Clone)]
pub struct PrerouterHead {
    pub owner_layer: usize,
    pub fc1: Vec<f32>,    // [hidden, feat_dim]
    pub fc2: Vec<f32>,    // [n_experts, hidden]
    pub linear: Vec<f32>, // [n_experts, feat_dim] residual init path
    pub feat_dim: usize,
    pub hidden: usize,
    pub n_experts: usize,
}

impl PrerouterHead {
    /// `fc1 → erf-GELU → fc2 + linear(x)`.
    pub fn forward(&self, features: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32; self.hidden];
        for i in 0..self.hidden {
            let row = &self.fc1[i * self.feat_dim..(i + 1) * self.feat_dim];
            let mut s = 0.0f32;
            for j in 0..self.feat_dim.min(features.len()) {
                s += features[j] * row[j];
            }
            // Exact GELU via erf: 0.5 * x * (1 + erf(x / sqrt(2)))
            h[i] = 0.5 * s * (1.0 + erf_approx(s * std::f32::consts::FRAC_1_SQRT_2));
        }
        let mut out = vec![0.0f32; self.n_experts];
        for e in 0..self.n_experts {
            let row = &self.fc2[e * self.hidden..(e + 1) * self.hidden];
            let mut s = 0.0f32;
            for i in 0..self.hidden {
                s += h[i] * row[i];
            }
            if !self.linear.is_empty() {
                let lrow = &self.linear[e * self.feat_dim..(e + 1) * self.feat_dim];
                for j in 0..self.feat_dim.min(features.len()) {
                    s += features[j] * lrow[j];
                }
            }
            out[e] = s;
        }
        out
    }
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Feature = concat[hidden, topk_t onehot, topk_tm1 onehot].
pub fn build_features(
    hidden: &[f32],
    topk_t: &[usize],
    topk_tm1: &[usize],
    n_experts: usize,
    top_k: usize,
) -> Vec<f32> {
    let mut feat = Vec::with_capacity(hidden.len() + 2 * n_experts);
    feat.extend_from_slice(hidden);
    let mut oh_t = vec![0.0f32; n_experts];
    let mut oh_tm1 = vec![0.0f32; n_experts];
    for &e in topk_t.iter().take(top_k) {
        if e < n_experts {
            oh_t[e] = 1.0;
        }
    }
    for &e in topk_tm1.iter().take(top_k) {
        if e < n_experts {
            oh_tm1[e] = 1.0;
        }
    }
    feat.extend_from_slice(&oh_t);
    feat.extend_from_slice(&oh_tm1);
    feat
}

#[derive(Debug, Clone, Default)]
pub struct PrerouterPrediction {
    /// Consumer layer → predicted (expert, weight) pairs.
    pub by_layer: Vec<Vec<(usize, f32)>>,
}

impl PrerouterPrediction {
    pub fn expert_ids(&self) -> Vec<ExpertId> {
        let mut out = Vec::new();
        for (layer, routed) in self.by_layer.iter().enumerate() {
            for &(e, _) in routed {
                out.push(ExpertId {
                    layer: layer as u32,
                    expert: e as u32,
                });
            }
        }
        out
    }
}

#[derive(Debug, Clone, Default)]
pub struct PrerouterHeads {
    pub spec: Option<PrerouterSpec>,
    pub heads: Vec<PrerouterHead>,
    pub n_experts: usize,
    pub model_dim: usize,
}

impl PrerouterHeads {
    pub fn load(
        path: &Path,
        spec: &PrerouterSpec,
        n_experts: usize,
        model_dim: usize,
    ) -> Result<Self, AllpacaError> {
        if !path.is_file() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "Edge0 prerouter missing: {}",
                path.display()
            ))));
        }
        let file = SafetensorsFile::open(path)?;
        let top_k = spec.top_k;
        let feat_dim = model_dim + 2 * n_experts;
        let mut heads = Vec::new();
        for owner in spec.owners.clone() {
            let prefix_candidates = [
                format!("layers.{owner}."),
                format!("prerouter.layers.{owner}."),
                format!("heads.{owner}."),
            ];
            let mut loaded = None;
            for pfx in &prefix_candidates {
                let fc1_name = format!("{pfx}fc1.weight");
                let fc2_name = format!("{pfx}fc2.weight");
                if file.tensor(&fc1_name).is_none() {
                    continue;
                }
                let fc1 = file.tensor_f32(&fc1_name)?;
                let fc2 = file.tensor_f32(&fc2_name)?;
                let linear = file
                    .tensor_f32(&format!("{pfx}linear.weight"))
                    .unwrap_or_default();
                loaded = Some(PrerouterHead {
                    owner_layer: owner,
                    fc1,
                    fc2,
                    linear,
                    feat_dim,
                    hidden: spec.hidden,
                    n_experts,
                });
                break;
            }
            if let Some(h) = loaded {
                heads.push(h);
            } else {
                // Synthetic zero head so stager still runs in tests / partial installs.
                heads.push(PrerouterHead {
                    owner_layer: owner,
                    fc1: vec![0.0; spec.hidden * feat_dim],
                    fc2: vec![0.0; n_experts * spec.hidden],
                    linear: vec![0.0; n_experts * feat_dim],
                    feat_dim,
                    hidden: spec.hidden,
                    n_experts,
                });
                let _ = top_k;
            }
        }
        Ok(Self {
            spec: Some(spec.clone()),
            heads,
            n_experts,
            model_dim,
        })
    }

    /// Build zero heads (no weight file) for synthetic Edge0 / `--no-prerouter` stubs.
    pub fn zeros(spec: &PrerouterSpec, n_experts: usize, model_dim: usize) -> Self {
        let feat_dim = model_dim + 2 * n_experts;
        let heads = spec
            .owners
            .clone()
            .map(|owner| PrerouterHead {
                owner_layer: owner,
                fc1: vec![0.0; spec.hidden * feat_dim],
                fc2: vec![0.0; n_experts * spec.hidden],
                linear: vec![0.0; n_experts * feat_dim],
                feat_dim,
                hidden: spec.hidden,
                n_experts,
            })
            .collect();
        Self {
            spec: Some(spec.clone()),
            heads,
            n_experts,
            model_dim,
        }
    }
}

/// Cross-token stager: owner N at token t predicts layer N+1 for token t+1.
#[derive(Debug, Clone)]
pub struct CrossTokenStager {
    pub spec: PrerouterSpec,
    pub heads: PrerouterHeads,
    pub router_kind: RouterKind,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub routed_scaling: Option<f32>,
    /// Pending predictions committed at step boundary: consumer_layer → experts.
    pub pending: Vec<Vec<(usize, f32)>>,
    pub prev_topk: Vec<Vec<usize>>,
    pub n_layers: usize,
}

impl CrossTokenStager {
    pub fn new(
        spec: PrerouterSpec,
        heads: PrerouterHeads,
        router_kind: RouterKind,
        n_layers: usize,
        n_group: Option<usize>,
        topk_group: Option<usize>,
        routed_scaling: Option<f32>,
    ) -> Self {
        Self {
            pending: vec![Vec::new(); n_layers],
            prev_topk: vec![Vec::new(); n_layers],
            spec,
            heads,
            router_kind,
            n_group,
            topk_group,
            routed_scaling,
            n_layers,
        }
    }

    /// After layer `owner` MoE executes with `exec_topk`, run head and stage consumer=owner+1.
    pub fn note_executed(
        &mut self,
        owner: usize,
        hidden: &[f32],
        exec_topk: &[usize],
    ) -> Option<Vec<(usize, f32)>> {
        let consumer = owner + 1;
        if consumer >= self.n_layers || !self.spec.owners.contains(&owner) {
            if owner < self.prev_topk.len() {
                self.prev_topk[owner] = exec_topk.to_vec();
            }
            return None;
        }
        let head = self.heads.heads.iter().find(|h| h.owner_layer == owner)?;
        let prev = self
            .prev_topk
            .get(owner)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let feat = build_features(
            hidden,
            exec_topk,
            prev,
            self.heads.n_experts,
            self.spec.top_k,
        );
        let logits = head.forward(&feat);
        let routed = route_logits(
            self.router_kind,
            &logits,
            self.spec.top_k,
            true,
            self.n_group,
            self.topk_group,
            self.routed_scaling,
            None,
        );
        if consumer < self.pending.len() {
            self.pending[consumer] = routed.clone();
        }
        if owner < self.prev_topk.len() {
            self.prev_topk[owner] = exec_topk.to_vec();
        }
        Some(routed)
    }

    pub fn take_prediction(&mut self, layer: usize) -> Option<Vec<(usize, f32)>> {
        if layer < self.pending.len() && !self.pending[layer].is_empty() {
            Some(std::mem::take(&mut self.pending[layer]))
        } else {
            None
        }
    }

    pub fn snapshot(&self) -> PrerouterPrediction {
        PrerouterPrediction {
            by_layer: self.pending.clone(),
        }
    }

    /// Whether this layer should consume staged prerouter indices (35b patch_call).
    pub fn consume_staged(&self, layer: usize) -> bool {
        self.spec.patch_call && layer >= self.spec.start_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge0::spec::PrerouterSpec;

    #[test]
    fn features_concat_onehots() {
        let f = build_features(&[0.5, -0.25], &[1], &[0], 4, 1);
        assert_eq!(f.len(), 2 + 8);
        assert_eq!(f[2 + 1], 1.0);
        assert_eq!(f[2 + 4 + 0], 1.0);
    }

    #[test]
    fn stager_stages_next_layer() {
        let spec = PrerouterSpec::qwen35();
        let heads = PrerouterHeads::zeros(&spec, 8, 4);
        let mut stager =
            CrossTokenStager::new(spec, heads, RouterKind::SoftmaxTopk, 40, None, None, None);
        let hidden = vec![0.1f32; 4];
        let pred = stager.note_executed(6, &hidden, &[0, 1, 2, 3]);
        assert!(pred.is_some());
        assert!(stager.take_prediction(7).is_some());
    }
}
