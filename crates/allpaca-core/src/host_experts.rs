//! Routed experts kept in RAM as the file holds them, for the CPU forward:
//! an expert is decoded to F32 when a token is routed to it. The experts
//! of a 30B MoE model are tens of GB packed and four times that in F32,
//! so a CPU reading (calibration, the sensitivity probe) of such a model
//! only fits this way. The bytes are the file's own (memory-mapped).

use std::collections::HashMap;
use std::sync::Arc;

use allpaca_gguf::GgufFile;
use allpaca_types::{AllpacaError, FatalError};

use crate::model::ModelConfig;
use crate::weight_set::LayerFfn;

/// The stacked tensors' names: gate, up, down.
const PARTS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];

pub(crate) struct PackedExperts {
    file: Arc<GgufFile>,
    n_experts: usize,
    /// Per local layer, the stacked gate, up and down tensor names (`None`
    /// for a part the layer does not have, or a layer without experts).
    layers: Vec<[Option<String>; 3]>,
    /// Stacks replaced by F32 weights, by (layer, part): the probe's swap.
    swapped: HashMap<(usize, usize), Vec<f32>>,
}

impl std::fmt::Debug for PackedExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedExperts")
            .field("n_experts", &self.n_experts)
            .field("layers", &self.layers.len())
            .field("swapped", &self.swapped.len())
            .finish()
    }
}

fn bad(msg: String) -> AllpacaError {
    AllpacaError::Fatal(FatalError::ModelLoad(msg))
}

impl PackedExperts {
    /// The stacked experts of every layer of `file`, or `None` when it has none.
    pub(crate) fn from_gguf(file: Arc<GgufFile>, cfg: &ModelConfig) -> Option<Self> {
        let layers: Vec<[Option<String>; 3]> = (0..cfg.n_layers)
            .map(|local| {
                let blk = cfg.gguf_blk(local);
                PARTS.map(|p| {
                    let name = format!("blk.{blk}.{p}.weight");
                    file.tensor(&name).is_some().then_some(name)
                })
            })
            .collect();
        let any = layers.iter().any(|l| l.iter().any(Option::is_some));
        (any && cfg.n_experts > 0).then(|| Self { file, n_experts: cfg.n_experts, layers, swapped: HashMap::new() })
    }

    pub(crate) fn has_layer(&self, layer: usize) -> bool {
        self.layers.get(layer).is_some_and(|l| l[1].is_some())
    }

    /// Weights in one expert's slice of stack `part` of `layer`, if it has one.
    fn per_expert(&self, layer: usize, part: usize) -> Option<usize> {
        let name = self.layers.get(layer)?[part].as_ref()?;
        Some(self.file.tensor(name)?.n_elements() as usize / self.n_experts)
    }

    /// Expert `e` of `layer`, decoded.
    pub(crate) fn expert(&self, layer: usize, e: usize) -> Result<LayerFfn, AllpacaError> {
        if e >= self.n_experts {
            return Err(bad(format!("no expert {e} on layer {layer}")));
        }
        let mut parts: [Vec<f32>; 3] = Default::default();
        for (part, out) in parts.iter_mut().enumerate() {
            let Some(n) = self.per_expert(layer, part) else { continue };
            if let Some(w) = self.swapped.get(&(layer, part)) {
                *out = w[e * n..(e + 1) * n].to_vec();
                continue;
            }
            let info = self.file.tensor(self.layers[layer][part].as_deref().unwrap_or_default()).expect("listed");
            let bytes = self.file.tensor_bytes_of(info)?;
            let stride = bytes.len() / self.n_experts;
            if stride * self.n_experts != bytes.len() {
                return Err(bad(format!("{}: {} bytes do not split into {} experts", info.name, bytes.len(), self.n_experts)));
            }
            *out = vec![0.0; n];
            allpaca_gguf::dequant_packed(info.ggml_type, &bytes[e * stride..(e + 1) * stride], n, out)?;
        }
        let [w_gate, w_up, w_down] = parts;
        Ok(LayerFfn { ffn_norm: Vec::new(), w_gate, w_up, w_down })
    }

    /// Weights in stack `part` of `layer`, or `None` when it has none.
    pub(crate) fn stack_len(&self, layer: usize, part: usize) -> Option<usize> {
        self.per_expert(layer, part).map(|n| n * self.n_experts)
    }

    /// Stack `part` of `layer` as it is now, decoded.
    pub(crate) fn stack(&self, layer: usize, part: usize) -> Result<Vec<f32>, AllpacaError> {
        if let Some(w) = self.swapped.get(&(layer, part)) {
            return Ok(w.clone());
        }
        let name = self.layers.get(layer).and_then(|l| l[part].as_deref()).ok_or_else(|| bad(format!("no stack {part} on layer {layer}")))?;
        self.file.tensor_f32(name)
    }

    /// Stacks held replaced by F32 weights.
    pub(crate) fn swapped(&self) -> usize {
        self.swapped.len()
    }

    /// Put `weights` in place of stack `part` of `layer` (`None`: the
    /// file's own again), returning what replaced it before, if anything.
    pub(crate) fn swap(&mut self, layer: usize, part: usize, weights: Option<Vec<f32>>) -> Option<Vec<f32>> {
        match weights {
            Some(w) => self.swapped.insert((layer, part), w),
            None => self.swapped.remove(&(layer, part)),
        }
    }
}
