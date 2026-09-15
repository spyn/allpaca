//! Edge0 MoE / prerouter / tier contracts (mirrors edge0 `moe/spec.py` + model defaults).

use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

/// Softmax top-k (Qwen3.5-MoE) vs sigmoid + group-limited top-k (Ling / DeepSeek-V3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RouterKind {
    #[default]
    SoftmaxTopk,
    SigmoidGroup,
}

/// Per-tensor default quantization for streamed experts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    pub bits: u32,
    pub group_size: u32,
    /// `"affine"` is the only supported mode in v1.
    pub mode: &'static str,
}

impl Default for QuantSpec {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 64,
            mode: "affine",
        }
    }
}

/// How expert projections are stacked in the checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightLayout {
    #[default]
    Separate,
    FusedGateUp,
}

impl WeightLayout {
    /// Bundle order for unfused experts: up, gate, down (Edge0 math order).
    pub fn bundle_projs(self) -> &'static [&'static str] {
        match self {
            Self::Separate => &["up_proj", "gate_proj", "down_proj"],
            Self::FusedGateUp => &["gate_up_proj", "down_proj"],
        }
    }
}

/// Everything MoE blocks + streaming need for one Edge0 tier.
#[derive(Debug, Clone, PartialEq)]
pub struct MoESpec {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub router: RouterKind,
    pub norm_topk_prob: bool,
    pub routed_scaling: Option<f32>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub shared_experts: usize,
    pub quant: QuantSpec,
    pub layout: WeightLayout,
    /// Safetensors key prefix with `{layer}`; consumer appends `.{proj}.{part}`.
    pub key_template: String,
}

impl MoESpec {
    pub fn keys(&self, layer: usize, proj: &str, part: &str) -> String {
        let prefix = self.key_template.replace("{layer}", &layer.to_string());
        format!("{prefix}.{proj}.{part}")
    }

    pub fn qwen35() -> Self {
        Self {
            num_experts: 256,
            top_k: 4,
            intermediate_size: 512,
            router: RouterKind::SoftmaxTopk,
            norm_topk_prob: true,
            routed_scaling: None,
            n_group: None,
            topk_group: None,
            shared_experts: 1,
            quant: QuantSpec::default(),
            layout: WeightLayout::Separate,
            key_template: "language_model.model.layers.{layer}.mlp.switch_mlp".into(),
        }
    }

    pub fn ling8() -> Self {
        Self {
            num_experts: 128,
            top_k: 8,
            intermediate_size: 512,
            router: RouterKind::SigmoidGroup,
            norm_topk_prob: true,
            routed_scaling: Some(2.5),
            n_group: Some(8),
            topk_group: Some(4),
            shared_experts: 1,
            quant: QuantSpec::default(),
            layout: WeightLayout::Separate,
            key_template: "model.layers.{layer}.mlp.experts".into(),
        }
    }
}

/// Prerouter head bank description.
#[derive(Debug, Clone, PartialEq)]
pub struct PrerouterSpec {
    pub kind: RouterKind,
    pub start_layer: usize,
    pub hidden: usize,
    pub owners: RangeInclusive<usize>,
    pub feature_topk: &'static str,
    pub weights_file: String,
    /// When true, MoE consumes prerouter indices (edge0-35b staged_replace).
    pub patch_call: bool,
    pub top_k: usize,
}

impl PrerouterSpec {
    pub fn qwen35() -> Self {
        Self {
            kind: RouterKind::SoftmaxTopk,
            start_layer: 7,
            hidden: 512,
            owners: 6..=38,
            feature_topk: "executed",
            weights_file: "prerouter_edge0_35b.safetensors".into(),
            patch_call: true,
            top_k: 4,
        }
    }

    pub fn ling8() -> Self {
        Self {
            kind: RouterKind::SigmoidGroup,
            start_layer: 7,
            hidden: 512,
            owners: 7..=22,
            feature_topk: "executed",
            weights_file: "prerouter_edge0_8b.safetensors".into(),
            patch_call: false,
            top_k: 8,
        }
    }

    pub fn n_heads(&self) -> usize {
        self.owners.end() - self.owners.start() + 1
    }
}

pub const EDGE0_35B_NAME: &str = "edge0-35b";
pub const EDGE0_8B_NAME: &str = "edge0-8b";

/// Detected Edge0 product tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge0Tier {
    Qwen35,
    Ling8,
}

impl Edge0Tier {
    pub fn name(self) -> &'static str {
        match self {
            Self::Qwen35 => EDGE0_35B_NAME,
            Self::Ling8 => EDGE0_8B_NAME,
        }
    }

    pub fn moe_spec(self) -> MoESpec {
        match self {
            Self::Qwen35 => MoESpec::qwen35(),
            Self::Ling8 => MoESpec::ling8(),
        }
    }

    pub fn prerouter_spec(self) -> PrerouterSpec {
        match self {
            Self::Qwen35 => PrerouterSpec::qwen35(),
            Self::Ling8 => PrerouterSpec::ling8(),
        }
    }

    pub fn lora_file(self) -> &'static str {
        match self {
            Self::Qwen35 => "lora_edge0_35b.safetensors",
            Self::Ling8 => "lora_edge0_8b.safetensors",
        }
    }

    pub fn n_layers(self) -> usize {
        match self {
            Self::Qwen35 => 40,
            Self::Ling8 => 24,
        }
    }

    /// Resolve from `config.json` `model_type` / architectures / directory basename.
    pub fn detect(model_type: &str, architectures: &[String], dir: &Path) -> Option<Self> {
        let mt = model_type.to_ascii_lowercase();
        if mt.contains("qwen3_5_moe") || mt.contains("qwen3.5_moe") || mt == "qwen3_5_moe" {
            return Some(Self::Qwen35);
        }
        if mt.contains("bailing") || mt.contains("ling") {
            return Some(Self::Ling8);
        }
        for a in architectures {
            let al = a.to_ascii_lowercase();
            if al.contains("qwen3_5moe") || al.contains("qwen3_5_moe") {
                return Some(Self::Qwen35);
            }
            if al.contains("bailing") {
                return Some(Self::Ling8);
            }
        }
        let base = dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if base.contains("edge0-35") || base.contains("edge0_35") {
            return Some(Self::Qwen35);
        }
        if base.contains("edge0-8") || base.contains("edge0_8") {
            return Some(Self::Ling8);
        }
        None
    }

    pub fn resolve_adapter(self, dir: &Path, file: &str) -> PathBuf {
        let primary = dir.join(file);
        if primary.is_file() {
            return primary;
        }
        let artifacts = dir.join("artifacts").join(file);
        if artifacts.is_file() {
            return artifacts;
        }
        primary
    }
}
