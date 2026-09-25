//! Edge0 streaming MoE recipe: MoESpec, Recover-LoRA, prerouter, HF safetensors load.
//!
//! Ports the contracts from [Edge0-AI/edge0](https://github.com/Edge0-AI/edge0) onto
//! allpaca's ExpertLru / HostExpertPool / HIP decode path.

pub mod loader;
pub mod lora;
pub mod mla;
pub mod prerouter;
pub mod quant;
pub mod routing;
pub mod spec;

pub use loader::{detect_edge0_dir, load_edge0_dir, Edge0LoadOptions, Edge0Loaded};
pub use lora::RecoverLora;
pub use mla::{MlaDims, MlaHostWeights};
pub use prerouter::{CrossTokenStager, PrerouterHeads, PrerouterPrediction};
pub use quant::{dequant_affine4, Affine4Tensor};
pub use routing::{group_select_from_logits, route_logits, select_from_logits};
pub use spec::{
    Edge0Tier, MoESpec, PrerouterSpec, QuantSpec, RouterKind, WeightLayout, EDGE0_35B_NAME,
    EDGE0_8B_NAME,
};
