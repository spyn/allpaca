use std::path::{Path, PathBuf};

use allpaca_types::{AllpacaError, FatalError};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
}

fn default_heads() -> usize {
    4
}

fn default_eps() -> f32 {
    1e-5
}

fn default_theta() -> f32 {
    10_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct VocabSidecar {
    pub tokens: Vec<String>,
    #[serde(default)]
    pub bos_id: Option<u32>,
    #[serde(default)]
    pub eos_id: Option<u32>,
}

pub fn resolve_sidecar(explicit: Option<&Path>, weights: &Path, name: &str) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    weights
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(name)
}

pub fn load_hf_config(path: &Path) -> Result<HfConfig, AllpacaError> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "SafeTensors config {}: {e}",
            path.display()
        )))
    })?;
    serde_json::from_str(&src).map_err(|e| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "SafeTensors config {}: {e}",
            path.display()
        )))
    })
}

pub fn load_vocab_sidecar(path: &Path) -> Result<VocabSidecar, AllpacaError> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "SafeTensors tokenizer {}: {e}",
            path.display()
        )))
    })?;
    serde_json::from_str(&src).map_err(|e| {
        AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "SafeTensors tokenizer {}: {e}",
            path.display()
        )))
    })
}
