use std::path::{Path, PathBuf};

use allpaca_types::{AllpacaError, FatalError};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
}

impl HfConfig {
    pub fn is_moe(&self) -> bool {
        self.num_local_experts.unwrap_or(0) > 0
            || self.n_routed_experts.unwrap_or(0) > 0
            || self.num_experts_per_tok.unwrap_or(0) > 0
    }

    pub fn hidden_act_is_silu(&self) -> bool {
        match self.hidden_act.as_deref() {
            None => true,
            Some(a) => {
                let a = a.to_ascii_lowercase();
                a == "silu" || a == "swish" || a == "silu_and_mul"
            }
        }
    }
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

/// Vocab JSON used by tests, or a HuggingFace `tokenizer.json` on disk.
#[derive(Debug, Clone)]
pub enum SidecarTokenizer {
    Vocab(VocabSidecar),
    HuggingFace {
        tokenizer_json: PathBuf,
        tokenizer_config: Option<PathBuf>,
        generation_config: Option<PathBuf>,
    },
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
    match load_tokenizer_sidecar(path)? {
        SidecarTokenizer::Vocab(v) => Ok(v),
        SidecarTokenizer::HuggingFace { .. } => Err(AllpacaError::Fatal(FatalError::Tokenizer(
            format!(
                "{} is a HuggingFace tokenizer.json (use load_tokenizer_sidecar)",
                path.display()
            ),
        ))),
    }
}

pub fn load_tokenizer_sidecar(path: &Path) -> Result<SidecarTokenizer, AllpacaError> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "tokenizer {}: {e}",
            path.display()
        )))
    })?;
    let value: Value = serde_json::from_str(&src).map_err(|e| {
        AllpacaError::Fatal(FatalError::Tokenizer(format!(
            "tokenizer {}: {e}",
            path.display()
        )))
    })?;
    if value.get("tokens").and_then(Value::as_array).is_some() {
        let vocab: VocabSidecar = serde_json::from_value(value).map_err(|e| {
            AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "tokenizer {}: {e}",
                path.display()
            )))
        })?;
        return Ok(SidecarTokenizer::Vocab(vocab));
    }
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let tokenizer_config = optional_file(dir.join("tokenizer_config.json"));
    let generation_config = optional_file(dir.join("generation_config.json"));
    Ok(SidecarTokenizer::HuggingFace {
        tokenizer_json: path.to_path_buf(),
        tokenizer_config,
        generation_config,
    })
}

fn optional_file(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}
