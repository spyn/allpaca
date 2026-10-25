use std::path::{Path, PathBuf};

use allpaca_types::{AllpacaError, FatalError};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HfQuantOverride {
    #[serde(default)]
    pub group_size: Option<u32>,
    #[serde(default)]
    pub bits: Option<u32>,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HfRopeParameters {
    #[serde(default)]
    pub mrope_interleaved: Option<bool>,
    #[serde(default)]
    pub mrope_section: Option<Vec<i32>>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Nested `text_config` used by Qwen3.5-MoE multimodal HF configs.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct HfTextConfig {
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub hidden_size: Option<usize>,
    #[serde(default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub num_attention_heads: Option<usize>,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub rms_norm_eps: Option<f32>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub rope_parameters: Option<HfRopeParameters>,
    #[serde(default)]
    pub model_type: Option<String>,
    // Ling / MLA
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub v_head_dim: Option<usize>,
    #[serde(default)]
    pub n_group: Option<usize>,
    #[serde(default)]
    pub topk_group: Option<usize>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
}

/// Flattened view used by loaders (top-level or nested `text_config`).
#[derive(Debug, Clone)]
pub struct EffectiveTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub shared_expert_intermediate_size: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub head_dim: Option<usize>,
    pub partial_rotary_factor: f32,
    pub layer_types: Vec<String>,
    pub rope_parameters: Option<HfRopeParameters>,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub qk_nope_head_dim: Option<usize>,
    pub qk_rope_head_dim: Option<usize>,
    pub v_head_dim: Option<usize>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    pub routed_scaling_factor: Option<f32>,
    pub norm_topk_prob: bool,
    pub n_shared_experts: Option<usize>,
}

impl EffectiveTextConfig {
    pub fn rope_sections(&self) -> [i32; 4] {
        let mut out = [0; 4];
        if let Some(rp) = &self.rope_parameters {
            if let Some(sec) = &rp.mrope_section {
                for (i, v) in sec.iter().take(4).enumerate() {
                    out[i] = *v;
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default = "default_vocab")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_layers")]
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
    pub text_config: Option<HfTextConfig>,
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<HfRopeParameters>,
    #[serde(default)]
    pub quantization: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
    // Ling MLA top-level
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    #[serde(default)]
    pub v_head_dim: Option<usize>,
    #[serde(default)]
    pub n_group: Option<usize>,
    #[serde(default)]
    pub topk_group: Option<usize>,
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    // DeepSeek-style MoE / act (dsv4)
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(default)]
    pub num_local_experts: Option<usize>,
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
}

fn default_vocab() -> usize {
    32
}
fn default_hidden() -> usize {
    32
}
fn default_layers() -> usize {
    1
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

impl HfConfig {
    pub fn is_moe(&self) -> bool {
        self.num_local_experts.unwrap_or(0) > 0
            || self.n_routed_experts.unwrap_or(0) > 0
            || self.num_experts.unwrap_or(0) > 0
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

    pub fn effective_text(&self) -> EffectiveTextConfig {
        let t = self.text_config.as_ref();
        EffectiveTextConfig {
            vocab_size: t
                .and_then(|x| x.vocab_size)
                .unwrap_or(self.vocab_size)
                .max(1),
            hidden_size: t
                .and_then(|x| x.hidden_size)
                .unwrap_or(self.hidden_size)
                .max(1),
            num_hidden_layers: t
                .and_then(|x| x.num_hidden_layers)
                .unwrap_or(self.num_hidden_layers)
                .max(1),
            num_attention_heads: t
                .and_then(|x| x.num_attention_heads)
                .unwrap_or(self.num_attention_heads)
                .max(1),
            num_key_value_heads: t
                .and_then(|x| x.num_key_value_heads)
                .or(self.num_key_value_heads),
            intermediate_size: t
                .and_then(|x| x.intermediate_size)
                .or(self.intermediate_size),
            moe_intermediate_size: t
                .and_then(|x| x.moe_intermediate_size)
                .or(self.moe_intermediate_size),
            shared_expert_intermediate_size: t
                .and_then(|x| x.shared_expert_intermediate_size)
                .or(self.shared_expert_intermediate_size),
            num_experts: t.and_then(|x| x.num_experts).or(self.num_experts),
            num_experts_per_tok: t
                .and_then(|x| x.num_experts_per_tok)
                .or(self.num_experts_per_tok),
            rms_norm_eps: t.and_then(|x| x.rms_norm_eps).unwrap_or(self.rms_norm_eps),
            rope_theta: t
                .and_then(|x| x.rope_parameters.as_ref().and_then(|r| r.rope_theta))
                .or(t.and_then(|x| x.rope_theta))
                .or(self.rope_parameters.as_ref().and_then(|r| r.rope_theta))
                .unwrap_or(self.rope_theta),
            head_dim: t.and_then(|x| x.head_dim).or(self.head_dim),
            partial_rotary_factor: t
                .and_then(|x| x.partial_rotary_factor)
                .or(self.partial_rotary_factor)
                .unwrap_or(1.0),
            layer_types: t
                .and_then(|x| x.layer_types.clone())
                .or_else(|| self.layer_types.clone())
                .unwrap_or_default(),
            rope_parameters: t
                .and_then(|x| x.rope_parameters.clone())
                .or_else(|| self.rope_parameters.clone()),
            q_lora_rank: t.and_then(|x| x.q_lora_rank).or(self.q_lora_rank),
            kv_lora_rank: t.and_then(|x| x.kv_lora_rank).or(self.kv_lora_rank),
            qk_nope_head_dim: t.and_then(|x| x.qk_nope_head_dim).or(self.qk_nope_head_dim),
            qk_rope_head_dim: t.and_then(|x| x.qk_rope_head_dim).or(self.qk_rope_head_dim),
            v_head_dim: t.and_then(|x| x.v_head_dim).or(self.v_head_dim),
            n_group: t.and_then(|x| x.n_group).or(self.n_group),
            topk_group: t.and_then(|x| x.topk_group).or(self.topk_group),
            routed_scaling_factor: t
                .and_then(|x| x.routed_scaling_factor)
                .or(self.routed_scaling_factor),
            norm_topk_prob: t
                .and_then(|x| x.norm_topk_prob)
                .or(self.norm_topk_prob)
                .unwrap_or(true),
            n_shared_experts: t.and_then(|x| x.n_shared_experts).or(self.n_shared_experts),
        }
    }
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
        SidecarTokenizer::HuggingFace { .. } => {
            Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "{} is a HuggingFace tokenizer.json (use load_tokenizer_sidecar)",
                path.display()
            ))))
        }
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
