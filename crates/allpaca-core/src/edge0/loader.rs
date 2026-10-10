//! Edge0 HF checkpoint directory loader (sharded safetensors + adapters).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use allpaca_gguf::HybridLayerKind;
use allpaca_safetensors::{
    load_hf_config, load_vocab_sidecar, resolve_sidecar, HfConfig, SafetensorsFile,
};
use allpaca_types::{AllpacaError, FatalError};

use super::lora::RecoverLora;
use super::mla::{MlaDims, MlaHostWeights};
use super::prerouter::{CrossTokenStager, PrerouterHeads};
use super::quant::{dequant_affine4, quant_affine4, Affine4Tensor};
use super::spec::{Edge0Tier, MoESpec, RouterKind};
use crate::model::{LayerWeights, ModelConfig, TinyModel};
use crate::weight_set::LayerFfn;

#[derive(Debug, Clone, Default)]
pub struct Edge0LoadOptions {
    pub no_lora: bool,
    pub no_prerouter: bool,
    pub lora_rank: usize,
    pub lora_alpha: f32,
}

impl Edge0LoadOptions {
    pub fn production() -> Self {
        Self {
            no_lora: false,
            no_prerouter: false,
            lora_rank: 16,
            lora_alpha: 32.0,
        }
    }
}

/// Result of loading an Edge0 directory (model + optional adapters).
#[derive(Debug)]
pub struct Edge0Loaded {
    pub tier: Edge0Tier,
    pub model: TinyModel,
    pub lora: Option<RecoverLora>,
    pub stager: Option<CrossTokenStager>,
    pub vocab_tokens: Vec<String>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

/// True when `dir` looks like an Edge0 HF checkpoint root.
pub fn detect_edge0_dir(dir: &Path) -> Option<Edge0Tier> {
    if !dir.is_dir() {
        return None;
    }
    let cfg_path = dir.join("config.json");
    if !cfg_path.is_file() {
        return None;
    }
    let hf = load_hf_config(&cfg_path).ok()?;
    let mt = hf.model_type.as_deref().unwrap_or("");
    Edge0Tier::detect(mt, &hf.architectures, dir).or_else(|| {
        // Co-located Edge0 adapters are a strong signal.
        if dir.join("lora_edge0_35b.safetensors").is_file()
            || dir.join("prerouter_edge0_35b.safetensors").is_file()
        {
            Some(Edge0Tier::Qwen35)
        } else if dir.join("lora_edge0_8b.safetensors").is_file()
            || dir.join("prerouter_edge0_8b.safetensors").is_file()
        {
            Some(Edge0Tier::Ling8)
        } else {
            None
        }
    })
}

/// Load Edge0 recipe from a checkpoint directory.
///
/// When weight shards are absent (CI / unit tests), builds a compact synthetic trunk
/// that still exercises MoE + shared expert + optional LoRA/prerouter wiring.
pub fn load_edge0_dir(dir: &Path, opts: &Edge0LoadOptions) -> Result<Edge0Loaded, AllpacaError> {
    let tier = detect_edge0_dir(dir).ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "not an Edge0 checkpoint directory: {}",
            dir.display()
        )))
    })?;
    let cfg_path = dir.join("config.json");
    let hf = load_hf_config(&cfg_path)?;
    let moe = tier.moe_spec();
    let index = load_weight_index(dir)?;
    let model = if index.is_empty() {
        synthetic_edge0_model(tier, &hf, &moe)?
    } else {
        load_edge0_weights(dir, tier, &hf, &moe, &index)?
    };

    let lora = if opts.no_lora {
        None
    } else {
        let path = tier.resolve_adapter(dir, tier.lora_file());
        if path.is_file() {
            Some(RecoverLora::load(&path, opts.lora_rank, opts.lora_alpha)?)
        } else {
            None
        }
    };

    let stager = if opts.no_prerouter {
        None
    } else {
        let pspec = tier.prerouter_spec();
        let path = tier.resolve_adapter(dir, &pspec.weights_file);
        let heads = if path.is_file() {
            PrerouterHeads::load(&path, &pspec, moe.num_experts, model.config.dim)?
        } else {
            PrerouterHeads::zeros(&pspec, moe.num_experts, model.config.dim)
        };
        Some(CrossTokenStager::new(
            pspec,
            heads,
            moe.router,
            model.config.n_layers,
            moe.n_group,
            moe.topk_group,
            moe.routed_scaling,
        ))
    };

    let tok_path = resolve_sidecar(None, &dir.join("dummy.safetensors"), "tokenizer.json");
    let (vocab_tokens, bos_id, eos_id) = if tok_path.is_file() {
        match load_vocab_sidecar(&tok_path) {
            Ok(v) => (v.tokens, v.bos_id, v.eos_id),
            Err(_) => synthetic_vocab(model.config.vocab_size),
        }
    } else {
        synthetic_vocab(model.config.vocab_size)
    };

    Ok(Edge0Loaded {
        tier,
        model,
        lora,
        stager,
        vocab_tokens,
        bos_id,
        eos_id,
    })
}

fn synthetic_vocab(n: usize) -> (Vec<String>, Option<u32>, Option<u32>) {
    let tokens = (0..n).map(|i| format!("t{i}")).collect();
    (tokens, Some(0), Some(1))
}

type WeightIndex = HashMap<String, PathBuf>;

fn load_weight_index(dir: &Path) -> Result<WeightIndex, AllpacaError> {
    let index_path = dir.join("model.safetensors.index.json");
    let mut map = WeightIndex::new();
    if index_path.is_file() {
        let src = std::fs::read_to_string(&index_path).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "Edge0 index {}: {e}",
                index_path.display()
            )))
        })?;
        let v: serde_json::Value = serde_json::from_str(&src).map_err(|e| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("Edge0 index JSON: {e}")))
        })?;
        if let Some(obj) = v.get("weight_map").and_then(|x| x.as_object()) {
            for (k, file) in obj {
                if let Some(name) = file.as_str() {
                    map.insert(k.clone(), dir.join(name));
                }
            }
        }
        return Ok(map);
    }
    // Single-file checkpoint.
    for name in ["model.safetensors", "model-00001-of-00001.safetensors"] {
        let p = dir.join(name);
        if p.is_file() {
            let file = SafetensorsFile::open(&p)?;
            for key in file.tensors.keys() {
                map.insert(key.clone(), p.clone());
            }
            break;
        }
    }
    Ok(map)
}

fn model_config_from_hf(tier: Edge0Tier, hf: &HfConfig, moe: &MoESpec) -> ModelConfig {
    let text = hf.effective_text();
    let n_heads = text.num_attention_heads.max(1);
    let n_kv = text.num_key_value_heads.unwrap_or(n_heads).max(1);
    let dim = text.hidden_size.max(1);
    let n_layers = text.num_hidden_layers.max(1);
    let head_dim = text.head_dim.unwrap_or(dim / n_heads);
    let rope_sections = text.rope_sections();
    ModelConfig {
        vocab_size: text.vocab_size.max(hf.vocab_size).max(1),
        dim,
        n_layers,
        n_heads,
        n_kv_heads: n_kv,
        ff_dim: text
            .moe_intermediate_size
            .or(text.intermediate_size)
            .unwrap_or(moe.intermediate_size)
            .max(1),
        n_experts: moe.num_experts,
        n_active: moe.top_k,
        eps: text.rms_norm_eps,
        rope_theta: text.rope_theta,
        head_dim_explicit: head_dim,
        rope_dims: ((head_dim as f32) * text.partial_rotary_factor) as usize,
        rope_sections,
        rope_neox: false,
        router_kind: moe.router,
        shared_experts: moe.shared_experts,
        n_group: moe.n_group.unwrap_or(0),
        topk_group: moe.topk_group.unwrap_or(0),
        routed_scaling: moe.routed_scaling.unwrap_or(1.0),
        moe_norm_topk: moe.norm_topk_prob,
        edge0_tier: Some(tier),
        ..ModelConfig::tiny()
    }
}

fn synthetic_edge0_model(
    tier: Edge0Tier,
    hf: &HfConfig,
    moe: &MoESpec,
) -> Result<TinyModel, AllpacaError> {
    let mut cfg = model_config_from_hf(tier, hf, moe);
    // Keep synthetic models tiny for CI when HF reports full 35B shapes without shards.
    if cfg.dim > 64 || cfg.n_layers > 4 || cfg.n_experts > 16 {
        cfg.dim = 32;
        cfg.n_layers = match tier {
            Edge0Tier::Qwen35 => 4,
            Edge0Tier::Ling8 => 3,
        };
        cfg.n_heads = 4;
        cfg.n_kv_heads = 2;
        cfg.head_dim_explicit = 8;
        cfg.ff_dim = 64;
        cfg.n_experts = 8;
        cfg.n_active = moe.top_k.min(4);
        cfg.vocab_size = cfg.vocab_size.min(128).max(32);
        if matches!(moe.router, RouterKind::SigmoidGroup) {
            cfg.n_group = 2;
            cfg.topk_group = 2;
            cfg.routed_scaling = 2.5;
        }
    }
    let mut model = TinyModel::synthetic(cfg.clone(), 0xE06E0u64, tier.name());
    // Attach shared experts + MLA on Ling layers.
    for (i, layer) in model.layers.iter_mut().enumerate() {
        if cfg.shared_experts > 0 && cfg.n_experts > 0 {
            layer.shared_expert = LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: vec![0.01; cfg.ff_dim * cfg.dim],
                w_up: vec![0.01; cfg.ff_dim * cfg.dim],
                w_down: vec![0.01; cfg.dim * cfg.ff_dim],
            };
        }
        if matches!(tier, Edge0Tier::Ling8) && i > 0 {
            let mla = MlaDims::from_hidden(cfg.dim, cfg.n_heads);
            layer.mla = Some(MlaHostWeights::synthetic(mla, 0.01));
        }
        // Synthetic Qwen35: keep Attention (not SSM) so CPU generate works without Mamba-2 weights.
        if matches!(tier, Edge0Tier::Ling8) && i == 0 {
            // L0 dense FFN — clear experts for dense path.
            layer.block = HybridLayerKind::DenseFfn;
            layer.experts.clear();
            layer.w_router.clear();
            layer.w_gate = vec![0.01; cfg.ff_dim * cfg.dim];
            layer.w_up = vec![0.01; cfg.ff_dim * cfg.dim];
            layer.w_down = vec![0.01; cfg.dim * cfg.ff_dim];
        }
    }
    if matches!(tier, Edge0Tier::Ling8) {
        // Only L1+ are MoE; keep config.n_experts for those layers' routers.
    }
    Ok(model)
}

fn load_edge0_weights(
    dir: &Path,
    tier: Edge0Tier,
    hf: &HfConfig,
    moe: &MoESpec,
    index: &WeightIndex,
) -> Result<TinyModel, AllpacaError> {
    let cfg = model_config_from_hf(tier, hf, moe);
    let mut cache: HashMap<PathBuf, Arc<SafetensorsFile>> = HashMap::new();
    let mut open = |path: &Path| -> Result<Arc<SafetensorsFile>, AllpacaError> {
        if let Some(f) = cache.get(path) {
            return Ok(Arc::clone(f));
        }
        let f = Arc::new(SafetensorsFile::open(path)?);
        cache.insert(path.to_path_buf(), Arc::clone(&f));
        Ok(f)
    };
    let mut get_f32 = |key: &str| -> Result<Option<Vec<f32>>, AllpacaError> {
        let Some(path) = index.get(key) else {
            return Ok(None);
        };
        let file = open(path)?;
        Ok(Some(file.tensor_f32(key)?))
    };

    let embed_keys = [
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "token_embd.weight",
    ];
    let mut embed = None;
    for k in embed_keys {
        if let Some(v) = get_f32(k)? {
            embed = Some(v);
            break;
        }
    }
    let embed = embed.ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "Edge0: embed_tokens not found".into(),
        ))
    })?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let layer = match tier {
            Edge0Tier::Qwen35 => load_qwen35_layer(i, &cfg, moe, &mut get_f32, hf)?,
            Edge0Tier::Ling8 => load_ling8_layer(i, &cfg, moe, &mut get_f32)?,
        };
        layers.push(layer);
    }

    let norm = get_f32("language_model.model.norm.weight")?
        .or(get_f32("model.norm.weight")?)
        .or(get_f32("output_norm.weight")?)
        .unwrap_or_else(|| vec![1.0; cfg.dim]);
    let lm_head = get_f32("language_model.lm_head.weight")?
        .or(get_f32("lm_head.weight")?)
        .or(get_f32("output.weight")?)
        .unwrap_or_else(|| embed.clone());

    let _ = dir;
    Ok(TinyModel {
        config: cfg,
        embed,
        layers,
        norm,
        lm_head,
        name: tier.name().into(),
        working_set: None,
        expert_lru: None,
        expert_host: None,
        early_exit: crate::early_exit::EarlyExitSpec::default(),
        exit_heads: Vec::new(),
        mtp_nextn: None,
        rpc: None,
        edge0: None,
        mla_cache: None,
        expert_controls: Default::default(),
        layer_capture: Default::default(),
        dsv4: None,
        q4x: None,
    })
}

fn load_qwen35_layer(
    i: usize,
    cfg: &ModelConfig,
    moe: &MoESpec,
    get_f32: &mut dyn FnMut(&str) -> Result<Option<Vec<f32>>, AllpacaError>,
    hf: &HfConfig,
) -> Result<LayerWeights, AllpacaError> {
    let layer_types = hf.effective_text().layer_types.clone();
    let is_linear = layer_types
        .get(i)
        .map(|t| t.contains("linear"))
        .unwrap_or(i % 4 != 3);
    let prefix = format!("language_model.model.layers.{i}");
    let attn_norm = get_f32(&format!("{prefix}.input_layernorm.weight"))?
        .or(get_f32(&format!("{prefix}.attn_norm.weight"))?)
        .unwrap_or_else(|| vec![1.0; cfg.dim]);
    let ffn_norm = get_f32(&format!("{prefix}.post_attention_layernorm.weight"))?
        .or(get_f32(&format!("{prefix}.ffn_norm.weight"))?)
        .unwrap_or_else(|| vec![1.0; cfg.dim]);

    let (wq, wk, wv, wo, q_norm, k_norm, block) = if is_linear {
        (
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            HybridLayerKind::Ssm,
        )
    } else {
        (
            get_f32(&format!("{prefix}.self_attn.q_proj.weight"))?.unwrap_or_default(),
            get_f32(&format!("{prefix}.self_attn.k_proj.weight"))?.unwrap_or_default(),
            get_f32(&format!("{prefix}.self_attn.v_proj.weight"))?.unwrap_or_default(),
            get_f32(&format!("{prefix}.self_attn.o_proj.weight"))?.unwrap_or_default(),
            get_f32(&format!("{prefix}.self_attn.q_norm.weight"))?.unwrap_or_default(),
            get_f32(&format!("{prefix}.self_attn.k_norm.weight"))?.unwrap_or_default(),
            HybridLayerKind::Attention,
        )
    };

    let w_router = get_f32(&format!("{prefix}.mlp.gate.weight"))?
        .unwrap_or_else(|| vec![0.0; moe.num_experts * cfg.dim]);

    let stacked_gate = get_f32(&format!(
        "language_model.model.layers.{i}.mlp.switch_mlp.gate_proj.weight"
    ))?;
    let stacked_up = get_f32(&format!(
        "language_model.model.layers.{i}.mlp.switch_mlp.up_proj.weight"
    ))?;
    let stacked_down = get_f32(&format!(
        "language_model.model.layers.{i}.mlp.switch_mlp.down_proj.weight"
    ))?;

    let mut experts = Vec::with_capacity(moe.num_experts);
    for e in 0..moe.num_experts {
        let per = get_f32(&format!(
            "{}.experts.{e}.gate_proj.weight",
            moe.key_template.replace("{layer}", &i.to_string())
        ))?;
        let w_gate = per.or_else(|| {
            stacked_gate
                .as_ref()
                .and_then(|s| slice_expert_row(s, e, moe.num_experts))
        });
        if let Some(w_gate) = w_gate {
            experts.push(LayerFfn {
                ffn_norm: Vec::new(),
                w_gate,
                w_up: stacked_up
                    .as_ref()
                    .and_then(|s| slice_expert_row(s, e, moe.num_experts))
                    .unwrap_or_default(),
                w_down: stacked_down
                    .as_ref()
                    .and_then(|s| slice_expert_row(s, e, moe.num_experts))
                    .unwrap_or_default(),
            });
        } else {
            experts.push(LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: Vec::new(),
                w_up: Vec::new(),
                w_down: Vec::new(),
            });
        }
    }

    let shared_expert =
        if let Some(w_gate) = get_f32(&format!("{prefix}.mlp.shared_expert.gate_proj.weight"))? {
            LayerFfn {
                ffn_norm: Vec::new(),
                w_gate,
                w_up: get_f32(&format!("{prefix}.mlp.shared_expert.up_proj.weight"))?
                    .unwrap_or_default(),
                w_down: get_f32(&format!("{prefix}.mlp.shared_expert.down_proj.weight"))?
                    .unwrap_or_default(),
            }
        } else {
            LayerFfn::default()
        };

    Ok(LayerWeights {
        block,
        ssm: None,
        gdn: None,
        mla: None,
        attn_norm,
        q_norm,
        k_norm,
        wq,
        wk,
        wv,
        wo,
        ffn_norm,
        w_gate: Vec::new(),
        w_up: Vec::new(),
        w_down: Vec::new(),
        w_router,
        experts,
        shared_expert,
        expert_bias: Vec::new(),
        attn_out_gate: Vec::new(),
        attn_sinks: Vec::new(),
        attn_post_norm: Vec::new(),
        shared_gate_inp: Vec::new(),
        attn_bias: Default::default(),
        ffn_post_norm: Vec::new(),
    })
}

fn load_ling8_layer(
    i: usize,
    cfg: &ModelConfig,
    moe: &MoESpec,
    get_f32: &mut dyn FnMut(&str) -> Result<Option<Vec<f32>>, AllpacaError>,
) -> Result<LayerWeights, AllpacaError> {
    let prefix = format!("model.layers.{i}");
    let attn_norm =
        get_f32(&format!("{prefix}.input_layernorm.weight"))?.unwrap_or_else(|| vec![1.0; cfg.dim]);
    let ffn_norm = get_f32(&format!("{prefix}.post_attention_layernorm.weight"))?
        .unwrap_or_else(|| vec![1.0; cfg.dim]);

    let mla_dims = MlaDims::from_hidden(cfg.dim, cfg.n_heads);
    let mla = Some(MlaHostWeights {
        dims: mla_dims,
        wdq: get_f32(&format!("{prefix}.self_attn.q_a_proj.weight"))?.unwrap_or_default(),
        wuq: get_f32(&format!("{prefix}.self_attn.q_b_proj.weight"))?.unwrap_or_default(),
        wdkv: get_f32(&format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"))?
            .unwrap_or_default(),
        wukv: get_f32(&format!("{prefix}.self_attn.kv_b_proj.weight"))?.unwrap_or_default(),
        wo: get_f32(&format!("{prefix}.self_attn.o_proj.weight"))?.unwrap_or_default(),
        q_norm: get_f32(&format!("{prefix}.self_attn.q_a_layernorm.weight"))?.unwrap_or_default(),
        kv_norm: get_f32(&format!("{prefix}.self_attn.kv_a_layernorm.weight"))?.unwrap_or_default(),
    });

    if i == 0 {
        return Ok(LayerWeights {
            block: HybridLayerKind::DenseFfn,
            ssm: None,
            gdn: None,
            mla,
            attn_norm,
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            ffn_norm,
            w_gate: get_f32(&format!("{prefix}.mlp.gate_proj.weight"))?.unwrap_or_default(),
            w_up: get_f32(&format!("{prefix}.mlp.up_proj.weight"))?.unwrap_or_default(),
            w_down: get_f32(&format!("{prefix}.mlp.down_proj.weight"))?.unwrap_or_default(),
            w_router: Vec::new(),
            experts: Vec::new(),
            shared_expert: LayerFfn::default(),
            expert_bias: Vec::new(),
            attn_out_gate: Vec::new(),
            attn_sinks: Vec::new(),
            attn_post_norm: Vec::new(),
            shared_gate_inp: Vec::new(),
            attn_bias: Default::default(),
            ffn_post_norm: Vec::new(),
        });
    }

    let w_router = get_f32(&format!("{prefix}.mlp.gate.weight"))?
        .unwrap_or_else(|| vec![0.0; moe.num_experts * cfg.dim]);
    let mut experts = Vec::with_capacity(moe.num_experts);
    for e in 0..moe.num_experts {
        experts.push(LayerFfn {
            ffn_norm: Vec::new(),
            w_gate: get_f32(&format!("{prefix}.mlp.experts.{e}.gate_proj.weight"))?
                .unwrap_or_default(),
            w_up: get_f32(&format!("{prefix}.mlp.experts.{e}.up_proj.weight"))?.unwrap_or_default(),
            w_down: get_f32(&format!("{prefix}.mlp.experts.{e}.down_proj.weight"))?
                .unwrap_or_default(),
        });
    }
    let shared_expert =
        if let Some(w_gate) = get_f32(&format!("{prefix}.mlp.shared_experts.gate_proj.weight"))? {
            LayerFfn {
                ffn_norm: Vec::new(),
                w_gate,
                w_up: get_f32(&format!("{prefix}.mlp.shared_experts.up_proj.weight"))?
                    .unwrap_or_default(),
                w_down: get_f32(&format!("{prefix}.mlp.shared_experts.down_proj.weight"))?
                    .unwrap_or_default(),
            }
        } else {
            LayerFfn::default()
        };

    Ok(LayerWeights {
        block: HybridLayerKind::Moe,
        ssm: None,
        gdn: None,
        mla,
        attn_norm,
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        wq: Vec::new(),
        wk: Vec::new(),
        wv: Vec::new(),
        wo: Vec::new(),
        ffn_norm,
        w_gate: Vec::new(),
        w_up: Vec::new(),
        w_down: Vec::new(),
        w_router,
        experts,
        shared_expert,
        expert_bias: get_f32(&format!("{prefix}.mlp.gate.e_score_correction_bias"))?
            .unwrap_or_default(),
        attn_out_gate: Vec::new(),
        attn_sinks: Vec::new(),
        attn_post_norm: Vec::new(),
        shared_gate_inp: Vec::new(),
        attn_bias: Default::default(),
        ffn_post_norm: Vec::new(),
    })
}

fn slice_expert_row(stacked: &[f32], expert: usize, n_experts: usize) -> Option<Vec<f32>> {
    if n_experts == 0 || stacked.len() % n_experts != 0 {
        return None;
    }
    let chunk = stacked.len() / n_experts;
    let start = expert * chunk;
    Some(stacked[start..start + chunk].to_vec())
}

/// Materialize one affine-4 expert from packed host bytes (for SSD fill path).
pub fn materialize_affine4_expert(
    gate: &Affine4Tensor,
    up: &Affine4Tensor,
    down: &Affine4Tensor,
) -> LayerFfn {
    LayerFfn {
        ffn_norm: Vec::new(),
        w_gate: dequant_affine4(gate),
        w_up: dequant_affine4(up),
        w_down: dequant_affine4(down),
    }
}

/// Pack a dense expert into affine-4 (tests / spill).
pub fn pack_expert_affine4(
    ffn: &LayerFfn,
    group_size: usize,
) -> (Affine4Tensor, Affine4Tensor, Affine4Tensor) {
    (
        quant_affine4(&ffn.w_gate, group_size),
        quant_affine4(&ffn.w_up, group_size),
        quant_affine4(&ffn.w_down, group_size),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detect_from_model_type_qwen() {
        let dir = std::env::temp_dir().join("allpaca_edge0_detect_qwen");
        let _ = fs::create_dir_all(&dir);
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_5_moe","architectures":["Qwen3_5MoeForConditionalGeneration"],"vocab_size":64,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4}"#,
        )
        .unwrap();
        assert_eq!(detect_edge0_dir(&dir), Some(Edge0Tier::Qwen35));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_synthetic_when_no_shards() {
        let dir = std::env::temp_dir().join("allpaca_edge0_load_syn");
        let _ = fs::create_dir_all(&dir);
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_5_moe","vocab_size":64,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,"text_config":{"vocab_size":64,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_experts":8,"num_experts_per_tok":4,"moe_intermediate_size":64}}"#,
        )
        .unwrap();
        let loaded = load_edge0_dir(&dir, &Edge0LoadOptions::production()).unwrap();
        assert_eq!(loaded.tier, Edge0Tier::Qwen35);
        assert_eq!(loaded.model.name, "edge0-35b");
        assert!(loaded.model.config.n_experts > 0);
        assert!(!loaded.model.layers[0].shared_expert.w_up.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_qwen35_forward_token_smoke() {
        let dir = std::env::temp_dir().join("allpaca_edge0_fwd_35b");
        let _ = fs::create_dir_all(&dir);
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_5_moe","vocab_size":64,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,"text_config":{"vocab_size":64,"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":64}}"#,
        )
        .unwrap();
        let loaded = load_edge0_dir(
            &dir,
            &Edge0LoadOptions {
                no_lora: true,
                no_prerouter: false,
                ..Edge0LoadOptions::production()
            },
        )
        .unwrap();
        let mut model = loaded.model;
        model.edge0 = Some(crate::model::Edge0Runtime {
            lora: loaded.lora,
            stager: loaded.stager.map(std::sync::Mutex::new),
        });
        for layer in &model.layers {
            assert_ne!(layer.block, HybridLayerKind::Ssm);
        }
        let n_layers = model.config.n_layers;
        let page_tokens = 8u32;
        let mut kv = crate::kv_tier::KvTierManager::new(0, 256 * 1024, 0.5);
        let metrics = crate::telemetry::Metrics::default();
        let mut page_ids = Vec::new();
        for layer in 0..n_layers {
            let id = kv
                .alloc_page(
                    layer as u32,
                    0,
                    page_tokens,
                    model.config.n_kv_heads,
                    model.config.head_dim(),
                    &metrics,
                )
                .unwrap();
            page_ids.push(vec![id]);
        }
        let mut logits = vec![0.0f32; model.config.vocab_size];
        model
            .forward_token(
                1,
                0,
                &page_ids,
                &mut kv,
                &metrics,
                &mut logits,
                None,
                None,
                false,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn synthetic_ling8_dense_l0_and_moe() {
        let dir = std::env::temp_dir().join("allpaca_edge0_fwd_8b");
        let _ = fs::create_dir_all(&dir);
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"ling","architectures":["BailingMoeV2ForCausalLM"],"vocab_size":64,"hidden_size":32,"num_hidden_layers":3,"num_attention_heads":4,"num_key_value_heads":2,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":64,"n_group":2,"topk_group":2}"#,
        )
        .unwrap();
        let loaded = load_edge0_dir(&dir, &Edge0LoadOptions::production()).unwrap();
        assert_eq!(loaded.tier, Edge0Tier::Ling8);
        assert!(loaded.model.layers[0].experts.is_empty());
        assert_eq!(loaded.model.layers[0].block, HybridLayerKind::DenseFfn);
        assert!(!loaded.model.layers[1].experts.is_empty());
        let mut model = loaded.model;
        model.edge0 = Some(crate::model::Edge0Runtime {
            lora: None,
            stager: loaded.stager.map(std::sync::Mutex::new),
        });
        let n_layers = model.config.n_layers;
        let page_tokens = 8u32;
        let mut kv = crate::kv_tier::KvTierManager::new(0, 256 * 1024, 0.5);
        let metrics = crate::telemetry::Metrics::default();
        let mut page_ids = Vec::new();
        for layer in 0..n_layers {
            let id = kv
                .alloc_page(
                    layer as u32,
                    0,
                    page_tokens,
                    model.config.n_kv_heads,
                    model.config.head_dim(),
                    &metrics,
                )
                .unwrap();
            page_ids.push(vec![id]);
        }
        let mut logits = vec![0.0f32; model.config.vocab_size];
        model
            .forward_token(
                1,
                0,
                &page_ids,
                &mut kv,
                &metrics,
                &mut logits,
                None,
                None,
                false,
                None,
                None,
                None,
                None,
            )
            .unwrap();
        let _ = fs::remove_dir_all(&dir);
    }
}
