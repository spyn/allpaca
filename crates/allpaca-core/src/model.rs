use crate::deltanet::{rmsnorm_silu_gate, GdnDims};
use crate::early_exit::EarlyExitSpec;
use crate::edge0::{
    CrossTokenStager, Edge0Tier, MlaDims, MlaHostWeights, RecoverLora, RouterKind,
};
use crate::forest::TokenForest;
use crate::ssm::{
    mamba2_step, validate_weights as validate_mamba2_weights, Mamba2Dims, SsmLayerWeights,
};
use allpaca_cpu::{
    attention, attention_softcap, attention_with_mask_softcap, gelu, matmul_nt, rmsnorm, rope,
    rope_imrope_neox, silu,
};
use allpaca_gguf::{GgufCompatReport, GgufFile, HybridLayerInfo, HybridLayerKind};
use allpaca_hip::{DeviceDecode, DeviceWeights, HipWorkspace};
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{HfConfig, SafetensorsCatalog};
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::anchors::RecurrentState;
use crate::direct_storage::{DirectStorageProbe, DirectStorageQueue};
use crate::expert_batch::MicroBatchExpertUnion;
use crate::kv_tier::KvTierManager;
use crate::micro_batch::LayerStagger;
use crate::packed::HostExpertPool;
use crate::telemetry::Metrics;
use crate::weight_set::{
    align_up, ExpertId, ExpertLru, FfnRefs, LayerFfn, MissPlan, WeightWorkingSet, DIRECT_ALIGN,
};

/// Optional decode-side hooks (layer stagger, elastic norm snapshot, NVMe→VRAM).
pub struct DeviceDecodeSide<'a> {
    pub request_id: u64,
    pub moe_norm: Option<&'a mut Vec<f32>>,
    pub layer_stagger: Option<&'a LayerStagger>,
    pub ds_probe: Option<&'a DirectStorageProbe>,
    pub ds_queue: Option<&'a DirectStorageQueue>,
    /// Target `output_norm` hidden (pre-lm-head) for MTP `h_nextn` handoff.
    pub last_hidden: Option<&'a mut Vec<f32>>,
    /// When false, skip lm_head GEMM / logits D2H (HIP decode cohort packs it).
    pub write_logits: bool,
}

impl<'a> DeviceDecodeSide<'a> {
    pub fn new(request_id: u64) -> Self {
        Self {
            request_id,
            moe_norm: None,
            layer_stagger: None,
            ds_probe: None,
            ds_queue: None,
            last_hidden: None,
            write_logits: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ff_dim: usize,
    pub n_experts: usize,
    pub n_active: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// Explicit attn head width from GGUF (`attention.key_length`). 0 → `dim/n_heads`.
    pub head_dim_explicit: usize,
    /// Partial RoPE width (`rope.dimension_count`). 0 → full head.
    pub rope_dims: usize,
    /// Qwen3.5 MRoPE/IMRoPE sections; zeros → classic RoPE.
    pub rope_sections: [i32; 4],
    /// Synthetic recurrent mix (semantic-anchor tracer). Not a KDA/FLA kernel.
    pub recurrent: bool,
    /// Trailing NextN/MTP draft layers excluded from primary `n_layers`.
    pub nextn_predict_layers: usize,
    /// GGUF `blk.{gguf_layer_base + i}` index base (MTP draft remaps trailing blocks to 0..n_layers).
    pub gguf_layer_base: usize,
    /// MoE router family (Edge0 / DeepSeek-style).
    pub router_kind: RouterKind,
    /// Always-resident shared experts (never streamed). Edge0 naming.
    pub shared_experts: usize,
    /// Sigmoid-group routing: number of expert groups.
    pub n_group: usize,
    /// Sigmoid-group routing: surviving groups.
    pub topk_group: usize,
    /// Sigmoid-group routed weight scale.
    pub routed_scaling: f32,
    /// Renormalize top-k router weights (Edge0).
    pub moe_norm_topk: bool,
    /// Set when loaded via Edge0 recipe.
    pub edge0_tier: Option<Edge0Tier>,
    /// FFN activation (Spark2_5 uses GELU; default SiLU/SwiGLU).
    pub ffn_gelu: bool,
    /// Spark2_5 / SWA: sliding window size (0 = disabled).
    pub sliding_window: usize,
    /// Per-layer SWA flag (true = sliding attention). Empty = all full.
    pub swa_pattern: Vec<bool>,
    /// RoPE dims for sliding-attention layers (0 = use `rope_dims`).
    pub rope_dims_swa: usize,
    /// RoPE theta for sliding-attention layers (0 = use `rope_theta`).
    pub rope_theta_swa: f32,
    /// Head-wise sigmoid attention output gate (Spark2_5).
    pub headwise_attn_gate: bool,
    /// Gemma attn logit softcap (0 = off).
    pub attn_logit_softcap: f32,
    /// Gemma final logit softcap (0 = off).
    pub final_logit_softcap: f32,
    /// Gemma: scale token embeddings by sqrt(dim).
    pub scale_embed: bool,
    /// DeepSeek2: first N layers use dense FFN.
    pub leading_dense_blocks: usize,
    /// DeepSeek2 MLA ranks (0 = unused).
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    /// DeepSeek-style expert groups. 1 = no group mask (Nemotron Lightning).
    pub n_expert_groups: usize,
    pub n_group_used: usize,
    /// Always-on shared experts (`ffn_*_shexp`). 0 = Mixtral-style routed-only.
    pub n_shared_experts: usize,
    pub shared_ff_dim: usize,
    /// llama.cpp `expert_weights_norm`: renormalize top-k sigmoid/softmax weights.
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    /// `true` = sigmoid router (Nemotron Lightning / DeepSeek); `false` = softmax.
    pub moe_sigmoid_gating: bool,
    /// Gate-less expert hidden act is ReLU² instead of SiLU.
    pub moe_relu_sqr: bool,
    /// DeepSeek-V4 `expert_gating_func=4` (sqrt-softplus).
    pub moe_sqrt_softplus: bool,
    /// Partial RoPE starts at this offset in the head (DeepSeek-V4 nope prefix).
    pub rope_offset: usize,
}

impl ModelConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            dim: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ff_dim: 64,
            n_experts: 0,
            n_active: 0,
            eps: 1e-5,
            rope_theta: 10_000.0,
            head_dim_explicit: 0,
            rope_dims: 0,
            rope_sections: [0; 4],
            recurrent: false,
            nextn_predict_layers: 0,
            gguf_layer_base: 0,
            router_kind: RouterKind::SoftmaxTopk,
            shared_experts: 0,
            n_group: 0,
            topk_group: 0,
            routed_scaling: 1.0,
            moe_norm_topk: true,
            edge0_tier: None,
            ffn_gelu: false,
            sliding_window: 0,
            swa_pattern: Vec::new(),
            rope_dims_swa: 0,
            rope_theta_swa: 0.0,
            headwise_attn_gate: false,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            scale_embed: false,
            leading_dense_blocks: 0,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            n_expert_groups: 1,
            n_group_used: 1,
            n_shared_experts: 0,
            shared_ff_dim: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            moe_sigmoid_gating: false,
            moe_relu_sqr: false,
            moe_sqrt_softplus: false,
            rope_offset: 0,
        }
    }

    pub fn moe_workspace(&self) -> usize {
        self.ff_dim.max(self.shared_ff_dim)
    }

    pub fn tiny_moe() -> Self {
        Self {
            n_experts: 4,
            n_active: 2,
            ..Self::tiny()
        }
    }

    pub fn tiny_hybrid() -> Self {
        Self {
            recurrent: true,
            ..Self::tiny()
        }
    }

    pub fn draft_tiny() -> Self {
        Self {
            n_layers: 1,
            ..Self::tiny()
        }
    }

    pub fn head_dim(&self) -> usize {
        if self.head_dim_explicit > 0 {
            self.head_dim_explicit
        } else {
            self.dim / self.n_heads.max(1)
        }
    }

    pub fn uses_imrope(&self) -> bool {
        self.rope_sections[0] > 0 && self.rope_sections[1] > 0
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let arch = file.architecture().unwrap_or("llama").to_string();
        let vocab = file
            .meta_u32(&format!("{arch}.vocab_size"))
            .or_else(|| {
                file.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(allpaca_gguf::GgufValue::as_string_array)
                    .map(|t| t.len() as u32)
            })
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing vocab_size".into()))
            })? as usize;
        let dim = file
            .meta_u32(&format!("{arch}.embedding_length"))
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing embedding_length".into()))
            })? as usize;
        let n_layers_all = file
            .meta_u32(&format!("{arch}.block_count"))
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing block_count".into()))
            })? as usize;
        // NextN/MTP draft blocks sit after the main stack (qwen35); llama.cpp
        // excludes them from `n_layer()` for the primary decode graph.
        let nextn = file
            .meta_u32(&format!("{arch}.nextn_predict_layers"))
            .unwrap_or(0) as usize;
        let n_layers = n_layers_all.saturating_sub(nextn).max(1);
        let n_heads = file
            .meta_u32(&format!("{arch}.attention.head_count"))
            .unwrap_or(4) as usize;
        let n_kv_heads = meta_u32_or_array_max(file, &format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        // MoE checkpoints (Nemotron Lightning) often omit dense feed_forward_length
        // or store per-layer arrays; prefer expert_feed_forward_length.
        let ff_dim = file
            .meta_u32(&format!("{arch}.expert_feed_forward_length"))
            .or_else(|| meta_u32_or_array_max(file, &format!("{arch}.feed_forward_length")))
            .unwrap_or((dim * 4) as u32) as usize;
        let eps = file
            .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .or_else(|| file.meta_f32(&format!("{arch}.attention.layer_norm_rms_eps")))
            .unwrap_or(1e-5);
        let rope_theta = file
            .meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10_000.0);
        let head_dim_explicit = file
            .meta_u32(&format!("{arch}.attention.key_length"))
            .or_else(|| file.meta_u32(&format!("{arch}.attention.value_length")))
            .unwrap_or(0) as usize;
        let rope_dims = file
            .meta_u32(&format!("{arch}.rope.dimension_count"))
            .unwrap_or(0) as usize;
        let mut rope_sections = [0i32; 4];
        if let Some(arr) = file
            .metadata
            .get(&format!("{arch}.rope.dimension_sections"))
            .and_then(allpaca_gguf::GgufValue::as_i32_array)
        {
            for (i, v) in arr.into_iter().take(4).enumerate() {
                rope_sections[i] = v;
            }
        }
        let n_shared_experts = file
            .meta_u32(&format!("{arch}.expert_shared_count"))
            .unwrap_or(0) as usize;
        let expert_weights_norm = file
            .meta_bool(&format!("{arch}.expert_weights_norm"))
            .unwrap_or(false);
        // Nemotron Lightning: sigmoid + ReLU² + shexp.
        // DeepSeek-V4: shexp + sqrt-softplus + SiLU (not Lightning ReLU²).
        let is_dsv4 = arch == "deepseek4" || arch == "deepseek_v4";
        let lightning_moe = !is_dsv4 && (n_shared_experts > 0 || expert_weights_norm);
        let rope_offset = if is_dsv4 {
            head_dim_explicit.saturating_sub(rope_dims)
        } else {
            0
        };
        Ok(Self {
            vocab_size: vocab,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            n_experts: file.meta_u32(&format!("{arch}.expert_count")).unwrap_or(0) as usize,
            n_active: file
                .meta_u32(&format!("{arch}.expert_used_count"))
                .unwrap_or(0) as usize,
            eps,
            rope_theta,
            head_dim_explicit,
            rope_dims,
            rope_sections,
            recurrent: file.tensors.iter().any(|t| t.name.contains(".ssm_")),
            nextn_predict_layers: nextn,
            gguf_layer_base: 0,
            router_kind: if arch == "deepseek2"
                && file.meta_u32(&format!("{arch}.expert_gating_func")).unwrap_or(0) == 2
            {
                RouterKind::SigmoidGroup
            } else {
                RouterKind::SoftmaxTopk
            },
            shared_experts: if arch == "deepseek2" {
                file.meta_u32(&format!("{arch}.expert_shared_count"))
                    .unwrap_or(0) as usize
            } else {
                0
            },
            n_group: 0,
            topk_group: 0,
            routed_scaling: if arch == "deepseek2" {
                file.meta_f32(&format!("{arch}.expert_weights_scale"))
                    .unwrap_or(1.0)
            } else {
                1.0
            },
            moe_norm_topk: true,
            edge0_tier: None,
            ffn_gelu: arch == "spark2_5" || arch == "gemma2" || arch == "gemma3",
            sliding_window: file
                .meta_u32(&format!("{arch}.attention.sliding_window"))
                .or_else(|| file.meta_u32("spark2_5.attention.sliding_window"))
                .unwrap_or(if arch == "gemma2" || arch == "gemma3" {
                    4096
                } else {
                    0
                }) as usize,
            swa_pattern: read_swa_pattern(file, &arch, n_layers),
            rope_dims_swa: file
                .meta_u32(&format!("{arch}.rope.dimension_count_swa"))
                .unwrap_or(0) as usize,
            rope_theta_swa: file
                .meta_f32(&format!("{arch}.rope.freq_base_swa"))
                .unwrap_or(0.0),
            headwise_attn_gate: arch == "spark2_5"
                || file.tensor("blk.0.attn_gate.weight").is_some() && arch != "qwen35",
            attn_logit_softcap: {
                let v = file
                    .meta_f32(&format!("{arch}.attn_logit_softcapping"))
                    .or_else(|| file.meta_f32(&format!("{arch}.attention.logit_softcapping")))
                    .unwrap_or(0.0);
                if v == 0.0 && arch == "gemma2" {
                    50.0
                } else {
                    v
                }
            },
            final_logit_softcap: {
                let v = file
                    .meta_f32(&format!("{arch}.final_logit_softcapping"))
                    .unwrap_or(0.0);
                if v == 0.0 && (arch == "gemma2" || arch == "gemma3") {
                    30.0
                } else {
                    v
                }
            },
            scale_embed: arch == "gemma2" || arch == "gemma3",
            leading_dense_blocks: file
                .meta_u32(&format!("{arch}.leading_dense_block_count"))
                .unwrap_or(0) as usize,
            q_lora_rank: file
                .meta_u32(&format!("{arch}.attention.q_lora_rank"))
                .unwrap_or(0) as usize,
            kv_lora_rank: file
                .meta_u32(&format!("{arch}.attention.kv_lora_rank"))
                .unwrap_or(0) as usize,
            n_expert_groups: file
                .meta_u32(&format!("{arch}.expert_group_count"))
                .unwrap_or(1)
                .max(1) as usize,
            n_group_used: file
                .meta_u32(&format!("{arch}.expert_group_used_count"))
                .unwrap_or(1)
                .max(1) as usize,
            n_shared_experts,
            shared_ff_dim: file
                .meta_u32(&format!("{arch}.expert_shared_feed_forward_length"))
                .unwrap_or(0) as usize,
            expert_weights_norm,
            expert_weights_scale: file
                .meta_f32(&format!("{arch}.expert_weights_scale"))
                .unwrap_or(1.0),
            moe_sigmoid_gating: lightning_moe,
            moe_relu_sqr: lightning_moe,
            moe_sqrt_softplus: is_dsv4,
            rope_offset,
        })
    }

    /// First GGUF block index for this config (`blk.{gguf_layer_base + local}`).
    pub fn gguf_blk(&self, local_layer: usize) -> usize {
        self.gguf_layer_base.saturating_add(local_layer)
    }

    pub fn layer_is_swa(&self, layer: usize) -> bool {
        self.swa_pattern.get(layer).copied().unwrap_or(false)
    }

    pub fn rope_dims_for_layer(&self, layer: usize) -> usize {
        if self.layer_is_swa(layer) && self.rope_dims_swa > 0 {
            self.rope_dims_swa
        } else {
            self.rope_dims
        }
    }

    pub fn rope_theta_for_layer(&self, layer: usize) -> f32 {
        if self.layer_is_swa(layer) && self.rope_theta_swa > 0.0 {
            self.rope_theta_swa
        } else {
            self.rope_theta
        }
    }
}

fn read_swa_pattern(file: &GgufFile, arch: &str, n_layers: usize) -> Vec<bool> {
    let key = format!("{arch}.attention.sliding_window_pattern");
    if let Some(arr) = file.metadata.get(&key).and_then(allpaca_gguf::GgufValue::as_u32_array)
    {
        let mut out: Vec<bool> = arr.into_iter().take(n_layers).map(|v| v != 0).collect();
        out.resize(n_layers, false);
        return out;
    }
    if let Some(arr) = file.metadata.get(&key).and_then(allpaca_gguf::GgufValue::as_i32_array)
    {
        let mut out: Vec<bool> = arr.into_iter().take(n_layers).map(|v| v != 0).collect();
        out.resize(n_layers, false);
        return out;
    }
    // Spark default: every 4th layer is full attention (HF layer_types pattern).
    if arch == "spark2_5" && n_layers > 0 {
        return (0..n_layers).map(|i| i % 4 != 3).collect();
    }
    // Gemma2: period 2; Gemma3: period 6 (llama.cpp defaults).
    if arch == "gemma2" && n_layers > 0 {
        return (0..n_layers).map(|i| i % 2 != 1).collect();
    }
    if arch == "gemma3" && n_layers > 0 {
        return (0..n_layers).map(|i| i % 6 != 5).collect();
    }
    Vec::new()
}

const TENSORS_PER_LAYER: usize = 9;
/// Host-materialize F32 only for tiny GGUFs. A 7B/8B Q8 file is ~32GiB as F32.
const HOST_F32_LIMIT: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum WeightKind {
    F32,
    F16,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    TQ1_0,
    TQ2_0,
    RocmFp4,
}

#[derive(Debug, Clone, Copy)]
struct WeightRef {
    off: usize,
    n: usize,
    kind: WeightKind,
}

impl WeightRef {
    fn f32(off: usize, n: usize) -> Self {
        Self {
            off,
            n,
            kind: WeightKind::F32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LayerOff {
    attn_norm: WeightRef,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    ffn_norm: WeightRef,
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
    #[allow(dead_code)]
    w_router: WeightRef,
    /// Qwen35 Gated DeltaNet projections (zero-sized when unused).
    w_qkv: WeightRef,
    w_z: WeightRef,
    w_alpha: WeightRef,
    w_beta: WeightRef,
    w_ssm_in: WeightRef,
    w_ssm_out: WeightRef,
    /// Optional per-head RMSNorm (zero-sized when absent).
    q_norm: WeightRef,
    k_norm: WeightRef,
    /// Spark2_5 head-wise attention output gate (zero-sized when absent).
    attn_out_gate: WeightRef,
    w_shared_up: WeightRef,
    w_shared_down: WeightRef,
}

/// Device-resident NextN/MTP seed weights (`blk.{i}.nextn.*`).
#[derive(Clone, Copy)]
struct MtpNextnOff {
    eh_proj: WeightRef,
    enorm: WeightRef,
    hnorm: WeightRef,
}

struct FfnRingState {
    bound: Vec<Option<usize>>,
    cursor: usize,
}

struct FfnRing {
    base: usize,
    slot_bytes: usize,
    state: Mutex<FfnRingState>,
}

enum DeviceFfn<'a> {
    Resident {
        ffn_norm: WeightRef,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
    Bound {
        _guard: std::sync::MutexGuard<'a, FfnRingState>,
        /// DMA source for async H2D; must outlive the GEMMs that read this slot.
        _keep_host: Option<LayerFfn>,
        ffn_norm: WeightRef,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
}

impl DeviceFfn<'_> {
    fn ffn_norm(&self) -> WeightRef {
        match self {
            Self::Resident { ffn_norm, .. } | Self::Bound { ffn_norm, .. } => *ffn_norm,
        }
    }
    fn w_gate(&self) -> WeightRef {
        match self {
            Self::Resident { w_gate, .. } | Self::Bound { w_gate, .. } => *w_gate,
        }
    }
    fn w_up(&self) -> WeightRef {
        match self {
            Self::Resident { w_up, .. } | Self::Bound { w_up, .. } => *w_up,
        }
    }
    fn w_down(&self) -> WeightRef {
        match self {
            Self::Resident { w_down, .. } | Self::Bound { w_down, .. } => *w_down,
        }
    }
}

struct ExpertRingState {
    bound: Vec<Option<ExpertId>>,
    used_at: Vec<u64>,
    clock: u64,
}

struct ExpertRing {
    base: usize,
    slot_bytes: usize,
    state: Mutex<ExpertRingState>,
}

/// MoE prefill GPU double-buffer: track async layer fills.
#[derive(Default)]
struct PrefillGpuPipe {
    ready: u64,
    inflight: Option<u32>,
}

fn prefill_expert_slot(layer: u32, expert: u32, n_exp: u32, n_slots: usize) -> usize {
    let group = (layer as usize % 2).saturating_mul(n_exp as usize);
    (group + expert as usize) % n_slots.max(1)
}

enum DeviceExpert<'a> {
    Resident {
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
    Bound {
        _guard: std::sync::MutexGuard<'a, ExpertRingState>,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
}

impl DeviceExpert<'_> {
    fn w_gate(&self) -> WeightRef {
        match self {
            Self::Resident { w_gate, .. } | Self::Bound { w_gate, .. } => *w_gate,
        }
    }
    fn w_up(&self) -> WeightRef {
        match self {
            Self::Resident { w_up, .. } | Self::Bound { w_up, .. } => *w_up,
        }
    }
    fn w_down(&self) -> WeightRef {
        match self {
            Self::Resident { w_down, .. } | Self::Bound { w_down, .. } => *w_down,
        }
    }
}

fn slot_expert_refs(
    base: usize,
    slot_bytes: usize,
    slot: usize,
    host: &LayerFfn,
) -> [WeightRef; 3] {
    let mut off = base + slot * slot_bytes + host.ffn_norm.len() * 4;
    let mut next = |n: usize| {
        let r = WeightRef::f32(off, n);
        off += n * 4;
        r
    };
    [
        next(host.w_gate.len()),
        next(host.w_up.len()),
        next(host.w_down.len()),
    ]
}

fn slot_ffn_refs(base: usize, slot_bytes: usize, slot: usize, host: &LayerFfn) -> [WeightRef; 4] {
    let mut off = base + slot * slot_bytes;
    let mut next = |n: usize| {
        let r = WeightRef::f32(off, n);
        off += n * 4;
        r
    };
    [
        next(host.ffn_norm.len()),
        next(host.w_gate.len()),
        next(host.w_up.len()),
        next(host.w_down.len()),
    ]
}

/// Host-layout map onto a once-uploaded [`DeviceWeights`] blob.
pub struct ResidentModel {
    weights: DeviceWeights,
    packed_h2d: u64,
    ffn_ring: Option<FfnRing>,
    expert_offs: Vec<Vec<[WeightRef; 3]>>,
    expert_ring: Option<ExpertRing>,
    prefill_pipe: Mutex<PrefillGpuPipe>,
    /// Side stream for expert H2D so next-layer fills overlap current-layer
    /// compute. `None` when the runtime cannot create one; fills then queue on
    /// the default stream exactly as before.
    h2d: Option<allpaca_hip::H2dStream>,
    /// Staging buffers for copies still in flight on `h2d`. Held until the
    /// stream is drained so the DMA source cannot be freed underneath it.
    h2d_pending: Mutex<Vec<Vec<u8>>>,
    embed: WeightRef,
    layers: Vec<LayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
    mtp_nextn: Option<MtpNextnOff>,
}

impl ResidentModel {
    pub fn upload(model: &TinyModel, device: i32) -> Result<Self, AllpacaError> {
        let mut slices: Vec<&[f32]> = Vec::new();
        let mut off = 0usize;
        fn push<'a>(slices: &mut Vec<&'a [f32]>, off: &mut usize, s: &'a [f32]) -> WeightRef {
            let o = *off;
            slices.push(s);
            *off += s.len() * 4;
            WeightRef::f32(o, s.len())
        }
        let embed = push(&mut slices, &mut off, model.embed.as_slice());
        let mut layers = Vec::with_capacity(model.layers.len());
        let mut expert_offs = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let attn_norm = push(&mut slices, &mut off, layer.attn_norm.as_slice());
            let wq = push(&mut slices, &mut off, layer.wq.as_slice());
            let wk = push(&mut slices, &mut off, layer.wk.as_slice());
            let wv = push(&mut slices, &mut off, layer.wv.as_slice());
            let wo = push(&mut slices, &mut off, layer.wo.as_slice());
            let ffn_norm = if layer.ffn_norm.is_empty() {
                WeightRef::f32(0, 0)
            } else {
                push(&mut slices, &mut off, layer.ffn_norm.as_slice())
            };
            let (w_gate, w_up, w_down) = if layer.w_gate.is_empty() {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                )
            } else {
                (
                    push(&mut slices, &mut off, layer.w_gate.as_slice()),
                    push(&mut slices, &mut off, layer.w_up.as_slice()),
                    push(&mut slices, &mut off, layer.w_down.as_slice()),
                )
            };
            let w_router = if layer.w_router.is_empty() {
                WeightRef::f32(0, 0)
            } else {
                push(&mut slices, &mut off, layer.w_router.as_slice())
            };
            let mut packed_ex = Vec::with_capacity(layer.experts.len());
            if model.expert_lru.is_none() && model.expert_host.is_none() {
                for expert in &layer.experts {
                    packed_ex.push([
                        push(&mut slices, &mut off, expert.w_gate.as_slice()),
                        push(&mut slices, &mut off, expert.w_up.as_slice()),
                        push(&mut slices, &mut off, expert.w_down.as_slice()),
                    ]);
                }
            }
            expert_offs.push(packed_ex);
            layers.push(LayerOff {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
                w_router,
                w_qkv: WeightRef::f32(0, 0),
                w_z: WeightRef::f32(0, 0),
                w_alpha: WeightRef::f32(0, 0),
                w_beta: WeightRef::f32(0, 0),
                w_ssm_in: WeightRef::f32(0, 0),
                w_ssm_out: WeightRef::f32(0, 0),
                q_norm: if layer.q_norm.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.q_norm.as_slice())
                },
                k_norm: if layer.k_norm.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.k_norm.as_slice())
                },
                attn_out_gate: if layer.attn_out_gate.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.attn_out_gate.as_slice())
                },
                w_shared_up: if layer.shared_expert.w_up.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.shared_expert.w_up.as_slice())
                },
                w_shared_down: if layer.shared_expert.w_down.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.shared_expert.w_down.as_slice())
                },
            });
        }
        let norm = push(&mut slices, &mut off, model.norm.as_slice());
        let lm_head = push(&mut slices, &mut off, model.lm_head.as_slice());
        let mtp_nextn = if let Some(n) = &model.mtp_nextn {
            Some(MtpNextnOff {
                eh_proj: push(&mut slices, &mut off, n.eh_proj.as_slice()),
                enorm: push(&mut slices, &mut off, n.enorm.as_slice()),
                hnorm: push(&mut slices, &mut off, n.hnorm.as_slice()),
            })
        } else {
            None
        };
        let packed = off;
        let (ring_slots, slot_bytes) = model
            .working_set
            .as_ref()
            .filter(|w| w.streams())
            .map(|w| (w.plan.ring_slots.max(1), w.plan.slot_bytes))
            .unwrap_or((0, 0));
        let ffn_tail = ring_slots.saturating_mul(slot_bytes);
        let (expert_slots, expert_slot_bytes) = {
            let eb = model.expert_slot_bytes();
            if eb == 0 {
                (0, 0)
            } else if let Some(lru) = &model.expert_lru {
                (lru.n_slots(), align_up(eb, DIRECT_ALIGN))
            } else if model.expert_host.is_some() {
                (1, align_up(eb, DIRECT_ALIGN))
            } else {
                (0, 0)
            }
        };
        let expert_tail = expert_slots.saturating_mul(expert_slot_bytes);
        let tail = ffn_tail.saturating_add(expert_tail);
        let weights = DeviceWeights::upload_f32_with_tail(device, &slices, tail)?;
        let ffn_ring = if ffn_tail > 0 {
            Some(FfnRing {
                base: packed,
                slot_bytes,
                state: Mutex::new(FfnRingState {
                    bound: vec![None; ring_slots],
                    cursor: 0,
                }),
            })
        } else {
            None
        };
        let expert_ring = if expert_tail > 0 {
            Some(ExpertRing {
                base: packed + ffn_tail,
                slot_bytes: expert_slot_bytes,
                state: Mutex::new(ExpertRingState {
                    bound: vec![None; expert_slots],
                    used_at: vec![0; expert_slots],
                    clock: 0,
                }),
            })
        } else {
            None
        };
        Ok(Self {
            packed_h2d: packed as u64,
            weights,
            ffn_ring,
            expert_offs,
            expert_ring,
            prefill_pipe: Mutex::new(PrefillGpuPipe::default()),
            h2d: allpaca_hip::H2dStream::new().ok(),
            h2d_pending: Mutex::new(Vec::new()),
            embed,
            layers,
            norm,
            lm_head,
            mtp_nextn,
        })
    }

    fn expert_ring(&self) -> Result<&ExpertRing, AllpacaError> {
        self.expert_ring.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "streamed expert has no device slot ring".into(),
            ))
        })
    }

    fn pack_expert_slot(host: &LayerFfn, slot_bytes: usize) -> Vec<u8> {
        let mut buf = vec![0u8; slot_bytes];
        let mut off = 0usize;
        for part in [&host.ffn_norm, &host.w_gate, &host.w_up, &host.w_down] {
            let n = part.len() * 4;
            if off + n > slot_bytes {
                break;
            }
            let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr().cast::<u8>(), n) };
            buf[off..off + n].copy_from_slice(bytes);
            off += n;
        }
        buf
    }

    pub(crate) fn copy_expert_h2d(&self, dev_off: usize, host: &[u8]) -> Result<(), AllpacaError> {
        self.weights.copy_h2d(dev_off, host)
    }

    /// Queue expert-slot H2D as one batched DMA on the side stream so it
    /// overlaps compute on the default stream. `staging` is retained until
    /// [`drain_expert_h2d`] confirms the copies landed. Without a side stream
    /// this degrades to per-region async copies on the default stream, which
    /// order implicitly against compute exactly as before.
    fn queue_expert_h2d(
        &self,
        staging: Vec<u8>,
        regions: &[(usize, usize, usize)],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let Some(h2d) = self.h2d.as_ref() else {
            for (dev, start, len) in regions {
                self.weights
                    .copy_h2d_async(*dev, &staging[*start..*start + *len])?;
            }
            return Ok(());
        };
        {
            let chunks: Vec<(usize, &[u8])> = regions
                .iter()
                .map(|(dev, start, len)| (*dev, &staging[*start..*start + *len]))
                .collect();
            // Safety: `staging` outlives the copies because it moves into
            // `h2d_pending` below (or the stream is drained before it drops).
            unsafe { self.weights.copy_h2d_batch_on(&chunks, h2d.raw())? };
        }
        match self.h2d_pending.lock() {
            Ok(mut pending) => pending.push(staging),
            Err(_) => h2d.synchronize()?,
        }
        if let Some(metrics) = metrics {
            metrics
                .expert_h2d_batched_copies
                .fetch_add(regions.len() as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Order the default compute stream behind the queued expert copies, then
    /// wait for them so the staging buffers can be released.
    fn drain_expert_h2d(&self) -> Result<(), AllpacaError> {
        let Some(h2d) = self.h2d.as_ref() else {
            return Ok(());
        };
        let idle = self
            .h2d_pending
            .lock()
            .map(|p| p.is_empty())
            .unwrap_or(false);
        if idle {
            return Ok(());
        }
        h2d.join_compute()?;
        h2d.synchronize()?;
        if let Ok(mut pending) = self.h2d_pending.lock() {
            pending.clear();
        }
        Ok(())
    }

    /// Contiguous bulk H2D for several experts (one DMA when slots are adjacent).
    /// When `nvme_direct` is set, tries NVMe→VRAM bind per expert before host pack.
    pub(crate) fn bulk_fill_experts(
        &self,
        pairs: &[(ExpertId, LayerFfn)],
        metrics: Option<&Metrics>,
        prefill_map: Option<(u32, u32)>,
        async_copy: bool,
        mut nvme_direct: Option<
            &mut dyn FnMut(ExpertId, usize, usize) -> Result<bool, AllpacaError>,
        >,
    ) -> Result<u64, AllpacaError> {
        if pairs.is_empty() {
            return Ok(0);
        }
        if !async_copy {
            // A blocking fill may reuse a ring slot an async fill is still
            // writing; the side stream and the default stream do not order
            // against each other on their own.
            self.drain_expert_h2d()?;
        }
        let ring = self.expert_ring()?;
        let n_slots = {
            let state = ring.state.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "device expert ring mutex poisoned".into(),
                ))
            })?;
            state.bound.len()
        };
        let mut slots: Vec<usize> = Vec::with_capacity(pairs.len());
        let mut nvme_filled: Vec<bool> = Vec::with_capacity(pairs.len());
        let mut staging: Vec<u8> = Vec::with_capacity(pairs.len() * ring.slot_bytes);
        {
            let mut state = ring.state.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "device expert ring mutex poisoned".into(),
                ))
            })?;
            state.clock = state.clock.saturating_add(1);
            for (id, host) in pairs {
                let packed = host.bytes();
                if packed > ring.slot_bytes {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "expert ({}, {}) is {packed} bytes, slot is {}",
                        id.layer, id.expert, ring.slot_bytes
                    ))));
                }
                let slot = if let Some((layer, n_exp)) = prefill_map {
                    if id.layer == layer {
                        prefill_expert_slot(layer, id.expert, n_exp, n_slots)
                    } else if let Some(pos) = state.bound.iter().position(|&b| b == Some(*id)) {
                        pos
                    } else {
                        state
                            .bound
                            .iter()
                            .position(|b| b.is_none())
                            .unwrap_or_else(|| {
                                state
                                    .used_at
                                    .iter()
                                    .enumerate()
                                    .min_by_key(|(_, t)| *t)
                                    .map(|(i, _)| i)
                                    .unwrap_or(0)
                            })
                    }
                } else if let Some(pos) = state.bound.iter().position(|&b| b == Some(*id)) {
                    pos
                } else {
                    state
                        .bound
                        .iter()
                        .position(|b| b.is_none())
                        .unwrap_or_else(|| {
                            state
                                .used_at
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, t)| *t)
                                .map(|(i, _)| i)
                                .unwrap_or(0)
                        })
                };
                state.bound[slot] = Some(*id);
                state.used_at[slot] = state.clock;
                slots.push(slot);
                let dev = ring.base + slot * ring.slot_bytes;
                let direct = nvme_direct
                    .as_mut()
                    .map(|f| f(*id, dev, ring.slot_bytes))
                    .transpose()?
                    .unwrap_or(false);
                nvme_filled.push(direct);
                if !direct {
                    staging.extend_from_slice(&Self::pack_expert_slot(host, ring.slot_bytes));
                }
            }
        }
        let host_slots: Vec<usize> = slots
            .iter()
            .copied()
            .zip(nvme_filled.iter().copied())
            .filter_map(|(s, d)| (!d).then_some(s))
            .collect();
        let bytes = staging.len() as u64;
        if !host_slots.is_empty() {
            let first = *host_slots.iter().min().unwrap_or(&0);
            let contiguous =
                host_slots.windows(2).all(|w| w[1] == w[0] + 1) && host_slots.len() > 1;
            // (device offset, staging offset, length) per copy. Adjacent slots
            // collapse to one region; scattered slots become one batched DMA.
            let regions: Vec<(usize, usize, usize)> = if contiguous {
                vec![(ring.base + first * ring.slot_bytes, 0, staging.len())]
            } else {
                host_slots
                    .iter()
                    .enumerate()
                    .map(|(i, slot)| {
                        (
                            ring.base + slot * ring.slot_bytes,
                            i * ring.slot_bytes,
                            ring.slot_bytes,
                        )
                    })
                    .collect()
            };
            if async_copy {
                self.queue_expert_h2d(staging, &regions, metrics)?;
            } else {
                for (dev, start, len) in &regions {
                    self.weights
                        .copy_h2d(*dev, &staging[*start..*start + *len])?;
                }
            }
        }
        let nvme_bytes: u64 = nvme_filled
            .iter()
            .zip(pairs.iter())
            .filter(|(d, _)| **d)
            .map(|(_, (_, h))| h.bytes() as u64)
            .sum();
        let total_bytes = bytes + nvme_bytes;
        if let Some(metrics) = metrics {
            metrics
                .weight_h2d_bytes
                .fetch_add(total_bytes, Ordering::Relaxed);
            if pairs.len() > 1 {
                metrics.expert_bulk_fills.fetch_add(1, Ordering::Relaxed);
                metrics
                    .expert_bulk_bytes
                    .fetch_add(total_bytes, Ordering::Relaxed);
            }
        }
        Ok(total_bytes)
    }

    pub(crate) fn prefill_layer_gpu_ready(&self, layer: u32) -> bool {
        self.prefill_pipe
            .lock()
            .map(|p| (p.ready & (1u64 << (layer.min(63)))) != 0)
            .unwrap_or(false)
    }

    pub(crate) fn prefill_wait_layer(&self, layer: u32) -> Result<(), AllpacaError> {
        let wait = self
            .prefill_pipe
            .lock()
            .map(|p| p.inflight == Some(layer))
            .unwrap_or(false);
        // Unconditional: a kickoff for a later layer moves `inflight`, and the
        // staging behind an earlier one must still be drained before its slots
        // are read.
        self.drain_expert_h2d()?;
        if wait {
            allpaca_hip::stream_synchronize()?;
            if let Ok(mut p) = self.prefill_pipe.lock() {
                if p.inflight == Some(layer) {
                    p.inflight = None;
                    p.ready |= 1u64 << layer.min(63);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn prefill_kickoff_layer(
        &self,
        layer: u32,
        n_exp: u32,
        hosts: &[LayerFfn],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if hosts.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(ExpertId, LayerFfn)> = hosts
            .iter()
            .enumerate()
            .map(|(e, h)| {
                (
                    ExpertId {
                        layer,
                        expert: e as u32,
                    },
                    h.clone(),
                )
            })
            .collect();
        self.bulk_fill_experts(&pairs, metrics, Some((layer, n_exp)), true, None)?;
        if let Ok(mut p) = self.prefill_pipe.lock() {
            p.inflight = Some(layer);
        }
        if let Some(metrics) = metrics {
            metrics
                .expert_prefill_overlaps
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    pub(crate) fn prefill_warm_layer_sync(
        &self,
        layer: u32,
        n_exp: u32,
        hosts: &[LayerFfn],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if hosts.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(ExpertId, LayerFfn)> = hosts
            .iter()
            .enumerate()
            .map(|(e, h)| {
                (
                    ExpertId {
                        layer,
                        expert: e as u32,
                    },
                    h.clone(),
                )
            })
            .collect();
        self.drain_expert_h2d()?;
        self.bulk_fill_experts(&pairs, metrics, Some((layer, n_exp)), false, None)?;
        allpaca_hip::stream_synchronize()?;
        if let Ok(mut p) = self.prefill_pipe.lock() {
            p.ready |= 1u64 << layer.min(63);
            if p.inflight == Some(layer) {
                p.inflight = None;
            }
        }
        Ok(())
    }

    /// Upload a real GGUF without host-materializing every tensor as F32.
    /// Packed tensors stay mmap-backed until H2D (no host duplicate of the GGUF).
    /// Large `token_embd.weight` stays packed (Q8_0 staged / ROCmFP4 native).
    /// Tied models reuse embed.
    pub fn upload_gguf(
        file: &GgufFile,
        cfg: &ModelConfig,
        device: i32,
    ) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        let mut parts: Vec<UploadPart<'_>> = Vec::new();
        let mut off = 0usize;
        let embed = append_gguf_tensor(file, "token_embd.weight", &mut parts, &mut off)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut expert_offs = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let i = cfg.gguf_blk(i);
            let reported = report
                .layers
                .get(i)
                .map(|l| l.block)
                .unwrap_or(HybridLayerKind::Attention);
            // NextN/MTP is dense attn+FFN even when the trunk is hybrid SSM.
            let block = if file.tensor(&format!("blk.{i}.attn_q.weight")).is_some() {
                HybridLayerKind::Attention
            } else {
                reported
            };
            let attn_norm = append_gguf_tensor(
                file,
                &format!("blk.{i}.attn_norm.weight"),
                &mut parts,
                &mut off,
            )?;
            let (wq, wk, wv, wo, w_qkv, w_z, w_alpha, w_beta, w_ssm_in, w_ssm_out) = match block {
                HybridLayerKind::Ssm => {
                    if file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some() {
                        (
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.attn_qkv.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.attn_gate.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.ssm_alpha.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.ssm_beta.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                            WeightRef::f32(0, 0),
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.ssm_out.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                        )
                    } else if file.tensor(&format!("blk.{i}.ssm_in.weight")).is_some() {
                        (
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            WeightRef::f32(0, 0),
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.ssm_in.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                            append_gguf_tensor(
                                file,
                                &format!("blk.{i}.ssm_out.weight"),
                                &mut parts,
                                &mut off,
                            )?,
                        )
                    } else {
                        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "blk.{i}: SSM layer has neither Qwen GDN nor Mamba-2 input projection"
                        ))));
                    }
                }
                HybridLayerKind::Attention | HybridLayerKind::DenseFfn | HybridLayerKind::Moe => {
                    let (wq, wk, wv, wo) =
                        if file.tensor(&format!("blk.{i}.attn_q.weight")).is_some() {
                            (
                                append_gguf_tensor(
                                    file,
                                    &format!("blk.{i}.attn_q.weight"),
                                    &mut parts,
                                    &mut off,
                                )?,
                                append_gguf_tensor(
                                    file,
                                    &format!("blk.{i}.attn_k.weight"),
                                    &mut parts,
                                    &mut off,
                                )?,
                                append_gguf_tensor(
                                    file,
                                    &format!("blk.{i}.attn_v.weight"),
                                    &mut parts,
                                    &mut off,
                                )?,
                                append_gguf_tensor(
                                    file,
                                    &format!("blk.{i}.attn_output.weight"),
                                    &mut parts,
                                    &mut off,
                                )?,
                            )
                        } else if file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some() {
                            let start = off;
                            let fused = append_gguf_tensor(
                                file,
                                &format!("blk.{i}.attn_qkv.weight"),
                                &mut parts,
                                &mut off,
                            )?;
                            let q_out = cfg.n_heads.saturating_mul(cfg.head_dim().max(1));
                            let kv_out = cfg.n_kv_heads.saturating_mul(cfg.head_dim().max(1));
                            let [wq, wk, wv] =
                                split_fused_qkv_ref(fused, off - start, cfg.dim, q_out, kv_out)?;
                            (
                                wq,
                                wk,
                                wv,
                                append_gguf_tensor(
                                    file,
                                    &format!("blk.{i}.attn_output.weight"),
                                    &mut parts,
                                    &mut off,
                                )?,
                            )
                        } else if matches!(block, HybridLayerKind::DenseFfn | HybridLayerKind::Moe)
                        {
                            (
                                WeightRef::f32(0, 0),
                                WeightRef::f32(0, 0),
                                WeightRef::f32(0, 0),
                                WeightRef::f32(0, 0),
                            )
                        } else {
                            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "blk.{i} attention layer missing attn_q/attn_qkv"
                            ))));
                        };
                    (
                        wq,
                        wk,
                        wv,
                        wo,
                        WeightRef::f32(0, 0),
                        WeightRef::f32(0, 0),
                        WeightRef::f32(0, 0),
                        WeightRef::f32(0, 0),
                        WeightRef::f32(0, 0),
                        WeightRef::f32(0, 0),
                    )
                }
            };
            let ffn_norm = if file.tensor(&format!("blk.{i}.ffn_norm.weight")).is_some() {
                append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_norm.weight"),
                    &mut parts,
                    &mut off,
                )?
            } else if file
                .tensor(&format!("blk.{i}.post_attention_norm.weight"))
                .is_some()
            {
                append_gguf_tensor(
                    file,
                    &format!("blk.{i}.post_attention_norm.weight"),
                    &mut parts,
                    &mut off,
                )?
            } else if matches!(block, HybridLayerKind::DenseFfn | HybridLayerKind::Moe) {
                // Nemotron Lightning MoE-only: FFN RMSNorm is the block's attn_norm.
                attn_norm
            } else if matches!(block, HybridLayerKind::Ssm)
                && file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_none()
            {
                WeightRef::f32(0, 0)
            } else if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_none()
                && file
                    .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                    .is_none()
                && file
                    .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                    .is_none()
            {
                // Attention-only hybrid block: no FFN.
                WeightRef::f32(0, 0)
            } else {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "missing ffn_norm/post_attention_norm on blk.{i}"
                ))));
            };
            let (w_gate, w_up, w_down, w_router, packed_ex) = if cfg.n_experts > 0
                && (file
                    .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                    .is_some()
                    || file.tensor(&format!("blk.{i}.ffn_router.weight")).is_some()
                    || file
                        .tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
                        .is_some()
                    || file
                        .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                        .is_some())
            {
                let w_router = if file
                    .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                    .is_some()
                {
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_gate_inp.weight"),
                        &mut parts,
                        &mut off,
                    )?
                } else {
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_router.weight"),
                        &mut parts,
                        &mut off,
                    )?
                };
                let packed_ex = append_gguf_experts(file, i, cfg.n_experts, &mut parts, &mut off)?;
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    w_router,
                    packed_ex,
                )
            } else if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some() {
                (
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_gate.weight"),
                        &mut parts,
                        &mut off,
                    )?,
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_up.weight"),
                        &mut parts,
                        &mut off,
                    )?,
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_down.weight"),
                        &mut parts,
                        &mut off,
                    )?,
                    WeightRef::f32(0, 0),
                    Vec::new(),
                )
            } else {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    Vec::new(),
                )
            };
            expert_offs.push(packed_ex);
            let q_norm = if file
                .tensor(&format!("blk.{i}.attn_q_norm.weight"))
                .is_some()
            {
                append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_q_norm.weight"),
                    &mut parts,
                    &mut off,
                )?
            } else {
                WeightRef::f32(0, 0)
            };
            let k_norm = if file
                .tensor(&format!("blk.{i}.attn_k_norm.weight"))
                .is_some()
            {
                append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_k_norm.weight"),
                    &mut parts,
                    &mut off,
                )?
            } else {
                WeightRef::f32(0, 0)
            };
            let attn_out_gate = if cfg.headwise_attn_gate
                && file
                    .tensor(&format!("blk.{i}.attn_gate.weight"))
                    .is_some()
                && matches!(
                    block,
                    HybridLayerKind::Attention | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
                )
            {
                append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_gate.weight"),
                    &mut parts,
                    &mut off,
                )?
            } else {
                WeightRef::f32(0, 0)
            };
            let (w_shared_up, w_shared_down) =
                append_gguf_shared_expert(file, i, &mut parts, &mut off)?;
            layers.push(LayerOff {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
                w_router,
                w_qkv,
                w_z,
                w_alpha,
                w_beta,
                w_ssm_in,
                w_ssm_out,
                q_norm,
                k_norm,
                attn_out_gate,
                w_shared_up,
                w_shared_down,
            });
        }
        let norm = if file.tensor("output_norm.weight").is_some() {
            append_gguf_tensor(file, "output_norm.weight", &mut parts, &mut off)?
        } else {
            let b = cfg.gguf_blk(0);
            append_gguf_tensor(
                file,
                &format!("blk.{b}.nextn.shared_head_norm.weight"),
                &mut parts,
                &mut off,
            )?
        };
        let lm_head = if file.tensor("output.weight").is_some() {
            append_gguf_tensor(file, "output.weight", &mut parts, &mut off)?
        } else {
            embed
        };
        let mtp_base = cfg.gguf_blk(0);
        let mtp_nextn = if file
            .tensor(&format!("blk.{mtp_base}.nextn.eh_proj.weight"))
            .is_some()
        {
            Some(MtpNextnOff {
                eh_proj: append_gguf_tensor(
                    file,
                    &format!("blk.{mtp_base}.nextn.eh_proj.weight"),
                    &mut parts,
                    &mut off,
                )?,
                enorm: append_gguf_tensor(
                    file,
                    &format!("blk.{mtp_base}.nextn.enorm.weight"),
                    &mut parts,
                    &mut off,
                )?,
                hnorm: append_gguf_tensor(
                    file,
                    &format!("blk.{mtp_base}.nextn.hnorm.weight"),
                    &mut parts,
                    &mut off,
                )?,
            })
        } else {
            None
        };

        let views: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        let weights = DeviceWeights::upload_bytes(device, &views)?;
        Ok(Self {
            packed_h2d: weights.bytes(),
            ffn_ring: None,
            expert_offs,
            expert_ring: None,
            prefill_pipe: Mutex::new(PrefillGpuPipe::default()),
            h2d: allpaca_hip::H2dStream::new().ok(),
            h2d_pending: Mutex::new(Vec::new()),
            weights,
            embed,
            layers,
            norm,
            lm_head,
            mtp_nextn,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.weights.bytes()
    }

    pub fn packed_h2d_bytes(&self) -> u64 {
        self.packed_h2d
    }

    pub fn has_mtp_nextn(&self) -> bool {
        self.mtp_nextn.is_some()
    }

    fn bind_ffn<'a>(
        &'a self,
        layer_idx: usize,
        host: Option<LayerFfn>,
        working_set: Option<&WeightWorkingSet>,
        metrics: Option<&Metrics>,
    ) -> Result<DeviceFfn<'a>, AllpacaError> {
        let layer = &self.layers[layer_idx];
        if layer.w_gate.n > 0 {
            return Ok(DeviceFfn::Resident {
                ffn_norm: layer.ffn_norm,
                w_gate: layer.w_gate,
                w_up: layer.w_up,
                w_down: layer.w_down,
            });
        }
        let ring = self.ffn_ring.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "streamed FFN layer has no device ring".into(),
            ))
        })?;
        let host = host.ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "streamed FFN layer {layer_idx} missing host pool"
            )))
        })?;
        let packed = host.bytes();
        if packed > ring.slot_bytes {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "FFN layer {layer_idx} is {packed} bytes, ring slot is {}",
                ring.slot_bytes
            ))));
        }
        let mut state = ring.state.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "device FFN ring mutex poisoned".into(),
            ))
        })?;
        let n_slots = state.bound.len();
        if let Some(slot) = state.bound.iter().position(|&b| b == Some(layer_idx)) {
            if let Some(ws) = working_set {
                ws.note_bind(true);
            }
            let [ffn_norm, w_gate, w_up, w_down] =
                slot_ffn_refs(ring.base, ring.slot_bytes, slot, &host);
            return Ok(DeviceFfn::Bound {
                _guard: state,
                _keep_host: None,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }
        let slot = state
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                let s = state.cursor % n_slots.max(1);
                state.cursor = state.cursor.wrapping_add(1);
                s
            });
        let mut copy_off = ring.base + slot * ring.slot_bytes;
        for part in [&host.ffn_norm, &host.w_gate, &host.w_up, &host.w_down] {
            let n = part.len() * 4;
            let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr().cast::<u8>(), n) };
            self.weights.copy_h2d(copy_off, bytes)?;
            copy_off += n;
        }
        // copy_h2d is hipMemcpyAsync. The host `LayerFfn` is a temporary; wait
        // so the slot is valid before GEMM and before this stack frame drops.
        allpaca_hip::stream_synchronize()?;
        state.bound[slot] = Some(layer_idx);
        if let Some(ws) = working_set {
            ws.note_bind(false);
        }
        if let Some(metrics) = metrics {
            metrics
                .weight_h2d_bytes
                .fetch_add(packed as u64, Ordering::Relaxed);
        }
        let [ffn_norm, w_gate, w_up, w_down] =
            slot_ffn_refs(ring.base, ring.slot_bytes, slot, &host);
        Ok(DeviceFfn::Bound {
            _guard: state,
            _keep_host: Some(host),
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        })
    }

    fn bind_expert<'a>(
        &'a self,
        id: ExpertId,
        host: &LayerFfn,
        metrics: Option<&Metrics>,
    ) -> Result<DeviceExpert<'a>, AllpacaError> {
        if let Some(refs) = self
            .expert_offs
            .get(id.layer as usize)
            .and_then(|layer| layer.get(id.expert as usize))
        {
            return Ok(DeviceExpert::Resident {
                w_gate: refs[0],
                w_up: refs[1],
                w_down: refs[2],
            });
        }
        let ring = self.expert_ring.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "streamed expert has no device slot ring".into(),
            ))
        })?;
        let packed = host.bytes();
        if packed > ring.slot_bytes {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "expert ({}, {}) is {packed} bytes, slot is {}",
                id.layer, id.expert, ring.slot_bytes
            ))));
        }
        let mut state = ring.state.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "device expert ring mutex poisoned".into(),
            ))
        })?;
        state.clock = state.clock.saturating_add(1);
        if let Some(slot) = state.bound.iter().position(|&b| b == Some(id)) {
            state.used_at[slot] = state.clock;
            let [w_gate, w_up, w_down] = slot_expert_refs(ring.base, ring.slot_bytes, slot, host);
            return Ok(DeviceExpert::Bound {
                _guard: state,
                w_gate,
                w_up,
                w_down,
            });
        }
        let victim = state
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                state
                    .used_at
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
        let mut copy_off = ring.base + victim * ring.slot_bytes;
        for part in [&host.ffn_norm, &host.w_gate, &host.w_up, &host.w_down] {
            let n = part.len() * 4;
            let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr().cast::<u8>(), n) };
            self.weights.copy_h2d(copy_off, bytes)?;
            copy_off += n;
        }
        state.bound[victim] = Some(id);
        state.used_at[victim] = state.clock;
        if let Some(metrics) = metrics {
            metrics
                .weight_h2d_bytes
                .fetch_add(packed as u64, Ordering::Relaxed);
        }
        let [w_gate, w_up, w_down] = slot_expert_refs(ring.base, ring.slot_bytes, victim, host);
        Ok(DeviceExpert::Bound {
            _guard: state,
            w_gate,
            w_up,
            w_down,
        })
    }

    fn f32_or_dequant(&self, hip: &HipWorkspace, w: WeightRef) -> Result<*const f32, AllpacaError> {
        match w.kind {
            WeightKind::F32 => Ok(self.weights.ptr_at(w.off)),
            WeightKind::Q8_0 => {
                let out = hip.ensure_dequant(w.n)?;
                hip.dequant_q8_0(self.weights.byte_ptr_at(w.off), w.n as i32, out)?;
                Ok(out as *const f32)
            }
            WeightKind::F16
            | WeightKind::Q4_K
            | WeightKind::Q5_K
            | WeightKind::Q6_K
            | WeightKind::TQ1_0
            | WeightKind::TQ2_0
            | WeightKind::RocmFp4 => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "F16/Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0/ROCmFP4 norm must be F32 at load".into(),
            ))),
        }
    }

    fn gemm_nt(
        &self,
        hip: &HipWorkspace,
        a: *const f32,
        w: WeightRef,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        match w.kind {
            WeightKind::F32 => hip.gemm_nt_dev(a, self.weights.ptr_at(w.off), c, m, n, k),
            WeightKind::F16 => {
                hip.gemm_nt_f16_dev(a, self.weights.byte_ptr_at(w.off) as *const u16, c, m, n, k)
            }
            WeightKind::Q8_0 => hip.gemm_nt_q8_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k),
            WeightKind::Q4_K => {
                hip.gemm_nt_q4_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::Q5_K => {
                hip.gemm_nt_q5_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::Q6_K => {
                hip.gemm_nt_q6_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::TQ2_0 => {
                hip.gemm_nt_tq2_0_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::TQ1_0 => {
                hip.gemm_nt_tq1_0_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::RocmFp4 => {
                hip.gemm_nt_rocmfp4_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
        }
    }

    fn embed_gather(
        &self,
        hip: &HipWorkspace,
        token: u32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        match self.embed.kind {
            WeightKind::F32 => hip.embed_gather(self.embed(), token, out, dim),
            WeightKind::F16 => hip.embed_gather_f16(
                self.weights.byte_ptr_at(self.embed.off) as *const u16,
                token,
                out,
                dim,
            ),
            WeightKind::Q8_0 => {
                hip.embed_gather_q8_0(self.weights.byte_ptr_at(self.embed.off), token, out, dim)
            }
            WeightKind::RocmFp4 => {
                hip.embed_gather_rocmfp4(self.weights.byte_ptr_at(self.embed.off), token, out, dim)
            }
            WeightKind::Q4_K
            | WeightKind::Q5_K
            | WeightKind::Q6_K
            | WeightKind::TQ1_0
            | WeightKind::TQ2_0 => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0 embed must be F32 at load".into(),
            ))),
        }
    }

    fn embed_gather_rows(
        &self,
        hip: &HipWorkspace,
        tokens: &[u32],
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        match self.embed.kind {
            WeightKind::F32 => hip.embed_gather_rows(self.embed(), tokens, out, dim),
            WeightKind::F16 => hip.embed_gather_f16_rows(
                self.weights.byte_ptr_at(self.embed.off) as *const u16,
                tokens,
                out,
                dim,
            ),
            WeightKind::Q8_0 => hip.embed_gather_q8_0_rows(
                self.weights.byte_ptr_at(self.embed.off),
                tokens,
                out,
                dim,
            ),
            WeightKind::RocmFp4 => hip.embed_gather_rocmfp4_rows(
                self.weights.byte_ptr_at(self.embed.off),
                tokens,
                out,
                dim,
            ),
            WeightKind::Q4_K
            | WeightKind::Q5_K
            | WeightKind::Q6_K
            | WeightKind::TQ1_0
            | WeightKind::TQ2_0 => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0 embed must be F32 at load".into(),
            ))),
        }
    }

    pub fn embed(&self) -> *const f32 {
        self.weights.ptr_at(self.embed.off)
    }

    pub fn attn_norm(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].attn_norm.off)
    }

    pub fn wq(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wq.off)
    }

    pub fn wk(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wk.off)
    }

    pub fn wv(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wv.off)
    }

    pub fn wo(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wo.off)
    }

    pub fn ffn_norm(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].ffn_norm.off)
    }

    pub fn w_gate(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_gate.off)
    }

    pub fn w_up(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_up.off)
    }

    pub fn w_down(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_down.off)
    }

    pub fn norm(&self) -> *const f32 {
        self.weights.ptr_at(self.norm.off)
    }

    pub fn lm_head(&self) -> *const f32 {
        self.weights.ptr_at(self.lm_head.off)
    }
}

fn relu_sqr(v: f32) -> f32 {
    if v > 0.0 {
        v * v
    } else {
        0.0
    }
}

fn sigmoid_stable(v: f32) -> f32 {
    if v >= 0.0 {
        let z = (-v).exp();
        1.0 / (1.0 + z)
    } else {
        let z = v.exp();
        z / (1.0 + z)
    }
}

pub(crate) fn expert_mlp(
    x: &[f32],
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    out: &mut [f32],
    relu_sqr_act: bool,
) -> Result<(), AllpacaError> {
    expert_mlp_rows(x, ffn, ff_dim, dim, 1, out, relu_sqr_act)
}

pub(crate) fn expert_mlp_rows(
    x: &[f32],
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    m: usize,
    out: &mut [f32],
    relu_sqr_act: bool,
) -> Result<(), AllpacaError> {
    if m == 0 {
        return Ok(());
    }
    let mut up = vec![0.0f32; m.saturating_mul(ff_dim)];
    matmul_nt(x, &ffn.w_up, &mut up, m, ff_dim, dim)?;
    // Gate-less MoE: Lightning uses ReLU²; tiny fixtures keep SiLU.
    if ffn.w_gate.is_empty() {
        if relu_sqr_act {
            for v in &mut up {
                *v = relu_sqr(*v);
            }
        } else {
            for v in &mut up {
                *v = silu(*v);
            }
        }
        return matmul_nt(&up, &ffn.w_down, out, m, dim, ff_dim);
    }
    let mut gate = vec![0.0f32; m.saturating_mul(ff_dim)];
    matmul_nt(x, &ffn.w_gate, &mut gate, m, ff_dim, dim)?;
    for i in 0..m.saturating_mul(ff_dim) {
        gate[i] = silu(gate[i]) * up[i];
    }
    matmul_nt(&gate, &ffn.w_down, out, m, dim, ff_dim)
}

pub(crate) fn measure_expert_host_bandwidth(ffn: &LayerFfn, ff_dim: usize, dim: usize) -> f64 {
    use std::time::Instant;
    if ffn.bytes() == 0 || dim == 0 || ff_dim == 0 {
        return 0.0;
    }
    let x = vec![0.01f32; dim];
    let mut out = vec![0.0f32; dim];
    let _ = expert_mlp(&x, ffn, ff_dim, dim, &mut out, false);
    let iters = 8usize;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = expert_mlp(&x, ffn, ff_dim, dim, &mut out, false);
    }
    let secs = t.elapsed().as_secs_f64().max(1e-9);
    (ffn.bytes() as f64 * iters as f64) / secs
}

/// Host GEMM bandwidth on the expert home node vs a cross-node read (for `q*`).
pub(crate) fn measure_expert_host_bandwidth_numa(
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    numa: &crate::numa::NumaTopology,
    id: ExpertId,
) -> f64 {
    let home = numa.preferred_node(id);
    let local = numa.run_on_node(home, || measure_expert_host_bandwidth(ffn, ff_dim, dim));
    if numa.n_nodes <= 1 {
        return local;
    }
    let remote = (home + 1) % numa.n_nodes;
    let cross = numa.run_on_node(remote, || measure_expert_host_bandwidth(ffn, ff_dim, dim));
    if cross <= 0.0 {
        local
    } else {
        cross.min(local)
    }
}

/// Profile fixed costs for overhead-aware `q*` (DMA setup + CPU dispatch).
pub(crate) fn measure_qstar_fixed_costs(expert_bytes: usize) -> (f64, f64) {
    use std::time::Instant;
    let n = expert_bytes.max(64);
    let src = vec![0u8; n];
    let mut dst = vec![0u8; n];
    // Warm.
    dst.copy_from_slice(&src);
    let iters = 32usize;
    let t = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&src);
    }
    let t_dma = t.elapsed().as_secs_f64() / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        std::thread::scope(|s| {
            s.spawn(|| ());
        });
    }
    let t_dispatch = t.elapsed().as_secs_f64() / iters as f64;
    (t_dma, t_dispatch)
}

/// llama.cpp `build_moe_ffn` clamp when `expert_weights_norm` is set (1/16384).
const EXPERT_WEIGHT_NORM_EPS: f32 = 6.103515625e-5;

fn sqrt_softplus_moe(x: f32) -> f32 {
    let sp = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    sp.max(0.0).sqrt()
}

fn moe_route(
    x: &[f32],
    layer: &LayerWeights,
    cfg: &ModelConfig,
) -> Vec<(usize, f32)> {
    let n_exp = cfg.n_experts.max(1);
    let n_act = cfg.n_active.max(1).min(n_exp);
    moe_route_logits(
        x,
        &layer.w_router,
        n_exp,
        n_act,
        cfg.router_kind,
        cfg.moe_norm_topk,
        if cfg.n_group > 0 { Some(cfg.n_group) } else { None },
        if cfg.topk_group > 0 {
            Some(cfg.topk_group)
        } else {
            None
        },
        Some(cfg.routed_scaling),
        if layer.expert_bias.is_empty() {
            None
        } else {
            Some(layer.expert_bias.as_slice())
        },
    )
}

fn moe_route_logits(
    x: &[f32],
    w_router: &[f32],
    n_experts: usize,
    n_active: usize,
    kind: RouterKind,
    norm: bool,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    routed_scaling: Option<f32>,
    expert_bias: Option<&[f32]>,
) -> Vec<(usize, f32)> {
    let dim = x.len();
    let mut logits = vec![0.0f32; n_experts];
    if w_router.len() >= n_experts.saturating_mul(dim) {
        for e in 0..n_experts {
            let row = &w_router[e * dim..(e + 1) * dim];
            let mut s = 0.0f32;
            for i in 0..dim {
                s += x[i] * row[i];
            }
            logits[e] = s;
        }
    }
    crate::edge0::route_logits(
        kind,
        &logits,
        n_active,
        norm,
        n_group,
        topk_group,
        routed_scaling,
        expert_bias,
    )
}

pub(crate) fn topk_route(
    x: &[f32],
    w_router: &[f32],
    n_experts: usize,
    n_active: usize,
    cfg: &ModelConfig,
) -> Vec<(usize, f32)> {
    let dim = x.len();
    let mut weights = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        let row = &w_router[e * dim..(e + 1) * dim];
        let mut s = 0.0f32;
        for i in 0..dim {
            s += x[i] * row[i];
        }
        weights[e] = s;
    }
    topk_from_logits(&mut weights, n_active, cfg)
}

pub(crate) fn topk_from_logits(
    weights: &mut [f32],
    n_active: usize,
    cfg: &ModelConfig,
) -> Vec<(usize, f32)> {
    let n_experts = weights.len();
    if cfg.moe_sqrt_softplus {
        for w in weights.iter_mut() {
            *w = sqrt_softplus_moe(*w);
        }
    } else if cfg.moe_sigmoid_gating {
        for w in weights.iter_mut() {
            *w = sigmoid_stable(*w);
        }
    } else {
        let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for w in weights.iter_mut() {
            *w = (*w - max).exp();
            sum += *w;
        }
        let sum = sum.max(1e-9);
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(n_active.max(1).min(n_experts));
    if !cfg.moe_sigmoid_gating || cfg.expert_weights_norm {
        let part: f32 = idx.iter().map(|&i| weights[i]).sum::<f32>();
        let denom = if cfg.expert_weights_norm {
            part.max(EXPERT_WEIGHT_NORM_EPS)
        } else {
            part.max(1e-9)
        };
        for &i in &idx {
            weights[i] /= denom;
        }
    }
    if cfg.expert_weights_scale != 0.0 && cfg.expert_weights_scale != 1.0 {
        for &i in &idx {
            weights[i] *= cfg.expert_weights_scale;
        }
    }
    idx.into_iter().map(|i| (i, weights[i])).collect()
}

fn empty_layer() -> LayerWeights {
    LayerWeights {
        block: HybridLayerKind::Attention,
        ssm: None,
        gdn: None,
       mla: None,
        attn_norm: Vec::new(),
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        wq: Vec::new(),
        wk: Vec::new(),
        wv: Vec::new(),
        wo: Vec::new(),
        ffn_norm: Vec::new(),
        w_gate: Vec::new(),
        w_up: Vec::new(),
        w_down: Vec::new(),
        w_router: Vec::new(),
        experts: Vec::new(),
        shared_expert: LayerFfn::default(),
        expert_bias: Vec::new(),
        attn_out_gate: Vec::new(),
        attn_post_norm: Vec::new(),
        ffn_post_norm: Vec::new(),
    }
}

fn split_fused_qkv_host(
    fused: Vec<f32>,
    input_dim: usize,
    q_out: usize,
    kv_out: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), AllpacaError> {
    let q_len = q_out.saturating_mul(input_dim);
    let kv_len = kv_out.saturating_mul(input_dim);
    let expected = q_len.saturating_add(kv_len.saturating_mul(2));
    if fused.len() != expected {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "fused attn_qkv shape mismatch: elements={} expected={expected}",
            fused.len()
        ))));
    }
    let q = fused[..q_len].to_vec();
    let k = fused[q_len..q_len + kv_len].to_vec();
    let v = fused[q_len + kv_len..].to_vec();
    Ok((q, k, v))
}

fn load_gguf_gdn_host(file: &GgufFile, layer: usize) -> Result<GdnHostWeights, AllpacaError> {
    let arch = file.architecture().unwrap_or("qwen35");
    let dims = GdnDims::from_meta(
        file.meta_u32(&format!("{arch}.ssm.state_size"))
            .unwrap_or(128) as usize,
        file.meta_u32(&format!("{arch}.ssm.inner_size"))
            .unwrap_or(0) as usize,
        file.meta_u32(&format!("{arch}.ssm.time_step_rank"))
            .unwrap_or(0) as usize,
        file.meta_u32(&format!("{arch}.ssm.group_count"))
            .unwrap_or(0) as usize,
        file.meta_u32(&format!("{arch}.ssm.conv_kernel"))
            .unwrap_or(4) as usize,
    )
    .ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "blk.{layer}: cannot infer GDN dims from GGUF meta"
        )))
    })?;
    if file
        .tensor(&format!("blk.{layer}.attn_qkv.weight"))
        .is_none()
        || file
            .tensor(&format!("blk.{layer}.ssm_out.weight"))
            .is_none()
    {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "blk.{layer}: not a Gated DeltaNet layer"
        ))));
    }
    let a = file.tensor_f32(&format!("blk.{layer}.ssm_a"))?;
    let mut dt_bias = if file.tensor(&format!("blk.{layer}.ssm_dt.bias")).is_some() {
        file.tensor_f32(&format!("blk.{layer}.ssm_dt.bias"))?
    } else {
        vec![0.0f32; dims.n_v_heads]
    };
    if dt_bias.len() < dims.n_v_heads {
        dt_bias.resize(dims.n_v_heads, 0.0);
    } else {
        dt_bias.truncate(dims.n_v_heads);
    }
    let norm = file.tensor_f32(&format!("blk.{layer}.ssm_norm.weight"))?;
    let conv1d = file.tensor_f32(&format!("blk.{layer}.ssm_conv1d.weight"))?;
    if a.len() != dims.n_v_heads
        || norm.len() != dims.head_v
        || conv1d.len() != dims.conv_kernel.saturating_mul(dims.qkv_dim)
    {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "blk.{layer}: GDN host tensor sizes mismatch dims={dims:?} a={} norm={} conv={}",
            a.len(),
            norm.len(),
            conv1d.len()
        ))));
    }
    // Host projections for CUDA / CPU host-step GDN (HIP also uploads these on-device).
    let w_qkv = file.tensor_f32(&format!("blk.{layer}.attn_qkv.weight"))?;
    let w_z = file.tensor_f32(&format!("blk.{layer}.attn_gate.weight"))?;
    let w_alpha = file.tensor_f32(&format!("blk.{layer}.ssm_alpha.weight"))?;
    let w_beta = file.tensor_f32(&format!("blk.{layer}.ssm_beta.weight"))?;
    let w_out = file.tensor_f32(&format!("blk.{layer}.ssm_out.weight"))?;
    let dim_in = w_qkv.len() / dims.qkv_dim.max(1);
    if dim_in == 0
        || w_qkv.len() != dims.qkv_dim.saturating_mul(dim_in)
        || w_z.len() != dims.value_dim.saturating_mul(dim_in)
        || w_alpha.len() != dims.n_v_heads.saturating_mul(dim_in)
        || w_beta.len() != dims.n_v_heads.saturating_mul(dim_in)
        || w_out.len() != dim_in.saturating_mul(dims.value_dim)
    {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "blk.{layer}: GDN projection sizes mismatch dims={dims:?} qkv={} z={} alpha={} beta={} out={}",
            w_qkv.len(),
            w_z.len(),
            w_alpha.len(),
            w_beta.len(),
            w_out.len()
        ))));
    }
    Ok(GdnHostWeights {
        dims,
        a,
        dt_bias,
        norm,
        conv1d,
        w_qkv,
        w_z,
        w_alpha,
        w_beta,
        w_out,
    })
}

fn load_gguf_ssm_layer(
    file: &GgufFile,
    layer: usize,
    dim: usize,
    materialize_projections: bool,
) -> Result<SsmLayerWeights, AllpacaError> {
    let arch = file.architecture().unwrap_or("nemotron_h_moe");
    let meta = |suffix: &str| {
        file.meta_u32(&format!("{arch}.ssm.{suffix}"))
            .map(|v| v as usize)
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "blk.{layer}: missing authoritative Mamba-2 metadata {arch}.ssm.{suffix}"
                )))
            })
    };
    let dims = Mamba2Dims::new(
        dim,
        meta("inner_size")?,
        meta("state_size")?,
        meta("time_step_rank")?,
        meta("group_count")?,
        meta("conv_kernel")?,
    )
    .ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "blk.{layer}: unsupported Mamba-2 metadata for {arch}"
        )))
    })?;
    let required = |suffix: &str| -> Result<Vec<f32>, AllpacaError> {
        file.tensor_f32(&format!("blk.{layer}.{suffix}"))
    };
    let weights = SsmLayerWeights {
        dims,
        w_in: if materialize_projections {
            required("ssm_in.weight")?
        } else {
            Vec::new()
        },
        w_out: if materialize_projections {
            required("ssm_out.weight")?
        } else {
            Vec::new()
        },
        a: required("ssm_a")?,
        d_skip: required("ssm_d")?,
        dt_bias: required("ssm_dt.bias")?,
        conv1d: required("ssm_conv1d.weight")?,
        conv_bias: if file
            .tensor(&format!("blk.{layer}.ssm_conv1d.bias"))
            .is_some()
        {
            required("ssm_conv1d.bias")?
        } else {
            Vec::new()
        },
        norm: required("ssm_norm.weight")?,
    };
    if materialize_projections {
        validate_mamba2_weights(&weights)
            .map_err(|e| AllpacaError::Fatal(FatalError::ModelLoad(format!("blk.{layer}: {e}"))))?;
    } else {
        for (name, got, want) in [
            ("ssm_a", weights.a.len(), dims.n_heads),
            ("ssm_d", weights.d_skip.len(), dims.n_heads),
            ("ssm_dt.bias", weights.dt_bias.len(), dims.n_heads),
            (
                "ssm_conv1d.weight",
                weights.conv1d.len(),
                dims.conv_dim * dims.conv_kernel,
            ),
            ("ssm_norm.weight", weights.norm.len(), dims.d_inner),
        ] {
            if got != want {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "blk.{layer}.{name} elements={got}, expected={want}"
                ))));
            }
        }
    }
    Ok(weights)
}

/// Prefer `ffn_norm` / `post_attention_norm`; MoE-only Lightning blocks reuse `attn_norm`.
fn gguf_ffn_norm(file: &GgufFile, layer: usize) -> Result<Vec<f32>, AllpacaError> {
    if file
        .tensor(&format!("blk.{layer}.ffn_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.ffn_norm.weight"))
    } else if file
        .tensor(&format!("blk.{layer}.post_attention_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.post_attention_norm.weight"))
    } else if file
        .tensor(&format!("blk.{layer}.attn_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.attn_norm.weight"))
    } else {
        Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "missing ffn_norm/post_attention_norm/attn_norm on blk.{layer}"
        ))))
    }
}

fn load_gguf_mla_host(
    file: &GgufFile,
    layer: usize,
    cfg: &ModelConfig,
) -> Result<Option<MlaHostWeights>, AllpacaError> {
    if file.tensor(&format!("blk.{layer}.attn_q_a.weight")).is_none()
        && file.tensor(&format!("blk.{layer}.attn_kv_a_mqa.weight")).is_none()
    {
        return Ok(None);
    }
    let q_lora = cfg.q_lora_rank.max(1);
    let kv_lora = cfg.kv_lora_rank.max(1);
    let n_heads = cfg.n_heads.max(1);
    let qk_rope = file
        .meta_u32(&format!(
            "{}.attention.key_length_mla",
            file.architecture().unwrap_or("deepseek2")
        ))
        .or_else(|| {
            file.meta_u32(&format!(
                "{}.rope.dimension_count",
                file.architecture().unwrap_or("deepseek2")
            ))
        })
        .unwrap_or(64) as usize;
    let qk_nope = (file
        .meta_u32(&format!(
            "{}.attention.key_length",
            file.architecture().unwrap_or("deepseek2")
        ))
        .unwrap_or(128) as usize)
        .saturating_sub(qk_rope)
        .max(64);
    let v_head = file
        .meta_u32(&format!(
            "{}.attention.value_length_mla",
            file.architecture().unwrap_or("deepseek2")
        ))
        .or_else(|| {
            file.meta_u32(&format!(
                "{}.attention.value_length",
                file.architecture().unwrap_or("deepseek2")
            ))
        })
        .unwrap_or(128) as usize;
    let dims = MlaDims {
        hidden_size: cfg.dim,
        num_heads: n_heads,
        q_lora_rank: q_lora,
        kv_lora_rank: kv_lora,
        qk_nope_head_dim: qk_nope,
        qk_rope_head_dim: qk_rope,
        v_head_dim: v_head,
    };
    let wdq = file.tensor_f32(&format!("blk.{layer}.attn_q_a.weight"))?;
    let wuq = file.tensor_f32(&format!("blk.{layer}.attn_q_b.weight"))?;
    let wdkv = file.tensor_f32(&format!("blk.{layer}.attn_kv_a_mqa.weight"))?;
    let wukv = if file.tensor(&format!("blk.{layer}.attn_kv_b.weight")).is_some() {
        file.tensor_f32(&format!("blk.{layer}.attn_kv_b.weight"))?
    } else {
        let k_b = file.tensor_f32(&format!("blk.{layer}.attn_k_b.weight"))?;
        let v_b = file.tensor_f32(&format!("blk.{layer}.attn_v_b.weight"))?;
        // Concatenate k_b and v_b along out dim for expand_kv compatibility when fused.
        let mut fused = k_b;
        fused.extend_from_slice(&v_b);
        fused
    };
    let wo = file
        .tensor_f32(&format!("blk.{layer}.attn_output.weight"))
        .or_else(|_| file.tensor_f32(&format!("blk.{layer}.attn_out.weight")))?;
    let q_norm = if file
        .tensor(&format!("blk.{layer}.attn_q_a_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.attn_q_a_norm.weight"))?
    } else {
        vec![1.0; q_lora]
    };
    let kv_norm = if file
        .tensor(&format!("blk.{layer}.attn_kv_a_norm.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.attn_kv_a_norm.weight"))?
    } else {
        vec![1.0; kv_lora]
    };
    Ok(Some(MlaHostWeights {
        dims,
        wdq,
        wuq,
        wdkv,
        wukv,
        wo,
        q_norm,
        kv_norm,
    }))
}

fn load_gguf_expert_bias(file: &GgufFile, layer: usize) -> Result<Vec<f32>, AllpacaError> {
    if file
        .tensor(&format!("blk.{layer}.ffn_exp_probs_b.bias"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.ffn_exp_probs_b.bias"))
    } else if file
        .tensor(&format!("blk.{layer}.ffn_exp_probs_b.weight"))
        .is_some()
    {
        file.tensor_f32(&format!("blk.{layer}.ffn_exp_probs_b.weight"))
    } else {
        Ok(Vec::new())
    }
}

/// Scalar meta, or max of a u32/i32 array (Nemotron stores per-layer head/ff dims).
fn meta_u32_or_array_max(file: &GgufFile, key: &str) -> Option<u32> {
    if let Some(v) = file.meta_u32(key) {
        return Some(v);
    }
    let items = file.metadata.get(key)?;
    match items {
        allpaca_gguf::GgufValue::Array(arr) => arr.iter().filter_map(|v| v.as_u32()).max(),
        _ => None,
    }
}

fn estimated_gguf_f32_bytes(file: &GgufFile, cfg: &ModelConfig) -> u64 {
    let mut names = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "output.weight".to_string(),
    ];
    for local in 0..cfg.n_layers {
        let i = cfg.gguf_blk(local);
        names.push(format!("blk.{i}.attn_norm.weight"));
        names.push(format!("blk.{i}.attn_q.weight"));
        names.push(format!("blk.{i}.attn_k.weight"));
        names.push(format!("blk.{i}.attn_v.weight"));
        names.push(format!("blk.{i}.attn_output.weight"));
        names.push(format!("blk.{i}.ffn_norm.weight"));
        if cfg.n_experts > 0 {
            names.push(format!("blk.{i}.ffn_gate_inp.weight"));
            names.push(format!("blk.{i}.ffn_gate_exps.weight"));
            names.push(format!("blk.{i}.ffn_up_exps.weight"));
            names.push(format!("blk.{i}.ffn_down_exps.weight"));
            for e in 0..cfg.n_experts {
                names.push(format!("blk.{i}.ffn_gate.{e}.weight"));
                names.push(format!("blk.{i}.ffn_up.{e}.weight"));
                names.push(format!("blk.{i}.ffn_down.{e}.weight"));
            }
        } else {
            names.push(format!("blk.{i}.ffn_gate.weight"));
            names.push(format!("blk.{i}.ffn_up.weight"));
            names.push(format!("blk.{i}.ffn_down.weight"));
        }
    }
    names
        .iter()
        .filter_map(|n| file.tensor(n))
        .map(|t| t.n_elements().saturating_mul(4))
        .sum()
}

fn require_gguf_tensor(file: &GgufFile, name: &str) -> Result<(), AllpacaError> {
    if file.tensor(name).is_none() {
        Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "missing tensor {name}"
        ))))
    } else {
        Ok(())
    }
}

fn gguf_blk_has_mla(file: &GgufFile, blk: usize) -> bool {
    file.tensor(&format!("blk.{blk}.attn_q_a.weight")).is_some()
}

fn attach_dsv4(model: &mut TinyModel, file: &GgufFile) -> Result<(), AllpacaError> {
    if !crate::deepseek4::file_is_deepseek4(file) {
        return Ok(());
    }
    let w = crate::deepseek4::Dsv4Weights::from_gguf(file, &model.config, false)?;
    model.dsv4 = Some(Arc::new(w));
    Ok(())
}

fn split_expert_rows(
    data: Vec<f32>,
    n_experts: usize,
    what: &str,
) -> Result<Vec<Vec<f32>>, AllpacaError> {
    if n_experts == 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{what} has no experts"
        ))));
    }
    if data.len() % n_experts != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{what} len {} is not divisible by {n_experts} experts",
            data.len()
        ))));
    }
    let chunk = data.len() / n_experts;
    Ok(data.chunks(chunk).map(|c| c.to_vec()).collect())
}

fn load_gguf_moe_layer(
    file: &GgufFile,
    layer: usize,
    n_experts: usize,
) -> Result<(Vec<f32>, Vec<LayerFfn>), AllpacaError> {
    let router_name = format!("blk.{layer}.ffn_gate_inp.weight");
    let w_router = file
        .tensor_f32(&router_name)
        .or_else(|_| file.tensor_f32(&format!("blk.{layer}.ffn_router.weight")))?;
    let stacked_gate = format!("blk.{layer}.ffn_gate_exps.weight");
    let stacked_up = format!("blk.{layer}.ffn_up_exps.weight");
    let experts = if file.tensor(&stacked_gate).is_some() {
        let gates = split_expert_rows(file.tensor_f32(&stacked_gate)?, n_experts, &stacked_gate)?;
        let ups = split_expert_rows(file.tensor_f32(&stacked_up)?, n_experts, "ffn_up_exps")?;
        let downs = split_expert_rows(
            file.tensor_f32(&format!("blk.{layer}.ffn_down_exps.weight"))?,
            n_experts,
            "ffn_down_exps",
        )?;
        (0..n_experts)
            .map(|e| LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: gates[e].clone(),
                w_up: ups[e].clone(),
                w_down: downs[e].clone(),
            })
            .collect()
    } else if file.tensor(&stacked_up).is_some() {
        // Gate-less stacked experts (Nemotron Lightning): up + down only.
        let ups = split_expert_rows(file.tensor_f32(&stacked_up)?, n_experts, "ffn_up_exps")?;
        let downs = split_expert_rows(
            file.tensor_f32(&format!("blk.{layer}.ffn_down_exps.weight"))?,
            n_experts,
            "ffn_down_exps",
        )?;
        (0..n_experts)
            .map(|e| LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: Vec::new(),
                w_up: ups[e].clone(),
                w_down: downs[e].clone(),
            })
            .collect()
    } else {
        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let gate_name = format!("blk.{layer}.ffn_gate.{e}.weight");
            let w_gate = if file.tensor(&gate_name).is_some() {
                file.tensor_f32(&gate_name)?
            } else {
                Vec::new()
            };
            experts.push(LayerFfn {
                ffn_norm: Vec::new(),
                w_gate,
                w_up: file.tensor_f32(&format!("blk.{layer}.ffn_up.{e}.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{layer}.ffn_down.{e}.weight"))?,
            });
        }
        experts
    };
    Ok((w_router, experts))
}

fn load_gguf_shared_expert(file: &GgufFile, layer: usize) -> Result<LayerFfn, AllpacaError> {
    let up_name = format!("blk.{layer}.ffn_up_shexp.weight");
    if file.tensor(&up_name).is_none() {
        return Ok(LayerFfn::default());
    }
    let gate_name = format!("blk.{layer}.ffn_gate_shexp.weight");
    Ok(LayerFfn {
        ffn_norm: Vec::new(),
        w_gate: if file.tensor(&gate_name).is_some() {
            file.tensor_f32(&gate_name)?
        } else {
            Vec::new()
        },
        w_up: file.tensor_f32(&up_name)?,
        w_down: file.tensor_f32(&format!("blk.{layer}.ffn_down_shexp.weight"))?,
    })
}

fn append_gguf_shared_expert<'a>(
    file: &'a GgufFile,
    layer: usize,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
) -> Result<(WeightRef, WeightRef), AllpacaError> {
    let up_name = format!("blk.{layer}.ffn_up_shexp.weight");
    if file.tensor(&up_name).is_none() {
        return Ok((WeightRef::f32(0, 0), WeightRef::f32(0, 0)));
    }
    Ok((
        append_gguf_tensor(file, &up_name, parts, off)?,
        append_gguf_tensor(
            file,
            &format!("blk.{layer}.ffn_down_shexp.weight"),
            parts,
            off,
        )?,
    ))
}

fn append_gguf_tensor<'a>(
    file: &'a GgufFile,
    name: &str,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
) -> Result<WeightRef, AllpacaError> {
    let info = file.tensor(name).ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
    })?;
    let n = info.n_elements() as usize;
    let (part, kind) = match info.ggml_type {
        allpaca_gguf::GgmlType::F32 => (
            UploadPart::Mapped(file.tensor_bytes(name)?),
            WeightKind::F32,
        ),
        allpaca_gguf::GgmlType::F16 => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 {
                let f32s = file.tensor_f32(name)?;
                let mut bytes = Vec::with_capacity(f32s.len() * 4);
                for v in &f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                (UploadPart::Mapped(packed), WeightKind::F16)
            }
        }
        allpaca_gguf::GgmlType::Q4_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_k(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q4_K GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (UploadPart::Mapped(packed), WeightKind::Q4_K)
            }
        }
        allpaca_gguf::GgmlType::TQ2_0 => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_tq2_0(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "TQ2_0 GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (UploadPart::Mapped(packed), WeightKind::TQ2_0)
            }
        }
        allpaca_gguf::GgmlType::TQ1_0 => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_tq1_0(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "TQ1_0 GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (UploadPart::Mapped(packed), WeightKind::TQ1_0)
            }
        }
        allpaca_gguf::GgmlType::RocmFp4 => {
            let packed = file.tensor_bytes(name)?;
            // Keep large embeds packed: vocab×dim F32 would be multi-GiB host RAM.
            if n <= 65_536 {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_rocmfp4(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "ROCmFP4 GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (UploadPart::Mapped(packed), WeightKind::RocmFp4)
            }
        }
        allpaca_gguf::GgmlType::Q5_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q5_k(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q5_K GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (UploadPart::Mapped(packed), WeightKind::Q5_K)
            }
        }
        allpaca_gguf::GgmlType::Q6_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q6_k(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q6_K GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (UploadPart::Mapped(packed), WeightKind::Q6_K)
            }
        }
        allpaca_gguf::GgmlType::Q8_0 => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q8_0(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (UploadPart::Owned(bytes), WeightKind::F32)
            } else if name == "token_embd.weight" {
                let staged = allpaca_hip::stage_q8_0_blocks(packed).map_err(|e| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!("Q8_0 stage {name}: {e}")))
                })?;
                (UploadPart::Owned(staged), WeightKind::Q8_0)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q8_0 GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                let n_rows = n / k;
                let staged = allpaca_hip::stage_q8_0_gemv(packed, n_rows, k).map_err(|e| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q8_0 GEMV stage {name}: {e}"
                    )))
                })?;
                (UploadPart::Owned(staged), WeightKind::Q8_0)
            }
        }
        _ => {
            let f32s = file.tensor_f32(name)?;
            let mut bytes = Vec::with_capacity(f32s.len() * 4);
            for v in &f32s {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (UploadPart::Owned(bytes), WeightKind::F32)
        }
    };
    if kind == WeightKind::Q8_0 && *off % allpaca_hip::Q8_STAGED_BLOCK != 0 {
        let pad = allpaca_hip::Q8_STAGED_BLOCK - (*off % allpaca_hip::Q8_STAGED_BLOCK);
        parts.push(UploadPart::Owned(vec![0u8; pad]));
        *off += pad;
    }
    if kind == WeightKind::F16 && *off % 16 != 0 {
        let pad = 16 - (*off % 16);
        parts.push(UploadPart::Owned(vec![0u8; pad]));
        *off += pad;
    }
    let w = WeightRef { off: *off, n, kind };
    *off += part.len();
    parts.push(part);
    Ok(w)
}

fn split_fused_qkv_ref(
    fused: WeightRef,
    uploaded_bytes: usize,
    input_dim: usize,
    q_out: usize,
    kv_out: usize,
) -> Result<[WeightRef; 3], AllpacaError> {
    let total_out = q_out.saturating_add(kv_out.saturating_mul(2));
    let expected = total_out.saturating_mul(input_dim);
    if total_out == 0 || fused.n != expected || uploaded_bytes % total_out != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "fused attn_qkv shape mismatch: elements={} expected={expected} bytes={uploaded_bytes}",
            fused.n
        ))));
    }
    let row_bytes = uploaded_bytes / total_out;
    let q = WeightRef {
        off: fused.off,
        n: q_out * input_dim,
        kind: fused.kind,
    };
    let k = WeightRef {
        off: q.off + q_out * row_bytes,
        n: kv_out * input_dim,
        kind: fused.kind,
    };
    let v = WeightRef {
        off: k.off + kv_out * row_bytes,
        n: kv_out * input_dim,
        kind: fused.kind,
    };
    Ok([q, k, v])
}

/// Host staging for GGUF→device upload: mmap borrow when possible, owned only
/// when staging/dequant forces a rewrite (keeps commit charge near one file size).
enum UploadPart<'a> {
    Mapped(&'a [u8]),
    Owned(Vec<u8>),
}

impl<'a> UploadPart<'a> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mapped(s) => s,
            Self::Owned(v) => v.as_slice(),
        }
    }

    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

fn split_packed_experts(
    w: WeightRef,
    n_experts: usize,
    what: &str,
) -> Result<Vec<WeightRef>, AllpacaError> {
    if n_experts == 0 || w.n % n_experts != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{what} n={} is not divisible by {n_experts} experts",
            w.n
        ))));
    }
    let n_per = w.n / n_experts;
    let stride = match w.kind {
        WeightKind::F32 => n_per.saturating_mul(4),
        WeightKind::F16 => n_per.saturating_mul(2),
        WeightKind::RocmFp4 => {
            if n_per % allpaca_cpu::ROCMFP4_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} ROCmFP4 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::ROCMFP4_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::ROCMFP4_BLOCK_ELEMS)
                .saturating_mul(allpaca_cpu::ROCMFP4_BLOCK_BYTES)
        }
        WeightKind::Q4_K => {
            if n_per % allpaca_cpu::Q4_K_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} Q4_K expert elems {n_per} not aligned to {}",
                    allpaca_cpu::Q4_K_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::Q4_K_BLOCK_ELEMS).saturating_mul(allpaca_cpu::Q4_K_BLOCK_BYTES)
        }
        WeightKind::TQ2_0 => {
            if n_per % allpaca_cpu::TQ2_0_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} TQ2_0 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::TQ2_0_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::TQ2_0_BLOCK_ELEMS).saturating_mul(allpaca_cpu::TQ2_0_BLOCK_BYTES)
        }
        WeightKind::TQ1_0 => {
            if n_per % allpaca_cpu::TQ1_0_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} TQ1_0 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::TQ1_0_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::TQ1_0_BLOCK_ELEMS).saturating_mul(allpaca_cpu::TQ1_0_BLOCK_BYTES)
        }
        WeightKind::Q5_K => {
            if n_per % allpaca_cpu::Q5_K_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} Q5_K expert elems {n_per} not aligned to {}",
                    allpaca_cpu::Q5_K_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::Q5_K_BLOCK_ELEMS).saturating_mul(allpaca_cpu::Q5_K_BLOCK_BYTES)
        }
        WeightKind::Q6_K => {
            if n_per % allpaca_cpu::Q6_K_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} Q6_K expert elems {n_per} not aligned to {}",
                    allpaca_cpu::Q6_K_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::Q6_K_BLOCK_ELEMS).saturating_mul(allpaca_cpu::Q6_K_BLOCK_BYTES)
        }
        WeightKind::Q8_0 => {
            if n_per % allpaca_cpu::Q8_0_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} Q8_0 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::Q8_0_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::Q8_0_BLOCK_ELEMS).saturating_mul(allpaca_cpu::Q8_0_BLOCK_BYTES)
        }
    };
    Ok((0..n_experts)
        .map(|e| WeightRef {
            off: w.off + e * stride,
            n: n_per,
            kind: w.kind,
        })
        .collect())
}

fn append_gguf_experts<'a>(
    file: &'a GgufFile,
    layer: usize,
    n_experts: usize,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
) -> Result<Vec<[WeightRef; 3]>, AllpacaError> {
    let stacked_gate = format!("blk.{layer}.ffn_gate_exps.weight");
    let stacked_up = format!("blk.{layer}.ffn_up_exps.weight");
    if file.tensor(&stacked_gate).is_some() || file.tensor(&stacked_up).is_some() {
        let gate = if file.tensor(&stacked_gate).is_some() {
            append_gguf_tensor(file, &stacked_gate, parts, off)?
        } else {
            WeightRef::f32(0, 0)
        };
        let up = append_gguf_tensor(file, &stacked_up, parts, off)?;
        let down = append_gguf_tensor(
            file,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            parts,
            off,
        )?;
        let ups = split_packed_experts(up, n_experts, "ffn_up_exps")?;
        let downs = split_packed_experts(down, n_experts, "ffn_down_exps")?;
        let gates = if gate.n > 0 {
            split_packed_experts(gate, n_experts, "ffn_gate_exps")?
        } else {
            vec![WeightRef::f32(0, 0); n_experts]
        };
        return Ok((0..n_experts)
            .map(|e| [gates[e], ups[e], downs[e]])
            .collect());
    }
    let mut out = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        let gate_name = format!("blk.{layer}.ffn_gate.{e}.weight");
        let w_gate = if file.tensor(&gate_name).is_some() {
            append_gguf_tensor(file, &gate_name, parts, off)?
        } else {
            WeightRef::f32(0, 0)
        };
        out.push([
            w_gate,
            append_gguf_tensor(file, &format!("blk.{layer}.ffn_up.{e}.weight"), parts, off)?,
            append_gguf_tensor(
                file,
                &format!("blk.{layer}.ffn_down.{e}.weight"),
                parts,
                off,
            )?,
        ]);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct GdnHostWeights {
    pub dims: GdnDims,
    pub a: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub conv1d: Vec<f32>,
    /// Optional host projections for synthetic / CUDA host-step path.
    /// Empty when HIP keeps projections on-device only.
    pub w_qkv: Vec<f32>,
    pub w_z: Vec<f32>,
    pub w_alpha: Vec<f32>,
    pub w_beta: Vec<f32>,
    pub w_out: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub block: HybridLayerKind,
    pub ssm: Option<SsmLayerWeights>,
    /// Qwen35 Gated DeltaNet host scalars / conv (projections stay on device).
    pub gdn: Option<GdnHostWeights>,
    /// Ling MLA host tensors (edge0-8b).
    pub mla: Option<MlaHostWeights>,
    pub attn_norm: Vec<f32>,
    /// Per-head Q/K RMSNorm (Qwen3.5 full-attn); empty when unused.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub w_router: Vec<f32>,
    pub experts: Vec<LayerFfn>,
    /// Always-on shared expert (`ffn_*_shexp`); empty when unused.
    pub shared_expert: LayerFfn,
    /// DeepSeek-style expert bias for sigmoid-group selection.
    pub expert_bias: Vec<f32>,
    /// Spark2_5 head-wise attention output gate (`attn_gate.weight`, shape `[n_heads, dim]`).
    pub attn_out_gate: Vec<f32>,
    /// Gemma post-attention RMSNorm (empty = skip).
    pub attn_post_norm: Vec<f32>,
    /// Gemma post-FFN RMSNorm (empty = skip).
    pub ffn_post_norm: Vec<f32>,
}

/// Runtime Edge0 adapters attached after load.
#[derive(Debug, Default)]
pub struct Edge0Runtime {
    pub lora: Option<RecoverLora>,
    /// Interior mutability: decode takes `&self` but stager updates every token.
    pub stager: Option<std::sync::Mutex<CrossTokenStager>>,
}

impl Clone for Edge0Runtime {
    fn clone(&self) -> Self {
        Self {
            lora: self.lora.clone(),
            stager: self.stager.as_ref().map(|m| {
                let inner = m
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                std::sync::Mutex::new(inner)
            }),
        }
    }
}

/// Host NextN/MTP seed tensors (`blk.{i}.nextn.*`). Device ROCmFP4 drafts skip these.
#[derive(Debug, Clone, Default)]
pub struct MtpNextnWeights {
    /// `[dim, 2*dim]` — maps concat(RMSNorm(tok), RMSNorm(h)) → seed.
    pub eh_proj: Vec<f32>,
    pub enorm: Vec<f32>,
    pub hnorm: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TinyModel {
    pub config: ModelConfig,
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub name: String,
    pub working_set: Option<Arc<WeightWorkingSet>>,
    pub expert_lru: Option<Arc<ExpertLru>>,
    pub expert_host: Option<Arc<HostExpertPool>>,
    /// Manifest early-exit capability (from GGUF `allpaca.early_exit.*` or test hook).
    pub early_exit: EarlyExitSpec,
    /// Optional dedicated exit heads (`blk.{i}.exit_head.weight`); empty = shared lm_head.
    pub exit_heads: Vec<Option<Vec<f32>>>,
    /// Present when this TinyModel is an MTP/NextN draft with host seed weights.
    pub mtp_nextn: Option<MtpNextnWeights>,
    /// DeepSeek-V4 MLA / hyper-connection / MoE sidecar (CPU decode).
    pub(crate) dsv4: Option<Arc<crate::deepseek4::Dsv4Weights>>,
    /// Shared RPC runtime when layers are offloaded to remote workers.
    pub rpc: Option<std::sync::Arc<crate::rpc_runtime::RpcRuntime>>,
    /// Edge0 Recover-LoRA + prerouter stager (when loaded via `load_edge0`).
    pub edge0: Option<Edge0Runtime>,
    /// DeepSeek2 / Ling MLA compressed KV cache: per-layer (ckv_seq, krope_seq).
    pub mla_cache: Option<Arc<std::sync::Mutex<Vec<(Vec<f32>, Vec<f32>)>>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AttnLayout {
    q_out: usize,
    kv_dim: usize,
    wo_k: usize,
    gated: bool,
    attn_dim: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
}

fn attn_layout(cfg: &ModelConfig, wq_n: usize, wk_n: usize, wo_n: usize) -> AttnLayout {
    let dim = cfg.dim.max(1);
    let q_out = if wq_n > 0 && wq_n % dim == 0 {
        wq_n / dim
    } else {
        dim
    };
    let kv_dim = if wk_n > 0 && wk_n % dim == 0 {
        wk_n / dim
    } else {
        cfg.n_kv_heads * cfg.head_dim().max(1)
    };
    let wo_k = if wo_n > 0 && wo_n % dim == 0 {
        wo_n / dim
    } else {
        dim
    };
    let gated = q_out == wo_k.saturating_mul(2) && wo_k > dim;
    let attn_dim = if gated {
        wo_k
    } else {
        q_out.min(wo_k).max(dim)
    };
    let head_dim = {
        let c = cfg.head_dim().max(1);
        if attn_dim % c == 0 && kv_dim % c == 0 {
            c
        } else if attn_dim % 256 == 0 && kv_dim % 256 == 0 {
            256
        } else {
            c
        }
    };
    AttnLayout {
        q_out,
        kv_dim,
        wo_k,
        gated,
        attn_dim,
        head_dim,
        n_heads: (attn_dim / head_dim).max(1),
        n_kv_heads: (kv_dim / head_dim).max(1),
    }
}

impl TinyModel {
    /// HIP `forward_tokens_device` is one GEMM with M=N on attention layers.
    /// MoE groups tokens that share an expert (`M` = group size); shared expert
    /// is `M = n_tok`. Mamba-2 / GDN blocks still walk token rows. One
    /// `gpu_forward` per chunk. Partial NeoX RoPE uses the same Q/K `q_out` /
    /// `n_heads` layout as decode. Gated-Q and IMRoPE stay per-token.
    pub fn supports_device_batched_prefill(&self) -> bool {
        if self.config.uses_imrope() {
            return false;
        }
        let dim = self.config.dim.max(1);
        !self.layers.iter().any(|l| {
            if l.wq.is_empty() || l.wo.is_empty() {
                return false;
            }
            let q_out = l.wq.len() / dim;
            let wo_k = l.wo.len() / dim;
            q_out == wo_k.saturating_mul(2) && wo_k > dim
        })
    }

    pub fn synthetic(cfg: ModelConfig, seed: u64, name: impl Into<String>) -> Self {
        let mut rng = Lcg::new(seed);
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let (w_gate, w_up, w_down, w_router, experts) = if cfg.n_experts > 0 {
                let experts = (0..cfg.n_experts)
                    .map(|_| LayerFfn {
                        ffn_norm: Vec::new(),
                        w_gate: rng.vec(cfg.ff_dim * dim),
                        w_up: rng.vec(cfg.ff_dim * dim),
                        w_down: rng.vec(dim * cfg.ff_dim),
                    })
                    .collect();
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    rng.vec(cfg.n_experts * dim),
                    experts,
                )
            } else {
                (
                    rng.vec(cfg.ff_dim * dim),
                    rng.vec(cfg.ff_dim * dim),
                    rng.vec(dim * cfg.ff_dim),
                    Vec::new(),
                    Vec::new(),
                )
            };
            layers.push(LayerWeights {
                block: HybridLayerKind::Attention,
                ssm: None,
                gdn: None,
                mla: None,
                attn_norm: ones(dim),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                wq: rng.vec(dim * dim),
                wk: rng.vec(kv_dim * dim),
                wv: rng.vec(kv_dim * dim),
                wo: rng.vec(dim * dim),
                ffn_norm: ones(dim),
                w_gate,
                w_up,
                w_down,
                w_router,
                experts,
                shared_expert: LayerFfn::default(),
                expert_bias: Vec::new(),
                attn_out_gate: Vec::new(),
                attn_post_norm: Vec::new(),
                ffn_post_norm: Vec::new(),
            });
        }
        Self {
            embed: rng.vec(vocab * dim),
            layers,
            norm: ones(dim),
            lm_head: rng.vec(vocab * dim),
            name: name.into(),
            config: cfg,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            rpc: None,
            edge0: None,
            mla_cache: None,
        }
    }

    /// Attention + Mamba-2 SSM hybrid for CUDA/host decode tracers.
    /// Layer 0 is SSM; remaining layers stay dense attention.
    pub fn synthetic_ssm_hybrid(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny_hybrid(), seed, "ssm-hybrid");
        let dim = model.config.dim;
        let dims = crate::ssm::Mamba2Dims::new(dim, dim, 2, 4, 2, 2)
            .expect("tiny SSM dims");
        let mut rng = Lcg::new(seed.wrapping_add(99));
        // Mildly negative A so selective scan stays stable under random inputs.
        let a: Vec<f32> = (0..dims.n_heads).map(|_| -0.5 + rng.next_f32()).collect();
        model.layers[0] = LayerWeights {
            block: HybridLayerKind::Ssm,
            ssm: Some(crate::ssm::SsmLayerWeights {
                dims,
                w_in: rng.vec(dims.in_proj_dim * dim),
                w_out: rng.vec(dim * dims.d_inner),
                a,
                d_skip: rng.vec(dims.n_heads),
                dt_bias: rng.vec(dims.n_heads),
                conv1d: rng.vec(dims.conv_dim * dims.conv_kernel),
                conv_bias: rng.vec(dims.conv_dim),
                norm: ones(dims.d_inner),
            }),
            gdn: None,
            mla: None,
            attn_norm: ones(dim),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            ffn_norm: Vec::new(),
            w_gate: Vec::new(),
            w_up: Vec::new(),
            w_down: Vec::new(),
            w_router: Vec::new(),
            experts: Vec::new(),
            shared_expert: LayerFfn::default(),
            expert_bias: Vec::new(),
            attn_out_gate: Vec::new(),
            attn_post_norm: Vec::new(),
            ffn_post_norm: Vec::new(),
        };
        model
    }

    /// Attention + Gated DeltaNet hybrid for CUDA/host decode tracers.
    /// Layer 0 is GDN (SSM block kind); remaining layers stay dense attention.
    pub fn synthetic_gdn_hybrid(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny_hybrid(), seed, "gdn-hybrid");
        let dim = model.config.dim;
        // head_v=4, n_v=4 → inner=16; n_k=2 → key_dim=8; qkv_dim=32 (=dim).
        let dims = GdnDims::from_meta(4, 16, 4, 2, 2).expect("tiny GDN dims");
        let mut rng = Lcg::new(seed.wrapping_add(77));
        let a: Vec<f32> = (0..dims.n_v_heads).map(|_| -0.5 + rng.next_f32()).collect();
        model.layers[0] = LayerWeights {
            block: HybridLayerKind::Ssm,
            ssm: None,
            gdn: Some(GdnHostWeights {
                dims,
                a,
                dt_bias: rng.vec(dims.n_v_heads),
                norm: ones(dims.head_v),
                conv1d: rng.vec(dims.conv_kernel * dims.qkv_dim),
                w_qkv: rng.vec(dims.qkv_dim * dim),
                w_z: rng.vec(dims.value_dim * dim),
                w_alpha: rng.vec(dims.n_v_heads * dim),
                w_beta: rng.vec(dims.n_v_heads * dim),
                w_out: rng.vec(dim * dims.value_dim),
            }),
            mla: None,
            attn_norm: ones(dim),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            ffn_norm: Vec::new(),
            w_gate: Vec::new(),
            w_up: Vec::new(),
            w_down: Vec::new(),
            w_router: Vec::new(),
            experts: Vec::new(),
            shared_expert: LayerFfn::default(),
            expert_bias: Vec::new(),
            attn_out_gate: Vec::new(),
            attn_post_norm: Vec::new(),
            ffn_post_norm: Vec::new(),
        };
        model
    }

    /// Attention layers with Multi-head Latent Attention on layer 0 (host MLA step).
    pub fn synthetic_mla_hybrid(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "mla-hybrid");
        let dims = crate::edge0::MlaDims::from_hidden(model.config.dim, model.config.n_heads);
        model.layers[0].mla = Some(crate::edge0::MlaHostWeights::synthetic(dims, 0.01));
        // Prefer MLA over dense QKV on layer 0.
        model.layers[0].wq.clear();
        model.layers[0].wk.clear();
        model.layers[0].wv.clear();
        model.layers[0].wo.clear();
        model.mla_cache = Some(Arc::new(std::sync::Mutex::new(
            vec![(Vec::new(), Vec::new()); model.layers.len()],
        )));
        model
    }

    /// Tiny MoE with an always-on shared expert (SwiGLU) on every layer.
    pub fn synthetic_moe_shared(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny_moe(), seed, "moe-shared");
        let dim = model.config.dim;
        let ff = model.config.ff_dim;
        let mut rng = Lcg::new(seed.wrapping_add(55));
        for layer in &mut model.layers {
            layer.shared_expert = LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: rng.vec(ff * dim),
                w_up: rng.vec(ff * dim),
                w_down: rng.vec(dim * ff),
            };
        }
        model
    }

    /// Layer 0 is DenseFfn-only (no attention); remaining layers stay dense attention.
    pub fn synthetic_dense_ffn_hybrid(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "dense-ffn-hybrid");
        let layer = &mut model.layers[0];
        layer.block = HybridLayerKind::DenseFfn;
        layer.wq.clear();
        layer.wk.clear();
        layer.wv.clear();
        layer.wo.clear();
        model
    }

    /// Tiny MoE with gate-less shared expert using Lightning ReLU² activation.
    pub fn synthetic_moe_shared_relu_sqr(seed: u64) -> Self {
        let mut cfg = ModelConfig::tiny_moe();
        cfg.moe_relu_sqr = true;
        let mut model = Self::synthetic(cfg, seed, "moe-shared-relu2");
        let dim = model.config.dim;
        let ff = model.config.ff_dim;
        let mut rng = Lcg::new(seed.wrapping_add(61));
        for layer in &mut model.layers {
            layer.shared_expert = LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: Vec::new(),
                w_up: rng.vec(ff * dim),
                w_down: rng.vec(dim * ff),
            };
        }
        model
    }

    /// Tiny dense model with Gemma-style attn + final logit softcaps.
    pub fn synthetic_softcap(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "softcap");
        model.config.attn_logit_softcap = 50.0;
        model.config.final_logit_softcap = 30.0;
        model
    }

    /// Tiny dense model with sliding-window attention on every layer.
    pub fn synthetic_swa(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "swa");
        model.config.sliding_window = 2;
        model.config.swa_pattern = vec![true; model.config.n_layers];
        model
    }

    pub fn host_weights_ready(&self) -> bool {
        !self.embed.is_empty()
    }

    /// Keep a GGUF handle for lazy DeepSeek-V4 MLA / expert / lm_head gathers.
    pub(crate) fn bind_gguf(&mut self, file: Arc<GgufFile>) {
        if let Some(dsv4) = &mut self.dsv4 {
            if let Some(inner) = Arc::get_mut(dsv4) {
                inner.attach_gguf(file);
            }
        }
    }

    pub(crate) fn expert_slot_bytes(&self) -> usize {
        if let Some(pool) = &self.expert_host {
            return pool.expert_bytes();
        }
        self.layers
            .iter()
            .find_map(|l| l.experts.first())
            .map(|e| e.bytes())
            .unwrap_or(0)
    }

    pub(crate) fn host_expert(
        &self,
        layer_idx: usize,
        expert: usize,
    ) -> Result<LayerFfn, AllpacaError> {
        let id = ExpertId {
            layer: layer_idx as u32,
            expert: expert as u32,
        };
        if let Some(pool) = &self.expert_host {
            return pool.get(id);
        }
        if let Some(ffn) = self
            .layers
            .get(layer_idx)
            .and_then(|l| l.experts.get(expert))
            .filter(|e| !e.w_up.is_empty() || !e.w_gate.is_empty())
        {
            return Ok(ffn.clone());
        }
        // Large MoE GGUFs skip host expert materialize; device-resident
        // `expert_offs` supply weights via `bind_expert`.
        if expert < self.config.n_experts {
            return Ok(LayerFfn::default());
        }
        Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "missing expert {expert} on layer {layer_idx}"
        ))))
    }

    /// Bind CPU expert MLP to the preferred NUMA node when a host pool is present.
    fn run_expert_mlp_numa(
        &self,
        id: ExpertId,
        x_norm: &[f32],
        ffn: &LayerFfn,
        ff_dim: usize,
        dim: usize,
        out: &mut [f32],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if let Some(pool) = &self.expert_host {
            let node = pool.preferred_numa_node(id);
            let r = pool.numa().run_on_node(node, || {
                expert_mlp(x_norm, ffn, ff_dim, dim, out, self.config.moe_relu_sqr)
            });
            if let Some(m) = metrics {
                HostExpertPool::record_numa(m, pool.numa());
            }
            return r;
        }
        expert_mlp(x_norm, ffn, ff_dim, dim, out, self.config.moe_relu_sqr)
    }

    fn host_experts_layer(&self, layer_idx: usize) -> Result<Vec<LayerFfn>, AllpacaError> {
        let n_exp = self.config.n_experts;
        if let Some(pool) = &self.expert_host {
            return pool.get_layer(layer_idx as u32, n_exp);
        }
        self.layers
            .get(layer_idx)
            .map(|l| l.experts.clone())
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "MoE missing host layer {layer_idx}"
                )))
            })
    }

    fn ffn<'a>(
        &'a self,
        layer_idx: usize,
        layer: &'a LayerWeights,
    ) -> Result<FfnRefs<'a>, AllpacaError> {
        if let Some(ws) = &self.working_set {
            if !ws.is_pinned(layer_idx) {
                return Ok(FfnRefs::Bound(ws.bind(layer_idx)?));
            }
        }
        Ok(FfnRefs::Resident(layer))
    }

    fn device_ffn<'a>(
        &'a self,
        resident: &'a ResidentModel,
        layer_idx: usize,
        metrics: Option<&Metrics>,
    ) -> Result<DeviceFfn<'a>, AllpacaError> {
        // Warm the host ring (and spare-slot L+1 prefetch) before H2D — same
        // policy as CPU `ffn()` → `WeightWorkingSet::bind`.
        let host = match self.working_set.as_ref() {
            Some(ws) if !ws.is_pinned(layer_idx) => {
                {
                    let _warm = ws.bind(layer_idx)?;
                }
                Some(ws.load_layer(layer_idx)?)
            }
            Some(ws) => ws.host_ffn(layer_idx).cloned(),
            None => None,
        };
        resident.bind_ffn(layer_idx, host, self.working_set.as_deref(), metrics)
    }

    fn streamed_ffn(&self, layer_idx: usize) -> bool {
        self.working_set
            .as_ref()
            .is_some_and(|ws| !ws.is_pinned(layer_idx))
    }

    fn apply_ffn(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &mut [f32],
        scratch: &mut [f32],
        hidden: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
        prefill: bool,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let use_moe = cfg.n_experts > 0
            && !layer.experts.is_empty()
            && !matches!(layer.block, HybridLayerKind::DenseFfn)
            && layer_idx >= cfg.leading_dense_blocks;
        if use_moe {
            if prefill {
                self.prefetch_moe_layers(layer_idx)?;
            }
            rmsnorm(x, &layer.ffn_norm, cfg.eps, &mut scratch[..dim])?;
            let moe_out = self.moe_hidden(layer_idx, layer, &scratch[..dim])?;
            for i in 0..dim {
                x[i] += moe_out[i];
            }
            return Ok(());
        }
        let ffn = self.ffn(layer_idx, layer)?;
        let ffn_dev = !self.streamed_ffn(layer_idx);
        rmsnorm_dispatch(
            hip,
            x,
            ffn.ffn_norm(),
            resident.filter(|_| ffn_dev).map(|r| r.ffn_norm(layer_idx)),
            cfg.eps,
            &mut scratch[..dim],
            metrics,
        )?;
        let mut gate = vec![0.0f32; cfg.ff_dim];
        let mut up = vec![0.0f32; cfg.ff_dim];
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            ffn.w_gate(),
            resident.filter(|_| ffn_dev).map(|r| r.w_gate(layer_idx)),
            &mut gate,
            1,
            cfg.ff_dim,
            dim,
        )?;
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            ffn.w_up(),
            resident.filter(|_| ffn_dev).map(|r| r.w_up(layer_idx)),
            &mut up,
            1,
            cfg.ff_dim,
            dim,
        )?;
        silu_mul_dispatch(hip, &mut gate, &up, metrics, cfg.ffn_gelu)?;
        gemm_nt_dispatch(
            hip,
            &gate,
            ffn.w_down(),
            resident.filter(|_| ffn_dev).map(|r| r.w_down(layer_idx)),
            hidden,
            1,
            dim,
            cfg.ff_dim,
        )?;
        if !layer.ffn_post_norm.is_empty() {
            let tmp = hidden.to_vec();
            rmsnorm(&tmp, &layer.ffn_post_norm, cfg.eps, hidden)?;
        }
        for i in 0..dim {
            x[i] += hidden[i];
        }
        Ok(())
    }

    fn moe_hidden(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        x_norm: &[f32],
    ) -> Result<Vec<f32>, AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let routed = if self.dsv4.is_some()
            || cfg.moe_sqrt_softplus
            || (cfg.moe_sigmoid_gating && cfg.edge0_tier.is_none())
        {
            let n_exp = cfg.n_experts.max(1);
            let n_act = cfg.n_active.max(1).min(n_exp);
            topk_route(x_norm, &layer.w_router, n_exp, n_act, cfg)
        } else {
            self.edge0_moe_route(layer_idx, layer, x_norm)
        };
        let ids: Vec<ExpertId> = routed
            .iter()
            .map(|&(e, _)| ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            })
            .collect();
        if let Some(lru) = &self.expert_lru {
            let plan = lru.plan(&ids);
            lru.note_plan(&plan);
            for id in &plan.fill {
                let host = self.host_expert(layer_idx, id.expert as usize)?;
                lru.admit(*id, &host)?;
            }
        }
        let mut acc = vec![0.0f32; dim];
        let mut scratch_h = vec![0.0f32; dim];
        for &(e, w) in &routed {
            let host = self.host_expert(layer_idx, e)?;
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            };
            let ffn = if let Some(lru) = &self.expert_lru {
                lru.materialize(id, &host)?
            } else {
                host
            };
            self.run_expert_mlp_numa(id, x_norm, &ffn, cfg.ff_dim, dim, &mut scratch_h, None)?;
            for i in 0..dim {
                acc[i] += w * scratch_h[i];
            }
        }
        if !layer.shared_expert.w_up.is_empty() {
            let sff = if cfg.shared_ff_dim > 0 {
                cfg.shared_ff_dim
            } else {
                layer.shared_expert.w_up.len() / dim.max(1)
            };
            expert_mlp(
                x_norm,
                &layer.shared_expert,
                sff,
                dim,
                &mut scratch_h,
                cfg.moe_relu_sqr,
            )?;
            for i in 0..dim {
                acc[i] += scratch_h[i];
            }
        }
        if let Some(e0) = &self.edge0 {
            if let Some(stager) = &e0.stager {
                let topk: Vec<usize> = routed.iter().map(|&(e, _)| e).collect();
                let mut g = stager.lock().unwrap_or_else(|e| e.into_inner());
                let _ = g.note_executed(layer_idx, x_norm, &topk);
            }
        }
        Ok(acc)
    }

    /// Edge0 prerouter: prefetch staged experts; optionally replace route (35b patch_call).
    fn edge0_moe_route(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        x_norm: &[f32],
    ) -> Vec<(usize, f32)> {
        let Some(e0) = &self.edge0 else {
            return moe_route(x_norm, layer, &self.config);
        };
        let Some(stager) = &e0.stager else {
            return moe_route(x_norm, layer, &self.config);
        };
        let mut g = stager.lock().unwrap_or_else(|e| e.into_inner());
        let staged = g.take_prediction(layer_idx);
        let patch = g.consume_staged(layer_idx);
        drop(g);
        if let Some(ref pred) = staged {
            if let Some(lru) = &self.expert_lru {
                for &(e, _) in pred {
                    if let Ok(host) = self.host_expert(layer_idx, e) {
                        let id = ExpertId {
                            layer: layer_idx as u32,
                            expert: e as u32,
                        };
                        let _ = lru.admit(id, &host);
                    }
                }
            }
            if patch {
                return pred.clone();
            }
        }
        moe_route(x_norm, layer, &self.config)
    }

    fn apply_edge0_lora(&self, layer_idx: usize, proj: &str, x: &[f32], y: &mut [f32]) {
        if let Some(e0) = &self.edge0 {
            if let Some(lora) = &e0.lora {
                lora.apply_named(
                    &format!("layers.{layer_idx}.self_attn.{proj}"),
                    x,
                    y,
                );
            }
        }
    }

    fn prefetch_moe_layers(&self, layer_idx: usize) -> Result<(), AllpacaError> {
        let Some(lru) = &self.expert_lru else {
            return Ok(());
        };
        let n_exp = self.config.n_experts;
        if n_exp == 0 || !lru.holds_layers(n_exp, 1) {
            return Ok(());
        }
        let experts = self.host_experts_layer(layer_idx)?;
        lru.admit_layer(layer_idx as u32, &experts)?;
        if lru.holds_layers(n_exp, 2) {
            if layer_idx + 1 < self.layers.len() {
                let next = self.host_experts_layer(layer_idx + 1)?;
                lru.admit_layer((layer_idx + 1) as u32, &next)?;
            }
        }
        Ok(())
    }

    fn warm_moe_slots(
        &self,
        layer_idx: usize,
        resident: &ResidentModel,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let Some(lru) = &self.expert_lru else {
            return Ok(());
        };
        let n_exp = self.config.n_experts;
        if n_exp == 0 || !lru.holds_layers(n_exp, 1) {
            return Ok(());
        }
        let warm = |idx: usize| -> Result<(), AllpacaError> {
            if resident.prefill_layer_gpu_ready(idx as u32) {
                return Ok(());
            }
            let experts = self.host_experts_layer(idx)?;
            resident.prefill_warm_layer_sync(idx as u32, n_exp as u32, &experts, metrics)
        };
        warm(layer_idx)?;
        if lru.holds_layers(n_exp, 2) && layer_idx + 1 < self.layers.len() {
            warm(layer_idx + 1)?;
        }
        Ok(())
    }

    pub(crate) fn try_nvme_expert_fill(
        &self,
        id: ExpertId,
        dev_off: usize,
        slot_bytes: usize,
        side: Option<&DeviceDecodeSide<'_>>,
        weights: &ResidentModel,
    ) -> Result<bool, AllpacaError> {
        let Some(side) = side else {
            return Ok(false);
        };
        let (probe, queue) = match (side.ds_probe, side.ds_queue) {
            (Some(p), Some(q)) => (p, q),
            _ => return Ok(false),
        };
        let pool = match &self.expert_host {
            Some(p) => p,
            None => return Ok(false),
        };
        let Some((path, offset, bytes)) = pool.slot_coords(id) else {
            return Ok(false);
        };
        let bytes = bytes.min(slot_bytes);
        queue.fill_to_device(
            probe,
            &path,
            offset,
            bytes,
            || {
                weights
                    .weights
                    .read_file_direct(&path, offset, dev_off, bytes)
            },
            |host| weights.copy_expert_h2d(dev_off, host),
        )
    }

    /// Router-only union of experts a draft verify span may touch (prefetch).
    pub fn routed_experts_for_verify_span(&self, span: &[u32]) -> Vec<ExpertId> {
        if self.config.n_experts == 0 || span.is_empty() {
            return Vec::new();
        }
        let dim = self.config.dim;
        let n_exp = self.config.n_experts.max(1);
        let n_act = self.config.n_active.max(1).min(n_exp);
        let mut union = std::collections::HashSet::new();
        for &tok in span {
            if tok as usize >= self.config.vocab_size {
                continue;
            }
            let x = &self.embed[tok as usize * dim..(tok as usize + 1) * dim];
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer.w_router.is_empty() {
                    continue;
                }
                for (e, _) in topk_route(x, &layer.w_router, n_exp, n_act, &self.config) {
                    union.insert(ExpertId {
                        layer: layer_idx as u32,
                        expert: e as u32,
                    });
                }
            }
        }
        let mut out: Vec<_> = union.into_iter().collect();
        out.sort_by_key(|id| (id.layer, id.expert));
        out
    }

    fn device_act_mul(
        &self,
        hip: &HipWorkspace,
        gate: *mut f32,
        up: *const f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if self.config.ffn_gelu {
            hip.gelu_mul_device(gate, up, n)
        } else {
            hip.silu_mul_device(gate, up, n)
        }
    }

    /// Spark2_5 head-wise sigmoid attention gate before `wo`.
    fn device_headwise_attn_gate(
        &self,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        layer: &LayerOff,
        buf: DeviceDecode,
        n_tok: usize,
        n_heads: usize,
        head_dim: usize,
        dim: usize,
    ) -> Result<(), AllpacaError> {
        if !self.config.headwise_attn_gate || layer.attn_out_gate.n == 0 || n_tok == 0 {
            return Ok(());
        }
        // scores[tok,h] = gate_w[h] · attn_inp[tok]; reuse buf.gate (n_tok*n_heads << n_tok*ff).
        resident.gemm_nt(
            hip,
            buf.normed,
            layer.attn_out_gate,
            buf.gate,
            n_tok as i32,
            n_heads as i32,
            dim as i32,
        )?;
        hip.scale_heads_sigmoid_device(
            buf.attn,
            buf.gate,
            n_tok as i32,
            n_heads as i32,
            head_dim as i32,
        )?;
        Ok(())
    }

    fn device_expert_gemm(
        &self,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        x: *const f32,
        m: i32,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
        ff_dim: usize,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        if m <= 0 {
            return Ok(());
        }
        let hidden = (m as usize).saturating_mul(ff_dim);
        resident.gemm_nt(hip, x, w_up, buf.up, m, ff_dim as i32, dim as i32)?;
        if w_gate.n == 0 {
            if cfg.moe_relu_sqr {
                hip.relu_sqr_device(buf.up, hidden as i32)?;
                resident.gemm_nt(hip, buf.up, w_down, buf.proj, m, dim as i32, ff_dim as i32)?;
                return Ok(());
            }
            // Gate-less tiny fixtures: silu(up) via silu_mul(gate←up, up=ones).
            hip.copy_d2d_f32(buf.gate, buf.up, hidden)?;
            let ones = vec![1.0f32; hidden];
            hip.copy_h2d_f32(buf.up, &ones)?;
            self.device_act_mul(hip, buf.gate, buf.up, hidden as i32)?;
        } else {
            resident.gemm_nt(hip, x, w_gate, buf.gate, m, ff_dim as i32, dim as i32)?;
            self.device_act_mul(hip, buf.gate, buf.up, hidden as i32)?;
        }
        resident.gemm_nt(
            hip,
            buf.gate,
            w_down,
            buf.proj,
            m,
            dim as i32,
            ff_dim as i32,
        )?;
        Ok(())
    }

    fn run_stagger_partner_cpu(
        &self,
        layer_idx: usize,
        x_norm: &[f32],
        metrics: &Metrics,
    ) -> Result<(), AllpacaError> {
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "stagger missing host layer {layer_idx}"
            )))
        })?;
        let cfg = &self.config;
        let dim = cfg.dim;
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let routed = topk_route(x_norm, &layer.w_router, n_exp, n_act, cfg);
        if routed.is_empty() {
            return Ok(());
        }
        let mut scratch = vec![0.0f32; dim];
        for &(e, w) in &routed {
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            };
            let host = self.host_expert(layer_idx, e)?;
            let ffn = if let Some(lru) = &self.expert_lru {
                let plan = lru.plan(&[id]);
                if !plan.fill.is_empty() {
                    lru.admit(id, &host)?;
                }
                lru.materialize(id, &host)?
            } else {
                host
            };
            if let Some(lru) = &self.expert_lru {
                if lru.plan(&[id]).cpu.iter().any(|&c| c == id) {
                    self.run_expert_mlp_numa(
                        id,
                        x_norm,
                        &ffn,
                        cfg.ff_dim,
                        dim,
                        &mut scratch,
                        None,
                    )?;
                    metrics.expert_cpu_runs.fetch_add(1, Ordering::Relaxed);
                    let _ = w;
                }
            }
        }
        Ok(())
    }

    fn device_moe(
        &self,
        layer_idx: usize,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        metrics: &Metrics,
        prefill: bool,
        micro_batch: Option<&MicroBatchExpertUnion>,
        mut side: Option<&mut DeviceDecodeSide<'_>>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        if prefill {
            self.prefetch_moe_layers(layer_idx)?;
            self.warm_moe_slots(layer_idx, resident, Some(metrics))?;
        }
        let host_layer = self.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MoE missing host layer {layer_idx}"
            )))
        })?;
        let off = resident.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MoE missing device layer {layer_idx}"
            )))
        })?;
        if off.ffn_norm.n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "MoE layer missing packed ffn_norm".into(),
            )));
        }
        hip.residual_rmsnorm(
            buf.x,
            buf.proj,
            resident.f32_or_dequant(hip, off.ffn_norm)?,
            buf.normed,
            dim as i32,
            cfg.eps,
        )?;
        metrics.residual_gpu_ops.fetch_add(1, Ordering::Relaxed);
        record_elementwise(Some(metrics));
        let mut x_norm = vec![0.0f32; dim];
        hip.copy_d2h_f32(buf.normed, &mut x_norm)?;
        let ds_probe = side.as_ref().and_then(|s| s.ds_probe);
        let ds_queue = side.as_ref().and_then(|s| s.ds_queue);
        let nvme_side = side.as_ref().map(|s| DeviceDecodeSide {
            request_id: s.request_id,
            moe_norm: None,
            layer_stagger: s.layer_stagger,
            ds_probe,
            ds_queue,
            last_hidden: None,
            write_logits: s.write_logits,
        });
        if let Some(s) = side.as_mut() {
            if let Some(norm) = s.moe_norm.as_mut() {
                if norm.len() == dim {
                    norm.copy_from_slice(&x_norm);
                }
            }
            if let Some(st) = s.layer_stagger.and_then(|ls| ls.active()) {
                if st.gpu_request == s.request_id && layer_idx == st.gpu_layer {
                    let partner_norm = s.layer_stagger.unwrap().partner_norm();
                    if partner_norm.len() == dim {
                        metrics.layer_stagger_steps.fetch_add(1, Ordering::Relaxed);
                        let _ = self.run_stagger_partner_cpu(st.cpu_layer, &partner_norm, metrics);
                    }
                }
            }
        }
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let routed = if self.edge0.is_some() || cfg.edge0_tier.is_some() {
            self.edge0_moe_route(layer_idx, host_layer, &x_norm)
        } else {
            topk_route(&x_norm, &host_layer.w_router, n_exp, n_act, cfg)
        };
        let ids: Vec<ExpertId> = routed
            .iter()
            .map(|&(e, _)| ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            })
            .collect();
        let plan = if let Some(lru) = &self.expert_lru {
            let route_ids = if !prefill {
                if let Some(u) = micro_batch {
                    let merged = u.union_with(layer_idx as u32, &ids);
                    if merged.len() == ids.len() {
                        u.stage_routed(layer_idx as u32, &ids);
                    }
                    merged
                } else {
                    ids.clone()
                }
            } else {
                ids.clone()
            };
            let union_plan = lru.plan(&route_ids);
            lru.note_plan(&union_plan);
            if !union_plan.fill.is_empty() {
                let mut pairs: Vec<(ExpertId, LayerFfn)> = Vec::new();
                for id in &union_plan.fill {
                    let host = self.host_expert(id.layer as usize, id.expert as usize)?;
                    pairs.push((*id, host));
                }
                if !pairs.is_empty() {
                    let nvme_side = nvme_side.as_ref();
                    let mut nvme = |id: ExpertId, dev_off: usize, slot_bytes: usize| {
                        self.try_nvme_expert_fill(id, dev_off, slot_bytes, nvme_side, resident)
                    };
                    resident.bulk_fill_experts(
                        &pairs,
                        Some(metrics),
                        None,
                        false,
                        Some(&mut nvme),
                    )?;
                }
                for (id, host) in &pairs {
                    lru.admit(*id, host)?;
                }
            }
            lru.plan(&ids)
        } else {
            MissPlan {
                hits: ids.clone(),
                fill: Vec::new(),
                cpu: Vec::new(),
            }
        };
        hip.copy_h2d_f32(buf.attn, &vec![0.0f32; dim])?;
        let mut cpu_jobs: Vec<(ExpertId, LayerFfn, f32)> = Vec::new();
        let mut gpu_jobs: Vec<(ExpertId, LayerFfn, f32)> = Vec::new();
        for &(e, w) in &routed {
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            };
            let host = self.host_expert(layer_idx, e)?;
            let ffn = if let Some(lru) = &self.expert_lru {
                lru.materialize(id, &host)?
            } else {
                host
            };
            if plan.cpu.iter().any(|&c| c == id) {
                cpu_jobs.push((id, ffn, w));
            } else {
                gpu_jobs.push((id, ffn, w));
            }
        }
        // GEM micro-batch: run CPU expert MLPs in parallel with the GPU expert loop.
        if !cpu_jobs.is_empty() && !gpu_jobs.is_empty() {
            metrics.micro_batch_plans.fetch_add(1, Ordering::Relaxed);
        }
        let cpu_outs: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = cpu_jobs
                .iter()
                .map(|(id, ffn, w)| {
                    let x = &x_norm;
                    scope.spawn(move || {
                        let mut scratch = vec![0.0f32; dim];
                        // NUMA bind inside thread for this expert.
                        if let Some(pool) = &self.expert_host {
                            let node = pool.preferred_numa_node(*id);
                            pool.numa()
                                .run_on_node(node, || {
                                    expert_mlp(
                                        x,
                                        ffn,
                                        cfg.ff_dim,
                                        dim,
                                        &mut scratch,
                                        cfg.moe_relu_sqr,
                                    )
                                })
                                .expect("expert_mlp");
                        } else {
                            expert_mlp(x, ffn, cfg.ff_dim, dim, &mut scratch, cfg.moe_relu_sqr)
                                .expect("expert_mlp");
                        }
                        for v in &mut scratch {
                            *v *= *w;
                        }
                        scratch
                    })
                })
                .collect();
            for (id, ffn, w) in &gpu_jobs {
                let expert = resident.bind_expert(*id, ffn, Some(metrics))?;
                self.device_expert_gemm(
                    hip,
                    resident,
                    buf,
                    buf.normed,
                    1,
                    expert.w_gate(),
                    expert.w_up(),
                    expert.w_down(),
                    cfg.ff_dim,
                )?;
                drop(expert);
                hip.scale_f32(buf.proj, dim, *w)?;
                hip.residual_add(buf.attn, buf.proj, dim as i32)?;
            }
            Ok::<Vec<Vec<f32>>, AllpacaError>(
                handles
                    .into_iter()
                    .map(|h| h.join().expect("cpu expert thread"))
                    .collect(),
            )
        })?;
        for scratch in cpu_outs {
            hip.copy_h2d_f32(buf.proj, &scratch)?;
            hip.residual_add(buf.attn, buf.proj, dim as i32)?;
        }
        if off.w_shared_up.n > 0 && off.w_shared_down.n > 0 {
            let sff = if cfg.shared_ff_dim > 0 {
                cfg.shared_ff_dim
            } else {
                off.w_shared_up.n / dim.max(1)
            };
            self.device_expert_gemm(
                hip,
                resident,
                buf,
                buf.normed,
                1,
                WeightRef::f32(0, 0),
                off.w_shared_up,
                off.w_shared_down,
                sff,
            )?;
            hip.residual_add(buf.attn, buf.proj, dim as i32)?;
        } else if !host_layer.shared_expert.w_up.is_empty() {
            let sff = if cfg.shared_ff_dim > 0 {
                cfg.shared_ff_dim
            } else {
                host_layer.shared_expert.w_up.len() / dim.max(1)
            };
            let mut scratch = vec![0.0f32; dim];
            expert_mlp(
                &x_norm,
                &host_layer.shared_expert,
                sff,
                dim,
                &mut scratch,
                cfg.moe_relu_sqr,
            )?;
            hip.copy_h2d_f32(buf.proj, &scratch)?;
            hip.residual_add(buf.attn, buf.proj, dim as i32)?;
        }
        hip.copy_d2d_f32(buf.proj, buf.attn, dim)?;
        record_elementwise(Some(metrics));
        Ok(())
    }

    /// Prefill MoE: one residual+RMSNorm, one D2H of all rows, then one expert
    /// GEMM per unique routed expert with `M = tokens that selected it`. Shared
    /// expert is `M = n_tok`. Falls back to per-token [`Self::device_moe`] when
    /// `n_tok <= 1` or any routed expert is a CPU miss.
    #[allow(clippy::too_many_arguments)]
    fn device_moe_rows(
        &self,
        layer_idx: usize,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        n_tok: usize,
        metrics: &Metrics,
        request_id: u64,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        if n_tok == 0 {
            return Ok(());
        }
        if n_tok == 1 {
            let mut side = DeviceDecodeSide::new(request_id);
            return self.device_moe(
                layer_idx,
                hip,
                resident,
                buf,
                metrics,
                true,
                None,
                Some(&mut side),
            );
        }
        self.prefetch_moe_layers(layer_idx)?;
        self.warm_moe_slots(layer_idx, resident, Some(metrics))?;
        let host_layer = self.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MoE missing host layer {layer_idx}"
            )))
        })?;
        let off = resident.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MoE missing device layer {layer_idx}"
            )))
        })?;
        if off.ffn_norm.n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "MoE layer missing packed ffn_norm".into(),
            )));
        }
        hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
        rmsnorm_device_rows(
            hip,
            buf.x,
            resident.f32_or_dequant(hip, off.ffn_norm)?,
            buf.normed,
            n_tok,
            dim,
            cfg.eps,
        )?;
        metrics.residual_gpu_ops.fetch_add(1, Ordering::Relaxed);
        record_elementwise(Some(metrics));
        let mut x_norm = vec![0.0f32; n_tok * dim];
        hip.copy_d2h_f32(buf.normed, &mut x_norm)?;
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tok);
        let mut union: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for t in 0..n_tok {
            let routed = topk_route(
                &x_norm[t * dim..(t + 1) * dim],
                &host_layer.w_router,
                n_exp,
                n_act,
                cfg,
            );
            for &(e, _) in &routed {
                union.insert(e);
            }
            routes.push(routed);
        }
        let ids: Vec<ExpertId> = union
            .iter()
            .map(|&e| ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            })
            .collect();
        let plan = if let Some(lru) = &self.expert_lru {
            let union_plan = lru.plan(&ids);
            lru.note_plan(&union_plan);
            if !union_plan.fill.is_empty() {
                let mut pairs: Vec<(ExpertId, LayerFfn)> = Vec::new();
                for id in &union_plan.fill {
                    let host = self.host_expert(id.layer as usize, id.expert as usize)?;
                    pairs.push((*id, host));
                }
                if !pairs.is_empty() {
                    resident.bulk_fill_experts(&pairs, Some(metrics), None, false, None)?;
                }
                for (id, host) in &pairs {
                    lru.admit(*id, host)?;
                }
            }
            lru.plan(&ids)
        } else {
            MissPlan {
                hits: ids.clone(),
                fill: Vec::new(),
                cpu: Vec::new(),
            }
        };
        hip.copy_h2d_f32(buf.attn, &vec![0.0f32; n_tok * dim])?;
        let mut groups: std::collections::BTreeMap<usize, Vec<(usize, f32)>> =
            std::collections::BTreeMap::new();
        for (t, routed) in routes.iter().enumerate() {
            for &(e, w) in routed {
                groups.entry(e).or_default().push((t, w));
            }
        }
        for (e, group) in &groups {
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: *e as u32,
            };
            let host = self.host_expert(layer_idx, *e)?;
            let ffn = if let Some(lru) = &self.expert_lru {
                lru.materialize(id, &host)?
            } else {
                host
            };
            if plan.cpu.iter().any(|&c| c == id) {
                for &(t, w) in group {
                    let mut scratch = vec![0.0f32; dim];
                    self.run_expert_mlp_numa(
                        id,
                        &x_norm[t * dim..(t + 1) * dim],
                        &ffn,
                        cfg.ff_dim,
                        dim,
                        &mut scratch,
                        None,
                    )?;
                    for v in &mut scratch {
                        *v *= w;
                    }
                    hip.copy_h2d_f32(buf.proj_row(t, dim), &scratch)?;
                    hip.residual_add(buf.attn_row(t, dim), buf.proj_row(t, dim), dim as i32)?;
                }
                continue;
            }
            let expert = resident.bind_expert(id, &ffn, Some(metrics))?;
            let m = group.len() as i32;
            for (i, &(t, _)) in group.iter().enumerate() {
                hip.copy_d2d_f32(buf.q_row(i, dim), buf.normed_row(t, dim), dim)?;
            }
            self.device_expert_gemm(
                hip,
                resident,
                buf,
                buf.q,
                m,
                expert.w_gate(),
                expert.w_up(),
                expert.w_down(),
                cfg.ff_dim,
            )?;
            drop(expert);
            for (i, &(t, w)) in group.iter().enumerate() {
                let src = buf.proj_row(i, dim);
                if (w - 1.0).abs() > f32::EPSILON {
                    hip.scale_f32(src, dim, w)?;
                }
                hip.residual_add(buf.attn_row(t, dim), src, dim as i32)?;
            }
        }
        if off.w_shared_up.n > 0 && off.w_shared_down.n > 0 {
            let sff = if cfg.shared_ff_dim > 0 {
                cfg.shared_ff_dim
            } else {
                off.w_shared_up.n / dim.max(1)
            };
            self.device_expert_gemm(
                hip,
                resident,
                buf,
                buf.normed,
                n_tok as i32,
                WeightRef::f32(0, 0),
                off.w_shared_up,
                off.w_shared_down,
                sff,
            )?;
            hip.residual_add(buf.attn, buf.proj, (n_tok * dim) as i32)?;
        } else if !host_layer.shared_expert.w_up.is_empty() {
            let sff = if cfg.shared_ff_dim > 0 {
                cfg.shared_ff_dim
            } else {
                host_layer.shared_expert.w_up.len() / dim.max(1)
            };
            for t in 0..n_tok {
                let mut scratch = vec![0.0f32; dim];
                expert_mlp(
                    &x_norm[t * dim..(t + 1) * dim],
                    &host_layer.shared_expert,
                    sff,
                    dim,
                    &mut scratch,
                    cfg.moe_relu_sqr,
                )?;
                hip.copy_h2d_f32(buf.proj_row(t, dim), &scratch)?;
                hip.residual_add(buf.attn_row(t, dim), buf.proj_row(t, dim), dim as i32)?;
            }
        }
        hip.copy_d2d_f32(buf.proj, buf.attn, n_tok * dim)?;
        record_elementwise(Some(metrics));
        Ok(())
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        if let Some(reason) = report.reject_reason {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(reason)));
        }
        let cfg = ModelConfig::from_gguf(file)?;
        Self::from_gguf_cfg(file, cfg, &report)
    }

    /// Load trailing NextN/MTP blocks as a standalone draft model (shared embed/lm_head).
    ///
    /// Seed uses `eh_proj(concat(RMSNorm(tok), RMSNorm(h_nextn)))` with `h_nextn`
    /// handed off from the target `output_norm` hidden (and updated after each MTP step).
    pub fn from_gguf_mtp(file: &GgufFile) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        if let Some(reason) = report.reject_reason {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(reason)));
        }
        let trunk = ModelConfig::from_gguf(file)?;
        if trunk.nextn_predict_layers == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "GGUF has no nextn_predict_layers for MTP draft".into(),
            )));
        }
        let mut cfg = trunk.clone();
        cfg.gguf_layer_base = trunk.n_layers;
        cfg.n_layers = trunk.nextn_predict_layers;
        cfg.nextn_predict_layers = 0;
        cfg.recurrent = false;
        let base = cfg.gguf_layer_base;
        if file
            .tensor(&format!("blk.{base}.nextn.eh_proj.weight"))
            .is_none()
            && file.tensor(&format!("blk.{base}.attn_q.weight")).is_none()
        {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MTP blk.{base} missing attn_q / nextn.eh_proj"
            ))));
        }
        let mut model = Self::from_gguf_cfg(file, cfg, &report)?;
        // Dense MTP heads (Qwen3.5/3.8) ship `ffn_gate/up/down`, not MoE exps.
        // Clear trunk expert count so draft decode stays on the dense device path.
        let base = model.config.gguf_layer_base;
        let dense_mtp = file
            .tensor(&format!("blk.{base}.ffn_gate.weight"))
            .is_some()
            && file
                .tensor(&format!("blk.{base}.ffn_gate_inp.weight"))
                .is_none()
            && file
                .tensor(&format!("blk.{base}.ffn_gate_exps.weight"))
                .is_none();
        if dense_mtp {
            model.config.n_experts = 0;
            model.config.n_active = 0;
        }
        // Prefer MTP shared head norm when present (llama.cpp graph_mtp).
        if let Ok(n) = file.tensor_f32(&format!("blk.{base}.nextn.shared_head_norm.weight")) {
            if model.host_weights_ready() {
                model.norm = n;
            }
        }
        if model.host_weights_ready()
            && file
                .tensor(&format!("blk.{base}.nextn.eh_proj.weight"))
                .is_some()
        {
            model.mtp_nextn = Some(MtpNextnWeights {
                eh_proj: file.tensor_f32(&format!("blk.{base}.nextn.eh_proj.weight"))?,
                enorm: file.tensor_f32(&format!("blk.{base}.nextn.enorm.weight"))?,
                hnorm: file.tensor_f32(&format!("blk.{base}.nextn.hnorm.weight"))?,
            });
        }
        model.name = format!("{}-mtp", model.name);
        Ok(model)
    }

    fn from_gguf_cfg(
        file: &GgufFile,
        cfg: ModelConfig,
        report: &GgufCompatReport,
    ) -> Result<Self, AllpacaError> {
        let name = file.meta_str("general.name").unwrap_or("gguf").to_string();
        for i in 0..cfg.n_layers {
            let i = cfg.gguf_blk(i);
            let info = report.layers.get(i).cloned().unwrap_or(HybridLayerInfo {
                index: i,
                block: HybridLayerKind::Attention,
                moe: cfg.n_experts > 0,
            });
            match info.block {
                HybridLayerKind::Ssm => {
                    if file.tensor(&format!("blk.{i}.ssm_in.weight")).is_none()
                        && file.tensor(&format!("blk.{i}.ssm_conv1d.weight")).is_none()
                        && file.tensor(&format!("blk.{i}.ssm_out.weight")).is_none()
                    {
                        // Dense MTP Attention blocks sometimes sit past an SSM trunk.
                        if file.tensor(&format!("blk.{i}.attn_q.weight")).is_none()
                            && file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_none()
                        {
                            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "missing SSM tensors on blk.{i}"
                            ))));
                        }
                    }
                    if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some() {
                        require_gguf_tensor(file, &format!("blk.{i}.ffn_up.weight"))?;
                        require_gguf_tensor(file, &format!("blk.{i}.ffn_down.weight"))?;
                    }
                }
                HybridLayerKind::Attention => {
                    require_gguf_tensor(file, &format!("blk.{i}.attn_norm.weight"))?;
                    if gguf_blk_has_mla(file, i) {
                        for suffix in [
                            "attn_q_a.weight",
                            "attn_q_b.weight",
                            "attn_kv.weight",
                            "attn_output_a.weight",
                            "attn_output_b.weight",
                        ] {
                            require_gguf_tensor(file, &format!("blk.{i}.{suffix}"))?;
                        }
                    } else {
                        require_gguf_tensor(file, &format!("blk.{i}.attn_output.weight"))?;
                        if file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_none() {
                            for suffix in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
                                require_gguf_tensor(file, &format!("blk.{i}.{suffix}"))?;
                            }
                        }
                    }
                    let has_dense_ffn = file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some();
                    let has_moe = file
                        .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                        .is_some()
                        || file
                            .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                            .is_some();
                    if has_dense_ffn {
                        for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                            require_gguf_tensor(file, &format!("blk.{i}.{suffix}"))?;
                        }
                    }
                    if has_dense_ffn || has_moe {
                        if file.tensor(&format!("blk.{i}.ffn_norm.weight")).is_none()
                            && file
                                .tensor(&format!("blk.{i}.post_attention_norm.weight"))
                                .is_none()
                            && file.tensor(&format!("blk.{i}.attn_norm.weight")).is_none()
                        {
                            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "missing ffn_norm/post_attention_norm/attn_norm on blk.{i}"
                            ))));
                        }
                    }
                    // else: attention-only block (Nemotron Lightning) — no FFN tensors.
                }
                HybridLayerKind::DenseFfn | HybridLayerKind::Moe => {
                    if file.tensor(&format!("blk.{i}.ffn_norm.weight")).is_none()
                        && file
                            .tensor(&format!("blk.{i}.post_attention_norm.weight"))
                            .is_none()
                        && file.tensor(&format!("blk.{i}.attn_norm.weight")).is_none()
                    {
                        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "missing ffn_norm/post_attention_norm/attn_norm on blk.{i}"
                        ))));
                    }
                    if info.moe || cfg.n_experts > 0 {
                        let has_stacked_up = file
                            .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                            .is_some();
                        let has_stacked_gate = file
                            .tensor(&format!("blk.{i}.ffn_gate_exps.weight"))
                            .is_some();
                        if has_stacked_up || has_stacked_gate {
                            require_gguf_tensor(file, &format!("blk.{i}.ffn_gate_inp.weight"))?;
                            require_gguf_tensor(file, &format!("blk.{i}.ffn_up_exps.weight"))?;
                            require_gguf_tensor(file, &format!("blk.{i}.ffn_down_exps.weight"))?;
                        } else {
                            for e in 0..cfg.n_experts.max(1) {
                                require_gguf_tensor(file, &format!("blk.{i}.ffn_up.{e}.weight"))?;
                                require_gguf_tensor(file, &format!("blk.{i}.ffn_down.{e}.weight"))?;
                            }
                        }
                    } else {
                        for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                            require_gguf_tensor(file, &format!("blk.{i}.{suffix}"))?;
                        }
                    }
                }
            }
        }
        if file.tensor("token_embd.weight").is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "missing tensor token_embd.weight".into(),
            )));
        }
        if file.tensor("output_norm.weight").is_none()
            && (0..cfg.n_layers).all(|local| {
                file.tensor(&format!(
                    "blk.{}.nextn.shared_head_norm.weight",
                    cfg.gguf_blk(local)
                ))
                .is_none()
            })
        {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "missing tensor output_norm.weight".into(),
            )));
        }
        let host_bytes = estimated_gguf_f32_bytes(file, &cfg);
        if host_bytes > HOST_F32_LIMIT {
            let mut model = Self::skeleton_hybrid(cfg, name, report, Some(file));
            attach_dsv4(&mut model, file)?;
            return Ok(model);
        }
        // Continue with original host-materialize body — call into remaining logic
        // by temporarily reconstructing what from_gguf did after validation.
        Self::from_gguf_host_materialize(file, cfg, report, name)
    }

    fn from_gguf_host_materialize(
        file: &GgufFile,
        cfg: ModelConfig,
        report: &GgufCompatReport,
        name: String,
    ) -> Result<Self, AllpacaError> {
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let i = cfg.gguf_blk(i);
            let info = report.layers.get(i).cloned().unwrap_or(HybridLayerInfo {
                index: i,
                block: HybridLayerKind::Attention,
                moe: cfg.n_experts > 0,
            });
            // Compat may mark MTP blocks as Ssm if neighboring heuristics misfire; prefer Attention
            // when Q projections exist (Qwen3.5 NextN is dense attn + FFN).
            let block = if file.tensor(&format!("blk.{i}.attn_q.weight")).is_some()
                || file.tensor(&format!("blk.{i}.attn_q_a.weight")).is_some()
            {
                HybridLayerKind::Attention
            } else {
                info.block
            };
            let layer = match block {
                HybridLayerKind::Ssm => {
                    let attn_norm = if file.tensor(&format!("blk.{i}.attn_norm.weight")).is_some() {
                        file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?
                    } else {
                        ones(cfg.dim)
                    };
                    let (w_gate, w_up, w_down, ffn_norm) =
                        if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some() {
                            (
                                file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                                file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                                file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                                gguf_ffn_norm(file, i).unwrap_or_else(|_| ones(cfg.dim)),
                            )
                        } else {
                            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                        };
                    LayerWeights {
                        block: HybridLayerKind::Ssm,
                        ssm: if file.architecture() == Some("qwen35") {
                            None
                        } else {
                            Some(load_gguf_ssm_layer(file, i, cfg.dim, true)?)
                        },
                        gdn: if file.architecture() == Some("qwen35") {
                            load_gguf_gdn_host(file, i).ok()
                        } else {
                            None
                        },
                        mla: None,
                        attn_norm,
                        q_norm: Vec::new(),
                        k_norm: Vec::new(),
                        wq: Vec::new(),
                        wk: Vec::new(),
                        wv: Vec::new(),
                        wo: Vec::new(),
                        ffn_norm,
                        w_gate,
                        w_up,
                        w_down,
                        w_router: Vec::new(),
                        experts: Vec::new(),
                        shared_expert: LayerFfn::default(),
                        expert_bias: Vec::new(),
                        attn_out_gate: Vec::new(),
                        attn_post_norm: Vec::new(),
                        ffn_post_norm: Vec::new(),
                    }
                }
                HybridLayerKind::DenseFfn | HybridLayerKind::Moe => {
                    let (w_gate, w_up, w_down, w_router, experts) = if info.moe || cfg.n_experts > 0
                    {
                        let n_exp = cfg.n_experts.max(1);
                        let (w_router, experts) = load_gguf_moe_layer(file, i, n_exp)?;
                        (Vec::new(), Vec::new(), Vec::new(), w_router, experts)
                    } else {
                        (
                            file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                            file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                            file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                            Vec::new(),
                            Vec::new(),
                        )
                    };
                    LayerWeights {
                        block: info.block,
                        ssm: None,
                        gdn: None,
                        mla: None,
                        attn_norm: ones(cfg.dim),
                        q_norm: Vec::new(),
                        k_norm: Vec::new(),
                        wq: Vec::new(),
                        wk: Vec::new(),
                        wv: Vec::new(),
                        wo: Vec::new(),
                        ffn_norm: gguf_ffn_norm(file, i)?,
                        w_gate,
                        w_up,
                        w_down,
                        w_router,
                        experts,
                        shared_expert: load_gguf_shared_expert(file, i)?,
                        expert_bias: load_gguf_expert_bias(file, i)?,
                        attn_out_gate: Vec::new(),
                        attn_post_norm: Vec::new(),
                        ffn_post_norm: Vec::new(),
                    }
                }
                HybridLayerKind::Attention => {
                    let has_moe = cfg.n_experts > 0
                        && (file
                            .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                            .is_some()
                            || file
                                .tensor(&format!("blk.{i}.ffn_up_exps.weight"))
                                .is_some());
                    let (w_gate, w_up, w_down, w_router, experts) = if has_moe {
                        let (w_router, experts) = load_gguf_moe_layer(file, i, cfg.n_experts)?;
                        (Vec::new(), Vec::new(), Vec::new(), w_router, experts)
                    } else if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some() {
                        (
                            file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                            file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                            file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                            Vec::new(),
                            Vec::new(),
                        )
                    } else {
                        // Attention-only hybrid block.
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
                    };
                    let (wq, wk, wv) = if gguf_blk_has_mla(file, i)
                        || file.tensor(&format!("blk.{i}.attn_q_a.weight")).is_some()
                    {
                        // MLA: leave dense QKV empty; weights live in `mla` / dsv4.
                        (Vec::new(), Vec::new(), Vec::new())
                    } else if file.tensor(&format!("blk.{i}.attn_qkv.weight")).is_some() {
                        split_fused_qkv_host(
                            file.tensor_f32(&format!("blk.{i}.attn_qkv.weight"))?,
                            cfg.dim,
                            cfg.n_heads.saturating_mul(cfg.head_dim().max(1)),
                            cfg.n_kv_heads.saturating_mul(cfg.head_dim().max(1)),
                        )?
                    } else {
                        (
                            file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                            file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                            file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                        )
                    };
                    let mla = load_gguf_mla_host(file, i, &cfg)?;
                    let shared_expert = load_gguf_shared_expert(file, i)?;
                    let expert_bias = load_gguf_expert_bias(file, i)?;
                    let ffn_norm = if !w_gate.is_empty() || !experts.is_empty() {
                        gguf_ffn_norm(file, i)?
                    } else {
                        Vec::new()
                    };
                    let attn_out_gate = if file.tensor(&format!("blk.{i}.attn_gate.weight")).is_some()
                        && cfg.headwise_attn_gate
                    {
                        file.tensor_f32(&format!("blk.{i}.attn_gate.weight"))?
                    } else {
                        Vec::new()
                    };
                    let wo = if mla.is_some() || gguf_blk_has_mla(file, i) {
                        Vec::new()
                    } else {
                        file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?
                    };
                    LayerWeights {
                        block: HybridLayerKind::Attention,
                        ssm: None,
                        gdn: None,
                        mla,
                        attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                        q_norm: if file
                            .tensor(&format!("blk.{i}.attn_q_norm.weight"))
                            .is_some()
                        {
                            file.tensor_f32(&format!("blk.{i}.attn_q_norm.weight"))?
                        } else {
                            Vec::new()
                        },
                        k_norm: if file
                            .tensor(&format!("blk.{i}.attn_k_norm.weight"))
                            .is_some()
                        {
                            file.tensor_f32(&format!("blk.{i}.attn_k_norm.weight"))?
                        } else {
                            Vec::new()
                        },
                        wq,
                        wk,
                        wv,
                        wo,
                        ffn_norm,
                        w_gate,
                        w_up,
                        w_down,
                        w_router,
                        experts,
                        shared_expert,
                        expert_bias,
                        attn_out_gate,
                        attn_post_norm: if file
                            .tensor(&format!("blk.{i}.attn_post_norm.weight"))
                            .is_some()
                        {
                            file.tensor_f32(&format!("blk.{i}.attn_post_norm.weight"))?
                        } else {
                            Vec::new()
                        },
                        ffn_post_norm: if file
                            .tensor(&format!("blk.{i}.ffn_post_norm.weight"))
                            .is_some()
                        {
                            file.tensor_f32(&format!("blk.{i}.ffn_post_norm.weight"))?
                        } else {
                            Vec::new()
                        },
                    }
                }
            };
            layers.push(layer);
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        let early_exit = crate::early_exit::spec_from_gguf(file, 0.9, 0.05);
        let mut exit_heads = vec![None; cfg.n_layers];
        if early_exit.supported {
            for &layer in &early_exit.exit_layers {
                let blk = cfg.gguf_blk(layer);
                let name = format!("blk.{blk}.exit_head.weight");
                if file.tensor(&name).is_some() {
                    if let Ok(w) = file.tensor_f32(&name) {
                        if layer < exit_heads.len() {
                            exit_heads[layer] = Some(w);
                        }
                    }
                }
            }
        }
        let mla_cache = if layers.iter().any(|l| l.mla.is_some()) {
            Some(Arc::new(std::sync::Mutex::new(vec![
                (Vec::new(), Vec::new());
                layers.len()
            ])))
        } else {
            None
        };
        let mut model = Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit,
            exit_heads,
            mtp_nextn: None,
            dsv4: None,
            rpc: None,
            edge0: None,
            mla_cache,
        };
        attach_dsv4(&mut model, file)?;
        Ok(model)
    }

    #[allow(dead_code)]
    fn skeleton(cfg: ModelConfig, name: impl Into<String>) -> Self {
        Self::skeleton_hybrid(
            cfg,
            name,
            &GgufCompatReport {
                architecture: String::new(),
                n_layers: 0,
                ssm_layers: 0,
                attention_layers: 0,
                moe_layers: 0,
                layers: Vec::new(),
                ggml_type_ids: Vec::new(),
                reject_reason: None,
            },
            None,
        )
    }

    fn skeleton_hybrid(
        cfg: ModelConfig,
        name: impl Into<String>,
        report: &allpaca_gguf::GgufCompatReport,
        file: Option<&GgufFile>,
    ) -> Self {
        let layers = (0..cfg.n_layers)
            .map(|local| {
                let i = cfg.gguf_blk(local);
                let info = report.layers.get(i);
                let reported = info.map(|l| l.block).unwrap_or(HybridLayerKind::Attention);
                let block = if file
                    .map(|f| {
                        f.tensor(&format!("blk.{i}.attn_q.weight")).is_some()
                            || f.tensor(&format!("blk.{i}.attn_q_a.weight")).is_some()
                    })
                    .unwrap_or(false)
                {
                    HybridLayerKind::Attention
                } else {
                    reported
                };
                let mut layer = empty_layer();
                layer.block = block;
                if block == HybridLayerKind::Ssm {
                    if let Some(f) = file {
                        if f.architecture() == Some("qwen35") {
                            if let Ok(gdn) = load_gguf_gdn_host(f, i) {
                                layer.gdn = Some(gdn);
                            }
                        }
                        if layer.gdn.is_none() {
                            layer.ssm = load_gguf_ssm_layer(f, i, cfg.dim, false).ok();
                        }
                    }
                } else if block == HybridLayerKind::Attention {
                    if let Some(f) = file {
                        if f.tensor(&format!("blk.{i}.attn_q_norm.weight")).is_some() {
                            layer.q_norm = f
                                .tensor_f32(&format!("blk.{i}.attn_q_norm.weight"))
                                .unwrap_or_default();
                        }
                        if f.tensor(&format!("blk.{i}.attn_k_norm.weight")).is_some() {
                            layer.k_norm = f
                                .tensor_f32(&format!("blk.{i}.attn_k_norm.weight"))
                                .unwrap_or_default();
                        }
                        if !f
                            .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                            .is_some_and(|t| t.ggml_type == allpaca_gguf::GgmlType::BF16)
                        {
                            if let Ok(router) = f
                                .tensor_f32(&format!("blk.{i}.ffn_gate_inp.weight"))
                                .or_else(|_| f.tensor_f32(&format!("blk.{i}.ffn_router.weight")))
                            {
                                layer.w_router = router;
                            }
                        }
                        if let Ok(norm) = gguf_ffn_norm(f, i) {
                            layer.ffn_norm = norm;
                        }
                    }
                } else if matches!(block, HybridLayerKind::DenseFfn | HybridLayerKind::Moe)
                    || (cfg.n_experts > 0
                        && file
                            .map(|f| {
                                f.tensor(&format!("blk.{i}.ffn_gate_inp.weight")).is_some()
                                    || f.tensor(&format!("blk.{i}.ffn_up_exps.weight")).is_some()
                            })
                            .unwrap_or(false))
                {
                    // MoE routing runs on host even when experts stay device-resident.
                    // Packed BF16 routers stay mmap'd; HIP GEMV routes per token.
                    if let Some(f) = file {
                        if !f
                            .tensor(&format!("blk.{i}.ffn_gate_inp.weight"))
                            .is_some_and(|t| t.ggml_type == allpaca_gguf::GgmlType::BF16)
                        {
                            if let Ok(router) = f
                                .tensor_f32(&format!("blk.{i}.ffn_gate_inp.weight"))
                                .or_else(|_| f.tensor_f32(&format!("blk.{i}.ffn_router.weight")))
                            {
                                layer.w_router = router;
                            }
                        }
                        if let Ok(norm) = gguf_ffn_norm(f, i) {
                            layer.ffn_norm = norm;
                        }
                    }
                }
                layer
            })
            .collect();
        Self {
            embed: Vec::new(),
            layers,
            norm: Vec::new(),
            lm_head: Vec::new(),
            name: name.into(),
            config: cfg,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            rpc: None,
            edge0: None,
            mla_cache: None,
        }
    }

    pub fn from_safetensors(
        file: &SafetensorsCatalog,
        hf: &HfConfig,
    ) -> Result<Self, AllpacaError> {
        if hf.is_moe() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "MoE HuggingFace SafeTensors is not supported (dense Llama/Qwen2/Mistral/Gemma only)"
                    .into(),
            )));
        }
        if !hf.hidden_act_is_silu() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "unsupported hidden_act {:?}",
                hf.hidden_act
            ))));
        }
        let n_heads = hf.num_attention_heads.max(1);
        let n_kv_heads = hf.num_key_value_heads.unwrap_or(n_heads).max(1);
        let ff_dim = hf.intermediate_size.unwrap_or(hf.hidden_size * 4);
        let cfg = ModelConfig {
            vocab_size: hf.vocab_size,
            dim: hf.hidden_size,
            n_layers: hf.num_hidden_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            n_experts: 0,
            n_active: 0,
            eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
            head_dim_explicit: hf.head_dim.unwrap_or(0),
            rope_dims: 0,
            rope_sections: [0; 4],
            recurrent: false,
            nextn_predict_layers: 0,
            gguf_layer_base: 0,
            router_kind: RouterKind::SoftmaxTopk,
            shared_experts: 0,
            n_group: 0,
            topk_group: 0,
            routed_scaling: 1.0,
            moe_norm_topk: true,
            edge0_tier: None,
            ffn_gelu: false,
            sliding_window: 0,
            swa_pattern: Vec::new(),
            rope_dims_swa: 0,
            rope_theta_swa: 0.0,
            headwise_attn_gate: false,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            scale_embed: false,
            leading_dense_blocks: 0,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            n_expert_groups: 1,
            n_group_used: 1,
            n_shared_experts: 0,
            shared_ff_dim: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            moe_sigmoid_gating: false,
            moe_relu_sqr: false,
            moe_sqrt_softplus: false,
            rope_offset: 0,
        };
        let embed = file.tensor_f32_key(allpaca_safetensors::WeightKey::Embed)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                block: HybridLayerKind::Attention,
                ssm: None,
                gdn: None,
                mla: None,
                attn_norm: file.tensor_f32_key(allpaca_safetensors::WeightKey::AttnNorm(i))?,
                q_norm: file.tensor_f32_key_optional(allpaca_safetensors::WeightKey::QNorm(i))?,
                k_norm: file.tensor_f32_key_optional(allpaca_safetensors::WeightKey::KNorm(i))?,
                wq: file.tensor_f32_key(allpaca_safetensors::WeightKey::Wq(i))?,
                wk: file.tensor_f32_key(allpaca_safetensors::WeightKey::Wk(i))?,
                wv: file.tensor_f32_key(allpaca_safetensors::WeightKey::Wv(i))?,
                wo: file.tensor_f32_key(allpaca_safetensors::WeightKey::Wo(i))?,
                ffn_norm: file.tensor_f32_key(allpaca_safetensors::WeightKey::FfnNorm(i))?,
                w_gate: file.tensor_f32_key(allpaca_safetensors::WeightKey::Gate(i))?,
                w_up: file.tensor_f32_key(allpaca_safetensors::WeightKey::Up(i))?,
                w_down: file.tensor_f32_key(allpaca_safetensors::WeightKey::Down(i))?,
                w_router: Vec::new(),
                experts: Vec::new(),
                shared_expert: LayerFfn::default(),
                expert_bias: Vec::new(),
                attn_out_gate: Vec::new(),
                attn_post_norm: Vec::new(),
                ffn_post_norm: Vec::new(),
            });
        }
        let norm = file.tensor_f32_key(allpaca_safetensors::WeightKey::OutputNorm)?;
        let lm_head = match file.tensor_f32_key(allpaca_safetensors::WeightKey::LmHead) {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        let name = hf
            .architectures
            .first()
            .cloned()
            .or_else(|| hf.model_type.clone())
            .unwrap_or_else(|| "safetensors".into());
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            rpc: None,
            edge0: None,
            mla_cache: None,
        })
    }

    pub fn from_onnx(file: &OnnxFile, hf: &HfConfig) -> Result<Self, AllpacaError> {
        let n_heads = hf.num_attention_heads.max(1);
        let n_kv_heads = hf.num_key_value_heads.unwrap_or(n_heads).max(1);
        let ff_dim = hf.intermediate_size.unwrap_or(hf.hidden_size * 4);
        let cfg = ModelConfig {
            vocab_size: hf.vocab_size,
            dim: hf.hidden_size,
            n_layers: hf.num_hidden_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            n_experts: 0,
            n_active: 0,
            eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
            head_dim_explicit: 0,
            rope_dims: 0,
            rope_sections: [0; 4],
            recurrent: false,
            nextn_predict_layers: 0,
            gguf_layer_base: 0,
            router_kind: RouterKind::SoftmaxTopk,
            shared_experts: 0,
            n_group: 0,
            topk_group: 0,
            routed_scaling: 1.0,
            moe_norm_topk: true,
            edge0_tier: None,
            ffn_gelu: false,
            sliding_window: 0,
            swa_pattern: Vec::new(),
            rope_dims_swa: 0,
            rope_theta_swa: 0.0,
            headwise_attn_gate: false,
            attn_logit_softcap: 0.0,
            final_logit_softcap: 0.0,
            scale_embed: false,
            leading_dense_blocks: 0,
            q_lora_rank: 0,
            kv_lora_rank: 0,
            n_expert_groups: 1,
            n_group_used: 1,
            n_shared_experts: 0,
            shared_ff_dim: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            moe_sigmoid_gating: false,
            moe_relu_sqr: false,
            moe_sqrt_softplus: false,
            rope_offset: 0,
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                block: HybridLayerKind::Attention,
                ssm: None,
                gdn: None,
                mla: None,
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                w_router: Vec::new(),
                experts: Vec::new(),
                shared_expert: LayerFfn::default(),
                expert_bias: Vec::new(),
                attn_out_gate: Vec::new(),
                attn_post_norm: Vec::new(),
                ffn_post_norm: Vec::new(),
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name: "onnx".into(),
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            rpc: None,
            edge0: None,
            mla_cache: None,
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        let n: usize = self.weight_slices().iter().map(|t| t.len()).sum();
        (n * 4) as u64
    }

    fn weight_slices(&self) -> Vec<&[f32]> {
        let mut v = Vec::with_capacity(2 + self.layers.len() * TENSORS_PER_LAYER);
        v.push(self.embed.as_slice());
        for layer in &self.layers {
            v.push(layer.attn_norm.as_slice());
            v.push(layer.wq.as_slice());
            v.push(layer.wk.as_slice());
            v.push(layer.wv.as_slice());
            v.push(layer.wo.as_slice());
            v.push(layer.ffn_norm.as_slice());
            v.push(layer.w_gate.as_slice());
            v.push(layer.w_up.as_slice());
            v.push(layer.w_down.as_slice());
        }
        v.push(self.norm.as_slice());
        v.push(self.lm_head.as_slice());
        v
    }

    /// Decode one token at `pos`, writing K/V into paged tiers. Logits written to `logits`.
    ///
    /// When `soft_x` is `Some`, it replaces the token embedding (vision soft prompt).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        prefill: bool,
        mut recurrent: Option<&mut RecurrentState>,
        micro_batch: Option<&MicroBatchExpertUnion>,
        mut side: Option<&mut DeviceDecodeSide<'_>>,
        soft_x: Option<&[f32]>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        if soft_x.is_none() && token as usize >= cfg.vocab_size {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "token {token} out of vocab"
            ))));
        }
        if let Some(x0) = soft_x {
            if x0.len() != dim {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "soft embed len {} != dim {dim}",
                    x0.len()
                ))));
            }
        }
        if self.dsv4.is_some() {
            return crate::deepseek4::forward_dsv4_token(
                self, token, pos, page_ids, kv, logits, metrics, None, hip, prefill,
            );
        }
        // Prefer on-device decode whenever resident weights exist — including hybrid
        // models that carry recurrent SSM/DeltaNet state (host stub inside device path).
        // RPC layer-split keeps activations on the host and bounces at device boundaries.
        if self.rpc.as_ref().is_some_and(|r| r.plan_has_remote()) && !self.host_weights_ready() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "RPC layer-split requires host F32 weights in v1".into(),
            )));
        }
        if let (Some(hip), Some(resident)) = (hip, resident) {
            if !self.rpc.as_ref().is_some_and(|r| r.plan_has_remote()) {
                match self.forward_token_device(
                    token,
                    pos,
                    page_ids,
                    kv,
                    metrics,
                    logits,
                    hip,
                    resident,
                    prefill,
                    micro_batch,
                    side.as_deref_mut(),
                    recurrent.as_deref_mut(),
                    soft_x,
                ) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        if !self.host_weights_ready() {
                            return Err(e);
                        }
                        tracing::warn!(pos, token, error = %e, "HIP on-device decode failed; host fallback");
                    }
                }
            }
        }
        if !self.host_weights_ready() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "large GGUF has no host F32 weights; HIP decode is required".into(),
            )));
        }
        let mut x = if let Some(x0) = soft_x {
            x0.to_vec()
        } else {
            self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec()
        };
        if cfg.scale_embed {
            let s = (dim as f32).sqrt();
            for v in &mut x {
                *v *= s;
            }
        }
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(rpc) = &self.rpc {
                if rpc.is_remote(layer_idx) {
                    rpc.run_layer(layer_idx, pos, &mut x)?;
                    continue;
                }
            }
            if matches!(
                layer.block,
                HybridLayerKind::Ssm | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
            ) {
                if layer.block == HybridLayerKind::Ssm {
                    let ssm = layer.ssm.as_ref().ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "SSM layer {layer_idx} missing production Mamba-2 weights"
                        )))
                    })?;
                    let norm_w = if layer.attn_norm.is_empty() {
                        ones(dim)
                    } else {
                        layer.attn_norm.clone()
                    };
                    rmsnorm_dispatch(
                        hip,
                        &x,
                        &norm_w,
                        None,
                        cfg.eps,
                        &mut scratch[..dim],
                        Some(metrics),
                    )?;
                    let state = recurrent.as_deref_mut().ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::ModelLoad(
                            "SSM layer requires recurrent state".into(),
                        ))
                    })?;
                    state.ensure_ssm(
                        layer_idx,
                        ssm.dims.state_floats(),
                        ssm.dims.conv_state_floats(),
                    );
                    mamba2_step(
                        &scratch[..dim],
                        &mut state.ssm_h[layer_idx],
                        &mut state.ssm_conv[layer_idx],
                        ssm,
                        cfg.eps,
                        &mut hidden,
                    )
                    .map_err(|e| {
                        AllpacaError::Fatal(FatalError::ModelLoad(format!(
                            "SSM layer {layer_idx}: {e}"
                        )))
                    })?;
                    for i in 0..dim {
                        x[i] += hidden[i];
                    }
                    if !layer.w_gate.is_empty() || !layer.experts.is_empty() {
                        self.apply_ffn(
                            layer_idx,
                            layer,
                            &mut x,
                            &mut scratch,
                            &mut hidden,
                            hip,
                            resident,
                            Some(metrics),
                            prefill,
                        )?;
                    }
                } else {
                    // DenseFfn / MoE-only block: FFN residual, no attention.
                    self.apply_ffn(
                        layer_idx,
                        layer,
                        &mut x,
                        &mut scratch,
                        &mut hidden,
                        hip,
                        resident,
                        Some(metrics),
                        prefill,
                    )?;
                }
                continue;
            }

            rmsnorm_dispatch(
                hip,
                &x,
                &layer.attn_norm,
                resident.map(|r| r.attn_norm(layer_idx)),
                cfg.eps,
                &mut scratch[..dim],
                Some(metrics),
            )?;
            if let Some(mla_w) = &layer.mla {
                if let Some(cache) = &self.mla_cache {
                    let mut g = cache.lock().unwrap_or_else(|e| e.into_inner());
                    if g.len() <= layer_idx {
                        g.resize(layer_idx + 1, (Vec::new(), Vec::new()));
                    }
                    let (ckv, krope) = &mut g[layer_idx];
                    crate::edge0::mla::mla_attend_cpu(
                        &scratch[..dim],
                        mla_w,
                        ckv,
                        krope,
                        cfg.rope_theta,
                        cfg.eps,
                        &mut hidden,
                    )?;
                } else {
                    let mut ckv = Vec::new();
                    let mut krope = Vec::new();
                    crate::edge0::mla::mla_attend_cpu(
                        &scratch[..dim],
                        mla_w,
                        &mut ckv,
                        &mut krope,
                        cfg.rope_theta,
                        cfg.eps,
                        &mut hidden,
                    )?;
                }
                for i in 0..dim {
                    x[i] += hidden[i];
                }
                self.apply_ffn(
                    layer_idx,
                    layer,
                    &mut x,
                    &mut scratch,
                    &mut hidden,
                    hip,
                    resident,
                    Some(metrics),
                    prefill,
                )?;
                continue;
            }
            gemm_nt_dispatch(
                hip,
                &scratch[..dim],
                &layer.wq,
                resident.map(|r| r.wq(layer_idx)),
                &mut q,
                1,
                dim,
                dim,
            )?;
            self.apply_edge0_lora(layer_idx, "q_proj", &scratch[..dim], &mut q);
            gemm_nt_dispatch(
                hip,
                &scratch[..dim],
                &layer.wk,
                resident.map(|r| r.wk(layer_idx)),
                &mut k,
                1,
                kv_dim,
                dim,
            )?;
            self.apply_edge0_lora(layer_idx, "k_proj", &scratch[..dim], &mut k);
            gemm_nt_dispatch(
                hip,
                &scratch[..dim],
                &layer.wv,
                resident.map(|r| r.wv(layer_idx)),
                &mut v,
                1,
                kv_dim,
                dim,
            )?;
            self.apply_edge0_lora(layer_idx, "v_proj", &scratch[..dim], &mut v);
            let n_rot = cfg.rope_dims_for_layer(layer_idx);
            let theta = cfg.rope_theta_for_layer(layer_idx);
            if n_rot > 0 && n_rot < head_dim && !cfg.uses_imrope() {
                for h in 0..cfg.n_heads {
                    let slice = &mut q[h * head_dim..(h + 1) * head_dim];
                    rope_imrope_neox(slice, pos, theta, n_rot, [0; 4])?;
                }
                for h in 0..cfg.n_kv_heads {
                    let slice = &mut k[h * head_dim..(h + 1) * head_dim];
                    rope_imrope_neox(slice, pos, theta, n_rot, [0; 4])?;
                }
            } else {
                rope_dispatch(hip, &mut q, cfg.n_heads, head_dim, pos, theta, Some(metrics))?;
                rope_dispatch(
                    hip,
                    &mut k,
                    cfg.n_kv_heads,
                    head_dim,
                    pos,
                    theta,
                    Some(metrics),
                )?;
            }

            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, cfg.n_kv_heads, head_dim, &k, &v)?;
            kv.touch(page_id, 1.0);

            let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_kv = k_all.len() / kv_dim;
            if cfg.layer_is_swa(layer_idx) && cfg.sliding_window > 0 {
                let mut mask = vec![true; n_kv];
                let start = n_kv.saturating_sub(cfg.sliding_window);
                for t in 0..start {
                    mask[t] = false;
                }
                attention_with_mask_softcap(
                    &q,
                    &k_all,
                    &v_all,
                    &mut attn_out,
                    1,
                    n_kv,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    head_dim,
                    &mask,
                    cfg.attn_logit_softcap,
                )?;
            } else {
                attention_dispatch(
                    hip,
                    &q,
                    &k_all,
                    &v_all,
                    &mut attn_out,
                    1,
                    n_kv,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    head_dim,
                    true,
                    cfg.attn_logit_softcap,
                )?;
            }
            // Spark2_5 head-wise sigmoid attention gate (applied before wo).
            if cfg.headwise_attn_gate && !layer.attn_out_gate.is_empty() {
                apply_headwise_attn_gate(
                    &scratch[..dim],
                    &layer.attn_out_gate,
                    &mut attn_out,
                    cfg.n_heads,
                    head_dim,
                    dim,
                );
            }
            gemm_nt_dispatch(
                hip,
                &attn_out,
                &layer.wo,
                resident.map(|r| r.wo(layer_idx)),
                &mut hidden,
                1,
                dim,
                dim,
            )?;
            self.apply_edge0_lora(layer_idx, "o_proj", &attn_out, &mut hidden);
            if !layer.attn_post_norm.is_empty() {
                let tmp = hidden.clone();
                rmsnorm(&tmp, &layer.attn_post_norm, cfg.eps, &mut hidden)?;
            }
            for i in 0..dim {
                x[i] += hidden[i];
            }

            self.apply_ffn(
                layer_idx,
                layer,
                &mut x,
                &mut scratch,
                &mut hidden,
                hip,
                resident,
                Some(metrics),
                prefill,
            )?;
            // Self-speculative / exit-head early exit: provisional logits; skip
            // remaining layers when confidence + margin clear the manifest gate.
            if crate::early_exit::is_exit_layer(&self.early_exit, layer_idx)
                && layer_idx + 1 < self.layers.len()
            {
                if logits.len() < cfg.vocab_size {
                    return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                        "logits buffer smaller than vocab".into(),
                    )));
                }
                rmsnorm_dispatch(
                    hip,
                    &x,
                    &self.norm,
                    resident.map(|r| r.norm()),
                    cfg.eps,
                    &mut scratch[..dim],
                    Some(metrics),
                )?;
                let dedicated = self
                    .exit_heads
                    .get(layer_idx)
                    .and_then(|h| h.as_ref())
                    .filter(|w| w.len() >= cfg.vocab_size * dim);
                if let Some(head) = dedicated {
                    // Host GEMM through the dedicated exit head (GGUF F32 materialize).
                    allpaca_cpu::matmul_nt(
                        &scratch[..dim],
                        head,
                        &mut logits[..cfg.vocab_size],
                        1,
                        cfg.vocab_size,
                        dim,
                    )?;
                } else {
                    gemm_nt_dispatch(
                        hip,
                        &scratch[..dim],
                        &self.lm_head,
                        resident.map(|r| r.lm_head()),
                        &mut logits[..cfg.vocab_size],
                        1,
                        cfg.vocab_size,
                        dim,
                    )?;
                }
                let pred = crate::early_exit::prediction_from_logits(&logits[..cfg.vocab_size]);
                if crate::early_exit::accept_exit(&pred, &self.early_exit) {
                    metrics
                        .early_exit_accepts
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(());
                }
            }
        }

        rmsnorm_dispatch(
            hip,
            &x,
            &self.norm,
            resident.map(|r| r.norm()),
            cfg.eps,
            &mut scratch[..dim],
            Some(metrics),
        )?;
        if let Some(side) = side.as_mut() {
            if let Some(out) = side.last_hidden.as_mut() {
                out.clear();
                out.extend_from_slice(&scratch[..dim]);
            }
        }
        let write_logits = side.as_ref().is_none_or(|s| s.write_logits);
        if write_logits {
            if logits.len() < cfg.vocab_size {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "logits buffer smaller than vocab".into(),
                )));
            }
            gemm_nt_dispatch(
                hip,
                &scratch[..dim],
                &self.lm_head,
                resident.map(|r| r.lm_head()),
                &mut logits[..cfg.vocab_size],
                1,
                cfg.vocab_size,
                dim,
            )?;
            softcap_inplace(&mut logits[..cfg.vocab_size], cfg.final_logit_softcap);
        }
        if hip.is_some() {
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .target_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Prefill a token chunk. HIP runs one forward with M=N (hipBLASLt when N>8 on F32).
    /// Decode stays on [`forward_token`]. CPU loops per token. Last-chunk logits are
    /// written for the last position only when `write_logits` is set. Packed linear
    /// verify passes `row_logits` of length `tokens.len() * vocab` to keep every row.
    /// When `out_row_hiddens` is set it receives `n_tok * dim` `output_norm` rows
    /// (MTP `h_nextn` handoff after packed verify). `request_id` keys device GDN.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        write_logits: bool,
        mut row_logits: Option<&mut [f32]>,
        mut recurrent: Option<&mut RecurrentState>,
        mut out_row_hiddens: Option<&mut Vec<f32>>,
        request_id: u64,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if self.dsv4.is_some() && tokens.len() > 1 {
            return crate::deepseek4::forward_dsv4_tokens(
                self,
                tokens,
                start_pos,
                page_ids,
                kv,
                logits,
                metrics,
                hip,
                write_logits,
            );
        }
        if self.rpc.as_ref().is_some_and(|r| r.plan_has_remote()) {
            for (i, tok) in tokens.iter().enumerate() {
                let last = i + 1 == tokens.len();
                let mut side = DeviceDecodeSide {
                    last_hidden: if last {
                        out_row_hiddens.as_deref_mut()
                    } else {
                        None
                    },
                    ..DeviceDecodeSide::new(request_id)
                };
                self.forward_token(
                    *tok,
                    start_pos + i,
                    page_ids,
                    kv,
                    metrics,
                    logits,
                    hip,
                    resident,
                    write_logits && last,
                    recurrent.as_deref_mut(),
                    None,
                    Some(&mut side),
                    None,
                )?;
            }
            return Ok(());
        }
        if tokens.len() == 1 {
            let mut side = DeviceDecodeSide {
                last_hidden: out_row_hiddens.as_deref_mut(),
                ..DeviceDecodeSide::new(request_id)
            };
            self.forward_token(
                tokens[0],
                start_pos,
                page_ids,
                kv,
                metrics,
                logits,
                hip,
                resident,
                false,
                recurrent,
                None,
                Some(&mut side),
                None,
            )?;
            if let Some(rows) = row_logits {
                let vocab = self.config.vocab_size;
                if rows.len() >= vocab {
                    if let Some(hip) = hip {
                        hip.wait_logits_d2h(&mut rows[..vocab])?;
                    } else {
                        rows[..vocab].copy_from_slice(&logits[..vocab]);
                    }
                }
            }
            return Ok(());
        }
        for tok in tokens {
            if *tok as usize >= self.config.vocab_size {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                    "token {tok} out of vocab"
                ))));
            }
        }
        if self.supports_device_batched_prefill() {
            if let (Some(hip), Some(resident)) = (hip, resident) {
                match self.forward_tokens_device(
                    tokens,
                    start_pos,
                    page_ids,
                    kv,
                    metrics,
                    logits,
                    hip,
                    resident,
                    write_logits,
                    row_logits.as_deref_mut(),
                    recurrent.as_deref_mut(),
                    out_row_hiddens.as_deref_mut(),
                    request_id,
                ) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        if !self.host_weights_ready() {
                            return Err(e);
                        }
                        tracing::warn!(
                            start_pos,
                            n = tokens.len(),
                            error = %e,
                            "HIP batched prefill failed; per-token host fallback"
                        );
                    }
                }
            }
        }
        let vocab = self.config.vocab_size;
        let dim = self.config.dim;
        if let Some(out) = out_row_hiddens.as_mut() {
            out.clear();
            out.reserve(tokens.len() * dim);
        }
        for (i, tok) in tokens.iter().enumerate() {
            let mut last = Vec::new();
            let mut side = DeviceDecodeSide {
                last_hidden: if out_row_hiddens.is_some() {
                    Some(&mut last)
                } else {
                    None
                },
                ..DeviceDecodeSide::new(request_id)
            };
            self.forward_token(
                *tok,
                start_pos + i,
                page_ids,
                kv,
                metrics,
                logits,
                hip,
                resident,
                true,
                recurrent.as_deref_mut(),
                None,
                Some(&mut side),
                None,
            )?;
            if let Some(rows) = row_logits.as_mut() {
                let off = i * vocab;
                if rows.len() >= off + vocab {
                    if let Some(hip) = hip {
                        hip.wait_logits_d2h(&mut rows[off..off + vocab])?;
                    } else {
                        rows[off..off + vocab].copy_from_slice(&logits[..vocab]);
                    }
                }
            }
            if let Some(out) = out_row_hiddens.as_mut() {
                if last.len() == dim {
                    out.extend_from_slice(&last);
                } else {
                    out.extend(std::iter::repeat(0.0f32).take(dim));
                }
            }
        }
        Ok(())
    }

    fn decode_workspaces(cfg: &ModelConfig, resident: &ResidentModel) -> (usize, usize) {
        let dim = cfg.dim;
        let mut q_workspace = dim;
        let mut kv_workspace = cfg.n_kv_heads * cfg.head_dim().max(1);
        for layer in &resident.layers {
            if layer.wq.n > 0 && layer.wq.n % dim == 0 {
                q_workspace = q_workspace.max(layer.wq.n / dim);
            }
            if layer.wo.n > 0 && layer.wo.n % dim == 0 {
                q_workspace = q_workspace.max(layer.wo.n / dim);
            }
            if layer.w_qkv.n > 0 && layer.w_qkv.n % dim == 0 {
                q_workspace = q_workspace.max(layer.w_qkv.n / dim);
            }
            if layer.w_ssm_in.n > 0 && layer.w_ssm_in.n % dim == 0 {
                q_workspace = q_workspace.max(layer.w_ssm_in.n / dim);
            }
            if layer.w_ssm_out.n > 0 && layer.w_ssm_out.n % dim == 0 {
                q_workspace = q_workspace.max(layer.w_ssm_out.n / dim);
            }
            if layer.w_z.n > 0 && layer.w_z.n % dim == 0 {
                q_workspace = q_workspace.max(layer.w_z.n / dim);
            }
            if layer.wk.n > 0 && layer.wk.n % dim == 0 {
                kv_workspace = kv_workspace.max(layer.wk.n / dim);
            }
        }
        (q_workspace, kv_workspace)
    }

    fn layer_attn_layout(cfg: &ModelConfig, layer: &LayerOff) -> AttnLayout {
        attn_layout(cfg, layer.wq.n, layer.wk.n, layer.wo.n)
    }

    fn decode_row_view(buf: DeviceDecode, token: usize, dim: usize) -> DeviceDecode {
        DeviceDecode {
            x: buf.x_row(token, dim),
            normed: buf.normed_row(token, dim),
            q: buf.q,
            k: buf.k,
            v: buf.v,
            attn: buf.attn,
            proj: buf.proj_row(token, dim),
            gate: buf.gate,
            up: buf.up,
            logits: buf.logits,
            packed_k: buf.packed_k,
            packed_v: buf.packed_v,
            max_kv: buf.max_kv,
        }
    }

    fn layer_has_device_moe(layer: &LayerOff, resident: &ResidentModel, layer_idx: usize) -> bool {
        layer.w_router.n > 0
            || resident
                .expert_offs
                .get(layer_idx)
                .is_some_and(|e| !e.is_empty())
    }

    /// SSM / GDN / MoE-only / dense-FFN hybrid block (no attention Q).
    /// `buf` is a single-token workspace; `x`/`normed`/`proj` may be a row view.
    #[allow(clippy::too_many_arguments)]
    fn device_hybrid_no_attn_layer(
        &self,
        layer_idx: usize,
        host_layer: &LayerWeights,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        metrics: &Metrics,
        prefill: bool,
        micro_batch: Option<&MicroBatchExpertUnion>,
        mut side: Option<&mut DeviceDecodeSide<'_>>,
        mut recurrent: Option<&mut RecurrentState>,
    ) -> Result<bool, AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let layer = &resident.layers[layer_idx];
        if !(matches!(
            host_layer.block,
            HybridLayerKind::Ssm | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
        ) && layer.wq.n == 0)
        {
            return Ok(false);
        }
        if host_layer.block == HybridLayerKind::Ssm && layer.w_ssm_in.n > 0 {
            let ssm = host_layer.ssm.as_ref().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "SSM layer {layer_idx} missing Mamba-2 metadata/extras"
                )))
            })?;
            let dims = ssm.dims;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.w_ssm_in,
                buf.q,
                1,
                dims.in_proj_dim as i32,
                dim as i32,
            )?;
            let (conv, bias, dt, a, d, norm) = hip.ensure_mamba2_weights(
                layer_idx,
                &ssm.conv1d,
                &ssm.conv_bias,
                &ssm.dt_bias,
                &ssm.a,
                &ssm.d_skip,
                &ssm.norm,
            )?;
            let state = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(
                    "Mamba-2 layer requires recurrent state".into(),
                ))
            })?;
            state.ensure_ssm(layer_idx, dims.state_floats(), dims.conv_state_floats());
            let req_id = side.as_ref().map(|s| s.request_id).unwrap_or(0);
            hip.mamba2_decode_step_resident(
                req_id,
                layer_idx,
                &mut state.ssm_h[layer_idx],
                &mut state.ssm_conv[layer_idx],
                buf.q,
                conv,
                bias,
                dt,
                a,
                d,
                norm,
                buf.attn,
                dims.d_inner as i32,
                dims.d_state as i32,
                dims.n_heads as i32,
                dims.n_groups as i32,
                dims.conv_kernel as i32,
                cfg.eps,
            )?;
            resident.gemm_nt(
                hip,
                buf.attn,
                layer.w_ssm_out,
                buf.proj,
                1,
                dim as i32,
                dims.d_inner as i32,
            )?;
            if layer_idx + 1 < resident.layers.len() {
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                hip.residual_add(buf.x, buf.proj, dim as i32)?;
            }
            return Ok(true);
        }
        if let (Some(gdn), true) = (host_layer.gdn.as_ref(), layer.w_qkv.n > 0) {
            let dims = &gdn.dims;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.w_qkv,
                buf.q,
                1,
                dims.qkv_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.w_z,
                buf.attn,
                1,
                dims.value_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.w_alpha,
                buf.gate,
                1,
                dims.n_v_heads as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.w_beta,
                buf.up,
                1,
                dims.n_v_heads as i32,
                dim as i32,
            )?;
            let state = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(
                    "Gated DeltaNet layer requires recurrent state".into(),
                ))
            })?;
            state.ensure_gdn(layer_idx, dims);
            let req_id = side.as_ref().map(|s| s.request_id).unwrap_or(0);
            let device_gdn = (|| {
                let (conv_w, dt, a, norm) = hip.ensure_gdn_weights(
                    layer_idx,
                    &gdn.conv1d,
                    &gdn.dt_bias,
                    &gdn.a,
                    &gdn.norm,
                )?;
                hip.gdn_decode_step_resident(
                    req_id,
                    layer_idx,
                    &mut state.gdn_state[layer_idx],
                    &mut state.gdn_conv[layer_idx],
                    buf.q,
                    buf.attn,
                    buf.gate,
                    buf.up,
                    conv_w,
                    dt,
                    a,
                    norm,
                    buf.attn,
                    dims.n_v_heads as i32,
                    dims.n_k_heads as i32,
                    dims.head_v as i32,
                    dims.head_k as i32,
                    dims.conv_kernel as i32,
                    cfg.eps,
                )
            })();
            if let Err(e) = device_gdn {
                let mut qkv = vec![0.0f32; dims.qkv_dim];
                let mut z = vec![0.0f32; dims.value_dim];
                let mut alpha = vec![0.0f32; dims.n_v_heads];
                let mut beta = vec![0.0f32; dims.n_v_heads];
                hip.copy_d2h_f32(buf.q, &mut qkv)?;
                hip.copy_d2h_f32(buf.attn, &mut z)?;
                hip.copy_d2h_f32(buf.gate, &mut alpha)?;
                hip.copy_d2h_f32(buf.up, &mut beta)?;
                let mut conv_out = vec![0.0f32; dims.qkv_dim];
                hip.gdn_conv1d_step_host(
                    &qkv,
                    &gdn.conv1d,
                    &mut state.gdn_conv[layer_idx],
                    &mut conv_out,
                    dims.conv_kernel as i32,
                )?;
                let q_raw = &conv_out[..dims.key_dim];
                let k_raw = &conv_out[dims.key_dim..dims.key_dim * 2];
                let v_raw = &conv_out[dims.key_dim * 2..];
                let mut q_exp = vec![0.0f32; dims.n_v_heads * dims.head_k];
                let mut k_exp = vec![0.0f32; dims.n_v_heads * dims.head_k];
                hip.expand_k_heads_host(
                    q_raw,
                    &mut q_exp,
                    dims.n_k_heads as i32,
                    dims.n_v_heads as i32,
                    dims.head_k as i32,
                )?;
                hip.expand_k_heads_host(
                    k_raw,
                    &mut k_exp,
                    dims.n_k_heads as i32,
                    dims.n_v_heads as i32,
                    dims.head_k as i32,
                )?;
                let mut decay = vec![0.0f32; dims.n_v_heads];
                hip.gdn_decay_beta_host(&alpha, &gdn.dt_bias, &gdn.a, &mut decay, &mut beta)?;
                let mut y = vec![0.0f32; dims.value_dim];
                hip.gated_delta_step_resident(
                    req_id,
                    layer_idx,
                    &mut state.gdn_state[layer_idx],
                    &mut q_exp,
                    &mut k_exp,
                    v_raw,
                    &decay,
                    &beta,
                    &mut y,
                    dims.n_v_heads as i32,
                    dims.head_v as i32,
                    dims.head_k as i32,
                    cfg.eps,
                )?;
                let mut gated = vec![0.0f32; dims.value_dim];
                rmsnorm_silu_gate(&y, &z, &gdn.norm, &mut gated, dims.head_v, cfg.eps);
                hip.copy_h2d_f32(buf.attn, &gated)?;
                let _ = e;
            }
            resident.gemm_nt(
                hip,
                buf.attn,
                layer.w_ssm_out,
                buf.proj,
                1,
                dim as i32,
                dims.value_dim as i32,
            )?;
            if layer.w_gate.n > 0 {
                let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
                let ffn_norm = ffn.ffn_norm();
                let w_gate = ffn.w_gate();
                let w_up = ffn.w_up();
                let w_down = ffn.w_down();
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, ffn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_gate,
                    buf.gate,
                    1,
                    cfg.ff_dim as i32,
                    dim as i32,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_up,
                    buf.up,
                    1,
                    cfg.ff_dim as i32,
                    dim as i32,
                )?;
                hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
                resident.gemm_nt(
                    hip,
                    buf.gate,
                    w_down,
                    buf.proj,
                    1,
                    dim as i32,
                    cfg.ff_dim as i32,
                )?;
                drop(ffn);
            }
            if layer_idx + 1 < resident.layers.len() {
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                hip.residual_add(buf.x, buf.proj, dim as i32)?;
            }
            return Ok(true);
        }
        if host_layer.block == HybridLayerKind::Ssm {
            let ssm = host_layer.ssm.as_ref().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "SSM layer {layer_idx} missing production Mamba-2 weights"
                )))
            })?;
            let mut x_host = vec![0.0f32; dim];
            let mut normed_host = vec![0.0f32; dim];
            let mut y_host = vec![0.0f32; dim];
            hip.copy_d2h_f32(buf.x, &mut x_host)?;
            hip.copy_d2h_f32(buf.normed, &mut normed_host)?;
            let state = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(
                    "SSM layer requires recurrent state".into(),
                ))
            })?;
            state.ensure_ssm(
                layer_idx,
                ssm.dims.state_floats(),
                ssm.dims.conv_state_floats(),
            );
            mamba2_step(
                &normed_host,
                &mut state.ssm_h[layer_idx],
                &mut state.ssm_conv[layer_idx],
                ssm,
                cfg.eps,
                &mut y_host,
            )
            .map_err(|e| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!("SSM layer {layer_idx}: {e}")))
            })?;
            for i in 0..dim {
                x_host[i] += y_host[i];
            }
            hip.copy_h2d_f32(buf.x, &x_host)?;
        }
        if layer.w_gate.n > 0 {
            let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
            let ffn_norm = ffn.ffn_norm();
            let w_gate = ffn.w_gate();
            let w_up = ffn.w_up();
            let w_down = ffn.w_down();
            if ffn_norm.n > 0 {
                hip.rmsnorm_device(
                    buf.x,
                    resident.f32_or_dequant(hip, ffn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                // SAFETY: `buf.x`/`buf.normed` are distinct decode rows of `dim` floats.
                unsafe {
                    std::ptr::copy_nonoverlapping(buf.x, buf.normed, dim);
                }
            }
            resident.gemm_nt(
                hip,
                buf.normed,
                w_gate,
                buf.gate,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                w_up,
                buf.up,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
            resident.gemm_nt(
                hip,
                buf.gate,
                w_down,
                buf.proj,
                1,
                dim as i32,
                cfg.ff_dim as i32,
            )?;
            drop(ffn);
            if layer_idx + 1 < resident.layers.len() {
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                hip.residual_add(buf.x, buf.proj, dim as i32)?;
            }
        } else if Self::layer_has_device_moe(layer, resident, layer_idx) {
            hip.copy_h2d_f32(buf.proj, &vec![0.0f32; dim])?;
            if let Some(s) = side.as_mut() {
                self.device_moe(
                    layer_idx,
                    hip,
                    resident,
                    buf,
                    metrics,
                    prefill,
                    micro_batch,
                    Some(s),
                )?;
            } else {
                self.device_moe(
                    layer_idx,
                    hip,
                    resident,
                    buf,
                    metrics,
                    prefill,
                    micro_batch,
                    None,
                )?;
            }
            if layer_idx + 1 < resident.layers.len() {
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                hip.residual_add(buf.x, buf.proj, dim as i32)?;
            }
        } else if layer_idx + 1 < resident.layers.len() {
            hip.rmsnorm_device(
                buf.x,
                resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
        }
        Ok(true)
    }

    /// Sequential target decode with device-resident hidden state and KV.
    #[allow(clippy::too_many_arguments)]
    fn forward_token_device(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        prefill: bool,
        micro_batch: Option<&MicroBatchExpertUnion>,
        mut side: Option<&mut DeviceDecodeSide<'_>>,
        mut recurrent: Option<&mut RecurrentState>,
        soft_x: Option<&[f32]>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        let (q_workspace, kv_workspace) = Self::decode_workspaces(cfg, resident);
        let page_tokens = page_ids
            .first()
            .and_then(|p| p.first())
            .and_then(|id| kv.get(*id).map(|p| p.capacity_tokens as usize))
            .unwrap_or(128);
        let max_kv = page_ids
            .iter()
            .map(|layer| layer.len().saturating_mul(page_tokens))
            .max()
            .unwrap_or(pos + 1)
            .max(pos + 1);
        hip.bind()?;
        let buf = hip.ensure_decode(
            q_workspace,
            kv_workspace,
            cfg.moe_workspace(),
            vocab,
            max_kv,
        )?;
        hip.begin_decode_profile();
        let n_exp = cfg.n_experts;
        let double_buffer = prefill
            && n_exp > 0
            && self
                .expert_lru
                .as_ref()
                .is_some_and(|lru| lru.holds_layers(n_exp, 2));
        if double_buffer && !resident.prefill_layer_gpu_ready(0) {
            if let Ok(experts) = self.host_experts_layer(0) {
                let _ = resident.prefill_warm_layer_sync(0, n_exp as u32, &experts, Some(metrics));
            }
        }
        let result = (|| {
            if let Some(x0) = soft_x {
                hip.copy_h2d_f32(buf.x, x0)?;
            } else {
                resident.embed_gather(hip, token, buf.x, dim as i32)?;
            }
            record_elementwise(Some(metrics));

            for (layer_idx, host_layer) in self.layers.iter().enumerate() {
                if double_buffer {
                    resident.prefill_wait_layer(layer_idx as u32)?;
                    if layer_idx + 1 < self.layers.len() {
                        if let Ok(next) = self.host_experts_layer(layer_idx + 1) {
                            let _ = resident.prefill_kickoff_layer(
                                (layer_idx + 1) as u32,
                                n_exp as u32,
                                &next,
                                Some(metrics),
                            );
                        }
                    }
                }
                let layer = &resident.layers[layer_idx];
                if layer_idx == 0 {
                    hip.rmsnorm_device(
                        buf.x,
                        resident.f32_or_dequant(hip, layer.attn_norm)?,
                        buf.normed,
                        dim as i32,
                        cfg.eps,
                    )?;
                    record_elementwise(Some(metrics));
                }

                if self.device_hybrid_no_attn_layer(
                    layer_idx,
                    host_layer,
                    hip,
                    resident,
                    buf,
                    metrics,
                    prefill,
                    micro_batch,
                    side.as_deref_mut(),
                    recurrent.as_deref_mut(),
                )? {
                    continue;
                }

                let al = Self::layer_attn_layout(cfg, layer);
                let q_out = al.q_out;
                let kv_dim = al.kv_dim;
                let wo_k = al.wo_k;
                let gated = al.gated;
                let attn_dim = al.attn_dim;
                let head_dim = al.head_dim;
                let n_heads = al.n_heads;
                let n_kv_heads = al.n_kv_heads;

                if layer.wq.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wq,
                            buf.q,
                            1,
                            q_out as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wk,
                            buf.k,
                            1,
                            kv_dim as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wv,
                            buf.v,
                            1,
                            kv_dim as i32,
                            dim as i32,
                        )?;
                        Ok(())
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        layer.wq,
                        buf.q,
                        1,
                        q_out as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        layer.wk,
                        buf.k,
                        1,
                        kv_dim as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        layer.wv,
                        buf.v,
                        1,
                        kv_dim as i32,
                        dim as i32,
                    )?;
                }
                // Qwen3.5 gated attention: Q projection is per-head interleaved [Q|gate].
                if gated {
                    hip.deinterleave_q_gate_device(
                        buf.q,
                        buf.proj,
                        buf.gate,
                        n_heads as i32,
                        head_dim as i32,
                    )?;
                    hip.copy_d2d_f32(buf.q, buf.proj, attn_dim)?;
                    if layer.q_norm.n == head_dim {
                        hip.rmsnorm_heads_device(
                            buf.q,
                            resident.f32_or_dequant(hip, layer.q_norm)?,
                            n_heads as i32,
                            head_dim as i32,
                            cfg.eps,
                        )?;
                    }
                    if layer.k_norm.n == head_dim {
                        hip.rmsnorm_heads_device(
                            buf.k,
                            resident.f32_or_dequant(hip, layer.k_norm)?,
                            n_kv_heads as i32,
                            head_dim as i32,
                            cfg.eps,
                        )?;
                    }
                }
                if cfg.uses_imrope() || cfg.rope_dims_for_layer(layer_idx) > 0 {
                    let n_rot = {
                        let r = cfg.rope_dims_for_layer(layer_idx);
                        if r > 0 {
                            r
                        } else {
                            head_dim
                        }
                    };
                    hip.rope_imrope_qk_device(
                        buf.q,
                        buf.k,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        n_rot as i32,
                        pos as i32,
                        cfg.rope_theta_for_layer(layer_idx),
                        cfg.rope_sections,
                    )?;
                } else {
                    hip.rope_qk_device(
                        buf.q,
                        buf.k,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        pos as i32,
                        cfg.rope_theta_for_layer(layer_idx),
                    )?;
                }
                record_elementwise(Some(metrics));

                let page_index = pos / page_tokens;
                let slot = (pos % page_tokens) as u32;
                let page_id = page_ids[layer_idx][page_index];
                kv.write_slot_from_device(page_id, slot, n_kv_heads, head_dim, buf.k, buf.v)?;
                kv.touch(page_id, 1.0);

                let (packed_k, packed_v, n_kv) = kv.kv_device_views(
                    &page_ids[layer_idx],
                    n_kv_heads,
                    head_dim,
                    buf.packed_k,
                    buf.packed_v,
                    buf.max_kv,
                )?;
                if cfg.layer_is_swa(layer_idx) && cfg.sliding_window > 0 {
                    let mut mask = vec![1u8; n_kv];
                    let start = n_kv.saturating_sub(cfg.sliding_window);
                    for t in 0..start {
                        mask[t] = 0;
                    }
                    hip.attention_device_masked_upload(
                        buf.q,
                        packed_k,
                        packed_v,
                        buf.attn,
                        1,
                        n_kv as i32,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        &mask,
                    )?;
                } else {
                    hip.attention_device(
                        buf.q,
                        packed_k,
                        packed_v,
                        buf.attn,
                        1,
                        n_kv as i32,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        true,
                    )?;
                }
                if gated {
                    hip.sigmoid_mul_device(buf.attn, buf.gate, attn_dim as i32)?;
                }
                self.device_headwise_attn_gate(
                    hip, resident, layer, buf, 1, n_heads, head_dim, dim,
                )?;
                if layer.wo.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.attn,
                            layer.wo,
                            buf.proj,
                            1,
                            dim as i32,
                            wo_k as i32,
                        )
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.attn,
                        layer.wo,
                        buf.proj,
                        1,
                        dim as i32,
                        wo_k as i32,
                    )?;
                }
                let moe_here = layer.w_router.n > 0
                    || resident
                        .expert_offs
                        .get(layer_idx)
                        .is_some_and(|e| !e.is_empty());
                if moe_here {
                    if let Some(s) = side.as_mut() {
                        self.device_moe(
                            layer_idx,
                            hip,
                            resident,
                            buf,
                            metrics,
                            prefill,
                            micro_batch,
                            Some(s),
                        )?;
                    } else {
                        self.device_moe(
                            layer_idx,
                            hip,
                            resident,
                            buf,
                            metrics,
                            prefill,
                            micro_batch,
                            None,
                        )?;
                    }
                } else if (layer.w_gate.n > 0 && layer.ffn_norm.n > 0)
                    || self.streamed_ffn(layer_idx)
                {
                    let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
                    let ffn_norm = ffn.ffn_norm();
                    let w_gate = ffn.w_gate();
                    let w_up = ffn.w_up();
                    let w_down = ffn.w_down();
                    hip.residual_rmsnorm(
                        buf.x,
                        buf.proj,
                        resident.f32_or_dequant(hip, ffn_norm)?,
                        buf.normed,
                        dim as i32,
                        cfg.eps,
                    )?;
                    metrics
                        .residual_gpu_ops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    record_elementwise(Some(metrics));
                    if w_gate.kind == WeightKind::Q8_0 {
                        hip.profile_q8(|| {
                            resident.gemm_nt(
                                hip,
                                buf.normed,
                                w_gate,
                                buf.gate,
                                1,
                                cfg.ff_dim as i32,
                                dim as i32,
                            )?;
                            resident.gemm_nt(
                                hip,
                                buf.normed,
                                w_up,
                                buf.up,
                                1,
                                cfg.ff_dim as i32,
                                dim as i32,
                            )?;
                            Ok(())
                        })?;
                    } else {
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_gate,
                            buf.gate,
                            1,
                            cfg.ff_dim as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_up,
                            buf.up,
                            1,
                            cfg.ff_dim as i32,
                            dim as i32,
                        )?;
                    }
                    self.device_act_mul(hip, buf.gate, buf.up, cfg.ff_dim as i32)?;
                    record_elementwise(Some(metrics));
                    if w_down.kind == WeightKind::Q8_0 {
                        hip.profile_q8(|| {
                            resident.gemm_nt(
                                hip,
                                buf.gate,
                                w_down,
                                buf.proj,
                                1,
                                dim as i32,
                                cfg.ff_dim as i32,
                            )
                        })?;
                    } else {
                        resident.gemm_nt(
                            hip,
                            buf.gate,
                            w_down,
                            buf.proj,
                            1,
                            dim as i32,
                            cfg.ff_dim as i32,
                        )?;
                    }
                    drop(ffn);
                }
                // else: attention-only hybrid block — `buf.proj` still holds attn out.
                if layer_idx + 1 < resident.layers.len() {
                    hip.residual_rmsnorm(
                        buf.x,
                        buf.proj,
                        resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                        buf.normed,
                        dim as i32,
                        cfg.eps,
                    )?;
                } else {
                    hip.residual_add(buf.x, buf.proj, dim as i32)?;
                }
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
            }

            hip.rmsnorm_device(
                buf.x,
                resident.f32_or_dequant(hip, resident.norm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            record_elementwise(Some(metrics));
            if let Some(side) = side.as_mut() {
                if let Some(out) = side.last_hidden.as_mut() {
                    out.resize(dim, 0.0);
                    hip.copy_d2h_f32(buf.normed, out)?;
                }
            }
            let write_logits = side.as_ref().is_none_or(|s| s.write_logits);
            if write_logits {
                {
                    let _lm = hip.lm_head_scope();
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        resident.lm_head,
                        buf.logits,
                        1,
                        vocab as i32,
                        dim as i32,
                    )?;
                }
                hip.start_logits_d2h(buf.logits, vocab)?;
            }
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .target_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })();
        hip.end_decode_profile();
        result
    }

    /// Batched HIP prefill: one GEMM with m = N per weight, attention n_q = N.
    /// MoE expert GEMMs group tokens that share an expert. Hybrid SSM/GDN
    /// layers walk token rows with decode kernels.
    #[allow(clippy::too_many_arguments)]
    fn forward_tokens_device(
        &self,
        tokens: &[u32],
        start_pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        write_logits: bool,
        row_logits: Option<&mut [f32]>,
        mut recurrent: Option<&mut RecurrentState>,
        mut out_row_hiddens: Option<&mut Vec<f32>>,
        request_id: u64,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        let n_tok = tokens.len();
        let (q_workspace, kv_workspace) = Self::decode_workspaces(cfg, resident);
        let kv_dim = kv_dim.max(kv_workspace);
        if write_logits && logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        if let Some(rows) = row_logits.as_ref() {
            if rows.len() < n_tok.saturating_mul(vocab) {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "row logits buffer smaller than n_tok * vocab".into(),
                )));
            }
        }
        let page_tokens = page_ids
            .first()
            .and_then(|p| p.first())
            .and_then(|id| kv.get(*id).map(|p| p.capacity_tokens as usize))
            .unwrap_or(128);
        let seq_end = start_pos + n_tok;
        let max_kv = page_ids
            .iter()
            .map(|layer| layer.len().saturating_mul(page_tokens))
            .max()
            .unwrap_or(seq_end)
            .max(seq_end);
        hip.bind()?;
        let buf = hip.ensure_decode_batch(
            dim,
            kv_dim,
            cfg.moe_workspace(),
            vocab,
            max_kv,
            n_tok,
            if row_logits.is_some() { n_tok } else { 1 },
            q_workspace,
        )?;
        hip.begin_decode_profile();
        let result = (|| {
            resident.embed_gather_rows(hip, tokens, buf.x, dim as i32)?;
            record_elementwise(Some(metrics));

            let m = n_tok as i32;
            for (layer_idx, host_layer) in self.layers.iter().enumerate() {
                let layer = &resident.layers[layer_idx];
                if layer_idx == 0 {
                    rmsnorm_device_rows(
                        hip,
                        buf.x,
                        resident.f32_or_dequant(hip, layer.attn_norm)?,
                        buf.normed,
                        n_tok,
                        dim,
                        cfg.eps,
                    )?;
                    record_elementwise(Some(metrics));
                }
                if matches!(
                    host_layer.block,
                    HybridLayerKind::Ssm | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
                ) && layer.wq.n == 0
                {
                    let moe_only = Self::layer_has_device_moe(layer, resident, layer_idx)
                        && host_layer.ssm.is_none()
                        && host_layer.gdn.is_none();
                    if moe_only {
                        hip.copy_h2d_f32(buf.proj, &vec![0.0f32; n_tok * dim])?;
                        self.device_moe_rows(
                            layer_idx, hip, resident, buf, n_tok, metrics, request_id,
                        )?;
                        if layer_idx + 1 < resident.layers.len() {
                            hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                            rmsnorm_device_rows(
                                hip,
                                buf.x,
                                resident.f32_or_dequant(
                                    hip,
                                    resident.layers[layer_idx + 1].attn_norm,
                                )?,
                                buf.normed,
                                n_tok,
                                dim,
                                cfg.eps,
                            )?;
                        } else {
                            hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                        }
                        continue;
                    }
                    for t in 0..n_tok {
                        let row = Self::decode_row_view(buf, t, dim);
                        let mut side = DeviceDecodeSide::new(request_id);
                        let handled = self.device_hybrid_no_attn_layer(
                            layer_idx,
                            host_layer,
                            hip,
                            resident,
                            row,
                            metrics,
                            true,
                            None,
                            Some(&mut side),
                            recurrent.as_deref_mut(),
                        )?;
                        if !handled {
                            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                "hybrid prefill layer {layer_idx} was not handled"
                            ))));
                        }
                    }
                    continue;
                }
                let al = Self::layer_attn_layout(cfg, layer);
                if layer.wq.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wq,
                            buf.q,
                            m,
                            al.q_out as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wk,
                            buf.k,
                            m,
                            al.kv_dim as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wv,
                            buf.v,
                            m,
                            al.kv_dim as i32,
                            dim as i32,
                        )?;
                        Ok(())
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        layer.wq,
                        buf.q,
                        m,
                        al.q_out as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        layer.wk,
                        buf.k,
                        m,
                        al.kv_dim as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        layer.wv,
                        buf.v,
                        m,
                        al.kv_dim as i32,
                        dim as i32,
                    )?;
                }
                let theta = cfg.rope_theta_for_layer(layer_idx);
                let n_rot = cfg.rope_dims_for_layer(layer_idx);
                if cfg.uses_imrope() || n_rot > 0 {
                    let n_rot = if n_rot > 0 { n_rot } else { al.head_dim };
                    for t in 0..n_tok {
                        hip.rope_imrope_qk_device(
                            buf.q_row(t, al.q_out),
                            buf.k_row(t, al.kv_dim),
                            al.n_heads as i32,
                            al.n_kv_heads as i32,
                            al.head_dim as i32,
                            n_rot as i32,
                            (start_pos + t) as i32,
                            theta,
                            cfg.rope_sections,
                        )?;
                    }
                } else {
                    hip.rope_qk_device_rows(
                        buf.q,
                        buf.k,
                        n_tok as i32,
                        al.n_heads as i32,
                        al.n_kv_heads as i32,
                        al.head_dim as i32,
                        start_pos as i32,
                        theta,
                    )?;
                }
                record_elementwise(Some(metrics));

                kv.write_slots_from_device(
                    &page_ids[layer_idx],
                    start_pos,
                    n_tok,
                    page_tokens,
                    al.n_kv_heads,
                    al.head_dim,
                    buf.k,
                    buf.v,
                )?;

                let (packed_k, packed_v, n_kv) = kv.kv_device_views(
                    &page_ids[layer_idx],
                    al.n_kv_heads,
                    al.head_dim,
                    buf.packed_k,
                    buf.packed_v,
                    buf.max_kv,
                )?;
                if cfg.layer_is_swa(layer_idx) && cfg.sliding_window > 0 {
                    let mut mask = vec![0u8; n_tok.saturating_mul(n_kv)];
                    for t in 0..n_tok {
                        let abs = start_pos + t;
                        let lo = abs.saturating_add(1).saturating_sub(cfg.sliding_window);
                        for s in lo..=abs.min(n_kv.saturating_sub(1)) {
                            mask[t * n_kv + s] = 1;
                        }
                    }
                    hip.attention_device_masked_upload(
                        buf.q,
                        packed_k,
                        packed_v,
                        buf.attn,
                        m,
                        n_kv as i32,
                        al.n_heads as i32,
                        al.n_kv_heads as i32,
                        al.head_dim as i32,
                        &mask,
                    )?;
                } else {
                    hip.attention_device(
                        buf.q,
                        packed_k,
                        packed_v,
                        buf.attn,
                        m,
                        n_kv as i32,
                        al.n_heads as i32,
                        al.n_kv_heads as i32,
                        al.head_dim as i32,
                        true,
                    )?;
                }
                self.device_headwise_attn_gate(
                    hip,
                    resident,
                    layer,
                    buf,
                    n_tok,
                    al.n_heads,
                    al.head_dim,
                    dim,
                )?;
                if layer.wo.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.attn,
                            layer.wo,
                            buf.proj,
                            m,
                            dim as i32,
                            al.wo_k as i32,
                        )
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.attn,
                        layer.wo,
                        buf.proj,
                        m,
                        dim as i32,
                        al.wo_k as i32,
                    )?;
                }
                let moe_here = Self::layer_has_device_moe(layer, resident, layer_idx);
                let dense_ffn = ((layer.w_gate.n > 0 && layer.ffn_norm.n > 0)
                    || self.streamed_ffn(layer_idx))
                    && !moe_here;
                if moe_here {
                    self.device_moe_rows(
                        layer_idx, hip, resident, buf, n_tok, metrics, request_id,
                    )?;
                    if layer_idx + 1 < resident.layers.len() {
                        hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                        rmsnorm_device_rows(
                            hip,
                            buf.x,
                            resident
                                .f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                            buf.normed,
                            n_tok,
                            dim,
                            cfg.eps,
                        )?;
                    } else {
                        hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                    }
                    continue;
                }
                hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
                if !dense_ffn {
                    if layer_idx + 1 < resident.layers.len() {
                        rmsnorm_device_rows(
                            hip,
                            buf.x,
                            resident
                                .f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                            buf.normed,
                            n_tok,
                            dim,
                            cfg.eps,
                        )?;
                        record_elementwise(Some(metrics));
                    }
                    continue;
                }
                let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
                let ffn_norm = ffn.ffn_norm();
                let w_gate = ffn.w_gate();
                let w_up = ffn.w_up();
                let w_down = ffn.w_down();
                rmsnorm_device_rows(
                    hip,
                    buf.x,
                    resident.f32_or_dequant(hip, ffn_norm)?,
                    buf.normed,
                    n_tok,
                    dim,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
                if w_gate.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_gate,
                            buf.gate,
                            m,
                            cfg.ff_dim as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_up,
                            buf.up,
                            m,
                            cfg.ff_dim as i32,
                            dim as i32,
                        )?;
                        Ok(())
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_gate,
                        buf.gate,
                        m,
                        cfg.ff_dim as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_up,
                        buf.up,
                        m,
                        cfg.ff_dim as i32,
                        dim as i32,
                    )?;
                }
                self.device_act_mul(hip, buf.gate, buf.up, (n_tok * cfg.ff_dim) as i32)?;
                record_elementwise(Some(metrics));
                if w_down.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.gate,
                            w_down,
                            buf.proj,
                            m,
                            dim as i32,
                            cfg.ff_dim as i32,
                        )
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.gate,
                        w_down,
                        buf.proj,
                        m,
                        dim as i32,
                        cfg.ff_dim as i32,
                    )?;
                }
                drop(ffn);
                hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
                if layer_idx + 1 < resident.layers.len() {
                    rmsnorm_device_rows(
                        hip,
                        buf.x,
                        resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                        buf.normed,
                        n_tok,
                        dim,
                        cfg.eps,
                    )?;
                    record_elementwise(Some(metrics));
                }
            }

            if row_logits.is_some() {
                rmsnorm_device_rows(
                    hip,
                    buf.x,
                    resident.f32_or_dequant(hip, resident.norm)?,
                    buf.normed,
                    n_tok,
                    dim,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
                if let Some(out) = out_row_hiddens.as_mut() {
                    out.resize(n_tok * dim, 0.0);
                    hip.copy_d2h_f32(buf.normed, out)?;
                }
                {
                    let _lm = hip.lm_head_scope();
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        resident.lm_head,
                        buf.logits,
                        n_tok as i32,
                        vocab as i32,
                        dim as i32,
                    )?;
                }
                hip.start_logits_d2h(buf.logits, n_tok * vocab)?;
            } else if write_logits {
                let x_last = buf.x_row(n_tok - 1, dim);
                hip.rmsnorm_device(
                    x_last,
                    resident.f32_or_dequant(hip, resident.norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
                if let Some(out) = out_row_hiddens.as_mut() {
                    out.resize(dim, 0.0);
                    hip.copy_d2h_f32(buf.normed, out)?;
                }
                {
                    let _lm = hip.lm_head_scope();
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        resident.lm_head,
                        buf.logits,
                        1,
                        vocab as i32,
                        dim as i32,
                    )?;
                }
                hip.start_logits_d2h(buf.logits, vocab)?;
            } else if let Some(out) = out_row_hiddens.as_mut() {
                let x_last = buf.x_row(n_tok - 1, dim);
                hip.rmsnorm_device(
                    x_last,
                    resident.f32_or_dequant(hip, resident.norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                out.resize(dim, 0.0);
                hip.copy_d2h_f32(buf.normed, out)?;
            }
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .target_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })();
        hip.end_decode_profile();
        result
    }

    /// Packed Token Forest verify: one target eval with an ancestry attention mask.
    /// When `hip` is present the ancestry mask runs on the target GPU.
    pub fn forward_forest(
        &self,
        forest: &TokenForest,
        page_ids: &[Vec<u64>],
        kv: &KvTierManager,
        prompt_len: usize,
        node_logits: &mut [Vec<f32>],
        hip: Option<&HipWorkspace>,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let n_tree = forest.nodes.len();
        if n_tree == 0 {
            return Ok(());
        }
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(n_tree);
        for n in &forest.nodes {
            let tok = n.token_id as usize;
            if tok >= cfg.vocab_size {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                    "token {tok} out of vocab"
                ))));
            }
            xs.push(self.embed[tok * dim..(tok + 1) * dim].to_vec());
        }
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let ancestry = forest.ancestry_mask();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (prompt_k, prompt_v) =
                gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_prompt = prompt_k.len().checked_div(kv_dim).unwrap_or(0);
            let n_kv = n_prompt + n_tree;
            let mut qs = vec![0.0f32; n_tree * dim];
            let mut ks = vec![0.0f32; n_tree * kv_dim];
            let mut vs = vec![0.0f32; n_tree * kv_dim];
            for (i, x) in xs.iter().enumerate() {
                rmsnorm(x, &layer.attn_norm, cfg.eps, &mut scratch[..dim])?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wq,
                    &mut qs[i * dim..(i + 1) * dim],
                    1,
                    dim,
                    dim,
                )?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wk,
                    &mut ks[i * kv_dim..(i + 1) * kv_dim],
                    1,
                    kv_dim,
                    dim,
                )?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wv,
                    &mut vs[i * kv_dim..(i + 1) * kv_dim],
                    1,
                    kv_dim,
                    dim,
                )?;
                let pos = prompt_len + forest.nodes[i].depth as usize - 1;
                for h in 0..cfg.n_heads {
                    rope(
                        &mut qs[i * dim + h * head_dim..i * dim + (h + 1) * head_dim],
                        pos,
                        cfg.rope_theta,
                    )?;
                }
                for h in 0..cfg.n_kv_heads {
                    rope(
                        &mut ks[i * kv_dim + h * head_dim..i * kv_dim + (h + 1) * head_dim],
                        pos,
                        cfg.rope_theta,
                    )?;
                }
            }
            let mut k_all = prompt_k;
            k_all.extend_from_slice(&ks);
            let mut v_all = prompt_v;
            v_all.extend_from_slice(&vs);
            let mut mask = vec![false; n_tree * n_kv];
            for i in 0..n_tree {
                for t in 0..n_prompt {
                    mask[i * n_kv + t] = true;
                }
                for j in 0..n_tree {
                    if ancestry[i * n_tree + j] {
                        mask[i * n_kv + n_prompt + j] = true;
                    }
                }
            }
            let mut attn_out = vec![0.0f32; n_tree * dim];
            let mut gpu_mask = false;
            if let Some(hip) = hip {
                let mask_u8: Vec<u8> = mask.iter().map(|b| u8::from(*b)).collect();
                if hip
                    .attention_masked(
                        &qs,
                        &k_all,
                        &v_all,
                        &mut attn_out,
                        n_tree as i32,
                        n_kv as i32,
                        cfg.n_heads as i32,
                        cfg.n_kv_heads as i32,
                        head_dim as i32,
                        &mask_u8,
                    )
                    .is_ok()
                {
                    gpu_mask = true;
                    if let Some(m) = metrics {
                        m.tree_gpu_verifies
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
            if !gpu_mask {
                attention_with_mask_softcap(
                    &qs,
                    &k_all,
                    &v_all,
                    &mut attn_out,
                    n_tree,
                    n_kv,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    head_dim,
                    &mask,
                    cfg.attn_logit_softcap,
                )?;
            }
            let ffn_scratch = &mut scratch;
            for i in 0..n_tree {
                let mut hidden = vec![0.0f32; dim];
                matmul_nt(
                    &attn_out[i * dim..(i + 1) * dim],
                    &layer.wo,
                    &mut hidden,
                    1,
                    dim,
                    dim,
                )?;
                for d in 0..dim {
                    xs[i][d] += hidden[d];
                }
                self.apply_ffn(
                    layer_idx,
                    layer,
                    &mut xs[i],
                    ffn_scratch,
                    &mut hidden,
                    None,
                    None,
                    metrics,
                    false,
                )?;
            }
        }

        for (i, x) in xs.iter().enumerate() {
            rmsnorm(x, &self.norm, cfg.eps, &mut scratch[..dim])?;
            if node_logits[i].len() < vocab {
                node_logits[i].resize(vocab, 0.0);
            }
            matmul_nt(
                &scratch[..dim],
                &self.lm_head,
                &mut node_logits[i][..vocab],
                1,
                vocab,
                dim,
            )?;
        }
        Ok(())
    }

    /// Draft-path forward with a dense (non-paged) KV cache.
    pub fn forward_dense(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
    ) -> Result<(), AllpacaError> {
        self.forward_dense_on(token, pos, cache, logits, None, None, None)
    }

    /// Draft forward. GEMM/attention run on `hip` when present (resident draft GPU).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_dense_on(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        cache.forwards = cache.forwards.saturating_add(1);
        // Prefer on-device draft whenever resident weights exist (incl. MTP / MoE).
        if let (Some(hip), Some(resident)) = (hip, resident) {
            match self.forward_dense_device(token, pos, cache, logits, hip, resident, metrics) {
                Ok(()) => return Ok(()),
                Err(e) if !self.host_weights_ready() => {
                    // ROCmFP4 / packed drafts are device-only skeletons — no host embed.
                    return Err(e);
                }
                Err(e) => {
                    tracing::warn!(pos, token, error = %e, "HIP draft on-device decode failed; host fallback");
                }
            }
        }
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        cache.ensure_layers(cfg.n_layers);
        let mut x = if let Some(nextn) = self.mtp_nextn.as_ref() {
            // llama.cpp graph_mtp: seed = eh_proj(concat(RMSNorm(tok), RMSNorm(h))).
            let tok = &self.embed[token as usize * dim..(token as usize + 1) * dim];
            if cache.h_nextn.len() != dim {
                cache.h_nextn = vec![0.0f32; dim];
            }
            let mut e_norm = vec![0.0f32; dim];
            let mut h_norm = vec![0.0f32; dim];
            rmsnorm(tok, &nextn.enorm, cfg.eps, &mut e_norm)?;
            rmsnorm(&cache.h_nextn, &nextn.hnorm, cfg.eps, &mut h_norm)?;
            let mut concat = vec![0.0f32; 2 * dim];
            concat[..dim].copy_from_slice(&e_norm);
            concat[dim..].copy_from_slice(&h_norm);
            let mut seed = vec![0.0f32; dim];
            matmul_nt(&concat, &nextn.eh_proj, &mut seed, 1, dim, 2 * dim)?;
            seed
        } else {
            self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec()
        };
        let mut normed = vec![0.0f32; dim];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm_dispatch(
                hip,
                &x,
                &layer.attn_norm,
                resident.map(|r| r.attn_norm(layer_idx)),
                cfg.eps,
                &mut normed,
                metrics,
            )?;
            gemm_nt_dispatch(
                hip,
                &normed,
                &layer.wq,
                resident.map(|r| r.wq(layer_idx)),
                &mut q,
                1,
                dim,
                dim,
            )?;
            gemm_nt_dispatch(
                hip,
                &normed,
                &layer.wk,
                resident.map(|r| r.wk(layer_idx)),
                &mut k,
                1,
                kv_dim,
                dim,
            )?;
            gemm_nt_dispatch(
                hip,
                &normed,
                &layer.wv,
                resident.map(|r| r.wv(layer_idx)),
                &mut v,
                1,
                kv_dim,
                dim,
            )?;
            rope_dispatch(
                hip,
                &mut q,
                cfg.n_heads,
                head_dim,
                pos,
                cfg.rope_theta,
                metrics,
            )?;
            rope_dispatch(
                hip,
                &mut k,
                cfg.n_kv_heads,
                head_dim,
                pos,
                cfg.rope_theta,
                metrics,
            )?;
            cache.layers[layer_idx].k.extend_from_slice(&k);
            cache.layers[layer_idx].v.extend_from_slice(&v);
            let n_kv = cache.layers[layer_idx].k.len() / kv_dim;
            attention_dispatch(
                hip,
                &q,
                &cache.layers[layer_idx].k,
                &cache.layers[layer_idx].v,
                &mut attn_out,
                1,
                n_kv,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
                true,
                cfg.attn_logit_softcap,
            )?;
            gemm_nt_dispatch(
                hip,
                &attn_out,
                &layer.wo,
                resident.map(|r| r.wo(layer_idx)),
                &mut hidden,
                1,
                dim,
                dim,
            )?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
            self.apply_ffn(
                layer_idx,
                layer,
                &mut x,
                &mut normed,
                &mut hidden,
                hip,
                resident,
                metrics,
                false,
            )?;
        }
        rmsnorm_dispatch(
            hip,
            &x,
            &self.norm,
            resident.map(|r| r.norm()),
            cfg.eps,
            &mut normed,
            metrics,
        )?;
        if self.mtp_nextn.is_some() {
            cache.h_nextn = normed.clone();
        }
        gemm_nt_dispatch(
            hip,
            &normed,
            &self.lm_head,
            resident.map(|r| r.lm_head()),
            &mut logits[..cfg.vocab_size],
            1,
            cfg.vocab_size,
            dim,
        )?;
        if hip.is_some() {
            if let Some(metrics) = metrics {
                metrics
                    .gpu_forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                metrics
                    .draft_gpu_forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Sequential draft decode with device-resident hidden state and DenseKv.
    #[allow(clippy::too_many_arguments)]
    fn forward_dense_device(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        // Match target decode workspace sizing (gated Q is often 2× attn_out).
        let mut q_workspace = dim;
        let mut kv_workspace = cfg.n_kv_heads * cfg.head_dim().max(1);
        for layer in &resident.layers {
            if layer.wq.n > 0 && layer.wq.n % dim == 0 {
                q_workspace = q_workspace.max(layer.wq.n / dim);
            }
            if layer.wo.n > 0 && layer.wo.n % dim == 0 {
                q_workspace = q_workspace.max(layer.wo.n / dim);
            }
            if layer.wk.n > 0 && layer.wk.n % dim == 0 {
                kv_workspace = kv_workspace.max(layer.wk.n / dim);
            }
        }
        hip.bind()?;
        // MTP draft KV is a short local window (seq.len()), not absolute pos.
        let mtp = resident.mtp_nextn.is_some() || self.mtp_nextn.is_some();
        let kv_slot = if mtp { cache.seq.len() } else { pos };
        cache.ensure_device(hip.device, cfg.n_layers, kv_workspace, kv_slot + 1)?;
        if let Some(metrics) = metrics {
            metrics
                .draft_kv_bytes_on_device
                .store(cache.device_bytes(), std::sync::atomic::Ordering::Relaxed);
        }
        let ff_scratch = cfg
            .moe_workspace()
            .max(2 * dim)
            .max(q_workspace.saturating_add(256));
        let buf = hip.ensure_decode(q_workspace, kv_workspace, ff_scratch, vocab, kv_slot + 1)?;

        if let Some(nextn) = resident.mtp_nextn {
            // graph_mtp: seed = eh_proj(concat(RMSNorm(tok), RMSNorm(h_nextn))).
            let h_ptr = cache.ensure_h_nextn_device(hip.device, dim)?;
            resident.embed_gather(hip, token, buf.x, dim as i32)?;
            hip.rmsnorm_device(
                buf.x,
                resident.f32_or_dequant(hip, nextn.enorm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            hip.rmsnorm_device(
                h_ptr,
                resident.f32_or_dequant(hip, nextn.hnorm)?,
                buf.attn,
                dim as i32,
                cfg.eps,
            )?;
            hip.copy_d2d_f32(buf.gate, buf.normed, dim)?;
            hip.copy_d2d_f32(unsafe { buf.gate.add(dim) }, buf.attn, dim)?;
            resident.gemm_nt(
                hip,
                buf.gate,
                nextn.eh_proj,
                buf.x,
                1,
                dim as i32,
                (2 * dim) as i32,
            )?;
        } else {
            resident.embed_gather(hip, token, buf.x, dim as i32)?;
        }
        record_elementwise(metrics);

        for layer_idx in 0..self.layers.len() {
            let layer = &resident.layers[layer_idx];
            let q_out = if layer.wq.n > 0 && layer.wq.n % dim == 0 {
                layer.wq.n / dim
            } else {
                dim
            };
            let kv_dim = if layer.wk.n > 0 && layer.wk.n % dim == 0 {
                layer.wk.n / dim
            } else {
                cfg.n_kv_heads * cfg.head_dim().max(1)
            };
            let wo_k = if layer.wo.n > 0 && layer.wo.n % dim == 0 {
                layer.wo.n / dim
            } else {
                dim
            };
            let gated = q_out == wo_k.saturating_mul(2) && wo_k > dim;
            let attn_dim = if gated {
                wo_k
            } else {
                q_out.min(wo_k).max(dim)
            };
            let head_dim = {
                let c = cfg.head_dim().max(1);
                if attn_dim % c == 0 && kv_dim % c == 0 {
                    c
                } else if attn_dim % 256 == 0 && kv_dim % 256 == 0 {
                    256
                } else {
                    c
                }
            };
            let n_heads = (attn_dim / head_dim).max(1);
            let n_kv_heads = (kv_dim / head_dim).max(1);

            if layer_idx == 0 {
                hip.rmsnorm_device(
                    buf.x,
                    resident.f32_or_dequant(hip, layer.attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                record_elementwise(metrics);
            }
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.wq,
                buf.q,
                1,
                q_out as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.wk,
                buf.k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.wv,
                buf.v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;

            if gated {
                // q_full in buf.q → compact q into buf.proj, gate into buf.gate; then q → buf.q.
                hip.deinterleave_q_gate_device(
                    buf.q,
                    buf.proj,
                    buf.gate,
                    n_heads as i32,
                    head_dim as i32,
                )?;
                hip.copy_d2d_f32(buf.q, buf.proj, attn_dim)?;
                if layer.q_norm.n == head_dim {
                    hip.rmsnorm_heads_device(
                        buf.q,
                        resident.f32_or_dequant(hip, layer.q_norm)?,
                        n_heads as i32,
                        head_dim as i32,
                        cfg.eps,
                    )?;
                }
                if layer.k_norm.n == head_dim {
                    hip.rmsnorm_heads_device(
                        buf.k,
                        resident.f32_or_dequant(hip, layer.k_norm)?,
                        n_kv_heads as i32,
                        head_dim as i32,
                        cfg.eps,
                    )?;
                }
            }
            if cfg.uses_imrope() || cfg.rope_dims_for_layer(layer_idx) > 0 {
                let n_rot = {
                    let r = cfg.rope_dims_for_layer(layer_idx);
                    if r > 0 {
                        r
                    } else {
                        head_dim
                    }
                };
                hip.rope_imrope_qk_device(
                    buf.q,
                    buf.k,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    n_rot as i32,
                    pos as i32,
                    cfg.rope_theta_for_layer(layer_idx),
                    cfg.rope_sections,
                )?;
            } else {
                hip.rope_qk_device(
                    buf.q,
                    buf.k,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    pos as i32,
                    cfg.rope_theta_for_layer(layer_idx),
                )?;
            }
            record_elementwise(metrics);
            cache.write_pos_from_device(layer_idx, kv_slot, buf.k, buf.v)?;
            let n_kv = kv_slot + 1;
            if cfg.layer_is_swa(layer_idx) && cfg.sliding_window > 0 {
                let mut mask = vec![1u8; n_kv];
                let start = n_kv.saturating_sub(cfg.sliding_window);
                for t in 0..start {
                    mask[t] = 0;
                }
                hip.attention_device_masked_upload(
                    buf.q,
                    cache.k_ptr(layer_idx),
                    cache.v_ptr(layer_idx),
                    buf.attn,
                    1,
                    n_kv as i32,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    &mask,
                )?;
            } else {
                hip.attention_device(
                    buf.q,
                    cache.k_ptr(layer_idx),
                    cache.v_ptr(layer_idx),
                    buf.attn,
                    1,
                    n_kv as i32,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    true,
                )?;
            }
            if gated {
                hip.sigmoid_mul_device(buf.attn, buf.gate, attn_dim as i32)?;
            }
            self.device_headwise_attn_gate(
                hip, resident, layer, buf, 1, n_heads, head_dim, dim,
            )?;
            resident.gemm_nt(
                hip,
                buf.attn,
                layer.wo,
                buf.proj,
                1,
                dim as i32,
                wo_k as i32,
            )?;
            let moe_layer = self.config.n_experts > 0 && layer.w_router.n > 0;
            if moe_layer {
                // MTP / draft MoE: same device_moe path as target decode (routed experts).
                let metrics_owned;
                let metrics_ref = match metrics {
                    Some(m) => m,
                    None => {
                        metrics_owned = Metrics::default();
                        &metrics_owned
                    }
                };
                self.device_moe(
                    layer_idx,
                    hip,
                    resident,
                    buf,
                    metrics_ref,
                    false,
                    None,
                    None,
                )?;
                // device_moe leaves MoE out in buf.proj; fold into residual like target.
                if layer_idx + 1 < resident.layers.len() {
                    hip.residual_rmsnorm(
                        buf.x,
                        buf.proj,
                        resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                        buf.normed,
                        dim as i32,
                        cfg.eps,
                    )?;
                } else {
                    hip.residual_add(buf.x, buf.proj, dim as i32)?;
                }
                if let Some(metrics) = metrics {
                    metrics
                        .residual_gpu_ops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                record_elementwise(metrics);
            } else {
                let ffn = self.device_ffn(resident, layer_idx, metrics)?;
                let ffn_norm = ffn.ffn_norm();
                let w_gate = ffn.w_gate();
                let w_up = ffn.w_up();
                let w_down = ffn.w_down();
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, ffn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                if let Some(metrics) = metrics {
                    metrics
                        .residual_gpu_ops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                record_elementwise(metrics);
                // FFN reuses buf.gate (overwrites MTP attn gate — already applied).
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_gate,
                    buf.gate,
                    1,
                    cfg.ff_dim as i32,
                    dim as i32,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_up,
                    buf.up,
                    1,
                    cfg.ff_dim as i32,
                    dim as i32,
                )?;
                self.device_act_mul(hip, buf.gate, buf.up, cfg.ff_dim as i32)?;
                record_elementwise(metrics);
                resident.gemm_nt(
                    hip,
                    buf.gate,
                    w_down,
                    buf.proj,
                    1,
                    dim as i32,
                    cfg.ff_dim as i32,
                )?;
                drop(ffn);
                if layer_idx + 1 < resident.layers.len() {
                    hip.residual_rmsnorm(
                        buf.x,
                        buf.proj,
                        resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                        buf.normed,
                        dim as i32,
                        cfg.eps,
                    )?;
                } else {
                    hip.residual_add(buf.x, buf.proj, dim as i32)?;
                }
                if let Some(metrics) = metrics {
                    metrics
                        .residual_gpu_ops
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                record_elementwise(metrics);
            }
        }

        hip.rmsnorm_device(
            buf.x,
            resident.f32_or_dequant(hip, resident.norm)?,
            buf.normed,
            dim as i32,
            cfg.eps,
        )?;
        record_elementwise(metrics);
        if resident.mtp_nextn.is_some() || self.mtp_nextn.is_some() {
            cache.store_h_nextn_from_device(buf.normed, dim)?;
        }
        {
            let _lm = hip.lm_head_scope();
            resident.gemm_nt(
                hip,
                buf.normed,
                resident.lm_head,
                buf.logits,
                1,
                vocab as i32,
                dim as i32,
            )?;
        }
        hip.publish_logits(buf.logits, &mut logits[..vocab])?;
        if let Some(metrics) = metrics {
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .draft_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

struct DeviceLayerKv {
    k: *mut u8,
    v: *mut u8,
    bytes: usize,
}

unsafe impl Send for DeviceLayerKv {}
unsafe impl Sync for DeviceLayerKv {}

impl std::fmt::Debug for DeviceLayerKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceLayerKv")
            .field("bytes", &self.bytes)
            .finish()
    }
}

#[derive(Debug)]
pub struct DenseKv {
    pub layers: Vec<DenseLayerKv>,
    device: Option<i32>,
    device_layers: Vec<DeviceLayerKv>,
    tokens: usize,
    capacity: usize,
    kv_dim: usize,
    seq: Vec<u32>,
    /// Draft decode steps on this cache (CPU or HIP). Locks prefix reuse.
    pub(crate) forwards: u64,
    /// Rolling MTP `h_nextn` host mirror (seed from target / CPU path).
    pub(crate) h_nextn: Vec<f32>,
    /// Device-resident `h_nextn` on the draft GPU (avoids H2D each draft step).
    h_nextn_dev: *mut u8,
    h_nextn_bytes: usize,
    h_nextn_dev_device: Option<i32>,
    /// Host `h_nextn` was refreshed from the target and needs H2D before device seed.
    pub(crate) h_nextn_host_dirty: bool,
}

// Device pointers; Engine moves DenseKv across threads under Mutex.
unsafe impl Send for DenseKv {}
unsafe impl Sync for DenseKv {}

impl Default for DenseKv {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            device: None,
            device_layers: Vec::new(),
            tokens: 0,
            capacity: 0,
            kv_dim: 0,
            seq: Vec::new(),
            forwards: 0,
            h_nextn: Vec::new(),
            h_nextn_dev: std::ptr::null_mut(),
            h_nextn_bytes: 0,
            h_nextn_dev_device: None,
            h_nextn_host_dirty: false,
        }
    }
}

impl Drop for DenseKv {
    fn drop(&mut self) {
        self.free_device();
    }
}

#[derive(Debug, Default, Clone)]
pub struct DenseLayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl DenseKv {
    fn ensure_layers(&mut self, n: usize) {
        self.layers.resize_with(n, DenseLayerKv::default);
    }

    pub fn device_bytes(&self) -> u64 {
        self.device_layers
            .iter()
            .map(|l| (l.bytes * 2) as u64)
            .sum()
    }

    pub fn reset(&mut self) {
        self.tokens = 0;
        self.seq.clear();
        self.h_nextn.clear();
        self.h_nextn_host_dirty = false;
        for layer in &mut self.layers {
            layer.k.clear();
            layer.v.clear();
        }
    }

    pub(crate) fn prefix_len(&self, ctx: &[u32]) -> usize {
        self.seq
            .iter()
            .zip(ctx)
            .take_while(|(a, b)| *a == *b)
            .count()
    }

    pub(crate) fn seq_len(&self) -> usize {
        self.seq.len()
    }

    pub(crate) fn seq_matches(&self, ctx: &[u32]) -> bool {
        self.seq.as_slice() == ctx
    }

    pub(crate) fn truncate(&mut self, n: usize, kv_dim: usize) {
        self.rewind_to(n, kv_dim);
        self.seq.truncate(n);
    }

    pub(crate) fn record(&mut self, token: u32) {
        self.seq.push(token);
    }

    pub(crate) fn seed_h_nextn_from_target(&mut self, hidden: &[f32]) {
        self.h_nextn = hidden.to_vec();
        self.h_nextn_host_dirty = true;
    }

    /// Independent copy of process KV (host seq/h_nextn/layers + D2D device K/V).
    /// Pointers are never aliased; the original cache stays usable for process().
    pub(crate) fn snapshot(&self) -> Result<DenseKv, AllpacaError> {
        let mut out = DenseKv {
            layers: self.layers.clone(),
            device: None,
            device_layers: Vec::new(),
            tokens: self.tokens,
            capacity: 0,
            kv_dim: self.kv_dim,
            seq: self.seq.clone(),
            forwards: self.forwards,
            h_nextn: self.h_nextn.clone(),
            h_nextn_dev: std::ptr::null_mut(),
            h_nextn_bytes: 0,
            h_nextn_dev_device: None,
            h_nextn_host_dirty: self.h_nextn_host_dirty,
        };
        let Some(device) = self.device else {
            return Ok(out);
        };
        if !self.device_layers.is_empty() && self.kv_dim > 0 {
            let n_layers = self.device_layers.len();
            let cap = self.capacity.max(self.tokens).max(1);
            out.ensure_device(device, n_layers, self.kv_dim, cap)?;
            let copy_tokens = self.tokens.min(out.capacity);
            let copy_bytes = copy_tokens.saturating_mul(self.kv_dim).saturating_mul(4);
            if copy_bytes > 0 {
                allpaca_hip::set_device(device)?;
                for i in 0..n_layers {
                    allpaca_hip::memcpy(
                        out.device_layers[i].k,
                        self.device_layers[i].k,
                        copy_bytes,
                        allpaca_hip::MEMCPY_D2D,
                    )?;
                    allpaca_hip::memcpy(
                        out.device_layers[i].v,
                        self.device_layers[i].v,
                        copy_bytes,
                        allpaca_hip::MEMCPY_D2D,
                    )?;
                }
            }
            out.tokens = self.tokens;
        }
        if !self.h_nextn_dev.is_null() && self.h_nextn_bytes > 0 {
            let dim = self.h_nextn.len().max(self.h_nextn_bytes / 4).max(1);
            if self.h_nextn_host_dirty {
                out.h_nextn_host_dirty = true;
            } else {
                let dst = out.ensure_h_nextn_device(device, dim)?;
                allpaca_hip::set_device(device)?;
                allpaca_hip::memcpy(
                    dst.cast(),
                    self.h_nextn_dev,
                    self.h_nextn_bytes.min(out.h_nextn_bytes),
                    allpaca_hip::MEMCPY_D2D,
                )?;
                out.h_nextn_host_dirty = false;
            }
        }
        Ok(out)
    }

    fn free_h_nextn_device(&mut self) {
        if !self.h_nextn_dev.is_null() {
            allpaca_hip::device_free(self.h_nextn_dev, self.h_nextn_bytes);
            self.h_nextn_dev = std::ptr::null_mut();
            self.h_nextn_bytes = 0;
            self.h_nextn_dev_device = None;
        }
    }

    fn ensure_h_nextn_device(&mut self, device: i32, dim: usize) -> Result<*mut f32, AllpacaError> {
        let bytes = dim * 4;
        let reuse = self.h_nextn_dev_device == Some(device)
            && self.h_nextn_bytes >= bytes
            && !self.h_nextn_dev.is_null();
        if !reuse {
            self.free_h_nextn_device();
            allpaca_hip::set_device(device)?;
            let p = allpaca_hip::device_malloc(bytes, device)?;
            self.h_nextn_dev = p;
            self.h_nextn_bytes = bytes;
            self.h_nextn_dev_device = Some(device);
        }
        if self.h_nextn_host_dirty {
            if self.h_nextn.len() != dim {
                self.h_nextn.resize(dim, 0.0);
            }
            allpaca_hip::set_device(device)?;
            allpaca_hip::memcpy(
                self.h_nextn_dev,
                self.h_nextn.as_ptr().cast(),
                bytes,
                allpaca_hip::MEMCPY_H2D,
            )?;
            self.h_nextn_host_dirty = false;
        } else if self.h_nextn.len() != dim {
            // First use without target seed: zeros on device.
            self.h_nextn = vec![0.0f32; dim];
            allpaca_hip::set_device(device)?;
            allpaca_hip::memcpy(
                self.h_nextn_dev,
                self.h_nextn.as_ptr().cast(),
                bytes,
                allpaca_hip::MEMCPY_H2D,
            )?;
        }
        Ok(self.h_nextn_dev.cast())
    }

    fn store_h_nextn_from_device(
        &mut self,
        src: *const f32,
        dim: usize,
    ) -> Result<(), AllpacaError> {
        let Some(device) = self.h_nextn_dev_device else {
            self.h_nextn.resize(dim, 0.0);
            allpaca_hip::memcpy(
                self.h_nextn.as_mut_ptr().cast(),
                src.cast(),
                dim * 4,
                allpaca_hip::MEMCPY_D2H,
            )?;
            return Ok(());
        };
        let dst = self.ensure_h_nextn_device(device, dim)?;
        allpaca_hip::memcpy(dst.cast(), src.cast(), dim * 4, allpaca_hip::MEMCPY_D2D)?;
        self.h_nextn_host_dirty = false;
        Ok(())
    }

    fn free_device(&mut self) {
        for layer in self.device_layers.drain(..) {
            allpaca_hip::device_free(layer.k, layer.bytes);
            allpaca_hip::device_free(layer.v, layer.bytes);
        }
        self.free_h_nextn_device();
        self.device = None;
        self.capacity = 0;
    }

    fn ensure_device(
        &mut self,
        device: i32,
        n_layers: usize,
        kv_dim: usize,
        min_tokens: usize,
    ) -> Result<(), AllpacaError> {
        allpaca_hip::set_device(device)?;
        self.ensure_layers(n_layers);
        let need = min_tokens.max(1);
        let reuse = self.device == Some(device)
            && self.kv_dim == kv_dim
            && self.device_layers.len() == n_layers
            && self.capacity >= need;
        if reuse {
            return Ok(());
        }
        let new_cap = need.max(self.capacity.saturating_mul(2)).max(8);
        let bytes = new_cap * kv_dim * 4;
        let copy_tokens = if self.device == Some(device) && self.kv_dim == kv_dim {
            self.tokens
        } else {
            0
        };
        let copy_bytes = copy_tokens * kv_dim * 4;
        let mut new_layers: Vec<DeviceLayerKv> = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let k = match allpaca_hip::device_malloc(bytes, device) {
                Ok(p) => p,
                Err(e) => {
                    for layer in new_layers.drain(..) {
                        allpaca_hip::device_free(layer.k, layer.bytes);
                        allpaca_hip::device_free(layer.v, layer.bytes);
                    }
                    return Err(e);
                }
            };
            let v = match allpaca_hip::device_malloc(bytes, device) {
                Ok(p) => p,
                Err(e) => {
                    allpaca_hip::device_free(k, bytes);
                    for layer in new_layers.drain(..) {
                        allpaca_hip::device_free(layer.k, layer.bytes);
                        allpaca_hip::device_free(layer.v, layer.bytes);
                    }
                    return Err(e);
                }
            };
            if copy_bytes > 0 && i < self.device_layers.len() {
                allpaca_hip::memcpy(
                    k,
                    self.device_layers[i].k,
                    copy_bytes,
                    allpaca_hip::MEMCPY_D2D,
                )?;
                allpaca_hip::memcpy(
                    v,
                    self.device_layers[i].v,
                    copy_bytes,
                    allpaca_hip::MEMCPY_D2D,
                )?;
            }
            new_layers.push(DeviceLayerKv { k, v, bytes });
        }
        self.free_device();
        self.device_layers = new_layers;
        self.device = Some(device);
        self.capacity = new_cap;
        self.kv_dim = kv_dim;
        Ok(())
    }

    fn write_pos_from_device(
        &mut self,
        layer: usize,
        pos: usize,
        k_dev: *const f32,
        v_dev: *const f32,
    ) -> Result<(), AllpacaError> {
        if layer >= self.device_layers.len() || k_dev.is_null() || v_dev.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "draft device KV pointer is null".into(),
            )));
        }
        if pos >= self.capacity || self.kv_dim == 0 {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "draft device KV capacity exceeded".into(),
            )));
        }
        let kv_bytes = self.kv_dim * 4;
        let off = pos * kv_bytes;
        let dest_k = unsafe { self.device_layers[layer].k.add(off) };
        let dest_v = unsafe { self.device_layers[layer].v.add(off) };
        allpaca_hip::memcpy(dest_k, k_dev.cast(), kv_bytes, allpaca_hip::MEMCPY_D2D)?;
        allpaca_hip::memcpy(dest_v, v_dev.cast(), kv_bytes, allpaca_hip::MEMCPY_D2D)?;
        self.tokens = self.tokens.max(pos + 1);
        Ok(())
    }

    fn k_ptr(&self, layer: usize) -> *const f32 {
        self.device_layers
            .get(layer)
            .map(|l| l.k.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null())
    }

    fn v_ptr(&self, layer: usize) -> *const f32 {
        self.device_layers
            .get(layer)
            .map(|l| l.v.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null())
    }

    pub fn rewind_to(&mut self, tokens: usize, kv_dim: usize) {
        for layer in &mut self.layers {
            let keep = tokens * kv_dim;
            layer.k.truncate(keep);
            layer.v.truncate(keep);
        }
        self.tokens = tokens;
    }
}

fn record_elementwise(metrics: Option<&Metrics>) {
    if let Some(metrics) = metrics {
        metrics
            .elementwise_gpu_ops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn rmsnorm_device_rows(
    hip: &HipWorkspace,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n_tok: usize,
    dim: usize,
    eps: f32,
) -> Result<(), AllpacaError> {
    hip.rmsnorm_device_rows(x, weight, out, n_tok as i32, dim as i32, eps)
}

fn rmsnorm_dispatch(
    hip: Option<&HipWorkspace>,
    x: &[f32],
    weight: &[f32],
    weight_dev: Option<*const f32>,
    eps: f32,
    out: &mut [f32],
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip.rmsnorm(x, weight, weight_dev, eps, out).is_ok() {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    rmsnorm(x, weight, eps, out)
}

fn rope_dispatch(
    hip: Option<&HipWorkspace>,
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip
            .rope(x, n_heads as i32, head_dim as i32, pos as i32, theta)
            .is_ok()
        {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    for h in 0..n_heads {
        rope(&mut x[h * head_dim..(h + 1) * head_dim], pos, theta)?;
    }
    Ok(())
}

fn silu_mul_dispatch(
    hip: Option<&HipWorkspace>,
    gate: &mut [f32],
    up: &[f32],
    metrics: Option<&Metrics>,
    use_gelu: bool,
) -> Result<(), AllpacaError> {
    if use_gelu {
        for i in 0..gate.len() {
            gate[i] = gelu(gate[i]) * up[i];
        }
        record_elementwise(metrics);
        return Ok(());
    }
    if let Some(hip) = hip {
        if hip.silu_mul(gate, up).is_ok() {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

fn softcap_inplace(x: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    for v in x {
        *v = (*v / cap).tanh() * cap;
    }
}

fn apply_headwise_attn_gate(
    attn_inp: &[f32],
    gate_w: &[f32],
    attn_out: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    dim: usize,
) {
    if n_heads == 0 || gate_w.len() < n_heads * dim || attn_out.len() < n_heads * head_dim {
        return;
    }
    for h in 0..n_heads {
        let row = &gate_w[h * dim..(h + 1) * dim];
        let mut s = 0.0f32;
        for i in 0..dim.min(attn_inp.len()) {
            s += attn_inp[i] * row[i];
        }
        let g = 1.0 / (1.0 + (-s).exp());
        let base = h * head_dim;
        for d in 0..head_dim {
            attn_out[base + d] *= g;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_nt_dispatch(
    hip: Option<&HipWorkspace>,
    a: &[f32],
    b: &[f32],
    b_dev: Option<*const f32>,
    c: &mut [f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    if let (Some(hip), Some(b_dev)) = (hip, b_dev) {
        if hip
            .gemm_nt_dev_b(a, b_dev, c, rows as i32, n as i32, k as i32)
            .is_ok()
        {
            return Ok(());
        }
    }
    if let Some(hip) = hip {
        if hip
            .gemm_nt(a, b, c, rows as i32, n as i32, k as i32)
            .is_ok()
        {
            return Ok(());
        }
    }
    matmul_nt(a, b, c, rows, n, k)
}

#[allow(clippy::too_many_arguments)]
fn attention_dispatch(
    hip: Option<&HipWorkspace>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
    softcap: f32,
) -> Result<(), AllpacaError> {
    if softcap <= 0.0 {
        if let Some(hip) = hip {
            if hip
                .attention(
                    q,
                    k,
                    v,
                    out,
                    n_q as i32,
                    n_kv as i32,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    causal,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
        return attention(
            q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
        );
    }
    // Softcap path is host-only until HIP kernels grow a softcap arg.
    attention_softcap(
        q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal, softcap,
    )
}

pub(crate) fn gather_kv(
    kv: &KvTierManager,
    pages: &[u64],
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    for id in pages {
        if kv.get(*id).map(|p| p.filled).unwrap_or(0) == 0 {
            continue;
        }
        let (k, v) = kv.read_kv(*id, n_kv_heads, head_dim)?;
        ks.extend_from_slice(&k);
        vs.extend_from_slice(&v);
    }
    Ok((ks, vs))
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

#[cfg(feature = "cuda")]
#[path = "cuda_decode.rs"]
mod cuda_decode;

#[cfg(feature = "cuda")]
pub use cuda_decode::CudaResidentModel;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((self.state >> 33) as u32) & 0x007F_FFFF;
        (bits as f32 / 8_388_608.0) * 0.2 - 0.1
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Allocate one page per layer covering `n_tokens` (rounded up to page_tokens).
pub fn alloc_pages_for_seq(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    n_tokens: usize,
    metrics: &Metrics,
) -> Result<Vec<Vec<u64>>, AllpacaError> {
    let n_pages = n_tokens.div_ceil(page_tokens as usize).max(1);
    let mut pages = Vec::with_capacity(cfg.n_layers);
    for layer in 0..cfg.n_layers {
        let mut layer_pages = Vec::with_capacity(n_pages);
        for p in 0..n_pages {
            layer_pages.push(kv.alloc_page(
                layer as u32,
                p as u32,
                page_tokens,
                cfg.n_kv_heads,
                cfg.head_dim(),
                metrics,
            )?);
        }
        pages.push(layer_pages);
    }
    Ok(pages)
}

pub fn ensure_pages(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    pages: &mut [Vec<u64>],
    pos: usize,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let need = pos / page_tokens as usize;
    for (layer, layer_pages) in pages.iter_mut().enumerate() {
        while layer_pages.len() <= need {
            let idx = layer_pages.len() as u32;
            layer_pages.push(kv.alloc_page(
                layer as u32,
                idx,
                page_tokens,
                cfg.n_kv_heads,
                cfg.head_dim(),
                metrics,
            )?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod spark2_5_tests {
    use super::*;
    use allpaca_gguf::{write_gguf, GgufFile, GgufValue};
    use std::collections::HashMap;

    #[test]
    fn spark2_5_config_enables_gelu_gate_and_swa() {
        let path = std::env::temp_dir().join("allpaca_spark2_5_cfg.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("spark2_5".into()),
        );
        meta.insert("spark2_5.block_count".into(), GgufValue::Uint32(8));
        meta.insert("spark2_5.embedding_length".into(), GgufValue::Uint32(64));
        meta.insert(
            "spark2_5.attention.head_count".into(),
            GgufValue::Uint32(4),
        );
        meta.insert(
            "spark2_5.attention.head_count_kv".into(),
            GgufValue::Uint32(2),
        );
        meta.insert(
            "spark2_5.feed_forward_length".into(),
            GgufValue::Uint32(128),
        );
        meta.insert(
            "spark2_5.attention.sliding_window".into(),
            GgufValue::Uint32(512),
        );
        meta.insert(
            "spark2_5.rope.dimension_count".into(),
            GgufValue::Uint32(16),
        );
        meta.insert(
            "spark2_5.rope.dimension_count_swa".into(),
            GgufValue::Uint32(64),
        );
        meta.insert(
            "spark2_5.rope.freq_base".into(),
            GgufValue::Float32(5_000_000.0),
        );
        meta.insert(
            "spark2_5.rope.freq_base_swa".into(),
            GgufValue::Float32(10_000.0),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("<unk>".into())]),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        assert!(cfg.ffn_gelu);
        assert!(cfg.headwise_attn_gate);
        assert_eq!(cfg.sliding_window, 512);
        assert_eq!(cfg.rope_dims_swa, 64);
        assert!((cfg.rope_theta_swa - 10_000.0).abs() < 1.0);
        assert!(cfg.layer_is_swa(0));
        assert!(!cfg.layer_is_swa(3));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gemma2_config_enables_gelu_softcap_and_swa() {
        let path = std::env::temp_dir().join("allpaca_gemma2_cfg.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("gemma2".into()),
        );
        meta.insert("gemma2.block_count".into(), GgufValue::Uint32(4));
        meta.insert("gemma2.embedding_length".into(), GgufValue::Uint32(64));
        meta.insert("gemma2.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert(
            "gemma2.attention.head_count_kv".into(),
            GgufValue::Uint32(2),
        );
        meta.insert("gemma2.feed_forward_length".into(), GgufValue::Uint32(128));
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("<unk>".into())]),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        assert!(cfg.ffn_gelu);
        assert!(cfg.scale_embed);
        assert!((cfg.final_logit_softcap - 30.0).abs() < 1e-3);
        assert!((cfg.attn_logit_softcap - 50.0).abs() < 1e-3);
        assert_eq!(cfg.sliding_window, 4096);
        assert!(cfg.layer_is_swa(0));
        assert!(!cfg.layer_is_swa(1));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn deepseek2_config_reads_mla_ranks() {
        let path = std::env::temp_dir().join("allpaca_deepseek2_cfg.gguf");
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("deepseek2".into()),
        );
        meta.insert("deepseek2.block_count".into(), GgufValue::Uint32(4));
        meta.insert("deepseek2.embedding_length".into(), GgufValue::Uint32(64));
        meta.insert(
            "deepseek2.attention.head_count".into(),
            GgufValue::Uint32(4),
        );
        meta.insert(
            "deepseek2.attention.head_count_kv".into(),
            GgufValue::Uint32(1),
        );
        meta.insert(
            "deepseek2.feed_forward_length".into(),
            GgufValue::Uint32(128),
        );
        meta.insert(
            "deepseek2.attention.q_lora_rank".into(),
            GgufValue::Uint32(16),
        );
        meta.insert(
            "deepseek2.attention.kv_lora_rank".into(),
            GgufValue::Uint32(16),
        );
        meta.insert(
            "deepseek2.leading_dense_block_count".into(),
            GgufValue::Uint32(1),
        );
        meta.insert(
            "deepseek2.expert_shared_count".into(),
            GgufValue::Uint32(2),
        );
        meta.insert(
            "deepseek2.expert_gating_func".into(),
            GgufValue::Uint32(2),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![GgufValue::String("<unk>".into())]),
        );
        write_gguf(&path, &meta, &[]).unwrap();
        let file = GgufFile::open(&path).unwrap();
        let cfg = ModelConfig::from_gguf(&file).unwrap();
        assert_eq!(cfg.q_lora_rank, 16);
        assert_eq!(cfg.kv_lora_rank, 16);
        assert_eq!(cfg.leading_dense_blocks, 1);
        assert_eq!(cfg.shared_experts, 2);
        assert_eq!(cfg.router_kind, RouterKind::SigmoidGroup);
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(all(test, feature = "hip"))]
#[test]
fn streamed_ffn_h2d_matches_host_tensors() {
    if allpaca_hip::device_count() == 0 {
        return;
    }
    let mut model = TinyModel::synthetic(ModelConfig::tiny(), 7, "ring");
    let expect_gate = model.layers[0].w_gate.clone();
    let expect_norm = model.layers[0].ffn_norm.clone();
    let sizes: Vec<usize> = model.layers.iter().map(|l| l.ffn_bytes()).collect();
    let plan = crate::weight_set::DenseTrunkPlan::plan(&sizes, 64, 1);
    assert!(plan.streams(), "tiny trunk must stream FFN");
    model.working_set = Some(WeightWorkingSet::detach_unpinned(&mut model.layers, plan).unwrap());
    let streamed = ResidentModel::upload(&model, 0).unwrap();
    let ffn = model.device_ffn(&streamed, 0, None).unwrap();
    let hip = HipWorkspace::new(0, 1 << 20, 64).unwrap();
    let mut gate = vec![0.0f32; expect_gate.len()];
    let mut norm = vec![0.0f32; expect_norm.len()];
    hip.copy_d2h_f32(streamed.weights.ptr_at(ffn.w_gate().off), &mut gate)
        .unwrap();
    hip.copy_d2h_f32(streamed.weights.ptr_at(ffn.ffn_norm().off), &mut norm)
        .unwrap();
    drop(ffn);
    assert_eq!(norm, expect_norm, "ring ffn_norm H2D");
    assert_eq!(gate, expect_gate, "ring w_gate H2D");
}

#[cfg(test)]
mod lightning_moe_tests {
    use super::*;

    #[test]
    fn softmax_topk_renormalizes_without_lightning_flags() {
        let cfg = ModelConfig::tiny_moe();
        let x = vec![1.0f32; cfg.dim];
        let mut w = vec![0.0f32; cfg.n_experts * cfg.dim];
        for i in 0..cfg.dim {
            w[i] = 1.0;
            w[cfg.dim + i] = 0.5;
        }
        let routed = topk_route(&x, &w, cfg.n_experts, cfg.n_active, &cfg);
        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].0, 0);
        assert_eq!(routed[1].0, 1);
        let sum: f32 = routed.iter().map(|(_, wt)| *wt).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    #[test]
    fn sigmoid_topk_renormalizes_with_clamp() {
        let mut cfg = ModelConfig::tiny_moe();
        cfg.moe_sigmoid_gating = true;
        cfg.expert_weights_norm = true;
        cfg.expert_weights_scale = 1.0;
        let x = vec![1.0f32; cfg.dim];
        let mut w = vec![0.0f32; cfg.n_experts * cfg.dim];
        for i in 0..cfg.dim {
            w[i] = 0.1;
            w[cfg.dim + i] = 0.05;
            w[2 * cfg.dim + i] = -0.1;
            w[3 * cfg.dim + i] = -0.2;
        }
        let routed = topk_route(&x, &w, cfg.n_experts, cfg.n_active, &cfg);
        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].0, 0);
        assert_eq!(routed[1].0, 1);
        let sum: f32 = routed.iter().map(|(_, wt)| *wt).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(routed[0].1 > routed[1].1);
    }

    #[test]
    fn batched_prefill_opts_in_for_moe_without_imrope() {
        let dense = TinyModel::synthetic(ModelConfig::tiny(), 1, "dense");
        assert!(dense.supports_device_batched_prefill());
        let moe = TinyModel::synthetic(ModelConfig::tiny_moe(), 1, "moe");
        assert!(moe.supports_device_batched_prefill());
        let hybrid = TinyModel::synthetic(ModelConfig::tiny_hybrid(), 1, "hybrid");
        assert!(hybrid.supports_device_batched_prefill());
        let mut partial = ModelConfig::tiny();
        partial.rope_dims = 2;
        let partial = TinyModel::synthetic(partial, 1, "partial-rope");
        assert!(
            partial.supports_device_batched_prefill(),
            "partial NeoX RoPE should batch with decode Q/K layout"
        );
        let mut imrope = ModelConfig::tiny();
        imrope.rope_sections = [16, 16, 16, 16];
        let blocked = TinyModel::synthetic(imrope, 1, "imrope");
        assert!(!blocked.supports_device_batched_prefill());
    }

    #[test]
    fn attn_layout_uses_weight_q_out_not_cfg_heads() {
        let mut cfg = ModelConfig::tiny();
        cfg.dim = 64;
        cfg.n_heads = 8;
        cfg.n_kv_heads = 2;
        cfg.head_dim_explicit = 32;
        let al = attn_layout(&cfg, 128 * 64, 64 * 64, 128 * 64);
        assert_eq!(al.q_out, 128);
        assert_eq!(al.wo_k, 128);
        assert_eq!(al.kv_dim, 64);
        assert_eq!(al.head_dim, 32);
        assert_eq!(al.n_heads, 4);
        assert_eq!(al.n_kv_heads, 2);
        assert!(!al.gated);
    }

    #[test]
    fn relu_sqr_gate_less_mlp() {
        let dim = 2usize;
        let ff = 2usize;
        let ffn = LayerFfn {
            ffn_norm: Vec::new(),
            w_gate: Vec::new(),
            w_up: vec![1.0, 0.0, 0.0, 1.0],
            w_down: vec![1.0, 0.0, 0.0, 1.0],
        };
        let x = [2.0f32, -3.0];
        let mut out = [0.0f32; 2];
        expert_mlp(&x, &ffn, ff, dim, &mut out, true).unwrap();
        assert!((out[0] - 4.0).abs() < 1e-5, "{out:?}");
        assert!(out[1].abs() < 1e-5, "{out:?}");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_softcap_tests {
    use super::*;

    #[test]
    fn cuda_final_logit_softcap_bounds_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let mut model = TinyModel::synthetic_softcap(31);
        // Tight cap so raw tiny-model logits exceed it without softcap.
        model.config.final_logit_softcap = 0.05;
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 8192).expect("cuda ws");
        let resident = CudaResidentModel::upload(&model, cuda.device).expect("upload");
        let metrics = Metrics::default();
        let mut kv = crate::kv_tier::KvTierManager::new(8 << 20, 8 << 20, 0.5);
        let pages =
            alloc_pages_for_seq(&mut kv, &model.config, 128, 1, &metrics).expect("pages");
        let mut logits = vec![0.0f32; model.config.vocab_size];
        model
            .forward_token_cuda(
                0,
                0,
                &pages,
                &mut kv,
                &metrics,
                &mut logits,
                &cuda,
                &resident,
                None,
                None,
            )
            .expect("forward");
        let max_abs = logits.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
        assert!(
            max_abs <= 0.05 + 1e-4,
            "final_logit_softcap must bound |logits|, got max_abs={max_abs}"
        );
    }

    fn cuda_decode_n_tokens(model: &TinyModel, n: usize) -> Vec<f32> {
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 8192).expect("cuda ws");
        let resident = CudaResidentModel::upload(model, cuda.device).expect("upload");
        let metrics = Metrics::default();
        let mut kv = crate::kv_tier::KvTierManager::new(8 << 20, 8 << 20, 0.5);
        let mut pages =
            alloc_pages_for_seq(&mut kv, &model.config, 128, n.max(1), &metrics).expect("pages");
        let mut logits = vec![0.0f32; model.config.vocab_size];
        for pos in 0..n {
            ensure_pages(&mut kv, &model.config, 128, &mut pages, pos, &metrics)
                .expect("ensure pages");
            model
                .forward_token_cuda(
                    (pos as u32) % model.config.vocab_size as u32,
                    pos,
                    &pages,
                    &mut kv,
                    &metrics,
                    &mut logits,
                    &cuda,
                    &resident,
                    None,
                    None,
                )
                .expect("forward");
        }
        logits
    }

    #[test]
    fn cuda_swa_window_changes_logits_vs_full_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let full = TinyModel::synthetic(ModelConfig::tiny(), 41, "swa-cmp-full");
        let mut swa = TinyModel::synthetic(ModelConfig::tiny(), 41, "swa-cmp-win");
        swa.config.sliding_window = 1;
        swa.config.swa_pattern = vec![true; swa.config.n_layers];
        let a = cuda_decode_n_tokens(&full, 4);
        let b = cuda_decode_n_tokens(&swa, 4);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "window=1 SWA should diverge from full attention after 4 tokens; max_diff={max_diff}"
        );
    }
}
