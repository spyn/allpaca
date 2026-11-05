use crate::deltanet::{rmsnorm_silu_gate, GdnDims};
use crate::early_exit::EarlyExitSpec;
use crate::edge0::{CrossTokenStager, Edge0Tier, MlaDims, MlaHostWeights, RecoverLora, RouterKind};
use crate::forest::TokenForest;
use crate::ssm::{
    mamba2_step, validate_weights as validate_mamba2_weights, Mamba2Dims, SsmLayerWeights,
};
use allpaca_cpu::{
    attention, attention_softcap, attention_with_mask_softcap, gelu, matmul_nt, rmsnorm, rope,
    rope_imrope_neox, silu,
};
use allpaca_gguf::{GgufCompatReport, GgufFile, HybridLayerInfo, HybridLayerKind};
use allpaca_hip::{copy_dev_f32, DeviceDecode, DeviceWeights, HipWorkspace};
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{HfConfig, SafetensorsCatalog};
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// One sequence's rows in a multi-sequence device forward: `n_rows`
/// consecutive positions from `start_pos`, with the sequence's own KV pages
/// and recurrent (GDN) state. Rows of all sequences share every GEMM and MoE
/// call; RoPE, KV writes, attention and recurrences stay per sequence.
pub struct SeqRows<'a> {
    pub start_pos: usize,
    pub n_rows: usize,
    pub page_ids: &'a [Vec<u64>],
    pub request_id: u64,
    pub recurrent: Option<&'a mut RecurrentState>,
}

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
    /// M-RoPE layout of a Request with images (rotary position != KV position).
    pub mrope: Option<&'a MropeLayout>,
    /// Match host routing and F16 reply rounding of separate remote experts.
    pub separate_moe_compat: bool,
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
            mrope: None,
            separate_moe_compat: false,
        }
    }
}

/// One image's soft tokens in a Request: `cols * rows` tokens from KV
/// position `start`, row-major over the merged patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub cols: usize,
    pub rows: usize,
}

impl ImageSpan {
    pub fn len(&self) -> usize {
        self.cols * self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// M-RoPE positions for a Request with images (llama.cpp `MTMD_POS_TYPE_MROPE`).
/// Image token `i` sits at `(base, base + i / cols, base + i % cols)`, and the
/// image advances the text position by `max(cols, rows)` rather than its token
/// count, so later tokens rotate at their KV position minus the saved span.
/// Attention still indexes KV slots by KV position.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MropeLayout {
    spans: Vec<ImageSpan>,
}

impl MropeLayout {
    /// `spans` in KV order, non-overlapping.
    pub fn new(mut spans: Vec<ImageSpan>) -> Self {
        spans.retain(|s| !s.is_empty());
        spans.sort_by_key(|s| s.start);
        Self { spans }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn spans(&self) -> &[ImageSpan] {
        &self.spans
    }

    /// `(t, h, w)` rotary position for the token at KV position `kv`.
    pub fn rope_pos(&self, kv: usize) -> [usize; 3] {
        let mut shift = 0usize;
        for s in &self.spans {
            if kv < s.start {
                break;
            }
            let base = s.start - shift;
            if kv < s.start + s.len() {
                let i = kv - s.start;
                return [base, base + i / s.cols, base + i % s.cols];
            }
            shift += s.len() - s.cols.max(s.rows);
        }
        [kv - shift; 3]
    }
}

/// A Request's soft (vision) tokens: which positions take encoder rows
/// instead of token embeddings, and the M-RoPE layout they imply.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisionSeq {
    /// Soft token spans in KV order.
    pub spans: Vec<ImageSpan>,
    /// `[soft tokens][dim]` encoder rows, spans in order.
    pub embeds: Vec<f32>,
    /// Empty when the soft tokens use ordinary sequential positions.
    pub mrope: MropeLayout,
}

impl VisionSeq {
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Encoder row replacing the token embedding at KV position `pos`.
    pub fn soft_row(&self, pos: usize, dim: usize) -> Option<&[f32]> {
        let mut row = 0usize;
        for span in &self.spans {
            if (span.start..span.start + span.len()).contains(&pos) {
                let r = row + pos - span.start;
                return self.embeds.get(r * dim..(r + 1) * dim);
            }
            row += span.len();
        }
        None
    }

    pub fn mrope(&self) -> Option<&MropeLayout> {
        (!self.mrope.is_empty()).then_some(&self.mrope)
    }

    /// Whether any soft token falls in KV positions `start..end`.
    pub fn overlaps(&self, start: usize, end: usize) -> bool {
        self.spans.iter().any(|s| s.start < end && start < s.start + s.len())
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
    /// RoPE pairing for full-head rotation: NeoX (`i`, `i + half`) or NORM
    /// (adjacent pairs), per llama.cpp's rope type for the GGUF architecture.
    /// Hugging Face checkpoints (unpermuted Q/K) are always NeoX.
    pub rope_neox: bool,
    /// Attention applies no rotary positions (Nemotron-H: the Mamba layers
    /// carry order; llama.cpp's graph never rotates Q/K there, whatever
    /// `rope.dimension_count` says).
    pub rope_none: bool,
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
    /// Qwen3.5-family attention: the Q projection is `[Q|gate]` per head.
    /// Needed when n_heads * head_dim == dim (Ornith-1.5-9B), where the
    /// shapes alone look like a plain double-width Q.
    pub gated_attn_q: bool,
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
    /// Width of the dense FFN blocks when it differs from the experts'
    /// (`ff_dim`): MoE models with leading dense blocks, e.g. Ling 2.0's
    /// 5120 against 512. 0 = the same as `ff_dim`.
    pub dense_ff_dim: usize,
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
            rope_neox: false,
            rope_none: false,
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
            gated_attn_q: false,
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
            dense_ff_dim: 0,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            moe_sigmoid_gating: false,
            moe_relu_sqr: false,
            moe_sqrt_softplus: false,
            rope_offset: 0,
        }
    }

    pub fn moe_workspace(&self) -> usize {
        self.ff_dim.max(self.shared_ff_dim).max(self.dense_ff_dim)
    }

    /// Hidden width of a dense (non-expert) FFN block.
    pub fn dense_ff(&self) -> usize {
        if self.dense_ff_dim > 0 {
            self.dense_ff_dim
        } else {
            self.ff_dim
        }
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
            rope_neox: gguf_rope_neox(&arch),
            rope_none: gguf_rope_none(&arch),
            // DeepSeek-V4 compressors keep per-sequence state too.
            recurrent: file
                .tensors
                .iter()
                .any(|t| t.name.contains(".ssm_") || t.name.contains(".attn_compressor_kv.")),
            nextn_predict_layers: nextn,
            gguf_layer_base: file.meta_u32("allpaca.gguf_layer_base").unwrap_or(0) as usize,
            router_kind: if arch == "deepseek2"
                && file
                    .meta_u32(&format!("{arch}.expert_gating_func"))
                    .unwrap_or(0)
                    == 2
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
                || file.tensor("blk.0.attn_gate.weight").is_some() && !is_qwen_gdn_arch(&arch),
            gated_attn_q: is_qwen_gdn_arch(&arch),
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
            // `ff_dim` is the expert width when the file has one; dense
            // blocks keep `feed_forward_length`.
            dense_ff_dim: meta_u32_or_array_max(file, &format!("{arch}.feed_forward_length"))
                .map(|v| v as usize)
                .filter(|&v| v != ff_dim)
                .unwrap_or(0),
            expert_weights_norm,
            expert_weights_scale: file
                .meta_f32(&format!("{arch}.expert_weights_scale"))
                .unwrap_or(1.0),
            moe_sigmoid_gating: lightning_moe,
            // Ling 2.0 (`bailingmoe2`) gates like Lightning but its experts are SiLU.
            moe_relu_sqr: lightning_moe && arch != "bailingmoe2",
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

    /// Rotary width when `layer` rotates NeoX-style (`i`, `i + half`), or
    /// `None` for a full-head NORM rotation. Partial rotation and IMRoPE are
    /// always NeoX.
    pub fn neox_rot_dims(&self, layer: usize, head_dim: usize) -> Option<usize> {
        let r = self.rope_dims_for_layer(layer);
        (self.uses_imrope() || (r > 0 && r < head_dim) || self.rope_neox)
            .then_some(if r > 0 { r } else { head_dim })
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
    if let Some(arr) = file
        .metadata
        .get(&key)
        .and_then(allpaca_gguf::GgufValue::as_u32_array)
    {
        let mut out: Vec<bool> = arr.into_iter().take(n_layers).map(|v| v != 0).collect();
        out.resize(n_layers, false);
        return out;
    }
    if let Some(arr) = file
        .metadata
        .get(&key)
        .and_then(allpaca_gguf::GgufValue::as_i32_array)
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
    Q4_0,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    TQ1_0,
    TQ2_0,
    RocmFp4,
    /// ROCmFP4 FAST (ggml type 101): 17-byte single-scale blocks.
    RocmFp4Fast,
    /// MoESQ paired-4:8 sparse FP4 (tensor type 200): 12-byte blocks.
    P48Fp4,
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

    #[cfg(feature = "vulkan")]
    fn f16(off: usize, n: usize) -> Self {
        Self {
            off,
            n,
            kind: WeightKind::F16,
        }
    }

    #[cfg(feature = "cuda")]
    fn q8(off: usize, n: usize) -> Self {
        Self::q8_0(off, n)
    }

    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    fn q8_0(off: usize, n: usize) -> Self {
        Self {
            off,
            n,
            kind: WeightKind::Q8_0,
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
    w_shared_gate: WeightRef,
    w_shared_up: WeightRef,
    w_shared_down: WeightRef,
    /// Qwen3-Next shared-expert gate `ffn_gate_inp_shexp` (`[dim]`).
    shared_gate_inp: WeightRef,
    /// Qwen2 Q/K/V projection biases (zero-sized when absent).
    bq: WeightRef,
    bk: WeightRef,
    bv: WeightRef,
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

/// Packed `token_embd` held in host RAM for quant types with no HIP gather
/// kernel (K-quants, IQ4, TQ, ...). A gather dequantizes `dim` floats per token
/// on the CPU and H2Ds them, instead of parking a vocab×dim F32 table in VRAM
/// (5 GiB for a 248k×5120 Q6_K embed) on every GPU.
struct HostEmbed {
    bytes: Vec<u8>,
    ty: allpaca_gguf::GgmlType,
    dim: usize,
    rows: usize,
    row_bytes: usize,
}

impl HostEmbed {
    /// `token_embd` types the HIP upload would otherwise expand to F32.
    fn wanted(ty: allpaca_gguf::GgmlType) -> bool {
        use allpaca_gguf::GgmlType as T;
        !matches!(
            ty,
            T::F32 | T::F16 | T::Q8_0 | T::RocmFp4 | T::RocmFp4Fast | T::I32 | T::Other
        )
    }

    /// `dim` is the model width: rows are `dim` elements whichever order the
    /// file lists the dims in (`[dim, vocab]` usually, `[vocab, dim]` in some
    /// writers).
    fn from_gguf(file: &GgufFile, name: &str, dim: usize) -> Result<Self, AllpacaError> {
        let info = file.tensor(name).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
        })?;
        let n = info.n_elements() as usize;
        let bytes = file.tensor_bytes(name)?;
        if dim == 0 || n % dim != 0 || bytes.len() % (n / dim).max(1) != 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "host embed {name} dims {:?} do not tile {} bytes",
                info.dims,
                bytes.len()
            ))));
        }
        let rows = n / dim;
        Ok(Self {
            row_bytes: bytes.len() / rows,
            bytes: bytes.to_vec(),
            ty: info.ggml_type,
            dim,
            rows,
        })
    }

    fn gather(&self, tokens: &[u32], out: &mut Vec<f32>) -> Result<(), AllpacaError> {
        out.clear();
        out.resize(tokens.len() * self.dim, 0.0);
        for (t, row) in tokens.iter().zip(out.chunks_exact_mut(self.dim)) {
            let r = *t as usize;
            if r >= self.rows {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "token {r} outside embed rows {}",
                    self.rows
                ))));
            }
            let src = &self.bytes[r * self.row_bytes..(r + 1) * self.row_bytes];
            allpaca_gguf::dequant_packed(self.ty, src, self.dim, row)?;
        }
        Ok(())
    }
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
    /// Grow-only scratch for the fused MoE kernels (gate/up/down outputs).
    moe_scratch: Mutex<Option<DeviceWeights>>,
    /// Pinned staging the GPU reads in place: MoE pair tables and remote
    /// expert sums. Rewritten only after the next layer's readback has
    /// synchronized the stream, so kernels never see it change underneath.
    moe_stage: Mutex<allpaca_hip::PinnedBuf>,
    /// This GPU's copies of the expert controls the router kernel reads.
    route_ctl: Mutex<DeviceRouteCtl>,
    embed: WeightRef,
    /// Set when `token_embd` stays packed on the host (see [`HostEmbed`]).
    host_embed: Option<HostEmbed>,
    layers: Vec<LayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
    mtp_nextn: Option<MtpNextnOff>,
}

/// Device side of [`crate::expert_controls::ExpertControls`] on one GPU:
/// per-layer disabled masks (n_exp bytes, re-uploaded when the controls'
/// generation moves) and routing-hit counts (`n_layers * n_exp` u32,
/// allocated on first use, zeroed when the usage epoch moves).
#[derive(Default)]
struct DeviceRouteCtl {
    mask_gen: Option<u64>,
    masks: Vec<Option<DeviceWeights>>,
    counts: Option<DeviceWeights>,
    counts_epoch: u64,
    shape: crate::expert_controls::ExpertShape,
}

/// A layer's stacked routed experts as the fused MoE kernels read them:
/// `[gate, up, down]`, each Q4_K/Q5_K/Q6_K, staged Q8_0 or paired-4:8 FP4
/// at a fixed per-expert stride.
fn stacked_moe_experts(
    weights: &DeviceWeights,
    offs: &[[WeightRef; 3]],
) -> Option<[allpaca_hip::MoeExperts; 3]> {
    if offs.len() < 2 {
        return None;
    }
    let last = offs.len() - 1;
    let mut out = [allpaca_hip::MoeExperts {
        ptr: std::ptr::null(),
        ggml_type: 0,
        stride: 0,
    }; 3];
    for (j, slot) in out.iter_mut().enumerate() {
        let (first, second) = (offs[0][j], offs[1][j]);
        let ggml_type = match first.kind {
            WeightKind::Q4_K => 12,
            WeightKind::Q5_K => 13,
            WeightKind::Q6_K => 14,
            WeightKind::Q8_0 => 8,
            WeightKind::P48Fp4 => 200,
            _ => return None,
        };
        let stride = second.off.checked_sub(first.off)?;
        if first.n == 0 || offs[last][j].off != first.off + last * stride {
            return None;
        }
        *slot = allpaca_hip::MoeExperts {
            ptr: weights.ptr_at(first.off).cast(),
            ggml_type,
            stride: stride as u64,
        };
    }
    allpaca_hip::MoeExperts::fused_set_ok(&out).then_some(out)
}

impl ResidentModel {
    /// Router-kernel pointers for `layer`: its disabled mask (null when none)
    /// and its slice of the hit counts (null unless recording).
    fn route_ctl_ptrs(
        &self,
        ctl: &crate::expert_controls::ExpertControls,
        layer: usize,
    ) -> Result<(*const u8, *mut u32), AllpacaError> {
        let shape = ctl.shape();
        if shape.n_experts == 0 {
            return Ok((std::ptr::null(), std::ptr::null_mut()));
        }
        let mut g = self
            .route_ctl
            .lock()
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidConfig("route ctl lock".into())))?;
        if g.shape != shape {
            *g = DeviceRouteCtl {
                shape,
                ..DeviceRouteCtl::default()
            };
        }
        let gen = ctl.generation();
        if g.mask_gen != Some(gen) {
            let mut masks = Vec::with_capacity(shape.n_layers);
            for l in 0..shape.n_layers {
                masks.push(match ctl.layer_mask(l) {
                    Some(mask) => {
                        let bytes: Vec<u8> = mask.iter().map(|&d| u8::from(d)).collect();
                        let dw = DeviceWeights::alloc(self.weights.device(), bytes.len().max(256))?;
                        dw.copy_h2d(0, &bytes)?;
                        Some(dw)
                    }
                    None => None,
                });
            }
            g.masks = masks;
            g.mask_gen = Some(gen);
        }
        let disabled = g
            .masks
            .get(layer)
            .and_then(Option::as_ref)
            .map_or(std::ptr::null(), |m| m.ptr_at(0) as *const u8);
        let counts = if ctl.recording() {
            let epoch = ctl.usage_epoch();
            let bytes = shape.n_layers * shape.n_experts * 4;
            if g.counts.is_none() {
                g.counts = Some(DeviceWeights::alloc(self.weights.device(), bytes.max(256))?);
                g.counts_epoch = epoch.wrapping_sub(1);
            }
            if g.counts_epoch != epoch {
                g.counts.as_ref().expect("counts").copy_h2d(0, &vec![0u8; bytes])?;
                g.counts_epoch = epoch;
            }
            let base = g.counts.as_ref().expect("counts").ptr_at(0) as *mut u32;
            unsafe { base.add(layer * shape.n_experts) }
        } else {
            std::ptr::null_mut()
        };
        Ok((disabled, counts))
    }

    /// This GPU's routing-hit counts (`n_layers * n_experts`), zeros when it
    /// has not counted since the last reset.
    pub(crate) fn device_expert_usage(
        &self,
        ctl: &crate::expert_controls::ExpertControls,
    ) -> Result<Vec<u64>, AllpacaError> {
        let shape = ctl.shape();
        let n = shape.n_layers * shape.n_experts;
        let g = self
            .route_ctl
            .lock()
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidConfig("route ctl lock".into())))?;
        let Some(counts) = g.counts.as_ref().filter(|_| g.counts_epoch == ctl.usage_epoch() && g.shape == shape) else {
            return Ok(vec![0; n]);
        };
        let mut raw = vec![0u8; n * 4];
        counts.copy_d2h(0, &mut raw)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| u64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect())
    }

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
                w_shared_gate: if layer.shared_expert.w_gate.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.shared_expert.w_gate.as_slice())
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
                shared_gate_inp: if layer.shared_gate_inp.is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.shared_gate_inp.as_slice())
                },
                bq: if layer.attn_bias[0].is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.attn_bias[0].as_slice())
                },
                bk: if layer.attn_bias[1].is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.attn_bias[1].as_slice())
                },
                bv: if layer.attn_bias[2].is_empty() {
                    WeightRef::f32(0, 0)
                } else {
                    push(&mut slices, &mut off, layer.attn_bias[2].as_slice())
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
            moe_scratch: Mutex::new(None),
            moe_stage: Mutex::new(allpaca_hip::PinnedBuf::default()),
            route_ctl: Mutex::new(DeviceRouteCtl::default()),
            embed,
            host_embed: None,
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
        Self::upload_gguf_filtered(file, cfg, device, &[], &[])
    }

    /// `skip_remote[local]` leaves that layer's weights on the RPC worker;
    /// `skip_experts[local]` leaves only its routed experts there.
    pub fn upload_gguf_filtered(
        file: &GgufFile,
        cfg: &ModelConfig,
        device: i32,
        skip_remote: &[bool],
        skip_experts: &[bool],
    ) -> Result<Self, AllpacaError> {
        Self::upload_gguf_split(file, cfg, device, skip_remote, skip_experts, true)
    }

    /// Layer weights only, for a second local GPU in the layer split.
    ///
    /// Embed, final norm, lm_head, and MTP NextN stay on the target GPU:
    /// decode switches back there before `output_norm`, so a copy here is
    /// VRAM the peer's layers cannot use.
    pub fn upload_gguf_peer_layers(
        file: &GgufFile,
        cfg: &ModelConfig,
        device: i32,
        skip: &[bool],
        skip_experts: &[bool],
    ) -> Result<Self, AllpacaError> {
        Self::upload_gguf_split(file, cfg, device, skip, skip_experts, false)
    }

    fn upload_gguf_split(
        file: &GgufFile,
        cfg: &ModelConfig,
        device: i32,
        skip_remote: &[bool],
        skip_experts: &[bool],
        globals: bool,
    ) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        let mut parts: Vec<UploadPart<'_>> = Vec::new();
        let mut off = 0usize;
        let none = WeightRef::f32(0, 0);
        let embed_ty = file.tensor("token_embd.weight").map(|t| t.ggml_type);
        let host_embed = match embed_ty {
            Some(ty) if globals && HostEmbed::wanted(ty) => {
                Some(HostEmbed::from_gguf(file, "token_embd.weight", cfg.dim)?)
            }
            _ => None,
        };
        let embed = if !globals || host_embed.is_some() {
            none
        } else {
            append_gguf_tensor(file, "token_embd.weight", &mut parts, &mut off)?
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut expert_offs = Vec::with_capacity(cfg.n_layers);
        for local in 0..cfg.n_layers {
            let i = cfg.gguf_blk(local);
            if skip_remote.get(local).copied().unwrap_or(false) {
                expert_offs.push(Vec::new());
                layers.push(LayerOff {
                    attn_norm: WeightRef::f32(0, 0),
                    wq: WeightRef::f32(0, 0),
                    wk: WeightRef::f32(0, 0),
                    wv: WeightRef::f32(0, 0),
                    wo: WeightRef::f32(0, 0),
                    ffn_norm: WeightRef::f32(0, 0),
                    w_gate: WeightRef::f32(0, 0),
                    w_up: WeightRef::f32(0, 0),
                    w_down: WeightRef::f32(0, 0),
                    w_router: WeightRef::f32(0, 0),
                    w_qkv: WeightRef::f32(0, 0),
                    w_z: WeightRef::f32(0, 0),
                    w_alpha: WeightRef::f32(0, 0),
                    w_beta: WeightRef::f32(0, 0),
                    w_ssm_in: WeightRef::f32(0, 0),
                    w_ssm_out: WeightRef::f32(0, 0),
                    q_norm: WeightRef::f32(0, 0),
                    k_norm: WeightRef::f32(0, 0),
                    attn_out_gate: WeightRef::f32(0, 0),
                    w_shared_gate: WeightRef::f32(0, 0),
                    w_shared_up: WeightRef::f32(0, 0),
                    w_shared_down: WeightRef::f32(0, 0),
                    shared_gate_inp: WeightRef::f32(0, 0),
                    bq: WeightRef::f32(0, 0),
                    bk: WeightRef::f32(0, 0),
                    bv: WeightRef::f32(0, 0),
                });
                continue;
            }
            let reported = report
                .layers
                .iter()
                .find(|layer| layer.index == i)
                .map(|layer| layer.block)
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
                        let (gdn_alpha, gdn_beta) =
                            append_gguf_gdn_alpha_beta(file, i, &mut parts, &mut off)?;
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
                            gdn_alpha,
                            gdn_beta,
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
                let packed_ex = if skip_experts.get(local).copied().unwrap_or(false) {
                    Vec::new()
                } else {
                    append_gguf_experts(file, i, cfg.n_experts, &mut parts, &mut off)?
                };
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
                && file.tensor(&format!("blk.{i}.attn_gate.weight")).is_some()
                && matches!(
                    block,
                    HybridLayerKind::Attention | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
                ) {
                append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_gate.weight"),
                    &mut parts,
                    &mut off,
                )?
            } else {
                WeightRef::f32(0, 0)
            };
            let (w_shared_gate, w_shared_up, w_shared_down) =
                append_gguf_shared_expert(file, i, &mut parts, &mut off)?;
            let (bq, bk, bv) = (
                append_gguf_bias(file, i, "attn_q", &mut parts, &mut off)?,
                append_gguf_bias(file, i, "attn_k", &mut parts, &mut off)?,
                append_gguf_bias(file, i, "attn_v", &mut parts, &mut off)?,
            );
            let shexp_gate_name = format!("blk.{i}.ffn_gate_inp_shexp.weight");
            let shared_gate_inp = if file.tensor(&shexp_gate_name).is_some() {
                append_gguf_tensor(file, &shexp_gate_name, &mut parts, &mut off)?
            } else {
                none
            };
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
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                shared_gate_inp,
                bq,
                bk,
                bv,
            });
        }
        let norm = if !globals {
            none
        } else if file.tensor("output_norm.weight").is_some() {
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
        let lm_head = if !globals {
            none
        } else if file.tensor("output.weight").is_some() {
            append_gguf_lm_head(file, cfg, "output.weight", &mut parts, &mut off)?
        } else if host_embed.is_some() || use_q4_0_draft_head(cfg) {
            // Tied weights, host gather: lm_head still needs the table on
            // device, packed in GEMV layout rather than an F32 expand.
            append_gguf_lm_head(file, cfg, "token_embd.weight", &mut parts, &mut off)?
        } else {
            embed
        };
        let mtp_base = cfg.gguf_blk(0);
        let mtp_nextn = if globals
            && file
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
            moe_scratch: Mutex::new(None),
            moe_stage: Mutex::new(allpaca_hip::PinnedBuf::default()),
            route_ctl: Mutex::new(DeviceRouteCtl::default()),
            weights,
            embed,
            host_embed,
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

    pub fn touch_resident(&self) -> Result<(), AllpacaError> {
        self.weights.touch()
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
            | WeightKind::Q4_0
            | WeightKind::Q3_K
            | WeightKind::Q4_K
            | WeightKind::Q5_K
            | WeightKind::Q6_K
            | WeightKind::TQ1_0
            | WeightKind::TQ2_0
            | WeightKind::RocmFp4
            | WeightKind::RocmFp4Fast
            | WeightKind::P48Fp4 => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "F16/Q4_0/Q3_K/Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0/ROCmFP4 norm must be F32 at load".into(),
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
            WeightKind::Q4_0 => {
                hip.gemm_nt_q4_0_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::Q3_K => {
                hip.gemm_nt_q3_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
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
            WeightKind::RocmFp4Fast => {
                hip.gemm_nt_rocmfp4_fast_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::P48Fp4 => {
                hip.gemm_nt_p48fp4_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
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
        if let Some(host) = &self.host_embed {
            let mut row = Vec::new();
            host.gather(&[token], &mut row)?;
            return hip.copy_h2d_f32(out, &row);
        }
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
            WeightKind::RocmFp4Fast => hip.embed_gather_rocmfp4_fast(
                self.weights.byte_ptr_at(self.embed.off),
                token,
                out,
                dim,
            ),
            WeightKind::P48Fp4
            | WeightKind::Q4_0
            | WeightKind::Q3_K
            | WeightKind::Q4_K
            | WeightKind::Q5_K
            | WeightKind::Q6_K
            | WeightKind::TQ1_0
            | WeightKind::TQ2_0 => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "Q4_0/Q3_K/Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0 embed must be F32 at load".into(),
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
        if let Some(host) = &self.host_embed {
            let mut rows = Vec::new();
            host.gather(tokens, &mut rows)?;
            return hip.copy_h2d_f32(out, &rows);
        }
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
            WeightKind::RocmFp4Fast => hip.embed_gather_rocmfp4_fast_rows(
                self.weights.byte_ptr_at(self.embed.off),
                tokens,
                out,
                dim,
            ),
            WeightKind::P48Fp4
            | WeightKind::Q4_0
            | WeightKind::Q3_K
            | WeightKind::Q4_K
            | WeightKind::Q5_K
            | WeightKind::Q6_K
            | WeightKind::TQ1_0
            | WeightKind::TQ2_0 => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "Q4_0/Q3_K/Q4_K/Q5_K/Q6_K/TQ1_0/TQ2_0 embed must be F32 at load".into(),
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

    /// Token embeddings into device rows `out` (`[tokens, dim]`) on `hip`.
    pub(crate) fn embed_rows_device(
        &self,
        hip: &HipWorkspace,
        tokens: &[u32],
        out: *mut f32,
        dim: usize,
    ) -> Result<(), AllpacaError> {
        self.embed_gather_rows(hip, tokens, out, dim as i32)
    }

    /// `out[rows, vocab] = a[rows, dim] · lm_headᵀ` on `hip` (the target GPU).
    pub(crate) fn lm_head_rows_device(
        &self,
        hip: &HipWorkspace,
        a: *const f32,
        out: *mut f32,
        rows: usize,
        vocab: usize,
        dim: usize,
    ) -> Result<(), AllpacaError> {
        self.gemm_nt(
            hip,
            a,
            self.lm_head,
            out,
            rows as i32,
            vocab as i32,
            dim as i32,
        )
    }
}

fn relu_sqr(v: f32) -> f32 {
    if v > 0.0 {
        v * v
    } else {
        0.0
    }
}

/// Host sigmoid with `det_expf`: the platform `exp` (MSVC CRT vs glibc)
/// differs in the last bit for some inputs, which split hosted RPC spans
/// from the separate path. Same bits on every coordinator and worker.
fn sigmoid_stable(v: f32) -> f32 {
    if v >= 0.0 {
        let z = allpaca_cpu::det_expf(-v);
        1.0 / (1.0 + z)
    } else {
        let z = allpaca_cpu::det_expf(v);
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

/// What an expert's down projection reads for input `x`: its gated
/// activations ([`expert_mlp`] without the last product).
pub(crate) fn expert_activations(
    x: &[f32],
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    relu_sqr_act: bool,
) -> Result<Vec<f32>, AllpacaError> {
    let mut up = vec![0.0f32; ff_dim];
    matmul_nt(x, &ffn.w_up, &mut up, 1, ff_dim, dim)?;
    if ffn.w_gate.is_empty() {
        let act = if relu_sqr_act { relu_sqr } else { silu };
        up.iter_mut().for_each(|v| *v = act(*v));
        return Ok(up);
    }
    let mut gate = vec![0.0f32; ff_dim];
    matmul_nt(x, &ffn.w_gate, &mut gate, 1, ff_dim, dim)?;
    for (g, u) in gate.iter_mut().zip(&up) {
        *g = silu(*g) * u;
    }
    Ok(gate)
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
    mask: Option<&[bool]>,
) -> Vec<(usize, f32)> {
    let n_exp = cfg.n_experts.max(1);
    let n_act = cfg.n_active.max(1).min(n_exp);
    moe_route_logits(
        x,
        mask,
        &layer.w_router,
        n_exp,
        n_act,
        cfg.router_kind,
        cfg.moe_norm_topk,
        if cfg.n_group > 0 {
            Some(cfg.n_group)
        } else {
            None
        },
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
    mask: Option<&[bool]>,
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
    // Disabled experts: -inf logit (softmax, plain sigmoid top-k) and -inf
    // selection bias (group routing selects on sigmoid + bias).
    let masked_bias: Option<Vec<f32>> = mask.map(|mask| {
        for (l, off) in logits.iter_mut().zip(mask) {
            if *off {
                *l = f32::NEG_INFINITY;
            }
        }
        (0..n_experts)
            .map(|e| {
                if mask.get(e).copied().unwrap_or(false) {
                    f32::NEG_INFINITY
                } else {
                    expert_bias.and_then(|b| b.get(e)).copied().unwrap_or(0.0)
                }
            })
            .collect()
    });
    crate::edge0::route_logits(
        kind,
        &logits,
        n_active,
        norm,
        n_group,
        topk_group,
        routed_scaling,
        masked_bias.as_deref().or(expert_bias),
    )
}

pub(crate) fn topk_route(
    x: &[f32],
    w_router: &[f32],
    n_experts: usize,
    n_active: usize,
    cfg: &ModelConfig,
    mask: Option<&[bool]>,
    select_bias: &[f32],
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
    topk_from_logits(&mut weights, n_active, cfg, mask, select_bias)
}

/// `mask[e]` = expert `e` is disabled: it takes no part in the softmax and
/// is never picked (scores are >= 0, a disabled one is set to -1). `None`
/// computes exactly the unmasked routing.
///
/// `select_bias` (`blk.N.exp_probs_b`, empty = none) is DeepSeek-V3's expert
/// selection bias for sigmoid routers: experts are picked on
/// `sigmoid(logit) + bias`, and weighted by the unbiased sigmoid.
/// The weight scale of [`topk_from_logits`] when the routing is the plain
/// kind a device kernel also does (softmax over all experts, the best by
/// weight, renormalised over the picks, no groups); `None` otherwise.
pub(crate) fn plain_softmax_routing(cfg: &ModelConfig) -> Option<f32> {
    let plain = !cfg.moe_sqrt_softplus
        && !cfg.moe_sigmoid_gating
        && !cfg.expert_weights_norm
        && cfg.n_expert_groups <= 1
        && cfg.n_active >= 1;
    let scale = if cfg.expert_weights_scale == 0.0 { 1.0 } else { cfg.expert_weights_scale };
    plain.then_some(scale)
}

pub(crate) fn topk_from_logits(
    weights: &mut [f32],
    n_active: usize,
    cfg: &ModelConfig,
    mask: Option<&[bool]>,
    select_bias: &[f32],
) -> Vec<(usize, f32)> {
    let n_experts = weights.len();
    let off = |e: usize| mask.is_some_and(|m| m.get(e).copied().unwrap_or(false));
    if cfg.moe_sqrt_softplus {
        for w in weights.iter_mut() {
            *w = sqrt_softplus_moe(*w);
        }
    } else if cfg.moe_sigmoid_gating {
        for w in weights.iter_mut() {
            *w = sigmoid_stable(*w);
        }
    } else if mask.is_some() {
        let mut max = f32::NEG_INFINITY;
        for (e, w) in weights.iter().enumerate() {
            if !off(e) {
                max = max.max(*w);
            }
        }
        let mut sum = 0.0f32;
        for (e, w) in weights.iter_mut().enumerate() {
            *w = if off(e) { 0.0 } else { allpaca_cpu::det_expf(*w - max) };
            sum += *w;
        }
        let sum = sum.max(1e-9);
        for w in weights.iter_mut() {
            *w /= sum;
        }
    } else {
        let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for w in weights.iter_mut() {
            *w = allpaca_cpu::det_expf(*w - max);
            sum += *w;
        }
        let sum = sum.max(1e-9);
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    if mask.is_some() {
        for (e, w) in weights.iter_mut().enumerate() {
            if off(e) {
                *w = -1.0;
            }
        }
    }
    let biased = cfg.moe_sigmoid_gating && !cfg.moe_sqrt_softplus && !select_bias.is_empty();
    let score = |e: usize| {
        if !biased {
            weights[e]
        } else if off(e) {
            f32::NEG_INFINITY
        } else {
            weights[e] + select_bias.get(e).copied().unwrap_or(0.0)
        }
    };
    let kept = expert_groups_kept(n_experts, cfg.n_expert_groups, cfg.n_group_used, &score);
    let mut idx: Vec<usize> = match &kept {
        Some((per_group, keep)) => (0..n_experts).filter(|e| keep[e / per_group]).collect(),
        None => (0..n_experts).collect(),
    };
    idx.sort_by(|&a, &b| {
        score(b)
            .partial_cmp(&score(a))
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

/// Two-tier (group-limited) routing, as DeepSeek-V3 and Ling 2.0 train it:
/// the experts form `n_groups` equal consecutive groups, a group scores the
/// sum of its two best selection scores, and only the `n_used` best groups
/// may supply experts. Returns `(experts per group, kept flag per group)`,
/// or `None` when the model routes flat (one group, all groups used, or a
/// group count that does not divide the experts).
pub(crate) fn expert_groups_kept(
    n_experts: usize,
    n_groups: usize,
    n_used: usize,
    score: &impl Fn(usize) -> f32,
) -> Option<(usize, Vec<bool>)> {
    if n_groups < 2 || n_used == 0 || n_used >= n_groups || n_experts % n_groups != 0 {
        return None;
    }
    let per_group = n_experts / n_groups;
    let group_score: Vec<f32> = (0..n_groups)
        .map(|g| {
            let mut top = [f32::NEG_INFINITY; 2];
            for e in g * per_group..(g + 1) * per_group {
                let v = score(e);
                if v > top[0] {
                    top = [v, top[0]];
                } else if v > top[1] {
                    top[1] = v;
                }
            }
            // A group of one expert (or one left enabled) scores that expert.
            if top[1] == f32::NEG_INFINITY {
                top[0]
            } else {
                top[0] + top[1]
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..n_groups).collect();
    order.sort_by(|&a, &b| {
        group_score[b]
            .partial_cmp(&group_score[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let mut keep = vec![false; n_groups];
    for &g in order.iter().take(n_used) {
        keep[g] = true;
    }
    Some((per_group, keep))
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
        attn_sinks: Vec::new(),
        attn_post_norm: Vec::new(),
        shared_gate_inp: Vec::new(),
        attn_bias: Default::default(),
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




/// llama.cpp's NeoX rope architectures (`LLAMA_ROPE_TYPE_NEOX`). GGUF
/// conversion permutes Q/K for the others (Llama, DeepSeek2, ...) so their
/// adjacent-pair NORM rotation matches.
fn gguf_rope_neox(arch: &str) -> bool {
    matches!(
        arch,
        "qwen2" | "qwen2moe" | "qwen3" | "qwen3moe" | "qwen35" | "qwen35moe" | "qwen3next" | "qwen4exp"
            | "phi2"
            | "phi3"
            | "gemma" | "gemma2" | "gemma3" | "gemma4" | "starcoder2" | "stablelm" | "gptneox" | "falcon"
            | "olmo2" | "nemotron" | "bailingmoe2"
    )
}

/// Architectures whose llama.cpp graph feeds Q/K to attention unrotated.
fn gguf_rope_none(arch: &str) -> bool {
    matches!(arch, "nemotron_h" | "nemotron_h_moe")
}

/// Qwen2-family Q/K/V projection biases (`blk.N.attn_{q,k,v}.bias`) onto
/// host layers. Without them every Qwen2/Qwen2.5 GGUF decoded garbage.
fn attach_gguf_attn_bias(
    layers: &mut [LayerWeights],
    file: &GgufFile,
    cfg: &ModelConfig,
) -> Result<(), AllpacaError> {
    for (local, layer) in layers.iter_mut().enumerate() {
        let i = cfg.gguf_blk(local);
        for (slot, part) in ["attn_q", "attn_k", "attn_v"].iter().enumerate() {
            let name = format!("blk.{i}.{part}.bias");
            if file.tensor(&name).is_some() {
                layer.attn_bias[slot] = file.tensor_f32(&name)?;
            }
        }
    }
    Ok(())
}

/// Upload `blk.{layer}.{part}.bias` when present (zero-sized ref otherwise).
fn append_gguf_bias<'a>(
    file: &'a GgufFile,
    layer: usize,
    part: &str,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
) -> Result<WeightRef, AllpacaError> {
    let name = format!("blk.{layer}.{part}.bias");
    if file.tensor(&name).is_some() {
        append_gguf_tensor(file, &name, parts, off)
    } else {
        Ok(WeightRef::f32(0, 0))
    }
}

/// Add a projection bias to each `bias.len()`-wide row of `x` (no-op when empty).
fn add_row_bias(x: &mut [f32], bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    for row in x.chunks_exact_mut(bias.len()) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// Qwen Gated DeltaNet checkpoints: Qwen3.5 (`qwen35`), its MoE variant
/// (`qwen35moe`: the same trunk with Qwen3-Next's routed + shared experts)
/// and Qwen3-Next (`qwen3next`).
pub(crate) fn is_qwen_gdn_arch(arch: &str) -> bool {
    matches!(arch, "qwen35" | "qwen35moe" | "qwen3next")
}

/// GDN dims from GGUF meta. Qwen3-Next repeat-interleaves K heads; Qwen3.5 tiles.
fn gguf_gdn_dims(file: &GgufFile, layer: usize) -> Result<GdnDims, AllpacaError> {
    let arch = file.architecture().unwrap_or("qwen35");
    let mut dims = GdnDims::from_meta(
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
    if arch == "qwen3next" {
        dims.k_repeat = crate::deltanet::KHeadRepeat::Interleave;
    }
    Ok(dims)
}

/// Qwen3-Next packs beta and alpha into one `ssm_ba` (`[dim, 2 * n_v]`): K
/// group `g` holds `r = n_v / n_k` beta rows, then `r` alpha rows, for V
/// heads `g * r ..`. Returns `(beta, alpha)`, each row-major `[n_v][dim]`.
fn split_ssm_ba(ba: &[f32], dim: usize, n_k: usize, n_v: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if n_k == 0 || dim == 0 || n_v % n_k != 0 || ba.len() != 2 * n_v * dim {
        return None;
    }
    let r = n_v / n_k;
    let row = |i: usize| &ba[i * dim..(i + 1) * dim];
    let mut beta = Vec::with_capacity(n_v * dim);
    let mut alpha = Vec::with_capacity(n_v * dim);
    for g in 0..n_k {
        for j in 0..r {
            beta.extend_from_slice(row(g * 2 * r + j));
        }
        for j in 0..r {
            alpha.extend_from_slice(row(g * 2 * r + r + j));
        }
    }
    Some((beta, alpha))
}

/// GDN `(beta, alpha)` projections as host F32, from `ssm_beta` / `ssm_alpha`
/// or Qwen3-Next's packed `ssm_ba`.
fn gguf_gdn_beta_alpha(
    file: &GgufFile,
    layer: usize,
    dims: &GdnDims,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let ba_name = format!("blk.{layer}.ssm_ba.weight");
    if file.tensor(&ba_name).is_none() {
        return Ok((
            file.tensor_f32(&format!("blk.{layer}.ssm_beta.weight"))?,
            file.tensor_f32(&format!("blk.{layer}.ssm_alpha.weight"))?,
        ));
    }
    let ba = file.tensor_f32(&ba_name)?;
    let dim = ba.len() / (2 * dims.n_v_heads).max(1);
    split_ssm_ba(&ba, dim, dims.n_k_heads, dims.n_v_heads).ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "blk.{layer}: ssm_ba has {} values, not 2 x {} heads x dim",
            ba.len(),
            dims.n_v_heads
        )))
    })
}

/// Device `(alpha, beta)` for a GDN layer. Packed `ssm_ba` splits on the host
/// into two small F32 tensors.
fn append_gguf_gdn_alpha_beta<'a>(
    file: &'a GgufFile,
    layer: usize,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
) -> Result<(WeightRef, WeightRef), AllpacaError> {
    if file.tensor(&format!("blk.{layer}.ssm_ba.weight")).is_none() {
        return Ok((
            append_gguf_tensor(file, &format!("blk.{layer}.ssm_alpha.weight"), parts, off)?,
            append_gguf_tensor(file, &format!("blk.{layer}.ssm_beta.weight"), parts, off)?,
        ));
    }
    let dims = gguf_gdn_dims(file, layer)?;
    let (beta, alpha) = gguf_gdn_beta_alpha(file, layer, &dims)?;
    Ok((append_f32_part(&alpha, parts, off), append_f32_part(&beta, parts, off)))
}

/// Upload host F32 values as their own 16-byte aligned part.
fn append_f32_part(data: &[f32], parts: &mut Vec<UploadPart<'_>>, off: &mut usize) -> WeightRef {
    if *off % 16 != 0 {
        let pad = 16 - (*off % 16);
        parts.push(UploadPart::Owned(vec![0u8; pad]));
        *off += pad;
    }
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let w = WeightRef::f32(*off, data.len());
    *off += bytes.len();
    parts.push(UploadPart::Owned(bytes));
    w
}

/// `materialize_projections` expands qkv/z/alpha/beta/out to host F32 (CPU
/// and host-weight CUDA paths). Device-resident loads upload them from the
/// file instead, so they stay empty (`gdn_needs_device_projections`).
fn load_gguf_gdn_host(
    file: &GgufFile,
    layer: usize,
    materialize_projections: bool,
) -> Result<GdnHostWeights, AllpacaError> {
    let dims = gguf_gdn_dims(file, layer)?;
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
    if !materialize_projections {
        return Ok(GdnHostWeights {
            dims,
            a,
            dt_bias,
            norm,
            conv1d,
            w_qkv: Vec::new(),
            w_z: Vec::new(),
            w_alpha: Vec::new(),
            w_beta: Vec::new(),
            w_out: Vec::new(),
        });
    }
    // Host projections for CUDA / CPU host-step GDN (HIP also uploads these on-device).
    let w_qkv = file.tensor_f32(&format!("blk.{layer}.attn_qkv.weight"))?;
    let w_z = file.tensor_f32(&format!("blk.{layer}.attn_gate.weight"))?;
    let (w_beta, w_alpha) = gguf_gdn_beta_alpha(file, layer, &dims)?;
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
    // DeepSeek-V4 MLA lives in `Dsv4Weights` (`attn_kv.weight` + output LoRA).
    // The DeepSeek-V2 host pack (`attn_kv_a_mqa` + dense `attn_output`) is absent.
    if matches!(file.architecture(), Some("deepseek4" | "deepseek_v4"))
        || (crate::deepseek4::file_is_deepseek4(file)
            && file
                .tensor(&format!("blk.{layer}.attn_kv_a_mqa.weight"))
                .is_none())
    {
        return Ok(None);
    }
    if file
        .tensor(&format!("blk.{layer}.attn_q_a.weight"))
        .is_none()
        && file
            .tensor(&format!("blk.{layer}.attn_kv_a_mqa.weight"))
            .is_none()
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
    let wukv = if file
        .tensor(&format!("blk.{layer}.attn_kv_b.weight"))
        .is_some()
    {
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
    // llama.cpp writes `blk.N.exp_probs_b.bias`; the `ffn_` spellings are
    // older converters.
    for name in ["exp_probs_b.bias", "ffn_exp_probs_b.bias", "ffn_exp_probs_b.weight"] {
        let name = format!("blk.{layer}.{name}");
        if file.tensor(&name).is_some() {
            return file.tensor_f32(&name);
        }
    }
    Ok(Vec::new())
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

/// The router of an MoE layer and an empty slot per expert: the experts
/// themselves stay packed ([`crate::host_experts::PackedExperts`]).
fn load_gguf_router_only(file: &GgufFile, layer: usize, n_experts: usize) -> Result<(Vec<f32>, Vec<LayerFfn>), AllpacaError> {
    let w_router = file
        .tensor_f32(&format!("blk.{layer}.ffn_gate_inp.weight"))
        .or_else(|_| file.tensor_f32(&format!("blk.{layer}.ffn_router.weight")))?;
    Ok((w_router, vec![LayerFfn::default(); n_experts]))
}

/// The shared expert's sigmoid gate (`ffn_gate_inp_shexp`, one weight per
/// input), or empty when the layer has none.
fn load_gguf_shared_gate(file: &GgufFile, layer: usize) -> Result<Vec<f32>, AllpacaError> {
    let name = format!("blk.{layer}.ffn_gate_inp_shexp.weight");
    if file.tensor(&name).is_none() {
        return Ok(Vec::new());
    }
    file.tensor_f32(&name)
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
) -> Result<(WeightRef, WeightRef, WeightRef), AllpacaError> {
    let none = WeightRef::f32(0, 0);
    let up_name = format!("blk.{layer}.ffn_up_shexp.weight");
    if file.tensor(&up_name).is_none() {
        return Ok((none, none, none));
    }
    // SwiGLU shared experts (Qwen3-Next) carry a gate; Lightning's ReLU² does not.
    let gate_name = format!("blk.{layer}.ffn_gate_shexp.weight");
    let gate = if file.tensor(&gate_name).is_some() {
        append_gguf_tensor(file, &gate_name, parts, off)?
    } else {
        none
    };
    Ok((
        gate,
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
    append_gguf_tensor_as(file, name, parts, off, name == "token_embd.weight")
}

fn append_gguf_lm_head<'a>(
    file: &'a GgufFile,
    cfg: &ModelConfig,
    name: &str,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
) -> Result<WeightRef, AllpacaError> {
    if !use_q4_0_draft_head(cfg) {
        return append_gguf_tensor_as(file, name, parts, off, false);
    }
    let info = file.tensor(name).ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
    })?;
    let n = info.n_elements() as usize;
    let part = UploadPart::Owned(stage_draft_head_q4_0(info, file.tensor_bytes(name)?)?);
    let w = WeightRef {
        off: *off,
        n,
        kind: WeightKind::Q4_0,
    };
    *off += part.len();
    tracing::info!(tensor = name, bytes = part.len(), "MTP draft LM head staged as Q4_0");
    parts.push(part);
    Ok(w)
}

/// Low-bit ggml types with no fused HIP GEMV. Large GEMM weights of these
/// types re-pack to Q8_0 at load: every value lands on (or within half a step
/// of) the Q8_0 grid, at ~1.1 B/elem instead of a 4 B/elem F32 expand.
fn requant_q8_0_on_load(ty: allpaca_gguf::GgmlType) -> bool {
    use allpaca_gguf::GgmlType as T;
    matches!(
        ty,
        T::IQ4_NL
            | T::IQ4_XS
            | T::Q4_0
            | T::Q4_1
            | T::Q5_0
            | T::Q5_1
            | T::Q2_K
            | T::Q3_K
            // No HIP kernels for the lattice i-quants either: a Q8_0 copy is
            // 8.5 bits per weight where the F32 fallback would be 32.
            | T::IQ2_XXS
            | T::IQ2_XS
            | T::IQ2_S
            | T::IQ3_XXS
            | T::IQ1_S
            | T::IQ1_M
    )
}

/// ROCmFPX two-scale types (102, 104, 109-111) have no HIP kernel; every
/// value they decode to is exact in f16, so they go to the device as F16
/// and the GPU reads the CPU dequant's values bit for bit.
fn rocmfpx_as_f16(ty: allpaca_gguf::GgmlType) -> bool {
    use allpaca_gguf::GgmlType as T;
    matches!(ty, T::RocmFpxQ6 | T::RocmFpxQ3 | T::RocmFpxQ5 | T::RocmFpxQ7 | T::RocmFpxQ2)
}

/// Q3_K GEMM weights with whole 256-value rows run on the native int8 GEMV
/// (the MTP draft lm_head: 0.55 GB instead of a 1.43 GB Q8_0 re-pack).
fn q3_k_native(info: &allpaca_gguf::TensorInfo) -> bool {
    let k = info.dims.first().copied().unwrap_or(0);
    k > 0 && k % 256 == 0 && info.n_elements() > 65_536
}

fn use_q4_0_draft_head(cfg: &ModelConfig) -> bool {
    cfg.gguf_layer_base > 0 && std::env::var("ALLPACA_DRAFT_HEAD_Q4_0").ok().as_deref() != Some("0")
}

pub(crate) fn stage_draft_head_q4_0(
    info: &allpaca_gguf::TensorInfo,
    bytes: &[u8],
) -> Result<Vec<u8>, AllpacaError> {
    let k = info.dims.first().copied().unwrap_or(0) as usize;
    let n = info.n_elements() as usize;
    if k == 0 || k % allpaca_cpu::Q4_0_BLOCK_ELEMS != 0 || n % k != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "Q4_0 draft head dims {:?} do not tile n={n}",
            info.dims
        ))));
    }
    let rows = n / k;
    if rows == 0 || bytes.len() != info.nbytes() || bytes.len() % rows != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "Q4_0 draft head {} has invalid packed bytes",
            info.name
        ))));
    }
    let row_bytes = bytes.len() / rows;
    let mut row = vec![0.0f32; k];
    let mut out = Vec::with_capacity(
        n.div_ceil(allpaca_cpu::Q4_0_BLOCK_ELEMS) * allpaca_cpu::Q4_0_BLOCK_BYTES,
    );
    for r in 0..rows {
        allpaca_gguf::dequant_packed(
            info.ggml_type,
            &bytes[r * row_bytes..(r + 1) * row_bytes],
            k,
            &mut row,
        )?;
        out.extend_from_slice(&allpaca_cpu::quantize_q4_0(&row));
    }
    Ok(out)
}

/// VRAM bytes [`append_gguf_tensor_as`] stages for one tensor (not `gather`).
fn staged_device_bytes(info: &allpaca_gguf::TensorInfo) -> u64 {
    use allpaca_gguf::GgmlType as T;
    let n = info.n_elements();
    let q8_staged = n.div_ceil(32) * allpaca_hip::Q8_STAGED_BLOCK as u64;
    match info.ggml_type {
        T::F32 => n * 4,
        _ if n <= 65_536 => n * 4,
        T::F16 => n * 2,
        T::Q4_K | T::Q5_K | T::Q6_K | T::TQ1_0 | T::TQ2_0 | T::RocmFp4 | T::RocmFp4Fast | T::P48Fp4 => {
            info.nbytes() as u64
        }
        T::Q3_K if q3_k_native(info) => info.nbytes() as u64,
        T::Q8_0 | T::RocmFpxQ8 => q8_staged,
        ty if rocmfpx_as_f16(ty) => n * 2,
        ty if requant_q8_0_on_load(ty) => q8_staged,
        _ => n * 4,
    }
}

fn staged_lm_head_bytes(info: &allpaca_gguf::TensorInfo, cfg: &ModelConfig) -> u64 {
    if use_q4_0_draft_head(cfg) {
        info.n_elements().div_ceil(allpaca_cpu::Q4_0_BLOCK_ELEMS as u64)
            * allpaca_cpu::Q4_0_BLOCK_BYTES as u64
    } else {
        staged_device_bytes(info)
    }
}

/// Per-layer VRAM footprint of a GGUF upload, plus the target-only globals
/// (device embed, `output_norm`, lm_head). Used to place layers by bytes.
pub fn gguf_device_bytes(file: &GgufFile, cfg: &ModelConfig) -> (Vec<u64>, u64) {
    let (mut layers, experts, globals) = gguf_device_bytes_split(file, cfg);
    for (l, e) in layers.iter_mut().zip(experts) {
        *l += e;
    }
    (layers, globals)
}

/// Stacked routed-expert tensors (`ffn_{gate,up,down}_exps`).
pub fn is_stacked_expert_tensor(name: &str) -> bool {
    name.ends_with("_exps.weight")
}

/// [`gguf_device_bytes`] with each layer's routed-expert bytes apart from
/// its body: `(body, experts, globals)`.
pub fn gguf_device_bytes_split(file: &GgufFile, cfg: &ModelConfig) -> (Vec<u64>, Vec<u64>, u64) {
    let prefixes: Vec<String> = (0..cfg.n_layers)
        .map(|l| format!("blk.{}.", cfg.gguf_blk(l)))
        .collect();
    let tied = file.tensor("output.weight").is_none();
    let mut layers = vec![0u64; cfg.n_layers];
    let mut experts = vec![0u64; cfg.n_layers];
    let mut globals = 0u64;
    for t in &file.tensors {
        if let Some(l) = prefixes.iter().position(|p| t.name.starts_with(p.as_str())) {
            if is_stacked_expert_tensor(&t.name) {
                experts[l] += staged_device_bytes(t);
            } else {
                layers[l] += staged_device_bytes(t);
            }
        } else if t.name == "output.weight" {
            globals += staged_lm_head_bytes(t, cfg);
        } else if t.name == "output_norm.weight" {
            globals += staged_device_bytes(t);
        } else if t.name == "token_embd.weight" && (tied || !HostEmbed::wanted(t.ggml_type)) {
            globals += if tied {
                staged_lm_head_bytes(t, cfg)
            } else {
                staged_device_bytes(t)
            };
            if tied && use_q4_0_draft_head(cfg) && !HostEmbed::wanted(t.ggml_type) {
                globals += staged_device_bytes(t);
            }
        }
    }
    (layers, experts, globals)
}

/// `gather` stages `token_embd` for row gathers (F32 expand for K-quants,
/// linear Q8_0 blocks) instead of the GEMV layout.
fn append_gguf_tensor_as<'a>(
    file: &'a GgufFile,
    name: &str,
    parts: &mut Vec<UploadPart<'a>>,
    off: &mut usize,
    gather: bool,
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
            if n <= 65_536 || gather {
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
            if n <= 65_536 || gather {
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
            if n <= 65_536 || gather {
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
        allpaca_gguf::GgmlType::P48Fp4 => {
            // Stays packed: the GEMV reads the sparse blocks as they are.
            let k = *info.dims.first().unwrap_or(&1) as usize;
            if k == 0 || k % allpaca_cpu::P48FP4_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "paired-4:8 FP4 {name} rows of {k} weights are not whole 32-weight blocks"
                ))));
            }
            (UploadPart::Mapped(file.tensor_bytes(name)?), WeightKind::P48Fp4)
        }
        ty @ (allpaca_gguf::GgmlType::RocmFp4 | allpaca_gguf::GgmlType::RocmFp4Fast) => {
            let fast = ty == allpaca_gguf::GgmlType::RocmFp4Fast;
            let packed = file.tensor_bytes(name)?;
            // Keep large embeds packed: vocab×dim F32 would be multi-GiB host RAM.
            if n <= 65_536 {
                let mut f32s = vec![0.0f32; n];
                if fast {
                    allpaca_cpu::dequant_rocmfp4_fast(packed, n, &mut f32s)?;
                } else {
                    allpaca_cpu::dequant_rocmfp4(packed, n, &mut f32s)?;
                }
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
                let kind = if fast {
                    WeightKind::RocmFp4Fast
                } else {
                    WeightKind::RocmFp4
                };
                (UploadPart::Mapped(packed), kind)
            }
        }
        allpaca_gguf::GgmlType::Q5_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || gather {
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
            if n <= 65_536 || gather {
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
            } else if gather {
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
        allpaca_gguf::GgmlType::Q3_K if !gather && q3_k_native(info) => {
            (UploadPart::Mapped(file.tensor_bytes(name)?), WeightKind::Q3_K)
        }
        allpaca_gguf::GgmlType::RocmFpxQ8 if !gather && n > 65_536 => {
            // Lossless: each block is a Q8_0 block with its UE4M3 scale as a half.
            let k = *info.dims.first().unwrap_or(&1) as usize;
            if k == 0 || k % 32 != 0 || n % k != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "ROCmFPX Q8_0 GEMV {name} dims {:?} do not tile n={n}",
                    info.dims
                ))));
            }
            let q8 = allpaca_cpu::repack_rocmfpx_q8_as_q8_0(file.tensor_bytes(name)?)?;
            let staged = allpaca_hip::stage_q8_0_gemv(&q8, n / k, k).map_err(|e| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "Q8_0 GEMV stage {name}: {e}"
                )))
            })?;
            (UploadPart::Owned(staged), WeightKind::Q8_0)
        }
        ty if !gather && n > 65_536 && rocmfpx_as_f16(ty) => {
            let f32s = file.tensor_f32(name)?;
            let mut bytes = Vec::with_capacity(n * 2);
            for v in &f32s {
                bytes.extend_from_slice(&allpaca_cpu::f32_to_f16(*v).to_le_bytes());
            }
            (UploadPart::Owned(bytes), WeightKind::F16)
        }
        ty if !gather && n > 65_536 && requant_q8_0_on_load(ty) => {
            let k = *info.dims.first().unwrap_or(&1) as usize;
            if k == 0 || k % 32 != 0 || n % k != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{ty:?} -> Q8_0 {name} dims {:?} do not tile n={n}",
                    info.dims
                ))));
            }
            let q8 = allpaca_cpu::quantize_q8_0(&file.tensor_f32(name)?);
            let staged = allpaca_hip::stage_q8_0_gemv(&q8, n / k, k).map_err(|e| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "Q8_0 GEMV stage {name}: {e}"
                )))
            })?;
            (UploadPart::Owned(staged), WeightKind::Q8_0)
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
        kind @ (WeightKind::RocmFp4 | WeightKind::RocmFp4Fast) => {
            let block_bytes = if kind == WeightKind::RocmFp4Fast {
                allpaca_cpu::ROCMFP4_FAST_BLOCK_BYTES
            } else {
                allpaca_cpu::ROCMFP4_BLOCK_BYTES
            };
            if n_per % allpaca_cpu::ROCMFP4_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} ROCmFP4 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::ROCMFP4_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::ROCMFP4_BLOCK_ELEMS).saturating_mul(block_bytes)
        }
        WeightKind::P48Fp4 => {
            if n_per % allpaca_cpu::P48FP4_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} paired-4:8 FP4 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::P48FP4_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::P48FP4_BLOCK_ELEMS).saturating_mul(allpaca_cpu::P48FP4_BLOCK_BYTES)
        }
        WeightKind::Q4_0 => {
            if n_per % allpaca_cpu::Q4_0_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} Q4_0 expert elems {n_per} not aligned to {}",
                    allpaca_cpu::Q4_0_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::Q4_0_BLOCK_ELEMS).saturating_mul(allpaca_cpu::Q4_0_BLOCK_BYTES)
        }
        WeightKind::Q3_K => {
            if n_per % allpaca_cpu::Q3_K_BLOCK_ELEMS != 0 {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{what} Q3_K expert elems {n_per} not aligned to {}",
                    allpaca_cpu::Q3_K_BLOCK_ELEMS
                ))));
            }
            (n_per / allpaca_cpu::Q3_K_BLOCK_ELEMS).saturating_mul(allpaca_cpu::Q3_K_BLOCK_BYTES)
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
            // Device Q8_0 is the 36-byte staged layout, not GGUF's 34-byte blocks.
            (n_per / allpaca_cpu::Q8_0_BLOCK_ELEMS).saturating_mul(allpaca_hip::Q8_STAGED_BLOCK)
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
    /// Qwen3-Next shared-expert gate (`ffn_gate_inp_shexp`, `[dim]`): the
    /// shared expert output is scaled by `sigmoid(x_norm · w)`. Empty = 1.
    pub shared_gate_inp: Vec<f32>,
    /// DeepSeek-style expert bias for sigmoid-group selection.
    pub expert_bias: Vec<f32>,
    /// Spark2_5 head-wise attention output gate (`attn_gate.weight`, shape `[n_heads, dim]`).
    pub attn_out_gate: Vec<f32>,
    /// Per-head attention sink logits (`attn_sinks.weight`, shape `[n_heads]`; empty = none).
    pub attn_sinks: Vec<f32>,
    /// Gemma post-attention RMSNorm (empty = skip).
    pub attn_post_norm: Vec<f32>,
    /// Gemma post-FFN RMSNorm (empty = skip).
    pub ffn_post_norm: Vec<f32>,
    /// Qwen2-style Q/K/V projection biases (`attn_{q,k,v}.bias`); empty = none.
    pub attn_bias: [Vec<f32>; 3],
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
                let inner = m.lock().unwrap_or_else(|e| e.into_inner()).clone();
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
    /// Routed experts left in the file's encoding for the CPU forward and
    /// decoded per routed expert ([`Self::from_gguf_host_packed_experts`]).
    pub(crate) packed_experts: Option<Arc<crate::host_experts::PackedExperts>>,
    /// Manifest early-exit capability (from GGUF `allpaca.early_exit.*` or test hook).
    pub early_exit: EarlyExitSpec,
    /// Optional dedicated exit heads (`blk.{i}.exit_head.weight`); empty = shared lm_head.
    pub exit_heads: Vec<Option<Vec<f32>>>,
    /// Present when this TinyModel is an MTP/NextN draft with host seed weights.
    pub mtp_nextn: Option<MtpNextnWeights>,
    /// DeepSeek-V4 MLA / hyper-connection / MoE sidecar (CPU decode).
    pub(crate) dsv4: Option<Arc<crate::deepseek4::Dsv4Weights>>,
    /// qwen4exp runs on its own host forward ([`crate::qwen4exp`]).
    pub(crate) q4x: Option<Arc<crate::qwen4exp::Qwen4Weights>>,
    /// gemma4 runs on its own forward ([`crate::gemma4`]).
    pub(crate) g4: Option<Arc<crate::gemma4::Gemma4Weights>>,
    /// Records matrix inputs during a calibration run ([`crate::calib`]).
    pub(crate) calib: Arc<crate::calib::Tap>,
    /// Shared RPC runtime when layers are offloaded to remote workers.
    pub rpc: Option<std::sync::Arc<crate::rpc_runtime::RpcRuntime>>,
    /// Edge0 Recover-LoRA + prerouter stager (when loaded via `load_edge0`).
    pub edge0: Option<Edge0Runtime>,
    /// DeepSeek2 / Ling MLA compressed KV cache: per-layer (ckv_seq, krope_seq).
    pub mla_cache: Option<Arc<std::sync::Mutex<Vec<(Vec<f32>, Vec<f32>)>>>>,
    /// Operator-disabled routed experts and routing-hit counts.
    pub expert_controls: Arc<crate::expert_controls::ExpertControls>,
    /// Residual rows entering chosen layers, for a DFlash2 draft (off by default).
    pub layer_capture: Arc<crate::dflash2::LayerCapture>,
    /// Set on a worker's hosted `q3n-layers` span: its routed MoE rounds like
    /// a separate `MoeFfn` reply (host top-k, F16 routed sum), so hosted
    /// output equals the separate expert-parallel path bit for bit.
    pub hosted_moe_compat: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttnLayout {
    pub(crate) q_out: usize,
    pub(crate) kv_dim: usize,
    pub(crate) wo_k: usize,
    pub(crate) gated: bool,
    pub(crate) attn_dim: usize,
    pub(crate) head_dim: usize,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
}

pub(crate) fn attn_layout(cfg: &ModelConfig, wq_n: usize, wk_n: usize, wo_n: usize) -> AttnLayout {
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
    let gated = q_out == wo_k.saturating_mul(2) && (wo_k > dim || cfg.gated_attn_q);
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
    /// CUDA MN prefill still rejects IMRoPE / gated-Q. HIP `forward_tokens_device`
    /// handles those per row after a batched Q/K/V GEMM.
    /// HIP M=N forward ([`Self::forward_tokens_device`]). Unlike the CUDA /
    /// Vulkan M=N paths it runs IMRoPE and gated `[Q|gate]` attention per row,
    /// so Qwen3.5/3.8 hybrids batch too. A `[Q|gate]` projection is only
    /// recognised by shape when n_heads * head_dim > dim; below that only
    /// `gated_attn_q` (the architecture) identifies it. An ambiguous Q without
    /// that flag is read differently by the per-token and M=N paths, so it
    /// stays per-token.
    pub fn supports_hip_batched_prefill(&self) -> bool {
        let dim = self.config.dim.max(1);
        !self.layers.iter().any(|l| {
            if l.wq.is_empty() || l.wo.is_empty() {
                return false;
            }
            let q_out = l.wq.len() / dim;
            let wo_k = l.wo.len() / dim;
            q_out == wo_k.saturating_mul(2) && wo_k <= dim && !self.config.gated_attn_q
        })
    }

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
            q_out == wo_k.saturating_mul(2) && (wo_k > dim || self.config.gated_attn_q)
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
                attn_sinks: Vec::new(),
                attn_post_norm: Vec::new(),
                shared_gate_inp: Vec::new(),
                attn_bias: Default::default(),
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
            packed_experts: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            q4x: None,
            g4: None,
            calib: Default::default(),
            rpc: None,
            edge0: None,
            mla_cache: None,
            expert_controls: Default::default(),
            layer_capture: Default::default(),
            hosted_moe_compat: false,
        }
    }

    /// Attention + Mamba-2 SSM hybrid for CUDA/host decode tracers.
    /// Layer 0 is SSM; remaining layers stay dense attention.
    pub fn synthetic_ssm_hybrid(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny_hybrid(), seed, "ssm-hybrid");
        let dim = model.config.dim;
        let dims = crate::ssm::Mamba2Dims::new(dim, dim, 2, 4, 2, 2).expect("tiny SSM dims");
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
            attn_sinks: Vec::new(),
            attn_post_norm: Vec::new(),
            shared_gate_inp: Vec::new(),
            attn_bias: Default::default(),
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
            attn_sinks: Vec::new(),
            attn_post_norm: Vec::new(),
            shared_gate_inp: Vec::new(),
            attn_bias: Default::default(),
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
        model.mla_cache = Some(Arc::new(std::sync::Mutex::new(vec![
            (
                Vec::new(),
                Vec::new()
            );
            model.layers.len()
        ])));
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

    /// Tiny dense model with Spark2_5 head-wise sigmoid attention gate.
    pub fn synthetic_headwise_gate(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "headwise-gate");
        model.config.headwise_attn_gate = true;
        let dim = model.config.dim;
        let n_heads = model.config.n_heads;
        let mut rng = Lcg::new(seed.wrapping_add(91));
        for layer in &mut model.layers {
            layer.attn_out_gate = rng.vec(n_heads * dim);
        }
        model
    }

    /// Tiny dense model with Gemma-style embed scale + post-attn/FFN RMSNorms.
    pub fn synthetic_gemma_norms(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "gemma-norms");
        model.config.scale_embed = true;
        let dim = model.config.dim;
        for layer in &mut model.layers {
            layer.attn_post_norm = ones(dim);
            layer.ffn_post_norm = ones(dim);
        }
        model
    }

    /// Tiny dense model with Gemma/Spark GELU FFN activation.
    pub fn synthetic_gelu_ffn(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "gelu-ffn");
        model.config.ffn_gelu = true;
        model
    }

    /// Tiny dense model with per-head Q/K RMSNorm (Qwen3-style).
    pub fn synthetic_qk_norms(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "qk-norms");
        let head_dim = model.config.head_dim();
        let mut rng = Lcg::new(seed.wrapping_add(73));
        for layer in &mut model.layers {
            layer.q_norm = rng.vec(head_dim);
            layer.k_norm = rng.vec(head_dim);
        }
        model
    }

    /// Tiny dense model with Qwen3.5 gated-attention Q (`[Q|gate]` interleaved).
    /// `attn_dim = 2*dim` so `wo_k > dim` trips the gated layout heuristic.
    pub fn synthetic_gated_attn(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "gated-attn");
        let dim = model.config.dim;
        let attn_dim = dim * 2;
        let q_out = attn_dim * 2;
        let mut rng = Lcg::new(seed.wrapping_add(107));
        for layer in &mut model.layers {
            layer.wq = rng.vec(q_out * dim);
            layer.wo = rng.vec(dim * attn_dim);
        }
        model
    }

    /// Zero wq output rows `[dim..]` so a non-gated CUDA path (N=dim) ignores the change.
    pub fn zero_wq_output_tail(&mut self) {
        let dim = self.config.dim.max(1);
        for layer in &mut self.layers {
            if layer.wq.len() <= dim * dim {
                continue;
            }
            for v in &mut layer.wq[dim * dim..] {
                *v = 0.0;
            }
        }
    }

    /// Tiny dense model with per-head attention sink logits.
    pub fn synthetic_attn_sinks(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "attn-sinks");
        let n_heads = model.config.n_heads;
        let mut rng = Lcg::new(seed.wrapping_add(113));
        for layer in &mut model.layers {
            layer.attn_sinks = rng.vec(n_heads);
        }
        model
    }

    /// Tiny dense model with partial NeoX RoPE (`rope_dims` < head_dim).
    pub fn synthetic_partial_rope(seed: u64) -> Self {
        let mut model = Self::synthetic(ModelConfig::tiny(), seed, "partial-rope");
        let head_dim = model.config.head_dim();
        model.config.rope_dims = (head_dim / 2).max(2);
        if model.config.rope_dims % 2 != 0 {
            model.config.rope_dims -= 1;
        }
        model
    }

    /// Tiny dense model with Qwen3.5-style IMRoPE sections.
    pub fn synthetic_imrope(seed: u64) -> Self {
        let mut model = Self::synthetic_partial_rope(seed);
        model.name = "imrope".into();
        // Non-zero sections trip `uses_imrope()` (t,h sectors required).
        model.config.rope_sections = [1, 1, 0, 0];
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

    /// [`Self::forward_forest`] runs on host F32 weights with attention-only
    /// branch state: a device-resident load has no host weights, and a
    /// recurrent (SSM/GDN) layer would need its state forked per branch.
    pub fn supports_forest_verify(&self) -> bool {
        self.host_weights_ready()
            && !self
                .layers
                .iter()
                .any(|l| l.block == HybridLayerKind::Ssm || l.gdn.is_some() || l.ssm.is_some())
    }

    /// True when a GDN layer has scalars/conv but no host F32 projections
    /// (Q8 / large GGUF — CUDA must `upload_gguf`).
    pub fn gdn_needs_device_projections(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.gdn.as_ref().is_some_and(|g| g.w_qkv.is_empty()))
    }

    /// Keep a GGUF handle for lazy DeepSeek-V4 MLA / expert / lm_head gathers.
    pub(crate) fn bind_gguf(&mut self, file: Arc<GgufFile>) {
        if let Some(g4) = &mut self.g4 {
            if let Some(inner) = Arc::get_mut(g4) {
                inner.attach_gguf(file.clone());
            }
        }
        if let Some(q4x) = &mut self.q4x {
            if let Some(inner) = Arc::get_mut(q4x) {
                inner.attach_gguf(file.clone());
            }
        }
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
        if let Some(packed) = self.packed_experts.as_ref().filter(|p| p.has_layer(layer_idx)) {
            return packed.expert(layer_idx, expert);
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
        if let Some(packed) = self.packed_experts.as_ref().filter(|p| p.has_layer(layer_idx)) {
            return (0..n_exp).map(|e| packed.expert(layer_idx, e)).collect();
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
            self.calib.observe(cfg.gguf_blk(layer_idx), crate::calib::Site::FfnIn, &scratch[..dim]);
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
        let mut gate = vec![0.0f32; cfg.dense_ff()];
        let mut up = vec![0.0f32; cfg.dense_ff()];
        self.calib.observe(cfg.gguf_blk(layer_idx), crate::calib::Site::FfnIn, &scratch[..dim]);
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            ffn.w_gate(),
            resident.filter(|_| ffn_dev).map(|r| r.w_gate(layer_idx)),
            &mut gate,
            1,
            cfg.dense_ff(),
            dim,
        )?;
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            ffn.w_up(),
            resident.filter(|_| ffn_dev).map(|r| r.w_up(layer_idx)),
            &mut up,
            1,
            cfg.dense_ff(),
            dim,
        )?;
        silu_mul_dispatch(hip, &mut gate, &up, metrics, cfg.ffn_gelu)?;
        self.calib.observe(cfg.gguf_blk(layer_idx), crate::calib::Site::FfnAct, &gate);
        gemm_nt_dispatch(
            hip,
            &gate,
            ffn.w_down(),
            resident.filter(|_| ffn_dev).map(|r| r.w_down(layer_idx)),
            hidden,
            1,
            dim,
            cfg.dense_ff(),
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
            let mask = self.route_mask(layer_idx);
            topk_route(x_norm, &layer.w_router, n_exp, n_act, cfg, mask.as_deref().map(Vec::as_slice), &layer.expert_bias)
        } else {
            self.edge0_moe_route(layer_idx, layer, x_norm)
        };
        self.expert_controls.record(layer_idx, &routed);
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
            if self.calib.wants() {
                // The output moves by the routing weight times the
                // expert's own error: weigh its inputs by its square.
                let blk = cfg.gguf_blk(layer_idx);
                let act = expert_activations(x_norm, &ffn, cfg.ff_dim, dim, cfg.moe_relu_sqr)?;
                self.calib.observe_weighted(blk, crate::calib::Site::ExpIn, x_norm, w * w);
                self.calib.observe_weighted(blk, crate::calib::Site::ExpAct, &act, w * w);
            }
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
            // Qwen3-Next / Qwen3.5 MoE gate the shared expert per token.
            let gate = if layer.shared_gate_inp.len() == dim {
                let logit: f32 = x_norm.iter().zip(&layer.shared_gate_inp).map(|(a, b)| a * b).sum();
                1.0 / (1.0 + (-logit).exp())
            } else {
                1.0
            };
            if self.calib.wants() {
                let blk = cfg.gguf_blk(layer_idx);
                let act = expert_activations(x_norm, &layer.shared_expert, sff, dim, cfg.moe_relu_sqr)?;
                self.calib.observe_weighted(blk, crate::calib::Site::ShexpIn, x_norm, gate * gate);
                self.calib.observe_weighted(blk, crate::calib::Site::ShexpAct, &act, gate * gate);
            }
            for i in 0..dim {
                acc[i] += gate * scratch_h[i];
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
        let mask = self.route_mask(layer_idx);
        let mask = mask.as_deref().map(Vec::as_slice);
        let Some(e0) = &self.edge0 else {
            return moe_route(x_norm, layer, &self.config, mask);
        };
        let Some(stager) = &e0.stager else {
            return moe_route(x_norm, layer, &self.config, mask);
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
            // A staged route that picks a disabled expert is not used.
            let picks_disabled = mask.is_some_and(|m| pred.iter().any(|&(e, _)| m.get(e).copied().unwrap_or(false)));
            if patch && !picks_disabled {
                return pred.clone();
            }
        }
        moe_route(x_norm, layer, &self.config, mask)
    }

    /// Layer `layer`'s disabled-expert mask, `None` when nothing is disabled.
    pub(crate) fn route_mask(&self, layer: usize) -> Option<Arc<Vec<bool>>> {
        self.expert_controls.layer_mask(layer)
    }

    fn apply_edge0_lora(&self, layer_idx: usize, proj: &str, x: &[f32], y: &mut [f32]) {
        if let Some(e0) = &self.edge0 {
            if let Some(lora) = &e0.lora {
                lora.apply_named(&format!("layers.{layer_idx}.self_attn.{proj}"), x, y);
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
        let dim = self.config.dim;
        // The hint feeds the expert LRU's prefetch, and routes host token
        // embeddings; large device-resident GGUFs keep neither.
        if self.config.n_experts == 0
            || span.is_empty()
            || self.expert_lru.is_none()
            || self.embed.len() < self.config.vocab_size * dim
        {
            return Vec::new();
        }
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
                let mask = self.route_mask(layer_idx);
                for (e, _) in topk_route(x, &layer.w_router, n_exp, n_act, &self.config, mask.as_deref().map(Vec::as_slice), &layer.expert_bias) {
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
        // Decode with Q8_0 gate and up: both GEMVs in one launch.
        let paired = m == 1
            && w_gate.n == w_up.n
            && w_gate.kind == WeightKind::Q8_0
            && w_up.kind == WeightKind::Q8_0
            && hip.gemm_nt_q8_pair_dev(
                x,
                resident.weights.byte_ptr_at(w_gate.off),
                buf.gate,
                resident.weights.byte_ptr_at(w_up.off),
                buf.up,
                m,
                ff_dim as i32,
                ff_dim as i32,
                dim as i32,
            )?;
        if paired {
            self.device_act_mul(hip, buf.gate, buf.up, hidden as i32)?;
            return resident.gemm_nt(
                hip,
                buf.gate,
                w_down,
                buf.proj,
                m,
                dim as i32,
                ff_dim as i32,
            );
        }
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
        let mask = self.route_mask(layer_idx);
        let routed = topk_route(x_norm, &layer.w_router, n_exp, n_act, cfg, mask.as_deref().map(Vec::as_slice), &layer.expert_bias);
        self.expert_controls.record(layer_idx, &routed);
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
        // GPU routing whenever this layer has it, stagger or not: the stagger
        // only warms a partner's CPU experts, and switching this Request to
        // host routing for it changed its tokens (the two routings round
        // differently), so concurrent output drifted from solo output.
        // `moe_norm` snapshots then come only from host-routed layers.
        let separate_moe_compat =
            self.hosted_moe_compat || side.as_ref().is_some_and(|s| s.separate_moe_compat);
        if self.gpu_routed_local_moe(layer_idx, hip, resident, off, host_layer, buf, 1, separate_moe_compat)? {
            record_elementwise(Some(metrics));
            return Ok(());
        }
        let (x_norm, mut router_logits) = self.device_moe_readback(hip, resident, off, buf, 1)?;
        let ds_probe = side.as_ref().and_then(|s| s.ds_probe);
        let ds_queue = side.as_ref().and_then(|s| s.ds_queue);
        let nvme_side = side.as_ref().map(|s| DeviceDecodeSide {
            request_id: s.request_id,
            mrope: s.mrope,
            moe_norm: None,
            layer_stagger: s.layer_stagger,
            ds_probe,
            ds_queue,
            last_hidden: None,
            write_logits: s.write_logits,
            separate_moe_compat: s.separate_moe_compat,
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
        let mask = self.route_mask(layer_idx);
        let mask = mask.as_deref().map(Vec::as_slice);
        let routed = if self.edge0.is_some() || cfg.edge0_tier.is_some() {
            self.edge0_moe_route(layer_idx, host_layer, &x_norm)
        } else if let Some(logits) = router_logits.as_mut() {
            topk_from_logits(logits, n_act, cfg, mask, &host_layer.expert_bias)
        } else {
            topk_route(&x_norm, &host_layer.w_router, n_exp, n_act, cfg, mask, &host_layer.expert_bias)
        };
        self.expert_controls.record(layer_idx, &routed);
        let ids: Vec<ExpertId> = routed
            .iter()
            .map(|&(e, _)| ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            })
            .collect();
        // Expert-parallel: this layer's experts may live on an RPC worker.
        let routes = std::slice::from_ref(&routed);
        if !self.remote_moe_rows(layer_idx, hip, resident, buf, &x_norm, routes)?
            && !self.fused_local_moe_rows(layer_idx, hip, resident, buf, routes)?
        {
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
        }
        if separate_moe_compat {
            hip.round_f16(buf.attn, dim)?;
        }
        if off.w_shared_up.n > 0
            && off.w_shared_down.n > 0
            && host_layer.shared_gate_inp.len() == dim
            && x_norm.len() >= dim
        {
            // Shared expert, then its sigmoid-gated scale, the residual add
            // and the copy back into proj in one launch.
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
                off.w_shared_gate,
                off.w_shared_up,
                off.w_shared_down,
                sff,
            )?;
            let logit: f32 = x_norm[..dim]
                .iter()
                .zip(&host_layer.shared_gate_inp)
                .map(|(a, b)| a * b)
                .sum();
            hip.scale_add_copy(buf.attn, buf.proj, sigmoid_stable(logit), dim)?;
            record_elementwise(Some(metrics));
            return Ok(());
        } else if off.w_shared_up.n > 0 && off.w_shared_down.n > 0 {
            self.device_shared_expert_rows(hip, resident, buf, off, host_layer, &x_norm, 1, None)?;
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

    /// Read back what host routing needs from `n_tok` rows of `buf.normed`:
    /// the rows themselves and, when the router is a device-resident
    /// `[n_experts, dim]` softmax router, its logits (GEMM into `buf.q`, free
    /// once the mixer is done). `buf.q` follows `buf.normed` in the decode
    /// workspace, so both come back in one D2H: on Windows every small copy
    /// costs ~60-80 us whatever its size.
    fn device_moe_readback(
        &self,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        off: &LayerOff,
        buf: DeviceDecode,
        n_tok: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), AllpacaError> {
        let cfg = &self.config;
        let (dim, n_exp) = (cfg.dim, cfg.n_experts);
        let xn = n_tok * dim;
        let router = n_exp > 0
            && off.w_router.n == n_exp * dim
            && cfg.router_kind == RouterKind::SoftmaxTopk
            && self.edge0.is_none()
            && cfg.edge0_tier.is_none();
        let mut x_norm = vec![0.0f32; xn];
        if !router {
            hip.copy_d2h_f32(buf.normed, &mut x_norm)?;
            return Ok((x_norm, None));
        }
        resident.gemm_nt(
            hip,
            buf.normed,
            off.w_router,
            buf.q,
            n_tok as i32,
            n_exp as i32,
            dim as i32,
        )?;
        let ln = n_tok * n_exp;
        let gap = (buf.q as usize)
            .checked_sub(buf.normed as usize)
            .map(|b| b / 4)
            .filter(|&g| g >= xn && g <= xn + (1 << 16));
        let Some(gap) = gap else {
            let mut logits = vec![0.0f32; ln];
            hip.copy_d2h_f32(buf.normed, &mut x_norm)?;
            hip.copy_d2h_f32(buf.q, &mut logits)?;
            return Ok((x_norm, Some(logits)));
        };
        // Copy by kernel into pinned memory, then wait: cheaper than a
        // hipMemcpy D2H on Windows (~48 vs ~70 us).
        let mut stage = resident.moe_stage.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("moe stage lock".into()))
        })?;
        let host = stage.ensure((gap + ln) * 4)?.cast::<f32>();
        hip.bind()?;
        allpaca_hip::copy_f32_kernel(host, buf.normed, gap + ln)?;
        allpaca_hip::stream_synchronize()?;
        let staged = unsafe { std::slice::from_raw_parts(host, gap + ln) };
        Ok((staged[..xn].to_vec(), Some(staged[gap..].to_vec())))
    }

    /// Expert-parallel MoE: when this layer's routed experts live on an RPC
    /// worker, run them there and write the weighted sums (`[rows][dim]`) to
    /// `buf.attn`. `false` means the experts are local.
    fn remote_moe_rows(
        &self,
        layer_idx: usize,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        x_norm: &[f32],
        routes: &[Vec<(usize, f32)>],
    ) -> Result<bool, AllpacaError> {
        let Some(rpc) = self.rpc.as_ref() else {
            return Ok(false);
        };
        let Some(out) = rpc.moe_ffn(layer_idx, x_norm, routes)? else {
            return Ok(false);
        };
        // Stage in pinned memory and copy by kernel: queued, where a
        // hipMemcpy would block the host for ~60 us.
        let mut stage = resident.moe_stage.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("moe stage lock".into()))
        })?;
        let host = stage.ensure(out.len() * 4)?.cast::<f32>();
        unsafe { std::ptr::copy_nonoverlapping(out.as_ptr(), host, out.len()) };
        hip.bind()?;
        allpaca_hip::copy_f32_kernel(buf.attn, host, out.len())?;
        Ok(true)
    }

    /// Qwen2 Q/K/V projection biases on `rows` rows of `buf.q` / `buf.k` /
    /// `buf.v` (no-op for layers without them).
    #[allow(clippy::too_many_arguments)]
    fn device_qkv_bias(
        &self,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        layer: &LayerOff,
        buf: DeviceDecode,
        rows: usize,
        q_out: usize,
        kv_dim: usize,
    ) -> Result<(), AllpacaError> {
        for (bias, x, n) in [(layer.bq, buf.q, q_out), (layer.bk, buf.k, kv_dim), (layer.bv, buf.v, kv_dim)] {
            if bias.n == n {
                hip.bias_act_rows_dev(x, resident.f32_or_dequant(hip, bias)?, rows as i32, n as i32, false)?;
            }
        }
        Ok(())
    }

    /// `n_tok` rows' MoE entirely on the GPU for a layer whose experts are
    /// local stacked K-quants: router GEMV, softmax top-k into a device pair
    /// table, fused experts, shared expert scaled by its on-device gate. No
    /// host round trip, so the host keeps queueing while the GPU runs (on
    /// Windows each sync after an idle gap costs 100-200 us). Leaves the MoE
    /// output in `buf.proj` like [`Self::device_moe`]. `false` when this layer
    /// needs host routing.
    #[allow(clippy::too_many_arguments)]
    fn gpu_routed_local_moe(
        &self,
        layer_idx: usize,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        off: &LayerOff,
        host_layer: &LayerWeights,
        buf: DeviceDecode,
        n_tok: usize,
        separate_moe_compat: bool,
    ) -> Result<bool, AllpacaError> {
        let cfg = &self.config;
        let (dim, ff, n_exp) = (cfg.dim, cfg.ff_dim, cfg.n_experts);
        let k = cfg.n_active.max(1).min(n_exp);
        let softmax_router = cfg.router_kind == RouterKind::SoftmaxTopk
            && !cfg.moe_sigmoid_gating
            && !cfg.moe_sqrt_softplus
            && (cfg.expert_weights_scale == 0.0 || cfg.expert_weights_scale == 1.0);
        if !softmax_router
            || separate_moe_compat
            || k > 64
            || off.w_router.n != n_exp * dim
            || self.edge0.is_some()
            || cfg.edge0_tier.is_some()
            || self.expert_lru.is_some()
            || resident.expert_ring.is_some()
            || cfg.moe_relu_sqr
            || cfg.ffn_gelu
            || dim % 256 != 0
            || ff % 256 != 0
            || self.rpc.as_ref().is_some_and(|r| r.experts_remote(layer_idx))
        {
            return Ok(false);
        }
        let Some(experts) = resident
            .expert_offs
            .get(layer_idx)
            .filter(|o| o.len() == n_exp)
            .and_then(|o| stacked_moe_experts(&resident.weights, o))
        else {
            return Ok(false);
        };
        let has_shared = off.w_shared_up.n > 0 && off.w_shared_down.n > 0;
        let shexp_gate = if host_layer.shared_gate_inp.is_empty() {
            std::ptr::null()
        } else if off.shared_gate_inp.n == dim && off.shared_gate_inp.kind == WeightKind::F32 {
            resident.weights.ptr_at(off.shared_gate_inp.off)
        } else {
            return Ok(false);
        };
        let m = n_tok as i32;
        resident.gemm_nt(hip, buf.normed, off.w_router, buf.q, m, n_exp as i32, dim as i32)?;
        let (_, table_bytes) = allpaca_hip::MoeDeviceRoutes::at(std::ptr::null_mut(), n_tok, k);
        let bytes = table_bytes + allpaca_hip::moe_ffn_scratch_bytes(n_tok, dim, ff, n_tok * k);
        let mut scratch = resident.moe_scratch.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("moe scratch lock".into()))
        })?;
        if scratch.as_ref().is_none_or(|s| s.bytes() < bytes as u64) {
            *scratch = None;
            *scratch = Some(DeviceWeights::alloc(resident.weights.device(), bytes.max(1 << 20))?);
        }
        let base = scratch.as_ref().expect("moe scratch").ptr_at(0) as *mut u8;
        let (routes, used) = allpaca_hip::MoeDeviceRoutes::at(base, n_tok, k);
        hip.bind()?;
        let (disabled, counts) = resident.route_ctl_ptrs(&self.expert_controls, layer_idx)?;
        allpaca_hip::moe_route_softmax_topk_device(
            buf.q, n_exp, buf.normed, shexp_gate, dim, &routes, disabled, counts,
        )?;
        allpaca_hip::moe_ffn_kquant_device(
            allpaca_hip::MoeInput::F32(buf.normed),
            n_tok,
            dim,
            ff,
            n_tok * k,
            routes.row,
            routes.expert,
            routes.w,
            routes.row_start,
            allpaca_hip::MoeGroups::per_pair(n_tok * k),
            experts[0],
            experts[1],
            experts[2],
            unsafe { base.add(used) },
            buf.attn,
        )?;
        drop(scratch);
        if separate_moe_compat {
            hip.round_f16(buf.attn, n_tok * dim)?;
        }
        if has_shared {
            let scale = (!shexp_gate.is_null()).then_some(routes.shexp_scale.cast_const());
            self.device_shared_expert_rows(hip, resident, buf, off, host_layer, &[], n_tok, scale)?;
        }
        hip.copy_d2d_f32(buf.proj, buf.attn, n_tok * dim)?;
        Ok(true)
    }

    /// Local routed experts through the fused K-quant MoE kernels: three
    /// launches per layer instead of ~4 per selected expert, with the same
    /// arithmetic. Rows of `buf.normed`; weighted sums into `buf.attn`.
    /// `false` when the layer's experts are not stacked K-quants or Q8_0
    /// resident here.
    fn fused_local_moe_rows(
        &self,
        layer_idx: usize,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        routes: &[Vec<(usize, f32)>],
    ) -> Result<bool, AllpacaError> {
        let cfg = &self.config;
        let (dim, ff) = (cfg.dim, cfg.ff_dim);
        if self.expert_lru.is_some()
            || resident.expert_ring.is_some()
            || cfg.moe_relu_sqr
            || cfg.ffn_gelu
            || dim % 256 != 0
            || ff % 256 != 0
        {
            return Ok(false);
        }
        let Some(experts) = resident
            .expert_offs
            .get(layer_idx)
            .filter(|o| o.len() == cfg.n_experts)
            .and_then(|o| stacked_moe_experts(&resident.weights, o))
        else {
            return Ok(false);
        };
        let pairs: Vec<(u32, u32, f32)> = routes
            .iter()
            .enumerate()
            .flat_map(|(r, route)| route.iter().map(move |&(e, w)| (r as u32, e as u32, w)))
            .collect();
        let rows = routes.len();
        let table = allpaca_hip::MoePairTable::new(rows, &pairs, &[]);
        let bytes = allpaca_hip::moe_ffn_scratch_bytes(rows, dim, ff, table.pairs);
        let mut scratch = resident.moe_scratch.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("moe scratch lock".into()))
        })?;
        if scratch.as_ref().is_none_or(|s| s.bytes() < bytes as u64) {
            *scratch = None;
            *scratch = Some(DeviceWeights::alloc(resident.weights.device(), bytes.max(1 << 20))?);
        }
        let work = scratch.as_ref().expect("moe scratch").ptr_at(0) as *mut u8;
        // The kernels read the pair table from pinned memory in place.
        let mut stage = resident.moe_stage.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("moe stage lock".into()))
        })?;
        let host = stage.ensure(table.host.len() * 4)?.cast::<f32>();
        unsafe { std::ptr::copy_nonoverlapping(table.host.as_ptr(), host, table.host.len()) };
        hip.bind()?;
        table.run(
            host,
            allpaca_hip::MoeInput::F32(buf.normed),
            rows,
            dim,
            ff,
            experts,
            work,
            buf.attn,
        )?;
        Ok(true)
    }

    /// Device shared expert (`ffn_*_shexp`) on `n_tok` rows of `buf.normed`,
    /// added into `buf.attn`. Qwen3-Next scales each row's output by
    /// `sigmoid(x_norm · ffn_gate_inp_shexp)`.
    #[allow(clippy::too_many_arguments)]
    fn device_shared_expert_rows(
        &self,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        off: &LayerOff,
        host_layer: &LayerWeights,
        x_norm: &[f32],
        n_tok: usize,
        device_scale: Option<*const f32>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
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
            off.w_shared_gate,
            off.w_shared_up,
            off.w_shared_down,
            sff,
        )?;
        let gate = &host_layer.shared_gate_inp;
        if let Some(scale) = device_scale {
            allpaca_hip::scale_rows_device(buf.proj, scale, n_tok, dim)?;
        } else if gate.len() == dim {
            for (t, x) in x_norm.chunks_exact(dim).take(n_tok).enumerate() {
                let logit: f32 = x.iter().zip(gate).map(|(a, b)| a * b).sum();
                hip.scale_f32(buf.proj_row(t, dim), dim, sigmoid_stable(logit))?;
            }
        }
        hip.residual_add(buf.attn, buf.proj, (n_tok * dim) as i32)?;
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
        // Same GPU routing as single-token decode, so verify spans and decode
        // produce identical tokens.
        if self.gpu_routed_local_moe(
            layer_idx,
            hip,
            resident,
            off,
            host_layer,
            buf,
            n_tok,
            self.hosted_moe_compat,
        )? {
            record_elementwise(Some(metrics));
            return Ok(());
        }
        let (x_norm, mut logits) = self.device_moe_readback(hip, resident, off, buf, n_tok)?;
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let mut routes: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_tok);
        let mut union: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let mask = self.route_mask(layer_idx);
        let mask = mask.as_deref().map(Vec::as_slice);
        for t in 0..n_tok {
            let routed = match logits.as_mut() {
                Some(l) => topk_from_logits(
                    &mut l[t * n_exp..(t + 1) * n_exp],
                    n_act,
                    cfg,
                    mask,
                    &host_layer.expert_bias,
                ),
                None => topk_route(
                    &x_norm[t * dim..(t + 1) * dim],
                    &host_layer.w_router,
                    n_exp,
                    n_act,
                    cfg,
                    mask,
                    &host_layer.expert_bias,
                ),
            };
            self.expert_controls.record(layer_idx, &routed);
            for &(e, _) in &routed {
                union.insert(e);
            }
            routes.push(routed);
        }
        if !self.remote_moe_rows(layer_idx, hip, resident, buf, &x_norm, &routes)?
            && !self.fused_local_moe_rows(layer_idx, hip, resident, buf, &routes)?
        {
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
        }
        if self.hosted_moe_compat {
            hip.round_f16(buf.attn, n_tok * dim)?;
        }
        if off.w_shared_up.n > 0 && off.w_shared_down.n > 0 {
            self.device_shared_expert_rows(hip, resident, buf, off, host_layer, &x_norm, n_tok, None)?;
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
        Self::from_gguf_sized(file, false)
    }

    /// [`Self::from_gguf`] with every weight decoded to F32 in RAM whatever
    /// the model's size, for the CPU forward.
    pub fn from_gguf_host(file: &GgufFile) -> Result<Self, AllpacaError> {
        Self::from_gguf_sized(file, true)
    }

    /// [`Self::from_gguf_host`] with the routed experts left in the file's
    /// encoding: each is decoded when a token is routed to it. The file
    /// must stay open (it is memory-mapped), hence the `Arc`.
    pub fn from_gguf_host_packed_experts(file: Arc<GgufFile>) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        if let Some(reason) = report.reject_reason {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(reason)));
        }
        let cfg = ModelConfig::from_gguf(&file)?;
        if crate::qwen4exp::file_is_qwen4exp(&file) {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "qwen4exp has its own forward: no packed host experts".into(),
            )));
        }
        let mut model = Self::from_gguf_cfg_with(&file, cfg, &report, true, true)?;
        model.packed_experts = crate::host_experts::PackedExperts::from_gguf(file, &model.config).map(Arc::new);
        Ok(model)
    }

    fn from_gguf_sized(file: &GgufFile, host_all: bool) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        if let Some(reason) = report.reject_reason {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(reason)));
        }
        let cfg = ModelConfig::from_gguf(file)?;
        if crate::qwen4exp::file_is_qwen4exp(file) {
            // Its own tensor set and forward: the generic layout checks do not apply.
            let weights = crate::qwen4exp::Qwen4Weights::from_gguf(file, cfg.n_layers)?;
            let name = file.meta_str("general.name").unwrap_or("gguf").to_string();
            let mut model = Self::skeleton_hybrid(cfg, name, &report, Some(file));
            model.config.recurrent = true;
            model.q4x = Some(Arc::new(weights));
            return Ok(model);
        }
        if crate::gemma4::file_is_gemma4(file) {
            // Its own K/V layout (two head dims, shared blocks) and forward.
            let weights = crate::gemma4::Gemma4Weights::from_gguf(file)?;
            let name = file.meta_str("general.name").unwrap_or("gguf").to_string();
            let mut model = Self::skeleton_hybrid(cfg, name, &report, Some(file));
            // The sequence state carries the K/V ([`crate::gemma4::Gemma4Memory`]).
            model.config.recurrent = true;
            model.g4 = Some(Arc::new(weights));
            return Ok(model);
        }
        Self::from_gguf_cfg(file, cfg, &report, host_all)
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
        let (base, count) = if trunk.nextn_predict_layers > 0 {
            (trunk.n_layers, trunk.nextn_predict_layers)
        } else {
            let mut blocks: Vec<usize> = file
                .tensors
                .iter()
                .filter_map(|t| {
                    let rest = t.name.strip_prefix("blk.")?;
                    let (block, suffix) = rest.split_once('.')?;
                    (suffix == "nextn.eh_proj.weight")
                        .then(|| block.parse::<usize>().ok())
                        .flatten()
                })
                .collect();
            blocks.sort_unstable();
            blocks.dedup();
            match (blocks.first(), blocks.last()) {
                (Some(first), Some(last)) => (*first, last - first + 1),
                _ => {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                        "GGUF has no nextn_predict_layers or NextN tensors for MTP draft".into(),
                    )))
                }
            }
        };
        let mut cfg = trunk.clone();
        cfg.gguf_layer_base = base;
        cfg.n_layers = count;
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
        if crate::qwen4exp::file_is_qwen4exp(file) {
            let mut model = Self::from_gguf(file)?;
            model.config.gguf_layer_base = cfg.gguf_layer_base;
            model.config.n_layers = cfg.n_layers;
            model.config.nextn_predict_layers = 0;
            model.config.recurrent = false;
            if let Some(q4) = model.q4x.as_mut().and_then(Arc::get_mut) {
                q4.spec.full_attention_interval = 1;
            }
            model.name = format!("{}-mtp", model.name);
            return Ok(model);
        }
        let mut model = Self::from_gguf_cfg(file, cfg, &report, false)?;
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
        host_all: bool,
    ) -> Result<Self, AllpacaError> {
        Self::from_gguf_cfg_with(file, cfg, report, host_all, false)
    }

    /// With `packed_experts`, stacked routed experts are not decoded: the
    /// caller attaches them packed.
    fn from_gguf_cfg_with(
        file: &GgufFile,
        cfg: ModelConfig,
        report: &GgufCompatReport,
        host_all: bool,
        packed_experts: bool,
    ) -> Result<Self, AllpacaError> {
        let name = file.meta_str("general.name").unwrap_or("gguf").to_string();
        for i in 0..cfg.n_layers {
            let i = cfg.gguf_blk(i);
            let info = report
                .layers
                .iter()
                .find(|layer| layer.index == i)
                .cloned()
                .unwrap_or(HybridLayerInfo {
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
        if host_bytes > HOST_F32_LIMIT && !host_all {
            let mut model = Self::skeleton_hybrid(cfg, name, report, Some(file));
            attach_dsv4(&mut model, file)?;
            return Ok(model);
        }
        // Continue with original host-materialize body — call into remaining logic
        // by temporarily reconstructing what from_gguf did after validation.
        Self::from_gguf_host_materialize(file, cfg, report, name, packed_experts)
    }

    fn from_gguf_host_materialize(
        file: &GgufFile,
        cfg: ModelConfig,
        report: &GgufCompatReport,
        name: String,
        packed_experts: bool,
    ) -> Result<Self, AllpacaError> {
        // Packed experts are decoded per routed expert: only the router here.
        let load_moe = |i: usize, n_exp: usize| {
            if packed_experts && file.tensor(&format!("blk.{i}.ffn_up_exps.weight")).is_some() {
                load_gguf_router_only(file, i, n_exp)
            } else {
                load_gguf_moe_layer(file, i, n_exp)
            }
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let i = cfg.gguf_blk(i);
            let info = report
                .layers
                .iter()
                .find(|layer| layer.index == i)
                .cloned()
                .unwrap_or(HybridLayerInfo {
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
                    let has_moe = cfg.n_experts > 0 && file.tensor(&format!("blk.{i}.ffn_gate_inp.weight")).is_some();
                    let (w_router, experts) = if has_moe { load_moe(i, cfg.n_experts)? } else { (Vec::new(), Vec::new()) };
                    let (w_gate, w_up, w_down, ffn_norm) =
                        if has_moe {
                            (Vec::new(), Vec::new(), Vec::new(), gguf_ffn_norm(file, i)?)
                        } else if file.tensor(&format!("blk.{i}.ffn_gate.weight")).is_some() {
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
                        ssm: if file.architecture().is_some_and(is_qwen_gdn_arch) {
                            None
                        } else {
                            Some(load_gguf_ssm_layer(file, i, cfg.dim, true)?)
                        },
                        gdn: if file.architecture().is_some_and(is_qwen_gdn_arch) {
                            load_gguf_gdn_host(file, i, true).ok()
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
                        w_router,
                        experts,
                        shared_expert: if has_moe { load_gguf_shared_expert(file, i)? } else { LayerFfn::default() },
                        expert_bias: if has_moe { load_gguf_expert_bias(file, i)? } else { Vec::new() },
                        attn_out_gate: Vec::new(),
                        attn_sinks: Vec::new(),
                        attn_post_norm: Vec::new(),
                        shared_gate_inp: load_gguf_shared_gate(file, i)?,
                        attn_bias: Default::default(),
                        ffn_post_norm: Vec::new(),
                    }
                }
                HybridLayerKind::DenseFfn | HybridLayerKind::Moe => {
                    let (w_gate, w_up, w_down, w_router, experts) = if info.moe || cfg.n_experts > 0
                    {
                        let n_exp = cfg.n_experts.max(1);
                        let (w_router, experts) = load_moe(i, n_exp)?;
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
                        attn_sinks: Vec::new(),
                        attn_post_norm: Vec::new(),
                        shared_gate_inp: load_gguf_shared_gate(file, i)?,
                        attn_bias: Default::default(),
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
                        let (w_router, experts) = load_moe(i, cfg.n_experts)?;
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
                    let attn_out_gate =
                        if file.tensor(&format!("blk.{i}.attn_gate.weight")).is_some()
                            && cfg.headwise_attn_gate
                        {
                            file.tensor_f32(&format!("blk.{i}.attn_gate.weight"))?
                        } else {
                            Vec::new()
                        };
                    let attn_sinks = if file.tensor(&format!("blk.{i}.attn_sinks.weight")).is_some()
                    {
                        file.tensor_f32(&format!("blk.{i}.attn_sinks.weight"))?
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
                        attn_sinks,
                        attn_post_norm: if file
                            .tensor(&format!("blk.{i}.attn_post_norm.weight"))
                            .is_some()
                        {
                            file.tensor_f32(&format!("blk.{i}.attn_post_norm.weight"))?
                        } else {
                            Vec::new()
                        },
                        shared_gate_inp: load_gguf_shared_gate(file, i)?,
                        attn_bias: Default::default(),
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
        attach_gguf_attn_bias(&mut layers, file, &cfg)?;
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
                (
                    Vec::new(),
                    Vec::new()
                );
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
            packed_experts: None,
            early_exit,
            exit_heads,
            mtp_nextn: None,
            dsv4: None,
            q4x: None,
            g4: None,
            calib: Default::default(),
            rpc: None,
            edge0: None,
            mla_cache,
            expert_controls: Default::default(),
            layer_capture: Default::default(),
            hosted_moe_compat: false,
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
                let info = report.layers.iter().find(|layer| layer.index == i);
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
                        if f.architecture().is_some_and(is_qwen_gdn_arch) {
                            if let Ok(gdn) = load_gguf_gdn_host(f, i, false) {
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
                if let Some(f) = file {
                    // GDN + MoE blocks (Qwen3-Next) route after the SSM mixer.
                    let routes = f.tensor(&format!("blk.{i}.ffn_gate_inp.weight"));
                    if block == HybridLayerKind::Ssm
                        && cfg.n_experts > 0
                        && routes.is_some_and(|t| t.ggml_type != allpaca_gguf::GgmlType::BF16)
                    {
                        if let Ok(router) = f.tensor_f32(&format!("blk.{i}.ffn_gate_inp.weight")) {
                            layer.w_router = router;
                        }
                        if let Ok(norm) = gguf_ffn_norm(f, i) {
                            layer.ffn_norm = norm;
                        }
                    }
                    let shexp_gate = format!("blk.{i}.ffn_gate_inp_shexp.weight");
                    if f.tensor(&shexp_gate).is_some() {
                        layer.shared_gate_inp = f.tensor_f32(&shexp_gate).unwrap_or_default();
                    }
                    // Routing runs on host: it needs the selection bias too.
                    if !layer.w_router.is_empty() {
                        layer.expert_bias = load_gguf_expert_bias(f, i).unwrap_or_default();
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
            packed_experts: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            q4x: None,
            g4: None,
            calib: Default::default(),
            rpc: None,
            edge0: None,
            mla_cache: None,
            expert_controls: Default::default(),
            layer_capture: Default::default(),
            hosted_moe_compat: false,
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
            rope_neox: true,
            rope_none: false,
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
            gated_attn_q: false,
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
            dense_ff_dim: 0,
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
                attn_sinks: Vec::new(),
                attn_post_norm: Vec::new(),
                shared_gate_inp: Vec::new(),
                attn_bias: Default::default(),
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
            packed_experts: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            q4x: None,
            g4: None,
            calib: Default::default(),
            rpc: None,
            edge0: None,
            mla_cache: None,
            expert_controls: Default::default(),
            layer_capture: Default::default(),
            hosted_moe_compat: false,
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
            rope_neox: true,
            rope_none: false,
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
            gated_attn_q: false,
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
            dense_ff_dim: 0,
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
                attn_sinks: Vec::new(),
                attn_post_norm: Vec::new(),
                shared_gate_inp: Vec::new(),
                attn_bias: Default::default(),
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
            packed_experts: None,
            early_exit: EarlyExitSpec::default(),
            exit_heads: Vec::new(),
            mtp_nextn: None,
            dsv4: None,
            q4x: None,
            g4: None,
            calib: Default::default(),
            rpc: None,
            edge0: None,
            mla_cache: None,
            expert_controls: Default::default(),
            layer_capture: Default::default(),
            hosted_moe_compat: false,
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
        if self.q4x.is_some() {
            let memory = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("qwen4exp needs the sequence's state".into()))
            })?;
            crate::qwen4exp::forward_qwen4_token(self, token, pos, logits, &mut memory.q4x)?;
            if let Some(last_hidden) = side.and_then(|s| s.last_hidden.as_deref_mut()) {
                if let Some(captured) = memory.q4x.row_hiddens.take() {
                    *last_hidden = captured;
                }
            }
            return Ok(());
        }
        if self.g4.is_some() {
            let memory = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("gemma4 needs the sequence's state".into()))
            })?;
            let r = crate::gemma4::forward_tokens(self, &[token], pos, logits, None, &mut memory.g4);
            metrics.kv_bytes_on_device.store(crate::gemma4_gpu::kv_device_bytes(), std::sync::atomic::Ordering::Relaxed);
            return r;
        }
        if self.dsv4.is_some() {
            return crate::deepseek4::forward_dsv4_token(
                self,
                token,
                pos,
                page_ids,
                kv,
                logits,
                metrics,
                None,
                hip,
                prefill,
                recurrent.as_deref_mut().map(|r| &mut r.dsv4),
            );
        }
        // Prefer on-device decode whenever resident weights exist — including hybrid
        // models that carry recurrent SSM/DeltaNet state (host stub inside device path).
        // RPC layer-split bounces activations only when the owner changes. A contiguous
        // remote span stays on the worker for one RunSpan RPC. Remote layers are skipped
        // inside the device loop.
        if let (Some(hip), Some(resident)) = (hip, resident) {
            {
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
                    None,
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
        let mut scratch = vec![0.0f32; dim.max(cfg.moe_workspace()).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if let Some(trace) = recurrent.as_deref_mut().and_then(|r| r.trace.as_mut()) {
                if trace.enter(pos, layer_idx, self.layers.len(), &mut x) {
                    continue;
                }
            }
            if let Some(rpc) = &self.rpc {
                if rpc.is_remote(layer_idx) {
                    if rpc.is_span_start(layer_idx) {
                        if rpc.is_hosted(layer_idx) {
                            rpc.run_hosted(layer_idx, pos, recurrent.as_deref_mut(), &mut x)?;
                        } else {
                            rpc.run_layer(layer_idx, pos, &mut x)?;
                        }
                    }
                    continue;
                }
            }
            if matches!(
                layer.block,
                HybridLayerKind::Ssm | HybridLayerKind::DenseFfn | HybridLayerKind::Moe
            ) {
                if let (HybridLayerKind::Ssm, Some(gdn)) =
                    (layer.block, layer.gdn.as_ref().filter(|g| !g.w_qkv.is_empty() && hip.is_none()))
                {
                    rmsnorm(&x, &layer.attn_norm, cfg.eps, &mut scratch[..dim])?;
                    let state = recurrent.as_deref_mut().ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::ModelLoad("a delta-net layer needs recurrent state".into()))
                    })?;
                    if state.gdn_state.len() <= layer_idx {
                        state.gdn_state.resize(layer_idx + 1, Vec::new());
                        state.gdn_conv.resize(layer_idx + 1, Vec::new());
                    }
                    let out = crate::qwen35_host::gdn_step(
                        gdn,
                        cfg,
                        &scratch[..dim],
                        &mut state.gdn_state[layer_idx],
                        &mut state.gdn_conv[layer_idx],
                        (&self.calib, cfg.gguf_blk(layer_idx)),
                    )?;
                    for i in 0..dim {
                        x[i] += out[i];
                    }
                    if !layer.w_gate.is_empty() || !layer.experts.is_empty() {
                        self.apply_ffn(layer_idx, layer, &mut x, &mut scratch, &mut hidden, hip, resident, Some(metrics), prefill)?;
                    }
                    continue;
                }
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
            if hip.is_none() && crate::qwen35_host::is_gated_attention(cfg, layer) {
                let state = recurrent.as_deref_mut().ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::ModelLoad("Qwen3.5 attention on the CPU needs recurrent state".into()))
                })?;
                if state.host_kv.len() <= layer_idx {
                    state.host_kv.resize(layer_idx + 1, Default::default());
                }
                let (keys, values) = &mut state.host_kv[layer_idx];
                let out = crate::qwen35_host::gated_attention_step(
                    layer,
                    cfg,
                    &scratch[..dim],
                    pos,
                    keys,
                    values,
                    (&self.calib, cfg.gguf_blk(layer_idx)),
                )?;
                for i in 0..dim {
                    x[i] += out[i];
                }
                self.apply_ffn(layer_idx, layer, &mut x, &mut scratch, &mut hidden, hip, resident, Some(metrics), prefill)?;
                continue;
            }
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
            self.calib.observe(cfg.gguf_blk(layer_idx), crate::calib::Site::AttnIn, &scratch[..dim]);
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
            add_row_bias(&mut q, &layer.attn_bias[0]);
            add_row_bias(&mut k, &layer.attn_bias[1]);
            add_row_bias(&mut v, &layer.attn_bias[2]);
            // Per-head RMS norm of the queries and keys (Qwen3, Ling), before RoPE.
            for (rows, gain, heads) in [(&mut q, &layer.q_norm, cfg.n_heads), (&mut k, &layer.k_norm, cfg.n_kv_heads)] {
                if gain.len() == head_dim {
                    for head in rows[..heads * head_dim].chunks_exact_mut(head_dim) {
                        let raw = head.to_vec();
                        rmsnorm(&raw, gain, cfg.eps, head)?;
                    }
                }
            }
            let theta = cfg.rope_theta_for_layer(layer_idx);
            let neox = cfg.neox_rot_dims(layer_idx, head_dim).filter(|_| !cfg.uses_imrope());
            if cfg.rope_none {
            } else if let Some(n_rot) = neox {
                for h in 0..cfg.n_heads {
                    let slice = &mut q[h * head_dim..(h + 1) * head_dim];
                    rope_imrope_neox(slice, pos, theta, n_rot, [0; 4])?;
                }
                for h in 0..cfg.n_kv_heads {
                    let slice = &mut k[h * head_dim..(h + 1) * head_dim];
                    rope_imrope_neox(slice, pos, theta, n_rot, [0; 4])?;
                }
            } else {
                rope_dispatch(
                    hip,
                    &mut q,
                    cfg.n_heads,
                    head_dim,
                    pos,
                    theta,
                    Some(metrics),
                )?;
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
            self.calib.observe(cfg.gguf_blk(layer_idx), crate::calib::Site::AttnOut, &attn_out);
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
        if let Some(trace) = recurrent.as_deref_mut().and_then(|r| r.trace.as_mut()) {
            // A trace that runs single blocks has no use for the output head.
            if trace.enter(pos, self.layers.len(), self.layers.len(), &mut x) {
                return Ok(());
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

    /// Whether [`Self::forward_batch`] can run several sequences in one pass
    /// on `hip`: the multi-row device path, with every local layer on `hip`
    /// or on the uploaded peer GPU.
    pub fn supports_forward_batch(&self, hip: &HipWorkspace) -> bool {
        // qwen4exp batches sequences to pipeline them through the worker
        // that hosts blocks; without one a batch would gain nothing.
        if let Some(q4) = self.q4x.as_ref() {
            return q4.gpu.lock().is_ok_and(|gpu| gpu.as_ref().is_some_and(|g| g.hosted().is_some()));
        }
        if let Some(g4) = self.g4.as_ref() {
            // Each sequence's rows run as their own pass, so a batch is exact.
            return g4.gpu.lock().is_ok_and(|gpu| gpu.is_some());
        }
        if self.dsv4.is_some() || !self.supports_hip_batched_prefill() {
            return false;
        }
        let Some(rpc) = self.rpc.as_ref() else {
            return true;
        };
        if rpc.placed_remote() {
            return false;
        }
        let peer = rpc.local_peer().map(|(dev, _, _)| dev);
        (0..self.layers.len()).all(|layer| {
            let want = rpc.local_device(layer);
            want < 0 || want == hip.device || Some(want) == peer
        })
    }

    /// One HIP forward over the rows of several sequences (continuous
    /// batching): `tokens` holds each sequence's rows in `segs` order, and
    /// `row_logits` receives `tokens.len() * vocab` logits. Weights, and the
    /// remote expert calls of expert-parallel MoE layers, are paid once for
    /// all rows instead of once per sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batch(
        &self,
        tokens: &[u32],
        segs: &mut [SeqRows<'_>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        row_logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
    ) -> Result<(), AllpacaError> {
        let vocab = self.config.vocab_size;
        if !self.supports_forward_batch(hip) || row_logits.len() < tokens.len() * vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "batched forward unsupported for this model or buffer".into(),
            )));
        }
        if let Some(tok) = tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "token {tok} out of vocab"
            ))));
        }
        if self.q4x.is_some() {
            return crate::qwen4exp::forward_qwen4_batch(self, tokens, segs, row_logits);
        }
        if self.g4.is_some() {
            let r = crate::gemma4::forward_batch(self, tokens, segs, row_logits);
            metrics.kv_bytes_on_device.store(crate::gemma4_gpu::kv_device_bytes(), std::sync::atomic::Ordering::Relaxed);
            return r;
        }
        let mut last = vec![0.0f32; vocab];
        // Each sequence's rows must match its solo run bit for bit.
        let _exact = allpaca_hip::ExactRows::enable();
        self.forward_tokens_device(
            tokens,
            segs,
            kv,
            metrics,
            &mut last,
            hip,
            resident,
            false,
            Some(&mut *row_logits),
            None,
            None,
        )?;
        hip.wait_logits_d2h(&mut row_logits[..tokens.len() * vocab])
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
        vision: Option<&VisionSeq>,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mrope = vision.and_then(VisionSeq::mrope);
        let dim = self.config.dim;
        if self.g4.is_some() {
            let memory = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("gemma4 needs the sequence's state".into()))
            })?;
            let r = crate::gemma4::forward_tokens(self, tokens, start_pos, logits, row_logits.as_deref_mut(), &mut memory.g4);
            metrics.kv_bytes_on_device.store(crate::gemma4_gpu::kv_device_bytes(), std::sync::atomic::Ordering::Relaxed);
            return r;
        }
        if self.q4x.is_some() {
            let memory = recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("qwen4exp needs the sequence's state".into()))
            })?;
            return crate::qwen4exp::forward_qwen4_tokens_with_hiddens(
                self,
                tokens,
                start_pos,
                logits,
                row_logits.as_deref_mut(),
                out_row_hiddens.unwrap_or(&mut Vec::new()),
                &mut memory.q4x,
            );
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
                true,
                recurrent.as_deref_mut().map(|r| &mut r.dsv4),
            );
        }
        // Layers on RPC workers: the batched device path below sends each
        // remote span `rpc.prefill_rows` rows per round trip. Without a local
        // GPU workspace the prompt runs one token at a time.
        let batched_remote = hip.is_some() && resident.is_some() && self.supports_hip_batched_prefill();
        if self.rpc.as_ref().is_some_and(|r| r.placed_remote()) && !batched_remote {
            for (i, tok) in tokens.iter().enumerate() {
                let last = i + 1 == tokens.len();
                let mut side = DeviceDecodeSide {
                    last_hidden: if last {
                        out_row_hiddens.as_deref_mut()
                    } else {
                        None
                    },
                    mrope,
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
                    vision.and_then(|v| v.soft_row(start_pos + i, dim)),
                )?;
            }
            return Ok(());
        }
        if tokens.len() == 1 {
            let mut side = DeviceDecodeSide {
                last_hidden: out_row_hiddens.as_deref_mut(),
                mrope,
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
                vision.and_then(|v| v.soft_row(start_pos, dim)),
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
        if self.supports_hip_batched_prefill() {
            if let (Some(hip), Some(resident)) = (hip, resident) {
                let mut segs = [SeqRows {
                    start_pos,
                    n_rows: tokens.len(),
                    page_ids,
                    request_id,
                    recurrent: recurrent.as_deref_mut(),
                }];
                match self.forward_tokens_device(
                tokens,
                &mut segs,
                kv,
                metrics,
                logits,
                hip,
                resident,
                write_logits,
                row_logits.as_deref_mut(),
                out_row_hiddens.as_deref_mut(),
                vision,
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
            if self.fuse_next_attn_norm(layer_idx, hip.device) {
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
            // qkv and z read the same normed row: one paired launch when both
            // are Q8_0 (bit-identical to two GEMVs).
            let qkv_z_paired = layer.w_qkv.kind == WeightKind::Q8_0
                && layer.w_z.kind == WeightKind::Q8_0
                && hip.gemm_nt_q8_pair_dev(
                    buf.normed,
                    resident.weights.byte_ptr_at(layer.w_qkv.off),
                    buf.q,
                    resident.weights.byte_ptr_at(layer.w_z.off),
                    buf.attn,
                    1,
                    dims.qkv_dim as i32,
                    dims.value_dim as i32,
                    dim as i32,
                )?;
            if !qkv_z_paired {
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
            }
            // Packed ssm_ba uploads alpha then beta as adjacent F32 rows: one
            // [2 n_v, dim] GEMV, beta read from its second half (each output
            // column is computed alone, so the bits match two GEMVs).
            let n_v = dims.n_v_heads;
            let ab_adjacent = layer.w_alpha.kind == WeightKind::F32
                && layer.w_beta.kind == WeightKind::F32
                && layer.w_alpha.n == n_v * dim
                && layer.w_beta.n == n_v * dim
                && layer.w_beta.off == layer.w_alpha.off + n_v * dim * 4;
            let beta_ptr = if ab_adjacent {
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    WeightRef::f32(layer.w_alpha.off, 2 * n_v * dim),
                    buf.gate,
                    1,
                    (2 * n_v) as i32,
                    dim as i32,
                )?;
                unsafe { buf.gate.add(n_v) }
            } else {
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    layer.w_alpha,
                    buf.gate,
                    1,
                    n_v as i32,
                    dim as i32,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    layer.w_beta,
                    buf.up,
                    1,
                    n_v as i32,
                    dim as i32,
                )?;
                buf.up
            };
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
                    beta_ptr,
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
                    dims.k_repeat == crate::deltanet::KHeadRepeat::Interleave,
                )
            })();
            if let Err(e) = device_gdn {
                static LOGGED: AtomicBool = AtomicBool::new(false);
                if !LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        layer_idx,
                        error = %e,
                        "GDN device step fell back to host"
                    );
                }
                let mut qkv = vec![0.0f32; dims.qkv_dim];
                let mut z = vec![0.0f32; dims.value_dim];
                let mut alpha = vec![0.0f32; dims.n_v_heads];
                let mut beta = vec![0.0f32; dims.n_v_heads];
                hip.copy_d2h_f32(buf.q, &mut qkv)?;
                hip.copy_d2h_f32(buf.attn, &mut z)?;
                hip.copy_d2h_f32(buf.gate, &mut alpha)?;
                hip.copy_d2h_f32(beta_ptr, &mut beta)?;
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
                    dims.k_repeat == crate::deltanet::KHeadRepeat::Interleave,
                )?;
                hip.expand_k_heads_host(
                    k_raw,
                    &mut k_exp,
                    dims.n_k_heads as i32,
                    dims.n_v_heads as i32,
                    dims.head_k as i32,
                    dims.k_repeat == crate::deltanet::KHeadRepeat::Interleave,
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
                    cfg.dense_ff() as i32,
                    dim as i32,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_up,
                    buf.up,
                    1,
                    cfg.dense_ff() as i32,
                    dim as i32,
                )?;
                hip.silu_mul_device(buf.gate, buf.up, cfg.dense_ff() as i32)?;
                resident.gemm_nt(
                    hip,
                    buf.gate,
                    w_down,
                    buf.proj,
                    1,
                    dim as i32,
                    cfg.dense_ff() as i32,
                )?;
                drop(ffn);
            } else if Self::layer_has_device_moe(layer, resident, layer_idx) {
                // Qwen3-Next: routed MoE after the GDN mixer.
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
            }
            if self.fuse_next_attn_norm(layer_idx, hip.device) {
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
                cfg.dense_ff() as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                w_up,
                buf.up,
                1,
                cfg.dense_ff() as i32,
                dim as i32,
            )?;
            hip.silu_mul_device(buf.gate, buf.up, cfg.dense_ff() as i32)?;
            resident.gemm_nt(
                hip,
                buf.gate,
                w_down,
                buf.proj,
                1,
                dim as i32,
                cfg.dense_ff() as i32,
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

    #[allow(clippy::too_many_arguments)]
    fn device_gdn_rows(
        &self,
        layer_idx: usize,
        host_layer: &LayerWeights,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        n_tok: usize,
        metrics: &Metrics,
        segs: &mut [SeqRows<'_>],
    ) -> Result<bool, AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let layer = &resident.layers[layer_idx];
        let Some(gdn) = host_layer.gdn.as_ref() else {
            return Ok(false);
        };
        if layer.w_qkv.n == 0 || n_tok == 0 {
            return Ok(false);
        }
        let dims = &gdn.dims;
        let m = n_tok as i32;
        resident.gemm_nt(
            hip,
            buf.normed,
            layer.w_qkv,
            buf.q,
            m,
            dims.qkv_dim as i32,
            dim as i32,
        )?;
        resident.gemm_nt(
            hip,
            buf.normed,
            layer.w_z,
            buf.attn,
            m,
            dims.value_dim as i32,
            dim as i32,
        )?;
        resident.gemm_nt(
            hip,
            buf.normed,
            layer.w_alpha,
            buf.gate,
            m,
            dims.n_v_heads as i32,
            dim as i32,
        )?;
        resident.gemm_nt(
            hip,
            buf.normed,
            layer.w_beta,
            buf.up,
            m,
            dims.n_v_heads as i32,
            dim as i32,
        )?;
        let (conv_w, dt, a, norm) = hip.ensure_gdn_weights(
            layer_idx,
            &gdn.conv1d,
            &gdn.dt_bias,
            &gdn.a,
            &gdn.norm,
        )?;
        let mut row0 = 0;
        for seg in segs.iter_mut() {
            let request_id = seg.request_id;
            let state = seg.recurrent.as_deref_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(
                    "Gated DeltaNet layer requires recurrent state".into(),
                ))
            })?;
            state.ensure_gdn(layer_idx, dims);
            let rows = row0..row0 + seg.n_rows;
            row0 += seg.n_rows;
            for t in rows {
                let q = buf.q_row(t, dims.qkv_dim);
                let z = buf.attn_row(t, dims.value_dim);
                let alpha = buf.gate_row(t, dims.n_v_heads);
                let beta = buf.up_row(t, dims.n_v_heads);
                if let Err(e) = hip.gdn_decode_step_resident(
                    request_id,
                    layer_idx,
                    &mut state.gdn_state[layer_idx],
                    &mut state.gdn_conv[layer_idx],
                    q,
                    z,
                    alpha,
                    beta,
                    conv_w,
                    dt,
                    a,
                    norm,
                    z,
                    dims.n_v_heads as i32,
                    dims.n_k_heads as i32,
                    dims.head_v as i32,
                    dims.head_k as i32,
                    dims.conv_kernel as i32,
                    cfg.eps,
                    dims.k_repeat == crate::deltanet::KHeadRepeat::Interleave,
                ) {
                    static LOGGED: AtomicBool = AtomicBool::new(false);
                    if !LOGGED.swap(true, Ordering::Relaxed) {
                        tracing::warn!(layer_idx, error = %e, "GDN batched step fell back to host");
                    }
                    let mut qkv = vec![0.0f32; dims.qkv_dim];
                    let mut z_h = vec![0.0f32; dims.value_dim];
                    let mut alpha_h = vec![0.0f32; dims.n_v_heads];
                    let mut beta_h = vec![0.0f32; dims.n_v_heads];
                    hip.copy_d2h_f32(q, &mut qkv)?;
                    hip.copy_d2h_f32(z, &mut z_h)?;
                    hip.copy_d2h_f32(alpha, &mut alpha_h)?;
                    hip.copy_d2h_f32(beta, &mut beta_h)?;
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
                        dims.k_repeat == crate::deltanet::KHeadRepeat::Interleave,
                    )?;
                    hip.expand_k_heads_host(
                        k_raw,
                        &mut k_exp,
                        dims.n_k_heads as i32,
                        dims.n_v_heads as i32,
                        dims.head_k as i32,
                        dims.k_repeat == crate::deltanet::KHeadRepeat::Interleave,
                    )?;
                    let mut decay = vec![0.0f32; dims.n_v_heads];
                    hip.gdn_decay_beta_host(&alpha_h, &gdn.dt_bias, &gdn.a, &mut decay, &mut beta_h)?;
                    let mut y = vec![0.0f32; dims.value_dim];
                    hip.gated_delta_step_resident(
                        request_id,
                        layer_idx,
                        &mut state.gdn_state[layer_idx],
                        &mut q_exp,
                        &mut k_exp,
                        v_raw,
                        &decay,
                        &beta_h,
                        &mut y,
                        dims.n_v_heads as i32,
                        dims.head_v as i32,
                        dims.head_k as i32,
                        cfg.eps,
                    )?;
                    let mut gated = vec![0.0f32; dims.value_dim];
                    rmsnorm_silu_gate(&y, &z_h, &gdn.norm, &mut gated, dims.head_v, cfg.eps);
                    hip.copy_h2d_f32(z, &gated)?;
                }
            }
        }
        resident.gemm_nt(
            hip,
            buf.attn,
            layer.w_ssm_out,
            buf.proj,
            m,
            dim as i32,
            dims.value_dim as i32,
        )?;
        if layer.w_gate.n > 0 {
            let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
            let ffn_norm = ffn.ffn_norm();
            let w_gate = ffn.w_gate();
            let w_up = ffn.w_up();
            let w_down = ffn.w_down();
            hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
            rmsnorm_device_rows(
                hip,
                buf.x,
                resident.f32_or_dequant(hip, ffn_norm)?,
                buf.normed,
                n_tok,
                dim,
                cfg.eps,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                w_gate,
                buf.gate,
                m,
                cfg.dense_ff() as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                w_up,
                buf.up,
                m,
                cfg.dense_ff() as i32,
                dim as i32,
            )?;
            hip.silu_mul_device(buf.gate, buf.up, (n_tok * cfg.dense_ff()) as i32)?;
            resident.gemm_nt(
                hip,
                buf.gate,
                w_down,
                buf.proj,
                m,
                dim as i32,
                cfg.dense_ff() as i32,
            )?;
            drop(ffn);
        } else if Self::layer_has_device_moe(layer, resident, layer_idx) {
            let request_id = segs.first().map_or(0, |s| s.request_id);
            self.device_moe_rows(layer_idx, hip, resident, buf, n_tok, metrics, request_id)?;
        }
        hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
        if self.fuse_next_attn_norm(layer_idx, hip.device) {
            rmsnorm_device_rows(
                hip,
                buf.x,
                resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                buf.normed,
                n_tok,
                dim,
                cfg.eps,
            )?;
        }
        Ok(true)
    }

    /// Sequential target decode with device-resident hidden state and KV.
    #[allow(clippy::too_many_arguments)]
    fn fuse_next_attn_norm(&self, layer_idx: usize, device: i32) -> bool {
        let next = layer_idx + 1;
        if next >= self.layers.len() {
            return false;
        }
        match self.rpc.as_ref() {
            Some(rpc) => !rpc.is_remote(next) && rpc.local_device(next) == device,
            None => true,
        }
    }

    fn layer_has_device_weights(layer: &LayerOff) -> bool {
        layer.attn_norm.n > 0 || layer.wq.n > 0 || layer.w_qkv.n > 0 || layer.w_ssm_in.n > 0
    }

    /// Copy `buf.x` onto `want` and retarget the decode workspace.
    ///
    /// Engine later waits for logits on the target HIP workspace (`primary`).
    /// Layers may run on a peer GPU; lm_head must return here first.
    fn switch_local_gpu<'a>(
        want: i32,
        hip: &mut &'a HipWorkspace,
        resident: &mut &'a ResidentModel,
        buf: &mut DeviceDecode,
        xfer: &mut [f32],
        primary_hip: &'a HipWorkspace,
        primary_resident: &'a ResidentModel,
        peer: &'a Option<(i32, Arc<HipWorkspace>, Arc<ResidentModel>)>,
        cfg: &ModelConfig,
        vocab: usize,
        max_kv: usize,
        primary_q: usize,
        primary_kv: usize,
        n_tok: usize,
        logits_rows: usize,
    ) -> Result<bool, AllpacaError> {
        if want < 0 || want == hip.device {
            return Ok(false);
        }
        fwd_segment_mark(hip.device, "switch");
        let src_x = buf.x;
        let src_dev = hip.device;
        if want == primary_hip.device {
            *hip = primary_hip;
            *resident = primary_resident;
        } else {
            let Some((dev, ph, pr)) = peer else {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "layer wants HIP {want} but no peer workspace was uploaded"
                ))));
            };
            if *dev != want {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "layer wants HIP {want} but peer workspace is {dev}"
                ))));
            }
            *hip = ph.as_ref();
            *resident = pr.as_ref();
        }
        hip.bind()?;
        let (qw, kw) = if hip.device == primary_hip.device {
            (primary_q, primary_kv)
        } else {
            Self::decode_workspaces(cfg, resident)
        };
        *buf = if n_tok > 1 {
            hip.ensure_decode_batch(
                cfg.dim,
                kw,
                cfg.moe_workspace(),
                vocab,
                max_kv,
                n_tok,
                logits_rows.max(1),
                qw,
            )?
        } else {
            hip.ensure_decode(qw, kw, cfg.moe_workspace(), vocab, max_kv)?
        };
        copy_dev_f32(
            buf.x,
            hip.device,
            src_x,
            src_dev,
            n_tok.max(1) * cfg.dim,
            xfer,
        )?;
        Ok(true)
    }

    /// Run the model's layers on an already-computed residual row and return
    /// that row after the last layer, without the final norm or LM head.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_hidden_token_device(
        &self,
        pos: usize,
        x: &mut [f32],
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        recurrent: &mut RecurrentState,
        request_id: u64,
    ) -> Result<(), AllpacaError> {
        if x.len() != self.config.dim {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "hidden input has {} floats, expected {}",
                x.len(),
                self.config.dim
            ))));
        }
        let mut out = Vec::new();
        let mut side = DeviceDecodeSide::new(request_id);
        side.separate_moe_compat = true;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.forward_token_device(
            0,
            pos,
            page_ids,
            kv,
            metrics,
            &mut logits,
            hip,
            resident,
            false,
            None,
            Some(&mut side),
            Some(recurrent),
            Some(x),
            Some(&mut out),
        )?;
        x.copy_from_slice(&out);
        Ok(())
    }

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
        mut hidden_out: Option<&mut Vec<f32>>,
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
        let rope_pos = side
            .as_ref()
            .and_then(|s| s.mrope)
            .map_or([pos; 3], |m| m.rope_pos(pos))
            .map(|p| p as i32);
        // The first layer with pages: a recurrent layer keeps none.
        let page_tokens = page_ids
            .iter()
            .find_map(|p| p.first())
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
        let fwd_t0 = std::time::Instant::now();
        fwd_segment_mark(hip.device, "start");
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
            let primary_hip = hip;
            let primary_resident = resident;
            let peer = self.rpc.as_ref().and_then(|r| r.local_peer());
            let mut hip = primary_hip;
            let mut resident = primary_resident;
            let mut buf = buf;
            let mut xfer = vec![0.0f32; dim];
            let mut normed_ready = false;
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
                if let Some(rpc) = &self.rpc {
                    if rpc.is_remote(layer_idx) {
                        if rpc.is_span_start(layer_idx) {
                            let mut host_x = vec![0.0f32; dim];
                            hip.copy_d2h_f32(buf.x, &mut host_x)?;
                            if rpc.is_hosted(layer_idx) {
                                rpc.run_hosted(layer_idx, pos, recurrent.as_deref_mut(), &mut host_x)?;
                            } else {
                                rpc.run_layer(layer_idx, pos, &mut host_x)?;
                            }
                            hip.copy_h2d_f32(buf.x, &host_x)?;
                        }
                        normed_ready = false;
                        continue;
                    }
                    let want = rpc.local_device(layer_idx);
                    if Self::switch_local_gpu(
                        want,
                        &mut hip,
                        &mut resident,
                        &mut buf,
                        &mut xfer,
                        primary_hip,
                        primary_resident,
                        &peer,
                        cfg,
                        vocab,
                        max_kv,
                        q_workspace,
                        kv_workspace,
                        1,
                        1,
                    )? {
                        normed_ready = false;
                    }
                }
                if let Some(k) = self.layer_capture.slot(layer_idx) {
                    let mut host = vec![0.0f32; dim];
                    hip.copy_d2h_f32(buf.x, &mut host)?;
                    self.layer_capture.record(k, &[(pos, token)], &host);
                }
                let layer = &resident.layers[layer_idx];
                if !Self::layer_has_device_weights(layer) {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "layer {layer_idx} has no weights on HIP {}",
                        hip.device
                    ))));
                }
                if !normed_ready {
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
                    normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
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
                self.device_qkv_bias(hip, resident, layer, buf, 1, q_out, kv_dim)?;
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
                } else if q_out == n_heads * head_dim {
                    // Plain attention with per-head Q/K RMSNorm before the
                    // rotation (Qwen3, Ling 2.0, Gemma 3).
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
                if cfg.rope_none {
                } else if let Some(n_rot) = cfg.neox_rot_dims(layer_idx, head_dim) {
                    hip.rope_imrope_qk_device(
                        buf.q,
                        buf.k,
                        n_heads as i32,
                        n_kv_heads as i32,
                        head_dim as i32,
                        n_rot as i32,
                        rope_pos,
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
                kv.write_slot_from_device(page_id, slot, n_kv_heads, head_dim, buf.k, buf.v, None)?;
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
                                cfg.dense_ff() as i32,
                                dim as i32,
                            )?;
                            resident.gemm_nt(
                                hip,
                                buf.normed,
                                w_up,
                                buf.up,
                                1,
                                cfg.dense_ff() as i32,
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
                            cfg.dense_ff() as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_up,
                            buf.up,
                            1,
                            cfg.dense_ff() as i32,
                            dim as i32,
                        )?;
                    }
                    self.device_act_mul(hip, buf.gate, buf.up, cfg.dense_ff() as i32)?;
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
                                cfg.dense_ff() as i32,
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
                            cfg.dense_ff() as i32,
                        )?;
                    }
                    drop(ffn);
                }
                // else: attention-only hybrid block — `buf.proj` still holds attn out.
                if self.fuse_next_attn_norm(layer_idx, hip.device) {
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
                normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
            }

            if let Some(out) = hidden_out.as_deref_mut() {
                out.resize(dim, 0.0);
                hip.copy_d2h_f32(buf.x, out)?;
                return Ok(());
            }
            Self::switch_local_gpu(
                primary_hip.device,
                &mut hip,
                &mut resident,
                &mut buf,
                &mut xfer,
                primary_hip,
                primary_resident,
                &peer,
                cfg,
                vocab,
                max_kv,
                q_workspace,
                kv_workspace,
                1,
                1,
            )?;
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
        fwd_segment_mark(hip.device, "end");
        if std::env::var("ALLPACA_FWD_LOG").is_ok_and(|v| v == "1" || v == "2") {
            let _ = allpaca_hip::stream_synchronize();
            eprintln!("FWD1 pos={pos} ms={:.2}", fwd_t0.elapsed().as_secs_f64() * 1e3);
        }
        result
    }

    /// Batched HIP prefill: one GEMM with m = N per weight, attention n_q = N.
    /// MoE expert GEMMs group tokens that share an expert. Hybrid SSM/GDN
    /// layers walk token rows with decode kernels.
    #[allow(clippy::too_many_arguments)]
    fn forward_tokens_device(
        &self,
        tokens: &[u32],
        segs: &mut [SeqRows<'_>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        write_logits: bool,
        row_logits: Option<&mut [f32]>,
        out_row_hiddens: Option<&mut Vec<f32>>,
        vision: Option<&VisionSeq>,
    ) -> Result<(), AllpacaError> {
        self.forward_tokens_device_impl(
            tokens,
            segs,
            kv,
            metrics,
            logits,
            hip,
            resident,
            write_logits,
            row_logits,
            out_row_hiddens,
            vision,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_hidden_tokens_device(
        &self,
        start_pos: usize,
        x: &mut [f32],
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        recurrent: &mut RecurrentState,
        request_id: u64,
    ) -> Result<(), AllpacaError> {
        let dim = self.config.dim;
        if x.is_empty() || x.len() % dim != 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "hidden rows must be a non-empty multiple of the model dimension".into(),
            )));
        }
        let tokens = vec![0; x.len() / dim];
        let mut segs = [SeqRows {
            start_pos,
            n_rows: tokens.len(),
            page_ids,
            request_id,
            recurrent: Some(recurrent),
        }];
        // Hosted rows need the same arithmetic as the coordinator's RPC prefill.
        let _exact_rows = allpaca_hip::ExactRows::enable();
        self.forward_tokens_device_impl(
            &tokens,
            &mut segs,
            kv,
            metrics,
            &mut [],
            hip,
            resident,
            false,
            None,
            None,
            None,
            Some(x),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_tokens_device_impl(
        &self,
        tokens: &[u32],
        segs: &mut [SeqRows<'_>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        write_logits: bool,
        row_logits: Option<&mut [f32]>,
        mut out_row_hiddens: Option<&mut Vec<f32>>,
        vision: Option<&VisionSeq>,
        hidden_rows: Option<&mut [f32]>,
    ) -> Result<(), AllpacaError> {
        // Every prefill takes exact rows: each row of a wide GEMM equals that
        // row computed alone, so the bits don't depend on where layers run.
        // RPC workers compute rows that way (`serve_connection_ext`); with the
        // BLAS/WMMA paths here, attaching a worker changed the local layers'
        // bits too, and Qwen3.8 text with them (Q23). MoE models needed it
        // with nothing remote as well: Ling-mini's local prefill changed
        // text between fresh starts without it (S2). The one exception is the
        // opt-in `experimental.row_variant_kernels` prefill of a dense model
        // with nothing remote, which exists to run the WMMA tiles instead.
        let moe = self.config.n_experts > 0;
        let remote = self.rpc.as_ref().is_some_and(|rpc| rpc.sends_work_remote());
        let _exact_rows = (moe || remote || !allpaca_hip::prefill_wmma_active())
            .then(allpaca_hip::ExactRows::enable);
        let cfg = &self.config;
        if segs.is_empty()
            || segs.iter().map(|s| s.n_rows).sum::<usize>() != tokens.len()
            || (vision.is_some() && segs.len() > 1)
        {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "sequence rows do not cover the batch".into(),
            )));
        }
        // Row r's absolute position, and each sequence's first row.
        let row_pos: Vec<usize> = segs
            .iter()
            .flat_map(|s| s.start_pos..s.start_pos + s.n_rows)
            .collect();
        let seg_row0: Vec<usize> = segs
            .iter()
            .scan(0, |r, s| {
                let r0 = *r;
                *r += s.n_rows;
                Some(r0)
            })
            .collect();
        let start_pos = segs[0].start_pos;
        let request_id = segs[0].request_id;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        let n_tok = tokens.len();
        let mrope = vision.and_then(VisionSeq::mrope);
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
        // The first layer with pages: a recurrent layer keeps none.
        let page_tokens = segs[0]
            .page_ids
            .iter()
            .find_map(|p| p.first())
            .and_then(|id| kv.get(*id).map(|p| p.capacity_tokens as usize))
            .unwrap_or(128);
        let max_kv = segs
            .iter()
            .map(|s| {
                let seq_end = s.start_pos + s.n_rows;
                s.page_ids
                    .iter()
                    .map(|layer| layer.len().saturating_mul(page_tokens))
                    .max()
                    .unwrap_or(seq_end)
                    .max(seq_end)
            })
            .max()
            .unwrap_or(0);
        hip.bind()?;
        let logits_rows = if row_logits.is_some() { n_tok } else { 1 };
        let buf = hip.ensure_decode_batch(
            dim,
            kv_dim,
            cfg.moe_workspace(),
            vocab,
            max_kv,
            n_tok,
            logits_rows,
            q_workspace,
        )?;
        let fwd_t0 = std::time::Instant::now();
        fwd_segment_mark(hip.device, "start");
        hip.begin_decode_profile();
        let result = (|| {
            let primary_hip = hip;
            let primary_resident = resident;
            let peer = self.rpc.as_ref().and_then(|r| r.local_peer());
            let mut hip = primary_hip;
            let mut resident = primary_resident;
            let mut buf = buf;
            let mut xfer = vec![0.0f32; n_tok * dim];
            let mut normed_ready = false;
            if let Some(input) = hidden_rows.as_deref() {
                hip.copy_h2d_f32(buf.x, input)?;
            } else {
                resident.embed_gather_rows(hip, tokens, buf.x, dim as i32)?;
                record_elementwise(Some(metrics));
                if let Some(v) = vision {
                    // Soft (image) rows replace their placeholder token embeddings.
                    for t in 0..n_tok {
                        if let Some(row) = v.soft_row(start_pos + t, dim) {
                            hip.copy_h2d_f32(buf.x_row(t, dim), row)?;
                        }
                    }
                }
            }

            let m = n_tok as i32;
            for (layer_idx, host_layer) in self.layers.iter().enumerate() {
                if let Some(rpc) = &self.rpc {
                    if rpc.is_remote(layer_idx) {
                        if rpc.is_span_start(layer_idx) {
                            let mut host_x = vec![0.0f32; n_tok * dim];
                            hip.copy_d2h_f32(buf.x, &mut host_x)?;
                            let consecutive = row_pos.windows(2).all(|w| w[1] == w[0] + 1);
                            if rpc.is_hosted(layer_idx) {
                                if consecutive && segs.len() == 1 {
                                    rpc.run_hosted_rows(
                                        layer_idx,
                                        row_pos[0],
                                        segs[0].recurrent.as_deref_mut(),
                                        &mut host_x,
                                    )?;
                                } else {
                                    for (si, seg) in segs.iter_mut().enumerate() {
                                        for t in 0..seg.n_rows {
                                            let r = seg_row0[si] + t;
                                            rpc.run_hosted(
                                                layer_idx,
                                                row_pos[r],
                                                seg.recurrent.as_deref_mut(),
                                                &mut host_x[r * dim..(r + 1) * dim],
                                            )?;
                                        }
                                    }
                                }
                            } else if consecutive {
                                // One prompt: the worker runs the rows together.
                                rpc.run_rows(layer_idx, row_pos[0], dim, &mut host_x)?;
                            } else {
                                for t in 0..n_tok {
                                    let row = &mut host_x[t * dim..(t + 1) * dim];
                                    rpc.run_layer(layer_idx, row_pos[t], row)?;
                                }
                            }
                            hip.copy_h2d_f32(buf.x, &host_x)?;
                        }
                        normed_ready = false;
                        continue;
                    }
                    let want = rpc.local_device(layer_idx);
                    if Self::switch_local_gpu(
                        want,
                        &mut hip,
                        &mut resident,
                        &mut buf,
                        &mut xfer,
                        primary_hip,
                        primary_resident,
                        &peer,
                        cfg,
                        vocab,
                        max_kv,
                        q_workspace,
                        kv_workspace,
                        n_tok,
                        logits_rows,
                    )? {
                        normed_ready = false;
                    }
                }
                if let Some(k) = self.layer_capture.slot(layer_idx) {
                    if segs.len() == 1 {
                        let rows: Vec<(usize, u32)> =
                            row_pos.iter().copied().zip(tokens.iter().copied()).collect();
                        let mut host = vec![0.0f32; n_tok * dim];
                        hip.copy_d2h_f32(buf.x, &mut host)?;
                        self.layer_capture.record(k, &rows, &host);
                    } else {
                        self.layer_capture.invalidate();
                    }
                }
                let layer = &resident.layers[layer_idx];
                if !Self::layer_has_device_weights(layer) {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "layer {layer_idx} has no weights on HIP {}",
                        hip.device
                    ))));
                }
                if !normed_ready {
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
                        hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                        normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
                        if normed_ready {
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
                        }
                        continue;
                    }
                    if self.device_gdn_rows(
                        layer_idx,
                        host_layer,
                        hip,
                        resident,
                        buf,
                        n_tok,
                        metrics,
                        segs,
                    )? {
                        normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
                        continue;
                    }
                    for (si, seg) in segs.iter_mut().enumerate() {
                        for t in seg_row0[si]..seg_row0[si] + seg.n_rows {
                            let row = Self::decode_row_view(buf, t, dim);
                            let mut side = DeviceDecodeSide::new(seg.request_id);
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
                                seg.recurrent.as_deref_mut(),
                            )?;
                            if !handled {
                                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                                    "hybrid prefill layer {layer_idx} was not handled"
                                ))));
                            }
                        }
                    }
                    normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
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
                self.device_qkv_bias(hip, resident, layer, buf, n_tok, al.q_out, al.kv_dim)?;
                if al.gated {
                    for t in 0..n_tok {
                        hip.deinterleave_q_gate_device(
                            buf.q_row(t, al.q_out),
                            buf.attn_row(t, al.attn_dim),
                            buf.gate_row(t, al.attn_dim),
                            al.n_heads as i32,
                            al.head_dim as i32,
                        )?;
                        if layer.q_norm.n == al.head_dim {
                            hip.rmsnorm_heads_device(
                                buf.attn_row(t, al.attn_dim),
                                resident.f32_or_dequant(hip, layer.q_norm)?,
                                al.n_heads as i32,
                                al.head_dim as i32,
                                cfg.eps,
                            )?;
                        }
                    }
                    hip.copy_d2d_f32(buf.q, buf.attn, n_tok * al.attn_dim)?;
                    if layer.k_norm.n == al.head_dim {
                        hip.rmsnorm_heads_device(
                            buf.k,
                            resident.f32_or_dequant(hip, layer.k_norm)?,
                            (n_tok * al.n_kv_heads) as i32,
                            al.head_dim as i32,
                            cfg.eps,
                        )?;
                    }
                } else if al.q_out == al.n_heads * al.head_dim {
                    // Plain attention with per-head Q/K RMSNorm before the
                    // rotation (Qwen3, Ling 2.0, Gemma 3). Rows are contiguous.
                    if layer.q_norm.n == al.head_dim {
                        hip.rmsnorm_heads_device(
                            buf.q,
                            resident.f32_or_dequant(hip, layer.q_norm)?,
                            (n_tok * al.n_heads) as i32,
                            al.head_dim as i32,
                            cfg.eps,
                        )?;
                    }
                    if layer.k_norm.n == al.head_dim {
                        hip.rmsnorm_heads_device(
                            buf.k,
                            resident.f32_or_dequant(hip, layer.k_norm)?,
                            (n_tok * al.n_kv_heads) as i32,
                            al.head_dim as i32,
                            cfg.eps,
                        )?;
                    }
                }
                let q_stride = if al.gated { al.attn_dim } else { al.q_out };
                let theta = cfg.rope_theta_for_layer(layer_idx);
                if cfg.rope_none {
                } else if let Some(n_rot) = cfg.neox_rot_dims(layer_idx, al.head_dim) {
                    for t in 0..n_tok {
                        hip.rope_imrope_qk_device(
                            buf.q_row(t, q_stride),
                            buf.k_row(t, al.kv_dim),
                            al.n_heads as i32,
                            al.n_kv_heads as i32,
                            al.head_dim as i32,
                            n_rot as i32,
                            mrope
                                .map_or([row_pos[t]; 3], |m| m.rope_pos(row_pos[t]))
                                .map(|p| p as i32),
                            theta,
                            cfg.rope_sections,
                        )?;
                    }
                } else {
                    for (si, seg) in segs.iter().enumerate() {
                        hip.rope_qk_device_rows(
                            buf.q_row(seg_row0[si], q_stride),
                            buf.k_row(seg_row0[si], al.kv_dim),
                            seg.n_rows as i32,
                            al.n_heads as i32,
                            al.n_kv_heads as i32,
                            al.head_dim as i32,
                            seg.start_pos as i32,
                            theta,
                        )?;
                    }
                }
                record_elementwise(Some(metrics));

                for (si, seg) in segs.iter().enumerate() {
                    let r0 = seg_row0[si];
                    let (start_pos, n_tok, m) = (seg.start_pos, seg.n_rows, seg.n_rows as i32);
                    kv.write_slots_from_device(
                        &seg.page_ids[layer_idx],
                        start_pos,
                        n_tok,
                        page_tokens,
                        al.n_kv_heads,
                        al.head_dim,
                        buf.k_row(r0, al.kv_dim),
                        buf.v_row(r0, al.kv_dim),
                        None,
                    )?;

                    let (packed_k, packed_v, n_kv) = kv.kv_device_views(
                        &seg.page_ids[layer_idx],
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
                            buf.q_row(r0, q_stride),
                            packed_k,
                            packed_v,
                            buf.attn_row(r0, al.attn_dim),
                            m,
                            n_kv as i32,
                            al.n_heads as i32,
                            al.n_kv_heads as i32,
                            al.head_dim as i32,
                            &mask,
                        )?;
                    } else {
                        hip.attention_device(
                            buf.q_row(r0, q_stride),
                            packed_k,
                            packed_v,
                            buf.attn_row(r0, al.attn_dim),
                            m,
                            n_kv as i32,
                            al.n_heads as i32,
                            al.n_kv_heads as i32,
                            al.head_dim as i32,
                            true,
                        )?;
                    }
                }
                if al.gated {
                    hip.sigmoid_mul_device(buf.attn, buf.gate, (n_tok * al.attn_dim) as i32)?;
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
                    hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                    normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
                    if normed_ready {
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
                    }
                    continue;
                }
                hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
                if !dense_ffn {
                    normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
                    if normed_ready {
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
                            cfg.dense_ff() as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_up,
                            buf.up,
                            m,
                            cfg.dense_ff() as i32,
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
                        cfg.dense_ff() as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_up,
                        buf.up,
                        m,
                        cfg.dense_ff() as i32,
                        dim as i32,
                    )?;
                }
                self.device_act_mul(hip, buf.gate, buf.up, (n_tok * cfg.dense_ff()) as i32)?;
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
                            cfg.dense_ff() as i32,
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
                        cfg.dense_ff() as i32,
                    )?;
                }
                drop(ffn);
                hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
                normed_ready = self.fuse_next_attn_norm(layer_idx, hip.device);
                if normed_ready {
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

            Self::switch_local_gpu(
                primary_hip.device,
                &mut hip,
                &mut resident,
                &mut buf,
                &mut xfer,
                primary_hip,
                primary_resident,
                &peer,
                cfg,
                vocab,
                max_kv,
                q_workspace,
                kv_workspace,
                n_tok,
                logits_rows,
            )?;

            if let Some(out) = hidden_rows {
                hip.copy_d2h_f32(buf.x, out)?;
            } else if row_logits.is_some() {
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
            } else if let Some(out) = out_row_hiddens.as_mut() {
                // Every row's output_norm hidden (an MTP draft's process KV is
                // built from them), plus last-row logits when asked.
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
                out.resize(n_tok * dim, 0.0);
                hip.copy_d2h_f32(buf.normed, out)?;
                if write_logits {
                    let _lm = hip.lm_head_scope();
                    resident.gemm_nt(
                        hip,
                        buf.normed_row(n_tok - 1, dim),
                        resident.lm_head,
                        buf.logits,
                        1,
                        vocab as i32,
                        dim as i32,
                    )?;
                    hip.start_logits_d2h(buf.logits, vocab)?;
                }
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
        fwd_segment_mark(hip.device, "end");
        if std::env::var("ALLPACA_FWD_LOG").is_ok_and(|v| v == "1" || v == "2") {
            let _ = allpaca_hip::stream_synchronize();
            eprintln!(
                "FWD rows={} ms={:.2}",
                tokens.len(),
                fwd_t0.elapsed().as_secs_f64() * 1e3
            );
        }
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
        let mut scratch = vec![0.0f32; dim.max(cfg.moe_workspace()).max(kv_dim)];
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
                add_row_bias(&mut qs[i * dim..(i + 1) * dim], &layer.attn_bias[0]);
                add_row_bias(&mut ks[i * kv_dim..(i + 1) * kv_dim], &layer.attn_bias[1]);
                add_row_bias(&mut vs[i * kv_dim..(i + 1) * kv_dim], &layer.attn_bias[2]);
                let pos = prompt_len + forest.nodes[i].depth as usize - 1;
                for h in 0..if cfg.rope_none { 0 } else { cfg.n_heads } {
                    rope(
                        &mut qs[i * dim + h * head_dim..i * dim + (h + 1) * head_dim],
                        pos,
                        cfg.rope_theta,
                    )?;
                }
                for h in 0..if cfg.rope_none { 0 } else { cfg.n_kv_heads } {
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
            match self.forward_dense_device(token, pos, cache, logits, hip, resident, metrics, true) {
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
        if !self.host_weights_ready() {
            // Large GGUF drafts skip the host F32 copy; without a resident
            // upload there is nothing to run on the host.
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "draft has no HIP resident weights and no host weights".into(),
            )));
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
            add_row_bias(&mut q, &layer.attn_bias[0]);
            add_row_bias(&mut k, &layer.attn_bias[1]);
            add_row_bias(&mut v, &layer.attn_bias[2]);
            if cfg.rope_none {
            } else if let Some(n_rot) = cfg.neox_rot_dims(layer_idx, head_dim) {
                for h in q.chunks_exact_mut(head_dim).chain(k.chunks_exact_mut(head_dim)) {
                    rope_imrope_neox(h, pos, cfg.rope_theta, n_rot, [0; 4])?;
                }
            } else {
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
            }
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
    /// MTP process step: append `token` to the draft KV (and roll `h_nextn`)
    /// without the vocab-wide lm_head. Only proposals need draft logits.
    pub fn process_dense_on(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if let (Some(hip), Some(resident)) = (hip, resident) {
            cache.forwards = cache.forwards.saturating_add(1);
            self.forward_dense_device(token, pos, cache, &mut [], hip, resident, metrics, false)?;
            if let Some(metrics) = metrics {
                metrics
                    .draft_logits_skipped
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            return Ok(());
        }
        let mut logits = vec![0.0f32; self.config.vocab_size];
        self.forward_dense_on(token, pos, cache, &mut logits, hip, resident, metrics)
    }

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
        write_logits: bool,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;
        if write_logits && logits.len() < vocab {
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
            let gated = q_out == wo_k.saturating_mul(2) && (wo_k > dim || cfg.gated_attn_q);
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
            self.device_qkv_bias(hip, resident, layer, buf, 1, q_out, kv_dim)?;

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
            } else if q_out == n_heads * head_dim {
                // Plain attention with per-head Q/K RMSNorm before the
                // rotation (Qwen3, Ling 2.0, Gemma 3).
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
            if cfg.rope_none {
            } else if let Some(n_rot) = cfg.neox_rot_dims(layer_idx, head_dim) {
                hip.rope_imrope_qk_device(
                    buf.q,
                    buf.k,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    n_rot as i32,
                    [pos as i32; 3],
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
            self.device_headwise_attn_gate(hip, resident, layer, buf, 1, n_heads, head_dim, dim)?;
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
                if self.fuse_next_attn_norm(layer_idx, hip.device) {
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
                    cfg.dense_ff() as i32,
                    dim as i32,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_up,
                    buf.up,
                    1,
                    cfg.dense_ff() as i32,
                    dim as i32,
                )?;
                self.device_act_mul(hip, buf.gate, buf.up, cfg.dense_ff() as i32)?;
                record_elementwise(metrics);
                resident.gemm_nt(
                    hip,
                    buf.gate,
                    w_down,
                    buf.proj,
                    1,
                    dim as i32,
                    cfg.dense_ff() as i32,
                )?;
                drop(ffn);
                if self.fuse_next_attn_norm(layer_idx, hip.device) {
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
            hip.publish_logits(buf.logits, &mut logits[..vocab])?;
        }
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

    pub(crate) fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
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

    /// The warmed K/V, its tokens and the target's last hidden state (`seed`)
    /// for a prompt-cache checkpoint. Reads device K/V back to the host.
    /// `h_nextn` is left out: the next process step seeds it from `seed`.
    pub(crate) fn to_cache_bytes(&self, seed: &[f32]) -> Result<Vec<u8>, AllpacaError> {
        let len = |n: usize| u32::try_from(n).expect("draft cache length fits in u32");
        // Only the device path records the width.
        let kv_dim = match self.layers.first() {
            Some(layer) if self.kv_dim == 0 => layer.k.len().checked_div(self.seq.len()).unwrap_or(0),
            _ => self.kv_dim,
        };
        let mut out = Vec::new();
        out.extend_from_slice(&len(kv_dim).to_le_bytes());
        out.extend_from_slice(&len(self.seq.len()).to_le_bytes());
        for t in &self.seq {
            out.extend_from_slice(&t.to_le_bytes());
        }
        crate::kv_codec::write_f32s(&mut out, seed);
        match self.device {
            Some(device) if !self.device_layers.is_empty() => {
                allpaca_hip::set_device(device)?;
                let n = self.tokens.saturating_mul(self.kv_dim);
                out.extend_from_slice(&len(self.device_layers.len()).to_le_bytes());
                let mut row = vec![0.0f32; n];
                for layer in &self.device_layers {
                    for src in [layer.k, layer.v] {
                        allpaca_hip::memcpy(
                            row.as_mut_ptr().cast(),
                            src,
                            n * 4,
                            allpaca_hip::MEMCPY_D2H,
                        )?;
                        crate::kv_codec::write_f32s(&mut out, &row);
                    }
                }
            }
            _ => {
                out.extend_from_slice(&len(self.layers.len()).to_le_bytes());
                for layer in &self.layers {
                    crate::kv_codec::write_f32s(&mut out, &layer.k);
                    crate::kv_codec::write_f32s(&mut out, &layer.v);
                }
            }
        }
        Ok(out)
    }

    /// What [`Self::to_cache_bytes`] wrote, with its seed: K/V on the host,
    /// which the first device step uploads. `None` when `bytes` is not one.
    pub(crate) fn from_cache_bytes(bytes: &[u8]) -> Option<(Self, Vec<f32>)> {
        let mut r = crate::kv_codec::Reader(bytes);
        let kv_dim = r.u32()? as usize;
        let n_seq = r.u32()? as usize;
        let seq = r
            .take(n_seq.checked_mul(4)?)?
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let seed = r.f32s()?;
        let n_layers = r.u32()? as usize;
        // Each layer's two lengths take 16 bytes.
        if n_layers > r.0.len() / 16 {
            return None;
        }
        let mut layers = Vec::with_capacity(n_layers);
        let want = n_seq.checked_mul(kv_dim)?;
        for _ in 0..n_layers {
            let (k, v) = (r.f32s()?, r.f32s()?);
            if k.len() != want || v.len() != want {
                return None;
            }
            layers.push(DenseLayerKv { k, v });
        }
        if !r.0.is_empty() {
            return None;
        }
        let mut kv = Self::default();
        kv.layers = layers;
        kv.kv_dim = kv_dim;
        kv.tokens = n_seq;
        kv.seq = seq;
        Some((kv, seed))
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
        let new_cap = need.max(self.tokens).max(self.capacity.saturating_mul(2)).max(8);
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
        if copy_tokens == 0 {
            self.upload_host_layers()?;
        }
        Ok(())
    }

    /// Move K/V restored from a prompt-cache checkpoint onto fresh device layers.
    fn upload_host_layers(&mut self) -> Result<(), AllpacaError> {
        let n = self.tokens.saturating_mul(self.kv_dim);
        if n == 0 || self.layers.iter().any(|l| l.k.len() != n || l.v.len() != n) {
            return Ok(());
        }
        for (host, dev) in self.layers.iter_mut().zip(&self.device_layers) {
            for (src, dst) in [(&host.k, dev.k), (&host.v, dev.v)] {
                allpaca_hip::memcpy(dst, src.as_ptr().cast(), n * 4, allpaca_hip::MEMCPY_H2D)?;
            }
            host.k = Vec::new();
            host.v = Vec::new();
        }
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

#[cfg(feature = "vulkan")]
#[path = "vulkan_decode.rs"]
mod vulkan_decode;

#[cfg(feature = "vulkan")]
pub use vulkan_decode::VulkanResidentModel;

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

/// Which layers keep KV pages: every layer except recurrent ones (SSM or
/// Gated DeltaNet), which never write KV. A model without a per-layer list
/// keeps pages on every layer.
pub fn kv_layer_mask(model: &TinyModel) -> Vec<bool> {
    if model.g4.is_some() {
        // gemma4 keeps its K/V in the sequence state.
        return vec![false; model.config.n_layers];
    }
    if model.layers.len() != model.config.n_layers {
        return vec![true; model.config.n_layers];
    }
    model
        .layers
        .iter()
        .map(|l| !(l.block == HybridLayerKind::Ssm || l.gdn.is_some() || l.ssm.is_some()))
        .collect()
}

/// Allocate one page per layer covering `n_tokens` (rounded up to page_tokens).
pub fn alloc_pages_for_seq(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    n_tokens: usize,
    metrics: &Metrics,
) -> Result<Vec<Vec<u64>>, AllpacaError> {
    alloc_pages_for_seq_masked(kv, cfg, page_tokens, n_tokens, None, metrics)
}

/// [`alloc_pages_for_seq`] for the layers `keeps` marks (see
/// [`kv_layer_mask`]); the others get an empty page list. A hybrid model with
/// 8 attention layers of 32 otherwise takes 4x the KV it uses.
pub fn alloc_pages_for_seq_masked(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    n_tokens: usize,
    keeps: Option<&[bool]>,
    metrics: &Metrics,
) -> Result<Vec<Vec<u64>>, AllpacaError> {
    let n_pages = n_tokens.div_ceil(page_tokens as usize).max(1);
    let mut pages = Vec::with_capacity(cfg.n_layers);
    for layer in 0..cfg.n_layers {
        if keeps.is_some_and(|k| !k.get(layer).copied().unwrap_or(true)) {
            pages.push(Vec::new());
            continue;
        }
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
        // An empty list is a layer without KV (see `kv_layer_mask`): a layer
        // that keeps KV always has at least one page.
        if layer_pages.is_empty() {
            continue;
        }
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

#[cfg(all(test, feature = "hip"))]
mod moe_prefill_rows_tests {
    use super::*;
    use allpaca_gguf::{write_gguf_quant, GgmlType, GgufWriteQuantTensor};

    /// A local MoE model at a width the wide GEMMs take (dim and ff 256) with
    /// stacked Q8_0 experts, like a real K-quant / Q8_0 MoE file.
    fn tiny_q8_moe(dir: &std::path::Path) -> std::path::PathBuf {
        let f32_src = dir.join("f32.gguf");
        crate::calib::tests::write_tiny_qwen35moe_gguf_sized(&f32_src, 4, false, 256, 32, 256);
        let f32_file = GgufFile::open(&f32_src).unwrap();
        let quant: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = f32_file
            .tensors
            .iter()
            .map(|info| {
                if info.name.ends_with("_exps.weight") {
                    let data = f32_file.tensor_f32(&info.name).unwrap();
                    (info.name.clone(), info.dims.clone(), GgmlType::Q8_0, allpaca_gguf::quantize_q8_0(&data))
                } else {
                    let bytes = f32_file.tensor_bytes(&info.name).unwrap().to_vec();
                    (info.name.clone(), info.dims.clone(), info.ggml_type, bytes)
                }
            })
            .collect();
        let tensors: Vec<GgufWriteQuantTensor<'_>> = quant
            .iter()
            .map(|(name, dims, ggml_type, bytes)| GgufWriteQuantTensor {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: *ggml_type,
                bytes,
            })
            .collect();
        let path = dir.join("q8.gguf");
        write_gguf_quant(&path, &f32_file.metadata, &[], &tensors).unwrap();
        path
    }

    /// Local MoE prefill (no RPC) gives each row the bits it gets alone.
    /// Without exact rows the wide GEMMs sum in another order, and the same
    /// prompt changed text between fresh starts on Ling-mini (S2, measured).
    #[test]
    fn local_moe_prefill_rows_equal_token_by_token() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let dir = std::env::temp_dir().join(format!("allpaca-moe-rows-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = GgufFile::open(&tiny_q8_moe(&dir)).unwrap();
        let model = TinyModel::from_gguf(&file).unwrap();
        assert!(model.config.n_experts > 0 && model.rpc.is_none());
        let resident = ResidentModel::upload_gguf(&file, &model.config, 0).unwrap();
        let hip = allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 256).unwrap();
        let cfg = &model.config;
        let tokens: Vec<u32> = (0..64u32).map(|i| 1 + (i * 7 + i / 3) % 5).collect();
        let run = |batched: bool, request_id: u64| -> Vec<f32> {
            let metrics = Metrics::default();
            let mut kv = crate::kv_tier::KvTierManager::with_hip(64 << 20, 64 << 20, 0.75, Some(hip.device));
            let keeps = kv_layer_mask(&model);
            let pages = alloc_pages_for_seq_masked(&mut kv, cfg, 128, tokens.len() + 1, Some(&keeps), &metrics).unwrap();
            let mut recurrent = crate::anchors::RecurrentState::zeros(cfg.n_layers, cfg.dim);
            let mut logits = vec![0.0f32; cfg.vocab_size];
            if batched {
                let mut segs = [SeqRows {
                    start_pos: 0,
                    n_rows: tokens.len(),
                    page_ids: &pages,
                    request_id,
                    recurrent: Some(&mut recurrent),
                }];
                model
                    .forward_tokens_device(
                        &tokens, &mut segs, &mut kv, &metrics, &mut logits, &hip, &resident, true, None,
                        None, None,
                    )
                    .unwrap();
            } else {
                for (pos, &tok) in tokens.iter().enumerate() {
                    model
                        .forward_token_device(
                            tok, pos, &pages, &mut kv, &metrics, &mut logits, &hip, &resident, true, None,
                            None, Some(&mut recurrent), None, None,
                        )
                        .unwrap();
                }
            }
            hip.gdn_clear_request(request_id);
            logits
        };
        let one_by_one = run(false, 11);
        let rows = run(true, 12);
        let differ = one_by_one.iter().zip(&rows).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(differ, 0, "batched MoE prefill changed {differ} of {} logits", rows.len());
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod host_exp_tests {
    use super::*;

    /// The shared-expert gate and host routing use `sigmoid_stable` on the
    /// coordinator and on workers. The platform `exp` (MSVC CRT vs glibc)
    /// differed in the last bit at one decode step of Qwen3-Coder-Next and
    /// split hosted RPC spans from the separate path, so these bits are
    /// pinned: they must hold on Windows and Linux alike.
    #[test]
    fn host_sigmoid_has_the_same_bits_on_every_platform() {
        let expect = [
            (-7.25f32, 0x3a3a_092d_u32),
            (-1.5, 0x3e3a_cdc1),
            (-0.3, 0x3ed9_e2ab),
            (0.0, 0x3f00_0000),
            (0.7, 0x3f2b_0e5a),
            (2.25, 0x3f67_972d),
            (9.5, 0x3f7f_fb18),
        ];
        for (v, bits) in expect {
            assert_eq!(sigmoid_stable(v).to_bits(), bits, "sigmoid({v})");
        }
    }
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
        meta.insert("spark2_5.attention.head_count".into(), GgufValue::Uint32(4));
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
        meta.insert("deepseek2.expert_shared_count".into(), GgufValue::Uint32(2));
        meta.insert("deepseek2.expert_gating_func".into(), GgufValue::Uint32(2));
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
        let routed = topk_route(&x, &w, cfg.n_experts, cfg.n_active, &cfg, None, &[]);
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
        let routed = topk_route(&x, &w, cfg.n_experts, cfg.n_active, &cfg, None, &[]);
        assert_eq!(routed.len(), 2);
        assert_eq!(routed[0].0, 0);
        assert_eq!(routed[1].0, 1);
        let sum: f32 = routed.iter().map(|(_, wt)| *wt).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(routed[0].1 > routed[1].1);
    }

    /// Nemotron-H / DeepSeek-V3 `exp_probs_b`: experts are picked on
    /// `sigmoid + bias` and weighted by the plain sigmoid; a disabled expert
    /// stays out whatever its bias.
    #[test]
    fn sigmoid_topk_selects_on_the_biased_score() {
        let mut cfg = ModelConfig::tiny_moe();
        cfg.moe_sigmoid_gating = true;
        cfg.expert_weights_norm = false;
        cfg.expert_weights_scale = 1.0;
        let logits = vec![2.0f32, 1.0, 0.0, -1.0];
        let plain = topk_from_logits(&mut logits.clone(), 2, &cfg, None, &[]);
        assert_eq!(plain.iter().map(|r| r.0).collect::<Vec<_>>(), vec![0, 1]);
        let bias = [0.0f32, 0.0, 0.0, 5.0];
        let got = topk_from_logits(&mut logits.clone(), 2, &cfg, None, &bias);
        assert_eq!(got.iter().map(|r| r.0).collect::<Vec<_>>(), vec![3, 0]);
        assert!((got[0].1 - sigmoid_stable(-1.0)).abs() < 1e-6, "weight is unbiased: {got:?}");
        assert!((got[1].1 - sigmoid_stable(2.0)).abs() < 1e-6, "{got:?}");
        let mask = [false, false, false, true];
        let got = topk_from_logits(&mut logits.clone(), 2, &cfg, Some(&mask), &bias);
        assert_eq!(got.iter().map(|r| r.0).collect::<Vec<_>>(), vec![0, 1]);
    }

    /// One Qwen3.5-style Gated DeltaNet layer (dim 16, 2 V heads of 4).
    fn write_one_gdn_layer_gguf(path: &std::path::Path) {
        use allpaca_gguf::{GgufValue, GgufWriteTensor};
        let mut meta = std::collections::HashMap::new();
        meta.insert("general.architecture".into(), GgufValue::String("qwen35".into()));
        for (k, v) in [
            ("state_size", 4u32),
            ("inner_size", 8),
            ("time_step_rank", 2),
            ("group_count", 1),
            ("conv_kernel", 4),
        ] {
            meta.insert(format!("qwen35.ssm.{k}"), GgufValue::Uint32(v));
        }
        let (dim, qkv, value, heads) = (16u64, 16u64, 8u64, 2u64);
        let data: Vec<Vec<f32>> = [heads, heads, 4, 4 * qkv, dim * qkv, dim * value, dim * heads, dim * heads, value * dim]
            .iter()
            .map(|&n| (0..n).map(|i| i as f32 * 0.01).collect())
            .collect();
        let shapes: [(&str, Vec<u64>); 9] = [
            ("ssm_a", vec![heads]),
            ("ssm_dt.bias", vec![heads]),
            ("ssm_norm.weight", vec![4]),
            ("ssm_conv1d.weight", vec![4, qkv]),
            ("attn_qkv.weight", vec![dim, qkv]),
            ("attn_gate.weight", vec![dim, value]),
            ("ssm_alpha.weight", vec![dim, heads]),
            ("ssm_beta.weight", vec![dim, heads]),
            ("ssm_out.weight", vec![value, dim]),
        ];
        let tensors: Vec<GgufWriteTensor<'_>> = shapes
            .into_iter()
            .zip(&data)
            .map(|((name, dims), d)| GgufWriteTensor {
                name: format!("blk.0.{name}"),
                dims,
                data: d,
            })
            .collect();
        allpaca_gguf::write_gguf(path, &meta, &tensors).unwrap();
    }

    /// GGUF conversion permutes Llama's Q/K for adjacent-pair (NORM) RoPE;
    /// Qwen2-family GGUFs keep NeoX pairing. Picking the wrong one garbled
    /// both Llama-3.2 and Qwen2.5 on HIP.
    #[test]
    fn rope_style_follows_gguf_architecture() {
        assert!(!super::gguf_rope_neox("llama"));
        assert!(!super::gguf_rope_neox("deepseek2"));
        assert!(super::gguf_rope_neox("qwen2"));
        assert!(super::gguf_rope_neox("qwen3next"));
        assert!(super::gguf_rope_none("nemotron_h_moe"));
        assert!(super::gguf_rope_none("nemotron_h"));
        assert!(!super::gguf_rope_none("nemotron"));
        assert!(!super::gguf_rope_none("llama"));
        let mut cfg = ModelConfig::tiny();
        let hd = cfg.head_dim();
        // Llama GGUFs set rope.dimension_count == head_dim: still NORM.
        cfg.rope_dims = hd;
        assert_eq!(cfg.neox_rot_dims(0, hd), None);
        cfg.rope_neox = true;
        assert_eq!(cfg.neox_rot_dims(0, hd), Some(hd));
        cfg.rope_neox = false;
        cfg.rope_dims = 0;
        assert_eq!(cfg.neox_rot_dims(0, hd), None);
        // Partial rotation is NeoX whatever the architecture.
        cfg.rope_dims = 2;
        assert_eq!(cfg.neox_rot_dims(0, hd), Some(2));
    }

    #[test]
    fn qkv_bias_adds_per_row() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        super::add_row_bias(&mut x, &[10.0, 20.0]);
        assert_eq!(x, [11.0, 22.0, 13.0, 24.0]);
        super::add_row_bias(&mut x, &[]);
        assert_eq!(x, [11.0, 22.0, 13.0, 24.0]);
    }

    /// The GPU router picks the host's experts in the host's order, with its
    /// weights, and computes the shared-expert gate.
    #[test]
    fn gpu_moe_routing_matches_host_topk() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let (rows, n_exp, k, dim) = (3usize, 512usize, 10usize, 256usize);
        let cfg = ModelConfig::tiny();
        let logits: Vec<f32> = (0..rows * n_exp)
            .map(|i| ((i * 7919 % 1009) as f32) * 0.013 - 6.0)
            .collect();
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i % 37) as f32) * 0.05 - 0.9).collect();
        let gate: Vec<f32> = (0..dim).map(|i| ((i % 11) as f32) * 0.02 - 0.1).collect();
        let up = |v: &[f32]| {
            let p = allpaca_hip::device_malloc(v.len() * 4, 0).unwrap();
            allpaca_hip::memcpy(p, v.as_ptr().cast(), v.len() * 4, allpaca_hip::MEMCPY_H2D).unwrap();
            p
        };
        let (dl, dx, dg) = (up(&logits), up(&x), up(&gate));
        let table = allpaca_hip::device_malloc(1 << 16, 0).unwrap();
        let (routes, _) = allpaca_hip::MoeDeviceRoutes::at(table, rows, k);
        allpaca_hip::moe_route_softmax_topk_device(dl.cast(), n_exp, dx.cast(), dg.cast(), dim, &routes, std::ptr::null(), std::ptr::null_mut())
            .unwrap();
        let down = |p: *mut u8, n: usize| {
            let mut v = vec![0u32; n];
            allpaca_hip::memcpy(v.as_mut_ptr().cast(), p, n * 4, allpaca_hip::MEMCPY_D2H).unwrap();
            v
        };
        let w = down(routes.w.cast(), rows * k);
        let e = down(routes.expert.cast(), rows * k);
        let rs = down(routes.row_start.cast(), rows + 1);
        let sc = down(routes.shexp_scale.cast(), rows);
        for r in 0..rows {
            let mut row = logits[r * n_exp..(r + 1) * n_exp].to_vec();
            let want = topk_from_logits(&mut row, k, &cfg, None, &[]);
            for (j, &(we, ww)) in want.iter().enumerate() {
                let p = r * k + j;
                assert_eq!(e[p] as usize, we, "row {r} rank {j} expert");
                let got = f32::from_bits(w[p]);
                assert!((got - ww).abs() <= 1e-6 * ww.abs().max(1.0), "row {r} rank {j}: {got} vs {ww}");
            }
            assert_eq!(rs[r] as usize, r * k);
            let dot: f32 = x[r * dim..(r + 1) * dim].iter().zip(&gate).map(|(a, b)| a * b).sum();
            let got = f32::from_bits(sc[r]);
            assert!((got - sigmoid_stable(dot)).abs() < 1e-5, "row {r} shexp {got}");
        }
        assert_eq!(rs[rows] as usize, rows * k);
        for (p, n) in [(dl, rows * n_exp), (dx, rows * dim), (dg, dim)] {
            allpaca_hip::device_free(p, n * 4);
        }
        allpaca_hip::device_free(table, 1 << 16);
    }

    /// Disabled experts: the GPU router matches the masked host top-k (never
    /// picking a disabled expert) and counts exactly the experts it picked.
    #[test]
    fn gpu_moe_routing_honours_disabled_experts_and_counts_picks() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let (rows, n_exp, k, dim) = (4usize, 512usize, 10usize, 256usize);
        let cfg = ModelConfig::tiny();
        let logits: Vec<f32> = (0..rows * n_exp)
            .map(|i| ((i * 7919 % 1009) as f32) * 0.013 - 6.0)
            .collect();
        // Disable every expert row 0 would pick unmasked, plus a stripe.
        let mut mask = vec![false; n_exp];
        for (e, _) in topk_from_logits(&mut logits[..n_exp].to_vec(), k, &cfg, None, &[]) {
            mask[e] = true;
        }
        for e in (0..n_exp).step_by(7) {
            mask[e] = true;
        }
        let up = |b: &[u8]| {
            let p = allpaca_hip::device_malloc(b.len().max(4), 0).unwrap();
            allpaca_hip::memcpy(p, b.as_ptr(), b.len(), allpaca_hip::MEMCPY_H2D).unwrap();
            p
        };
        let lbytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dl = up(&lbytes);
        let dm = up(&mask.iter().map(|&d| u8::from(d)).collect::<Vec<u8>>());
        let dc = up(&vec![0u8; n_exp * 4]);
        let table = allpaca_hip::device_malloc(1 << 16, 0).unwrap();
        let (routes, _) = allpaca_hip::MoeDeviceRoutes::at(table, rows, k);
        allpaca_hip::moe_route_softmax_topk_device(
            dl.cast(),
            n_exp,
            std::ptr::null(),
            std::ptr::null(),
            dim,
            &routes,
            dm.cast_const(),
            dc.cast(),
        )
        .unwrap();
        let down = |p: *mut u8, n: usize| {
            let mut v = vec![0u32; n];
            allpaca_hip::memcpy(v.as_mut_ptr().cast(), p, n * 4, allpaca_hip::MEMCPY_D2H).unwrap();
            v
        };
        let w = down(routes.w.cast(), rows * k);
        let e = down(routes.expert.cast(), rows * k);
        let counts = down(dc, n_exp);
        let mut want_counts = vec![0u32; n_exp];
        for r in 0..rows {
            let mut row = logits[r * n_exp..(r + 1) * n_exp].to_vec();
            let want = topk_from_logits(&mut row, k, &cfg, Some(&mask), &[]);
            assert_eq!(want.len(), k);
            for (j, &(we, ww)) in want.iter().enumerate() {
                assert!(!mask[we], "host picked disabled expert {we}");
                let p = r * k + j;
                assert_eq!(e[p] as usize, we, "row {r} rank {j} expert");
                let got = f32::from_bits(w[p]);
                assert!((got - ww).abs() <= 1e-6 * ww.abs().max(1.0), "row {r} rank {j}: {got} vs {ww}");
                want_counts[we] += 1;
            }
        }
        assert_eq!(counts, want_counts);
        allpaca_hip::device_free(dl, lbytes.len());
        allpaca_hip::device_free(dm, n_exp);
        allpaca_hip::device_free(dc, n_exp * 4);
        allpaca_hip::device_free(table, 1 << 16);
    }

    /// Host routing with a mask never picks a disabled expert, renormalizes
    /// over the rest, and leaves unmasked routing untouched.
    #[test]
    fn host_routing_skips_disabled_experts() {
        let cfg = ModelConfig::tiny();
        let logits = vec![3.0f32, 2.5, 2.0, -1.0, 0.5, 1.0];
        let plain = topk_from_logits(&mut logits.clone(), 2, &cfg, None, &[]);
        assert_eq!(plain.iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1]);
        let none = vec![false; 6];
        assert_eq!(topk_from_logits(&mut logits.clone(), 2, &cfg, Some(&none), &[]), plain);
        let mask = vec![true, false, false, false, false, false];
        let got = topk_from_logits(&mut logits.clone(), 2, &cfg, Some(&mask), &[]);
        assert_eq!(got.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2]);
        let sum: f32 = got.iter().map(|p| p.1).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ssm_ba_splits_per_k_group_into_beta_then_alpha() {
        // 2 K groups x 2 V heads, dim 1: rows are [b0 b1 a0 a1 | b2 b3 a2 a3].
        let ba = [10.0, 11.0, 20.0, 21.0, 12.0, 13.0, 22.0, 23.0];
        let (beta, alpha) = super::split_ssm_ba(&ba, 1, 2, 4).expect("split");
        assert_eq!(beta, [10.0, 11.0, 12.0, 13.0]);
        assert_eq!(alpha, [20.0, 21.0, 22.0, 23.0]);
        assert!(super::split_ssm_ba(&ba, 1, 3, 4).is_none());
    }

    #[test]
    fn device_resident_gdn_layers_skip_host_f32_projections() {
        // A large GGUF loads as a skeleton whose GDN projections are uploaded
        // from the file; expanding them to host F32 held ~22 GB for the 27B.
        let dir = std::env::temp_dir().join(format!("allpaca-gdn-lean-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gdn.gguf");
        write_one_gdn_layer_gguf(&path);
        let file = GgufFile::open(&path).unwrap();

        let lean = load_gguf_gdn_host(&file, 0, false).unwrap();
        assert!(
            lean.w_qkv.is_empty()
                && lean.w_z.is_empty()
                && lean.w_alpha.is_empty()
                && lean.w_beta.is_empty()
                && lean.w_out.is_empty(),
            "lean GDN layer must not hold host projections"
        );
        assert_eq!((lean.a.len(), lean.norm.len(), lean.conv1d.len()), (2, 4, 64));

        let host = load_gguf_gdn_host(&file, 0, true).unwrap();
        assert_eq!((host.w_qkv.len(), host.w_out.len()), (256, 128));
        drop(file);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mrope_layout_positions_image_grid_and_shifts_later_text() {
        // Two images: 2 cols x 3 rows at KV 5, then 4 x 1 at KV 13.
        let layout = MropeLayout::new(vec![
            ImageSpan { start: 5, cols: 2, rows: 3 },
            ImageSpan { start: 13, cols: 4, rows: 1 },
        ]);
        assert_eq!(layout.rope_pos(4), [4, 4, 4], "text before the image");
        assert_eq!(layout.rope_pos(5), [5, 5, 5]);
        assert_eq!(layout.rope_pos(6), [5, 5, 6], "row 0, column 1");
        assert_eq!(layout.rope_pos(10), [5, 7, 6], "row 2, column 1");
        // The image spans max(cols, rows) = 3 positions, not 6 tokens.
        assert_eq!(layout.rope_pos(11), [8, 8, 8]);
        assert_eq!(layout.rope_pos(12), [9, 9, 9]);
        assert_eq!(layout.rope_pos(13), [10, 10, 10]);
        assert_eq!(layout.rope_pos(16), [10, 10, 13]);
        // Second image: 4 tokens over max(4, 1) = 4 positions (no extra shift).
        assert_eq!(layout.rope_pos(17), [14, 14, 14]);
        assert_eq!(MropeLayout::default().rope_pos(9), [9, 9, 9]);
    }

    #[test]
    fn q3_k_gemm_weights_stay_native_on_device() {
        let info = |k: u64| allpaca_gguf::TensorInfo {
            name: "output.weight".into(),
            dims: vec![k, 248_320],
            ggml_type: allpaca_gguf::GgmlType::Q3_K,
            ggml_type_id: 11,
            offset: 0,
            shard: 0,
        };
        // The MTP draft lm_head: 110 B per 256 values, not a 36 B/32 Q8_0 re-pack.
        let native = info(5120);
        assert_eq!(staged_device_bytes(&native), native.nbytes() as u64);
        // Rows that are not whole super-blocks still re-pack to Q8_0.
        let ragged = info(5120 + 32);
        let q8 = ragged.n_elements().div_ceil(32) * allpaca_hip::Q8_STAGED_BLOCK as u64;
        assert_eq!(staged_device_bytes(&ragged), q8);
    }

    /// IQ1/IQ2/IQ3_XXS GEMM weights go to the device as the Q8_0 re-pack
    /// the IQ4 types use (no native kernel), not as 4-byte floats.
    #[test]
    fn low_bit_iq_gemm_weights_repack_to_q8_0_on_device() {
        use allpaca_gguf::GgmlType as T;
        for (ty, id) in [(T::IQ2_XXS, 16), (T::IQ2_XS, 17), (T::IQ3_XXS, 18), (T::IQ1_S, 19), (T::IQ2_S, 22), (T::IQ1_M, 29)] {
            let info = allpaca_gguf::TensorInfo {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![4096, 1024],
                ggml_type: ty,
                ggml_type_id: id,
                offset: 0,
                shard: 0,
            };
            let q8 = info.n_elements().div_ceil(32) * allpaca_hip::Q8_STAGED_BLOCK as u64;
            assert_eq!(staged_device_bytes(&info), q8, "{ty:?}");
        }
    }

    /// ROCmFPX GEMM weights reach the device without changing a value: type
    /// 103 as the staged Q8_0 its blocks re-pack to, the two-scale types
    /// (102, 104, 109-111) as F16, which holds every value they decode to.
    #[test]
    fn rocmfpx_gemm_weights_stage_losslessly() {
        use allpaca_gguf::GgmlType as T;
        let (rows, k) = (512usize, 256usize);
        let n = rows * k;
        let path = std::env::temp_dir().join(format!("allpaca-p13-rocmfpx-{}.gguf", std::process::id()));
        let types = [
            (T::RocmFpxQ8, allpaca_cpu::ROCMFPX_Q8_BLOCK_BYTES),
            (T::RocmFpxQ6, allpaca_cpu::ROCMFPX_Q6_BLOCK_BYTES),
            (T::RocmFpxQ3, allpaca_cpu::ROCMFPX_Q3_BLOCK_BYTES),
            (T::RocmFpxQ5, allpaca_cpu::ROCMFPX_Q5_BLOCK_BYTES),
            (T::RocmFpxQ7, allpaca_cpu::ROCMFPX_Q7_BLOCK_BYTES),
            (T::RocmFpxQ2, allpaca_cpu::ROCMFPX_Q2_BLOCK_BYTES),
        ];
        let payloads: Vec<Vec<u8>> = types
            .iter()
            .map(|&(_, bb)| (0..n / 32 * bb).map(|i| (i * 131 + 7) as u8 % 0x7f).collect())
            .collect();
        let tensors: Vec<allpaca_gguf::GgufWriteQuantTensor> = types
            .iter()
            .zip(&payloads)
            .map(|(&(ty, _), bytes)| allpaca_gguf::GgufWriteQuantTensor {
                name: format!("w{}.weight", ty as u32),
                dims: vec![k as u64, rows as u64],
                ggml_type: ty,
                bytes,
            })
            .collect();
        allpaca_gguf::write_gguf_quant(&path, &Default::default(), &[], &tensors).unwrap();
        let file = GgufFile::open(&path).unwrap();
        for ((ty, _), packed) in types.iter().zip(&payloads) {
            let name = format!("w{}.weight", *ty as u32);
            let info = file.tensor(&name).unwrap();
            let (mut parts, mut off) = (Vec::new(), 0usize);
            let w = append_gguf_tensor_as(&file, &name, &mut parts, &mut off, false).unwrap();
            let staged = parts.last().unwrap().as_slice().to_vec();
            assert_eq!(staged.len() as u64, staged_device_bytes(info), "{ty:?}");
            let mut want = vec![0.0f32; n];
            allpaca_gguf::dequant_packed(*ty, packed, n, &mut want).unwrap();
            if *ty == T::RocmFpxQ8 {
                assert_eq!(w.kind, WeightKind::Q8_0);
                let q8 = allpaca_cpu::repack_rocmfpx_q8_as_q8_0(packed).unwrap();
                assert_eq!(staged, allpaca_hip::stage_q8_0_gemv(&q8, rows, k).unwrap());
                let mut got = vec![0.0f32; n];
                allpaca_cpu::dequant_q8_0(&q8, n, &mut got).unwrap();
                assert!(got.iter().zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()));
            } else {
                assert_eq!(w.kind, WeightKind::F16, "{ty:?}");
                let got = staged.chunks_exact(2).map(|h| allpaca_cpu::f16_to_f32(u16::from_le_bytes([h[0], h[1]])));
                assert!(got.zip(&want).all(|(g, w)| g.to_bits() == w.to_bits()), "{ty:?}");
            }
        }
        drop(file);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn only_the_mtp_draft_head_uses_a_q4_0_copy() {
        let target = ModelConfig::tiny();
        let mut draft = target.clone();
        draft.gguf_layer_base = 64;
        assert!(!use_q4_0_draft_head(&target));
        assert!(use_q4_0_draft_head(&draft));

        let info = allpaca_gguf::TensorInfo {
            name: "output.weight".into(),
            dims: vec![32, 2],
            ggml_type: allpaca_gguf::GgmlType::F32,
            ggml_type_id: 0,
            offset: 0,
            shard: 0,
        };
        let source: Vec<f32> = (0..64)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 16.0)
            .collect();
        let bytes: Vec<u8> = source.iter().flat_map(|v| v.to_le_bytes()).collect();
        let packed = stage_draft_head_q4_0(&info, &bytes).unwrap();
        assert_eq!(packed.len(), 2 * allpaca_cpu::Q4_0_BLOCK_BYTES);

        let mut roundtrip = vec![0.0; source.len()];
        allpaca_cpu::dequant_q4_0(&packed, source.len(), &mut roundtrip).unwrap();
        for (original, actual) in source.iter().zip(roundtrip) {
            assert!((original - actual).abs() < 0.125, "{original} vs {actual}");
        }
    }

    #[test]
    fn lm_head_upload_quantises_only_the_mtp_draft() {
        let dir = std::env::temp_dir().join(format!("allpaca-p1-head-upload-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("head.gguf");
        let source: Vec<f32> = (0..64).map(|i| (i % 17) as f32 / 32.0).collect();
        allpaca_gguf::write_gguf(
            &path,
            &std::collections::HashMap::new(),
            &[allpaca_gguf::GgufWriteTensor {
                name: "output.weight".into(),
                dims: vec![32, 2],
                data: &source,
            }],
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        let target = ModelConfig::tiny();
        let mut draft = target.clone();
        draft.gguf_layer_base = 64;

        let mut target_parts = Vec::new();
        let mut draft_parts = Vec::new();
        let target_ref = append_gguf_lm_head(&file, &target, "output.weight", &mut target_parts, &mut 0).unwrap();
        let draft_ref = append_gguf_lm_head(&file, &draft, "output.weight", &mut draft_parts, &mut 0).unwrap();
        assert!(matches!(target_ref.kind, WeightKind::F32));
        assert!(matches!(draft_ref.kind, WeightKind::Q4_0));
        assert_eq!(target_parts[0].len(), source.len() * 4);
        assert_eq!(draft_parts[0].len(), source.len() / 32 * allpaca_cpu::Q4_0_BLOCK_BYTES);
        let info = file.tensor("output.weight").unwrap();
        assert_eq!(staged_lm_head_bytes(info, &target), (source.len() * 4) as u64);
        assert_eq!(staged_lm_head_bytes(info, &draft), draft_parts[0].len() as u64);

        drop(file);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hip_batched_prefill_takes_imrope_but_not_undetectable_gated_q() {
        let mut imrope = ModelConfig::tiny();
        imrope.rope_sections = [16, 16, 16, 16];
        let plain = TinyModel::synthetic(imrope.clone(), 1, "imrope");
        assert!(plain.supports_hip_batched_prefill(), "HIP batches IMRoPE rows");
        // `[Q|gate]` with n_heads * head_dim == dim: q_out == 2 * wo_k cannot
        // be told apart from a wider plain Q, so keep it per-token.
        let mut ambiguous = TinyModel::synthetic(imrope, 1, "gated-q-eq-dim");
        let dim = ambiguous.config.dim;
        for layer in &mut ambiguous.layers {
            layer.wq = vec![0.0; 2 * dim * dim];
        }
        assert!(!ambiguous.supports_hip_batched_prefill());
    }

    /// Ling-mini-2.0's shape in small: 8 experts in 4 groups, 2 groups used,
    /// top-3, sigmoid gating with normalised and scaled weights.
    #[test]
    fn group_limited_routing_keeps_the_best_groups_only() {
        let mut cfg = ModelConfig::tiny_moe();
        cfg.n_experts = 8;
        cfg.n_active = 3;
        cfg.moe_sigmoid_gating = true;
        cfg.expert_weights_norm = true;
        cfg.expert_weights_scale = 2.5;
        cfg.n_expert_groups = 4;
        cfg.n_group_used = 2;
        // Groups: {0,1} {2,3} {4,5} {6,7}. Expert 2 is the single best, but
        // its group's top-2 sum loses to groups 0 and 3.
        let logits = [1.0f32, 0.9, 1.6, -4.0, 0.2, 0.1, 1.2, 0.8];
        let got = topk_from_logits(&mut logits.clone(), cfg.n_active, &cfg, None, &[]);
        let picked: Vec<usize> = got.iter().map(|(e, _)| *e).collect();
        assert_eq!(picked, vec![6, 0, 1], "experts only from the two best groups");
        let sum: f32 = got.iter().map(|(_, w)| *w).sum();
        assert!((sum - 2.5).abs() < 1e-5, "weights normalise to the scale, got {sum}");
        // Flat routing would have picked expert 2 first.
        cfg.n_expert_groups = 1;
        let flat = topk_from_logits(&mut logits.clone(), cfg.n_active, &cfg, None, &[]);
        assert_eq!(flat[0].0, 2);
        // Disabled experts drop out of their group's score: with group 3
        // off, groups 0 and 2 are the best two (expert 2's group still loses).
        cfg.n_expert_groups = 4;
        let mut mask = [false; 8];
        mask[6] = true;
        mask[7] = true;
        let masked = topk_from_logits(&mut logits.clone(), cfg.n_active, &cfg, Some(&mask), &[]);
        let picked: Vec<usize> = masked.iter().map(|(e, _)| *e).collect();
        assert_eq!(picked, vec![0, 1, 4]);
    }

    /// Ornith-1.5-9B: 16 heads x 256 = dim 4096, so only the architecture
    /// flag says the double-width Q is `[Q|gate]`.
    #[test]
    fn attn_layout_reads_gated_q_at_attn_dim_equal_to_dim_from_the_arch_flag() {
        let mut cfg = ModelConfig::tiny();
        let dim = cfg.dim;
        let (wq, wk, wo) = (2 * dim * dim, dim * dim, dim * dim);
        let plain = attn_layout(&cfg, wq, wk, wo);
        assert!(!plain.gated);
        cfg.gated_attn_q = true;
        let gated = attn_layout(&cfg, wq, wk, wo);
        assert!(gated.gated);
        assert_eq!((gated.q_out, gated.attn_dim), (2 * dim, dim));
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

#[cfg(test)]
mod kv_page_mask_tests {
    use super::*;

    #[test]
    fn only_recurrent_layers_go_without_kv_pages() {
        for model in [TinyModel::synthetic_gdn_hybrid(5), TinyModel::synthetic_ssm_hybrid(5)] {
            let keeps = kv_layer_mask(&model);
            assert_eq!(keeps.len(), model.config.n_layers);
            for (layer, keep) in model.layers.iter().zip(&keeps) {
                let recurrent =
                    layer.block == HybridLayerKind::Ssm || layer.gdn.is_some() || layer.ssm.is_some();
                assert_eq!(*keep, !recurrent);
            }
            assert!(keeps.iter().any(|k| *k) && keeps.iter().any(|k| !*k), "a hybrid has both");
            let metrics = Metrics::default();
            let mut kv = crate::kv_tier::KvTierManager::new(8 << 20, 8 << 20, 1.0);
            let mut pages = alloc_pages_for_seq_masked(&mut kv, &model.config, 4, 9, Some(&keeps), &metrics)
                .expect("pages");
            for (layer_pages, keep) in pages.iter().zip(&keeps) {
                assert_eq!(layer_pages.len(), if *keep { 3 } else { 0 });
            }
            // Growing the sequence adds pages only where KV is kept.
            ensure_pages(&mut kv, &model.config, 4, &mut pages, 16, &metrics).expect("grow");
            for (layer_pages, keep) in pages.iter().zip(&keeps) {
                assert_eq!(layer_pages.len(), if *keep { 5 } else { 0 });
            }
        }
        // Attention that is not the `Attention` block (MLA, MoE layers) keeps its pages.
        for model in [TinyModel::synthetic_mla_hybrid(5), TinyModel::synthetic_moe_shared(5)] {
            let keeps = kv_layer_mask(&model);
            for (layer, keep) in model.layers.iter().zip(&keeps) {
                if layer.gdn.is_none() && layer.ssm.is_none() && layer.block != HybridLayerKind::Ssm {
                    assert!(*keep);
                }
            }
        }
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
        let pages = alloc_pages_for_seq(&mut kv, &model.config, 128, 1, &metrics).expect("pages");
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

    #[test]
    fn cuda_prefill_loop_matches_token_loop_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let model = TinyModel::synthetic(ModelConfig::tiny(), 7, "parity-loop");
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 8 * 1024 * 1024).expect("ws");
        let resident = CudaResidentModel::upload(&model, cuda.device).expect("upload");
        let n = 8usize;
        let tokens: Vec<u32> = (0..n)
            .map(|pos| (pos as u32) % model.config.vocab_size as u32)
            .collect();

        let run = |label: &str| -> Vec<f32> {
            let metrics = Metrics::default();
            let mut kv = crate::kv_tier::KvTierManager::with_devices(
                8 << 20,
                8 << 20,
                0.5,
                None,
                Some(cuda.device),
            );
            let pages =
                alloc_pages_for_seq(&mut kv, &model.config, 128, n, &metrics).expect("pages");
            let mut logits = vec![0.0f32; model.config.vocab_size];
            if label == "batch" {
                model
                    .forward_tokens_cuda(
                        &tokens,
                        0,
                        &pages,
                        &mut kv,
                        &metrics,
                        &mut logits,
                        &cuda,
                        &resident,
                        true,
                        None,
                    )
                    .expect("batch");
            } else {
                for (pos, &tok) in tokens.iter().enumerate() {
                    model
                        .forward_token_cuda(
                            tok,
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
                        .expect("tok");
                }
            }
            logits
        };

        let a = run("tok");
        let b = run("tok");
        let det = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(det < 1e-5, "token loop not deterministic max_diff={det}");

        let c = run("batch");
        let diff = a
            .iter()
            .zip(c.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "batch vs tok max_diff={diff}");
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

    #[test]
    fn cuda_swa_softcap_changes_logits_vs_plain_swa_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let plain = TinyModel::synthetic_swa(85);
        let mut capped = TinyModel::synthetic_swa(85);
        capped.config.attn_logit_softcap = 1.5;
        let a = cuda_decode_n_tokens(&plain, 4);
        let b = cuda_decode_n_tokens(&capped, 4);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "SWA+attn_logit_softcap should change logits vs plain SWA; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_attn_softcap_changes_logits_vs_plain_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let plain = TinyModel::synthetic(ModelConfig::tiny(), 91, "attn-sc-plain");
        let mut capped = TinyModel::synthetic(ModelConfig::tiny(), 91, "attn-sc-cap");
        capped.config.attn_logit_softcap = 1.5;
        // Keep final softcap off so divergence is from attn path only.
        capped.config.final_logit_softcap = 0.0;
        let a = cuda_decode_n_tokens(&plain, 3);
        let b = cuda_decode_n_tokens(&capped, 3);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "attn_logit_softcap should change logits vs plain; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_moe_q8_experts_pack_smaller_than_f32_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let model = TinyModel::synthetic(ModelConfig::tiny_moe(), 43, "moe-q8-pack");
        let mut f32_bytes = 0usize;
        f32_bytes += model.embed.len() * 4;
        f32_bytes += model.norm.len() * 4;
        f32_bytes += model.lm_head.len() * 4;
        for layer in &model.layers {
            f32_bytes += layer.attn_norm.len() * 4;
            f32_bytes += layer.wq.len() * 4;
            f32_bytes += layer.wk.len() * 4;
            f32_bytes += layer.wv.len() * 4;
            f32_bytes += layer.wo.len() * 4;
            f32_bytes += layer.ffn_norm.len() * 4;
            for expert in &layer.experts {
                f32_bytes += expert.w_gate.len() * 4;
                f32_bytes += expert.w_up.len() * 4;
                f32_bytes += expert.w_down.len() * 4;
            }
        }
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 8192).expect("cuda ws");
        let resident = CudaResidentModel::upload(&model, cuda.device).expect("upload");
        assert!(
            (resident.packed_h2d_bytes() as usize) < f32_bytes,
            "Q8 MoE experts should shrink device pack: packed={} f32_est={}",
            resident.packed_h2d_bytes(),
            f32_bytes
        );
    }

    #[test]
    fn cuda_headwise_gate_changes_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let plain = TinyModel::synthetic(ModelConfig::tiny(), 53, "hw-plain");
        let gated = TinyModel::synthetic_headwise_gate(53);
        let a = cuda_decode_n_tokens(&plain, 2);
        let b = cuda_decode_n_tokens(&gated, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "headwise attn gate should change logits; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_moe_shared_expert_uploads_as_q8_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let model = TinyModel::synthetic_moe_shared(47);
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 8192).expect("cuda ws");
        let resident = CudaResidentModel::upload(&model, cuda.device).expect("upload");
        assert!(
            resident.shared_expert_is_q8(),
            "shared expert up should be staged Q8_0"
        );
        let mut f32_bytes = 0usize;
        f32_bytes += model.embed.len() * 4;
        f32_bytes += model.norm.len() * 4;
        f32_bytes += model.lm_head.len() * 4;
        for layer in &model.layers {
            f32_bytes += layer.attn_norm.len() * 4;
            f32_bytes += layer.wq.len() * 4;
            f32_bytes += layer.wk.len() * 4;
            f32_bytes += layer.wv.len() * 4;
            f32_bytes += layer.wo.len() * 4;
            f32_bytes += layer.ffn_norm.len() * 4;
            for expert in &layer.experts {
                f32_bytes += expert.w_gate.len() * 4;
                f32_bytes += expert.w_up.len() * 4;
                f32_bytes += expert.w_down.len() * 4;
            }
            f32_bytes += layer.shared_expert.w_gate.len() * 4;
            f32_bytes += layer.shared_expert.w_up.len() * 4;
            f32_bytes += layer.shared_expert.w_down.len() * 4;
        }
        assert!(
            (resident.packed_h2d_bytes() as usize) < f32_bytes,
            "Q8 shared+routed should shrink pack: packed={} f32_est={}",
            resident.packed_h2d_bytes(),
            f32_bytes
        );
    }

    #[test]
    fn cuda_gdn_projections_upload_resident_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let model = TinyModel::synthetic_gdn_hybrid(11);
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 64 * 1024).expect("cuda ws");
        let resident = CudaResidentModel::upload(&model, cuda.device).expect("upload");
        assert!(
            resident.gdn_projections_resident(),
            "GDN qkv/z/out should pack onto the device"
        );
        let gdn = model.layers[0].gdn.as_ref().expect("gdn");
        let proj_bytes = (gdn.w_qkv.len()
            + gdn.w_z.len()
            + gdn.w_alpha.len()
            + gdn.w_beta.len()
            + gdn.w_out.len())
            * 4;
        assert!(
            (resident.packed_h2d_bytes() as usize) >= proj_bytes,
            "pack should include GDN projections: packed={} proj={}",
            resident.packed_h2d_bytes(),
            proj_bytes
        );
    }

    #[test]
    fn cuda_ssm_projections_upload_resident_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let model = TinyModel::synthetic_ssm_hybrid(7);
        let cuda = allpaca_cuda::CudaWorkspace::new(0, 64 * 1024).expect("cuda ws");
        let resident = CudaResidentModel::upload(&model, cuda.device).expect("upload");
        assert!(
            resident.ssm_projections_resident(),
            "SSM w_in/w_out should pack onto the device"
        );
        let ssm = model.layers[0].ssm.as_ref().expect("ssm");
        let proj_bytes = (ssm.w_in.len() + ssm.w_out.len()) * 4;
        assert!(
            (resident.packed_h2d_bytes() as usize) >= proj_bytes,
            "pack should include SSM projections: packed={} proj={}",
            resident.packed_h2d_bytes(),
            proj_bytes
        );
    }

    #[test]
    fn cuda_gemma_norms_change_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let plain = TinyModel::synthetic(ModelConfig::tiny(), 59, "gemma-plain");
        let gemma = TinyModel::synthetic_gemma_norms(59);
        let a = cuda_decode_n_tokens(&plain, 2);
        let b = cuda_decode_n_tokens(&gemma, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "scale_embed + post-norms should change logits; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_gelu_ffn_changes_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let silu = TinyModel::synthetic(ModelConfig::tiny(), 61, "silu-ffn");
        let gelu = TinyModel::synthetic_gelu_ffn(61);
        let a = cuda_decode_n_tokens(&silu, 2);
        let b = cuda_decode_n_tokens(&gelu, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "ffn_gelu should change logits vs SiLU; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_qk_norms_change_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let plain = TinyModel::synthetic(ModelConfig::tiny(), 67, "qk-plain");
        let qk = TinyModel::synthetic_qk_norms(67);
        let a = cuda_decode_n_tokens(&plain, 2);
        let b = cuda_decode_n_tokens(&qk, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "q_norm/k_norm should change logits; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_gated_attn_uses_q_tail_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let full = TinyModel::synthetic_gated_attn(71);
        let mut tailed = TinyModel::synthetic_gated_attn(71);
        tailed.zero_wq_output_tail();
        let a = cuda_decode_n_tokens(&full, 2);
        let b = cuda_decode_n_tokens(&tailed, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "gated Q must consume wq rows beyond dim; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_attn_sinks_change_logits_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let plain = TinyModel::synthetic(ModelConfig::tiny(), 79, "sinks-plain");
        let sinks = TinyModel::synthetic_attn_sinks(79);
        let a = cuda_decode_n_tokens(&plain, 2);
        let b = cuda_decode_n_tokens(&sinks, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "attn_sinks should change logits; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_partial_rope_changes_logits_vs_full_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        let full = TinyModel::synthetic(ModelConfig::tiny(), 89, "rope-full");
        let partial = TinyModel::synthetic_partial_rope(89);
        let a = cuda_decode_n_tokens(&full, 2);
        let b = cuda_decode_n_tokens(&partial, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "partial rope_dims should change logits vs full RoPE; max_diff={max_diff}"
        );
    }

    #[test]
    fn cuda_imrope_changes_logits_vs_full_when_device_present() {
        if !allpaca_cuda::runtime_linked() || allpaca_cuda::device_count() == 0 {
            return;
        }
        // Text positions make IMRoPE == plain partial NeoX; both must still
        // diverge from full-head device RoPE.
        let full = TinyModel::synthetic(ModelConfig::tiny(), 91, "imrope-full");
        let imrope = TinyModel::synthetic_imrope(91);
        let a = cuda_decode_n_tokens(&full, 2);
        let b = cuda_decode_n_tokens(&imrope, 2);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "IMRoPE partial dims should change logits vs full RoPE; max_diff={max_diff}"
        );
    }
}

thread_local! {
    static FWD_SEG_T0: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
}

/// `ALLPACA_FWD_LOG=2`: wall time of each device's run of layers in a
/// forward (syncs the device at every boundary, so only for diagnosis).
pub(crate) fn fwd_segment_mark(device: i32, what: &str) {
    if !std::env::var("ALLPACA_FWD_LOG").is_ok_and(|v| v == "2") {
        return;
    }
    let _ = allpaca_hip::stream_synchronize();
    let now = std::time::Instant::now();
    FWD_SEG_T0.with(|t| {
        if let Some(t0) = t.get() {
            if what != "start" {
                eprintln!(
                    "SEG dev={device} {what} ms={:.2}",
                    now.duration_since(t0).as_secs_f64() * 1e3
                );
            }
        }
        t.set(Some(now));
    });
}

