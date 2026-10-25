//! Allpaca Engine: scheduling, memory tiers, draft providers, and decode.

mod anchors;
mod batch_gemm;
pub mod calib;
mod compute;
pub mod decision;
mod constraint;
mod deepseek4;
mod dsv4_attn;
mod dflash2;
pub mod deltanet;
mod device_manager;
mod direct_storage;
mod draft;
mod dspark;
mod dsv4_gpu;
pub mod prompt_store;
mod qsa;
mod qwen4_gpu;
pub mod qwen4_remote;
mod dsv4_ram;
mod early_exit;
pub mod edge0;
mod engine;
mod gemma4;
mod gemma4_gpu;
mod expert_batch;
pub mod expert_controls;
mod forest;
mod grammar;
mod hf;
mod kv_nvme;
mod kv_codec;
mod kv_radix;
mod kv_tier;
mod layer_split;
mod logits_ring;
mod memory_tier;
mod micro_batch;
mod modality;
mod model;
mod numa;
mod otel;
mod packed;
mod qwen4exp;
pub mod gptq;
pub mod gsq;
pub(crate) mod host_experts;
pub(crate) mod qwen35_host;
pub mod propagate;
pub mod rco;
pub mod requant;
mod rpc_runtime;
mod sampling;
mod scheduler;
pub mod sensitivity;
mod ssm;
mod telemetry;
mod tokenizer;
mod transfer;
pub mod tune;
pub mod qwen3vl;
mod vision;
mod weight_set;

pub use batch_gemm::{
    apply_packed_attn, apply_packed_attn_hip, apply_packed_ffn, apply_packed_ffn_hip,
    apply_packed_lm_head, batch_gemm_other_ids, decode_layer_attn_shared_a_weights,
    decode_layer_shared_a_weights, packed_ffn_shared_a, packed_gemm_nt, packed_gemm_nt_cpu,
    packed_gemm_nt_hip, packed_gemm_nt_hip_diagnostic, plan_batch_gemm, residual_add_rows,
    write_residual_into_hidden, BatchGemmCandidate, BatchGemmPhase, BatchGemmPlan, ModelShapeKey,
    PackedFfnWeights, PackedGemmWeight,
};
pub use compute::ComputeDev;
#[doc(hidden)]
pub use deepseek4::{
    write_tiny_deepseek4_ctx_gguf, write_tiny_deepseek4_gguf, write_tiny_deepseek4_packed_gguf,
    write_tiny_deepseek4_q8_gguf, TinyDsv4Ctx,
};
#[doc(hidden)]
pub use gemma4::write_tiny_gemma4_gguf;
pub use device_manager::{detect_pipeline_mode, suggested_kv_vram_bytes, DeviceInventory};
pub use draft::{DraftModelProvider, DraftProvider, NGramProvider};
pub use early_exit::{
    accept_exit, has_calibration_data, spec_from_gguf, spec_from_metadata, EarlyExitSpec,
    ExitPrediction,
};
pub use edge0::{
    detect_edge0_dir, load_edge0_dir, Edge0LoadOptions, Edge0Loaded, Edge0Tier, MoESpec,
    RecoverLora, RouterKind,
};
pub use engine::{
    mtp_text_is_garbled, DevicePlacement, Engine, ExpertLayerHome, ExpertPlacement, KvEstimate,
    RequestHandle,
};
pub use rpc_runtime::RemoteGpu;
pub use kv_tier::KvTierManager;
pub use layer_split::{plan_layer_owners, LayerPlan};
pub use memory_tier::{
    CxlMemBackend, MemoryTierBackend, PimBackend, PinnedDramBackend, ReadTicket,
};
pub use qwen3vl::{Qwen3VlConfig, Qwen3VlVision};
pub use modality::{EncoderPlacement, ImageInput, NoopVisionEncoder, VisionEncoder, VisionTokens};
#[cfg(feature = "cuda")]
pub use model::CudaResidentModel;
#[cfg(feature = "vulkan")]
pub use model::VulkanResidentModel;
pub use model::{LayerWeights, ModelConfig, MtpNextnWeights, TinyModel};
pub use otel::export_otlp_json;
pub use packed::{HostExpertPool, PackedStore};
pub use scheduler::Scheduler;
pub use telemetry::Metrics;
pub use tokenizer::{format_chat_colon, format_chatml, EngineTokenizer};
pub use transfer::{select_transfer_path, select_transfer_path_with_bench};
pub use tune::{
    available_groups, available_profiles, cache_bytes_for, discover_models, estimate_budget,
    expand_groups, keep_experts_from_ids, keep_ids_for_entry, load_catalog_beside,
    probe_checkpoint, DiscoveredModel, KeepsetCatalog, MoeProbe, MoeShape, ResourceSnapshot,
    TuneBudget, TuneGroup, TuneProfile, CTX_STEPS,
};
pub use vision::PatchVisionEncoder;
pub use weight_set::{
    expert_lru_slots, host_pool_slots, qstar_fill_count, DenseTrunkPlan, ExpertId, ExpertLru,
    WeightWorkingSet,
};
