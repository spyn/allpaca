//! Allpaca Engine: scheduling, memory tiers, draft providers, and decode.

mod anchors;
mod batch_gemm;
mod constraint;
pub mod deltanet;
mod device_manager;
mod compute;
mod direct_storage;
mod draft;
mod early_exit;
mod engine;
mod expert_batch;
mod forest;
mod grammar;
mod kv_nvme;
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
mod rpc_runtime;
mod sampling;
mod scheduler;
mod ssm;
mod telemetry;
mod tokenizer;
mod transfer;
mod vision;
mod weight_set;

pub use batch_gemm::{
    batch_gemm_other_ids, packed_gemm_nt, packed_gemm_nt_cpu, packed_gemm_nt_hip,
    packed_gemm_nt_hip_diagnostic, plan_batch_gemm, BatchGemmCandidate, BatchGemmPhase,
    BatchGemmPlan, ModelShapeKey,
};
pub use device_manager::{detect_pipeline_mode, DeviceInventory};
pub use compute::ComputeDev;
pub use layer_split::{plan_layer_owners, LayerPlan};
pub use draft::{DraftModelProvider, DraftProvider, NGramProvider};
pub use early_exit::{
    accept_exit, has_calibration_data, spec_from_gguf, spec_from_metadata, EarlyExitSpec,
    ExitPrediction,
};
pub use engine::{mtp_text_is_garbled, Engine, RequestHandle};
pub use kv_tier::KvTierManager;
pub use memory_tier::{
    CxlMemBackend, MemoryTierBackend, PimBackend, PinnedDramBackend, ReadTicket,
};
pub use modality::{
    EncoderPlacement, ImageInput, NoopVisionEncoder, VisionEncoder, VisionTokens,
};
pub use model::{LayerWeights, ModelConfig, MtpNextnWeights, TinyModel};
#[cfg(feature = "cuda")]
pub use model::CudaResidentModel;
pub use packed::{HostExpertPool, PackedStore};
pub use scheduler::Scheduler;
pub use otel::export_otlp_json;
pub use telemetry::Metrics;
pub use tokenizer::EngineTokenizer;
pub use transfer::{select_transfer_path, select_transfer_path_with_bench};
pub use vision::PatchVisionEncoder;
pub use weight_set::{
    expert_lru_slots, host_pool_slots, qstar_fill_count, DenseTrunkPlan, ExpertId, ExpertLru,
    WeightWorkingSet,
};
