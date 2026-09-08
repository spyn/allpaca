//! Allpaca Engine: scheduling, memory tiers, draft providers, and decode.

mod anchors;
mod constraint;
pub mod deltanet;
mod device_manager;
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
mod logits_ring;
mod micro_batch;
mod model;
mod numa;
mod packed;
mod sampling;
mod scheduler;
mod ssm;
mod telemetry;
mod tokenizer;
mod transfer;
mod weight_set;

pub use device_manager::{detect_pipeline_mode, DeviceInventory};
pub use draft::{DraftModelProvider, DraftProvider, NGramProvider};
pub use early_exit::{accept_exit, EarlyExitSpec, ExitPrediction, spec_from_gguf};
pub use engine::{mtp_text_is_garbled, Engine, RequestHandle};
pub use kv_tier::KvTierManager;
pub use model::{LayerWeights, ModelConfig, MtpNextnWeights, TinyModel};
pub use packed::{HostExpertPool, PackedStore};
pub use scheduler::Scheduler;
pub use telemetry::Metrics;
pub use tokenizer::EngineTokenizer;
pub use transfer::{select_transfer_path, select_transfer_path_with_bench};
pub use weight_set::{
    expert_lru_slots, host_pool_slots, qstar_fill_count, DenseTrunkPlan, ExpertId, ExpertLru,
    WeightWorkingSet,
};
