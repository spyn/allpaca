//! Allpaca Engine: scheduling, memory tiers, draft providers, and decode.

mod anchors;
mod constraint;
mod direct_storage;
mod expert_batch;
mod grammar;
mod device_manager;
mod draft;
mod engine;
mod forest;
mod kv_nvme;
mod kv_radix;
mod kv_tier;
mod logits_ring;
mod micro_batch;
mod model;
mod numa;
mod sampling;
mod scheduler;
mod telemetry;
mod packed;
mod tokenizer;
mod transfer;
mod weight_set;

pub use device_manager::{detect_pipeline_mode, DeviceInventory};
pub use draft::{DraftModelProvider, DraftProvider, NGramProvider};
pub use engine::{Engine, RequestHandle};
pub use kv_tier::KvTierManager;
pub use model::{LayerWeights, ModelConfig, TinyModel};
pub use scheduler::Scheduler;
pub use telemetry::Metrics;
pub use tokenizer::EngineTokenizer;
pub use transfer::{select_transfer_path, select_transfer_path_with_bench};
pub use packed::{HostExpertPool, PackedStore};
pub use weight_set::{
    expert_lru_slots, host_pool_slots, qstar_fill_count, DenseTrunkPlan, ExpertId, ExpertLru,
    WeightWorkingSet,
};
