//! Allpaca Engine: scheduling, memory tiers, draft providers, and decode.

mod constraint;
mod device_manager;
mod draft;
mod engine;
mod forest;
mod kv_tier;
mod logits_ring;
mod model;
mod sampling;
mod scheduler;
mod telemetry;
mod tokenizer;
mod transfer;

pub use device_manager::{detect_pipeline_mode, DeviceInventory};
pub use draft::{DraftModelProvider, DraftProvider, NGramProvider};
pub use engine::{Engine, RequestHandle};
pub use kv_tier::KvTierManager;
pub use model::{LayerWeights, ModelConfig, TinyModel};
pub use scheduler::Scheduler;
pub use telemetry::Metrics;
pub use tokenizer::EngineTokenizer;
pub use transfer::select_transfer_path;
