use serde::{Deserialize, Serialize};

/// In-memory telemetry snapshot exposed by `GET /stats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub pipeline_mode: String,
    pub transfer_path: String,
    pub target_device: i32,
    pub draft_device: i32,
    pub requests_total: u64,
    pub active_requests: u64,
    pub prompt_tokens: u64,
    pub tokens_generated: u64,
    pub draft_proposed: u64,
    pub draft_accepted: u64,
    pub acceptance_rate: f64,
    pub page_migrations: u64,
    pub vram_pages: u64,
    pub dram_pages: u64,
    pub ttft_ms_last: u64,
    pub itl_ms_ema: f64,
    pub degraded_events: u64,
    pub tree_nodes: u64,
    pub tree_accepted_path_len: u64,
    pub tree_path_valid: bool,
    pub accepted_tokens_per_target_eval: f64,
    pub model_id: String,
}
