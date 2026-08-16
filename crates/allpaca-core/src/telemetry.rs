use std::sync::atomic::{AtomicU64, Ordering};

use allpaca_types::StatsSnapshot;

#[derive(Debug, Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub active_requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub tokens_generated: AtomicU64,
    pub draft_proposed: AtomicU64,
    pub draft_accepted: AtomicU64,
    pub page_migrations: AtomicU64,
    pub vram_pages: AtomicU64,
    pub dram_pages: AtomicU64,
    pub ttft_ms_last: AtomicU64,
    pub itl_ms_ema_x100: AtomicU64,
    pub degraded_events: AtomicU64,
    pub tree_nodes: AtomicU64,
    pub tree_accepted_path_len: AtomicU64,
    pub tree_path_invalid: AtomicU64,
    pub target_evals: AtomicU64,
    pub gpu_forwards: AtomicU64,
    pub target_gpu_forwards: AtomicU64,
    pub draft_gpu_forwards: AtomicU64,
    pub host_staging_bytes: AtomicU64,
    pub draft_tokens_copied: AtomicU64,
    pub weight_bytes_migrated: AtomicU64,
    pub device_memory_bytes: AtomicU64,
    pub weight_bytes_on_device: AtomicU64,
    pub weight_h2d_bytes: AtomicU64,
    pub kv_bytes_on_device: AtomicU64,
    pub elementwise_gpu_ops: AtomicU64,
}

impl Metrics {
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot(
        &self,
        pipeline_mode: &str,
        transfer_path: &str,
        p2p_capable: bool,
        p2p_enabled: bool,
        p2p_status: &str,
        p2p_gbps: f64,
        host_staging_gbps: f64,
        p2p_us: f64,
        host_staging_us: f64,
        target_device: i32,
        draft_device: i32,
        model_id: String,
        backend: String,
    ) -> StatsSnapshot {
        let proposed = self.draft_proposed.load(Ordering::Relaxed);
        let accepted = self.draft_accepted.load(Ordering::Relaxed);
        let acceptance_rate = if proposed == 0 {
            0.0
        } else {
            accepted as f64 / proposed as f64
        };
        let target_evals = self.target_evals.load(Ordering::Relaxed);
        let generated = self.tokens_generated.load(Ordering::Relaxed);
        let accepted_tokens_per_target_eval = if target_evals == 0 {
            0.0
        } else {
            generated as f64 / target_evals as f64
        };
        StatsSnapshot {
            pipeline_mode: pipeline_mode.to_string(),
            transfer_path: transfer_path.to_string(),
            p2p_capable,
            p2p_enabled,
            p2p_status: p2p_status.to_string(),
            p2p_gbps,
            host_staging_gbps,
            p2p_us,
            host_staging_us,
            target_device,
            draft_device,
            requests_total: self.requests_total.load(Ordering::Relaxed),
            active_requests: self.active_requests.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            tokens_generated: self.tokens_generated.load(Ordering::Relaxed),
            draft_proposed: proposed,
            draft_accepted: accepted,
            acceptance_rate,
            page_migrations: self.page_migrations.load(Ordering::Relaxed),
            vram_pages: self.vram_pages.load(Ordering::Relaxed),
            dram_pages: self.dram_pages.load(Ordering::Relaxed),
            ttft_ms_last: self.ttft_ms_last.load(Ordering::Relaxed),
            itl_ms_ema: self.itl_ms_ema_x100.load(Ordering::Relaxed) as f64 / 100.0,
            degraded_events: self.degraded_events.load(Ordering::Relaxed),
            tree_nodes: self.tree_nodes.load(Ordering::Relaxed),
            tree_accepted_path_len: self.tree_accepted_path_len.load(Ordering::Relaxed),
            tree_path_valid: self.tree_nodes.load(Ordering::Relaxed) > 0
                && self.tree_path_invalid.load(Ordering::Relaxed) == 0,
            accepted_tokens_per_target_eval,
            model_id,
            backend,
            gpu_forwards: self.gpu_forwards.load(Ordering::Relaxed),
            target_gpu_forwards: self.target_gpu_forwards.load(Ordering::Relaxed),
            draft_gpu_forwards: self.draft_gpu_forwards.load(Ordering::Relaxed),
            host_staging_bytes: self.host_staging_bytes.load(Ordering::Relaxed),
            draft_tokens_copied: self.draft_tokens_copied.load(Ordering::Relaxed),
            weight_bytes_migrated: self.weight_bytes_migrated.load(Ordering::Relaxed),
            device_memory_bytes: self.device_memory_bytes.load(Ordering::Relaxed),
            weight_bytes_on_device: self.weight_bytes_on_device.load(Ordering::Relaxed),
            weight_h2d_bytes: self.weight_h2d_bytes.load(Ordering::Relaxed),
            kv_bytes_on_device: self.kv_bytes_on_device.load(Ordering::Relaxed),
            elementwise_gpu_ops: self.elementwise_gpu_ops.load(Ordering::Relaxed),
        }
    }

    pub fn record_itl_ms(&self, ms: u64) {
        let prev = self.itl_ms_ema_x100.load(Ordering::Relaxed);
        let sample = ms.saturating_mul(100);
        let next = if prev == 0 {
            sample
        } else {
            (prev * 8 + sample * 2) / 10
        };
        self.itl_ms_ema_x100.store(next, Ordering::Relaxed);
    }
}
