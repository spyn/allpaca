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
}

impl Metrics {
    pub fn snapshot(
        &self,
        pipeline_mode: &str,
        transfer_path: &str,
        target_device: i32,
        draft_device: i32,
        model_id: String,
    ) -> StatsSnapshot {
        let proposed = self.draft_proposed.load(Ordering::Relaxed);
        let accepted = self.draft_accepted.load(Ordering::Relaxed);
        let acceptance_rate = if proposed == 0 {
            0.0
        } else {
            accepted as f64 / proposed as f64
        };
        StatsSnapshot {
            pipeline_mode: pipeline_mode.to_string(),
            transfer_path: transfer_path.to_string(),
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
            model_id,
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
