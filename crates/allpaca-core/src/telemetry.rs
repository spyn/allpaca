use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use allpaca_types::StatsSnapshot;

/// Default GPU sample rate. The plan's window is 1–10 Hz; 10 Hz reacts inside a
/// handful of tokens while keeping the driver round-trip far off the hot path.
pub const GPU_SAMPLE_INTERVAL_MS: u64 = 100;

/// Rate limiter for telemetry that costs a driver round-trip.
///
/// `hipMemGetInfo` is cheap but not free, and decode calls its neighbours
/// thousands of times a second, so VRAM pressure is sampled on a clock instead:
/// scheduler safe points and `/stats` ask whether a sample is [`due`], and at
/// most one caller per window wins.
///
/// [`due`]: GpuSampler::due
#[derive(Debug)]
pub struct GpuSampler {
    origin: Instant,
    /// Milliseconds since `origin` of the last sample, biased by 1 so that 0
    /// unambiguously means "never sampled" and the first call always wins.
    last_ms_plus_one: AtomicU64,
    min_interval_ms: u64,
}

impl Default for GpuSampler {
    fn default() -> Self {
        Self::new(GPU_SAMPLE_INTERVAL_MS)
    }
}

impl GpuSampler {
    pub fn new(min_interval_ms: u64) -> Self {
        Self {
            origin: Instant::now(),
            last_ms_plus_one: AtomicU64::new(0),
            min_interval_ms,
        }
    }

    /// True for exactly one caller once `min_interval_ms` has passed. Losers of
    /// the race get `false` and keep the stale sample rather than block.
    pub fn due(&self) -> bool {
        let now = self.origin.elapsed().as_millis() as u64;
        let prev = self.last_ms_plus_one.load(Ordering::Relaxed);
        if prev != 0 && now.saturating_sub(prev - 1) < self.min_interval_ms {
            return false;
        }
        self.last_ms_plus_one
            .compare_exchange(prev, now + 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }
}

#[derive(Debug, Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub active_requests: AtomicU64,
    pub prompt_tokens: AtomicU64,
    pub prefill_chunks: AtomicU64,
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
    pub tree_gpu_verifies: AtomicU64,
    pub linear_packed_verifies: AtomicU64,
    pub draft_ahead_hits: AtomicU64,
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
    pub draft_kv_bytes_on_device: AtomicU64,
    pub elementwise_gpu_ops: AtomicU64,
    pub residual_gpu_ops: AtomicU64,
    pub activation_h2d_bytes: AtomicU64,
    pub activation_d2h_bytes: AtomicU64,
    pub q8_gemm_ns: AtomicU64,
    pub attention_ns: AtomicU64,
    pub lm_head_ns: AtomicU64,
    pub other_gpu_ns: AtomicU64,
    pub host_ns: AtomicU64,
    pub weight_pinned_layers: AtomicU64,
    pub weight_ring_slots: AtomicU64,
    pub weight_ring_binds: AtomicU64,
    pub weight_ring_hits: AtomicU64,
    pub expert_slots: AtomicU64,
    pub expert_lru_hits: AtomicU64,
    pub expert_lru_misses: AtomicU64,
    pub expert_lru_fills: AtomicU64,
    pub expert_cpu_runs: AtomicU64,
    pub expert_pcie_bps: AtomicU64,
    pub expert_host_bps: AtomicU64,
    pub expert_host_slots: AtomicU64,
    pub expert_numa_nodes: AtomicU64,
    pub expert_numa_binds: AtomicU64,
    pub expert_numa_pins: AtomicU64,
    pub nvme_reads: AtomicU64,
    pub nvme_bytes: AtomicU64,
    pub nvme_direct: AtomicU64,
    pub anchor_hits: AtomicU64,
    pub anchor_skip_tokens: AtomicU64,
    pub radix_hits: AtomicU64,
    pub radix_skip_tokens: AtomicU64,
    pub ds_enqueued: AtomicU64,
    pub ds_bytes: AtomicU64,
    pub ds_pinned: AtomicU64,
    pub ds_pinned_bytes: AtomicU64,
    /// NVMe→host→VRAM expert fills with bound device slots.
    pub ds_gpu_dma: AtomicU64,
    pub ds_gpu_dma_bytes: AtomicU64,
    pub ds_com_binds: AtomicU64,
    pub ds_available: AtomicU64,
    pub micro_batch_plans: AtomicU64,
    /// Poll ticks that co-advanced a stagger partner (same-tick dual step).
    pub micro_batch_dual_steps: AtomicU64,
    /// MoE prefill: next-layer expert H2D queued while current layer computes.
    pub expert_prefill_overlaps: AtomicU64,
    /// Decode/micro-batch: expert miss fills merged into one contiguous DMA.
    pub expert_bulk_fills: AtomicU64,
    pub expert_bulk_bytes: AtomicU64,
    /// Slot copies issued through the batched H2D ABI on the side stream.
    pub expert_h2d_batched_copies: AtomicU64,
    /// Draft-span expert union prefetches before packed verify.
    pub draft_expert_prefetches: AtomicU64,
    /// Cross-request layer-stagger CPU expert steps during GPU forward.
    pub layer_stagger_steps: AtomicU64,
    /// Soft plans for packed cross-request HIP GEMM (execute still one-at-a-time).
    pub batch_gemm_plans: AtomicU64,
    pub radix_evictions: AtomicU64,
    pub radix_bytes: AtomicU64,
    pub gemv_tile_cols: AtomicU64,
    pub vram_rebalances: AtomicU64,
    /// Driver-measured whole-board VRAM, refreshed by [`GpuSampler`] off the
    /// decode path. `device_memory_bytes` is this process only; these include
    /// every other GPU client, which is what VRAM rebalancing must respect.
    pub gpu_vram_used_bytes: AtomicU64,
    pub gpu_vram_total_bytes: AtomicU64,
    pub draft_gpu_vram_used_bytes: AtomicU64,
    pub draft_gpu_vram_total_bytes: AtomicU64,
    /// Target-GPU busy fraction over the last sample window, scaled by 1e4.
    pub gpu_util_x1e4: AtomicU64,
    pub gpu_samples: AtomicU64,
    /// Soft attention-trim spills of cold VRAM KV pages to DRAM.
    pub kv_attention_trims: AtomicU64,
    /// Config requested early exit but the loaded model has no supported manifest.
    pub early_exit_gated: AtomicU64,
    /// Accepted early exits (self-speculate or dedicated exit heads).
    pub early_exit_accepts: AtomicU64,
    pub kv_nvme_pages: AtomicU64,
    pub kv_nvme_reads: AtomicU64,
    pub kv_nvme_bytes: AtomicU64,
    pub kv_nvme_direct: AtomicU64,
    pub kv_nvme_prefetches: AtomicU64,
    /// Continuous-batching `Engine::poll` invocations.
    pub poll_ticks: AtomicU64,
    pub poll_prefill_picks: AtomicU64,
    pub poll_decode_picks: AtomicU64,
    pub poll_wait_ms_ema_x100: AtomicU64,
    /// Grammar masks served from a completed background prefetch.
    pub async_mask_hits: AtomicU64,
    /// MTP process() skipped because draft KV seq_len did not match pos.
    pub mtp_process_skips: AtomicU64,
    pub rpc_roundtrips: AtomicU64,
    pub rpc_activation_bytes: AtomicU64,
    pub rpc_h2d_bytes: AtomicU64,
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
        draft_depth: u64,
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
            prefill_chunks: self.prefill_chunks.load(Ordering::Relaxed),
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
            tree_gpu_verifies: self.tree_gpu_verifies.load(Ordering::Relaxed),
            linear_packed_verifies: self.linear_packed_verifies.load(Ordering::Relaxed),
            draft_ahead_hits: self.draft_ahead_hits.load(Ordering::Relaxed),
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
            kv_bytes_on_device: self.kv_bytes_on_device.load(Ordering::Relaxed)
                + self.draft_kv_bytes_on_device.load(Ordering::Relaxed),
            draft_kv_bytes_on_device: self.draft_kv_bytes_on_device.load(Ordering::Relaxed),
            elementwise_gpu_ops: self.elementwise_gpu_ops.load(Ordering::Relaxed),
            residual_gpu_ops: self.residual_gpu_ops.load(Ordering::Relaxed),
            activation_h2d_bytes: self.activation_h2d_bytes.load(Ordering::Relaxed),
            activation_d2h_bytes: self.activation_d2h_bytes.load(Ordering::Relaxed),
            q8_gemm_us: self.q8_gemm_ns.load(Ordering::Relaxed) as f64 / 1_000.0,
            attention_us: self.attention_ns.load(Ordering::Relaxed) as f64 / 1_000.0,
            lm_head_us: self.lm_head_ns.load(Ordering::Relaxed) as f64 / 1_000.0,
            other_gpu_us: self.other_gpu_ns.load(Ordering::Relaxed) as f64 / 1_000.0,
            host_us: self.host_ns.load(Ordering::Relaxed) as f64 / 1_000.0,
            weight_pinned_layers: self.weight_pinned_layers.load(Ordering::Relaxed),
            weight_ring_slots: self.weight_ring_slots.load(Ordering::Relaxed),
            weight_ring_binds: self.weight_ring_binds.load(Ordering::Relaxed),
            weight_ring_hits: self.weight_ring_hits.load(Ordering::Relaxed),
            expert_slots: self.expert_slots.load(Ordering::Relaxed),
            expert_lru_hits: self.expert_lru_hits.load(Ordering::Relaxed),
            expert_lru_misses: self.expert_lru_misses.load(Ordering::Relaxed),
            expert_lru_fills: self.expert_lru_fills.load(Ordering::Relaxed),
            expert_cpu_runs: self.expert_cpu_runs.load(Ordering::Relaxed),
            expert_pcie_bps: self.expert_pcie_bps.load(Ordering::Relaxed),
            expert_host_bps: self.expert_host_bps.load(Ordering::Relaxed),
            expert_host_slots: self.expert_host_slots.load(Ordering::Relaxed),
            expert_numa_nodes: self.expert_numa_nodes.load(Ordering::Relaxed),
            expert_numa_binds: self.expert_numa_binds.load(Ordering::Relaxed),
            expert_numa_pins: self.expert_numa_pins.load(Ordering::Relaxed),
            nvme_reads: self.nvme_reads.load(Ordering::Relaxed),
            nvme_bytes: self.nvme_bytes.load(Ordering::Relaxed),
            nvme_direct: self.nvme_direct.load(Ordering::Relaxed) != 0,
            anchor_hits: self.anchor_hits.load(Ordering::Relaxed),
            anchor_skip_tokens: self.anchor_skip_tokens.load(Ordering::Relaxed),
            radix_hits: self.radix_hits.load(Ordering::Relaxed),
            radix_skip_tokens: self.radix_skip_tokens.load(Ordering::Relaxed),
            ds_enqueued: self.ds_enqueued.load(Ordering::Relaxed),
            ds_bytes: self.ds_bytes.load(Ordering::Relaxed),
            ds_pinned: self.ds_pinned.load(Ordering::Relaxed),
            ds_pinned_bytes: self.ds_pinned_bytes.load(Ordering::Relaxed),
            ds_gpu_dma: self.ds_gpu_dma.load(Ordering::Relaxed),
            ds_gpu_dma_bytes: self.ds_gpu_dma_bytes.load(Ordering::Relaxed),
            ds_com_binds: self.ds_com_binds.load(Ordering::Relaxed),
            ds_available: self.ds_available.load(Ordering::Relaxed) != 0,
            micro_batch_plans: self.micro_batch_plans.load(Ordering::Relaxed),
            micro_batch_dual_steps: self.micro_batch_dual_steps.load(Ordering::Relaxed),
            expert_prefill_overlaps: self.expert_prefill_overlaps.load(Ordering::Relaxed),
            expert_bulk_fills: self.expert_bulk_fills.load(Ordering::Relaxed),
            expert_bulk_bytes: self.expert_bulk_bytes.load(Ordering::Relaxed),
            expert_h2d_batched_copies: self.expert_h2d_batched_copies.load(Ordering::Relaxed),
            draft_expert_prefetches: self.draft_expert_prefetches.load(Ordering::Relaxed),
            layer_stagger_steps: self.layer_stagger_steps.load(Ordering::Relaxed),
            batch_gemm_plans: self.batch_gemm_plans.load(Ordering::Relaxed),
            radix_evictions: self.radix_evictions.load(Ordering::Relaxed),
            radix_bytes: self.radix_bytes.load(Ordering::Relaxed),
            gemv_tile_cols: self.gemv_tile_cols.load(Ordering::Relaxed),
            vram_rebalances: self.vram_rebalances.load(Ordering::Relaxed),
            gpu_vram_used_bytes: self.gpu_vram_used_bytes.load(Ordering::Relaxed),
            gpu_vram_total_bytes: self.gpu_vram_total_bytes.load(Ordering::Relaxed),
            draft_gpu_vram_used_bytes: self.draft_gpu_vram_used_bytes.load(Ordering::Relaxed),
            draft_gpu_vram_total_bytes: self.draft_gpu_vram_total_bytes.load(Ordering::Relaxed),
            gpu_util: self.gpu_util_x1e4.load(Ordering::Relaxed) as f64 / 10_000.0,
            gpu_samples: self.gpu_samples.load(Ordering::Relaxed),
            kv_attention_trims: self.kv_attention_trims.load(Ordering::Relaxed),
            early_exit_gated: self.early_exit_gated.load(Ordering::Relaxed),
            early_exit_accepts: self.early_exit_accepts.load(Ordering::Relaxed),
            kv_nvme_pages: self.kv_nvme_pages.load(Ordering::Relaxed),
            kv_nvme_reads: self.kv_nvme_reads.load(Ordering::Relaxed),
            kv_nvme_bytes: self.kv_nvme_bytes.load(Ordering::Relaxed),
            kv_nvme_direct: self.kv_nvme_direct.load(Ordering::Relaxed) != 0,
            kv_nvme_prefetches: self.kv_nvme_prefetches.load(Ordering::Relaxed),
            poll_ticks: self.poll_ticks.load(Ordering::Relaxed),
            poll_prefill_picks: self.poll_prefill_picks.load(Ordering::Relaxed),
            poll_decode_picks: self.poll_decode_picks.load(Ordering::Relaxed),
            poll_wait_ms_ema: self.poll_wait_ms_ema_x100.load(Ordering::Relaxed) as f64 / 100.0,
            async_mask_hits: self.async_mask_hits.load(Ordering::Relaxed),
            mtp_process_skips: self.mtp_process_skips.load(Ordering::Relaxed),
            draft_depth,
            rpc_roundtrips: self.rpc_roundtrips.load(Ordering::Relaxed),
            rpc_activation_bytes: self.rpc_activation_bytes.load(Ordering::Relaxed),
            rpc_h2d_bytes: self.rpc_h2d_bytes.load(Ordering::Relaxed),
        }
    }

    pub fn record_itl_ms(&self, ms: f64) {
        let sample = (ms.max(0.0) * 100.0).round() as u64;
        let prev = self.itl_ms_ema_x100.load(Ordering::Relaxed);
        let next = if prev == 0 {
            sample
        } else {
            (prev * 8 + sample * 2) / 10
        };
        self.itl_ms_ema_x100.store(next, Ordering::Relaxed);
    }

    /// Store one GPU sample. `util` is the share of decode wall time spent in
    /// allpaca kernels on the target device, clamped to 0.0..=1.0.
    pub fn record_gpu_sample(
        &self,
        vram_used: u64,
        vram_total: u64,
        draft_vram_used: u64,
        draft_vram_total: u64,
        util: f64,
    ) {
        self.gpu_vram_used_bytes.store(vram_used, Ordering::Relaxed);
        self.gpu_vram_total_bytes
            .store(vram_total, Ordering::Relaxed);
        self.draft_gpu_vram_used_bytes
            .store(draft_vram_used, Ordering::Relaxed);
        self.draft_gpu_vram_total_bytes
            .store(draft_vram_total, Ordering::Relaxed);
        let util = if util.is_finite() { util } else { 0.0 };
        self.gpu_util_x1e4.store(
            (util.clamp(0.0, 1.0) * 10_000.0).round() as u64,
            Ordering::Relaxed,
        );
        self.gpu_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Target-GPU busy fraction from the last sample, 0.0..=1.0.
    pub fn gpu_util(&self) -> f64 {
        self.gpu_util_x1e4.load(Ordering::Relaxed) as f64 / 10_000.0
    }

    /// Free VRAM on the target device from the last sample, or `None` when the
    /// sampler has not run (CPU path, or HIP without a driver reading).
    pub fn gpu_vram_free_bytes(&self) -> Option<u64> {
        let total = self.gpu_vram_total_bytes.load(Ordering::Relaxed);
        if total == 0 {
            return None;
        }
        Some(total.saturating_sub(self.gpu_vram_used_bytes.load(Ordering::Relaxed)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sample per window, and never two in the same window even under
    /// concurrent safe points, so decode cannot be taxed by a burst of callers.
    #[test]
    fn sampler_admits_one_caller_per_window() {
        let s = GpuSampler::new(50);
        assert!(s.due(), "first call must always sample");
        assert!(!s.due(), "second call inside the window must be refused");
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(s.due(), "window elapsed, one caller must win again");
        assert!(!s.due());
    }

    /// A zero interval still serializes callers but never blocks them.
    #[test]
    fn sampler_with_zero_interval_always_admits() {
        let s = GpuSampler::new(0);
        for _ in 0..4 {
            assert!(s.due());
        }
    }

    /// Util is stored lossily as x1e4 and must survive the round trip clamped,
    /// including the garbage a divide-by-zero profile can produce.
    #[test]
    fn gpu_sample_clamps_util_and_reports_free_vram() {
        let m = Metrics::default();
        assert_eq!(m.gpu_vram_free_bytes(), None, "unsampled must be None");

        m.record_gpu_sample(3 << 30, 16 << 30, 1 << 30, 8 << 30, 0.375);
        assert!((m.gpu_util() - 0.375).abs() < 1e-4);
        assert_eq!(m.gpu_vram_free_bytes(), Some(13 << 30));
        assert_eq!(m.gpu_samples.load(Ordering::Relaxed), 1);

        m.record_gpu_sample(0, 16 << 30, 0, 0, f64::NAN);
        assert_eq!(m.gpu_util(), 0.0, "NaN util must not poison the metric");
        m.record_gpu_sample(0, 16 << 30, 0, 0, 4.2);
        assert_eq!(m.gpu_util(), 1.0, "util must clamp to 1.0");
        assert_eq!(m.gpu_samples.load(Ordering::Relaxed), 3);
    }
}
