use serde::{Deserialize, Serialize};

/// In-memory telemetry snapshot exposed by `GET /stats`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatsSnapshot {
    pub pipeline_mode: String,
    pub transfer_path: String,
    /// True when a GPU pair can access peers after enable + re-query.
    pub p2p_capable: bool,
    /// True when `hipDeviceEnablePeerAccess` succeeded in both directions.
    pub p2p_enabled: bool,
    /// `"measured"` after a successful enable+peer-copy trial, or `"p2p_unavailable"`.
    pub p2p_status: String,
    pub p2p_gbps: f64,
    pub host_staging_gbps: f64,
    pub p2p_us: f64,
    pub host_staging_us: f64,
    pub target_device: i32,
    pub draft_device: i32,
    pub requests_total: u64,
    pub active_requests: u64,
    pub prompt_tokens: u64,
    /// Prompt prefill chunks completed (`prefill.chunk_tokens` each).
    pub prefill_chunks: u64,
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
    /// Packed Token Forest attention launches that ran the ancestry mask on HIP.
    pub tree_gpu_verifies: u64,
    /// Linear draft-span verifies that ran one Target Model forward over the proposed tokens.
    pub linear_packed_verifies: u64,
    /// Next-span Draft Model proposes that were reused after overlapping packed verify.
    #[serde(default)]
    pub draft_ahead_hits: u64,
    pub accepted_tokens_per_target_eval: f64,
    pub model_id: String,
    /// `"cpu"` or `"hip"` — which kernel backend ran target forwards.
    pub backend: String,
    pub gpu_forwards: u64,
    /// HIP forwards of the target model (fast GPU).
    pub target_gpu_forwards: u64,
    /// HIP forwards of the draft model (slow GPU when two devices exist).
    pub draft_gpu_forwards: u64,
    /// Host-staged bytes that crossed the device boundary (token ids / short logits).
    pub host_staging_bytes: u64,
    /// Draft token ids copied off the draft device via host staging (or P2P).
    pub draft_tokens_copied: u64,
    /// Full weight tensors moved between GPUs. Must stay 0 for resident speculation.
    pub weight_bytes_migrated: u64,
    /// Bytes currently owned by `hipMalloc` (0 on the CPU path).
    pub device_memory_bytes: u64,
    /// Model weight tensors resident on a HIP device after load (0 on CPU).
    pub weight_bytes_on_device: u64,
    /// Cumulative H2D bytes of weight tensors. Grows at load only, not decode GEMM.
    pub weight_h2d_bytes: u64,
    /// KV page bytes currently in `hipMalloc` (0 when pages are host Vecs).
    /// Includes draft DenseKv pages when they are device-resident.
    pub kv_bytes_on_device: u64,
    /// Draft DenseKv bytes in `hipMalloc` on the draft device (0 on 1 GPU / CPU).
    pub draft_kv_bytes_on_device: u64,
    /// HIP RoPE / RMSNorm / SiLU launches (0 on the CPU fallback path).
    pub elementwise_gpu_ops: u64,
    /// HIP residual-add launches (0 on the CPU fallback path).
    pub residual_gpu_ops: u64,
    /// Cumulative H2D bytes of activations (hidden, QKV, KV gather). Not weights.
    pub activation_h2d_bytes: u64,
    /// Cumulative D2H bytes of activations. Decode should grow only by logits.
    pub activation_d2h_bytes: u64,
    /// Last/EMA HIP decode: Q8_0 GEMM/GEMV time in microseconds (0 on cpu_only).
    pub q8_gemm_us: f64,
    /// Last/EMA HIP decode: attention kernel time in microseconds (0 on cpu_only).
    pub attention_us: f64,
    /// Last/EMA HIP decode: F32 lm_head GEMM/GEMV time in microseconds (0 on cpu_only).
    pub lm_head_us: f64,
    /// Last/EMA HIP decode: other GPU kernels (RoPE, RMSNorm, residual, F32 layer GEMM).
    pub other_gpu_us: f64,
    /// Last/EMA HIP decode: host-side time (D2H logits, bookkeeping) in microseconds.
    pub host_us: f64,
    /// Dense FFN layers pinned in the weight working set (0 = full resident / unset).
    #[serde(default)]
    pub weight_pinned_layers: u64,
    /// Ring slots granted for streamed FFN (0 when fully pinned).
    #[serde(default)]
    pub weight_ring_slots: u64,
    /// Bind attempts on streamed FFN layers.
    #[serde(default)]
    pub weight_ring_binds: u64,
    /// Binds served from a ring slot without copying from the host pool.
    #[serde(default)]
    pub weight_ring_hits: u64,
    /// GPU/hot expert LRU slots (0 = host pool only).
    #[serde(default)]
    pub expert_slots: u64,
    #[serde(default)]
    pub expert_lru_hits: u64,
    #[serde(default)]
    pub expert_lru_misses: u64,
    #[serde(default)]
    pub expert_lru_fills: u64,
    #[serde(default)]
    pub expert_cpu_runs: u64,
    /// Measured expert H2D bandwidth (bytes/s) used by `q*`. 0 = unmeasured.
    #[serde(default)]
    pub expert_pcie_bps: u64,
    /// Measured CPU expert-kernel bandwidth (bytes/s) used by `q*`. 0 = unmeasured.
    #[serde(default)]
    pub expert_host_bps: u64,
    /// DRAM slots in the NVMe-backed expert host pool (0 = full RAM pool).
    #[serde(default)]
    pub expert_host_slots: u64,
    /// Host NUMA nodes visible to expert placement.
    #[serde(default)]
    pub expert_numa_nodes: u64,
    /// Successful CPU thread binds to an expert's preferred NUMA node.
    #[serde(default)]
    pub expert_numa_binds: u64,
    /// Expert weight clones first-touched onto a preferred NUMA node.
    #[serde(default)]
    pub expert_numa_pins: u64,
    /// Direct I/O (or buffered fallback) reads from the packed expert file.
    #[serde(default)]
    pub nvme_reads: u64,
    /// Bytes read from the packed expert file (aligned slot size, not payload).
    #[serde(default)]
    pub nvme_bytes: u64,
    /// True when packed expert reads used Direct I/O rather than the page cache.
    #[serde(default)]
    pub nvme_direct: bool,
    /// Semantic-anchor restores (recurrent state, not KV LRU).
    #[serde(default)]
    pub anchor_hits: u64,
    /// Prompt tokens skipped because a surviving anchor restored the prefix.
    #[serde(default)]
    pub anchor_skip_tokens: u64,
    /// Radix-prefix KV restores (mid-prompt edit LCP).
    #[serde(default)]
    pub radix_hits: u64,
    /// Prompt tokens skipped via radix LCP restore.
    #[serde(default)]
    pub radix_skip_tokens: u64,
    /// DirectStorage DMA intents + host-staged/pinned H2D transfers.
    #[serde(default)]
    pub ds_enqueued: u64,
    #[serde(default)]
    pub ds_bytes: u64,
    /// Aligned Direct I/O fills into pinned-shaped host memory (H2D source).
    #[serde(default)]
    pub ds_pinned: u64,
    #[serde(default)]
    pub ds_pinned_bytes: u64,
    /// True when DirectStorage/GDS probe succeeded at engine start.
    #[serde(default)]
    pub ds_available: bool,
    /// Cross-request GPU/CPU stagger plans recorded under concurrency.
    #[serde(default)]
    pub micro_batch_plans: u64,
    /// Poll ticks that co-advanced a stagger partner (same-tick dual step).
    #[serde(default)]
    pub micro_batch_dual_steps: u64,
    /// MoE prefill: next-layer expert H2D overlapped with current layer GPU work.
    #[serde(default)]
    pub expert_prefill_overlaps: u64,
    /// Expert miss fills merged into one contiguous DMA (decode micro-batch union).
    #[serde(default)]
    pub expert_bulk_fills: u64,
    #[serde(default)]
    pub expert_bulk_bytes: u64,
    /// Radix forest nodes evicted to stay under the node/byte caps.
    #[serde(default)]
    pub radix_evictions: u64,
    /// Approximate checkpoint bytes currently held in the radix forest.
    #[serde(default)]
    pub radix_bytes: u64,
    /// Last §12 MoE GEMV column tile (32/64/128) chosen for Q5_K/Q6_K.
    #[serde(default)]
    pub gemv_tile_cols: u64,
    /// Elastic KV→expert VRAM rebalances at scheduler safe points.
    #[serde(default)]
    pub vram_rebalances: u64,
    /// KV pages currently resident on the NVMe cold tier.
    #[serde(default)]
    pub kv_nvme_pages: u64,
    /// Direct I/O (or buffered fallback) reads of spilled KV pages.
    #[serde(default)]
    pub kv_nvme_reads: u64,
    /// Bytes moved to/from the KV swap file (aligned slot size).
    #[serde(default)]
    pub kv_nvme_bytes: u64,
    /// True when KV cold-tier I/O used Direct I/O rather than the page cache.
    #[serde(default)]
    pub kv_nvme_direct: bool,
    /// NVMe → DRAM restores of the next page_index (prefetch, not a decode fault).
    #[serde(default)]
    pub kv_nvme_prefetches: u64,
    /// Continuous-batching `Engine::poll` ticks.
    #[serde(default)]
    pub poll_ticks: u64,
    /// Poll picks while the Request was still prefilling.
    #[serde(default)]
    pub poll_prefill_picks: u64,
    /// Poll picks while the Request was decoding (or emitting).
    #[serde(default)]
    pub poll_decode_picks: u64,
    /// EMA of wait-ms before a poll pick (longest-waiting fairness).
    #[serde(default)]
    pub poll_wait_ms_ema: f64,
    /// Grammar masks served from a completed background prefetch.
    #[serde(default)]
    pub async_mask_hits: u64,
    /// Current adaptive draft span (from the scheduler).
    #[serde(default)]
    pub draft_depth: u64,
}
