use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use allpaca_gguf::{extract_tokenizer, GgufFile};
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{
    load_hf_config, load_vocab_sidecar, resolve_sidecar, SafetensorsFile,
};
use allpaca_types::{
    AllpacaError, Config, DraftProviderKind, FatalError, FeatureFlags, FinishReason,
    GenerationOutput, GenerationParams, KvHandle, StatsSnapshot, StepResult, TokenEvent,
    TransientError,
};
use futures::Stream;
use rand::rngs::StdRng;

use crate::anchors::{restore_kv, AnchorCache, RecurrentState};
use crate::constraint::JsonConstraint;
use crate::device_manager::DeviceInventory;
use crate::direct_storage::{
    probe_direct_storage, DirectStorageBridge, DirectStorageProbe, DirectStorageQueue,
};
use crate::draft::{DraftModelProvider, DraftProvider, NGramProvider};
use crate::forest::{build_token_forest, TokenForest};
use crate::kv_radix::KvRadixForest;
use crate::kv_tier::KvTierManager;
use crate::logits_ring::LogitRing;
use crate::expert_batch::MicroBatchExpertUnion;
use crate::micro_batch::{plan_micro_batch, ActiveLayerHint, StaggerPhase};
use crate::model::{
    alloc_pages_for_seq, ensure_pages, DenseKv, ModelConfig, ResidentModel, TinyModel,
};
use crate::sampling::{rng_from_params, sample};
use crate::scheduler::Scheduler;
use crate::telemetry::Metrics;
use crate::tokenizer::EngineTokenizer;
use crate::packed::{temp_pack_path, PackedStore, HostExpertPool};
use crate::weight_set::{
    expert_lru_slots, host_pool_slots, DenseTrunkPlan, ExpertId, ExpertLru, LayerFfn,
    WeightWorkingSet,
};
use allpaca_hip::HipWorkspace;

struct ActiveRequest {
    id: u64,
    tokens: Vec<u32>,
    prompt_len: usize,
    prefill_pos: usize,
    generated: usize,
    params: GenerationParams,
    pages: Vec<Vec<u64>>,
    pending_logits: Vec<f32>,
    emit_queue: Vec<TokenEvent>,
    /// StepResults parked by a peer thread's fair `poll` demux.
    parked: VecDeque<StepResult>,
    /// Prefetched grammar mask from a background thread (`structured_output.async`).
    async_mask_rx: Option<std::sync::mpsc::Receiver<Result<allpaca_types::TokenMask, String>>>,
    json_prefix: String,
    grammar: Option<llguidance::Matcher>,
    rng: StdRng,
    started: Instant,
    last_token_at: Instant,
    first_emitted: bool,
    finished: bool,
    recurrent: Option<RecurrentState>,
}

struct EngineInner {
    flags: FeatureFlags,
    scheduler: Scheduler,
    kv: KvTierManager,
    ring: LogitRing,
    tokenizer: Option<EngineTokenizer>,
    target: Option<Arc<TinyModel>>,
    draft_model: Option<Arc<TinyModel>>,
    ngram: NGramProvider,
    active: HashMap<u64, ActiveRequest>,
    next_request_id: u64,
    model_id: String,
    target_resident: Option<Arc<ResidentModel>>,
    draft_resident: Option<Arc<ResidentModel>>,
    draft_kv: DenseKv,
    draft_ahead: Option<DraftAhead>,
    anchors: AnchorCache,
    /// GEM radix prefix KV reuse (arbitrary mid-prompt edits).
    kv_radix: KvRadixForest,
    /// DirectStorage / GDS capability probe at engine start.
    ds_probe: DirectStorageProbe,
    grammar_factory: Option<llguidance::ParserFactory>,
    /// Round-robin cursor for continuous-batching ties.
    last_poll_id: Option<u64>,
    /// Consecutive prefill picks under `PrefillBudget` schedule.
    consecutive_prefill_picks: usize,
    /// Next poll prefers this request (cross-req micro-batch partner).
    prefer_next: Option<u64>,
}

struct DraftAhead {
    ctx: Vec<u32>,
    tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<EngineInner>>,
    config: Config,
    inventory: DeviceInventory,
    metrics: Arc<Metrics>,
    hip: Option<Arc<HipWorkspace>>,
    draft_hip: Option<Arc<HipWorkspace>>,
    /// Shared DirectStorage enqueue ledger (file-coordinate + staged H2D).
    ds_queue: Arc<DirectStorageQueue>,
    ds_probe: DirectStorageProbe,
    micro_batch: Arc<MicroBatchExpertUnion>,
}

pub struct RequestHandle {
    id: u64,
    inner: Arc<Mutex<EngineInner>>,
    metrics: Arc<Metrics>,
}

impl RequestHandle {
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn kv_handle(&self) -> Result<KvHandle, AllpacaError> {
        let guard = self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
        })?;
        let active = guard
            .active
            .get(&self.id)
            .ok_or(AllpacaError::Transient(TransientError::RequestBusy))?;
        Ok(KvHandle {
            request_id: active.id,
            epoch: 0,
            token_count: active.tokens.len(),
            pages: active.pages.iter().flatten().copied().collect(),
        })
    }
}

impl std::fmt::Debug for RequestHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestHandle").field("id", &self.id).finish()
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(active) = guard.active.remove(&self.id) {
                for layer in active.pages {
                    for page in layer {
                        guard.kv.release(page, &self.metrics);
                    }
                }
                self.metrics.active_requests.store(
                    guard.active.len() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }
    }
}

impl Engine {
    pub fn new(config: Config) -> Result<Self, AllpacaError> {
        let mut config = config;
        if config.model.draft_path.is_none() {
            if let Ok(raw) = std::env::var("ALLPACA_DRAFT_GGUF") {
                let p = std::path::PathBuf::from(raw);
                if p.is_file() {
                    config.model.draft_path = Some(p);
                }
            }
        }
        config.validate()?;
        let inventory = DeviceInventory::discover(&config);
        let mut flags = FeatureFlags::from_mode(inventory.pipeline_mode);
        flags.overlay(&config.features);
        apply_provider_flags(&mut flags, &config);
        let scheduler = Scheduler::new(inventory.pipeline_mode, flags.clone(), &config.speculative);
        let hip = open_hip_workspace(&config, &inventory).map(Arc::new);
        let draft_hip = open_draft_hip_workspace(&config, &inventory).map(Arc::new);
        let mut kv = KvTierManager::with_hip(
            config.kv.vram_bytes,
            config.kv.dram_bytes,
            config.kv.hot_fraction,
            hip.as_ref().map(|h| h.device),
        );
        kv.open_nvme(config.kv.nvme_bytes, config.kv.nvme_path.as_deref())?;
        let ring = LogitRing::new(config.sampling.pinned_ring_slots, 64);
        let ds_probe = probe_direct_storage();
        let ds_queue = Arc::new(DirectStorageQueue::new());
        let inner = EngineInner {
            flags,
            scheduler,
            kv,
            ring,
            tokenizer: None,
            target: None,
            draft_model: None,
            ngram: NGramProvider::new(config.speculative.ngram_n),
            active: HashMap::new(),
            next_request_id: 1,
            model_id: config.model.id.clone(),
            target_resident: None,
            draft_resident: None,
            draft_kv: DenseKv::default(),
            draft_ahead: None,
            anchors: AnchorCache::new(config.anchors.slots, config.anchors.token_ids.clone()),
            kv_radix: KvRadixForest::with_limits(4096, 64 << 20),
            ds_probe: ds_probe.clone(),
            grammar_factory: None,
            last_poll_id: None,
            consecutive_prefill_picks: 0,
            prefer_next: None,
        };
        let engine = Self {
            inner: Arc::new(Mutex::new(inner)),
            config: config.clone(),
            inventory,
            metrics: Arc::new(Metrics::default()),
            hip,
            draft_hip,
            ds_queue,
            ds_probe,
            micro_batch: Arc::new(MicroBatchExpertUnion::new()),
        };
        if config.model.synthetic {
            engine.load_synthetic(ModelConfig::tiny())?;
        } else if let Some(path) = &config.model.path {
            engine.load_model_path(path)?;
        }
        Ok(engine)
    }

    pub fn inventory(&self) -> &DeviceInventory {
        &self.inventory
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn stats(&self) -> StatsSnapshot {
        let (model_id, kv_h2d, kv_d2h, draft_kv_bytes) = match self.lock() {
            Ok(g) => {
                if let Some(ws) = g.target.as_ref().and_then(|m| m.working_set.as_ref()) {
                    self.metrics
                        .weight_ring_binds
                        .store(ws.ring_binds(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics
                        .weight_ring_hits
                        .store(ws.ring_hits(), std::sync::atomic::Ordering::Relaxed);
                }
                if let Some(pool) = g.target.as_ref().and_then(|m| m.expert_host.as_ref()) {
                    self.metrics
                        .expert_host_slots
                        .store(pool.n_slots() as u64, std::sync::atomic::Ordering::Relaxed);
                    HostExpertPool::record_numa(&self.metrics, pool.numa());
                    self.metrics
                        .nvme_reads
                        .store(pool.nvme_reads(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics
                        .nvme_bytes
                        .store(pool.nvme_bytes(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics.nvme_direct.store(
                        u64::from(pool.is_direct()),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                } else if let Some(ws) = g.target.as_ref().and_then(|m| m.working_set.as_ref()) {
                    self.metrics
                        .nvme_reads
                        .store(ws.nvme_reads(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics
                        .nvme_bytes
                        .store(ws.nvme_bytes(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics.nvme_direct.store(
                        u64::from(ws.nvme_direct()),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                if let Some(lru) = g.target.as_ref().and_then(|m| m.expert_lru.as_ref()) {
                    self.metrics
                        .expert_lru_hits
                        .store(lru.hits(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics
                        .expert_lru_misses
                        .store(lru.misses(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics
                        .expert_lru_fills
                        .store(lru.fills(), std::sync::atomic::Ordering::Relaxed);
                    self.metrics
                        .expert_cpu_runs
                        .store(lru.cpu_runs(), std::sync::atomic::Ordering::Relaxed);
                }
                self.metrics
                    .anchor_hits
                    .store(g.anchors.hits(), std::sync::atomic::Ordering::Relaxed);
                self.metrics.anchor_skip_tokens.store(
                    g.anchors.skip_tokens(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics
                    .radix_hits
                    .store(g.kv_radix.hits(), std::sync::atomic::Ordering::Relaxed);
                self.metrics.radix_skip_tokens.store(
                    g.kv_radix.skip_tokens(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics.radix_evictions.store(
                    g.kv_radix.evictions(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics.radix_bytes.store(
                    g.kv_radix.checkpoint_bytes() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics.ds_available.store(
                    u64::from(g.ds_probe.available),
                    std::sync::atomic::Ordering::Relaxed,
                );
                // DMA intents + host-staged/pinned H2D from the shared queue.
                let q = &self.ds_queue;
                self.metrics.ds_enqueued.store(
                    q.enqueued() + q.staged() + q.pinned(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics.ds_bytes.store(
                    q.bytes() + q.staged_bytes() + q.pinned_bytes(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics
                    .ds_pinned
                    .store(q.pinned(), std::sync::atomic::Ordering::Relaxed);
                self.metrics.ds_pinned_bytes.store(
                    q.pinned_bytes(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics
                    .kv_nvme_pages
                    .store(g.kv.nvme_count(), std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .kv_nvme_reads
                    .store(g.kv.nvme_reads(), std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .kv_nvme_bytes
                    .store(g.kv.nvme_bytes(), std::sync::atomic::Ordering::Relaxed);
                self.metrics.kv_nvme_direct.store(
                    u64::from(g.kv.nvme_direct()),
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics.kv_nvme_prefetches.store(
                    g.kv.nvme_prefetches(),
                    std::sync::atomic::Ordering::Relaxed,
                );
                (
                g.target
                    .as_ref()
                    .map(|m| m.name.clone())
                    .unwrap_or_else(|| self.config.model.id.clone()),
                g.kv.activation_h2d_bytes(),
                g.kv.activation_d2h_bytes(),
                g.draft_kv.device_bytes(),
                )
            }
            Err(_) => (self.config.model.id.clone(), 0, 0, 0),
        };
        let (backend, device_memory_bytes) = if self.hip.is_some() {
            ("hip".into(), allpaca_hip::allocated_bytes())
        } else {
            ("cpu".into(), 0)
        };
        let hip_h2d = self
            .hip
            .as_ref()
            .map(|h| h.activation_h2d_bytes())
            .unwrap_or(0);
        let hip_d2h = self
            .hip
            .as_ref()
            .map(|h| h.activation_d2h_bytes())
            .unwrap_or(0);
        let draft_h2d = self
            .draft_hip
            .as_ref()
            .map(|h| h.activation_h2d_bytes())
            .unwrap_or(0);
        let draft_d2h = self
            .draft_hip
            .as_ref()
            .map(|h| h.activation_d2h_bytes())
            .unwrap_or(0);
        self.metrics
            .device_memory_bytes
            .store(device_memory_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .draft_kv_bytes_on_device
            .store(draft_kv_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .activation_h2d_bytes
            .store(hip_h2d + kv_h2d + draft_h2d, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .activation_d2h_bytes
            .store(hip_d2h + kv_d2h + draft_d2h, std::sync::atomic::Ordering::Relaxed);
        if let Some(hip) = &self.hip {
            let p = hip.decode_profile();
            self.metrics
                .q8_gemm_ns
                .store((p.q8_gemm_us * 1_000.0).round() as u64, std::sync::atomic::Ordering::Relaxed);
            self.metrics
                .attention_ns
                .store((p.attention_us * 1_000.0).round() as u64, std::sync::atomic::Ordering::Relaxed);
            self.metrics
                .lm_head_ns
                .store((p.lm_head_us * 1_000.0).round() as u64, std::sync::atomic::Ordering::Relaxed);
            self.metrics
                .other_gpu_ns
                .store((p.other_gpu_us * 1_000.0).round() as u64, std::sync::atomic::Ordering::Relaxed);
            let gpu_us = p.q8_gemm_us + p.attention_us + p.lm_head_us + p.other_gpu_us;
            let itl_us = self.metrics.itl_ms_ema_x100.load(std::sync::atomic::Ordering::Relaxed)
                as f64
                * 10.0;
            let host_us = if itl_us > gpu_us {
                itl_us - gpu_us
            } else {
                p.host_us
            };
            self.metrics
                .host_ns
                .store((host_us * 1_000.0).round() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        let bench = self.inventory.transfer_benchmark;
        let p2p_copy_ok = bench.map(|b| b.p2p_copy_ok).unwrap_or(false);
        let p2p_status = if p2p_copy_ok {
            "measured"
        } else {
            "p2p_unavailable"
        };
        let draft_depth = self
            .lock()
            .ok()
            .map(|g| g.scheduler.draft_depth as u64)
            .unwrap_or(0);
        self.metrics.snapshot(
            &format!("{:?}", self.inventory.pipeline_mode),
            &format!("{:?}", self.inventory.transfer_path),
            self.inventory.p2p_capable,
            self.inventory.p2p_enabled,
            p2p_status,
            bench.map(|b| b.p2p_gbps).unwrap_or(0.0),
            bench.map(|b| b.host_staging_gbps).unwrap_or(0.0),
            bench.map(|b| b.p2p_us).unwrap_or(0.0),
            bench.map(|b| b.host_staging_us).unwrap_or(0.0),
            self.inventory.target_device,
            self.inventory.draft_device,
            model_id,
            backend,
            draft_depth,
        )
    }

    pub fn model_id(&self) -> String {
        self.lock()
            .ok()
            .map(|g| g.model_id.clone())
            .unwrap_or_else(|| self.config.model.id.clone())
    }

    pub fn has_model(&self) -> bool {
        self.lock()
            .map(|g| g.target.is_some() && g.tokenizer.is_some())
            .unwrap_or(false)
    }

    pub fn load_synthetic(&self, cfg: ModelConfig) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let seed = self.config.engine.determinism_seed.unwrap_or(42);
        let tokenizer = EngineTokenizer::synthetic(cfg.vocab_size)?;
        inner.ring.resize_vocab(cfg.vocab_size);
        let mut target = TinyModel::synthetic(cfg.clone(), seed, "synthetic");
        self.apply_weight_working_set(&mut target)?;
        self.apply_expert_lru(&mut target)?;
        inner.target = Some(Arc::new(target));
        if inner.flags.draft_model_speculation {
            let mut draft = TinyModel::synthetic(
                ModelConfig::draft_tiny(),
                seed.wrapping_add(1),
                "synthetic-draft",
            );
            self.apply_weight_working_set(&mut draft)?;
            self.apply_expert_lru(&mut draft)?;
            inner.draft_model = Some(Arc::new(draft));
        }
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = "synthetic".into();
        drop(inner);
        self.upload_resident_weights()?;
        tracing::info!(vocab = cfg.vocab_size, layers = cfg.n_layers, "loaded synthetic model");
        Ok(())
    }

    fn apply_weight_working_set(&self, model: &mut TinyModel) -> Result<(), AllpacaError> {
        let cfg = &self.config.weights;
        if cfg.trunk_bytes == 0 || !model.host_weights_ready() {
            return Ok(());
        }
        let sizes: Vec<usize> = model.layers.iter().map(|l| l.ffn_bytes()).collect();
        if sizes.iter().all(|&b| b == 0) {
            return Ok(());
        }
        let requested = cfg.ring_slots.max(1);
        let plan = DenseTrunkPlan::plan(&sizes, cfg.trunk_bytes, requested);
        WeightWorkingSet::record_plan(&self.metrics, &plan);
        if !plan.streams() {
            return Ok(());
        }
        model.working_set = Some(WeightWorkingSet::detach_unpinned(&mut model.layers, plan)?);
        Ok(())
    }

    fn apply_expert_lru(&self, model: &mut TinyModel) -> Result<(), AllpacaError> {
        if model.config.n_experts == 0 || !model.host_weights_ready() {
            return Ok(());
        }
        let expert_bytes = model
            .layers
            .first()
            .and_then(|l| l.experts.first())
            .map(|e| e.bytes())
            .unwrap_or(0);
        if expert_bytes == 0 {
            return Ok(());
        }
        let sample = model.layers.first().and_then(|l| l.experts.first());
        let sample_id = ExpertId { layer: 0, expert: 0 };
        let b_host = sample
            .map(|e| {
                if let Some(pool) = &model.expert_host {
                    crate::model::measure_expert_host_bandwidth_numa(
                        e,
                        model.config.ff_dim,
                        model.config.dim,
                        pool.numa(),
                        sample_id,
                    )
                } else {
                    crate::model::measure_expert_host_bandwidth(
                        e,
                        model.config.ff_dim,
                        model.config.dim,
                    )
                }
            })
            .unwrap_or(0.0);
        let b_pcie = self
            .hip
            .as_ref()
            .and_then(|hip| {
                allpaca_hip::benchmark_h2d_bytes_per_sec(hip.device, expert_bytes).ok()
            })
            .unwrap_or(0.0);
        self.metrics
            .expert_host_bps
            .store(b_host as u64, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .expert_pcie_bps
            .store(b_pcie as u64, std::sync::atomic::Ordering::Relaxed);

        let pool_bytes = expert_bytes
            .saturating_mul(model.layers.len())
            .saturating_mul(model.config.n_experts);
        let host_bytes = self.config.weights.host_bytes;
        if host_bytes > 0 && pool_bytes > host_bytes {
            let n_host = host_pool_slots(expert_bytes, host_bytes);
            let path = temp_pack_path();
            let pairs: Vec<(ExpertId, &LayerFfn)> = model
                .layers
                .iter()
                .enumerate()
                .flat_map(|(li, layer)| {
                    layer.experts.iter().enumerate().map(move |(ei, expert)| {
                        (
                            ExpertId {
                                layer: li as u32,
                                expert: ei as u32,
                            },
                            expert,
                        )
                    })
                })
                .collect();
            let store = PackedStore::pack(&path, pairs)?;
            for layer in &mut model.layers {
                for expert in &mut layer.experts {
                    *expert = LayerFfn::default();
                }
            }
            HostExpertPool::record_slots(&self.metrics, n_host);
            let ds = DirectStorageBridge {
                probe: self.ds_probe.clone(),
                queue: Arc::clone(&self.ds_queue),
            };
            model.expert_host = Some(HostExpertPool::with_direct_storage(store, n_host, ds));
        }

        let n_slots = expert_lru_slots(expert_bytes, self.config.weights.cache_bytes);
        ExpertLru::record_slots(&self.metrics, n_slots);
        if n_slots == 0 {
            return Ok(());
        }
        let (t_dma, t_dispatch) = crate::model::measure_qstar_fixed_costs(expert_bytes);
        model.expert_lru = Some(ExpertLru::with_overhead(
            n_slots,
            self.config.weights.miss_policy,
            b_pcie,
            b_host,
            t_dma,
            t_dispatch,
            expert_bytes as f64,
        ));
        if let Some(pool) = &model.expert_host {
            HostExpertPool::record_numa(&self.metrics, pool.numa());
        }
        Ok(())
    }

    fn load_model_path(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => self.load_safetensors(path),
            Some("onnx") => self.load_onnx(path),
            _ => self.load_gguf(path),
        }
    }

    pub fn load_safetensors(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = SafetensorsFile::open(path)?;
        let config_path = resolve_sidecar(
            self.config.model.config_path.as_deref(),
            path,
            "config.json",
        );
        let tokenizer_path = resolve_sidecar(
            self.config.model.tokenizer_path.as_deref(),
            path,
            "tokenizer.json",
        );
        let hf = load_hf_config(&config_path)?;
        let vocab = load_vocab_sidecar(&tokenizer_path)?;
        let mut model = TinyModel::from_safetensors(&file, &hf)?;
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        let tokenizer = EngineTokenizer::from_vocab(vocab.tokens, vocab.bos_id, vocab.eos_id)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()
    }

    pub fn load_onnx(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = OnnxFile::open(path)?;
        file.ensure_weight_export()?;
        let config_path = resolve_sidecar(
            self.config.model.config_path.as_deref(),
            path,
            "config.json",
        );
        let tokenizer_path = resolve_sidecar(
            self.config.model.tokenizer_path.as_deref(),
            path,
            "tokenizer.json",
        );
        let hf = load_hf_config(&config_path)?;
        let vocab = load_vocab_sidecar(&tokenizer_path)?;
        let mut model = TinyModel::from_onnx(&file, &hf)?;
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        let tokenizer = EngineTokenizer::from_vocab(vocab.tokens, vocab.bos_id, vocab.eos_id)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()
    }

    pub fn load_gguf(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = GgufFile::open(path)?;
        let tok = extract_tokenizer(&file)?;
        let mut model = TinyModel::from_gguf(&file)?;
        if !model.host_weights_ready() && self.hip.is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "large GGUF requires HIP (host F32 materialize is skipped to avoid OOM)".into(),
            )));
        }
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        let mut draft_file = None;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, EngineTokenizer::from_gguf(tok));
        if let Some(draft_path) = &self.config.model.draft_path {
            match GgufFile::open(draft_path) {
                Ok(f) => match TinyModel::from_gguf(&f) {
                    Ok(draft) => {
                        inner.draft_model = Some(Arc::new(draft));
                        draft_file = Some(f);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "draft GGUF failed; disabling draft-model speculation");
                        inner.flags.draft_model_speculation = false;
                        self.metrics
                            .degraded_events
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, "draft GGUF failed; disabling draft-model speculation");
                    inner.flags.draft_model_speculation = false;
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_from_gguf(Some(&file), draft_file.as_ref())
    }

    fn upload_resident_weights(&self) -> Result<(), AllpacaError> {
        self.upload_resident_from_gguf(None, None)
    }

    fn upload_resident_from_gguf(
        &self,
        target_gguf: Option<&GgufFile>,
        draft_gguf: Option<&GgufFile>,
    ) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let mut weight_bytes = 0u64;
        let mut h2d = 0u64;
        let target = inner.target.clone();
        let draft = inner.draft_model.clone();
        inner.target_resident = if let (Some(hip), Some(target)) = (&self.hip, target) {
            let uploaded = if target.config.n_experts > 0 && target.host_weights_ready() {
                ResidentModel::upload(&target, hip.device)
            } else if let Some(file) = target_gguf {
                ResidentModel::upload_gguf(file, &target.config, hip.device)
            } else if target.host_weights_ready() {
                ResidentModel::upload(&target, hip.device)
            } else {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "large GGUF missing file handle for HIP upload".into(),
                )))
            };
            match uploaded {
                Ok(resident) => {
                    let n = resident.bytes();
                    weight_bytes += n;
                    h2d += resident.packed_h2d_bytes();
                    Some(Arc::new(resident))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "HIP target weight upload failed; GEMM will H2D per op");
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        } else {
            None
        };
        inner.draft_resident = if let (Some(hip), Some(draft)) = (&self.draft_hip, draft) {
            tracing::info!(device = hip.device, "HIP draft weight upload start");
            let uploaded = if draft.config.n_experts > 0 && draft.host_weights_ready() {
                ResidentModel::upload(&draft, hip.device)
            } else if let Some(file) = draft_gguf {
                ResidentModel::upload_gguf(file, &draft.config, hip.device)
            } else if draft.host_weights_ready() {
                ResidentModel::upload(&draft, hip.device)
            } else {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "large draft GGUF missing file handle for HIP upload".into(),
                )))
            };
            match uploaded {
                Ok(resident) => {
                    let n = resident.bytes();
                    weight_bytes += n;
                    h2d += resident.packed_h2d_bytes();
                    Some(Arc::new(resident))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "HIP draft weight upload failed; GEMM will H2D per op");
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
            }
        } else {
            None
        };
        self.metrics
            .weight_bytes_on_device
            .store(weight_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .weight_h2d_bytes
            .store(h2d, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationOutput, AllpacaError> {
        let mut handle = self.begin_request(prompt, params)?;
        let mut token_ids = Vec::new();
        let finish = self.drive_request(&mut handle, |ev| {
            token_ids.push(ev.token_id);
            ev.finish_reason
        })?;
        let text = {
            let inner = self.lock()?;
            inner
                .tokenizer
                .as_ref()
                .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
                .decode(&token_ids)?
        };
        let prompt_tokens = handle_prompt_len(&handle, &self.inner)?;
        Ok(GenerationOutput {
            text,
            completion_tokens: token_ids.len(),
            token_ids,
            prompt_tokens,
            finish_reason: finish,
        })
    }

    pub fn generate_stream(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> impl Stream<Item = Result<TokenEvent, AllpacaError>> + Send {
        let engine = self.clone();
        let use_fair = engine.config.engine.max_active_requests > 1;
        async_stream::stream! {
            let mut handle = match engine.begin_request(&prompt, params) {
                Ok(h) => h,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            if use_fair {
                loop {
                    match engine.next_fair(&mut handle) {
                        Ok(None) => break,
                        Ok(Some(ev)) => {
                            let done = ev.finish_reason.is_some();
                            yield Ok(ev);
                            if done {
                                break;
                            }
                        }
                        Err(e) => {
                            yield Err(e);
                            break;
                        }
                    }
                }
            } else {
                loop {
                    match engine.step(&mut handle) {
                        Ok(StepResult::Token(ev)) => {
                            let done = ev.finish_reason.is_some();
                            yield Ok(ev);
                            if done {
                                break;
                            }
                        }
                        Ok(StepResult::Prefill { .. }) => {}
                        Ok(StepResult::Finished(_)) => break,
                        Err(e) => {
                            yield Err(e);
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Drive one Request to completion. When `max_active_requests > 1`, uses
    /// `poll` and parks foreign events so concurrent callers share fair scheduling.
    fn drive_request(
        &self,
        handle: &mut RequestHandle,
        mut on_token: impl FnMut(TokenEvent) -> Option<FinishReason>,
    ) -> Result<FinishReason, AllpacaError> {
        if self.config.engine.max_active_requests <= 1 {
            loop {
                match self.step(handle)? {
                    StepResult::Token(ev) => {
                        if let Some(reason) = on_token(ev) {
                            return Ok(reason);
                        }
                    }
                    StepResult::Prefill { .. } => {}
                    StepResult::Finished(reason) => return Ok(reason),
                }
            }
        }
        loop {
            match self.next_fair(handle)? {
                None => {
                    return Ok(FinishReason::Stop);
                }
                Some(ev) => {
                    if let Some(reason) = on_token(ev) {
                        return Ok(reason);
                    }
                }
            }
        }
    }

    /// One fair tick for `handle`: drain parked results, else `poll` (parking peers).
    fn next_fair(&self, handle: &mut RequestHandle) -> Result<Option<TokenEvent>, AllpacaError> {
        let id = handle.id;
        loop {
            if let Some(parked) = self.take_parked(id)? {
                match parked {
                    StepResult::Token(ev) => return Ok(Some(ev)),
                    StepResult::Prefill { .. } => continue,
                    StepResult::Finished(_) => return Ok(None),
                }
            }
            match self.poll()? {
                None => {
                    // Idle: if we are finished, stop; else retry (peer may unblock).
                    let finished = {
                        let inner = self.lock()?;
                        inner
                            .active
                            .get(&id)
                            .map(|r| r.finished && r.emit_queue.is_empty() && r.parked.is_empty())
                            .unwrap_or(true)
                    };
                    if finished {
                        return Ok(None);
                    }
                    std::thread::yield_now();
                    continue;
                }
                Some(ev) if ev.request_id == id => match ev.result {
                    StepResult::Token(tok) => return Ok(Some(tok)),
                    StepResult::Prefill { .. } => continue,
                    StepResult::Finished(_) => return Ok(None),
                },
                Some(ev) => {
                    self.park_result(ev.request_id, ev.result)?;
                }
            }
        }
    }

    fn take_parked(&self, id: u64) -> Result<Option<StepResult>, AllpacaError> {
        let mut inner = self.lock()?;
        Ok(inner
            .active
            .get_mut(&id)
            .and_then(|r| r.parked.pop_front()))
    }

    fn park_result(&self, id: u64, result: StepResult) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        if let Some(active) = inner.active.get_mut(&id) {
            active.parked.push_back(result);
        }
        Ok(())
    }

    pub fn begin_request(
        &self,
        prompt: &str,
        mut params: GenerationParams,
    ) -> Result<RequestHandle, AllpacaError> {
        if params.max_tokens == 0 {
            params.max_tokens = 1;
        }
        let mut inner = self.lock()?;
        if inner.active.len() >= self.config.engine.max_active_requests {
            return Err(AllpacaError::Transient(TransientError::RequestBusy));
        }
        let tokenizer = inner
            .tokenizer
            .as_ref()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let mut tokens = tokenizer.encode(prompt)?;
        if tokens.is_empty() {
            if let Some(bos) = tokenizer.bos_id {
                tokens.push(bos);
            } else {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(
                    "empty prompt encoding".into(),
                )));
            }
        }
        if let Some(eos) = tokenizer.eos_id {
            if !params.stop_token_ids.contains(&eos) {
                params.stop_token_ids.push(eos);
            }
        }
        let vocab = tokenizer.vocab_size();
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let page_tokens = self.config.kv.page_tokens;
        let pages = alloc_pages_for_seq(
            &mut inner.kv,
            &target.config,
            page_tokens,
            tokens.len() + params.max_tokens,
            &self.metrics,
        )?;
        let mut recurrent = if target.config.recurrent {
            Some(RecurrentState::zeros(
                target.config.n_layers,
                target.config.dim,
            ))
        } else {
            None
        };
        let mut prefill_pos = 0;
        if let Some((state, rows, n)) = inner.anchors.lookup(&tokens) {
            restore_kv(&mut inner.kv, &pages, &rows, &target.config)?;
            if let Some(dst) = recurrent.as_mut() {
                *dst = state;
            }
            prefill_pos = n;
        }
        if let Some((n, cp)) = inner
            .kv_radix
            .lookup_checkpoint_capped(&tokens, tokens.len().saturating_sub(1))
        {
            if n > prefill_pos {
                restore_kv(&mut inner.kv, &pages, &cp.kv, &target.config)?;
                if let Some(dst) = recurrent.as_mut() {
                    *dst = cp.recurrent;
                } else if target.config.recurrent {
                    recurrent = Some(cp.recurrent);
                }
                prefill_pos = n;
            }
        }
        let rng = rng_from_params(&params, self.config.engine.determinism_seed);
        let grammar = if self.config.structured_output.enabled
            && inner.flags.json_constraints
            && (params.json_schema.is_some() || params.grammar.is_some())
        {
            if inner.grammar_factory.is_none() {
                let factory = {
                    let tok = inner
                        .tokenizer
                        .as_ref()
                        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
                    crate::grammar::factory_from_tokenizer(tok)?
                };
                inner.grammar_factory = Some(factory);
            }
            Some(crate::grammar::matcher_from_factory(
                inner.grammar_factory.as_ref().unwrap(),
                params.json_schema.as_deref(),
                params.grammar.as_deref(),
            )?)
        } else {
            None
        };
        let id = inner.next_request_id;
        inner.next_request_id += 1;
        inner.draft_ahead = None;
        inner.draft_kv.reset();
        let prompt_len = tokens.len();
        inner.active.insert(
            id,
            ActiveRequest {
                id,
                prompt_len,
                prefill_pos,
                tokens,
                generated: 0,
                params,
                pages,
                pending_logits: vec![0.0f32; vocab],
                emit_queue: Vec::new(),
                parked: VecDeque::new(),
                async_mask_rx: None,
                json_prefix: String::new(),
                grammar,
                rng,
                started: Instant::now(),
                last_token_at: Instant::now(),
                first_emitted: false,
                finished: false,
                recurrent,
            },
        );
        if self.config.structured_output.async_masks {
            if let Some(active) = inner.active.get_mut(&id) {
                if active.grammar.is_some() {
                    kick_async_mask(active, vocab);
                }
            }
        }
        self.metrics
            .requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics.active_requests.store(
            inner.active.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.metrics.prompt_tokens.fetch_add(
            prompt_len as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // One chunk only — remaining prompt tokens continue in `step` so a
        // second Request can admit / decode between chunks.
        self.prefill_chunk(&mut inner, id)?;
        Ok(RequestHandle {
            id,
            inner: Arc::clone(&self.inner),
            metrics: Arc::clone(&self.metrics),
        })
    }

    pub fn step(&self, handle: &mut RequestHandle) -> Result<StepResult, AllpacaError> {
        self.step_id(handle.id)
    }

    /// Advance the longest-waiting runnable Request. `None` when nothing is runnable
    /// (idle, or only finished Requests remain until their handles are dropped).
    ///
    /// Under concurrency (`≥2` runnable), also advances a stagger partner in the
    /// same tick. The partner's `StepResult` is **parked** so it still surfaces as
    /// a later `PollEvent` (prior silent dual-step broke interleave observability).
    pub fn poll(&self) -> Result<Option<allpaca_types::PollEvent>, AllpacaError> {
        loop {
            let (id, waited_ms, is_prefill, stagger_partner) = {
                let mut inner = self.lock()?;
                let schedule = self.config.engine.poll_schedule;
                let max_pref = self.config.engine.max_consecutive_prefill_picks;
                let streak = inner.consecutive_prefill_picks;
                let candidates: Vec<(u64, Instant, bool)> = inner
                    .active
                    .iter()
                    .filter(|(_, r)| !(r.finished && r.emit_queue.is_empty() && r.parked.is_empty()))
                    .map(|(id, r)| {
                        let prefill = r.prefill_pos < r.prompt_len
                            && r.emit_queue.is_empty()
                            && r.parked.is_empty();
                        (*id, r.last_token_at, prefill)
                    })
                    .collect();
                let mut stagger_partner: Option<u64> = None;
                if candidates.len() >= 2 {
                    let hints: Vec<ActiveLayerHint> = candidates
                        .iter()
                        .enumerate()
                        .map(|(i, (rid, _, pref))| ActiveLayerHint {
                            request_id: *rid,
                            phase: if *pref {
                                StaggerPhase::Prefill
                            } else {
                                StaggerPhase::Decode
                            },
                            layer: 0,
                            cpu_expert_pending: i > 0
                                || inner
                                    .target
                                    .as_ref()
                                    .and_then(|m| m.expert_lru.as_ref())
                                    .is_some_and(|lru| lru.cpu_runs() > 0),
                        })
                        .collect();
                    let plan = plan_micro_batch(&hints);
                    if plan.gpu.is_some() && plan.cpu.is_some() {
                        self.metrics
                            .micro_batch_plans
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if let Some(c) = plan.cpu {
                            stagger_partner = Some(c.request_id);
                        }
                    }
                }
                // Soft affinity from a prior failed same-tick partner step.
                let preferred = inner
                    .prefer_next
                    .take()
                    .filter(|pid| candidates.iter().any(|(cid, _, _)| cid == pid));
                let id = preferred.or_else(|| {
                    inner.scheduler.pick_runnable(
                        &candidates,
                        inner.last_poll_id,
                        schedule,
                        streak,
                        max_pref,
                    )
                });
                let Some(id) = id else {
                    return Ok(None);
                };
                if let Some(partner) = stagger_partner {
                    if partner == id {
                        stagger_partner = candidates
                            .iter()
                            .map(|(cid, _, _)| *cid)
                            .find(|cid| *cid != id);
                    }
                }
                let active = inner.active.get(&id).unwrap();
                let waited_ms = active.last_token_at.elapsed().as_secs_f64() * 1000.0;
                let is_prefill = active.prefill_pos < active.prompt_len
                    && active.emit_queue.is_empty()
                    && active.parked.is_empty();
                (id, waited_ms, is_prefill, stagger_partner)
            };
            {
                let mut inner = self.lock()?;
                inner.last_poll_id = Some(id);
                if is_prefill {
                    inner.consecutive_prefill_picks =
                        inner.consecutive_prefill_picks.saturating_add(1);
                } else {
                    inner.consecutive_prefill_picks = 0;
                }
            }
            self.metrics
                .poll_ticks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if is_prefill {
                self.metrics
                    .poll_prefill_picks
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                self.metrics
                    .poll_decode_picks
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let prev = self
                .metrics
                .poll_wait_ms_ema_x100
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                / 100.0;
            let ema = if prev <= 0.0 {
                waited_ms
            } else {
                prev * 0.9 + waited_ms * 0.1
            };
            self.metrics.poll_wait_ms_ema_x100.store(
                (ema * 100.0) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let dual_partner = stagger_partner.filter(|&p| p != id);
            if dual_partner.is_some() {
                self.micro_batch.begin_dual_step(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0),
                );
            }
            let primary = match self.step_id(id) {
                Ok(result) => result,
                // Peer Drop removed the Request between pick and step.
                Err(AllpacaError::Transient(TransientError::RequestBusy)) => {
                    if dual_partner.is_some() {
                        self.micro_batch.end_dual_step();
                    }
                    continue;
                }
                Err(e) => {
                    if dual_partner.is_some() {
                        self.micro_batch.end_dual_step();
                    }
                    return Err(e);
                }
            };
            // Same-tick partner: co-progress then park so the result remains a
            // PollEvent when that request is next picked (observability intact).
            if let Some(partner) = dual_partner {
                match self.step_id(partner) {
                    Ok(partner_result) => {
                        self.park_result(partner, partner_result)?;
                        self.metrics
                            .micro_batch_dual_steps
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Err(AllpacaError::Transient(TransientError::RequestBusy)) => {
                        let mut inner = self.lock()?;
                        inner.prefer_next = Some(partner);
                    }
                    Err(e) => {
                        self.micro_batch.end_dual_step();
                        return Err(e);
                    }
                }
                self.micro_batch.end_dual_step();
            }
            return Ok(Some(allpaca_types::PollEvent {
                request_id: id,
                result: primary,
            }));
        }
    }

    fn step_id(&self, id: u64) -> Result<StepResult, AllpacaError> {
        let mut inner = self.lock()?;
        {
            let active = request_mut(&mut inner, id)?;
            // Drain same-tick micro-batch parks before fresh work.
            if let Some(parked) = active.parked.pop_front() {
                return Ok(parked);
            }
            if !active.emit_queue.is_empty() {
                return Ok(StepResult::Token(active.emit_queue.remove(0)));
            }
            if active.finished {
                return Ok(StepResult::Finished(FinishReason::Stop));
            }
            if active.generated >= active.params.max_tokens {
                active.finished = true;
                return Ok(StepResult::Finished(FinishReason::Length));
            }
        }
        if request(&inner, id)?.prefill_pos < request(&inner, id)?.prompt_len {
            let (done, total) = self.prefill_chunk(&mut inner, id)?;
            if done < total {
                if let Some(active) = inner.active.get_mut(&id) {
                    active.last_token_at = Instant::now();
                }
                return Ok(StepResult::Prefill {
                    tokens_done: done,
                    tokens_total: total,
                });
            }
        }

        let json_constraints = inner.flags.json_constraints;
        let structured = json_constraints && self.config.structured_output.enabled;
        let speculate = self.config.speculative.enabled
            && (inner.flags.ngram_speculation || inner.flags.draft_model_speculation);
        let (json_mode, json_prefix, temperature, remaining, depth, has_grammar) = {
            let active = request(&inner, id)?;
            let remaining = active.params.max_tokens.saturating_sub(active.generated);
            let depth = inner.scheduler.draft_depth.min(remaining.saturating_sub(1));
            (
                active.params.json_mode && structured && active.grammar.is_none(),
                active.json_prefix.clone(),
                active.params.temperature,
                remaining,
                depth,
                active.grammar.is_some(),
            )
        };

        let mask = if has_grammar {
            grammar_mask(&mut inner, id, &self.metrics)?
        } else if json_mode {
            let vocab_tokens = inner
                .tokenizer
                .as_ref()
                .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
                .tokens
                .clone();
            let constraint = JsonConstraint::new(vocab_tokens);
            let mut m = constraint.mask(&json_prefix);
            if !m.any() {
                tracing::warn!("JSON constraint mask empty; allowing all tokens (degraded)");
                self.metrics
                    .degraded_events
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                m = allpaca_types::TokenMask::allow_all(inner.tokenizer.as_ref().unwrap().vocab_size());
            }
            Some(m)
        } else {
            None
        };

        let use_tree = speculate && inner.flags.tree_speculation && depth > 0;
        self.metrics
            .target_evals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let micro_batch = if self.micro_batch.dual_active() {
            Some(self.micro_batch.as_ref())
        } else {
            None
        };
        let produced = if use_tree {
            let ctx = request(&inner, id)?.tokens.clone();
            let max_nodes = self.config.speculative.tree.max_nodes.max(1);
            let draft = if inner.flags.draft_model_speculation {
                inner.draft_model.as_deref()
            } else {
                None
            };
            let forest = build_token_forest(
                &inner.ngram,
                draft,
                &ctx,
                max_nodes,
                depth.max(1),
            )?;
            self.metrics
                .tree_nodes
                .fetch_add(forest.nodes.len() as u64, std::sync::atomic::Ordering::Relaxed);
            verify_forest(&mut inner, id, &forest, mask.as_ref(), remaining, &self.metrics, self.hip.as_deref(), micro_batch)?
        } else {
            let drafts = if speculate && depth > 0 {
                propose_drafts(
                    &mut inner,
                    id,
                    depth,
                    temperature,
                    self.draft_hip.as_deref(),
                    &self.metrics,
                )?
            } else {
                Vec::new()
            };
            verify_and_sample(
                    &mut inner,
                    id,
                    drafts,
                    mask.as_ref(),
                    remaining,
                    &self.metrics,
                    self.hip.as_deref(),
                    self.draft_hip.as_deref(),
                    depth,
                    temperature,
                    micro_batch,
                )?
        };
        let accepted = produced.accepted;
        let proposed = produced.proposed;
        let active_n = inner.active.len();
        inner
            .scheduler
            .record_acceptance_with_pressure(accepted, proposed, active_n);
        self.metrics
            .draft_proposed
            .fetch_add(proposed, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .draft_accepted
            .fetch_add(accepted, std::sync::atomic::Ordering::Relaxed);

        let last_idx = produced.tokens.len().saturating_sub(1);
        for (i, tok) in produced.tokens.into_iter().enumerate() {
            let piece = inner
                .tokenizer
                .as_ref()
                .map(|t| t.decode_one(tok))
                .unwrap_or_default();
            let (stop, maxed, first) = {
                let active = request(&inner, id)?;
                (
                    is_stop(tok, &active.params),
                    active.generated + 1 >= active.params.max_tokens && i == last_idx,
                    !active.first_emitted,
                )
            };
            let finish = if stop {
                Some(FinishReason::Stop)
            } else if maxed {
                Some(FinishReason::Length)
            } else {
                None
            };
            let now = Instant::now();
            let vocab = inner
                .tokenizer
                .as_ref()
                .map(|t| t.vocab_size())
                .unwrap_or(0);
            let async_masks = self.config.structured_output.async_masks;
            if let Some(active) = inner.active.get_mut(&id) {
                if let Some(ref mut g) = active.grammar {
                    if !g.is_stopped() {
                        if let Err(e) = g.consume_token(tok) {
                            tracing::warn!(token = tok, error = %e, "grammar consume_token failed");
                            self.metrics
                                .degraded_events
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        } else if async_masks {
                            kick_async_mask(active, vocab);
                        }
                    }
                }
                if !is_special(&piece) {
                    active.json_prefix.push_str(&piece);
                }
                if first {
                    let ttft = now.duration_since(active.started).as_millis() as u64;
                    self.metrics
                        .ttft_ms_last
                        .store(ttft, std::sync::atomic::Ordering::Relaxed);
                    active.first_emitted = true;
                } else {
                    let itl = now.duration_since(active.last_token_at).as_secs_f64() * 1_000.0;
                    self.metrics.record_itl_ms(itl);
                }
                active.last_token_at = now;
                active.generated += 1;
                if finish.is_some() {
                    active.finished = true;
                }
                active.emit_queue.push(TokenEvent {
                    token_id: tok,
                    text: piece,
                    finish_reason: finish,
                });
            }
            self.metrics
                .tokens_generated
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        let ev = inner
            .active
            .get_mut(&id)
            .and_then(|a| {
                if a.emit_queue.is_empty() {
                    None
                } else {
                    Some(a.emit_queue.remove(0))
                }
            })
            .ok_or_else(|| AllpacaError::Fatal(FatalError::InvalidConfig("empty step".into())))?;
        Ok(StepResult::Token(ev))
    }

    fn prefill_chunk(
        &self,
        inner: &mut EngineInner,
        id: u64,
    ) -> Result<(usize, usize), AllpacaError> {
        let chunk = self.config.prefill.chunk_tokens.max(1);
        let (start, total, toks, pages) = {
            let active = request_mut(inner, id)?;
            let start = active.prefill_pos;
            let total = active.prompt_len;
            if start >= total {
                return Ok((start, total));
            }
            let end = (start + chunk).min(total);
            (
                start,
                total,
                active.tokens[start..end].to_vec(),
                active.pages.clone(),
            )
        };
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let resident = inner.target_resident.clone();
        let mut logits = {
            let active = request_mut(inner, id)?;
            std::mem::take(&mut active.pending_logits)
        };
        let end = start + toks.len();
        let write_logits = end >= total;
        let mut recurrent = {
            let active = request_mut(inner, id)?;
            std::mem::take(&mut active.recurrent)
        };
        // Per-token prefill so radix (and anchors) get correct mid-prefix checkpoints.
        for (i, tok) in toks.iter().enumerate() {
            let pos = start + i;
            let last = pos + 1 >= total;
            target.forward_token(
                *tok,
                pos,
                &pages,
                &mut inner.kv,
                &self.metrics,
                &mut logits,
                self.hip.as_deref(),
                resident.as_deref(),
                last || write_logits,
                recurrent.as_mut(),
                None,
            )?;
            let prefix = request(inner, id)?.tokens[..=pos].to_vec();
            let pages_now = request(inner, id)?.pages.clone();
            let state_ref = recurrent
                .clone()
                .unwrap_or_else(|| RecurrentState::zeros(target.config.n_layers, target.config.dim));
            inner.kv_radix.capture(
                &prefix,
                &state_ref,
                &inner.kv,
                &pages_now,
                &target.config,
            )?;
            if inner.anchors.enabled() && inner.anchors.is_anchor(*tok) {
                if let Some(state) = recurrent.as_ref() {
                    inner.anchors.capture(
                        &prefix,
                        state,
                        &inner.kv,
                        &pages_now,
                        &target.config,
                    )?;
                }
            }
        }
        if let Some(hip) = self.hip.as_deref() {
            hip.wait_logits_d2h(&mut logits)?;
        }
        {
            let active = request_mut(inner, id)?;
            active.prefill_pos = end;
            active.pending_logits = logits;
            active.recurrent = recurrent;
        }
        self.metrics
            .prefill_chunks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if end >= total && inner.flags.ngram_speculation {
            let tokens = request(inner, id)?.tokens.clone();
            inner.ngram.observe(&tokens);
        }
        Ok((end, total))
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, EngineInner>, AllpacaError> {
        self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
        })
    }

    /// Rebuild expert LRU slots vs KV VRAM at a scheduler safe point.
    /// Host (or NVMe) pool is not reloaded. GPU residency is a speed dial.
    pub fn rebalance_vram(
        &self,
        cache_bytes: usize,
        vram_bytes: usize,
    ) -> Result<(), AllpacaError> {
        {
            let mut inner = self.lock()?;
            if let Some(model) = inner.target.as_mut() {
                let model = Arc::get_mut(model).ok_or_else(|| {
                    AllpacaError::Transient(TransientError::RequestBusy)
                })?;
                self.rebuild_expert_slots(model, cache_bytes);
            }
            inner.kv.set_vram_bytes(vram_bytes, &self.metrics)?;
        }
        if self.hip.is_some() {
            self.upload_resident_weights()?;
        }
        Ok(())
    }

    fn rebuild_expert_slots(&self, model: &mut TinyModel, cache_bytes: usize) {
        if model.config.n_experts == 0 {
            return;
        }
        let expert_bytes = model.expert_slot_bytes();
        let n_slots = expert_lru_slots(expert_bytes, cache_bytes);
        ExpertLru::record_slots(&self.metrics, n_slots);
        if n_slots == 0 {
            model.expert_lru = None;
            return;
        }
        let b_pcie = self
            .metrics
            .expert_pcie_bps
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let b_host = self
            .metrics
            .expert_host_bps
            .load(std::sync::atomic::Ordering::Relaxed) as f64;
        let (t_dma, t_dispatch) = crate::model::measure_qstar_fixed_costs(expert_bytes);
        model.expert_lru = Some(ExpertLru::with_overhead(
            n_slots,
            self.config.weights.miss_policy,
            b_pcie,
            b_host,
            t_dma,
            t_dispatch,
            expert_bytes as f64,
        ));
    }
}

fn open_hip_workspace(config: &Config, inventory: &DeviceInventory) -> Option<HipWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_hip::runtime_linked() {
        return None;
    }
    let gpus: Vec<_> = inventory.devices.iter().filter(|d| d.is_gpu()).collect();
    if gpus.is_empty() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "hip" {
        return None;
    }
    let device = if inventory.target_device >= 0 {
        inventory.target_device
    } else {
        gpus[0].id
    };
    match HipWorkspace::new(device, 8 * 1024 * 1024, 256) {
        Ok(ws) => {
            tracing::info!(device, bytes = ws.allocated_bytes(), "HIP workspace allocated");
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "HIP workspace failed; CPU fallback");
            None
        }
    }
}

/// Resident draft workspace on the slow GPU. Absent on 1 GPU / CPU (degraded).
fn open_draft_hip_workspace(config: &Config, inventory: &DeviceInventory) -> Option<HipWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_hip::runtime_linked() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "hip" {
        return None;
    }
    let draft = inventory.draft_device;
    let target = inventory.target_device;
    if draft < 0 || draft == target {
        return None;
    }
    match HipWorkspace::new(draft, 8 * 1024 * 1024, 256) {
        Ok(ws) => {
            tracing::info!(
                device = draft,
                bytes = ws.allocated_bytes(),
                "HIP draft workspace allocated (resident; P2P not required)"
            );
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "HIP draft workspace failed; CPU draft fallback");
            None
        }
    }
}

fn apply_provider_flags(flags: &mut FeatureFlags, config: &Config) {
    if config.features.tree_speculation.is_none() {
        flags.tree_speculation = config.speculative.enabled && config.speculative.tree.enabled;
    }
    if !config.speculative.enabled {
        flags.ngram_speculation = false;
        flags.draft_model_speculation = false;
        flags.tree_speculation = false;
        return;
    }
    let providers = &config.speculative.providers;
    if !providers.contains(&DraftProviderKind::Ngram) {
        flags.ngram_speculation = false;
    }
    if !providers.contains(&DraftProviderKind::DraftModel) {
        flags.draft_model_speculation = false;
    }
}

fn request(inner: &EngineInner, id: u64) -> Result<&ActiveRequest, AllpacaError> {
    inner
        .active
        .get(&id)
        .ok_or(AllpacaError::Transient(TransientError::RequestBusy))
}

fn request_mut(inner: &mut EngineInner, id: u64) -> Result<&mut ActiveRequest, AllpacaError> {
    inner
        .active
        .get_mut(&id)
        .ok_or(AllpacaError::Transient(TransientError::RequestBusy))
}

fn set_tokenizer(inner: &mut EngineInner, tokenizer: EngineTokenizer) {
    inner.tokenizer = Some(tokenizer);
    inner.grammar_factory = None;
}

fn grammar_mask(
    inner: &mut EngineInner,
    id: u64,
    metrics: &Metrics,
) -> Result<Option<allpaca_types::TokenMask>, AllpacaError> {
    let vocab = inner
        .tokenizer
        .as_ref()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?
        .vocab_size();
    let active = request_mut(inner, id)?;
    if let Some(rx) = active.async_mask_rx.take() {
        match rx.recv() {
            Ok(Ok(mut mask)) => {
                metrics
                    .async_mask_hits
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !mask.any() {
                    tracing::warn!("async grammar mask empty; allowing all tokens (degraded)");
                    metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    mask = allpaca_types::TokenMask::allow_all(vocab);
                }
                return Ok(Some(mask));
            }
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "async grammar mask failed; falling back to sync");
            }
            Err(_) => {
                tracing::warn!("async grammar mask worker dropped; falling back to sync");
            }
        }
    }
    let Some(matcher) = active.grammar.as_mut() else {
        return Ok(None);
    };
    match matcher.compute_mask_or_eos() {
        Ok(vob) => {
            let mut mask = crate::grammar::vob_to_mask(&vob, vocab);
            if !mask.any() {
                tracing::warn!("grammar constraint mask empty; allowing all tokens (degraded)");
                metrics
                    .degraded_events
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                mask = allpaca_types::TokenMask::allow_all(vocab);
            }
            Ok(Some(mask))
        }
        Err(e) => {
            tracing::warn!(error = %e, "grammar mask failed; allowing all tokens (degraded)");
            metrics
                .degraded_events
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(Some(allpaca_types::TokenMask::allow_all(vocab)))
        }
    }
}

fn kick_async_mask(active: &mut ActiveRequest, vocab: usize) {
    let Some(matcher) = active.grammar.as_ref() else {
        return;
    };
    if matcher.is_stopped() {
        return;
    }
    let mut matcher = matcher.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    active.async_mask_rx = Some(rx);
    std::thread::spawn(move || {
        let result = match matcher.compute_mask_or_eos() {
            Ok(vob) => Ok(crate::grammar::vob_to_mask(&vob, vocab)),
            Err(e) => Err(e.to_string()),
        };
        let _ = tx.send(result);
    });
}

fn propose_drafts(
    inner: &mut EngineInner,
    id: u64,
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<Vec<u32>, AllpacaError> {
    let ctx = request(inner, id)?.tokens.clone();
    if let Some(ahead) = inner.draft_ahead.take() {
        if ahead.ctx == ctx && !ahead.tokens.is_empty() {
            metrics
                .draft_ahead_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            host_stage_draft_tokens(&ahead.tokens, draft_hip, metrics);
            return Ok(ahead.tokens);
        }
    }
    if inner.flags.draft_model_speculation {
        if let Some(model) = inner.draft_model.clone() {
            let mut provider = DraftModelProvider::new((*model).clone());
            let resident = inner.draft_resident.clone();
            let draft = provider.propose_on(
                &ctx,
                depth,
                temperature,
                &mut inner.draft_kv,
                draft_hip,
                resident.as_deref(),
                Some(metrics),
            )?;
            if !draft.is_empty() {
                host_stage_draft_tokens(&draft, draft_hip, metrics);
                return Ok(draft);
            }
        }
    }
    if inner.flags.ngram_speculation {
        return inner.ngram.propose(&ctx, depth, temperature);
    }
    Ok(Vec::new())
}

/// Tokens (and only tokens) cross the device boundary via host. Weights stay put.
fn host_stage_draft_tokens(tokens: &[u32], draft_hip: Option<&HipWorkspace>, metrics: &Metrics) {
    if tokens.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(tokens) as u64;
    metrics
        .draft_tokens_copied
        .fetch_add(tokens.len() as u64, std::sync::atomic::Ordering::Relaxed);
    metrics
        .host_staging_bytes
        .fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    if let Some(hip) = draft_hip {
        let _ = hip.bind();
    }
}

struct Produced {
    tokens: Vec<u32>,
    accepted: u64,
    proposed: u64,
}

fn verify_and_sample(
    inner: &mut EngineInner,
    id: u64,
    drafts: Vec<u32>,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    draft_hip: Option<&HipWorkspace>,
    depth: usize,
    temperature: f32,
    micro_batch: Option<&MicroBatchExpertUnion>,
) -> Result<Produced, AllpacaError> {
    let drafts_len = drafts.len() as u64;
    if remaining == 0 {
        return Ok(Produced {
            tokens: Vec::new(),
            accepted: 0,
            proposed: drafts_len,
        });
    }
    if let Some(hip) = hip {
        let active = request_mut(inner, id)?;
        hip.wait_logits_d2h(&mut active.pending_logits)?;
    }
    let sampled = {
        let active = request_mut(inner, id)?;
        sample(
            &mut active.pending_logits,
            mask,
            &active.params,
            &mut active.rng,
        )
    };
    let first_match = drafts.first().copied() == Some(sampled);
    let stop = is_stop(sampled, &request(inner, id)?.params);
    if first_match && drafts.len() >= 2 && !stop && remaining > 1 {
        return packed_linear_verify(
            inner,
            id,
            drafts,
            sampled,
            mask,
            remaining,
            metrics,
            hip,
            draft_hip,
            depth,
            temperature,
        );
    }
    let mut produced = vec![sampled];
    let mut accepted = u64::from(first_match);
    let mut draft_iter = drafts.into_iter().peekable();
    if first_match {
        draft_iter.next();
    }
    append_and_forward(inner, id, sampled, metrics, hip, micro_batch)?;
    if stop || !first_match || draft_iter.peek().is_none() {
        return finish_linear_verify(
            inner,
            id,
            Produced {
                tokens: produced,
                accepted,
                proposed: drafts_len,
            },
            depth,
            temperature,
            draft_hip,
            metrics,
        );
    }
    while produced.len() < remaining {
        if let Some(hip) = hip {
            let active = request_mut(inner, id)?;
            hip.wait_logits_d2h(&mut active.pending_logits)?;
        }
        let sampled = {
            let active = request_mut(inner, id)?;
            sample(
                &mut active.pending_logits,
                mask,
                &active.params,
                &mut active.rng,
            )
        };
        produced.push(sampled);
        let matched = draft_iter.peek().copied() == Some(sampled);
        if matched {
            accepted += 1;
            draft_iter.next();
        }
        append_and_forward(inner, id, sampled, metrics, hip, micro_batch)?;
        let stop = is_stop(sampled, &request(inner, id)?.params);
        if stop || !matched || draft_iter.peek().is_none() {
            break;
        }
    }
    finish_linear_verify(
        inner,
        id,
        Produced {
            tokens: produced,
            accepted,
            proposed: drafts_len,
        },
        depth,
        temperature,
        draft_hip,
        metrics,
    )
}

fn finish_linear_verify(
    inner: &mut EngineInner,
    id: u64,
    produced: Produced,
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<Produced, AllpacaError> {
    if !produced.tokens.is_empty() {
        let ctx = request(inner, id)?.tokens.clone();
        store_draft_ahead_from_ctx(inner, &ctx, depth, temperature, draft_hip, metrics)?;
    }
    Ok(produced)
}

fn packed_linear_verify(
    inner: &mut EngineInner,
    id: u64,
    span: Vec<u32>,
    first: u32,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    draft_hip: Option<&HipWorkspace>,
    depth: usize,
    temperature: f32,
) -> Result<Produced, AllpacaError> {
    let span_len = span.len();
    let next_ctx = {
        let mut ctx = request(inner, id)?.tokens.clone();
        ctx.extend_from_slice(&span);
        ctx
    };
    let can_ahead = depth > 0
        && inner.flags.draft_model_speculation
        && inner.draft_model.is_some();
    if !can_ahead {
        return packed_linear_verify_span(inner, id, span, first, mask, remaining, metrics, hip);
    }
    let mut draft_kv = std::mem::take(&mut inner.draft_kv);
    let model = inner.draft_model.clone().expect("draft-ahead needs a Draft Model");
    let resident = inner.draft_resident.clone();
    let next_ctx_worker = next_ctx.clone();
    std::thread::scope(|s| {
        let worker = s.spawn(|| {
            if let Some(h) = draft_hip {
                let _ = h.bind();
            }
            let mut provider = DraftModelProvider::new((*model).clone());
            let tokens = provider.propose_on(
                &next_ctx_worker,
                depth,
                temperature,
                &mut draft_kv,
                draft_hip,
                resident.as_deref(),
                Some(metrics),
            );
            (draft_kv, tokens)
        });
        let produced = packed_linear_verify_span(
            inner, id, span, first, mask, remaining, metrics, hip,
        );
        let (kv, ahead) = worker.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
        inner.draft_kv = kv;
        let produced = produced?;
        let ahead = ahead?;
        let actual_ctx = request(inner, id)?.tokens.clone();
        let full_accept = produced.accepted == produced.proposed
            && produced.proposed == span_len as u64
            && actual_ctx == next_ctx;
        if full_accept && !ahead.is_empty() {
            inner.draft_ahead = Some(DraftAhead {
                ctx: next_ctx,
                tokens: ahead,
            });
        } else {
            store_draft_ahead_from_ctx(
                inner,
                &actual_ctx,
                depth,
                temperature,
                draft_hip,
                metrics,
            )?;
        }
        Ok(produced)
    })
}

fn store_draft_ahead_from_ctx(
    inner: &mut EngineInner,
    ctx: &[u32],
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    if depth == 0 || ctx.is_empty() {
        return Ok(());
    }
    let Some(model) = inner.draft_model.clone() else {
        return Ok(());
    };
    let mut provider = DraftModelProvider::new((*model).clone());
    let resident = inner.draft_resident.clone();
    let tokens = provider.propose_on(
        ctx,
        depth,
        temperature,
        &mut inner.draft_kv,
        draft_hip,
        resident.as_deref(),
        Some(metrics),
    )?;
    if !tokens.is_empty() {
        inner.draft_ahead = Some(DraftAhead {
            ctx: ctx.to_vec(),
            tokens,
        });
    }
    Ok(())
}

fn packed_linear_verify_span(
    inner: &mut EngineInner,
    id: u64,
    span: Vec<u32>,
    first: u32,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
) -> Result<Produced, AllpacaError> {
    let drafts_len = span.len() as u64;
    let vocab = request(inner, id)?.pending_logits.len();
    let start_pos = request(inner, id)?.tokens.len();
    let page_tokens = request(inner, id)?
        .pages
        .first()
        .and_then(|layer| layer.first())
        .and_then(|page_id| inner.kv.get(*page_id))
        .map(|p| p.capacity_tokens)
        .unwrap_or(128);
    let last_pos = start_pos + span.len() - 1;
    let target = inner
        .target
        .clone()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    let mut pages = request(inner, id)?.pages.clone();
    ensure_pages(
        &mut inner.kv,
        &target.config,
        page_tokens,
        &mut pages,
        last_pos,
        metrics,
    )?;
    request_mut(inner, id)?.pages = pages;
    let mut last_logits = std::mem::take(&mut request_mut(inner, id)?.pending_logits);
    let mut row_logits = vec![0.0f32; span.len() * vocab];
    let pages = request(inner, id)?.pages.clone();
    let resident = inner.target_resident.clone();
    let mut recurrent = std::mem::take(&mut request_mut(inner, id)?.recurrent);
    target.forward_tokens(
        &span,
        start_pos,
        &pages,
        &mut inner.kv,
        metrics,
        &mut last_logits,
        hip,
        resident.as_deref(),
        true,
        Some(&mut row_logits),
        recurrent.as_mut(),
    )?;
    if let Some(hip) = hip {
        hip.wait_logits_d2h(&mut row_logits)?;
    }
    request_mut(inner, id)?.pending_logits = last_logits;
    request_mut(inner, id)?.recurrent = recurrent;
    metrics
        .linear_packed_verifies
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let mut produced = vec![first];
    let mut accepted = 1u64;
    let mut keep = 1usize;
    let params_stop = request(inner, id)?.params.clone();
    for i in 0..span.len().saturating_sub(1) {
        if produced.len() >= remaining {
            break;
        }
        if is_stop(produced[produced.len() - 1], &params_stop) {
            break;
        }
        let off = i * vocab;
        let row = &mut row_logits[off..off + vocab];
        let next = {
            let active = request_mut(inner, id)?;
            sample(row, mask, &active.params, &mut active.rng)
        };
        produced.push(next);
        if next == span[i + 1] {
            accepted += 1;
            keep = i + 2;
        } else {
            break;
        }
    }

    let pages = request(inner, id)?.pages.clone();
    inner
        .kv
        .rewind_filled(&pages, start_pos + keep, page_tokens as usize);
    {
        let active = request_mut(inner, id)?;
        active.tokens.extend_from_slice(&span[..keep]);
        active.pages = pages;
    }
    if inner.flags.ngram_speculation {
        let ctx = request(inner, id)?.tokens.clone();
        inner.ngram.observe(&ctx);
    }

    if produced.len() > keep {
        append_and_forward(inner, id, produced[keep], metrics, hip, None)?;
    } else {
        let off = keep.saturating_sub(1) * vocab;
        let pending = row_logits[off..off + vocab].to_vec();
        request_mut(inner, id)?.pending_logits = pending;
    }
    Ok(Produced {
        tokens: produced,
        accepted,
        proposed: drafts_len,
    })
}

fn verify_forest(
    inner: &mut EngineInner,
    id: u64,
    forest: &TokenForest,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    micro_batch: Option<&MicroBatchExpertUnion>,
) -> Result<Produced, AllpacaError> {
    let vocab = inner.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(1);
    let mut node_logits = vec![vec![0.0f32; vocab]; forest.nodes.len()];
    if !forest.nodes.is_empty() {
        let pages = request(inner, id)?.pages.clone();
        let prompt_len = request(inner, id)?.tokens.len();
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        target.forward_forest(
            forest,
            &pages,
            &inner.kv,
            prompt_len,
            &mut node_logits,
            hip,
            Some(metrics),
        )?;
    }

    let mut produced = Vec::new();
    let mut accepted_tokens = Vec::new();
    let mut parent = -1i32;
    while produced.len() < remaining {
        if let Some(hip) = hip {
            let active = request_mut(inner, id)?;
            hip.wait_logits_d2h(&mut active.pending_logits)?;
        }
        if parent >= 0 {
            if let Some(nl) = node_logits.get(parent as usize) {
                let active = request_mut(inner, id)?;
                if nl.len() == active.pending_logits.len() {
                    active.pending_logits.copy_from_slice(nl);
                }
            }
        }
        let sampled = {
            let active = request_mut(inner, id)?;
            sample(
                &mut active.pending_logits,
                mask,
                &active.params,
                &mut active.rng,
            )
        };
        produced.push(sampled);
        let child = forest.children(parent).find(|&i| forest.nodes[i].token_id as u32 == sampled);
        if let Some(idx) = child {
            accepted_tokens.push(sampled);
            parent = idx as i32;
        }
        append_and_forward(inner, id, sampled, metrics, hip, micro_batch)?;
        let stop = is_stop(sampled, &request(inner, id)?.params);
        if stop || child.is_none() || forest.children(parent).next().is_none()
        {
            break;
        }
    }
    let path_valid = forest.is_valid_path(&accepted_tokens);
    if !path_valid {
        metrics
            .tree_path_invalid
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    metrics.tree_accepted_path_len.fetch_add(
        accepted_tokens.len() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    Ok(Produced {
        tokens: produced,
        accepted: accepted_tokens.len() as u64,
        proposed: forest.nodes.len() as u64,
    })
}

fn append_and_forward(
    inner: &mut EngineInner,
    id: u64,
    token: u32,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    micro_batch: Option<&MicroBatchExpertUnion>,
) -> Result<(), AllpacaError> {
    let pos = request(inner, id)?.tokens.len();
    let page_tokens = request(inner, id)?
        .pages
        .first()
        .and_then(|layer| layer.first())
        .and_then(|page_id| inner.kv.get(*page_id))
        .map(|p| p.capacity_tokens)
        .unwrap_or(128);
    let target = inner
        .target
        .clone()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    let mut pages = request(inner, id)?.pages.clone();
    ensure_pages(
        &mut inner.kv,
        &target.config,
        page_tokens,
        &mut pages,
        pos,
        metrics,
    )?;
    {
        let active = request_mut(inner, id)?;
        active.pages = pages;
        active.tokens.push(token);
    }
    if inner.flags.ngram_speculation {
        let ctx = request(inner, id)?.tokens.clone();
        inner.ngram.observe(&ctx);
    }
    let mut logits = std::mem::take(&mut request_mut(inner, id)?.pending_logits);
    let pages = request(inner, id)?.pages.clone();
    let resident = inner.target_resident.clone();
    let mut recurrent = std::mem::take(&mut request_mut(inner, id)?.recurrent);
    target.forward_token(
        token,
        pos,
        &pages,
        &mut inner.kv,
        metrics,
        &mut logits,
        hip,
        resident.as_deref(),
        false,
        recurrent.as_mut(),
        micro_batch,
    )?;
    request_mut(inner, id)?.pending_logits = logits;
    request_mut(inner, id)?.recurrent = recurrent;
    Ok(())
}

fn is_stop(token: u32, params: &GenerationParams) -> bool {
    params.stop_token_ids.contains(&token)
}

fn is_special(piece: &str) -> bool {
    piece.starts_with('<') && piece.ends_with('>')
}

fn handle_prompt_len(
    handle: &RequestHandle,
    inner: &Arc<Mutex<EngineInner>>,
) -> Result<usize, AllpacaError> {
    let guard = inner.lock().map_err(|_| {
        AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
    })?;
    if let Some(active) = guard.active.get(&handle.id) {
        return Ok(active.prompt_len);
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_types::{CliOverrides, MissPolicy};

    fn test_config() -> Config {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(7),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.kv.page_tokens = 8;
        cfg.kv.vram_bytes = 8 * 1024;
        cfg.kv.dram_bytes = 64 * 1024;
        cfg
    }

    fn test_engine() -> Engine {
        Engine::new(test_config()).unwrap()
    }

    #[test]
    fn generate_produces_tokens() {
        let engine = test_engine();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        assert_eq!(out.completion_tokens, out.token_ids.len());
        assert!(out.completion_tokens <= 8);
        assert!(out.prompt_tokens >= 1);
        assert!(matches!(
            out.finish_reason,
            FinishReason::Length | FinishReason::Stop
        ));
    }

    #[test]
    fn streamed_ffn_greedy_tokens_match_fully_resident() {
        let mut resident_cfg = test_config();
        resident_cfg.speculative.enabled = false;
        let resident = Engine::new(resident_cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let full = resident.generate("hello", params.clone()).unwrap();

        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.weights.trunk_bytes = 64;
        cfg.weights.ring_slots = 1;
        let streamed = Engine::new(cfg).unwrap();
        let part = streamed.generate("hello", params).unwrap();
        assert_eq!(
            part.token_ids, full.token_ids,
            "GPU residency is a speed dial: greedy tokens must match"
        );
        let stats = streamed.stats();
        assert!(
            stats.weight_ring_slots >= 1,
            "tiny trunk budget must grant a ring"
        );
        assert!(
            stats.weight_ring_binds > 0,
            "streamed FFN must actually bind through the ring"
        );
        assert!(
            stats.nvme_reads > 0,
            "unpinned FFN must page from the packed trunk, not stay in RAM; stats={stats:?}"
        );
        assert!(
            stats.weight_pinned_layers < 2,
            "tiny budget should not pin both synthetic layers"
        );
    }

    #[test]
    fn moe_lru_greedy_tokens_match_fully_resident() {
        let mut resident_cfg = test_config();
        resident_cfg.model.synthetic = false;
        resident_cfg.speculative.enabled = false;
        let resident = Engine::new(resident_cfg).unwrap();
        resident.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let full = resident.generate("hello", params.clone()).unwrap();

        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.cache_bytes = 64;
        cfg.weights.miss_policy = MissPolicy::Fill;
        let streamed = Engine::new(cfg).unwrap();
        streamed.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let part = streamed.generate("hello", params).unwrap();
        assert_eq!(
            part.token_ids, full.token_ids,
            "GPU residency is a speed dial: greedy tokens must match"
        );
        let stats = streamed.stats();
        assert!(
            stats.expert_slots >= 1,
            "tiny cache_bytes must grant at least one expert slot"
        );
        assert!(
            stats.expert_lru_misses > 0,
            "a one-slot LRU over 4 experts must miss"
        );
        assert!(
            stats.expert_lru_fills > 0,
            "Fill policy must admit misses into the LRU"
        );
    }

    #[test]
    fn moe_nvme_host_pool_greedy_tokens_match_fully_resident() {
        let mut resident_cfg = test_config();
        resident_cfg.model.synthetic = false;
        resident_cfg.speculative.enabled = false;
        let resident = Engine::new(resident_cfg).unwrap();
        resident.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let full = resident.generate("hello", params.clone()).unwrap();

        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.host_bytes = 64;
        cfg.weights.cache_bytes = 0;
        let streamed = Engine::new(cfg).unwrap();
        streamed.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let part = streamed.generate("hello", params).unwrap();
        assert_eq!(
            part.token_ids, full.token_ids,
            "NVMe host pool is a speed dial: greedy tokens must match"
        );
        let stats = streamed.stats();
        assert!(
            stats.expert_host_slots >= 1,
            "tiny host_bytes must grant at least one DRAM slot; stats={stats:?}"
        );
        assert!(
            stats.nvme_reads > 0,
            "experts that miss the DRAM pool must page from the packed file; stats={stats:?}"
        );
        #[cfg(windows)]
        assert!(
            stats.nvme_direct,
            "packed expert reads must use FILE_FLAG_NO_BUFFERING, not the page cache; stats={stats:?}"
        );
    }

    #[test]
    fn elastic_vram_rebalance_keeps_tokens_and_host_pool() {
        let mut resident_cfg = test_config();
        resident_cfg.model.synthetic = false;
        resident_cfg.speculative.enabled = false;
        let resident = Engine::new(resident_cfg).unwrap();
        resident.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let full = resident.generate("hello", params.clone()).unwrap();

        let expert_bytes = 64 * 32 * 3 * 4;
        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.host_bytes = 64;
        cfg.weights.cache_bytes = expert_bytes * 4;
        cfg.kv.vram_bytes = 8 * 1024;
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let first = engine.generate("hello", params.clone()).unwrap();
        assert_eq!(first.token_ids, full.token_ids);
        let before = engine.stats();
        assert!(
            before.expert_slots >= 4,
            "wide cache_bytes must grant several expert slots; stats={before:?}"
        );
        assert!(
            before.nvme_reads > 0,
            "host_bytes overflow must page from NVMe; stats={before:?}"
        );
        let host_reads = before.nvme_reads;

        engine
            .rebalance_vram(64, 1024)
            .expect("safe-point rebalance must succeed with no in-flight forward");
        let mid = engine.stats();
        assert_eq!(
            mid.nvme_reads, host_reads,
            "rebalance must not reload the host/NVMe pool; stats={mid:?}"
        );
        assert_eq!(
            mid.expert_slots, 1,
            "shrinking cache_bytes must rebuild a one-slot LRU; stats={mid:?}"
        );

        let second = engine.generate("hello", params).unwrap();
        assert_eq!(
            second.token_ids, full.token_ids,
            "elastic VRAM is a speed dial: greedy tokens must match"
        );
    }

    #[test]
    fn semantic_anchor_skips_prefix_and_matches_full_prefill() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut full_cfg = test_config();
        full_cfg.model.synthetic = false;
        full_cfg.speculative.enabled = false;
        let full_engine = Engine::new(full_cfg).unwrap();
        full_engine
            .load_synthetic(ModelConfig::tiny_hybrid())
            .unwrap();
        let full = full_engine
            .generate("hello true", params.clone())
            .unwrap();

        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.anchors.slots = 4;
        cfg.anchors.token_ids = vec![3];
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_hybrid()).unwrap();
        let _warm = engine.generate("hello world", params.clone()).unwrap();
        let after_warm = engine.stats();
        assert_eq!(
            after_warm.anchor_hits, 0,
            "first prompt has no prior checkpoint; stats={after_warm:?}"
        );

        let part = engine.generate("hello true", params).unwrap();
        assert_eq!(
            part.token_ids, full.token_ids,
            "restoring the deepest surviving anchor must match a full prefill"
        );
        let stats = engine.stats();
        assert!(
            stats.anchor_hits >= 1,
            "edited prompt must hit the hello-boundary checkpoint; stats={stats:?}"
        );
        assert!(
            stats.anchor_skip_tokens >= 2,
            "bos+hello must be skipped; stats={stats:?}"
        );
    }

    #[test]
    fn radix_prefix_skips_shared_tokens_after_mid_edit() {
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let _warm = engine.generate("hello world", params.clone()).unwrap();

        let _edited = engine.generate("hello there", params).unwrap();
        let stats = engine.stats();
        assert!(
            stats.radix_hits >= 1,
            "edited prompt must restore LCP via radix; stats={stats:?}"
        );
        assert!(
            stats.radix_skip_tokens >= 1,
            "at least one shared token must be skipped; stats={stats:?}"
        );
    }

    #[test]
    fn ds_probe_surfaces_on_stats() {
        let engine = Engine::new(test_config()).unwrap();
        let stats = engine.stats();
        // Probe always runs; availability is host-dependent.
        let _ = stats.ds_available;
        assert_eq!(stats.ds_enqueued, 0, "no expert H2D yet; stats={stats:?}");
        assert_eq!(stats.ds_pinned, 0, "no pinned staging yet; stats={stats:?}");
    }

    #[test]
    fn qstar_records_measured_host_bandwidth() {
        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.cache_bytes = 64;
        cfg.weights.miss_policy = MissPolicy::Qstar;
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let stats = engine.stats();
        assert!(
            stats.expert_host_bps > 0,
            "q* must measure CPU expert bandwidth; stats={stats:?}"
        );
        assert_eq!(
            stats.expert_pcie_bps, 0,
            "cpu_only has no PCIe expert H2D"
        );
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let after = engine.stats();
        assert!(
            after.expert_cpu_runs > 0,
            "measured q* with no PCIe must CPU-run residual misses; stats={after:?}"
        );
    }

    #[test]
    fn moe_prefill_admits_full_layer_and_matches_resident() {
        let expert_bytes = 64 * 32 * 3 * 4;
        let mut resident_cfg = test_config();
        resident_cfg.model.synthetic = false;
        resident_cfg.speculative.enabled = false;
        let resident = Engine::new(resident_cfg).unwrap();
        resident.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let prompt = "hello ".repeat(16);
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let full = resident.generate(&prompt, params.clone()).unwrap();

        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.cache_bytes = expert_bytes * 4;
        cfg.weights.miss_policy = MissPolicy::Fill;
        let buffered = Engine::new(cfg).unwrap();
        buffered.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let part = buffered.generate(&prompt, params).unwrap();
        assert_eq!(
            part.token_ids, full.token_ids,
            "GPU residency is a speed dial: greedy tokens must match"
        );
        let stats = buffered.stats();
        assert!(
            stats.expert_slots >= 4,
            "one full expert layer must fit; slots={}",
            stats.expert_slots
        );
        assert!(
            stats.expert_lru_fills >= 4,
            "prefill must admit a whole layer, not only top-k; fills={}",
            stats.expert_lru_fills
        );
    }

    #[test]
    fn moe_prefill_double_buffer_hits_on_live_path() {
        let expert_bytes = 64 * 32 * 3 * 4;
        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.cache_bytes = expert_bytes * 8;
        cfg.weights.miss_policy = MissPolicy::Fill;
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let prompt = "hello ".repeat(16);
        let _ = engine
            .generate(
                &prompt,
                GenerationParams {
                    max_tokens: 2,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let stats = engine.stats();
        assert!(
            stats.expert_slots >= 8,
            "two full layers must fit; slots={}",
            stats.expert_slots
        );
        assert!(
            stats.expert_lru_hits > stats.expert_lru_fills / 4,
            "double-buffered prefill must hit admitted experts; stats={stats:?}"
        );
    }

    #[test]
    fn numa_host_bandwidth_uses_cross_node_floor() {
        use crate::numa::NumaTopology;
        use crate::weight_set::{ExpertId, LayerFfn};
        let topo = NumaTopology::with_nodes(2);
        let ffn = LayerFfn {
            w_gate: vec![1.0; 32],
            w_up: vec![1.0; 32],
            w_down: vec![1.0; 32],
            ..LayerFfn::default()
        };
        let id = ExpertId { layer: 0, expert: 1 };
        let b = crate::model::measure_expert_host_bandwidth_numa(&ffn, 32, 64, &topo, id);
        assert!(b > 0.0, "NUMA-aware host measure must be positive");
        assert!(
            topo.binds_ok() + topo.binds_skip() >= 2,
            "local + remote probes must attempt binds"
        );
    }

    #[test]
    fn generate_with_prompt_longer_than_eight_tokens_still_produces_tokens() {
        let engine = test_engine();
        let prompt = "hello ".repeat(16);
        let out = engine
            .generate(
                &prompt,
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(
            out.prompt_tokens >= 16,
            "need a prompt of at least 16 tokens, got {}",
            out.prompt_tokens
        );
        assert!(!out.token_ids.is_empty());
    }

    #[tokio::test]
    async fn generate_stream_emits_tokens() {
        use futures::StreamExt;

        let engine = test_engine();
        let mut stream = std::pin::pin!(engine.generate_stream(
            "hello".into(),
            GenerationParams {
                max_tokens: 6,
                temperature: 0.0,
                seed: Some(7),
                ..GenerationParams::default()
            },
        ));
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            let ev = item.expect("stream token");
            events.push(ev.clone());
            if ev.finish_reason.is_some() {
                break;
            }
        }
        assert!(!events.is_empty());
        assert!(events.iter().all(|e| e.token_id < 64));
        assert!(events.last().and_then(|e| e.finish_reason).is_some());
    }

    #[test]
    fn begin_step_matches_generate() {
        let engine = test_engine();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut handle = engine.begin_request("hello", params.clone()).unwrap();
        let mut n = 0;
        while let StepResult::Token(_) = engine.step(&mut handle).unwrap() {
            n += 1;
            if n > 8 {
                break;
            }
        }
        assert!(n >= 1);
    }

    #[test]
    fn exceeding_cap_is_transient_and_drop_frees_slot() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let first = engine.begin_request("hello", params.clone()).unwrap();
        let second = engine.begin_request("world", params.clone()).unwrap();
        let err = engine.begin_request("third", params.clone()).unwrap_err();
        assert!(
            matches!(
                err,
                AllpacaError::Transient(TransientError::RequestBusy)
                    | AllpacaError::Transient(TransientError::Backpressure(_))
            ),
            "saturated scheduler must be Transient, got {err:?}"
        );
        assert_eq!(err.class(), allpaca_types::ErrorClass::Transient);
        drop(first);
        engine
            .begin_request("third", params.clone())
            .expect("dropping a Request must free a scheduler slot");
        drop(second);
        let out = engine
            .generate("hello", params)
            .expect("completed generate must free its slot");
        assert!(!out.token_ids.is_empty());
    }

    #[test]
    fn each_request_keeps_its_own_kv_pages() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.kv.page_tokens = 2;
        cfg.kv.vram_bytes = 1024;
        cfg.kv.dram_bytes = 256 * 1024;
        cfg.kv.hot_fraction = 0.35;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut first = engine.begin_request("hello", params.clone()).unwrap();
        let mut second = engine.begin_request("world", params).unwrap();
        assert!(matches!(engine.step(&mut first).unwrap(), StepResult::Token(_)));
        assert!(matches!(engine.step(&mut second).unwrap(), StepResult::Token(_)));

        let kv_a = first.kv_handle().expect("first Request KvHandle");
        let kv_b = second.kv_handle().expect("second Request KvHandle");
        assert_ne!(kv_a.request_id, kv_b.request_id);
        assert!(!kv_a.pages.is_empty());
        assert!(!kv_b.pages.is_empty());
        for page in &kv_a.pages {
            assert!(
                !kv_b.pages.contains(page),
                "Requests must not share KV Pages; a={kv_a:?} b={kv_b:?}"
            );
        }

        let pages_both = engine.stats().vram_pages + engine.stats().dram_pages;
        assert!(pages_both > 0);
        drop(first);
        let pages_after = engine.stats().vram_pages + engine.stats().dram_pages;
        assert!(
            pages_after > 0,
            "finishing one Request must not wipe the other's KV; stats={:?}",
            engine.stats()
        );
        assert!(
            pages_after < pages_both,
            "dropped Request must release its KV Pages; before={pages_both} after={pages_after}"
        );
        assert!(matches!(
            engine.step(&mut second).unwrap(),
            StepResult::Token(_) | StepResult::Finished(_)
        ));
        let stats = engine.stats();
        assert!(
            stats.dram_pages > 0 || stats.page_migrations > 0 || stats.vram_pages > 0,
            "KV spilling/stats must still work with two Requests; stats={stats:?}"
        );
    }

    #[tokio::test]
    async fn two_concurrent_streams_progress_independently() {
        use futures::StreamExt;

        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 6,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut a = std::pin::pin!(engine.generate_stream("hello".into(), params.clone()));
        let mut b = std::pin::pin!(engine.generate_stream("world".into(), params));
        let ev_a = a
            .next()
            .await
            .expect("stream a")
            .expect("first Request stream must not be RequestBusy");
        let ev_b = b
            .next()
            .await
            .expect("stream b")
            .expect("second Request stream must not be RequestBusy");
        assert!(ev_a.token_id < 64);
        assert!(ev_b.token_id < 64);
        let mut n_a = 1usize;
        let mut n_b = 1usize;
        while let Some(item) = a.next().await {
            item.expect("stream a token");
            n_a += 1;
        }
        while let Some(item) = b.next().await {
            item.expect("stream b token");
            n_b += 1;
        }
        assert!(n_a >= 1);
        assert!(n_b >= 1);
    }

    #[test]
    fn two_concurrent_generate_progress_independently() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 6,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let a = {
            let engine = engine.clone();
            let params = params.clone();
            std::thread::spawn(move || engine.generate("hello", params))
        };
        let b = {
            let engine = engine.clone();
            std::thread::spawn(move || engine.generate("world", params))
        };
        let out_a = a
            .join()
            .expect("thread a")
            .expect("first Request must not be RequestBusy");
        let out_b = b
            .join()
            .expect("thread b")
            .expect("second Request must not be RequestBusy");
        assert!(!out_a.token_ids.is_empty());
        assert!(!out_b.token_ids.is_empty());
        assert!(out_a.prompt_tokens >= 1);
        assert!(out_b.prompt_tokens >= 1);
        assert_ne!(
            out_a.token_ids, out_b.token_ids,
            "different prompts must produce independently progressing Requests"
        );
    }

    #[test]
    fn concurrent_generate_uses_poll_when_max_active_above_one() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let a = {
            let engine = engine.clone();
            let params = params.clone();
            std::thread::spawn(move || engine.generate("hello", params))
        };
        let b = {
            let engine = engine.clone();
            std::thread::spawn(move || engine.generate("world", params))
        };
        let out_a = a.join().unwrap().unwrap();
        let out_b = b.join().unwrap().unwrap();
        assert!(!out_a.token_ids.is_empty());
        assert!(!out_b.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.poll_ticks > 0,
            "max_active_requests>1 must drive generate via poll; stats={stats:?}"
        );
        assert!(
            stats.poll_decode_picks > 0,
            "fair generate must record decode picks; stats={stats:?}"
        );
    }

    #[test]
    fn poll_interleaves_two_decode_requests() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let a = engine.begin_request("hello", params.clone()).unwrap();
        let b = engine.begin_request("world", params).unwrap();
        let id_a = a.id();
        let id_b = b.id();

        let mut progress_a = 0usize;
        let mut progress_b = 0usize;
        let mut switches = 0usize;
        let mut last: Option<u64> = None;
        let mut done_a = false;
        let mut done_b = false;
        for _ in 0..64 {
            let Some(ev) = engine.poll().expect("poll") else {
                break;
            };
            if last.map(|l| l != ev.request_id).unwrap_or(false) {
                switches += 1;
            }
            last = Some(ev.request_id);
            match ev.result {
                StepResult::Token(tok) => {
                    if ev.request_id == id_a {
                        progress_a += 1;
                        if tok.finish_reason.is_some() {
                            done_a = true;
                        }
                    } else if ev.request_id == id_b {
                        progress_b += 1;
                        if tok.finish_reason.is_some() {
                            done_b = true;
                        }
                    }
                }
                StepResult::Prefill { .. } => {
                    if ev.request_id == id_a {
                        progress_a += 1;
                    } else if ev.request_id == id_b {
                        progress_b += 1;
                    }
                }
                StepResult::Finished(_) => {
                    if ev.request_id == id_a {
                        done_a = true;
                    } else if ev.request_id == id_b {
                        done_b = true;
                    }
                }
            }
            if done_a && done_b {
                break;
            }
        }
        assert!(
            progress_a >= 1 && progress_b >= 1,
            "poll must advance both Requests; a={progress_a} b={progress_b}"
        );
        assert!(
            switches >= 1,
            "poll must interleave Requests instead of draining one; switches={switches}"
        );
        assert!(done_a && done_b, "both Requests must finish under poll-only driving");
        let stats = engine.stats();
        assert!(
            stats.micro_batch_plans >= 1,
            "concurrent poll must plan stagger; stats={stats:?}"
        );
        assert!(
            stats.micro_batch_dual_steps >= 1,
            "concurrent poll must same-tick dual-step partner; stats={stats:?}"
        );
        drop(a);
        drop(b);
    }

    #[test]
    fn poll_lets_decode_progress_while_peer_prefills() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.prefill.chunk_tokens = 1;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 3,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let long = "hello world hello world hello world hello world";
        let long_req = engine.begin_request(long, params.clone()).unwrap();
        let short = engine.begin_request("hello", params).unwrap();
        let id_long = long_req.id();
        let id_short = short.id();

        let mut short_tokens = 0usize;
        let mut long_prefill_after_short_token = false;
        for _ in 0..96 {
            let Some(ev) = engine.poll().expect("poll") else {
                break;
            };
            match &ev.result {
                StepResult::Token(_) if ev.request_id == id_short => {
                    short_tokens += 1;
                }
                StepResult::Prefill {
                    tokens_done,
                    tokens_total,
                } if ev.request_id == id_long && *tokens_done < *tokens_total => {
                    if short_tokens >= 1 {
                        long_prefill_after_short_token = true;
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(
            short_tokens >= 1,
            "decode Request must emit tokens under poll while a peer is still prefilling"
        );
        assert!(
            long_prefill_after_short_token,
            "long Request must still be mid-prefill after the decode Request produced a token"
        );
        drop(long_req);
        drop(short);
    }

    #[test]
    fn decode_priority_poll_prefers_decoding_over_older_prefill() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.engine.poll_schedule = allpaca_types::PollSchedule::DecodePriority;
        cfg.prefill.chunk_tokens = 1;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let long = "hello world hello world hello world hello world";
        let _long = engine.begin_request(long, params.clone()).unwrap();
        let short = engine.begin_request("hi", params).unwrap();
        let id_short = short.id();

        let mut short_tokens = 0usize;
        for _ in 0..32 {
            let Some(ev) = engine.poll().expect("poll") else {
                break;
            };
            if ev.request_id == id_short && matches!(ev.result, StepResult::Token(_)) {
                short_tokens += 1;
            }
            if short_tokens >= 2 {
                break;
            }
        }
        assert!(
            short_tokens >= 2,
            "decode_priority must let short Request decode; got {short_tokens}"
        );
    }

    #[test]
    fn prefill_budget_forces_decode_under_mixed_load() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.engine.poll_schedule = allpaca_types::PollSchedule::PrefillBudget;
        cfg.engine.max_consecutive_prefill_picks = 1;
        cfg.prefill.chunk_tokens = 1;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let long = "hello world hello world hello world hello world";
        let _long = engine.begin_request(long, params.clone()).unwrap();
        let short = engine.begin_request("hi", params).unwrap();
        let id_short = short.id();
        let mut short_tokens = 0usize;
        let mut saw_prefill = false;
        for _ in 0..48 {
            let Some(ev) = engine.poll().expect("poll") else {
                break;
            };
            match &ev.result {
                StepResult::Prefill { .. } => saw_prefill = true,
                StepResult::Token(_) if ev.request_id == id_short => {
                    short_tokens += 1;
                    if short_tokens >= 1 && saw_prefill {
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(
            short_tokens >= 1 && saw_prefill,
            "prefill_budget must still serve decode after a prefill streak; tokens={short_tokens} saw_prefill={saw_prefill}"
        );
    }

    #[test]
    fn queue_pressure_shrinks_draft_depth() {
        let mut cfg = test_config();
        cfg.speculative.enabled = true;
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 12;
        cfg.features.adaptive_draft_depth = Some(true);
        let engine = Engine::new(cfg).unwrap();
        {
            let mut inner = engine.lock().unwrap();
            inner.scheduler.record_acceptance(9, 10);
            let raised = inner.scheduler.draft_depth;
            assert!(raised > 2);
            // Neutral acceptance keeps depth; queue pressure alone must shrink.
            inner
                .scheduler
                .record_acceptance_with_pressure(5, 10, 3);
            assert!(
                inner.scheduler.draft_depth < raised,
                "active_requests>1 must shrink depth; raised={raised} now={}",
                inner.scheduler.draft_depth
            );
            assert!(inner.scheduler.draft_depth >= 2);
        }
    }

    #[test]
    fn adaptive_draft_depth_moves_during_generate() {
        let mut cfg = test_config();
        cfg.speculative.enabled = true;
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 12;
        cfg.features.adaptive_draft_depth = Some(true);
        cfg.features.ngram_speculation = Some(true);
        let engine = Engine::new(cfg).unwrap();
        let initial = engine.stats().draft_depth;
        // Repeating prompt trains the ngram; long decode should propose and adapt.
        let prompt = "hello world hello world hello world hello world";
        let out = engine
            .generate(
                prompt,
                GenerationParams {
                    max_tokens: 24,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.draft_proposed > 0,
            "live generate must propose drafts; stats={stats:?}"
        );
        assert!(
            stats.draft_depth != initial || stats.acceptance_rate > 0.0,
            "adaptive path must move depth or record acceptance; initial={initial} stats={stats:?}"
        );
    }

    #[test]
    fn adaptive_draft_depth_observable_on_stats() {
        let mut cfg = test_config();
        cfg.speculative.enabled = true;
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 12;
        cfg.features.adaptive_draft_depth = Some(true);
        let engine = Engine::new(cfg).unwrap();
        let initial = engine.stats().draft_depth;
        assert!(initial >= 2, "initial draft depth from min_depth; stats={initial}");
        {
            let mut inner = engine.lock().unwrap();
            inner.scheduler.record_acceptance(9, 10);
        }
        let raised = engine.stats().draft_depth;
        assert!(
            raised > initial,
            "high acceptance must raise draft_depth; before={initial} after={raised}"
        );
        {
            let mut inner = engine.lock().unwrap();
            inner.scheduler.record_acceptance(1, 10);
        }
        let lowered = engine.stats().draft_depth;
        assert!(
            lowered < raised,
            "low acceptance must shrink draft_depth; raised={raised} lowered={lowered}"
        );
    }

    #[test]
    fn two_begin_request_succeed_when_max_active_requests_is_two() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let first = engine.begin_request("hello", params.clone()).unwrap();
        let second = engine
            .begin_request("world", params)
            .expect("second Request must be admitted when under the cap");
        assert_eq!(engine.stats().active_requests, 2);
        drop(first);
        drop(second);
    }

    #[test]
    fn chunked_prefill_lets_second_request_admit_and_is_observable() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.prefill.chunk_tokens = 1;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 2,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let long = "hello world hello world hello world hello world";
        let mut first = engine.begin_request(long, params.clone()).unwrap();
        let after_first = engine.stats();
        assert_eq!(
            after_first.prefill_chunks, 1,
            "begin_request must release after one prefill chunk; stats={after_first:?}"
        );
        let mut second = engine
            .begin_request("hello", params.clone())
            .expect("second Request must admit before the first prompt finishes prefill");
        assert_eq!(engine.stats().active_requests, 2);
        match engine.step(&mut first).unwrap() {
            StepResult::Prefill {
                tokens_done,
                tokens_total,
            } => {
                assert!(tokens_done < tokens_total);
                assert!(tokens_done >= 2);
            }
            other => panic!("long prompt with chunk_tokens=1 must keep prefilling, got {other:?}"),
        }
        match engine.step(&mut second).unwrap() {
            StepResult::Prefill { .. } | StepResult::Token(_) | StepResult::Finished(_) => {}
        }
        drop(first);
        drop(second);
        let out = engine
            .generate(long, params)
            .expect("generate must finish remaining prefill chunks");
        assert!(!out.token_ids.is_empty());
        assert!(
            engine.stats().prefill_chunks > 1,
            "chunked prefill must be visible on stats; stats={:?}",
            engine.stats()
        );
    }

    #[test]
    fn second_request_is_busy() {
        let engine = test_engine();
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationParams::default()
        };
        let first = engine.begin_request("hello", params.clone()).unwrap();
        let err = engine.begin_request("world", params.clone()).unwrap_err();
        assert!(matches!(
            err,
            AllpacaError::Transient(TransientError::RequestBusy)
        ));
        assert_eq!(err.class(), allpaca_types::ErrorClass::Transient);
        drop(first);
        engine
            .begin_request("world", params)
            .expect("lock released after Request handle drop");
    }

    #[test]
    fn generate_spills_kv_pages_to_pinned_dram() {
        let mut cfg = test_config();
        cfg.kv.page_tokens = 2;
        cfg.kv.vram_bytes = 1024;
        cfg.kv.dram_bytes = 256 * 1024;
        cfg.kv.hot_fraction = 0.35;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 16,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut handle = engine.begin_request("hello", params).unwrap();
        let mut tokens = 0usize;
        while let StepResult::Token(ev) = engine.step(&mut handle).unwrap() {
            tokens += 1;
            if ev.finish_reason.is_some() {
                break;
            }
        }
        assert!(tokens >= 1);
        let stats = engine.stats();
        assert!(
            stats.dram_pages > 0 || stats.page_migrations > 0,
            "KV Pages must move VRAM → Pinned DRAM under pressure; stats={stats:?}"
        );
        drop(handle);
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty(), "generate still works after KV spill");
    }

    #[test]
    fn generate_with_nvme_kv_matches_resident() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let prompt = "hello world hello world hello world hello true";

        let mut full_cfg = test_config();
        full_cfg.speculative.enabled = false;
        full_cfg.kv.page_tokens = 2;
        let full = Engine::new(full_cfg).unwrap();
        let full_out = full.generate(prompt, params.clone()).unwrap();

        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.kv.page_tokens = 2;
        cfg.kv.vram_bytes = 256;
        cfg.kv.dram_bytes = 256;
        cfg.kv.hot_fraction = 0.4;
        cfg.kv.nvme_bytes = 1024 * 1024;
        let engine = Engine::new(cfg).unwrap();
        let part = engine.generate(prompt, params).unwrap();
        assert_eq!(
            part.token_ids, full_out.token_ids,
            "NVMe cold KV is capacity, not a different model"
        );
        let stats = engine.stats();
        assert!(
            stats.kv_nvme_pages > 0 || stats.kv_nvme_reads > 0,
            "tight DRAM must page KV to NVMe; stats={stats:?}"
        );
    }

    #[test]
    fn stats_snapshot_includes_nvme_kv_fields() {
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.kv.page_tokens = 2;
        cfg.kv.vram_bytes = 256;
        cfg.kv.dram_bytes = 256;
        cfg.kv.hot_fraction = 0.4;
        cfg.kv.nvme_bytes = 1024 * 1024;
        let engine = Engine::new(cfg).unwrap();
        let _ = engine
            .generate(
                "hello world hello world hello world hello true",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let stats = engine.stats();
        assert!(
            stats.kv_nvme_reads > 0 || stats.kv_nvme_pages > 0,
            "NVMe cold KV must be observable on stats; stats={stats:?}"
        );
    }

    #[test]
    fn disabling_tree_falls_back_to_linear_draft() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut cfg = test_config();
        cfg.speculative.tree.enabled = true;
        cfg.features.tree_speculation = Some(false);
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(true);
        let engine = Engine::new(cfg).expect("disabling Token Forest must not be Fatal");
        let out = engine.generate("hello", params).expect("linear Draft Provider fallback");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert_eq!(
            stats.tree_nodes, 0,
            "disabled tree Feature Flag must not build a Token Forest; stats={stats:?}"
        );
        assert!(
            stats.draft_proposed > 0,
            "Engine must fall back to linear NGram/DraftModel providers; stats={stats:?}"
        );
    }

    #[test]
    fn tree_accepted_prefix_is_valid_forest_path() {
        let mut cfg = test_config();
        cfg.speculative.tree.enabled = true;
        cfg.speculative.tree.max_nodes = 16;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(true);
        let engine = Engine::new(cfg).unwrap();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.tree_nodes > 0,
            "Token Forest must be built when the Feature Flag is on"
        );
        assert!(
            stats.tree_path_valid,
            "accepted prefix must be a valid ancestry path; stats={stats:?}"
        );
        assert!(
            stats.tree_accepted_path_len <= stats.tree_nodes,
            "accepted path cannot be longer than the forest; stats={stats:?}"
        );
    }

    #[test]
    fn tree_speculation_proposes_multiple_branches() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut linear_cfg = test_config();
        linear_cfg.speculative.tree.enabled = false;
        linear_cfg.features.draft_model_speculation = Some(true);
        linear_cfg.features.ngram_speculation = Some(true);
        let linear = Engine::new(linear_cfg).unwrap();
        linear.generate("hello", params.clone()).unwrap();
        let linear_stats = linear.stats();

        let mut tree_cfg = test_config();
        tree_cfg.speculative.tree.enabled = true;
        tree_cfg.speculative.tree.max_nodes = 16;
        tree_cfg.features.draft_model_speculation = Some(true);
        tree_cfg.features.ngram_speculation = Some(true);
        let tree = Engine::new(tree_cfg).unwrap();
        tree.generate("hello", params).unwrap();
        let tree_stats = tree.stats();

        assert!(
            tree_stats.tree_nodes > 1,
            "Token Forest must propose multiple candidate nodes; stats={tree_stats:?}"
        );
        assert!(
            tree_stats.tree_nodes > linear_stats.draft_proposed
                || tree_stats.draft_proposed > linear_stats.draft_proposed
                || tree_stats.accepted_tokens_per_target_eval
                    > linear_stats.accepted_tokens_per_target_eval,
            "tree must expose more candidates than linear-only; tree={tree_stats:?} linear={linear_stats:?}"
        );
    }

    /// Operator soak: turn on `speculative.tree` and watch `tree_*` on `/stats`.
    #[test]
    fn tree_speculation_soak_exposes_stats_for_operators() {
        let mut cfg = test_config();
        cfg.speculative.tree.enabled = true;
        cfg.speculative.tree.max_nodes = 16;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(true);
        let engine = Engine::new(cfg).unwrap();
        let out = engine
            .generate(
                "hello world hello",
                GenerationParams {
                    max_tokens: 12,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.tree_nodes > 1,
            "enable speculative.tree for forest soak; missing tree_nodes: {stats:?}"
        );
        assert!(
            stats.tree_path_valid
                || stats.tree_accepted_path_len > 0
                || stats.tokens_generated > 0,
            "tree soak must finish with path/tokens; stats={stats:?}"
        );
        assert!(
            stats.draft_proposed > 0 || stats.accepted_tokens_per_target_eval >= 1.0,
            "operator /stats must show draft activity under tree; {stats:?}"
        );
    }

    #[test]
    fn tree_speculation_generate_matches_greedy_without_tree() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut with_tree = test_config();
        with_tree.speculative.tree.enabled = true;
        with_tree.speculative.tree.max_nodes = 16;
        with_tree.features.draft_model_speculation = Some(true);
        with_tree.features.ngram_speculation = Some(true);
        let tree_engine = Engine::new(with_tree).unwrap();
        let tree_out = tree_engine.generate("hello", params.clone()).unwrap();
        assert!(
            !tree_out.token_ids.is_empty(),
            "tree-enabled generate must still produce tokens"
        );

        let mut no_tree = test_config();
        no_tree.speculative.tree.enabled = false;
        no_tree.features.draft_model_speculation = Some(true);
        no_tree.features.ngram_speculation = Some(true);
        let baseline = Engine::new(no_tree).unwrap();
        let base_out = baseline.generate("hello", params).unwrap();
        assert_eq!(
            tree_out.token_ids, base_out.token_ids,
            "Target Model must decide tokens when Token Forest is enabled"
        );
    }

    #[test]
    fn target_decides_tokens_with_speculation() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };

        let mut with_spec = test_config();
        with_spec.features.draft_model_speculation = Some(true);
        with_spec.features.ngram_speculation = Some(true);
        let speculated = Engine::new(with_spec).unwrap();
        let with_out = speculated.generate("hello", params.clone()).unwrap();
        assert!(
            speculated.stats().draft_proposed > 0,
            "Draft Provider must propose under speculation"
        );

        let mut no_spec = test_config();
        no_spec.merge_cli(&CliOverrides {
            speculative: Some(false),
            ..CliOverrides::default()
        });
        let baseline = Engine::new(no_spec).unwrap();
        let without_out = baseline.generate("hello", params).unwrap();
        assert_eq!(
            with_out.token_ids, without_out.token_ids,
            "Target Model must decide tokens; Draft Provider only proposes"
        );
    }

    fn is_special_token(piece: &str) -> bool {
        piece.starts_with('<') && piece.ends_with('>')
    }

    /// Public-API oracle: a prefix is JSON-legal if it is valid JSON or a
    /// truncated value (object/array/string/number/literal) with no junk.
    fn json_prefix_legal(s: &str) -> bool {
        if s.is_empty() {
            return true;
        }
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        let mut chars = s.chars().peekable();
        let mut started = false;
        while let Some(c) = chars.next() {
            if in_string {
                if escape {
                    escape = false;
                    continue;
                }
                match c {
                    '\\' => escape = true,
                    '"' => in_string = false,
                    ch if ch.is_control() => return false,
                    _ => {}
                }
                continue;
            }
            if c.is_whitespace() {
                continue;
            }
            started = true;
            match c {
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth -= 1;
                    if depth < 0 {
                        return false;
                    }
                }
                '"' => in_string = true,
                ':' | ',' => {
                    if depth == 0 {
                        return false;
                    }
                }
                't' => {
                    let rest: String = std::iter::once('t').chain(chars.clone()).take(4).collect();
                    if !"true".starts_with(&rest) && rest != "true"[..rest.len().min(4)] {
                        return false;
                    }
                    for _ in 0..rest.len().saturating_sub(1).min(3) {
                        chars.next();
                    }
                }
                'f' => {
                    let rest: String = std::iter::once('f').chain(chars.clone()).take(5).collect();
                    if !"false".starts_with(&rest) {
                        return false;
                    }
                    for _ in 0..rest.len().saturating_sub(1).min(4) {
                        chars.next();
                    }
                }
                'n' => {
                    let rest: String = std::iter::once('n').chain(chars.clone()).take(4).collect();
                    if !"null".starts_with(&rest) {
                        return false;
                    }
                    for _ in 0..rest.len().saturating_sub(1).min(3) {
                        chars.next();
                    }
                }
                '-' | '0'..='9' => {
                    while matches!(chars.peek(), Some('0'..='9' | '.' | 'e' | 'E' | '+' | '-')) {
                        chars.next();
                    }
                }
                _ => return false,
            }
        }
        started || s.chars().all(char::is_whitespace)
    }

    fn write_tiny_target_gguf(path: &std::path::Path) {
        write_tiny_llama_gguf(path, 0.01, 0.01);
    }

    fn write_tiny_llama_gguf(path: &std::path::Path, w: f32, ffn_down: f32) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
        let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight".into(), vec![vocab as u64, dim as u64], vec![w; vocab * dim]),
            ("blk.0.attn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
            ("blk.0.attn_q.weight".into(), vec![dim as u64, dim as u64], vec![w; dim * dim]),
            ("blk.0.attn_k.weight".into(), vec![2, dim as u64], vec![w; 2 * dim]),
            ("blk.0.attn_v.weight".into(), vec![2, dim as u64], vec![w; 2 * dim]),
            ("blk.0.attn_output.weight".into(), vec![dim as u64, dim as u64], vec![w; dim * dim]),
            ("blk.0.ffn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
            ("blk.0.ffn_gate.weight".into(), vec![ff as u64, dim as u64], vec![w; ff * dim]),
            ("blk.0.ffn_up.weight".into(), vec![ff as u64, dim as u64], vec![w; ff * dim]),
            ("blk.0.ffn_down.weight".into(), vec![dim as u64, ff as u64], vec![ffn_down; dim * ff]),
            ("output_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
        ];
        let views: Vec<GgufWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data,
            })
            .collect();
        write_gguf(path, &meta, &views).unwrap();
    }

    fn write_tiny_moe_gguf(path: &std::path::Path) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let n_exp = 4usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert("llama.expert_count".into(), GgufValue::Uint32(n_exp as u32));
        meta.insert("llama.expert_used_count".into(), GgufValue::Uint32(2));
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
        let w = 0.01f32;
        let mut gate_exps = Vec::new();
        let mut up_exps = Vec::new();
        let mut down_exps = Vec::new();
        for _ in 0..n_exp {
            gate_exps.extend(std::iter::repeat(w).take(ff * dim));
            up_exps.extend(std::iter::repeat(w).take(ff * dim));
            down_exps.extend(std::iter::repeat(w).take(dim * ff));
        }
        let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight".into(), vec![vocab as u64, dim as u64], vec![w; vocab * dim]),
            ("blk.0.attn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
            ("blk.0.attn_q.weight".into(), vec![dim as u64, dim as u64], vec![w; dim * dim]),
            ("blk.0.attn_k.weight".into(), vec![2, dim as u64], vec![w; 2 * dim]),
            ("blk.0.attn_v.weight".into(), vec![2, dim as u64], vec![w; 2 * dim]),
            ("blk.0.attn_output.weight".into(), vec![dim as u64, dim as u64], vec![w; dim * dim]),
            ("blk.0.ffn_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
            (
                "blk.0.ffn_gate_inp.weight".into(),
                vec![n_exp as u64, dim as u64],
                vec![w; n_exp * dim],
            ),
            (
                "blk.0.ffn_gate_exps.weight".into(),
                vec![n_exp as u64, ff as u64, dim as u64],
                gate_exps,
            ),
            (
                "blk.0.ffn_up_exps.weight".into(),
                vec![n_exp as u64, ff as u64, dim as u64],
                up_exps,
            ),
            (
                "blk.0.ffn_down_exps.weight".into(),
                vec![n_exp as u64, dim as u64, ff as u64],
                down_exps,
            ),
            ("output_norm.weight".into(), vec![dim as u64], vec![1.0; dim]),
        ];
        let views: Vec<GgufWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data,
            })
            .collect();
        write_gguf(path, &meta, &views).unwrap();
    }

    fn write_tiny_quant_gguf(path: &std::path::Path) {
        use allpaca_gguf::{
            quantize_q4_0, quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
            GgufWriteTensor,
        };
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

        let embd = vec![0.01f32; vocab * dim];
        let q = vec![0.01f32; dim * dim];
        let k = vec![0.01f32; 2 * dim];
        let v = vec![0.01f32; 2 * dim];
        let o = vec![0.01f32; dim * dim];
        let gate = vec![0.01f32; ff * dim];
        let up = vec![0.01f32; ff * dim];
        let down = vec![0.01f32; dim * ff];
        let embd_q = quantize_q8_0(&embd);
        let q_q = quantize_q4_0(&q);
        let k_q = quantize_q4_0(&k);
        let v_q = quantize_q4_0(&v);
        let o_q = quantize_q4_0(&o);
        let gate_q = quantize_q4_0(&gate);
        let up_q = quantize_q4_0(&up);
        let down_q = quantize_q4_0(&down);
        let attn_norm = vec![1.0f32; dim];
        let ffn_norm = vec![1.0f32; dim];
        let output_norm = vec![1.0f32; dim];

        let f32_tensors = [
            GgufWriteTensor {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &attn_norm,
            },
            GgufWriteTensor {
                name: "blk.0.ffn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &ffn_norm,
            },
            GgufWriteTensor {
                name: "output_norm.weight".into(),
                dims: vec![dim as u64],
                data: &output_norm,
            },
        ];
        let quant_tensors = [
            GgufWriteQuantTensor {
                name: "token_embd.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type: GgmlType::Q8_0,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::Q4_0,
                bytes: &down_q,
            },
        ];
        write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).unwrap();
    }

    #[test]
    fn engine_fatal_without_model_or_valid_config() {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        let engine = Engine::new(cfg).unwrap();
        let err = engine
            .generate("hello", GenerationParams::default())
            .unwrap_err();
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);

        let mut bad = Config::compiled_defaults();
        bad.kv.page_tokens = 0;
        let err = match Engine::new(bad) {
            Ok(_) => panic!("invalid config must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(matches!(err, AllpacaError::Fatal(FatalError::InvalidConfig(_))));

        let junk = std::env::temp_dir().join("allpaca_corrupt.gguf");
        std::fs::write(&junk, b"not a gguf").unwrap();
        let mut corrupt = Config::compiled_defaults();
        corrupt.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        corrupt.model.path = Some(junk.clone());
        let err = match Engine::new(corrupt) {
            Ok(_) => panic!("corrupt GGUF must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let _ = std::fs::remove_file(junk);
    }

    #[test]
    fn engine_fatal_on_missing_gguf() {
        let mut missing = Config::compiled_defaults();
        missing.merge_cli(&CliOverrides {
            cpu_only: true,
            model: Some(std::env::temp_dir().join("allpaca_missing_target.gguf")),
            ..CliOverrides::default()
        });
        assert_eq!(
            missing.model.path.as_deref(),
            Some(std::path::Path::new(
                &std::env::temp_dir().join("allpaca_missing_target.gguf")
            ))
        );
        let err = match Engine::new(missing) {
            Ok(_) => panic!("missing GGUF must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
    }

    #[test]
    fn nemotron_hybrid_fatal_reason_stable_synthetic_and_real() {
        // Probe path (always): synthetic hybrid SSM GGUF must Fatal with a stable
        // named reason — no silent skip when the real ROCmFP4 file is absent.
        {
            use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
            let path = std::env::temp_dir().join(format!(
                "allpaca_nemotron_hybrid_probe-{}-{}.gguf",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "general.architecture".into(),
                GgufValue::String("nemotron_h_moe".into()),
            );
            meta.insert(
                "nemotron_h_moe.block_count".into(),
                GgufValue::Uint32(1),
            );
            meta.insert(
                "tokenizer.ggml.model".into(),
                GgufValue::String("wordlevel".into()),
            );
            meta.insert(
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array(vec![
                    GgufValue::String("<unk>".into()),
                    GgufValue::String("<s>".into()),
                    GgufValue::String("</s>".into()),
                    GgufValue::String("hi".into()),
                ]),
            );
            meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
            meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
            let ssm = vec![0.0f32; 4];
            write_gguf(
                &path,
                &meta,
                &[GgufWriteTensor {
                    name: "blk.0.ssm_in.weight".into(),
                    dims: vec![4, 1],
                    data: &ssm,
                }],
            )
            .unwrap();
            let nvme_path = std::env::temp_dir().join(format!(
                "allpaca-nemotron-nvme-{}-{}.bin",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut cfg = Config::compiled_defaults();
            cfg.merge_cli(&CliOverrides {
                cpu_only: true,
                ..CliOverrides::default()
            });
            cfg.model.synthetic = false;
            cfg.model.path = Some(path.clone());
            // MoE + NVMe cold tier (GEM hybrid serving posture).
            cfg.weights.cache_bytes = 1 << 20;
            cfg.weights.host_bytes = 1 << 20;
            cfg.kv.nvme_bytes = 1 << 20;
            cfg.kv.nvme_path = Some(nvme_path.clone());
            let err = match Engine::new(cfg) {
                Ok(_) => panic!("synthetic nemotron_h_moe SSM must Fatal"),
                Err(e) => e,
            };
            assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
            let msg = err.to_string();
            assert!(
                msg.contains("nemotron_h_moe") && msg.contains("SSM"),
                "Fatal reason must stay stable (arch+SSM); got {msg}"
            );
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&nvme_path);
        }

        // Real GGUF under MoE+NVMe: clear Fatal (SSM + ggml_type 100) — not a green
        // generate path until SSM/FP4 land. Named skip if the file is absent.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf",
        );
        if !path.exists() {
            eprintln!(
                "skip nemotron real GGUF: place ROCmFP4 at {}; synthetic probe already asserted",
                path.display()
            );
            return;
        }
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path);
        cfg.weights.cache_bytes = 64 << 20;
        cfg.weights.host_bytes = 64 << 20;
        cfg.kv.nvme_bytes = 64 << 20;
        cfg.kv.nvme_path = Some(std::env::temp_dir().join(format!(
            "allpaca-nemotron-real-nvme-{}.bin",
            std::process::id()
        )));
        let nvme = cfg.kv.nvme_path.clone();
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("Nemotron hybrid/FP4 must not load for generate"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let msg = err.to_string();
        assert!(
            msg.contains("nemotron_h_moe") && msg.contains("SSM") && msg.contains("100"),
            "named Fatal must cite arch, SSM, and ggml_type_id 100; got {msg}"
        );
        if let Some(p) = nvme {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn engine_fatal_on_missing_or_invalid_safetensors() {
        let mut missing = Config::compiled_defaults();
        missing.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        missing.model.path = Some(std::env::temp_dir().join("allpaca_missing.safetensors"));
        let err = match Engine::new(missing) {
            Ok(_) => panic!("missing SafeTensors must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "a .safetensors path must not be treated as GGUF: {err}"
        );

        let junk = std::env::temp_dir().join("allpaca_corrupt.safetensors");
        std::fs::write(&junk, b"not a safetensors file").unwrap();
        let mut corrupt = Config::compiled_defaults();
        corrupt.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        corrupt.model.path = Some(junk.clone());
        let err = match Engine::new(corrupt) {
            Ok(_) => panic!("invalid SafeTensors must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "invalid SafeTensors must not surface as corrupt GGUF: {err}"
        );
        let _ = std::fs::remove_file(junk);
    }

    #[test]
    fn engine_fatal_on_missing_or_invalid_onnx() {
        let mut missing = Config::compiled_defaults();
        missing.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        missing.model.path = Some(std::env::temp_dir().join("allpaca_missing.onnx"));
        let err = match Engine::new(missing) {
            Ok(_) => panic!("missing ONNX must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "a .onnx path must not be treated as GGUF: {err}"
        );

        let junk = std::env::temp_dir().join("allpaca_corrupt.onnx");
        std::fs::write(&junk, b"not an onnx file").unwrap();
        let mut corrupt = Config::compiled_defaults();
        corrupt.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        corrupt.model.path = Some(junk.clone());
        let err = match Engine::new(corrupt) {
            Ok(_) => panic!("invalid ONNX must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(
            !matches!(err, AllpacaError::Fatal(FatalError::CorruptGguf(_))),
            "invalid ONNX must not surface as corrupt GGUF: {err}"
        );
        let _ = std::fs::remove_file(junk);
    }

    fn write_tiny_safetensors_checkpoint(dir: &std::path::Path) -> std::path::PathBuf {
        use allpaca_safetensors::{write_safetensors, SafetensorsDtype, SafetensorsWriteTensor};

        std::fs::create_dir_all(dir).unwrap();
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let kv_dim = 2usize;
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures":["LlamaForCausalLM"],"vocab_size":{vocab},"hidden_size":{dim},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":1,"intermediate_size":{ff},"rms_norm_eps":1e-5,"rope_theta":10000.0}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();

        let embd = vec![0.01f32; vocab * dim];
        let ones = vec![1.0f32; dim];
        let q = vec![0.01f32; dim * dim];
        let k = vec![0.01f32; kv_dim * dim];
        let v = vec![0.01f32; kv_dim * dim];
        let o = vec![0.01f32; dim * dim];
        let gate = vec![0.01f32; ff * dim];
        let up = vec![0.01f32; ff * dim];
        let down = vec![0.01f32; dim * ff];
        let path = dir.join("model.safetensors");
        let owned = [
            ("token_embd.weight", vec![vocab as u64, dim as u64], &embd, SafetensorsDtype::F32),
            ("blk.0.attn_norm.weight", vec![dim as u64], &ones, SafetensorsDtype::F32),
            ("blk.0.attn_q.weight", vec![dim as u64, dim as u64], &q, SafetensorsDtype::F32),
            ("blk.0.attn_k.weight", vec![kv_dim as u64, dim as u64], &k, SafetensorsDtype::F32),
            ("blk.0.attn_v.weight", vec![kv_dim as u64, dim as u64], &v, SafetensorsDtype::F32),
            ("blk.0.attn_output.weight", vec![dim as u64, dim as u64], &o, SafetensorsDtype::F32),
            ("blk.0.ffn_norm.weight", vec![dim as u64], &ones, SafetensorsDtype::F32),
            ("blk.0.ffn_gate.weight", vec![ff as u64, dim as u64], &gate, SafetensorsDtype::F32),
            ("blk.0.ffn_up.weight", vec![ff as u64, dim as u64], &up, SafetensorsDtype::F32),
            ("blk.0.ffn_down.weight", vec![dim as u64, ff as u64], &down, SafetensorsDtype::F32),
            ("output_norm.weight", vec![dim as u64], &ones, SafetensorsDtype::F32),
        ];
        let views: Vec<SafetensorsWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data, dtype)| SafetensorsWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                dtype: *dtype,
                data,
            })
            .collect();
        write_safetensors(&path, &views).unwrap();
        path
    }

    #[test]
    fn engine_generate_from_tiny_safetensors() {
        let dir = std::env::temp_dir().join("allpaca_tiny_safetensors_ckpt");
        let path = write_tiny_safetensors_checkpoint(&dir);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("SafeTensors checkpoint must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on SafeTensors weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    fn write_tiny_onnx_checkpoint(dir: &std::path::Path) -> std::path::PathBuf {
        use allpaca_onnx::{write_onnx_weights, OnnxWriteTensor};

        std::fs::create_dir_all(dir).unwrap();
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let kv_dim = 2usize;
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures":["LlamaForCausalLM"],"vocab_size":{vocab},"hidden_size":{dim},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":1,"intermediate_size":{ff},"rms_norm_eps":1e-5,"rope_theta":10000.0}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();
        let embd = vec![0.01f32; vocab * dim];
        let ones = vec![1.0f32; dim];
        let q = vec![0.01f32; dim * dim];
        let k = vec![0.01f32; kv_dim * dim];
        let v = vec![0.01f32; kv_dim * dim];
        let o = vec![0.01f32; dim * dim];
        let gate = vec![0.01f32; ff * dim];
        let up = vec![0.01f32; ff * dim];
        let down = vec![0.01f32; dim * ff];
        let path = dir.join("model.onnx");
        let owned = [
            ("token_embd.weight", vec![vocab as i64, dim as i64], &embd),
            ("blk.0.attn_norm.weight", vec![dim as i64], &ones),
            ("blk.0.attn_q.weight", vec![dim as i64, dim as i64], &q),
            ("blk.0.attn_k.weight", vec![kv_dim as i64, dim as i64], &k),
            ("blk.0.attn_v.weight", vec![kv_dim as i64, dim as i64], &v),
            ("blk.0.attn_output.weight", vec![dim as i64, dim as i64], &o),
            ("blk.0.ffn_norm.weight", vec![dim as i64], &ones),
            ("blk.0.ffn_gate.weight", vec![ff as i64, dim as i64], &gate),
            ("blk.0.ffn_up.weight", vec![ff as i64, dim as i64], &up),
            ("blk.0.ffn_down.weight", vec![dim as i64, ff as i64], &down),
            ("output_norm.weight", vec![dim as i64], &ones),
        ];
        let views: Vec<OnnxWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| OnnxWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                data,
            })
            .collect();
        write_onnx_weights(&path, "tiny-llama", &views, &[]).unwrap();
        path
    }

    #[test]
    fn engine_generate_from_tiny_onnx() {
        let dir = std::env::temp_dir().join("allpaca_tiny_onnx_ckpt");
        let path = write_tiny_onnx_checkpoint(&dir);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("ONNX weight-export must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on ONNX initializer weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_fatal_on_unsupported_onnx_graph() {
        use allpaca_onnx::{write_onnx_weights, OnnxWriteNode, OnnxWriteTensor};

        let dir = std::env::temp_dir().join("allpaca_onnx_bad_op");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":4,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":4,"intermediate_size":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();
        let w = [1.0f32, 2.0];
        let path = dir.join("model.onnx");
        write_onnx_weights(
            &path,
            "conv-net",
            &[OnnxWriteTensor {
                name: "w".into(),
                dims: vec![2],
                data: &w,
            }],
            &[OnnxWriteNode {
                op_type: "Conv".into(),
                name: "conv0".into(),
            }],
        )
        .unwrap();
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("unsupported ONNX graph must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let msg = err.to_string();
        assert!(
            msg.contains("Conv") && !msg.to_lowercase().contains("panic"),
            "Fatal error must name the unsupported op: {msg}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn engine_fatal_on_unsupported_safetensors_dtype() {
        let dir = std::env::temp_dir().join("allpaca_st_bad_dtype");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"vocab_size":4,"hidden_size":8,"num_hidden_layers":1,"num_attention_heads":4,"intermediate_size":16}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"tokens":["<unk>","<s>","</s>","hello"],"bos_id":1,"eos_id":2}"#,
        )
        .unwrap();
        let header = r#"{"token_embd.weight":{"dtype":"I32","shape":[4,8],"data_offsets":[0,128]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 128]);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path);
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("unsupported SafeTensors dtype must be Fatal"),
            Err(e) => e,
        };
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        let msg = err.to_string();
        assert!(
            msg.contains("I32") && msg.to_lowercase().contains("unsupported"),
            "Fatal error must name the unsupported dtype: {msg}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    fn write_tiny_q4k_gguf(path: &std::path::Path) {
        use allpaca_gguf::{
            quantize_q4_k, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
            GgufWriteTensor,
        };
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

        let embd_q = quantize_q4_k(&vec![0.01f32; vocab * dim]);
        let q_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
        let k_q = quantize_q4_k(&vec![0.01f32; 2 * dim]);
        let v_q = quantize_q4_k(&vec![0.01f32; 2 * dim]);
        let o_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
        let gate_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
        let up_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
        let down_q = quantize_q4_k(&vec![0.01f32; dim * ff]);
        let attn_norm = vec![1.0f32; dim];
        let ffn_norm = vec![1.0f32; dim];
        let output_norm = vec![1.0f32; dim];

        let f32_tensors = [
            GgufWriteTensor {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &attn_norm,
            },
            GgufWriteTensor {
                name: "blk.0.ffn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &ffn_norm,
            },
            GgufWriteTensor {
                name: "output_norm.weight".into(),
                dims: vec![dim as u64],
                data: &output_norm,
            },
        ];
        let quant_tensors = [
            GgufWriteQuantTensor {
                name: "token_embd.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &down_q,
            },
        ];
        write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).unwrap();
    }

    fn write_tiny_uniform_kquant_gguf(
        path: &std::path::Path,
        ggml_type: allpaca_gguf::GgmlType,
        quantize: fn(&[f32]) -> Vec<u8>,
    ) {
        use allpaca_gguf::{write_gguf_quant, GgufValue, GgufWriteQuantTensor, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

        let embd_q = quantize(&vec![0.01f32; vocab * dim]);
        let q_q = quantize(&vec![0.01f32; dim * dim]);
        let k_q = quantize(&vec![0.01f32; 2 * dim]);
        let v_q = quantize(&vec![0.01f32; 2 * dim]);
        let o_q = quantize(&vec![0.01f32; dim * dim]);
        let gate_q = quantize(&vec![0.01f32; ff * dim]);
        let up_q = quantize(&vec![0.01f32; ff * dim]);
        let down_q = quantize(&vec![0.01f32; dim * ff]);
        let attn_norm = vec![1.0f32; dim];
        let ffn_norm = vec![1.0f32; dim];
        let output_norm = vec![1.0f32; dim];

        let f32_tensors = [
            GgufWriteTensor {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &attn_norm,
            },
            GgufWriteTensor {
                name: "blk.0.ffn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &ffn_norm,
            },
            GgufWriteTensor {
                name: "output_norm.weight".into(),
                dims: vec![dim as u64],
                data: &output_norm,
            },
        ];
        let quant_tensors = [
            GgufWriteQuantTensor {
                name: "token_embd.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type,
                bytes: &down_q,
            },
        ];
        write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).unwrap();
    }

    #[test]
    fn engine_generate_from_q4k_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_q4k.gguf");
        write_tiny_q4k_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("Q4_K GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on dequantized Q4_K weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_generate_from_f16_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_f16.gguf");
        write_tiny_uniform_kquant_gguf(&path, allpaca_gguf::GgmlType::F16, allpaca_gguf::quantize_f16);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("F16 GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on F16 weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_generate_from_q6k_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_q6k.gguf");
        write_tiny_uniform_kquant_gguf(&path, allpaca_gguf::GgmlType::Q6_K, allpaca_gguf::quantize_q6_k);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("Q6_K GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on dequantized Q6_K weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_generate_from_q5k_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_q5k.gguf");
        write_tiny_uniform_kquant_gguf(&path, allpaca_gguf::GgmlType::Q5_K, allpaca_gguf::quantize_q5_k);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("Q5_K GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on dequantized Q5_K weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    fn write_tiny_q4k_q6k_gguf(path: &std::path::Path) {
        use allpaca_gguf::{
            quantize_q4_k, quantize_q6_k, write_gguf_quant, GgmlType, GgufValue,
            GgufWriteQuantTensor, GgufWriteTensor,
        };
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(vec![
                GgufValue::String("<unk>".into()),
                GgufValue::String("<s>".into()),
                GgufValue::String("</s>".into()),
                GgufValue::String("hello".into()),
            ]),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

        let embd_q = quantize_q4_k(&vec![0.01f32; vocab * dim]);
        let q_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
        let k_q = quantize_q4_k(&vec![0.01f32; 2 * dim]);
        let v_q = quantize_q6_k(&vec![0.01f32; 2 * dim]);
        let o_q = quantize_q4_k(&vec![0.01f32; dim * dim]);
        let gate_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
        let up_q = quantize_q4_k(&vec![0.01f32; ff * dim]);
        let down_q = quantize_q6_k(&vec![0.01f32; dim * ff]);
        let output_q = quantize_q6_k(&vec![0.01f32; vocab * dim]);
        let attn_norm = vec![1.0f32; dim];
        let ffn_norm = vec![1.0f32; dim];
        let output_norm = vec![1.0f32; dim];

        let f32_tensors = [
            GgufWriteTensor {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &attn_norm,
            },
            GgufWriteTensor {
                name: "blk.0.ffn_norm.weight".into(),
                dims: vec![dim as u64],
                data: &ffn_norm,
            },
            GgufWriteTensor {
                name: "output_norm.weight".into(),
                dims: vec![dim as u64],
                data: &output_norm,
            },
        ];
        let quant_tensors = [
            GgufWriteQuantTensor {
                name: "token_embd.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &embd_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &q_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_k.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &k_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_v.weight".into(),
                dims: vec![2, dim as u64],
                ggml_type: GgmlType::Q6_K,
                bytes: &v_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.attn_output.weight".into(),
                dims: vec![dim as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &o_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_gate.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &gate_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_up.weight".into(),
                dims: vec![ff as u64, dim as u64],
                ggml_type: GgmlType::Q4_K,
                bytes: &up_q,
            },
            GgufWriteQuantTensor {
                name: "blk.0.ffn_down.weight".into(),
                dims: vec![dim as u64, ff as u64],
                ggml_type: GgmlType::Q6_K,
                bytes: &down_q,
            },
            GgufWriteQuantTensor {
                name: "output.weight".into(),
                dims: vec![vocab as u64, dim as u64],
                ggml_type: GgmlType::Q6_K,
                bytes: &output_q,
            },
        ];
        write_gguf_quant(path, &meta, &f32_tensors, &quant_tensors).unwrap();
    }

    #[test]
    fn engine_generate_from_mixed_q4k_q6k_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_q4k_q6k.gguf");
        write_tiny_q4k_q6k_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("mixed Q4_K+Q6_K GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on mixed Q4_K+Q6_K weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_generate_from_q4_q8_gguf() {
        let path = std::env::temp_dir().join("allpaca_target_q4q8.gguf");
        write_tiny_quant_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("Q4_0/Q8_0 GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("generate must work on dequantized Q4/Q8 weights");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_speculative_linear_verify_packs_draft_span() {
        let target = std::env::temp_dir().join("allpaca_pack_verify_tgt.gguf");
        let draft = std::env::temp_dir().join("allpaca_pack_verify_dft.gguf");
        write_tiny_target_gguf(&target);
        write_tiny_target_gguf(&draft);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(target.clone());
        cfg.model.draft_path = Some(draft.clone());
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 4;
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("identical tiny target+draft must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("packed linear verify generate must run");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.linear_packed_verifies > 0,
            "greedy identical target+draft must pack the linear draft span; gen={} acceptance={:.3} stats={stats:?}",
            out.token_ids.len(),
            stats.acceptance_rate
        );
        assert!(
            stats.draft_ahead_hits > 0,
            "full-accept packed verify must overlap the next Draft Model propose; gen={} stats={stats:?}",
            out.token_ids.len()
        );
        let mut plain = Config::compiled_defaults();
        plain.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        plain.model.path = Some(target.clone());
        plain.kv.page_tokens = 8;
        let plain_engine = Engine::new(plain).expect("plain target must load");
        let plain_out = plain_engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("plain generate must run");
        assert_eq!(
            out.token_ids, plain_out.token_ids,
            "packed linear verify must match greedy Target Model tokens"
        );
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(draft);
    }

    #[test]
    fn engine_speculative_reuses_draft_ahead_after_partial_accept() {
        let target = std::env::temp_dir().join("allpaca_pack_partial_tgt.gguf");
        let draft = std::env::temp_dir().join("allpaca_pack_partial_dft.gguf");
        write_tiny_llama_gguf(&target, 0.01, 0.01);
        write_tiny_llama_gguf(&draft, 0.01, 2.0);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(target.clone());
        cfg.model.draft_path = Some(draft.clone());
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 4;
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("mismatched tiny target+draft must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 16,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("partial-accept packed verify must generate");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.linear_packed_verifies > 0,
            "first draft token must match often enough to pack; stats={stats:?}"
        );
        assert!(
            stats.draft_ahead_hits > 0,
            "packed verify must queue the next Draft span even when later tokens diverge; gen={} acceptance={:.3} stats={stats:?}",
            out.token_ids.len(),
            stats.acceptance_rate
        );
        let mut plain = Config::compiled_defaults();
        plain.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        plain.model.path = Some(target.clone());
        plain.kv.page_tokens = 8;
        let plain_engine = Engine::new(plain).expect("plain target must load");
        let plain_out = plain_engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 16,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("plain generate must run");
        assert_eq!(
            out.token_ids, plain_out.token_ids,
            "partial-accept verify must keep Target Model tokens"
        );
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(draft);
    }

    #[test]
    fn engine_speculative_generate_from_tiny_written_draft_gguf() {
        let target = std::env::temp_dir().join("allpaca_target_for_draft.gguf");
        let draft = std::env::temp_dir().join("allpaca_tiny_draft.gguf");
        write_tiny_target_gguf(&target);
        write_tiny_target_gguf(&draft);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(target.clone());
        cfg.model.draft_path = Some(draft.clone());
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("tiny target+draft GGUF must load");
        assert_eq!(engine.stats().degraded_events, 0);
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("speculative generate with a written draft GGUF must run");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(draft);
    }

    #[test]
    fn engine_degraded_when_draft_gguf_missing() {
        let target = std::env::temp_dir().join("allpaca_target_tiny.gguf");
        write_tiny_target_gguf(&target);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(target.clone());
        cfg.model.draft_path = Some(std::env::temp_dir().join("allpaca_missing_draft.gguf"));
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("target GGUF loads; missing draft is Degraded");
        assert!(engine.stats().degraded_events > 0);
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("Degraded draft failure must continue on the Target Model");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(target);
    }

    #[test]
    fn engine_generate_from_gguf_still_works_alongside_safetensors() {
        let path = std::env::temp_dir().join("allpaca_gguf_regression.gguf");
        write_tiny_target_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("GGUF path must still load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("GGUF generate must still work");
        assert!(!out.token_ids.is_empty());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_generate_from_moe_gguf_loads_stacked_experts() {
        let path = std::env::temp_dir().join("allpaca_moe_gguf.gguf");
        write_tiny_moe_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.kv.page_tokens = 8;
        cfg.weights.cache_bytes = 64;
        cfg.weights.miss_policy = MissPolicy::Fill;
        let engine = Engine::new(cfg).expect("Mixtral-style GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("MoE GGUF generate must work");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.expert_slots >= 1,
            "cache_bytes on a Mixtral-style GGUF must grant expert slots; stats={stats:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn json_mode_emits_only_legal_tokens() {
        let engine = test_engine();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 12,
                    temperature: 0.0,
                    json_mode: true,
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let tok = EngineTokenizer::synthetic(64).unwrap();
        let mut prefix = String::new();
        for id in &out.token_ids {
            let piece = tok.decode_one(*id);
            if is_special_token(&piece) {
                continue;
            }
            prefix.push_str(&piece);
            assert!(
                json_prefix_legal(&prefix),
                "Constraint Mask allowed illegal JSON after {prefix:?} (+{piece:?})"
            );
        }
        assert!(
            prefix.starts_with('{')
                || prefix.starts_with('[')
                || prefix.starts_with('"')
                || prefix.starts_with('-')
                || prefix.starts_with(|c: char| c.is_ascii_digit())
                || prefix.starts_with("true")
                || prefix.starts_with("false")
                || prefix.starts_with("null")
                || prefix.is_empty(),
            "JSON mode must start with a JSON value, got {prefix:?}"
        );
    }

    #[test]
    fn async_schema_mask_hits_and_stays_valid() {
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.structured_output.async_masks = true;
        let engine = Engine::new(cfg).unwrap();
        let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 16,
                    temperature: 0.0,
                    json_schema: Some(schema.into()),
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let value: serde_json::Value = serde_json::from_str(&out.text).unwrap();
        assert!(value.get("ok").is_some_and(|v| v.is_boolean()));
        let stats = engine.stats();
        assert!(
            stats.async_mask_hits > 0,
            "structured_output.async must serve at least one prefetched mask; stats={stats:?}"
        );
    }

    #[test]
    fn json_schema_emits_only_schema_valid_object() {
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let schema = r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#;
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 16,
                    temperature: 0.0,
                    json_schema: Some(schema.into()),
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.text.is_empty(), "schema generate must emit text");
        let value: serde_json::Value =
            serde_json::from_str(&out.text).unwrap_or_else(|e| panic!("schema output must parse as JSON: {e}; text={:?}", out.text));
        let obj = value
            .as_object()
            .unwrap_or_else(|| panic!("schema output must be an object, got {value}"));
        assert_eq!(obj.len(), 1, "additionalProperties:false, got {value}");
        assert!(
            obj.get("ok").is_some_and(|v| v.is_boolean()),
            "required boolean field ok, got {value}"
        );
    }

    #[test]
    fn lark_grammar_emits_only_true_or_false() {
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    grammar: Some(r#"start: "true" | "false""#.into()),
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let text = out.text.trim();
        assert!(
            text == "true" || text == "false",
            "lark grammar must force true|false, got {text:?}"
        );
    }
}
