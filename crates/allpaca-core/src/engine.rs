use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use allpaca_gguf::{discover_dspark_sidecar, extract_tokenizer, is_dspark_filename, GgufFile};
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{
    discover_checkpoint, load_hf_config, load_tokenizer_sidecar, resolve_sidecar, CheckpointKind,
    SafetensorsCatalog,
};
use allpaca_types::{
    AllpacaError, Config, DegradedError, DraftProviderKind, FatalError, FeatureFlags, FinishReason,
    GenerationOutput, GenerationParams, KvHandle, StatsSnapshot, StepResult, TokenEvent,
    TransientError,
};
use futures::Stream;
use rand::rngs::StdRng;

use crate::anchors::{restore_kv, AnchorCache, RecurrentState};
use crate::batch_gemm::{
    apply_packed_attn_hip, apply_packed_ffn_hip, apply_packed_lm_head, batch_gemm_other_ids,
    decode_layer_attn_shared_a_weights, packed_gemm_nt_hip, plan_batch_gemm, residual_add_rows,
    write_residual_into_hidden, BatchGemmCandidate, BatchGemmPhase, BatchGemmPlan, ModelShapeKey,
};
use crate::constraint::JsonConstraint;
use crate::device_manager::DeviceInventory;
use crate::direct_storage::{
    probe_direct_storage, DirectStorageBridge, DirectStorageProbe, DirectStorageQueue,
};
use crate::draft::{DraftModelProvider, DraftProvider, NGramProvider};
use crate::early_exit::gate_unsupported;
#[cfg(test)]
use crate::early_exit::EarlyExitSpec;
use crate::expert_batch::MicroBatchExpertUnion;
use crate::forest::{build_token_forest, TokenForest};
use crate::kv_radix::KvRadixForest;
use crate::kv_tier::KvTierManager;
use crate::logits_ring::LogitRing;
use crate::micro_batch::{plan_micro_batch, ActiveLayerHint, LayerStagger, StaggerPhase};
use crate::model::{
    alloc_pages_for_seq, ensure_pages, DenseKv, DeviceDecodeSide, ModelConfig, ResidentModel,
    TinyModel,
};
use crate::packed::{temp_pack_path, HostExpertPool, PackedStore};
use crate::sampling::{rng_from_params, sample};
use crate::scheduler::Scheduler;
use crate::telemetry::{GpuSampler, Metrics};
use crate::tokenizer::EngineTokenizer;
use crate::weight_set::{
    expert_lru_slots, host_pool_slots, DenseTrunkPlan, ExpertId, ExpertLru, LayerFfn,
    WeightWorkingSet,
};
use allpaca_hip::HipWorkspace;

/// What a decision request reads while its prompt is forwarded.
struct DecisionProbe {
    /// Tokens whose logits after the last prompt token are the scores.
    labels: Vec<u32>,
    /// Prompt positions whose final hidden state is kept, ascending; the
    /// last one is the last prompt token.
    positions: Vec<usize>,
    /// One `dim` row per position reached so far.
    rows: Vec<f32>,
    label_logits: Vec<f32>,
    done: bool,
}

impl DecisionProbe {
    fn new(read: &crate::decision::DecisionRead, tokens: &[u32]) -> Self {
        let (labels, positions) = match read {
            crate::decision::DecisionRead::Labels(labels) => (labels.clone(), Vec::new()),
            crate::decision::DecisionRead::Pointer { marker } => {
                let last = tokens.len().saturating_sub(1);
                let mut positions: Vec<usize> = (0..last).filter(|&i| tokens[i] == *marker).collect();
                positions.push(last);
                (Vec::new(), positions)
            }
        };
        Self {
            labels,
            positions,
            rows: Vec::new(),
            label_logits: Vec::new(),
            done: false,
        }
    }
}

struct ActiveRequest {
    id: u64,
    tokens: Vec<u32>,
    prompt_len: usize,
    prefill_pos: usize,
    generated: usize,
    params: GenerationParams,
    pages: Vec<Vec<u64>>,
    pending_logits: Vec<f32>,
    /// Decision request: no token is sampled, the prompt's forward pass is
    /// read instead.
    probe: Option<DecisionProbe>,
    emit_queue: Vec<TokenEvent>,
    /// StepResults parked by a peer thread's fair `poll` demux.
    parked: VecDeque<StepResult>,
    /// Prefetched grammar mask from a background thread (`structured_output.async`).
    async_mask_rx: Option<std::sync::mpsc::Receiver<Result<allpaca_types::TokenMask, String>>>,
    json_prefix: String,
    /// Holds a UTF-8 character split across tokens until it is complete.
    utf8: crate::tokenizer::Utf8Stream,
    grammar: Option<llguidance::Matcher>,
    rng: StdRng,
    started: Instant,
    last_token_at: Instant,
    first_emitted: bool,
    finished: bool,
    recurrent: Option<RecurrentState>,
    /// Decode steps completed (for layer-stagger micro-batch hints).
    decode_step: u64,
    /// Last MoE router input (FFN RMSNorm) from the most recent decode forward.
    moe_norm: Vec<f32>,
    /// Target hidden: `output_norm` after decode; packed HIP residual may
    /// overwrite with `x+WO+FFN` when MTP is off (`h_nextn` stays `output_norm`).
    last_hidden: Vec<f32>,
    /// `last_hidden` was written by the most recent decode forward.
    last_hidden_fresh: bool,
    /// This tick's HIP decode cohort will pack lm_head; skip the per-request GEMM.
    defer_lm_head: bool,
    /// Correction token from a rejected packed verify: already emitted, not yet
    /// forwarded (`pending_logits` are stale while set). The next step verifies
    /// it at the head of the drafted span, or forwards it alone.
    deferred: Option<u32>,
    /// Soft vision tokens (image rows instead of token embeddings) and their
    /// M-RoPE layout; empty for text-only Requests.
    vision: Arc<crate::model::VisionSeq>,
    /// When vision encode landed on the draft GPU, skip draft propose for this Request.
    suppress_draft: bool,
}

/// What one [`Engine::poll_for`] tick yielded for the calling Request.
enum PollFor {
    /// This Request's next result.
    Result(StepResult),
    /// Finished with nothing left to hand out.
    Done,
    /// The tick served another Request (or found nothing runnable).
    Idle,
}

/// A Request's rows awaiting the batched decode forward: its sampled token,
/// or a packed n-gram verify span (`span` then holds the rest of the verify).
struct StagedDecode {
    id: u64,
    tokens: Vec<u32>,
    pos: usize,
    span: Option<Box<SpanTail>>,
}

/// Rows one batched decode forward may carry. `forward_batch` runs its GEMMs
/// row-exact (`allpaca_hip::ExactRows`: wide M as chunks on the small-M
/// kernels), so any count reproduces solo rows; this bounds the per-tick
/// workspace and keeps four Requests' n-gram spans (up to 8 rows each) in
/// one forward.
const BATCH_EXACT_ROWS: usize = 32;

/// Whether `rows` more fit in the batched forward being staged (`None`: not
/// staging at all).
fn stage_fits(inner: &EngineInner, rows: usize) -> bool {
    inner.staged_decode.as_ref().is_some_and(|staged| {
        staged.iter().map(|st| st.tokens.len()).sum::<usize>() + rows <= BATCH_EXACT_ROWS
    })
}

/// What a packed linear verify carries from its forward to its accept walk
/// and rewind ([`finish_linear_span`]).
struct SpanTail {
    span: Vec<u32>,
    first: u32,
    remaining: usize,
    start_pos: usize,
    page_tokens: u32,
    last_logits: Vec<f32>,
    rec_before: Option<RecurrentState>,
    device_gdn_cloned: usize,
    journal_rewind: bool,
    pending_h: Vec<f32>,
    /// The span is headed by an already-emitted deferred correction, which
    /// is dropped from the produced tokens.
    drop_first: bool,
}

struct EngineInner {
    flags: FeatureFlags,
    scheduler: Scheduler,
    kv: KvTierManager,
    ring: LogitRing,
    tokenizer: Option<EngineTokenizer>,
    /// Set when the loaded GGUF is a decision model (`<arch>.decision.type`).
    decision: Option<Arc<crate::decision::DecisionModel>>,
    target: Option<Arc<TinyModel>>,
    draft_model: Option<Arc<TinyModel>>,
    ngram: NGramProvider,
    dspark: Option<crate::dspark::DsparkProvider>,
    dflash2: Option<crate::dflash2::Dflash2Provider>,
    active: HashMap<u64, ActiveRequest>,
    next_request_id: u64,
    model_id: String,
    target_resident: Option<Arc<ResidentModel>>,
    draft_resident: Option<Arc<ResidentModel>>,
    #[cfg(feature = "cuda")]
    target_cuda_resident: Option<Arc<crate::CudaResidentModel>>,
    #[cfg(feature = "vulkan")]
    target_vulkan_resident: Option<Arc<crate::VulkanResidentModel>>,
    draft_kv: DenseKv,
    draft_ahead: Option<DraftAhead>,
    anchors: AnchorCache,
    /// GEM radix prefix KV reuse (arbitrary mid-prompt edits).
    kv_radix: KvRadixForest,
    /// Model and expert mask the prompt cache holds checkpoints for.
    prompt_cache_key: Option<u64>,
    /// DirectStorage / GDS capability probe at engine start.
    ds_probe: DirectStorageProbe,
    grammar_factory: Option<llguidance::ParserFactory>,
    /// Round-robin cursor for continuous-batching ties.
    last_poll_id: Option<u64>,
    /// Consecutive prefill picks under `PrefillBudget` schedule.
    consecutive_prefill_picks: usize,
    /// Next poll prefers this request (cross-req micro-batch partner).
    prefer_next: Option<u64>,
    /// Request whose next-token logits may still be in flight on the HIP
    /// workspace's single pinned logits buffer (decode starts the D2H and the
    /// next step of the same Request waits on it).
    logits_owner: Option<u64>,
    /// Set while `poll` batches decode: `append_and_forward` records the
    /// token here instead of running its own forward.
    staged_decode: Option<Vec<StagedDecode>>,
    /// Consecutive decode steps with expert LRU misses (elastic pressure).
    elastic_miss_streak: u32,
    /// Run [`Engine::rebalance_experts_elastic`] on next idle safe point.
    elastic_rebalance_pending: bool,
    /// Prior expert LRU miss counter for streak detection.
    elastic_last_misses: u64,
    /// Optional vision encoder (placement + encode); text-only ignores it.
    vision_encoder: Option<Arc<dyn crate::VisionEncoder>>,
}

struct DraftAhead {
    ctx: Vec<u32>,
    tokens: Vec<u32>,
}

struct DecodePollHooks<'a> {
    layer_stagger: Option<&'a LayerStagger>,
    ds_probe: &'a DirectStorageProbe,
    ds_queue: &'a DirectStorageQueue,
    #[cfg(feature = "cuda")]
    cuda: Option<&'a allpaca_cuda::CudaWorkspace>,
    #[cfg(feature = "vulkan")]
    vulkan: Option<&'a allpaca_vulkan::VulkanWorkspace>,
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<Mutex<EngineInner>>,
    config: Config,
    inventory: DeviceInventory,
    metrics: Arc<Metrics>,
    hip: Option<Arc<HipWorkspace>>,
    draft_hip: Option<Arc<HipWorkspace>>,
    #[cfg(feature = "cuda")]
    cuda: Option<Arc<allpaca_cuda::CudaWorkspace>>,
    #[cfg(feature = "vulkan")]
    vulkan: Option<Arc<allpaca_vulkan::VulkanWorkspace>>,
    /// Shared DirectStorage enqueue ledger (file-coordinate + staged H2D).
    ds_queue: Arc<DirectStorageQueue>,
    ds_probe: DirectStorageProbe,
    micro_batch: Arc<MicroBatchExpertUnion>,
    layer_stagger: Arc<LayerStagger>,
    /// Clock for driver VRAM / GPU-busy sampling. Shared across clones so the
    /// rate limit is per engine, not per handle.
    gpu_sampler: Arc<GpuSampler>,
    rpc: Option<Arc<crate::rpc_runtime::RpcRuntime>>,
    /// Serializes `poll` ticks across threads: a batched decode tick steps
    /// several Requests with `inner` released in between, and no other tick
    /// may step them before their batched forward lands.
    poll_tick: Arc<Mutex<()>>,
    /// Last values `stats` / `model_id` read under `inner`, served while a
    /// prefill holds the engine lock.
    read_side: Arc<Mutex<ReadSide>>,
}

/// Starts the GPU BLAS library on `device` from a background thread while the
/// model loads, so its one-time start-up is not paid by the first request. A
/// request that arrives first waits for it, as it would have anyway.
fn spawn_blas_warm_up(device: i32) {
    std::thread::spawn(move || {
        let t = std::time::Instant::now();
        match allpaca_hip::warm_up_blas(device) {
            Ok(()) => {
                tracing::info!(device, ms = t.elapsed().as_millis() as u64, "GPU BLAS warmed up")
            }
            Err(e) => tracing::debug!(device, error = %e, "GPU BLAS warm-up skipped"),
        }
    });
}

/// How long a monitoring read waits for the engine lock. Decode releases it
/// between tokens; a prefill holds it for the whole time to first token, and
/// a dashboard poll must not hang that long.
const READ_LOCK_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

#[derive(Clone, Default)]
struct ReadSide {
    model_id: Option<String>,
    target_name: Option<String>,
    kv_h2d: u64,
    kv_d2h: u64,
    draft_kv_bytes: u64,
    draft_depth: u64,
}

pub struct RequestHandle {
    id: u64,
    inner: Arc<Mutex<EngineInner>>,
    metrics: Arc<Metrics>,
    hip: Option<Arc<HipWorkspace>>,
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
        f.debug_struct("RequestHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if let Some(hip) = self.hip.as_ref() {
            let target = self.inner.lock().ok().and_then(|g| g.target.clone());
            let _ = for_each_gdn_workspace(hip, target.as_deref(), |ws| {
                ws.gdn_clear_request(self.id);
                Ok(())
            });
        }
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
            // Safe point: last client gone while elastic pressure was recorded.
            if guard.active.is_empty() && guard.elastic_rebalance_pending {
                guard.elastic_rebalance_pending = true;
            }
        }
    }
}

fn rpc_hello_token(config: &Config) -> String {
    if let Some(t) = config
        .rpc
        .token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return t.to_string();
    }
    std::env::var("ALLPACA_RPC_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default()
}

fn scale_kv_vram_from_inventory(config: &mut Config, inventory: &DeviceInventory) {
    let compiled = allpaca_types::KvSection::default().vram_bytes;
    if config.kv.vram_bytes != compiled {
        return;
    }
    let gpu_mem = inventory
        .devices
        .iter()
        .filter(|d| d.is_gpu() && !d.is_rpc())
        .map(|d| d.memory_bytes)
        .max()
        .unwrap_or(0);
    let suggested = crate::suggested_kv_vram_bytes(gpu_mem);
    if suggested > config.kv.vram_bytes {
        tracing::info!(
            suggested,
            gpu_mem,
            mode = ?inventory.pipeline_mode,
            "scaled kv.vram_bytes from GPU capacity"
        );
        config.kv.vram_bytes = suggested;
    }
}

#[cfg(test)]
mod scale_kv_tests {
    use super::scale_kv_vram_from_inventory;
    use crate::device_manager::DeviceInventory;
    use allpaca_types::{Config, DeviceInfo, DeviceKind, PipelineMode, TransferPath};

    fn cuda_12gb_inventory() -> DeviceInventory {
        DeviceInventory {
            devices: vec![
                DeviceInfo::cpu(),
                DeviceInfo {
                    id: 0,
                    kind: DeviceKind::CudaGpu,
                    name: "RTX 3060".into(),
                    memory_bytes: 12 << 30,
                    bandwidth_gbps: 360.0,
                    peers: vec![],
                },
            ],
            pipeline_mode: PipelineMode::MobileTiered,
            transfer_path: TransferPath::HostStaging,
            transfer_benchmark: None,
            p2p_capable: false,
            p2p_enabled: false,
            target_device: 0,
            draft_device: -1,
            vision_device: None,
        }
    }

    #[test]
    fn scales_compiled_default_kv_for_12gb_cuda() {
        let mut cfg = Config::compiled_defaults();
        assert_eq!(cfg.kv.vram_bytes, 4 * 1024 * 1024);
        scale_kv_vram_from_inventory(&mut cfg, &cuda_12gb_inventory());
        assert_eq!(cfg.kv.vram_bytes, 4usize << 30);
    }

    #[test]
    fn scales_8gb_laptop_to_half_vram() {
        let mut inv = cuda_12gb_inventory();
        inv.devices[1].name = "Ada laptop".into();
        inv.devices[1].memory_bytes = 8 << 30;
        let mut cfg = Config::compiled_defaults();
        scale_kv_vram_from_inventory(&mut cfg, &inv);
        assert_eq!(cfg.kv.vram_bytes, 4usize << 30);
    }

    #[test]
    fn scales_6gb_card_below_cap() {
        let mut inv = cuda_12gb_inventory();
        inv.devices[1].memory_bytes = 6 << 30;
        let mut cfg = Config::compiled_defaults();
        scale_kv_vram_from_inventory(&mut cfg, &inv);
        assert_eq!(cfg.kv.vram_bytes, 3usize << 30);
    }

    #[test]
    fn preserves_explicit_kv_override() {
        let mut cfg = Config::compiled_defaults();
        cfg.kv.vram_bytes = 256 * 1024 * 1024;
        scale_kv_vram_from_inventory(&mut cfg, &cuda_12gb_inventory());
        assert_eq!(cfg.kv.vram_bytes, 256 * 1024 * 1024);
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
        config.apply_tune_overlay();
        config.validate()?;
        let mut inventory = DeviceInventory::discover(&config);
        scale_kv_vram_from_inventory(&mut config, &inventory);
        let rpc = if config.rpc.endpoints.is_empty() {
            None
        } else {
            let token = rpc_hello_token(&config);
            let rt = crate::rpc_runtime::RpcRuntime::connect(
                &config.rpc.endpoints,
                if token.is_empty() {
                    None
                } else {
                    Some(token.as_str())
                },
            )?;
            rt.apply_hello_to_inventory(&mut inventory.devices)?;
            Some(Arc::new(rt))
        };
        let mut flags = FeatureFlags::from_mode(inventory.pipeline_mode);
        flags.overlay(&config.features);
        apply_provider_flags(&mut flags, &config);
        let scheduler = Scheduler::new(inventory.pipeline_mode, flags.clone(), &config.speculative);
        let hip = open_hip_workspace(&config, &inventory).map(Arc::new);
        let draft_hip = open_draft_hip_workspace(&config, &inventory).map(Arc::new);
        #[cfg(feature = "cuda")]
        let cuda = open_cuda_workspace(&config, &inventory).map(Arc::new);
        #[cfg(feature = "vulkan")]
        let vulkan = open_vulkan_workspace(&config, &inventory).map(Arc::new);
        let cuda_kv = {
            #[cfg(feature = "cuda")]
            {
                if hip.is_none() {
                    cuda.as_ref().map(|c| c.device)
                } else {
                    None
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        };
        let mut kv = KvTierManager::with_devices(
            config.kv.vram_bytes,
            config.kv.dram_bytes,
            config.kv.hot_fraction,
            hip.as_ref().map(|h| h.device),
            cuda_kv,
            {
                #[cfg(feature = "vulkan")]
                {
                    if hip.is_none() {
                        #[cfg(feature = "cuda")]
                        let skip = cuda.is_some();
                        #[cfg(not(feature = "cuda"))]
                        let skip = false;
                        if skip {
                            None
                        } else {
                            vulkan.clone()
                        }
                    } else {
                        None
                    }
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    None
                }
            },
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
            decision: None,
            target: None,
            draft_model: None,
            ngram: NGramProvider::new(config.speculative.ngram_n),
            dspark: None,
            dflash2: None,
            active: HashMap::new(),
            next_request_id: 1,
            model_id: config.model.id.clone(),
            target_resident: None,
            draft_resident: None,
            #[cfg(feature = "cuda")]
            target_cuda_resident: None,
            #[cfg(feature = "vulkan")]
            target_vulkan_resident: None,
            draft_kv: DenseKv::default(),
            draft_ahead: None,
            anchors: AnchorCache::new(config.anchors.slots, config.anchors.token_ids.clone()),
            // One node per token: the node cap must allow long prompts.
            kv_radix: KvRadixForest::with_limits(1 << 20, config.prompt_cache.ram_bytes)
                .with_codec(config.prompt_cache.codec),
            prompt_cache_key: None,
            ds_probe: ds_probe.clone(),
            grammar_factory: None,
            last_poll_id: None,
            consecutive_prefill_picks: 0,
            prefer_next: None,
            logits_owner: None,
            staged_decode: None,
            elastic_miss_streak: 0,
            elastic_rebalance_pending: false,
            elastic_last_misses: 0,
            vision_encoder: None,
        };
        let engine = Self {
            inner: Arc::new(Mutex::new(inner)),
            config: config.clone(),
            inventory,
            metrics: Arc::new(Metrics::default()),
            hip,
            draft_hip,
            #[cfg(feature = "cuda")]
            cuda,
            #[cfg(feature = "vulkan")]
            vulkan,
            ds_queue,
            ds_probe,
            micro_batch: Arc::new(MicroBatchExpertUnion::new()),
            layer_stagger: Arc::new(LayerStagger::new()),
            gpu_sampler: Arc::new(GpuSampler::default()),
            rpc,
            poll_tick: Arc::new(Mutex::new(())),
            read_side: Arc::new(Mutex::new(ReadSide::default())),
        };
        if config.model.synthetic {
            engine.load_synthetic(ModelConfig::tiny())?;
        } else if let Some(path) = &config.model.path {
            engine.load_resolved_model(path)?;
        }
        engine.install_vision_encoder_from_config()?;
        Ok(engine)
    }

    fn install_vision_encoder_from_config(&self) -> Result<(), AllpacaError> {
        let mm = &self.config.multimodal;
        let out_dim = {
            let inner = self.lock()?;
            inner.target.as_ref().map(|m| m.config.dim)
        };
        let Some(out_dim) = out_dim else {
            return Ok(());
        };
        let device = self.inventory.placement_vision_device();
        let image_size = if mm.vision_image_size == 0 {
            crate::vision::DEFAULT_IMAGE_SIZE
        } else {
            mm.vision_image_size
        };
        let patch = if mm.vision_patch == 0 {
            crate::vision::DEFAULT_PATCH_SIZE
        } else {
            mm.vision_patch
        };
        let n_blocks = mm.vision_n_blocks;
        let is_gguf = mm
            .vision_path
            .as_ref()
            .and_then(|p| p.extension())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        if let (Some(path), true) = (&mm.vision_path, is_gguf) {
            let mut enc = crate::Qwen3VlVision::load_gguf(path)?;
            if mm.vision_max_tokens > 0 {
                enc = enc.with_max_image_tokens(mm.vision_max_tokens as usize);
            }
            if enc.config().proj_dim != out_dim {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "mmproj {} projects to {} but the model width is {out_dim}",
                    path.display(),
                    enc.config().proj_dim
                ))));
            }
            tracing::info!(path = %path.display(), cfg = ?enc.config(), "loaded Qwen3-VL mmproj vision encoder");
            self.set_vision_encoder(Arc::new(enc))?;
        } else if let Some(path) = &mm.vision_path {
            let enc = crate::PatchVisionEncoder::load(
                path, device, out_dim, image_size, patch, n_blocks,
            )?;
            tracing::info!(
                path = %path.display(),
                out_dim,
                image_size,
                patch,
                n_blocks,
                ?device,
                "loaded patch vision encoder weights"
            );
            self.set_vision_encoder(Arc::new(enc))?;
        } else if self.config.model.synthetic && (n_blocks > 0 || mm.vision_device.is_some()) {
            let seed = self.config.engine.determinism_seed.unwrap_or(42);
            let enc = crate::PatchVisionEncoder::synthetic_sized(
                device, out_dim, image_size, patch, seed, n_blocks,
            )?;
            tracing::info!(
                out_dim,
                image_size,
                patch,
                n_blocks,
                ?device,
                "installed synthetic patch vision encoder"
            );
            self.set_vision_encoder(Arc::new(enc))?;
        }
        Ok(())
    }

    pub fn inventory(&self) -> &DeviceInventory {
        &self.inventory
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Live stats for every GPU on the RPC workers (empty without RPC).
    pub fn remote_gpus(&self) -> Vec<crate::rpc_runtime::RemoteGpu> {
        self.rpc.as_ref().map(|r| r.remote_gpu_stats()).unwrap_or_default()
    }

    /// The loaded target's expert controls, sized to its routed experts.
    /// `None` when no model is loaded.
    pub fn expert_controls(&self) -> Option<Arc<crate::expert_controls::ExpertControls>> {
        let inner = self.lock().ok()?;
        let target = inner.target.as_ref()?;
        let cfg = &target.config;
        let routed = target.layers.iter().any(|l| !l.w_router.is_empty()) || cfg.n_experts > 1;
        let shape = crate::expert_controls::ExpertShape {
            n_layers: if routed { target.layers.len() } else { 0 },
            n_experts: if routed { cfg.n_experts } else { 0 },
            n_active: cfg.n_active.max(1).min(cfg.n_experts.max(1)),
        };
        let ctl = Arc::clone(&target.expert_controls);
        if ctl.shape() != shape {
            ctl.configure(shape);
        }
        Some(ctl)
    }

    /// Routing hits per (layer, expert), layer-major: host-routed layers plus
    /// every local GPU's device counts. Zeros unless recording is on.
    pub fn expert_usage(&self) -> Result<Vec<u64>, AllpacaError> {
        let Some(ctl) = self.expert_controls() else {
            return Ok(Vec::new());
        };
        let mut total = ctl.host_usage();
        let (resident, peer) = {
            let inner = self.lock()?;
            let peer = inner
                .target
                .as_ref()
                .and_then(|t| t.rpc.as_ref())
                .and_then(|r| r.local_peer())
                .map(|(_, _, res)| res);
            (inner.target_resident.clone(), peer)
        };
        for res in resident.into_iter().chain(peer) {
            let dev = res.device_expert_usage(&ctl)?;
            for (t, d) in total.iter_mut().zip(dev) {
                *t += d;
            }
        }
        Ok(total)
    }

    /// Register a vision encoder for multimodal Requests. Text-only generate is unchanged.
    /// When the encoder's placement matches a HIP workspace, uploads weights once and
    /// records [`Metrics::vision_weight_bytes_on_device`].
    pub fn set_vision_encoder(
        &self,
        encoder: Arc<dyn crate::VisionEncoder>,
    ) -> Result<(), AllpacaError> {
        let hip_dev = encoder
            .device()
            .and_then(|d| self.hip_workspace_for_device(d).map(|h| h.device));
        let bytes = encoder.ensure_device_resident(hip_dev)?;
        if bytes > 0 {
            self.metrics
                .vision_weight_bytes_on_device
                .store(bytes, std::sync::atomic::Ordering::Relaxed);
        }
        self.lock()?.vision_encoder = Some(encoder);
        Ok(())
    }

    /// The installed vision encoder's in-prompt image placeholder, if it uses one.
    pub fn image_marker(&self) -> Option<&'static str> {
        self.lock().ok()?.vision_encoder.as_ref()?.image_marker()
    }

    /// Start a Request with optional images. Empty images match [`Self::begin_request`].
    pub fn begin_request_multimodal(
        &self,
        prompt: &str,
        images: &[crate::ImageInput],
        params: GenerationParams,
    ) -> Result<RequestHandle, AllpacaError> {
        let vision = if images.is_empty() {
            None
        } else {
            let enc = {
                let inner = self.lock()?;
                inner.vision_encoder.clone()
            };
            match enc {
                Some(enc) => Some(enc.encode(images)?),
                None => {
                    return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "vision encoder not registered".into(),
                    )));
                }
            }
        };
        self.begin_request_with_vision(prompt, vision, params)
    }

    /// Refresh driver VRAM and GPU busy into [`Metrics`], at most once per
    /// [`GpuSampler`] window, and hand the busy fraction to `q*`.
    ///
    /// `hipMemGetInfo` / `cudaMemGetInfo` are driver round-trips, so this belongs
    /// to scheduler safe points and `/stats` only. Decode must never call it.
    fn sample_gpu(&self) {
        if !self.gpu_sampler.due() {
            return;
        }
        #[cfg(feature = "hip")]
        if let Some(hip) = self.hip.as_ref() {
            let read = |dev: i32| {
                allpaca_hip::device_memory_info(dev)
                    .map(|(free, total)| (total.saturating_sub(free), total))
                    .unwrap_or((0, 0))
            };
            let target_dev = self.inventory.target_device.max(0);
            let (used, total) = read(target_dev);
            let draft_dev = self.inventory.draft_device;
            // GPU 1 is the draft card *and* the local layer-split peer. Sample
            // it whenever that workspace exists — not only after an MTP draft
            // forward — so the UI shows 9060 VRAM during `--speculative false`.
            let (draft_used, draft_total) =
                if self.draft_hip.is_some() && draft_dev >= 0 && draft_dev != target_dev {
                    read(draft_dev)
                } else {
                    (0, 0)
                };
            let p = hip.decode_profile();
            let gpu_us = p.q8_gemm_us + p.attention_us + p.lm_head_us + p.other_gpu_us;
            let wall_us = gpu_us + p.host_us;
            let util = if wall_us > 0.0 { gpu_us / wall_us } else { 0.0 };
            self.metrics
                .record_gpu_sample(used, total, draft_used, draft_total, util);
            if let Some(g) = self.lock_for_read() {
                if let Some(lru) = g.target.as_ref().and_then(|m| m.expert_lru.as_ref()) {
                    lru.set_gpu_busy(util);
                }
            }
            return;
        }
        #[cfg(feature = "cuda")]
        if self.cuda.is_some() {
            let target_dev = self.inventory.target_device.max(0);
            let (used, total) = allpaca_cuda::device_memory_info(target_dev)
                .map(|(free, total)| (total.saturating_sub(free), total))
                .unwrap_or((0, 0));
            let p = self
                .cuda
                .as_ref()
                .map(|c| c.decode_profile())
                .unwrap_or_default();
            let wall = p.gpu_us + p.host_us;
            let util = if wall > 0.0 { p.gpu_us / wall } else { 0.0 };
            self.metrics.record_gpu_sample(used, total, 0, 0, util);
        }
    }

    /// Configured KV VRAM budget after boot-time GPU capacity scaling.
    pub fn kv_vram_budget_bytes(&self) -> usize {
        self.config.kv.vram_bytes
    }

    /// Which device runs which layers of the loaded target, in layer order,
    /// for the operator UI. Empty when no model is loaded.
    pub fn placement(&self) -> Vec<DevicePlacement> {
        let (n_layers, rpc, target_bytes, draft, dflash2_dev) = {
            let Ok(inner) = self.lock() else {
                return Vec::new();
            };
            let Some(target) = inner.target.as_ref() else {
                return Vec::new();
            };
            let draft = inner.draft_resident.as_ref().map(|r| {
                let role = if mtp_draft_active(&inner) { "MTP draft" } else { "draft model" };
                (r.bytes(), role)
            });
            (
                target.config.n_layers,
                target.rpc.clone(),
                inner.target_resident.as_ref().map(|r| r.bytes()),
                draft,
                inner.dflash2.as_ref().map(|p| p.device()),
            )
        };
        let primary = match self.hip.as_ref() {
            Some(h) => ("hip", h.device),
            None => {
                #[cfg(feature = "cuda")]
                if let Some(c) = self.cuda.as_ref() {
                    return placement_rows(n_layers, ("cuda", c.device), self, None, None);
                }
                ("cpu", -1)
            }
        };
        let draft_dev = self.draft_hip.as_ref().map(|h| h.device);
        let extras = PlacementExtras {
            target_bytes,
            draft: draft.map(|(b, r)| (draft_dev, b, r)),
            dflash2_dev,
        };
        placement_rows(n_layers, primary, self, rpc.as_deref(), Some(extras))
    }

    /// The expert pins (`rpc.expert_homes`) this engine was built with.
    pub fn expert_homes(&self) -> &std::collections::BTreeMap<String, String> {
        &self.config.rpc.expert_homes
    }

    /// Where each layer's routed experts sit and where they may be pinned
    /// (`rpc.expert_homes`). `None` unless the loaded model runs
    /// expert-parallel across local GPUs and RPC workers.
    pub fn expert_placement(&self) -> Option<ExpertPlacement> {
        let rpc = self.lock().ok()?.target.as_ref()?.rpc.clone()?;
        let (bytes, workers) = rpc.expert_layout()?;
        let layers = bytes
            .iter()
            .enumerate()
            .map(|(l, &bytes)| ExpertLayerHome {
                device: match rpc.experts_home(l) {
                    Some((endpoint, device)) => format!("rpc:{endpoint}/{device}"),
                    None => format!("hip:{}", rpc.local_device(l)),
                },
                bytes,
            })
            .collect();
        Some(ExpertPlacement { layers, workers })
    }

    /// KV cost per context token of the loaded target, for sizing
    /// `kv.vram_bytes` from a context length. `None` when no model is loaded.
    pub fn kv_estimate(&self) -> Option<KvEstimate> {
        let inner = self.lock().ok()?;
        let target = inner.target.as_ref()?;
        let cfg = &target.config;
        // KV pages are f32 K and V per attention layer.
        let per_layer = (cfg.n_kv_heads.max(1) * cfg.head_dim().max(1) * 2 * 4) as u64;
        let mut est = KvEstimate {
            budget_bytes: self.config.kv.vram_bytes as u64,
            page_tokens: self.config.kv.page_tokens,
            ..KvEstimate::default()
        };
        for (l, layer) in target.layers.iter().enumerate() {
            let attends = layer.gdn.is_none()
                && layer.ssm.is_none()
                && layer.block == allpaca_gguf::HybridLayerKind::Attention;
            if !attends {
                continue;
            }
            est.attention_layers += 1;
            if cfg.sliding_window > 0 && cfg.swa_pattern.get(l).copied().unwrap_or(false) {
                est.swa_window = cfg.sliding_window as u64;
                est.swa_bytes_per_token += per_layer;
            } else {
                est.full_bytes_per_token += per_layer;
            }
        }
        Some(est)
    }

    pub fn stats(&self) -> StatsSnapshot {
        self.sample_gpu();
        let (model_id, kv_h2d, kv_d2h, draft_kv_bytes, draft_depth) = match self.lock_for_read() {
            Some(g) => {
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
                self.metrics
                    .radix_evictions
                    .store(g.kv_radix.evictions(), std::sync::atomic::Ordering::Relaxed);
                self.metrics.radix_bytes.store(
                    g.kv_radix.checkpoint_bytes() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.metrics.gemv_tile_cols.store(
                    allpaca_hip::last_gemv_tile_cols(),
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
                self.metrics
                    .ds_pinned_bytes
                    .store(q.pinned_bytes(), std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .ds_gpu_dma
                    .store(q.gpu_dma(), std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .ds_gpu_dma_bytes
                    .store(q.gpu_dma_bytes(), std::sync::atomic::Ordering::Relaxed);
                self.metrics
                    .ds_com_binds
                    .store(q.com_binds(), std::sync::atomic::Ordering::Relaxed);
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
                self.metrics
                    .kv_nvme_prefetches
                    .store(g.kv.nvme_prefetches(), std::sync::atomic::Ordering::Relaxed);
                let fresh = (
                    g.target
                        .as_ref()
                        .map(|m| m.name.clone())
                        .unwrap_or_else(|| self.config.model.id.clone()),
                    g.kv.activation_h2d_bytes(),
                    g.kv.activation_d2h_bytes(),
                    g.draft_kv.device_bytes(),
                    g.scheduler.draft_depth as u64,
                );
                drop(g);
                if let Ok(mut c) = self.read_side.lock() {
                    c.target_name = Some(fresh.0.clone());
                    (c.kv_h2d, c.kv_d2h, c.draft_kv_bytes, c.draft_depth) =
                        (fresh.1, fresh.2, fresh.3, fresh.4);
                }
                fresh
            }
            None => {
                let c = self.read_side.lock().map(|c| c.clone()).unwrap_or_default();
                (
                    c.target_name.unwrap_or_else(|| self.config.model.id.clone()),
                    c.kv_h2d,
                    c.kv_d2h,
                    c.draft_kv_bytes,
                    c.draft_depth,
                )
            }
        };
        let (backend, device_memory_bytes) =
            if self.rpc.as_ref().is_some_and(|r| r.placed_remote()) {
                if let Some(rt) = &self.rpc {
                    self.metrics.rpc_roundtrips.store(
                        rt.roundtrips.load(std::sync::atomic::Ordering::Relaxed),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    self.metrics.rpc_activation_bytes.store(
                        rt.activation_bytes
                            .load(std::sync::atomic::Ordering::Relaxed),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    self.metrics.rpc_h2d_bytes.store(
                        rt.h2d_bytes.load(std::sync::atomic::Ordering::Relaxed),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                ("rpc".into(), allpaca_hip::allocated_bytes())
            } else if self.hip.is_some() {
                ("hip".into(), allpaca_hip::allocated_bytes())
            } else {
                #[cfg(feature = "cuda")]
                {
                    if let Some(cuda) = &self.cuda {
                        ("cuda".into(), cuda.allocated_bytes())
                    } else {
                        #[cfg(feature = "vulkan")]
                        {
                            if let Some(vk) = &self.vulkan {
                                ("vulkan".into(), vk.allocated_bytes())
                            } else {
                                ("cpu".into(), 0)
                            }
                        }
                        #[cfg(not(feature = "vulkan"))]
                        {
                            ("cpu".into(), 0)
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    #[cfg(feature = "vulkan")]
                    {
                        if let Some(vk) = &self.vulkan {
                            ("vulkan".into(), vk.allocated_bytes())
                        } else {
                            ("cpu".into(), 0)
                        }
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        ("cpu".into(), 0)
                    }
                }
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
        #[cfg(feature = "cuda")]
        let (cuda_h2d, cuda_d2h) = self
            .cuda
            .as_ref()
            .map(|c| (c.activation_h2d_bytes(), c.activation_d2h_bytes()))
            .unwrap_or((0, 0));
        #[cfg(not(feature = "cuda"))]
        let (cuda_h2d, cuda_d2h) = (0u64, 0u64);
        #[cfg(feature = "vulkan")]
        let (vk_h2d, vk_d2h) = self
            .vulkan
            .as_ref()
            .map(|c| (c.activation_h2d_bytes(), c.activation_d2h_bytes()))
            .unwrap_or((0, 0));
        #[cfg(not(feature = "vulkan"))]
        let (vk_h2d, vk_d2h) = (0u64, 0u64);
        self.metrics
            .device_memory_bytes
            .store(device_memory_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .draft_kv_bytes_on_device
            .store(draft_kv_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics.activation_h2d_bytes.store(
            hip_h2d + kv_h2d + draft_h2d + cuda_h2d + vk_h2d,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.metrics.activation_d2h_bytes.store(
            hip_d2h + kv_d2h + draft_d2h + cuda_d2h + vk_d2h,
            std::sync::atomic::Ordering::Relaxed,
        );
        if let Some(hip) = &self.hip {
            let mut p = hip.decode_profile();
            let itl_us = self
                .metrics
                .itl_ms_ema_x100
                .load(std::sync::atomic::Ordering::Relaxed) as f64
                * 10.0;
            let gpu_us = p.q8_gemm_us + p.attention_us + p.lm_head_us + p.other_gpu_us;
            // The event-timed sample forward runs slower than an untimed one
            // (a marker per kernel; on Windows each can split the command
            // buffer). When its buckets overshoot the measured ITL, report
            // them as shares of the ITL instead.
            let sampled_us = gpu_us + p.host_us;
            if itl_us > 0.0 && sampled_us > itl_us {
                let k = itl_us / sampled_us;
                p.q8_gemm_us *= k;
                p.attention_us *= k;
                p.lm_head_us *= k;
                p.other_gpu_us *= k;
                p.host_us *= k;
            }
            self.metrics.q8_gemm_ns.store(
                (p.q8_gemm_us * 1_000.0).round() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.metrics.attention_ns.store(
                (p.attention_us * 1_000.0).round() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.metrics.lm_head_ns.store(
                (p.lm_head_us * 1_000.0).round() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.metrics.other_gpu_ns.store(
                (p.other_gpu_us * 1_000.0).round() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let gpu_us = p.q8_gemm_us + p.attention_us + p.lm_head_us + p.other_gpu_us;
            let host_us = if itl_us > gpu_us {
                itl_us - gpu_us
            } else {
                p.host_us
            };
            self.metrics.host_ns.store(
                (host_us * 1_000.0).round() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let bench = self.inventory.transfer_benchmark;
        let p2p_copy_ok = bench.map(|b| b.p2p_copy_ok).unwrap_or(false);
        let p2p_status = if p2p_copy_ok {
            "measured"
        } else {
            "p2p_unavailable"
        };
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
        match self.lock_for_read() {
            Some(g) => {
                let id = g.model_id.clone();
                drop(g);
                if let Ok(mut c) = self.read_side.lock() {
                    c.model_id = Some(id.clone());
                }
                id
            }
            None => self
                .read_side
                .lock()
                .ok()
                .and_then(|c| c.model_id.clone())
                .unwrap_or_else(|| self.config.model.id.clone()),
        }
    }

    /// The engine lock for a monitoring read: gives up after
    /// [`READ_LOCK_WAIT`] (or on a poisoned lock) so the caller serves its
    /// last values instead.
    fn lock_for_read(&self) -> Option<std::sync::MutexGuard<'_, EngineInner>> {
        let deadline = std::time::Instant::now() + READ_LOCK_WAIT;
        loop {
            match self.inner.try_lock() {
                Ok(g) => return Some(g),
                Err(std::sync::TryLockError::WouldBlock)
                    if std::time::Instant::now() < deadline =>
                {
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                Err(_) => return None,
            }
        }
    }

    pub fn has_model(&self) -> bool {
        self.lock()
            .map(|g| g.target.is_some() && g.tokenizer.is_some())
            .unwrap_or(false)
    }

    pub fn load_synthetic(&self, cfg: ModelConfig) -> Result<(), AllpacaError> {
        let seed = self.config.engine.determinism_seed.unwrap_or(42);
        let vocab = cfg.vocab_size;
        let layers = cfg.n_layers;
        let target = TinyModel::synthetic(cfg, seed, "synthetic");
        self.load_tiny_model(target, "synthetic")?;
        tracing::info!(vocab, layers, "loaded synthetic model");
        Ok(())
    }

    /// Load a host-ready TinyModel (tests / synthetic variants).
    pub fn load_tiny_model(
        &self,
        mut target: TinyModel,
        model_id: &str,
    ) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let seed = self.config.engine.determinism_seed.unwrap_or(42);
        let tokenizer = EngineTokenizer::synthetic(target.config.vocab_size)?;
        inner.ring.resize_vocab(target.config.vocab_size);
        self.apply_weight_working_set(&mut target)?;
        self.apply_expert_lru(&mut target)?;
        self.attach_rpc(&mut target, None)?;
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
        inner.model_id = model_id.into();
        drop(inner);
        self.upload_resident_weights()?;
        self.note_early_exit_gate()?;
        Ok(())
    }

    /// Install a ready synthetic `TinyModel` (e.g. SSM hybrid) and upload resident weights.
    pub fn load_synthetic_model(&self, mut target: TinyModel) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let vocab = target.config.vocab_size;
        let n_layers = target.config.n_layers;
        let tokenizer = EngineTokenizer::synthetic(vocab)?;
        inner.ring.resize_vocab(vocab);
        self.apply_weight_working_set(&mut target)?;
        self.apply_expert_lru(&mut target)?;
        self.attach_rpc(&mut target, None)?;
        inner.target = Some(Arc::new(target));
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = "synthetic".into();
        drop(inner);
        self.upload_resident_weights()?;
        self.note_early_exit_gate()?;
        tracing::info!(vocab, layers = n_layers, "loaded synthetic TinyModel");
        Ok(())
    }

    fn attach_rpc(
        &self,
        model: &mut TinyModel,
        gguf: Option<&GgufFile>,
    ) -> Result<(), AllpacaError> {
        self.attach_rpc_with_draft(model, gguf, 0)
    }

    /// `draft_bytes` is the draft model's VRAM footprint; the local layer
    /// planner keeps it free on the draft GPU.
    fn attach_rpc_with_draft(
        &self,
        model: &mut TinyModel,
        gguf: Option<&GgufFile>,
        draft_bytes: u64,
    ) -> Result<(), AllpacaError> {
        let mut local_devices = Vec::new();
        if let Some(hip) = &self.hip {
            let total = allpaca_hip::device_memory_info(hip.device)
                .map(|(_, total)| total)
                .unwrap_or(1 << 30);
            local_devices.push((hip.device, total));
        }
        if let Some(hip) = &self.draft_hip {
            let total = allpaca_hip::device_memory_info(hip.device)
                .map(|(_, total)| total)
                .unwrap_or(1 << 30);
            local_devices.push((hip.device, total));
        }
        if local_devices.is_empty() {
            local_devices.push((0, 8 << 30));
        }
        let rpc_max_seq = rpc_kv_max_seq(
            self.config.rpc.max_seq,
            self.config.kv.vram_bytes as u64,
            model,
        );
        if let Some(rt) = &self.rpc {
            rt.set_local_budgets_mib(&self.config.engine.local_vram_budget_mib);
            rt.set_expert_homes(&self.config.rpc.expert_homes);
            rt.set_prefill_rows(self.config.rpc.prefill_rows);
            rt.prepare(
                model,
                &local_devices,
                self.config.rpc.tensor_split.as_deref(),
                rpc_max_seq,
                gguf,
                self.config.kv.vram_bytes as u64,
                draft_bytes,
            )?;
            model.rpc = Some(Arc::clone(rt));
            tracing::info!(
                remote_layers = rt.plan_has_remote().then_some("yes").unwrap_or("no"),
                sessions = rt.session_count(),
                "rpc layer plan ready"
            );
            return Ok(());
        }
        if local_devices.len() > 1 {
            let rt = Arc::new(crate::rpc_runtime::RpcRuntime::connect(&[], None)?);
            rt.set_local_budgets_mib(&self.config.engine.local_vram_budget_mib);
            rt.set_prefill_rows(self.config.rpc.prefill_rows);
            rt.prepare(
                model,
                &local_devices,
                self.config.rpc.tensor_split.as_deref(),
                rpc_max_seq,
                gguf,
                self.config.kv.vram_bytes as u64,
                draft_bytes,
            )?;
            tracing::info!(
                local_gpus = local_devices.len(),
                "local GPU pool layer plan ready"
            );
            model.rpc = Some(rt);
        }
        Ok(())
    }

    /// Record Degraded gate when early_exit is enabled but the model has no manifest.
    fn note_early_exit_gate(&self) -> Result<(), AllpacaError> {
        let enabled = self.config.early_exit.enabled;
        let spec = {
            let inner = self.lock()?;
            inner
                .target
                .as_ref()
                .map(|m| m.early_exit.clone())
                .unwrap_or_default()
        };
        if gate_unsupported(enabled, &spec) {
            self.metrics
                .early_exit_gated
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.metrics
                .degraded_events
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::warn!(
                "early_exit.enabled but model has no supported exit-head manifest; gating"
            );
        }
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
        let sample_id = ExpertId {
            layer: 0,
            expert: 0,
        };
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
            .and_then(|hip| allpaca_hip::benchmark_h2d_bytes_per_sec(hip.device, expert_bytes).ok())
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

        let keep = self.tune_keep_ids(model);
        let pin_bytes = keep.len().saturating_mul(expert_bytes);
        let cache_bytes = if keep.is_empty() {
            self.config.weights.cache_bytes
        } else {
            self.config.weights.cache_bytes.max(pin_bytes)
        };
        let n_slots = expert_lru_slots(expert_bytes, cache_bytes);
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
        self.warmup_expert_keep_set(model)?;
        Ok(())
    }

    fn tune_keep_ids(&self, model: &TinyModel) -> Vec<ExpertId> {
        let Some(entry) = self.config.tune_for_model() else {
            return Vec::new();
        };
        let shape = crate::tune::MoeShape::from_model_config(&model.config);
        let catalog = self
            .config
            .model
            .path
            .as_ref()
            .and_then(|p| crate::tune::load_catalog_beside(p));
        crate::tune::keep_ids_for_entry(entry, &shape, catalog.as_ref())
    }

    fn warmup_expert_keep_set(&self, model: &TinyModel) -> Result<(), AllpacaError> {
        let Some(lru) = model.expert_lru.as_ref() else {
            return Ok(());
        };
        let keep = self.tune_keep_ids(model);
        lru.set_keep_set(&keep);
        let hosts: Vec<(ExpertId, LayerFfn)> = keep
            .iter()
            .filter_map(|id| {
                model
                    .host_expert(id.layer as usize, id.expert as usize)
                    .ok()
                    .map(|ffn| (*id, ffn))
            })
            .collect();
        let refs: Vec<(ExpertId, &LayerFfn)> = hosts.iter().map(|(id, ffn)| (*id, ffn)).collect();
        lru.warmup_keep_set(&refs)
    }

    fn resolve_configured_path(
        &self,
        path: &std::path::Path,
    ) -> Result<(std::path::PathBuf, Option<String>), AllpacaError> {
        let spec = path.to_string_lossy();
        let opts =
            crate::hf::HubOptions::from_env().with_revision(self.config.model.hf_revision.clone());
        crate::hf::resolve_model_spec(&spec, &opts)
    }

    fn load_resolved_model(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let (resolved, preferred) = self.resolve_configured_path(path)?;
        self.load_model_path_discovered(&resolved, preferred.as_deref())
    }

    fn load_model_path_discovered(
        &self,
        path: &std::path::Path,
        preferred: Option<&str>,
    ) -> Result<(), AllpacaError> {
        if path.is_dir() && crate::edge0::detect_edge0_dir(path).is_some() {
            return self.load_edge0(path);
        }
        let discovered = discover_checkpoint(path, preferred)?;
        match discovered.kind {
            CheckpointKind::Safetensors => self.load_safetensors(&discovered.weights),
            CheckpointKind::Onnx => self.load_onnx(&discovered.weights),
            CheckpointKind::Gguf => self.load_gguf(&discovered.weights),
        }
    }

    /// Load an Edge0 HF checkpoint directory (35b / 8b recipe).
    pub fn load_edge0(&self, dir: &std::path::Path) -> Result<(), AllpacaError> {
        let opts = crate::edge0::Edge0LoadOptions {
            no_lora: self.config.model.edge0_no_lora,
            no_prerouter: self.config.model.edge0_no_prerouter,
            lora_rank: 16,
            lora_alpha: 32.0,
        };
        let loaded = crate::edge0::load_edge0_dir(dir, &opts)?;
        let mut model = loaded.model;
        model.edge0 = Some(crate::model::Edge0Runtime {
            lora: loaded.lora,
            stager: loaded.stager.map(std::sync::Mutex::new),
        });
        model.name = loaded.tier.name().into();
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        self.attach_rpc(&mut model, None)?;
        let tokenizer =
            EngineTokenizer::from_vocab(loaded.vocab_tokens, loaded.bos_id, loaded.eos_id)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()?;
        self.note_early_exit_gate()
    }

    pub fn load_safetensors(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = SafetensorsCatalog::open(path)?;
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
        let vocab = load_tokenizer_sidecar(&tokenizer_path)?;
        let mut model = TinyModel::from_safetensors(&file, &hf)?;
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        self.attach_rpc(&mut model, None)?;
        let tokenizer = EngineTokenizer::from_sidecar(vocab)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()?;
        self.note_early_exit_gate()
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
        let vocab = load_tokenizer_sidecar(&tokenizer_path)?;
        let mut model = TinyModel::from_onnx(&file, &hf)?;
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        self.attach_rpc(&mut model, None)?;
        let tokenizer = EngineTokenizer::from_sidecar(vocab)?;
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, tokenizer);
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        self.upload_resident_weights()?;
        self.note_early_exit_gate()
    }

    pub fn load_gguf(&self, path: &std::path::Path) -> Result<(), AllpacaError> {
        let file = Arc::new(GgufFile::open(path)?);
        let tok = extract_tokenizer(&file)?;
        let mut model = if self.config.weights.host_f32 {
            TinyModel::from_gguf_host(&file)?
        } else {
            TinyModel::from_gguf(&file)?
        };
        model.bind_gguf(file.clone());
        #[cfg(feature = "cuda")]
        let has_cuda = self.cuda.is_some();
        #[cfg(not(feature = "cuda"))]
        let has_cuda = false;
        if !model.host_weights_ready()
            && self.hip.is_none()
            && !has_cuda
            && model.dsv4.is_none()
            && model.q4x.is_none()
        {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "large GGUF requires HIP or CUDA (host F32 materialize is skipped to avoid OOM)"
                    .into(),
            )));
        }
        let skip_hip = model.dsv4.is_some() || model.q4x.is_some();
        self.apply_weight_working_set(&mut model)?;
        self.apply_expert_lru(&mut model)?;
        let mut draft_file = None;
        let mut draft_reuses_target_gguf = false;
        // A DFlash2 draft's file name also reads as a DSpark sidecar.
        let draft_path_is_dspark = !self
            .config
            .speculative
            .providers
            .contains(&DraftProviderKind::Dflash2)
            && self.config.model.draft_path.as_ref().is_some_and(|p| {
                is_dspark_filename(p.file_name().and_then(|s| s.to_str()).unwrap_or(""))
            });
        let mut inner = self.lock()?;
        inner.ring.resize_vocab(model.config.vocab_size);
        set_tokenizer(&mut inner, EngineTokenizer::from_gguf(tok));
        inner.decision = match inner.tokenizer.as_ref() {
            Some(tok) => crate::decision::DecisionModel::from_gguf(&file, tok)?.map(Arc::new),
            None => None,
        };
        let draft_path_is_dflash2 = self
            .config
            .speculative
            .providers
            .contains(&DraftProviderKind::Dflash2);
        if let Some(draft_path) = &self.config.model.draft_path {
            if !draft_path_is_dspark && !draft_path_is_dflash2 {
                let opened =
                    self.resolve_configured_path(draft_path)
                        .and_then(|(resolved, preferred)| {
                            discover_checkpoint(&resolved, preferred.as_deref())
                                .and_then(|d| GgufFile::open(&d.weights))
                        });
                match opened {
                    Ok(f) => match load_draft_gguf(&f) {
                        Ok(draft) => {
                            if draft.config.gguf_layer_base > 0 {
                                tracing::info!(
                                    layers = draft.config.n_layers,
                                    base = draft.config.gguf_layer_base,
                                    "loaded MTP/NextN sidecar as draft model"
                                );
                            }
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
        } else if self
            .config
            .speculative
            .providers
            .contains(&DraftProviderKind::Mtp)
            && model.config.nextn_predict_layers > 0
        {
            match TinyModel::from_gguf_mtp(&file) {
                Ok(draft) => {
                    tracing::info!(
                        layers = draft.config.n_layers,
                        base = draft.config.gguf_layer_base,
                        "loaded MTP/NextN blocks as draft model"
                    );
                    inner.draft_model = Some(Arc::new(draft));
                    draft_reuses_target_gguf = true;
                    inner.flags.draft_model_speculation = true;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "MTP draft load failed; continuing without MTP");
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        let dspark_path = if draft_path_is_dspark {
            self.config.model.draft_path.clone()
        } else if draft_path_is_dflash2 {
            None
        } else {
            discover_dspark_sidecar(path)
        };
        if let Some(dsp) = dspark_path {
            match GgufFile::open(&dsp) {
                Ok(f) if crate::dflash2::Dflash2Spec::from_gguf(&f).is_some() => {
                    // A DFlash2 draft is not a DSpark sidecar; it attaches
                    // only through `providers = ["dflash2"]`.
                    tracing::info!(path = %dsp.display(), "skipping DFlash2 draft found as a DSpark sidecar");
                }
                Ok(f) => {
                    let draft = if crate::dspark::file_is_dspark(&f) {
                        None
                    } else {
                        TinyModel::from_gguf(&f)
                            .ok()
                            .filter(|m| m.host_weights_ready())
                    };
                    match crate::dspark::DsparkProvider::from_gguf(&f, draft) {
                        Ok(p) => {
                            tracing::info!(path = %dsp.display(), "loaded DSpark sidecar");
                            inner.dspark = Some(p);
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "DSpark sidecar failed; continuing without DSpark");
                            self.metrics
                                .degraded_events
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "DSpark sidecar open failed; continuing without DSpark");
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        } else if self
            .config
            .speculative
            .providers
            .contains(&DraftProviderKind::Dspark)
        {
            tracing::warn!("DSpark requested but no dspark-*.gguf sidecar next to the target");
            self.metrics
                .degraded_events
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if model.early_exit.supported {
            model.early_exit.min_confidence = self.config.early_exit.min_confidence;
            model.early_exit.min_calibration_margin = self.config.early_exit.min_calibration_margin;
        }
        let draft_bytes = inner.draft_model.as_ref().map_or(0, |draft| {
            let draft_gguf = if draft_reuses_target_gguf {
                Some(file.as_ref())
            } else {
                draft_file.as_ref()
            };
            draft_gguf.map_or(0, |f| {
                let (layers, globals) = crate::model::gguf_device_bytes(f, &draft.config);
                layers.iter().sum::<u64>() + globals
            })
        });
        self.attach_rpc_with_draft(&mut model, Some(file.as_ref()), draft_bytes)?;
        inner.model_id = model.name.clone();
        inner.target = Some(Arc::new(model));
        drop(inner);
        if skip_hip {
            if let Some(hip) = &self.hip {
                let target = {
                    let g = self.lock()?;
                    g.target.clone()
                };
                if let Some(target) = target {
                    match crate::dsv4_gpu::attach(
                        &target,
                        hip,
                        self.draft_hip.as_ref(),
                        self.config.weights.host_bytes as u64,
                    ) {
                        Ok(()) => {
                            let keep = self.tune_keep_ids(&target);
                            if let Err(e) = crate::dsv4_gpu::apply_keep_set(&target, &keep) {
                                tracing::warn!(error = %e, "DeepSeek-V4 keep-set warmup failed");
                            }
                            tracing::info!(
                                "DeepSeek-V4: HIP BF16 MLA + MXFP4 expert LRU (no generic GGUF upload)"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "DeepSeek-V4 HIP working set failed; CPU MLA/MoE");
                            self.metrics
                                .degraded_events
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            } else {
                tracing::info!("DeepSeek-V4: CPU MLA/MoE decode (no HIP workspace)");
            }
            if let Some(hip) = self.draft_hip.as_ref().or(self.hip.as_ref()) {
                let mut g = self.lock()?;
                if let Some(p) = g.dspark.as_mut() {
                    p.attach_hip(hip.clone());
                }
            }
            // DSV4 skips generic HIP GGUF upload (`dsv4_gpu::attach` above). CUDA-only
            // still attempts resident upload so MLA/DSV4 is explicitly rejected.
            #[cfg(feature = "cuda")]
            if self.hip.is_none() {
                self.upload_resident_from_gguf(None, None)?;
            }
            #[cfg(feature = "vulkan")]
            if self.hip.is_none() {
                #[cfg(feature = "cuda")]
                let skip = self.cuda.is_some();
                #[cfg(not(feature = "cuda"))]
                let skip = false;
                if !skip {
                    self.upload_resident_from_gguf(None, None)?;
                }
            }
            return self.note_early_exit_gate();
        }
        let draft_ref = if draft_reuses_target_gguf {
            Some(file.as_ref())
        } else {
            draft_file.as_ref()
        };
        self.upload_resident_from_gguf(Some(file.as_ref()), draft_ref)?;
        self.attach_dflash2();
        self.note_early_exit_gate()
    }

    /// Loads the DFlash2 draft named by `model.draft_path` onto the draft GPU
    /// (the target GPU when there is none). It reads the target's embeddings
    /// and lm_head in place and taps its layer inputs.
    fn attach_dflash2(&self) {
        if !self.config.speculative.enabled
            || !self
                .config
                .speculative
                .providers
                .contains(&DraftProviderKind::Dflash2)
        {
            return;
        }
        let loaded = (|| -> Result<crate::dflash2::Dflash2Provider, AllpacaError> {
            let path = self.config.model.draft_path.clone().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "speculative provider dflash2 needs model.draft_path".into(),
                ))
            })?;
            let (resolved, preferred) = self.resolve_configured_path(&path)?;
            let file = GgufFile::open(&discover_checkpoint(&resolved, preferred.as_deref())?.weights)?;
            let spec = crate::dflash2::Dflash2Spec::from_gguf(&file).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "{} is not a DFlash2 draft",
                    resolved.display()
                )))
            })?;
            let (target, resident) = {
                let g = self.lock()?;
                (g.target.clone(), g.target_resident.clone())
            };
            let target = target.ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
            let no_hip = || {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "DFlash2 needs the target resident on HIP".into(),
                ))
            };
            let resident = resident.ok_or_else(no_hip)?;
            let target_hip = self.hip.clone().ok_or_else(no_hip)?;
            if spec.dim != target.config.dim {
                return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "DFlash2 width {} != target {}",
                    spec.dim, target.config.dim
                ))));
            }
            let device = std::env::var("ALLPACA_DFLASH2_DEVICE")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or_else(|| {
                    self.draft_hip
                        .as_ref()
                        .map_or(target_hip.device, |h| h.device)
                });
            let hip = Arc::new(HipWorkspace::new(device, 8 * 1024 * 1024, 256)?);
            let head = crate::dflash2::ResidentHead::new(
                target_hip,
                resident,
                target.config.dim,
                target.config.vocab_size,
                spec.block_size,
            )?;
            crate::dflash2::Dflash2Provider::load(
                &file,
                hip,
                Arc::clone(&target.layer_capture),
                Box::new(head),
            )
        })();
        match loaded {
            Ok(p) => {
                if let Ok(mut g) = self.lock() {
                    g.dflash2 = Some(p);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "DFlash2 draft failed; continuing without it");
                self.metrics
                    .degraded_events
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
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
        let target_for_peer = inner.target.clone();
        let mut peer_uploaded = None;
        if let (Some(peer_hip), Some(target_m), Some(file)) =
            (self.draft_hip.as_ref(), target_for_peer, target_gguf)
        {
            if let Some(rpc) = target_m.rpc.as_ref() {
                let dev = peer_hip.device;
                let any = (0..target_m.config.n_layers).any(|i| rpc.local_device(i) == dev);
                if any {
                    let skip: Vec<bool> = (0..target_m.config.n_layers)
                        .map(|i| rpc.is_remote(i) || rpc.local_device(i) != dev)
                        .collect();
                    let skip_experts: Vec<bool> = (0..target_m.config.n_layers)
                        .map(|i| rpc.experts_remote(i))
                        .collect();
                    match ResidentModel::upload_gguf_peer_layers(
                        file,
                        &target_m.config,
                        dev,
                        &skip,
                        &skip_experts,
                    ) {
                        Ok(resident) => {
                            let hip_used = allpaca_hip::device_memory_info(dev)
                                .ok()
                                .map(|(free, total)| total.saturating_sub(free));
                            tracing::info!(
                                device = dev,
                                bytes = resident.bytes(),
                                hip_used,
                                "local GPU pool: second device layers uploaded"
                            );
                            weight_bytes += resident.bytes();
                            h2d += resident.packed_h2d_bytes();
                            peer_uploaded = Some(resident);
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                device = dev,
                                "second local GPU layer upload failed; those layers stay on the target GPU"
                            );
                        }
                    }
                }
            }
        }
        let peer_ok = peer_uploaded.is_some();
        let peer_dev = self.draft_hip.as_ref().map(|h| h.device);
        if !peer_ok {
            // Nothing was uploaded to the second GPU (no GGUF to read its
            // layers from, no draft workspace, or the upload failed): its
            // layers run on the target GPU, which uploads them below.
            if let (Some(hip), Some(rpc)) = (
                self.hip.as_ref(),
                inner.target.as_ref().and_then(|m| m.rpc.clone()),
            ) {
                let n_layers = inner.target.as_ref().map_or(0, |m| m.config.n_layers);
                let others: Vec<i32> = (0..n_layers)
                    .map(|i| rpc.local_device(i))
                    .filter(|&d| d >= 0 && d != hip.device)
                    .collect();
                for d in others {
                    rpc.fold_local_device(d, hip.device);
                }
            }
        }
        let target = inner.target.clone();
        let draft = inner.draft_model.clone();
        inner.target_resident = if let (Some(hip), Some(target)) = (&self.hip, target) {
            let host_ready = target.host_weights_ready();
            let uploaded = if target.config.n_experts > 0 && host_ready {
                ResidentModel::upload(&target, hip.device)
            } else if let Some(file) = target_gguf {
                let skip: Vec<bool> = (0..target.config.n_layers)
                    .map(|i| {
                        target.rpc.as_ref().is_some_and(|r| {
                            r.is_remote(i)
                                || (peer_ok && peer_dev.is_some_and(|d| r.local_device(i) == d))
                        })
                    })
                    .collect();
                let skip_experts: Vec<bool> = (0..target.config.n_layers)
                    .map(|i| target.rpc.as_ref().is_some_and(|r| r.experts_remote(i)))
                    .collect();
                ResidentModel::upload_gguf_filtered(
                    file,
                    &target.config,
                    hip.device,
                    &skip,
                    &skip_experts,
                )
            } else if host_ready {
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
                    let hip_used = self.hip.as_ref().and_then(|h| {
                        allpaca_hip::device_memory_info(h.device)
                            .ok()
                            .map(|(free, total)| total.saturating_sub(free))
                    });
                    tracing::info!(
                        device = self.hip.as_ref().map(|h| h.device),
                        bytes = n,
                        hip_used,
                        "local GPU pool: target device layers uploaded"
                    );
                    Some(Arc::new(resident))
                }
                Err(e) => {
                    // Large GGUFs skip host F32 materialize — without a resident
                    // upload there is no decode path left.
                    if !host_ready {
                        return Err(e);
                    }
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
        if let (Some(resident), Some(draft_hip)) = (peer_uploaded, self.draft_hip.as_ref()) {
            if let Some(rpc) = inner.target.as_ref().and_then(|m| m.rpc.clone()) {
                // Own workspace for the target's layers on this GPU: the draft
                // model keeps `draft_hip`, and draft-ahead runs it on a worker
                // thread while the target verifies, so sharing decode buffers
                // would race.
                let hip = match HipWorkspace::new(draft_hip.device, 8 * 1024 * 1024, 256) {
                    Ok(ws) => Arc::new(ws),
                    Err(e) => {
                        tracing::warn!(error = %e, "peer HIP workspace failed; sharing the draft workspace");
                        Arc::clone(draft_hip)
                    }
                };
                let _ = resident.touch_resident();
                let _ = hip.bind();
                let _ = allpaca_hip::stream_synchronize();
                if let Some(t) = inner.target.as_ref() {
                    let dim = t.config.dim.max(1);
                    let kv = t.config.n_kv_heads * t.config.head_dim().max(1);
                    let _ = hip.ensure_decode(
                        dim,
                        kv,
                        t.config.moe_workspace(),
                        t.config.vocab_size.max(1),
                        256,
                    );
                }
                rpc.set_local_peer(hip.device, hip, Arc::new(resident));
            }
        }
        inner.draft_resident = if let (Some(hip), Some(draft)) = (&self.draft_hip, draft) {
            tracing::info!(
                device = hip.device,
                shares_target_layers = peer_ok,
                "HIP draft weight upload start"
            );
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
        #[cfg(feature = "cuda")]
        {
            let target = inner.target.clone();
            inner.target_cuda_resident = if self.hip.is_none() {
                if let (Some(cuda), Some(target)) = (&self.cuda, target) {
                    let host_ready = target.host_weights_ready();
                    let gdn_device = target.gdn_needs_device_projections();
                    let uploaded = if host_ready && !gdn_device {
                        crate::CudaResidentModel::upload(&target, cuda.device)
                    } else if let Some(file) = target_gguf {
                        crate::CudaResidentModel::upload_gguf(file, &target.config, cuda.device)
                    } else {
                        Err(AllpacaError::Fatal(FatalError::ModelLoad(
                            "large GGUF missing file handle for CUDA upload".into(),
                        )))
                    };
                    match uploaded {
                        Ok(resident) => {
                            weight_bytes += resident.bytes();
                            h2d += resident.packed_h2d_bytes();
                            tracing::info!(
                                device = cuda.device,
                                bytes = resident.bytes(),
                                "CUDA target weight upload"
                            );
                            Some(Arc::new(resident))
                        }
                        Err(e) => {
                            if !host_ready || gdn_device {
                                return Err(e);
                            }
                            tracing::warn!(
                                error = %e,
                                "CUDA target weight upload failed; CPU forward"
                            );
                            self.metrics
                                .degraded_events
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            None
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };
        }
        #[cfg(feature = "vulkan")]
        {
            let target = inner.target.clone();
            inner.target_vulkan_resident = if self.hip.is_none() {
                #[cfg(feature = "cuda")]
                let cuda_busy = self.cuda.is_some();
                #[cfg(not(feature = "cuda"))]
                let cuda_busy = false;
                if cuda_busy {
                    None
                } else if let (Some(vk), Some(target)) = (&self.vulkan, target) {
                    if target.host_weights_ready() {
                        let uploaded = if let Some(file) = target_gguf {
                            crate::VulkanResidentModel::upload_gguf(&target, file, vk)
                        } else {
                            crate::VulkanResidentModel::upload(&target, vk)
                        };
                        match uploaded {
                            Ok(resident) => {
                                weight_bytes += resident.bytes();
                                h2d += resident.packed_h2d_bytes();
                                tracing::info!(
                                    device = vk.device,
                                    bytes = resident.bytes(),
                                    "Vulkan target weight upload"
                                );
                                Some(Arc::new(resident))
                            }
                            Err(e) => {
                                tracing::warn!(
                                    error = %e,
                                    "Vulkan target weight upload failed; CPU forward"
                                );
                                self.metrics
                                    .degraded_events
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
        }
        self.metrics
            .weight_bytes_on_device
            .store(weight_bytes, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .weight_h2d_bytes
            .store(h2d, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Read each text with the loaded model and return the second moments
    /// of its matrix inputs ([`crate::calib`]). Only the CPU forward records.
    pub fn capture_input_stats(&self, texts: &[&str]) -> Result<crate::calib::InputStats, AllpacaError> {
        let model = self.lock_for_read().and_then(|g| g.target.clone()).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad("calibration needs a loaded model".into()))
        })?;
        model.calib.start();
        let mut run = Ok(());
        for text in texts {
            let params = GenerationParams { max_tokens: 1, temperature: 0.0, seed: Some(7), ..GenerationParams::default() };
            if let Err(e) = self.generate(text, params) {
                run = Err(e);
                break;
            }
        }
        let stats = model.calib.finish();
        run.map(|()| stats)
    }

    pub fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationOutput, AllpacaError> {
        self.generate_multimodal(prompt, &[], params)
    }

    /// ChatML when the loaded vocab has `<|im_start|>` / `<|im_end|>`; else `role:`.
    pub fn format_chat(&self, messages: &[(&str, &str)]) -> String {
        self.lock()
            .ok()
            .and_then(|inner| {
                inner
                    .tokenizer
                    .as_ref()
                    .map(|t| t.format_chat(messages, true))
            })
            .unwrap_or_else(|| crate::tokenizer::format_chat_colon(messages, true))
    }

    pub fn generate_multimodal(
        &self,
        prompt: &str,
        images: &[crate::ImageInput],
        params: GenerationParams,
    ) -> Result<GenerationOutput, AllpacaError> {
        let mut handle = self.begin_request_multimodal(prompt, images, params)?;
        let mut token_ids = Vec::new();
        // The reply text is the stream's text, so a blocking reply and a
        // streamed one are the same string: special tokens the model emits
        // (`<think>`) are kept, the stop token's text is not, and characters
        // split across tokens are joined.
        let mut text = String::new();
        let finish = self.drive_request(&mut handle, |ev| {
            token_ids.push(ev.token_id);
            text.push_str(&ev.text);
            ev.finish_reason
        })?;
        tracing::info!(
            n = token_ids.len(),
            first = token_ids.first().copied(),
            last = token_ids.last().copied(),
            ?finish,
            "generate tokens"
        );
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
        self.generate_stream_multimodal(prompt, Vec::new(), params)
    }

    pub fn generate_stream_multimodal(
        &self,
        prompt: String,
        images: Vec<crate::ImageInput>,
        params: GenerationParams,
    ) -> impl Stream<Item = Result<TokenEvent, AllpacaError>> + Send {
        let engine = self.clone();
        let use_fair = engine.config.engine.max_active_requests > 1;
        async_stream::stream! {
            let mut handle = match engine.begin_request_multimodal(&prompt, &images, params) {
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
            let result = match self.take_parked(id)? {
                Some(parked) => Some(parked),
                None => match self.poll_for(id)? {
                    PollFor::Result(r) => Some(r),
                    PollFor::Done => return Ok(None),
                    PollFor::Idle => None,
                },
            };
            match result {
                Some(StepResult::Token(ev)) => return Ok(Some(ev)),
                Some(StepResult::Prefill { .. }) => continue,
                Some(StepResult::Finished(_)) => return Ok(None),
                None => std::thread::yield_now(),
            }
        }
    }

    fn request_fully_done(&self, id: u64) -> Result<bool, AllpacaError> {
        let inner = self.lock()?;
        Ok(inner
            .active
            .get(&id)
            .map(|r| r.finished && r.emit_queue.is_empty() && r.parked.is_empty())
            .unwrap_or(true))
    }

    fn take_parked(&self, id: u64) -> Result<Option<StepResult>, AllpacaError> {
        let mut inner = self.lock()?;
        Ok(inner.active.get_mut(&id).and_then(|r| r.parked.pop_front()))
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
        params: GenerationParams,
    ) -> Result<RequestHandle, AllpacaError> {
        self.begin_request_with_vision(prompt, None, params)
    }

    fn begin_request_with_vision(
        &self,
        prompt: &str,
        vision: Option<crate::VisionTokens>,
        params: GenerationParams,
    ) -> Result<RequestHandle, AllpacaError> {
        self.begin_request_probed(prompt, vision, params, None)
    }

    /// `read`: start a decision request, which ends with its prompt.
    fn begin_request_probed(
        &self,
        prompt: &str,
        vision: Option<crate::VisionTokens>,
        mut params: GenerationParams,
        read: Option<&crate::decision::DecisionRead>,
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
        let dim = inner
            .target
            .as_ref()
            .map(|m| m.config.dim)
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let mut suppress_draft = false;
        let vision_seq = match vision {
            None => crate::model::VisionSeq::default(),
            Some(v) => {
                let need = v.n_tokens.checked_mul(dim).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::InvalidConfig(
                        "vision soft token size overflow".into(),
                    ))
                })?;
                if v.embeddings.len() != need {
                    return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                        "vision embeddings len {} != n_tokens {} * dim {}",
                        v.embeddings.len(),
                        v.n_tokens,
                        dim
                    ))));
                }
                let draft_dev = self.inventory.draft_device;
                if draft_dev >= 0 && v.device == Some(draft_dev) {
                    suppress_draft = true;
                    self.metrics
                        .degraded_events
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                if let Some(vdev) = v.device {
                    if let Some(hip) = self.hip_workspace_for_device(vdev) {
                        let bytes = v.embeddings.len().saturating_mul(4);
                        if bytes > 0 {
                            if let Ok(buf) = allpaca_hip::DeviceWeights::alloc(vdev, bytes) {
                                let _ = hip.copy_h2d_f32(buf.ptr_at(0) as *mut f32, &v.embeddings);
                                self.metrics
                                    .vision_bytes_on_device
                                    .fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                    }
                }
                let grids = if v.grids.is_empty() {
                    vec![(v.n_tokens, 1)]
                } else {
                    v.grids.clone()
                };
                if grids.iter().map(|(c, r)| c * r).sum::<usize>() != v.n_tokens {
                    return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                        "vision grids {grids:?} do not cover {} tokens",
                        v.n_tokens
                    ))));
                }
                let pad = (!v.grids.is_empty())
                    .then(|| tokenizer.token_id("<|image_pad|>"))
                    .flatten();
                let (expanded, spans) = expand_image_pads(&tokens, pad, &grids);
                tokens = expanded;
                let mrope = if v.grids.is_empty() {
                    crate::model::MropeLayout::default()
                } else {
                    crate::model::MropeLayout::new(spans.clone())
                };
                crate::model::VisionSeq {
                    spans,
                    embeds: v.embeddings,
                    mrope,
                }
            }
        };
        for id in tokenizer.stop_token_ids() {
            if !params.stop_token_ids.contains(&id) {
                params.stop_token_ids.push(id);
            }
        }
        let vocab = tokenizer.vocab_size();
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let page_tokens = self.config.kv.page_tokens;
        if let Some(rpc) = target.rpc.as_ref() {
            let devs: Vec<i32> = (0..target.config.n_layers)
                .map(|i| rpc.local_device(i))
                .collect();
            inner.kv.set_layer_devices(&devs);
        }
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
        self.sync_prompt_cache(&mut inner, &target);
        let probe = read.map(|read| DecisionProbe::new(read, &tokens));
        let mut prefill_pos = 0;
        // Soft vision tokens are not text anchors / radix keys yet. A
        // decision request that reads hidden states forwards every token.
        if vision_seq.is_empty() && !probe.as_ref().is_some_and(|p| !p.positions.is_empty()) {
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
                    restore_kv(&mut inner.kv, &pages, &cp.kv_rows(), &target.config)?;
                    if let Some(dst) = recurrent.as_mut() {
                        *dst = cp.recurrent;
                    } else if target.config.recurrent {
                        recurrent = Some(cp.recurrent);
                    }
                    prefill_pos = n;
                }
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
                probe,
                emit_queue: Vec::new(),
                parked: VecDeque::new(),
                async_mask_rx: None,
                json_prefix: String::new(),
                utf8: Default::default(),
                grammar,
                rng,
                started: Instant::now(),
                last_token_at: Instant::now(),
                first_emitted: false,
                finished: false,
                recurrent,
                decode_step: 0,
                moe_norm: vec![0.0f32; dim],
                last_hidden: vec![0.0f32; dim],
                last_hidden_fresh: false,
                defer_lm_head: false,
                deferred: None,
                vision: Arc::new(vision_seq),
                suppress_draft,
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
        self.metrics
            .prompt_tokens
            .fetch_add(prompt_len as u64, std::sync::atomic::Ordering::Relaxed);
        // One chunk only — remaining prompt tokens continue in `step` so a
        // second Request can admit / decode between chunks.
        // Another Request's decode logits may still be in flight: land them
        // before this prefill's forward reuses the pinned logits slot.
        self.drain_foreign_logits(&mut inner, id)?;
        self.prefill_chunk(&mut inner, id)?;
        Ok(RequestHandle {
            id,
            inner: Arc::clone(&self.inner),
            metrics: Arc::clone(&self.metrics),
            hip: self.hip.clone(),
        })
    }

    fn hip_workspace_for_device(&self, device: i32) -> Option<&HipWorkspace> {
        if let Some(hip) = self.hip.as_ref() {
            if hip.device == device {
                return Some(hip.as_ref());
            }
        }
        if let Some(hip) = self.draft_hip.as_ref() {
            if hip.device == device {
                return Some(hip.as_ref());
            }
        }
        None
    }

    /// The decision side of the loaded model, when it is a decision model.
    pub fn decision(&self) -> Option<Arc<crate::decision::DecisionModel>> {
        self.lock().ok().and_then(|inner| inner.decision.clone())
    }

    /// Run `prompt` through the model once and return the scores `read`
    /// asks for, with the number of prompt tokens. No token is generated.
    pub fn decision_scores(
        &self,
        prompt: &str,
        read: &crate::decision::DecisionRead,
    ) -> Result<(Vec<f32>, usize), AllpacaError> {
        let params = GenerationParams {
            max_tokens: 1,
            ..GenerationParams::default()
        };
        let handle = self.begin_request_probed(prompt, None, params, Some(read))?;
        let id = handle.id;
        loop {
            let mut inner = self.lock()?;
            let active = request(&inner, id)?;
            if active.probe.as_ref().is_some_and(|p| p.done) {
                break;
            }
            if active.prefill_pos >= active.prompt_len {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "decision request ended without a result".into(),
                )));
            }
            self.drain_foreign_logits(&mut inner, id)?;
            self.prefill_chunk(&mut inner, id)?;
        }
        let inner = self.lock()?;
        let active = request(&inner, id)?;
        let prompt_tokens = active.prompt_len;
        let probe = active.probe.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("not a decision request".into()))
        })?;
        let scores = if probe.positions.is_empty() {
            probe.label_logits.clone()
        } else {
            let dim = active.last_hidden.len();
            if probe.rows.len() != probe.positions.len() * dim {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "decision request read {} of {} hidden states; this backend does not return them",
                    probe.rows.len() / dim.max(1),
                    probe.positions.len()
                ))));
            }
            let model = inner.decision.as_ref().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::InvalidConfig("not a decision model".into()))
            })?;
            model.pointer_scores(&probe.rows)?
        };
        Ok((scores, prompt_tokens))
    }

    pub fn step(&self, handle: &mut RequestHandle) -> Result<StepResult, AllpacaError> {
        self.step_id(handle.id)
    }

    /// Advance the longest-waiting runnable Request. `None` when nothing is runnable
    /// (idle, or only finished Requests remain until their handles are dropped).
    ///
    /// Under concurrency (`≥2` runnable), also advances a stagger partner and any
    /// soft batch-GEMM plan peers in the same tick. Peer `StepResult`s are **parked**
    /// so they still surface as later `PollEvent`s (prior silent dual-step broke
    /// interleave observability).
    pub fn poll(&self) -> Result<Option<allpaca_types::PollEvent>, AllpacaError> {
        let _tick = self.poll_tick.lock().unwrap_or_else(|e| e.into_inner());
        self.poll_tick_locked()
    }

    /// One `poll` tick for the caller driving `id`: another Request's event
    /// goes back to the front of that Request's queue before the tick ends.
    /// It was the oldest undelivered result for that Request, and while it
    /// sat outside the queue that Request could be stepped again and hand
    /// its next token to its own caller first.
    fn poll_for(&self, id: u64) -> Result<PollFor, AllpacaError> {
        let _tick = self.poll_tick.lock().unwrap_or_else(|e| e.into_inner());
        // A tick that just ended may have parked this Request's result, or
        // finished it: check both with no tick in flight, so a result still
        // held by a running tick is never mistaken for the end of the stream.
        if let Some(parked) = self.take_parked(id)? {
            return Ok(PollFor::Result(parked));
        }
        if self.request_fully_done(id)? {
            return Ok(PollFor::Done);
        }
        match self.poll_tick_locked()? {
            Some(ev) if ev.request_id == id => Ok(PollFor::Result(ev.result)),
            Some(ev) => {
                let mut inner = self.lock()?;
                if let Some(active) = inner.active.get_mut(&ev.request_id) {
                    active.parked.push_front(ev.result);
                }
                Ok(PollFor::Idle)
            }
            None => Ok(PollFor::Idle),
        }
    }

    fn poll_tick_locked(&self) -> Result<Option<allpaca_types::PollEvent>, AllpacaError> {
        if let Some(ev) = self.poll_decode_batch()? {
            return Ok(Some(ev));
        }
        loop {
            let (id, waited_ms, is_prefill, stagger_partner, batch_gemm_plan) = {
                let mut inner = self.lock()?;
                let schedule = self.config.engine.poll_schedule;
                let max_pref = self.config.engine.max_consecutive_prefill_picks;
                let streak = inner.consecutive_prefill_picks;
                let candidates: Vec<(u64, Instant, bool)> = inner
                    .active
                    .iter()
                    .filter(|(_, r)| {
                        !(r.finished && r.emit_queue.is_empty() && r.parked.is_empty())
                    })
                    .map(|(id, r)| {
                        let prefill = r.prefill_pos < r.prompt_len
                            && r.emit_queue.is_empty()
                            && r.parked.is_empty();
                        (*id, r.last_token_at, prefill)
                    })
                    .collect();
                let mut stagger_partner: Option<u64> = None;
                let mut batch_gemm_plan: Option<BatchGemmPlan> = None;
                if candidates.len() >= 2 {
                    let n_layers = inner
                        .target
                        .as_ref()
                        .map(|m| m.layers.len())
                        .unwrap_or(1)
                        .max(1);
                    let hints: Vec<ActiveLayerHint> = candidates
                        .iter()
                        .map(|(rid, _, pref)| {
                            let decode_layer = inner
                                .active
                                .get(rid)
                                .map(|r| (r.decode_step as usize) % n_layers)
                                .unwrap_or(0);
                            ActiveLayerHint {
                                request_id: *rid,
                                phase: if *pref {
                                    StaggerPhase::Prefill
                                } else {
                                    StaggerPhase::Decode
                                },
                                layer: if *pref { 0 } else { decode_layer },
                                cpu_expert_pending: inner
                                    .target
                                    .as_ref()
                                    .and_then(|m| m.expert_lru.as_ref())
                                    .is_some_and(|lru| lru.cpu_runs() > 0),
                            }
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
                    // Soft packed-GEMM: plan + same-tick co-step of other members.
                    let shape = inner
                        .target
                        .as_ref()
                        .map(|m| ModelShapeKey {
                            n_layers: m.config.n_layers,
                            dim: m.config.dim,
                            vocab_size: m.config.vocab_size,
                        })
                        .unwrap_or(ModelShapeKey {
                            n_layers: n_layers,
                            dim: 0,
                            vocab_size: 0,
                        });
                    let gemm_cands: Vec<BatchGemmCandidate> = candidates
                        .iter()
                        .map(|(rid, _, pref)| BatchGemmCandidate {
                            request_id: *rid,
                            phase: if *pref {
                                BatchGemmPhase::Prefill
                            } else {
                                BatchGemmPhase::Decode
                            },
                            shape,
                            m_tokens: if *pref { 0 } else { 1 },
                        })
                        .collect();
                    if let Some(plan) = plan_batch_gemm(&gemm_cands) {
                        self.metrics
                            .batch_gemm_plans
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        batch_gemm_plan = Some(plan);
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
                if let Some(id) = id {
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
                    (
                        Some(id),
                        waited_ms,
                        is_prefill,
                        stagger_partner,
                        batch_gemm_plan,
                    )
                } else {
                    // Drop the mutex before rebalance (it re-locks).
                    (None, 0.0, false, None, None)
                }
            };
            let Some(id) = id else {
                let _ = self.maybe_auto_elastic_rebalance();
                return Ok(None);
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
            self.metrics
                .poll_wait_ms_ema_x100
                .store((ema * 100.0) as u64, std::sync::atomic::Ordering::Relaxed);
            let dual_partner = stagger_partner.filter(|&p| p != id);
            let batch_partners = batch_gemm_plan
                .as_ref()
                .map(|p| batch_gemm_other_ids(p, id))
                .unwrap_or_default();
            if let Some(plan) = batch_gemm_plan.as_ref() {
                self.mark_cohort_defer_lm_head(plan);
            }
            if dual_partner.is_some() {
                self.micro_batch.begin_dual_step(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0),
                );
            }
            if let Some(partner) = dual_partner {
                let stagger_plan = {
                    let inner = self.lock()?;
                    let n_layers = inner
                        .target
                        .as_ref()
                        .map(|m| m.layers.len())
                        .unwrap_or(1)
                        .max(1);
                    let hints: Vec<ActiveLayerHint> = inner
                        .active
                        .iter()
                        .filter(|(rid, r)| {
                            (**rid == id || **rid == partner)
                                && !(r.finished && r.emit_queue.is_empty() && r.parked.is_empty())
                        })
                        .map(|(rid, r)| {
                            let prefill = r.prefill_pos < r.prompt_len
                                && r.emit_queue.is_empty()
                                && r.parked.is_empty();
                            let decode_layer = (r.decode_step as usize) % n_layers;
                            ActiveLayerHint {
                                request_id: *rid,
                                phase: if prefill {
                                    StaggerPhase::Prefill
                                } else {
                                    StaggerPhase::Decode
                                },
                                layer: if prefill { 0 } else { decode_layer },
                                cpu_expert_pending: inner
                                    .target
                                    .as_ref()
                                    .and_then(|m| m.expert_lru.as_ref())
                                    .is_some_and(|lru| lru.cpu_runs() > 0),
                            }
                        })
                        .collect();
                    plan_micro_batch(&hints)
                };
                if let (Some(gpu), Some(cpu)) = (stagger_plan.gpu, stagger_plan.cpu) {
                    let partner_norm = {
                        let inner = self.lock()?;
                        inner
                            .active
                            .get(&cpu.request_id)
                            .map(|r| r.moe_norm.clone())
                            .unwrap_or_default()
                    };
                    self.layer_stagger.begin(
                        gpu.request_id,
                        gpu.layer,
                        cpu.request_id,
                        cpu.layer,
                        &partner_norm,
                    );
                }
            }
            let primary = match self.step_id(id) {
                Ok(result) => result,
                // Peer Drop removed the Request between pick and step.
                Err(AllpacaError::Transient(TransientError::RequestBusy)) => {
                    if dual_partner.is_some() {
                        self.micro_batch.end_dual_step();
                        self.layer_stagger.end();
                    }
                    continue;
                }
                Err(e) => {
                    if dual_partner.is_some() {
                        self.micro_batch.end_dual_step();
                        self.layer_stagger.end();
                    }
                    return Err(e);
                }
            };
            // Same-tick partner: co-progress then park so the result remains a
            // PollEvent when that request is next picked (observability intact).
            let mut batch_gemm_co_stepped = false;
            if let Some(partner) = dual_partner {
                match self.step_id(partner) {
                    Ok(partner_result) => {
                        self.park_result(partner, partner_result)?;
                        self.metrics
                            .micro_batch_dual_steps
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if batch_partners.contains(&partner) {
                            batch_gemm_co_stepped = true;
                        }
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
            for partner in batch_partners {
                if dual_partner == Some(partner) {
                    continue;
                }
                match self.step_id(partner) {
                    Ok(partner_result) => {
                        self.park_result(partner, partner_result)?;
                        batch_gemm_co_stepped = true;
                    }
                    Err(AllpacaError::Transient(TransientError::RequestBusy)) => {
                        let mut inner = self.lock()?;
                        inner.prefer_next = Some(partner);
                    }
                    Err(e) => return Err(e),
                }
            }
            if batch_gemm_co_stepped {
                self.metrics
                    .batch_gemm_executes
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Layer-fused packed HIP: one gemm_nt per shared-A decode weight.
                if let Some(hip) = self.hip.as_ref() {
                    let m = batch_gemm_plan.as_ref().map(|p| p.m_tokens).unwrap_or(0);
                    if m >= 2 {
                        if let Some(plan) = batch_gemm_plan.as_ref() {
                            let (launched, applied, hidden) =
                                self.try_packed_batch_gemm_hip(hip, plan);
                            if launched > 0 {
                                self.metrics
                                    .batch_gemm_hip_kernels
                                    .fetch_add(launched, std::sync::atomic::Ordering::Relaxed);
                            }
                            if applied > 0 {
                                self.metrics
                                    .batch_gemm_hip_applied
                                    .fetch_add(applied, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                self.fallback_deferred_lm_head(plan);
                            }
                            if hidden > 0 {
                                self.metrics
                                    .batch_gemm_residual_hidden
                                    .fetch_add(hidden, std::sync::atomic::Ordering::Relaxed);
                            }
                            self.clear_cohort_defer_lm_head(plan);
                        }
                    }
                }
            } else if let Some(plan) = batch_gemm_plan.as_ref() {
                self.fallback_deferred_lm_head(plan);
                self.clear_cohort_defer_lm_head(plan);
            }
            self.layer_stagger.end();
            self.note_elastic_pressure_after_decode()?;
            return Ok(Some(allpaca_types::PollEvent {
                request_id: id,
                result: primary,
            }));
        }
    }

    fn mark_cohort_defer_lm_head(&self, plan: &BatchGemmPlan) {
        if self.hip.is_none() || plan.m_tokens < 2 || self.config.speculative.enabled {
            return;
        }
        let Ok(mut inner) = self.lock() else {
            return;
        };
        let Some(target) = inner.target.as_ref() else {
            return;
        };
        let dim = target.config.dim;
        if dim == 0 || target.lm_head.is_empty() || target.lm_head.len() % dim != 0 {
            return;
        }
        for &rid in &plan.request_ids {
            if let Some(req) = inner.active.get_mut(&rid) {
                req.defer_lm_head = true;
            }
        }
    }

    fn clear_cohort_defer_lm_head(&self, plan: &BatchGemmPlan) {
        let Ok(mut inner) = self.lock() else {
            return;
        };
        for &rid in &plan.request_ids {
            if let Some(req) = inner.active.get_mut(&rid) {
                req.defer_lm_head = false;
            }
        }
    }

    /// CPU lm_head from fresh `output_norm` when packed HIP apply misses.
    fn fallback_deferred_lm_head(&self, plan: &BatchGemmPlan) {
        let Ok(mut inner) = self.lock() else {
            return;
        };
        let Some(target) = inner.target.clone() else {
            return;
        };
        let dim = target.config.dim;
        if dim == 0 || target.lm_head.is_empty() || target.lm_head.len() % dim != 0 {
            return;
        }
        let mut ids = Vec::new();
        let mut rows = Vec::new();
        for &rid in &plan.request_ids {
            let Some(req) = inner.active.get(&rid) else {
                continue;
            };
            if req.defer_lm_head && req.last_hidden_fresh && req.last_hidden.len() == dim {
                ids.push(rid);
                rows.push(req.last_hidden.clone());
            }
        }
        if ids.is_empty() {
            return;
        }
        let n = target.lm_head.len() / dim;
        let mut c_rows: Vec<Vec<f32>> = ids.iter().map(|_| vec![0.0f32; n]).collect();
        let a_refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
        let mut c_refs: Vec<&mut [f32]> = c_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
        if apply_packed_lm_head(&a_refs, &target.lm_head, dim, &mut c_refs).is_err() {
            return;
        }
        let vocab = target.config.vocab_size;
        for (i, rid) in ids.into_iter().enumerate() {
            let Some(req) = inner.active.get_mut(&rid) else {
                continue;
            };
            let row = &c_rows[i];
            let n_logits = n.min(vocab).min(row.len());
            if n_logits == 0 {
                continue;
            }
            if req.pending_logits.len() < n_logits {
                req.pending_logits.resize(n_logits, 0.0);
            }
            req.pending_logits[..n_logits].copy_from_slice(&row[..n_logits]);
            req.last_hidden_fresh = false;
        }
    }

    /// Returns `(kernels_launched, applied, residual_hidden)` where `applied`
    /// is 1 when packed lm_head logits were written into at least one Request's
    /// `pending_logits` (consumed by the next sample). Only uses fresh
    /// `output_norm` rows for lm_head. `residual_hidden` counts Requests whose
    /// `last_hidden` was overwritten with the packed `x+WO+FFN` stream.
    fn try_packed_batch_gemm_hip(
        &self,
        hip: &HipWorkspace,
        plan: &BatchGemmPlan,
    ) -> (u64, u64, u64) {
        let Ok(inner) = self.lock() else {
            return (0, 0, 0);
        };
        let Some(target) = inner.target.clone() else {
            return (0, 0, 0);
        };
        let dim = target.config.dim;
        let vocab = target.config.vocab_size;
        let n_heads = target.config.n_heads;
        let n_kv_heads = target.config.n_kv_heads.max(1);
        let head_dim = target.config.head_dim();
        if dim == 0 || plan.request_ids.len() < 2 {
            return (0, 0, 0);
        }
        let attn_w = target.layers.first().map(|layer| {
            (
                layer.wq.as_slice().to_vec(),
                layer.wk.as_slice().to_vec(),
                layer.wv.as_slice().to_vec(),
                layer.wo.as_slice().to_vec(),
            )
        });
        let weights: Vec<(Vec<f32>, usize, usize)> = if let Some(layer) = target.layers.first() {
            decode_layer_attn_shared_a_weights(
                dim,
                layer.wq.as_slice(),
                layer.wk.as_slice(),
                layer.wv.as_slice(),
                layer.wo.as_slice(),
            )
            .into_iter()
            .map(|w| (w.b.to_vec(), w.n, w.k))
            .collect()
        } else {
            Vec::new()
        };
        let ffn_weights = target.layers.first().map(|layer| {
            (
                layer.w_gate.as_slice().to_vec(),
                layer.w_up.as_slice().to_vec(),
                layer.w_down.as_slice().to_vec(),
            )
        });
        let lm_head = if !target.lm_head.is_empty() && target.lm_head.len() % dim == 0 {
            let n = target.lm_head.len() / dim;
            Some((target.lm_head.clone(), n, dim))
        } else {
            None
        };
        let has_ffn = ffn_weights
            .as_ref()
            .is_some_and(|(g, u, d)| !g.is_empty() && !u.is_empty() && !d.is_empty());
        if weights.is_empty() && !has_ffn && lm_head.is_none() {
            return (0, 0, 0);
        }
        let mut apply_ids: Vec<u64> = Vec::new();
        let mut a_rows: Vec<Vec<f32>> = Vec::new();
        for &rid in &plan.request_ids {
            let Some(req) = inner.active.get(&rid) else {
                continue;
            };
            if req.last_hidden_fresh && req.last_hidden.len() == dim {
                apply_ids.push(rid);
                a_rows.push(req.last_hidden.clone());
            }
        }
        drop(inner);
        if a_rows.len() < 2 {
            return (0, 0, 0);
        }

        let a_refs: Vec<&[f32]> = a_rows.iter().map(|v| v.as_slice()).collect();
        let mut fused_rows = a_rows.clone();
        let mut launched = 0u64;
        let mut attn_fused = false;
        if let Some((wq, wk, wv, wo)) = attn_w.as_ref() {
            let mut attn_out: Vec<Vec<f32>> = a_rows
                .iter()
                .map(|a| {
                    let m = a.len() / dim;
                    vec![0.0f32; m.saturating_mul(dim)]
                })
                .collect();
            let mut out_refs: Vec<&mut [f32]> =
                attn_out.iter_mut().map(|v| v.as_mut_slice()).collect();
            if let Ok(n) = apply_packed_attn_hip(
                hip,
                &a_refs,
                wq,
                wk,
                wv,
                wo,
                dim,
                n_heads,
                n_kv_heads,
                head_dim,
                &mut out_refs,
            ) {
                launched += n;
                attn_fused = true;
                let y_refs: Vec<&[f32]> = attn_out.iter().map(|v| v.as_slice()).collect();
                let mut x_refs: Vec<&mut [f32]> =
                    fused_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
                let _ = residual_add_rows(&mut x_refs, &y_refs);
            }
        }
        if !attn_fused {
            for (b, n, k) in &weights {
                let mut c_rows: Vec<Vec<f32>> = a_rows
                    .iter()
                    .map(|a| vec![0.0f32; (a.len() / k) * n])
                    .collect();
                let mut c_refs: Vec<&mut [f32]> =
                    c_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
                if packed_gemm_nt_hip(hip, &a_refs, b, &mut c_refs, *n, *k).is_ok() {
                    launched += 1;
                }
            }
        }

        if let Some((w_gate, w_up, w_down)) = ffn_weights {
            let mut ffn_out: Vec<Vec<f32>> = fused_rows
                .iter()
                .map(|a| {
                    let m = a.len() / dim;
                    vec![0.0f32; m * dim]
                })
                .collect();
            let ffn_ok = {
                let ffn_a: Vec<&[f32]> = fused_rows.iter().map(|v| v.as_slice()).collect();
                let mut out_refs: Vec<&mut [f32]> =
                    ffn_out.iter_mut().map(|v| v.as_mut_slice()).collect();
                apply_packed_ffn_hip(hip, &ffn_a, &w_gate, &w_up, &w_down, dim, &mut out_refs)
            };
            if let Ok(n) = ffn_ok {
                launched += n;
                let y_refs: Vec<&[f32]> = ffn_out.iter().map(|v| v.as_slice()).collect();
                let mut x_refs: Vec<&mut [f32]> =
                    fused_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
                let _ = residual_add_rows(&mut x_refs, &y_refs);
            }
        }

        let mut applied = 0u64;
        let mut hidden_writes = 0u64;
        if let Some((b, n, k)) = lm_head {
            let mut c_rows: Vec<Vec<f32>> = a_rows
                .iter()
                .map(|a| vec![0.0f32; (a.len() / k) * n])
                .collect();
            let mut c_refs: Vec<&mut [f32]> = c_rows.iter_mut().map(|v| v.as_mut_slice()).collect();
            if packed_gemm_nt_hip(hip, &a_refs, &b, &mut c_refs, n, k).is_ok() {
                launched += 1;
                // Consume C: install packed logits for the next sample path.
                if let Ok(mut inner) = self.lock() {
                    let mut wrote = false;
                    for (i, &rid) in apply_ids.iter().enumerate() {
                        let Some(req) = inner.active.get_mut(&rid) else {
                            continue;
                        };
                        let row = &c_rows[i];
                        let n_logits = n.min(vocab).min(row.len());
                        if n_logits == 0 {
                            continue;
                        }
                        if req.pending_logits.len() < n_logits {
                            req.pending_logits.resize(n_logits, 0.0);
                        }
                        req.pending_logits[..n_logits].copy_from_slice(&row[..n_logits]);
                        // This hidden row is spent: a later tick in which the
                        // Request does not forward must not re-apply it (it
                        // would sample the previous position's logits again).
                        req.last_hidden_fresh = false;
                        wrote = true;
                    }
                    if wrote {
                        applied = 1;
                        // lm_head already consumed `a_rows` (`output_norm`). Persist
                        // fused `x+WO+FFN` into decode hidden unless MTP needs
                        // the original output_norm for `h_nextn`.
                        if !mtp_draft_active(&inner) {
                            for (i, &rid) in apply_ids.iter().enumerate() {
                                let Some(req) = inner.active.get_mut(&rid) else {
                                    continue;
                                };
                                if write_residual_into_hidden(
                                    &mut [req.last_hidden.as_mut_slice()],
                                    &[fused_rows[i].as_slice()],
                                )
                                .is_ok()
                                {
                                    hidden_writes += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        (launched, applied, hidden_writes)
    }

    fn step_id(&self, id: u64) -> Result<StepResult, AllpacaError> {
        let mut inner = self.lock()?;
        self.drain_foreign_logits(&mut inner, id)?;
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
            if request(&inner, id)?.finished {
                // A decision request: its prompt is all there is.
                return Ok(StepResult::Finished(FinishReason::Stop));
            }
        }

        let json_constraints = inner.flags.json_constraints;
        let structured = json_constraints && self.config.structured_output.enabled;
        let suppress_draft = request(&inner, id)?.suppress_draft;
        let spec_limit = self.config.speculative.max_concurrent_requests;
        let live = inner.active.values().filter(|r| !r.finished).count();
        // DFlash2 keeps one sequence's context ring: with company it steps
        // aside and the requests batch-decode instead.
        let crowded = (spec_limit > 0 && live > spec_limit) || (inner.dflash2.is_some() && live > 1);
        let speculate = self.config.speculative.enabled
            && (inner.flags.ngram_speculation
                || inner.flags.draft_model_speculation
                || inner.dspark.is_some()
                || inner.dflash2.is_some())
            && !suppress_draft
            && !crowded;
        // One DFlash2 pass drafts the whole block at the same cost, so it
        // always proposes the full block; verify then takes what matches.
        let step_depth = match (speculate, inner.dflash2.as_ref()) {
            (false, _) => 0,
            (true, Some(p)) => (p.spec().block_size - 1).min(self.config.speculative.max_depth),
            (true, None) => inner.scheduler.step_depth(),
        };
        let (json_mode, json_prefix, temperature, remaining, depth, has_grammar) = {
            let active = request(&inner, id)?;
            let remaining = active.params.max_tokens.saturating_sub(active.generated);
            let depth = step_depth.min(remaining.saturating_sub(1));
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
                m = allpaca_types::TokenMask::allow_all(
                    inner.tokenizer.as_ref().unwrap().vocab_size(),
                );
            }
            Some(m)
        } else {
            None
        };

        let ctx_now = request(&inner, id)?.tokens.clone();
        // Prefer linear MTP/draft-ahead when a warm next span is queued; tree
        // rebuild would discard the overlapped proposal.
        let ahead_ready = inner
            .draft_ahead
            .as_ref()
            .is_some_and(|a| a.ctx == ctx_now && !a.tokens.is_empty());
        let forest_ok = inner
            .target
            .as_ref()
            .is_some_and(|t| t.supports_forest_verify());
        if inner.flags.tree_speculation && !forest_ok {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "speculative.tree needs host weights and no recurrent layers; using linear verify"
                );
            }
        }
        let use_tree =
            speculate && inner.flags.tree_speculation && forest_ok && depth > 0 && !ahead_ready;
        self.metrics
            .target_evals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let micro_batch = if self.micro_batch.dual_active() {
            Some(self.micro_batch.as_ref())
        } else {
            None
        };
        let hooks = DecodePollHooks {
            layer_stagger: Some(self.layer_stagger.as_ref()),
            ds_probe: &self.ds_probe,
            ds_queue: &self.ds_queue,
            #[cfg(feature = "cuda")]
            cuda: self.cuda.as_deref(),
            #[cfg(feature = "vulkan")]
            vulkan: self.vulkan.as_deref(),
        };
        let step_started = Instant::now();
        let produced = if use_tree {
            let max_nodes = self.config.speculative.tree.max_nodes.max(1);
            let draft = if inner.flags.draft_model_speculation {
                inner.draft_model.as_deref()
            } else {
                None
            };
            let forest =
                build_token_forest(&inner.ngram, draft, &ctx_now, max_nodes, depth.max(1))?;
            self.metrics.tree_nodes.fetch_add(
                forest.nodes.len() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            verify_forest(
                &mut inner,
                id,
                &forest,
                mask.as_ref(),
                remaining,
                &self.metrics,
                self.hip.as_deref(),
                self.draft_hip.as_deref(),
                depth,
                temperature,
                micro_batch,
                &hooks,
            )?
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
                &hooks,
            )?
        };
        // Cost of this step by rows verified, for the throughput-aware depth
        // (a staged batched span finishes elsewhere and is not a solo step).
        if speculate && !use_tree && !produced.tokens.is_empty() {
            inner.scheduler.record_step_time(
                produced.proposed as usize + 1,
                step_started.elapsed().as_secs_f64() * 1e6,
            );
        }
        if produced.tokens.is_empty()
            && inner
                .staged_decode
                .as_ref()
                .is_some_and(|s| s.iter().any(|st| st.id == id && st.span.is_some()))
        {
            // Staged verify span: the batched decode tick finishes and emits it.
            return Ok(StepResult::Prefill {
                tokens_done: 0,
                tokens_total: 0,
            });
        }
        self.emit_produced(&mut inner, id, produced)
    }

    /// Record a step's acceptance and queue its tokens; returns the first.
    fn emit_produced(
        &self,
        inner: &mut EngineInner,
        id: u64,
        produced: Produced,
    ) -> Result<StepResult, AllpacaError> {
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
            let piece_bytes = inner
                .tokenizer
                .as_ref()
                .map(|t| t.decode_one_bytes(tok))
                .unwrap_or_default();
            let (stop, maxed, first) = {
                let active = request(inner, id)?;
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
                // The stop token ends the reply; its text (`<|im_end|>`) is
                // not part of it. Otherwise emit only complete characters.
                let mut piece = if stop {
                    String::new()
                } else {
                    active.utf8.push(&piece_bytes)
                };
                if finish.is_some() {
                    piece.push_str(&active.utf8.finish());
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
        #[cfg(feature = "cuda")]
        let cuda_resident = inner.target_cuda_resident.clone();
        #[cfg(feature = "vulkan")]
        let vulkan_resident = inner.target_vulkan_resident.clone();
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
        let mut last_hidden = {
            let active = request_mut(inner, id)?;
            std::mem::take(&mut active.last_hidden)
        };
        let (vision, dim) = (request(inner, id)?.vision.clone(), target.config.dim);
        let vision_ref = (!vision.is_empty()).then_some(&*vision);
        let mrope_ref = vision.mrope();
        let soft_row = |pos: usize| vision.soft_row(pos, dim);
        let vision_in_chunk = vision.overlaps(start, end);
        // The HIP batched forward writes soft rows itself; CUDA and DSv4 do not.
        let batch_prefill_hip = !(vision_in_chunk && target.dsv4.is_some())
            && toks.len() > 1
            && ((self.hip.is_some()
                && resident.is_some()
                && target.supports_hip_batched_prefill())
                || target.dsv4.is_some());
        #[cfg(feature = "cuda")]
        let batch_prefill_cuda = !vision_in_chunk
            && toks.len() > 1
            && self.hip.is_none()
            && self.cuda.is_some()
            && cuda_resident.is_some()
            && !mtp_draft_active(inner)
            && target.supports_device_batched_prefill();
        #[cfg(not(feature = "cuda"))]
        let batch_prefill_cuda = false;
        let mut did_batch = false;
        let probed: Vec<usize> = request(inner, id)?
            .probe
            .as_ref()
            .map(|p| p.positions.iter().copied().filter(|p| (start..end).contains(p)).collect())
            .unwrap_or_default();
        let mut probed_rows: Vec<f32> = Vec::new();
        if batch_prefill_hip {
            let mtp = mtp_draft_active(inner);
            let mut row_hiddens = Vec::new();
            target.forward_tokens(
                &toks,
                start,
                &pages,
                &mut inner.kv,
                &self.metrics,
                &mut logits,
                self.hip.as_deref(),
                resident.as_deref(),
                write_logits,
                None,
                recurrent.as_mut(),
                (mtp || !probed.is_empty()).then_some(&mut row_hiddens),
                id,
                vision_ref,
            )?;
            if row_hiddens.len() == toks.len() * dim {
                for pos in &probed {
                    probed_rows.extend_from_slice(&row_hiddens[(pos - start) * dim..][..dim]);
                }
            }
            if mtp && row_hiddens.len() == toks.len() * dim {
                // Same pairing as per-token prefill: token `pos` with h_{pos-1}.
                for (i, tok) in toks.iter().enumerate() {
                    let h_prev = if i == 0 {
                        std::mem::take(&mut last_hidden)
                    } else {
                        row_hiddens[(i - 1) * dim..i * dim].to_vec()
                    };
                    mtp_process_token(
                        inner,
                        *tok,
                        start + i,
                        &h_prev,
                        self.draft_hip.as_deref(),
                        &self.metrics,
                    )?;
                }
                last_hidden = row_hiddens[(toks.len() - 1) * dim..].to_vec();
            }
            // Device KV stays on-GPU. Host radix/anchor snapshots would D2H every
            // page (growing with seq) and break logits-only decode accounting.
            did_batch = true;
        } else if batch_prefill_cuda {
            #[cfg(feature = "cuda")]
            {
                if let (Some(cuda), Some(cres)) = (self.cuda.as_deref(), cuda_resident.as_deref()) {
                    match target.forward_tokens_cuda(
                        &toks,
                        start,
                        &pages,
                        &mut inner.kv,
                        &self.metrics,
                        &mut logits,
                        cuda,
                        cres,
                        write_logits,
                        recurrent.as_mut(),
                    ) {
                        Ok(()) => did_batch = true,
                        Err(e) => {
                            tracing::warn!(
                                start,
                                n = toks.len(),
                                error = %e,
                                "CUDA batched prefill failed; per-token fallback"
                            );
                        }
                    }
                }
            }
        }
        if !did_batch {
            let want_hidden = mtp_draft_active(inner) || !probed.is_empty();
            for (i, tok) in toks.iter().enumerate() {
                let pos = start + i;
                let last = pos + 1 >= total;
                // MTP process: (token_pos, h_{pos-1}) before target updates last_hidden.
                mtp_process_token(
                    inner,
                    *tok,
                    pos,
                    &last_hidden,
                    self.draft_hip.as_deref(),
                    &self.metrics,
                )?;
                let mut decode_side = DeviceDecodeSide {
                    last_hidden: if want_hidden {
                        Some(&mut last_hidden)
                    } else {
                        None
                    },
                    mrope: mrope_ref,
                    ..DeviceDecodeSide::new(id)
                };
                let soft = soft_row(pos);
                let used_cuda = {
                    #[cfg(feature = "cuda")]
                    {
                        let mut used = false;
                        if self.hip.is_none() {
                            if let (Some(cuda), Some(cres)) =
                                (self.cuda.as_deref(), cuda_resident.as_deref())
                            {
                                match target.forward_token_cuda(
                                    *tok,
                                    pos,
                                    &pages,
                                    &mut inner.kv,
                                    &self.metrics,
                                    &mut logits,
                                    cuda,
                                    cres,
                                    soft,
                                    recurrent.as_mut(),
                                ) {
                                    Ok(()) => used = true,
                                    Err(e) => {
                                        tracing::warn!(
                                            pos,
                                            token = *tok,
                                            error = %e,
                                            "CUDA on-device decode failed; host fallback"
                                        );
                                    }
                                }
                            }
                        }
                        used
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        false
                    }
                };
                let used_vulkan = {
                    #[cfg(feature = "vulkan")]
                    {
                        let mut used = false;
                        if self.hip.is_none() && !used_cuda {
                            if let (Some(vk), Some(vres)) =
                                (self.vulkan.as_deref(), vulkan_resident.as_deref())
                            {
                                match target.forward_token_vulkan(
                                    *tok,
                                    pos,
                                    &pages,
                                    &mut inner.kv,
                                    &self.metrics,
                                    &mut logits,
                                    vk,
                                    vres,
                                    soft,
                                ) {
                                    Ok(()) => used = true,
                                    Err(e) => {
                                        tracing::warn!(
                                            pos,
                                            token = *tok,
                                            error = %e,
                                            "Vulkan on-device decode failed; host fallback"
                                        );
                                    }
                                }
                            }
                        }
                        used
                    }
                    #[cfg(not(feature = "vulkan"))]
                    {
                        false
                    }
                };
                if !used_cuda && !used_vulkan {
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
                        Some(&mut decode_side),
                        soft,
                    )?;
                    if probed.contains(&pos) && last_hidden.len() == dim {
                        probed_rows.extend_from_slice(&last_hidden);
                    }
                }
                let device_gdn = self
                    .hip
                    .as_ref()
                    .is_some_and(|h| h.gdn_request_resident(id));
                let is_anchor = inner.anchors.enabled() && inner.anchors.is_anchor(*tok);
                // Device GDN slots are source of truth. Skip per-token D2H; sync
                // host (and store radix/anchors) only at real anchors or request end.
                let sync_host_gdn = !device_gdn || last || is_anchor;
                if sync_host_gdn {
                    if let (Some(hip), Some(state)) = (self.hip.as_deref(), recurrent.as_mut()) {
                        for_each_gdn_workspace(hip, Some(&target), |ws| {
                            ws.gdn_sync_request_to_host(id, &mut state.gdn_state)?;
                            ws.gdn_sync_conv_to_host(id, &mut state.gdn_conv)?;
                            ws.mamba2_sync_request_to_host(id, &mut state.ssm_h, &mut state.ssm_conv)
                        })?;
                    }
                }
                if sync_host_gdn && self.hip.is_none() {
                    let prefix = request(inner, id)?.tokens[..=pos].to_vec();
                    let pages_now = request(inner, id)?.pages.clone();
                    let state_ref = recurrent.clone().unwrap_or_else(|| {
                        RecurrentState::zeros(target.config.n_layers, target.config.dim)
                    });
                    inner.kv_radix.capture(
                        &prefix,
                        &state_ref,
                        &inner.kv,
                        &pages_now,
                        &target.config,
                    )?;
                    if is_anchor {
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
            }
        }
        if let Some(hip) = self.hip.as_deref() {
            hip.wait_logits_d2h(&mut logits)?;
        }
        {
            let active = request_mut(inner, id)?;
            active.prefill_pos = end;
            if let Some(probe) = active.probe.as_mut() {
                probe.rows.extend_from_slice(&probed_rows);
                if end >= total {
                    probe.label_logits = probe
                        .labels
                        .iter()
                        .map(|&t| logits.get(t as usize).copied().unwrap_or(f32::NAN))
                        .collect();
                    probe.done = true;
                    active.finished = true;
                }
            }
            active.pending_logits = logits;
            active.recurrent = recurrent;
            active.last_hidden = last_hidden;
        }
        self.metrics
            .prefill_chunks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if end >= total {
            self.capture_prompt(inner, id, &target)?;
        } else {
            let every = self.config.prompt_cache.boundary_tokens;
            if target.config.recurrent && every > 0 && end / every > start / every {
                self.capture_state(inner, id, &target, end)?;
            }
        }
        if end >= total && inner.flags.ngram_speculation {
            let tokens = request(inner, id)?.tokens.clone();
            inner.ngram.observe(&tokens);
        }
        Ok((end, total))
    }

    /// Point the prompt cache at this model and expert mask: a change clears
    /// it and opens that pair's disk folder.
    fn sync_prompt_cache(&self, inner: &mut EngineInner, target: &TinyModel) {
        let disabled = target.expert_controls.disabled();
        let key = prompt_cache_key(&target.config, self.config.model.path.as_deref(), &disabled);
        if inner.prompt_cache_key == Some(key) {
            return;
        }
        inner.prompt_cache_key = Some(key);
        inner.kv_radix.reset();
        let pc = &self.config.prompt_cache;
        let Some(dir) = pc.disk_dir.as_ref() else {
            return;
        };
        // Their state is not part of a checkpoint file.
        if target.dsv4.is_some() || target.q4x.is_some() {
            return;
        }
        let dir = dir.join(format!("{key:016x}"));
        if let Err(e) = inner.kv_radix.open_disk(&dir, pc.disk_bytes) {
            tracing::warn!(error = %e, "prompt cache: disk tier not opened");
        }
    }

    /// Keep a hybrid's recurrent state after its first `end` prompt tokens
    /// (a state checkpoint; the keys and values come from the full prompt's
    /// checkpoint when prefill ends).
    fn capture_state(
        &self,
        inner: &mut EngineInner,
        id: u64,
        target: &TinyModel,
        end: usize,
    ) -> Result<(), AllpacaError> {
        if !self.prompt_cache_reads_back(inner, id, target)? {
            return Ok(());
        }
        let active = request(inner, id)?;
        let tokens = active.tokens[..end].to_vec();
        let Some(state) = active.recurrent.clone() else {
            return Ok(());
        };
        inner.kv_radix.insert_state(&tokens, state);
        Ok(())
    }

    /// Whether the prompt cache can take this Request's state on HIP, after
    /// reading its GDN/Mamba state back to the host. Left out: models whose
    /// state the host cannot read back yet (DeepSeek-V4, qwen4exp, layers on
    /// RPC workers), and vision or decision prompts.
    fn prompt_cache_reads_back(
        &self,
        inner: &mut EngineInner,
        id: u64,
        target: &TinyModel,
    ) -> Result<bool, AllpacaError> {
        let Some(hip) = self.hip.as_deref() else {
            return Ok(false);
        };
        let layers_remote = target.rpc.as_ref().is_some_and(|rpc| rpc.placed_remote());
        if target.dsv4.is_some() || target.q4x.is_some() || layers_remote {
            return Ok(false);
        }
        let active = request_mut(inner, id)?;
        if !active.vision.is_empty() || active.probe.is_some() {
            return Ok(false);
        }
        if let Some(state) = active.recurrent.as_mut() {
            for_each_gdn_workspace(hip, Some(target), |ws| {
                ws.gdn_sync_request_to_host(id, &mut state.gdn_state)?;
                ws.gdn_sync_conv_to_host(id, &mut state.gdn_conv)?;
                ws.mamba2_sync_request_to_host(id, &mut state.ssm_h, &mut state.ssm_conv)
            })?;
        }
        Ok(true)
    }

    /// Keep a prefilled prompt in the prompt cache. The CPU path captures as
    /// it goes; on HIP the keys, values and recurrent state are read back once
    /// here (see [`Self::prompt_cache_reads_back`] for what is left out).
    fn capture_prompt(
        &self,
        inner: &mut EngineInner,
        id: u64,
        target: &TinyModel,
    ) -> Result<(), AllpacaError> {
        if !self.prompt_cache_reads_back(inner, id, target)? {
            return Ok(());
        }
        let active = request(inner, id)?;
        let tokens = active.tokens[..active.prompt_len].to_vec();
        let pages = active.pages.clone();
        let state = active
            .recurrent
            .clone()
            .unwrap_or_else(|| RecurrentState::zeros(target.config.n_layers, target.config.dim));
        inner
            .kv_radix
            .capture(&tokens, &state, &inner.kv, &pages, &target.config)
    }

    /// Whether this Engine can batch decode at all: enabled, several Request
    /// slots, a HIP-resident target that supports multi-sequence forwards, and
    /// no draft model (its per-Request state needs its own forward).
    fn batch_decode_enabled(&self, inner: &EngineInner) -> bool {
        self.config.engine.batch_decode
            && self.config.engine.max_active_requests >= 2
            && inner.target_resident.is_some()
            && inner.draft_model.is_none()
            && match (self.hip.as_deref(), inner.target.as_ref()) {
                (Some(hip), Some(t)) => t.supports_forward_batch(hip),
                _ => false,
            }
    }

    /// Requests whose next step ends in one forward the tick can batch: a plain
    /// decode token or an n-gram verify span. Prefill done, nothing queued to
    /// emit, and no grammar, JSON mode, vision, deferred lm_head or speculation
    /// that runs its own forwards.
    fn batch_decode_ids(&self, inner: &EngineInner) -> Vec<u64> {
        let cfg = &self.config;
        // N-gram verify spans ride along in the batched forward; draft-model,
        // DSpark and tree speculation run their own forwards.
        let live = inner.active.values().filter(|r| !r.finished).count();
        let own_forward_spec = cfg.speculative.enabled
            && (inner.flags.draft_model_speculation
                || inner.dspark.is_some()
                || (inner.dflash2.is_some() && live <= 1)
                || inner.flags.tree_speculation);
        let structured = inner.flags.json_constraints && cfg.structured_output.enabled;
        let mut ids: Vec<(Instant, u64)> = inner
            .active
            .iter()
            .filter(|(_, r)| {
                !r.finished
                    && r.prefill_pos >= r.prompt_len
                    && r.emit_queue.is_empty()
                    && r.parked.is_empty()
                    && r.generated < r.params.max_tokens
                    && (!own_forward_spec || r.suppress_draft)
                    && r.grammar.is_none()
                    && !(r.params.json_mode && structured)
                    && r.vision.is_empty()
                    && !r.defer_lm_head
            })
            .map(|(id, r)| (r.last_token_at, *id))
            .collect();
        // Longest-waiting first, so the returned event is the fair pick.
        ids.sort();
        ids.into_iter().map(|(_, id)| id).collect()
    }

    /// One continuous-batching tick: when two or more Requests are in plain
    /// decode, sample each one's next token and run a single batched forward
    /// for all of them. The longest-waiting Request's token is returned; the
    /// others are parked and come out on their next pick.
    fn poll_decode_batch(&self) -> Result<Option<allpaca_types::PollEvent>, AllpacaError> {
        let ids = {
            let mut inner = self.lock()?;
            if !self.batch_decode_enabled(&inner) {
                return Ok(None);
            }
            // Hand out the previous batch's parked tokens before the next
            // batch, so its members stay in step with each other.
            let parked = inner
                .active
                .iter()
                .filter(|(_, r)| !r.parked.is_empty())
                .min_by_key(|(_, r)| r.last_token_at)
                .map(|(id, _)| *id);
            if let Some(id) = parked {
                drop(inner);
                let result = self.step_id(id)?;
                return Ok(Some(allpaca_types::PollEvent {
                    request_id: id,
                    result,
                }));
            }
            let ids = self.batch_decode_ids(&inner);
            if ids.len() < 2 {
                return Ok(None);
            }
            inner.staged_decode = Some(Vec::new());
            ids
        };
        let mut results = Vec::with_capacity(ids.len());
        for &id in &ids {
            match self.step_id(id) {
                Ok(r) => results.push((id, r)),
                // Dropped between pick and step.
                Err(AllpacaError::Transient(TransientError::RequestBusy)) => {}
                Err(e) => {
                    self.lock()?.staged_decode = None;
                    return Err(e);
                }
            }
        }
        {
            let mut inner = self.lock()?;
            let staged = inner.staged_decode.take().unwrap_or_default();
            let spans = self.forward_staged_decode(&mut inner, staged)?;
            // Finish the staged n-gram verifies now that their rows' logits
            // are in: accept walk, rewind, correction, then emit.
            let hooks = DecodePollHooks {
                layer_stagger: Some(self.layer_stagger.as_ref()),
                ds_probe: &self.ds_probe,
                ds_queue: &self.ds_queue,
                #[cfg(feature = "cuda")]
                cuda: self.cuda.as_deref(),
                #[cfg(feature = "vulkan")]
                vulkan: self.vulkan.as_deref(),
            };
            for (id, tail, mut rows) in spans {
                // A previous finish may have left its correction's logits in flight.
                self.drain_foreign_logits(&mut inner, id)?;
                let walk = accept_span(
                    &mut inner,
                    id,
                    &tail.span,
                    tail.first,
                    &mut rows,
                    None,
                    tail.remaining,
                )?;
                let recurrent = request_mut(&mut inner, id)?.recurrent.take();
                let produced = finish_linear_span(
                    &mut inner,
                    id,
                    *tail,
                    walk,
                    rows,
                    Vec::new(),
                    recurrent,
                    false,
                    &self.metrics,
                    self.hip.as_deref(),
                    self.draft_hip.as_deref(),
                    &hooks,
                    None,
                )?;
                let result = self.emit_produced(&mut inner, id, produced)?;
                if let Some(slot) = results.iter_mut().find(|(rid, _)| *rid == id) {
                    slot.1 = result;
                }
            }
            inner.last_poll_id = Some(ids[0]);
            inner.consecutive_prefill_picks = 0;
        }
        let mut results = results.into_iter();
        let Some((primary_id, primary)) = results.next() else {
            return Ok(None);
        };
        for (id, r) in results {
            self.park_result(id, r)?;
        }
        self.metrics
            .poll_ticks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .poll_decode_picks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(allpaca_types::PollEvent {
            request_id: primary_id,
            result: primary,
        }))
    }

    /// Run the batched decode forward for `staged` and hand each Request its
    /// next-token logits.
    fn forward_staged_decode(
        &self,
        inner: &mut EngineInner,
        mut staged: Vec<StagedDecode>,
    ) -> Result<Vec<(u64, Box<SpanTail>, Vec<f32>)>, AllpacaError> {
        staged.retain(|st| inner.active.contains_key(&st.id));
        if staged.is_empty() {
            return Ok(Vec::new());
        }
        // Nothing may still be in flight in the pinned logits slot.
        self.drain_foreign_logits(inner, u64::MAX)?;
        let hip = self
            .hip
            .as_deref()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let target = inner
            .target
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let resident = inner
            .target_resident
            .clone()
            .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
        let vocab = target.config.vocab_size;
        let pages: Vec<Vec<Vec<u64>>> = staged
            .iter()
            .map(|st| request(inner, st.id).map(|r| r.pages.clone()))
            .collect::<Result<_, _>>()?;
        let mut recurrent: Vec<Option<RecurrentState>> = staged
            .iter()
            .map(|st| request_mut(inner, st.id).map(|r| r.recurrent.take()))
            .collect::<Result<_, _>>()?;
        let tokens: Vec<u32> = staged.iter().flat_map(|st| st.tokens.iter().copied()).collect();
        let mut row_logits = vec![0.0f32; tokens.len() * vocab];
        let result = {
            let mut segs: Vec<crate::model::SeqRows<'_>> = staged
                .iter()
                .zip(&pages)
                .zip(recurrent.iter_mut())
                .map(|((st, pages), rec)| crate::model::SeqRows {
                    start_pos: st.pos,
                    n_rows: st.tokens.len(),
                    page_ids: pages,
                    request_id: st.id,
                    recurrent: rec.as_mut(),
                })
                .collect();
            target.forward_batch(
                &tokens,
                &mut segs,
                &mut inner.kv,
                &self.metrics,
                &mut row_logits,
                hip,
                &resident,
            )
        };
        let mut spans = Vec::new();
        let mut row0 = 0usize;
        for (st, rec) in staged.into_iter().zip(recurrent) {
            let rows = row_logits[row0 * vocab..(row0 + st.tokens.len()) * vocab].to_vec();
            row0 += st.tokens.len();
            let Some(active) = inner.active.get_mut(&st.id) else {
                continue;
            };
            active.recurrent = rec;
            match st.span {
                Some(tail) => spans.push((st.id, tail, rows)),
                None => {
                    active.pending_logits = rows;
                    active.decode_step = active.decode_step.saturating_add(1);
                    active.last_hidden_fresh = false;
                }
            }
        }
        result?;
        self.metrics
            .batched_decode_steps
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.metrics
            .batched_decode_rows
            .fetch_add(row0 as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(spans)
    }

    /// Land logits still in flight for another Request in that Request's
    /// buffer before `id` runs a forward (the workspace has one pinned logits
    /// slot, so `id`'s own wait would otherwise take them).
    fn drain_foreign_logits(&self, inner: &mut EngineInner, id: u64) -> Result<(), AllpacaError> {
        let Some(owner) = inner.logits_owner.filter(|&o| o != id) else {
            return Ok(());
        };
        inner.logits_owner = None;
        let Some(hip) = self.hip.as_deref() else {
            return Ok(());
        };
        match inner.active.get_mut(&owner) {
            Some(active) => hip.wait_logits_d2h(&mut active.pending_logits),
            None => hip.wait_logits_d2h(&mut []),
        }
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, EngineInner>, AllpacaError> {
        self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig("engine mutex poisoned".into()))
        })
    }

    /// Shrink a requested `(expert cache, KV VRAM)` pair to what the board can
    /// actually hold, scaling both by the same factor so their ratio survives.
    ///
    /// Headroom is measured free VRAM plus what this process already owns, less
    /// a reserve, because the compositor and other GPU clients move underneath
    /// us. Returns the request unchanged when the sampler has no reading, so
    /// the CPU path and unsampled HIP behave exactly as before.
    fn clamp_to_measured_vram(&self, cache_bytes: usize, vram_bytes: usize) -> (usize, usize) {
        /// Leave the driver room for context, command buffers and the desktop.
        const RESERVE_BYTES: u64 = 512 << 20;
        let Some(free) = self.metrics.gpu_vram_free_bytes() else {
            return (cache_bytes, vram_bytes);
        };
        let owned = allpaca_hip::allocated_bytes();
        let headroom = free.saturating_add(owned).saturating_sub(RESERVE_BYTES);
        let want = (cache_bytes as u64).saturating_add(vram_bytes as u64);
        if want <= headroom || want == 0 {
            return (cache_bytes, vram_bytes);
        }
        let scale = headroom as f64 / want as f64;
        (
            (cache_bytes as f64 * scale) as usize,
            (vram_bytes as f64 * scale) as usize,
        )
    }

    /// Rebuild expert LRU slots vs KV VRAM at a scheduler safe point.
    /// Host (or NVMe) pool is not reloaded. GPU residency is a speed dial.
    ///
    /// The request is capped by measured board pressure: the expert pool is a
    /// speed dial, and overshooting it into an OOM costs far more than the
    /// slots would have won.
    pub fn rebalance_vram(
        &self,
        cache_bytes: usize,
        vram_bytes: usize,
    ) -> Result<(), AllpacaError> {
        self.sample_gpu();
        let (cache_bytes, vram_bytes) = self.clamp_to_measured_vram(cache_bytes, vram_bytes);
        {
            let mut inner = self.lock()?;
            if let Some(model) = inner.target.as_mut() {
                let model = Arc::get_mut(model)
                    .ok_or_else(|| AllpacaError::Transient(TransientError::RequestBusy))?;
                self.rebuild_expert_slots(model, cache_bytes);
            }
            inner.kv.set_vram_bytes(vram_bytes, &self.metrics)?;
        }
        if self.hip.is_some() {
            self.upload_resident_weights()?;
        }
        Ok(())
    }

    /// Safe-point: trim cold radix checkpoints and cold VRAM KV pages, then grow
    /// the expert LRU pool from the freed budget (OPPORTUNITY §8.2 elastic VRAM).
    /// No-op when Requests are in flight.
    pub fn rebalance_experts_elastic(&self) -> Result<(), AllpacaError> {
        self.sample_gpu();
        let (freed, new_cache, new_vram) = {
            let mut inner = self.lock()?;
            if !inner.active.is_empty() {
                return Err(AllpacaError::Transient(TransientError::RequestBusy));
            }
            let before = inner.kv_radix.checkpoint_bytes();
            // Soft pressure: trim cold half so expert LRU can expand.
            let target = before / 2;
            let mut freed = if before > target {
                inner.kv_radix.trim_to_bytes(target)
            } else {
                0
            };
            let expert_bytes = inner
                .target
                .as_ref()
                .map(|m| m.expert_slot_bytes())
                .unwrap_or(0);
            // Attention-trim: spill cold VRAM KV → DRAM for more expert slots.
            if expert_bytes > 0 {
                let want = expert_bytes.max(256);
                freed = freed.saturating_add(inner.kv.trim_cold_vram_bytes(want, &self.metrics)?);
            }
            let base_cache = self.config.weights.cache_bytes;
            let new_cache = base_cache.saturating_add(freed);
            // Give expert pool headroom by shrinking KV VRAM when we reclaimed.
            let cur_vram = self.config.kv.vram_bytes;
            let new_vram = if freed > 0 && expert_bytes > 0 {
                cur_vram.saturating_sub(freed.min(cur_vram / 4))
            } else {
                cur_vram
            };
            // Reclaimed budget is only spendable if the board still has it.
            let (new_cache, new_vram) = self.clamp_to_measured_vram(new_cache, new_vram);
            if let Some(model) = inner.target.as_mut() {
                if let Some(model) = Arc::get_mut(model) {
                    self.rebuild_expert_slots(model, new_cache);
                }
            }
            if new_vram != cur_vram {
                inner.kv.set_vram_bytes(new_vram, &self.metrics)?;
            }
            (freed, new_cache, new_vram)
        };
        if freed > 0 || new_cache > self.config.weights.cache_bytes {
            self.metrics
                .vram_rebalances
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let _ = new_vram;
        if self.hip.is_some() {
            self.upload_resident_weights()?;
        }
        Ok(())
    }

    /// Auto safe-point rebalance when radix + expert miss pressure was recorded.
    fn maybe_auto_elastic_rebalance(&self) -> Result<(), AllpacaError> {
        // Every tick reaches here, but the sampler's own clock keeps the driver
        // call at 10 Hz, so VRAM and `q*`'s busy input stay fresh without
        // anyone polling `/stats`.
        self.sample_gpu();
        let pending = {
            let mut inner = self.lock()?;
            if !inner.active.is_empty() || !inner.elastic_rebalance_pending {
                return Ok(());
            }
            inner.elastic_rebalance_pending = false;
            true
        };
        if pending {
            let _ = self.rebalance_experts_elastic();
        }
        Ok(())
    }

    #[cfg(test)]
    fn test_mark_elastic_pending(&self) {
        if let Ok(mut inner) = self.lock() {
            inner.elastic_rebalance_pending = true;
        }
    }

    /// Seed orphan VRAM KV pages for elastic attention-trim tests (no active Request).
    #[cfg(test)]
    fn test_seed_vram_kv_pages(&self, n: u32) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let page_tokens = self.config.kv.page_tokens;
        for i in 0..n {
            let _ = inner
                .kv
                .alloc_page(0, i, page_tokens, 1, 4, &self.metrics)?;
        }
        Ok(())
    }

    /// Enable a synthetic early-exit manifest for tracer tests.
    #[cfg(test)]
    fn test_enable_early_exit_manifest(&self, exit_layers: Vec<usize>) {
        if let Ok(mut inner) = self.lock() {
            if let Some(model) = inner.target.as_mut() {
                if let Some(model) = Arc::get_mut(model) {
                    model.early_exit = EarlyExitSpec {
                        supported: true,
                        exit_layers,
                        min_confidence: self.config.early_exit.min_confidence,
                        min_calibration_margin: self.config.early_exit.min_calibration_margin,
                        calibration_version: Some("test-v1".into()),
                        requires_verification: true,
                    };
                }
            }
        }
    }

    fn note_elastic_pressure_after_decode(&self) -> Result<(), AllpacaError> {
        let mut inner = self.lock()?;
        let Some(model) = inner.target.as_ref() else {
            return Ok(());
        };
        let Some(lru) = model.expert_lru.as_ref() else {
            return Ok(());
        };
        let misses = lru.misses();
        if misses > inner.elastic_last_misses {
            inner.elastic_miss_streak = inner.elastic_miss_streak.saturating_add(1);
        } else {
            inner.elastic_miss_streak = 0;
        }
        inner.elastic_last_misses = misses;
        let max_bytes = inner.kv_radix.max_bytes();
        let radix_bytes = inner.kv_radix.checkpoint_bytes();
        let radix_hot = max_bytes > 0 && radix_bytes > max_bytes * 3 / 4;
        let (vram_pages, _) = inner.kv.counts();
        let kv_hot = vram_pages >= 2;
        if (radix_hot || kv_hot) && inner.elastic_miss_streak >= 4 {
            inner.elastic_rebalance_pending = true;
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
        let keep = self.tune_keep_ids(model);
        if let Some(lru) = model.expert_lru.as_ref() {
            lru.set_keep_set(&keep);
        }
    }
}

fn open_hip_workspace(config: &Config, inventory: &DeviceInventory) -> Option<HipWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_hip::runtime_linked() {
        return None;
    }
    let gpus: Vec<_> = inventory
        .devices
        .iter()
        .filter(|d| d.kind == allpaca_types::DeviceKind::HipGpu)
        .collect();
    if gpus.is_empty() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "hip" {
        return None;
    }
    let device =
        if inventory.target_device >= 0 && gpus.iter().any(|d| d.id == inventory.target_device) {
            inventory.target_device
        } else {
            gpus[0].id
        };
    match HipWorkspace::new(device, 8 * 1024 * 1024, 256) {
        Ok(ws) => {
            tracing::info!(
                device,
                bytes = ws.allocated_bytes(),
                "HIP workspace allocated"
            );
            spawn_blas_warm_up(device);
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
    if !inventory
        .devices
        .iter()
        .any(|d| d.id == draft && d.kind == allpaca_types::DeviceKind::HipGpu)
    {
        return None;
    }
    match HipWorkspace::new(draft, 8 * 1024 * 1024, 256) {
        Ok(ws) => {
            tracing::info!(
                device = draft,
                bytes = ws.allocated_bytes(),
                "HIP draft workspace allocated (resident; P2P not required)"
            );
            spawn_blas_warm_up(draft);
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "HIP draft workspace failed; CPU draft fallback");
            None
        }
    }
}

#[cfg(feature = "cuda")]
fn open_cuda_workspace(
    config: &Config,
    inventory: &DeviceInventory,
) -> Option<allpaca_cuda::CudaWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_cuda::runtime_linked() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "cuda" {
        return None;
    }
    // Prefer HIP when both exist under auto; explicit `cuda` selects CUDA.
    if config.engine.gpu_backend == "auto"
        && inventory
            .devices
            .iter()
            .any(|d| d.kind == allpaca_types::DeviceKind::HipGpu)
    {
        return None;
    }
    let gpus: Vec<_> = inventory
        .devices
        .iter()
        .filter(|d| d.kind == allpaca_types::DeviceKind::CudaGpu)
        .collect();
    if gpus.is_empty() {
        return None;
    }
    let device =
        if inventory.target_device >= 0 && gpus.iter().any(|d| d.id == inventory.target_device) {
            inventory.target_device
        } else {
            gpus[0].id
        };
    match allpaca_cuda::CudaWorkspace::new(device, 8 * 1024 * 1024) {
        Ok(ws) => {
            tracing::info!(
                device,
                bytes = ws.allocated_bytes(),
                "CUDA workspace allocated"
            );
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "CUDA workspace failed; CPU fallback");
            None
        }
    }
}

#[cfg(feature = "vulkan")]
fn open_vulkan_workspace(
    config: &Config,
    inventory: &DeviceInventory,
) -> Option<allpaca_vulkan::VulkanWorkspace> {
    if config.engine.cpu_only || config.engine.gpu_backend == "cpu" {
        return None;
    }
    if !allpaca_vulkan::runtime_linked() {
        return None;
    }
    if config.engine.gpu_backend != "auto" && config.engine.gpu_backend != "vulkan" {
        return None;
    }
    if config.engine.gpu_backend == "auto"
        && inventory
            .devices
            .iter()
            .any(|d| d.kind == allpaca_types::DeviceKind::HipGpu)
    {
        return None;
    }
    if config.engine.gpu_backend == "auto"
        && inventory
            .devices
            .iter()
            .any(|d| d.kind == allpaca_types::DeviceKind::CudaGpu)
    {
        return None;
    }
    let gpus: Vec<_> = inventory
        .devices
        .iter()
        .filter(|d| d.kind == allpaca_types::DeviceKind::VulkanGpu)
        .collect();
    if gpus.is_empty() {
        return None;
    }
    let device =
        if inventory.target_device >= 0 && gpus.iter().any(|d| d.id == inventory.target_device) {
            inventory.target_device
        } else {
            gpus[0].id
        };
    match allpaca_vulkan::VulkanWorkspace::new(device, 8 * 1024 * 1024) {
        Ok(ws) => {
            tracing::info!(
                device,
                bytes = ws.allocated_bytes(),
                "Vulkan workspace allocated"
            );
            Some(ws)
        }
        Err(e) => {
            tracing::warn!(error = %e, "Vulkan workspace failed; CPU fallback");
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
    if !providers.contains(&DraftProviderKind::DraftModel)
        && !providers.contains(&DraftProviderKind::Mtp)
    {
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

/// True when MTP decode text is empty, whitespace-only, or dropped-word junk.
pub fn mtp_text_is_garbled(text: &str) -> bool {
    text.chars().all(char::is_whitespace) || mtp_text_has_dropped_words(text)
}

fn mtp_text_has_dropped_words(text: &str) -> bool {
    let t = text.to_ascii_lowercase();
    // Live Qwen MTP soaks drop a digit from "400-word" → "00-word".
    // Do not use raw `contains("00-word")`: that also matches "400-word".
    if t.match_indices("00-word")
        .any(|(i, _)| i == 0 || !t.as_bytes()[i - 1].is_ascii_digit())
    {
        return true;
    }
    // Determiner + finite verb with the subject noun missing ("The wants").
    if t.contains("the wants ") || t.contains("the is ") || t.contains("the needs ") {
        return true;
    }
    // Preposition stacked on "to " after a dropped gerund ("for to code").
    if t.contains("for to ") || t.contains("in to code") {
        return true;
    }
    // "Let me" with the verb dropped ("Let me about", "Let me words").
    t.contains("let me about ") || t.contains("let me words ")
}

/// Run `f` on every HIP workspace that holds device GDN / Mamba-2 state for
/// the target: the target GPU, then the local layer-split peer. Recurrent
/// slots live on the workspace that ran the layer, so clone / rewind / clear
/// must reach both or the peer's layers keep state from rejected tokens (and
/// leak it per request).
/// Names the prompt cache's disk folder: checkpoints only resume the model
/// file, shape and expert mask that wrote them.
fn prompt_cache_key(
    model: &ModelConfig,
    file: Option<&std::path::Path>,
    disabled: &[Vec<u32>],
) -> u64 {
    let mut text = format!("{model:?}|{disabled:?}");
    if let Some((file, meta)) = file.and_then(|f| std::fs::metadata(f).ok().map(|m| (f, m))) {
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_nanos());
        text.push_str(&format!("|{}|{}|{modified}", file.display(), meta.len()));
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn for_each_gdn_workspace(
    hip: &HipWorkspace,
    target: Option<&TinyModel>,
    mut f: impl FnMut(&HipWorkspace) -> Result<(), AllpacaError>,
) -> Result<(), AllpacaError> {
    f(hip)?;
    let peer = target
        .and_then(|t| t.rpc.as_ref())
        .and_then(|rpc| rpc.local_peer());
    if let Some((_, peer_hip, _)) = peer {
        if !std::ptr::eq(peer_hip.as_ref(), hip) {
            f(&peer_hip)?;
        }
    }
    Ok(())
}

/// Every workspace holding the target's GDN state can roll a clone forward
/// `keep` journaled steps (see [`HipWorkspace::gdn_rewind_request_to`]).
fn gdn_all_can_replay(hip: &HipWorkspace, target: &TinyModel, id: u64, keep: usize) -> bool {
    let mut all = true;
    let _ = for_each_gdn_workspace(hip, Some(target), |ws| {
        all &= ws.gdn_can_replay(id, keep);
        Ok(())
    });
    all
}


/// Load a `--draft-model` GGUF. MTP sidecars (`mtp-*.gguf`) ship only the
/// trailing NextN block (`blk.{n_layers}`), so they load as an MTP draft.
fn load_draft_gguf(file: &GgufFile) -> Result<TinyModel, AllpacaError> {
    if file.tensor("blk.0.attn_norm.weight").is_none() {
        if let Ok(mtp) = TinyModel::from_gguf_mtp(file) {
            return Ok(mtp);
        }
    }
    TinyModel::from_gguf(file)
}

fn mtp_draft_active(inner: &EngineInner) -> bool {
    let Some(model) = inner.draft_model.as_ref() else {
        return false;
    };
    model.config.gguf_layer_base > 0
        || model.mtp_nextn.is_some()
        || inner
            .draft_resident
            .as_ref()
            .is_some_and(|r| r.has_mtp_nextn())
}

/// llama.cpp `process()`: pair `(token_k, h_{k-1})` into the MTP draft KV.
fn mtp_process_token(
    inner: &mut EngineInner,
    token: u32,
    pos: usize,
    h_prev: &[f32],
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    if !mtp_draft_active(inner) {
        return Ok(());
    }
    let model = inner
        .draft_model
        .clone()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    let dim = model.config.dim;
    let kv_dim = model.config.n_kv_heads * model.config.head_dim();
    if h_prev.len() != dim {
        return Ok(());
    }
    if inner.draft_kv.seq_len() > pos {
        inner.draft_kv.truncate(pos, kv_dim);
    }
    if inner.draft_kv.seq_len() != pos {
        // Missing earlier process steps — skip rather than corrupt slots.
        metrics
            .mtp_process_skips
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("ALLPACA_MTP_PROBE").ok().as_deref() == Some("1") {
            eprintln!(
                "MTP_PROBE process skip seq_len={} pos={}",
                inner.draft_kv.seq_len(),
                pos
            );
        }
        return Ok(());
    }
    inner.draft_kv.seed_h_nextn_from_target(h_prev);
    let resident = inner.draft_resident.clone();
    model.process_dense_on(
        token,
        pos,
        &mut inner.draft_kv,
        draft_hip,
        resident.as_deref(),
        Some(metrics),
    )?;
    inner.draft_kv.record(token);
    Ok(())
}

/// Process an assumed-accepted span into a process-KV snapshot.
/// Token 0 pairs with target `pending_h`; token `k>0` pairs with target
/// `row_hiddens[(k-1)*dim..]` (`h_{k-1}`). Empty `row_hiddens` keeps rolling
/// MTP `h_nextn` after token 0 (legacy). Live process KV is rebuilt from
/// target row hiddens after verify.
fn mtp_process_assumed_span(
    model: &TinyModel,
    cache: &mut DenseKv,
    span: &[u32],
    start_pos: usize,
    pending_h: &[f32],
    row_hiddens: &[f32],
    draft_hip: Option<&HipWorkspace>,
    resident: Option<&ResidentModel>,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if span.is_empty() {
        return Ok(());
    }
    let kv_dim = model.config.n_kv_heads * model.config.head_dim();
    let dim = model.config.dim;
    if cache.seq_len() > start_pos {
        cache.truncate(start_pos, kv_dim);
    }
    if cache.seq_len() != start_pos {
        return Ok(());
    }
    for (i, &token) in span.iter().enumerate() {
        let h_prev = if i == 0 {
            pending_h
        } else {
            let off = (i - 1) * dim;
            if row_hiddens.len() >= off + dim {
                &row_hiddens[off..off + dim]
            } else {
                &[]
            }
        };
        if h_prev.len() == dim {
            cache.seed_h_nextn_from_target(h_prev);
        }
        model.process_dense_on(token, start_pos + i, cache, draft_hip, resident, metrics)?;
        cache.record(token);
    }
    Ok(())
}

/// Drafts after a deferred correction `tok`: `[tok, d1, ..]`, or empty when no
/// provider can continue past it. Any draft-ahead span predicted `tok`'s slot
/// from the pre-correction context and is dropped.
#[allow(clippy::too_many_arguments)]
fn propose_after_deferred(
    inner: &mut EngineInner,
    id: u64,
    ctx: &[u32],
    tok: u32,
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<Vec<u32>, AllpacaError> {
    inner.draft_ahead = None;
    if let Some(provider) = inner.dflash2.as_mut() {
        let mut after = ctx.to_vec();
        after.push(tok);
        match provider.propose(&after, depth, temperature) {
            Ok(more) if !more.is_empty() => {
                let mut draft = vec![tok];
                draft.extend(more);
                host_stage_draft_tokens(&draft[1..], draft_hip, metrics);
                return Ok(draft);
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, "DFlash2 propose failed"),
        }
    }
    if inner.flags.draft_model_speculation {
        if let Some(model) = inner.draft_model.clone() {
            let mtp = mtp_draft_active(inner);
            let seed = request(inner, id)?.last_hidden.clone();
            if mtp && seed.len() == model.config.dim {
                inner.draft_kv.seed_h_nextn_from_target(&seed);
            }
            let mut provider = DraftModelProvider::new((*model).clone());
            let resident = inner.draft_resident.clone();
            let draft = provider.propose_after(
                ctx,
                tok,
                depth + 1,
                temperature,
                &mut inner.draft_kv,
                draft_hip,
                resident.as_deref(),
                Some(metrics),
            )?;
            if mtp {
                // Draft AR writes are speculative; process() owns the prompt KV.
                let kv_dim = model.config.n_kv_heads * model.config.head_dim();
                inner.draft_kv.truncate(ctx.len(), kv_dim);
                if seed.len() == model.config.dim {
                    inner.draft_kv.seed_h_nextn_from_target(&seed);
                }
            }
            if draft.len() >= 2 {
                host_stage_draft_tokens(&draft[1..], draft_hip, metrics);
                return Ok(draft);
            }
        }
    }
    if inner.flags.ngram_speculation {
        let mut after = ctx.to_vec();
        after.push(tok);
        let more = inner.ngram.propose(&after, depth, temperature)?;
        if !more.is_empty() {
            let mut draft = vec![tok];
            draft.extend(more);
            return Ok(draft);
        }
    }
    Ok(Vec::new())
}

fn propose_drafts(
    inner: &mut EngineInner,
    id: u64,
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<Vec<u32>, AllpacaError> {
    if request(inner, id)?.suppress_draft {
        return Ok(Vec::new());
    }
    let ctx = request(inner, id)?.tokens.clone();
    if let Some(tok) = request(inner, id)?.deferred {
        return propose_after_deferred(inner, id, &ctx, tok, depth, temperature, draft_hip, metrics);
    }
    if let Some(ahead) = inner.draft_ahead.take() {
        if ahead.ctx == ctx && !ahead.tokens.is_empty() {
            metrics
                .draft_ahead_hits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            host_stage_draft_tokens(&ahead.tokens, draft_hip, metrics);
            return Ok(ahead.tokens);
        }
    }
    if let Some(provider) = inner.dflash2.as_mut() {
        match provider.propose(&ctx, depth, temperature) {
            Ok(draft) if !draft.is_empty() => {
                host_stage_draft_tokens(&draft, draft_hip, metrics);
                return Ok(draft);
            }
            Ok(_) => {}
            Err(e) => tracing::warn!(error = %e, "DFlash2 propose failed"),
        }
    }
    if let Some(provider) = inner.dspark.as_mut() {
        let draft = provider.propose(&ctx, depth, temperature)?;
        if !draft.is_empty() {
            host_stage_draft_tokens(&draft, draft_hip, metrics);
            return Ok(draft);
        }
    }
    if inner.flags.draft_model_speculation {
        if let Some(model) = inner.draft_model.clone() {
            let mtp = model.config.gguf_layer_base > 0
                || model.mtp_nextn.is_some()
                || inner
                    .draft_resident
                    .as_ref()
                    .is_some_and(|r| r.has_mtp_nextn());
            if mtp {
                let dim = model.config.dim;
                let seed = request(inner, id)
                    .map(|a| a.last_hidden.clone())
                    .unwrap_or_default();
                if seed.len() == dim {
                    inner.draft_kv.seed_h_nextn_from_target(&seed);
                }
            }
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
            if mtp {
                // Draft AR writes are speculative; process() owns the prompt KV.
                let kv_dim = model.config.n_kv_heads * model.config.head_dim();
                inner.draft_kv.truncate(ctx.len(), kv_dim);
                let seed = request(inner, id)
                    .map(|a| a.last_hidden.clone())
                    .unwrap_or_default();
                if seed.len() == model.config.dim {
                    inner.draft_kv.seed_h_nextn_from_target(&seed);
                }
            }
            if !draft.is_empty() {
                if std::env::var("ALLPACA_MTP_PROBE").ok().as_deref() == Some("1") {
                    eprintln!(
                        "MTP_PROBE ctx_len={} n_past={} drafts={:?} draft_kv_seq={}",
                        ctx.len(),
                        ctx.len(),
                        &draft[..draft.len().min(8)],
                        inner.draft_kv.seq_len()
                    );
                }
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

fn prefetch_draft_expert_union(
    inner: &EngineInner,
    span: &[u32],
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    hooks: &DecodePollHooks<'_>,
) -> Result<(), AllpacaError> {
    let target = inner
        .target
        .as_ref()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    let Some(resident) = inner.target_resident.as_ref() else {
        return Ok(());
    };
    if target.config.n_experts == 0 || span.len() < 2 {
        return Ok(());
    }
    let ids = target.routed_experts_for_verify_span(span);
    if ids.len() < 2 {
        return Ok(());
    }
    let Some(lru) = target.expert_lru.as_ref() else {
        return Ok(());
    };
    let plan = lru.plan(&ids);
    if plan.fill.is_empty() {
        return Ok(());
    }
    let mut pairs = Vec::new();
    for id in &plan.fill {
        let host = target.host_expert(id.layer as usize, id.expert as usize)?;
        pairs.push((*id, host));
    }
    let side = DeviceDecodeSide {
        layer_stagger: hooks.layer_stagger,
        ds_probe: Some(hooks.ds_probe),
        ds_queue: Some(hooks.ds_queue),
        ..DeviceDecodeSide::new(0)
    };
    let mut nvme = |id: ExpertId, dev_off: usize, slot_bytes: usize| {
        target.try_nvme_expert_fill(id, dev_off, slot_bytes, Some(&side), resident)
    };
    resident.bulk_fill_experts(&pairs, Some(metrics), None, false, Some(&mut nvme))?;
    for (id, host) in &pairs {
        lru.admit(*id, host)?;
    }
    metrics
        .draft_expert_prefetches
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _ = hip;
    Ok(())
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
    hooks: &DecodePollHooks<'_>,
) -> Result<Produced, AllpacaError> {
    let mut drafts = drafts;
    let drafts_len = drafts.len() as u64;
    if let Some(tok) = request_mut(inner, id)?.deferred.take() {
        // Drafts were proposed from the forwarded context, so drafts[0]
        // predicts this slot. On a match the correction heads the packed span
        // (it was already emitted, so it is not produced again).
        if drafts.len() >= 2 && drafts[0] == tok && remaining > 0 {
            metrics
                .deferred_corrections_packed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            prefetch_draft_expert_union(inner, &drafts, metrics, hip, hooks)?;
            let mut out = packed_linear_verify(
                inner,
                id,
                drafts,
                tok,
                mask,
                remaining + 1,
                metrics,
                hip,
                draft_hip,
                depth,
                temperature,
                hooks,
            )?;
            if let Some(tail) = inner
                .staged_decode
                .as_mut()
                .and_then(|s| s.iter_mut().find(|st| st.id == id))
                .and_then(|st| st.span.as_mut())
            {
                // Staged: the batched tick drops the correction when it finishes.
                tail.drop_first = true;
                return Ok(out);
            }
            // The correction was already emitted and is not a drafted token.
            out.tokens.remove(0);
            out.accepted = out.accepted.saturating_sub(1);
            out.proposed = out.proposed.saturating_sub(1);
            return Ok(out);
        }
        // This step samples from the correction's logits right away, so its
        // forward cannot wait for a batched decode tick.
        let staged = inner.staged_decode.take();
        let forwarded = append_and_forward(
            inner,
            id,
            tok,
            metrics,
            hip,
            draft_hip,
            micro_batch,
            hooks.layer_stagger,
            Some(hooks.ds_probe),
            Some(hooks.ds_queue),
            #[cfg(feature = "cuda")]
            hooks.cuda,
            #[cfg(feature = "vulkan")]
            hooks.vulkan,
        );
        inner.staged_decode = staged;
        forwarded?;
        // They were aligned one slot behind the context that now includes `tok`.
        drafts.clear();
    }
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
        if std::env::var("ALLPACA_DUMP_TOPK").ok().as_deref() == Some("1") {
            static DUMPED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                let logits = &active.pending_logits;
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| {
                    logits[b]
                        .partial_cmp(&logits[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let k = idx.len().min(10);
                let top: Vec<(usize, f32)> = idx[..k].iter().map(|&i| (i, logits[i])).collect();
                eprintln!("ALLPACA_DUMP_TOPK first-sample top10={top:?}");
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
    let first_match = drafts.first().copied() == Some(sampled);
    let stop = is_stop(sampled, &request(inner, id)?.params);
    if first_match && drafts.len() >= 2 && !stop && remaining > 1 {
        prefetch_draft_expert_union(inner, &drafts, metrics, hip, hooks)?;
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
            hooks,
        );
    }
    let mut produced = vec![sampled];
    let mut accepted = u64::from(first_match);
    let mut draft_iter = drafts.into_iter().peekable();
    if first_match {
        draft_iter.next();
    }
    append_and_forward(
        inner,
        id,
        sampled,
        metrics,
        hip,
        draft_hip,
        micro_batch,
        hooks.layer_stagger,
        Some(hooks.ds_probe),
        Some(hooks.ds_queue),
        #[cfg(feature = "cuda")]
        hooks.cuda,
        #[cfg(feature = "vulkan")]
        hooks.vulkan,
    )?;
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
        append_and_forward(
            inner,
            id,
            sampled,
            metrics,
            hip,
            draft_hip,
            micro_batch,
            hooks.layer_stagger,
            Some(hooks.ds_probe),
            Some(hooks.ds_queue),
            #[cfg(feature = "cuda")]
            hooks.cuda,
            #[cfg(feature = "vulkan")]
            hooks.vulkan,
        )?;
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
        store_draft_ahead_from_ctx(inner, id, &ctx, depth, temperature, draft_hip, metrics)?;
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
    hooks: &DecodePollHooks<'_>,
) -> Result<Produced, AllpacaError> {
    let span_len = span.len();
    let next_ctx = {
        let mut ctx = request(inner, id)?.tokens.clone();
        ctx.extend_from_slice(&span);
        ctx
    };
    let can_ahead = depth > 0 && inner.flags.draft_model_speculation && inner.draft_model.is_some();
    if !can_ahead {
        return packed_linear_verify_span(
            inner, id, span, first, mask, remaining, metrics, hip, draft_hip, hooks, None, None,
        );
    }
    if mtp_draft_active(inner) {
        return packed_mtp_verify_ahead(
            inner,
            id,
            span,
            first,
            mask,
            remaining,
            metrics,
            hip,
            draft_hip,
            depth,
            temperature,
            hooks,
            span_len,
            next_ctx,
        );
    }
    let mut draft_kv = std::mem::take(&mut inner.draft_kv);
    let model = inner
        .draft_model
        .clone()
        .expect("draft-ahead needs a Draft Model");
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
            inner, id, span, first, mask, remaining, metrics, hip, draft_hip, hooks, None, None,
        );
        let (kv, ahead) = worker
            .join()
            .unwrap_or_else(|p| std::panic::resume_unwind(p));
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
                id,
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

fn packed_mtp_verify_ahead(
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
    hooks: &DecodePollHooks<'_>,
    span_len: usize,
    next_ctx: Vec<u32>,
) -> Result<Produced, AllpacaError> {
    let pending_h = request(inner, id)?.last_hidden.clone();
    let start_pos = request(inner, id)?.tokens.len();
    let snap = match inner.draft_kv.snapshot() {
        Ok(s) => s,
        Err(_) => {
            let produced = packed_linear_verify_span(
                inner, id, span, first, mask, remaining, metrics, hip, draft_hip, hooks, None, None,
            )?;
            let actual_ctx = request(inner, id)?.tokens.clone();
            store_draft_ahead_from_ctx(
                inner,
                id,
                &actual_ctx,
                depth,
                temperature,
                draft_hip,
                metrics,
            )?;
            return Ok(produced);
        }
    };
    let model = inner
        .draft_model
        .clone()
        .expect("draft-ahead needs a Draft Model");
    let resident = inner.draft_resident.clone();
    let span_worker = span.clone();
    let next_ctx_worker = next_ctx.clone();
    let model_w = model.clone();
    let resident_w = resident.clone();
    let (row_tx, row_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let row_tx_err = row_tx.clone();
    std::thread::scope(|s| {
        let worker = s.spawn(move || {
            if let Some(h) = draft_hip {
                let _ = h.bind();
            }
            let rows = row_rx.recv().unwrap_or_default();
            if rows.is_empty() {
                return Ok(Vec::new());
            }
            let mut cache = snap;
            mtp_process_assumed_span(
                model_w.as_ref(),
                &mut cache,
                &span_worker,
                start_pos,
                &pending_h,
                &rows,
                draft_hip,
                resident_w.as_deref(),
                Some(metrics),
            )?;
            if !cache.seq_matches(&next_ctx_worker) {
                return Ok(Vec::new());
            }
            let mut provider = DraftModelProvider::new((*model_w).clone());
            provider.propose_on(
                &next_ctx_worker,
                depth,
                temperature,
                &mut cache,
                draft_hip,
                resident_w.as_deref(),
                Some(metrics),
            )
        });
        let pending = std::cell::RefCell::new(Some(worker));
        let joined = std::cell::RefCell::new(None);
        let join_ahead = || {
            if let Some(h) = pending.borrow_mut().take() {
                let r = h.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
                *joined.borrow_mut() = Some(r);
            }
        };
        let send_rows = |rows: &[f32]| {
            let _ = row_tx.send(rows.to_vec());
        };
        let produced = packed_linear_verify_span(
            inner,
            id,
            span,
            first,
            mask,
            remaining,
            metrics,
            hip,
            draft_hip,
            hooks,
            Some(&join_ahead),
            Some(&send_rows),
        );
        if produced.is_err() {
            let _ = row_tx_err.send(Vec::new());
        }
        join_ahead();
        let produced = produced?;
        let ahead = match joined.borrow_mut().take() {
            Some(Ok(tokens)) => tokens,
            _ => Vec::new(),
        };
        let actual_ctx = request(inner, id)?.tokens.clone();
        let full_accept = produced.accepted == produced.proposed
            && produced.proposed == span_len as u64
            && actual_ctx == next_ctx;
        if full_accept && !ahead.is_empty() {
            inner.draft_ahead = Some(DraftAhead {
                ctx: next_ctx,
                tokens: ahead,
            });
            let dim = model.config.dim;
            let seed = request(inner, id)
                .map(|a| a.last_hidden.clone())
                .unwrap_or_default();
            if seed.len() == dim {
                inner.draft_kv.seed_h_nextn_from_target(&seed);
            }
        } else {
            store_draft_ahead_from_ctx(
                inner,
                id,
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
    id: u64,
    ctx: &[u32],
    depth: usize,
    temperature: f32,
    draft_hip: Option<&HipWorkspace>,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    // A deferred correction occupies the next slot; a span drafted from `ctx`
    // would predict that slot and is never used (`propose_after_deferred`).
    if depth == 0 || ctx.is_empty() || request(inner, id)?.deferred.is_some() {
        return Ok(());
    }
    let Some(model) = inner.draft_model.clone() else {
        return Ok(());
    };
    let mtp = mtp_draft_active(inner);
    if mtp {
        let dim = model.config.dim;
        let seed = request(inner, id)
            .map(|a| a.last_hidden.clone())
            .unwrap_or_default();
        if seed.len() == dim {
            inner.draft_kv.seed_h_nextn_from_target(&seed);
        }
    }
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
    if mtp {
        let kv_dim = model.config.n_kv_heads * model.config.head_dim();
        inner.draft_kv.truncate(ctx.len(), kv_dim);
        let seed = request(inner, id)
            .map(|a| a.last_hidden.clone())
            .unwrap_or_default();
        if seed.len() == model.config.dim {
            inner.draft_kv.seed_h_nextn_from_target(&seed);
        }
    }
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
    draft_hip: Option<&HipWorkspace>,
    hooks: &DecodePollHooks<'_>,
    before_draft_gpu: Option<&dyn Fn()>,
    after_row_hiddens: Option<&dyn Fn(&[f32])>,
) -> Result<Produced, AllpacaError> {
    prefetch_draft_expert_union(inner, &span, metrics, hip, hooks)?;
    let drafts_len = span.len() as u64;
    let vocab = request(inner, id)?.pending_logits.len();
    let start_pos = request(inner, id)?.tokens.len();
    let vision = request(inner, id)?.vision.clone();
    let vision_ref = (!vision.is_empty()).then_some(&*vision);
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
    let mut row_hiddens = Vec::new();
    let pages = request(inner, id)?.pages.clone();
    let resident = inner.target_resident.clone();
    let mut recurrent = std::mem::take(&mut request_mut(inner, id)?.recurrent);
    let mut device_gdn_cloned = 0usize;
    if let (Some(hip), Some(state)) = (hip, recurrent.as_mut()) {
        for_each_gdn_workspace(hip, Some(&target), |ws| {
            device_gdn_cloned += ws.gdn_clone_request(id)?;
            Ok(())
        })?;
        if device_gdn_cloned == 0 {
            for_each_gdn_workspace(hip, Some(&target), |ws| {
                ws.gdn_sync_request_to_host(id, &mut state.gdn_state)?;
                ws.gdn_sync_conv_to_host(id, &mut state.gdn_conv)?;
                ws.mamba2_sync_request_to_host(id, &mut state.ssm_h, &mut state.ssm_conv)
            })?;
        }
    }
    let rec_before = recurrent.clone();
    let pending_h = request(inner, id)?.last_hidden.clone();
    // One packed forward over the span when a partial accept can be rewound
    // from the device GDN journal. Otherwise (host GDN state, Mamba-2 slots)
    // forward one token at a time and stop at the reject, so no rewind is needed.
    let journal_rewind = device_gdn_cloned > 0
        && hip.is_some_and(|h| gdn_all_can_replay(h, &target, id, 0));
    let incremental_gdn = recurrent.is_some() && !journal_rewind;
    let mut produced = vec![first];
    let mut accepted = 1u64;
    let mut keep = 1usize;
    if incremental_gdn {
        // Device clone is the pre-span GDN state. Forward only until reject so
        // rewind is enough (state already at keep); do not replay the prefix.
        let params_stop = request(inner, id)?.params.clone();
        let mut one_logits = vec![0.0f32; vocab];
        let mut one_hidden = Vec::new();
        target.forward_tokens(
            &[span[0]],
            start_pos,
            &pages,
            &mut inner.kv,
            metrics,
            &mut last_logits,
            hip,
            resident.as_deref(),
            true,
            Some(&mut one_logits),
            recurrent.as_mut(),
            Some(&mut one_hidden),
            id,
            vision_ref,
        )?;
        if let Some(hip) = hip {
            hip.wait_logits_d2h(&mut one_logits)?;
        }
        row_logits[..vocab].copy_from_slice(&one_logits);
        row_hiddens.extend_from_slice(&one_hidden);
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
            if next != span[i + 1] {
                break;
            }
            accepted += 1;
            keep = i + 2;
            one_logits.fill(0.0);
            one_hidden.clear();
            target.forward_tokens(
                &[span[i + 1]],
                start_pos + i + 1,
                &pages,
                &mut inner.kv,
                metrics,
                &mut last_logits,
                hip,
                resident.as_deref(),
                true,
                Some(&mut one_logits),
                recurrent.as_mut(),
                Some(&mut one_hidden),
                id,
                vision_ref,
            )?;
            if let Some(hip) = hip {
                hip.wait_logits_d2h(&mut one_logits)?;
            }
            let dest = (i + 1) * vocab;
            row_logits[dest..dest + vocab].copy_from_slice(&one_logits);
            row_hiddens.extend_from_slice(&one_hidden);
        }
    } else {
        if stage_fits(inner, span.len())
            && before_draft_gpu.is_none()
            && after_row_hiddens.is_none()
            && vision_ref.is_none()
            && mask.is_none()
        {
            // Batched decode tick: these rows go out with the other Requests'
            // in one forward, then the tick runs `finish_linear_span`.
            request_mut(inner, id)?.recurrent = recurrent;
            let tail = SpanTail {
                span: span.clone(),
                first,
                remaining,
                start_pos,
                page_tokens,
                last_logits,
                rec_before,
                device_gdn_cloned,
                journal_rewind,
                pending_h,
                drop_first: false,
            };
            if let Some(staged) = inner.staged_decode.as_mut() {
                staged.push(StagedDecode {
                    id,
                    tokens: span,
                    pos: start_pos,
                    span: Some(Box::new(tail)),
                });
            }
            return Ok(Produced {
                tokens: Vec::new(),
                accepted: 0,
                proposed: drafts_len,
            });
        }
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
            Some(&mut row_hiddens),
            id,
            vision_ref,
        )?;
        if let Some(hip) = hip {
            hip.bind()?;
            hip.wait_logits_d2h(&mut row_logits)?;
        }
        (produced, accepted, keep) =
            accept_span(inner, id, &span, first, &mut row_logits, mask, remaining)?;
    }
    if let Some(send) = after_row_hiddens {
        // Draft-ahead assumes the whole span lands; after a reject its work
        // would be discarded, so hand it nothing and let it exit.
        send(if keep == span.len() { &row_hiddens } else { &[] });
    }
    finish_linear_span(
        inner,
        id,
        SpanTail {
            span,
            first,
            remaining,
            start_pos,
            page_tokens,
            last_logits,
            rec_before,
            device_gdn_cloned,
            journal_rewind,
            pending_h,
            drop_first: false,
        },
        (produced, accepted, keep),
        row_logits,
        row_hiddens,
        recurrent,
        incremental_gdn,
        metrics,
        hip,
        draft_hip,
        hooks,
        before_draft_gpu,
    )
}

/// Accept walk over a forwarded span's row logits: sample each row, stop at
/// the first token that differs from the draft. Returns `(produced, accepted, keep)`.
fn accept_span(
    inner: &mut EngineInner,
    id: u64,
    span: &[u32],
    first: u32,
    row_logits: &mut [f32],
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
) -> Result<(Vec<u32>, u64, usize), AllpacaError> {
    let vocab = row_logits.len() / span.len().max(1);
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
    Ok((produced, accepted, keep))
}

/// After a verify span's forward and accept walk: rewind KV and recurrent
/// state to the accepted prefix, extend the context, and forward (or defer)
/// the correction.
#[allow(clippy::too_many_arguments)]
fn finish_linear_span(
    inner: &mut EngineInner,
    id: u64,
    tail: SpanTail,
    (produced, accepted, keep): (Vec<u32>, u64, usize),
    row_logits: Vec<f32>,
    row_hiddens: Vec<f32>,
    mut recurrent: Option<RecurrentState>,
    incremental_gdn: bool,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    draft_hip: Option<&HipWorkspace>,
    hooks: &DecodePollHooks<'_>,
    before_draft_gpu: Option<&dyn Fn()>,
) -> Result<Produced, AllpacaError> {
    let SpanTail {
        span,
        remaining,
        start_pos,
        page_tokens,
        last_logits,
        rec_before,
        device_gdn_cloned,
        journal_rewind,
        pending_h,
        drop_first,
        ..
    } = tail;
    let drafts_len = span.len() as u64;
    let vocab = last_logits.len();
    let target = inner
        .target
        .clone()
        .ok_or(AllpacaError::Fatal(FatalError::NoModel))?;
    let resident = inner.target_resident.clone();
    let vision = request(inner, id)?.vision.clone();
    let vision_ref = (!vision.is_empty()).then_some(&*vision);
    let dim = target.config.dim;
    request_mut(inner, id)?.pending_logits = last_logits;
    metrics
        .linear_packed_verifies
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let pages = request(inner, id)?.pages.clone();
    inner
        .kv
        .rewind_filled(&pages, start_pos + keep, page_tokens as usize);
    if journal_rewind && keep < span.len() {
        let hip = hip.expect("journal rewind needs HIP");
        if gdn_all_can_replay(hip, &target, id, keep) {
            for_each_gdn_workspace(hip, Some(&target), |ws| ws.gdn_rewind_request_to(id, keep))?;
        } else {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "GDN journal lost rows mid-verify".into(),
            )));
        }
    } else if !incremental_gdn && keep < span.len() {
        if let Some(saved) = rec_before {
            recurrent = Some(saved);
            if let Some(hip) = hip {
                for_each_gdn_workspace(hip, Some(&target), |ws| {
                    if device_gdn_cloned > 0 {
                        ws.gdn_rewind_request(id)
                    } else {
                        ws.gdn_invalidate_request(id);
                        Ok(())
                    }
                })?;
            }
            let mut replay_logits = vec![0.0f32; vocab];
            target.forward_tokens(
                &span[..keep],
                start_pos,
                &pages,
                &mut inner.kv,
                metrics,
                &mut replay_logits,
                hip,
                resident.as_deref(),
                false,
                None,
                recurrent.as_mut(),
                None,
                id,
                vision_ref,
            )?;
            if let Some(hip) = hip {
                let _ = hip.wait_logits_d2h(&mut replay_logits);
            }
        }
    }
    if let Some(hip) = hip {
        let _ = for_each_gdn_workspace(hip, Some(&target), |ws| {
            ws.gdn_drop_clone(id);
            Ok(())
        });
    }
    request_mut(inner, id)?.recurrent = recurrent;
    if let Some(join) = before_draft_gpu {
        join();
    }
    // Rebuild MTP draft KV for accepted prefix with target hiddens (llama process).
    if mtp_draft_active(inner) && dim > 0 && row_hiddens.len() >= keep * dim {
        let kv_dim = inner
            .draft_model
            .as_ref()
            .map(|m| m.config.n_kv_heads * m.config.head_dim())
            .unwrap_or(0);
        if kv_dim > 0 {
            inner.draft_kv.truncate(start_pos, kv_dim);
        }
        for i in 0..keep {
            let h_prev = if i == 0 {
                pending_h.as_slice()
            } else {
                let off = (i - 1) * dim;
                &row_hiddens[off..off + dim]
            };
            mtp_process_token(inner, span[i], start_pos + i, h_prev, draft_hip, metrics)?;
        }
    }
    {
        let active = request_mut(inner, id)?;
        active.tokens.extend_from_slice(&span[..keep]);
        active.pages = pages;
        // MTP handoff: last accepted row's output_norm hidden (not the rejected suffix).
        if dim > 0 && row_hiddens.len() >= keep * dim {
            let start = (keep - 1) * dim;
            active.last_hidden = row_hiddens[start..start + dim].to_vec();
        }
    }
    if inner.flags.ngram_speculation {
        let ctx = request(inner, id)?.tokens.clone();
        inner.ngram.observe(&ctx);
    }

    let correction = produced.get(keep).copied();
    let params_end = request(inner, id)?.params.clone();
    let defer = correction.is_some_and(|tok| {
        !is_stop(tok, &params_end)
            && produced.len() < remaining
            && !inner.flags.tree_speculation
            && (inner.flags.draft_model_speculation
                || inner.flags.ngram_speculation
                || inner.dflash2.is_some())
    });
    if defer {
        // Forwarded next step at the head of the drafted span (or alone).
        request_mut(inner, id)?.deferred = correction;
    } else if produced.len() > keep {
        if let Some(join) = before_draft_gpu {
            join();
        }
        append_and_forward(
            inner,
            id,
            produced[keep],
            metrics,
            hip,
            draft_hip,
            None,
            hooks.layer_stagger,
            Some(hooks.ds_probe),
            Some(hooks.ds_queue),
            #[cfg(feature = "cuda")]
            hooks.cuda,
            #[cfg(feature = "vulkan")]
            hooks.vulkan,
        )?;
    } else {
        let off = keep.saturating_sub(1) * vocab;
        let pending = row_logits[off..off + vocab].to_vec();
        request_mut(inner, id)?.pending_logits = pending;
    }
    let mut out = Produced {
        tokens: produced,
        accepted,
        proposed: drafts_len,
    };
    if drop_first {
        // The correction was already emitted and is not a drafted token.
        out.tokens.remove(0);
        out.accepted = out.accepted.saturating_sub(1);
        out.proposed = out.proposed.saturating_sub(1);
    }
    Ok(out)
}

fn verify_forest(
    inner: &mut EngineInner,
    id: u64,
    forest: &TokenForest,
    mask: Option<&allpaca_types::TokenMask>,
    remaining: usize,
    metrics: &Metrics,
    hip: Option<&HipWorkspace>,
    draft_hip: Option<&HipWorkspace>,
    depth: usize,
    temperature: f32,
    micro_batch: Option<&MicroBatchExpertUnion>,
    hooks: &DecodePollHooks<'_>,
) -> Result<Produced, AllpacaError> {
    let vocab = inner
        .tokenizer
        .as_ref()
        .map(|t| t.vocab_size())
        .unwrap_or(1);
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
        let child = forest
            .children(parent)
            .find(|&i| forest.nodes[i].token_id as u32 == sampled);
        if let Some(idx) = child {
            accepted_tokens.push(sampled);
            parent = idx as i32;
        }
        append_and_forward(
            inner,
            id,
            sampled,
            metrics,
            hip,
            draft_hip,
            micro_batch,
            hooks.layer_stagger,
            Some(hooks.ds_probe),
            Some(hooks.ds_queue),
            #[cfg(feature = "cuda")]
            hooks.cuda,
            #[cfg(feature = "vulkan")]
            hooks.vulkan,
        )?;
        let stop = is_stop(sampled, &request(inner, id)?.params);
        if stop || child.is_none() || forest.children(parent).next().is_none() {
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
    if !produced.is_empty() {
        let ctx = request(inner, id)?.tokens.clone();
        store_draft_ahead_from_ctx(inner, id, &ctx, depth, temperature, draft_hip, metrics)?;
    }
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
    draft_hip: Option<&HipWorkspace>,
    micro_batch: Option<&MicroBatchExpertUnion>,
    layer_stagger: Option<&LayerStagger>,
    ds_probe: Option<&DirectStorageProbe>,
    ds_queue: Option<&DirectStorageQueue>,
    #[cfg(feature = "cuda")] cuda: Option<&allpaca_cuda::CudaWorkspace>,
    #[cfg(feature = "vulkan")] vulkan: Option<&allpaca_vulkan::VulkanWorkspace>,
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
    // Batched decode tick with room left: the tick forwards this row.
    if stage_fits(inner, 1) {
        if let Some(staged) = inner.staged_decode.as_mut() {
            staged.push(StagedDecode {
                id,
                tokens: vec![token],
                pos,
                span: None,
            });
            return Ok(());
        }
    }
    let mut logits = std::mem::take(&mut request_mut(inner, id)?.pending_logits);
    let pages = request(inner, id)?.pages.clone();
    let resident = inner.target_resident.clone();
    #[cfg(feature = "cuda")]
    let cuda_resident = inner.target_cuda_resident.clone();
    #[cfg(feature = "vulkan")]
    let vulkan_resident = inner.target_vulkan_resident.clone();
    let mut recurrent = std::mem::take(&mut request_mut(inner, id)?.recurrent);
    let mut moe_norm = std::mem::take(&mut request_mut(inner, id)?.moe_norm);
    let mut last_hidden = std::mem::take(&mut request_mut(inner, id)?.last_hidden);
    let defer_lm_head = request(inner, id)?.defer_lm_head;
    mtp_process_token(inner, token, pos, &last_hidden, draft_hip, metrics)?;
    let want_hidden = mtp_draft_active(inner) || defer_lm_head;
    let vision = request(inner, id)?.vision.clone();
    let mut side = DeviceDecodeSide {
        moe_norm: Some(&mut moe_norm),
        layer_stagger,
        ds_probe,
        ds_queue,
        last_hidden: if want_hidden {
            Some(&mut last_hidden)
        } else {
            None
        },
        write_logits: !defer_lm_head,
        mrope: vision.mrope(),
        ..DeviceDecodeSide::new(id)
    };
    let used_cuda = {
        #[cfg(feature = "cuda")]
        {
            let mut used = false;
            if hip.is_none() {
                if let (Some(cuda), Some(cres)) = (cuda, cuda_resident.as_deref()) {
                    match target.forward_token_cuda(
                        token,
                        pos,
                        &pages,
                        &mut inner.kv,
                        metrics,
                        &mut logits,
                        cuda,
                        cres,
                        None,
                        recurrent.as_mut(),
                    ) {
                        Ok(()) => used = true,
                        Err(e) => {
                            tracing::warn!(
                                pos,
                                token,
                                error = %e,
                                "CUDA on-device decode failed; host fallback"
                            );
                        }
                    }
                }
            }
            used
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    };
    let used_vulkan = {
        #[cfg(feature = "vulkan")]
        {
            let mut used = false;
            if hip.is_none() && !used_cuda {
                if let (Some(vk), Some(vres)) = (vulkan, vulkan_resident.as_deref()) {
                    match target.forward_token_vulkan(
                        token,
                        pos,
                        &pages,
                        &mut inner.kv,
                        metrics,
                        &mut logits,
                        vk,
                        vres,
                        None,
                    ) {
                        Ok(()) => used = true,
                        Err(e) => {
                            tracing::warn!(
                                pos,
                                token,
                                error = %e,
                                "Vulkan on-device decode failed; host fallback"
                            );
                        }
                    }
                }
            }
            used
        }
        #[cfg(not(feature = "vulkan"))]
        {
            false
        }
    };
    if !used_cuda && !used_vulkan {
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
            Some(&mut side),
            None,
        )?;
    }
    request_mut(inner, id)?.pending_logits = logits;
    if hip.is_some() {
        inner.logits_owner = Some(id);
    }
    request_mut(inner, id)?.recurrent = recurrent;
    request_mut(inner, id)?.moe_norm = moe_norm;
    request_mut(inner, id)?.last_hidden = last_hidden;
    if let Some(active) = inner.active.get_mut(&id) {
        active.decode_step = active.decode_step.saturating_add(1);
        active.last_hidden_fresh = want_hidden;
        if defer_lm_head {
            metrics
                .batch_gemm_lm_head_skipped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    Ok(())
}

/// Replaces the i-th `pad` token in `tokens` with `grids[i]` = `(cols, rows)`
/// image tokens and returns the spans they occupy. Images without a pad
/// (none left, or no pad token in the vocab) go first, before the prompt.
fn expand_image_pads(
    tokens: &[u32],
    pad: Option<u32>,
    grids: &[(usize, usize)],
) -> (Vec<u32>, Vec<crate::model::ImageSpan>) {
    let fill = pad.unwrap_or(0);
    let n_pads = pad.map_or(0, |p| tokens.iter().filter(|&&t| t == p).count());
    let (front, inline) = grids.split_at(grids.len().saturating_sub(n_pads));
    let mut out = Vec::with_capacity(tokens.len() + grids.iter().map(|(c, r)| c * r).sum::<usize>());
    let mut spans = Vec::with_capacity(grids.len());
    let mut push_image = |out: &mut Vec<u32>, &(cols, rows): &(usize, usize)| {
        spans.push(crate::model::ImageSpan { start: out.len(), cols, rows });
        out.extend(std::iter::repeat_n(fill, cols * rows));
    };
    for g in front {
        push_image(&mut out, g);
    }
    let mut inline = inline.iter();
    for &t in tokens {
        match (Some(t) == pad).then(|| inline.next()).flatten() {
            Some(g) => push_image(&mut out, g),
            None => out.push(t),
        }
    }
    (out, spans)
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

/// Smallest remote KV capacity the auto setting picks.
const RPC_AUTO_MIN_SEQ: usize = 2048;

/// Remote layers' KV capacity in tokens: `configured`, or when that is 0 the
/// context `kv_vram_bytes` holds across this model's attention layers (the
/// same sizing the local KV uses), at least [`RPC_AUTO_MIN_SEQ`].
fn rpc_kv_max_seq(configured: usize, kv_vram_bytes: u64, model: &TinyModel) -> usize {
    if configured > 0 {
        return configured;
    }
    let cfg = &model.config;
    let per_layer = (cfg.n_kv_heads.max(1) * cfg.head_dim().max(1) * 2 * 4) as u64;
    let attn_layers = model
        .layers
        .iter()
        // By layer type: large GGUFs keep no host weights to look at.
        .filter(|l| {
            l.gdn.is_none()
                && l.ssm.is_none()
                && l.block == allpaca_gguf::HybridLayerKind::Attention
        })
        .count()
        .max(1) as u64;
    let tokens = (kv_vram_bytes / (per_layer * attn_layers).max(1)) as usize;
    tokens.max(RPC_AUTO_MIN_SEQ)
}

/// One device's share of the loaded model ([`Engine::placement`]).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DevicePlacement {
    /// `hip:0`, `cuda:0`, `cpu`, or `rpc:<host:port>/<gpu>`.
    pub device: String,
    /// Card name when known.
    pub name: String,
    /// Inclusive layer ranges whose bodies run here.
    pub layers: Vec<[usize; 2]>,
    pub n_layers: usize,
    /// Layers whose routed experts run here (expert-parallel MoE).
    pub expert_layers: usize,
    /// Weight bytes uploaded here, when known.
    pub weight_bytes: Option<u64>,
    /// Other parts held here, e.g. "output head", "MTP draft".
    pub roles: Vec<String>,
}

/// One layer's routed experts ([`Engine::expert_placement`]).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExpertLayerHome {
    /// `hip:<n>` (beside the layer body) or `rpc:<host:port>/<gpu>`.
    pub device: String,
    pub bytes: u64,
}

/// Expert-parallel layout: every layer's experts move as one block.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExpertPlacement {
    /// In layer order.
    pub layers: Vec<ExpertLayerHome>,
    /// Worker GPUs a layer's experts may be pinned to.
    pub workers: Vec<String>,
}

/// Per-device rows in first-use order, found by device key.
#[derive(Default)]
struct PlacementRows {
    rows: Vec<DevicePlacement>,
}

impl PlacementRows {
    fn entry(&mut self, key: &str, name: &str) -> &mut DevicePlacement {
        let i = match self.rows.iter().position(|r| r.device == key) {
            Some(i) => i,
            None => {
                self.rows.push(DevicePlacement {
                    device: key.to_string(),
                    name: name.to_string(),
                    layers: Vec::new(),
                    n_layers: 0,
                    expert_layers: 0,
                    weight_bytes: None,
                    roles: Vec::new(),
                });
                self.rows.len() - 1
            }
        };
        &mut self.rows[i]
    }
}

/// What else the engine holds, for [`placement_rows`]: target bytes on the
/// primary device, the draft model's (device, bytes, role), DFlash2's device.
struct PlacementExtras {
    target_bytes: Option<u64>,
    draft: Option<(Option<i32>, u64, &'static str)>,
    dflash2_dev: Option<i32>,
}

/// Group layers by device (see [`Engine::placement`]). `(kind, dev)` is the
/// backend device that runs local layers when there is no layer split.
fn placement_rows(
    n_layers: usize,
    (kind, primary_dev): (&str, i32),
    engine: &Engine,
    rpc: Option<&crate::rpc_runtime::RpcRuntime>,
    extras: Option<PlacementExtras>,
) -> Vec<DevicePlacement> {
    use crate::rpc_runtime::LayerHome;
    let key_local = |dev: i32| {
        if kind == "cpu" {
            "cpu".to_string()
        } else {
            format!("{kind}:{dev}")
        }
    };
    let local_name = |dev: i32| {
        engine
            .inventory
            .devices
            .iter()
            .find(|d| d.is_gpu() && !d.is_rpc() && d.id == dev)
            .map(|d| d.name.clone())
            .unwrap_or_default()
    };
    let remote_names = rpc.map(|r| r.remote_gpu_stats()).unwrap_or_default();
    let remote_name = |endpoint: &str, gpu: u32| {
        remote_names
            .iter()
            .find(|g| g.endpoint == endpoint && g.index == gpu)
            .map(|g| g.name.clone())
            .unwrap_or_default()
    };
    let mut out = PlacementRows::default();
    for l in 0..n_layers {
        let (key, name) = match rpc.map(|r| r.layer_home(l)) {
            Some(LayerHome::Remote { endpoint, device }) => (
                format!("rpc:{endpoint}/{device}"),
                remote_name(&endpoint, device),
            ),
            Some(LayerHome::Local(dev)) if dev >= 0 && kind != "cpu" => {
                (key_local(dev), local_name(dev))
            }
            _ => (key_local(primary_dev), local_name(primary_dev)),
        };
        let row = out.entry(&key, &name);
        row.n_layers += 1;
        match row.layers.last_mut() {
            Some(last) if last[1] + 1 == l => last[1] = l,
            _ => row.layers.push([l, l]),
        }
    }
    if let Some(rpc) = rpc {
        for l in 0..n_layers {
            if let Some((endpoint, device)) = rpc.experts_home(l) {
                let name = remote_name(&endpoint, device);
                out.entry(&format!("rpc:{endpoint}/{device}"), &name).expert_layers += 1;
            }
        }
    }
    if let Some(x) = extras {
        let row = out.entry(&key_local(primary_dev), &local_name(primary_dev));
        row.weight_bytes = x.target_bytes;
        row.roles.push("output head".into());
        if let Some((dev, _, peer)) = rpc.and_then(|r| r.local_peer()) {
            out.entry(&key_local(dev), &local_name(dev)).weight_bytes = Some(peer.bytes());
        }
        if let Some((dev, bytes, role)) = x.draft {
            let dev = dev.unwrap_or(primary_dev);
            out.entry(&key_local(dev), &local_name(dev))
                .roles
                .push(format!("{role} ({} MiB)", bytes >> 20));
        }
        if let Some(dev) = x.dflash2_dev {
            out.entry(&key_local(dev), &local_name(dev)).roles.push("DFlash2 draft".into());
        }
    }
    out.rows
}

/// What a context length costs in KV for the loaded model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct KvEstimate {
    /// Bytes per token over the full-attention layers.
    pub full_bytes_per_token: u64,
    /// Bytes per token over sliding-window layers, which stop growing at
    /// `swa_window` tokens.
    pub swa_bytes_per_token: u64,
    pub swa_window: u64,
    pub attention_layers: usize,
    /// The engine's current `kv.vram_bytes`.
    pub budget_bytes: u64,
    pub page_tokens: u32,
}

impl KvEstimate {
    /// KV bytes for `tokens` of context.
    pub fn bytes_for(&self, tokens: u64) -> u64 {
        let swa = if self.swa_window > 0 { tokens.min(self.swa_window) } else { tokens };
        tokens * self.full_bytes_per_token + swa * self.swa_bytes_per_token
    }

    /// Context tokens that fit in `bytes`.
    pub fn tokens_for(&self, bytes: u64) -> u64 {
        let (mut lo, mut hi) = (0u64, 1u64 << 24);
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if self.bytes_for(mid) <= bytes {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
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
    fn decision_scores_read_the_prompt_and_generate_nothing() {
        use crate::decision::{DecisionModel, DecisionRead};

        let prompt = "alpha beta alpha gamma alpha delta";
        let engine = test_engine();
        let (tokens, dim) = {
            let inner = engine.lock().unwrap();
            (
                inner.tokenizer.as_ref().unwrap().encode(prompt).unwrap(),
                inner.target.as_ref().unwrap().config.dim,
            )
        };
        let marker = tokens[0];
        let n_markers = tokens[..tokens.len() - 1].iter().filter(|t| **t == marker).count();
        assert!(n_markers >= 1 && tokens.len() >= 3, "{tokens:?}");

        // One prompt token per prefill chunk must read the same values.
        let mut cfg = test_config();
        cfg.prefill.chunk_tokens = 1;
        let chunked = Engine::new(cfg).unwrap();
        for e in [&engine, &chunked] {
            e.lock().unwrap().decision = Some(Arc::new(DecisionModel::test_pointer(marker, dim)));
        }

        // Label logits after the last prompt token.
        let labels = DecisionRead::Labels(vec![3, 5]);
        let (scores, n) = engine.decision_scores(prompt, &labels).unwrap();
        assert_eq!(n, tokens.len());
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|s| s.is_finite()) && scores[0] != scores[1], "{scores:?}");
        assert_eq!(engine.decision_scores(prompt, &labels).unwrap().0, scores);
        assert_eq!(chunked.decision_scores(prompt, &labels).unwrap().0, scores);

        // One score per marker token, from the hidden states.
        let pointer = DecisionRead::Pointer { marker };
        let (scores, n) = engine.decision_scores(prompt, &pointer).unwrap();
        assert_eq!(n, tokens.len());
        assert_eq!(scores.len(), n_markers);
        assert!(scores.iter().all(|s| s.is_finite()), "{scores:?}");
        assert_eq!(engine.decision_scores(prompt, &pointer).unwrap().0, scores);
        assert_eq!(chunked.decision_scores(prompt, &pointer).unwrap().0, scores);

        // Nothing was sampled and no request is left behind.
        assert!(engine.lock().unwrap().active.is_empty());
        assert_eq!(engine.stats().tokens_generated, 0);
    }

    #[test]
    fn placement_lists_every_layer_once_by_device() {
        let engine = test_engine();
        let n_layers = engine.lock().unwrap().target.as_ref().map(|t| t.config.n_layers).unwrap();
        let rows = engine.placement();
        assert_eq!(rows.len(), 1, "a CPU-only engine has one device: {rows:?}");
        assert_eq!(rows[0].device, "cpu");
        assert_eq!(rows[0].n_layers, n_layers);
        assert_eq!(rows[0].layers, vec![[0, n_layers - 1]]);
        assert!(rows[0].roles.iter().any(|r| r == "output head"));
    }

    fn spawn_hello_stub(reply: allpaca_rpc::HelloReply) -> (String, std::thread::JoinHandle<()>) {
        use std::io::Read;
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{encode_hello_reply, write_reply, STATUS_OK};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut head = [0u8; 9];
            stream.read_exact(&mut head).unwrap();
            let size = u64::from_le_bytes(head[1..9].try_into().unwrap()) as usize;
            if size > 0 {
                let mut payload = vec![0u8; size];
                stream.read_exact(&mut payload).unwrap();
            }
            write_reply(&mut stream, STATUS_OK, &encode_hello_reply(&reply)).unwrap();
        });
        (addr, worker)
    }

    fn hello_stub_reply(major: u16, minor: u16, kernel_abi_rev: u32) -> allpaca_rpc::HelloReply {
        allpaca_rpc::HelloReply {
            major,
            minor,
            patch: 0,
            kernel_abi_rev,
            backend: allpaca_rpc::RpcBackendKind::Cpu,
            dtype_mask: 1,
            device_name: "mismatch-stub".into(),
            memory_bytes: 64 << 20,
            devices: Vec::new(),
        }
    }

    #[test]
    fn rpc_loopback_synthetic_generate() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![addr.to_string()];
        cfg.rpc.max_seq = 64;
        cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let engine = Engine::new(cfg).unwrap();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.rpc_roundtrips > 0,
            "expected remote layer launches, got {}",
            stats.rpc_roundtrips
        );
        drop(engine);
        worker.join().unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn rpc_loopback_cuda_worker_synthetic_generate() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CudaWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let backend = WorkerBackend::Cuda(CudaWorker::new(0).expect("cuda worker"));
            serve_connection(stream, backend).unwrap();
        });

        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![addr.to_string()];
        cfg.rpc.max_seq = 64;
        cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let engine = Engine::new(cfg).unwrap();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.rpc_roundtrips > 0,
            "expected CUDA RPC layer launches, got {}",
            stats.rpc_roundtrips
        );
        drop(engine);
        worker.join().unwrap();
    }

    #[test]
    fn rpc_layer_split_greedy_matches_cpu() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(2),
            ..GenerationParams::default()
        };
        let mut cpu_cfg = test_config();
        cpu_cfg.speculative.enabled = false;
        let cpu = Engine::new(cpu_cfg).unwrap();
        let cpu_ids = cpu.generate("hello", params.clone()).unwrap().token_ids;

        let mut rpc_cfg = test_config();
        rpc_cfg.speculative.enabled = false;
        rpc_cfg.rpc.endpoints = vec![addr.to_string()];
        rpc_cfg.rpc.max_seq = 64;
        rpc_cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let rpc = Engine::new(rpc_cfg).unwrap();
        let rpc_out = rpc.generate("hello", params).unwrap();
        let trips = rpc.stats().rpc_roundtrips;
        drop(rpc);
        worker.join().unwrap();

        assert!(
            trips > 0,
            "parity test must exercise remote layers, rpc_roundtrips={trips}"
        );
        assert_eq!(rpc_out.token_ids, cpu_ids);
    }

    /// Two-layer Llama GGUF with Q8_0 weights and varied values, written to
    /// the temp dir as `name`.
    fn tiny_llama_q8_gguf(name: &str) -> std::path::PathBuf {
        tiny_dense_q8_gguf(name, "llama")
    }

    /// [`tiny_llama_q8_gguf`] under another dense architecture name (its RoPE
    /// style follows the architecture).
    fn tiny_dense_q8_gguf(name: &str, arch: &str) -> std::path::PathBuf {
        use std::collections::HashMap;

        use allpaca_gguf::{
            quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor,
            GgufWriteTensor,
        };

        let dim = 32usize;
        let vocab = 8usize;
        let ff = 64usize;
        let n_kv = 16usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String(arch.into()),
        );
        meta.insert(format!("{arch}.vocab_size"), GgufValue::Uint32(vocab as u32));
        meta.insert(
            format!("{arch}.embedding_length"),
            GgufValue::Uint32(dim as u32),
        );
        meta.insert(format!("{arch}.block_count"), GgufValue::Uint32(2));
        meta.insert(
            format!("{arch}.attention.head_count"),
            GgufValue::Uint32(4),
        );
        meta.insert(
            format!("{arch}.attention.head_count_kv"),
            GgufValue::Uint32(2),
        );
        meta.insert(
            format!("{arch}.feed_forward_length"),
            GgufValue::Uint32(ff as u32),
        );
        // As real Llama GGUFs: dimension_count == head_dim, which is still
        // the standard (not NeoX) rotation. Remote layers once took this as
        // a NeoX partial rotation and diverged as positions grew.
        meta.insert(
            format!("{arch}.rope.dimension_count"),
            GgufValue::Uint32((dim / 4) as u32),
        );
        meta.insert(
            "tokenizer.ggml.model".into(),
            GgufValue::String("wordlevel".into()),
        );
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                ["<unk>", "<s>", "</s>", "hello", "a", "b", "c", "d"]
                    .into_iter()
                    .map(|s| GgufValue::String(s.into()))
                    .collect(),
            ),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

        // Varied weights: with constant ones every head and position looks
        // alike and a wrong rotation cannot show.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * scale
                })
                .collect()
        };
        let mut quant = Vec::new();
        let mut f32s: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        let embd = rnd(vocab * dim, 2.0);
        quant.push((
            "token_embd.weight".to_string(),
            vec![vocab as u64, dim as u64],
            quantize_q8_0(&embd),
        ));
        for layer in 0..2 {
            let q = rnd(dim * dim, 0.6);
            let kv = rnd(n_kv * dim, 0.6);
            let ffn_up = rnd(ff * dim, 0.3);
            let ffn_down = rnd(dim * ff, 0.3);
            let mut push = |suffix: &str, dims: Vec<u64>, data: &[f32]| {
                quant.push((
                    format!("blk.{layer}.{suffix}"),
                    dims,
                    quantize_q8_0(data),
                ));
            };
            push("attn_q.weight", vec![dim as u64, dim as u64], &q);
            push("attn_k.weight", vec![dim as u64, n_kv as u64], &kv);
            push("attn_v.weight", vec![dim as u64, n_kv as u64], &kv);
            push("attn_output.weight", vec![dim as u64, dim as u64], &q);
            push("ffn_gate.weight", vec![dim as u64, ff as u64], &ffn_up);
            push("ffn_up.weight", vec![dim as u64, ff as u64], &ffn_up);
            push("ffn_down.weight", vec![ff as u64, dim as u64], &ffn_down);
            f32s.push((
                format!("blk.{layer}.attn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
            f32s.push((
                format!("blk.{layer}.ffn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
        }
        f32s.push((
            "output_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        let head = rnd(vocab * dim, 1.0);
        quant.push((
            "output.weight".into(),
            vec![vocab as u64, dim as u64],
            quantize_q8_0(&head),
        ));
        let f32_tensors: Vec<GgufWriteTensor> = f32s
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data: data.as_slice(),
            })
            .collect();
        let quant_tensors: Vec<GgufWriteQuantTensor> = quant
            .iter()
            .map(|(name, dims, bytes)| GgufWriteQuantTensor {
                name: name.clone(),
                dims: dims.clone(),
                ggml_type: GgmlType::Q8_0,
                bytes: bytes.as_slice(),
            })
            .collect();
        let path = std::env::temp_dir().join(name);
        write_gguf_quant(&path, &meta, &f32_tensors, &quant_tensors).unwrap();
        path
    }

    #[test]
    fn rpc_packed_q8_layer_matches_cpu() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let path = tiny_llama_q8_gguf("allpaca_rpc_packed_q8.gguf");

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });
        // Long enough that positions matter (12 prompt tokens + 8 generated).
        let prompt = "hello a b c d a b c d hello a b";
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(2),
            ..GenerationParams::default()
        };
        let mut cpu_cfg = Config::compiled_defaults();
        cpu_cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(2),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cpu_cfg.model.synthetic = false;
        cpu_cfg.model.path = Some(path.clone());
        cpu_cfg.kv.page_tokens = 8;
        let cpu = Engine::new(cpu_cfg).unwrap();
        let cpu_ids = cpu.generate(prompt, params.clone()).unwrap().token_ids;

        let mut rpc_cfg = Config::compiled_defaults();
        rpc_cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(2),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        rpc_cfg.model.synthetic = false;
        rpc_cfg.model.path = Some(path);
        rpc_cfg.kv.page_tokens = 8;
        rpc_cfg.rpc.endpoints = vec![addr.to_string()];
        rpc_cfg.rpc.max_seq = 32;
        rpc_cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let rpc = Engine::new(rpc_cfg).unwrap();
        let rpc_out = rpc.generate(prompt, params).unwrap();
        let stats = rpc.stats();
        drop(rpc);
        worker.join().unwrap();
        assert!(
            stats.rpc_roundtrips > 0,
            "packed layer must run, trips={}",
            stats.rpc_roundtrips
        );
        assert!(stats.rpc_h2d_bytes > 0, "packed upload missing");
        assert!(
            stats.rpc_h2d_bytes < 20_000,
            "packed Q8 upload should stay under the F32 size, got {}",
            stats.rpc_h2d_bytes
        );
        assert_eq!(rpc_out.token_ids, cpu_ids);
    }

    /// A HIP host prefills a prompt through layers on an RPC worker in
    /// chunks of `rpc.prefill_rows` rows per round trip, and decodes the same
    /// tokens as one round trip per prompt token, on a CPU and a HIP worker.
    #[cfg(feature = "hip")]
    #[test]
    fn rpc_batched_prefill_matches_per_token() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        if allpaca_hip::device_count() == 0 {
            return;
        }
        for arch in ["llama", "qwen2"] {
            // qwen2 rotates NeoX-style: the worker's partial-RoPE row op.
            let path = tiny_dense_q8_gguf(&format!("allpaca_rpc_batched_prefill_{arch}.gguf"), arch);
            // 12 prompt tokens: chunks of 5 rows cover 5 + 5 + 2.
            let prompt = "hello a b c d a b c d hello a b";
            let params = GenerationParams {
                max_tokens: 8,
                temperature: 0.0,
                seed: Some(2),
                ..GenerationParams::default()
            };
            let run = |prefill_rows: usize, hip_worker: bool| {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let addr = listener.local_addr().unwrap();
                let worker = thread::spawn(move || {
                    let (stream, _) = listener.accept().unwrap();
                    let backend = if hip_worker {
                        WorkerBackend::Hip(allpaca_rpc::HipWorker::new(0).unwrap())
                    } else {
                        WorkerBackend::Cpu(CpuWorker::new(64 << 20))
                    };
                    serve_connection(stream, backend).unwrap();
                });
                let mut cfg = Config::compiled_defaults();
                cfg.merge_cli(&CliOverrides {
                    seed: Some(2),
                    speculative: Some(false),
                    ..CliOverrides::default()
                });
                cfg.engine.gpu_backend = "hip".into();
                cfg.model.synthetic = false;
                cfg.model.path = Some(path.clone());
                cfg.kv.page_tokens = 8;
                cfg.rpc.endpoints = vec![addr.to_string()];
                cfg.rpc.max_seq = 32;
                cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
                cfg.rpc.prefill_rows = prefill_rows;
                let engine = Engine::new(cfg).unwrap();
                let out = engine.generate(prompt, params.clone()).unwrap();
                let trips = engine.stats().rpc_roundtrips;
                drop(engine);
                worker.join().unwrap();
                (out.token_ids, trips)
            };
            for hip_worker in [false, true] {
                let (per_token, per_token_trips) = run(0, hip_worker);
                let (batched, batched_trips) = run(5, hip_worker);
                assert!(per_token_trips > 0, "a layer must run on the worker");
                assert_eq!(
                    batched, per_token,
                    "batched prefill must decode the same tokens (hip worker: {hip_worker})"
                );
                // Prompt: 12 trips one by one, 3 in chunks of 5; decode is the same.
                assert_eq!(
                    per_token_trips - batched_trips,
                    12 - 3,
                    "trips: per-token {per_token_trips}, batched {batched_trips}"
                );
            }
        }
    }

    #[test]
    fn rpc_run_span_fuses_contiguous_remote_layers() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(2),
            ..GenerationParams::default()
        };
        let mut wide = ModelConfig::tiny();
        wide.n_layers = 4;

        let mut cpu_cfg = test_config();
        cpu_cfg.model.synthetic = false;
        cpu_cfg.speculative.enabled = false;
        let cpu = Engine::new(cpu_cfg).unwrap();
        cpu.load_synthetic(wide.clone()).unwrap();
        let cpu_ids = cpu.generate("hello", params.clone()).unwrap().token_ids;

        let mut rpc_cfg = test_config();
        rpc_cfg.model.synthetic = false;
        rpc_cfg.speculative.enabled = false;
        rpc_cfg.rpc.endpoints = vec![addr.to_string()];
        rpc_cfg.rpc.max_seq = 64;
        rpc_cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let rpc = Engine::new(rpc_cfg).unwrap();
        rpc.load_synthetic(wide).unwrap();
        let rpc_out = rpc.generate("hello", params).unwrap();
        let stats = rpc.stats();
        drop(rpc);
        worker.join().unwrap();

        assert_eq!(rpc_out.token_ids, cpu_ids);
        assert_eq!(
            stats.rpc_roundtrips, 6,
            "two contiguous remote layers must be one RunSpan per forward"
        );
        assert_eq!(stats.rpc_activation_bytes, 1536);
    }

    #[test]
    fn tensor_split_changes_which_layers_run_on_rpc_server() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
            }
        });

        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(2),
            ..GenerationParams::default()
        };
        let mut base = test_config();
        base.speculative.enabled = false;
        base.rpc.endpoints = vec![addr.to_string()];
        base.rpc.max_seq = 64;

        let mut local_cfg = base.clone();
        local_cfg.rpc.tensor_split = None;
        let local = Engine::new(local_cfg).unwrap();
        assert!(!local
            .generate("hello", params.clone())
            .unwrap()
            .token_ids
            .is_empty());
        let local_trips = local.stats().rpc_roundtrips;
        drop(local);

        let mut split_cfg = base;
        split_cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let split = Engine::new(split_cfg).unwrap();
        assert!(!split
            .generate("hello", params)
            .unwrap()
            .token_ids
            .is_empty());
        let split_trips = split.stats().rpc_roundtrips;
        drop(split);

        assert_eq!(
            local_trips, 0,
            "default memory split keeps tiny layers local; got {local_trips}"
        );
        assert!(
            split_trips > local_trips,
            "even --tensor-split must run layers on the Rpc Server; local={local_trips} split={split_trips}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn engine_new_is_fatal_when_rpc_hello_major_mismatches() {
        use allpaca_rpc::{KERNEL_ABI_REV, PROTO_MAJOR};

        let (addr, worker) = spawn_hello_stub(hello_stub_reply(
            PROTO_MAJOR.wrapping_add(1),
            0,
            KERNEL_ABI_REV,
        ));
        let mut cfg = test_config();
        cfg.rpc.endpoints = vec![addr];
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("HELLO major mismatch must refuse Engine::new"),
            Err(e) => e,
        };
        assert!(
            err.is_fatal(),
            "HELLO major mismatch must be Fatal, got {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("version mismatch"),
            "Fatal must name the version mismatch, got {msg}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn engine_new_is_fatal_when_rpc_hello_minor_is_newer() {
        use allpaca_rpc::{KERNEL_ABI_REV, PROTO_MAJOR, PROTO_MINOR};

        let (addr, worker) = spawn_hello_stub(hello_stub_reply(
            PROTO_MAJOR,
            PROTO_MINOR.wrapping_add(1),
            KERNEL_ABI_REV,
        ));
        let mut cfg = test_config();
        cfg.rpc.endpoints = vec![addr];
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("newer HELLO minor must refuse Engine::new"),
            Err(e) => e,
        };
        assert!(err.is_fatal(), "newer HELLO minor must be Fatal, got {err}");
        let msg = err.to_string();
        assert!(
            msg.contains("version mismatch"),
            "Fatal must name the version mismatch, got {msg}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn engine_new_is_fatal_when_rpc_kernel_abi_is_newer() {
        use allpaca_rpc::{KERNEL_ABI_REV, PROTO_MAJOR, PROTO_MINOR};

        let (addr, worker) = spawn_hello_stub(hello_stub_reply(
            PROTO_MAJOR,
            PROTO_MINOR,
            KERNEL_ABI_REV.saturating_add(1),
        ));
        let mut cfg = test_config();
        cfg.rpc.endpoints = vec![addr];
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("newer kernel ABI must refuse Engine::new"),
            Err(e) => e,
        };
        assert!(err.is_fatal(), "newer kernel ABI must be Fatal, got {err}");
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("abi"),
            "Fatal must name the kernel ABI, got {msg}"
        );
        worker.join().unwrap();
    }

    #[test]
    fn engine_new_is_fatal_when_rpc_server_unreachable() {
        let mut cfg = test_config();
        cfg.rpc.endpoints = vec!["127.0.0.1:1".into()];
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("unreachable Rpc Server must refuse Engine::new"),
            Err(e) => e,
        };
        assert!(
            err.is_fatal(),
            "unreachable Rpc Server must be Fatal, got {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("rpc"),
            "Fatal must name the Rpc path, got {msg}"
        );
    }

    #[test]
    fn engine_new_is_fatal_when_tensor_split_len_mismatches_devices() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![addr.to_string()];
        cfg.rpc.tensor_split = Some(vec![1.0]);
        let err = match Engine::new(cfg) {
            Ok(_) => panic!("tensor_split length mismatch must refuse Engine::new"),
            Err(e) => e,
        };
        let _ = worker.join();
        assert!(
            err.is_fatal(),
            "tensor_split length mismatch must be Fatal, got {err}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("tensor_split"),
            "Fatal must name tensor_split, got {msg}"
        );
    }

    #[test]
    fn generate_is_fatal_when_rpc_server_drops() {
        use std::net::{Shutdown, TcpListener};
        use std::sync::mpsc;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            tx.send(stream.try_clone().unwrap()).unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![addr.to_string()];
        cfg.rpc.max_seq = 64;
        cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
        let engine = Engine::new(cfg).unwrap();
        let peer = rx.recv().unwrap();
        peer.shutdown(Shutdown::Both).unwrap();
        drop(peer);

        let err = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )
            .unwrap_err();
        assert!(
            err.is_fatal(),
            "Rpc Server drop mid-generate must be Fatal, got {err}"
        );
        let _ = worker.join();
    }

    #[test]
    fn two_rpc_servers_greedy_matches_cpu() {
        use std::net::TcpListener;
        use std::thread;

        use allpaca_rpc::{serve_connection, CpuWorker, WorkerBackend};

        let l1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let l2 = TcpListener::bind("127.0.0.1:0").unwrap();
        let a1 = l1.local_addr().unwrap();
        let a2 = l2.local_addr().unwrap();
        let w1 = thread::spawn(move || {
            let (stream, _) = l1.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });
        let w2 = thread::spawn(move || {
            let (stream, _) = l2.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(2),
            ..GenerationParams::default()
        };
        let mut cpu_cfg = test_config();
        cpu_cfg.speculative.enabled = false;
        let cpu_ids = Engine::new(cpu_cfg)
            .unwrap()
            .generate("hello", params.clone())
            .unwrap()
            .token_ids;

        let mut rpc_cfg = test_config();
        rpc_cfg.speculative.enabled = false;
        rpc_cfg.rpc.endpoints = vec![a1.to_string(), a2.to_string()];
        rpc_cfg.rpc.max_seq = 64;
        rpc_cfg.rpc.tensor_split = Some(vec![0.0, 1.0, 1.0]);
        let rpc = Engine::new(rpc_cfg).unwrap();
        let rpc_out = rpc.generate("hello", params).unwrap();
        let trips = rpc.stats().rpc_roundtrips;
        drop(rpc);
        w1.join().unwrap();
        w2.join().unwrap();

        assert!(
            trips >= 2,
            "both Rpc Servers must run layers, rpc_roundtrips={trips}"
        );
        assert_eq!(rpc_out.token_ids, cpu_ids);
    }

    #[test]
    fn early_exit_self_speculate_records_accepts() {
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.early_exit.enabled = true;
        cfg.early_exit.min_confidence = 0.51;
        cfg.early_exit.min_calibration_margin = 0.0;
        let engine = Engine::new(cfg).unwrap();
        // Gate fired on load (unsupported); enable manifest for decode.
        engine.test_enable_early_exit_manifest(vec![0]);
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.early_exit_accepts >= 1,
            "supported exit layer must self-speculate; stats={stats:?}"
        );
    }

    #[test]
    fn early_exit_gates_when_model_lacks_manifest() {
        let mut cfg = test_config();
        cfg.early_exit.enabled = true;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let stats = engine.stats();
        assert!(
            stats.early_exit_gated >= 1,
            "unsupported synthetic must gate early_exit; stats={stats:?}"
        );
        assert!(
            stats.degraded_events >= 1,
            "gate must count as Degraded; stats={stats:?}"
        );
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 2,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty(), "gate must not block generate");
        assert_eq!(engine.stats().early_exit_accepts, 0);
    }

    #[test]
    fn early_exit_loads_gguf_manifest_and_dedicated_head() {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let path = std::env::temp_dir().join("allpaca_early_exit_heads.gguf");
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let w = 0.01f32;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
        meta.insert("llama.block_count".into(), GgufValue::Uint32(2));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert(
            "llama.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert("allpaca.early_exit.supported".into(), GgufValue::Bool(true));
        meta.insert(
            "allpaca.early_exit.exit_layers".into(),
            GgufValue::Array(vec![GgufValue::Uint32(0)]),
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
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        owned.push((
            "token_embd.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ));
        for i in 0..2usize {
            owned.push((
                format!("blk.{i}.attn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
            owned.push((
                format!("blk.{i}.attn_q.weight"),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_k.weight"),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_v.weight"),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_output.weight"),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ));
            owned.push((
                format!("blk.{i}.ffn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
            owned.push((
                format!("blk.{i}.ffn_gate.weight"),
                vec![dim as u64, ff as u64],
                vec![w; dim * ff],
            ));
            owned.push((
                format!("blk.{i}.ffn_up.weight"),
                vec![dim as u64, ff as u64],
                vec![w; dim * ff],
            ));
            owned.push((
                format!("blk.{i}.ffn_down.weight"),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ));
        }
        // Sharp exit head: large bias toward token 3 for any input.
        let mut exit_head = vec![0.0f32; vocab * dim];
        for d in 0..dim {
            exit_head[3 * dim + d] = 10.0;
        }
        owned.push((
            "blk.0.exit_head.weight".into(),
            vec![vocab as u64, dim as u64],
            exit_head,
        ));
        owned.push((
            "output_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        owned.push((
            "output.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ));
        let views: Vec<GgufWriteTensor> = owned
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: name.clone(),
                dims: dims.clone(),
                data: data.as_slice(),
            })
            .collect();
        write_gguf(&path, &meta, &views).unwrap();

        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(2),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.synthetic = false;
        cfg.early_exit.enabled = true;
        cfg.early_exit.min_confidence = 0.51;
        cfg.early_exit.min_calibration_margin = 0.0;
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("early-exit GGUF must load");
        assert_eq!(
            engine.stats().early_exit_gated,
            0,
            "GGUF manifest must not gate; stats={:?}",
            engine.stats()
        );
        {
            let g = engine.lock().unwrap();
            let m = g.target.as_ref().unwrap();
            assert!(m.early_exit.supported);
            assert_eq!(m.early_exit.exit_layers, vec![0]);
            assert!(
                m.exit_heads.first().and_then(|h| h.as_ref()).is_some(),
                "dedicated exit head must load"
            );
        }
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )
            .expect("generate");
        assert!(!out.token_ids.is_empty());
        assert!(
            engine.stats().early_exit_accepts >= 1,
            "dedicated exit head must accept; stats={:?}",
            engine.stats()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn text_only_generate_unchanged_with_vision_encoder() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let baseline = test_engine();
        let out_a = baseline.generate("hello", params.clone()).unwrap();

        let with_enc = test_engine();
        with_enc
            .set_vision_encoder(std::sync::Arc::new(crate::NoopVisionEncoder::new(Some(1))))
            .unwrap();
        let out_b = with_enc.generate("hello", params).unwrap();

        assert_eq!(out_a.token_ids, out_b.token_ids);
        assert_eq!(out_a.text, out_b.text);
    }

    #[test]
    fn cpu_vision_install_leaves_weight_residency_zero() {
        let engine = test_engine();
        let dim = ModelConfig::tiny().dim;
        let enc = crate::PatchVisionEncoder::synthetic(Some(0), dim, 3).unwrap();
        engine.set_vision_encoder(std::sync::Arc::new(enc)).unwrap();
        let stats = engine.stats();
        assert_eq!(stats.vision_weight_bytes_on_device, 0);
        assert_eq!(stats.vision_bytes_on_device, 0);
    }

    #[derive(Clone)]
    struct StubVisionEncoder {
        n_tokens: usize,
        dim: usize,
        fill: f32,
        device: Option<i32>,
    }

    impl crate::EncoderPlacement for StubVisionEncoder {
        fn device(&self) -> Option<i32> {
            self.device
        }
    }

    impl crate::VisionEncoder for StubVisionEncoder {
        fn encode(
            &self,
            images: &[crate::ImageInput],
        ) -> Result<crate::VisionTokens, AllpacaError> {
            if images.is_empty() {
                return Ok(crate::VisionTokens::empty(self.device));
            }
            Ok(crate::VisionTokens {
                embeddings: vec![self.fill; self.n_tokens * self.dim],
                n_tokens: self.n_tokens,
                device: self.device,
                grids: Vec::new(),
            })
        }
    }

    #[test]
    fn multimodal_soft_prefix_counts_in_prompt_tokens() {
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let baseline = test_engine();
        let out_text = baseline.generate("hi", params.clone()).unwrap();

        let engine = test_engine();
        engine
            .set_vision_encoder(std::sync::Arc::new(StubVisionEncoder {
                n_tokens: 2,
                dim: 32,
                fill: 0.25,
                device: Some(1),
            }))
            .unwrap();
        let images = [crate::ImageInput::new(vec![1, 2, 3], "image/png")];
        let mut handle = engine
            .begin_request_multimodal("hi", &images, params)
            .unwrap();
        let mut token_ids = Vec::new();
        let finish = engine
            .drive_request(&mut handle, |ev| {
                token_ids.push(ev.token_id);
                ev.finish_reason
            })
            .unwrap();
        let prompt_tokens = handle_prompt_len(&handle, &engine.inner).unwrap();
        assert_eq!(
            prompt_tokens,
            out_text.prompt_tokens + 2,
            "vision soft tokens must prepend the text prompt"
        );
        assert!(!token_ids.is_empty());
        assert!(matches!(finish, FinishReason::Length | FinishReason::Stop));
    }

    #[test]
    fn multimodal_soft_embeds_change_generation() {
        let params = GenerationParams {
            max_tokens: 8,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let images = [crate::ImageInput::new(vec![1], "image/png")];
        let run = |fill: f32| {
            let engine = test_engine();
            engine
                .set_vision_encoder(std::sync::Arc::new(StubVisionEncoder {
                    n_tokens: 2,
                    dim: 32,
                    fill,
                    device: Some(1),
                }))
                .unwrap();
            let mut handle = engine
                .begin_request_multimodal("hi", &images, params.clone())
                .unwrap();
            let mut token_ids = Vec::new();
            engine
                .drive_request(&mut handle, |ev| {
                    token_ids.push(ev.token_id);
                    ev.finish_reason
                })
                .unwrap();
            token_ids
        };
        let a = run(0.0);
        let b = run(1.0);
        assert_ne!(a, b, "distinct soft embeds must change greedy completions");
    }

    #[test]
    fn multimodal_vision_shape_mismatch_is_fatal() {
        let engine = test_engine();
        engine
            .set_vision_encoder(std::sync::Arc::new(StubVisionEncoder {
                n_tokens: 2,
                dim: 8, // tiny model dim is 32
                fill: 0.5,
                device: Some(1),
            }))
            .unwrap();
        let images = [crate::ImageInput::new(vec![1], "image/png")];
        let err = engine
            .begin_request_multimodal(
                "hi",
                &images,
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap_err();
        assert_eq!(err.class(), allpaca_types::ErrorClass::Fatal);
        assert!(err.to_string().contains("vision embeddings"), "err={err}");
    }

    #[test]
    fn multimodal_images_with_noop_are_degraded() {
        let engine = test_engine();
        engine
            .set_vision_encoder(std::sync::Arc::new(crate::NoopVisionEncoder::new(Some(1))))
            .unwrap();
        let images = [crate::ImageInput::new(vec![1, 2, 3], "image/png")];
        let err = engine
            .begin_request_multimodal(
                "hello",
                &images,
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap_err();
        assert_eq!(err.class(), allpaca_types::ErrorClass::Degraded);
        assert!(err.to_string().contains("vision encoder"), "err={err}");
    }

    #[test]
    fn multimodal_images_without_encoder_are_degraded() {
        let engine = test_engine();
        let images = [crate::ImageInput::new(vec![9], "image/jpeg")];
        let err = engine
            .begin_request_multimodal(
                "hello",
                &images,
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .unwrap_err();
        assert_eq!(err.class(), allpaca_types::ErrorClass::Degraded);
        assert!(err.to_string().contains("not registered"), "err={err}");
    }

    #[test]
    fn vision_on_draft_device_suppresses_draft_propose() {
        let params = GenerationParams {
            max_tokens: 6,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let mut engine = test_engine();
        engine.inventory.draft_device = 1;
        engine
            .set_vision_encoder(std::sync::Arc::new(StubVisionEncoder {
                n_tokens: 2,
                dim: 32,
                fill: 0.1,
                device: Some(1),
            }))
            .unwrap();
        let degraded_before = engine.stats().degraded_events;
        let images = [crate::ImageInput::new(vec![1, 2, 3], "image/png")];
        let mut handle = engine
            .begin_request_multimodal("hi", &images, params)
            .unwrap();
        {
            let guard = engine.inner.lock().unwrap();
            let active = guard.active.get(&handle.id).unwrap();
            assert!(
                active.suppress_draft,
                "vision on draft GPU must set suppress_draft"
            );
        }
        assert!(
            engine.stats().degraded_events > degraded_before,
            "contention must bump degraded_events"
        );
        let proposed_before = engine.stats().draft_proposed;
        engine
            .drive_request(&mut handle, |ev| ev.finish_reason)
            .unwrap();
        assert_eq!(
            engine.stats().draft_proposed,
            proposed_before,
            "suppress_draft must skip draft propose"
        );
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
    fn dense_ring_double_buffer_hits_on_live_path() {
        // tiny FFN ≈ 24 KiB/layer; two ring slots fit, neither layer pins.
        let mut cfg = test_config();
        cfg.speculative.enabled = false;
        cfg.weights.trunk_bytes = 64 * 1024;
        cfg.weights.ring_slots = 2;
        let engine = Engine::new(cfg).unwrap();
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 6,
                    temperature: 0.0,
                    seed: Some(9),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.weight_ring_slots >= 2,
            "double-buffer plan must grant two ring slots; stats={stats:?}"
        );
        assert!(
            stats.weight_ring_binds > 0,
            "streamed dense FFN must bind; stats={stats:?}"
        );
        assert!(
            stats.weight_ring_hits > 0,
            "spare-slot L+1 prefetch must produce ring hits on cyclic walk; stats={stats:?}"
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
    fn elastic_rebalance_trims_radix_and_grows_expert_slots() {
        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.host_bytes = 64;
        cfg.weights.cache_bytes = 64; // one expert slot initially
        cfg.kv.vram_bytes = 8 * 1024;
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        let params = GenerationParams {
            max_tokens: 2,
            temperature: 0.0,
            seed: Some(3),
            ..GenerationParams::default()
        };
        // Fill radix with several prompt prefixes.
        for prompt in ["hello", "hello world", "hello there", "goodbye friend"] {
            let _ = engine.generate(prompt, params.clone()).unwrap();
        }
        let before = engine.stats();
        assert!(
            before.radix_bytes > 0,
            "prefills must seed radix checkpoints; stats={before:?}"
        );
        let slots_before = before.expert_slots;

        engine
            .rebalance_experts_elastic()
            .expect("safe-point elastic rebalance");
        let after = engine.stats();
        assert!(
            after.vram_rebalances >= 1,
            "elastic rebalance must record; stats={after:?}"
        );
        assert!(
            after.radix_bytes < before.radix_bytes,
            "radix must shrink; before={} after={}",
            before.radix_bytes,
            after.radix_bytes
        );
        assert!(
            after.expert_slots >= slots_before,
            "expert pool must not shrink; before={slots_before} after={}",
            after.expert_slots
        );
    }

    /// Without a HIP device there is nothing to read, so the sampler must stay
    /// silent and every budget must pass through untouched. A CPU-only run that
    /// silently shrank its expert pool would be a nasty regression.
    #[test]
    fn unsampled_vram_leaves_budgets_and_stats_untouched() {
        let mut cfg = test_config();
        cfg.model.synthetic = true;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let s = engine.stats();
        assert_eq!(s.gpu_samples, 0, "CPU path must not sample; stats={s:?}");
        assert_eq!(s.gpu_vram_total_bytes, 0);
        assert_eq!(s.gpu_util, 0.0);
        assert_eq!(
            engine.clamp_to_measured_vram(1 << 40, 1 << 40),
            (1 << 40, 1 << 40),
            "an unsampled board must not clamp, however large the request"
        );
    }

    /// Once a reading exists, an over-budget request scales down but keeps its
    /// expert-cache to KV ratio, and a request inside headroom is left alone.
    #[test]
    fn measured_vram_scales_over_budget_request_and_keeps_ratio() {
        let mut cfg = test_config();
        cfg.model.synthetic = true;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        // 4 GiB board, 3 GiB already in use by anyone.
        engine
            .metrics
            .record_gpu_sample(3 << 30, 4 << 30, 0, 0, 0.5);

        let (cache, kv) = engine.clamp_to_measured_vram(256 << 20, 256 << 20);
        assert!(
            cache + kv <= 1 << 30,
            "must fit measured headroom; got cache={cache} kv={kv}"
        );
        assert_eq!(cache, kv, "equal halves must stay equal after scaling");

        let (cache, kv) = engine.clamp_to_measured_vram(2 << 30, 1 << 30);
        assert!(cache + kv < 3 << 30, "over-budget request must shrink");
        let ratio = cache as f64 / kv.max(1) as f64;
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "2:1 cache:kv ratio must survive scaling; got {ratio}"
        );

        let small = (4 << 20, 4 << 20);
        assert_eq!(
            engine.clamp_to_measured_vram(small.0, small.1),
            small,
            "a request inside headroom must pass through"
        );
    }

    #[test]
    fn elastic_attention_trim_spills_kv_and_grows_experts() {
        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        cfg.weights.host_bytes = 64;
        cfg.weights.cache_bytes = 64; // one expert slot initially
        cfg.kv.vram_bytes = 8 * 1024;
        cfg.kv.dram_bytes = 64 * 1024;
        cfg.kv.page_tokens = 4;
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        // Request drop releases KV; seed orphan pages so idle rebalance can trim.
        engine.test_seed_vram_kv_pages(4).unwrap();
        let before = engine.stats();
        assert!(
            before.vram_pages >= 2,
            "seeded VRAM KV must be visible; stats={before:?}"
        );
        let slots_before = before.expert_slots;
        let dram_before = before.dram_pages;

        engine
            .rebalance_experts_elastic()
            .expect("safe-point attention-trim elastic");
        let after = engine.stats();
        assert!(
            after.kv_attention_trims >= 1,
            "attention-trim must record cold KV spills; stats={after:?}"
        );
        assert!(
            after.dram_pages > dram_before,
            "trimmed pages must land in DRAM; before={dram_before} after={}",
            after.dram_pages
        );
        assert!(
            after.expert_slots >= slots_before,
            "trimmed KV budget must not shrink experts; before={slots_before} after={}",
            after.expert_slots
        );
        assert!(
            after.vram_rebalances >= 1,
            "elastic must record; stats={after:?}"
        );
    }

    #[test]
    fn auto_elastic_rebalance_on_idle_poll() {
        let mut cfg = test_config();
        cfg.model.synthetic = false;
        cfg.speculative.enabled = false;
        // Keep experts host-resident so this tracer is not an NVMe soak.
        cfg.weights.host_bytes = 0;
        cfg.weights.cache_bytes = 64;
        let engine = Engine::new(cfg).unwrap();
        engine.load_synthetic(ModelConfig::tiny_moe()).unwrap();
        for prompt in ["hello", "hello world", "hello there"] {
            let _ = engine
                .generate(
                    prompt,
                    GenerationParams {
                        max_tokens: 1,
                        temperature: 0.0,
                        seed: Some(1),
                        ..GenerationParams::default()
                    },
                )
                .unwrap();
        }
        let before = engine.stats().vram_rebalances;
        engine.test_mark_elastic_pending();
        assert!(engine.poll().unwrap().is_none());
        assert!(
            engine.stats().vram_rebalances > before,
            "idle poll must run auto elastic rebalance"
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
        let full = full_engine.generate("hello true", params.clone()).unwrap();

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
    fn prompt_cache_on_disk_serves_a_new_engine() {
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(7),
            ..GenerationParams::default()
        };
        let dir = std::env::temp_dir()
            .join("allpaca_prompt_cache_tests")
            .join(format!("engine_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = || {
            let mut cfg = test_config();
            cfg.speculative.enabled = false;
            cfg.prompt_cache.disk_dir = Some(dir.clone());
            cfg
        };
        let full = {
            let mut cfg = test_config();
            cfg.speculative.enabled = false;
            Engine::new(cfg).unwrap().generate("hello world hello true", params.clone()).unwrap()
        };
        Engine::new(config()).unwrap().generate("hello world hello", params.clone()).unwrap();

        let engine = Engine::new(config()).unwrap();
        let hit = engine.generate("hello world hello true", params).unwrap();
        let stats = engine.stats();
        assert!(stats.radix_hits >= 1, "checkpoints on disk must serve a new engine; {stats:?}");
        assert_eq!(hit.token_ids, full.token_ids);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prompt_cache_folders_follow_the_model_and_the_expert_mask() {
        let cfg = ModelConfig::tiny();
        let base = prompt_cache_key(&cfg, None, &[]);
        assert_eq!(base, prompt_cache_key(&cfg, None, &[]));
        assert_ne!(base, prompt_cache_key(&cfg, None, &[vec![3]]));
        assert_ne!(base, prompt_cache_key(&ModelConfig::tiny_hybrid(), None, &[]));
        let file = std::env::temp_dir().join(format!("allpaca_pc_key_{}.bin", std::process::id()));
        std::fs::write(&file, b"one").unwrap();
        let one = prompt_cache_key(&cfg, Some(&file), &[]);
        std::fs::write(&file, b"other").unwrap();
        assert_ne!(one, prompt_cache_key(&cfg, Some(&file), &[]));
        let _ = std::fs::remove_file(&file);
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
        assert_eq!(stats.expert_pcie_bps, 0, "cpu_only has no PCIe expert H2D");
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
        let id = ExpertId {
            layer: 0,
            expert: 1,
        };
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
        assert!(matches!(
            engine.step(&mut first).unwrap(),
            StepResult::Token(_)
        ));
        assert!(matches!(
            engine.step(&mut second).unwrap(),
            StepResult::Token(_)
        ));

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

    /// Two Requests decoded together: speculation stops while more Requests
    /// are active than `speculative.max_concurrent_requests` (0 = no limit).
    #[test]
    fn speculation_pauses_above_max_concurrent_requests() {
        let drafted = |limit: usize| {
            let mut cfg = test_config();
            cfg.engine.max_active_requests = 2;
            cfg.speculative.max_concurrent_requests = limit;
            let engine = Engine::new(cfg).unwrap();
            let params = GenerationParams {
                max_tokens: 24,
                temperature: 0.0,
                seed: Some(7),
                ..GenerationParams::default()
            };
            let prompt = "hello world hello world hello world hello world";
            let a = engine.begin_request(prompt, params.clone()).unwrap();
            let b = engine.begin_request(prompt, params).unwrap();
            for _ in 0..512 {
                if engine.poll().expect("poll").is_none() {
                    break;
                }
            }
            drop((a, b));
            engine.stats().draft_proposed
        };
        assert_eq!(drafted(1), 0, "two active Requests exceed a limit of 1: no drafts");
        assert!(drafted(0) > 0, "without a limit the same Requests draft");
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
        assert!(
            done_a && done_b,
            "both Requests must finish under poll-only driving"
        );
        let stats = engine.stats();
        assert!(
            stats.micro_batch_plans >= 1,
            "concurrent poll must plan stagger; stats={stats:?}"
        );
        assert!(
            stats.micro_batch_dual_steps >= 1,
            "concurrent poll must same-tick dual-step partner; stats={stats:?}"
        );
        assert!(
            stats.batch_gemm_plans >= 1,
            "concurrent decode peers must plan batch GEMM; stats={stats:?}"
        );
        assert!(
            stats.batch_gemm_executes >= 1,
            "concurrent decode peers must soft-execute batch GEMM; stats={stats:?}"
        );
        drop(a);
        drop(b);
    }

    #[test]
    fn poll_batch_gemm_soft_execute_counts() {
        let mut cfg = test_config();
        cfg.engine.max_active_requests = 2;
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 3,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let a = engine.begin_request("hello", params.clone()).unwrap();
        let b = engine.begin_request("world", params).unwrap();
        for _ in 0..64 {
            if engine.poll().expect("poll").is_none() {
                break;
            }
            let stats = engine.stats();
            if stats.batch_gemm_executes >= 1 {
                drop(a);
                drop(b);
                return;
            }
        }
        let stats = engine.stats();
        assert!(
            stats.batch_gemm_executes >= 1,
            "two concurrent decode requests must soft-execute batch GEMM; stats={stats:?}"
        );
        drop(a);
        drop(b);
    }

    /// HIP synthetic: packed decode GEMM increments kernels; lm_head C is applied.
    #[test]
    fn poll_batch_gemm_hip_kernels_with_real_activations_or_skips() {
        if !allpaca_hip::runtime_linked() || allpaca_hip::device_count() == 0 {
            return;
        }
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: false,
            seed: Some(7),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.engine.max_active_requests = 2;
        cfg.engine.gpu_backend = "auto".into();
        cfg.kv.page_tokens = 8;
        cfg.kv.vram_bytes = 8 * 1024;
        cfg.kv.dram_bytes = 64 * 1024;
        let engine = match Engine::new(cfg) {
            Ok(e) if e.stats().backend == "hip" => e,
            _ => return,
        };
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
        let a = engine.begin_request("hello", params.clone()).unwrap();
        let b = engine.begin_request("world", params).unwrap();
        for _ in 0..96 {
            if engine.poll().expect("poll").is_none() {
                break;
            }
            let stats = engine.stats();
            if stats.batch_gemm_hip_kernels >= 7 && stats.batch_gemm_hip_applied >= 1 {
                assert!(
                    stats.batch_gemm_executes >= 1,
                    "HIP packed kernel implies soft execute; stats={stats:?}"
                );
                assert!(
                    stats.batch_gemm_residual_hidden >= 1,
                    "fused residual must land in last_hidden; stats={stats:?}"
                );
                drop(a);
                drop(b);
                return;
            }
        }
        let stats = engine.stats();
        // The device batched decode (one forward over both rows) supersedes
        // the soft packed-GEMM co-step whenever the target is GPU-resident.
        if stats.batched_decode_steps >= 1
            && stats.batched_decode_rows >= 2 * stats.batched_decode_steps
        {
            drop(a);
            drop(b);
            return;
        }
        assert!(
            stats.batch_gemm_hip_kernels >= 7,
            "packed Q/K/V C must feed attention then WO plus FFN(gate/up/down); stats={stats:?}"
        );
        assert!(
            stats.batch_gemm_hip_applied >= 1,
            "packed lm_head logits must be applied to pending_logits; stats={stats:?}"
        );
        assert!(
            stats.batch_gemm_residual_hidden >= 1,
            "fused residual must land in last_hidden; stats={stats:?}"
        );
        drop(a);
        drop(b);
    }

    /// HIP: packed lm_head C is consumed (applied metric) when cohort co-steps.
    #[test]
    fn poll_batch_gemm_hip_applied_or_skips() {
        if !allpaca_hip::runtime_linked() || allpaca_hip::device_count() == 0 {
            return;
        }
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: false,
            seed: Some(11),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.engine.max_active_requests = 2;
        cfg.engine.gpu_backend = "auto".into();
        cfg.kv.page_tokens = 8;
        cfg.kv.vram_bytes = 8 * 1024;
        cfg.kv.dram_bytes = 64 * 1024;
        let engine = match Engine::new(cfg) {
            Ok(e) if e.stats().backend == "hip" => e,
            _ => return,
        };
        let params = GenerationParams {
            max_tokens: 4,
            temperature: 0.0,
            seed: Some(2),
            ..GenerationParams::default()
        };
        let a = engine.begin_request("hello", params.clone()).unwrap();
        let b = engine.begin_request("world", params).unwrap();
        for _ in 0..96 {
            if engine.poll().expect("poll").is_none() {
                break;
            }
            let stats = engine.stats();
            if stats.batch_gemm_hip_applied >= 1 && stats.batch_gemm_lm_head_skipped >= 1 {
                assert!(
                    stats.batch_gemm_hip_kernels >= 1,
                    "applied implies at least one packed HIP kernel; stats={stats:?}"
                );
                assert!(
                    stats.batch_gemm_residual_hidden >= 1,
                    "applied must write fused residual into last_hidden; stats={stats:?}"
                );
                drop(a);
                drop(b);
                return;
            }
        }
        let stats = engine.stats();
        // The device batched decode (one forward over both rows) supersedes
        // the soft packed-GEMM co-step whenever the target is GPU-resident.
        if stats.batched_decode_steps >= 1
            && stats.batched_decode_rows >= 2 * stats.batched_decode_steps
        {
            drop(a);
            drop(b);
            return;
        }
        assert!(
            stats.batch_gemm_hip_applied >= 1,
            "packed lm_head must be applied on HIP cohort; stats={stats:?}"
        );
        assert!(
            stats.batch_gemm_lm_head_skipped >= 1,
            "cohort decode must skip per-request lm_head; stats={stats:?}"
        );
        assert!(
            stats.batch_gemm_residual_hidden >= 1,
            "packed residual must land in last_hidden; stats={stats:?}"
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
    fn image_pads_expand_in_place_into_image_tokens() {
        const PAD: u32 = 99;
        // "a <vs> <pad> <ve> b <vs> <pad> <ve>" with a 2x1 and a 1x3 image.
        let tokens = [1, 50, PAD, 51, 2, 50, PAD, 51];
        let (out, spans) = expand_image_pads(&tokens, Some(PAD), &[(2, 1), (1, 3)]);
        assert_eq!(out, [1, 50, PAD, PAD, 51, 2, 50, PAD, PAD, PAD, 51]);
        assert_eq!(
            spans,
            [
                crate::model::ImageSpan { start: 2, cols: 2, rows: 1 },
                crate::model::ImageSpan { start: 7, cols: 1, rows: 3 },
            ]
        );
        // No pads in the prompt: images go first, in order.
        let (out, spans) = expand_image_pads(&[7, 8], Some(PAD), &[(2, 2)]);
        assert_eq!(out, [PAD, PAD, PAD, PAD, 7, 8]);
        assert_eq!(spans, [crate::model::ImageSpan { start: 0, cols: 2, rows: 2 }]);
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
            for _ in 0..10 {
                inner.scheduler.record_acceptance(9, 10);
            }
            let raised = inner.scheduler.draft_depth;
            assert!(raised > 2);
            // Neutral acceptance keeps depth; queue pressure alone must shrink.
            inner.scheduler.record_acceptance_with_pressure(5, 10, 3);
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
        assert!(
            initial >= 2,
            "initial draft depth from min_depth; stats={initial}"
        );
        {
            // Depth follows sustained acceptance, not one span.
            let mut inner = engine.lock().unwrap();
            for _ in 0..10 {
                inner.scheduler.record_acceptance(9, 10);
            }
        }
        let raised = engine.stats().draft_depth;
        assert!(
            raised > initial,
            "sustained high acceptance must raise draft_depth; before={initial} after={raised}"
        );
        {
            let mut inner = engine.lock().unwrap();
            for _ in 0..30 {
                inner.scheduler.record_acceptance(1, 10);
            }
        }
        let lowered = engine.stats().draft_depth;
        assert!(
            lowered < raised,
            "sustained low acceptance must shrink draft_depth; raised={raised} lowered={lowered}"
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
        assert!(
            !out.token_ids.is_empty(),
            "generate still works after KV spill"
        );
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
        let out = engine
            .generate("hello", params)
            .expect("linear Draft Provider fallback");
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
            stats.tree_path_valid || stats.tree_accepted_path_len > 0 || stats.tokens_generated > 0,
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
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
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
            (
                "token_embd.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
            (
                "blk.0.attn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.0.attn_q.weight".into(),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ),
            (
                "blk.0.attn_k.weight".into(),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ),
            (
                "blk.0.attn_v.weight".into(),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ),
            (
                "blk.0.attn_output.weight".into(),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ),
            (
                "blk.0.ffn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.0.ffn_gate.weight".into(),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ),
            (
                "blk.0.ffn_up.weight".into(),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ),
            (
                "blk.0.ffn_down.weight".into(),
                vec![dim as u64, ff as u64],
                vec![ffn_down; dim * ff],
            ),
            (
                "output_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
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

    fn write_tiny_mtp_gguf(path: &std::path::Path) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let w = 0.01f32;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("llama".into()),
        );
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
        meta.insert("llama.block_count".into(), GgufValue::Uint32(2));
        meta.insert("llama.nextn_predict_layers".into(), GgufValue::Uint32(1));
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
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            (
                "token_embd.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
            (
                "output_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
        ];
        for i in 0..2usize {
            owned.push((
                format!("blk.{i}.attn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
            owned.push((
                format!("blk.{i}.attn_q.weight"),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_k.weight"),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_v.weight"),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_output.weight"),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ));
            owned.push((
                format!("blk.{i}.ffn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
            owned.push((
                format!("blk.{i}.ffn_gate.weight"),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ));
            owned.push((
                format!("blk.{i}.ffn_up.weight"),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ));
            owned.push((
                format!("blk.{i}.ffn_down.weight"),
                vec![dim as u64, ff as u64],
                vec![w; dim * ff],
            ));
        }
        // NextN seed tensors on the MTP block (blk.1). eh_proj: [dim, 2*dim].
        owned.push((
            "blk.1.nextn.eh_proj.weight".into(),
            vec![dim as u64, (2 * dim) as u64],
            vec![w; dim * 2 * dim],
        ));
        owned.push((
            "blk.1.nextn.enorm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        owned.push((
            "blk.1.nextn.hnorm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        owned.push((
            "blk.1.nextn.shared_head_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ));
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
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
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
            (
                "token_embd.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
            (
                "blk.0.attn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.0.attn_q.weight".into(),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ),
            (
                "blk.0.attn_k.weight".into(),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ),
            (
                "blk.0.attn_v.weight".into(),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ),
            (
                "blk.0.attn_output.weight".into(),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ),
            (
                "blk.0.ffn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
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
            (
                "output_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
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
            quantize_q4_0, quantize_q8_0, write_gguf_quant, GgmlType, GgufValue,
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
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
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
        assert!(matches!(
            err,
            AllpacaError::Fatal(FatalError::InvalidConfig(_))
        ));

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

    fn write_tiny_nemotron_hybrid_gguf(path: &std::path::Path) {
        write_tiny_nemotron_hybrid_gguf_ssm_d(path, 0.25);
    }

    fn write_tiny_nemotron_hybrid_gguf_ssm_d(path: &std::path::Path, ssm_d: f32) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        // Layer 0: full Mamba-2 SSD; layer 1: attention + dense FFN.
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let d_inner = 4usize;
        let d_state = 2usize;
        let n_heads = 2usize;
        let n_groups = 1usize;
        let conv_kernel = 2usize;
        let conv_dim = d_inner + 2 * n_groups * d_state;
        let in_proj = 2 * d_inner + 2 * n_groups * d_state + n_heads;
        let w = 0.01f32;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("nemotron_h_moe".into()),
        );
        meta.insert(
            "nemotron_h_moe.vocab_size".into(),
            GgufValue::Uint32(vocab as u32),
        );
        meta.insert(
            "nemotron_h_moe.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
        meta.insert("nemotron_h_moe.block_count".into(), GgufValue::Uint32(2));
        meta.insert(
            "nemotron_h_moe.attention.head_count".into(),
            GgufValue::Uint32(4),
        );
        meta.insert(
            "nemotron_h_moe.attention.head_count_kv".into(),
            GgufValue::Uint32(1),
        );
        meta.insert(
            "nemotron_h_moe.feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.inner_size".into(),
            GgufValue::Uint32(d_inner as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.state_size".into(),
            GgufValue::Uint32(d_state as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.time_step_rank".into(),
            GgufValue::Uint32(n_heads as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.group_count".into(),
            GgufValue::Uint32(n_groups as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.conv_kernel".into(),
            GgufValue::Uint32(conv_kernel as u32),
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
        let owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            (
                "token_embd.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
            (
                "blk.0.attn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.0.ssm_in.weight".into(),
                vec![dim as u64, in_proj as u64],
                vec![w; dim * in_proj],
            ),
            (
                "blk.0.ssm_out.weight".into(),
                vec![d_inner as u64, dim as u64],
                vec![w; dim * d_inner],
            ),
            (
                "blk.0.ssm_conv1d.weight".into(),
                vec![conv_kernel as u64, conv_dim as u64],
                vec![0.5; conv_kernel * conv_dim],
            ),
            (
                "blk.0.ssm_conv1d.bias".into(),
                vec![conv_dim as u64],
                vec![0.01; conv_dim],
            ),
            (
                "blk.0.ssm_a".into(),
                vec![1, n_heads as u64],
                vec![-1.0; n_heads],
            ),
            (
                "blk.0.ssm_d".into(),
                vec![1, n_heads as u64],
                vec![ssm_d; n_heads],
            ),
            (
                "blk.0.ssm_dt.bias".into(),
                vec![n_heads as u64],
                vec![0.0; n_heads],
            ),
            (
                "blk.0.ssm_norm.weight".into(),
                vec![(d_inner / n_groups) as u64, n_groups as u64],
                vec![1.0; d_inner],
            ),
            (
                "blk.1.attn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.1.attn_q.weight".into(),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ),
            (
                "blk.1.attn_k.weight".into(),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ),
            (
                "blk.1.attn_v.weight".into(),
                vec![2, dim as u64],
                vec![w; 2 * dim],
            ),
            (
                "blk.1.attn_output.weight".into(),
                vec![dim as u64, dim as u64],
                vec![w; dim * dim],
            ),
            (
                "blk.1.ffn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.1.ffn_gate.weight".into(),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ),
            (
                "blk.1.ffn_up.weight".into(),
                vec![ff as u64, dim as u64],
                vec![w; ff * dim],
            ),
            (
                "blk.1.ffn_down.weight".into(),
                vec![dim as u64, ff as u64],
                vec![w; dim * ff],
            ),
            (
                "output_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "output.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
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

    /// Nemotron-Lightning MoE block: `attn_norm` (no `ffn_norm`), stacked `up`/`down`
    /// experts without `gate_exps`.
    fn write_tiny_nemotron_lightning_moe_gguf(path: &std::path::Path) {
        write_tiny_nemotron_lightning_moe_gguf_opts(path, false);
    }

    fn write_tiny_nemotron_lightning_moe_gguf_opts(path: &std::path::Path, with_shexp: bool) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 8usize;
        let vocab = 4usize;
        let ff = 8usize;
        let n_exp = 2usize;
        let d_inner = 4usize;
        let d_state = 2usize;
        let n_heads = 2usize;
        let n_groups = 1usize;
        let conv_kernel = 2usize;
        let conv_dim = d_inner + 2 * n_groups * d_state;
        let in_proj = 2 * d_inner + 2 * n_groups * d_state + n_heads;
        let w = 0.01f32;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("nemotron_h_moe".into()),
        );
        meta.insert(
            "nemotron_h_moe.vocab_size".into(),
            GgufValue::Uint32(vocab as u32),
        );
        meta.insert(
            "nemotron_h_moe.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
        meta.insert("nemotron_h_moe.block_count".into(), GgufValue::Uint32(2));
        meta.insert(
            "nemotron_h_moe.attention.head_count".into(),
            GgufValue::Uint32(4),
        );
        meta.insert(
            "nemotron_h_moe.attention.head_count_kv".into(),
            GgufValue::Uint32(1),
        );
        meta.insert(
            "nemotron_h_moe.expert_feed_forward_length".into(),
            GgufValue::Uint32(ff as u32),
        );
        meta.insert(
            "nemotron_h_moe.expert_count".into(),
            GgufValue::Uint32(n_exp as u32),
        );
        meta.insert(
            "nemotron_h_moe.expert_used_count".into(),
            GgufValue::Uint32(1),
        );
        meta.insert(
            "nemotron_h_moe.ssm.inner_size".into(),
            GgufValue::Uint32(d_inner as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.state_size".into(),
            GgufValue::Uint32(d_state as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.time_step_rank".into(),
            GgufValue::Uint32(n_heads as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.group_count".into(),
            GgufValue::Uint32(n_groups as u32),
        );
        meta.insert(
            "nemotron_h_moe.ssm.conv_kernel".into(),
            GgufValue::Uint32(conv_kernel as u32),
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
        let shared_ff = 12usize;
        if with_shexp {
            meta.insert(
                "nemotron_h_moe.expert_shared_count".into(),
                GgufValue::Uint32(1),
            );
            meta.insert(
                "nemotron_h_moe.expert_shared_feed_forward_length".into(),
                GgufValue::Uint32(shared_ff as u32),
            );
            meta.insert(
                "nemotron_h_moe.expert_weights_norm".into(),
                GgufValue::Bool(true),
            );
            meta.insert(
                "nemotron_h_moe.expert_weights_scale".into(),
                GgufValue::Float32(2.5),
            );
        }
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![
            (
                "token_embd.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
            (
                "blk.0.attn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.0.ssm_in.weight".into(),
                vec![dim as u64, in_proj as u64],
                vec![w; dim * in_proj],
            ),
            (
                "blk.0.ssm_out.weight".into(),
                vec![d_inner as u64, dim as u64],
                vec![w; dim * d_inner],
            ),
            (
                "blk.0.ssm_conv1d.weight".into(),
                vec![conv_kernel as u64, conv_dim as u64],
                vec![0.5; conv_kernel * conv_dim],
            ),
            (
                "blk.0.ssm_conv1d.bias".into(),
                vec![conv_dim as u64],
                vec![0.01; conv_dim],
            ),
            (
                "blk.0.ssm_a".into(),
                vec![1, n_heads as u64],
                vec![-1.0; n_heads],
            ),
            (
                "blk.0.ssm_d".into(),
                vec![1, n_heads as u64],
                vec![0.25; n_heads],
            ),
            (
                "blk.0.ssm_dt.bias".into(),
                vec![n_heads as u64],
                vec![0.0; n_heads],
            ),
            (
                "blk.0.ssm_norm.weight".into(),
                vec![(d_inner / n_groups) as u64, n_groups as u64],
                vec![1.0; d_inner],
            ),
            // MoE-only: Lightning uses attn_norm, up/down exps, no ffn_norm / gate_exps.
            (
                "blk.1.attn_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "blk.1.ffn_gate_inp.weight".into(),
                vec![dim as u64, n_exp as u64],
                vec![w; dim * n_exp],
            ),
            (
                "blk.1.ffn_up_exps.weight".into(),
                vec![dim as u64, ff as u64, n_exp as u64],
                vec![w; dim * ff * n_exp],
            ),
            (
                "blk.1.ffn_down_exps.weight".into(),
                vec![ff as u64, dim as u64, n_exp as u64],
                vec![w; ff * dim * n_exp],
            ),
            (
                "output_norm.weight".into(),
                vec![dim as u64],
                vec![1.0; dim],
            ),
            (
                "output.weight".into(),
                vec![vocab as u64, dim as u64],
                vec![w; vocab * dim],
            ),
        ];
        if with_shexp {
            owned.push((
                "blk.1.ffn_up_shexp.weight".into(),
                vec![dim as u64, shared_ff as u64],
                vec![w; dim * shared_ff],
            ));
            owned.push((
                "blk.1.ffn_down_shexp.weight".into(),
                vec![shared_ff as u64, dim as u64],
                vec![w; shared_ff * dim],
            ));
        }
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

    #[test]
    fn nemotron_lightning_moe_loads_without_ffn_norm_or_gate_exps() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_nemotron_lightning_moe-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_nemotron_lightning_moe_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).expect("Lightning MoE layout must load");
        let out = engine
            .generate(
                "hi",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("Lightning MoE must generate");
        assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn nemotron_lightning_shexp_sets_routing_flags_and_generates() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_nemotron_lightning_shexp-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_nemotron_lightning_moe_gguf_opts(&path, true);
        let file = allpaca_gguf::GgufFile::open(&path).expect("shexp fixture");
        let mcfg = crate::model::ModelConfig::from_gguf(&file).expect("lightning cfg");
        assert!(mcfg.moe_sigmoid_gating, "{mcfg:?}");
        assert!(mcfg.moe_relu_sqr, "{mcfg:?}");
        assert!(mcfg.expert_weights_norm, "{mcfg:?}");
        assert!((mcfg.expert_weights_scale - 2.5).abs() < 1e-6, "{mcfg:?}");
        assert_eq!(mcfg.shared_ff_dim, 12, "{mcfg:?}");
        assert_eq!(mcfg.n_shared_experts, 1, "{mcfg:?}");
        let model = crate::model::TinyModel::from_gguf(&file).expect("lightning model");
        assert!(
            model.supports_device_batched_prefill(),
            "Lightning hybrid/MoE should opt into batched prefill"
        );
        assert!(
            !model.layers[1].shared_expert.w_up.is_empty(),
            "shexp must materialize on the MoE block"
        );
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).expect("Lightning shexp layout must load");
        let out = engine
            .generate(
                "hi",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("Lightning shexp must generate");
        assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn tiny_deepseek4_cpu_generates() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4_gen-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        crate::deepseek4::write_tiny_deepseek4_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).expect("tiny DeepSeek-V4 must load");
        let out = engine
            .generate(
                "hi",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("tiny DeepSeek-V4 must generate");
        assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
        let _ = std::fs::remove_file(path);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiny_deepseek4_gguf_rejects_upload() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_cuda_tiny_dsv4-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        crate::deepseek4::write_tiny_deepseek4_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: false,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg.engine.gpu_backend = "cuda".into();
        cfg.speculative.enabled = false;
        let engine = match Engine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                panic!("tiny DeepSeek-V4 GGUF must load: {e}");
            }
        };
        let stats = engine.stats();
        let _ = std::fs::remove_file(&path);
        if stats.backend != "cuda" {
            return;
        }
        assert_eq!(
            stats.weight_bytes_on_device, 0,
            "MLA/DSV4 must not CUDA-upload; stats={stats:?}"
        );
        assert!(
            stats.degraded_events >= 1,
            "CUDA MLA upload reject must degrade; stats={stats:?}"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_tiny_deepseek4_gguf_rejects_upload() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_vulkan_tiny_dsv4-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        crate::deepseek4::write_tiny_deepseek4_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: false,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.clone());
        cfg.engine.gpu_backend = "vulkan".into();
        cfg.speculative.enabled = false;
        let engine = match Engine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                panic!("tiny DeepSeek-V4 GGUF must load: {e}");
            }
        };
        let stats = engine.stats();
        let _ = std::fs::remove_file(&path);
        if stats.backend != "vulkan" {
            return;
        }
        assert_eq!(
            stats.weight_bytes_on_device, 0,
            "MLA/DSV4 must not Vulkan-upload; stats={stats:?}"
        );
        assert!(
            stats.degraded_events >= 1,
            "Vulkan MLA upload reject must degrade; stats={stats:?}"
        );
    }

    #[cfg(feature = "hip")]
    fn dsv4_generate_tokens(path: &std::path::Path, cpu_only: bool) -> (Vec<u32>, StatsSnapshot) {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path.to_path_buf());
        cfg.speculative.enabled = false;
        let engine = Engine::new(cfg).expect("tiny DeepSeek-V4 must load");
        let out = engine
            .generate(
                "hi",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("tiny DeepSeek-V4 must generate");
        (out.token_ids, engine.stats())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn tiny_deepseek4_hip_f32_stays_cpu() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip tiny_deepseek4_hip_f32_stays_cpu: no HIP device");
            return;
        }
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4_f32_hip-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        crate::deepseek4::write_tiny_deepseek4_gguf(&path);
        let (cpu_ids, _) = dsv4_generate_tokens(&path, true);
        let (hip_ids, hip_stats) = dsv4_generate_tokens(&path, false);
        let _ = std::fs::remove_file(&path);
        assert!(!cpu_ids.is_empty(), "CPU F32 fixture must emit tokens");
        assert_eq!(
            hip_ids, cpu_ids,
            "F32 DeepSeek-V4 must stay on CPU gemv; hip={hip_ids:?} cpu={cpu_ids:?}"
        );
        assert_eq!(
            hip_stats.gpu_forwards, 0,
            "F32 fixture must not count GPU forwards; stats={hip_stats:?}"
        );
    }

    #[cfg(feature = "hip")]
    #[test]
    fn tiny_deepseek4_hip_packed_matches_cpu() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip tiny_deepseek4_hip_packed_matches_cpu: no HIP device");
            return;
        }
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_dsv4_packed_hip-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        crate::deepseek4::write_tiny_deepseek4_packed_gguf(&path);
        let (cpu_ids, _) = dsv4_generate_tokens(&path, true);
        let (hip_ids, hip_stats) = dsv4_generate_tokens(&path, false);
        let _ = std::fs::remove_file(&path);
        assert!(!cpu_ids.is_empty(), "packed fixture must emit tokens");
        assert_eq!(
            hip_ids, cpu_ids,
            "packed BF16/MXFP4 HIP must match CPU; hip={hip_ids:?} cpu={cpu_ids:?} stats={hip_stats:?}"
        );
        if hip_stats.backend == "hip" {
            assert!(
                hip_stats.gpu_forwards > 0,
                "packed HIP generate should record gpu_forwards; stats={hip_stats:?}"
            );
        }
    }

    fn unsloth_dsv4_q8_path() -> Option<std::path::PathBuf> {
        if let Ok(p) = std::env::var("ALLPACA_DSV4_GGUF") {
            let pb = std::path::PathBuf::from(p);
            return pb.is_file().then_some(pb);
        }
        let default = std::path::PathBuf::from(
            "/zepool/cache/huggingface/hub/models--unsloth--DeepSeek-V4-Flash-0731-GGUF/snapshots/fbbb5b93fb787c21338159b0af3318bb3f4d9768/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf",
        );
        default.is_file().then_some(default)
    }

    #[test]
    fn unsloth_dsv4_q8_loads_when_env_set() {
        if std::env::var("ALLPACA_DSV4").ok().as_deref() != Some("1") {
            return;
        }
        let path = unsloth_dsv4_q8_path().expect("ALLPACA_DSV4=1 but GGUF missing");
        {
            let file = allpaca_gguf::GgufFile::open(&path).expect("probe open");
            for name in [
                "token_embd.weight",
                "output.weight",
                "output_hc_fn.weight",
                "output_hc_base.weight",
                "output_hc_scale.weight",
                "blk.0.attn_q_a.weight",
                "blk.0.attn_q_a_norm.weight",
                "blk.0.attn_q_b.weight",
                "blk.0.attn_kv.weight",
                "blk.0.attn_kv_a_norm.weight",
                "blk.0.attn_output_a.weight",
                "blk.0.attn_output_b.weight",
                "blk.0.hc_attn_fn.weight",
                "blk.0.hc_attn_base.weight",
                "blk.0.hc_attn_scale.weight",
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_gate_exps.weight",
                "blk.0.ffn_up_exps.weight",
                "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_up_shexp.weight",
                "blk.0.ffn_down_shexp.weight",
            ] {
                match file.tensor(name) {
                    Some(t) => eprintln!(
                        "dsv4 tensor {name} dims={:?} type={:?} n={}",
                        t.dims,
                        t.ggml_type,
                        t.n_elements()
                    ),
                    None => eprintln!("dsv4 tensor {name} MISSING"),
                }
            }
        }
        if let Some(dsp) = allpaca_gguf::discover_dspark_sidecar(&path) {
            let file = allpaca_gguf::GgufFile::open(&dsp).expect("dspark probe open");
            eprintln!(
                "dspark sidecar {} arch={:?} n_tensors={}",
                dsp.display(),
                file.architecture(),
                file.tensors.len()
            );
            for name in [
                "markov_w1.weight",
                "markov_w2.weight",
                "dspark.markov_w1.weight",
                "dspark.markov_w2.weight",
            ] {
                match file.tensor(name) {
                    Some(t) => eprintln!(
                        "dspark tensor {name} dims={:?} type={:?}",
                        t.dims, t.ggml_type
                    ),
                    None => eprintln!("dspark tensor {name} MISSING"),
                }
            }
        } else {
            eprintln!("dspark sidecar not found next to {}", path.display());
        }
        let mut cfg = Config::compiled_defaults();
        let hip = std::env::var("ALLPACA_DSV4_HIP").ok().as_deref() == Some("1");
        cfg.merge_cli(&CliOverrides {
            cpu_only: !hip,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path);
        cfg.speculative.enabled = true;
        cfg.speculative.providers = vec![DraftProviderKind::Dspark, DraftProviderKind::Ngram];
        let engine = Engine::new(cfg).expect("Unsloth DeepSeek-V4-Flash Q8 must load");
        let stats = engine.stats();
        eprintln!(
            "dsv4 load ok backend={} degraded={}",
            stats.backend, stats.degraded_events
        );
        if std::env::var("ALLPACA_DSV4_GENERATE").ok().as_deref() != Some("1") {
            return;
        }
        let chatml = std::env::var("ALLPACA_DSV4_CHATML").ok().as_deref() == Some("1");
        let prompt = if chatml {
            crate::format_chatml(&[("user", "hi")], true)
        } else {
            "hi".to_string()
        };
        let out = engine
            .generate(
                &prompt,
                GenerationParams {
                    max_tokens: 1,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("Unsloth DeepSeek-V4-Flash must emit a token");
        assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
        let stats = engine.stats();
        eprintln!(
            "dsv4 generate ok chatml={} prompt_chars={} tokens={:?} text={:?} ttft_ms={} itl_ms={:.2} backend={} gpu_forwards={} target_gpu_forwards={} degraded={} dsv4_h2d_us={:.0} dsv4_h2d_bytes={} dsv4_gemv_us={:.0} dsv4_silu_us={:.0} dsv4_attn_us={:.0} dsv4_norm_us={:.0} dsv4_d2h_us={:.0} dsv4_pack_us={:.0} dsv4_hc_us={:.0} dsv4_router_us={:.0} dsv4_embed_us={:.0} q8_gemm_us={:.0} other_gpu_us={:.0} activation_h2d_bytes={} activation_d2h_bytes={} expert_lru_hits={} expert_lru_misses={} expert_lru_fills={} expert_prefill_overlaps={} draft_expert_prefetches={}",
            chatml,
            prompt.len(),
            out.token_ids,
            out.text,
            stats.ttft_ms_last,
            stats.itl_ms_ema,
            stats.backend,
            stats.gpu_forwards,
            stats.target_gpu_forwards,
            stats.degraded_events,
            stats.dsv4_h2d_us,
            stats.dsv4_h2d_bytes,
            stats.dsv4_gemv_us,
            stats.dsv4_silu_us,
            stats.dsv4_attn_us,
            stats.dsv4_norm_us,
            stats.dsv4_d2h_us,
            stats.dsv4_pack_us,
            stats.dsv4_hc_us,
            stats.dsv4_router_us,
            stats.dsv4_embed_us,
            stats.q8_gemm_us,
            stats.other_gpu_us,
            stats.activation_h2d_bytes,
            stats.activation_d2h_bytes,
            stats.expert_lru_hits,
            stats.expert_lru_misses,
            stats.expert_lru_fills,
            stats.expert_prefill_overlaps,
            stats.draft_expert_prefetches,
        );
    }

    #[test]
    fn tiny_dspark_sidecar_is_used_when_present() {
        let dir = std::env::temp_dir().join(format!(
            "allpaca_dspark_pair-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let target = dir.join("tiny-dsv4.gguf");
        let sidecar = dir.join("dspark-tiny.gguf");
        crate::deepseek4::write_tiny_deepseek4_gguf(&target);
        {
            use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
            use std::collections::HashMap;
            let mut meta = HashMap::new();
            meta.insert(
                "general.architecture".into(),
                GgufValue::String("dflash".into()),
            );
            meta.insert("dspark.block_size".into(), GgufValue::Uint32(2));
            meta.insert("dspark.markov_rank".into(), GgufValue::Uint32(2));
            let w1 = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
            let w2 = vec![0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0];
            write_gguf(
                &sidecar,
                &meta,
                &[
                    GgufWriteTensor {
                        name: "dspark.markov_w1.weight".into(),
                        dims: vec![2, 4],
                        data: &w1,
                    },
                    GgufWriteTensor {
                        name: "dspark.markov_w2.weight".into(),
                        dims: vec![2, 4],
                        data: &w2,
                    },
                ],
            )
            .unwrap();
        }
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(target.clone());
        cfg.speculative.enabled = true;
        cfg.speculative.providers = vec![DraftProviderKind::Dspark];
        let engine = Engine::new(cfg).expect("DSpark pair must load");
        let out = engine
            .generate(
                "hi",
                GenerationParams {
                    max_tokens: 3,
                    temperature: 0.0,
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .expect("DSpark pair must generate");
        assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(sidecar);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_lightning_shexp_long_prompt_batches_prefill() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip hip_lightning_shexp_long_prompt_batches_prefill: no HIP device");
            return;
        }
        let path = std::env::temp_dir().join(format!(
            "allpaca_nemotron_lightning_shexp_hip-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_nemotron_lightning_moe_gguf_opts(&path, true);
        let prompt = "hi hi hi hi hi hi hi hi";
        let run = |cpu_only: bool| {
            let mut cfg = Config::compiled_defaults();
            cfg.model.path = Some(path.clone());
            cfg.model.synthetic = false;
            cfg.engine.cpu_only = cpu_only;
            cfg.speculative.enabled = false;
            let engine = Engine::new(cfg).expect("Lightning shexp HIP load");
            let out = engine
                .generate(
                    prompt,
                    GenerationParams {
                        max_tokens: 4,
                        temperature: 0.0,
                        seed: Some(7),
                        ..GenerationParams::default()
                    },
                )
                .expect("Lightning shexp HIP generate");
            (engine, out)
        };
        let (cpu_engine, cpu_out) = run(true);
        drop(cpu_engine);
        let (hip_engine, hip_out) = run(false);
        let stats = hip_engine.stats();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            hip_out.token_ids, cpu_out.token_ids,
            "batched Lightning prefill must match CPU; hip={:?} cpu={:?}",
            hip_out.token_ids, cpu_out.token_ids
        );
        assert!(
            hip_out.prompt_tokens >= 8,
            "need a multi-token prompt, got {}",
            hip_out.prompt_tokens
        );
        if stats.backend == "hip" {
            assert!(
                stats.gpu_forwards < hip_out.prompt_tokens as u64,
                "hybrid batched prefill must not run one forward per prompt token; gpu_forwards={} prompt_tokens={} stats={stats:?}",
                stats.gpu_forwards,
                hip_out.prompt_tokens
            );
        }
    }

    fn write_tiny_qwen_gdn_gguf(path: &std::path::Path, w: f32) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;

        let dim = 16usize;
        let vocab = 4usize;
        let n_v = 2usize;
        let n_k = 2usize;
        let head = 8usize;
        let inner = n_v * head;
        let qkv_dim = n_k * head * 2 + inner;
        let conv_k = 2usize;
        let mut meta = HashMap::new();
        meta.insert(
            "general.architecture".into(),
            GgufValue::String("qwen35".into()),
        );
        meta.insert("qwen35.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert(
            "qwen35.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
        meta.insert("qwen35.block_count".into(), GgufValue::Uint32(2));
        meta.insert("qwen35.attention.head_count".into(), GgufValue::Uint32(2));
        meta.insert(
            "qwen35.attention.head_count_kv".into(),
            GgufValue::Uint32(1),
        );
        meta.insert(
            "qwen35.feed_forward_length".into(),
            GgufValue::Uint32((dim * 2) as u32),
        );
        meta.insert(
            "qwen35.ssm.state_size".into(),
            GgufValue::Uint32(head as u32),
        );
        meta.insert(
            "qwen35.ssm.inner_size".into(),
            GgufValue::Uint32(inner as u32),
        );
        meta.insert(
            "qwen35.ssm.time_step_rank".into(),
            GgufValue::Uint32(n_v as u32),
        );
        meta.insert(
            "qwen35.ssm.group_count".into(),
            GgufValue::Uint32(n_k as u32),
        );
        meta.insert(
            "qwen35.ssm.conv_kernel".into(),
            GgufValue::Uint32(conv_k as u32),
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
        let mut owned: Vec<(String, Vec<u64>, Vec<f32>)> = vec![(
            "token_embd.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        )];
        for i in 0..2 {
            owned.push((
                format!("blk.{i}.attn_norm.weight"),
                vec![dim as u64],
                vec![1.0; dim],
            ));
            owned.push((
                format!("blk.{i}.attn_qkv.weight"),
                vec![qkv_dim as u64, dim as u64],
                vec![w; qkv_dim * dim],
            ));
            owned.push((
                format!("blk.{i}.attn_gate.weight"),
                vec![inner as u64, dim as u64],
                vec![w; inner * dim],
            ));
            owned.push((
                format!("blk.{i}.ssm_alpha.weight"),
                vec![n_v as u64, dim as u64],
                vec![w; n_v * dim],
            ));
            owned.push((
                format!("blk.{i}.ssm_beta.weight"),
                vec![n_v as u64, dim as u64],
                vec![w; n_v * dim],
            ));
            owned.push((
                format!("blk.{i}.ssm_out.weight"),
                vec![dim as u64, inner as u64],
                vec![w; dim * inner],
            ));
            owned.push((format!("blk.{i}.ssm_a"), vec![n_v as u64], vec![-0.5; n_v]));
            owned.push((
                format!("blk.{i}.ssm_dt.bias"),
                vec![n_v as u64],
                vec![0.0; n_v],
            ));
            owned.push((
                format!("blk.{i}.ssm_norm.weight"),
                vec![head as u64],
                vec![1.0; head],
            ));
            owned.push((
                format!("blk.{i}.ssm_conv1d.weight"),
                vec![(conv_k * qkv_dim) as u64],
                vec![w; conv_k * qkv_dim],
            ));
        }
        owned.push((
            "output_norm.weight".into(),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        owned.push((
            "output.weight".into(),
            vec![vocab as u64, dim as u64],
            vec![w; vocab * dim],
        ));
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

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_tiny_gdn_gguf_rejects_upload() {
        let path = std::env::temp_dir().join("allpaca_cuda_tiny_gdn.gguf");
        write_tiny_qwen_gdn_gguf(&path, 0.02);
        let mut cfg = Config::compiled_defaults();
        cfg.model.path = Some(path.clone());
        cfg.model.synthetic = false;
        cfg.engine.cpu_only = false;
        cfg.engine.gpu_backend = "cuda".into();
        cfg.engine.determinism_seed = Some(7);
        cfg.speculative.enabled = false;
        cfg.kv.page_tokens = 8;
        let engine = match Engine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                panic!("tiny GDN GGUF must load: {e}");
            }
        };
        let stats = engine.stats();
        let _ = std::fs::remove_file(&path);
        if stats.backend != "cuda" {
            return;
        }
        assert_eq!(
            stats.weight_bytes_on_device, 0,
            "GDN/SSM must not CUDA-upload; stats={stats:?}"
        );
        assert!(
            stats.degraded_events >= 1,
            "CUDA GDN upload reject must degrade; stats={stats:?}"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_tiny_gdn_gguf_rejects_upload() {
        let path = std::env::temp_dir().join(format!(
            "allpaca_vulkan_tiny_gdn-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_qwen_gdn_gguf(&path, 0.02);
        let mut cfg = Config::compiled_defaults();
        cfg.model.path = Some(path.clone());
        cfg.model.synthetic = false;
        cfg.engine.cpu_only = false;
        cfg.engine.gpu_backend = "vulkan".into();
        cfg.engine.determinism_seed = Some(7);
        cfg.speculative.enabled = false;
        cfg.kv.page_tokens = 8;
        let engine = match Engine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                panic!("tiny GDN GGUF must load: {e}");
            }
        };
        let stats = engine.stats();
        let _ = std::fs::remove_file(&path);
        if stats.backend != "vulkan" {
            return;
        }
        assert_eq!(
            stats.weight_bytes_on_device, 0,
            "GDN/SSM must not Vulkan-upload; stats={stats:?}"
        );
        assert!(
            stats.degraded_events >= 1,
            "Vulkan GDN upload reject must degrade; stats={stats:?}"
        );
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_mamba2_matches_host_and_keeps_state_resident() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip hip_mamba2 parity: no HIP device");
            return;
        }
        use crate::ssm::{mamba2_step, Mamba2Dims, SsmLayerWeights};
        let dims = Mamba2Dims::new(2, 2, 1, 1, 1, 2).unwrap();
        let mut w_in = vec![0.0; dims.in_proj_dim * dims.d_model];
        for (row, col, value) in [
            (0, 0, 1.0),
            (1, 1, 1.0),
            (2, 0, 1.0),
            (3, 1, 1.0),
            (4, 0, 0.5),
            (5, 1, 1.0),
            (6, 0, 1.0),
        ] {
            w_in[row * dims.d_model + col] = value;
        }
        let weights = SsmLayerWeights {
            dims,
            w_in,
            w_out: vec![1.0, 0.0, 0.0, 1.0],
            a: vec![-1.0],
            d_skip: vec![0.25],
            dt_bias: vec![0.0],
            conv1d: vec![0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
            conv_bias: vec![0.0; 4],
            norm: vec![1.0; 2],
        };
        let input = [1.0f32, -0.5];
        let projected = [1.0f32, -0.5, 1.0, -0.5, 0.5, -0.5, 1.0];
        let mut host_state = vec![0.0; dims.state_floats()];
        let mut host_conv = vec![0.0; dims.conv_state_floats()];
        let mut host_out = vec![0.0; dims.d_model];
        mamba2_step(
            &input,
            &mut host_state,
            &mut host_conv,
            &weights,
            1e-5,
            &mut host_out,
        )
        .unwrap();

        let hip = allpaca_hip::HipWorkspace::new(0, 4096, 4).unwrap();
        let projection = allpaca_hip::DeviceWeights::alloc(0, projected.len() * 4).unwrap();
        let output = allpaca_hip::DeviceWeights::alloc(0, dims.d_inner * 4).unwrap();
        hip.copy_h2d_f32(projection.ptr_at(0) as *mut f32, &projected)
            .unwrap();
        let (conv, bias, dt, a, d, norm) = hip
            .ensure_mamba2_weights(
                0,
                &weights.conv1d,
                &weights.conv_bias,
                &weights.dt_bias,
                &weights.a,
                &weights.d_skip,
                &weights.norm,
            )
            .unwrap();
        let mut device_state = vec![0.0; dims.state_floats()];
        let mut device_conv = vec![0.0; dims.conv_state_floats()];
        hip.mamba2_decode_step_resident(
            77,
            0,
            &mut device_state,
            &mut device_conv,
            projection.ptr_at(0),
            conv,
            bias,
            dt,
            a,
            d,
            norm,
            output.ptr_at(0) as *mut f32,
            dims.d_inner as i32,
            dims.d_state as i32,
            dims.n_heads as i32,
            dims.n_groups as i32,
            dims.conv_kernel as i32,
            1e-5,
        )
        .unwrap();
        let mut device_out = vec![0.0; dims.d_inner];
        hip.copy_d2h_f32(output.ptr_at(0), &mut device_out).unwrap();
        hip.mamba2_sync_request_to_host(
            77,
            std::slice::from_mut(&mut device_state),
            std::slice::from_mut(&mut device_conv),
        )
        .unwrap();
        for (got, want) in device_out.iter().zip(&host_out) {
            assert!((got - want).abs() < 2e-4, "HIP={got} host={want}");
        }
        for (got, want) in device_state.iter().zip(&host_state) {
            assert!((got - want).abs() < 2e-4, "state HIP={got} host={want}");
        }
        assert_eq!(device_conv, host_conv);

        let snapshot_state = device_state.clone();
        let snapshot_conv = device_conv.clone();
        assert_eq!(hip.gdn_clone_request(77).unwrap(), 1);
        let d2h_before = hip.activation_d2h_bytes();
        hip.mamba2_decode_step_resident(
            77,
            0,
            &mut device_state,
            &mut device_conv,
            projection.ptr_at(0),
            conv,
            bias,
            dt,
            a,
            d,
            norm,
            output.ptr_at(0) as *mut f32,
            dims.d_inner as i32,
            dims.d_state as i32,
            dims.n_heads as i32,
            dims.n_groups as i32,
            dims.conv_kernel as i32,
            1e-5,
        )
        .unwrap();
        assert_eq!(
            hip.activation_d2h_bytes(),
            d2h_before,
            "resident second step must not D2H state or activations"
        );
        hip.gdn_rewind_request(77).unwrap();
        hip.mamba2_sync_request_to_host(
            77,
            std::slice::from_mut(&mut device_state),
            std::slice::from_mut(&mut device_conv),
        )
        .unwrap();
        assert_eq!(
            device_state, snapshot_state,
            "D2D rewind must restore SSD state"
        );
        assert_eq!(
            device_conv, snapshot_conv,
            "D2D rewind must restore conv history"
        );
        // Every verify clones and drops; the snapshot slots must be reused.
        hip.gdn_drop_clone(77);
        let held = hip.allocated_bytes();
        assert_eq!(hip.gdn_clone_request(77).unwrap(), 1);
        assert_eq!(hip.allocated_bytes(), held, "re-clone must not allocate");
        hip.gdn_drop_clone(77);
        hip.gdn_clear_request(77);
        assert!(hip.allocated_bytes() < held, "clear must free the snapshot");
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_nemotron_mamba2_greedy_matches_cpu() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip HIP Nemotron end-to-end: no HIP device");
            return;
        }
        let path = std::env::temp_dir().join(format!(
            "allpaca-nemotron-mamba2-parity-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_nemotron_hybrid_gguf(&path);
        let run = |cpu_only: bool| {
            let mut cfg = Config::compiled_defaults();
            cfg.model.path = Some(path.clone());
            cfg.model.synthetic = false;
            cfg.engine.cpu_only = cpu_only;
            cfg.speculative.enabled = false;
            Engine::new(cfg)
                .unwrap()
                .generate(
                    "hi",
                    GenerationParams {
                        max_tokens: 6,
                        temperature: 0.0,
                        seed: Some(7),
                        ..GenerationParams::default()
                    },
                )
                .unwrap()
                .token_ids
        };
        let cpu = run(true);
        let gpu = run(false);
        let _ = std::fs::remove_file(path);
        assert_eq!(gpu, cpu);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_tiny_gdn_partial_accept_matches_target_without_keep_replay() {
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip hip_tiny_gdn_partial_accept: no HIP device");
            return;
        }
        let path = std::env::temp_dir().join(format!(
            "allpaca_tiny_gdn_pack-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_tiny_qwen_gdn_gguf(&path, 0.02);
        let prompt = "hi hi hi";
        let n = 8usize;
        let mut greedy_cfg = Config::compiled_defaults();
        greedy_cfg.model.path = Some(path.clone());
        greedy_cfg.model.synthetic = false;
        greedy_cfg.engine.cpu_only = false;
        greedy_cfg.engine.gpu_backend = "auto".into();
        greedy_cfg.engine.determinism_seed = Some(7);
        greedy_cfg.speculative.enabled = false;
        greedy_cfg.kv.page_tokens = 32;
        let greedy_engine = match Engine::new(greedy_cfg) {
            Ok(e) => e,
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                eprintln!("skip hip_tiny_gdn_partial_accept: Engine {e}");
                return;
            }
        };
        let greedy = greedy_engine
            .generate(
                prompt,
                GenerationParams {
                    max_tokens: n,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("target-only");
        let greedy_fwd = greedy_engine.stats().target_gpu_forwards;
        drop(greedy_engine);

        let mut spec_cfg = Config::compiled_defaults();
        spec_cfg.model.path = Some(path.clone());
        spec_cfg.model.synthetic = false;
        spec_cfg.engine.cpu_only = false;
        spec_cfg.engine.gpu_backend = "auto".into();
        spec_cfg.engine.determinism_seed = Some(7);
        spec_cfg.speculative.enabled = true;
        spec_cfg.speculative.min_depth = 4;
        spec_cfg.speculative.max_depth = 4;
        spec_cfg.features.draft_model_speculation = Some(false);
        spec_cfg.features.ngram_speculation = Some(true);
        spec_cfg.kv.page_tokens = 32;
        let spec_engine = Engine::new(spec_cfg).expect("spec Engine");
        let mut handle = spec_engine
            .begin_request(
                prompt,
                GenerationParams {
                    max_tokens: n,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("begin");
        {
            let mut inner = spec_engine.lock().expect("lock");
            let active = inner
                .active
                .get(&handle.id())
                .expect("active request after prefill");
            let ctx = active.tokens.clone();
            // Target-only is all 0s; draft a matching first token and a rejected suffix.
            let mut observed = ctx.clone();
            observed.extend_from_slice(&[0, 1, 1, 1]);
            inner.ngram.observe(&observed);
            inner.draft_ahead = Some(DraftAhead {
                ctx,
                tokens: vec![0, 1, 1, 1],
            });
        }
        let mut spec_ids = Vec::new();
        let mut packs = Vec::new();
        loop {
            let before = spec_engine.stats();
            match spec_engine.step(&mut handle).expect("step") {
                allpaca_types::StepResult::Prefill { .. } => {}
                allpaca_types::StepResult::Token(ev) => spec_ids.push(ev.token_id),
                allpaca_types::StepResult::Finished(_) => break,
            }
            let after = spec_engine.stats();
            if after.linear_packed_verifies > before.linear_packed_verifies {
                packs.push((
                    after
                        .target_gpu_forwards
                        .saturating_sub(before.target_gpu_forwards),
                    after.draft_accepted.saturating_sub(before.draft_accepted),
                    after.draft_proposed.saturating_sub(before.draft_proposed),
                ));
            }
        }
        let spec = spec_engine.stats();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            spec_ids, greedy.token_ids,
            "partial accept must match target-only"
        );
        assert!(
            spec.target_gpu_forwards <= greedy_fwd.saturating_add(2),
            "partial-accept GDN must not cost a keep-prefix replay vs target-only; spec_fwd={} greedy_fwd={greedy_fwd} spec={spec:?}",
            spec.target_gpu_forwards
        );
        let (delta_fwd, dacc, dprop) = packs
            .iter()
            .copied()
            .find(|(_, dacc, dprop)| dacc < dprop)
            .expect("expected a partial-accept packed step");
        // Incremental GDN verify forwards only keep + the bonus sample.
        // Full-span + keep-prefix replay is span + keep + bonus.
        assert!(
            delta_fwd <= dacc.saturating_add(1),
            "kept-prefix GDN replay must not re-forward accepted tokens; delta_fwd={delta_fwd} dacc={dacc} dprop={dprop} packs={packs:?} spec={spec:?}"
        );
    }

    #[test]
    fn nemotron_hybrid_synthetic_generates_and_real_fp4_loads() {
        // Synthetic F32 hybrid (SSM + attention) must generate greedily past Fatal.
        {
            let path = std::env::temp_dir().join(format!(
                "allpaca_nemotron_hybrid_green-{}-{}.gguf",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            write_tiny_nemotron_hybrid_gguf(&path);
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
            cfg.speculative.enabled = false;
            cfg.weights.cache_bytes = 1 << 20;
            cfg.weights.host_bytes = 1 << 20;
            cfg.kv.nvme_bytes = 1 << 20;
            cfg.kv.nvme_path = Some(nvme_path.clone());
            let engine = Engine::new(cfg).expect("synthetic nemotron_h_moe F32 hybrid must load");
            let out = engine
                .generate(
                    "hi",
                    GenerationParams {
                        max_tokens: 4,
                        temperature: 0.0,
                        seed: Some(7),
                        ..GenerationParams::default()
                    },
                )
                .expect("synthetic hybrid generate");
            assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
            let again = engine
                .generate(
                    "hi",
                    GenerationParams {
                        max_tokens: 4,
                        temperature: 0.0,
                        seed: Some(7),
                        ..GenerationParams::default()
                    },
                )
                .unwrap();
            assert_eq!(
                out.token_ids, again.token_ids,
                "greedy hybrid path must be deterministic"
            );
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&nvme_path);
        }

        // Real ROCmFP4: type 100 no longer gates the loader. The header check is
        // cheap enough to always run; the 30B load/generate is opt-in because it
        // materializes an 18 GiB checkpoint.
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-COHERENT.gguf");
        if !path.exists() {
            eprintln!(
                "skip nemotron real GGUF: place ROCmFP4 at {}; synthetic generate already asserted",
                path.display()
            );
            return;
        }
        let report = allpaca_gguf::GgufFile::open(&path)
            .expect("mmap Nemotron header")
            .compat_report();
        assert!(
            report.ggml_type_ids.contains(&100),
            "expected ROCmFP4 tensors; ids={:?}",
            report.ggml_type_ids
        );
        assert!(
            report.reject_reason.is_none(),
            "ROCmFP4 (ggml_type_id=100) must no longer gate the loader; got {:?}",
            report.reject_reason
        );

        if std::env::var("ALLPACA_NEMOTRON").ok().as_deref() != Some("1") {
            eprintln!(
                "skip nemotron real ROCmFP4 generate: set ALLPACA_NEMOTRON=1 (loads {} GiB)",
                18
            );
            return;
        }
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: !cfg!(feature = "hip"),
            ..CliOverrides::default()
        });
        cfg.model.synthetic = false;
        cfg.model.path = Some(path);
        cfg.speculative.enabled = false;
        cfg.weights.cache_bytes = 64 << 20;
        cfg.weights.host_bytes = 64 << 20;
        cfg.kv.nvme_bytes = 64 << 20;
        cfg.kv.nvme_path = Some(
            std::env::temp_dir().join(format!("allpaca-nemotron-real-nvme-{}", std::process::id())),
        );
        let nvme = cfg.kv.nvme_path.clone();
        let engine = Engine::new(cfg).expect("Nemotron ROCmFP4 must load once dequant lands");
        let out = engine
            .generate(
                "hi",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(7),
                    ..GenerationParams::default()
                },
            )
            .expect("Nemotron ROCmFP4 generate");
        assert!(!out.token_ids.is_empty(), "must emit tokens; got {out:?}");
        if let Some(p) = nvme {
            let _ = std::fs::remove_file(p);
        }
    }

    /// `rpc.max_seq = 0` sizes remote KV from the engine's KV budget; an
    /// explicit value is used as given.
    #[test]
    fn rpc_kv_capacity_follows_the_kv_budget() {
        let cfg = ModelConfig::tiny();
        let model = TinyModel::synthetic(cfg.clone(), 1, "synthetic");
        assert_eq!(rpc_kv_max_seq(64, 1 << 30, &model), 64);
        let per_token = (cfg.n_kv_heads * cfg.head_dim() * 2 * 4 * cfg.n_layers) as u64;
        assert_eq!(rpc_kv_max_seq(0, per_token * 50_000, &model), 50_000);
        // A tiny budget still leaves the floor.
        assert_eq!(rpc_kv_max_seq(0, 1024, &model), RPC_AUTO_MIN_SEQ);
    }

    /// A dashboard poll during a prefill (engine lock held for the whole time
    /// to first token) returns the last readings instead of hanging.
    #[test]
    fn stats_and_model_id_do_not_wait_for_a_busy_engine() {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(3),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        let engine = Engine::new(cfg).unwrap();
        engine.generate("hello", GenerationParams { max_tokens: 4, ..GenerationParams::default() }).unwrap();
        let idle = engine.stats();
        let id = engine.model_id();
        let busy = engine.inner.lock().unwrap();
        let started = std::time::Instant::now();
        let during = engine.stats();
        assert_eq!(engine.model_id(), id);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "monitoring reads waited {:?} on a busy engine",
            started.elapsed()
        );
        drop(busy);
        assert_eq!(during.model_id, idle.model_id);
        assert_eq!(during.tokens_generated, idle.tokens_generated);
        assert_eq!(during.draft_depth, idle.draft_depth);
    }

    /// A blocking reply is the same string a streamed reply adds up to, so
    /// the text does not depend on which path served the request (special
    /// pieces the model emits are kept in both).
    #[test]
    fn generate_text_equals_streamed_text() {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(3),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        let engine = Engine::new(cfg).unwrap();
        let params = GenerationParams {
            max_tokens: 48,
            temperature: 0.0,
            seed: Some(3),
            ..GenerationParams::default()
        };
        let out = engine.generate("hello world", params.clone()).unwrap();
        let mut handle = engine.begin_request("hello world", params).unwrap();
        let (mut streamed, mut ids) = (String::new(), Vec::new());
        engine
            .drive_request(&mut handle, |ev| {
                streamed.push_str(&ev.text);
                ids.push(ev.token_id);
                ev.finish_reason
            })
            .unwrap();
        assert_eq!(ids, out.token_ids);
        assert_eq!(out.text, streamed);
        assert!(!out.text.is_empty());
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
            (
                "token_embd.weight",
                vec![vocab as u64, dim as u64],
                &embd,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.attn_norm.weight",
                vec![dim as u64],
                &ones,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.attn_q.weight",
                vec![dim as u64, dim as u64],
                &q,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.attn_k.weight",
                vec![kv_dim as u64, dim as u64],
                &k,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.attn_v.weight",
                vec![kv_dim as u64, dim as u64],
                &v,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.attn_output.weight",
                vec![dim as u64, dim as u64],
                &o,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.ffn_norm.weight",
                vec![dim as u64],
                &ones,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.ffn_gate.weight",
                vec![ff as u64, dim as u64],
                &gate,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.ffn_up.weight",
                vec![ff as u64, dim as u64],
                &up,
                SafetensorsDtype::F32,
            ),
            (
                "blk.0.ffn_down.weight",
                vec![dim as u64, ff as u64],
                &down,
                SafetensorsDtype::F32,
            ),
            (
                "output_norm.weight",
                vec![dim as u64],
                &ones,
                SafetensorsDtype::F32,
            ),
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

    fn write_tiny_hf_safetensors_checkpoint(dir: &std::path::Path) -> std::path::PathBuf {
        use allpaca_safetensors::{write_safetensors, SafetensorsDtype, SafetensorsWriteTensor};

        std::fs::create_dir_all(dir).unwrap();
        let dim = 8usize;
        let vocab = 4usize;
        let ff = 16usize;
        let kv_dim = 2usize;
        std::fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"architectures":["LlamaForCausalLM"],"model_type":"llama","vocab_size":{vocab},"hidden_size":{dim},"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":1,"intermediate_size":{ff},"rms_norm_eps":1e-5,"rope_theta":10000.0,"hidden_act":"silu","tie_word_embeddings":true}}"#
            ),
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{
                "version": "1.0",
                "added_tokens": [
                    {"id": 0, "special": true, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                    {"id": 1, "special": true, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false},
                    {"id": 2, "special": true, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}
                ],
                "pre_tokenizer": {"type": "Whitespace"},
                "model": {
                    "type": "WordLevel",
                    "unk_token": "<unk>",
                    "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}
                }
            }"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": "<s>", "eos_token": "</s>", "add_bos_token": false}"#,
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
            (
                "model.embed_tokens.weight",
                vec![vocab as u64, dim as u64],
                &embd,
            ),
            (
                "model.layers.0.input_layernorm.weight",
                vec![dim as u64],
                &ones,
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![dim as u64, dim as u64],
                &q,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![kv_dim as u64, dim as u64],
                &k,
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                vec![kv_dim as u64, dim as u64],
                &v,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                vec![dim as u64, dim as u64],
                &o,
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                vec![dim as u64],
                &ones,
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![ff as u64, dim as u64],
                &gate,
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                vec![ff as u64, dim as u64],
                &up,
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![dim as u64, ff as u64],
                &down,
            ),
            ("model.norm.weight", vec![dim as u64], &ones),
        ];
        let views: Vec<SafetensorsWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| SafetensorsWriteTensor {
                name: (*name).into(),
                dims: dims.clone(),
                dtype: SafetensorsDtype::BF16,
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

    #[test]
    fn engine_generate_from_hf_named_bf16_safetensors_dir() {
        let dir = std::env::temp_dir().join("allpaca_tiny_hf_safetensors_ckpt");
        let _ = std::fs::remove_dir_all(&dir);
        let _path = write_tiny_hf_safetensors_checkpoint(&dir);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(dir.clone());
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("HuggingFace SafeTensors dir must load");
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
            .expect("generate must work on HuggingFace BF16 SafeTensors");
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
        let header =
            r#"{"token_embd.weight":{"dtype":"I32","shape":[4,8],"data_offsets":[0,128]}}"#;
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
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
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
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
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
        write_tiny_uniform_kquant_gguf(
            &path,
            allpaca_gguf::GgmlType::F16,
            allpaca_gguf::quantize_f16,
        );
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
        write_tiny_uniform_kquant_gguf(
            &path,
            allpaca_gguf::GgmlType::Q6_K,
            allpaca_gguf::quantize_q6_k,
        );
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
        write_tiny_uniform_kquant_gguf(
            &path,
            allpaca_gguf::GgmlType::Q5_K,
            allpaca_gguf::quantize_q5_k,
        );
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
        meta.insert(
            "llama.embedding_length".into(),
            GgufValue::Uint32(dim as u32),
        );
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
    fn engine_hybrid_packed_verify_partial_accept_matches_target_only() {
        let target = std::env::temp_dir().join("allpaca_pack_hybrid_tgt.gguf");
        let draft = std::env::temp_dir().join("allpaca_pack_hybrid_dft.gguf");
        write_tiny_nemotron_hybrid_gguf_ssm_d(&target, 0.25);
        write_tiny_llama_gguf(&draft, 0.01, 2.0);
        let params = GenerationParams {
            max_tokens: 16,
            temperature: 0.0,
            seed: Some(1),
            ..GenerationParams::default()
        };
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
        let engine = Engine::new(cfg).expect("hybrid target+draft must load");
        let mut handle = engine
            .begin_request("hi", params.clone())
            .expect("begin hybrid spec");
        loop {
            match engine.step(&mut handle).expect("spec step") {
                StepResult::Prefill { .. } => {}
                StepResult::Token(_) => {
                    if engine.stats().linear_packed_verifies > 0 {
                        break;
                    }
                }
                StepResult::Finished(_) => break,
            }
        }
        let stats = engine.stats();
        assert!(
            stats.linear_packed_verifies > 0,
            "hybrid first draft token must match often enough to pack; stats={stats:?}"
        );
        let (spec_tokens, spec_ssm) = {
            let g = engine.lock().expect("lock spec");
            let active = g.active.get(&handle.id()).expect("active spec request");
            let ssm = active
                .recurrent
                .as_ref()
                .map(|r| r.ssm_h.clone())
                .unwrap_or_default();
            (active.tokens.clone(), ssm)
        };
        let mut plain = Config::compiled_defaults();
        plain.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(false),
            ..CliOverrides::default()
        });
        plain.model.path = Some(target.clone());
        plain.kv.page_tokens = 8;
        let plain_engine = Engine::new(plain).expect("plain hybrid target must load");
        let mut plain_handle = plain_engine
            .begin_request("hi", params)
            .expect("begin plain hybrid");
        loop {
            match plain_engine.step(&mut plain_handle).expect("plain step") {
                StepResult::Prefill { .. } => {}
                StepResult::Token(_) | StepResult::Finished(_) => {
                    let n = {
                        let g = plain_engine.lock().expect("lock plain");
                        g.active
                            .get(&plain_handle.id())
                            .map(|a| a.tokens.len())
                            .unwrap_or(0)
                    };
                    if n >= spec_tokens.len() {
                        break;
                    }
                }
            }
        }
        let (plain_tokens, plain_ssm) = {
            let g = plain_engine.lock().expect("lock plain inner");
            let active = g
                .active
                .get(&plain_handle.id())
                .expect("active plain request");
            let ssm = active
                .recurrent
                .as_ref()
                .map(|r| r.ssm_h.clone())
                .unwrap_or_default();
            (active.tokens.clone(), ssm)
        };
        assert_eq!(
            spec_tokens, plain_tokens,
            "partial-accept packed verify must keep Target Model tokens"
        );
        assert_eq!(
            spec_ssm, plain_ssm,
            "packed verify must rewind recurrent SSM to the accepted keep, not the rejected suffix"
        );
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(draft);
    }

    #[test]
    fn engine_mtp_draft_loads_from_target_gguf_when_provider_set() {
        let path = std::env::temp_dir().join("allpaca_tiny_mtp_target.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("MTP target GGUF must load");
        assert_eq!(engine.stats().degraded_events, 0);
        {
            let g = engine.lock().expect("lock");
            let draft = g.draft_model.as_ref().expect("MTP draft attached");
            assert_eq!(draft.config.n_layers, 1);
            assert_eq!(draft.config.gguf_layer_base, 1);
            assert!(draft.name.contains("mtp"));
        }
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
            .expect("MTP speculative generate");
        assert!(!out.token_ids.is_empty());
        assert!(
            engine.stats().draft_tokens_copied > 0 || engine.stats().draft_gpu_forwards > 0,
            "MTP draft must propose tokens; stats={:?}",
            engine.stats()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_tree_plus_mtp_builds_forest_and_generates() {
        let path = std::env::temp_dir().join("allpaca_tree_mtp.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(3),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.tree.enabled = true;
        cfg.speculative.tree.max_nodes = 8;
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 2;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(true);
        cfg.features.tree_speculation = Some(true);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("tree+MTP GGUF must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 6,
                    temperature: 0.0,
                    seed: Some(3),
                    ..GenerationParams::default()
                },
            )
            .expect("tree+MTP generate");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.tree_nodes > 0,
            "Token Forest must build when tree+MTP are both on; stats={stats:?}"
        );
        assert!(
            stats.tree_path_valid,
            "accepted tree path must be valid; stats={stats:?}"
        );
        assert!(
            stats.draft_proposed > 0,
            "MTP/tree must propose draft nodes; stats={stats:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// Tree verify must keep MTP process KV in lockstep (no skips) and still
    /// branch from the MTP draft when ngram is off.
    #[test]
    fn engine_tree_plus_mtp_keeps_process_kv_without_ngram() {
        let path = std::env::temp_dir().join("allpaca_tree_mtp_process.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(5),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.tree.enabled = true;
        cfg.speculative.tree.max_nodes = 8;
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 2;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(true);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("tree+MTP must load");
        let out = engine
            .generate(
                "hello",
                GenerationParams {
                    max_tokens: 6,
                    temperature: 0.0,
                    seed: Some(5),
                    ..GenerationParams::default()
                },
            )
            .expect("tree+MTP generate without ngram");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.tree_nodes > 0,
            "MTP draft alone must populate the Token Forest; stats={stats:?}"
        );
        assert_eq!(
            stats.mtp_process_skips, 0,
            "tree accept path must keep MTP process KV sequential; stats={stats:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// Productize tree+MTP: after forest verify, overlap the next MTP draft
    /// span the same way linear packed verify does.
    #[test]
    fn engine_tree_plus_mtp_reuses_draft_ahead() {
        let path = std::env::temp_dir().join("allpaca_tree_mtp_ahead.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.tree.enabled = true;
        cfg.speculative.tree.max_nodes = 8;
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 2;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(true);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("tree+MTP must load");
        let mut handle = engine
            .begin_request(
                "hello",
                GenerationParams {
                    max_tokens: 12,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("begin tree+MTP request");
        loop {
            match engine.step(&mut handle).expect("step") {
                StepResult::Prefill { .. } => {}
                StepResult::Token(_) => {
                    if engine.stats().draft_ahead_hits > 0 {
                        break;
                    }
                }
                StepResult::Finished(_) => break,
            }
        }
        let stats = engine.stats();
        assert!(stats.tree_nodes > 0, "tree path must run; stats={stats:?}");
        assert!(
            stats.draft_ahead_hits > 0,
            "tree+MTP must reuse overlapped MTP drafts; stats={stats:?}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_mtp_packed_verify_ahead_uses_warm_process_kv() {
        let path = std::env::temp_dir().join("allpaca_mtp_ahead_warm.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 2;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("MTP target GGUF must load");
        let mut handle = engine
            .begin_request(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("begin MTP request");
        loop {
            match engine.step(&mut handle).expect("step") {
                StepResult::Prefill { .. } => {}
                StepResult::Token(_) => {
                    if engine.stats().linear_packed_verifies > 0 {
                        break;
                    }
                }
                StepResult::Finished(_) => break,
            }
        }
        let stats = engine.stats();
        assert!(
            stats.linear_packed_verifies > 0,
            "MTP must pack a linear draft span; stats={stats:?}"
        );
        let depth = 2u64;
        let g = engine.lock().expect("lock");
        let ctx = g
            .active
            .get(&handle.id())
            .expect("active request")
            .tokens
            .clone();
        assert!(
            g.draft_kv.seq_matches(&ctx),
            "process KV must stay warm after packed verify; seq_len={} ctx_len={}",
            g.draft_kv.seq_len(),
            ctx.len()
        );
        let processed = g.draft_kv.seq_len() as u64;
        let extras = g.draft_kv.forwards.saturating_sub(processed);
        // First propose (1+depth) on live process KV. Next-span ahead runs on a snapshot.
        assert!(
            extras >= 1 + depth,
            "packed-verify must keep the first propose on warm process KV (1 seed + {depth} drafts); forwards={} seq={} extras={extras}",
            g.draft_kv.forwards,
            processed
        );
        let ahead = g
            .draft_ahead
            .as_ref()
            .expect("packed verify must queue next-span tokens");
        assert_eq!(ahead.ctx, ctx, "ahead ctx must match actual process KV");
        assert!(!ahead.tokens.is_empty());
        let last_hidden = g
            .active
            .get(&handle.id())
            .expect("active request")
            .last_hidden
            .clone();
        assert_eq!(
            g.draft_kv.h_nextn, last_hidden,
            "ahead propose is speculative; process KV must keep target last_hidden as h_nextn"
        );
        assert!(
            last_hidden.iter().any(|x| *x != 0.0),
            "last accepted output_norm must not be an all-zero rejected suffix"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_mtp_packed_verify_ahead_proposes_from_process_kv_snapshot() {
        let path = std::env::temp_dir().join("allpaca_mtp_ahead_snap.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 2;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("MTP target GGUF must load");
        let mut handle = engine
            .begin_request(
                "hello",
                GenerationParams {
                    max_tokens: 8,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("begin MTP request");
        let mut proposed0 = 0u64;
        let mut accepted0 = 0u64;
        loop {
            let p0 = engine.stats().draft_proposed;
            let a0 = engine.stats().draft_accepted;
            match engine.step(&mut handle).expect("step") {
                StepResult::Prefill { .. } => {}
                StepResult::Token(_) => {
                    if engine.stats().linear_packed_verifies > 0 {
                        proposed0 = p0;
                        accepted0 = a0;
                        break;
                    }
                }
                StepResult::Finished(_) => break,
            }
        }
        let stats = engine.stats();
        assert!(
            stats.linear_packed_verifies > 0,
            "MTP must pack a linear draft span; stats={stats:?}"
        );
        let depth = 2u64;
        let g = engine.lock().expect("lock");
        let ctx = g
            .active
            .get(&handle.id())
            .expect("active request")
            .tokens
            .clone();
        assert!(
            g.draft_kv.seq_matches(&ctx),
            "live process KV must stay on the engine (not moved into the ahead worker); seq_len={} ctx_len={}",
            g.draft_kv.seq_len(),
            ctx.len()
        );
        let processed = g.draft_kv.seq_len() as u64;
        let extras = g.draft_kv.forwards.saturating_sub(processed);
        let ahead = g
            .draft_ahead
            .as_ref()
            .expect("packed verify must queue next-span tokens");
        assert_eq!(
            ahead.ctx, ctx,
            "queued ahead must be for the actual ctx after verify, not a discarded assumed span"
        );
        assert!(!ahead.tokens.is_empty());
        let dprop = stats.draft_proposed.saturating_sub(proposed0);
        let dacc = stats.draft_accepted.saturating_sub(accepted0);
        if dacc == dprop && dprop > 0 {
            assert!(
                extras < 2 * (1 + depth),
                "full-accept ahead must propose on a process-KV snapshot, not the live cache; forwards={} seq={} extras={extras}",
                g.draft_kv.forwards,
                processed
            );
        }
        assert!(
            extras >= 1 + depth,
            "first propose still runs on live process KV; extras={extras}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_mtp_packed_verify_partial_accept_discards_assumed_ahead() {
        let path = std::env::temp_dir().join("allpaca_mtp_ahead_partial.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 4;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("MTP target GGUF must load");
        let mut handle = engine
            .begin_request(
                "hello",
                GenerationParams {
                    max_tokens: 16,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("begin MTP request");
        loop {
            let p0 = engine.stats().draft_proposed;
            let a0 = engine.stats().draft_accepted;
            let packs0 = engine.stats().linear_packed_verifies;
            match engine.step(&mut handle).expect("step") {
                StepResult::Prefill { .. } => {}
                StepResult::Token(_) => {
                    let stats = engine.stats();
                    if stats.linear_packed_verifies > packs0 {
                        let dprop = stats.draft_proposed.saturating_sub(p0);
                        let dacc = stats.draft_accepted.saturating_sub(a0);
                        let g = engine.lock().expect("lock");
                        let Some(active) = g.active.get(&handle.id()) else {
                            break;
                        };
                        let ctx = active.tokens.clone();
                        assert!(
                            g.draft_kv.seq_matches(&ctx),
                            "partial accept must not apply the ahead snapshot KV; seq_len={} ctx_len={}",
                            g.draft_kv.seq_len(),
                            ctx.len()
                        );
                        if dacc < dprop {
                            if let Some(ahead) = g.draft_ahead.as_ref() {
                                assert_eq!(
                                    ahead.ctx, ctx,
                                    "partial-accept ahead must be rebuilt for actual_ctx, not the assumed full span"
                                );
                            }
                        }
                    }
                }
                StepResult::Finished(_) => break,
            }
        }
        assert_eq!(
            engine.stats().mtp_process_skips,
            0,
            "partial accept must rebuild process KV without skips; stats={:?}",
            engine.stats()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_mtp_packed_verify_full_accept_reuses_draft_ahead() {
        let path = std::env::temp_dir().join("allpaca_mtp_ahead_hits.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.min_depth = 2;
        cfg.speculative.max_depth = 2;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.features.tree_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("MTP target GGUF must load");
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
            .expect("MTP packed-verify generate");
        assert!(!out.token_ids.is_empty());
        let stats = engine.stats();
        assert!(
            stats.linear_packed_verifies > 0,
            "MTP must pack a linear draft span; stats={stats:?}"
        );
        assert_eq!(
            stats.mtp_process_skips, 0,
            "process KV must stay aligned; stats={stats:?}"
        );
        assert!(
            stats.draft_ahead_hits > 0,
            "full-accept packed verify must reuse next-span ahead tokens (ctx == actual_ctx); gen={} acceptance={:.3} stats={stats:?}",
            out.token_ids.len(),
            stats.acceptance_rate
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_mtp_process_desync_increments_skip_count() {
        let path = std::env::temp_dir().join("allpaca_mtp_process_skips.gguf");
        write_tiny_mtp_gguf(&path);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            speculative: Some(true),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(path.clone());
        cfg.model.draft_path = None;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.kv.page_tokens = 8;
        let engine = Engine::new(cfg).expect("MTP target GGUF must load");
        let handle = engine
            .begin_request(
                "hello",
                GenerationParams {
                    max_tokens: 2,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("begin MTP request");
        assert_eq!(
            engine.stats().mtp_process_skips,
            0,
            "matching process() during prefill must not skip; stats={:?}",
            engine.stats()
        );
        let (token, dim) = {
            let g = engine.lock().expect("lock");
            let req = g.active.get(&handle.id()).expect("active");
            let token = *req.tokens.last().expect("prompt token");
            let dim = g.draft_model.as_ref().expect("MTP draft").config.dim;
            (token, dim)
        };
        let h = vec![0.0f32; dim];
        {
            let mut g = engine.lock().expect("lock");
            let pos = g.draft_kv.seq_len();
            mtp_process_token(&mut g, token, pos, &h, None, engine.metrics())
                .expect("matching process");
        }
        assert_eq!(
            engine.stats().mtp_process_skips,
            0,
            "process() at seq_len must not skip; stats={:?}",
            engine.stats()
        );
        {
            let mut g = engine.lock().expect("lock");
            let pos = g.draft_kv.seq_len().saturating_add(3);
            mtp_process_token(&mut g, token, pos, &h, None, engine.metrics())
                .expect("desync process returns Ok");
        }
        assert_eq!(
            engine.stats().mtp_process_skips,
            1,
            "process() with seq_len != pos must increment mtp_process_skips; stats={:?}",
            engine.stats()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mtp_quality_gate_rejects_empty_and_newline_only_text() {
        assert!(
            crate::mtp_text_is_garbled(""),
            "empty decode text is garbled"
        );
        assert!(
            crate::mtp_text_is_garbled("\n\n"),
            "newline-only decode text is garbled"
        );
        assert!(
            crate::mtp_text_is_garbled(" \n \t"),
            "whitespace-only decode text is garbled"
        );
        assert!(
            !crate::mtp_text_is_garbled("Hello world"),
            "visible text must pass the quality gate"
        );
    }

    #[test]
    fn mtp_quality_gate_rejects_dropped_word_soak_prefix() {
        const POST_BOS_SOAK: &str = "The wants a 404-word essay why \"hello world\" matter for to code Let me about what to cover:1. significance of world programs and their importance in to code";
        const PRE_BOS_SOAK: &str = "user wants detailed 00-word about why world programs for learning code. me write complete essay Let me words carefully hit around 00 words";
        assert!(
            crate::mtp_text_is_garbled(POST_BOS_SOAK),
            "post-BOS overnight prefix is dropped-word junk"
        );
        assert!(
            crate::mtp_text_is_garbled(PRE_BOS_SOAK),
            "pre-BOS overnight prefix (00-word) is dropped-word junk"
        );
        assert!(
            !crate::mtp_text_is_garbled(
                "The user wants a 400-word essay about why hello world programs matter for learning to code."
            ),
            "a complete English sentence must still pass the quality gate"
        );
    }

    #[cfg(feature = "hip")]
    fn gated_qwen_gguf_path() -> Option<std::path::PathBuf> {
        for key in ["ALLPACA_ROCMFP4_GGUF", "ALLPACA_QWEN_GGUF"] {
            if let Ok(raw) = std::env::var(key) {
                let path = std::path::PathBuf::from(raw);
                if path.is_file() {
                    return Some(path);
                }
            }
        }
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let qwen = dir
            .join("Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF")
            .join("Qwen3.8-27B-Q4_0_ROCMFP4_COHERENT.gguf");
        qwen.is_file().then_some(qwen)
    }

    #[cfg(feature = "hip")]
    fn llama_cli_bin() -> Option<std::path::PathBuf> {
        if let Ok(raw) = std::env::var("ALLPACA_LLAMA_CLI") {
            let path = std::path::PathBuf::from(raw);
            if path.is_file() {
                return Some(path);
            }
        }
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let nearby = [
            "ROCmFPX/build/bin/llama-cli.exe",
            "ROCmFPX-build/bin/llama-cli.exe",
            "ROCmFPX/build/bin/llama-cli",
            "ROCmFPX-build/bin/llama-cli",
        ];
        for rel in nearby {
            let path = root.join(rel);
            if path.is_file() {
                return Some(path);
            }
        }
        let finder = if cfg!(windows) { "where" } else { "which" };
        let out = std::process::Command::new(finder)
            .arg("llama-cli")
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let line = String::from_utf8_lossy(&out.stdout);
        let path = std::path::PathBuf::from(line.lines().next()?.trim());
        path.is_file().then_some(path)
    }

    #[cfg(feature = "hip")]
    #[test]
    fn llama_cli_bin_discovers_sibling_rocmfpx_when_present() {
        if std::env::var("ALLPACA_LLAMA_CLI").is_ok() {
            eprintln!("skip llama_cli discovery: ALLPACA_LLAMA_CLI is set");
            return;
        }
        let sibling = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../ROCmFPX/build/bin/llama-cli.exe");
        if !sibling.is_file() {
            eprintln!("skip llama_cli discovery: no {}", sibling.display());
            return;
        }
        let found = llama_cli_bin().expect("must discover sibling ROCmFPX llama-cli");
        assert_eq!(
            found.canonicalize().unwrap_or(found),
            sibling.canonicalize().unwrap_or(sibling),
            "ROCmFPX llama-cli must win over stock PATH/build_hip binaries"
        );
    }

    #[test]
    fn mtp_llamacpp_draft_id_parser_reads_draft_token_line() {
        let log = "sampled token 1\ndraft tokens: 151644 8948\ndraft acceptance rate = 0.77 (10 accepted / 13 generated)\n";
        assert_eq!(parse_llamacpp_draft_ids(log), Some(vec![151644, 8948]));
    }

    #[test]
    fn mtp_llamacpp_draft_id_parser_reads_rocmfpx_candidate_lines() {
        let log = "\
0.21.602.640 I common_speculative_state_draft_mtp: - n_max=2, n_min=0, p_min=0.75, n_embd=5120, backend_sampling=1
0.22.900.000 D  - seq_id 0, draft candidate   0, pos   0:  151644 (   0.910) '<|im_start|>'
0.22.900.001 D  - seq_id 0, draft candidate   1, pos   0:    8948 (   0.020) 'Hello'
0.22.900.002 D  - seq_id 0, draft candidate   0, pos   1:    8948 (   0.880) 'Hello'
draft acceptance rate = 0.77 (10 accepted / 13 generated)
";
        assert_eq!(
            parse_llamacpp_draft_ids(log),
            Some(vec![151644, 8948]),
            "parser must take top (k=0) ROCmFPX draft candidates, not init-line digits"
        );
    }

    #[test]
    fn mtp_llamacpp_draft_id_parser_keeps_first_draft_span_only() {
        let log = "\
0.18.273.744 D  - seq_id 0, draft candidate   0, pos   0:    271 (   0.745) '\\n\\n'
0.18.279.870 D  - seq_id 0, draft candidate   0, pos   1: 248069 (   1.000) '</think>'
0.18.528.920 D  - seq_id 0, draft candidate   0, pos   0:    760 (   0.994) 'The'
";
        assert_eq!(
            parse_llamacpp_draft_ids(log),
            Some(vec![271, 248069]),
            "later llama.cpp draft() calls must not mix into the first-step span"
        );
    }

    fn parse_llamacpp_draft_ids(log: &str) -> Option<Vec<u32>> {
        let mut by_pos: Vec<(u32, u32)> = Vec::new();
        for line in log.lines() {
            let lower = line.to_ascii_lowercase();
            if !lower.contains("draft") || lower.contains("acceptance") {
                continue;
            }
            if let Some(ids) = parse_llamacpp_draft_candidate_id(&lower) {
                if ids.0 == 0 && !by_pos.is_empty() {
                    break;
                }
                by_pos.push(ids);
                continue;
            }
            if !by_pos.is_empty() {
                continue;
            }
            if !lower.contains("token") {
                continue;
            }
            if lower.contains("specul") || lower.contains("n_max") || lower.contains("n_embd") {
                continue;
            }
            let ids: Vec<u32> = line
                .split(|c: char| !c.is_ascii_digit())
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.parse().ok())
                .collect();
            if !ids.is_empty() {
                return Some(ids);
            }
        }
        if by_pos.is_empty() {
            return None;
        }
        by_pos.sort_by_key(|(pos, _)| *pos);
        Some(by_pos.into_iter().map(|(_, id)| id).collect())
    }

    fn parse_llamacpp_draft_candidate_id(lower: &str) -> Option<(u32, u32)> {
        if !lower.contains("draft candidate") {
            return None;
        }
        let after = lower.split("draft candidate").nth(1)?;
        let mut nums = after
            .split(|c: char| !c.is_ascii_digit())
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<u32>().ok());
        let rank = nums.next()?;
        if rank != 0 {
            return None;
        }
        let pos = nums.next()?;
        let id = nums.next()?;
        Some((pos, id))
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_qwen_mtp_first_draft_tokens_match_llamacpp_when_opt_in() {
        if std::env::var("ALLPACA_QWEN").ok().as_deref() != Some("1") {
            eprintln!("skip llama.cpp MTP parity: set ALLPACA_QWEN=1");
            return;
        }
        let Some(gguf) = gated_qwen_gguf_path() else {
            eprintln!("skip llama.cpp MTP parity: no Qwen GGUF");
            return;
        };
        let Ok(log_path) = std::env::var("ALLPACA_LLAMA_DRAFT_LOG") else {
            eprintln!(
                "skip llama.cpp MTP parity: set ALLPACA_LLAMA_DRAFT_LOG to a llama-cli log from a separate process (HIP tests must not spawn llama-cli)"
            );
            return;
        };
        let prompt = std::env::var("ALLPACA_QWEN_PROMPT").unwrap_or_else(|_| {
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n".into()
        });
        let depth = 2usize;
        eprintln!("llama.cpp MTP parity reading {log_path}");
        let log =
            std::fs::read_to_string(&log_path).unwrap_or_else(|e| panic!("read {log_path}: {e}"));
        let llama_drafts = parse_llamacpp_draft_ids(&log).unwrap_or_else(|| {
            let tail: Vec<&str> = log.lines().rev().take(80).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            panic!(
                "could not parse draft token ids from {log_path} tail:\n{}",
                tail.join("\n")
            )
        });
        eprintln!("llama.cpp first-step MTP drafts={llama_drafts:?}");
        if allpaca_hip::device_count() == 0 {
            eprintln!("skip llama.cpp MTP parity: no HIP device");
            return;
        }
        let mut cfg = Config::compiled_defaults();
        cfg.model.path = Some(gguf.clone());
        cfg.model.synthetic = false;
        cfg.engine.cpu_only = false;
        cfg.engine.gpu_backend = "auto".into();
        cfg.speculative.enabled = true;
        cfg.speculative.providers = vec![DraftProviderKind::Mtp];
        cfg.speculative.min_depth = depth;
        cfg.speculative.max_depth = depth;
        cfg.features.draft_model_speculation = Some(true);
        cfg.features.ngram_speculation = Some(false);
        cfg.kv.vram_bytes = 256 * 1024 * 1024;
        cfg.kv.dram_bytes = 256 * 1024 * 1024;
        cfg.kv.page_tokens = 128;
        let engine = Engine::new(cfg).expect("Qwen MTP Engine");
        let mut handle = engine
            .begin_request(
                &prompt,
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .expect("begin Qwen request");
        loop {
            let prefill_done = {
                let g = engine.lock().expect("lock");
                let req = g.active.get(&handle.id()).expect("active");
                req.prefill_pos >= req.prompt_len
            };
            if prefill_done {
                break;
            }
            match engine.step(&mut handle).expect("prefill step") {
                StepResult::Prefill { .. } => {}
                other => panic!("expected remaining prefill, got {other:?}"),
            }
        }
        let prompt_len = {
            let g = engine.lock().expect("lock");
            g.active.get(&handle.id()).expect("active").prompt_len
        };
        match engine.step(&mut handle).expect("first decode step") {
            StepResult::Token(ev) => {
                eprintln!(
                    "Allpaca first decode token={} text={:?} prompt_len={prompt_len}",
                    ev.token_id, ev.text
                );
            }
            other => panic!("expected first decode token after prefill, got {other:?}"),
        }
        let allpaca_drafts = {
            let mut g = engine.lock().expect("lock");
            let req = g.active.get(&handle.id()).expect("active");
            let mut ctx = req.tokens.clone();
            // llama.cpp first MTP span is after the first target sample, not after
            // packed verify may have already appended extra accepted drafts.
            ctx.truncate(prompt_len.saturating_add(1));
            let model = (**g.draft_model.as_ref().expect("MTP draft")).clone();
            let kv_dim = model.config.n_kv_heads * model.config.head_dim();
            g.draft_kv.truncate(ctx.len(), kv_dim);
            eprintln!(
                "Allpaca MTP ctx n={} last={:?}",
                ctx.len(),
                ctx.iter().rev().take(6).copied().collect::<Vec<_>>()
            );
            assert!(
                g.draft_kv.seq_matches(&ctx),
                "draft KV must match prompt+first-token prefix before first-step propose"
            );
            let resident = g.draft_resident.clone();
            let mut provider = crate::draft::DraftModelProvider::new(model);
            provider
                .propose_on(
                    &ctx,
                    depth,
                    0.0,
                    &mut g.draft_kv,
                    engine.draft_hip.as_deref(),
                    resident.as_deref(),
                    Some(engine.metrics()),
                )
                .expect("Allpaca first-step MTP propose")
        };
        eprintln!("Allpaca first-step MTP drafts={allpaca_drafts:?}");
        assert!(
            !allpaca_drafts.is_empty(),
            "Allpaca first-step MTP drafts must be non-empty"
        );
        assert_eq!(
            allpaca_drafts, llama_drafts,
            "first-step MTP draft token ids must match llama.cpp"
        );
    }

    #[test]
    fn mtp_eh_proj_seed_diverges_from_plain_token_embed() {
        let path = std::env::temp_dir().join("allpaca_mtp_eh_proj.gguf");
        write_tiny_mtp_gguf(&path);
        let file = allpaca_gguf::GgufFile::open(&path).expect("open tiny MTP");
        let mtp = TinyModel::from_gguf_mtp(&file).expect("from_gguf_mtp");
        assert!(
            mtp.mtp_nextn.is_some(),
            "host MTP must load nextn.eh_proj/enorm/hnorm"
        );
        let mut plain = mtp.clone();
        plain.mtp_nextn = None;
        let mut cache_m = DenseKv::default();
        let mut cache_p = DenseKv::default();
        let mut logits_m = vec![0.0f32; mtp.config.vocab_size];
        let mut logits_p = vec![0.0f32; plain.config.vocab_size];
        mtp.forward_dense(1, 0, &mut cache_m, &mut logits_m)
            .expect("mtp forward");
        plain
            .forward_dense(1, 0, &mut cache_p, &mut logits_p)
            .expect("plain forward");
        assert_ne!(
            logits_m, logits_p,
            "eh_proj seed must change logits vs raw token embed"
        );
        assert_eq!(cache_m.h_nextn.len(), mtp.config.dim);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mtp_eh_proj_seed_uses_h_nextn_from_target() {
        let path = std::env::temp_dir().join("allpaca_mtp_h_nextn.gguf");
        write_tiny_mtp_gguf(&path);
        let file = allpaca_gguf::GgufFile::open(&path).expect("open");
        let mtp = TinyModel::from_gguf_mtp(&file).expect("mtp");
        let dim = mtp.config.dim;
        let mut cache0 = DenseKv::default();
        cache0.h_nextn = vec![0.0f32; dim];
        let mut cache1 = DenseKv::default();
        cache1.h_nextn = vec![1.0f32; dim];
        let mut a = vec![0.0f32; mtp.config.vocab_size];
        let mut b = vec![0.0f32; mtp.config.vocab_size];
        mtp.forward_dense(1, 0, &mut cache0, &mut a).unwrap();
        mtp.forward_dense(1, 0, &mut cache1, &mut b).unwrap();
        assert_ne!(a, b, "different target h_nextn must change MTP logits");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mtp_dense_head_clears_trunk_expert_count() {
        let path = std::env::temp_dir().join("allpaca_mtp_dense_experts.gguf");
        write_tiny_mtp_gguf(&path);
        let file = allpaca_gguf::GgufFile::open(&path).expect("open");
        let mtp = TinyModel::from_gguf_mtp(&file).expect("mtp");
        assert_eq!(
            mtp.config.n_experts, 0,
            "dense MTP ffn_gate must clear trunk n_experts for device draft"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mtp_assumed_span_ahead_matches_process_from_row_hiddens() {
        let path = std::env::temp_dir().join("allpaca_mtp_assumed_rows.gguf");
        write_tiny_mtp_gguf(&path);
        let file = allpaca_gguf::GgufFile::open(&path).expect("open");
        let mtp = TinyModel::from_gguf_mtp(&file).expect("mtp");
        let dim = mtp.config.dim;
        let prefix = vec![1u32, 2];
        let span = vec![3u32, 1, 2];
        let pending_h = vec![0.25f32; dim];
        let mut row_hiddens = vec![0.0f32; span.len() * dim];
        for (i, row) in row_hiddens.chunks_mut(dim).enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = 0.1 * (i + 1) as f32 + 0.01 * j as f32;
            }
        }

        let mut warm = DenseKv::default();
        let mut logits = vec![0.0f32; mtp.config.vocab_size];
        for (pos, &tok) in prefix.iter().enumerate() {
            let h = if pos == 0 {
                vec![0.0f32; dim]
            } else {
                vec![0.05f32 * pos as f32; dim]
            };
            warm.seed_h_nextn_from_target(&h);
            mtp.forward_dense(tok, pos, &mut warm, &mut logits).unwrap();
            warm.record(tok);
        }
        assert!(warm.seq_matches(&prefix));
        assert!(warm.seq_len() > 0, "must not use empty DenseKv");

        let mut from_rows = warm.snapshot().expect("snapshot for process-from-rows");
        for (i, &token) in span.iter().enumerate() {
            let h_prev = if i == 0 {
                pending_h.as_slice()
            } else {
                let off = (i - 1) * dim;
                &row_hiddens[off..off + dim]
            };
            from_rows.seed_h_nextn_from_target(h_prev);
            mtp.forward_dense(token, prefix.len() + i, &mut from_rows, &mut logits)
                .unwrap();
            from_rows.record(token);
        }
        let mut next_ctx = prefix.clone();
        next_ctx.extend_from_slice(&span);
        assert!(from_rows.seq_matches(&next_ctx));
        let h_want = from_rows.h_nextn.clone();

        let mut assumed = warm.snapshot().expect("snapshot for assumed span");
        mtp_process_assumed_span(
            &mtp,
            &mut assumed,
            &span,
            prefix.len(),
            &pending_h,
            &row_hiddens,
            None,
            None,
            None,
        )
        .expect("assumed span");
        assert!(
            assumed.seq_matches(&next_ctx),
            "assumed span must stay a warm process KV"
        );
        assert_eq!(
            assumed.h_nextn, h_want,
            "assumed span must seed each token from target rows, not rolling MTP h_nextn"
        );
        let mut provider = crate::draft::DraftModelProvider::new(mtp.clone());
        let want = provider
            .propose_on(&next_ctx, 2, 0.0, &mut from_rows, None, None, None)
            .expect("propose from row-processed KV");
        let mut provider = crate::draft::DraftModelProvider::new(mtp);
        let got = provider
            .propose_on(&next_ctx, 2, 0.0, &mut assumed, None, None, None)
            .expect("propose from assumed span");
        assert_eq!(
            got, want,
            "full-accept ahead must match process() pairing (token_k, h_{{k-1}}) from target rows"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mtp_seed_h_nextn_marks_dirty_for_all_zero_hidden() {
        let mut cache = DenseKv::default();
        cache.seed_h_nextn_from_target(&vec![0.0f32; 8]);
        assert!(cache.h_nextn_host_dirty);
        assert_eq!(cache.h_nextn.len(), 8);
    }

    #[test]
    fn mtp_propose_warm_kv_forwards_at_n_past() {
        let path = std::env::temp_dir().join("allpaca_mtp_propose_warm.gguf");
        write_tiny_mtp_gguf(&path);
        let file = allpaca_gguf::GgufFile::open(&path).expect("open");
        let mtp = TinyModel::from_gguf_mtp(&file).expect("mtp");
        let dim = mtp.config.dim;
        let mut cache = DenseKv::default();
        let ctx = vec![1u32, 2, 3];
        // Simulate process(): (tok_i, h_{i-1}) into draft KV.
        let mut logits = vec![0.0f32; mtp.config.vocab_size];
        for (pos, &tok) in ctx.iter().enumerate() {
            let h = if pos == 0 {
                vec![0.0f32; dim]
            } else {
                vec![0.1f32 * pos as f32; dim]
            };
            cache.seed_h_nextn_from_target(&h);
            mtp.forward_dense(tok, pos, &mut cache, &mut logits)
                .unwrap();
            cache.record(tok);
        }
        assert!(cache.seq_matches(&ctx));
        let before = cache.forwards;
        cache.seed_h_nextn_from_target(&vec![0.5f32; dim]);
        let mut provider = crate::draft::DraftModelProvider::new(mtp);
        let draft = provider
            .propose_on(&ctx, 2, 0.0, &mut cache, None, None, None)
            .expect("warm propose");
        assert_eq!(draft.len(), 2);
        // Warm path: 1 seed at n_past + 2 draft steps (no full replay).
        assert_eq!(cache.forwards, before + 3);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn mtp_propose_forwards_only_last_ctx_plus_drafts() {
        let path = std::env::temp_dir().join("allpaca_mtp_propose_window.gguf");
        write_tiny_mtp_gguf(&path);
        let file = allpaca_gguf::GgufFile::open(&path).expect("open");
        let mtp = TinyModel::from_gguf_mtp(&file).expect("mtp");
        let dim = mtp.config.dim;
        let mut provider = crate::draft::DraftModelProvider::new(mtp);
        let mut cache = DenseKv::default();
        cache.seed_h_nextn_from_target(&vec![0.25f32; dim]);
        let ctx = vec![1u32, 2, 3, 1, 2];
        let draft = provider
            .propose_on(&ctx, 3, 0.0, &mut cache, None, None, None)
            .expect("mtp propose");
        assert_eq!(draft.len(), 3);
        // 1 last-ctx forward + 3 draft forwards — not ctx.len()+3.
        assert_eq!(
            cache.forwards, 4,
            "MTP must not replay full prompt; forwards={}",
            cache.forwards
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn engine_speculative_generate_from_tiny_written_draft_gguf() {
        // Two folders down from the temp folder: sidecar discovery looks beside
        // the target and at the files of the folder above, and other tests
        // write `allpaca_dspark_*.gguf` files into the temp folder.
        let dir = std::env::temp_dir().join(format!("allpaca_draft_gguf_{}", std::process::id()));
        let model_dir = dir.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let target = model_dir.join("target.gguf");
        let draft = model_dir.join("draft.gguf");
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
        let _ = std::fs::remove_dir_all(dir);
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
        let value: serde_json::Value = serde_json::from_str(&out.text).unwrap_or_else(|e| {
            panic!("schema output must parse as JSON: {e}; text={:?}", out.text)
        });
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
