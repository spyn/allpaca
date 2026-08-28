use allpaca_cpu::{attention, attention_with_mask, matmul_nt, rmsnorm, rope, silu};
use allpaca_hip::{DeviceDecode, DeviceWeights, HipWorkspace};
use crate::early_exit::EarlyExitSpec;
use crate::forest::TokenForest;
use allpaca_gguf::GgufFile;
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{HfConfig, SafetensorsFile};
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::anchors::RecurrentState;
use crate::direct_storage::{DirectStorageProbe, DirectStorageQueue};
use crate::kv_tier::KvTierManager;
use crate::micro_batch::LayerStagger;
use crate::telemetry::Metrics;
use crate::packed::HostExpertPool;
use crate::expert_batch::MicroBatchExpertUnion;
use crate::weight_set::{
    align_up, ExpertId, ExpertLru, FfnRefs, LayerFfn, MissPlan, WeightWorkingSet, DIRECT_ALIGN,
};

/// Optional decode-side hooks (layer stagger, elastic norm snapshot, NVMe→VRAM).
pub struct DeviceDecodeSide<'a> {
    pub request_id: u64,
    pub moe_norm: Option<&'a mut Vec<f32>>,
    pub layer_stagger: Option<&'a LayerStagger>,
    pub ds_probe: Option<&'a DirectStorageProbe>,
    pub ds_queue: Option<&'a DirectStorageQueue>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ff_dim: usize,
    pub n_experts: usize,
    pub n_active: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// Synthetic recurrent mix (semantic-anchor tracer). Not a KDA/FLA kernel.
    pub recurrent: bool,
}

impl ModelConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 64,
            dim: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            ff_dim: 64,
            n_experts: 0,
            n_active: 0,
            eps: 1e-5,
            rope_theta: 10_000.0,
            recurrent: false,
        }
    }

    pub fn tiny_moe() -> Self {
        Self {
            n_experts: 4,
            n_active: 2,
            ..Self::tiny()
        }
    }

    pub fn tiny_hybrid() -> Self {
        Self {
            recurrent: true,
            ..Self::tiny()
        }
    }

    pub fn draft_tiny() -> Self {
        Self {
            n_layers: 1,
            ..Self::tiny()
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let arch = file.architecture().unwrap_or("llama").to_string();
        let vocab = file
            .meta_u32(&format!("{arch}.vocab_size"))
            .or_else(|| {
                file.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(allpaca_gguf::GgufValue::as_string_array)
                    .map(|t| t.len() as u32)
            })
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing vocab_size".into()))
            })? as usize;
        let dim = file
            .meta_u32(&format!("{arch}.embedding_length"))
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("missing embedding_length".into()))
            })? as usize;
        let n_layers = file
            .meta_u32(&format!("{arch}.block_count"))
            .ok_or_else(|| AllpacaError::Fatal(FatalError::ModelLoad("missing block_count".into())))?
            as usize;
        let n_heads = file
            .meta_u32(&format!("{arch}.attention.head_count"))
            .unwrap_or(4) as usize;
        let n_kv_heads = file
            .meta_u32(&format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let ff_dim = file
            .meta_u32(&format!("{arch}.feed_forward_length"))
            .unwrap_or((dim * 4) as u32) as usize;
        let eps = file
            .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_theta = file
            .meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10_000.0);
        Ok(Self {
            vocab_size: vocab,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            n_experts: file
                .meta_u32(&format!("{arch}.expert_count"))
                .unwrap_or(0) as usize,
            n_active: file
                .meta_u32(&format!("{arch}.expert_used_count"))
                .unwrap_or(0) as usize,
            eps,
            rope_theta,
            recurrent: false,
        })
    }
}

const TENSORS_PER_LAYER: usize = 9;
/// Host-materialize F32 only for tiny GGUFs. A 7B/8B Q8 file is ~32GiB as F32.
const HOST_F32_LIMIT: u64 = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum WeightKind {
    F32,
    F16,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

#[derive(Debug, Clone, Copy)]
struct WeightRef {
    off: usize,
    n: usize,
    kind: WeightKind,
}

impl WeightRef {
    fn f32(off: usize, n: usize) -> Self {
        Self {
            off,
            n,
            kind: WeightKind::F32,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LayerOff {
    attn_norm: WeightRef,
    wq: WeightRef,
    wk: WeightRef,
    wv: WeightRef,
    wo: WeightRef,
    ffn_norm: WeightRef,
    w_gate: WeightRef,
    w_up: WeightRef,
    w_down: WeightRef,
    #[allow(dead_code)]
    w_router: WeightRef,
}

struct FfnRingState {
    bound: Vec<Option<usize>>,
    cursor: usize,
}

struct FfnRing {
    base: usize,
    slot_bytes: usize,
    state: Mutex<FfnRingState>,
}

enum DeviceFfn<'a> {
    Resident {
        ffn_norm: WeightRef,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
    Bound {
        _guard: std::sync::MutexGuard<'a, FfnRingState>,
        ffn_norm: WeightRef,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
}

impl DeviceFfn<'_> {
    fn ffn_norm(&self) -> WeightRef {
        match self {
            Self::Resident { ffn_norm, .. } | Self::Bound { ffn_norm, .. } => *ffn_norm,
        }
    }
    fn w_gate(&self) -> WeightRef {
        match self {
            Self::Resident { w_gate, .. } | Self::Bound { w_gate, .. } => *w_gate,
        }
    }
    fn w_up(&self) -> WeightRef {
        match self {
            Self::Resident { w_up, .. } | Self::Bound { w_up, .. } => *w_up,
        }
    }
    fn w_down(&self) -> WeightRef {
        match self {
            Self::Resident { w_down, .. } | Self::Bound { w_down, .. } => *w_down,
        }
    }
}

struct ExpertRingState {
    bound: Vec<Option<ExpertId>>,
    used_at: Vec<u64>,
    clock: u64,
}

struct ExpertRing {
    base: usize,
    slot_bytes: usize,
    state: Mutex<ExpertRingState>,
}

/// MoE prefill GPU double-buffer: track async layer fills.
#[derive(Default)]
struct PrefillGpuPipe {
    ready: u64,
    inflight: Option<u32>,
}

fn prefill_expert_slot(layer: u32, expert: u32, n_exp: u32, n_slots: usize) -> usize {
    let group = (layer as usize % 2).saturating_mul(n_exp as usize);
    (group + expert as usize) % n_slots.max(1)
}

enum DeviceExpert<'a> {
    Resident {
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
    Bound {
        _guard: std::sync::MutexGuard<'a, ExpertRingState>,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    },
}

impl DeviceExpert<'_> {
    fn w_gate(&self) -> WeightRef {
        match self {
            Self::Resident { w_gate, .. } | Self::Bound { w_gate, .. } => *w_gate,
        }
    }
    fn w_up(&self) -> WeightRef {
        match self {
            Self::Resident { w_up, .. } | Self::Bound { w_up, .. } => *w_up,
        }
    }
    fn w_down(&self) -> WeightRef {
        match self {
            Self::Resident { w_down, .. } | Self::Bound { w_down, .. } => *w_down,
        }
    }
}

fn slot_expert_refs(base: usize, slot_bytes: usize, slot: usize, host: &LayerFfn) -> [WeightRef; 3] {
    let mut off = base + slot * slot_bytes + host.ffn_norm.len() * 4;
    let mut next = |n: usize| {
        let r = WeightRef::f32(off, n);
        off += n * 4;
        r
    };
    [
        next(host.w_gate.len()),
        next(host.w_up.len()),
        next(host.w_down.len()),
    ]
}

fn slot_ffn_refs(base: usize, slot_bytes: usize, slot: usize, host: &LayerFfn) -> [WeightRef; 4] {
    let mut off = base + slot * slot_bytes;
    let mut next = |n: usize| {
        let r = WeightRef::f32(off, n);
        off += n * 4;
        r
    };
    [
        next(host.ffn_norm.len()),
        next(host.w_gate.len()),
        next(host.w_up.len()),
        next(host.w_down.len()),
    ]
}

/// Host-layout map onto a once-uploaded [`DeviceWeights`] blob.
pub struct ResidentModel {
    weights: DeviceWeights,
    packed_h2d: u64,
    ffn_ring: Option<FfnRing>,
    expert_offs: Vec<Vec<[WeightRef; 3]>>,
    expert_ring: Option<ExpertRing>,
    prefill_pipe: Mutex<PrefillGpuPipe>,
    embed: WeightRef,
    layers: Vec<LayerOff>,
    norm: WeightRef,
    lm_head: WeightRef,
}

impl ResidentModel {
    pub fn upload(model: &TinyModel, device: i32) -> Result<Self, AllpacaError> {
        let mut slices: Vec<&[f32]> = Vec::new();
        let mut off = 0usize;
        fn push<'a>(slices: &mut Vec<&'a [f32]>, off: &mut usize, s: &'a [f32]) -> WeightRef {
            let o = *off;
            slices.push(s);
            *off += s.len() * 4;
            WeightRef::f32(o, s.len())
        }
        let embed = push(&mut slices, &mut off, model.embed.as_slice());
        let mut layers = Vec::with_capacity(model.layers.len());
        let mut expert_offs = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let attn_norm = push(&mut slices, &mut off, layer.attn_norm.as_slice());
            let wq = push(&mut slices, &mut off, layer.wq.as_slice());
            let wk = push(&mut slices, &mut off, layer.wk.as_slice());
            let wv = push(&mut slices, &mut off, layer.wv.as_slice());
            let wo = push(&mut slices, &mut off, layer.wo.as_slice());
            let ffn_norm = if layer.ffn_norm.is_empty() {
                WeightRef::f32(0, 0)
            } else {
                push(&mut slices, &mut off, layer.ffn_norm.as_slice())
            };
            let (w_gate, w_up, w_down) = if layer.w_gate.is_empty() {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                )
            } else {
                (
                    push(&mut slices, &mut off, layer.w_gate.as_slice()),
                    push(&mut slices, &mut off, layer.w_up.as_slice()),
                    push(&mut slices, &mut off, layer.w_down.as_slice()),
                )
            };
            let w_router = if layer.w_router.is_empty() {
                WeightRef::f32(0, 0)
            } else {
                push(&mut slices, &mut off, layer.w_router.as_slice())
            };
            let mut packed_ex = Vec::with_capacity(layer.experts.len());
            if model.expert_lru.is_none() && model.expert_host.is_none() {
                for expert in &layer.experts {
                    packed_ex.push([
                        push(&mut slices, &mut off, expert.w_gate.as_slice()),
                        push(&mut slices, &mut off, expert.w_up.as_slice()),
                        push(&mut slices, &mut off, expert.w_down.as_slice()),
                    ]);
                }
            }
            expert_offs.push(packed_ex);
            layers.push(LayerOff {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
                w_router,
            });
        }
        let norm = push(&mut slices, &mut off, model.norm.as_slice());
        let lm_head = push(&mut slices, &mut off, model.lm_head.as_slice());
        let packed = off;
        let (ring_slots, slot_bytes) = model
            .working_set
            .as_ref()
            .filter(|w| w.streams())
            .map(|w| (w.plan.ring_slots.max(1), w.plan.slot_bytes))
            .unwrap_or((0, 0));
        let ffn_tail = ring_slots.saturating_mul(slot_bytes);
        let (expert_slots, expert_slot_bytes) = {
            let eb = model.expert_slot_bytes();
            if eb == 0 {
                (0, 0)
            } else if let Some(lru) = &model.expert_lru {
                (lru.n_slots(), align_up(eb, DIRECT_ALIGN))
            } else if model.expert_host.is_some() {
                (1, align_up(eb, DIRECT_ALIGN))
            } else {
                (0, 0)
            }
        };
        let expert_tail = expert_slots.saturating_mul(expert_slot_bytes);
        let tail = ffn_tail.saturating_add(expert_tail);
        let weights = DeviceWeights::upload_f32_with_tail(device, &slices, tail)?;
        let ffn_ring = if ffn_tail > 0 {
            Some(FfnRing {
                base: packed,
                slot_bytes,
                state: Mutex::new(FfnRingState {
                    bound: vec![None; ring_slots],
                    cursor: 0,
                }),
            })
        } else {
            None
        };
        let expert_ring = if expert_tail > 0 {
            Some(ExpertRing {
                base: packed + ffn_tail,
                slot_bytes: expert_slot_bytes,
                state: Mutex::new(ExpertRingState {
                    bound: vec![None; expert_slots],
                    used_at: vec![0; expert_slots],
                    clock: 0,
                }),
            })
        } else {
            None
        };
        Ok(Self {
            packed_h2d: packed as u64,
            weights,
            ffn_ring,
            expert_offs,
            expert_ring,
            prefill_pipe: Mutex::new(PrefillGpuPipe::default()),
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    fn expert_ring(&self) -> Result<&ExpertRing, AllpacaError> {
        self.expert_ring.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "streamed expert has no device slot ring".into(),
            ))
        })
    }

    fn pack_expert_slot(host: &LayerFfn, slot_bytes: usize) -> Vec<u8> {
        let mut buf = vec![0u8; slot_bytes];
        let mut off = 0usize;
        for part in [&host.ffn_norm, &host.w_gate, &host.w_up, &host.w_down] {
            let n = part.len() * 4;
            if off + n > slot_bytes {
                break;
            }
            let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr().cast::<u8>(), n) };
            buf[off..off + n].copy_from_slice(bytes);
            off += n;
        }
        buf
    }

    pub(crate) fn copy_expert_h2d(&self, dev_off: usize, host: &[u8]) -> Result<(), AllpacaError> {
        self.weights.copy_h2d(dev_off, host)
    }

    /// Contiguous bulk H2D for several experts (one DMA when slots are adjacent).
    /// When `nvme_direct` is set, tries NVMe→VRAM bind per expert before host pack.
    pub(crate) fn bulk_fill_experts(
        &self,
        pairs: &[(ExpertId, LayerFfn)],
        metrics: Option<&Metrics>,
        prefill_map: Option<(u32, u32)>,
        async_copy: bool,
        mut nvme_direct: Option<
            &mut dyn FnMut(ExpertId, usize, usize) -> Result<bool, AllpacaError>,
        >,
    ) -> Result<u64, AllpacaError> {
        if pairs.is_empty() {
            return Ok(0);
        }
        let ring = self.expert_ring()?;
        let n_slots = {
            let state = ring.state.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "device expert ring mutex poisoned".into(),
                ))
            })?;
            state.bound.len()
        };
        let mut slots: Vec<usize> = Vec::with_capacity(pairs.len());
        let mut nvme_filled: Vec<bool> = Vec::with_capacity(pairs.len());
        let mut staging: Vec<u8> = Vec::with_capacity(pairs.len() * ring.slot_bytes);
        {
            let mut state = ring.state.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "device expert ring mutex poisoned".into(),
                ))
            })?;
            state.clock = state.clock.saturating_add(1);
            for (id, host) in pairs {
                let packed = host.bytes();
                if packed > ring.slot_bytes {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "expert ({}, {}) is {packed} bytes, slot is {}",
                        id.layer, id.expert, ring.slot_bytes
                    ))));
                }
                let slot = if let Some((layer, n_exp)) = prefill_map {
                    if id.layer == layer {
                        prefill_expert_slot(layer, id.expert, n_exp, n_slots)
                    } else if let Some(pos) = state.bound.iter().position(|&b| b == Some(*id)) {
                        pos
                    } else {
                        state
                            .bound
                            .iter()
                            .position(|b| b.is_none())
                            .unwrap_or_else(|| {
                                state
                                    .used_at
                                    .iter()
                                    .enumerate()
                                    .min_by_key(|(_, t)| *t)
                                    .map(|(i, _)| i)
                                    .unwrap_or(0)
                            })
                    }
                } else if let Some(pos) = state.bound.iter().position(|&b| b == Some(*id)) {
                    pos
                } else {
                    state
                        .bound
                        .iter()
                        .position(|b| b.is_none())
                        .unwrap_or_else(|| {
                            state
                                .used_at
                                .iter()
                                .enumerate()
                                .min_by_key(|(_, t)| *t)
                                .map(|(i, _)| i)
                                .unwrap_or(0)
                        })
                };
                state.bound[slot] = Some(*id);
                state.used_at[slot] = state.clock;
                slots.push(slot);
                let dev = ring.base + slot * ring.slot_bytes;
                let direct = nvme_direct
                    .as_mut()
                    .map(|f| f(*id, dev, ring.slot_bytes))
                    .transpose()?
                    .unwrap_or(false);
                nvme_filled.push(direct);
                if !direct {
                    staging.extend_from_slice(&Self::pack_expert_slot(host, ring.slot_bytes));
                }
            }
        }
        let host_slots: Vec<usize> = slots
            .iter()
            .copied()
            .zip(nvme_filled.iter().copied())
            .filter_map(|(s, d)| (!d).then_some(s))
            .collect();
        if !host_slots.is_empty() {
            let first = *host_slots.iter().min().unwrap_or(&0);
            let contiguous = host_slots.windows(2).all(|w| w[1] == w[0] + 1)
                && host_slots.len() > 1;
            if contiguous {
                let dev_off = ring.base + first * ring.slot_bytes;
                if async_copy {
                    self.weights.copy_h2d_async(dev_off, &staging)?;
                } else {
                    self.weights.copy_h2d(dev_off, &staging)?;
                }
            } else {
                let mut staging_off = 0usize;
                for slot in &host_slots {
                    let chunk = &staging[staging_off..staging_off + ring.slot_bytes];
                    let dev = ring.base + slot * ring.slot_bytes;
                    if async_copy {
                        self.weights.copy_h2d_async(dev, chunk)?;
                    } else {
                        self.weights.copy_h2d(dev, chunk)?;
                    }
                    staging_off += ring.slot_bytes;
                }
            }
        }
        let bytes = staging.len() as u64;
        let nvme_bytes: u64 = nvme_filled
            .iter()
            .zip(pairs.iter())
            .filter(|(d, _)| **d)
            .map(|(_, (_, h))| h.bytes() as u64)
            .sum();
        let total_bytes = bytes + nvme_bytes;
        if let Some(metrics) = metrics {
            metrics
                .weight_h2d_bytes
                .fetch_add(total_bytes, Ordering::Relaxed);
            if pairs.len() > 1 {
                metrics
                    .expert_bulk_fills
                    .fetch_add(1, Ordering::Relaxed);
                metrics
                    .expert_bulk_bytes
                    .fetch_add(total_bytes, Ordering::Relaxed);
            }
        }
        Ok(total_bytes)
    }

    pub(crate) fn prefill_layer_gpu_ready(&self, layer: u32) -> bool {
        self.prefill_pipe
            .lock()
            .map(|p| (p.ready & (1u64 << (layer.min(63)))) != 0)
            .unwrap_or(false)
    }

    pub(crate) fn prefill_wait_layer(&self, layer: u32) -> Result<(), AllpacaError> {
        let wait = self
            .prefill_pipe
            .lock()
            .map(|p| p.inflight == Some(layer))
            .unwrap_or(false);
        if wait {
            allpaca_hip::stream_synchronize()?;
            if let Ok(mut p) = self.prefill_pipe.lock() {
                if p.inflight == Some(layer) {
                    p.inflight = None;
                    p.ready |= 1u64 << layer.min(63);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn prefill_kickoff_layer(
        &self,
        layer: u32,
        n_exp: u32,
        hosts: &[LayerFfn],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if hosts.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(ExpertId, LayerFfn)> = hosts
            .iter()
            .enumerate()
            .map(|(e, h)| {
                (
                    ExpertId {
                        layer,
                        expert: e as u32,
                    },
                    h.clone(),
                )
            })
            .collect();
        self.bulk_fill_experts(
            &pairs,
            metrics,
            Some((layer, n_exp)),
            true,
            None,
        )?;
        if let Ok(mut p) = self.prefill_pipe.lock() {
            p.inflight = Some(layer);
        }
        if let Some(metrics) = metrics {
            metrics
                .expert_prefill_overlaps
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    pub(crate) fn prefill_warm_layer_sync(
        &self,
        layer: u32,
        n_exp: u32,
        hosts: &[LayerFfn],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if hosts.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(ExpertId, LayerFfn)> = hosts
            .iter()
            .enumerate()
            .map(|(e, h)| {
                (
                    ExpertId {
                        layer,
                        expert: e as u32,
                    },
                    h.clone(),
                )
            })
            .collect();
        self.bulk_fill_experts(&pairs, metrics, Some((layer, n_exp)), false, None)?;
        allpaca_hip::stream_synchronize()?;
        if let Ok(mut p) = self.prefill_pipe.lock() {
            p.ready |= 1u64 << layer.min(63);
            if p.inflight == Some(layer) {
                p.inflight = None;
            }
        }
        Ok(())
    }

    /// Upload a real GGUF without host-materializing every tensor as F32.
    /// Q8_0 / large F16 GEMM weights stay packed; `output.weight` and large
    /// `token_embd.weight` use the same path (F16 lm_head/embed stay F16).
    /// Tied models reuse embed.
    pub fn upload_gguf(file: &GgufFile, cfg: &ModelConfig, device: i32) -> Result<Self, AllpacaError> {
        let mut parts: Vec<Vec<u8>> = Vec::new();
        let mut off = 0usize;
        let embed = append_gguf_tensor(file, "token_embd.weight", &mut parts, &mut off)?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut expert_offs = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let attn_norm = append_gguf_tensor(
                file,
                &format!("blk.{i}.attn_norm.weight"),
                &mut parts,
                &mut off,
            )?;
            let wq = append_gguf_tensor(file, &format!("blk.{i}.attn_q.weight"), &mut parts, &mut off)?;
            let wk = append_gguf_tensor(file, &format!("blk.{i}.attn_k.weight"), &mut parts, &mut off)?;
            let wv = append_gguf_tensor(file, &format!("blk.{i}.attn_v.weight"), &mut parts, &mut off)?;
            let wo = append_gguf_tensor(
                file,
                &format!("blk.{i}.attn_output.weight"),
                &mut parts,
                &mut off,
            )?;
            let ffn_norm = append_gguf_tensor(
                file,
                &format!("blk.{i}.ffn_norm.weight"),
                &mut parts,
                &mut off,
            )?;
            let (w_gate, w_up, w_down, w_router, packed_ex) = if cfg.n_experts > 0 {
                let w_router = if file.tensor(&format!("blk.{i}.ffn_gate_inp.weight")).is_some() {
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_gate_inp.weight"),
                        &mut parts,
                        &mut off,
                    )?
                } else {
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_router.weight"),
                        &mut parts,
                        &mut off,
                    )?
                };
                let packed_ex = append_gguf_experts(file, i, cfg.n_experts, &mut parts, &mut off)?;
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    w_router,
                    packed_ex,
                )
            } else {
                (
                    append_gguf_tensor(file, &format!("blk.{i}.ffn_gate.weight"), &mut parts, &mut off)?,
                    append_gguf_tensor(file, &format!("blk.{i}.ffn_up.weight"), &mut parts, &mut off)?,
                    append_gguf_tensor(
                        file,
                        &format!("blk.{i}.ffn_down.weight"),
                        &mut parts,
                        &mut off,
                    )?,
                    WeightRef::f32(0, 0),
                    Vec::new(),
                )
            };
            expert_offs.push(packed_ex);
            layers.push(LayerOff {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
                w_router,
            });
        }
        let norm = append_gguf_tensor(file, "output_norm.weight", &mut parts, &mut off)?;
        let lm_head = if file.tensor("output.weight").is_some() {
            append_gguf_tensor(file, "output.weight", &mut parts, &mut off)?
        } else {
            embed
        };

        let views: Vec<&[u8]> = parts.iter().map(|p| p.as_slice()).collect();
        let weights = DeviceWeights::upload_bytes(device, &views)?;
        Ok(Self {
            packed_h2d: weights.bytes(),
            ffn_ring: None,
            expert_offs,
            expert_ring: None,
            prefill_pipe: Mutex::new(PrefillGpuPipe::default()),
            weights,
            embed,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn bytes(&self) -> u64 {
        self.weights.bytes()
    }

    pub fn packed_h2d_bytes(&self) -> u64 {
        self.packed_h2d
    }

    fn bind_ffn<'a>(
        &'a self,
        layer_idx: usize,
        host: Option<&LayerFfn>,
        working_set: Option<&WeightWorkingSet>,
        metrics: Option<&Metrics>,
    ) -> Result<DeviceFfn<'a>, AllpacaError> {
        let layer = &self.layers[layer_idx];
        if layer.w_gate.n > 0 {
            return Ok(DeviceFfn::Resident {
                ffn_norm: layer.ffn_norm,
                w_gate: layer.w_gate,
                w_up: layer.w_up,
                w_down: layer.w_down,
            });
        }
        let ring = self.ffn_ring.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "streamed FFN layer has no device ring".into(),
            ))
        })?;
        let host = host.ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "streamed FFN layer {layer_idx} missing host pool"
            )))
        })?;
        let packed = host.bytes();
        if packed > ring.slot_bytes {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "FFN layer {layer_idx} is {packed} bytes, ring slot is {}",
                ring.slot_bytes
            ))));
        }
        let mut state = ring.state.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "device FFN ring mutex poisoned".into(),
            ))
        })?;
        let n_slots = state.bound.len();
        if let Some(slot) = state.bound.iter().position(|&b| b == Some(layer_idx)) {
            if let Some(ws) = working_set {
                ws.note_bind(true);
            }
            let [ffn_norm, w_gate, w_up, w_down] =
                slot_ffn_refs(ring.base, ring.slot_bytes, slot, host);
            return Ok(DeviceFfn::Bound {
                _guard: state,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }
        let slot = state
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                let s = state.cursor % n_slots.max(1);
                state.cursor = state.cursor.wrapping_add(1);
                s
            });
        let mut copy_off = ring.base + slot * ring.slot_bytes;
        for part in [&host.ffn_norm, &host.w_gate, &host.w_up, &host.w_down] {
            let n = part.len() * 4;
            let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr().cast::<u8>(), n) };
            self.weights.copy_h2d(copy_off, bytes)?;
            copy_off += n;
        }
        state.bound[slot] = Some(layer_idx);
        if let Some(ws) = working_set {
            ws.note_bind(false);
        }
        if let Some(metrics) = metrics {
            metrics
                .weight_h2d_bytes
                .fetch_add(packed as u64, Ordering::Relaxed);
        }
        let [ffn_norm, w_gate, w_up, w_down] =
            slot_ffn_refs(ring.base, ring.slot_bytes, slot, host);
        Ok(DeviceFfn::Bound {
            _guard: state,
            ffn_norm,
            w_gate,
            w_up,
            w_down,
        })
    }

    fn bind_expert<'a>(
        &'a self,
        id: ExpertId,
        host: &LayerFfn,
        metrics: Option<&Metrics>,
    ) -> Result<DeviceExpert<'a>, AllpacaError> {
        if let Some(refs) = self
            .expert_offs
            .get(id.layer as usize)
            .and_then(|layer| layer.get(id.expert as usize))
        {
            return Ok(DeviceExpert::Resident {
                w_gate: refs[0],
                w_up: refs[1],
                w_down: refs[2],
            });
        }
        let ring = self.expert_ring.as_ref().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(
                "streamed expert has no device slot ring".into(),
            ))
        })?;
        let packed = host.bytes();
        if packed > ring.slot_bytes {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "expert ({}, {}) is {packed} bytes, slot is {}",
                id.layer, id.expert, ring.slot_bytes
            ))));
        }
        let mut state = ring.state.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "device expert ring mutex poisoned".into(),
            ))
        })?;
        state.clock = state.clock.saturating_add(1);
        if let Some(slot) = state.bound.iter().position(|&b| b == Some(id)) {
            state.used_at[slot] = state.clock;
            let [w_gate, w_up, w_down] = slot_expert_refs(ring.base, ring.slot_bytes, slot, host);
            return Ok(DeviceExpert::Bound {
                _guard: state,
                w_gate,
                w_up,
                w_down,
            });
        }
        let victim = state
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                state
                    .used_at
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
        let mut copy_off = ring.base + victim * ring.slot_bytes;
        for part in [&host.ffn_norm, &host.w_gate, &host.w_up, &host.w_down] {
            let n = part.len() * 4;
            let bytes = unsafe { std::slice::from_raw_parts(part.as_ptr().cast::<u8>(), n) };
            self.weights.copy_h2d(copy_off, bytes)?;
            copy_off += n;
        }
        state.bound[victim] = Some(id);
        state.used_at[victim] = state.clock;
        if let Some(metrics) = metrics {
            metrics
                .weight_h2d_bytes
                .fetch_add(packed as u64, Ordering::Relaxed);
        }
        let [w_gate, w_up, w_down] = slot_expert_refs(ring.base, ring.slot_bytes, victim, host);
        Ok(DeviceExpert::Bound {
            _guard: state,
            w_gate,
            w_up,
            w_down,
        })
    }

    fn f32_or_dequant(
        &self,
        hip: &HipWorkspace,
        w: WeightRef,
    ) -> Result<*const f32, AllpacaError> {
        match w.kind {
            WeightKind::F32 => Ok(self.weights.ptr_at(w.off)),
            WeightKind::Q8_0 => {
                let out = hip.ensure_dequant(w.n)?;
                hip.dequant_q8_0(self.weights.byte_ptr_at(w.off), w.n as i32, out)?;
                Ok(out as *const f32)
            }
            WeightKind::F16 | WeightKind::Q4_K | WeightKind::Q5_K | WeightKind::Q6_K => {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "F16/Q4_K/Q5_K/Q6_K norm must be F32 at load".into(),
                )))
            }
        }
    }

    fn gemm_nt(
        &self,
        hip: &HipWorkspace,
        a: *const f32,
        w: WeightRef,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        match w.kind {
            WeightKind::F32 => hip.gemm_nt_dev(a, self.weights.ptr_at(w.off), c, m, n, k),
            WeightKind::F16 => hip.gemm_nt_f16_dev(
                a,
                self.weights.byte_ptr_at(w.off) as *const u16,
                c,
                m,
                n,
                k,
            ),
            WeightKind::Q8_0 => {
                hip.gemm_nt_q8_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::Q4_K => {
                hip.gemm_nt_q4_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::Q5_K => {
                hip.gemm_nt_q5_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
            WeightKind::Q6_K => {
                hip.gemm_nt_q6_k_dev(a, self.weights.byte_ptr_at(w.off), c, m, n, k)
            }
        }
    }

    fn embed_gather(
        &self,
        hip: &HipWorkspace,
        token: u32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        match self.embed.kind {
            WeightKind::F32 => hip.embed_gather(self.embed(), token, out, dim),
            WeightKind::F16 => hip.embed_gather_f16(
                self.weights.byte_ptr_at(self.embed.off) as *const u16,
                token,
                out,
                dim,
            ),
            WeightKind::Q8_0 => hip.embed_gather_q8_0(
                self.weights.byte_ptr_at(self.embed.off),
                token,
                out,
                dim,
            ),
            WeightKind::Q4_K | WeightKind::Q5_K | WeightKind::Q6_K => {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "Q4_K/Q5_K/Q6_K embed must be F32 at load".into(),
                )))
            }
        }
    }

    fn embed_gather_rows(
        &self,
        hip: &HipWorkspace,
        tokens: &[u32],
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        match self.embed.kind {
            WeightKind::F32 => hip.embed_gather_rows(self.embed(), tokens, out, dim),
            WeightKind::F16 => hip.embed_gather_f16_rows(
                self.weights.byte_ptr_at(self.embed.off) as *const u16,
                tokens,
                out,
                dim,
            ),
            WeightKind::Q8_0 => hip.embed_gather_q8_0_rows(
                self.weights.byte_ptr_at(self.embed.off),
                tokens,
                out,
                dim,
            ),
            WeightKind::Q4_K | WeightKind::Q5_K | WeightKind::Q6_K => {
                Err(AllpacaError::Fatal(FatalError::ModelLoad(
                    "Q4_K/Q5_K/Q6_K embed must be F32 at load".into(),
                )))
            }
        }
    }

    pub fn embed(&self) -> *const f32 {
        self.weights.ptr_at(self.embed.off)
    }

    pub fn attn_norm(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].attn_norm.off)
    }

    pub fn wq(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wq.off)
    }

    pub fn wk(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wk.off)
    }

    pub fn wv(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wv.off)
    }

    pub fn wo(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].wo.off)
    }

    pub fn ffn_norm(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].ffn_norm.off)
    }

    pub fn w_gate(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_gate.off)
    }

    pub fn w_up(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_up.off)
    }

    pub fn w_down(&self, layer: usize) -> *const f32 {
        self.weights.ptr_at(self.layers[layer].w_down.off)
    }

    pub fn norm(&self) -> *const f32 {
        self.weights.ptr_at(self.norm.off)
    }

    pub fn lm_head(&self) -> *const f32 {
        self.weights.ptr_at(self.lm_head.off)
    }
}

fn expert_mlp(
    x: &[f32],
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    out: &mut [f32],
) -> Result<(), AllpacaError> {
    let mut gate = vec![0.0f32; ff_dim];
    let mut up = vec![0.0f32; ff_dim];
    matmul_nt(x, &ffn.w_gate, &mut gate, 1, ff_dim, dim)?;
    matmul_nt(x, &ffn.w_up, &mut up, 1, ff_dim, dim)?;
    for i in 0..ff_dim {
        gate[i] = silu(gate[i]) * up[i];
    }
    matmul_nt(&gate, &ffn.w_down, out, 1, dim, ff_dim)
}

pub(crate) fn measure_expert_host_bandwidth(
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
) -> f64 {
    use std::time::Instant;
    if ffn.bytes() == 0 || dim == 0 || ff_dim == 0 {
        return 0.0;
    }
    let x = vec![0.01f32; dim];
    let mut out = vec![0.0f32; dim];
    let _ = expert_mlp(&x, ffn, ff_dim, dim, &mut out);
    let iters = 8usize;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = expert_mlp(&x, ffn, ff_dim, dim, &mut out);
    }
    let secs = t.elapsed().as_secs_f64().max(1e-9);
    (ffn.bytes() as f64 * iters as f64) / secs
}

/// Host GEMM bandwidth on the expert home node vs a cross-node read (for `q*`).
pub(crate) fn measure_expert_host_bandwidth_numa(
    ffn: &LayerFfn,
    ff_dim: usize,
    dim: usize,
    numa: &crate::numa::NumaTopology,
    id: ExpertId,
) -> f64 {
    let home = numa.preferred_node(id);
    let local = numa.run_on_node(home, || measure_expert_host_bandwidth(ffn, ff_dim, dim));
    if numa.n_nodes <= 1 {
        return local;
    }
    let remote = (home + 1) % numa.n_nodes;
    let cross = numa.run_on_node(remote, || measure_expert_host_bandwidth(ffn, ff_dim, dim));
    if cross <= 0.0 {
        local
    } else {
        cross.min(local)
    }
}

/// Profile fixed costs for overhead-aware `q*` (DMA setup + CPU dispatch).
pub(crate) fn measure_qstar_fixed_costs(expert_bytes: usize) -> (f64, f64) {
    use std::time::Instant;
    let n = expert_bytes.max(64);
    let src = vec![0u8; n];
    let mut dst = vec![0u8; n];
    // Warm.
    dst.copy_from_slice(&src);
    let iters = 32usize;
    let t = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(&src);
    }
    let t_dma = t.elapsed().as_secs_f64() / iters as f64;

    let t = Instant::now();
    for _ in 0..iters {
        std::thread::scope(|s| {
            s.spawn(|| ());
        });
    }
    let t_dispatch = t.elapsed().as_secs_f64() / iters as f64;
    (t_dma, t_dispatch)
}

fn topk_route(x: &[f32], w_router: &[f32], n_experts: usize, n_active: usize) -> Vec<(usize, f32)> {
    let dim = x.len();
    let mut logits = vec![0.0f32; n_experts];
    for e in 0..n_experts {
        let row = &w_router[e * dim..(e + 1) * dim];
        let mut s = 0.0f32;
        for i in 0..dim {
            s += x[i] * row[i];
        }
        logits[e] = s;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = logits
        .iter()
        .map(|v| (*v - max).exp())
        .collect();
    let sum: f32 = weights.iter().sum::<f32>().max(1e-9);
    for w in &mut weights {
        *w /= sum;
    }
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    idx.truncate(n_active.max(1).min(n_experts));
    let part: f32 = idx.iter().map(|&i| weights[i]).sum::<f32>().max(1e-9);
    idx.into_iter().map(|i| (i, weights[i] / part)).collect()
}

fn empty_layer() -> LayerWeights {
    LayerWeights {
        attn_norm: Vec::new(),
        wq: Vec::new(),
        wk: Vec::new(),
        wv: Vec::new(),
        wo: Vec::new(),
        ffn_norm: Vec::new(),
        w_gate: Vec::new(),
        w_up: Vec::new(),
        w_down: Vec::new(),
        w_router: Vec::new(),
        experts: Vec::new(),
    }
}

fn estimated_gguf_f32_bytes(file: &GgufFile, cfg: &ModelConfig) -> u64 {
    let mut names = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "output.weight".to_string(),
    ];
    for i in 0..cfg.n_layers {
        names.push(format!("blk.{i}.attn_norm.weight"));
        names.push(format!("blk.{i}.attn_q.weight"));
        names.push(format!("blk.{i}.attn_k.weight"));
        names.push(format!("blk.{i}.attn_v.weight"));
        names.push(format!("blk.{i}.attn_output.weight"));
        names.push(format!("blk.{i}.ffn_norm.weight"));
        if cfg.n_experts > 0 {
            names.push(format!("blk.{i}.ffn_gate_inp.weight"));
            names.push(format!("blk.{i}.ffn_gate_exps.weight"));
            names.push(format!("blk.{i}.ffn_up_exps.weight"));
            names.push(format!("blk.{i}.ffn_down_exps.weight"));
            for e in 0..cfg.n_experts {
                names.push(format!("blk.{i}.ffn_gate.{e}.weight"));
                names.push(format!("blk.{i}.ffn_up.{e}.weight"));
                names.push(format!("blk.{i}.ffn_down.{e}.weight"));
            }
        } else {
            names.push(format!("blk.{i}.ffn_gate.weight"));
            names.push(format!("blk.{i}.ffn_up.weight"));
            names.push(format!("blk.{i}.ffn_down.weight"));
        }
    }
    names
        .iter()
        .filter_map(|n| file.tensor(n))
        .map(|t| t.n_elements().saturating_mul(4))
        .sum()
}

fn require_gguf_tensor(file: &GgufFile, name: &str) -> Result<(), AllpacaError> {
    if file.tensor(name).is_none() {
        Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "missing tensor {name}"
        ))))
    } else {
        Ok(())
    }
}

fn split_expert_rows(data: Vec<f32>, n_experts: usize, what: &str) -> Result<Vec<Vec<f32>>, AllpacaError> {
    if n_experts == 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{what} has no experts"
        ))));
    }
    if data.len() % n_experts != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{what} len {} is not divisible by {n_experts} experts",
            data.len()
        ))));
    }
    let chunk = data.len() / n_experts;
    Ok(data.chunks(chunk).map(|c| c.to_vec()).collect())
}

fn load_gguf_moe_layer(
    file: &GgufFile,
    layer: usize,
    n_experts: usize,
) -> Result<(Vec<f32>, Vec<LayerFfn>), AllpacaError> {
    let router_name = format!("blk.{layer}.ffn_gate_inp.weight");
    let w_router = file.tensor_f32(&router_name).or_else(|_| {
        file.tensor_f32(&format!("blk.{layer}.ffn_router.weight"))
    })?;
    let stacked_gate = format!("blk.{layer}.ffn_gate_exps.weight");
    let experts = if file.tensor(&stacked_gate).is_some() {
        let gates = split_expert_rows(file.tensor_f32(&stacked_gate)?, n_experts, &stacked_gate)?;
        let ups = split_expert_rows(
            file.tensor_f32(&format!("blk.{layer}.ffn_up_exps.weight"))?,
            n_experts,
            "ffn_up_exps",
        )?;
        let downs = split_expert_rows(
            file.tensor_f32(&format!("blk.{layer}.ffn_down_exps.weight"))?,
            n_experts,
            "ffn_down_exps",
        )?;
        (0..n_experts)
            .map(|e| LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: gates[e].clone(),
                w_up: ups[e].clone(),
                w_down: downs[e].clone(),
            })
            .collect()
    } else {
        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            experts.push(LayerFfn {
                ffn_norm: Vec::new(),
                w_gate: file.tensor_f32(&format!("blk.{layer}.ffn_gate.{e}.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{layer}.ffn_up.{e}.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{layer}.ffn_down.{e}.weight"))?,
            });
        }
        experts
    };
    Ok((w_router, experts))
}

fn append_gguf_tensor(
    file: &GgufFile,
    name: &str,
    parts: &mut Vec<Vec<u8>>,
    off: &mut usize,
) -> Result<WeightRef, AllpacaError> {
    let info = file.tensor(name).ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(format!("missing tensor {name}")))
    })?;
    let n = info.n_elements() as usize;
    let (bytes, kind) = match info.ggml_type {
        allpaca_gguf::GgmlType::F32 => (file.tensor_bytes(name)?.to_vec(), WeightKind::F32),
        allpaca_gguf::GgmlType::F16 => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 {
                let f32s = file.tensor_f32(name)?;
                let mut bytes = Vec::with_capacity(f32s.len() * 4);
                for v in &f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (bytes, WeightKind::F32)
            } else {
                (packed.to_vec(), WeightKind::F16)
            }
        }
        allpaca_gguf::GgmlType::Q4_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q4_k(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (bytes, WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q4_K GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (packed.to_vec(), WeightKind::Q4_K)
            }
        }
        allpaca_gguf::GgmlType::Q5_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q5_k(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (bytes, WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q5_K GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (packed.to_vec(), WeightKind::Q5_K)
            }
        }
        allpaca_gguf::GgmlType::Q6_K => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 || name == "token_embd.weight" {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q6_k(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (bytes, WeightKind::F32)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q6_K GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                (packed.to_vec(), WeightKind::Q6_K)
            }
        }
        allpaca_gguf::GgmlType::Q8_0 => {
            let packed = file.tensor_bytes(name)?;
            if n <= 65_536 {
                let mut f32s = vec![0.0f32; n];
                allpaca_cpu::dequant_q8_0(packed, n, &mut f32s)?;
                let mut bytes = Vec::with_capacity(n * 4);
                for v in f32s {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                (bytes, WeightKind::F32)
            } else if name == "token_embd.weight" {
                let staged = allpaca_hip::stage_q8_0_blocks(packed).map_err(|e| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!("Q8_0 stage {name}: {e}")))
                })?;
                (staged, WeightKind::Q8_0)
            } else {
                let k = *info.dims.first().unwrap_or(&1) as usize;
                if k == 0 || n % k != 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "Q8_0 GEMV {name} dims {:?} do not tile n={n}",
                        info.dims
                    ))));
                }
                let n_rows = n / k;
                let staged = allpaca_hip::stage_q8_0_gemv(packed, n_rows, k).map_err(|e| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!("Q8_0 GEMV stage {name}: {e}")))
                })?;
                (staged, WeightKind::Q8_0)
            }
        }
        _ => {
            let f32s = file.tensor_f32(name)?;
            let mut bytes = Vec::with_capacity(f32s.len() * 4);
            for v in &f32s {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (bytes, WeightKind::F32)
        }
    };
    if kind == WeightKind::Q8_0 && *off % allpaca_hip::Q8_STAGED_BLOCK != 0 {
        let pad = allpaca_hip::Q8_STAGED_BLOCK - (*off % allpaca_hip::Q8_STAGED_BLOCK);
        parts.push(vec![0u8; pad]);
        *off += pad;
    }
    if kind == WeightKind::F16 && *off % 16 != 0 {
        let pad = 16 - (*off % 16);
        parts.push(vec![0u8; pad]);
        *off += pad;
    }
    let w = WeightRef {
        off: *off,
        n,
        kind,
    };
    *off += bytes.len();
    parts.push(bytes);
    Ok(w)
}

fn split_packed_experts(
    w: WeightRef,
    n_experts: usize,
    what: &str,
) -> Result<Vec<WeightRef>, AllpacaError> {
    if n_experts == 0 || w.n % n_experts != 0 {
        return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
            "{what} n={} is not divisible by {n_experts} experts",
            w.n
        ))));
    }
    let n_per = w.n / n_experts;
    let stride = match w.kind {
        WeightKind::F32 => n_per.saturating_mul(4),
        WeightKind::F16 => n_per.saturating_mul(2),
        _ => {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "{what} packed quant experts cannot be sliced; use host F32"
            ))))
        }
    };
    Ok((0..n_experts)
        .map(|e| WeightRef {
            off: w.off + e * stride,
            n: n_per,
            kind: w.kind,
        })
        .collect())
}

fn append_gguf_experts(
    file: &GgufFile,
    layer: usize,
    n_experts: usize,
    parts: &mut Vec<Vec<u8>>,
    off: &mut usize,
) -> Result<Vec<[WeightRef; 3]>, AllpacaError> {
    let stacked = format!("blk.{layer}.ffn_gate_exps.weight");
    if file.tensor(&stacked).is_some() {
        let gate = append_gguf_tensor(file, &stacked, parts, off)?;
        let up = append_gguf_tensor(
            file,
            &format!("blk.{layer}.ffn_up_exps.weight"),
            parts,
            off,
        )?;
        let down = append_gguf_tensor(
            file,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            parts,
            off,
        )?;
        let gates = split_packed_experts(gate, n_experts, "ffn_gate_exps")?;
        let ups = split_packed_experts(up, n_experts, "ffn_up_exps")?;
        let downs = split_packed_experts(down, n_experts, "ffn_down_exps")?;
        return Ok((0..n_experts)
            .map(|e| [gates[e], ups[e], downs[e]])
            .collect());
    }
    let mut out = Vec::with_capacity(n_experts);
    for e in 0..n_experts {
        out.push([
            append_gguf_tensor(
                file,
                &format!("blk.{layer}.ffn_gate.{e}.weight"),
                parts,
                off,
            )?,
            append_gguf_tensor(file, &format!("blk.{layer}.ffn_up.{e}.weight"), parts, off)?,
            append_gguf_tensor(
                file,
                &format!("blk.{layer}.ffn_down.{e}.weight"),
                parts,
                off,
            )?,
        ]);
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub w_router: Vec<f32>,
    pub experts: Vec<LayerFfn>,
}

#[derive(Debug, Clone)]
pub struct TinyModel {
    pub config: ModelConfig,
    pub embed: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub name: String,
    pub working_set: Option<Arc<WeightWorkingSet>>,
    pub expert_lru: Option<Arc<ExpertLru>>,
    pub expert_host: Option<Arc<HostExpertPool>>,
    /// Manifest early-exit capability (unsupported for synthetic / most GGUF).
    pub early_exit: EarlyExitSpec,
}

impl TinyModel {
    pub fn synthetic(cfg: ModelConfig, seed: u64, name: impl Into<String>) -> Self {
        let mut rng = Lcg::new(seed);
        let dim = cfg.dim;
        let vocab = cfg.vocab_size;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let (w_gate, w_up, w_down, w_router, experts) = if cfg.n_experts > 0 {
                let experts = (0..cfg.n_experts)
                    .map(|_| LayerFfn {
                        ffn_norm: Vec::new(),
                        w_gate: rng.vec(cfg.ff_dim * dim),
                        w_up: rng.vec(cfg.ff_dim * dim),
                        w_down: rng.vec(dim * cfg.ff_dim),
                    })
                    .collect();
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    rng.vec(cfg.n_experts * dim),
                    experts,
                )
            } else {
                (
                    rng.vec(cfg.ff_dim * dim),
                    rng.vec(cfg.ff_dim * dim),
                    rng.vec(dim * cfg.ff_dim),
                    Vec::new(),
                    Vec::new(),
                )
            };
            layers.push(LayerWeights {
                attn_norm: ones(dim),
                wq: rng.vec(dim * dim),
                wk: rng.vec(kv_dim * dim),
                wv: rng.vec(kv_dim * dim),
                wo: rng.vec(dim * dim),
                ffn_norm: ones(dim),
                w_gate,
                w_up,
                w_down,
                w_router,
                experts,
            });
        }
        Self {
            embed: rng.vec(vocab * dim),
            layers,
            norm: ones(dim),
            lm_head: rng.vec(vocab * dim),
            name: name.into(),
            config: cfg,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
        }
    }

    pub fn host_weights_ready(&self) -> bool {
        !self.embed.is_empty()
    }

    pub(crate) fn expert_slot_bytes(&self) -> usize {
        if let Some(pool) = &self.expert_host {
            return pool.expert_bytes();
        }
        self.layers
            .iter()
            .find_map(|l| l.experts.first())
            .map(|e| e.bytes())
            .unwrap_or(0)
    }

    pub(crate) fn host_expert(&self, layer_idx: usize, expert: usize) -> Result<LayerFfn, AllpacaError> {
        let id = ExpertId {
            layer: layer_idx as u32,
            expert: expert as u32,
        };
        if let Some(pool) = &self.expert_host {
            return pool.get(id);
        }
        self.layers
            .get(layer_idx)
            .and_then(|l| l.experts.get(expert))
            .cloned()
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "missing expert {expert} on layer {layer_idx}"
                )))
            })
    }

    /// Bind CPU expert MLP to the preferred NUMA node when a host pool is present.
    fn run_expert_mlp_numa(
        &self,
        id: ExpertId,
        x_norm: &[f32],
        ffn: &LayerFfn,
        ff_dim: usize,
        dim: usize,
        out: &mut [f32],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if let Some(pool) = &self.expert_host {
            let node = pool.preferred_numa_node(id);
            let r = pool.numa().run_on_node(node, || expert_mlp(x_norm, ffn, ff_dim, dim, out));
            if let Some(m) = metrics {
                HostExpertPool::record_numa(m, pool.numa());
            }
            return r;
        }
        expert_mlp(x_norm, ffn, ff_dim, dim, out)
    }

    fn host_experts_layer(&self, layer_idx: usize) -> Result<Vec<LayerFfn>, AllpacaError> {
        let n_exp = self.config.n_experts;
        if let Some(pool) = &self.expert_host {
            return pool.get_layer(layer_idx as u32, n_exp);
        }
        self.layers
            .get(layer_idx)
            .map(|l| l.experts.clone())
            .ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!(
                    "MoE missing host layer {layer_idx}"
                )))
            })
    }

    fn ffn<'a>(
        &'a self,
        layer_idx: usize,
        layer: &'a LayerWeights,
    ) -> Result<FfnRefs<'a>, AllpacaError> {
        if let Some(ws) = &self.working_set {
            if !ws.is_pinned(layer_idx) {
                return Ok(FfnRefs::Bound(ws.bind(layer_idx)?));
            }
        }
        Ok(FfnRefs::Resident(layer))
    }

    fn device_ffn<'a>(
        &'a self,
        resident: &'a ResidentModel,
        layer_idx: usize,
        metrics: Option<&Metrics>,
    ) -> Result<DeviceFfn<'a>, AllpacaError> {
        // Warm the host ring (and spare-slot L+1 prefetch) before H2D — same
        // policy as CPU `ffn()` → `WeightWorkingSet::bind`.
        let host = match self.working_set.as_ref() {
            Some(ws) if !ws.is_pinned(layer_idx) => {
                {
                    let _warm = ws.bind(layer_idx)?;
                }
                Some(ws.load_layer(layer_idx)?)
            }
            Some(ws) => ws.host_ffn(layer_idx).cloned(),
            None => None,
        };
        resident.bind_ffn(
            layer_idx,
            host.as_ref(),
            self.working_set.as_deref(),
            metrics,
        )
    }

    fn streamed_ffn(&self, layer_idx: usize) -> bool {
        self.working_set
            .as_ref()
            .is_some_and(|ws| !ws.is_pinned(layer_idx))
    }

    fn apply_ffn(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        x: &mut [f32],
        scratch: &mut [f32],
        hidden: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
        prefill: bool,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        if cfg.n_experts > 0 {
            if prefill {
                self.prefetch_moe_layers(layer_idx)?;
            }
            rmsnorm(x, &layer.ffn_norm, cfg.eps, &mut scratch[..dim])?;
            let moe_out = self.moe_hidden(layer_idx, layer, &scratch[..dim])?;
            for i in 0..dim {
                x[i] += moe_out[i];
            }
            return Ok(());
        }
        let ffn = self.ffn(layer_idx, layer)?;
        let ffn_dev = !self.streamed_ffn(layer_idx);
        rmsnorm_dispatch(
            hip,
            x,
            ffn.ffn_norm(),
            resident.filter(|_| ffn_dev).map(|r| r.ffn_norm(layer_idx)),
            cfg.eps,
            &mut scratch[..dim],
            metrics,
        )?;
        let mut gate = vec![0.0f32; cfg.ff_dim];
        let mut up = vec![0.0f32; cfg.ff_dim];
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            ffn.w_gate(),
            resident.filter(|_| ffn_dev).map(|r| r.w_gate(layer_idx)),
            &mut gate,
            1,
            cfg.ff_dim,
            dim,
        )?;
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            ffn.w_up(),
            resident.filter(|_| ffn_dev).map(|r| r.w_up(layer_idx)),
            &mut up,
            1,
            cfg.ff_dim,
            dim,
        )?;
        silu_mul_dispatch(hip, &mut gate, &up, metrics)?;
        gemm_nt_dispatch(
            hip,
            &gate,
            ffn.w_down(),
            resident.filter(|_| ffn_dev).map(|r| r.w_down(layer_idx)),
            hidden,
            1,
            dim,
            cfg.ff_dim,
        )?;
        for i in 0..dim {
            x[i] += hidden[i];
        }
        Ok(())
    }

    fn moe_hidden(
        &self,
        layer_idx: usize,
        layer: &LayerWeights,
        x_norm: &[f32],
    ) -> Result<Vec<f32>, AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let routed = topk_route(x_norm, &layer.w_router, n_exp, n_act);
        let ids: Vec<ExpertId> = routed
            .iter()
            .map(|&(e, _)| ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            })
            .collect();
        if let Some(lru) = &self.expert_lru {
            let plan = lru.plan(&ids);
            lru.note_plan(&plan);
            for id in &plan.fill {
                let host = self.host_expert(layer_idx, id.expert as usize)?;
                lru.admit(*id, &host)?;
            }
        }
        let mut acc = vec![0.0f32; dim];
        let mut scratch_h = vec![0.0f32; dim];
        for &(e, w) in &routed {
            let host = self.host_expert(layer_idx, e)?;
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            };
            let ffn = if let Some(lru) = &self.expert_lru {
                lru.materialize(id, &host)?
            } else {
                host
            };
            self.run_expert_mlp_numa(id, x_norm, &ffn, cfg.ff_dim, dim, &mut scratch_h, None)?;
            for i in 0..dim {
                acc[i] += w * scratch_h[i];
            }
        }
        Ok(acc)
    }

    fn prefetch_moe_layers(&self, layer_idx: usize) -> Result<(), AllpacaError> {
        let Some(lru) = &self.expert_lru else {
            return Ok(());
        };
        let n_exp = self.config.n_experts;
        if n_exp == 0 || !lru.holds_layers(n_exp, 1) {
            return Ok(());
        }
        let experts = self.host_experts_layer(layer_idx)?;
        lru.admit_layer(layer_idx as u32, &experts)?;
        if lru.holds_layers(n_exp, 2) {
            if layer_idx + 1 < self.layers.len() {
                let next = self.host_experts_layer(layer_idx + 1)?;
                lru.admit_layer((layer_idx + 1) as u32, &next)?;
            }
        }
        Ok(())
    }

    fn warm_moe_slots(
        &self,
        layer_idx: usize,
        resident: &ResidentModel,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let Some(lru) = &self.expert_lru else {
            return Ok(());
        };
        let n_exp = self.config.n_experts;
        if n_exp == 0 || !lru.holds_layers(n_exp, 1) {
            return Ok(());
        }
        let warm = |idx: usize| -> Result<(), AllpacaError> {
            if resident.prefill_layer_gpu_ready(idx as u32) {
                return Ok(());
            }
            let experts = self.host_experts_layer(idx)?;
            resident.prefill_warm_layer_sync(idx as u32, n_exp as u32, &experts, metrics)
        };
        warm(layer_idx)?;
        if lru.holds_layers(n_exp, 2) && layer_idx + 1 < self.layers.len() {
            warm(layer_idx + 1)?;
        }
        Ok(())
    }

    pub(crate) fn try_nvme_expert_fill(
        &self,
        id: ExpertId,
        dev_off: usize,
        slot_bytes: usize,
        side: Option<&DeviceDecodeSide<'_>>,
        weights: &ResidentModel,
    ) -> Result<bool, AllpacaError> {
        let Some(side) = side else {
            return Ok(false);
        };
        let (probe, queue) = match (side.ds_probe, side.ds_queue) {
            (Some(p), Some(q)) => (p, q),
            _ => return Ok(false),
        };
        let pool = match &self.expert_host {
            Some(p) => p,
            None => return Ok(false),
        };
        let Some((path, offset, bytes)) = pool.slot_coords(id) else {
            return Ok(false);
        };
        let bytes = bytes.min(slot_bytes);
        queue.fill_to_device(probe, &path, offset, bytes, |host| {
            weights.copy_expert_h2d(dev_off, host)
        })
    }

    /// Router-only union of experts a draft verify span may touch (prefetch).
    pub fn routed_experts_for_verify_span(&self, span: &[u32]) -> Vec<ExpertId> {
        if self.config.n_experts == 0 || span.is_empty() {
            return Vec::new();
        }
        let dim = self.config.dim;
        let n_exp = self.config.n_experts.max(1);
        let n_act = self.config.n_active.max(1).min(n_exp);
        let mut union = std::collections::HashSet::new();
        for &tok in span {
            if tok as usize >= self.config.vocab_size {
                continue;
            }
            let x = &self.embed[tok as usize * dim..(tok as usize + 1) * dim];
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                if layer.w_router.is_empty() {
                    continue;
                }
                for (e, _) in topk_route(x, &layer.w_router, n_exp, n_act) {
                    union.insert(ExpertId {
                        layer: layer_idx as u32,
                        expert: e as u32,
                    });
                }
            }
        }
        let mut out: Vec<_> = union.into_iter().collect();
        out.sort_by_key(|id| (id.layer, id.expert));
        out
    }

    fn device_expert_gemm(
        &self,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        w_gate: WeightRef,
        w_up: WeightRef,
        w_down: WeightRef,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        resident.gemm_nt(
            hip,
            buf.normed,
            w_gate,
            buf.gate,
            1,
            cfg.ff_dim as i32,
            dim as i32,
        )?;
        resident.gemm_nt(
            hip,
            buf.normed,
            w_up,
            buf.up,
            1,
            cfg.ff_dim as i32,
            dim as i32,
        )?;
        hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
        resident.gemm_nt(
            hip,
            buf.gate,
            w_down,
            buf.proj,
            1,
            dim as i32,
            cfg.ff_dim as i32,
        )?;
        Ok(())
    }

    fn run_stagger_partner_cpu(
        &self,
        layer_idx: usize,
        x_norm: &[f32],
        metrics: &Metrics,
    ) -> Result<(), AllpacaError> {
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "stagger missing host layer {layer_idx}"
            )))
        })?;
        let cfg = &self.config;
        let dim = cfg.dim;
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let routed = topk_route(x_norm, &layer.w_router, n_exp, n_act);
        if routed.is_empty() {
            return Ok(());
        }
        let mut scratch = vec![0.0f32; dim];
        for &(e, w) in &routed {
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            };
            let host = self.host_expert(layer_idx, e)?;
            let ffn = if let Some(lru) = &self.expert_lru {
                let plan = lru.plan(&[id]);
                if !plan.fill.is_empty() {
                    lru.admit(id, &host)?;
                }
                lru.materialize(id, &host)?
            } else {
                host
            };
            if let Some(lru) = &self.expert_lru {
                if lru.plan(&[id]).cpu.iter().any(|&c| c == id) {
                    self.run_expert_mlp_numa(id, x_norm, &ffn, cfg.ff_dim, dim, &mut scratch, None)?;
                    metrics
                        .expert_cpu_runs
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = w;
                }
            }
        }
        Ok(())
    }

    fn device_moe(
        &self,
        layer_idx: usize,
        hip: &HipWorkspace,
        resident: &ResidentModel,
        buf: DeviceDecode,
        metrics: &Metrics,
        prefill: bool,
        micro_batch: Option<&MicroBatchExpertUnion>,
        mut side: Option<&mut DeviceDecodeSide<'_>>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        if prefill {
            self.prefetch_moe_layers(layer_idx)?;
            self.warm_moe_slots(layer_idx, resident, Some(metrics))?;
        }
        let host_layer = self.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MoE missing host layer {layer_idx}"
            )))
        })?;
        let off = resident.layers.get(layer_idx).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "MoE missing device layer {layer_idx}"
            )))
        })?;
        if off.ffn_norm.n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "MoE layer missing packed ffn_norm".into(),
            )));
        }
        hip.residual_rmsnorm(
            buf.x,
            buf.proj,
            resident.f32_or_dequant(hip, off.ffn_norm)?,
            buf.normed,
            dim as i32,
            cfg.eps,
        )?;
        metrics
            .residual_gpu_ops
            .fetch_add(1, Ordering::Relaxed);
        record_elementwise(Some(metrics));
        let mut x_norm = vec![0.0f32; dim];
        hip.copy_d2h_f32(buf.normed, &mut x_norm)?;
        let ds_probe = side.as_ref().and_then(|s| s.ds_probe);
        let ds_queue = side.as_ref().and_then(|s| s.ds_queue);
        let nvme_side = side.as_ref().map(|s| DeviceDecodeSide {
            request_id: s.request_id,
            moe_norm: None,
            layer_stagger: s.layer_stagger,
            ds_probe,
            ds_queue,
        });
        if let Some(s) = side.as_mut() {
            if let Some(norm) = s.moe_norm.as_mut() {
                if norm.len() == dim {
                    norm.copy_from_slice(&x_norm);
                }
            }
            if let Some(st) = s.layer_stagger.and_then(|ls| ls.active()) {
                if st.gpu_request == s.request_id && layer_idx == st.gpu_layer {
                    let partner_norm = s.layer_stagger.unwrap().partner_norm();
                    if partner_norm.len() == dim {
                        metrics
                            .layer_stagger_steps
                            .fetch_add(1, Ordering::Relaxed);
                        let _ = self.run_stagger_partner_cpu(
                            st.cpu_layer,
                            &partner_norm,
                            metrics,
                        );
                    }
                }
            }
        }
        let n_exp = cfg.n_experts.max(1);
        let n_act = cfg.n_active.max(1).min(n_exp);
        let routed = topk_route(&x_norm, &host_layer.w_router, n_exp, n_act);
        let ids: Vec<ExpertId> = routed
            .iter()
            .map(|&(e, _)| ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            })
            .collect();
        let plan = if let Some(lru) = &self.expert_lru {
            let route_ids = if !prefill {
                if let Some(u) = micro_batch {
                    let merged = u.union_with(layer_idx as u32, &ids);
                    if merged.len() == ids.len() {
                        u.stage_routed(layer_idx as u32, &ids);
                    }
                    merged
                } else {
                    ids.clone()
                }
            } else {
                ids.clone()
            };
            let union_plan = lru.plan(&route_ids);
            lru.note_plan(&union_plan);
                if !union_plan.fill.is_empty() {
                let mut pairs: Vec<(ExpertId, LayerFfn)> = Vec::new();
                for id in &union_plan.fill {
                    let host = self.host_expert(id.layer as usize, id.expert as usize)?;
                    pairs.push((*id, host));
                }
                if !pairs.is_empty() {
                    let nvme_side = nvme_side.as_ref();
                    let mut nvme = |id: ExpertId, dev_off: usize, slot_bytes: usize| {
                        self.try_nvme_expert_fill(
                            id,
                            dev_off,
                            slot_bytes,
                            nvme_side,
                            resident,
                        )
                    };
                    resident.bulk_fill_experts(
                        &pairs,
                        Some(metrics),
                        None,
                        false,
                        Some(&mut nvme),
                    )?;
                }
                for (id, host) in &pairs {
                    lru.admit(*id, host)?;
                }
            }
            lru.plan(&ids)
        } else {
            MissPlan {
                hits: ids.clone(),
                fill: Vec::new(),
                cpu: Vec::new(),
            }
        };
        hip.copy_h2d_f32(buf.attn, &vec![0.0f32; dim])?;
        let mut cpu_jobs: Vec<(ExpertId, LayerFfn, f32)> = Vec::new();
        let mut gpu_jobs: Vec<(ExpertId, LayerFfn, f32)> = Vec::new();
        for &(e, w) in &routed {
            let id = ExpertId {
                layer: layer_idx as u32,
                expert: e as u32,
            };
            let host = self.host_expert(layer_idx, e)?;
            let ffn = if let Some(lru) = &self.expert_lru {
                lru.materialize(id, &host)?
            } else {
                host
            };
            if plan.cpu.iter().any(|&c| c == id) {
                cpu_jobs.push((id, ffn, w));
            } else {
                gpu_jobs.push((id, ffn, w));
            }
        }
        // GEM micro-batch: run CPU expert MLPs in parallel with the GPU expert loop.
        if !cpu_jobs.is_empty() && !gpu_jobs.is_empty() {
            metrics
                .micro_batch_plans
                .fetch_add(1, Ordering::Relaxed);
        }
        let cpu_outs: Vec<Vec<f32>> = std::thread::scope(|scope| {
            let handles: Vec<_> = cpu_jobs
                .iter()
                .map(|(id, ffn, w)| {
                    let x = &x_norm;
                    scope.spawn(move || {
                        let mut scratch = vec![0.0f32; dim];
                        // NUMA bind inside thread for this expert.
                        if let Some(pool) = &self.expert_host {
                            let node = pool.preferred_numa_node(*id);
                            pool.numa()
                                .run_on_node(node, || {
                                    expert_mlp(x, ffn, cfg.ff_dim, dim, &mut scratch)
                                })
                                .expect("expert_mlp");
                        } else {
                            expert_mlp(x, ffn, cfg.ff_dim, dim, &mut scratch).expect("expert_mlp");
                        }
                        for v in &mut scratch {
                            *v *= *w;
                        }
                        scratch
                    })
                })
                .collect();
            for (id, ffn, w) in &gpu_jobs {
                let expert = resident.bind_expert(*id, ffn, Some(metrics))?;
                self.device_expert_gemm(
                    hip,
                    resident,
                    buf,
                    expert.w_gate(),
                    expert.w_up(),
                    expert.w_down(),
                )?;
                drop(expert);
                hip.scale_f32(buf.proj, dim, *w)?;
                hip.residual_add(buf.attn, buf.proj, dim as i32)?;
            }
            Ok::<Vec<Vec<f32>>, AllpacaError>(
                handles
                    .into_iter()
                    .map(|h| h.join().expect("cpu expert thread"))
                    .collect(),
            )
        })?;
        for scratch in cpu_outs {
            hip.copy_h2d_f32(buf.proj, &scratch)?;
            hip.residual_add(buf.attn, buf.proj, dim as i32)?;
        }
        hip.copy_d2d_f32(buf.proj, buf.attn, dim)?;
        record_elementwise(Some(metrics));
        Ok(())
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
        let report = file.compat_report();
        if let Some(reason) = report.reject_reason {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(reason)));
        }
        let cfg = ModelConfig::from_gguf(file)?;
        let name = file
            .meta_str("general.name")
            .unwrap_or("gguf")
            .to_string();
        for i in 0..cfg.n_layers {
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
            ] {
                require_gguf_tensor(file, &format!("blk.{i}.{suffix}"))?;
            }
            if cfg.n_experts > 0 {
                if file.tensor(&format!("blk.{i}.ffn_gate_exps.weight")).is_some() {
                    require_gguf_tensor(file, &format!("blk.{i}.ffn_gate_inp.weight"))?;
                    require_gguf_tensor(file, &format!("blk.{i}.ffn_up_exps.weight"))?;
                    require_gguf_tensor(file, &format!("blk.{i}.ffn_down_exps.weight"))?;
                } else {
                    require_gguf_tensor(file, &format!("blk.{i}.ffn_gate_inp.weight"))
                        .or_else(|_| require_gguf_tensor(file, &format!("blk.{i}.ffn_router.weight")))?;
                    for e in 0..cfg.n_experts {
                        require_gguf_tensor(file, &format!("blk.{i}.ffn_gate.{e}.weight"))?;
                        require_gguf_tensor(file, &format!("blk.{i}.ffn_up.{e}.weight"))?;
                        require_gguf_tensor(file, &format!("blk.{i}.ffn_down.{e}.weight"))?;
                    }
                }
            } else {
                for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    require_gguf_tensor(file, &format!("blk.{i}.{suffix}"))?;
                }
            }
        }
        if file.tensor("token_embd.weight").is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "missing tensor token_embd.weight".into(),
            )));
        }
        if file.tensor("output_norm.weight").is_none() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "missing tensor output_norm.weight".into(),
            )));
        }
        let host_bytes = estimated_gguf_f32_bytes(file, &cfg);
        if host_bytes > HOST_F32_LIMIT {
            return Ok(Self::skeleton(cfg, name));
        }
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let (w_gate, w_up, w_down, w_router, experts) = if cfg.n_experts > 0 {
                let (w_router, experts) = load_gguf_moe_layer(file, i, cfg.n_experts)?;
                (Vec::new(), Vec::new(), Vec::new(), w_router, experts)
            } else {
                (
                    file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                    file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                    file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                    Vec::new(),
                    Vec::new(),
                )
            };
            layers.push(LayerWeights {
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate,
                w_up,
                w_down,
                w_router,
                experts,
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
        })
    }

    fn skeleton(cfg: ModelConfig, name: impl Into<String>) -> Self {
        Self {
            embed: Vec::new(),
            layers: (0..cfg.n_layers).map(|_| empty_layer()).collect(),
            norm: Vec::new(),
            lm_head: Vec::new(),
            name: name.into(),
            config: cfg,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
        }
    }

    pub fn from_safetensors(file: &SafetensorsFile, hf: &HfConfig) -> Result<Self, AllpacaError> {
        let n_heads = hf.num_attention_heads.max(1);
        let n_kv_heads = hf.num_key_value_heads.unwrap_or(n_heads).max(1);
        let ff_dim = hf.intermediate_size.unwrap_or(hf.hidden_size * 4);
        let cfg = ModelConfig {
            vocab_size: hf.vocab_size,
            dim: hf.hidden_size,
            n_layers: hf.num_hidden_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            n_experts: 0,
            n_active: 0,
            eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
            recurrent: false,
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                w_router: Vec::new(),
                experts: Vec::new(),
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        let name = hf
            .architectures
            .first()
            .cloned()
            .unwrap_or_else(|| "safetensors".into());
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name,
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
        })
    }

    pub fn from_onnx(file: &OnnxFile, hf: &HfConfig) -> Result<Self, AllpacaError> {
        let n_heads = hf.num_attention_heads.max(1);
        let n_kv_heads = hf.num_key_value_heads.unwrap_or(n_heads).max(1);
        let ff_dim = hf.intermediate_size.unwrap_or(hf.hidden_size * 4);
        let cfg = ModelConfig {
            vocab_size: hf.vocab_size,
            dim: hf.hidden_size,
            n_layers: hf.num_hidden_layers,
            n_heads,
            n_kv_heads,
            ff_dim,
            n_experts: 0,
            n_active: 0,
            eps: hf.rms_norm_eps,
            rope_theta: hf.rope_theta,
            recurrent: false,
        };
        let embed = file.tensor_f32("token_embd.weight")?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(LayerWeights {
                attn_norm: file.tensor_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: file.tensor_f32(&format!("blk.{i}.attn_q.weight"))?,
                wk: file.tensor_f32(&format!("blk.{i}.attn_k.weight"))?,
                wv: file.tensor_f32(&format!("blk.{i}.attn_v.weight"))?,
                wo: file.tensor_f32(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: file.tensor_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                w_gate: file.tensor_f32(&format!("blk.{i}.ffn_gate.weight"))?,
                w_up: file.tensor_f32(&format!("blk.{i}.ffn_up.weight"))?,
                w_down: file.tensor_f32(&format!("blk.{i}.ffn_down.weight"))?,
                w_router: Vec::new(),
                experts: Vec::new(),
            });
        }
        let norm = file.tensor_f32("output_norm.weight")?;
        let lm_head = match file.tensor_f32("output.weight") {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };
        Ok(Self {
            config: cfg,
            embed,
            layers,
            norm,
            lm_head,
            name: "onnx".into(),
            working_set: None,
            expert_lru: None,
            expert_host: None,
            early_exit: EarlyExitSpec::default(),
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        let n: usize = self
            .weight_slices()
            .iter()
            .map(|t| t.len())
            .sum();
        (n * 4) as u64
    }

    fn weight_slices(&self) -> Vec<&[f32]> {
        let mut v = Vec::with_capacity(2 + self.layers.len() * TENSORS_PER_LAYER);
        v.push(self.embed.as_slice());
        for layer in &self.layers {
            v.push(layer.attn_norm.as_slice());
            v.push(layer.wq.as_slice());
            v.push(layer.wk.as_slice());
            v.push(layer.wv.as_slice());
            v.push(layer.wo.as_slice());
            v.push(layer.ffn_norm.as_slice());
            v.push(layer.w_gate.as_slice());
            v.push(layer.w_up.as_slice());
            v.push(layer.w_down.as_slice());
        }
        v.push(self.norm.as_slice());
        v.push(self.lm_head.as_slice());
        v
    }

    /// Decode one token at `pos`, writing K/V into paged tiers. Logits written to `logits`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_token(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        prefill: bool,
        mut recurrent: Option<&mut RecurrentState>,
        micro_batch: Option<&MicroBatchExpertUnion>,
        side: Option<&mut DeviceDecodeSide<'_>>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        if token as usize >= cfg.vocab_size {
            return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                "token {token} out of vocab"
            ))));
        }
        if recurrent.is_none() {
            if let (Some(hip), Some(resident)) = (hip, resident) {
                match self.forward_token_device(
                    token, pos, page_ids, kv, metrics, logits, hip, resident, prefill,
                    micro_batch, side,
                ) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        if !self.host_weights_ready() {
                            return Err(e);
                        }
                        tracing::warn!(pos, token, error = %e, "HIP on-device decode failed; host fallback");
                    }
                }
            }
        }
        if !self.host_weights_ready() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "large GGUF has no host F32 weights; HIP decode is required".into(),
            )));
        }
        let mut x = self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec();
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm_dispatch(
                hip,
                &x,
                &layer.attn_norm,
                resident.map(|r| r.attn_norm(layer_idx)),
                cfg.eps,
                &mut scratch[..dim],
                Some(metrics),
            )?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.wq, resident.map(|r| r.wq(layer_idx)), &mut q, 1, dim, dim)?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.wk, resident.map(|r| r.wk(layer_idx)), &mut k, 1, kv_dim, dim)?;
            gemm_nt_dispatch(hip, &scratch[..dim], &layer.wv, resident.map(|r| r.wv(layer_idx)), &mut v, 1, kv_dim, dim)?;
            rope_dispatch(hip, &mut q, cfg.n_heads, head_dim, pos, cfg.rope_theta, Some(metrics))?;
            rope_dispatch(hip, &mut k, cfg.n_kv_heads, head_dim, pos, cfg.rope_theta, Some(metrics))?;

            let page_tokens = kv
                .get(page_ids[layer_idx][0])
                .map(|p| p.capacity_tokens)
                .unwrap_or(128);
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot(page_id, slot, cfg.n_kv_heads, head_dim, &k, &v)?;
            kv.touch(page_id, 1.0);

            let (k_all, v_all) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_kv = k_all.len() / kv_dim;
            attention_dispatch(
                hip,
                &q,
                &k_all,
                &v_all,
                &mut attn_out,
                1,
                n_kv,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
                true,
            )?;
            gemm_nt_dispatch(hip, &attn_out, &layer.wo, resident.map(|r| r.wo(layer_idx)), &mut hidden, 1, dim, dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }

            self.apply_ffn(
                layer_idx,
                layer,
                &mut x,
                &mut scratch,
                &mut hidden,
                hip,
                resident,
                Some(metrics),
                prefill,
            )?;
            if let Some(state) = recurrent.as_deref_mut() {
                state.mix_layer(layer_idx, &mut x);
            }
        }

        rmsnorm_dispatch(
            hip,
            &x,
            &self.norm,
            resident.map(|r| r.norm()),
            cfg.eps,
            &mut scratch[..dim],
            Some(metrics),
        )?;
        if logits.len() < cfg.vocab_size {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        gemm_nt_dispatch(
            hip,
            &scratch[..dim],
            &self.lm_head,
            resident.map(|r| r.lm_head()),
            &mut logits[..cfg.vocab_size],
            1,
            cfg.vocab_size,
            dim,
        )?;
        if hip.is_some() {
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .target_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Prefill a token chunk. HIP runs one forward with M=N (hipBLASLt when N>8 on F32).
    /// Decode stays on [`forward_token`]. CPU loops per token. Last-chunk logits are
    /// written for the last position only when `write_logits` is set. Packed linear
    /// verify passes `row_logits` of length `tokens.len() * vocab` to keep every row.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens(
        &self,
        tokens: &[u32],
        start_pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        write_logits: bool,
        mut row_logits: Option<&mut [f32]>,
        mut recurrent: Option<&mut RecurrentState>,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if tokens.len() == 1 {
            self.forward_token(
                tokens[0],
                start_pos,
                page_ids,
                kv,
                metrics,
                logits,
                hip,
                resident,
                false,
                recurrent,
                None,
                None,
            )?;
            if let Some(rows) = row_logits {
                let vocab = self.config.vocab_size;
                if rows.len() >= vocab {
                    rows[..vocab].copy_from_slice(&logits[..vocab]);
                }
            }
            return Ok(());
        }
        for tok in tokens {
            if *tok as usize >= self.config.vocab_size {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                    "token {tok} out of vocab"
                ))));
            }
        }
        if recurrent.is_none() && self.config.n_experts == 0 {
            if let (Some(hip), Some(resident)) = (hip, resident) {
            match self.forward_tokens_device(
                tokens,
                start_pos,
                page_ids,
                kv,
                metrics,
                logits,
                hip,
                resident,
                write_logits,
                row_logits.as_deref_mut(),
            ) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if !self.host_weights_ready() {
                        return Err(e);
                    }
                    tracing::warn!(
                        start_pos,
                        n = tokens.len(),
                        error = %e,
                        "HIP batched prefill failed; per-token host fallback"
                    );
                }
            }
            }
        }
        let vocab = self.config.vocab_size;
        for (i, tok) in tokens.iter().enumerate() {
            self.forward_token(
                *tok,
                start_pos + i,
                page_ids,
                kv,
                metrics,
                logits,
                hip,
                resident,
                true,
                recurrent.as_deref_mut(),
                None,
                None,
            )?;
            if let Some(rows) = row_logits.as_mut() {
                let off = i * vocab;
                if rows.len() >= off + vocab {
                    rows[off..off + vocab].copy_from_slice(&logits[..vocab]);
                }
            }
        }
        Ok(())
    }

    /// Sequential target decode with device-resident hidden state and KV.
    #[allow(clippy::too_many_arguments)]
    fn forward_token_device(
        &self,
        token: u32,
        pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        prefill: bool,
        micro_batch: Option<&MicroBatchExpertUnion>,
        mut side: Option<&mut DeviceDecodeSide<'_>>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        let page_tokens = page_ids
            .first()
            .and_then(|p| p.first())
            .and_then(|id| kv.get(*id).map(|p| p.capacity_tokens as usize))
            .unwrap_or(128);
        let max_kv = page_ids
            .iter()
            .map(|layer| layer.len().saturating_mul(page_tokens))
            .max()
            .unwrap_or(pos + 1)
            .max(pos + 1);
        hip.bind()?;
        let buf = hip.ensure_decode(dim, kv_dim, cfg.ff_dim, vocab, max_kv)?;
        hip.begin_decode_profile();
        let n_exp = cfg.n_experts;
        let double_buffer = prefill
            && n_exp > 0
            && self
                .expert_lru
                .as_ref()
                .is_some_and(|lru| lru.holds_layers(n_exp, 2));
        if double_buffer && !resident.prefill_layer_gpu_ready(0) {
            if let Ok(experts) = self.host_experts_layer(0) {
                let _ = resident.prefill_warm_layer_sync(0, n_exp as u32, &experts, Some(metrics));
            }
        }
        let result = (|| {
        resident.embed_gather(hip, token, buf.x, dim as i32)?;
        record_elementwise(Some(metrics));

        for (layer_idx, _) in self.layers.iter().enumerate() {
            if double_buffer {
                resident.prefill_wait_layer(layer_idx as u32)?;
                if layer_idx + 1 < self.layers.len() {
                    if let Ok(next) = self.host_experts_layer(layer_idx + 1) {
                        let _ = resident.prefill_kickoff_layer(
                            (layer_idx + 1) as u32,
                            n_exp as u32,
                            &next,
                            Some(metrics),
                        );
                    }
                }
            }
            let layer = &resident.layers[layer_idx];
            if layer_idx == 0 {
                hip.rmsnorm_device(
                    buf.x,
                    resident.f32_or_dequant(hip, layer.attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
            }
            if layer.wq.kind == WeightKind::Q8_0 {
                hip.profile_q8(|| {
                    resident.gemm_nt(hip, buf.normed, layer.wq, buf.q, 1, dim as i32, dim as i32)?;
                    resident.gemm_nt(hip, buf.normed, layer.wk, buf.k, 1, kv_dim as i32, dim as i32)?;
                    resident.gemm_nt(hip, buf.normed, layer.wv, buf.v, 1, kv_dim as i32, dim as i32)?;
                    Ok(())
                })?;
            } else {
                resident.gemm_nt(hip, buf.normed, layer.wq, buf.q, 1, dim as i32, dim as i32)?;
                resident.gemm_nt(hip, buf.normed, layer.wk, buf.k, 1, kv_dim as i32, dim as i32)?;
                resident.gemm_nt(hip, buf.normed, layer.wv, buf.v, 1, kv_dim as i32, dim as i32)?;
            }
            hip.rope_qk_device(
                buf.q,
                buf.k,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            record_elementwise(Some(metrics));

            let page_index = pos / page_tokens;
            let slot = (pos % page_tokens) as u32;
            let page_id = page_ids[layer_idx][page_index];
            kv.write_slot_from_device(page_id, slot, cfg.n_kv_heads, head_dim, buf.k, buf.v)?;
            kv.touch(page_id, 1.0);

            let (packed_k, packed_v, n_kv) = kv.kv_device_views(
                &page_ids[layer_idx],
                cfg.n_kv_heads,
                head_dim,
                buf.packed_k,
                buf.packed_v,
                buf.max_kv,
            )?;
            hip.attention_device(
                buf.q,
                packed_k,
                packed_v,
                buf.attn,
                1,
                n_kv as i32,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                true,
            )?;
            if layer.wo.kind == WeightKind::Q8_0 {
                hip.profile_q8(|| {
                    resident.gemm_nt(hip, buf.attn, layer.wo, buf.proj, 1, dim as i32, dim as i32)
                })?;
            } else {
                resident.gemm_nt(hip, buf.attn, layer.wo, buf.proj, 1, dim as i32, dim as i32)?;
            }
            if self.config.n_experts > 0 {
                if let Some(s) = side.as_mut() {
                    self.device_moe(
                        layer_idx,
                        hip,
                        resident,
                        buf,
                        metrics,
                        prefill,
                        micro_batch,
                        Some(s),
                    )?;
                } else {
                    self.device_moe(
                        layer_idx,
                        hip,
                        resident,
                        buf,
                        metrics,
                        prefill,
                        micro_batch,
                        None,
                    )?;
                }
            } else {
            let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
            let ffn_norm = ffn.ffn_norm();
            let w_gate = ffn.w_gate();
            let w_up = ffn.w_up();
            let w_down = ffn.w_down();
            hip.residual_rmsnorm(
                buf.x,
                buf.proj,
                resident.f32_or_dequant(hip, ffn_norm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            metrics
                .residual_gpu_ops
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            record_elementwise(Some(metrics));
            if w_gate.kind == WeightKind::Q8_0 {
                hip.profile_q8(|| {
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_gate,
                        buf.gate,
                        1,
                        cfg.ff_dim as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_up,
                        buf.up,
                        1,
                        cfg.ff_dim as i32,
                        dim as i32,
                    )?;
                    Ok(())
                })?;
            } else {
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_gate,
                    buf.gate,
                    1,
                    cfg.ff_dim as i32,
                    dim as i32,
                )?;
                resident.gemm_nt(
                    hip,
                    buf.normed,
                    w_up,
                    buf.up,
                    1,
                    cfg.ff_dim as i32,
                    dim as i32,
                )?;
            }
            hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
            record_elementwise(Some(metrics));
            if w_down.kind == WeightKind::Q8_0 {
                hip.profile_q8(|| {
                    resident.gemm_nt(
                        hip,
                        buf.gate,
                        w_down,
                        buf.proj,
                        1,
                        dim as i32,
                        cfg.ff_dim as i32,
                    )
                })?;
            } else {
                resident.gemm_nt(
                    hip,
                    buf.gate,
                    w_down,
                    buf.proj,
                    1,
                    dim as i32,
                    cfg.ff_dim as i32,
                )?;
            }
            drop(ffn);
            }
            if layer_idx + 1 < resident.layers.len() {
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                hip.residual_add(buf.x, buf.proj, dim as i32)?;
            }
            metrics
                .residual_gpu_ops
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            record_elementwise(Some(metrics));
        }

        hip.rmsnorm_device(
            buf.x,
            resident.f32_or_dequant(hip, resident.norm)?,
            buf.normed,
            dim as i32,
            cfg.eps,
        )?;
        record_elementwise(Some(metrics));
        {
            let _lm = hip.lm_head_scope();
            resident.gemm_nt(
                hip,
                buf.normed,
                resident.lm_head,
                buf.logits,
                1,
                vocab as i32,
                dim as i32,
            )?;
        }
        hip.start_logits_d2h(buf.logits, vocab)?;
        metrics
            .gpu_forwards
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        metrics
            .target_gpu_forwards
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
        })();
        hip.end_decode_profile();
        result
    }

    /// Batched HIP prefill: one GEMM with m = N per weight, attention n_q = N.
    #[allow(clippy::too_many_arguments)]
    fn forward_tokens_device(
        &self,
        tokens: &[u32],
        start_pos: usize,
        page_ids: &[Vec<u64>],
        kv: &mut KvTierManager,
        metrics: &Metrics,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        write_logits: bool,
        row_logits: Option<&mut [f32]>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        let n_tok = tokens.len();
        if write_logits && logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        if let Some(rows) = row_logits.as_ref() {
            if rows.len() < n_tok.saturating_mul(vocab) {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "row logits buffer smaller than n_tok * vocab".into(),
                )));
            }
        }
        let page_tokens = page_ids
            .first()
            .and_then(|p| p.first())
            .and_then(|id| kv.get(*id).map(|p| p.capacity_tokens as usize))
            .unwrap_or(128);
        let seq_end = start_pos + n_tok;
        let max_kv = page_ids
            .iter()
            .map(|layer| layer.len().saturating_mul(page_tokens))
            .max()
            .unwrap_or(seq_end)
            .max(seq_end);
        hip.bind()?;
        let buf = hip.ensure_decode_batch(
            dim,
            kv_dim,
            cfg.ff_dim,
            vocab,
            max_kv,
            n_tok,
            if row_logits.is_some() { n_tok } else { 1 },
        )?;
        hip.begin_decode_profile();
        let result = (|| {
            resident.embed_gather_rows(hip, tokens, buf.x, dim as i32)?;
            record_elementwise(Some(metrics));

            let m = n_tok as i32;
            for (layer_idx, _) in self.layers.iter().enumerate() {
                let layer = &resident.layers[layer_idx];
                if layer_idx == 0 {
                    rmsnorm_device_rows(
                        hip,
                        buf.x,
                        resident.f32_or_dequant(hip, layer.attn_norm)?,
                        buf.normed,
                        n_tok,
                        dim,
                        cfg.eps,
                    )?;
                    record_elementwise(Some(metrics));
                }
                if layer.wq.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(hip, buf.normed, layer.wq, buf.q, m, dim as i32, dim as i32)?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wk,
                            buf.k,
                            m,
                            kv_dim as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            layer.wv,
                            buf.v,
                            m,
                            kv_dim as i32,
                            dim as i32,
                        )?;
                        Ok(())
                    })?;
                } else {
                    resident.gemm_nt(hip, buf.normed, layer.wq, buf.q, m, dim as i32, dim as i32)?;
                    resident.gemm_nt(hip, buf.normed, layer.wk, buf.k, m, kv_dim as i32, dim as i32)?;
                    resident.gemm_nt(hip, buf.normed, layer.wv, buf.v, m, kv_dim as i32, dim as i32)?;
                }
                hip.rope_qk_device_rows(
                    buf.q,
                    buf.k,
                    n_tok as i32,
                    cfg.n_heads as i32,
                    cfg.n_kv_heads as i32,
                    head_dim as i32,
                    start_pos as i32,
                    cfg.rope_theta,
                )?;
                record_elementwise(Some(metrics));

                kv.write_slots_from_device(
                    &page_ids[layer_idx],
                    start_pos,
                    n_tok,
                    page_tokens,
                    cfg.n_kv_heads,
                    head_dim,
                    buf.k,
                    buf.v,
                )?;

                let (packed_k, packed_v, n_kv) = kv.kv_device_views(
                    &page_ids[layer_idx],
                    cfg.n_kv_heads,
                    head_dim,
                    buf.packed_k,
                    buf.packed_v,
                    buf.max_kv,
                )?;
                hip.attention_device(
                    buf.q,
                    packed_k,
                    packed_v,
                    buf.attn,
                    m,
                    n_kv as i32,
                    cfg.n_heads as i32,
                    cfg.n_kv_heads as i32,
                    head_dim as i32,
                    true,
                )?;
                if layer.wo.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(hip, buf.attn, layer.wo, buf.proj, m, dim as i32, dim as i32)
                    })?;
                } else {
                    resident.gemm_nt(hip, buf.attn, layer.wo, buf.proj, m, dim as i32, dim as i32)?;
                }
                hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
                let ffn = self.device_ffn(resident, layer_idx, Some(metrics))?;
                let ffn_norm = ffn.ffn_norm();
                let w_gate = ffn.w_gate();
                let w_up = ffn.w_up();
                let w_down = ffn.w_down();
                rmsnorm_device_rows(
                    hip,
                    buf.x,
                    resident.f32_or_dequant(hip, ffn_norm)?,
                    buf.normed,
                    n_tok,
                    dim,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
                if w_gate.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_gate,
                            buf.gate,
                            m,
                            cfg.ff_dim as i32,
                            dim as i32,
                        )?;
                        resident.gemm_nt(
                            hip,
                            buf.normed,
                            w_up,
                            buf.up,
                            m,
                            cfg.ff_dim as i32,
                            dim as i32,
                        )?;
                        Ok(())
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_gate,
                        buf.gate,
                        m,
                        cfg.ff_dim as i32,
                        dim as i32,
                    )?;
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        w_up,
                        buf.up,
                        m,
                        cfg.ff_dim as i32,
                        dim as i32,
                    )?;
                }
                hip.silu_mul_device(buf.gate, buf.up, (n_tok * cfg.ff_dim) as i32)?;
                record_elementwise(Some(metrics));
                if w_down.kind == WeightKind::Q8_0 {
                    hip.profile_q8(|| {
                        resident.gemm_nt(
                            hip,
                            buf.gate,
                            w_down,
                            buf.proj,
                            m,
                            dim as i32,
                            cfg.ff_dim as i32,
                        )
                    })?;
                } else {
                    resident.gemm_nt(
                        hip,
                        buf.gate,
                        w_down,
                        buf.proj,
                        m,
                        dim as i32,
                        cfg.ff_dim as i32,
                    )?;
                }
                drop(ffn);
                hip.residual_add(buf.x, buf.proj, (n_tok * dim) as i32)?;
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                record_elementwise(Some(metrics));
                if layer_idx + 1 < resident.layers.len() {
                    rmsnorm_device_rows(
                        hip,
                        buf.x,
                        resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                        buf.normed,
                        n_tok,
                        dim,
                        cfg.eps,
                    )?;
                    record_elementwise(Some(metrics));
                }
            }

            if row_logits.is_some() {
                rmsnorm_device_rows(
                    hip,
                    buf.x,
                    resident.f32_or_dequant(hip, resident.norm)?,
                    buf.normed,
                    n_tok,
                    dim,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
                {
                    let _lm = hip.lm_head_scope();
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        resident.lm_head,
                        buf.logits,
                        n_tok as i32,
                        vocab as i32,
                        dim as i32,
                    )?;
                }
                hip.start_logits_d2h(buf.logits, n_tok * vocab)?;
            } else if write_logits {
                let x_last = buf.x_row(n_tok - 1, dim);
                hip.rmsnorm_device(
                    x_last,
                    resident.f32_or_dequant(hip, resident.norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                record_elementwise(Some(metrics));
                {
                    let _lm = hip.lm_head_scope();
                    resident.gemm_nt(
                        hip,
                        buf.normed,
                        resident.lm_head,
                        buf.logits,
                        1,
                        vocab as i32,
                        dim as i32,
                    )?;
                }
                hip.start_logits_d2h(buf.logits, vocab)?;
            }
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .target_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        })();
        hip.end_decode_profile();
        result
    }

    /// Packed Token Forest verify: one target eval with an ancestry attention mask.
    /// When `hip` is present the ancestry mask runs on the target GPU.
    pub fn forward_forest(
        &self,
        forest: &TokenForest,
        page_ids: &[Vec<u64>],
        kv: &KvTierManager,
        prompt_len: usize,
        node_logits: &mut [Vec<f32>],
        hip: Option<&HipWorkspace>,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let n_tree = forest.nodes.len();
        if n_tree == 0 {
            return Ok(());
        }
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(n_tree);
        for n in &forest.nodes {
            let tok = n.token_id as usize;
            if tok >= cfg.vocab_size {
                return Err(AllpacaError::Fatal(FatalError::Tokenizer(format!(
                    "token {tok} out of vocab"
                ))));
            }
            xs.push(self.embed[tok * dim..(tok + 1) * dim].to_vec());
        }
        let mut scratch = vec![0.0f32; dim.max(cfg.ff_dim).max(kv_dim)];
        let ancestry = forest.ancestry_mask();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (prompt_k, prompt_v) = gather_kv(kv, &page_ids[layer_idx], cfg.n_kv_heads, head_dim)?;
            let n_prompt = prompt_k.len().checked_div(kv_dim).unwrap_or(0);
            let n_kv = n_prompt + n_tree;
            let mut qs = vec![0.0f32; n_tree * dim];
            let mut ks = vec![0.0f32; n_tree * kv_dim];
            let mut vs = vec![0.0f32; n_tree * kv_dim];
            for (i, x) in xs.iter().enumerate() {
                rmsnorm(x, &layer.attn_norm, cfg.eps, &mut scratch[..dim])?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wq,
                    &mut qs[i * dim..(i + 1) * dim],
                    1,
                    dim,
                    dim,
                )?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wk,
                    &mut ks[i * kv_dim..(i + 1) * kv_dim],
                    1,
                    kv_dim,
                    dim,
                )?;
                matmul_nt(
                    &scratch[..dim],
                    &layer.wv,
                    &mut vs[i * kv_dim..(i + 1) * kv_dim],
                    1,
                    kv_dim,
                    dim,
                )?;
                let pos = prompt_len + forest.nodes[i].depth as usize - 1;
                for h in 0..cfg.n_heads {
                    rope(&mut qs[i * dim + h * head_dim..i * dim + (h + 1) * head_dim], pos, cfg.rope_theta)?;
                }
                for h in 0..cfg.n_kv_heads {
                    rope(
                        &mut ks[i * kv_dim + h * head_dim..i * kv_dim + (h + 1) * head_dim],
                        pos,
                        cfg.rope_theta,
                    )?;
                }
            }
            let mut k_all = prompt_k;
            k_all.extend_from_slice(&ks);
            let mut v_all = prompt_v;
            v_all.extend_from_slice(&vs);
            let mut mask = vec![false; n_tree * n_kv];
            for i in 0..n_tree {
                for t in 0..n_prompt {
                    mask[i * n_kv + t] = true;
                }
                for j in 0..n_tree {
                    if ancestry[i * n_tree + j] {
                        mask[i * n_kv + n_prompt + j] = true;
                    }
                }
            }
            let mut attn_out = vec![0.0f32; n_tree * dim];
            let mut gpu_mask = false;
            if let Some(hip) = hip {
                let mask_u8: Vec<u8> = mask.iter().map(|b| u8::from(*b)).collect();
                if hip
                    .attention_masked(
                        &qs,
                        &k_all,
                        &v_all,
                        &mut attn_out,
                        n_tree as i32,
                        n_kv as i32,
                        cfg.n_heads as i32,
                        cfg.n_kv_heads as i32,
                        head_dim as i32,
                        &mask_u8,
                    )
                    .is_ok()
                {
                    gpu_mask = true;
                    if let Some(m) = metrics {
                        m.tree_gpu_verifies
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
            if !gpu_mask {
                attention_with_mask(
                    &qs,
                    &k_all,
                    &v_all,
                    &mut attn_out,
                    n_tree,
                    n_kv,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    head_dim,
                    &mask,
                )?;
            }
            let ffn_scratch = &mut scratch;
            for i in 0..n_tree {
                let mut hidden = vec![0.0f32; dim];
                matmul_nt(
                    &attn_out[i * dim..(i + 1) * dim],
                    &layer.wo,
                    &mut hidden,
                    1,
                    dim,
                    dim,
                )?;
                for d in 0..dim {
                    xs[i][d] += hidden[d];
                }
                self.apply_ffn(
                    layer_idx,
                    layer,
                    &mut xs[i],
                    ffn_scratch,
                    &mut hidden,
                    None,
                    None,
                    metrics,
                    false,
                )?;
            }
        }

        for (i, x) in xs.iter().enumerate() {
            rmsnorm(x, &self.norm, cfg.eps, &mut scratch[..dim])?;
            if node_logits[i].len() < vocab {
                node_logits[i].resize(vocab, 0.0);
            }
            matmul_nt(
                &scratch[..dim],
                &self.lm_head,
                &mut node_logits[i][..vocab],
                1,
                vocab,
                dim,
            )?;
        }
        Ok(())
    }

    /// Draft-path forward with a dense (non-paged) KV cache.
    pub fn forward_dense(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
    ) -> Result<(), AllpacaError> {
        self.forward_dense_on(token, pos, cache, logits, None, None, None)
    }

    /// Draft forward. GEMM/attention run on `hip` when present (resident draft GPU).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_dense_on(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
        hip: Option<&HipWorkspace>,
        resident: Option<&ResidentModel>,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        cache.forwards = cache.forwards.saturating_add(1);
        if self.config.n_experts == 0 {
            if let (Some(hip), Some(resident)) = (hip, resident) {
                match self.forward_dense_device(token, pos, cache, logits, hip, resident, metrics) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        tracing::warn!(pos, token, error = %e, "HIP draft on-device decode failed; host fallback");
                    }
                }
            }
        }
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        cache.ensure_layers(cfg.n_layers);
        let mut x = self.embed[token as usize * dim..(token as usize + 1) * dim].to_vec();
        let mut normed = vec![0.0f32; dim];
        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut hidden = vec![0.0f32; dim];
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm_dispatch(
                hip,
                &x,
                &layer.attn_norm,
                resident.map(|r| r.attn_norm(layer_idx)),
                cfg.eps,
                &mut normed,
                metrics,
            )?;
            gemm_nt_dispatch(hip, &normed, &layer.wq, resident.map(|r| r.wq(layer_idx)), &mut q, 1, dim, dim)?;
            gemm_nt_dispatch(hip, &normed, &layer.wk, resident.map(|r| r.wk(layer_idx)), &mut k, 1, kv_dim, dim)?;
            gemm_nt_dispatch(hip, &normed, &layer.wv, resident.map(|r| r.wv(layer_idx)), &mut v, 1, kv_dim, dim)?;
            rope_dispatch(hip, &mut q, cfg.n_heads, head_dim, pos, cfg.rope_theta, metrics)?;
            rope_dispatch(hip, &mut k, cfg.n_kv_heads, head_dim, pos, cfg.rope_theta, metrics)?;
            cache.layers[layer_idx].k.extend_from_slice(&k);
            cache.layers[layer_idx].v.extend_from_slice(&v);
            let n_kv = cache.layers[layer_idx].k.len() / kv_dim;
            attention_dispatch(
                hip,
                &q,
                &cache.layers[layer_idx].k,
                &cache.layers[layer_idx].v,
                &mut attn_out,
                1,
                n_kv,
                cfg.n_heads,
                cfg.n_kv_heads,
                head_dim,
                true,
            )?;
            gemm_nt_dispatch(hip, &attn_out, &layer.wo, resident.map(|r| r.wo(layer_idx)), &mut hidden, 1, dim, dim)?;
            for i in 0..dim {
                x[i] += hidden[i];
            }
            self.apply_ffn(
                layer_idx,
                layer,
                &mut x,
                &mut normed,
                &mut hidden,
                hip,
                resident,
                metrics,
                false,
            )?;
        }
        rmsnorm_dispatch(
            hip,
            &x,
            &self.norm,
            resident.map(|r| r.norm()),
            cfg.eps,
            &mut normed,
            metrics,
        )?;
        gemm_nt_dispatch(
            hip,
            &normed,
            &self.lm_head,
            resident.map(|r| r.lm_head()),
            &mut logits[..cfg.vocab_size],
            1,
            cfg.vocab_size,
            dim,
        )?;
        if hip.is_some() {
            if let Some(metrics) = metrics {
                metrics
                    .gpu_forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                metrics
                    .draft_gpu_forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Sequential draft decode with device-resident hidden state and DenseKv.
    #[allow(clippy::too_many_arguments)]
    fn forward_dense_device(
        &self,
        token: u32,
        pos: usize,
        cache: &mut DenseKv,
        logits: &mut [f32],
        hip: &HipWorkspace,
        resident: &ResidentModel,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_kv_heads * head_dim;
        let vocab = cfg.vocab_size;
        if logits.len() < vocab {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "logits buffer smaller than vocab".into(),
            )));
        }
        hip.bind()?;
        cache.ensure_device(hip.device, cfg.n_layers, kv_dim, pos + 1)?;
        if let Some(metrics) = metrics {
            metrics
                .draft_kv_bytes_on_device
                .store(cache.device_bytes(), std::sync::atomic::Ordering::Relaxed);
        }
        let buf = hip.ensure_decode(dim, kv_dim, cfg.ff_dim, vocab, pos + 1)?;
        resident.embed_gather(hip, token, buf.x, dim as i32)?;
        record_elementwise(metrics);

        // Same dtype dispatch as target decode (`resident.gemm_nt`). F32 GEMV on
        // packed F16/Q8 reads past the weight allocation and hangs the draft GPU.
        for layer_idx in 0..self.layers.len() {
            let layer = &resident.layers[layer_idx];
            if layer_idx == 0 {
                hip.rmsnorm_device(
                    buf.x,
                    resident.f32_or_dequant(hip, layer.attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
                record_elementwise(metrics);
            }
            resident.gemm_nt(hip, buf.normed, layer.wq, buf.q, 1, dim as i32, dim as i32)?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.wk,
                buf.k,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                layer.wv,
                buf.v,
                1,
                kv_dim as i32,
                dim as i32,
            )?;
            hip.rope_qk_device(
                buf.q,
                buf.k,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                pos as i32,
                cfg.rope_theta,
            )?;
            record_elementwise(metrics);
            cache.write_pos_from_device(layer_idx, pos, buf.k, buf.v)?;
            let n_kv = pos + 1;
            hip.attention_device(
                buf.q,
                cache.k_ptr(layer_idx),
                cache.v_ptr(layer_idx),
                buf.attn,
                1,
                n_kv as i32,
                cfg.n_heads as i32,
                cfg.n_kv_heads as i32,
                head_dim as i32,
                true,
            )?;
            resident.gemm_nt(
                hip,
                buf.attn,
                layer.wo,
                buf.proj,
                1,
                dim as i32,
                dim as i32,
            )?;
            let ffn = self.device_ffn(resident, layer_idx, metrics)?;
            let ffn_norm = ffn.ffn_norm();
            let w_gate = ffn.w_gate();
            let w_up = ffn.w_up();
            let w_down = ffn.w_down();
            hip.residual_rmsnorm(
                buf.x,
                buf.proj,
                resident.f32_or_dequant(hip, ffn_norm)?,
                buf.normed,
                dim as i32,
                cfg.eps,
            )?;
            if let Some(metrics) = metrics {
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            record_elementwise(metrics);
            resident.gemm_nt(
                hip,
                buf.normed,
                w_gate,
                buf.gate,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            resident.gemm_nt(
                hip,
                buf.normed,
                w_up,
                buf.up,
                1,
                cfg.ff_dim as i32,
                dim as i32,
            )?;
            hip.silu_mul_device(buf.gate, buf.up, cfg.ff_dim as i32)?;
            record_elementwise(metrics);
            resident.gemm_nt(
                hip,
                buf.gate,
                w_down,
                buf.proj,
                1,
                dim as i32,
                cfg.ff_dim as i32,
            )?;
            drop(ffn);
            if layer_idx + 1 < resident.layers.len() {
                hip.residual_rmsnorm(
                    buf.x,
                    buf.proj,
                    resident.f32_or_dequant(hip, resident.layers[layer_idx + 1].attn_norm)?,
                    buf.normed,
                    dim as i32,
                    cfg.eps,
                )?;
            } else {
                hip.residual_add(buf.x, buf.proj, dim as i32)?;
            }
            if let Some(metrics) = metrics {
                metrics
                    .residual_gpu_ops
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            record_elementwise(metrics);
        }

        hip.rmsnorm_device(
            buf.x,
            resident.f32_or_dequant(hip, resident.norm)?,
            buf.normed,
            dim as i32,
            cfg.eps,
        )?;
        record_elementwise(metrics);
        {
            let _lm = hip.lm_head_scope();
            resident.gemm_nt(
                hip,
                buf.normed,
                resident.lm_head,
                buf.logits,
                1,
                vocab as i32,
                dim as i32,
            )?;
        }
        hip.publish_logits(buf.logits, &mut logits[..vocab])?;
        if let Some(metrics) = metrics {
            metrics
                .gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            metrics
                .draft_gpu_forwards
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }
}

struct DeviceLayerKv {
    k: *mut u8,
    v: *mut u8,
    bytes: usize,
}

unsafe impl Send for DeviceLayerKv {}
unsafe impl Sync for DeviceLayerKv {}

impl std::fmt::Debug for DeviceLayerKv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceLayerKv")
            .field("bytes", &self.bytes)
            .finish()
    }
}

#[derive(Debug)]
pub struct DenseKv {
    pub layers: Vec<DenseLayerKv>,
    device: Option<i32>,
    device_layers: Vec<DeviceLayerKv>,
    tokens: usize,
    capacity: usize,
    kv_dim: usize,
    seq: Vec<u32>,
    /// Draft decode steps on this cache (CPU or HIP). Locks prefix reuse.
    pub(crate) forwards: u64,
}

impl Default for DenseKv {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            device: None,
            device_layers: Vec::new(),
            tokens: 0,
            capacity: 0,
            kv_dim: 0,
            seq: Vec::new(),
            forwards: 0,
        }
    }
}

impl Drop for DenseKv {
    fn drop(&mut self) {
        self.free_device();
    }
}

#[derive(Debug, Default, Clone)]
pub struct DenseLayerKv {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

impl DenseKv {
    fn ensure_layers(&mut self, n: usize) {
        self.layers.resize_with(n, DenseLayerKv::default);
    }

    pub fn device_bytes(&self) -> u64 {
        self.device_layers.iter().map(|l| (l.bytes * 2) as u64).sum()
    }

    pub fn reset(&mut self) {
        self.tokens = 0;
        self.seq.clear();
        for layer in &mut self.layers {
            layer.k.clear();
            layer.v.clear();
        }
    }

    pub(crate) fn prefix_len(&self, ctx: &[u32]) -> usize {
        self.seq
            .iter()
            .zip(ctx)
            .take_while(|(a, b)| *a == *b)
            .count()
    }

    pub(crate) fn truncate(&mut self, n: usize, kv_dim: usize) {
        self.rewind_to(n, kv_dim);
        self.seq.truncate(n);
    }

    pub(crate) fn record(&mut self, token: u32) {
        self.seq.push(token);
    }

    fn free_device(&mut self) {
        for layer in self.device_layers.drain(..) {
            allpaca_hip::device_free(layer.k, layer.bytes);
            allpaca_hip::device_free(layer.v, layer.bytes);
        }
        self.device = None;
        self.capacity = 0;
    }

    fn ensure_device(
        &mut self,
        device: i32,
        n_layers: usize,
        kv_dim: usize,
        min_tokens: usize,
    ) -> Result<(), AllpacaError> {
        allpaca_hip::set_device(device)?;
        self.ensure_layers(n_layers);
        let need = min_tokens.max(1);
        let reuse = self.device == Some(device)
            && self.kv_dim == kv_dim
            && self.device_layers.len() == n_layers
            && self.capacity >= need;
        if reuse {
            return Ok(());
        }
        let new_cap = need.max(self.capacity.saturating_mul(2)).max(8);
        let bytes = new_cap * kv_dim * 4;
        let copy_tokens = if self.device == Some(device) && self.kv_dim == kv_dim {
            self.tokens
        } else {
            0
        };
        let copy_bytes = copy_tokens * kv_dim * 4;
        let mut new_layers: Vec<DeviceLayerKv> = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let k = match allpaca_hip::device_malloc(bytes, device) {
                Ok(p) => p,
                Err(e) => {
                    for layer in new_layers.drain(..) {
                        allpaca_hip::device_free(layer.k, layer.bytes);
                        allpaca_hip::device_free(layer.v, layer.bytes);
                    }
                    return Err(e);
                }
            };
            let v = match allpaca_hip::device_malloc(bytes, device) {
                Ok(p) => p,
                Err(e) => {
                    allpaca_hip::device_free(k, bytes);
                    for layer in new_layers.drain(..) {
                        allpaca_hip::device_free(layer.k, layer.bytes);
                        allpaca_hip::device_free(layer.v, layer.bytes);
                    }
                    return Err(e);
                }
            };
            if copy_bytes > 0 && i < self.device_layers.len() {
                allpaca_hip::memcpy(k, self.device_layers[i].k, copy_bytes, allpaca_hip::MEMCPY_D2D)?;
                allpaca_hip::memcpy(v, self.device_layers[i].v, copy_bytes, allpaca_hip::MEMCPY_D2D)?;
            }
            new_layers.push(DeviceLayerKv { k, v, bytes });
        }
        self.free_device();
        self.device_layers = new_layers;
        self.device = Some(device);
        self.capacity = new_cap;
        self.kv_dim = kv_dim;
        Ok(())
    }

    fn write_pos_from_device(
        &mut self,
        layer: usize,
        pos: usize,
        k_dev: *const f32,
        v_dev: *const f32,
    ) -> Result<(), AllpacaError> {
        if layer >= self.device_layers.len() || k_dev.is_null() || v_dev.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "draft device KV pointer is null".into(),
            )));
        }
        if pos >= self.capacity || self.kv_dim == 0 {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "draft device KV capacity exceeded".into(),
            )));
        }
        let kv_bytes = self.kv_dim * 4;
        let off = pos * kv_bytes;
        let dest_k = unsafe { self.device_layers[layer].k.add(off) };
        let dest_v = unsafe { self.device_layers[layer].v.add(off) };
        allpaca_hip::memcpy(dest_k, k_dev.cast(), kv_bytes, allpaca_hip::MEMCPY_D2D)?;
        allpaca_hip::memcpy(dest_v, v_dev.cast(), kv_bytes, allpaca_hip::MEMCPY_D2D)?;
        self.tokens = self.tokens.max(pos + 1);
        Ok(())
    }

    fn k_ptr(&self, layer: usize) -> *const f32 {
        self.device_layers
            .get(layer)
            .map(|l| l.k.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null())
    }

    fn v_ptr(&self, layer: usize) -> *const f32 {
        self.device_layers
            .get(layer)
            .map(|l| l.v.cast::<f32>() as *const f32)
            .unwrap_or(std::ptr::null())
    }

    pub fn rewind_to(&mut self, tokens: usize, kv_dim: usize) {
        for layer in &mut self.layers {
            let keep = tokens * kv_dim;
            layer.k.truncate(keep);
            layer.v.truncate(keep);
        }
        self.tokens = tokens;
    }
}

fn record_elementwise(metrics: Option<&Metrics>) {
    if let Some(metrics) = metrics {
        metrics
            .elementwise_gpu_ops
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn rmsnorm_device_rows(
    hip: &HipWorkspace,
    x: *const f32,
    weight: *const f32,
    out: *mut f32,
    n_tok: usize,
    dim: usize,
    eps: f32,
) -> Result<(), AllpacaError> {
    hip.rmsnorm_device_rows(x, weight, out, n_tok as i32, dim as i32, eps)
}

fn rmsnorm_dispatch(
    hip: Option<&HipWorkspace>,
    x: &[f32],
    weight: &[f32],
    weight_dev: Option<*const f32>,
    eps: f32,
    out: &mut [f32],
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip.rmsnorm(x, weight, weight_dev, eps, out).is_ok() {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    rmsnorm(x, weight, eps, out)
}

fn rope_dispatch(
    hip: Option<&HipWorkspace>,
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip
            .rope(x, n_heads as i32, head_dim as i32, pos as i32, theta)
            .is_ok()
        {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    for h in 0..n_heads {
        rope(&mut x[h * head_dim..(h + 1) * head_dim], pos, theta)?;
    }
    Ok(())
}

fn silu_mul_dispatch(
    hip: Option<&HipWorkspace>,
    gate: &mut [f32],
    up: &[f32],
    metrics: Option<&Metrics>,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip.silu_mul(gate, up).is_ok() {
            record_elementwise(metrics);
            return Ok(());
        }
    }
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gemm_nt_dispatch(
    hip: Option<&HipWorkspace>,
    a: &[f32],
    b: &[f32],
    b_dev: Option<*const f32>,
    c: &mut [f32],
    rows: usize,
    n: usize,
    k: usize,
) -> Result<(), AllpacaError> {
    if let (Some(hip), Some(b_dev)) = (hip, b_dev) {
        if hip
            .gemm_nt_dev_b(a, b_dev, c, rows as i32, n as i32, k as i32)
            .is_ok()
        {
            return Ok(());
        }
    }
    if let Some(hip) = hip {
        if hip
            .gemm_nt(a, b, c, rows as i32, n as i32, k as i32)
            .is_ok()
        {
            return Ok(());
        }
    }
    matmul_nt(a, b, c, rows, n, k)
}

#[allow(clippy::too_many_arguments)]
fn attention_dispatch(
    hip: Option<&HipWorkspace>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Result<(), AllpacaError> {
    if let Some(hip) = hip {
        if hip
            .attention(
                q,
                k,
                v,
                out,
                n_q as i32,
                n_kv as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                causal,
            )
            .is_ok()
        {
            return Ok(());
        }
    }
    attention(q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal)
}

pub(crate) fn gather_kv(
    kv: &KvTierManager,
    pages: &[u64],
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
    let mut ks = Vec::new();
    let mut vs = Vec::new();
    for id in pages {
        if kv.get(*id).map(|p| p.filled).unwrap_or(0) == 0 {
            continue;
        }
        let (k, v) = kv.read_kv(*id, n_kv_heads, head_dim)?;
        ks.extend_from_slice(&k);
        vs.extend_from_slice(&v);
    }
    Ok((ks, vs))
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed | 1,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let bits = ((self.state >> 33) as u32) & 0x007F_FFFF;
        (bits as f32 / 8_388_608.0) * 0.2 - 0.1
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Allocate one page per layer covering `n_tokens` (rounded up to page_tokens).
pub fn alloc_pages_for_seq(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    n_tokens: usize,
    metrics: &Metrics,
) -> Result<Vec<Vec<u64>>, AllpacaError> {
    let n_pages = n_tokens.div_ceil(page_tokens as usize).max(1);
    let mut pages = Vec::with_capacity(cfg.n_layers);
    for layer in 0..cfg.n_layers {
        let mut layer_pages = Vec::with_capacity(n_pages);
        for p in 0..n_pages {
            layer_pages.push(kv.alloc_page(
                layer as u32,
                p as u32,
                page_tokens,
                cfg.n_kv_heads,
                cfg.head_dim(),
                metrics,
            )?);
        }
        pages.push(layer_pages);
    }
    Ok(pages)
}

pub fn ensure_pages(
    kv: &mut KvTierManager,
    cfg: &ModelConfig,
    page_tokens: u32,
    pages: &mut [Vec<u64>],
    pos: usize,
    metrics: &Metrics,
) -> Result<(), AllpacaError> {
    let need = pos / page_tokens as usize;
    for (layer, layer_pages) in pages.iter_mut().enumerate() {
        while layer_pages.len() <= need {
            let idx = layer_pages.len() as u32;
            layer_pages.push(kv.alloc_page(
                layer as u32,
                idx,
                page_tokens,
                cfg.n_kv_heads,
                cfg.head_dim(),
                metrics,
            )?);
        }
    }
    Ok(())
}
