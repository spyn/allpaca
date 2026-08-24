use allpaca_cpu::{attention, attention_with_mask, matmul_nt, rmsnorm, rope, silu};
use allpaca_hip::{DeviceWeights, HipWorkspace};
use crate::forest::TokenForest;
use allpaca_gguf::GgufFile;
use allpaca_onnx::OnnxFile;
use allpaca_safetensors::{HfConfig, SafetensorsFile};
use allpaca_types::{AllpacaError, FatalError};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use crate::kv_tier::KvTierManager;
use crate::telemetry::Metrics;
use crate::weight_set::{ExpertId, ExpertLru, FfnRefs, LayerFfn, WeightWorkingSet};

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
        }
    }

    pub fn tiny_moe() -> Self {
        Self {
            n_experts: 4,
            n_active: 2,
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
            n_experts: 0,
            n_active: 0,
            eps,
            rope_theta,
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
        for layer in &model.layers {
            let attn_norm = push(&mut slices, &mut off, layer.attn_norm.as_slice());
            let wq = push(&mut slices, &mut off, layer.wq.as_slice());
            let wk = push(&mut slices, &mut off, layer.wk.as_slice());
            let wv = push(&mut slices, &mut off, layer.wv.as_slice());
            let wo = push(&mut slices, &mut off, layer.wo.as_slice());
            let (ffn_norm, w_gate, w_up, w_down) = if layer.w_gate.is_empty() {
                (
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                    WeightRef::f32(0, 0),
                )
            } else {
                (
                    push(&mut slices, &mut off, layer.ffn_norm.as_slice()),
                    push(&mut slices, &mut off, layer.w_gate.as_slice()),
                    push(&mut slices, &mut off, layer.w_up.as_slice()),
                    push(&mut slices, &mut off, layer.w_down.as_slice()),
                )
            };
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
        let tail = ring_slots.saturating_mul(slot_bytes);
        let weights = DeviceWeights::upload_f32_with_tail(device, &slices, tail)?;
        let ffn_ring = if tail > 0 {
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
        Ok(Self {
            packed_h2d: packed as u64,
            weights,
            ffn_ring,
            embed,
            layers,
            norm,
            lm_head,
        })
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
        for i in 0..cfg.n_layers {
            layers.push(LayerOff {
                attn_norm: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_norm.weight"),
                    &mut parts,
                    &mut off,
                )?,
                wq: append_gguf_tensor(file, &format!("blk.{i}.attn_q.weight"), &mut parts, &mut off)?,
                wk: append_gguf_tensor(file, &format!("blk.{i}.attn_k.weight"), &mut parts, &mut off)?,
                wv: append_gguf_tensor(file, &format!("blk.{i}.attn_v.weight"), &mut parts, &mut off)?,
                wo: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.attn_output.weight"),
                    &mut parts,
                    &mut off,
                )?,
                ffn_norm: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_norm.weight"),
                    &mut parts,
                    &mut off,
                )?,
                w_gate: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_gate.weight"),
                    &mut parts,
                    &mut off,
                )?,
                w_up: append_gguf_tensor(file, &format!("blk.{i}.ffn_up.weight"), &mut parts, &mut off)?,
                w_down: append_gguf_tensor(
                    file,
                    &format!("blk.{i}.ffn_down.weight"),
                    &mut parts,
                    &mut off,
                )?,
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
            WeightKind::F16 | WeightKind::Q4_K => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "F16/Q4_K norm must be F32 at load".into(),
            ))),
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
            WeightKind::Q4_K => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "Q4_K embed must be F32 at load".into(),
            ))),
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
            WeightKind::Q4_K => Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "Q4_K embed must be F32 at load".into(),
            ))),
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

fn estimated_gguf_f32_bytes(file: &GgufFile, n_layers: usize) -> u64 {
    let mut names = vec![
        "token_embd.weight".to_string(),
        "output_norm.weight".to_string(),
        "output.weight".to_string(),
    ];
    for i in 0..n_layers {
        names.push(format!("blk.{i}.attn_norm.weight"));
        names.push(format!("blk.{i}.attn_q.weight"));
        names.push(format!("blk.{i}.attn_k.weight"));
        names.push(format!("blk.{i}.attn_v.weight"));
        names.push(format!("blk.{i}.attn_output.weight"));
        names.push(format!("blk.{i}.ffn_norm.weight"));
        names.push(format!("blk.{i}.ffn_gate.weight"));
        names.push(format!("blk.{i}.ffn_up.weight"));
        names.push(format!("blk.{i}.ffn_down.weight"));
    }
    names
        .iter()
        .filter_map(|n| file.tensor(n))
        .map(|t| t.n_elements().saturating_mul(4))
        .sum()
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
        }
    }

    pub fn host_weights_ready(&self) -> bool {
        !self.embed.is_empty()
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
        let host = self
            .working_set
            .as_ref()
            .and_then(|ws| ws.host_ffn(layer_idx));
        resident.bind_ffn(
            layer_idx,
            host,
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
    ) -> Result<(), AllpacaError> {
        let cfg = &self.config;
        let dim = cfg.dim;
        if cfg.n_experts > 0 {
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
                let host = layer.experts.get(id.expert as usize).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "missing expert {}",
                        id.expert
                    )))
                })?;
                lru.admit(*id, host)?;
            }
        }
        let mut acc = vec![0.0f32; dim];
        let mut scratch_h = vec![0.0f32; dim];
        for &(e, w) in &routed {
            let host = layer.experts.get(e).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad(format!("missing expert {e}")))
            })?;
            let ffn = if let Some(lru) = &self.expert_lru {
                lru.materialize(
                    ExpertId {
                        layer: layer_idx as u32,
                        expert: e as u32,
                    },
                    host,
                )?
            } else {
                host.clone()
            };
            expert_mlp(x_norm, &ffn, cfg.ff_dim, dim, &mut scratch_h)?;
            for i in 0..dim {
                acc[i] += w * scratch_h[i];
            }
        }
        Ok(acc)
    }

    pub fn from_gguf(file: &GgufFile) -> Result<Self, AllpacaError> {
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
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("blk.{i}.{suffix}");
                if file.tensor(&name).is_none() {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "missing tensor {name}"
                    ))));
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
        let host_bytes = estimated_gguf_f32_bytes(file, cfg.n_layers);
        if host_bytes > HOST_F32_LIMIT {
            return Ok(Self::skeleton(cfg, name));
        }
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
            name,
            working_set: None,
            expert_lru: None,
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
        if self.config.n_experts == 0 {
            if let (Some(hip), Some(resident)) = (hip, resident) {
                match self.forward_token_device(
                    token, pos, page_ids, kv, metrics, logits, hip, resident,
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
            )?;
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
        if self.config.n_experts == 0 {
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
        let result = (|| {
        resident.embed_gather(hip, token, buf.x, dim as i32)?;
        record_elementwise(Some(metrics));

        for (layer_idx, _) in self.layers.iter().enumerate() {
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

fn gather_kv(
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
