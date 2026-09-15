//! DeepSeek-V4 HIP working set: resident BF16 MLA/shexp/lm_head + MXFP4 expert LRU.
//!
//! Generic GGUF→F32 HIP upload is skipped for these weights. gfx906 expands BF16
//! to F32 in registers. Experts stay packed on NVMe and H2D as 17-byte MXFP4
//! blocks. F32 tensors (tiny fixtures) stay on the CPU gemv path.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use allpaca_cpu::{MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};
use allpaca_gguf::{GgmlType, GgufFile};
use allpaca_hip::{
    device_memory_info, host_register, host_unregister, DeviceWeights, H2dStream, HipWorkspace,
};
use allpaca_types::{AllpacaError, DegradedError, FatalError};

use crate::model::{ModelConfig, TinyModel};
use crate::telemetry::Metrics;
use crate::weight_set::{qstar_fill_count, ExpertId};

const STREAM_SLOT_BYTES: usize = 128 * 1024 * 1024;
const PREFILL_RESERVE: u64 = 4 * 1024 * 1024 * 1024;
const ACT_SLOT_FLOATS: usize = 131072;
const ACT_SLOTS: usize = 3;

#[derive(Clone, Copy)]
struct DenseEntry {
    offset: usize,
    bytes: usize,
}

struct ExpertSlot {
    key: Option<(usize, usize)>,
    used: u64,
    pinned: bool,
}

struct RegisteredMmap {
    ptrs: Vec<*mut u8>,
}

unsafe impl Send for RegisteredMmap {}
unsafe impl Sync for RegisteredMmap {}

impl RegisteredMmap {
    fn clear(&mut self) {
        for ptr in self.ptrs.drain(..) {
            host_unregister(ptr);
        }
    }
}

impl Drop for RegisteredMmap {
    fn drop(&mut self) {
        self.clear();
    }
}

pub struct Dsv4Gpu {
    hip: Arc<HipWorkspace>,
    draft: Option<Arc<HipWorkspace>>,
    dense: Option<DeviceWeights>,
    dense_map: HashMap<String, DenseEntry>,
    stream: Option<DeviceWeights>,
    stream_off: usize,
    experts: Option<DeviceWeights>,
    expert_slot: usize,
    slots: Vec<ExpertSlot>,
    gpu1: Option<DeviceWeights>,
    gpu1_device: i32,
    gpu1_slots: Vec<ExpertSlot>,
    prefill: Option<DeviceWeights>,
    prefill_slot: usize,
    prefill_n: usize,
    prefill_layers: [Option<usize>; 2],
    prefill_used: [u64; 2],
    prefill_device: i32,
    clock: u64,
    h2d: Option<H2dStream>,
    h2d1: Option<H2dStream>,
    h2d_keep: Vec<Vec<u8>>,
    h2d_inflight: bool,
    h2d1_inflight: bool,
    h2d_reg: RegisteredMmap,
    act: Option<DeviceWeights>,
    pub used: AtomicBool,
    keep_set: HashSet<(usize, usize)>,
}

impl Dsv4Gpu {
    pub fn build(
        file: &GgufFile,
        cfg: &ModelConfig,
        hip: Arc<HipWorkspace>,
        draft: Option<Arc<HipWorkspace>>,
    ) -> Result<Self, AllpacaError> {
        hip.bind()?;
        let names = dense_bf16_names(cfg);
        let mut parts: Vec<(&str, &[u8])> = Vec::new();
        let mut owned: Vec<(String, Vec<u8>)> = Vec::new();
        for name in &names {
            let Some(info) = file.tensor(name) else {
                continue;
            };
            if info.ggml_type != GgmlType::BF16 {
                continue;
            }
            let bytes = file.tensor_bytes_of(info)?;
            owned.push((name.clone(), bytes.to_vec()));
        }
        for (name, bytes) in &owned {
            parts.push((name.as_str(), bytes.as_slice()));
        }
        let mut dense_map = HashMap::new();
        let dense = if parts.is_empty() {
            None
        } else {
            match upload_named(hip.device, &parts) {
                Ok((blob, map)) => {
                    tracing::info!(
                        bytes = blob.bytes(),
                        n = map.len(),
                        "DeepSeek-V4 HIP BF16 resident weights"
                    );
                    dense_map = map;
                    Some(blob)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "DeepSeek-V4 BF16 resident upload failed; stream slots");
                    None
                }
            }
        };
        let stream = DeviceWeights::alloc(hip.device, STREAM_SLOT_BYTES * 2).ok();
        let expert_slot = packed_expert_bytes(cfg.dim, cfg.ff_dim.max(1));
        let gpu1_device = draft.as_ref().map(|d| d.device).unwrap_or(-1);
        let prefill_device = draft.as_ref().map(|d| d.device).unwrap_or(hip.device);
        let mut gpu = Self {
            hip: hip.clone(),
            draft: draft.clone(),
            dense,
            dense_map,
            stream,
            stream_off: 0,
            experts: None,
            expert_slot,
            slots: Vec::new(),
            gpu1: None,
            gpu1_device,
            gpu1_slots: Vec::new(),
            prefill: None,
            prefill_slot: 0,
            prefill_n: 0,
            prefill_layers: [None, None],
            prefill_used: [0, 0],
            prefill_device,
            clock: 0,
            h2d: H2dStream::new().ok(),
            h2d1: None,
            h2d_keep: Vec::new(),
            h2d_inflight: false,
            h2d1_inflight: false,
            h2d_reg: RegisteredMmap { ptrs: Vec::new() },
            act: DeviceWeights::alloc(hip.device, ACT_SLOTS * ACT_SLOT_FLOATS * 4).ok(),
            used: AtomicBool::new(false),
            keep_set: HashSet::new(),
        };
        gpu.alloc_expert_pools(cfg)?;
        Ok(gpu)
    }

    fn pin_keep(&self, key: (usize, usize)) -> bool {
        self.keep_set.contains(&key)
    }

    /// Durable pin-prefer keep-set. Empty means no reserved slots (true LRU).
    pub fn set_keep_set(&mut self, ids: &[ExpertId]) {
        self.keep_set = ids
            .iter()
            .map(|id| (id.layer as usize, id.expert as usize))
            .collect();
        for slot in self.slots.iter_mut().chain(self.gpu1_slots.iter_mut()) {
            slot.pinned = slot.key.is_some_and(|k| self.keep_set.contains(&k));
        }
    }

    fn alloc_expert_pools(&mut self, cfg: &ModelConfig) -> Result<(), AllpacaError> {
        if cfg.n_experts == 0 || self.expert_slot == 0 {
            return Ok(());
        }
        let hip = self.hip.clone();
        let draft = self.draft.clone();
        let (free0, _) = device_memory_info(hip.device).unwrap_or((0, 0));
        let budget0 = (free0 / 2) as usize;
        let n0 = (budget0 / self.expert_slot.max(1)).min(512);
        if n0 > 0 {
            if let Ok(blob) = DeviceWeights::alloc(hip.device, n0 * self.expert_slot) {
                tracing::info!(
                    slots = n0,
                    bytes = blob.bytes(),
                    "DeepSeek-V4 GPU0 expert LRU"
                );
                self.slots = (0..n0)
                    .map(|_| ExpertSlot {
                        key: None,
                        used: 0,
                        pinned: false,
                    })
                    .collect();
                self.experts = Some(blob);
            }
        }
        if let Some(dhip) = draft {
            let _ = dhip.bind();
            let (free1, _) = device_memory_info(dhip.device).unwrap_or((0, 0));
            let n_exp = cfg.n_experts.max(1);
            let layer_bytes = n_exp.saturating_mul(self.expert_slot);
            let mut remain = free1 as usize;
            if remain as u64 > PREFILL_RESERVE && layer_bytes > 0 {
                let mut n_buf = 1usize;
                if remain.saturating_sub(layer_bytes) as u64 > PREFILL_RESERVE {
                    n_buf = 2;
                }
                if let Ok(blob) = DeviceWeights::alloc(dhip.device, n_buf * layer_bytes) {
                    tracing::info!(
                        bytes = blob.bytes(),
                        buffers = n_buf,
                        "DeepSeek-V4 GPU1 prefill expert layer buffer"
                    );
                    self.prefill = Some(blob);
                    self.prefill_slot = layer_bytes;
                    self.prefill_n = n_buf;
                    self.prefill_device = dhip.device;
                    remain = remain.saturating_sub(n_buf * layer_bytes);
                }
            }
            let n1 = (remain / 2 / self.expert_slot).min(512);
            if n1 > 0 {
                if let Ok(blob) = DeviceWeights::alloc(dhip.device, n1 * self.expert_slot) {
                    tracing::info!(
                        slots = n1,
                        bytes = blob.bytes(),
                        "DeepSeek-V4 GPU1 expert LRU"
                    );
                    self.gpu1_slots = (0..n1)
                        .map(|_| ExpertSlot {
                            key: None,
                            used: 0,
                            pinned: false,
                        })
                        .collect();
                    self.gpu1 = Some(blob);
                    self.gpu1_device = dhip.device;
                }
            }
            self.h2d1 = H2dStream::new().ok();
            hip.bind()?;
        }
        if self.prefill.is_none() {
            let n_exp = cfg.n_experts.max(1);
            let layer_bytes = n_exp.saturating_mul(self.expert_slot);
            let (free0, _) = device_memory_info(hip.device).unwrap_or((0, 0));
            if free0 > PREFILL_RESERVE && layer_bytes > 0 {
                let mut n_buf = 1usize;
                if free0.saturating_sub(layer_bytes as u64) > PREFILL_RESERVE {
                    n_buf = 2;
                }
                if let Ok(blob) = DeviceWeights::alloc(hip.device, n_buf * layer_bytes) {
                    tracing::info!(
                        bytes = blob.bytes(),
                        buffers = n_buf,
                        "DeepSeek-V4 GPU0 prefill expert layer buffer"
                    );
                    self.prefill = Some(blob);
                    self.prefill_slot = layer_bytes;
                    self.prefill_n = n_buf;
                    self.prefill_device = hip.device;
                }
            }
        }
        Ok(())
    }

    pub fn gemv_bf16(
        &mut self,
        file: &GgufFile,
        name: &str,
        x: &[f32],
        out: &mut [f32],
        n: usize,
        k: usize,
        elem_off: usize,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        self.gemm_bf16(file, name, x, out, n, k, elem_off, 1, metrics)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(
        &mut self,
        file: &GgufFile,
        name: &str,
        x: &[f32],
        out: &mut [f32],
        n: usize,
        k: usize,
        elem_off: usize,
        m: usize,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if m == 0 {
            return Ok(());
        }
        if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "BF16 GEMM shape exceeds i32".into(),
            )));
        }
        if x.len() < m.saturating_mul(k) || out.len() < m.saturating_mul(n) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "BF16 GEMM row buffer size mismatch".into(),
            )));
        }
        let hip = self.hip.clone();
        let info = file.tensor(name).ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled(format!(
                "missing {name} for HIP BF16"
            )))
        })?;
        if info.ggml_type != GgmlType::BF16 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("{name} is not BF16"),
            )));
        }
        let packed_need = n.saturating_mul(k).saturating_mul(2);
        let byte_off = elem_off.saturating_mul(2);
        let mi = m as i32;
        let ni = n as i32;
        let ki = k as i32;
        if let Some(ent) = self.dense_map.get(name).copied() {
            if byte_off.saturating_add(packed_need) <= ent.bytes {
                if let Some(blob) = &self.dense {
                    let ptr = blob.byte_ptr_at(ent.offset + byte_off) as *const u16;
                    hip.bind()?;
                    let t = Instant::now();
                    hip.gemm_nt_bf16_dev_b(x, ptr, out, mi, ni, ki)?;
                    add_ns(metrics, |met| &met.dsv4_gemv_ns, t);
                    self.used.store(true, Ordering::Relaxed);
                    return Ok(());
                }
            }
        }
        let bytes = file.tensor_bytes_of(info)?;
        let slice = bytes.get(byte_off..byte_off + packed_need).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!("{name} BF16 slice OOB")))
        })?;
        let slot = self.stream.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("no BF16 stream slot".into()))
        })?;
        if packed_need > STREAM_SLOT_BYTES {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("{name} BF16 {packed_need} exceeds stream slot"),
            )));
        }
        let off = if self.stream_off == 0 {
            0
        } else {
            STREAM_SLOT_BYTES
        };
        self.stream_off = 1 - self.stream_off;
        let t = Instant::now();
        slot.copy_h2d(off, slice)?;
        add_ns(metrics, |met| &met.dsv4_h2d_ns, t);
        add_bytes(metrics, slice.len() as u64);
        hip.bind()?;
        let t = Instant::now();
        hip.gemm_nt_bf16_dev_b(x, slot.byte_ptr_at(off) as *const u16, out, mi, ni, ki)?;
        add_ns(metrics, |met| &met.dsv4_gemv_ns, t);
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn expert_mlp_gpu(
        &mut self,
        file: &GgufFile,
        layer: usize,
        expert: usize,
        cfg: &ModelConfig,
        x: &[f32],
        y: &mut [f32],
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        self.expert_mlp_gpu_rows(file, layer, expert, cfg, x, y, 1, metrics)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn expert_mlp_gpu_rows(
        &mut self,
        file: &GgufFile,
        layer: usize,
        expert: usize,
        cfg: &ModelConfig,
        x: &[f32],
        y: &mut [f32],
        m: usize,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if m == 0 {
            return Ok(());
        }
        if m > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "expert GEMM M exceeds i32".into(),
            )));
        }
        let hip = self.hip.clone();
        let draft = self.draft.clone();
        let dim = cfg.dim;
        let ff = cfg.ff_dim.max(1);
        if x.len() < m.saturating_mul(dim) || y.len() < m.saturating_mul(dim) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "expert GEMM row buffer size mismatch".into(),
            )));
        }
        let (ptr, device) = self.admit_expert(file, layer, expert, cfg, metrics)?;
        let gate_bytes = mxfp4_matrix_bytes(ff, dim);
        let up_bytes = gate_bytes;
        let ws = if device == hip.device {
            hip.as_ref()
        } else {
            draft.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "expert on GPU1 without draft HIP".into(),
                ))
            })?
        };
        ws.bind()?;
        // SAFETY: `ptr` is an expert slot of `gate|up|down` packed bytes.
        let up_ptr = unsafe { ptr.add(gate_bytes) };
        let down_ptr = unsafe { ptr.add(gate_bytes + up_bytes) };
        let mi = m as i32;
        let t = Instant::now();
        let swiglu = ws.mxfp4_swiglu_gemm(x, ptr, up_ptr, down_ptr, y, mi, dim as i32, ff as i32);
        if swiglu.is_ok() {
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        } else {
            let mut gate = vec![0.0f32; m.saturating_mul(ff)];
            let mut up = vec![0.0f32; m.saturating_mul(ff)];
            let t = Instant::now();
            ws.gemm_nt_mxfp4_dev_b(x, ptr, &mut gate, mi, ff as i32, dim as i32)?;
            ws.gemm_nt_mxfp4_dev_b(x, up_ptr, &mut up, mi, ff as i32, dim as i32)?;
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
            let t = Instant::now();
            for i in 0..m.saturating_mul(ff) {
                gate[i] = allpaca_cpu::silu(gate[i]) * up[i];
            }
            add_ns(metrics, |m| &m.dsv4_silu_ns, t);
            let t = Instant::now();
            ws.gemm_nt_mxfp4_dev_b(&gate, down_ptr, y, mi, dim as i32, ff as i32)?;
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        }
        hip.bind()?;
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn admit_expert(
        &mut self,
        file: &GgufFile,
        layer: usize,
        expert: usize,
        cfg: &ModelConfig,
        metrics: Option<&Metrics>,
    ) -> Result<(*const u8, i32), AllpacaError> {
        let hip = self.hip.clone();
        let draft = self.draft.clone();
        let key = (layer, expert);
        self.clock = self.clock.saturating_add(1);
        if let Some(i) = self.slots.iter().position(|s| s.key == Some(key)) {
            self.slots[i].used = self.clock;
            self.slots[i].pinned = self.pin_keep(key);
            if let Some(m) = metrics {
                m.expert_lru_hits.fetch_add(1, Ordering::Relaxed);
            }
            let off = i * self.expert_slot;
            let ptr = self.experts.as_ref().unwrap().byte_ptr_at(off);
            return Ok((ptr, hip.device));
        }
        if let Some(i) = self.gpu1_slots.iter().position(|s| s.key == Some(key)) {
            self.gpu1_slots[i].used = self.clock;
            self.gpu1_slots[i].pinned = self.pin_keep(key);
            if let Some(m) = metrics {
                m.expert_lru_hits.fetch_add(1, Ordering::Relaxed);
            }
            let off = i * self.expert_slot;
            let ptr = self.gpu1.as_ref().unwrap().byte_ptr_at(off);
            return Ok((ptr, self.gpu1_device));
        }
        if let Some(i) = self
            .prefill_layers
            .iter()
            .take(self.prefill_n)
            .position(|l| *l == Some(layer))
        {
            if let Some(blob) = &self.prefill {
                if let Some(m) = metrics {
                    m.expert_lru_hits.fetch_add(1, Ordering::Relaxed);
                }
                let off = i * self.prefill_slot + expert * self.expert_slot;
                return Ok((blob.byte_ptr_at(off), self.prefill_device));
            }
        }
        if let Some(m) = metrics {
            m.expert_lru_misses.fetch_add(1, Ordering::Relaxed);
        }
        let packed = packed_expert_triple(file, layer, expert, cfg)?;
        if packed.len() > self.expert_slot {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "expert packed larger than slot".into(),
            )));
        }
        if !self.slots.is_empty() {
            let victim = lru_victim(&self.slots);
            let off = victim * self.expert_slot;
            let blob = self.experts.as_ref().unwrap();
            hip.bind()?;
            let t = Instant::now();
            blob.copy_h2d(off, &packed)?;
            add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
            add_bytes(metrics, packed.len() as u64);
            if let Some(m) = metrics {
                m.expert_lru_fills.fetch_add(1, Ordering::Relaxed);
            }
            self.slots[victim].key = Some(key);
            self.slots[victim].used = self.clock;
            self.slots[victim].pinned = self.pin_keep(key);
            return Ok((blob.byte_ptr_at(off), hip.device));
        }
        if !self.gpu1_slots.is_empty() {
            let victim = lru_victim(&self.gpu1_slots);
            let off = victim * self.expert_slot;
            let blob = self.gpu1.as_ref().unwrap();
            if let Some(d) = draft.as_ref() {
                d.bind()?;
            }
            let t = Instant::now();
            blob.copy_h2d(off, &packed)?;
            add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
            add_bytes(metrics, packed.len() as u64);
            if let Some(m) = metrics {
                m.expert_lru_fills.fetch_add(1, Ordering::Relaxed);
            }
            self.gpu1_slots[victim].key = Some(key);
            self.gpu1_slots[victim].used = self.clock;
            self.gpu1_slots[victim].pinned = self.pin_keep(key);
            hip.bind()?;
            return Ok((blob.byte_ptr_at(off), self.gpu1_device));
        }
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "no expert GPU slot".into(),
        )))
    }

    pub fn has_prefill_layer(&self, layer: usize) -> bool {
        self.prefill_layers
            .iter()
            .take(self.prefill_n)
            .any(|l| *l == Some(layer))
    }

    pub fn finish_h2d_keep(&mut self) {
        self.h2d_keep.clear();
    }

    pub fn has_expert_slots(&self) -> bool {
        !self.slots.is_empty() || !self.gpu1_slots.is_empty() || self.prefill.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_sinks(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        out: &mut [f32],
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        self.hip.attention_sinks(
            q, k, v, sinks, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
        )
    }

    pub fn prefetch_layer_experts(
        &mut self,
        file: &GgufFile,
        layer: usize,
        cfg: &ModelConfig,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        let Some(blob) = &self.prefill else {
            return Ok(());
        };
        if self.prefill_n == 0 {
            return Ok(());
        }
        if self.has_prefill_layer(layer) {
            return Ok(());
        }
        let hip = self.hip.clone();
        let draft = self.draft.clone();
        let ws = if self.prefill_device == hip.device {
            hip.as_ref()
        } else {
            draft.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "prefill blob on GPU1 without draft HIP".into(),
                ))
            })?
        };
        ws.bind()?;
        let victim = (0..self.prefill_n)
            .min_by_key(|&i| (self.prefill_layers[i].is_some(), self.prefill_used[i]))
            .unwrap_or(0);
        let base = victim * self.prefill_slot;
        let t = Instant::now();
        let mut nbytes = 0u64;
        for e in 0..cfg.n_experts {
            let packed = packed_expert_triple(file, layer, e, cfg)?;
            let off = base + e * self.expert_slot;
            nbytes = nbytes.saturating_add(packed.len() as u64);
            blob.copy_h2d(off, &packed)?;
        }
        add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
        add_bytes(metrics, nbytes);
        if let Some(m) = metrics {
            m.expert_bulk_fills.fetch_add(1, Ordering::Relaxed);
            m.expert_bulk_bytes.fetch_add(nbytes, Ordering::Relaxed);
        }
        self.clock = self.clock.saturating_add(1);
        self.prefill_layers[victim] = Some(layer);
        self.prefill_used[victim] = self.clock;
        hip.bind()?;
        Ok(())
    }

    /// Pin resident routed experts and batch-H2D misses from GGUF mmap (no host pack Vec).
    /// Copies stay on the side stream until [`Self::join_expert_h2d`].
    pub fn warmup_experts(
        &mut self,
        file: &GgufFile,
        layer: usize,
        ids: &[usize],
        cfg: &ModelConfig,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        self.h2d_keep.clear();
        self.h2d_inflight = false;
        self.h2d_reg.clear();
        if ids.is_empty() || self.has_prefill_layer(layer) {
            return Ok(());
        }
        let hip = self.hip.clone();
        let draft = self.draft.clone();
        let mut n0 = 0usize;
        let mut n1 = 0usize;
        let mut parts0: Vec<(usize, &[u8])> = Vec::new();
        let mut parts1: Vec<(usize, &[u8])> = Vec::new();
        let t_pack = Instant::now();
        for &expert in ids {
            let key = (layer, expert);
            self.clock = self.clock.saturating_add(1);
            if let Some(i) = self.slots.iter().position(|s| s.key == Some(key)) {
                self.slots[i].used = self.clock;
                self.slots[i].pinned = self.pin_keep(key);
                continue;
            }
            if let Some(i) = self.gpu1_slots.iter().position(|s| s.key == Some(key)) {
                self.gpu1_slots[i].used = self.clock;
                self.gpu1_slots[i].pinned = self.pin_keep(key);
                continue;
            }
            if let Some(m) = metrics {
                m.expert_lru_misses.fetch_add(1, Ordering::Relaxed);
            }
            let slices = expert_mxfp4_parts(file, layer, expert, cfg)?;
            let packed_len: usize = slices.iter().map(|(_, s)| s.len()).sum();
            if packed_len > self.expert_slot {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "expert packed larger than slot".into(),
                )));
            }
            let use_gpu1 = assign_miss_gpu1(self.slots.len(), self.gpu1_slots.len(), n0, n1);
            if use_gpu1 {
                let victim = lru_victim(&self.gpu1_slots);
                self.gpu1_slots[victim].key = Some(key);
                self.gpu1_slots[victim].used = self.clock;
                self.gpu1_slots[victim].pinned = self.pin_keep(key);
                let base = victim * self.expert_slot;
                for (rel, sl) in slices {
                    parts1.push((base + rel, sl));
                }
                n1 += 1;
                continue;
            }
            if self.slots.is_empty() {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "no expert GPU slot".into(),
                )));
            }
            let victim = lru_victim(&self.slots);
            self.slots[victim].key = Some(key);
            self.slots[victim].used = self.clock;
            self.slots[victim].pinned = self.pin_keep(key);
            let base = victim * self.expert_slot;
            for (rel, sl) in slices {
                parts0.push((base + rel, sl));
            }
            n0 += 1;
        }
        add_ns(metrics, |m| &m.dsv4_pack_ns, t_pack);
        let mut all_parts = Vec::with_capacity(parts0.len() + parts1.len());
        all_parts.extend_from_slice(&parts0);
        all_parts.extend_from_slice(&parts1);
        if !all_parts.is_empty() {
            let t_pin = Instant::now();
            let registered = register_mmap_parts(&all_parts);
            if registered.is_ok() {
                add_ns(metrics, |m| &m.dsv4_pack_ns, t_pin);
            } else {
                static REG_WARN: AtomicBool = AtomicBool::new(false);
                if !REG_WARN.swap(true, Ordering::Relaxed) {
                    tracing::warn!("hipHostRegister on GGUF mmap failed; pageable expert H2D");
                }
                for (_, sl) in &all_parts {
                    prefault_mmap(sl);
                }
                add_ns(metrics, |m| &m.dsv4_pack_ns, t_pin);
            }
            if let Ok(ptrs) = registered {
                self.h2d_reg.ptrs = ptrs;
            }
        }
        let nbytes0: u64 = parts0.iter().map(|(_, s)| s.len() as u64).sum();
        let nbytes1: u64 = parts1.iter().map(|(_, s)| s.len() as u64).sum();
        let t_h2d = Instant::now();
        let mut inflight0 = false;
        let mut inflight1 = false;
        std::thread::scope(|s| {
            let j0 = (!parts0.is_empty()).then(|| {
                let blob = self.experts.as_ref();
                let stream = self.h2d.as_ref();
                let hip = hip.clone();
                let parts = &parts0;
                s.spawn(move || {
                    let blob = blob.ok_or_else(|| {
                        AllpacaError::Degraded(DegradedError::FeatureDisabled(
                            "GPU0 expert blob missing".into(),
                        ))
                    })?;
                    hip.bind()?;
                    copy_expert_parts(blob, parts, stream)
                })
            });
            let j1 = (!parts1.is_empty()).then(|| {
                let blob = self.gpu1.as_ref();
                let stream = self.h2d1.as_ref();
                let draft = draft.clone();
                let parts = &parts1;
                s.spawn(move || {
                    let blob = blob.ok_or_else(|| {
                        AllpacaError::Degraded(DegradedError::FeatureDisabled(
                            "GPU1 expert blob missing".into(),
                        ))
                    })?;
                    let d = draft.as_ref().ok_or_else(|| {
                        AllpacaError::Degraded(DegradedError::FeatureDisabled(
                            "GPU1 expert H2D without draft HIP".into(),
                        ))
                    })?;
                    d.bind()?;
                    copy_expert_parts(blob, parts, stream)
                })
            });
            if let Some(j) = j0 {
                match j.join() {
                    Ok(Ok(async_copy)) => inflight0 = async_copy,
                    Ok(Err(e)) => tracing::warn!(error = %e, "GPU0 expert H2D failed"),
                    Err(_) => tracing::warn!("GPU0 expert H2D thread panicked"),
                }
            }
            if let Some(j) = j1 {
                match j.join() {
                    Ok(Ok(async_copy)) => inflight1 = async_copy,
                    Ok(Err(e)) => tracing::warn!(error = %e, "GPU1 expert H2D failed"),
                    Err(_) => tracing::warn!("GPU1 expert H2D thread panicked"),
                }
            }
        });
        let _ = hip.bind();
        add_ns(metrics, |m| &m.dsv4_h2d_ns, t_h2d);
        add_bytes(metrics, nbytes0.saturating_add(nbytes1));
        self.h2d_inflight = inflight0;
        self.h2d1_inflight = inflight1;
        if let Some(m) = metrics {
            if n0 > 0 {
                m.expert_lru_fills.fetch_add(n0 as u64, Ordering::Relaxed);
                m.expert_bulk_fills.fetch_add(1, Ordering::Relaxed);
                m.expert_bulk_bytes.fetch_add(nbytes0, Ordering::Relaxed);
                m.expert_h2d_batched_copies
                    .fetch_add(parts0.len() as u64, Ordering::Relaxed);
            }
            if n1 > 0 {
                m.expert_lru_fills.fetch_add(n1 as u64, Ordering::Relaxed);
                m.expert_bulk_fills.fetch_add(1, Ordering::Relaxed);
                m.expert_bulk_bytes.fetch_add(nbytes1, Ordering::Relaxed);
                m.expert_h2d_batched_copies
                    .fetch_add(parts1.len() as u64, Ordering::Relaxed);
            }
            if inflight0 || inflight1 {
                m.expert_prefill_overlaps.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    pub fn join_expert_h2d(&mut self) {
        if self.h2d_inflight {
            if let Some(h2d) = self.h2d.as_ref() {
                let _ = self.hip.bind();
                let _ = h2d.join_compute();
            }
            self.h2d_inflight = false;
        }
        if self.h2d1_inflight {
            if let (Some(h2d1), Some(d)) = (self.h2d1.as_ref(), self.draft.as_ref()) {
                let _ = d.bind();
                let _ = h2d1.join_compute();
            }
            self.h2d1_inflight = false;
            let _ = self.hip.bind();
        }
    }

    pub fn resident_expert(&self, layer: usize, expert: usize) -> Option<(*const u8, i32)> {
        let key = (layer, expert);
        if let Some(i) = self.slots.iter().position(|s| s.key == Some(key)) {
            let off = i * self.expert_slot;
            let ptr = self.experts.as_ref()?.byte_ptr_at(off);
            return Some((ptr, self.hip.device));
        }
        if let Some(i) = self.gpu1_slots.iter().position(|s| s.key == Some(key)) {
            let off = i * self.expert_slot;
            let ptr = self.gpu1.as_ref()?.byte_ptr_at(off);
            return Some((ptr, self.gpu1_device));
        }
        if let Some(i) = self
            .prefill_layers
            .iter()
            .take(self.prefill_n)
            .position(|l| *l == Some(layer))
        {
            let blob = self.prefill.as_ref()?;
            let off = i * self.prefill_slot + expert * self.expert_slot;
            return Some((blob.byte_ptr_at(off), self.prefill_device));
        }
        None
    }

    pub fn hip_pair(&self) -> (Arc<HipWorkspace>, Option<Arc<HipWorkspace>>) {
        (self.hip.clone(), self.draft.clone())
    }

    fn act_ptr(&self, slot: usize) -> Result<*mut f32, AllpacaError> {
        let blob = self.act.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("no DSV4 act blob".into()))
        })?;
        if slot >= ACT_SLOTS {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "act slot OOB".into(),
            )));
        }
        Ok(blob.byte_ptr_at(slot * ACT_SLOT_FLOATS * 4) as *mut f32)
    }

    fn dense_bf16_ptr(
        &mut self,
        file: &GgufFile,
        name: &str,
        n: usize,
        k: usize,
        elem_off: usize,
        metrics: Option<&Metrics>,
    ) -> Result<*const u16, AllpacaError> {
        let packed_need = n.saturating_mul(k).saturating_mul(2);
        let byte_off = elem_off.saturating_mul(2);
        if let Some(ent) = self.dense_map.get(name).copied() {
            if byte_off.saturating_add(packed_need) <= ent.bytes {
                if let Some(blob) = &self.dense {
                    return Ok(blob.byte_ptr_at(ent.offset + byte_off) as *const u16);
                }
            }
        }
        let info = file.tensor(name).ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled(format!(
                "missing {name} for HIP BF16"
            )))
        })?;
        if info.ggml_type != GgmlType::BF16 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("{name} is not BF16"),
            )));
        }
        let bytes = file.tensor_bytes_of(info)?;
        let slice = bytes.get(byte_off..byte_off + packed_need).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!("{name} BF16 slice OOB")))
        })?;
        let slot = self.stream.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("no BF16 stream slot".into()))
        })?;
        if packed_need > STREAM_SLOT_BYTES {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("{name} BF16 {packed_need} exceeds stream slot"),
            )));
        }
        let off = if self.stream_off == 0 {
            0
        } else {
            STREAM_SLOT_BYTES
        };
        self.stream_off = 1 - self.stream_off;
        let t = Instant::now();
        slot.copy_h2d(off, slice)?;
        add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
        add_bytes(metrics, slice.len() as u64);
        Ok(slot.byte_ptr_at(off) as *const u16)
    }

    pub fn mla_q_proj(
        &mut self,
        file: &GgufFile,
        blk: usize,
        cur: &[f32],
        q_a_norm: &[f32],
        eps: f32,
        q_lora: usize,
        q_out: usize,
        dim: usize,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        if q_out > ACT_SLOT_FLOATS || q_lora > ACT_SLOT_FLOATS || dim > ACT_SLOT_FLOATS {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA Q exceeds act slot".into(),
            )));
        }
        let hip = self.hip.clone();
        let s0 = self.act_ptr(0)?;
        let s1 = self.act_ptr(1)?;
        hip.bind()?;
        let t = Instant::now();
        hip.copy_h2d_f32(s0, cur)?;
        add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
        add_bytes(metrics, (cur.len() * 4) as u64);
        let qa = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.attn_q_a.weight"),
            q_lora,
            dim,
            0,
            metrics,
        )?;
        let t = Instant::now();
        hip.gemm_nt_bf16_dev(s0, qa, s1, 1, q_lora as i32, dim as i32)?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        if !q_a_norm.is_empty() {
            let nw = self.act_ptr(2)?;
            hip.copy_h2d_f32(nw, q_a_norm)?;
            let t = Instant::now();
            hip.rmsnorm_device(s1, nw, s1, q_lora as i32, eps)?;
            add_ns(metrics, |m| &m.dsv4_norm_ns, t);
        }
        let qb = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.attn_q_b.weight"),
            q_out,
            q_lora,
            0,
            metrics,
        )?;
        let t = Instant::now();
        hip.gemm_nt_bf16_dev(s1, qb, s0, 1, q_out as i32, q_lora as i32)?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        let mut q = vec![0.0f32; q_out];
        let t = Instant::now();
        hip.copy_d2h_f32(s0, &mut q)?;
        add_ns(metrics, |m| &m.dsv4_d2h_ns, t);
        self.used.store(true, Ordering::Relaxed);
        Ok(q)
    }

    pub fn mla_kv_proj(
        &mut self,
        file: &GgufFile,
        blk: usize,
        cur: &[f32],
        kv_norm: &[f32],
        eps: f32,
        head_dim: usize,
        dim: usize,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        if head_dim > ACT_SLOT_FLOATS || dim > ACT_SLOT_FLOATS {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA KV exceeds act slot".into(),
            )));
        }
        let hip = self.hip.clone();
        let s0 = self.act_ptr(0)?;
        let s1 = self.act_ptr(1)?;
        hip.bind()?;
        let t = Instant::now();
        hip.copy_h2d_f32(s0, cur)?;
        add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
        add_bytes(metrics, (cur.len() * 4) as u64);
        let w = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.attn_kv.weight"),
            head_dim,
            dim,
            0,
            metrics,
        )?;
        let t = Instant::now();
        hip.gemm_nt_bf16_dev(s0, w, s1, 1, head_dim as i32, dim as i32)?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        if !kv_norm.is_empty() {
            let nw = self.act_ptr(2)?;
            hip.copy_h2d_f32(nw, kv_norm)?;
            let t = Instant::now();
            hip.rmsnorm_device(s1, nw, s1, head_dim as i32, eps)?;
            add_ns(metrics, |m| &m.dsv4_norm_ns, t);
        }
        let mut kv = vec![0.0f32; head_dim];
        let t = Instant::now();
        hip.copy_d2h_f32(s1, &mut kv)?;
        add_ns(metrics, |m| &m.dsv4_d2h_ns, t);
        self.used.store(true, Ordering::Relaxed);
        Ok(kv)
    }

    pub fn shexp_swiglu(
        &mut self,
        file: &GgufFile,
        blk: usize,
        x: &[f32],
        dim: usize,
        ff: usize,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        self.shexp_swiglu_rows(file, blk, x, 1, dim, ff, metrics)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn shexp_swiglu_rows(
        &mut self,
        file: &GgufFile,
        blk: usize,
        x: &[f32],
        n_tok: usize,
        dim: usize,
        ff: usize,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        if n_tok == 0 {
            return Ok(Vec::new());
        }
        if n_tok > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "shexp GEMM M exceeds i32".into(),
            )));
        }
        if x.len() < n_tok.saturating_mul(dim) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "shexp row buffer size mismatch".into(),
            )));
        }
        let hip = self.hip.clone();
        let gate = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.ffn_gate_shexp.weight"),
            ff,
            dim,
            0,
            metrics,
        )?;
        let up = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.ffn_up_shexp.weight"),
            ff,
            dim,
            0,
            metrics,
        )?;
        let down = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.ffn_down_shexp.weight"),
            dim,
            ff,
            0,
            metrics,
        )?;
        let mut y = vec![0.0f32; n_tok.saturating_mul(dim)];
        hip.bind()?;
        let t = Instant::now();
        hip.bf16_swiglu_gemm(
            x,
            gate,
            up,
            down,
            &mut y,
            n_tok as i32,
            dim as i32,
            ff as i32,
        )?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        self.used.store(true, Ordering::Relaxed);
        Ok(y)
    }

    pub fn qstar_gpu_count(&self, misses: usize) -> usize {
        if self.slots.is_empty() && self.gpu1_slots.is_empty() {
            return 0;
        }
        qstar_fill_count(misses, 12.0e9, 8.0e9)
    }
}

/// Full-layer expert H2D is cheaper than 6-of-256 only when most experts fire.
pub(crate) fn want_full_layer_prefetch(prefill: bool, cfg: &ModelConfig) -> bool {
    prefill && cfg.n_experts > 0 && cfg.n_active.saturating_mul(8) >= cfg.n_experts
}

fn upload_named(
    device: i32,
    parts: &[(&str, &[u8])],
) -> Result<(DeviceWeights, HashMap<String, DenseEntry>), AllpacaError> {
    let views: Vec<&[u8]> = parts.iter().map(|(_, b)| *b).collect();
    let blob = DeviceWeights::upload_bytes(device, &views)?;
    let mut map = HashMap::new();
    let mut off = 0usize;
    for (name, bytes) in parts {
        map.insert(
            (*name).to_string(),
            DenseEntry {
                offset: off,
                bytes: bytes.len(),
            },
        );
        off += bytes.len();
    }
    Ok((blob, map))
}

fn dense_bf16_names(cfg: &ModelConfig) -> Vec<String> {
    let mut names = vec!["output.weight".into()];
    for i in 0..cfg.n_layers {
        let blk = cfg.gguf_blk(i);
        for suffix in [
            "attn_q_a.weight",
            "attn_q_b.weight",
            "attn_kv.weight",
            "attn_output_a.weight",
            "attn_output_b.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_inp.weight",
        ] {
            names.push(format!("blk.{blk}.{suffix}"));
        }
    }
    names
}

fn mxfp4_matrix_bytes(n: usize, k: usize) -> usize {
    n.saturating_mul(
        k.div_ceil(MXFP4_BLOCK_ELEMS)
            .saturating_mul(MXFP4_BLOCK_BYTES),
    )
}

fn packed_expert_bytes(dim: usize, ff: usize) -> usize {
    mxfp4_matrix_bytes(ff, dim)
        .saturating_mul(2)
        .saturating_add(mxfp4_matrix_bytes(dim, ff))
}

fn assign_miss_gpu1(n0_slots: usize, n1_slots: usize, assigned0: usize, assigned1: usize) -> bool {
    if n1_slots == 0 {
        return false;
    }
    if n0_slots == 0 {
        return true;
    }
    if assigned1 >= n1_slots && assigned0 < n0_slots {
        return false;
    }
    if assigned0 >= n0_slots {
        return true;
    }
    assigned1 < assigned0
}

fn copy_expert_parts(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    stream: Option<&H2dStream>,
) -> Result<bool, AllpacaError> {
    if parts.is_empty() {
        return Ok(false);
    }
    if let Some(h2d) = stream {
        // SAFETY: GGUF mmap (or hipHostRegister'd pages) outlives join_expert_h2d.
        unsafe { blob.copy_h2d_batch_on(parts, h2d.raw())? };
        Ok(true)
    } else {
        for (off, sl) in parts {
            blob.copy_h2d(*off, sl)?;
        }
        Ok(false)
    }
}

pub(crate) struct ExpertSwigluJob {
    pub expert: usize,
    pub ptr: usize,
    pub x: Vec<f32>,
    pub m: usize,
    pub group: Vec<(usize, f32)>,
}

pub(crate) fn run_expert_swiglu_jobs(
    ws: &HipWorkspace,
    jobs: Vec<ExpertSwigluJob>,
    dim: usize,
    ff: usize,
    metrics: Option<&Metrics>,
) -> (Vec<(Vec<f32>, Vec<(usize, f32)>)>, Vec<ExpertSwigluJob>) {
    let gate_bytes = mxfp4_matrix_bytes(ff, dim);
    let mut out = Vec::with_capacity(jobs.len());
    let mut failed = Vec::new();
    if ws.bind().is_err() {
        return (out, jobs);
    }
    for job in jobs {
        let ptr = job.ptr as *const u8;
        // SAFETY: `ptr` is a packed gate|up|down expert slot on `ws`.
        let up_ptr = unsafe { ptr.add(gate_bytes) };
        let down_ptr = unsafe { ptr.add(gate_bytes.saturating_add(gate_bytes)) };
        let mut y = vec![0.0f32; job.x.len()];
        let t = Instant::now();
        if ws
            .mxfp4_swiglu_gemm(
                &job.x,
                ptr,
                up_ptr,
                down_ptr,
                &mut y,
                job.m as i32,
                dim as i32,
                ff as i32,
            )
            .is_ok()
        {
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
            out.push((y, job.group));
        } else {
            failed.push(job);
        }
    }
    (out, failed)
}

fn expert_mxfp4_parts<'a>(
    file: &'a GgufFile,
    layer: usize,
    expert: usize,
    cfg: &ModelConfig,
) -> Result<Vec<(usize, &'a [u8])>, AllpacaError> {
    let dim = cfg.dim;
    let ff = cfg.ff_dim.max(1);
    let n_exp = cfg.n_experts.max(1);
    let mut out = Vec::with_capacity(3);
    let mut rel = 0usize;
    for (name, need_n, need_k) in [
        (format!("blk.{layer}.ffn_gate_exps.weight"), ff, dim),
        (format!("blk.{layer}.ffn_up_exps.weight"), ff, dim),
        (format!("blk.{layer}.ffn_down_exps.weight"), dim, ff),
    ] {
        let info = file
            .tensor(&name)
            .ok_or_else(|| AllpacaError::Fatal(FatalError::ModelLoad(format!("missing {name}"))))?;
        if info.ggml_type != GgmlType::MXFP4 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("{name} is not MXFP4"),
            )));
        }
        let bytes = file.tensor_bytes_of(info)?;
        let n = info.n_elements() as usize;
        let per = n / n_exp;
        let stride = per
            .div_ceil(MXFP4_BLOCK_ELEMS)
            .saturating_mul(MXFP4_BLOCK_BYTES);
        let start = expert.saturating_mul(stride);
        let slice = bytes.get(start..start + stride).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::CorruptGguf(format!(
                "{name} expert {expert} OOB (n_exp={n_exp} need_n={need_n} need_k={need_k})"
            )))
        })?;
        out.push((rel, slice));
        rel += slice.len();
    }
    Ok(out)
}

const HOST_PAGE: usize = 4096;

fn page_range(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.is_empty() {
        return None;
    }
    let start = bytes.as_ptr() as usize;
    let end = start.saturating_add(bytes.len());
    let aligned = start & !(HOST_PAGE - 1);
    let aligned_end = end.saturating_add(HOST_PAGE - 1) & !(HOST_PAGE - 1);
    let n = aligned_end.saturating_sub(aligned);
    (n > 0).then_some((aligned, n))
}

fn merge_page_ranges(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.len() <= 1 {
        return ranges;
    }
    ranges.sort_unstable_by_key(|&(start, _)| start);
    let mut out = Vec::with_capacity(ranges.len());
    let mut cur = ranges[0];
    for &(start, n) in ranges.iter().skip(1) {
        let cur_end = cur.0.saturating_add(cur.1);
        let end = start.saturating_add(n);
        if start <= cur_end {
            cur.1 = end.max(cur_end).saturating_sub(cur.0);
        } else {
            out.push(cur);
            cur = (start, n);
        }
    }
    out.push(cur);
    out
}

fn register_mmap_parts(parts: &[(usize, &[u8])]) -> Result<Vec<*mut u8>, AllpacaError> {
    let mut ranges = Vec::with_capacity(parts.len());
    for (_, sl) in parts {
        if let Some(range) = page_range(sl) {
            ranges.push(range);
        }
    }
    let merged = merge_page_ranges(ranges);
    let mut ptrs = Vec::with_capacity(merged.len());
    for (start, n) in merged {
        let ptr = start as *mut u8;
        if let Err(e) = host_register(ptr, n, 0) {
            for p in ptrs {
                host_unregister(p);
            }
            return Err(e);
        }
        ptrs.push(ptr);
    }
    Ok(ptrs)
}

fn packed_expert_triple(
    file: &GgufFile,
    layer: usize,
    expert: usize,
    cfg: &ModelConfig,
) -> Result<Vec<u8>, AllpacaError> {
    let parts = expert_mxfp4_parts(file, layer, expert, cfg)?;
    let mut out = Vec::with_capacity(packed_expert_bytes(cfg.dim, cfg.ff_dim.max(1)));
    for (_, sl) in parts {
        out.extend_from_slice(sl);
    }
    Ok(out)
}

pub fn attach(
    model: &TinyModel,
    hip: &Arc<HipWorkspace>,
    draft: Option<&Arc<HipWorkspace>>,
) -> Result<(), AllpacaError> {
    let dsv4 = model.dsv4.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing DeepSeek-V4 weights".into()))
    })?;
    let file = dsv4.gguf.clone().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "DeepSeek-V4 GGUF handle missing for HIP attach".into(),
        ))
    })?;
    let gpu = Dsv4Gpu::build(file.as_ref(), &model.config, hip.clone(), draft.cloned())?;
    let mut guard = dsv4
        .gpu
        .lock()
        .map_err(|_| AllpacaError::Fatal(FatalError::ModelLoad("dsv4 gpu lock poisoned".into())))?;
    *guard = Some(gpu);
    Ok(())
}

/// Apply a pin-prefer keep-set to an attached DeepSeek-V4 HIP working set.
pub fn apply_keep_set(model: &TinyModel, ids: &[ExpertId]) -> Result<(), AllpacaError> {
    if ids.is_empty() {
        return Ok(());
    }
    let Some(dsv4) = model.dsv4.as_ref() else {
        return Ok(());
    };
    let file = dsv4.gguf.clone();
    let mut guard = dsv4.gpu.lock().map_err(|_| {
        AllpacaError::Fatal(FatalError::ModelLoad("dsv4 gpu lock poisoned".into()))
    })?;
    let Some(gpu) = guard.as_mut() else {
        return Ok(());
    };
    gpu.set_keep_set(ids);
    let Some(file) = file else {
        return Ok(());
    };
    let mut by_layer: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for id in ids {
        by_layer
            .entry(id.layer as usize)
            .or_default()
            .push(id.expert as usize);
    }
    for (layer, experts) in by_layer {
        gpu.warmup_experts(file.as_ref(), layer, &experts, &model.config, None)?;
    }
    Ok(())
}

fn add_ns(metrics: Option<&Metrics>, field: impl Fn(&Metrics) -> &AtomicU64, t: Instant) {
    if let Some(m) = metrics {
        field(m).fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

fn add_bytes(metrics: Option<&Metrics>, n: u64) {
    if let Some(m) = metrics {
        m.dsv4_h2d_bytes.fetch_add(n, Ordering::Relaxed);
    }
}

fn lru_victim(slots: &[ExpertSlot]) -> usize {
    slots
        .iter()
        .enumerate()
        .min_by_key(|(_, s)| {
            let empty = u8::from(s.key.is_some());
            let pin = u8::from(s.pinned);
            (empty, pin, s.used)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Fault GGUF mmap pages into DRAM before HIP DMA. Cold mmap H2D runs at
/// NVMe speed (~300 MB/s); hot pages hit PCIe.
fn prefault_mmap(bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let ptr = bytes.as_ptr();
    let len = bytes.len();
    let mut off = 0usize;
    while off < len {
        // SAFETY: `off < len`, so this is in-bounds.
        unsafe {
            std::ptr::read_volatile(ptr.add(off));
        }
        off = off.saturating_add(4096);
    }
    if len > 1 {
        unsafe {
            std::ptr::read_volatile(ptr.add(len - 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::want_full_layer_prefetch;
    use crate::model::ModelConfig;

    #[test]
    fn unsloth_short_prefill_skips_full_layer() {
        let cfg = ModelConfig {
            n_experts: 256,
            n_active: 6,
            ..ModelConfig::tiny()
        };
        assert!(!want_full_layer_prefetch(true, &cfg));
        assert!(!want_full_layer_prefetch(false, &cfg));
    }

    #[test]
    fn tiny_moe_prefill_uses_full_layer() {
        let cfg = ModelConfig::tiny_moe();
        assert!(want_full_layer_prefetch(true, &cfg));
        assert!(!want_full_layer_prefetch(false, &cfg));
    }

    #[test]
    fn merge_page_ranges_coalesces_adjacent() {
        let page = super::HOST_PAGE;
        let merged = super::merge_page_ranges(vec![(2 * page, page), (0, page), (page, page)]);
        assert_eq!(merged, vec![(0, 3 * page)]);
        let split = super::merge_page_ranges(vec![(0, page), (3 * page, page)]);
        assert_eq!(split, vec![(0, page), (3 * page, page)]);
    }

    #[test]
    fn assign_miss_gpu1_balances_and_respects_caps() {
        assert!(!super::assign_miss_gpu1(8, 0, 0, 0));
        assert!(super::assign_miss_gpu1(0, 8, 0, 0));
        assert!(!super::assign_miss_gpu1(8, 8, 0, 0));
        assert!(super::assign_miss_gpu1(8, 8, 1, 0));
        assert!(!super::assign_miss_gpu1(8, 8, 1, 1));
        assert!(!super::assign_miss_gpu1(2, 2, 1, 2));
        assert!(super::assign_miss_gpu1(2, 2, 2, 1));
    }
}
