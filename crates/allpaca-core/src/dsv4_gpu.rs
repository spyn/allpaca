//! DeepSeek-V4 HIP working set: resident BF16 MLA/shexp/lm_head, F32 HC mix
//! weights, and MXFP4 expert LRU packed into leftover HBM.
//!
//! Generic GGUF→F32 HIP upload is skipped for MLA/MoE. gfx906 expands BF16 to
//! F32 in registers. Experts stay packed on NVMe and H2D as 17-byte MXFP4
//! blocks. Full-layer prefill buffers are only allocated when the active set is
//! dense enough that they would be used. F32 tensors other than HC mix stay on
//! the CPU gemv path.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::Instant;

use allpaca_cpu::{MXFP4_BLOCK_BYTES, MXFP4_BLOCK_ELEMS};
use allpaca_gguf::{GgmlType, GgufFile};
use allpaca_hip::{
    device_memory_info, gemm_nt_device, host_free, host_malloc, host_register, host_unregister,
    memcpy, DeviceWeights, H2dStream, HipWorkspace, H2D_READY_RING, MEMCPY_D2H, MEMCPY_H2D,
};
use allpaca_types::{AllpacaError, DegradedError, FatalError};

use crate::model::{ModelConfig, TinyModel};
use crate::telemetry::Metrics;
use crate::weight_set::{qstar_fill_count, ExpertId};

const STREAM_SLOT_BYTES: usize = 128 * 1024 * 1024;
const PREFILL_RESERVE: u64 = 4 * 1024 * 1024 * 1024;
/// Leave HBM for KV pages, HIP scratch, and allocator fragmentation.
const HBM_RESERVE: u64 = 2 * 1024 * 1024 * 1024;
/// Per-GPU expert LRU cap. Filling both 32 GiB heaps evicts the GGUF page cache
/// (host RAM is 128 GiB; the Unsloth shard is 149 GiB) and turns mmap prefault
/// into NVMe. 12 GiB ×2 holds the ChatML routed working set (~23 GiB).
const EXPERT_LRU_MAX_BYTES: u64 = 12 * 1024 * 1024 * 1024;
/// Routing counts fade by `HEAT_DECAY` every this many routed experts
/// (about eight tokens of DeepSeek-V4-Flash).
const HEAT_DECAY_ROUTES: u64 = 2048;
const HEAT_DECAY: f32 = 0.8;
/// An expert displaces a resident one only when it has been routed this
/// much more often: a copy costs several milliseconds, a miss on the CPU
/// well under one.
const HEAT_MARGIN: f32 = 2.0;
const ACT_SLOT_FLOATS: usize = 131072;
const ACT_SLOTS: usize = 3;
/// Bounce / mmap-window size. Two in-flight windows × two GPUs must fit
/// `ulimit -l` (8 MiB here). Registering the whole GGUF mmap still fails;
/// attach probes one window and uses ping-pong `hipHostRegister` when it works.
const PIN_CHUNK_TRY: [usize; 2] = [2 * 1024 * 1024, 1024 * 1024];
/// Bounce buffer sizes, largest first. Where pinned memory is not rationed
/// by `ulimit -l`, one buffer takes a whole expert (13.4 MB for
/// DeepSeek-V4-Flash) in a single copy.
#[cfg(windows)]
const PIN_BOUNCE_TRY: [usize; 4] = [16 * 1024 * 1024, 4 * 1024 * 1024, 2 * 1024 * 1024, 1024 * 1024];
#[cfg(not(windows))]
const PIN_BOUNCE_TRY: [usize; 2] = PIN_CHUNK_TRY;
const HOST_PAGE: usize = 4096;
/// Prefault this many host bytes, then decide if the rest is already in RAM.
const PREFAULT_WARM_PROBE_BYTES: usize = 12 * 1024 * 1024;
/// 12 MiB in under this many ms is DRAM (skip the remaining 22 GiB touch).
const PREFAULT_WARM_MS: u128 = 8;

#[derive(Clone, Copy)]
struct DenseEntry {
    offset: usize,
    bytes: usize,
    /// Staged Q8_0 blocks (36 bytes per 32 values) instead of BF16.
    q8: bool,
}

impl DenseEntry {
    /// Byte offset and length of `n` rows of `k` values starting at element
    /// `elem_off` (a whole number of rows), when the entry holds them.
    fn span(&self, n: usize, k: usize, elem_off: usize) -> Option<(usize, usize)> {
        let (off, len) = if self.q8 {
            if k % 32 != 0 || elem_off % 32 != 0 {
                return None;
            }
            (elem_off / 32 * allpaca_hip::Q8_STAGED_BLOCK, n * (k / 32) * allpaca_hip::Q8_STAGED_BLOCK)
        } else {
            (elem_off * 2, n * k * 2)
        };
        (off + len <= self.bytes).then_some((off, len))
    }
}

struct ExpertSlot {
    key: Option<(usize, usize)>,
    used: u64,
    pinned: bool,
}

pub(crate) struct ExpertCopyPlan<'a> {
    parts0: Vec<(usize, &'a [u8])>,
    parts1: Vec<(usize, &'a [u8])>,
    /// `(expert, part_count)` covering `parts0` / `parts1` in order.
    groups0: Vec<(usize, usize)>,
    groups1: Vec<(usize, usize)>,
    n0: usize,
    n1: usize,
    prefetch: bool,
}

impl ExpertCopyPlan<'_> {
    fn empty() -> Self {
        Self {
            parts0: Vec::new(),
            parts1: Vec::new(),
            groups0: Vec::new(),
            groups1: Vec::new(),
            n0: 0,
            n1: 0,
            prefetch: false,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.parts0.is_empty() && self.parts1.is_empty()
    }
}

enum PlanMode {
    Warmup,
    Prefetch { keep_layer: usize },
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

/// Two `hipHostMalloc` staging buffers for mmap→pin memcpy overlapping pin→HBM DMA,
/// or (when `window`) ping-pong `hipHostRegister` of GGUF mmap page windows.
/// `live` / `which` persist across `submit_copy_plan` so the next layer waits a
/// slot event instead of host-syncing the whole H2D stream.
struct PinPair {
    a: *mut u8,
    b: *mut u8,
    bytes: usize,
    live: [AtomicBool; 2],
    which: AtomicUsize,
    window: bool,
    win_ptr: [AtomicUsize; 2],
    win_len: [AtomicUsize; 2],
}

unsafe impl Send for PinPair {}
unsafe impl Sync for PinPair {}

impl Drop for PinPair {
    fn drop(&mut self) {
        for i in 0..2 {
            let p = self.win_ptr[i].swap(0, Ordering::Relaxed);
            if p != 0 {
                host_unregister(p as *mut u8);
            }
        }
        host_free(self.a);
        host_free(self.b);
        self.a = std::ptr::null_mut();
        self.b = std::ptr::null_mut();
    }
}

fn empty_pin_atomics() -> (
    [AtomicBool; 2],
    AtomicUsize,
    [AtomicUsize; 2],
    [AtomicUsize; 2],
) {
    (
        [AtomicBool::new(false), AtomicBool::new(false)],
        AtomicUsize::new(0),
        [AtomicUsize::new(0), AtomicUsize::new(0)],
        [AtomicUsize::new(0), AtomicUsize::new(0)],
    )
}

fn try_window_pin_pair() -> PinPair {
    let (live, which, win_ptr, win_len) = empty_pin_atomics();
    PinPair {
        a: std::ptr::null_mut(),
        b: std::ptr::null_mut(),
        bytes: PIN_CHUNK_TRY[0],
        live,
        which,
        window: true,
        win_ptr,
        win_len,
    }
}

fn try_pin_pair() -> Option<PinPair> {
    for bytes in PIN_BOUNCE_TRY {
        let a = match host_malloc(bytes) {
            Ok(p) => p,
            Err(_) => continue,
        };
        match host_malloc(bytes) {
            Ok(b) => {
                let (live, which, win_ptr, win_len) = empty_pin_atomics();
                return Some(PinPair {
                    a,
                    b,
                    bytes,
                    live,
                    which,
                    window: false,
                    win_ptr,
                    win_len,
                });
            }
            Err(_) => host_free(a),
        }
    }
    None
}

/// One page-aligned window of a GGUF mmap slice. `reg_*` is what HIP locks;
/// `data` is the bytes inside that window that belong to `sl`.
struct MmapWindow<'a> {
    reg_ptr: usize,
    reg_len: usize,
    data: &'a [u8],
    data_off: usize,
}

/// Split `sl` into non-overlapping page-aligned register windows of `window` bytes.
fn mmap_windows(sl: &[u8], window: usize) -> Vec<MmapWindow<'_>> {
    let mut out = Vec::new();
    let Some((base, cover)) = page_range(sl) else {
        return out;
    };
    let sl_start = sl.as_ptr() as usize;
    let sl_end = sl_start.saturating_add(sl.len());
    let window = window.max(HOST_PAGE);
    let cover_end = base.saturating_add(cover);
    let mut win0 = base;
    while win0 < cover_end {
        let win_len = window.min(cover_end.saturating_sub(win0));
        let h0 = sl_start.max(win0);
        let h1 = sl_end.min(win0.saturating_add(win_len));
        if h1 > h0 {
            let data_off = h0.saturating_sub(sl_start);
            out.push(MmapWindow {
                reg_ptr: win0,
                reg_len: win_len,
                data: &sl[data_off..data_off + (h1 - h0)],
                data_off,
            });
        }
        win0 = win0.saturating_add(win_len);
    }
    out
}

/// `hipHostRegister` on a 2 MiB GGUF window. File-backed ZFS mmaps fail here
/// (`/zepool/...`); bounce stays the Unsloth path. ext4/tmpfs maps can succeed.
fn probe_windowed_mmap_register(file: &GgufFile) -> bool {
    for info in &file.tensors {
        let Ok(sl) = file.tensor_bytes_of(info) else {
            continue;
        };
        if sl.len() < HOST_PAGE {
            continue;
        }
        let n = sl.len().min(PIN_CHUNK_TRY[0]);
        let Some((ptr, len)) = page_range(&sl[..n]) else {
            continue;
        };
        let ptr = ptr as *mut u8;
        return match host_register(ptr, len, 0) {
            Ok(()) => {
                host_unregister(ptr);
                true
            }
            Err(_) => false,
        };
    }
    false
}

fn try_expert_pin(file: &GgufFile) -> Option<PinPair> {
    if probe_windowed_mmap_register(file) {
        Some(try_window_pin_pair())
    } else {
        try_pin_pair()
    }
}

fn release_reg_slot(pin: &PinPair, slot: usize) {
    let p = pin.win_ptr[slot].swap(0, Ordering::AcqRel);
    if p != 0 {
        host_unregister(p as *mut u8);
        pin.win_len[slot].store(0, Ordering::Release);
    }
}

pub struct Dsv4Gpu {
    hip: Arc<HipWorkspace>,
    draft: Option<Arc<HipWorkspace>>,
    dense: Option<DeviceWeights>,
    dense_map: HashMap<String, DenseEntry>,
    f32_dense: Option<DeviceWeights>,
    f32_map: HashMap<String, DenseEntry>,
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
    pin0: Option<PinPair>,
    pin1: Option<PinPair>,
    h2d_keep: Vec<Vec<u8>>,
    h2d_inflight: bool,
    h2d1_inflight: bool,
    h2d_reg: RegisteredMmap,
    act: Option<DeviceWeights>,
    q_ws: Option<DeviceWeights>,
    attn_ws: Option<DeviceWeights>,
    hc_ones: Option<DeviceWeights>,
    hc_act: Option<DeviceWeights>,
    hc_live: usize,
    hc_rows: usize,
    hc_hc_dim: usize,
    hc_on_device: bool,
    hc_curs: usize,
    hc_curs_n: usize,
    hc_curs_host: usize,
    hc_curs_synced: bool,
    o_out: usize,
    o_out_n: usize,
    warmup_miss: Vec<(usize, usize)>,
    pub used: AtomicBool,
    keep_set: HashSet<(usize, usize)>,
    /// Decayed routing count per (layer, expert): what earns a cache slot.
    heat: HashMap<(usize, usize), f32>,
    heat_routes: u64,
    /// Cache misses run on the CPU from the mapped file, and an expert
    /// takes a slot only when it is routed clearly more often than the
    /// coldest resident one. Off: every miss is copied in (plain LRU).
    cpu_miss: bool,
    /// Host RAM tier the CPU experts read through.
    ram: Option<Arc<std::sync::Mutex<crate::dsv4_ram::ExpertRam>>>,
    /// Mapped expert slices handed to a GPU copy, released once it landed.
    mapped_used: Vec<(usize, usize)>,
    /// RAM-tier experts pinned as the source of a GPU copy.
    ram_pinned: Vec<(usize, usize)>,
    /// Reads the next layer's cold experts ahead of use.
    read_ahead: Option<crate::dsv4_ram::Prefetcher>,
}

impl Dsv4Gpu {
    pub fn build(
        file: &GgufFile,
        cfg: &ModelConfig,
        hip: Arc<HipWorkspace>,
        draft: Option<Arc<HipWorkspace>>,
        host_bytes: u64,
    ) -> Result<Self, AllpacaError> {
        hip.bind()?;
        // Before the dense weights pass through this process on their way
        // to the GPU.
        let ram_free = crate::dsv4_ram::available_ram_bytes();
        let names = dense_bf16_names(cfg);
        let mut parts: Vec<(&str, &[u8])> = Vec::new();
        let mut owned: Vec<(String, Vec<u8>)> = Vec::new();
        let mut q8_names: HashSet<String> = HashSet::new();
        for name in &names {
            let Some(info) = file.tensor(name) else {
                continue;
            };
            let bytes = file.tensor_bytes_of(info)?;
            match info.ggml_type {
                GgmlType::BF16 => owned.push((name.clone(), bytes.to_vec())),
                // `dims` are [k, n]: n rows of k values, whole blocks per row.
                GgmlType::Q8_0 if info.dims.len() == 2 && info.dims[0] % 32 == 0 => {
                    let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
                    owned.push((name.clone(), allpaca_hip::stage_q8_0_gemv(bytes, n, k)?));
                    q8_names.insert(name.clone());
                }
                _ => continue,
            }
            crate::dsv4_ram::release_mapped(bytes);
        }
        for (name, bytes) in &owned {
            parts.push((name.as_str(), bytes.as_slice()));
        }
        let mut dense_map = HashMap::new();
        let dense = if parts.is_empty() {
            None
        } else {
            match upload_named(hip.device, &parts) {
                Ok((blob, mut map)) => {
                    for (name, ent) in map.iter_mut() {
                        if q8_names.contains(name) {
                            ent.q8 = true;
                            hip.register_q8_dense(blob.byte_ptr_at(ent.offset), ent.bytes);
                        }
                    }
                    tracing::info!(
                        bytes = blob.bytes(),
                        n = map.len(),
                        q8_0 = q8_names.len(),
                        "DeepSeek-V4 HIP dense resident weights"
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
        let f32_names = dense_f32_hc_names(cfg);
        let mut f32_parts: Vec<(&str, &[u8])> = Vec::new();
        for name in &f32_names {
            let Some(info) = file.tensor(name) else {
                continue;
            };
            if info.ggml_type != GgmlType::F32 {
                continue;
            }
            let Ok(bytes) = file.tensor_bytes_of(info) else {
                continue;
            };
            f32_parts.push((name.as_str(), bytes));
        }
        let mut f32_map = HashMap::new();
        let f32_dense = if f32_parts.is_empty() {
            None
        } else {
            match upload_named(hip.device, &f32_parts) {
                Ok((blob, map)) => {
                    tracing::info!(
                        bytes = blob.bytes(),
                        n = map.len(),
                        "DeepSeek-V4 HIP F32 HC mix weights"
                    );
                    f32_map = map;
                    Some(blob)
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "DeepSeek-V4 F32 HC upload failed; mix GEMM uses host weights"
                    );
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
            f32_dense,
            f32_map,
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
            pin0: None,
            pin1: None,
            h2d_keep: Vec::new(),
            h2d_inflight: false,
            h2d1_inflight: false,
            h2d_reg: RegisteredMmap { ptrs: Vec::new() },
            act: DeviceWeights::alloc(hip.device, ACT_SLOTS * ACT_SLOT_FLOATS * 4).ok(),
            q_ws: None,
            attn_ws: None,
            hc_ones: None,
            hc_act: None,
            hc_live: 0,
            hc_rows: 0,
            hc_hc_dim: 0,
            hc_on_device: false,
            hc_curs: 0,
            hc_curs_n: 0,
            hc_curs_host: 0,
            hc_curs_synced: false,
            o_out: 0,
            o_out_n: 0,
            warmup_miss: Vec::new(),
            used: AtomicBool::new(false),
            keep_set: HashSet::new(),
            heat: HashMap::new(),
            heat_routes: 0,
            cpu_miss: std::env::var("ALLPACA_DSV4_CPU_MISS").map_or(true, |v| v != "0")
                && cpu_expert_kernel_fits(file, cfg),
            ram: None,
            mapped_used: Vec::new(),
            ram_pinned: Vec::new(),
            read_ahead: None,
        };
        if gpu.cpu_miss {
            gpu.ram = open_expert_ram(file, cfg, gpu.expert_slot, ram_free, host_bytes);
        }
        if gpu.h2d.is_some() {
            gpu.pin0 = try_expert_pin(file);
            if let Some(pin) = gpu.pin0.as_ref() {
                if pin.window {
                    eprintln!(
                        "dsv4 h2d pin gpu=0 windowed hipHostRegister bytes={}",
                        pin.bytes
                    );
                } else {
                    eprintln!("dsv4 h2d pin gpu=0 bounce bytes={}", pin.bytes);
                }
            } else {
                eprintln!("dsv4 h2d pin gpu=0 pageable");
            }
        }
        gpu.alloc_expert_pools(cfg)?;
        Ok(gpu)
    }

    /// Misses are computed on the CPU instead of being copied into the cache.
    pub(crate) fn cpu_miss(&self) -> bool {
        self.cpu_miss && self.has_expert_slots()
    }

    /// The host RAM tier, when the model file allows direct reads.
    pub(crate) fn expert_ram(&self) -> Option<Arc<std::sync::Mutex<crate::dsv4_ram::ExpertRam>>> {
        self.ram.clone()
    }

    /// Start the read-ahead worker when `ALLPACA_DSV4_READAHEAD=1`. Off by
    /// default: on DeepSeek-V4-Flash about a quarter of its reads were used,
    /// SSD traffic nearly doubled and throughput did not improve.
    fn start_read_ahead(&mut self, file: &Arc<GgufFile>, cfg: &ModelConfig) {
        let Some(ram) = self.ram.clone() else {
            return;
        };
        if !std::env::var("ALLPACA_DSV4_READAHEAD").is_ok_and(|v| v == "1") {
            return;
        }
        let ff = cfg.ff_dim.max(1);
        let max_slice = mxfp4_matrix_bytes(ff, cfg.dim).max(mxfp4_matrix_bytes(cfg.dim, ff));
        let spans_cfg = cfg.clone();
        match crate::dsv4_ram::Prefetcher::start(
            file.clone(),
            ram,
            cfg.dim,
            cfg.n_experts,
            cfg.n_active.max(1),
            max_slice,
            move |f, layer, expert| expert_mxfp4_spans(f, layer, expert, &spans_cfg),
        ) {
            Ok(p) => self.read_ahead = Some(p),
            Err(e) => tracing::warn!(error = %e, "DeepSeek-V4 expert read-ahead unavailable"),
        }
    }

    /// Predict the experts of `next_layer` from `rows`, this layer's expert
    /// inputs, and read the cold ones ahead.
    pub(crate) fn read_ahead(&self, next_layer: usize, rows: &[f32]) {
        let Some(p) = self.read_ahead.as_ref() else {
            return;
        };
        let on_gpu: Vec<usize> = self
            .slots
            .iter()
            .chain(&self.gpu1_slots)
            .filter_map(|s| s.key)
            .filter(|k| k.0 == next_layer)
            .map(|k| k.1)
            .collect();
        p.submit(next_layer, rows, on_gpu);
    }

    /// Count one routing of `(layer, expert)`.
    pub(crate) fn note_route(&mut self, layer: usize, expert: usize) {
        *self.heat.entry((layer, expert)).or_insert(0.0) += 1.0;
        self.heat_routes += 1;
        if self.heat_routes % HEAT_DECAY_ROUTES == 0 {
            self.heat.retain(|_, h| {
                *h *= HEAT_DECAY;
                *h > 0.05
            });
        }
    }

    /// Whether `(layer, expert)`, which is not resident, takes a cache slot
    /// now: always while slots are free, else only when it is routed
    /// clearly more often than the coldest resident expert of each cache,
    /// which then becomes the next one evicted.
    pub(crate) fn admits(&mut self, layer: usize, expert: usize) -> bool {
        if !self.cpu_miss {
            return true;
        }
        if self.slots.iter().chain(&self.gpu1_slots).any(|s| s.key.is_none()) {
            return true;
        }
        let heat = &self.heat;
        let coldest = |slots: &[ExpertSlot]| -> Option<(usize, f32)> {
            slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.pinned)
                .filter_map(|(i, s)| s.key.map(|k| (i, heat.get(&k).copied().unwrap_or(0.0))))
                .min_by(|a, b| a.1.total_cmp(&b.1))
        };
        let (c0, c1) = (coldest(&self.slots), coldest(&self.gpu1_slots));
        let floor = match (c0, c1) {
            (Some(a), Some(b)) => a.1.max(b.1),
            (Some(a), None) | (None, Some(a)) => a.1,
            (None, None) => return false,
        };
        if heat.get(&(layer, expert)).copied().unwrap_or(0.0) < floor + HEAT_MARGIN {
            return false;
        }
        if let Some((i, _)) = c0 {
            self.slots[i].used = 0;
        }
        if let Some((i, _)) = c1 {
            self.gpu1_slots[i].used = 0;
        }
        true
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
        let full_prefill = want_full_layer_prefetch(true, cfg);
        if let Some(dhip) = draft.as_ref() {
            let _ = dhip.bind();
            if full_prefill {
                self.try_alloc_prefill_layer(dhip.device, cfg);
            }
            let free1 = device_free_bytes(dhip.device);
            if let Some((blob, n1)) = try_alloc_expert_lru(
                dhip.device,
                expert_lru_slots(free1, self.expert_slot),
                self.expert_slot,
            ) {
                tracing::info!(
                    slots = n1,
                    bytes = blob.bytes(),
                    "DeepSeek-V4 GPU1 expert LRU"
                );
                self.gpu1_slots = empty_expert_slots(n1);
                self.gpu1 = Some(blob);
                self.gpu1_device = dhip.device;
            }
            self.h2d1 = H2dStream::new().ok();
            if self.h2d1.is_some() {
                self.pin1 = match self.pin0.as_ref() {
                    Some(p) if p.window => Some(try_window_pin_pair()),
                    _ => try_pin_pair(),
                };
                if let Some(pin) = self.pin1.as_ref() {
                    if pin.window {
                        eprintln!(
                            "dsv4 h2d pin gpu=1 windowed hipHostRegister bytes={}",
                            pin.bytes
                        );
                    } else {
                        eprintln!("dsv4 h2d pin gpu=1 bounce bytes={}", pin.bytes);
                    }
                } else {
                    eprintln!("dsv4 h2d pin gpu=1 pageable");
                }
            }
            hip.bind()?;
        }
        if full_prefill && self.prefill.is_none() {
            self.try_alloc_prefill_layer(hip.device, cfg);
        }
        let free0 = device_free_bytes(hip.device);
        if let Some((blob, n0)) = try_alloc_expert_lru(
            hip.device,
            expert_lru_slots(free0, self.expert_slot),
            self.expert_slot,
        ) {
            tracing::info!(
                slots = n0,
                bytes = blob.bytes(),
                "DeepSeek-V4 GPU0 expert LRU"
            );
            self.slots = empty_expert_slots(n0);
            self.experts = Some(blob);
        }
        Ok(())
    }

    fn try_alloc_prefill_layer(&mut self, device: i32, cfg: &ModelConfig) {
        let n_exp = cfg.n_experts.max(1);
        let layer_bytes = n_exp.saturating_mul(self.expert_slot);
        if layer_bytes == 0 {
            return;
        }
        let (free, _) = device_memory_info(device).unwrap_or((0, 0));
        if free <= PREFILL_RESERVE {
            return;
        }
        let mut n_buf = 1usize;
        if free.saturating_sub(layer_bytes as u64) > PREFILL_RESERVE {
            n_buf = 2;
        }
        if let Ok(blob) = DeviceWeights::alloc(device, n_buf.saturating_mul(layer_bytes)) {
            tracing::info!(
                bytes = blob.bytes(),
                buffers = n_buf,
                device,
                "DeepSeek-V4 prefill expert layer buffer"
            );
            self.prefill = Some(blob);
            self.prefill_slot = layer_bytes;
            self.prefill_n = n_buf;
            self.prefill_device = device;
        }
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
        let resident = self
            .dense_map
            .get(name)
            .and_then(|ent| ent.span(n, k, elem_off).map(|(off, _)| (*ent, off)));
        // A Q8_0 tensor is served from its staged resident copy only.
        if info.ggml_type != GgmlType::BF16 && resident.is_none() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("{name} is not BF16"),
            )));
        }
        let packed_need = n.saturating_mul(k).saturating_mul(2);
        let byte_off = resident.map_or(elem_off.saturating_mul(2), |(_, off)| off);
        let mi = m as i32;
        let ni = n as i32;
        let ki = k as i32;
        if let Some((ent, _)) = resident {
            {
                if let Some(blob) = &self.dense {
                    let ptr = blob.byte_ptr_at(ent.offset + byte_off) as *const u16;
                    let da = self.device_a_if_curs(x, m, k);
                    hip.bind()?;
                    let t = Instant::now();
                    match da {
                        Some(a) => hip.gemm_nt_bf16_from_device_a(a, ptr, out, mi, ni, ki)?,
                        None => hip.gemm_nt_bf16_dev_b(x, ptr, out, mi, ni, ki)?,
                    }
                    add_ns(metrics, |met| &met.dsv4_gemv_ns, t);
                    self.used.store(true, Ordering::Relaxed);
                    return Ok(());
                }
            }
        }
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
        add_ns(metrics, |met| &met.dsv4_h2d_ns, t);
        add_bytes(metrics, slice.len() as u64);
        hip.bind()?;
        let t = Instant::now();
        let b = slot.byte_ptr_at(off) as *const u16;
        match self.device_a_if_curs(x, m, k) {
            Some(a) => hip.gemm_nt_bf16_from_device_a(a, b, out, mi, ni, ki)?,
            None => hip.gemm_nt_bf16_dev_b(x, b, out, mi, ni, ki)?,
        }
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
        // Copies of this layer have landed: their source pages can go.
        if !self.h2d_inflight && !self.h2d1_inflight {
            for (ptr, len) in self.mapped_used.drain(..) {
                // SAFETY: a slice of the model's mapping, which outlives the engine's GPU state.
                crate::dsv4_ram::release_mapped(unsafe { std::slice::from_raw_parts(ptr as *const u8, len) });
            }
            if !self.ram_pinned.is_empty() {
                if let Some(ram) = self.ram.as_ref() {
                    ram.lock().unwrap_or_else(|e| e.into_inner()).release(&self.ram_pinned);
                }
                self.ram_pinned.clear();
            }
        }
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

    fn reset_expert_h2d_host(&mut self) {
        self.join_expert_h2d();
        self.h2d_keep.clear();
        self.h2d_reg.clear();
    }

    fn admit_expert_slot(
        &mut self,
        key: (usize, usize),
        n0: &mut usize,
        n1: &mut usize,
        pin: bool,
        keep_layer: Option<usize>,
    ) -> Option<(bool, usize)> {
        let prefer_gpu1 = assign_miss_gpu1(self.slots.len(), self.gpu1_slots.len(), *n0, *n1);
        let order = if prefer_gpu1 {
            [true, false]
        } else {
            [false, true]
        };
        for use_gpu1 in order {
            if use_gpu1 {
                let Some(victim) = lru_victim_keep(&self.gpu1_slots, keep_layer) else {
                    continue;
                };
                self.gpu1_slots[victim].key = Some(key);
                self.gpu1_slots[victim].used = self.clock;
                self.gpu1_slots[victim].pinned = pin || self.pin_keep(key);
                *n1 = n1.saturating_add(1);
                return Some((true, victim));
            }
            let Some(victim) = lru_victim_keep(&self.slots, keep_layer) else {
                continue;
            };
            self.slots[victim].key = Some(key);
            self.slots[victim].used = self.clock;
            self.slots[victim].pinned = pin || self.pin_keep(key);
            *n0 = n0.saturating_add(1);
            return Some((false, victim));
        }
        None
    }

    fn plan_expert_copies<'a>(
        &mut self,
        file: &'a GgufFile,
        layer: usize,
        ids: &[usize],
        cfg: &ModelConfig,
        metrics: Option<&Metrics>,
        mode: PlanMode,
    ) -> Result<ExpertCopyPlan<'a>, AllpacaError> {
        let (pin, count_misses, keep_layer, prefetch) = match mode {
            PlanMode::Warmup => (true, true, None, false),
            PlanMode::Prefetch { keep_layer } => (false, false, Some(keep_layer), true),
        };
        let mut plan = ExpertCopyPlan {
            prefetch,
            ..ExpertCopyPlan::empty()
        };
        if !prefetch {
            self.warmup_miss.clear();
        }
        if ids.is_empty() || self.has_prefill_layer(layer) {
            return Ok(plan);
        }
        let t_pack = Instant::now();
        for &expert in ids {
            let key = (layer, expert);
            self.clock = self.clock.saturating_add(1);
            if let Some(i) = self.slots.iter().position(|s| s.key == Some(key)) {
                self.slots[i].used = self.clock;
                if pin || self.pin_keep(key) {
                    self.slots[i].pinned = true;
                }
                continue;
            }
            if let Some(i) = self.gpu1_slots.iter().position(|s| s.key == Some(key)) {
                self.gpu1_slots[i].used = self.clock;
                if pin || self.pin_keep(key) {
                    self.gpu1_slots[i].pinned = true;
                }
                continue;
            }
            if count_misses {
                if let Some(m) = metrics {
                    m.expert_lru_misses.fetch_add(1, Ordering::Relaxed);
                }
            }
            let slices = expert_mxfp4_parts(file, layer, expert, cfg)?;
            let packed_len: usize = slices.iter().map(|(_, s)| s.len()).sum();
            if packed_len > self.expert_slot {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "expert packed larger than slot".into(),
                )));
            }
            let Some((use_gpu1, victim)) =
                self.admit_expert_slot(key, &mut plan.n0, &mut plan.n1, pin, keep_layer)
            else {
                if matches!(mode, PlanMode::Warmup)
                    && self.slots.is_empty()
                    && self.gpu1_slots.is_empty()
                {
                    return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "no expert GPU slot".into(),
                    )));
                }
                continue;
            };
            if !prefetch {
                self.warmup_miss.push(key);
            }
            let base = victim * self.expert_slot;
            let n_parts = slices.len();
            // An expert that earned a GPU slot has usually been run on the
            // CPU, so its bytes are in the RAM tier: copy from there, not
            // through the mapped file.
            let cached = self.ram.as_ref().and_then(|ram| {
                let mut ram = ram.lock().unwrap_or_else(|e| e.into_inner());
                ram.pin(key).filter(|(_, len)| *len == packed_len)
            });
            if cached.is_some() {
                self.ram_pinned.push(key);
            }
            let dest = if use_gpu1 {
                &mut plan.parts1
            } else {
                &mut plan.parts0
            };
            for (rel, sl) in slices {
                let src: &'a [u8] = match cached {
                    // SAFETY: pinned in the RAM tier until `finish_h2d_keep`,
                    // which runs after this plan's copies have landed.
                    Some((ptr, _)) => unsafe { std::slice::from_raw_parts(ptr.add(rel), sl.len()) },
                    None => {
                        self.mapped_used.push((sl.as_ptr() as usize, sl.len()));
                        sl
                    }
                };
                dest.push((base + rel, src));
            }
            if use_gpu1 {
                plan.groups1.push((expert, n_parts));
            } else {
                plan.groups0.push((expert, n_parts));
            }
        }
        add_ns(metrics, |m| &m.dsv4_pack_ns, t_pack);
        Ok(plan)
    }

    fn register_copy_plan(&mut self, plan: &ExpertCopyPlan<'_>, metrics: Option<&Metrics>) {
        let mut all_parts = Vec::with_capacity(plan.parts0.len() + plan.parts1.len());
        all_parts.extend_from_slice(&plan.parts0);
        all_parts.extend_from_slice(&plan.parts1);
        if all_parts.is_empty() {
            return;
        }
        let t_pin = Instant::now();
        let registered = register_mmap_parts(&all_parts);
        if registered.is_ok() {
            add_ns(metrics, |m| &m.dsv4_pack_ns, t_pin);
        } else {
            let bounce = self.pin0.is_some() || self.pin1.is_some();
            static REG_WARN: AtomicBool = AtomicBool::new(false);
            if !REG_WARN.swap(true, Ordering::Relaxed) {
                if bounce {
                    tracing::warn!("hipHostRegister on GGUF mmap failed; pinned bounce expert H2D");
                } else {
                    tracing::warn!("hipHostRegister on GGUF mmap failed; pageable expert H2D");
                }
            }
            if !bounce {
                for (_, sl) in &all_parts {
                    prefault_mmap(sl);
                }
            }
            add_ns(metrics, |m| &m.dsv4_pack_ns, t_pin);
        }
        if let Ok(ptrs) = registered {
            self.h2d_reg.ptrs = ptrs;
        }
    }

    pub(crate) fn submit_copy_plan(
        &self,
        plan: &ExpertCopyPlan<'_>,
        metrics: Option<&Metrics>,
    ) -> (bool, bool) {
        self.submit_copy_plan_notifying(plan, metrics, None)
    }

    /// Same as [`Self::submit_copy_plan`], optionally `send`ing `(expert, ready_idx)`
    /// after that expert's DMA is queued. The compute stream waits via
    /// [`Self::wait_miss_ready`]; the bounce thread does not host-sync the stream.
    /// `ready` is dropped as soon as DMA is queued so miss GEMM is not blocked on
    /// leftover mmap prefault.
    pub(crate) fn submit_copy_plan_notifying(
        &self,
        plan: &ExpertCopyPlan<'_>,
        metrics: Option<&Metrics>,
        ready: Option<Sender<(usize, usize)>>,
    ) -> (bool, bool) {
        if plan.is_empty() {
            return (false, false);
        }
        let hip = self.hip.clone();
        let draft = self.draft.clone();
        let nbytes0: u64 = plan.parts0.iter().map(|(_, s)| s.len() as u64).sum();
        let nbytes1: u64 = plan.parts1.iter().map(|(_, s)| s.len() as u64).sum();
        let t_h2d = Instant::now();
        let mut inflight0 = false;
        let mut inflight1 = false;
        let stop_pf = AtomicBool::new(false);
        std::thread::scope(|s| {
            let bounce = self.h2d_reg.ptrs.is_empty();
            if bounce {
                if !plan.parts0.is_empty() {
                    let parts = &plan.parts0;
                    let stop = &stop_pf;
                    s.spawn(move || prefault_parts_until(parts, stop));
                }
                if !plan.parts1.is_empty() {
                    let parts = &plan.parts1;
                    let stop = &stop_pf;
                    s.spawn(move || prefault_parts_until(parts, stop));
                }
            }
            let j0 = (!plan.parts0.is_empty()).then(|| {
                let blob = self.experts.as_ref();
                let stream = self.h2d.as_ref();
                let pin = bounce.then(|| self.pin0.as_ref()).flatten();
                let hip = hip.clone();
                let parts = &plan.parts0;
                let groups = &plan.groups0;
                let ready = ready.clone();
                s.spawn(move || {
                    let blob = blob.ok_or_else(|| {
                        AllpacaError::Degraded(DegradedError::FeatureDisabled(
                            "GPU0 expert blob missing".into(),
                        ))
                    })?;
                    hip.bind()?;
                    copy_expert_groups(blob, parts, groups, stream, pin, ready.as_ref())
                })
            });
            let j1 = (!plan.parts1.is_empty()).then(|| {
                let blob = self.gpu1.as_ref();
                let stream = self.h2d1.as_ref();
                let pin = bounce.then(|| self.pin1.as_ref()).flatten();
                let draft = draft.clone();
                let parts = &plan.parts1;
                let groups = &plan.groups1;
                let ready = ready.clone();
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
                    copy_expert_groups(blob, parts, groups, stream, pin, ready.as_ref())
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
            add_ns(metrics, |m| &m.dsv4_h2d_ns, t_h2d);
            stop_pf.store(true, Ordering::Release);
            drop(ready);
        });
        let _ = hip.bind();
        add_bytes(metrics, nbytes0.saturating_add(nbytes1));
        if let Some(m) = metrics {
            if plan.n0 > 0 {
                m.expert_lru_fills
                    .fetch_add(plan.n0 as u64, Ordering::Relaxed);
                m.expert_bulk_fills.fetch_add(1, Ordering::Relaxed);
                m.expert_bulk_bytes.fetch_add(nbytes0, Ordering::Relaxed);
                m.expert_h2d_batched_copies
                    .fetch_add(plan.parts0.len() as u64, Ordering::Relaxed);
            }
            if plan.n1 > 0 {
                m.expert_lru_fills
                    .fetch_add(plan.n1 as u64, Ordering::Relaxed);
                m.expert_bulk_fills.fetch_add(1, Ordering::Relaxed);
                m.expert_bulk_bytes.fetch_add(nbytes1, Ordering::Relaxed);
                m.expert_h2d_batched_copies
                    .fetch_add(plan.parts1.len() as u64, Ordering::Relaxed);
            }
            if plan.prefetch && (plan.n0 > 0 || plan.n1 > 0) {
                m.draft_expert_prefetches.fetch_add(1, Ordering::Relaxed);
            }
        }
        (inflight0, inflight1)
    }

    pub(crate) fn mark_expert_h2d(&mut self, inflight0: bool, inflight1: bool) {
        self.h2d_inflight = inflight0;
        self.h2d1_inflight = inflight1;
    }

    pub(crate) fn unpin_layer(&mut self, layer: usize) {
        for slot in self.slots.iter_mut().chain(self.gpu1_slots.iter_mut()) {
            if slot.key.is_some_and(|(l, _)| l == layer) {
                slot.pinned = false;
            }
        }
    }

    pub(crate) fn expert_copy_pending(&self, layer: usize, expert: usize) -> bool {
        self.warmup_miss
            .iter()
            .any(|&(l, e)| l == layer && e == expert)
    }

    /// Pin resident routed experts and register mmap for miss H2D. Does not copy;
    /// caller issues [`Self::submit_copy_plan`] so DMA can overlap shexp / hit GEMM.
    pub fn warmup_experts<'a>(
        &mut self,
        file: &'a GgufFile,
        layer: usize,
        ids: &[usize],
        cfg: &ModelConfig,
        metrics: Option<&Metrics>,
    ) -> Result<ExpertCopyPlan<'a>, AllpacaError> {
        self.reset_expert_h2d_host();
        self.warmup_miss.clear();
        if ids.is_empty() || self.has_prefill_layer(layer) {
            return Ok(ExpertCopyPlan::empty());
        }
        let plan = self.plan_expert_copies(file, layer, ids, cfg, metrics, PlanMode::Warmup)?;
        self.register_copy_plan(&plan, metrics);
        Ok(plan)
    }

    /// Assign L+1 hint IDs into slots that are not `keep_layer`, register mmap, do not copy.
    /// Caller issues [`Self::submit_copy_plan`] on a side thread overlapping L compute.
    pub(crate) fn begin_hint_prefetch<'a>(
        &mut self,
        file: &'a GgufFile,
        layer: usize,
        ids: &[usize],
        cfg: &ModelConfig,
        metrics: Option<&Metrics>,
        keep_layer: usize,
    ) -> Result<ExpertCopyPlan<'a>, AllpacaError> {
        self.reset_expert_h2d_host();
        if ids.is_empty() || self.has_prefill_layer(layer) {
            return Ok(ExpertCopyPlan::empty());
        }
        let plan = self.plan_expert_copies(
            file,
            layer,
            ids,
            cfg,
            metrics,
            PlanMode::Prefetch { keep_layer },
        )?;
        self.register_copy_plan(&plan, metrics);
        Ok(plan)
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

    /// Fold GPU1 `moe_acc` into GPU0 through a pinned bounce (no P2P).
    /// Returns true when GPU1 was inactive or the fold landed on GPU0.
    pub(crate) fn fold_gpu1_moe_into_gpu0(&self) -> bool {
        let Some(h1) = self.draft.as_ref() else {
            return true;
        };
        if !h1.moe_active() {
            return true;
        }
        let h0 = &self.hip;
        if !h0.moe_active() {
            return false;
        }
        if h1.moe_join_compute().is_err() {
            return false;
        }
        let (Ok(src), Ok(n)) = (h1.moe_acc_device(), h1.moe_floats()) else {
            return false;
        };
        let (Ok(staging), Ok(acc)) = (h0.moe_staging_device(), h0.moe_acc_device()) else {
            return false;
        };
        if n == 0 || n > i32::MAX as usize {
            h1.moe_release();
            return true;
        }
        let bytes = n.saturating_mul(4);
        let pin = self.pin0.as_ref();
        let bounce = pin.filter(|p| !p.window && !p.a.is_null() && p.bytes > 0);
        if let Some(pin) = bounce {
            if let Some(h2d) = self.h2d.as_ref() {
                wait_pin_slots(h2d, pin);
            }
            let mut done = 0usize;
            while done < bytes {
                let chunk = (bytes - done).min(pin.bytes);
                let ok = (|| -> Result<(), AllpacaError> {
                    h1.bind()?;
                    // SAFETY: `src`/`staging` are live moe buffers of `bytes`. `pin.a`
                    // is `pin.bytes`. Slot events were waited above so the bounce
                    // buffer is not in flight. `done + chunk <= bytes`.
                    unsafe {
                        memcpy(pin.a, src.cast::<u8>().add(done), chunk, MEMCPY_D2H)?;
                        h0.bind()?;
                        memcpy(staging.add(done), pin.a, chunk, MEMCPY_H2D)?;
                    }
                    Ok(())
                })();
                if ok.is_err() {
                    return false;
                }
                done += chunk;
            }
        } else {
            let mut tmp = vec![0u8; bytes];
            let ok = (|| -> Result<(), AllpacaError> {
                h1.bind()?;
                // `src`/`staging` are live moe buffers of `bytes`. `tmp` is a
                // host bounce of the same length (window-register pin has no
                // `hipHostMalloc` slot).
                memcpy(tmp.as_mut_ptr(), src.cast::<u8>(), bytes, MEMCPY_D2H)?;
                h0.bind()?;
                memcpy(staging, tmp.as_ptr(), bytes, MEMCPY_H2D)?;
                Ok(())
            })();
            if ok.is_err() {
                return false;
            }
        }
        if h0.residual_add(acc, staging.cast(), n as i32).is_err() {
            return false;
        }
        h1.moe_release();
        true
    }

    /// GPU-wait on the miss-ready event recorded for this expert's DMA.
    pub(crate) fn wait_miss_ready(&self, gpu1: bool, idx: usize) -> Result<(), AllpacaError> {
        if idx == usize::MAX {
            return Ok(());
        }
        if gpu1 {
            let d = self.draft.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "GPU1 miss ready without draft HIP".into(),
                ))
            })?;
            d.bind()?;
            self.h2d1
                .as_ref()
                .ok_or_else(|| {
                    AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "GPU1 H2D stream missing".into(),
                    ))
                })?
                .wait_ready_compute(idx)
        } else {
            self.hip.bind()?;
            self.h2d
                .as_ref()
                .ok_or_else(|| {
                    AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "GPU0 H2D stream missing".into(),
                    ))
                })?
                .wait_ready_compute(idx)
        }
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

    fn ensure_q_ws(&mut self, bytes: usize) -> Result<*mut u8, AllpacaError> {
        let bytes = bytes.max(4);
        let reuse = self
            .q_ws
            .as_ref()
            .is_some_and(|blob| blob.bytes() as usize >= bytes);
        if !reuse {
            self.q_ws = Some(DeviceWeights::alloc(self.hip.device, bytes)?);
        }
        self.q_ws
            .as_ref()
            .map(|blob| blob.byte_ptr_at(0) as *mut u8)
            .ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "MLA Q workspace missing".into(),
                ))
            })
    }

    fn ensure_attn_ws(&mut self, bytes: usize) -> Result<*mut u8, AllpacaError> {
        let bytes = bytes.max(4);
        let reuse = self
            .attn_ws
            .as_ref()
            .is_some_and(|blob| blob.bytes() as usize >= bytes);
        if !reuse {
            self.attn_ws = Some(DeviceWeights::alloc(self.hip.device, bytes)?);
        }
        self.attn_ws
            .as_ref()
            .map(|blob| blob.byte_ptr_at(0) as *mut u8)
            .ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "MLA attn workspace missing".into(),
                ))
            })
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
            if let (Some((off, _)), Some(blob)) = (ent.span(n, k, elem_off), &self.dense) {
                return Ok(blob.byte_ptr_at(ent.offset + off) as *const u16);
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
        self.mla_q_rows(
            file, blk, cur, q_a_norm, eps, q_lora, q_out, dim, 1, metrics,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_q_rows(
        &mut self,
        file: &GgufFile,
        blk: usize,
        cur: &[f32],
        q_a_norm: &[f32],
        eps: f32,
        q_lora: usize,
        q_out: usize,
        dim: usize,
        n_tok: usize,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        if n_tok == 0 {
            return Ok(Vec::new());
        }
        let (dq, q_n) = self.mla_q_into_device(
            file, blk, cur, q_a_norm, eps, q_lora, q_out, dim, n_tok, metrics,
        )?;
        let hip = self.hip.clone();
        let mut q = vec![0.0f32; q_n];
        let t = Instant::now();
        hip.copy_d2h_f32(dq, &mut q)?;
        add_ns(metrics, |m| &m.dsv4_d2h_ns, t);
        Ok(q)
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_q_into_device(
        &mut self,
        file: &GgufFile,
        blk: usize,
        cur: &[f32],
        q_a_norm: &[f32],
        eps: f32,
        q_lora: usize,
        q_out: usize,
        dim: usize,
        n_tok: usize,
        metrics: Option<&Metrics>,
    ) -> Result<(*mut f32, usize), AllpacaError> {
        if n_tok == 0 {
            return Ok((std::ptr::null_mut(), 0));
        }
        if n_tok > i32::MAX as usize
            || q_lora > i32::MAX as usize
            || q_out > i32::MAX as usize
            || dim > i32::MAX as usize
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA Q rows shape exceeds i32".into(),
            )));
        }
        let a_need = n_tok.saturating_mul(dim);
        if cur.len() < a_need {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA Q rows buffer size mismatch".into(),
            )));
        }
        let da = self.device_a_if_curs(cur, n_tok, dim);
        let qa = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.attn_q_a.weight"),
            q_lora,
            dim,
            0,
            metrics,
        )?;
        let qb = self.dense_bf16_ptr(
            file,
            &format!("blk.{blk}.attn_q_b.weight"),
            q_out,
            q_lora,
            0,
            metrics,
        )?;
        let hip = self.hip.clone();
        hip.bind()?;
        let qr_n = n_tok.saturating_mul(q_lora);
        let q_n = n_tok.saturating_mul(q_out);
        let mi = n_tok as i32;
        if n_tok == 1
            && q_out <= ACT_SLOT_FLOATS
            && q_lora <= ACT_SLOT_FLOATS
            && dim <= ACT_SLOT_FLOATS
        {
            let s0 = self.act_ptr(0)?;
            let s1 = self.act_ptr(1)?;
            let dcur = match da {
                Some(p) => p,
                None => {
                    let t = Instant::now();
                    hip.copy_h2d_f32(s0, cur)?;
                    add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
                    add_bytes(metrics, (cur.len() * 4) as u64);
                    s0 as *const f32
                }
            };
            let t = Instant::now();
            hip.gemm_nt_bf16_from_device_a_into(dcur, qa, s1, 1, q_lora as i32, dim as i32)?;
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
            if !q_a_norm.is_empty() {
                let nw = self.act_ptr(2)?;
                hip.copy_h2d_f32(nw, q_a_norm)?;
                let t = Instant::now();
                hip.rmsnorm_device_rows(s1, nw, s1, 1, q_lora as i32, eps)?;
                add_ns(metrics, |m| &m.dsv4_norm_ns, t);
            }
            let t = Instant::now();
            hip.gemm_nt_bf16_from_device_a_into(s1, qb, s0, 1, q_out as i32, q_lora as i32)?;
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
            self.used.store(true, Ordering::Relaxed);
            return Ok((s0, q_n));
        }
        let a_n = if da.is_some() { 0 } else { a_need };
        let nw_n = if q_a_norm.is_empty() { 0 } else { q_lora };
        let need = (qr_n
            .saturating_add(q_n)
            .saturating_add(nw_n)
            .saturating_add(a_n))
        .saturating_mul(4);
        let base = self.ensure_q_ws(need)?;
        // SAFETY: `ensure_q_ws` allocated at least `need` bytes; offsets are
        // qr | q | norm | optional host-A staging.
        let dqr = base.cast::<f32>();
        let dq = unsafe { base.add(qr_n.saturating_mul(4)) }.cast::<f32>();
        let dnw = unsafe { base.add((qr_n.saturating_add(q_n)).saturating_mul(4)) }.cast::<f32>();
        let dcur =
            unsafe { base.add((qr_n.saturating_add(q_n).saturating_add(nw_n)).saturating_mul(4)) }
                .cast::<f32>();
        let a_ptr = match da {
            Some(p) => p,
            None => {
                let t = Instant::now();
                hip.copy_h2d_f32(dcur, &cur[..a_need])?;
                add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
                add_bytes(metrics, (a_need * 4) as u64);
                dcur as *const f32
            }
        };
        let t = Instant::now();
        hip.gemm_nt_bf16_from_device_a_into(a_ptr, qa, dqr, mi, q_lora as i32, dim as i32)?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        if nw_n > 0 {
            hip.copy_h2d_f32(dnw, &q_a_norm[..q_lora])?;
            let t = Instant::now();
            hip.rmsnorm_device_rows(dqr, dnw, dqr, mi, q_lora as i32, eps)?;
            add_ns(metrics, |m| &m.dsv4_norm_ns, t);
        }
        let t = Instant::now();
        hip.gemm_nt_bf16_from_device_a_into(dqr, qb, dq, mi, q_out as i32, q_lora as i32)?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        self.used.store(true, Ordering::Relaxed);
        Ok((dq, q_n))
    }

    /// Q projection + per-head ones-RMSNorm + partial RoPE, left on device.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_q_norm_rope_device(
        &mut self,
        file: &GgufFile,
        blk: usize,
        cur: &[f32],
        q_a_norm: &[f32],
        eps: f32,
        q_lora: usize,
        q_out: usize,
        dim: usize,
        n_tok: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        theta: f32,
        n_rot: usize,
        rope_off: usize,
        metrics: Option<&Metrics>,
    ) -> Result<*const f32, AllpacaError> {
        if n_heads == 0
            || head_dim == 0
            || q_out != n_heads.saturating_mul(head_dim)
            || n_tok > i32::MAX as usize
            || n_heads > i32::MAX as usize
            || head_dim > i32::MAX as usize
            || start_pos > i32::MAX as usize
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA Q rope device shape mismatch".into(),
            )));
        }
        let (dq, q_n) = self.mla_q_into_device(
            file, blk, cur, q_a_norm, eps, q_lora, q_out, dim, n_tok, metrics,
        )?;
        if dq.is_null() || q_n != n_tok.saturating_mul(q_out) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA Q device pointer missing".into(),
            )));
        }
        let hip = self.hip.clone();
        hip.bind()?;
        let ones = vec![1.0f32; head_dim];
        let dones = if n_tok == 1 && q_out <= ACT_SLOT_FLOATS {
            let p = self.act_ptr(2)?;
            hip.copy_h2d_f32(p, &ones)?;
            p
        } else {
            let qr_n = n_tok.saturating_mul(q_lora);
            if qr_n < head_dim {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "MLA Q workspace too small for ones".into(),
                )));
            }
            let base = self.ensure_q_ws((qr_n.saturating_add(q_n)).saturating_mul(4))?;
            let p = base.cast::<f32>();
            hip.copy_h2d_f32(p, &ones)?;
            p
        };
        let rows = n_tok.saturating_mul(n_heads);
        let t = Instant::now();
        hip.rmsnorm_device_rows(dq, dones, dq, rows as i32, head_dim as i32, eps)?;
        add_ns(metrics, |m| &m.dsv4_norm_ns, t);
        hip.rope_offset_device_rows(
            dq,
            n_tok as i32,
            n_heads as i32,
            head_dim as i32,
            n_rot as i32,
            rope_off as i32,
            start_pos as i32,
            theta,
            false,
        )?;
        Ok(dq as *const f32)
    }

    pub fn copy_d2h_f32(&self, src: *const f32, dst: &mut [f32]) -> Result<(), AllpacaError> {
        self.hip.copy_d2h_f32(src, dst)
    }

    pub fn attention_sinks_from_device_q(
        &self,
        q: *const f32,
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
        self.hip.attention_sinks_from_device_q(
            q, k, v, sinks, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
        )
    }

    /// Attention + inverse RoPE, left on device for `mla_o_rows_from_device`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_rope_back_from_device_q(
        &mut self,
        q: *const f32,
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        n_tok: usize,
        n_kv: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_pos: usize,
        theta: f32,
        n_rot: usize,
        rope_off: usize,
    ) -> Result<*const f32, AllpacaError> {
        if q.is_null()
            || n_tok == 0
            || n_heads == 0
            || head_dim == 0
            || n_tok > i32::MAX as usize
            || n_kv > i32::MAX as usize
            || n_heads > i32::MAX as usize
            || n_kv_heads > i32::MAX as usize
            || head_dim > i32::MAX as usize
            || start_pos > i32::MAX as usize
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "device attn rope_back shape mismatch".into(),
            )));
        }
        let q_n = n_tok.saturating_mul(n_heads).saturating_mul(head_dim);
        let base = self.ensure_attn_ws(q_n.saturating_mul(4))?;
        let dout = base.cast::<f32>();
        self.hip.attention_sinks_from_device_q_into(
            q,
            k,
            v,
            sinks,
            dout,
            n_tok as i32,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            true,
        )?;
        self.hip.rope_offset_device_rows(
            dout,
            n_tok as i32,
            n_heads as i32,
            head_dim as i32,
            n_rot as i32,
            rope_off as i32,
            start_pos as i32,
            theta,
            true,
        )?;
        self.used.store(true, Ordering::Relaxed);
        Ok(dout as *const f32)
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

    #[allow(clippy::too_many_arguments)]
    pub fn mla_o_rows(
        &mut self,
        file: &GgufFile,
        blk: usize,
        attn_raws: &[f32],
        n_tok: usize,
        q_out: usize,
        n_heads: usize,
        head_dim: usize,
        groups: usize,
        per: usize,
        dim: usize,
        metrics: Option<&Metrics>,
        skip_d2h: bool,
    ) -> Result<Vec<f32>, AllpacaError> {
        let groups = groups.max(1);
        let per = per.max(1);
        let heads_g = (n_heads / groups).max(1);
        let o_group_dim = heads_g.saturating_mul(head_dim);
        let oa_name = format!("blk.{blk}.attn_output_a.weight");
        let ob_name = format!("blk.{blk}.attn_output_b.weight");
        self.mla_o_rows_core(
            attn_raws,
            None,
            n_tok,
            q_out,
            o_group_dim,
            groups,
            per,
            dim,
            metrics,
            |gpu, g| {
                gpu.dense_bf16_ptr(
                    file,
                    &oa_name,
                    per,
                    o_group_dim,
                    g.saturating_mul(per).saturating_mul(o_group_dim),
                    metrics,
                )
            },
            |gpu| gpu.dense_bf16_ptr(file, &ob_name, dim, per.saturating_mul(groups), 0, metrics),
            skip_d2h,
        )
    }

    pub fn last_o_device(&self) -> Option<*const f32> {
        if self.o_out == 0 || self.o_out_n == 0 {
            None
        } else {
            Some(self.o_out as *const f32)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_o_rows_from_device(
        &mut self,
        file: &GgufFile,
        blk: usize,
        dattn: *const f32,
        n_tok: usize,
        q_out: usize,
        n_heads: usize,
        head_dim: usize,
        groups: usize,
        per: usize,
        dim: usize,
        metrics: Option<&Metrics>,
        skip_d2h: bool,
    ) -> Result<Vec<f32>, AllpacaError> {
        let groups = groups.max(1);
        let per = per.max(1);
        let heads_g = (n_heads / groups).max(1);
        let o_group_dim = heads_g.saturating_mul(head_dim);
        let oa_name = format!("blk.{blk}.attn_output_a.weight");
        let ob_name = format!("blk.{blk}.attn_output_b.weight");
        self.mla_o_rows_core(
            &[],
            Some(dattn),
            n_tok,
            q_out,
            o_group_dim,
            groups,
            per,
            dim,
            metrics,
            |gpu, g| {
                gpu.dense_bf16_ptr(
                    file,
                    &oa_name,
                    per,
                    o_group_dim,
                    g.saturating_mul(per).saturating_mul(o_group_dim),
                    metrics,
                )
            },
            |gpu| gpu.dense_bf16_ptr(file, &ob_name, dim, per.saturating_mul(groups), 0, metrics),
            skip_d2h,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_o_rows_with_ptrs(
        &mut self,
        attn_raws: &[f32],
        n_tok: usize,
        q_out: usize,
        o_group_dim: usize,
        groups: usize,
        per: usize,
        dim: usize,
        o_a: &[*const u16],
        o_b: *const u16,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        if o_a.len() != groups || o_b.is_null() || o_a.iter().any(|p| p.is_null()) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA O rows weight pointer mismatch".into(),
            )));
        }
        self.mla_o_rows_core(
            attn_raws,
            None,
            n_tok,
            q_out,
            o_group_dim,
            groups,
            per,
            dim,
            metrics,
            |_, g| Ok(o_a[g]),
            |_| Ok(o_b),
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_o_rows_with_ptrs_from_device(
        &mut self,
        dattn: *const f32,
        n_tok: usize,
        q_out: usize,
        o_group_dim: usize,
        groups: usize,
        per: usize,
        dim: usize,
        o_a: &[*const u16],
        o_b: *const u16,
        metrics: Option<&Metrics>,
        skip_d2h: bool,
    ) -> Result<Vec<f32>, AllpacaError> {
        if o_a.len() != groups || o_b.is_null() || o_a.iter().any(|p| p.is_null()) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA O rows weight pointer mismatch".into(),
            )));
        }
        self.mla_o_rows_core(
            &[],
            Some(dattn),
            n_tok,
            q_out,
            o_group_dim,
            groups,
            per,
            dim,
            metrics,
            |_, g| Ok(o_a[g]),
            |_| Ok(o_b),
            skip_d2h,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_o_rows_core<Fa, Fb>(
        &mut self,
        attn_raws: &[f32],
        dattn: Option<*const f32>,
        n_tok: usize,
        q_out: usize,
        o_group_dim: usize,
        groups: usize,
        per: usize,
        dim: usize,
        metrics: Option<&Metrics>,
        mut o_a_at: Fa,
        mut o_b_at: Fb,
        skip_d2h: bool,
    ) -> Result<Vec<f32>, AllpacaError>
    where
        Fa: FnMut(&mut Self, usize) -> Result<*const u16, AllpacaError>,
        Fb: FnMut(&mut Self) -> Result<*const u16, AllpacaError>,
    {
        if n_tok == 0 {
            return Ok(Vec::new());
        }
        let oa_out = per.saturating_mul(groups);
        if n_tok > i32::MAX as usize
            || o_group_dim > i32::MAX as usize
            || per > i32::MAX as usize
            || dim > i32::MAX as usize
            || oa_out > i32::MAX as usize
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA O rows shape exceeds i32".into(),
            )));
        }
        let host_ok = attn_raws.len() >= n_tok.saturating_mul(q_out);
        let dev_ok = dattn.is_some_and(|p| !p.is_null());
        if groups == 0 || q_out < groups.saturating_mul(o_group_dim) || !(host_ok || dev_ok) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA O rows buffer size mismatch".into(),
            )));
        }
        let hip = self.hip.clone();
        hip.bind()?;
        let xs_n = n_tok.saturating_mul(o_group_dim);
        let ys_n = n_tok.saturating_mul(per);
        let oa_n = n_tok.saturating_mul(oa_out);
        let out_n = n_tok.saturating_mul(dim);
        let need = (xs_n
            .saturating_add(ys_n)
            .saturating_add(oa_n)
            .saturating_add(out_n))
        .saturating_mul(4);
        let base = self.ensure_q_ws(need)?;
        // SAFETY: `ensure_q_ws` allocated at least `need` bytes; layout is
        // xs | ys | oa | out.
        let dxs = base.cast::<f32>();
        let dys = unsafe { base.add(xs_n.saturating_mul(4)) }.cast::<f32>();
        let doa = unsafe { base.add((xs_n.saturating_add(ys_n)).saturating_mul(4)) }.cast::<f32>();
        let dout =
            unsafe { base.add((xs_n.saturating_add(ys_n).saturating_add(oa_n)).saturating_mul(4)) }
                .cast::<f32>();
        let mi = n_tok as i32;
        for g in 0..groups {
            let oa_w = o_a_at(self, g)?;
            if oa_w.is_null() {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "MLA O rows o_a pointer is null".into(),
                )));
            }
            if let Some(p) = dattn {
                for t in 0..n_tok {
                    // SAFETY: `p` is `[n_tok, q_out]` on device; `dxs` is `[n_tok, o_group_dim]`.
                    let src = unsafe {
                        p.add(
                            t.saturating_mul(q_out)
                                .saturating_add(g.saturating_mul(o_group_dim)),
                        )
                    };
                    let dst = unsafe { dxs.add(t.saturating_mul(o_group_dim)) };
                    hip.copy_d2d_f32(dst, src, o_group_dim)?;
                }
            } else {
                let mut xs = vec![0.0f32; xs_n];
                for t in 0..n_tok {
                    let src = t
                        .saturating_mul(q_out)
                        .saturating_add(g.saturating_mul(o_group_dim));
                    let dst = t.saturating_mul(o_group_dim);
                    if src + o_group_dim <= attn_raws.len() && dst + o_group_dim <= xs.len() {
                        xs[dst..dst + o_group_dim]
                            .copy_from_slice(&attn_raws[src..src + o_group_dim]);
                    }
                }
                let t = Instant::now();
                hip.copy_h2d_f32(dxs, &xs)?;
                add_ns(metrics, |m| &m.dsv4_h2d_ns, t);
                add_bytes(metrics, (xs_n * 4) as u64);
            }
            let t = Instant::now();
            hip.gemm_nt_bf16_from_device_a_into(
                dxs as *const f32,
                oa_w,
                dys,
                mi,
                per as i32,
                o_group_dim as i32,
            )?;
            add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
            for tok in 0..n_tok {
                // SAFETY: `doa` is `[n_tok, oa_out]`; `dys` is `[n_tok, per]`.
                let dst = unsafe {
                    doa.add(
                        tok.saturating_mul(oa_out)
                            .saturating_add(g.saturating_mul(per)),
                    )
                };
                let src = unsafe { dys.add(tok.saturating_mul(per)) };
                hip.copy_d2d_f32(dst, src, per)?;
            }
        }
        let o_b = o_b_at(self)?;
        if o_b.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MLA O rows o_b pointer is null".into(),
            )));
        }
        let t = Instant::now();
        hip.gemm_nt_bf16_from_device_a_into(
            doa as *const f32,
            o_b,
            dout,
            mi,
            dim as i32,
            oa_out as i32,
        )?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        self.o_out = dout as usize;
        self.o_out_n = out_n;
        self.used.store(true, Ordering::Relaxed);
        if skip_d2h {
            return Ok(Vec::new());
        }
        let mut out = vec![0.0f32; out_n];
        let t = Instant::now();
        hip.copy_d2h_f32(dout, &mut out)?;
        add_ns(metrics, |m| &m.dsv4_d2h_ns, t);
        Ok(out)
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
        if let Some((gate, up, down)) = self.shexp_resident_ptrs(blk, dim, ff) {
            return self.shexp_swiglu_with_ptrs(x, n_tok, dim, ff, gate, up, down, metrics);
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

    pub(crate) fn shexp_resident_ptrs(
        &self,
        blk: usize,
        dim: usize,
        ff: usize,
    ) -> Option<(*const u16, *const u16, *const u16)> {
        let gate_n = ff.saturating_mul(dim).saturating_mul(2);
        let down_n = dim.saturating_mul(ff).saturating_mul(2);
        let gate = self.dense_entry_ptr(&format!("blk.{blk}.ffn_gate_shexp.weight"), gate_n)?;
        let up = self.dense_entry_ptr(&format!("blk.{blk}.ffn_up_shexp.weight"), gate_n)?;
        let down = self.dense_entry_ptr(&format!("blk.{blk}.ffn_down_shexp.weight"), down_n)?;
        Some((gate, up, down))
    }

    /// `packed_need`: the tensor's size as BF16 (2 bytes per value).
    fn dense_entry_ptr(&self, name: &str, packed_need: usize) -> Option<*const u16> {
        let ent = self.dense_map.get(name).copied()?;
        let need = if ent.q8 {
            packed_need / 2 / 32 * allpaca_hip::Q8_STAGED_BLOCK
        } else {
            packed_need
        };
        if need > ent.bytes {
            return None;
        }
        let blob = self.dense.as_ref()?;
        Some(blob.byte_ptr_at(ent.offset).cast())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shexp_swiglu_with_ptrs(
        &self,
        x: &[f32],
        n_tok: usize,
        dim: usize,
        ff: usize,
        gate: *const u16,
        up: *const u16,
        down: *const u16,
        metrics: Option<&Metrics>,
    ) -> Result<Vec<f32>, AllpacaError> {
        if n_tok == 0 {
            return Ok(Vec::new());
        }
        if n_tok > i32::MAX as usize || x.len() < n_tok.saturating_mul(dim) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "shexp GEMM buffer mismatch".into(),
            )));
        }
        let hip = self.hip.clone();
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

    /// BF16 shared-expert SwiGLU from device `moe_x` into `moe_acc`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn shexp_accumulate_into_moe(
        &self,
        n_tok: usize,
        dim: usize,
        ff: usize,
        gate: *const u16,
        up: *const u16,
        down: *const u16,
        metrics: Option<&Metrics>,
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 {
            return Ok(());
        }
        if n_tok > i32::MAX as usize || gate.is_null() || up.is_null() || down.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "shexp accumulate mismatch".into(),
            )));
        }
        let hip = self.hip.clone();
        hip.bind()?;
        let t = Instant::now();
        hip.bf16_swiglu_accumulate(gate, up, down, n_tok as i32, dim as i32, ff as i32)?;
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn qstar_gpu_count(&self, misses: usize) -> usize {
        if self.slots.is_empty() && self.gpu1_slots.is_empty() {
            return 0;
        }
        qstar_fill_count(misses, 12.0e9, 8.0e9)
    }

    fn f32_ptr(&self, name: &str) -> Option<*const f32> {
        let ent = self.f32_map.get(name)?;
        let blob = self.f32_dense.as_ref()?;
        if ent.bytes == 0 {
            return None;
        }
        Some(blob.byte_ptr_at(ent.offset).cast())
    }

    fn hc_workspace_floats(n_tok: usize, hc: usize, dim: usize) -> usize {
        let a_n = n_tok.saturating_mul(hc.saturating_mul(dim));
        let mix_dim = (2usize.saturating_add(hc)).saturating_mul(hc);
        let c_n = n_tok.saturating_mul(mix_dim);
        let aux_n = n_tok
            .saturating_mul(dim)
            .saturating_add(n_tok.saturating_mul(hc))
            .saturating_add(n_tok.saturating_mul(hc).saturating_mul(hc))
            .saturating_add(dim)
            .saturating_add(n_tok.saturating_mul(dim));
        a_n.saturating_mul(2)
            .saturating_add(c_n)
            .max(a_n.saturating_mul(3).saturating_add(aux_n))
    }

    fn ensure_hc_ones(&mut self, n: usize) -> Result<*const f32, AllpacaError> {
        let bytes = n.saturating_mul(4).max(4);
        let reuse = self
            .hc_ones
            .as_ref()
            .is_some_and(|blob| blob.bytes() as usize >= bytes);
        if !reuse {
            let blob = DeviceWeights::alloc(self.hip.device, bytes)?;
            let ones = vec![1.0f32; n.max(1)];
            let raw = unsafe {
                std::slice::from_raw_parts(ones.as_ptr().cast::<u8>(), ones.len().saturating_mul(4))
            };
            blob.copy_h2d(0, raw)?;
            self.hc_ones = Some(blob);
        }
        self.hc_ones
            .as_ref()
            .map(|blob| blob.byte_ptr_at(0).cast())
            .ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "HC ones buffer missing".into(),
                ))
            })
    }

    fn ensure_hc_act(&mut self, floats: usize) -> Result<*mut u8, AllpacaError> {
        let bytes = floats.saturating_mul(4).max(4);
        let reuse = self
            .hc_act
            .as_ref()
            .is_some_and(|blob| blob.bytes() as usize >= bytes);
        if !reuse {
            self.hc_on_device = false;
            self.hc_curs = 0;
            self.hc_curs_n = 0;
            self.hc_curs_host = 0;
            self.hc_act = Some(DeviceWeights::alloc(self.hip.device, bytes)?);
        }
        self.hc_act
            .as_ref()
            .map(|blob| blob.byte_ptr_at(0) as *mut u8)
            .ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "HC act buffer missing".into(),
                ))
            })
    }

    pub fn invalidate_hc_device(&mut self) {
        self.hc_on_device = false;
        self.hc_curs = 0;
        self.hc_curs_n = 0;
        self.hc_curs_host = 0;
        self.hc_curs_synced = false;
    }

    /// D2H mix_pre `curs` once when a host reader needs the bytes.
    pub fn ensure_hc_curs_host(&mut self, curs: &mut [f32]) -> Result<(), AllpacaError> {
        if self.hc_curs == 0 || self.hc_curs_n == 0 || self.hc_curs_synced {
            return Ok(());
        }
        if curs.as_ptr() as usize != self.hc_curs_host || curs.len() < self.hc_curs_n {
            return Ok(());
        }
        let p = self.hc_curs as *const f32;
        let n = self.hc_curs_n;
        self.hip.copy_d2h_f32(p, &mut curs[..n])?;
        self.hc_curs_synced = true;
        Ok(())
    }

    pub(crate) fn hc_curs_device(&self) -> Option<(*const f32, usize)> {
        if self.hc_curs == 0 || self.hc_curs_n == 0 {
            None
        } else {
            Some((self.hc_curs as *const f32, self.hc_curs_n))
        }
    }

    fn device_a_if_curs(&self, x: &[f32], m: usize, k: usize) -> Option<*const f32> {
        let need = m.saturating_mul(k);
        if self.hc_curs == 0
            || self.hc_curs_n == 0
            || self.hc_curs_n != need
            || x.len() != need
            || x.as_ptr() as usize != self.hc_curs_host
        {
            None
        } else {
            Some(self.hc_curs as *const f32)
        }
    }

    pub fn download_hc_packed(&mut self, packed: &mut [f32]) -> Result<(), AllpacaError> {
        if !self.hc_on_device {
            return Ok(());
        }
        let a_n = self.hc_rows.saturating_mul(self.hc_hc_dim);
        if a_n == 0 || packed.len() < a_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC packed download size mismatch".into(),
            )));
        }
        let base = self.ensure_hc_act(a_n.saturating_mul(2))?;
        let live = self.hc_live.min(1);
        let src = unsafe { base.add(live.saturating_mul(a_n).saturating_mul(4)) }.cast::<f32>();
        self.hip.copy_d2h_f32(src, &mut packed[..a_n])?;
        Ok(())
    }

    fn mix_weight_ptr(
        &mut self,
        fn_w: &[f32],
        fn_name: Option<&str>,
    ) -> Result<*const f32, AllpacaError> {
        if let Some(name) = fn_name {
            if let Some(ptr) = self.f32_ptr(name) {
                return Ok(ptr);
            }
        }
        let nbytes = fn_w.len().saturating_mul(4);
        let slot = self.stream.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "no HC mix stream slot".into(),
            ))
        })?;
        if nbytes > STREAM_SLOT_BYTES {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC mix weight exceeds stream slot".into(),
            )));
        }
        let raw = unsafe { std::slice::from_raw_parts(fn_w.as_ptr().cast::<u8>(), nbytes) };
        slot.copy_h2d(0, raw)?;
        Ok(slot.byte_ptr_at(0).cast())
    }

    /// Unit RMSNorm of packed HC streams then mix GEMM `M=n_tok`, `N=mix_dim`, `K=hc_dim`.
    pub fn hc_mix_logits(
        &mut self,
        packed: &[f32],
        fn_w: &[f32],
        fn_name: Option<&str>,
        n_tok: usize,
        hc_dim: usize,
        mix_dim: usize,
        eps: f32,
        mixes: &mut [f32],
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 {
            return Ok(());
        }
        if n_tok > i32::MAX as usize || hc_dim > i32::MAX as usize || mix_dim > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC mix shape exceeds i32".into(),
            )));
        }
        let a_n = n_tok.saturating_mul(hc_dim);
        let c_n = n_tok.saturating_mul(mix_dim);
        let b_n = mix_dim.saturating_mul(hc_dim);
        if packed.len() < a_n || mixes.len() < c_n || fn_w.len() < b_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC mix buffer size mismatch".into(),
            )));
        }
        if let Err(e) = self.hc_mix_logits_fused(
            packed, fn_w, fn_name, n_tok, hc_dim, mix_dim, eps, mixes, a_n, c_n,
        ) {
            tracing::debug!(error = %e, "fused HC mix failed; scratch GEMM");
            if self.hc_on_device {
                return Err(e);
            }
            self.hc_mix_logits_scratch(packed, fn_w, fn_name, n_tok, hc_dim, mix_dim, eps, mixes)?;
        }
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hc_mix_logits_fused(
        &mut self,
        packed: &[f32],
        fn_w: &[f32],
        fn_name: Option<&str>,
        n_tok: usize,
        hc_dim: usize,
        mix_dim: usize,
        eps: f32,
        mixes: &mut [f32],
        a_n: usize,
        c_n: usize,
    ) -> Result<(), AllpacaError> {
        let ones = self.ensure_hc_ones(hc_dim)?;
        let base = self.ensure_hc_act(a_n.saturating_mul(4).saturating_add(c_n))?;
        let bptr = self.mix_weight_ptr(fn_w, fn_name)?;
        let hip = self.hip.clone();
        let shape_ok = self.hc_on_device && self.hc_rows == n_tok && self.hc_hc_dim == hc_dim;
        if !shape_ok {
            self.hc_live = 0;
            self.hc_on_device = false;
        }
        let live = self.hc_live.min(1);
        let dx = unsafe { base.add(live.saturating_mul(a_n).saturating_mul(4)) }.cast::<f32>();
        let dnorm =
            unsafe { base.add((1 - live).saturating_mul(a_n).saturating_mul(4)) }.cast::<f32>();
        let dc = unsafe { base.add(a_n.saturating_mul(8)) }.cast::<f32>();
        hip.bind()?;
        if !shape_ok {
            hip.copy_h2d_f32(dx, &packed[..a_n])?;
        }
        hip.rmsnorm_device_rows(dx, ones, dnorm, n_tok as i32, hc_dim as i32, eps)?;
        gemm_nt_device(dnorm, bptr, dc, n_tok as i32, mix_dim as i32, hc_dim as i32)?;
        hip.copy_d2h_f32(dc, &mut mixes[..c_n])?;
        self.hc_rows = n_tok;
        self.hc_hc_dim = hc_dim;
        self.hc_on_device = true;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn hc_mix_logits_scratch(
        &mut self,
        packed: &[f32],
        fn_w: &[f32],
        fn_name: Option<&str>,
        n_tok: usize,
        hc_dim: usize,
        mix_dim: usize,
        eps: f32,
        mixes: &mut [f32],
    ) -> Result<(), AllpacaError> {
        let hip = self.hip.clone();
        let a_n = n_tok.saturating_mul(hc_dim);
        let mut normed = vec![0.0f32; a_n];
        let ones = vec![1.0f32; hc_dim];
        hip.rmsnorm_rows(packed, &ones, &mut normed, n_tok as i32, hc_dim as i32, eps)?;
        if let Some(ptr) = fn_name.and_then(|n| self.f32_ptr(n)) {
            hip.gemm_nt_dev_b(
                &normed,
                ptr,
                mixes,
                n_tok as i32,
                mix_dim as i32,
                hc_dim as i32,
            )
        } else {
            hip.gemm_nt(
                &normed,
                fn_w,
                mixes,
                n_tok as i32,
                mix_dim as i32,
                hc_dim as i32,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_mix_pre(
        &mut self,
        pre: &[f32],
        n_tok: usize,
        hc: usize,
        dim: usize,
        curs: &mut [f32],
        norm_w: Option<&[f32]>,
        eps: f32,
        copy_host: bool,
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 {
            return Ok(());
        }
        if !self.hc_on_device || self.hc_rows != n_tok || hc == 0 || hc > 8 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC mix_pre needs resident packed streams".into(),
            )));
        }
        let hc_dim = hc.saturating_mul(dim);
        if self.hc_hc_dim != hc_dim {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC mix_pre hc_dim mismatch".into(),
            )));
        }
        let a_n = n_tok.saturating_mul(hc_dim);
        let curs_n = n_tok.saturating_mul(dim);
        let pre_n = n_tok.saturating_mul(hc);
        if pre.len() < pre_n || curs.len() < curs_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC mix_pre buffer size mismatch".into(),
            )));
        }
        let base = self.ensure_hc_act(Self::hc_workspace_floats(n_tok, hc, dim))?;
        if !self.hc_on_device {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC act realloc dropped packed streams".into(),
            )));
        }
        let live = self.hc_live.min(1);
        let packed = unsafe { base.add(live.saturating_mul(a_n).saturating_mul(4)) }.cast::<f32>();
        let aux = unsafe { base.add(a_n.saturating_mul(12)) };
        let dcurs = aux.cast::<f32>();
        let dpre = unsafe { aux.add(curs_n.saturating_mul(4)) }.cast::<f32>();
        let hip = self.hip.clone();
        hip.bind()?;
        hip.copy_h2d_f32(dpre, &pre[..pre_n])?;
        hip.hc_mix_pre_device(packed, dpre, dcurs, n_tok as i32, hc as i32, dim as i32)?;
        if let Some(w) = norm_w {
            if w.len() >= dim {
                let dw = unsafe { aux.add((curs_n.saturating_add(pre_n)).saturating_mul(4)) }
                    .cast::<f32>();
                hip.copy_h2d_f32(dw, &w[..dim])?;
                hip.rmsnorm_device_rows(dcurs, dw, dcurs, n_tok as i32, dim as i32, eps)?;
            }
        }
        self.hc_curs = dcurs as usize;
        self.hc_curs_n = curs_n;
        self.hc_curs_host = curs.as_ptr() as usize;
        if copy_host {
            hip.copy_d2h_f32(dcurs, &mut curs[..curs_n])?;
            self.hc_curs_synced = true;
        } else {
            self.hc_curs_synced = false;
        }
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post_packed(
        &mut self,
        packed: &[f32],
        xs: &[f32],
        posts: &[f32],
        combs: &[f32],
        n_tok: usize,
        hc: usize,
        dim: usize,
        xs_dev: Option<*const f32>,
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 {
            return Ok(());
        }
        if hc == 0 || hc > 8 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC post hc out of range".into(),
            )));
        }
        let hc_dim = hc.saturating_mul(dim);
        let a_n = n_tok.saturating_mul(hc_dim);
        let xs_n = n_tok.saturating_mul(dim);
        let post_n = n_tok.saturating_mul(hc);
        let comb_n = n_tok.saturating_mul(hc).saturating_mul(hc);
        if packed.len() < a_n || posts.len() < post_n || combs.len() < comb_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC post buffer size mismatch".into(),
            )));
        }
        if xs_dev.is_none() && xs.len() < xs_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HC post buffer size mismatch".into(),
            )));
        }
        let base = self.ensure_hc_act(Self::hc_workspace_floats(n_tok, hc, dim))?;
        let hip = self.hip.clone();
        hip.bind()?;
        let shape_ok = self.hc_on_device && self.hc_rows == n_tok && self.hc_hc_dim == hc_dim;
        if !shape_ok {
            self.hc_live = 0;
            self.hc_on_device = false;
            let dx = base.cast::<f32>();
            hip.copy_h2d_f32(dx, &packed[..a_n])?;
            self.hc_rows = n_tok;
            self.hc_hc_dim = hc_dim;
            self.hc_on_device = true;
        }
        let live = self.hc_live.min(1);
        let other = 1 - live;
        let src = unsafe { base.add(live.saturating_mul(a_n).saturating_mul(4)) }.cast::<f32>();
        let dst = unsafe { base.add(other.saturating_mul(a_n).saturating_mul(4)) }.cast::<f32>();
        let aux = unsafe { base.add(a_n.saturating_mul(12)) };
        let dxs_scratch = aux.cast::<f32>();
        let dpost = unsafe { aux.add(xs_n.saturating_mul(4)) }.cast::<f32>();
        let dcomb =
            unsafe { aux.add((xs_n.saturating_add(post_n)).saturating_mul(4)) }.cast::<f32>();
        let dxs = match xs_dev {
            Some(p) if !p.is_null() => p,
            Some(_) => {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "HC post null device xs".into(),
                )));
            }
            None => {
                hip.copy_h2d_f32(dxs_scratch, &xs[..xs_n])?;
                dxs_scratch
            }
        };
        hip.copy_h2d_f32(dpost, &posts[..post_n])?;
        hip.copy_h2d_f32(dcomb, &combs[..comb_n])?;
        hip.hc_post_device(
            dxs,
            src,
            dpost,
            dcomb,
            dst,
            n_tok as i32,
            hc as i32,
            dim as i32,
        )?;
        self.hc_live = other;
        self.hc_on_device = true;
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn rmsnorm_rows(
        &mut self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > i32::MAX as usize || n > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm_rows shape exceeds i32".into(),
            )));
        }
        self.hip
            .rmsnorm_rows(x, weight, out, rows as i32, n as i32, eps)?;
        self.used.store(true, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(all(test, feature = "hip"))]
impl Dsv4Gpu {
    fn for_hc_test(hip: std::sync::Arc<HipWorkspace>) -> Self {
        let stream = DeviceWeights::alloc(hip.device, STREAM_SLOT_BYTES).ok();
        Self {
            hip,
            draft: None,
            dense: None,
            dense_map: HashMap::new(),
            f32_dense: None,
            f32_map: HashMap::new(),
            stream,
            stream_off: 0,
            experts: None,
            expert_slot: 0,
            slots: Vec::new(),
            gpu1: None,
            gpu1_device: -1,
            gpu1_slots: Vec::new(),
            prefill: None,
            prefill_slot: 0,
            prefill_n: 0,
            prefill_layers: [None, None],
            prefill_used: [0, 0],
            prefill_device: 0,
            clock: 0,
            keep_set: HashSet::new(),
            heat: HashMap::new(),
            heat_routes: 0,
            cpu_miss: false,
            ram: None,
            mapped_used: Vec::new(),
            ram_pinned: Vec::new(),
            read_ahead: None,
            h2d: None,
            h2d1: None,
            pin0: None,
            pin1: None,
            h2d_keep: Vec::new(),
            h2d_inflight: false,
            h2d1_inflight: false,
            h2d_reg: RegisteredMmap { ptrs: Vec::new() },
            act: None,
            q_ws: None,
            attn_ws: None,
            hc_ones: None,
            hc_act: None,
            hc_live: 0,
            hc_rows: 0,
            hc_hc_dim: 0,
            hc_on_device: false,
            hc_curs: 0,
            hc_curs_n: 0,
            hc_curs_host: 0,
            hc_curs_synced: false,
            o_out: 0,
            o_out_n: 0,
            warmup_miss: Vec::new(),
            used: AtomicBool::new(false),
        }
    }
}

/// Full-layer expert H2D is cheaper than 6-of-256 only when most experts fire.
pub(crate) fn want_full_layer_prefetch(prefill: bool, cfg: &ModelConfig) -> bool {
    prefill && cfg.n_experts > 0 && cfg.n_active.saturating_mul(8) >= cfg.n_experts
}

/// Hint-prefetch L's routed IDs as L+1 only when the active set is dense enough
/// that consecutive layers are likely to overlap. Unsloth 6/256 stays off.
pub(crate) fn want_hint_prefetch(cfg: &ModelConfig) -> bool {
    cfg.n_experts > 0 && cfg.n_active.saturating_mul(8) >= cfg.n_experts
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
                q8: false,
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

fn dense_f32_hc_names(cfg: &ModelConfig) -> Vec<String> {
    let mut names = vec!["output_hc_fn.weight".into()];
    for i in 0..cfg.n_layers {
        let blk = cfg.gguf_blk(i);
        names.push(format!("blk.{blk}.hc_attn_fn.weight"));
        names.push(format!("blk.{blk}.hc_ffn_fn.weight"));
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

/// VRAM an expert cache may take on `device`: what the driver reports, capped
/// by what the OS says no process holds. On Windows `hipMemGetInfo` ignores
/// the desktop's share, and a cache sized from it spills into system RAM.
fn device_free_bytes(device: i32) -> u64 {
    let (free, _) = device_memory_info(device).unwrap_or((0, 0));
    allpaca_hip::device_local_headroom(device).map_or(free, |h| h.min(free))
}

/// RAM left to the rest of the system when the host expert tier is sized.
const RAM_TIER_RESERVE: u64 = 8 * 1024 * 1024 * 1024;

/// Bytes the host RAM expert tier may take, and where the figure comes
/// from. In order: `ALLPACA_DSV4_RAM_MIB` (0 = no tier, experts are read
/// through the mapped file), then `weights.host_bytes` when it is set,
/// then the RAM free at load minus [`RAM_TIER_RESERVE`]. Never more than
/// all the experts, `all`.
///
/// The automatic figure is a snapshot: a server started while another large
/// process is still exiting sees less. Set `weights.host_bytes` for a tier
/// that is the same size on every start.
fn ram_tier_budget(
    env_mib: Option<u64>,
    host_bytes: u64,
    ram_free: Option<u64>,
    all: u64,
) -> Option<(u64, &'static str)> {
    let (bytes, source) = match env_mib {
        Some(mib) => (mib << 20, "ALLPACA_DSV4_RAM_MIB"),
        None if host_bytes > 0 => (host_bytes, "weights.host_bytes"),
        None => (ram_free?.saturating_sub(RAM_TIER_RESERVE), "free RAM at load"),
    };
    Some((bytes.min(all), source))
}

/// The host RAM tier for experts that are not in VRAM.
fn open_expert_ram(
    file: &GgufFile,
    cfg: &ModelConfig,
    slot: usize,
    ram_free: Option<u64>,
    host_bytes: u64,
) -> Option<Arc<std::sync::Mutex<crate::dsv4_ram::ExpertRam>>> {
    let paths = file.shard_paths()?;
    let all = (cfg.n_layers * cfg.n_experts) as u64 * slot as u64;
    let env_mib = std::env::var("ALLPACA_DSV4_RAM_MIB").ok().and_then(|v| v.parse::<u64>().ok());
    let (budget, source) = ram_tier_budget(env_mib, host_bytes, ram_free, all)?;
    let max_slice = mxfp4_matrix_bytes(cfg.ff_dim.max(1), cfg.dim).max(mxfp4_matrix_bytes(cfg.dim, cfg.ff_dim.max(1)));
    match crate::dsv4_ram::ExpertRam::new(paths, slot, max_slice, budget) {
        Ok(Some(ram)) => {
            tracing::info!(
                slots = ram.slots(),
                bytes = ram.slots() as u64 * slot as u64,
                budget_from = source,
                "DeepSeek-V4 host RAM expert tier"
            );
            Some(Arc::new(std::sync::Mutex::new(ram)))
        }
        Ok(None) => None,
        Err(e) => {
            tracing::warn!(error = %e, "DeepSeek-V4 host RAM expert tier unavailable; reading through the mapped file");
            None
        }
    }
}

/// Where the three packed slices of one expert are in the model's files.
pub(crate) fn expert_mxfp4_spans(
    file: &GgufFile,
    layer: usize,
    expert: usize,
    cfg: &ModelConfig,
) -> Option<Vec<crate::dsv4_ram::FileSpan>> {
    let n_exp = cfg.n_experts.max(1);
    let mut out = Vec::with_capacity(3);
    for part in ["gate", "up", "down"] {
        let info = file.tensor(&format!("blk.{layer}.ffn_{part}_exps.weight"))?;
        if info.ggml_type != GgmlType::MXFP4 {
            return None;
        }
        let per = info.n_elements() as usize / n_exp;
        let stride = per.div_ceil(MXFP4_BLOCK_ELEMS).saturating_mul(MXFP4_BLOCK_BYTES);
        let (shard, start) = file.tensor_file_offset(info)?;
        out.push(crate::dsv4_ram::FileSpan {
            shard,
            offset: start + (expert * stride) as u64,
            len: stride,
        });
    }
    Some(out)
}

/// The packed CPU expert kernel takes MXFP4 experts in whole 32-value blocks.
fn cpu_expert_kernel_fits(file: &GgufFile, cfg: &ModelConfig) -> bool {
    let blk = cfg.gguf_blk(0);
    cfg.dim % MXFP4_BLOCK_ELEMS == 0
        && cfg.ff_dim > 0
        && cfg.ff_dim % MXFP4_BLOCK_ELEMS == 0
        && ["gate", "up", "down"].iter().all(|part| {
            file.tensor(&format!("blk.{blk}.ffn_{part}_exps.weight"))
                .is_some_and(|t| t.ggml_type == GgmlType::MXFP4)
        })
}

fn expert_lru_slots(free: u64, slot: usize) -> usize {
    if slot == 0 {
        return 0;
    }
    let budget = free.saturating_sub(HBM_RESERVE).min(EXPERT_LRU_MAX_BYTES);
    (budget as usize) / slot
}

fn try_alloc_expert_lru(device: i32, mut n: usize, slot: usize) -> Option<(DeviceWeights, usize)> {
    if slot == 0 {
        return None;
    }
    while n > 0 {
        let bytes = n.saturating_mul(slot);
        if bytes / slot != n {
            n /= 2;
            continue;
        }
        if let Ok(blob) = DeviceWeights::alloc(device, bytes) {
            return Some((blob, n));
        }
        n /= 2;
    }
    None
}

fn empty_expert_slots(n: usize) -> Vec<ExpertSlot> {
    (0..n)
        .map(|_| ExpertSlot {
            key: None,
            used: 0,
            pinned: false,
        })
        .collect()
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
    pin: Option<&PinPair>,
) -> Result<bool, AllpacaError> {
    if parts.is_empty() {
        return Ok(false);
    }
    if let (Some(h2d), Some(pin)) = (stream, pin) {
        copy_expert_parts_pinned(blob, parts, h2d, pin)
    } else if let Some(h2d) = stream {
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

/// Copy `parts` grouped by expert. When `ready` is set, record a ready-ring event
/// after each expert (GPU wait, not host stream sync) so miss GEMM overlaps later copies.
fn copy_expert_groups(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    groups: &[(usize, usize)],
    stream: Option<&H2dStream>,
    pin: Option<&PinPair>,
    ready: Option<&Sender<(usize, usize)>>,
) -> Result<bool, AllpacaError> {
    if ready.is_none() || groups.is_empty() {
        return copy_expert_parts(blob, parts, stream, pin);
    }
    if let (Some(h2d), Some(pin)) = (stream, pin) {
        return copy_expert_groups_pinned(blob, parts, groups, h2d, pin, ready);
    }
    let mut off = 0usize;
    let mut seq = 0usize;
    let mut inflight = false;
    for &(expert, n) in groups {
        let end = off.saturating_add(n);
        if end > parts.len() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "expert H2D group overrun".into(),
            )));
        }
        let async_copy = copy_expert_parts(blob, &parts[off..end], stream, pin)?;
        signal_miss_ready(stream, ready, expert, async_copy, &mut seq)?;
        inflight |= async_copy;
        off = end;
    }
    if off != parts.len() {
        let async_copy = copy_expert_parts(blob, &parts[off..], stream, pin)?;
        inflight |= async_copy;
    }
    Ok(inflight)
}

fn signal_miss_ready(
    stream: Option<&H2dStream>,
    ready: Option<&Sender<(usize, usize)>>,
    expert: usize,
    async_copy: bool,
    seq: &mut usize,
) -> Result<(), AllpacaError> {
    let Some(tx) = ready else {
        return Ok(());
    };
    if !async_copy {
        let _ = tx.send((expert, usize::MAX));
        return Ok(());
    }
    let h2d = stream.ok_or_else(|| {
        AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "async expert H2D without stream".into(),
        ))
    })?;
    let idx = *seq % H2D_READY_RING;
    if *seq >= H2D_READY_RING {
        h2d.synchronize_ready(idx)?;
    }
    h2d.record_ready(idx)?;
    let _ = tx.send((expert, idx));
    *seq = seq.saturating_add(1);
    Ok(())
}

fn copy_expert_groups_pinned(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    groups: &[(usize, usize)],
    h2d: &H2dStream,
    pin: &PinPair,
    ready: Option<&Sender<(usize, usize)>>,
) -> Result<bool, AllpacaError> {
    if !pin.window && (pin.bytes == 0 || pin.a.is_null() || pin.b.is_null()) {
        let mut off = 0usize;
        let mut seq = 0usize;
        let mut inflight = false;
        for &(expert, n) in groups {
            let end = off.saturating_add(n);
            if end > parts.len() {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "expert H2D group overrun".into(),
                )));
            }
            let async_copy = copy_expert_parts(blob, &parts[off..end], Some(h2d), None)?;
            signal_miss_ready(Some(h2d), ready, expert, async_copy, &mut seq)?;
            inflight |= async_copy;
            off = end;
        }
        return Ok(inflight);
    }
    for (offset, src) in parts {
        if offset.saturating_add(src.len()) > blob.bytes() as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "weight batch H2D overflow: off={offset} n={} cap={}",
                    src.len(),
                    blob.bytes()
                ),
            )));
        }
    }
    let mut seq = 0usize;
    let mut off = 0usize;
    let mut any = false;
    for &(expert, n) in groups {
        let end = off.saturating_add(n);
        if end > parts.len() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "expert H2D group overrun".into(),
            )));
        }
        let queued = copy_pinned_chunks(blob, &parts[off..end], h2d, pin)?;
        any |= queued;
        signal_miss_ready(Some(h2d), ready, expert, queued, &mut seq)?;
        off = end;
    }
    if off != parts.len() {
        any |= copy_pinned_chunks(blob, &parts[off..], h2d, pin)?;
    }
    Ok(any)
}

fn copy_expert_parts_pinned(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    h2d: &H2dStream,
    pin: &PinPair,
) -> Result<bool, AllpacaError> {
    if !pin.window && (pin.bytes == 0 || pin.a.is_null() || pin.b.is_null()) {
        // SAFETY: same mmap lifetime as the pageable async path.
        unsafe { blob.copy_h2d_batch_on(parts, h2d.raw())? };
        return Ok(true);
    }
    for (offset, src) in parts {
        if offset.saturating_add(src.len()) > blob.bytes() as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "weight batch H2D overflow: off={offset} n={} cap={}",
                    src.len(),
                    blob.bytes()
                ),
            )));
        }
    }
    copy_pinned_chunks(blob, parts, h2d, pin)
}

fn wait_pin_slots(h2d: &H2dStream, pin: &PinPair) {
    for i in 0..2 {
        if pin.live[i].swap(false, Ordering::AcqRel) {
            let _ = h2d.synchronize_slot(i);
        }
        if pin.window {
            release_reg_slot(pin, i);
        }
    }
}

/// True when the probe window faulted at DRAM speed. The bounce thread can
/// memcpy the rest without a second 22 GiB walk fighting it for bandwidth.
fn prefault_probe_is_warm(probed_bytes: usize, elapsed_ms: u128) -> bool {
    probed_bytes >= PREFAULT_WARM_PROBE_BYTES && elapsed_ms < PREFAULT_WARM_MS
}

/// Host mmap touch that stops once DMA has landed (`stop`). Used so bounce
/// memcpy does not serialise NVMe page-in on the H2D thread. After one expert
/// (~12 MiB) at DRAM speed, skip the remaining walk (warm page cache).
fn prefault_parts_until(parts: &[(usize, &[u8])], stop: &AtomicBool) {
    let t0 = Instant::now();
    let mut probed = 0usize;
    let mut i = 0usize;
    while i < parts.len() {
        if stop.load(Ordering::Acquire) {
            return;
        }
        prefault_mmap(parts[i].1);
        probed = probed.saturating_add(parts[i].1.len());
        i += 1;
        if probed >= PREFAULT_WARM_PROBE_BYTES {
            if prefault_probe_is_warm(probed, t0.elapsed().as_millis()) {
                return;
            }
            break;
        }
    }
    while i < parts.len() {
        if stop.load(Ordering::Acquire) {
            return;
        }
        prefault_mmap(parts[i].1);
        i += 1;
    }
}

/// Bytes of one bounce slot gathered from GGUF slices. Two threads copy when
/// the slot is large enough that a spawn is cheaper than one serial memcpy.
const PIN_GATHER_PARALLEL_BYTES: usize = 32 * 1024;

struct PinSpan {
    /// Host address stored as `usize` so the span is `Send` (Rust 2024 pointers are not).
    src: usize,
    pin_off: usize,
    len: usize,
}

fn pin_gather_spans(
    parts: &[(usize, &[u8])],
    mut pi: usize,
    mut poff: usize,
    n: usize,
) -> Result<(Vec<PinSpan>, usize, usize), AllpacaError> {
    let mut spans = Vec::new();
    let mut packed = 0usize;
    while packed < n {
        if pi >= parts.len() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "packed H2D host run underrun".into(),
            )));
        }
        let sl = parts[pi].1;
        if poff > sl.len() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "packed H2D host cursor past slice".into(),
            )));
        }
        let take = (sl.len() - poff).min(n - packed);
        if take == 0 {
            pi += 1;
            poff = 0;
            continue;
        }
        // SAFETY: `poff < sl.len()` because `take > 0`. The address is only
        // read for `take` bytes, and `poff + take <= sl.len()`.
        let src = unsafe { sl.as_ptr().add(poff) as usize };
        spans.push(PinSpan {
            src,
            pin_off: packed,
            len: take,
        });
        packed += take;
        poff += take;
        if poff == sl.len() {
            pi += 1;
            poff = 0;
        }
    }
    Ok((spans, pi, poff))
}

fn copy_spans_to_pin(dst: *mut u8, spans: &[PinSpan]) {
    let total: usize = spans.iter().map(|s| s.len).sum();
    if total == 0 {
        return;
    }
    let dst = dst as usize;
    if total < PIN_GATHER_PARALLEL_BYTES {
        copy_span_range(dst, spans, 0, total);
        return;
    }
    let mid = total / 2;
    std::thread::scope(|scope| {
        scope.spawn(|| copy_span_range(dst, spans, 0, mid));
        copy_span_range(dst, spans, mid, total);
    });
}

fn copy_span_range(dst: usize, spans: &[PinSpan], from: usize, to: usize) {
    for s in spans {
        let s_end = s.pin_off + s.len;
        if s_end <= from {
            continue;
        }
        if s.pin_off >= to {
            break;
        }
        let lo = from.max(s.pin_off);
        let hi = to.min(s_end);
        let off = lo - s.pin_off;
        let len = hi - lo;
        // SAFETY: `s.src` addresses `s.len` bytes of the GGUF slice that built
        // this span. `off + len <= s.len`. `dst + lo` lies in the idle bounce
        // slot. Callers split `[from, to)` so concurrent copies do not alias.
        unsafe {
            std::ptr::copy_nonoverlapping((s.src + off) as *const u8, (dst + lo) as *mut u8, len);
        }
    }
}

/// Length of the contiguous device-offset run starting at `start`.
fn packed_run_parts(parts: &[(usize, &[u8])], start: usize) -> usize {
    if start >= parts.len() {
        return 0;
    }
    let mut n = 1usize;
    let mut end = parts[start].0.saturating_add(parts[start].1.len());
    while start + n < parts.len() {
        let (off, sl) = parts[start + n];
        if off != end {
            break;
        }
        end = end.saturating_add(sl.len());
        n += 1;
    }
    n
}

fn copy_pinned_run(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    dst0: usize,
    total: usize,
    h2d: &H2dStream,
    pin: &PinPair,
) -> Result<bool, AllpacaError> {
    if total == 0 {
        return Ok(false);
    }
    if dst0.saturating_add(total) > blob.bytes() as usize {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            format!(
                "packed H2D overflow: off={dst0} n={total} cap={}",
                blob.bytes()
            ),
        )));
    }
    let mut done = 0usize;
    let mut pi = 0usize;
    let mut poff = 0usize;
    while done < total {
        let which = pin.which.load(Ordering::Relaxed) & 1;
        if pin.live[which].load(Ordering::Acquire) {
            h2d.synchronize_slot(which)?;
            pin.live[which].store(false, Ordering::Release);
        }
        let n = (total - done).min(pin.bytes);
        let src_ptr = if which == 0 { pin.a } else { pin.b };
        let (spans, npi, npoff) = pin_gather_spans(parts, pi, poff, n)?;
        copy_spans_to_pin(src_ptr, &spans);
        pi = npi;
        poff = npoff;
        let dst_off = dst0 + done;
        // SAFETY: `src_ptr` is pin slot `which`, not in flight (`synchronize_slot`
        // above). `n <= pin.bytes`. Device range `[dst_off, dst_off+n)` was
        // bounds-checked. DMA stays queued until this slot is waited.
        unsafe {
            let chunk = std::slice::from_raw_parts(src_ptr, n);
            blob.copy_h2d_batch_on(&[(dst_off, chunk)], h2d.raw())?;
        }
        h2d.record_slot(which)?;
        pin.live[which].store(true, Ordering::Release);
        pin.which.store(which ^ 1, Ordering::Relaxed);
        done += n;
    }
    Ok(true)
}

fn copy_pinned_chunks(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    h2d: &H2dStream,
    pin: &PinPair,
) -> Result<bool, AllpacaError> {
    if pin.window {
        return copy_registered_parts(blob, parts, h2d, pin);
    }
    let mut any = false;
    let mut i = 0usize;
    while i < parts.len() {
        let nrun = packed_run_parts(parts, i);
        let run = &parts[i..i + nrun];
        let total: usize = run.iter().map(|(_, sl)| sl.len()).sum();
        any |= copy_pinned_run(blob, run, run[0].0, total, h2d, pin)?;
        i += nrun;
    }
    Ok(any)
}

fn copy_registered_parts(
    blob: &DeviceWeights,
    parts: &[(usize, &[u8])],
    h2d: &H2dStream,
    pin: &PinPair,
) -> Result<bool, AllpacaError> {
    if pin.bytes == 0 {
        // SAFETY: GGUF mmap outlives join_expert_h2d.
        unsafe { blob.copy_h2d_batch_on(parts, h2d.raw())? };
        return Ok(true);
    }
    let mut any = false;
    for (dst0, sl) in parts {
        if sl.is_empty() {
            continue;
        }
        if dst0.saturating_add(sl.len()) > blob.bytes() as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "window H2D overflow: off={dst0} n={} cap={}",
                    sl.len(),
                    blob.bytes()
                ),
            )));
        }
        for win in mmap_windows(sl, pin.bytes) {
            let which = pin.which.load(Ordering::Relaxed) & 1;
            if pin.live[which].load(Ordering::Acquire) {
                h2d.synchronize_slot(which)?;
                pin.live[which].store(false, Ordering::Release);
            }
            release_reg_slot(pin, which);
            let reg = win.reg_ptr as *mut u8;
            if host_register(reg, win.reg_len, 0).is_ok() {
                pin.win_ptr[which].store(win.reg_ptr, Ordering::Release);
                pin.win_len[which].store(win.reg_len, Ordering::Release);
            }
            let dst_off = dst0.saturating_add(win.data_off);
            // SAFETY: `win.data` is inside the GGUF mmap (registered when
            // `host_register` succeeded). Device range was bounds-checked.
            // DMA stays queued until this slot is waited.
            unsafe {
                blob.copy_h2d_batch_on(&[(dst_off, win.data)], h2d.raw())?;
            }
            h2d.record_slot(which)?;
            pin.live[which].store(true, Ordering::Release);
            pin.which.store(which ^ 1, Ordering::Relaxed);
            any = true;
        }
    }
    Ok(any)
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
    let slot_bytes = ws.swiglu_slot_bytes();
    let width = ws.swiglu_pipeline_slots();
    let moe = ws.moe_active();
    let fits = jobs
        .iter()
        .all(|j| HipWorkspace::swiglu_workspace_bytes(j.m, dim, ff) <= slot_bytes);
    let pipeline =
        width >= 1 && fits && !jobs.is_empty() && (moe || (width >= 2 && jobs.len() >= 2));
    if pipeline {
        return run_expert_swiglu_pipelined(ws, jobs, dim, ff, gate_bytes, width, moe, metrics);
    }
    for job in jobs {
        let ptr = job.ptr as *const u8;
        // SAFETY: `ptr` is a packed gate|up|down expert slot on `ws`.
        let up_ptr = unsafe { ptr.add(gate_bytes) };
        let down_ptr = unsafe { ptr.add(gate_bytes.saturating_add(gate_bytes)) };
        let t = Instant::now();
        if moe {
            let (idx, w) = expert_row_meta(&job.group);
            if ws
                .mxfp4_swiglu_accumulate_gemm(
                    &idx,
                    &w,
                    ptr,
                    up_ptr,
                    down_ptr,
                    job.m as i32,
                    dim as i32,
                    ff as i32,
                )
                .is_ok()
            {
                add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
            } else {
                failed.push(job);
            }
            continue;
        }
        let mut y = vec![0.0f32; job.x.len()];
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

struct SwigluInflight {
    job: ExpertSwigluJob,
    y: Vec<f32>,
}

fn expert_row_meta(group: &[(usize, f32)]) -> (Vec<i32>, Vec<f32>) {
    let mut idx = Vec::with_capacity(group.len());
    let mut w = Vec::with_capacity(group.len());
    for &(t, wt) in group {
        idx.push(t as i32);
        w.push(wt);
    }
    (idx, w)
}

fn collect_swiglu_slot(
    ws: &HipWorkspace,
    inflight: &mut [Option<SwigluInflight>],
    slot: usize,
    dim: usize,
    ff: usize,
    device_acc: bool,
    out: &mut Vec<(Vec<f32>, Vec<(usize, f32)>)>,
    failed: &mut Vec<ExpertSwigluJob>,
    acc_ok: &mut usize,
) {
    let Some(mut live) = inflight[slot].take() else {
        return;
    };
    if device_acc {
        if ws
            .mxfp4_swiglu_accumulate_slot(slot, live.job.m as i32, dim as i32, ff as i32)
            .is_ok()
        {
            *acc_ok = acc_ok.saturating_add(1);
        } else {
            failed.push(live.job);
        }
        return;
    }
    if ws
        .mxfp4_swiglu_collect(slot, &mut live.y, live.job.m as i32, dim as i32, ff as i32)
        .is_ok()
    {
        out.push((live.y, live.job.group));
    } else {
        failed.push(live.job);
    }
}

fn run_expert_swiglu_pipelined(
    ws: &HipWorkspace,
    jobs: Vec<ExpertSwigluJob>,
    dim: usize,
    ff: usize,
    gate_bytes: usize,
    width: usize,
    device_acc: bool,
    metrics: Option<&Metrics>,
) -> (Vec<(Vec<f32>, Vec<(usize, f32)>)>, Vec<ExpertSwigluJob>) {
    let width = width.max(1);
    let mut out = Vec::with_capacity(jobs.len());
    let mut failed = Vec::new();
    let mut inflight: Vec<Option<SwigluInflight>> = (0..width).map(|_| None).collect();
    let mut which = 0usize;
    let mut acc_ok = 0usize;
    let t = Instant::now();
    for job in jobs {
        collect_swiglu_slot(
            ws,
            &mut inflight,
            which,
            dim,
            ff,
            device_acc,
            &mut out,
            &mut failed,
            &mut acc_ok,
        );
        let ptr = job.ptr as *const u8;
        // SAFETY: `ptr` is a packed gate|up|down expert slot on `ws`.
        let up_ptr = unsafe { ptr.add(gate_bytes) };
        let down_ptr = unsafe { ptr.add(gate_bytes.saturating_add(gate_bytes)) };
        let enqueued = if device_acc {
            let (idx, w) = expert_row_meta(&job.group);
            ws.mxfp4_swiglu_enqueue_from_rows(
                which,
                &idx,
                &w,
                ptr,
                up_ptr,
                down_ptr,
                job.m as i32,
                dim as i32,
                ff as i32,
            )
            .is_ok()
        } else {
            ws.mxfp4_swiglu_enqueue(
                which,
                &job.x,
                ptr,
                up_ptr,
                down_ptr,
                job.m as i32,
                dim as i32,
                ff as i32,
            )
            .is_ok()
        };
        if enqueued {
            let y = if device_acc {
                Vec::new()
            } else {
                vec![0.0f32; job.x.len()]
            };
            inflight[which] = Some(SwigluInflight { job, y });
            which += 1;
            if which == width {
                which = 0;
            }
        } else {
            failed.push(job);
        }
    }
    for slot in 0..width {
        collect_swiglu_slot(
            ws,
            &mut inflight,
            slot,
            dim,
            ff,
            device_acc,
            &mut out,
            &mut failed,
            &mut acc_ok,
        );
    }
    if device_acc {
        let _ = ws.moe_join_compute();
    }
    if !out.is_empty() || acc_ok > 0 {
        add_ns(metrics, |m| &m.dsv4_gemv_ns, t);
    }
    (out, failed)
}

pub(crate) fn expert_mxfp4_parts<'a>(
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
    host_bytes: u64,
) -> Result<(), AllpacaError> {
    let dsv4 = model.dsv4.as_ref().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad("missing DeepSeek-V4 weights".into()))
    })?;
    let file = dsv4.gguf.clone().ok_or_else(|| {
        AllpacaError::Fatal(FatalError::ModelLoad(
            "DeepSeek-V4 GGUF handle missing for HIP attach".into(),
        ))
    })?;
    let mut gpu = Dsv4Gpu::build(file.as_ref(), &model.config, hip.clone(), draft.cloned(), host_bytes)?;
    gpu.start_read_ahead(&file, &model.config);
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
    let mut guard = dsv4
        .gpu
        .lock()
        .map_err(|_| AllpacaError::Fatal(FatalError::ModelLoad("dsv4 gpu lock poisoned".into())))?;
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
    lru_victim_keep(slots, None).unwrap_or(0)
}

fn lru_victim_keep(slots: &[ExpertSlot], keep_layer: Option<usize>) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, s)| match (keep_layer, s.key) {
            (Some(keep), Some((layer, _))) if layer == keep => false,
            _ => true,
        })
        .min_by_key(|(_, s)| {
            let empty = u8::from(s.key.is_some());
            let pin = u8::from(s.pinned);
            (empty, pin, s.used)
        })
        .map(|(i, _)| i)
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
    use super::{
        expert_lru_slots, want_full_layer_prefetch, want_hint_prefetch, EXPERT_LRU_MAX_BYTES,
        HBM_RESERVE,
    };
    use crate::model::ModelConfig;

    #[test]
    fn ram_tier_budget_prefers_the_env_then_the_config_then_free_ram() {
        const GIB: u64 = 1 << 30;
        let all = 140 * GIB;
        // Automatic: what is free minus the reserve; unknown free RAM = no tier.
        assert_eq!(super::ram_tier_budget(None, 0, Some(60 * GIB), all), Some((52 * GIB, "free RAM at load")));
        assert_eq!(super::ram_tier_budget(None, 0, Some(4 * GIB), all), Some((0, "free RAM at load")));
        assert_eq!(super::ram_tier_budget(None, 0, None, all), None);
        // `weights.host_bytes` does not depend on what happens to be free.
        assert_eq!(
            super::ram_tier_budget(None, 40 * GIB, Some(10 * GIB), all),
            Some((40 * GIB, "weights.host_bytes"))
        );
        assert_eq!(super::ram_tier_budget(None, 40 * GIB, None, all), Some((40 * GIB, "weights.host_bytes")));
        // The environment overrides both; 0 turns the tier off.
        assert_eq!(
            super::ram_tier_budget(Some(1024), 40 * GIB, Some(60 * GIB), all),
            Some((GIB, "ALLPACA_DSV4_RAM_MIB"))
        );
        assert_eq!(
            super::ram_tier_budget(Some(0), 40 * GIB, Some(60 * GIB), all),
            Some((0, "ALLPACA_DSV4_RAM_MIB"))
        );
        // Never more than the model's experts.
        assert_eq!(super::ram_tier_budget(None, 500 * GIB, None, all), Some((all, "weights.host_bytes")));
    }

    #[test]
    fn unsloth_short_prefill_skips_full_layer() {
        let cfg = ModelConfig {
            n_experts: 256,
            n_active: 6,
            ..ModelConfig::tiny()
        };
        assert!(!want_full_layer_prefetch(true, &cfg));
        assert!(!want_full_layer_prefetch(false, &cfg));
        assert!(!want_hint_prefetch(&cfg));
    }

    #[test]
    fn tiny_moe_prefill_uses_full_layer() {
        let cfg = ModelConfig::tiny_moe();
        assert!(want_full_layer_prefetch(true, &cfg));
        assert!(!want_full_layer_prefetch(false, &cfg));
        assert!(want_hint_prefetch(&cfg));
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
    fn pin_gather_spans_tile_one_slot_across_slices() {
        let a = vec![1u8; 40_000];
        let b = vec![2u8; 40_000];
        let parts = [(0usize, a.as_slice()), (40_000usize, b.as_slice())];
        let (spans, pi, poff) = super::pin_gather_spans(&parts, 0, 3, 70_000).expect("spans");
        assert_eq!(pi, 1);
        assert_eq!(poff, 30_003);
        let mut covered = 0usize;
        for w in spans.windows(2) {
            assert_eq!(w[0].pin_off + w[0].len, w[1].pin_off);
            covered += w[0].len;
        }
        covered += spans.last().map(|s| s.len).unwrap_or(0);
        assert_eq!(covered, 70_000);
        assert_eq!(spans[0].pin_off, 0);
        let mut buf = vec![0u8; 70_000];
        super::copy_spans_to_pin(buf.as_mut_ptr(), &spans);
        assert!(buf[..39_997].iter().all(|&x| x == 1));
        assert!(buf[39_997..].iter().all(|&x| x == 2));
    }

    #[test]
    fn packed_run_parts_joins_contiguous_device_offsets() {
        let a = [1u8; 4];
        let b = [2u8; 6];
        let c = [3u8; 2];
        let parts = [
            (0usize, a.as_slice()),
            (4usize, b.as_slice()),
            (20usize, c.as_slice()),
        ];
        assert_eq!(super::packed_run_parts(&parts, 0), 2);
        assert_eq!(super::packed_run_parts(&parts, 2), 1);
        assert_eq!(super::packed_run_parts(&parts, 3), 0);
    }

    #[test]
    fn prefault_probe_is_warm_only_after_full_window_at_dram_speed() {
        assert!(!super::prefault_probe_is_warm(4 * 1024 * 1024, 1));
        assert!(super::prefault_probe_is_warm(
            super::PREFAULT_WARM_PROBE_BYTES,
            1
        ));
        assert!(super::prefault_probe_is_warm(
            super::PREFAULT_WARM_PROBE_BYTES,
            super::PREFAULT_WARM_MS - 1
        ));
        assert!(!super::prefault_probe_is_warm(
            super::PREFAULT_WARM_PROBE_BYTES,
            super::PREFAULT_WARM_MS
        ));
        assert!(!super::prefault_probe_is_warm(
            super::PREFAULT_WARM_PROBE_BYTES,
            40
        ));
    }

    #[test]
    fn mmap_windows_cover_unaligned_slice_without_overlap() {
        let page = super::HOST_PAGE;
        let raw = vec![0u8; page * 3 + 100];
        let sl = &raw[50..page * 2 + 80];
        let wins = super::mmap_windows(sl, page);
        assert!(wins.len() >= 2);
        let mut covered = 0usize;
        let mut prev_end = 0usize;
        for (i, w) in wins.iter().enumerate() {
            assert_eq!(w.reg_ptr % page, 0, "window {i} not page-aligned");
            assert_eq!(w.reg_len % page, 0);
            assert!(!w.data.is_empty());
            assert!(w.reg_ptr >= prev_end || i == 0);
            prev_end = w.reg_ptr.saturating_add(w.reg_len);
            covered += w.data.len();
        }
        assert_eq!(covered, sl.len());
        assert_eq!(wins[0].data_off, 0);
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

    #[test]
    fn lru_victim_keep_skips_current_layer() {
        let slots = [
            super::ExpertSlot {
                key: Some((3, 1)),
                used: 9,
                pinned: true,
            },
            super::ExpertSlot {
                key: Some((3, 2)),
                used: 8,
                pinned: true,
            },
            super::ExpertSlot {
                key: Some((1, 0)),
                used: 1,
                pinned: false,
            },
            super::ExpertSlot {
                key: None,
                used: 0,
                pinned: false,
            },
        ];
        assert_eq!(super::lru_victim_keep(&slots, Some(3)), Some(3));
        let full_keep = [
            super::ExpertSlot {
                key: Some((4, 0)),
                used: 1,
                pinned: true,
            },
            super::ExpertSlot {
                key: Some((4, 1)),
                used: 2,
                pinned: true,
            },
        ];
        assert_eq!(super::lru_victim_keep(&full_keep, Some(4)), None);
        assert_eq!(super::lru_victim_keep(&full_keep, Some(3)), Some(0));
    }

    #[test]
    fn expert_lru_slots_uses_hbm_reserve_without_512_cap() {
        let slot = 12 * 1024 * 1024;
        assert_eq!(expert_lru_slots(0, slot), 0);
        assert_eq!(expert_lru_slots(HBM_RESERVE, slot), 0);
        assert_eq!(expert_lru_slots(HBM_RESERVE + slot as u64 * 10, slot), 10);
        let uncapped = expert_lru_slots(30u64 << 30, slot);
        assert!(uncapped > 512);
        assert!(uncapped <= (EXPERT_LRU_MAX_BYTES as usize) / slot);
        assert_eq!(expert_lru_slots(8 << 30, 0), 0);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hc_mix_logits_matches_cpu_rmsnorm_gemm() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip);
        let n_tok = 3usize;
        let hc_dim = 16usize;
        let mix_dim = 8usize;
        let eps = 1e-5f32;
        let packed: Vec<f32> = (0..n_tok * hc_dim)
            .map(|i| 0.05 * (i as f32 % 11.0) - 0.2)
            .collect();
        let fn_w: Vec<f32> = (0..mix_dim * hc_dim)
            .map(|i| 0.01 * (i as f32 % 7.0) - 0.03)
            .collect();
        let mut gpu_mixes = vec![0.0f32; n_tok * mix_dim];
        gpu.hc_mix_logits(
            &packed,
            &fn_w,
            None,
            n_tok,
            hc_dim,
            mix_dim,
            eps,
            &mut gpu_mixes,
        )
        .expect("gpu hc mix");
        let mut normed = vec![0.0f32; n_tok * hc_dim];
        let ones = vec![1.0f32; hc_dim];
        for t in 0..n_tok {
            let a = t * hc_dim;
            allpaca_cpu::rmsnorm(
                &packed[a..a + hc_dim],
                &ones,
                eps,
                &mut normed[a..a + hc_dim],
            )
            .expect("cpu rmsnorm");
        }
        let mut cpu_mixes = vec![0.0f32; n_tok * mix_dim];
        allpaca_cpu::matmul_nt(&normed, &fn_w, &mut cpu_mixes, n_tok, mix_dim, hc_dim)
            .expect("cpu gemm");
        for i in 0..n_tok * mix_dim {
            let diff = (gpu_mixes[i] - cpu_mixes[i]).abs();
            assert!(
                diff < 2e-4,
                "mix[{i}] gpu={} cpu={} diff={diff}",
                gpu_mixes[i],
                cpu_mixes[i]
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hc_mix_pre_post_match_cpu_and_skip_stale_host() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip);
        let n_tok = 3usize;
        let hc = 4usize;
        let dim = 8usize;
        let hc_dim = hc * dim;
        let mix_dim = (2 + hc) * hc;
        let eps = 1e-5f32;
        let packed: Vec<f32> = (0..n_tok * hc_dim)
            .map(|i| 0.05 * (i as f32 % 11.0) - 0.2)
            .collect();
        let fn_w: Vec<f32> = (0..mix_dim * hc_dim)
            .map(|i| 0.01 * (i as f32 % 7.0) - 0.03)
            .collect();
        let mut mixes = vec![0.0f32; n_tok * mix_dim];
        gpu.hc_mix_logits(
            &packed, &fn_w, None, n_tok, hc_dim, mix_dim, eps, &mut mixes,
        )
        .expect("gpu hc mix");
        let pre: Vec<f32> = (0..n_tok * hc).map(|i| 0.1 + 0.02 * i as f32).collect();
        let mut gpu_curs = vec![0.0f32; n_tok * dim];
        gpu.hc_mix_pre(&pre, n_tok, hc, dim, &mut gpu_curs, None, eps, true)
            .expect("gpu mix_pre");
        let mut cpu_curs = vec![0.0f32; n_tok * dim];
        for t in 0..n_tok {
            let dst = &mut cpu_curs[t * dim..(t + 1) * dim];
            dst.fill(0.0);
            let tok = &packed[t * hc_dim..(t + 1) * hc_dim];
            for s in 0..hc {
                let w = pre[t * hc + s];
                for i in 0..dim {
                    dst[i] += tok[s * dim + i] * w;
                }
            }
        }
        for i in 0..n_tok * dim {
            let diff = (gpu_curs[i] - cpu_curs[i]).abs();
            assert!(
                diff < 2e-4,
                "curs[{i}] gpu={} cpu={} diff={diff}",
                gpu_curs[i],
                cpu_curs[i]
            );
        }
        let mut mixes2 = vec![0.0f32; n_tok * mix_dim];
        let garbage = vec![99.0f32; packed.len()];
        gpu.hc_mix_logits(
            &garbage,
            &fn_w,
            None,
            n_tok,
            hc_dim,
            mix_dim,
            eps,
            &mut mixes2,
        )
        .expect("skip-h2d mix");
        for i in 0..n_tok * mix_dim {
            let diff = (mixes[i] - mixes2[i]).abs();
            assert!(
                diff < 2e-5,
                "resident mix used stale host at {i}: {} vs {}",
                mixes[i],
                mixes2[i]
            );
        }
        let xs: Vec<f32> = (0..n_tok * dim).map(|i| 0.03 * i as f32).collect();
        let posts: Vec<f32> = (0..n_tok * hc).map(|i| 0.4 + 0.05 * i as f32).collect();
        let combs: Vec<f32> = (0..n_tok * hc * hc)
            .map(|i| 0.02 * ((i % 7) as f32) - 0.05)
            .collect();
        gpu.hc_post_packed(&garbage, &xs, &posts, &combs, n_tok, hc, dim, None)
            .expect("gpu post ignores stale host");
        let mut got = vec![0.0f32; packed.len()];
        gpu.download_hc_packed(&mut got).expect("download post");
        let mut expect = vec![0.0f32; packed.len()];
        for t in 0..n_tok {
            for dst in 0..hc {
                for i in 0..dim {
                    let mut acc = xs[t * dim + i] * posts[t * hc + dst];
                    for src in 0..hc {
                        let w = combs[t * hc * hc + dst + src * hc];
                        acc += packed[t * hc_dim + src * dim + i] * w;
                    }
                    expect[t * hc_dim + dst * dim + i] = acc;
                }
            }
        }
        for i in 0..packed.len() {
            let diff = (got[i] - expect[i]).abs();
            assert!(
                diff < 2e-4,
                "post[{i}] gpu={} cpu={} diff={diff}",
                got[i],
                expect[i]
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hc_mix_pre_skip_host_copy_still_materializes_on_ensure() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip.clone());
        let n_tok = 3usize;
        let hc = 4usize;
        let dim = 8usize;
        let hc_dim = hc * dim;
        let packed: Vec<f32> = (0..n_tok * hc_dim)
            .map(|i| 0.05 * (i as f32 % 11.0) - 0.2)
            .collect();
        let mix_dim = (2 + hc) * hc;
        let fn_w: Vec<f32> = (0..mix_dim * hc_dim)
            .map(|i| 0.01 * (i as f32 % 7.0) - 0.03)
            .collect();
        let mut mixes = vec![0.0f32; n_tok * mix_dim];
        gpu.hc_mix_logits(
            &packed, &fn_w, None, n_tok, hc_dim, mix_dim, 1e-5, &mut mixes,
        )
        .expect("install packed");
        let pre: Vec<f32> = (0..n_tok * hc).map(|i| 0.1 + 0.02 * i as f32).collect();
        let mut curs = vec![0.0f32; n_tok * dim];
        let d2h_before = hip.activation_d2h_bytes();
        gpu.hc_mix_pre(&pre, n_tok, hc, dim, &mut curs, None, 1e-5, false)
            .expect("mix_pre skip host");
        assert_eq!(
            hip.activation_d2h_bytes().saturating_sub(d2h_before),
            0,
            "mix_pre skip must not D2H curs"
        );
        assert!(curs.iter().all(|v| *v == 0.0), "host curs stays untouched");
        gpu.ensure_hc_curs_host(&mut curs).expect("materialize");
        let mut cpu = vec![0.0f32; n_tok * dim];
        for t in 0..n_tok {
            let dst = &mut cpu[t * dim..(t + 1) * dim];
            let tok = &packed[t * hc_dim..(t + 1) * hc_dim];
            for s in 0..hc {
                let w = pre[t * hc + s];
                for i in 0..dim {
                    dst[i] += tok[s * dim + i] * w;
                }
            }
        }
        for i in 0..cpu.len() {
            let diff = (curs[i] - cpu[i]).abs();
            assert!(
                diff < 2e-3,
                "curs[{i}] {} vs {} diff={diff}",
                curs[i],
                cpu[i]
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hc_mix_pre_device_curs_feeds_bf16_gemm_and_moe_without_activation_h2d() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip);
        let n_tok = 3usize;
        let hc = 4usize;
        let dim = 8usize;
        let hc_dim = hc * dim;
        let mix_dim = (2 + hc) * hc;
        let eps = 1e-5f32;
        let packed: Vec<f32> = (0..n_tok * hc_dim)
            .map(|i| 0.05 * (i as f32 % 11.0) - 0.2)
            .collect();
        let fn_w: Vec<f32> = (0..mix_dim * hc_dim)
            .map(|i| 0.01 * (i as f32 % 7.0) - 0.03)
            .collect();
        let mut mixes = vec![0.0f32; n_tok * mix_dim];
        gpu.hc_mix_logits(
            &packed, &fn_w, None, n_tok, hc_dim, mix_dim, eps, &mut mixes,
        )
        .expect("gpu hc mix");
        let pre: Vec<f32> = (0..n_tok * hc).map(|i| 0.1 + 0.02 * i as f32).collect();
        let mut gpu_curs = vec![0.0f32; n_tok * dim];
        gpu.hc_mix_pre(&pre, n_tok, hc, dim, &mut gpu_curs, None, eps, true)
            .expect("gpu mix_pre");
        let (curs_dev, curs_n) = gpu
            .hc_curs_device()
            .expect("mix_pre must leave device curs");
        assert_eq!(curs_n, n_tok * dim);
        assert!(
            gpu.device_a_if_curs(&gpu_curs, n_tok, dim).is_some(),
            "mix_pre host curs must feed device-A GEMM"
        );
        let other = gpu_curs.clone();
        assert!(
            gpu.device_a_if_curs(&other, n_tok, dim).is_none(),
            "same-length clone must not steal mix_pre device A"
        );
        let (hip, _) = gpu.hip_pair();
        let n = 16usize;
        let mut b_f = vec![0.0f32; n * dim];
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        let b_bf16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let db = allpaca_hip::device_malloc(b_bf16.len() * 2, hip.device).expect("device b");
        allpaca_hip::memcpy(
            db,
            b_bf16.as_ptr().cast(),
            b_bf16.len() * 2,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("h2d b");
        let mut from_dev = vec![0.0f32; n_tok * n];
        let mut from_host = vec![0.0f32; n_tok * n];
        let h2d_before = hip.activation_h2d_bytes();
        hip.gemm_nt_bf16_from_device_a(
            curs_dev,
            db.cast(),
            &mut from_dev,
            n_tok as i32,
            n as i32,
            dim as i32,
        )
        .expect("device-A gemm from mix_pre curs");
        assert_eq!(
            hip.activation_h2d_bytes(),
            h2d_before,
            "mix_pre curs GEMM must not H2D A"
        );
        hip.gemm_nt_bf16_dev_b(
            &gpu_curs,
            db.cast(),
            &mut from_host,
            n_tok as i32,
            n as i32,
            dim as i32,
        )
        .expect("host-A gemm");
        for i in 0..n_tok * n {
            let diff = (from_dev[i] - from_host[i]).abs();
            assert!(
                diff < 2e-3,
                "curs gemm[{i}] device={} host={} diff={diff}",
                from_dev[i],
                from_host[i]
            );
        }
        let h2d_moe = hip.activation_h2d_bytes();
        hip.moe_begin_from_device(curs_dev, n_tok, dim)
            .expect("moe from mix_pre curs");
        assert_eq!(
            hip.activation_h2d_bytes(),
            h2d_moe,
            "moe_begin_from_device must not H2D mix_pre curs"
        );
        let mut moe_x = vec![0.0f32; curs_n];
        hip.copy_d2h_f32(hip.moe_x_device().expect("moe_x"), &mut moe_x)
            .expect("d2h moe_x");
        for i in 0..curs_n {
            let diff = (moe_x[i] - gpu_curs[i]).abs();
            assert!(
                diff < 2e-4,
                "moe_x[{i}] {} vs curs {} diff={diff}",
                moe_x[i],
                gpu_curs[i]
            );
        }
        allpaca_hip::device_free(db, b_bf16.len() * 2);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mla_o_rows_keeps_grouped_oa_on_device_without_oa_h2d() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip.clone());
        let n_tok = 3usize;
        let groups = 2usize;
        let o_group_dim = 8usize;
        let per = 4usize;
        let dim = 16usize;
        let q_out = groups * o_group_dim;
        let oa_out = per * groups;
        let mut attn = vec![0.0f32; n_tok * q_out];
        let mut oa_f = vec![0.0f32; groups * per * o_group_dim];
        let mut ob_f = vec![0.0f32; dim * oa_out];
        for (i, v) in attn.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in oa_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in ob_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        let oa_bf16: Vec<u16> = oa_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let ob_bf16: Vec<u16> = ob_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let oa_deq: Vec<f32> = oa_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let ob_deq: Vec<f32> = ob_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let mut cpu_oa = vec![0.0f32; n_tok * oa_out];
        for g in 0..groups {
            let mut xs = vec![0.0f32; n_tok * o_group_dim];
            for t in 0..n_tok {
                let src = t * q_out + g * o_group_dim;
                let dst = t * o_group_dim;
                xs[dst..dst + o_group_dim].copy_from_slice(&attn[src..src + o_group_dim]);
            }
            let w = &oa_deq[g * per * o_group_dim..(g + 1) * per * o_group_dim];
            let mut ys = vec![0.0f32; n_tok * per];
            for t in 0..n_tok {
                for n in 0..per {
                    let mut acc = 0.0f32;
                    for k in 0..o_group_dim {
                        acc += xs[t * o_group_dim + k] * w[n * o_group_dim + k];
                    }
                    ys[t * per + n] = acc;
                }
            }
            for t in 0..n_tok {
                let src = t * per;
                let dst = t * oa_out + g * per;
                cpu_oa[dst..dst + per].copy_from_slice(&ys[src..src + per]);
            }
        }
        let mut cpu_out = vec![0.0f32; n_tok * dim];
        for t in 0..n_tok {
            for n in 0..dim {
                let mut acc = 0.0f32;
                for k in 0..oa_out {
                    acc += cpu_oa[t * oa_out + k] * ob_deq[n * oa_out + k];
                }
                cpu_out[t * dim + n] = acc;
            }
        }
        hip.bind().expect("bind");
        let doa_w: Vec<*mut u8> = (0..groups)
            .map(|_| allpaca_hip::device_malloc(per * o_group_dim * 2, hip.device).expect("doa_w"))
            .collect();
        let dob = allpaca_hip::device_malloc(ob_bf16.len() * 2, hip.device).expect("dob");
        allpaca_hip::memcpy(
            dob,
            ob_bf16.as_ptr().cast(),
            ob_bf16.len() * 2,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("h2d ob");
        for g in 0..groups {
            let w_off = g * per * o_group_dim;
            allpaca_hip::memcpy(
                doa_w[g],
                oa_bf16[w_off..].as_ptr().cast(),
                per * o_group_dim * 2,
                allpaca_hip::MEMCPY_H2D,
            )
            .expect("h2d oa_w");
        }
        let o_a: Vec<*const u16> = doa_w.iter().map(|p| (*p) as *const u16).collect();
        let h2d_before = hip.activation_h2d_bytes();
        let d2h_before = hip.activation_d2h_bytes();
        let got = gpu
            .mla_o_rows_with_ptrs(
                &attn,
                n_tok,
                q_out,
                o_group_dim,
                groups,
                per,
                dim,
                &o_a,
                dob.cast(),
                None,
            )
            .expect("mla o rows");
        let xs_bytes = (n_tok * q_out * 4) as u64;
        let out_bytes = (n_tok * dim * 4) as u64;
        assert_eq!(
            hip.activation_h2d_bytes().saturating_sub(h2d_before),
            xs_bytes,
            "MLA O must H2D grouped attn slices only, not oa"
        );
        assert_eq!(
            hip.activation_d2h_bytes().saturating_sub(d2h_before),
            out_bytes,
            "MLA O must D2H attn_out only, not grouped ys/oa"
        );
        for i in 0..cpu_out.len() {
            let diff = (got[i] - cpu_out[i]).abs();
            assert!(
                diff < 2e-3,
                "mla o[{i}] gpu={} cpu={} diff={diff}",
                got[i],
                cpu_out[i]
            );
        }
        for p in doa_w {
            allpaca_hip::device_free(p, per * o_group_dim * 2);
        }
        allpaca_hip::device_free(dob, ob_bf16.len() * 2);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mla_o_rows_from_device_attn_skips_xs_h2d() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip.clone());
        let n_tok = 3usize;
        let groups = 2usize;
        let o_group_dim = 8usize;
        let per = 4usize;
        let dim = 16usize;
        let q_out = groups * o_group_dim;
        let oa_out = per * groups;
        let mut attn = vec![0.0f32; n_tok * q_out];
        let mut oa_f = vec![0.0f32; groups * per * o_group_dim];
        let mut ob_f = vec![0.0f32; dim * oa_out];
        for (i, v) in attn.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in oa_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in ob_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        let oa_bf16: Vec<u16> = oa_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let ob_bf16: Vec<u16> = ob_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let oa_deq: Vec<f32> = oa_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let ob_deq: Vec<f32> = ob_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let mut cpu_oa = vec![0.0f32; n_tok * oa_out];
        for g in 0..groups {
            let mut xs = vec![0.0f32; n_tok * o_group_dim];
            for t in 0..n_tok {
                let src = t * q_out + g * o_group_dim;
                let dst = t * o_group_dim;
                xs[dst..dst + o_group_dim].copy_from_slice(&attn[src..src + o_group_dim]);
            }
            let w = &oa_deq[g * per * o_group_dim..(g + 1) * per * o_group_dim];
            let mut ys = vec![0.0f32; n_tok * per];
            for t in 0..n_tok {
                for n in 0..per {
                    let mut acc = 0.0f32;
                    for k in 0..o_group_dim {
                        acc += xs[t * o_group_dim + k] * w[n * o_group_dim + k];
                    }
                    ys[t * per + n] = acc;
                }
            }
            for t in 0..n_tok {
                let src = t * per;
                let dst = t * oa_out + g * per;
                cpu_oa[dst..dst + per].copy_from_slice(&ys[src..src + per]);
            }
        }
        let mut cpu_out = vec![0.0f32; n_tok * dim];
        for t in 0..n_tok {
            for n in 0..dim {
                let mut acc = 0.0f32;
                for k in 0..oa_out {
                    acc += cpu_oa[t * oa_out + k] * ob_deq[n * oa_out + k];
                }
                cpu_out[t * dim + n] = acc;
            }
        }
        hip.bind().expect("bind");
        let dattn = allpaca_hip::device_malloc(attn.len() * 4, hip.device).expect("dattn");
        allpaca_hip::memcpy(
            dattn,
            attn.as_ptr().cast(),
            attn.len() * 4,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("h2d attn");
        let doa_w: Vec<*mut u8> = (0..groups)
            .map(|_| allpaca_hip::device_malloc(per * o_group_dim * 2, hip.device).expect("doa_w"))
            .collect();
        let dob = allpaca_hip::device_malloc(ob_bf16.len() * 2, hip.device).expect("dob");
        allpaca_hip::memcpy(
            dob,
            ob_bf16.as_ptr().cast(),
            ob_bf16.len() * 2,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("h2d ob");
        for g in 0..groups {
            let w_off = g * per * o_group_dim;
            allpaca_hip::memcpy(
                doa_w[g],
                oa_bf16[w_off..].as_ptr().cast(),
                per * o_group_dim * 2,
                allpaca_hip::MEMCPY_H2D,
            )
            .expect("h2d oa_w");
        }
        let o_a: Vec<*const u16> = doa_w.iter().map(|p| (*p) as *const u16).collect();
        let h2d_before = hip.activation_h2d_bytes();
        let d2h_before = hip.activation_d2h_bytes();
        let got = gpu
            .mla_o_rows_with_ptrs_from_device(
                dattn.cast(),
                n_tok,
                q_out,
                o_group_dim,
                groups,
                per,
                dim,
                &o_a,
                dob.cast(),
                None,
                false,
            )
            .expect("mla o from device attn");
        assert_eq!(
            hip.activation_h2d_bytes().saturating_sub(h2d_before),
            0,
            "MLA O from device attn must not H2D grouped xs"
        );
        assert_eq!(
            hip.activation_d2h_bytes().saturating_sub(d2h_before),
            (n_tok * dim * 4) as u64,
            "MLA O must D2H attn_out only"
        );
        for i in 0..cpu_out.len() {
            let diff = (got[i] - cpu_out[i]).abs();
            assert!(
                diff < 2e-3,
                "mla o device[{i}] gpu={} cpu={} diff={diff}",
                got[i],
                cpu_out[i]
            );
        }
        allpaca_hip::device_free(dattn, attn.len() * 4);
        for p in doa_w {
            allpaca_hip::device_free(p, per * o_group_dim * 2);
        }
        allpaca_hip::device_free(dob, ob_bf16.len() * 2);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mla_o_rows_from_device_keep_skips_out_d2h() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip.clone());
        let n_tok = 3usize;
        let groups = 2usize;
        let o_group_dim = 8usize;
        let per = 4usize;
        let dim = 16usize;
        let q_out = groups * o_group_dim;
        let oa_out = per * groups;
        let mut attn = vec![0.0f32; n_tok * q_out];
        let mut oa_f = vec![0.0f32; groups * per * o_group_dim];
        let mut ob_f = vec![0.0f32; dim * oa_out];
        for (i, v) in attn.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in oa_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in ob_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        let oa_bf16: Vec<u16> = oa_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let ob_bf16: Vec<u16> = ob_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        hip.bind().expect("bind");
        let dattn = allpaca_hip::device_malloc(attn.len() * 4, hip.device).expect("dattn");
        allpaca_hip::memcpy(
            dattn,
            attn.as_ptr().cast(),
            attn.len() * 4,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("h2d attn");
        let doa_w: Vec<*mut u8> = (0..groups)
            .map(|_| allpaca_hip::device_malloc(per * o_group_dim * 2, hip.device).expect("doa_w"))
            .collect();
        let dob = allpaca_hip::device_malloc(ob_bf16.len() * 2, hip.device).expect("dob");
        allpaca_hip::memcpy(
            dob,
            ob_bf16.as_ptr().cast(),
            ob_bf16.len() * 2,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("h2d ob");
        for g in 0..groups {
            let w_off = g * per * o_group_dim;
            allpaca_hip::memcpy(
                doa_w[g],
                oa_bf16[w_off..].as_ptr().cast(),
                per * o_group_dim * 2,
                allpaca_hip::MEMCPY_H2D,
            )
            .expect("h2d oa_w");
        }
        let o_a: Vec<*const u16> = doa_w.iter().map(|p| (*p) as *const u16).collect();
        let d2h_before = hip.activation_d2h_bytes();
        let got = gpu
            .mla_o_rows_with_ptrs_from_device(
                dattn.cast(),
                n_tok,
                q_out,
                o_group_dim,
                groups,
                per,
                dim,
                &o_a,
                dob.cast(),
                None,
                true,
            )
            .expect("mla o keep");
        assert!(got.is_empty(), "keep-out must skip host Vec");
        assert_eq!(
            hip.activation_d2h_bytes().saturating_sub(d2h_before),
            0,
            "keep-out must not D2H attn_out"
        );
        let p = gpu.last_o_device().expect("device O pointer");
        let mut pulled = vec![0.0f32; n_tok * dim];
        gpu.copy_d2h_f32(p, &mut pulled).expect("d2h check");
        let host = gpu
            .mla_o_rows_with_ptrs_from_device(
                dattn.cast(),
                n_tok,
                q_out,
                o_group_dim,
                groups,
                per,
                dim,
                &o_a,
                dob.cast(),
                None,
                false,
            )
            .expect("mla o host");
        for i in 0..host.len() {
            let diff = (pulled[i] - host[i]).abs();
            assert!(
                diff < 2e-3,
                "keep-out[{i}] {} vs {} diff={diff}",
                pulled[i],
                host[i]
            );
        }
        allpaca_hip::device_free(dattn, attn.len() * 4);
        for p in doa_w {
            allpaca_hip::device_free(p, per * o_group_dim * 2);
        }
        allpaca_hip::device_free(dob, ob_bf16.len() * 2);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hc_post_packed_from_device_xs_skips_xs_h2d() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        let mut gpu = super::Dsv4Gpu::for_hc_test(hip.clone());
        let n_tok = 3usize;
        let hc = 4usize;
        let dim = 8usize;
        let hc_dim = hc * dim;
        let packed: Vec<f32> = (0..n_tok * hc_dim)
            .map(|i| 0.05 * (i as f32 % 11.0) - 0.2)
            .collect();
        let xs: Vec<f32> = (0..n_tok * dim).map(|i| 0.03 * i as f32).collect();
        let posts: Vec<f32> = (0..n_tok * hc).map(|i| 0.4 + 0.05 * i as f32).collect();
        let combs: Vec<f32> = (0..n_tok * hc * hc)
            .map(|i| 0.02 * ((i % 7) as f32) - 0.05)
            .collect();
        hip.bind().expect("bind");
        let dxs = allpaca_hip::device_malloc(xs.len() * 4, hip.device).expect("dxs");
        allpaca_hip::memcpy(
            dxs,
            xs.as_ptr().cast(),
            xs.len() * 4,
            allpaca_hip::MEMCPY_H2D,
        )
        .expect("seed xs");
        let h2d_before = hip.activation_h2d_bytes();
        gpu.hc_post_packed(
            &packed,
            &[],
            &posts,
            &combs,
            n_tok,
            hc,
            dim,
            Some(dxs.cast()),
        )
        .expect("device xs post");
        let h2d_delta = hip.activation_h2d_bytes().saturating_sub(h2d_before);
        assert_eq!(
            h2d_delta,
            ((packed.len() + posts.len() + combs.len()) * 4) as u64,
            "hc_post from device xs H2D should be residual+posts+combs, not xs"
        );
        let mut got = vec![0.0f32; packed.len()];
        gpu.download_hc_packed(&mut got).expect("download post");
        let mut expect = vec![0.0f32; packed.len()];
        for t in 0..n_tok {
            for dst in 0..hc {
                for i in 0..dim {
                    let mut acc = xs[t * dim + i] * posts[t * hc + dst];
                    for src in 0..hc {
                        let w = combs[t * hc * hc + dst + src * hc];
                        acc += packed[t * hc_dim + src * dim + i] * w;
                    }
                    expect[t * hc_dim + dst * dim + i] = acc;
                }
            }
        }
        for i in 0..packed.len() {
            let diff = (got[i] - expect[i]).abs();
            assert!(
                diff < 2e-4,
                "device-xs post[{i}] gpu={} cpu={} diff={diff}",
                got[i],
                expect[i]
            );
        }
        allpaca_hip::device_free(dxs, xs.len() * 4);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn expert_pinned_bounce_h2d_matches_direct() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let Some(pin) = super::try_pin_pair() else {
            return;
        };
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        hip.bind().expect("bind");
        let blob = allpaca_hip::DeviceWeights::alloc(hip.device, 4 * 1024 * 1024).expect("blob");
        let h2d = allpaca_hip::H2dStream::new().expect("h2d");
        let a: Vec<u8> = (0..1_500_000).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..800_000).map(|i| (i % 241) as u8).collect();
        let parts = [(0usize, a.as_slice()), (2_000_000usize, b.as_slice())];
        super::copy_expert_parts(&blob, &parts, Some(&h2d), Some(&pin)).expect("bounce h2d");
        h2d.synchronize().expect("sync");
        let mut got_a = vec![0u8; a.len()];
        let mut got_b = vec![0u8; b.len()];
        allpaca_hip::memcpy(
            got_a.as_mut_ptr(),
            blob.byte_ptr_at(0),
            got_a.len(),
            allpaca_hip::MEMCPY_D2H,
        )
        .expect("d2h a");
        allpaca_hip::memcpy(
            got_b.as_mut_ptr(),
            blob.byte_ptr_at(2_000_000),
            got_b.len(),
            allpaca_hip::MEMCPY_D2H,
        )
        .expect("d2h b");
        assert_eq!(got_a, a);
        assert_eq!(got_b, b);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn expert_pinned_bounce_packs_contiguous_parts() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let Some(pin) = super::try_pin_pair() else {
            return;
        };
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        hip.bind().expect("bind");
        let blob = allpaca_hip::DeviceWeights::alloc(hip.device, 4 * 1024 * 1024).expect("blob");
        let h2d = allpaca_hip::H2dStream::new().expect("h2d");
        let a: Vec<u8> = (0..1_500_000).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..800_000).map(|i| (i % 241) as u8).collect();
        let parts = [(0usize, a.as_slice()), (a.len(), b.as_slice())];
        assert_eq!(super::packed_run_parts(&parts, 0), 2);
        super::copy_expert_parts(&blob, &parts, Some(&h2d), Some(&pin)).expect("packed bounce");
        h2d.synchronize().expect("sync");
        let mut got = vec![0u8; a.len() + b.len()];
        allpaca_hip::memcpy(
            got.as_mut_ptr(),
            blob.byte_ptr_at(0),
            got.len(),
            allpaca_hip::MEMCPY_D2H,
        )
        .expect("d2h packed");
        assert_eq!(&got[..a.len()], a.as_slice());
        assert_eq!(&got[a.len()..], b.as_slice());
    }

    #[cfg(feature = "hip")]
    #[test]
    fn expert_windowed_register_h2d_matches_direct() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let path = std::env::temp_dir().join("allpaca-window-h2d.bin");
        let a: Vec<u8> = (0..1_500_000).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..800_000).map(|i| (i % 241) as u8).collect();
        {
            let mut file = std::fs::File::create(&path).expect("create mmap file");
            use std::io::Write;
            file.write_all(&a).expect("write a");
            file.write_all(&b).expect("write b");
        }
        let file = std::fs::File::open(&path).expect("open mmap file");
        // SAFETY: `path` is a private temp file we just wrote; no other maps.
        let map = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
        let sl_a = &map[..a.len()];
        let sl_b = &map[a.len()..];
        let Some((ptr, len)) = super::page_range(&sl_a[..sl_a.len().min(4096)]) else {
            let _ = std::fs::remove_file(&path);
            return;
        };
        // Pinning a read-only file mapping is platform-dependent (Windows HIP
        // refuses it); the copy below must be right either way.
        match allpaca_hip::host_register(ptr as *mut u8, len, 0) {
            Ok(()) => allpaca_hip::host_unregister(ptr as *mut u8),
            Err(e) => eprintln!("file mmap window not pinnable here ({e}); checking the fallback copy"),
        }
        let pin = super::try_window_pin_pair();
        let hip = std::sync::Arc::new(
            allpaca_hip::HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace"),
        );
        hip.bind().expect("bind");
        let blob = allpaca_hip::DeviceWeights::alloc(hip.device, 4 * 1024 * 1024).expect("blob");
        let h2d = allpaca_hip::H2dStream::new().expect("h2d");
        let parts = [(0usize, sl_a), (2_000_000usize, sl_b)];
        super::copy_expert_parts(&blob, &parts, Some(&h2d), Some(&pin)).expect("window h2d");
        h2d.synchronize().expect("sync");
        let mut got_a = vec![0u8; a.len()];
        let mut got_b = vec![0u8; b.len()];
        allpaca_hip::memcpy(
            got_a.as_mut_ptr(),
            blob.byte_ptr_at(0),
            got_a.len(),
            allpaca_hip::MEMCPY_D2H,
        )
        .expect("d2h a");
        allpaca_hip::memcpy(
            got_b.as_mut_ptr(),
            blob.byte_ptr_at(2_000_000),
            got_b.len(),
            allpaca_hip::MEMCPY_D2H,
        )
        .expect("d2h b");
        let _ = std::fs::remove_file(&path);
        assert_eq!(got_a, a);
        assert_eq!(got_b, b);
    }
}
