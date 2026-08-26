use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use allpaca_cpu::copy_async;
use allpaca_types::{
    AllpacaError, DeviceBuffer, DeviceBufferKind, FatalError, KvPage, KvResidency, PageId,
    TransientError,
};

use crate::kv_nvme::{temp_kv_nvme_path, NvmeKvStore};
use crate::telemetry::Metrics;

struct HipPageAlloc {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for HipPageAlloc {}
unsafe impl Sync for HipPageAlloc {}

struct BufferPool {
    data: Vec<f32>,
    bump: usize,
    live: usize,
    capacity: usize,
    free: Vec<(usize, usize)>,
}

impl BufferPool {
    fn new(capacity_bytes: usize) -> Self {
        let capacity = (capacity_bytes / 4).max(1);
        Self {
            data: vec![0.0; capacity],
            bump: 0,
            live: 0,
            capacity,
            free: Vec::new(),
        }
    }

    fn alloc(&mut self, len: usize) -> Result<usize, AllpacaError> {
        if let Some(idx) = self.free.iter().position(|(_, l)| *l >= len) {
            let (off, l) = self.free.remove(idx);
            if l > len {
                self.free.push((off + len, l - len));
            }
            self.live += len;
            return Ok(off);
        }
        if self.bump + len > self.capacity {
            return Err(AllpacaError::Transient(TransientError::Backpressure(
                "KV pool exhausted".into(),
            )));
        }
        let off = self.bump;
        self.bump += len;
        self.live += len;
        Ok(off)
    }

    fn free(&mut self, off: usize, len: usize) {
        self.free.push((off, len));
        self.live = self.live.saturating_sub(len);
    }

    fn slice(&self, off: usize, len: usize) -> &[f32] {
        &self.data[off..off + len]
    }

    fn slice_mut(&mut self, off: usize, len: usize) -> &mut [f32] {
        &mut self.data[off..off + len]
    }
}

/// Three-tier KV manager: VRAM (hot) <-> Pinned DRAM (warm) <-> NVMe (cold).
/// NVMe is off until [`Self::open_nvme`]. When `hip_device` is set, VRAM pages
/// are real `hipMalloc` buffers.
pub struct KvTierManager {
    pages: HashMap<PageId, KvPage>,
    vram: BufferPool,
    dram: BufferPool,
    nvme: Option<NvmeKvStore>,
    hip_pages: HashMap<PageId, HipPageAlloc>,
    hip_device: Option<i32>,
    activation_h2d_bytes: AtomicU64,
    activation_d2h_bytes: AtomicU64,
    nvme_prefetches: AtomicU64,
    next_id: u64,
    clock: u64,
    hot_fraction: f32,
    vram_capacity: usize,
}

impl KvTierManager {
    pub fn new(vram_bytes: usize, dram_bytes: usize, hot_fraction: f32) -> Self {
        Self::with_hip(vram_bytes, dram_bytes, hot_fraction, None)
    }

    pub fn with_hip(
        vram_bytes: usize,
        dram_bytes: usize,
        hot_fraction: f32,
        hip_device: Option<i32>,
    ) -> Self {
        Self {
            pages: HashMap::new(),
            vram: BufferPool::new(vram_bytes),
            dram: BufferPool::new(dram_bytes),
            nvme: None,
            hip_pages: HashMap::new(),
            hip_device,
            activation_h2d_bytes: AtomicU64::new(0),
            activation_d2h_bytes: AtomicU64::new(0),
            nvme_prefetches: AtomicU64::new(0),
            next_id: 1,
            clock: 1,
            hot_fraction,
            vram_capacity: (vram_bytes / 4).max(1),
        }
    }

    /// Enable the NVMe cold tier. `path = None` uses a temp file removed on drop.
    pub fn open_nvme(
        &mut self,
        nvme_bytes: usize,
        path: Option<&std::path::Path>,
    ) -> Result<(), AllpacaError> {
        if nvme_bytes == 0 {
            self.nvme = None;
            return Ok(());
        }
        let (store_path, remove) = match path {
            Some(p) => (p.to_path_buf(), false),
            None => (temp_kv_nvme_path(), true),
        };
        self.nvme = Some(NvmeKvStore::open(&store_path, nvme_bytes, remove)?);
        Ok(())
    }

    pub fn page_floats(page_tokens: u32, n_kv_heads: usize, head_dim: usize) -> usize {
        page_tokens as usize * n_kv_heads * head_dim * 2
    }

    pub fn alloc_page(
        &mut self,
        layer: u32,
        page_index: u32,
        page_tokens: u32,
        n_kv_heads: usize,
        head_dim: usize,
        metrics: &Metrics,
    ) -> Result<PageId, AllpacaError> {
        let len = Self::page_floats(page_tokens, n_kv_heads, head_dim);
        self.evict_if_needed(len, metrics)?;
        let id = self.next_id;
        let (tier, offset, kind, device_id) = if let Some(dev) = self.hip_device {
            let bytes = len * 4;
            match allpaca_hip::device_malloc(bytes, dev) {
                Ok(ptr) => {
                    self.hip_pages.insert(id, HipPageAlloc { ptr, bytes });
                    (KvResidency::Vram, 0, DeviceBufferKind::Hip, dev)
                }
                Err(_) => {
                    let off = self.dram_alloc(len, metrics).map_err(|_| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(
                            "HIP KV alloc failed and pinned DRAM is full".into(),
                        ))
                    })?;
                    (KvResidency::PinnedDram, off, DeviceBufferKind::CpuPinnedDram, -1)
                }
            }
        } else {
            match self.vram.alloc(len) {
                Ok(off) => (KvResidency::Vram, off, DeviceBufferKind::CpuVram, -1),
                Err(_) => {
                    let off = self.dram_alloc(len, metrics).map_err(|_| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(
                            "VRAM and pinned DRAM KV pools are full".into(),
                        ))
                    })?;
                    (KvResidency::PinnedDram, off, DeviceBufferKind::CpuPinnedDram, -1)
                }
            }
        };
        self.next_id += 1;
        self.clock += 1;
        self.pages.insert(
            id,
            KvPage {
                id,
                layer,
                page_index,
                filled: 0,
                capacity_tokens: page_tokens,
                tier,
                refcount: 1,
                last_access: self.clock,
                predicted_reuse: 1.0,
                location: DeviceBuffer {
                    kind,
                    device_id,
                    offset,
                    len,
                },
            },
        );
        self.refresh_metrics(metrics);
        Ok(id)
    }

    pub fn get(&self, id: PageId) -> Option<&KvPage> {
        self.pages.get(&id)
    }

    pub fn get_mut(&mut self, id: PageId) -> Option<&mut KvPage> {
        self.pages.get_mut(&id)
    }

    pub fn touch(&mut self, id: PageId, reuse: f32) {
        self.clock += 1;
        if let Some(page) = self.pages.get_mut(&id) {
            page.last_access = self.clock;
            page.predicted_reuse = reuse;
        }
    }

    pub fn retain(&mut self, id: PageId) {
        if let Some(page) = self.pages.get_mut(&id) {
            page.refcount = page.refcount.saturating_add(1);
        }
    }

    pub fn release(&mut self, id: PageId, metrics: &Metrics) {
        let Some(page) = self.pages.get_mut(&id) else {
            return;
        };
        page.refcount = page.refcount.saturating_sub(1);
        if page.refcount > 0 {
            return;
        }
        let loc = page.location.clone();
        match loc.kind {
            DeviceBufferKind::Hip => {
                if let Some(alloc) = self.hip_pages.remove(&id) {
                    allpaca_hip::device_free(alloc.ptr, alloc.bytes);
                }
            }
            DeviceBufferKind::CpuVram => self.vram.free(loc.offset, loc.len),
            DeviceBufferKind::CpuPinnedDram => self.dram.free(loc.offset, loc.len),
            DeviceBufferKind::Nvme => {
                if let Some(store) = self.nvme.as_ref() {
                    store.free_page(id);
                }
            }
        }
        self.pages.remove(&id);
        self.refresh_metrics(metrics);
    }

    pub fn write_slot(
        &mut self,
        id: PageId,
        slot: u32,
        n_kv_heads: usize,
        head_dim: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), AllpacaError> {
        self.ensure_dram(id)?;
        let page = self.pages.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
        })?;
        let kv_width = n_kv_heads * head_dim;
        let k_off = page.location.offset + slot as usize * kv_width;
        let v_off = page.location.offset + page.capacity_tokens as usize * kv_width + slot as usize * kv_width;
        match page.location.kind {
            DeviceBufferKind::Hip => {
                let alloc = self.hip_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "HIP KV page {id} missing device buffer"
                    )))
                })?;
                let k_bytes = kv_width * 4;
                let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                allpaca_hip::memcpy(
                    k_ptr,
                    k.as_ptr().cast(),
                    k_bytes,
                    allpaca_hip::MEMCPY_H2D,
                )?;
                allpaca_hip::memcpy(
                    v_ptr,
                    v.as_ptr().cast(),
                    k_bytes,
                    allpaca_hip::MEMCPY_H2D,
                )?;
                self.activation_h2d_bytes
                    .fetch_add((k_bytes * 2) as u64, Ordering::Relaxed);
            }
            DeviceBufferKind::CpuVram => {
                self.vram.slice_mut(k_off, kv_width).copy_from_slice(k);
                self.vram.slice_mut(v_off, kv_width).copy_from_slice(v);
            }
            DeviceBufferKind::CpuPinnedDram => {
                self.dram.slice_mut(k_off, kv_width).copy_from_slice(k);
                self.dram.slice_mut(v_off, kv_width).copy_from_slice(v);
            }
            DeviceBufferKind::Nvme => {
                return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                    "NVMe KV page must be restored to DRAM before write".into(),
                )));
            }
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.filled = page.filled.max(slot + 1);
        }
        self.prefetch_next(id);
        Ok(())
    }

    pub fn read_kv(
        &self,
        id: PageId,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
        let page = self.pages.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
        })?;
        let kv_width = n_kv_heads * head_dim;
        let filled = page.filled as usize;
        let cap = page.capacity_tokens as usize;
        let k_off = page.location.offset;
        let v_off = page.location.offset + cap * kv_width;
        let n = filled * kv_width;
        let (k, v) = match page.location.kind {
            DeviceBufferKind::Hip => {
                let alloc = self.hip_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "HIP KV page {id} missing device buffer"
                    )))
                })?;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                allpaca_hip::memcpy(
                    k.as_mut_ptr().cast(),
                    unsafe { alloc.ptr.add(k_off * 4) },
                    n * 4,
                    allpaca_hip::MEMCPY_D2H,
                )?;
                allpaca_hip::memcpy(
                    v.as_mut_ptr().cast(),
                    unsafe { alloc.ptr.add(v_off * 4) },
                    n * 4,
                    allpaca_hip::MEMCPY_D2H,
                )?;
                self.activation_d2h_bytes
                    .fetch_add((n * 8) as u64, Ordering::Relaxed);
                (k, v)
            }
            DeviceBufferKind::CpuVram => (
                self.vram.slice(k_off, n).to_vec(),
                self.vram.slice(v_off, n).to_vec(),
            ),
            DeviceBufferKind::CpuPinnedDram => (
                self.dram.slice(k_off, n).to_vec(),
                self.dram.slice(v_off, n).to_vec(),
            ),
            DeviceBufferKind::Nvme => {
                let store = self.nvme.as_ref().ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(
                        "NVMe KV page with no cold-tier store".into(),
                    ))
                })?;
                let data = store.read_page(page.location.offset, page.location.len)?;
                let v0 = cap * kv_width;
                let k = data.get(..n).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV K slice".into()))
                })?;
                let v = data.get(v0..v0 + n).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV V slice".into()))
                })?;
                (k.to_vec(), v.to_vec())
            }
        };
        Ok((k, v))
    }

    pub fn activation_h2d_bytes(&self) -> u64 {
        self.activation_h2d_bytes.load(Ordering::Relaxed)
    }

    pub fn activation_d2h_bytes(&self) -> u64 {
        self.activation_d2h_bytes.load(Ordering::Relaxed)
    }

    /// D2D write of a K/V slot from device pointers (no host bounce).
    pub(crate) fn write_slot_from_device(
        &mut self,
        id: PageId,
        slot: u32,
        n_kv_heads: usize,
        head_dim: usize,
        k_dev: *const f32,
        v_dev: *const f32,
    ) -> Result<(), AllpacaError> {
        if k_dev.is_null() || v_dev.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "device K/V pointer is null".into(),
            )));
        }
        self.ensure_dram(id)?;
        let page = self.pages.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
        })?;
        let kv_width = n_kv_heads * head_dim;
        let k_off = page.location.offset + slot as usize * kv_width;
        let v_off = page.location.offset
            + page.capacity_tokens as usize * kv_width
            + slot as usize * kv_width;
        let k_bytes = kv_width * 4;
        match page.location.kind {
            DeviceBufferKind::Hip => {
                let alloc = self.hip_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "HIP KV page {id} missing device buffer"
                    )))
                })?;
                let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                allpaca_hip::memcpy(k_ptr, k_dev.cast(), k_bytes, allpaca_hip::MEMCPY_D2D)?;
                allpaca_hip::memcpy(v_ptr, v_dev.cast(), k_bytes, allpaca_hip::MEMCPY_D2D)?;
            }
            DeviceBufferKind::CpuVram | DeviceBufferKind::CpuPinnedDram => {
                let mut k = vec![0.0f32; kv_width];
                let mut v = vec![0.0f32; kv_width];
                allpaca_hip::memcpy(
                    k.as_mut_ptr().cast(),
                    k_dev.cast(),
                    k_bytes,
                    allpaca_hip::MEMCPY_D2H,
                )?;
                allpaca_hip::memcpy(
                    v.as_mut_ptr().cast(),
                    v_dev.cast(),
                    k_bytes,
                    allpaca_hip::MEMCPY_D2H,
                )?;
                self.activation_d2h_bytes
                    .fetch_add((k_bytes * 2) as u64, Ordering::Relaxed);
                return self.write_slot(id, slot, n_kv_heads, head_dim, &k, &v);
            }
            DeviceBufferKind::Nvme => {
                return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                    "NVMe KV page must be restored to DRAM before device write".into(),
                )));
            }
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.filled = page.filled.max(slot + 1);
        }
        self.prefetch_next(id);
        Ok(())
    }

    /// D2D write of packed K/V rows `[n_tok, kv_width]`, one copy per page run.
    pub(crate) fn write_slots_from_device(
        &mut self,
        page_ids: &[PageId],
        start_pos: usize,
        n_tok: usize,
        page_tokens: usize,
        n_kv_heads: usize,
        head_dim: usize,
        k_dev: *const f32,
        v_dev: *const f32,
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 {
            return Ok(());
        }
        if page_tokens == 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "KV page_tokens must be > 0".into(),
            )));
        }
        if k_dev.is_null() || v_dev.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "device K/V pointer is null".into(),
            )));
        }
        let kv_width = n_kv_heads * head_dim;
        let mut i = 0;
        while i < n_tok {
            let pos = start_pos + i;
            let page_index = pos / page_tokens;
            let slot = pos % page_tokens;
            let page_id = *page_ids.get(page_index).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                    "KV page index {page_index} out of range"
                )))
            })?;
            self.ensure_dram(page_id)?;
            let run = (n_tok - i).min(page_tokens - slot);
            let (kind, offset, cap) = {
                let page = self.pages.get(&page_id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {page_id}")))
                })?;
                (
                    page.location.kind,
                    page.location.offset,
                    page.capacity_tokens as usize,
                )
            };
            match kind {
                DeviceBufferKind::Hip => {
                    let k_off = offset + slot * kv_width;
                    let v_off = offset + cap * kv_width + slot * kv_width;
                    let alloc = self.hip_pages.get(&page_id).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                            "HIP KV page {page_id} missing device buffer"
                        )))
                    })?;
                    let run_bytes = run * kv_width * 4;
                    let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                    let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                    let k_src = unsafe { k_dev.add(i * kv_width) };
                    let v_src = unsafe { v_dev.add(i * kv_width) };
                    allpaca_hip::memcpy(k_ptr, k_src.cast(), run_bytes, allpaca_hip::MEMCPY_D2D)?;
                    allpaca_hip::memcpy(v_ptr, v_src.cast(), run_bytes, allpaca_hip::MEMCPY_D2D)?;
                    if let Some(page) = self.pages.get_mut(&page_id) {
                        page.filled = page.filled.max((slot + run) as u32);
                    }
                }
                DeviceBufferKind::CpuVram | DeviceBufferKind::CpuPinnedDram => {
                    for t in 0..run {
                        self.write_slot_from_device(
                            page_id,
                            (slot + t) as u32,
                            n_kv_heads,
                            head_dim,
                            unsafe { k_dev.add((i + t) * kv_width) },
                            unsafe { v_dev.add((i + t) * kv_width) },
                        )?;
                    }
                }
                DeviceBufferKind::Nvme => {
                    return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                        "NVMe KV page must be restored to DRAM before device write".into(),
                    )));
                }
            }
            self.touch(page_id, 1.0);
            self.prefetch_next(page_id);
            i += run;
        }
        Ok(())
    }

    /// Keep KV `filled` equal to `seq_len` tokens across `pages` (one vec per layer).
    /// Packed linear verify writes a speculative span, then rewinds the rejected suffix.
    pub(crate) fn rewind_filled(
        &mut self,
        pages: &[Vec<PageId>],
        seq_len: usize,
        page_tokens: usize,
    ) {
        if page_tokens == 0 {
            return;
        }
        for layer in pages {
            for (i, id) in layer.iter().enumerate() {
                let start = i * page_tokens;
                let filled = if seq_len <= start {
                    0
                } else {
                    (seq_len - start).min(page_tokens) as u32
                };
                if let Some(page) = self.pages.get_mut(id) {
                    page.filled = filled;
                }
            }
        }
    }

    /// Device K/V views for attention. A single resident HIP page is used in-place
    /// (no D2D pack). Multiple or host pages still pack into `dest_k` / `dest_v`.
    pub(crate) fn kv_device_views(
        &self,
        pages: &[u64],
        n_kv_heads: usize,
        head_dim: usize,
        dest_k: *mut f32,
        dest_v: *mut f32,
        max_kv: usize,
    ) -> Result<(*const f32, *const f32, usize), AllpacaError> {
        if pages.len() == 1 {
            let id = pages[0];
            if let Some(page) = self.pages.get(&id) {
                let filled = page.filled as usize;
                if filled > 0
                    && filled <= max_kv
                    && page.location.kind == DeviceBufferKind::Hip
                {
                    let alloc = self.hip_pages.get(&id).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                            "HIP KV page {id} missing device buffer"
                        )))
                    })?;
                    let kv_width = n_kv_heads * head_dim;
                    let k_off = page.location.offset;
                    let v_off = page.location.offset + page.capacity_tokens as usize * kv_width;
                    let k = unsafe { alloc.ptr.add(k_off * 4).cast::<f32>() };
                    let v = unsafe { alloc.ptr.add(v_off * 4).cast::<f32>() };
                    return Ok((k, v, filled));
                }
            }
        }
        let n_kv = self.pack_kv_device(pages, n_kv_heads, head_dim, dest_k, dest_v, max_kv)?;
        Ok((dest_k as *const f32, dest_v as *const f32, n_kv))
    }

    /// Pack filled tokens from `pages` into device K/V buffers (D2D when resident).
    pub(crate) fn pack_kv_device(
        &self,
        pages: &[u64],
        n_kv_heads: usize,
        head_dim: usize,
        dest_k: *mut f32,
        dest_v: *mut f32,
        max_kv: usize,
    ) -> Result<usize, AllpacaError> {
        if dest_k.is_null() || dest_v.is_null() || max_kv == 0 {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "packed device KV destination is empty".into(),
            )));
        }
        let kv_width = n_kv_heads * head_dim;
        let mut n_kv = 0usize;
        for id in pages {
            let page = self.pages.get(id).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
            })?;
            let filled = page.filled as usize;
            if filled == 0 {
                continue;
            }
            if n_kv + filled > max_kv {
                return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                    "packed device KV exceeds workspace".into(),
                )));
            }
            let n = filled * kv_width;
            let k_off = page.location.offset;
            let v_off = page.location.offset + page.capacity_tokens as usize * kv_width;
            let dk = unsafe { dest_k.add(n_kv * kv_width).cast::<u8>() };
            let dv = unsafe { dest_v.add(n_kv * kv_width).cast::<u8>() };
            match page.location.kind {
                DeviceBufferKind::Hip => {
                    let alloc = self.hip_pages.get(id).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                            "HIP KV page {id} missing device buffer"
                        )))
                    })?;
                    allpaca_hip::memcpy(
                        dk,
                        unsafe { alloc.ptr.add(k_off * 4) },
                        n * 4,
                        allpaca_hip::MEMCPY_D2D,
                    )?;
                    allpaca_hip::memcpy(
                        dv,
                        unsafe { alloc.ptr.add(v_off * 4) },
                        n * 4,
                        allpaca_hip::MEMCPY_D2D,
                    )?;
                }
                DeviceBufferKind::CpuVram => {
                    allpaca_hip::memcpy(
                        dk,
                        self.vram.slice(k_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_hip::MEMCPY_H2D,
                    )?;
                    allpaca_hip::memcpy(
                        dv,
                        self.vram.slice(v_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_hip::MEMCPY_H2D,
                    )?;
                    self.activation_h2d_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                }
                DeviceBufferKind::CpuPinnedDram => {
                    allpaca_hip::memcpy(
                        dk,
                        self.dram.slice(k_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_hip::MEMCPY_H2D,
                    )?;
                    allpaca_hip::memcpy(
                        dv,
                        self.dram.slice(v_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_hip::MEMCPY_H2D,
                    )?;
                    self.activation_h2d_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                }
                DeviceBufferKind::Nvme => {
                    let store = self.nvme.as_ref().ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(
                            "NVMe KV page with no cold-tier store".into(),
                        ))
                    })?;
                    let data = store.read_page(page.location.offset, page.location.len)?;
                    let v0 = page.capacity_tokens as usize * kv_width;
                    let k = data.get(..n).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV K pack".into()))
                    })?;
                    let v = data.get(v0..v0 + n).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV V pack".into()))
                    })?;
                    allpaca_hip::memcpy(dk, k.as_ptr().cast(), n * 4, allpaca_hip::MEMCPY_H2D)?;
                    allpaca_hip::memcpy(dv, v.as_ptr().cast(), n * 4, allpaca_hip::MEMCPY_H2D)?;
                    self.activation_h2d_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                }
            }
            n_kv += filled;
        }
        Ok(n_kv)
    }

    pub fn migrate_to_dram(&mut self, id: PageId, metrics: &Metrics) -> Result<(), AllpacaError> {
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        if page.tier == KvResidency::PinnedDram {
            return Ok(());
        }
        let len = page.location.len;
        let dst = self.dram_alloc(len, metrics)?;
        match page.location.kind {
            DeviceBufferKind::Hip => {
                if let Some(alloc) = self.hip_pages.remove(&id) {
                    allpaca_hip::memcpy(
                        self.dram.slice_mut(dst, len).as_mut_ptr().cast(),
                        alloc.ptr,
                        alloc.bytes,
                        allpaca_hip::MEMCPY_D2H,
                    )?;
                    allpaca_hip::device_free(alloc.ptr, alloc.bytes);
                }
            }
            DeviceBufferKind::CpuVram => {
                copy_f32(
                    self.dram.slice_mut(dst, len),
                    self.vram.slice(page.location.offset, len),
                )?;
                self.vram.free(page.location.offset, len);
            }
            DeviceBufferKind::Nvme => {
                let data = {
                    let store = self.nvme.as_ref().ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(
                            "NVMe KV page with no cold-tier store".into(),
                        ))
                    })?;
                    store.read_page(page.location.offset, len)?
                };
                self.dram.slice_mut(dst, len).copy_from_slice(&data);
                if let Some(store) = self.nvme.as_ref() {
                    store.free_page(id);
                }
            }
            DeviceBufferKind::CpuPinnedDram => {}
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.tier = KvResidency::PinnedDram;
            page.location.kind = DeviceBufferKind::CpuPinnedDram;
            page.location.device_id = -1;
            page.location.offset = dst;
        }
        metrics
            .page_migrations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.refresh_metrics(metrics);
        Ok(())
    }

    pub fn migrate_to_vram(&mut self, id: PageId, metrics: &Metrics) -> Result<(), AllpacaError> {
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        if page.tier == KvResidency::Vram {
            return Ok(());
        }
        if page.tier == KvResidency::Nvme {
            self.migrate_to_dram(id, metrics)?;
        }
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        let len = page.location.len;
        self.evict_if_needed(len, metrics)?;
        if let Some(dev) = self.hip_device {
            let bytes = len * 4;
            let ptr = allpaca_hip::device_malloc(bytes, dev)?;
            allpaca_hip::memcpy(
                ptr,
                self.dram.slice(page.location.offset, len).as_ptr().cast(),
                bytes,
                allpaca_hip::MEMCPY_H2D,
            )?;
            self.dram.free(page.location.offset, len);
            self.hip_pages.insert(id, HipPageAlloc { ptr, bytes });
            if let Some(page) = self.pages.get_mut(&id) {
                page.tier = KvResidency::Vram;
                page.location.kind = DeviceBufferKind::Hip;
                page.location.device_id = dev;
                page.location.offset = 0;
            }
        } else {
            let dst = self.vram.alloc(len)?;
            copy_f32(
                self.vram.slice_mut(dst, len),
                self.dram.slice(page.location.offset, len),
            )?;
            self.dram.free(page.location.offset, len);
            if let Some(page) = self.pages.get_mut(&id) {
                page.tier = KvResidency::Vram;
                page.location.kind = DeviceBufferKind::CpuVram;
                page.location.offset = dst;
            }
        }
        metrics
            .page_migrations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.refresh_metrics(metrics);
        Ok(())
    }

    fn migrate_to_nvme(&mut self, id: PageId, metrics: &Metrics) -> Result<(), AllpacaError> {
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        if page.tier == KvResidency::Nvme {
            return Ok(());
        }
        if page.tier == KvResidency::Vram {
            self.migrate_to_dram(id, metrics)?;
        }
        let page = self.pages.get(&id).cloned().ok_or_else(|| {
            AllpacaError::Transient(TransientError::PageMigration(format!("missing {id}")))
        })?;
        let len = page.location.len;
        let data = self.dram.slice(page.location.offset, len).to_vec();
        let slot = {
            let store = self.nvme.as_ref().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::OutOfMemory(
                    "NVMe KV cold tier is not open".into(),
                ))
            })?;
            store.write_page(id, &data)?
        };
        self.dram.free(page.location.offset, len);
        if let Some(page) = self.pages.get_mut(&id) {
            page.tier = KvResidency::Nvme;
            page.location.kind = DeviceBufferKind::Nvme;
            page.location.device_id = -1;
            page.location.offset = slot;
        }
        metrics
            .page_migrations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.refresh_metrics(metrics);
        Ok(())
    }

    fn dram_alloc(&mut self, len: usize, metrics: &Metrics) -> Result<usize, AllpacaError> {
        loop {
            match self.dram.alloc(len) {
                Ok(off) => return Ok(off),
                Err(e) => {
                    if self.nvme.is_none() {
                        return Err(e);
                    }
                    let Some(victim) = self.lowest_dram_page() else {
                        return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                            "pinned DRAM is full and no DRAM page can spill to NVMe".into(),
                        )));
                    };
                    self.migrate_to_nvme(victim, metrics)?;
                }
            }
        }
    }

    fn ensure_dram(&mut self, id: PageId) -> Result<(), AllpacaError> {
        let tier = self.pages.get(&id).map(|p| p.tier);
        if tier != Some(KvResidency::Nvme) {
            return Ok(());
        }
        let metrics = Metrics::default();
        self.migrate_to_dram(id, &metrics)
    }

    fn prefetch_next(&mut self, id: PageId) {
        let Some(page) = self.pages.get(&id).cloned() else {
            return;
        };
        let want = page.page_index.saturating_add(1);
        let Some(next_id) = self.pages.values().find(|p| {
            p.layer == page.layer
                && p.page_index == want
                && p.tier == KvResidency::Nvme
                && p.id != id
        }).map(|p| p.id) else {
            return;
        };
        if self.migrate_to_dram(next_id, &Metrics::default()).is_ok() {
            self.nvme_prefetches.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn lowest_dram_page(&self) -> Option<PageId> {
        self.pages
            .values()
            .filter(|p| p.tier == KvResidency::PinnedDram)
            .min_by(|a, b| {
                a.priority(self.clock, 1.0, 0.4)
                    .partial_cmp(&b.priority(self.clock, 1.0, 0.4))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
    }

    fn vram_live_floats(&self) -> usize {
        if self.hip_device.is_some() {
            self.hip_pages.values().map(|p| p.bytes / 4).sum()
        } else {
            self.vram.live
        }
    }

    /// Shrink or grow the VRAM KV budget. Excess hot pages spill to DRAM.
    /// Does not touch the weight host pool.
    pub fn set_vram_bytes(
        &mut self,
        vram_bytes: usize,
        metrics: &Metrics,
    ) -> Result<(), AllpacaError> {
        self.vram_capacity = (vram_bytes / 4).max(1);
        self.evict_if_needed(0, metrics)?;
        while self.vram_live_floats() > self.vram_capacity {
            let Some(id) = self.lowest_vram_page() else {
                break;
            };
            self.migrate_to_dram(id, metrics)?;
        }
        Ok(())
    }

    fn evict_if_needed(&mut self, needed: usize, metrics: &Metrics) -> Result<(), AllpacaError> {
        let hot_cap = ((self.vram_capacity as f32) * self.hot_fraction).max(1.0) as usize;
        while self.vram_live_floats() + needed > hot_cap {
            let victim = self.lowest_vram_page();
            let Some(id) = victim else {
                break;
            };
            self.migrate_to_dram(id, metrics)?;
        }
        Ok(())
    }

    fn lowest_vram_page(&self) -> Option<PageId> {
        self.pages
            .values()
            .filter(|p| p.tier == KvResidency::Vram)
            .min_by(|a, b| {
                a.priority(self.clock, 1.0, 0.1)
                    .partial_cmp(&b.priority(self.clock, 1.0, 0.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
    }

    fn refresh_metrics(&self, metrics: &Metrics) {
        let vram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::Vram)
            .count() as u64;
        let dram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::PinnedDram)
            .count() as u64;
        metrics
            .vram_pages
            .store(vram, std::sync::atomic::Ordering::Relaxed);
        metrics
            .dram_pages
            .store(dram, std::sync::atomic::Ordering::Relaxed);
        let nvme = self.nvme_count();
        metrics
            .kv_nvme_pages
            .store(nvme, std::sync::atomic::Ordering::Relaxed);
        if let Some(store) = self.nvme.as_ref() {
            metrics
                .kv_nvme_reads
                .store(store.reads(), std::sync::atomic::Ordering::Relaxed);
            metrics
                .kv_nvme_bytes
                .store(store.bytes(), std::sync::atomic::Ordering::Relaxed);
            metrics.kv_nvme_direct.store(
                u64::from(store.is_direct()),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        metrics.kv_nvme_prefetches.store(
            self.nvme_prefetches.load(std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
        let kv_dev: u64 = self.hip_pages.values().map(|p| p.bytes as u64).sum();
        metrics
            .kv_bytes_on_device
            .store(kv_dev, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn counts(&self) -> (u64, u64) {
        let vram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::Vram)
            .count() as u64;
        let dram = self
            .pages
            .values()
            .filter(|p| p.tier == KvResidency::PinnedDram)
            .count() as u64;
        (vram, dram)
    }

    pub fn nvme_count(&self) -> u64 {
        self.pages
            .values()
            .filter(|p| p.tier == KvResidency::Nvme)
            .count() as u64
    }

    pub fn nvme_reads(&self) -> u64 {
        self.nvme.as_ref().map(|s| s.reads()).unwrap_or(0)
    }

    pub fn nvme_bytes(&self) -> u64 {
        self.nvme.as_ref().map(|s| s.bytes()).unwrap_or(0)
    }

    pub fn nvme_direct(&self) -> bool {
        self.nvme.as_ref().map(|s| s.is_direct()).unwrap_or(false)
    }

    pub fn nvme_prefetches(&self) -> u64 {
        self.nvme_prefetches.load(Ordering::Relaxed)
    }
}

fn copy_f32(dst: &mut [f32], src: &[f32]) -> Result<(), AllpacaError> {
    let mut src_b = vec![0u8; src.len() * 4];
    let mut dst_b = vec![0u8; dst.len() * 4];
    for (i, v) in src.iter().enumerate() {
        src_b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    copy_async(&mut dst_b, &src_b)?;
    for i in 0..dst.len() {
        let mut b = [0u8; 4];
        b.copy_from_slice(&dst_b[i * 4..i * 4 + 4]);
        dst[i] = f32::from_le_bytes(b);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::Metrics;

    #[test]
    fn rewind_filled_drops_tail_slots() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::new(4096, 4096, 0.4);
        let id = kv.alloc_page(0, 0, 8, 1, 4, &metrics).unwrap();
        let kv_width = 4usize;
        let zero = vec![0.0f32; kv_width];
        for slot in 0..4u32 {
            kv.write_slot(id, slot, 1, 4, &zero, &zero).unwrap();
        }
        let (k, _) = kv.read_kv(id, 1, 4).unwrap();
        assert_eq!(k.len(), 16);
        kv.rewind_filled(&[vec![id]], 2, 8);
        let (k, _) = kv.read_kv(id, 1, 4).unwrap();
        assert_eq!(k.len(), 8, "seq_len=2 must leave two KV slots");
    }

    #[test]
    fn spills_to_dram_when_vram_hot_set_full() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::new(256, 4096, 0.4);
        let mut ids = Vec::new();
        for i in 0..8 {
            ids.push(kv.alloc_page(0, i, 2, 1, 4, &metrics).unwrap());
        }
        let (vram, dram) = kv.counts();
        assert!(dram > 0, "expected DRAM spill, vram={vram} dram={dram}");
        assert!(metrics.page_migrations.load(std::sync::atomic::Ordering::Relaxed) > 0);
    }

    #[test]
    fn shrinking_vram_budget_spills_hot_pages_to_dram() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::new(4096, 16 * 1024, 1.0);
        let mut ids = Vec::new();
        for i in 0..4 {
            ids.push(kv.alloc_page(0, i, 4, 1, 4, &metrics).unwrap());
        }
        let (vram_before, dram_before) = kv.counts();
        assert!(vram_before >= 4, "pages must start in VRAM; vram={vram_before}");
        assert_eq!(dram_before, 0);
        kv.set_vram_bytes(128, &metrics).unwrap();
        let (vram_after, dram_after) = kv.counts();
        assert!(
            dram_after > 0,
            "elastic shrink must spill KV to DRAM; vram={vram_after} dram={dram_after}"
        );
        assert!(vram_after < vram_before);
    }

    #[test]
    fn spills_to_nvme_when_dram_full_and_restores_bytes() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::new(256, 256, 0.4);
        kv.open_nvme(256 * 1024, None).unwrap();
        let mut ids = Vec::new();
        for i in 0..12u32 {
            ids.push(kv.alloc_page(0, i, 2, 1, 4, &metrics).unwrap());
        }
        let marker = vec![1.5f32, 2.5, 3.5, 4.5];
        kv.write_slot(ids[0], 0, 1, 4, &marker, &marker).unwrap();
        for i in 12..24u32 {
            ids.push(kv.alloc_page(0, i, 2, 1, 4, &metrics).unwrap());
        }
        assert!(
            kv.nvme_count() > 0,
            "DRAM overflow must spill to NVMe; counts={:?} nvme={}",
            kv.counts(),
            kv.nvme_count()
        );
        let (k, v) = kv.read_kv(ids[0], 1, 4).unwrap();
        assert_eq!(&k[..4], marker.as_slice(), "cold-tier read must keep K bytes");
        assert_eq!(&v[..4], marker.as_slice(), "cold-tier read must keep V bytes");
        kv.write_slot(ids[0], 1, 1, 4, &marker, &marker).unwrap();
        let (k, _) = kv.read_kv(ids[0], 1, 4).unwrap();
        assert_eq!(
            k.len(),
            8,
            "write after NVMe restore must keep both slots; filled len={}",
            k.len()
        );
        assert!(
            kv.nvme_reads() >= 1,
            "spilled pages must be read from the swap file"
        );
    }

    /// Packed device K/V rows written as page runs must match per-slot HIP copies.
    #[cfg(feature = "hip")]
    #[test]
    fn write_slots_from_device_matches_per_slot_on_hip() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let metrics = Metrics::default();
        let mut batched = KvTierManager::with_hip(1 << 20, 1 << 20, 0.9, Some(0));
        let mut per_slot = KvTierManager::with_hip(1 << 20, 1 << 20, 0.9, Some(0));
        let page_tokens = 8u32;
        let n_kv_heads = 2usize;
        let head_dim = 4usize;
        let kv_width = n_kv_heads * head_dim;
        let start_pos = 6usize;
        let n_tok = 4usize;
        let n_pages = 2u32;
        let mut batch_ids = Vec::new();
        let mut slot_ids = Vec::new();
        for i in 0..n_pages {
            batch_ids.push(
                batched
                    .alloc_page(0, i, page_tokens, n_kv_heads, head_dim, &metrics)
                    .unwrap(),
            );
            slot_ids.push(
                per_slot
                    .alloc_page(0, i, page_tokens, n_kv_heads, head_dim, &metrics)
                    .unwrap(),
            );
        }
        let mut k = vec![0.0f32; n_tok * kv_width];
        let mut v = vec![0.0f32; n_tok * kv_width];
        for (i, x) in k.iter_mut().enumerate() {
            *x = i as f32 * 0.01;
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = 1.0 + i as f32 * 0.01;
        }
        let k_bytes = k.len() * 4;
        let v_bytes = v.len() * 4;
        allpaca_hip::set_device(0).unwrap();
        let dk = allpaca_hip::device_malloc(k_bytes, 0).unwrap();
        let dv = allpaca_hip::device_malloc(v_bytes, 0).unwrap();
        allpaca_hip::memcpy(dk, k.as_ptr().cast(), k_bytes, allpaca_hip::MEMCPY_H2D).unwrap();
        allpaca_hip::memcpy(dv, v.as_ptr().cast(), v_bytes, allpaca_hip::MEMCPY_H2D).unwrap();
        batched
            .write_slots_from_device(
                &batch_ids,
                start_pos,
                n_tok,
                page_tokens as usize,
                n_kv_heads,
                head_dim,
                dk.cast(),
                dv.cast(),
            )
            .expect("batched KV write");
        for i in 0..n_tok {
            let pos = start_pos + i;
            let page_index = pos / page_tokens as usize;
            let slot = (pos % page_tokens as usize) as u32;
            per_slot
                .write_slot_from_device(
                    slot_ids[page_index],
                    slot,
                    n_kv_heads,
                    head_dim,
                    unsafe { dk.cast::<f32>().add(i * kv_width) },
                    unsafe { dv.cast::<f32>().add(i * kv_width) },
                )
                .expect("per-slot KV write");
        }
        allpaca_hip::device_free(dk, k_bytes);
        allpaca_hip::device_free(dv, v_bytes);
        for (a, b) in batch_ids.iter().zip(slot_ids.iter()) {
            let (bk, bv) = batched.read_kv(*a, n_kv_heads, head_dim).unwrap();
            let (sk, sv) = per_slot.read_kv(*b, n_kv_heads, head_dim).unwrap();
            assert_eq!(bk, sk, "K page mismatch batched vs per-slot");
            assert_eq!(bv, sv, "V page mismatch batched vs per-slot");
        }
    }
}
