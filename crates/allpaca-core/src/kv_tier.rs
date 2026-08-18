use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use allpaca_cpu::copy_async;
use allpaca_types::{
    AllpacaError, DeviceBuffer, DeviceBufferKind, FatalError, KvPage, KvResidency, PageId,
    TransientError,
};

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

/// Two-tier KV manager: VRAM (hot) <-> Pinned DRAM (warm). No NVMe in MVP.
/// When `hip_device` is set, VRAM pages are real `hipMalloc` buffers.
pub struct KvTierManager {
    pages: HashMap<PageId, KvPage>,
    vram: BufferPool,
    dram: BufferPool,
    hip_pages: HashMap<PageId, HipPageAlloc>,
    hip_device: Option<i32>,
    activation_h2d_bytes: AtomicU64,
    activation_d2h_bytes: AtomicU64,
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
            hip_pages: HashMap::new(),
            hip_device,
            activation_h2d_bytes: AtomicU64::new(0),
            activation_d2h_bytes: AtomicU64::new(0),
            next_id: 1,
            clock: 1,
            hot_fraction,
            vram_capacity: (vram_bytes / 4).max(1),
        }
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
                    let off = self.dram.alloc(len).map_err(|_| {
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
                    let off = self.dram.alloc(len).map_err(|_| {
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
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.filled = page.filled.max(slot + 1);
        }
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
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.filled = page.filled.max(slot + 1);
        }
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
            }
            self.touch(page_id, 1.0);
            i += run;
        }
        Ok(())
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
        let dst = self.dram.alloc(len)?;
        if page.location.kind == DeviceBufferKind::Hip {
            if let Some(alloc) = self.hip_pages.remove(&id) {
                allpaca_hip::memcpy(
                    self.dram.slice_mut(dst, len).as_mut_ptr().cast(),
                    alloc.ptr,
                    alloc.bytes,
                    allpaca_hip::MEMCPY_D2H,
                )?;
                allpaca_hip::device_free(alloc.ptr, alloc.bytes);
            }
        } else {
            copy_f32(
                self.dram.slice_mut(dst, len),
                self.vram.slice(page.location.offset, len),
            )?;
            self.vram.free(page.location.offset, len);
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

    fn vram_live_floats(&self) -> usize {
        if self.hip_device.is_some() {
            self.hip_pages.values().map(|p| p.bytes / 4).sum()
        } else {
            self.vram.live
        }
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
