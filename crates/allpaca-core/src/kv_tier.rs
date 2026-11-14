use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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

    /// Add capacity. Offsets stay valid; the buffer may move, so no pointer
    /// into it may be held across this call.
    fn grow(&mut self, extra_bytes: usize) {
        self.capacity += extra_bytes / 4;
        self.data.resize(self.capacity, 0.0);
    }
}

struct CudaPageAlloc {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for CudaPageAlloc {}
unsafe impl Sync for CudaPageAlloc {}

/// Three-tier KV manager: VRAM (hot) <-> Pinned DRAM (warm) <-> NVMe (cold).
/// NVMe is off until [`Self::open_nvme`]. When `hip_device` is set, VRAM pages
/// are real `hipMalloc` buffers. CUDA-only engines use `cudaMalloc` pages.
/// Vulkan-only engines use `VulkanKvPage` buffers.
pub struct KvTierManager {
    pages: HashMap<PageId, KvPage>,
    vram: BufferPool,
    dram: BufferPool,
    nvme: Option<NvmeKvStore>,
    hip_pages: HashMap<PageId, HipPageAlloc>,
    hip_device: Option<i32>,
    /// Per-layer HIP device. Empty uses `hip_device`. `-1` uses `hip_device`.
    layer_hip: Vec<i32>,
    cuda_pages: HashMap<PageId, CudaPageAlloc>,
    cuda_device: Option<i32>,
    vulkan_pages: HashMap<PageId, allpaca_vulkan::VulkanKvPage>,
    vulkan: Option<Arc<allpaca_vulkan::VulkanWorkspace>>,
    activation_h2d_bytes: AtomicU64,
    activation_d2h_bytes: AtomicU64,
    cache_d2h_bytes: AtomicU64,
    nvme_prefetches: AtomicU64,
    next_id: u64,
    clock: u64,
    hot_fraction: f32,
    vram_capacity: usize,
    /// Latent sparse attention index of a sequence's layer, keyed by the
    /// layer's first page (`experimental.lsa`). Derived from the cached
    /// keys: dropped with the page, rebuilt from the pages when missing.
    lsa: HashMap<PageId, crate::lsa::LsaIndex>,
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
        Self::with_devices(vram_bytes, dram_bytes, hot_fraction, hip_device, None, None)
    }

    pub fn with_devices(
        vram_bytes: usize,
        dram_bytes: usize,
        hot_fraction: f32,
        hip_device: Option<i32>,
        cuda_device: Option<i32>,
        vulkan: Option<Arc<allpaca_vulkan::VulkanWorkspace>>,
    ) -> Self {
        Self {
            pages: HashMap::new(),
            vram: BufferPool::new(vram_bytes),
            dram: BufferPool::new(dram_bytes),
            nvme: None,
            hip_pages: HashMap::new(),
            hip_device,
            layer_hip: Vec::new(),
            cuda_pages: HashMap::new(),
            cuda_device: if hip_device.is_some() {
                None
            } else {
                cuda_device
            },
            vulkan_pages: HashMap::new(),
            vulkan: if hip_device.is_some() || cuda_device.is_some() {
                None
            } else {
                vulkan
            },
            activation_h2d_bytes: AtomicU64::new(0),
            activation_d2h_bytes: AtomicU64::new(0),
            cache_d2h_bytes: AtomicU64::new(0),
            nvme_prefetches: AtomicU64::new(0),
            next_id: 1,
            clock: 1,
            hot_fraction,
            vram_capacity: (vram_bytes / 4).max(1),
            lsa: HashMap::new(),
        }
    }

    /// Take the LSA index kept under `first_page` (a sequence layer's first
    /// page), if any. Hand it back with [`Self::lsa_put`].
    pub(crate) fn lsa_take(&mut self, first_page: PageId) -> Option<crate::lsa::LsaIndex> {
        self.lsa.remove(&first_page)
    }

    pub(crate) fn lsa_put(&mut self, first_page: PageId, index: crate::lsa::LsaIndex) {
        if self.pages.contains_key(&first_page) {
            self.lsa.insert(first_page, index);
        }
    }

    /// LSA indexes held (one per sequence layer that went sparse).
    pub fn lsa_indexes(&self) -> usize {
        self.lsa.len()
    }

    /// Key rows `first..first + n` of a sequence layer (`pages` in order),
    /// read back to the host.
    pub(crate) fn lsa_read_keys(
        &self,
        pages: &[PageId],
        first: usize,
        n: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>, AllpacaError> {
        let mut out = Vec::with_capacity(n * n_kv_heads * head_dim);
        let mut page_start = 0usize;
        for id in pages {
            let cap = self.pages.get(id).map_or(0, |p| p.capacity_tokens as usize);
            let (lo, hi) = (first.max(page_start), (first + n).min(page_start + cap));
            if lo < hi {
                let (k, _) = self.read_kv_slot_range(
                    *id,
                    (lo - page_start) as u32,
                    (hi - lo) as u32,
                    n_kv_heads,
                    head_dim,
                    &self.activation_d2h_bytes,
                )?;
                out.extend_from_slice(&k);
            }
            page_start += cap;
        }
        if out.len() != n * n_kv_heads * head_dim {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "lsa: keys {first}..{} are not all cached ({} floats read)",
                first + n,
                out.len()
            ))));
        }
        Ok(out)
    }

    /// The device address of each page's key rows and the rows a page
    /// holds, for a layer's `pages`. Every page must be resident on the HIP
    /// device: a page in host memory is refused, because which pages are
    /// there depends on memory pressure and what a row reads must not.
    fn lsa_page_table(&self, pages: &[PageId]) -> Result<(Vec<u64>, usize), AllpacaError> {
        let mut table = Vec::with_capacity(pages.len());
        let mut page_rows = 0usize;
        for id in pages {
            let page = self.pages.get(id).ok_or_else(|| {
                AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
            })?;
            let cap = page.capacity_tokens as usize;
            if page_rows == 0 {
                page_rows = cap;
            }
            let alloc = self.hip_pages.get(id).filter(|_| {
                page.location.kind == DeviceBufferKind::Hip && cap == page_rows && cap > 0
            });
            let Some(alloc) = alloc else {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "experimental.lsa needs every KV page of a layer on its GPU, and page {id} is {:?} (raise kv.vram_bytes, or lower the context)",
                    page.location.kind
                ))));
            };
            // SAFETY: the page's key rows start `offset` floats into its buffer.
            table.push(unsafe { alloc.ptr.add(page.location.offset * 4) } as u64);
        }
        Ok((table, page_rows))
    }

    /// Gathers the key and value rows `keys` (positions in the sequence) of
    /// a layer's `pages` into `dst_k` / `dst_v` on the card, in the order
    /// given.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lsa_gather_device(
        &self,
        hip: &allpaca_hip::HipWorkspace,
        pages: &[PageId],
        n_kv_heads: usize,
        head_dim: usize,
        keys: &[u32],
        dst_k: *mut f32,
        dst_v: *mut f32,
    ) -> Result<(), AllpacaError> {
        let kv_width = n_kv_heads * head_dim;
        let (table, page_rows) = self.lsa_page_table(pages)?;
        hip.gather_paged_kv(dst_k, dst_v, &table, page_rows * kv_width, keys, kv_width, page_rows)
    }

    /// Attention of the query rows at `q` (`q_stride` floats apart) that
    /// each read their own keys `lists[row]` of a layer's `pages`, into
    /// `out` (`out_stride` apart). A row's bits are those of
    /// [`Self::lsa_gather_device`] and one attention over what it gathered.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lsa_attend_rows_device(
        &self,
        hip: &allpaca_hip::HipWorkspace,
        pages: &[PageId],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        lists: &[&[u32]],
        q: *const f32,
        q_stride: usize,
        out: *mut f32,
        out_stride: usize,
    ) -> Result<(), AllpacaError> {
        let (table, page_rows) = self.lsa_page_table(pages)?;
        hip.attention_indexed_device(
            q,
            q_stride,
            out,
            out_stride,
            &table,
            page_rows * n_kv_heads * head_dim,
            page_rows,
            lists,
            n_heads,
            n_kv_heads,
            head_dim,
        )
    }

    /// Place later KV pages for `layer` on that HIP device.
    pub fn set_layer_devices(&mut self, devices: &[i32]) {
        self.layer_hip = devices.to_vec();
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
        let hip_dev = if self.hip_device.is_some() {
            self.layer_hip
                .get(layer as usize)
                .copied()
                .filter(|d| *d >= 0)
                .or(self.hip_device)
        } else {
            None
        };
        let (tier, offset, kind, device_id) = if let Some(dev) = hip_dev {
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
                    (
                        KvResidency::PinnedDram,
                        off,
                        DeviceBufferKind::CpuPinnedDram,
                        -1,
                    )
                }
            }
        } else if let Some(dev) = self.cuda_device {
            let bytes = len * 4;
            match allpaca_cuda::device_malloc(bytes, dev) {
                Ok(ptr) => {
                    self.cuda_pages.insert(
                        id,
                        CudaPageAlloc {
                            ptr: ptr.cast(),
                            bytes,
                        },
                    );
                    (KvResidency::Vram, 0, DeviceBufferKind::Cuda, dev)
                }
                Err(_) => {
                    let off = self.dram_alloc(len, metrics).map_err(|_| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(
                            "CUDA KV alloc failed and pinned DRAM is full".into(),
                        ))
                    })?;
                    (
                        KvResidency::PinnedDram,
                        off,
                        DeviceBufferKind::CpuPinnedDram,
                        -1,
                    )
                }
            }
        } else if let Some(ws) = self.vulkan.as_ref() {
            let bytes = len * 4;
            match ws.alloc_kv_page(bytes) {
                Ok(page) => {
                    let dev = ws.device;
                    self.vulkan_pages.insert(id, page);
                    (KvResidency::Vram, 0, DeviceBufferKind::Vulkan, dev)
                }
                Err(_) => {
                    let off = self.dram_alloc(len, metrics).map_err(|_| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(
                            "Vulkan KV alloc failed and pinned DRAM is full".into(),
                        ))
                    })?;
                    (
                        KvResidency::PinnedDram,
                        off,
                        DeviceBufferKind::CpuPinnedDram,
                        -1,
                    )
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
                    (
                        KvResidency::PinnedDram,
                        off,
                        DeviceBufferKind::CpuPinnedDram,
                        -1,
                    )
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
            DeviceBufferKind::Cuda => {
                if let Some(alloc) = self.cuda_pages.remove(&id) {
                    allpaca_cuda::device_free(alloc.ptr.cast());
                }
            }
            DeviceBufferKind::Vulkan => {
                let _ = self.vulkan_pages.remove(&id);
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
        self.lsa.remove(&id);
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
        self.write_slots(id, slot, n_kv_heads, head_dim, k, v)
    }

    /// Write consecutive rows from `first_slot` on: `k` and `v` hold one row
    /// of `n_kv_heads * head_dim` values each after another. A page keeps
    /// its rows' keys together and their values together, so this is one copy
    /// for each, where a device page took two synchronous copies per row.
    pub fn write_slots(
        &mut self,
        id: PageId,
        first_slot: u32,
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
        let n = if kv_width == 0 { 0 } else { k.len() / kv_width };
        if k.len() != n * kv_width || v.len() != k.len() || first_slot as usize + n > page.capacity_tokens as usize {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "KV page {id}: {} key / {} value floats from slot {first_slot} do not fit {} slots of {kv_width}",
                k.len(),
                v.len(),
                page.capacity_tokens
            ))));
        }
        if n == 0 {
            return Ok(());
        }
        let width = n * kv_width;
        let k_off = page.location.offset + first_slot as usize * kv_width;
        let v_off = page.location.offset
            + page.capacity_tokens as usize * kv_width
            + first_slot as usize * kv_width;
        match page.location.kind {
            DeviceBufferKind::Hip => {
                let alloc = self.hip_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "HIP KV page {id} missing device buffer"
                    )))
                })?;
                let k_bytes = width * 4;
                let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                allpaca_hip::memcpy(k_ptr, k.as_ptr().cast(), k_bytes, allpaca_hip::MEMCPY_H2D)?;
                allpaca_hip::memcpy(v_ptr, v.as_ptr().cast(), k_bytes, allpaca_hip::MEMCPY_H2D)?;
                self.activation_h2d_bytes
                    .fetch_add((k_bytes * 2) as u64, Ordering::Relaxed);
            }
            DeviceBufferKind::Cuda => {
                let alloc = self.cuda_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "CUDA KV page {id} missing device buffer"
                    )))
                })?;
                let k_bytes = width * 4;
                let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                allpaca_cuda::memcpy(
                    k_ptr.cast(),
                    k.as_ptr().cast(),
                    k_bytes,
                    allpaca_cuda::MEMCPY_H2D,
                )?;
                allpaca_cuda::memcpy(
                    v_ptr.cast(),
                    v.as_ptr().cast(),
                    k_bytes,
                    allpaca_cuda::MEMCPY_H2D,
                )?;
                self.activation_h2d_bytes
                    .fetch_add((k_bytes * 2) as u64, Ordering::Relaxed);
            }
            DeviceBufferKind::Vulkan => {
                let page_buf = self.vulkan_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "Vulkan KV page {id} missing device buffer"
                    )))
                })?;
                let ws = self.vulkan.as_ref().ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(
                        "Vulkan KV page with no workspace".into(),
                    ))
                })?;
                ws.kv_write_f32(page_buf, k_off, k)?;
                ws.kv_write_f32(page_buf, v_off, v)?;
                self.activation_h2d_bytes
                    .fetch_add((k.len() * 8) as u64, Ordering::Relaxed);
            }
            DeviceBufferKind::CpuVram => {
                self.vram.slice_mut(k_off, width).copy_from_slice(k);
                self.vram.slice_mut(v_off, width).copy_from_slice(v);
            }
            DeviceBufferKind::CpuPinnedDram => {
                self.dram.slice_mut(k_off, width).copy_from_slice(k);
                self.dram.slice_mut(v_off, width).copy_from_slice(v);
            }
            DeviceBufferKind::Nvme => {
                return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                    "NVMe KV page must be restored to DRAM before write".into(),
                )));
            }
        }
        if let Some(page) = self.pages.get_mut(&id) {
            page.filled = page.filled.max(first_slot + n as u32);
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
        self.read_kv_counted(id, n_kv_heads, head_dim, &self.activation_d2h_bytes)
    }

    /// [`Self::read_kv`] for a prompt-cache checkpoint: counted apart from
    /// the forward's activation traffic.
    pub fn read_kv_for_cache(
        &self,
        id: PageId,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
        self.read_kv_counted(id, n_kv_heads, head_dim, &self.cache_d2h_bytes)
    }

    /// `n_slots` rows from `first_slot`, not the rest of the page.
    ///
    /// Counted on [`Self::cache_d2h_bytes`] for every residency, including
    /// host pages, so a window checkpoint's cost stays the window.
    pub fn read_kv_slots_for_cache(
        &self,
        id: PageId,
        first_slot: u32,
        n_slots: u32,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
        self.read_kv_slot_range(
            id,
            first_slot,
            n_slots,
            n_kv_heads,
            head_dim,
            &self.cache_d2h_bytes,
        )
    }

    fn read_kv_slot_range(
        &self,
        id: PageId,
        first_slot: u32,
        n_slots: u32,
        n_kv_heads: usize,
        head_dim: usize,
        counter: &AtomicU64,
    ) -> Result<(Vec<f32>, Vec<f32>), AllpacaError> {
        let page = self.pages.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory(format!("unknown KV page {id}")))
        })?;
        let kv_width = n_kv_heads.saturating_mul(head_dim);
        let n_slots = n_slots as usize;
        let first = first_slot as usize;
        if kv_width == 0 || n_slots == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let filled = page.filled as usize;
        let cap = page.capacity_tokens as usize;
        let end = first.saturating_add(n_slots);
        if end > filled || end > cap {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                "kv page {id}: slots {first}+{n_slots} past filled {filled}"
            ))));
        }
        let n = n_slots.saturating_mul(kv_width);
        let k_at = first.saturating_mul(kv_width);
        let k_off = page.location.offset.saturating_add(k_at);
        let v_off = page
            .location
            .offset
            .saturating_add(cap.saturating_mul(kv_width))
            .saturating_add(k_at);
        let kind = page.location.kind;
        let page_off = page.location.offset;
        let page_len = page.location.len;
        let (k, v) = match kind {
            DeviceBufferKind::Hip => {
                let alloc = self.hip_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "HIP KV page {id} missing device buffer"
                    )))
                })?;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                let bytes = n.saturating_mul(4);
                // SAFETY: `alloc.ptr` is this page's device buffer of `alloc.bytes`.
                // `k_off` and `v_off` are float indexes of `n` f32s inside it:
                // the caller rejected a span past `filled` and `capacity_tokens`.
                let k_ptr = unsafe { alloc.ptr.add(k_off.saturating_mul(4)) };
                let v_ptr = unsafe { alloc.ptr.add(v_off.saturating_mul(4)) };
                allpaca_hip::memcpy(k.as_mut_ptr().cast(), k_ptr, bytes, allpaca_hip::MEMCPY_D2H)?;
                allpaca_hip::memcpy(v.as_mut_ptr().cast(), v_ptr, bytes, allpaca_hip::MEMCPY_D2H)?;
                (k, v)
            }
            DeviceBufferKind::Cuda => {
                let alloc = self.cuda_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "CUDA KV page {id} missing device buffer"
                    )))
                })?;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                let bytes = n.saturating_mul(4);
                // SAFETY: same span invariant as the HIP arm, for this page's cudaMalloc.
                let k_ptr = unsafe { alloc.ptr.add(k_off.saturating_mul(4)) };
                let v_ptr = unsafe { alloc.ptr.add(v_off.saturating_mul(4)) };
                allpaca_cuda::memcpy(
                    k.as_mut_ptr().cast(),
                    k_ptr.cast(),
                    bytes,
                    allpaca_cuda::MEMCPY_D2H,
                )?;
                allpaca_cuda::memcpy(
                    v.as_mut_ptr().cast(),
                    v_ptr.cast(),
                    bytes,
                    allpaca_cuda::MEMCPY_D2H,
                )?;
                (k, v)
            }
            DeviceBufferKind::Vulkan => {
                let page_buf = self.vulkan_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "Vulkan KV page {id} missing device buffer"
                    )))
                })?;
                let ws = self.vulkan.as_ref().ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(
                        "Vulkan KV page with no workspace".into(),
                    ))
                })?;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                ws.kv_read_f32(page_buf, k_off, &mut k)?;
                ws.kv_read_f32(page_buf, v_off, &mut v)?;
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
                let data = store.read_page(page_off, page_len)?;
                let v_at = cap.saturating_mul(kv_width).saturating_add(k_at);
                let k = data.get(k_at..k_at + n).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV K slice".into()))
                })?;
                let v = data.get(v_at..v_at + n).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV V slice".into()))
                })?;
                (k.to_vec(), v.to_vec())
            }
        };
        counter.fetch_add(n.saturating_mul(8) as u64, Ordering::Relaxed);
        Ok((k, v))
    }

    /// Device-to-host bytes read for prompt-cache checkpoints.
    pub fn cache_d2h_bytes(&self) -> u64 {
        self.cache_d2h_bytes.load(Ordering::Relaxed)
    }

    fn read_kv_counted(
        &self,
        id: PageId,
        n_kv_heads: usize,
        head_dim: usize,
        counter: &AtomicU64,
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
                counter.fetch_add((n * 8) as u64, Ordering::Relaxed);
                (k, v)
            }
            DeviceBufferKind::Cuda => {
                let alloc = self.cuda_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "CUDA KV page {id} missing device buffer"
                    )))
                })?;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                allpaca_cuda::memcpy(
                    k.as_mut_ptr().cast(),
                    unsafe { alloc.ptr.add(k_off * 4) }.cast(),
                    n * 4,
                    allpaca_cuda::MEMCPY_D2H,
                )?;
                allpaca_cuda::memcpy(
                    v.as_mut_ptr().cast(),
                    unsafe { alloc.ptr.add(v_off * 4) }.cast(),
                    n * 4,
                    allpaca_cuda::MEMCPY_D2H,
                )?;
                counter.fetch_add((n * 8) as u64, Ordering::Relaxed);
                (k, v)
            }
            DeviceBufferKind::Vulkan => {
                let page_buf = self.vulkan_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "Vulkan KV page {id} missing device buffer"
                    )))
                })?;
                let ws = self.vulkan.as_ref().ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(
                        "Vulkan KV page with no workspace".into(),
                    ))
                })?;
                let mut k = vec![0.0f32; n];
                let mut v = vec![0.0f32; n];
                ws.kv_read_f32(page_buf, k_off, &mut k)?;
                ws.kv_read_f32(page_buf, v_off, &mut v)?;
                counter.fetch_add((n * 8) as u64, Ordering::Relaxed);
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
    ///
    /// `cuda_stream`: when `Some`, CUDA copies enqueue on that stream so they
    /// order with prior workspace GEMM/RoPE. `None` uses the default stream.
    pub(crate) fn write_slot_from_device(
        &mut self,
        id: PageId,
        slot: u32,
        n_kv_heads: usize,
        head_dim: usize,
        k_dev: *const f32,
        v_dev: *const f32,
        cuda_stream: Option<*mut std::ffi::c_void>,
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
            DeviceBufferKind::Cuda => {
                let alloc = self.cuda_pages.get(&id).ok_or_else(|| {
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "CUDA KV page {id} missing device buffer"
                    )))
                })?;
                let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                let stream = cuda_stream.unwrap_or(std::ptr::null_mut());
                allpaca_cuda::memcpy_on(
                    k_ptr.cast(),
                    k_dev.cast(),
                    k_bytes,
                    allpaca_cuda::MEMCPY_D2D,
                    stream,
                )?;
                allpaca_cuda::memcpy_on(
                    v_ptr.cast(),
                    v_dev.cast(),
                    k_bytes,
                    allpaca_cuda::MEMCPY_D2D,
                    stream,
                )?;
            }
            DeviceBufferKind::Vulkan => {
                return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                    "Vulkan KV D2D from raw device pointers is not wired".into(),
                )));
            }
            DeviceBufferKind::CpuVram | DeviceBufferKind::CpuPinnedDram => {
                let mut k = vec![0.0f32; kv_width];
                let mut v = vec![0.0f32; kv_width];
                if self.cuda_device.is_some() {
                    let stream = cuda_stream.unwrap_or(std::ptr::null_mut());
                    allpaca_cuda::memcpy_on(
                        k.as_mut_ptr().cast(),
                        k_dev.cast(),
                        k_bytes,
                        allpaca_cuda::MEMCPY_D2H,
                        stream,
                    )?;
                    allpaca_cuda::memcpy_on(
                        v.as_mut_ptr().cast(),
                        v_dev.cast(),
                        k_bytes,
                        allpaca_cuda::MEMCPY_D2H,
                        stream,
                    )?;
                    if !stream.is_null() {
                        // Host needs the bytes before write_slot; wait this stream only.
                        unsafe { allpaca_cuda::stream_synchronize_on(stream)? };
                    }
                } else {
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
                }
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
        cuda_stream: Option<*mut std::ffi::c_void>,
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
                    AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                        "unknown KV page {page_id}"
                    )))
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
                DeviceBufferKind::Cuda => {
                    let k_off = offset + slot * kv_width;
                    let v_off = offset + cap * kv_width + slot * kv_width;
                    let alloc = self.cuda_pages.get(&page_id).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                            "CUDA KV page {page_id} missing device buffer"
                        )))
                    })?;
                    let run_bytes = run * kv_width * 4;
                    let k_ptr = unsafe { alloc.ptr.add(k_off * 4) };
                    let v_ptr = unsafe { alloc.ptr.add(v_off * 4) };
                    let k_src = unsafe { k_dev.add(i * kv_width) };
                    let v_src = unsafe { v_dev.add(i * kv_width) };
                    let stream = cuda_stream.unwrap_or(std::ptr::null_mut());
                    allpaca_cuda::memcpy_on(
                        k_ptr.cast(),
                        k_src.cast(),
                        run_bytes,
                        allpaca_cuda::MEMCPY_D2D,
                        stream,
                    )?;
                    allpaca_cuda::memcpy_on(
                        v_ptr.cast(),
                        v_src.cast(),
                        run_bytes,
                        allpaca_cuda::MEMCPY_D2D,
                        stream,
                    )?;
                    if let Some(page) = self.pages.get_mut(&page_id) {
                        page.filled = page.filled.max((slot + run) as u32);
                    }
                }
                DeviceBufferKind::Vulkan => {
                    return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                        "Vulkan KV D2D from raw device pointers is not wired".into(),
                    )));
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
                            cuda_stream,
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
                if filled > 0 && filled <= max_kv && page.location.kind == DeviceBufferKind::Hip {
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

    /// Device K/V for CUDA attention. One resident CUDA page is in-place;
    /// multiple or host pages pack into `dest_k` / `dest_v` (D2D / H2D).
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_kv_views(
        &self,
        pages: &[u64],
        n_kv_heads: usize,
        head_dim: usize,
        dest_k: *mut f32,
        dest_v: *mut f32,
        max_kv: usize,
    ) -> Result<(*const f32, *const f32, usize), AllpacaError> {
        if pages.len() == 1 {
            if let Some((k, v, n)) = self.cuda_single_page_view(pages, n_kv_heads, head_dim) {
                if n > 0 && n <= max_kv {
                    return Ok((k, v, n));
                }
            }
        }
        let n_kv = self.pack_kv_cuda(pages, n_kv_heads, head_dim, dest_k, dest_v, max_kv)?;
        Ok((dest_k as *const f32, dest_v as *const f32, n_kv))
    }

    /// In-place CUDA K/V for a single resident page (no D2H pack).
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_single_page_view(
        &self,
        pages: &[u64],
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Option<(*const f32, *const f32, usize)> {
        if pages.len() != 1 {
            return None;
        }
        let id = pages[0];
        let page = self.pages.get(&id)?;
        if page.location.kind != DeviceBufferKind::Cuda || page.filled == 0 {
            return None;
        }
        let alloc = self.cuda_pages.get(&id)?;
        let kv_width = n_kv_heads * head_dim;
        let k_off = page.location.offset;
        let v_off = page.location.offset + page.capacity_tokens as usize * kv_width;
        let k = unsafe { alloc.ptr.add(k_off * 4).cast::<f32>() };
        let v = unsafe { alloc.ptr.add(v_off * 4).cast::<f32>() };
        Some((k, v, page.filled as usize))
    }

    /// Pack filled tokens into CUDA device K/V (D2D for CUDA pages, H2D for host).
    #[cfg(feature = "cuda")]
    pub(crate) fn pack_kv_cuda(
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
                "packed CUDA KV destination is empty".into(),
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
                    "packed CUDA KV exceeds workspace".into(),
                )));
            }
            let n = filled * kv_width;
            let k_off = page.location.offset;
            let v_off = page.location.offset + page.capacity_tokens as usize * kv_width;
            let dk = unsafe { dest_k.add(n_kv * kv_width).cast::<u8>() };
            let dv = unsafe { dest_v.add(n_kv * kv_width).cast::<u8>() };
            match page.location.kind {
                DeviceBufferKind::Cuda => {
                    let alloc = self.cuda_pages.get(id).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                            "CUDA KV page {id} missing device buffer"
                        )))
                    })?;
                    allpaca_cuda::memcpy(
                        dk.cast(),
                        unsafe { alloc.ptr.add(k_off * 4) }.cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_D2D,
                    )?;
                    allpaca_cuda::memcpy(
                        dv.cast(),
                        unsafe { alloc.ptr.add(v_off * 4) }.cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_D2D,
                    )?;
                }
                DeviceBufferKind::CpuVram => {
                    allpaca_cuda::memcpy(
                        dk.cast(),
                        self.vram.slice(k_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_H2D,
                    )?;
                    allpaca_cuda::memcpy(
                        dv.cast(),
                        self.vram.slice(v_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_H2D,
                    )?;
                    self.activation_h2d_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                }
                DeviceBufferKind::CpuPinnedDram => {
                    allpaca_cuda::memcpy(
                        dk.cast(),
                        self.dram.slice(k_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_H2D,
                    )?;
                    allpaca_cuda::memcpy(
                        dv.cast(),
                        self.dram.slice(v_off, n).as_ptr().cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_H2D,
                    )?;
                    self.activation_h2d_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                }
                DeviceBufferKind::Hip | DeviceBufferKind::Nvme => {
                    return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                        "CUDA KV pack cannot read HIP/NVMe pages".into(),
                    )));
                }
                DeviceBufferKind::Vulkan => {
                    return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                        "CUDA KV pack cannot read Vulkan pages".into(),
                    )));
                }
            }
            n_kv += filled;
        }
        Ok(n_kv)
    }

    /// In-place Vulkan K/V for a single resident page (no D2H pack).
    #[cfg(feature = "vulkan")]
    pub(crate) fn vulkan_single_page_view(
        &self,
        pages: &[u64],
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Option<(
        allpaca_vulkan::DeviceSlice,
        allpaca_vulkan::DeviceSlice,
        usize,
    )> {
        if pages.len() != 1 {
            return None;
        }
        let id = pages[0];
        let page = self.pages.get(&id)?;
        if page.location.kind != DeviceBufferKind::Vulkan || page.filled == 0 {
            return None;
        }
        let page_buf = self.vulkan_pages.get(&id)?;
        let ws = self.vulkan.as_ref()?;
        let kv_width = n_kv_heads * head_dim;
        let k_off = page.location.offset;
        let v_off = page.location.offset + page.capacity_tokens as usize * kv_width;
        let n = page.filled as usize * kv_width;
        let k = ws.kv_slice(page_buf, k_off, n).ok()?;
        let v = ws.kv_slice(page_buf, v_off, n).ok()?;
        Some((k, v, page.filled as usize))
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
                DeviceBufferKind::Cuda => {
                    let alloc = self.cuda_pages.get(id).ok_or_else(|| {
                        AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                            "CUDA KV page {id} missing device buffer"
                        )))
                    })?;
                    let mut k = vec![0.0f32; n];
                    let mut v = vec![0.0f32; n];
                    allpaca_cuda::memcpy(
                        k.as_mut_ptr().cast(),
                        unsafe { alloc.ptr.add(k_off * 4) }.cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_D2H,
                    )?;
                    allpaca_cuda::memcpy(
                        v.as_mut_ptr().cast(),
                        unsafe { alloc.ptr.add(v_off * 4) }.cast(),
                        n * 4,
                        allpaca_cuda::MEMCPY_D2H,
                    )?;
                    allpaca_hip::memcpy(dk, k.as_ptr().cast(), n * 4, allpaca_hip::MEMCPY_H2D)?;
                    allpaca_hip::memcpy(dv, v.as_ptr().cast(), n * 4, allpaca_hip::MEMCPY_H2D)?;
                    self.activation_h2d_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                    self.activation_d2h_bytes
                        .fetch_add((n * 8) as u64, Ordering::Relaxed);
                }
                DeviceBufferKind::Vulkan => {
                    return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                        "Vulkan KV pack into raw device pointers is not wired".into(),
                    )));
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
            DeviceBufferKind::Cuda => {
                if let Some(alloc) = self.cuda_pages.remove(&id) {
                    allpaca_cuda::memcpy(
                        self.dram.slice_mut(dst, len).as_mut_ptr().cast(),
                        alloc.ptr.cast(),
                        alloc.bytes,
                        allpaca_cuda::MEMCPY_D2H,
                    )?;
                    allpaca_cuda::device_free(alloc.ptr.cast());
                }
            }
            DeviceBufferKind::Vulkan => {
                if let Some(page_buf) = self.vulkan_pages.remove(&id) {
                    if let Some(ws) = self.vulkan.clone() {
                        ws.kv_read_f32(&page_buf, 0, self.dram.slice_mut(dst, len))?;
                    }
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
        } else if let Some(dev) = self.cuda_device {
            let bytes = len * 4;
            let ptr = allpaca_cuda::device_malloc(bytes, dev)?;
            allpaca_cuda::memcpy(
                ptr,
                self.dram.slice(page.location.offset, len).as_ptr().cast(),
                bytes,
                allpaca_cuda::MEMCPY_H2D,
            )?;
            self.dram.free(page.location.offset, len);
            self.cuda_pages.insert(
                id,
                CudaPageAlloc {
                    ptr: ptr.cast(),
                    bytes,
                },
            );
            if let Some(page) = self.pages.get_mut(&id) {
                page.tier = KvResidency::Vram;
                page.location.kind = DeviceBufferKind::Cuda;
                page.location.device_id = dev;
                page.location.offset = 0;
            }
        } else if let Some(ws) = self.vulkan.clone() {
            let bytes = len * 4;
            let page_buf = ws.alloc_kv_page(bytes)?;
            ws.kv_write_f32(&page_buf, 0, self.dram.slice(page.location.offset, len))?;
            self.dram.free(page.location.offset, len);
            let dev = ws.device;
            self.vulkan_pages.insert(id, page_buf);
            if let Some(page) = self.pages.get_mut(&id) {
                page.tier = KvResidency::Vram;
                page.location.kind = DeviceBufferKind::Vulkan;
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
        let Some(next_id) = self
            .pages
            .values()
            .find(|p| {
                p.layer == page.layer
                    && p.page_index == want
                    && p.tier == KvResidency::Nvme
                    && p.id != id
            })
            .map(|p| p.id)
        else {
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
        } else if self.cuda_device.is_some() {
            self.cuda_pages.values().map(|p| p.bytes / 4).sum()
        } else if self.vulkan.is_some() {
            self.vulkan_pages.values().map(|p| p.bytes() / 4).sum()
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

    /// Give the DRAM tier `extra_bytes` more room, for KV that a lower VRAM
    /// budget no longer holds, so the context a request can use stays the same.
    pub fn grow_dram_bytes(&mut self, extra_bytes: usize) {
        self.dram.grow(extra_bytes);
    }

    /// Bytes the DRAM tier can hold.
    pub fn dram_capacity_bytes(&self) -> usize {
        self.dram.capacity * 4
    }

    /// Bytes of KV that stay in VRAM before pages move to DRAM: the VRAM
    /// budget times `hot_fraction`.
    pub fn vram_hot_bytes(&self) -> usize {
        (((self.vram_capacity as f32) * self.hot_fraction) as usize) * 4
    }

    /// Spill lowest-priority VRAM KV pages to DRAM until `want_bytes` are freed
    /// (attention-trim / H2O-style cold shrink for expert VRAM headroom).
    /// Returns bytes actually migrated. Capacity is left unchanged — this is a
    /// soft trim of live residency, not a permanent budget cut.
    pub fn trim_cold_vram_bytes(
        &mut self,
        want_bytes: usize,
        metrics: &Metrics,
    ) -> Result<usize, AllpacaError> {
        if want_bytes == 0 {
            return Ok(0);
        }
        let mut freed = 0usize;
        while freed < want_bytes {
            let Some(id) = self.lowest_vram_page() else {
                break;
            };
            let page_bytes = self
                .pages
                .get(&id)
                .map(|p| p.location.len.saturating_mul(4))
                .unwrap_or(0);
            if page_bytes == 0 {
                break;
            }
            self.migrate_to_dram(id, metrics)?;
            freed = freed.saturating_add(page_bytes);
            metrics
                .kv_attention_trims
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.refresh_metrics(metrics);
        Ok(freed)
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
            self.nvme_prefetches
                .load(std::sync::atomic::Ordering::Relaxed),
            std::sync::atomic::Ordering::Relaxed,
        );
        let kv_dev: u64 = self.hip_pages.values().map(|p| p.bytes as u64).sum::<u64>()
            + self
                .cuda_pages
                .values()
                .map(|p| p.bytes as u64)
                .sum::<u64>()
            + self
                .vulkan_pages
                .values()
                .map(|p| p.bytes() as u64)
                .sum::<u64>();
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

    /// Rows written together land where rows written one at a time do.
    fn write_slots_matches_row_by_row_on(mut kv: KvTierManager) {
        let metrics = Metrics::default();
        let (heads, head_dim) = (2usize, 4usize);
        let width = heads * head_dim;
        let rows = 5usize;
        let k: Vec<f32> = (0..rows * width).map(|i| i as f32 * 0.5).collect();
        let v: Vec<f32> = (0..rows * width).map(|i| -(i as f32) - 1.0).collect();
        let together = kv.alloc_page(0, 0, 8, heads, head_dim, &metrics).unwrap();
        let one_by_one = kv.alloc_page(0, 1, 8, heads, head_dim, &metrics).unwrap();
        kv.write_slots(together, 2, heads, head_dim, &k, &v).unwrap();
        let zero = vec![0.0f32; width];
        for slot in 0..2u32 {
            kv.write_slot(one_by_one, slot, heads, head_dim, &zero, &zero).unwrap();
        }
        for r in 0..rows {
            let span = r * width..(r + 1) * width;
            kv.write_slot(one_by_one, 2 + r as u32, heads, head_dim, &k[span.clone()], &v[span]).unwrap();
        }
        let a = kv.read_kv(together, heads, head_dim).unwrap();
        let b = kv.read_kv(one_by_one, heads, head_dim).unwrap();
        assert_eq!(a.0.len(), 7 * width, "filled up to slot 7");
        assert_eq!(a.0[2 * width..], b.0[2 * width..]);
        assert_eq!(a.1[2 * width..], b.1[2 * width..]);
        assert!(
            kv.write_slots(together, 6, heads, head_dim, &k, &v).is_err(),
            "5 rows from slot 6 overflow an 8-slot page"
        );
    }

    #[test]
    fn write_slots_matches_row_by_row() {
        write_slots_matches_row_by_row_on(KvTierManager::new(1 << 20, 1 << 20, 0.9));
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_write_slots_matches_row_by_row() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        write_slots_matches_row_by_row_on(KvTierManager::with_hip(1 << 20, 1 << 20, 0.9, Some(0)));
    }

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
    fn cuda_kv_alloc_write_read_or_dram_fallback() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::with_devices(1 << 20, 1 << 20, 0.9, None, Some(0), None);
        let id = kv.alloc_page(0, 0, 8, 1, 4, &metrics).unwrap();
        let page = kv.get(id).expect("page");
        assert!(
            matches!(
                page.location.kind,
                DeviceBufferKind::Cuda | DeviceBufferKind::CpuPinnedDram
            ),
            "CUDA runtime uses device pages; no driver falls back to DRAM; kind={:?}",
            page.location.kind
        );
        let marker = vec![1.5f32, -0.25, 0.5, 2.0];
        kv.write_slot(id, 0, 1, 4, &marker, &marker).unwrap();
        let (k, v) = kv.read_kv(id, 1, 4).unwrap();
        assert_eq!(k, marker);
        assert_eq!(v, marker);
        kv.release(id, &metrics);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_pack_kv_concatenates_two_device_pages() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        let metrics = Metrics::default();
        let mut kv = KvTierManager::with_devices(1 << 20, 1 << 20, 0.9, None, Some(0));
        let a = kv.alloc_page(0, 0, 2, 1, 4, &metrics).unwrap();
        let b = kv.alloc_page(0, 1, 2, 1, 4, &metrics).unwrap();
        if kv.get(a).map(|p| p.location.kind) != Some(DeviceBufferKind::Cuda) {
            kv.release(a, &metrics);
            kv.release(b, &metrics);
            return;
        }
        let m0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let m1 = vec![5.0f32, 6.0, 7.0, 8.0];
        kv.write_slot(a, 0, 1, 4, &m0, &m0).unwrap();
        kv.write_slot(b, 0, 1, 4, &m1, &m1).unwrap();
        let dest = allpaca_cuda::device_malloc(64, 0).expect("pack dest");
        let dest_k = dest.cast::<f32>();
        let dest_v = unsafe { dest_k.add(8) };
        let (k_ptr, v_ptr, n) = kv
            .cuda_kv_views(&[a, b], 1, 4, dest_k, dest_v, 2)
            .expect("cuda pack");
        assert_eq!(n, 2);
        let mut k = vec![0.0f32; 8];
        let mut v = vec![0.0f32; 8];
        allpaca_cuda::memcpy(
            k.as_mut_ptr().cast(),
            k_ptr.cast(),
            32,
            allpaca_cuda::MEMCPY_D2H,
        )
        .unwrap();
        allpaca_cuda::memcpy(
            v.as_mut_ptr().cast(),
            v_ptr.cast(),
            32,
            allpaca_cuda::MEMCPY_D2H,
        )
        .unwrap();
        assert_eq!(k, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(v, k);
        allpaca_cuda::device_free(dest);
        kv.release(a, &metrics);
        kv.release(b, &metrics);
    }

    #[test]
    fn vulkan_kv_alloc_write_read_or_dram_fallback() {
        let metrics = Metrics::default();
        let vulkan = allpaca_vulkan::VulkanWorkspace::new(0, 4096)
            .ok()
            .map(std::sync::Arc::new);
        let mut kv = KvTierManager::with_devices(1 << 20, 1 << 20, 0.9, None, None, vulkan);
        let id = kv.alloc_page(0, 0, 8, 1, 4, &metrics).unwrap();
        let page = kv.get(id).expect("page");
        assert!(
            matches!(
                page.location.kind,
                DeviceBufferKind::Vulkan
                    | DeviceBufferKind::CpuPinnedDram
                    | DeviceBufferKind::CpuVram
            ),
            "Vulkan runtime uses device pages; no device falls back to host; kind={:?}",
            page.location.kind
        );
        let marker = vec![1.5f32, -0.25, 0.5, 2.0];
        kv.write_slot(id, 0, 1, 4, &marker, &marker).unwrap();
        let (k, v) = kv.read_kv(id, 1, 4).unwrap();
        assert_eq!(k, marker);
        assert_eq!(v, marker);
        kv.release(id, &metrics);
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
        assert!(
            metrics
                .page_migrations
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0
        );
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
        assert!(
            vram_before >= 4,
            "pages must start in VRAM; vram={vram_before}"
        );
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
    fn capping_vram_and_growing_dram_keeps_the_kv_capacity() {
        // One page here is 4 tokens × 1 head × 4 dims × K+V = 32 floats = 128 B.
        let page = 128usize;
        let fill = |kv: &mut KvTierManager, n: u32| {
            let metrics = Metrics::default();
            (0..n).all(|i| kv.alloc_page(0, i, 4, 1, 4, &metrics).is_ok())
        };
        // 8 pages of VRAM budget and 1 of DRAM: 8 pages fit.
        let mut kv = KvTierManager::new(8 * page, page, 1.0);
        assert!(fill(&mut kv, 8));
        // Capped to 4 pages of VRAM without growing DRAM, 8 pages no longer fit.
        let mut kv = KvTierManager::new(8 * page, page, 1.0);
        kv.set_vram_bytes(4 * page, &Metrics::default()).unwrap();
        assert!(!fill(&mut kv, 8), "the cap alone shrinks the context");
        // The engine's cap moves the difference to DRAM: 8 pages fit again.
        let mut kv = KvTierManager::new(8 * page, page, 1.0);
        kv.set_vram_bytes(4 * page, &Metrics::default()).unwrap();
        kv.grow_dram_bytes(4 * page);
        assert!(fill(&mut kv, 8));
        let (vram, dram) = kv.counts();
        assert!(vram <= 4 && dram >= 4, "vram={vram} dram={dram}");
        // Only `hot_fraction` of the VRAM budget holds pages before DRAM takes over.
        let kv = KvTierManager::new(8 * page, page, 0.75);
        assert_eq!(kv.vram_hot_bytes(), 6 * page);
        assert_eq!(kv.dram_capacity_bytes(), page);
    }

    #[test]
    fn attention_trim_spills_cold_vram_without_changing_capacity() {
        let metrics = Metrics::default();
        let mut kv = KvTierManager::new(4096, 16 * 1024, 1.0);
        for i in 0..4 {
            kv.alloc_page(0, i, 4, 1, 4, &metrics).unwrap();
        }
        let (vram_before, _) = kv.counts();
        assert!(vram_before >= 4);
        let cap_before = kv.vram_capacity;
        let freed = kv.trim_cold_vram_bytes(128, &metrics).unwrap();
        assert!(freed >= 128, "must free at least one page; freed={freed}");
        let (vram_after, dram_after) = kv.counts();
        assert!(dram_after > 0, "trim must spill to DRAM");
        assert!(vram_after < vram_before);
        assert_eq!(kv.vram_capacity, cap_before, "soft trim leaves capacity");
        assert!(
            metrics
                .kv_attention_trims
                .load(std::sync::atomic::Ordering::Relaxed)
                >= 1
        );
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
        assert_eq!(
            &k[..4],
            marker.as_slice(),
            "cold-tier read must keep K bytes"
        );
        assert_eq!(
            &v[..4],
            marker.as_slice(),
            "cold-tier read must keep V bytes"
        );
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
                None,
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
                    None,
                )
                .expect("per-slot KV write");
        }
        allpaca_hip::device_free(dk, k_bytes);
        allpaca_hip::device_free(dv, v_bytes);
        // Only written slots are defined. device_malloc pages are dirty, so
        // unread slots must not be compared (parallel HIP tests reuse VRAM).
        for page_i in 0..n_pages as usize {
            let (bk, bv) = batched
                .read_kv(batch_ids[page_i], n_kv_heads, head_dim)
                .unwrap();
            let (sk, sv) = per_slot
                .read_kv(slot_ids[page_i], n_kv_heads, head_dim)
                .unwrap();
            let page_start = page_i * page_tokens as usize;
            let page_end = page_start + page_tokens as usize;
            let write_lo = start_pos.max(page_start);
            let write_hi = (start_pos + n_tok).min(page_end);
            if write_lo >= write_hi {
                continue;
            }
            let lo = (write_lo - page_start) * kv_width;
            let hi = (write_hi - page_start) * kv_width;
            assert_eq!(
                &bk[lo..hi],
                &sk[lo..hi],
                "K page {page_i} written slots mismatch"
            );
            assert_eq!(
                &bv[lo..hi],
                &sv[lo..hi],
                "V page {page_i} written slots mismatch"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_write_slot_from_device_orders_on_workspace_stream() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        let metrics = Metrics::default();
        let mut kv = KvTierManager::with_devices(1 << 20, 1 << 20, 0.9, None, Some(0));
        let id = kv.alloc_page(0, 0, 4, 1, 4, &metrics).unwrap();
        if kv.get(id).map(|p| p.location.kind) != Some(DeviceBufferKind::Cuda) {
            kv.release(id, &metrics);
            return;
        }
        let ws = allpaca_cuda::CudaWorkspace::new(0, 1 << 16).expect("workspace");
        let stream = ws.raw_stream();
        let pattern = [1.0f32, 2.0, 3.0, 4.0];
        // Async H2D on workspace stream, then stream-ordered slot write (no mid sync).
        let dk = allpaca_cuda::device_malloc(16, 0).unwrap();
        let dv = allpaca_cuda::device_malloc(16, 0).unwrap();
        allpaca_cuda::memcpy_on(
            dk,
            pattern.as_ptr().cast(),
            16,
            allpaca_cuda::MEMCPY_H2D,
            stream,
        )
        .unwrap();
        allpaca_cuda::memcpy_on(
            dv,
            pattern.as_ptr().cast(),
            16,
            allpaca_cuda::MEMCPY_H2D,
            stream,
        )
        .unwrap();
        kv.write_slot_from_device(id, 0, 1, 4, dk.cast(), dv.cast(), Some(stream))
            .expect("stream-ordered KV write");
        ws.synchronize().unwrap();
        let (k, v) = kv.read_kv(id, 1, 4).unwrap();
        assert_eq!(&k[..4], &pattern);
        assert_eq!(&v[..4], &pattern);
        allpaca_cuda::device_free(dk);
        allpaca_cuda::device_free(dv);
        kv.release(id, &metrics);
    }
}
