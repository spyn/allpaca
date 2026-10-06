//! DeepSeek-V4 host expert tier: a RAM cache of packed experts between the
//! GPU caches and the model file.
//!
//! Experts the GPU caches do not hold run on the CPU. Their bytes come from
//! this cache when they are in it, else from the file with Direct I/O in a
//! few large parallel reads (a page fault through the mapped file reads the
//! same bytes at a fraction of the drive's speed, and leaves the OS deciding
//! what stays in RAM). The cache has a fixed budget and keeps the experts
//! that are routed most.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};

use allpaca_gguf::{GgmlType, GgufFile};

use allpaca_types::{AllpacaError, FatalError};

use crate::telemetry::Metrics;

/// Direct I/O wants sector-aligned offsets, lengths and buffers.
const ALIGN: usize = 4096;
/// Parallel readers: an NVMe drive serves several large reads at once.
const READERS: usize = 6;
/// Routing counts fade by `HEAT_DECAY` every this many requested experts.
const HEAT_DECAY_EVERY: u64 = 4096;
const HEAT_DECAY: f32 = 0.8;
/// A cached expert is replaced only by one requested this much more often.
const HEAT_MARGIN: f32 = 1.0;
/// Read-ahead experts waiting to be asked for; the oldest give way.
const STAGE_MAX: usize = 24;
/// Parallel readers of the read-ahead worker.
const AHEAD_READERS: usize = 4;
/// A read-ahead predicts for at most this many rows.
pub(crate) const AHEAD_MAX_ROWS: usize = 4;

/// Where one tensor slice of an expert lives on disk.
#[derive(Debug, Clone)]
pub(crate) struct FileSpan {
    pub shard: usize,
    pub offset: u64,
    pub len: usize,
}

/// A page-aligned heap buffer.
struct AlignedBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    fn new(bytes: usize) -> Result<Self, AllpacaError> {
        let layout = std::alloc::Layout::from_size_align(bytes.max(ALIGN), ALIGN)
            .map_err(|_| AllpacaError::Fatal(FatalError::InvalidConfig("expert RAM tier layout".into())))?;
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                "expert RAM tier ({bytes} bytes)"
            ))));
        }
        Ok(Self { ptr, layout })
    }

    fn len(&self) -> usize {
        self.layout.size()
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// One reader thread's file handles and its aligned landing buffer.
struct Reader {
    files: HashMap<usize, (File, bool)>,
    scratch: AlignedBuf,
}

impl Reader {
    /// Read `span` and copy it to `dst`. Returns whether Direct I/O served it.
    fn read(&mut self, paths: &[PathBuf], span: &FileSpan, dst: *mut u8) -> Result<bool, AllpacaError> {
        let path = paths.get(span.shard).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!("expert shard {} has no path", span.shard)))
        })?;
        if !self.files.contains_key(&span.shard) {
            let opened = match crate::direct_storage::open_direct(path, false) {
                Ok(f) => (f, true),
                Err(_) => (File::open(path)?, false),
            };
            self.files.insert(span.shard, opened);
        }
        let start = span.offset as usize & !(ALIGN - 1);
        let head = span.offset as usize - start;
        let want = (head + span.len).div_ceil(ALIGN) * ALIGN;
        if want > self.scratch.len() {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "expert slice larger than the read buffer".into(),
            )));
        }
        // SAFETY: `want` bytes of the reader's own buffer.
        let buf = unsafe { std::slice::from_raw_parts_mut(self.scratch.ptr, want) };
        let mut direct = self.files[&span.shard].1;
        let mut filled = read_at(&mut self.files.get_mut(&span.shard).unwrap().0, start as u64, buf);
        if direct && !matches!(filled, Ok(n) if n >= head + span.len) {
            // A filesystem that refuses unbuffered reads: use the page cache.
            let mut f = File::open(path)?;
            filled = read_at(&mut f, start as u64, buf);
            self.files.insert(span.shard, (f, false));
            direct = false;
        }
        let filled = filled?;
        if filled < head + span.len {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "short read of expert slice at {} in {}",
                span.offset,
                path.display()
            ))));
        }
        // SAFETY: `dst` has room for `span.len` bytes and is written by this thread only.
        unsafe { std::ptr::copy_nonoverlapping(self.scratch.ptr.add(head), dst, span.len) };
        Ok(direct)
    }
}

/// Bytes read at `offset`; fewer than `buf.len()` only at the end of the file.
fn read_at(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    file.seek(SeekFrom::Start(offset))?;
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Where a fetched expert's bytes are until the next `fetch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Loc {
    Slot(usize),
    Temp(usize),
}

#[derive(Clone, Copy)]
struct Dst(*mut u8);
unsafe impl Send for Dst {}
unsafe impl Sync for Dst {}

pub(crate) struct ExpertRam {
    paths: Vec<PathBuf>,
    /// Packed bytes of one expert: its tensor slices back to back.
    slot_bytes: usize,
    arena: AlignedBuf,
    slots: Vec<Option<(usize, usize)>>,
    index: HashMap<(usize, usize), usize>,
    /// Buffers for experts that are read but not kept.
    temps: Vec<Vec<u8>>,
    readers: Vec<Reader>,
    heat: HashMap<(usize, usize), f32>,
    requests: u64,
    /// Slots being copied to a GPU cache: not evicted until released.
    pinned: Vec<usize>,
    /// Experts read ahead of use, oldest first.
    staged: VecDeque<((usize, usize), Vec<u8>)>,
    /// Expert-sized buffers to reuse.
    spare: Vec<Vec<u8>>,
    staged_total: u64,
    staged_reported: u64,
}

impl ExpertRam {
    /// A cache of at most `budget_bytes` for experts of `slot_bytes`, read
    /// from `paths` (one per GGUF shard). `None` when not even one fits.
    pub(crate) fn new(
        paths: Vec<PathBuf>,
        slot_bytes: usize,
        max_slice: usize,
        budget_bytes: u64,
    ) -> Result<Option<Self>, AllpacaError> {
        let n = (budget_bytes / slot_bytes.max(1) as u64) as usize;
        if n == 0 || slot_bytes == 0 {
            return Ok(None);
        }
        let mut readers = Vec::with_capacity(READERS);
        for _ in 0..READERS {
            readers.push(Reader {
                files: HashMap::new(),
                scratch: AlignedBuf::new(max_slice + 2 * ALIGN)?,
            });
        }
        Ok(Some(Self {
            paths,
            slot_bytes,
            // Zeroed pages are committed when first written.
            arena: AlignedBuf::new(n * slot_bytes)?,
            slots: vec![None; n],
            index: HashMap::new(),
            temps: Vec::new(),
            readers,
            heat: HashMap::new(),
            requests: 0,
            pinned: Vec::new(),
            staged: VecDeque::new(),
            spare: Vec::new(),
            staged_total: 0,
            staged_reported: 0,
        }))
    }

    pub(crate) fn slots(&self) -> usize {
        self.slots.len()
    }

    /// The packed bytes of a fetched expert. Valid until the next `fetch`.
    pub(crate) fn bytes(&self, loc: Loc) -> &[u8] {
        match loc {
            // SAFETY: slot `i` lies inside the arena.
            Loc::Slot(i) => unsafe {
                std::slice::from_raw_parts(self.arena.ptr.add(i * self.slot_bytes), self.slot_bytes)
            },
            Loc::Temp(i) => &self.temps[i][..self.slot_bytes],
        }
    }

    /// The experts of `layer` among `experts` that are neither cached nor
    /// already read ahead.
    pub(crate) fn missing(&self, layer: usize, experts: &[usize]) -> Vec<usize> {
        experts
            .iter()
            .copied()
            .filter(|e| {
                let key = (layer, *e);
                !self.index.contains_key(&key) && !self.staged.iter().any(|(k, _)| *k == key)
            })
            .collect()
    }

    /// An expert-sized buffer for a read-ahead.
    pub(crate) fn spare_buf(&mut self) -> Vec<u8> {
        self.spare.pop().unwrap_or_else(|| vec![0u8; self.slot_bytes])
    }

    /// Keep the packed bytes of `key`, read ahead of use, for the next
    /// [`Self::fetch`] that asks for it.
    pub(crate) fn stage(&mut self, key: (usize, usize), buf: Vec<u8>) {
        if self.staged.len() >= STAGE_MAX {
            if let Some((_, old)) = self.staged.pop_front() {
                self.spare.push(old);
            }
        }
        self.staged.push_back((key, buf));
        self.staged_total += 1;
    }

    /// The cached bytes of `key`, held in place until [`Self::release`]:
    /// the source of a copy into a GPU cache. `None` when it is not cached.
    pub(crate) fn pin(&mut self, key: (usize, usize)) -> Option<(*const u8, usize)> {
        let slot = *self.index.get(&key)?;
        if !self.pinned.contains(&slot) {
            self.pinned.push(slot);
        }
        // SAFETY: slot `slot` lies inside the arena.
        Some((unsafe { self.arena.ptr.add(slot * self.slot_bytes).cast_const() }, self.slot_bytes))
    }

    /// The copies of `keys` have landed: unpin them, and drop them from the
    /// cache, since a GPU holds them now.
    pub(crate) fn release(&mut self, keys: &[(usize, usize)]) {
        for key in keys {
            if let Some(slot) = self.index.remove(key) {
                self.slots[slot] = None;
                self.pinned.retain(|s| *s != slot);
            }
        }
    }

    /// Make every expert of `wanted` (`(expert, its slices on disk)`, all of
    /// `layer`) available in RAM: cached ones are hits, the rest are read from
    /// the file in parallel, and kept when a slot is free or the expert is
    /// requested clearly more often than the coldest cached one.
    pub(crate) fn fetch(
        &mut self,
        layer: usize,
        wanted: &[(usize, Vec<FileSpan>)],
        metrics: &Metrics,
    ) -> Result<Vec<Loc>, AllpacaError> {
        let mut locs = Vec::with_capacity(wanted.len());
        let mut reads: Vec<(Dst, &FileSpan)> = Vec::new();
        let mut n_temp = 0usize;
        let mut hits = 0u64;
        let mut ahead_hits = 0u64;
        for (expert, spans) in wanted {
            let key = (layer, *expert);
            *self.heat.entry(key).or_insert(0.0) += 1.0;
            self.requests += 1;
            if self.requests % HEAT_DECAY_EVERY == 0 {
                self.heat.retain(|_, h| {
                    *h *= HEAT_DECAY;
                    *h > 0.05
                });
            }
            if let Some(&slot) = self.index.get(&key) {
                hits += 1;
                locs.push(Loc::Slot(slot));
                continue;
            }
            if spans.iter().map(|s| s.len).sum::<usize>() != self.slot_bytes {
                return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                    "expert slices do not fill a RAM slot".into(),
                )));
            }
            let loc = match self.take_slot(layer, key, wanted) {
                Some(slot) => Loc::Slot(slot),
                None => {
                    if self.temps.len() <= n_temp {
                        self.temps.push(vec![0u8; self.slot_bytes]);
                    }
                    n_temp += 1;
                    Loc::Temp(n_temp - 1)
                }
            };
            // Already read ahead: no read, the bytes move into place.
            let ahead = self.staged.iter().position(|(k, _)| *k == key).and_then(|i| self.staged.remove(i));
            if let Some((_, mut buf)) = ahead {
                match loc {
                    // SAFETY: slot `i` lies inside the arena; `buf` holds one expert.
                    Loc::Slot(i) => unsafe {
                        std::ptr::copy_nonoverlapping(
                            buf.as_ptr(),
                            self.arena.ptr.add(i * self.slot_bytes),
                            self.slot_bytes,
                        );
                    },
                    Loc::Temp(i) => std::mem::swap(&mut self.temps[i], &mut buf),
                }
                self.spare.push(buf);
                ahead_hits += 1;
                locs.push(loc);
                continue;
            }
            let base = match loc {
                // SAFETY: inside the arena / the temp buffer.
                Loc::Slot(i) => unsafe { self.arena.ptr.add(i * self.slot_bytes) },
                Loc::Temp(i) => self.temps[i].as_mut_ptr(),
            };
            let mut at = 0usize;
            for span in spans {
                // SAFETY: the slices sum to `slot_bytes`.
                reads.push((Dst(unsafe { base.add(at) }), span));
                at += span.len;
            }
            locs.push(loc);
        }
        metrics.expert_ram_hits.fetch_add(hits, Ordering::Relaxed);
        metrics.expert_prefetch_hits.fetch_add(ahead_hits, Ordering::Relaxed);
        metrics
            .expert_prefetch_reads
            .fetch_add(self.staged_total - self.staged_reported, Ordering::Relaxed);
        self.staged_reported = self.staged_total;
        if reads.is_empty() {
            return Ok(locs);
        }
        let reads = &reads;
        let direct = match read_all(&mut self.readers, &self.paths, reads) {
            Ok(n) => n,
            Err(e) => {
                {
                    // Whatever was being read into a slot is not an expert.
                    for (loc, (expert, _)) in locs.iter().zip(wanted) {
                        if let Loc::Slot(i) = loc {
                            if self.index.get(&(layer, *expert)) == Some(i) {
                                self.index.remove(&(layer, *expert));
                                self.slots[*i] = None;
                            }
                        }
                    }
                    return Err(e);
                }
            }
        };
        let bytes: usize = reads.iter().map(|(_, s)| s.len).sum();
        metrics.nvme_reads.fetch_add(reads.len() as u64, Ordering::Relaxed);
        metrics.nvme_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        metrics
            .nvme_direct
            .store(u64::from(direct == reads.len() as u64), Ordering::Relaxed);
        Ok(locs)
    }

    /// A slot for `key`: a free one, or the coldest cached expert's when
    /// `key` is clearly hotter and that expert is not wanted right now.
    fn take_slot(&mut self, layer: usize, key: (usize, usize), wanted: &[(usize, Vec<FileSpan>)]) -> Option<usize> {
        let slot = match self.slots.iter().position(Option::is_none) {
            Some(free) => free,
            None => {
                let (victim, cold) = self
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, k)| k.map(|k| (i, k)))
                    .filter(|(i, k)| {
                        !self.pinned.contains(i) && !(k.0 == layer && wanted.iter().any(|(e, _)| *e == k.1))
                    })
                    .map(|(i, k)| (i, self.heat.get(&k).copied().unwrap_or(0.0)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))?;
                if self.heat.get(&key).copied().unwrap_or(0.0) < cold + HEAT_MARGIN {
                    return None;
                }
                if let Some(old) = self.slots[victim].take() {
                    self.index.remove(&old);
                }
                victim
            }
        };
        self.slots[slot] = Some(key);
        self.index.insert(key, slot);
        Some(slot)
    }
}

/// Read-ahead: a worker that predicts the experts of the next layer and
/// reads the ones that are in neither VRAM nor RAM while the current layer
/// still computes.
///
/// The prediction applies the next layer's router to the current layer's
/// expert input. It only decides what is read early: which experts run, and
/// their result, do not depend on it.
pub(crate) struct Prefetcher {
    tx: Option<SyncSender<AheadJob>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

struct AheadJob {
    layer: usize,
    rows: Vec<f32>,
    on_gpu: Vec<usize>,
}

impl Prefetcher {
    /// `spans(layer, expert)` gives an expert's slices on disk; `top` is how
    /// many experts a row routes to.
    pub(crate) fn start(
        file: Arc<GgufFile>,
        ram: Arc<Mutex<ExpertRam>>,
        dim: usize,
        n_experts: usize,
        top: usize,
        max_slice: usize,
        spans: impl Fn(&GgufFile, usize, usize) -> Option<Vec<FileSpan>> + Send + 'static,
    ) -> Result<Self, AllpacaError> {
        let paths = file.shard_paths().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::InvalidConfig("read-ahead needs the model's files".into()))
        })?;
        let mut readers = Vec::with_capacity(AHEAD_READERS);
        for _ in 0..AHEAD_READERS {
            readers.push(Reader {
                files: HashMap::new(),
                scratch: AlignedBuf::new(max_slice + 2 * ALIGN)?,
            });
        }
        // One job at a time: a prediction that had to wait is stale.
        let (tx, rx) = sync_channel::<AheadJob>(1);
        let worker = std::thread::Builder::new()
            .name("dsv4-read-ahead".into())
            .spawn(move || {
                let mut routers: HashMap<usize, Vec<f32>> = HashMap::new();
                while let Ok(job) = rx.recv() {
                    let router = routers
                        .entry(job.layer)
                        .or_insert_with(|| load_router(&file, job.layer, dim, n_experts));
                    if router.len() != dim * n_experts {
                        continue;
                    }
                    let mut want = predict(router, &job.rows, dim, n_experts, top);
                    want.retain(|e| !job.on_gpu.contains(e));
                    let lock = || ram.lock().unwrap_or_else(|e| e.into_inner());
                    let missing = lock().missing(job.layer, &want);
                    let mut bufs: Vec<(usize, Vec<FileSpan>, Vec<u8>)> = Vec::new();
                    for e in missing {
                        if let Some(s) = spans(&file, job.layer, e) {
                            let buf = lock().spare_buf();
                            if s.iter().map(|x| x.len).sum::<usize>() == buf.len() {
                                bufs.push((e, s, buf));
                            }
                        }
                    }
                    if bufs.is_empty() {
                        continue;
                    }
                    let mut reads: Vec<(Dst, &FileSpan)> = Vec::new();
                    for (_, s, buf) in bufs.iter_mut() {
                        let mut at = 0usize;
                        for span in s.iter() {
                            // SAFETY: the slices sum to the buffer's length.
                            reads.push((Dst(unsafe { buf.as_mut_ptr().add(at) }), span));
                            at += span.len;
                        }
                    }
                    let ok = read_all(&mut readers, &paths, &reads).is_ok();
                    drop(reads);
                    let mut ram = lock();
                    for (e, _, buf) in bufs {
                        if ok {
                            ram.stage((job.layer, e), buf);
                        } else {
                            ram.spare.push(buf);
                        }
                    }
                }
            })
            .map_err(|e| AllpacaError::Fatal(FatalError::InvalidConfig(format!("read-ahead thread: {e}"))))?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    /// Predict from `rows` (the current layer's expert inputs, `dim` each)
    /// and read ahead for `layer`. Dropped when the worker is still busy.
    pub(crate) fn submit(&self, layer: usize, rows: &[f32], on_gpu: Vec<usize>) {
        if let Some(tx) = self.tx.as_ref() {
            let _ = tx.try_send(AheadJob {
                layer,
                rows: rows.to_vec(),
                on_gpu,
            });
        }
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The router of `layer` as F32, `n_experts` rows of `dim`; empty when the
/// file has none this worker can read.
fn load_router(file: &GgufFile, layer: usize, dim: usize, n_experts: usize) -> Vec<f32> {
    let name = format!("blk.{layer}.ffn_gate_inp.weight");
    let Some(info) = file.tensor(&name) else {
        return Vec::new();
    };
    if info.n_elements() as usize != dim * n_experts {
        return Vec::new();
    }
    match info.ggml_type {
        GgmlType::BF16 => file
            .tensor_bytes_of(info)
            .map(|b| {
                b.chunks_exact(2)
                    .map(|c| allpaca_cpu::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            })
            .unwrap_or_default(),
        _ => file.tensor_f32(&name).unwrap_or_default(),
    }
}

/// The `top` highest-scoring experts of each row, without duplicates.
fn predict(router: &[f32], rows: &[f32], dim: usize, n_experts: usize, top: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for x in rows.chunks_exact(dim) {
        let mut scores: Vec<(f32, usize)> = (0..n_experts)
            .map(|e| {
                let w = &router[e * dim..(e + 1) * dim];
                (w.iter().zip(x).map(|(a, b)| a * b).sum::<f32>(), e)
            })
            .collect();
        let k = top.min(n_experts);
        if k == 0 {
            continue;
        }
        scores.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        for (_, e) in &scores[..k] {
            if !out.contains(e) {
                out.push(*e);
            }
        }
    }
    out
}

/// Read every `(destination, span)` with `readers` in parallel. Returns how
/// many were served by Direct I/O.
fn read_all(readers: &mut [Reader], paths: &[PathBuf], reads: &[(Dst, &FileSpan)]) -> Result<u64, AllpacaError> {
    let next = AtomicUsize::new(0);
    let n_threads = readers.len().min(reads.len());
    let results: Vec<Result<u64, AllpacaError>> = std::thread::scope(|s| {
        let handles: Vec<_> = readers
            .iter_mut()
            .take(n_threads)
            .map(|reader| {
                let next = &next;
                s.spawn(move || -> Result<u64, AllpacaError> {
                    let mut direct = 0u64;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((dst, span)) = reads.get(i) else {
                            return Ok(direct);
                        };
                        direct += u64::from(reader.read(paths, span, dst.0)?);
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().unwrap_or_else(|_| {
                    Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                        "expert reader thread panicked".into(),
                    )))
                })
            })
            .collect()
    });
    let mut direct = 0u64;
    for r in results {
        direct += r?;
    }
    Ok(direct)
}

/// Drop the pages of a mapped-file range from this process's working set.
/// They stay in the OS file cache while memory lasts; what this stops is a
/// file that was only read once (weights uploaded to a GPU) counting as this
/// process's memory until the system is short.
pub(crate) fn release_mapped(bytes: &[u8]) {
    let start = (bytes.as_ptr() as usize).div_ceil(ALIGN) * ALIGN;
    let end = (bytes.as_ptr() as usize + bytes.len()) & !(ALIGN - 1);
    if end <= start {
        return;
    }
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn VirtualUnlock(address: *mut std::ffi::c_void, size: usize) -> i32;
        }
        // SAFETY: a range inside a live mapping. Unlocking pages that are not
        // locked fails with ERROR_NOT_LOCKED and removes them from the
        // working set, which is the documented way to do that.
        unsafe {
            VirtualUnlock(start as *mut std::ffi::c_void, end - start);
        }
    }
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        const MADV_DONTNEED: i32 = 4;
        // SAFETY: a range inside a live read-only file mapping; the pages are
        // read again from the page cache on the next access.
        unsafe {
            madvise(start as *mut std::ffi::c_void, end - start, MADV_DONTNEED);
        }
    }
}

/// RAM this process can still take without pushing the system into paging.
pub(crate) fn available_ram_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
        }
        // SAFETY: a correctly sized, zeroed MEMORYSTATUSEX with its length set.
        unsafe {
            let mut st: MemoryStatusEx = std::mem::zeroed();
            st.length = std::mem::size_of::<MemoryStatusEx>() as u32;
            (GlobalMemoryStatusEx(&mut st) != 0).then_some(st.avail_phys)
        }
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file of `n` experts of three slices, each byte naming its expert.
    fn fixture(dir: &std::path::Path, n: usize, part: usize) -> (PathBuf, Vec<Vec<FileSpan>>) {
        let path = dir.join(format!("allpaca_dsv4_ram_{}_{n}_{part}.bin", std::process::id()));
        let head = 123usize; // slices start unaligned, as in a GGUF
        let mut data = vec![0xEEu8; head];
        let mut spans = Vec::new();
        for e in 0..n {
            let mut s = Vec::new();
            for p in 0..3 {
                s.push(FileSpan {
                    shard: 0,
                    offset: data.len() as u64,
                    len: part,
                });
                data.extend(std::iter::repeat_n((e * 3 + p) as u8, part));
            }
            spans.push(s);
        }
        std::fs::write(&path, &data).unwrap();
        (path, spans)
    }

    #[test]
    fn experts_are_read_once_then_served_from_ram_and_the_hot_ones_stay() {
        let part = 5000usize;
        let (path, spans) = fixture(&std::env::temp_dir(), 4, part);
        let metrics = Metrics::default();
        // Room for two experts.
        let mut ram = ExpertRam::new(vec![path.clone()], 3 * part, part, (2 * 3 * part) as u64)
            .unwrap()
            .unwrap();
        assert_eq!(ram.slots(), 2);
        let check = |ram: &ExpertRam, loc: Loc, e: usize| {
            let b = ram.bytes(loc);
            for p in 0..3 {
                assert!(b[p * part..(p + 1) * part].iter().all(|v| *v == (e * 3 + p) as u8), "expert {e} part {p}");
            }
        };
        let want = |ids: &[usize]| -> Vec<(usize, Vec<FileSpan>)> { ids.iter().map(|&e| (e, spans[e].clone())).collect() };
        let reads = |m: &Metrics| m.nvme_reads.load(Ordering::Relaxed);

        // Two experts fill the cache: six slices read.
        let locs = ram.fetch(7, &want(&[0, 1]), &metrics).unwrap();
        assert!(matches!(locs[..], [Loc::Slot(_), Loc::Slot(_)]));
        check(&ram, locs[0], 0);
        check(&ram, locs[1], 1);
        assert_eq!(reads(&metrics), 6);

        // Again: served from RAM, nothing read.
        let locs = ram.fetch(7, &want(&[1, 0]), &metrics).unwrap();
        check(&ram, locs[0], 1);
        check(&ram, locs[1], 0);
        assert_eq!(reads(&metrics), 6);
        assert_eq!(metrics.expert_ram_hits.load(Ordering::Relaxed), 2);

        // A third expert is read but not kept while it is no hotter...
        let locs = ram.fetch(7, &want(&[2]), &metrics).unwrap();
        assert!(matches!(locs[0], Loc::Temp(_)));
        check(&ram, locs[0], 2);
        assert_eq!(reads(&metrics), 9);
        // ...and takes a slot once it has been asked for more than a cached one.
        ram.fetch(7, &want(&[2]), &metrics).unwrap();
        let locs = ram.fetch(7, &want(&[2]), &metrics).unwrap();
        assert!(matches!(locs[0], Loc::Slot(_)), "{locs:?}");
        check(&ram, locs[0], 2);
        let locs = ram.fetch(7, &want(&[2]), &metrics).unwrap();
        check(&ram, locs[0], 2);
        assert_eq!(reads(&metrics), 15, "the fourth request is a RAM hit");

        // A pinned expert keeps its bytes while others are fetched, and
        // leaves the cache when released.
        let (ptr, len) = ram.pin((7, 2)).expect("cached");
        for _ in 0..6 {
            ram.fetch(7, &want(&[3]), &metrics).unwrap();
        }
        // SAFETY: pinned, so the slot was not reused.
        let held = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert!(held[..part].iter().all(|v| *v == 6) && held[2 * part..].iter().all(|v| *v == 8));
        ram.release(&[(7, 2)]);
        assert!(ram.pin((7, 2)).is_none());

        // The same expert number in another layer is a different expert.
        let locs = ram.fetch(8, &want(&[3]), &metrics).unwrap();
        check(&ram, locs[0], 3);
        let _ = std::fs::remove_file(path);
    }
}
