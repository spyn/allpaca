//! Packed expert banks on NVMe. Direct I/O into a bounded DRAM host pool.
//!
//! Not KV. Host (or this file) is source of truth when the expert set does not
//! fit RAM; GPU LRU slots remain a speed dial.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use allpaca_types::{AllpacaError, FatalError};

use crate::telemetry::Metrics;
use crate::weight_set::{align_up, ExpertId, LayerFfn, DIRECT_ALIGN};

#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBuf {
    fn new(size: usize, align: usize) -> Result<Self, AllpacaError> {
        let size = size.max(align);
        let layout = Layout::from_size_align(size, align).map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "aligned Direct I/O layout".into(),
            ))
        })?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                "aligned Direct I/O buffer ({size} bytes)"
            ))));
        }
        Ok(Self { ptr, layout })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

#[derive(Debug, Clone, Copy)]
struct PackedSlotMeta {
    id: ExpertId,
    payload: usize,
    n_norm: u32,
    n_gate: u32,
    n_up: u32,
    n_down: u32,
}

/// Aligned FTW-style expert file. One Direct I/O slot per `(layer, expert)`.
pub struct PackedStore {
    path: PathBuf,
    file: Mutex<Option<File>>,
    slot_bytes: usize,
    entries: Vec<PackedSlotMeta>,
    index: HashMap<ExpertId, usize>,
    expert_bytes: usize,
    direct: bool,
    reads: AtomicU64,
    bytes: AtomicU64,
    remove_on_drop: bool,
}

impl std::fmt::Debug for PackedStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedStore")
            .field("path", &self.path)
            .field("slot_bytes", &self.slot_bytes)
            .field("n_slots", &self.entries.len())
            .field("direct", &self.direct)
            .finish()
    }
}

impl PackedStore {
    pub fn pack<'a, I>(path: &Path, experts: I) -> Result<Self, AllpacaError>
    where
        I: IntoIterator<Item = (ExpertId, &'a LayerFfn)>,
    {
        let mut entries = Vec::new();
        let mut blobs = Vec::new();
        let mut max_payload = 0usize;
        let mut expert_bytes = 0usize;
        for (id, ffn) in experts {
            let payload = ffn.pack_bytes();
            max_payload = max_payload.max(payload.len());
            expert_bytes = expert_bytes.max(ffn.bytes());
            entries.push(PackedSlotMeta {
                id,
                payload: payload.len(),
                n_norm: ffn.ffn_norm.len() as u32,
                n_gate: ffn.w_gate.len() as u32,
                n_up: ffn.w_up.len() as u32,
                n_down: ffn.w_down.len() as u32,
            });
            blobs.push(payload);
        }
        if entries.is_empty() {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "packed store has no experts".into(),
            )));
        }
        let slot_bytes = align_up(max_payload.max(1), DIRECT_ALIGN);
        {
            let mut out = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)?;
            for blob in &blobs {
                let mut slot = vec![0u8; slot_bytes];
                slot[..blob.len()].copy_from_slice(blob);
                out.write_all(&slot)?;
            }
            out.sync_all()?;
        }
        let (file, direct) = open_packed_reader(path, slot_bytes)?;
        let mut index = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            index.insert(e.id, i);
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(Some(file)),
            slot_bytes,
            entries,
            index,
            expert_bytes,
            direct,
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            remove_on_drop: true,
        })
    }

    pub fn read_expert(&self, id: ExpertId) -> Result<LayerFfn, AllpacaError> {
        let slot = *self.index.get(&id).ok_or_else(|| {
            AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "packed store missing expert ({}, {})",
                id.layer, id.expert
            )))
        })?;
        let meta = self.entries[slot];
        let mut buf = AlignedBuf::new(self.slot_bytes, DIRECT_ALIGN)?;
        {
            let mut guard = self.file.lock().map_err(|_| {
                AllpacaError::Fatal(FatalError::InvalidConfig(
                    "packed store mutex poisoned".into(),
                ))
            })?;
            let file = guard.as_mut().ok_or_else(|| {
                AllpacaError::Fatal(FatalError::ModelLoad("packed store file closed".into()))
            })?;
            let off = (slot as u64).saturating_mul(self.slot_bytes as u64);
            file.seek(SeekFrom::Start(off))?;
            let dest = buf.as_mut();
            let mut filled = 0usize;
            while filled < self.slot_bytes {
                let n = file.read(&mut dest[filled..])?;
                if n == 0 {
                    return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                        "packed store short read at slot {slot}"
                    ))));
                }
                filled += n;
            }
        }
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes
            .fetch_add(self.slot_bytes as u64, Ordering::Relaxed);
        LayerFfn::unpack_bytes(
            &buf.as_mut()[..meta.payload],
            meta.n_norm as usize,
            meta.n_gate as usize,
            meta.n_up as usize,
            meta.n_down as usize,
        )
    }

    pub fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    pub fn bytes_read(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }
}

impl Drop for PackedStore {
    fn drop(&mut self) {
        if !self.remove_on_drop {
            return;
        }
        if let Ok(mut guard) = self.file.lock() {
            *guard = None;
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

fn open_packed_reader(path: &Path, slot_bytes: usize) -> Result<(File, bool), AllpacaError> {
    if let Ok(mut file) = open_direct(path) {
        if probe_slot_read(&mut file, slot_bytes).is_ok() {
            file.seek(SeekFrom::Start(0))?;
            return Ok((file, true));
        }
    }
    Ok((File::open(path)?, false))
}

fn probe_slot_read(file: &mut File, slot_bytes: usize) -> Result<(), AllpacaError> {
    let mut buf = AlignedBuf::new(slot_bytes, DIRECT_ALIGN)?;
    file.seek(SeekFrom::Start(0))?;
    let dest = buf.as_mut();
    let mut filled = 0usize;
    while filled < slot_bytes {
        let n = file.read(&mut dest[filled..])?;
        if n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(
                "packed store probe short read".into(),
            )));
        }
        filled += n;
    }
    Ok(())
}

pub fn temp_pack_path() -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let id = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "allpaca-pack-{}-{id}.bin",
        std::process::id()
    ))
}

fn open_direct(path: &Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        return OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING)
            .open(path);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use std::os::unix::fs::OpenOptionsExt;
        return OpenOptions::new()
            .read(true)
            .custom_flags(0o40000)
            .open(path);
    }
    #[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
    {
        let _ = path;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Direct I/O is not available on this platform",
        ))
    }
}

struct HostPoolInner {
    bound: Vec<Option<ExpertId>>,
    used_at: Vec<u64>,
    slots: Vec<LayerFfn>,
    clock: u64,
}

/// Bounded DRAM LRU over [`PackedStore`]. GPU LRU fills from here.
pub struct HostExpertPool {
    store: PackedStore,
    n_slots: usize,
    inner: Mutex<HostPoolInner>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl std::fmt::Debug for HostExpertPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostExpertPool")
            .field("n_slots", &self.n_slots)
            .field("direct", &self.store.direct)
            .finish()
    }
}

impl HostExpertPool {
    pub fn new(store: PackedStore, n_slots: usize) -> Arc<Self> {
        let n = n_slots.max(1);
        Arc::new(Self {
            store,
            n_slots: n,
            inner: Mutex::new(HostPoolInner {
                bound: vec![None; n],
                used_at: vec![0; n],
                slots: vec![LayerFfn::default(); n],
                clock: 0,
            }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    pub fn expert_bytes(&self) -> usize {
        self.store.expert_bytes()
    }

    pub fn is_direct(&self) -> bool {
        self.store.is_direct()
    }

    pub fn nvme_reads(&self) -> u64 {
        self.store.reads()
    }

    pub fn nvme_bytes(&self) -> u64 {
        self.store.bytes_read()
    }

    pub fn record_slots(metrics: &Metrics, n_slots: usize) {
        metrics
            .expert_host_slots
            .store(n_slots as u64, Ordering::Relaxed);
    }

    pub fn get(&self, id: ExpertId) -> Result<LayerFfn, AllpacaError> {
        let mut inner = self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "host expert pool mutex poisoned".into(),
            ))
        })?;
        inner.clock = inner.clock.saturating_add(1);
        if let Some(slot) = inner.bound.iter().position(|&b| b == Some(id)) {
            inner.used_at[slot] = inner.clock;
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(inner.slots[slot].clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let victim = inner
            .bound
            .iter()
            .position(|b| b.is_none())
            .unwrap_or_else(|| {
                inner
                    .used_at
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });
        let ffn = self.store.read_expert(id)?;
        inner.slots[victim] = ffn.clone();
        inner.bound[victim] = Some(id);
        inner.used_at[victim] = inner.clock;
        Ok(ffn)
    }

    pub fn get_layer(&self, layer: u32, n_experts: usize) -> Result<Vec<LayerFfn>, AllpacaError> {
        let mut out = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            out.push(self.get(ExpertId {
                layer,
                expert: e as u32,
            })?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_path() -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(1);
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "allpaca-pack-test-{}-{id}.bin",
            std::process::id()
        ))
    }

    fn sample_ffn() -> LayerFfn {
        LayerFfn {
            ffn_norm: vec![],
            w_gate: vec![1.0, 2.0, 3.0],
            w_up: vec![4.0, 5.0],
            w_down: vec![6.0],
        }
    }

    #[test]
    fn packed_expert_roundtrip_uses_aligned_slots() {
        let ffn = sample_ffn();
        let path = unique_path();
        let store = PackedStore::pack(
            &path,
            [(
                ExpertId {
                    layer: 1,
                    expert: 2,
                },
                &ffn,
            )],
        )
        .unwrap();
        assert_eq!(store.slot_bytes, DIRECT_ALIGN);
        let got = store
            .read_expert(ExpertId {
                layer: 1,
                expert: 2,
            })
            .unwrap();
        assert_eq!(got.w_gate, ffn.w_gate);
        assert_eq!(got.w_up, ffn.w_up);
        assert_eq!(got.w_down, ffn.w_down);
        assert!(store.reads() >= 1);
        assert_eq!(store.bytes_read(), DIRECT_ALIGN as u64);
    }

    #[test]
    fn host_pool_pages_from_packed_store() {
        let a = LayerFfn {
            w_gate: vec![1.0],
            w_up: vec![2.0],
            w_down: vec![3.0],
            ..LayerFfn::default()
        };
        let b = LayerFfn {
            w_gate: vec![9.0],
            w_up: vec![8.0],
            w_down: vec![7.0],
            ..LayerFfn::default()
        };
        let path = unique_path();
        let store = PackedStore::pack(
            &path,
            [
                (
                    ExpertId {
                        layer: 0,
                        expert: 0,
                    },
                    &a,
                ),
                (
                    ExpertId {
                        layer: 0,
                        expert: 1,
                    },
                    &b,
                ),
            ],
        )
        .unwrap();
        let pool = HostExpertPool::new(store, 1);
        let got_a = pool
            .get(ExpertId {
                layer: 0,
                expert: 0,
            })
            .unwrap();
        let got_b = pool
            .get(ExpertId {
                layer: 0,
                expert: 1,
            })
            .unwrap();
        assert_eq!(got_a.w_gate, a.w_gate);
        assert_eq!(got_b.w_down, b.w_down);
        assert!(
            pool.nvme_reads() >= 2,
            "one DRAM slot over two experts must miss to NVMe"
        );
    }
}
