//! Cold KV pages on NVMe. Direct I/O into aligned slots; not the weight packed store.
//!
//! VRAM stays hot, pinned DRAM stays warm. This file is capacity for pages that
//! lost the DRAM working set. Prefetch into DRAM before a decode write when the
//! next page_index is already known.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use allpaca_types::{AllpacaError, FatalError, PageId};

use crate::direct_storage::open_direct;
use crate::weight_set::{align_up, DIRECT_ALIGN};

struct AlignedBuf {
    ptr: *mut u8,
    layout: Layout,
}

impl AlignedBuf {
    fn new(size: usize, align: usize) -> Result<Self, AllpacaError> {
        let size = size.max(align);
        let layout = Layout::from_size_align(size, align).map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "aligned KV Direct I/O layout".into(),
            ))
        })?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                "aligned KV Direct I/O buffer ({size} bytes)"
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

struct StoreInner {
    file: Option<File>,
    slot_bytes: usize,
    n_slots: usize,
    next: usize,
    free: Vec<usize>,
    bound: HashMap<PageId, usize>,
}

/// Aligned swap file of KV pages. One Direct I/O slot per spilled page.
pub struct NvmeKvStore {
    path: PathBuf,
    budget_bytes: usize,
    inner: Mutex<StoreInner>,
    direct: bool,
    reads: AtomicU64,
    writes: AtomicU64,
    bytes: AtomicU64,
    remove_on_drop: bool,
}

impl std::fmt::Debug for NvmeKvStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NvmeKvStore")
            .field("path", &self.path)
            .field("budget_bytes", &self.budget_bytes)
            .field("direct", &self.direct)
            .finish()
    }
}

impl NvmeKvStore {
    pub fn open(
        path: &Path,
        budget_bytes: usize,
        remove_on_drop: bool,
    ) -> Result<Self, AllpacaError> {
        if budget_bytes == 0 {
            return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
                "kv.nvme_bytes must be > 0 to open a cold tier".into(),
            )));
        }
        {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(path)?;
        }
        let (file, direct) = open_rw(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            budget_bytes,
            inner: Mutex::new(StoreInner {
                file: Some(file),
                slot_bytes: 0,
                n_slots: 0,
                next: 0,
                free: Vec::new(),
                bound: HashMap::new(),
            }),
            direct,
            reads: AtomicU64::new(0),
            writes: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            remove_on_drop,
        })
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    #[allow(dead_code)]
    pub fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn write_page(&self, id: PageId, data: &[f32]) -> Result<usize, AllpacaError> {
        let need = align_up(data.len().saturating_mul(4).max(1), DIRECT_ALIGN);
        let mut inner = self.lock()?;
        if inner.slot_bytes == 0 {
            inner.slot_bytes = need;
            inner.n_slots = (self.budget_bytes / need).max(1);
        } else if need > inner.slot_bytes {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "KV page larger than NVMe slot".into(),
            )));
        }
        if let Some(existing) = inner.bound.get(&id).copied() {
            self.write_slot(&mut inner, existing, data)?;
            return Ok(existing);
        }
        let slot = if let Some(free) = inner.free.pop() {
            free
        } else if inner.next < inner.n_slots {
            let s = inner.next;
            inner.next += 1;
            s
        } else {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(
                "NVMe KV cold tier is full".into(),
            )));
        };
        self.write_slot(&mut inner, slot, data)?;
        inner.bound.insert(id, slot);
        Ok(slot)
    }

    pub fn read_page(&self, slot: usize, n_floats: usize) -> Result<Vec<f32>, AllpacaError> {
        let mut inner = self.lock()?;
        if inner.slot_bytes == 0 || n_floats == 0 {
            return Ok(vec![0.0; n_floats]);
        }
        let slot_bytes = inner.slot_bytes;
        let mut buf = AlignedBuf::new(slot_bytes, DIRECT_ALIGN)?;
        let off = (slot as u64).saturating_mul(slot_bytes as u64);
        let file = inner.file.as_mut().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV file closed".into()))
        })?;
        file.seek(SeekFrom::Start(off))?;
        let dest = buf.as_mut();
        let mut filled = 0usize;
        while filled < slot_bytes {
            let n = file.read(&mut dest[filled..])?;
            if n == 0 {
                return Err(AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                    "NVMe KV short read at slot {slot}"
                ))));
            }
            filled += n;
        }
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(slot_bytes as u64, Ordering::Relaxed);
        let mut out = vec![0.0f32; n_floats];
        let src = buf.as_mut();
        for (i, dst) in out.iter_mut().enumerate() {
            let a = i * 4;
            *dst = f32::from_le_bytes(src[a..a + 4].try_into().unwrap());
        }
        Ok(out)
    }

    pub fn free_page(&self, id: PageId) {
        if let Ok(mut inner) = self.inner.lock() {
            if let Some(slot) = inner.bound.remove(&id) {
                inner.free.push(slot);
            }
        }
    }

    fn write_slot(
        &self,
        inner: &mut StoreInner,
        slot: usize,
        data: &[f32],
    ) -> Result<(), AllpacaError> {
        let mut buf = AlignedBuf::new(inner.slot_bytes, DIRECT_ALIGN)?;
        {
            let dest = buf.as_mut();
            for (i, v) in data.iter().enumerate() {
                let a = i * 4;
                dest[a..a + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let off = (slot as u64).saturating_mul(inner.slot_bytes as u64);
        let file = inner.file.as_mut().ok_or_else(|| {
            AllpacaError::Fatal(FatalError::OutOfMemory("NVMe KV file closed".into()))
        })?;
        file.seek(SeekFrom::Start(off))?;
        file.write_all(buf.as_mut())?;
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.bytes
            .fetch_add(inner.slot_bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, StoreInner>, AllpacaError> {
        self.inner.lock().map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "NVMe KV store mutex poisoned".into(),
            ))
        })
    }
}

impl Drop for NvmeKvStore {
    fn drop(&mut self) {
        if !self.remove_on_drop {
            return;
        }
        if let Ok(mut inner) = self.inner.lock() {
            inner.file = None;
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

pub fn temp_kv_nvme_path() -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    let id = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("allpaca-kv-{}-{id}.bin", std::process::id()))
}

/// The slot file, unbuffered when the platform allows: a buffered handle
/// kept open for random page reads is slow on Windows.
fn open_rw(path: &Path) -> Result<(File, bool), AllpacaError> {
    match open_direct(path, true) {
        Ok(file) => return Ok((file, true)),
        Err(e) => tracing::warn!(
            error = %e,
            path = %path.display(),
            "KV NVMe tier: unbuffered open failed, using a buffered handle"
        ),
    }
    Ok((OpenOptions::new().read(true).write(true).open(path)?, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvme_kv_slot_roundtrip() {
        let path = temp_kv_nvme_path();
        let store = NvmeKvStore::open(&path, 64 * 1024, true).unwrap();
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.25).collect();
        let slot = store.write_page(7, &data).unwrap();
        let got = store.read_page(slot, data.len()).unwrap();
        assert_eq!(got, data);
        assert!(store.writes() >= 1);
        assert!(store.reads() >= 1);
        #[cfg(windows)]
        assert!(store.is_direct(), "pages are read unbuffered on Windows");
    }
}
