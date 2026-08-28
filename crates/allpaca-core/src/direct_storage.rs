//! DirectStorage / GPU-direct NVMe path (GEM).
//!
//! Today expert and KV cold tiers bounce NVMe → host RAM → PCIe → VRAM.
//! DirectStorage (Windows) / GDS (Linux) can DMA into device memory and skip
//! the host bounce for large expert fills.
//!
//! Vertical (deepest feasible without inventing GPU DMA):
//! 1. Probe `dstorage.dll` and resolve `DStorageGetFactory` via GetProcAddress
//!    (no NuGet / no COM call — factory export proves SDK surface).
//! 2. File-coordinate DMA **intents** when the probe is hot (`ds_enqueued`).
//! 3. Real host path: aligned Direct I/O into a pinned-shaped buffer, then
//!    record `ds_pinned_*` as the H2D source (overlapped host→device still
//!    needs HIP; we do not claim NVMe→VRAM DMA completed).
//! 4. When a device destination is bound, stage aligned host read then H2D
//!    into the expert ring slot (`ds_gpu_dma_*` — deepest path without COM).

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use allpaca_types::{AllpacaError, FatalError};

use crate::weight_set::DIRECT_ALIGN;

/// Result of probing host DirectStorage / GDS availability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectStorageProbe {
    pub available: bool,
    pub backend: DirectStorageBackend,
    pub detail: String,
    /// Windows: `DStorageGetFactory` export resolved (SDK surface without COM).
    pub factory_resolved: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // LinuxGds used on Linux builds; kept for API symmetry on Windows.
pub enum DirectStorageBackend {
    /// No direct path; keep host-staged NVMe reads.
    HostStaging,
    /// Windows DirectStorage (`dstorage.dll`) present.
    WindowsDirectStorage,
    /// Linux GPUDirect Storage marker (future).
    LinuxGds,
}

/// Probe + queue shared with the NVMe expert host pool.
#[derive(Debug, Clone)]
pub struct DirectStorageBridge {
    pub probe: DirectStorageProbe,
    pub queue: Arc<DirectStorageQueue>,
}

/// Records DMA intents, pinned Direct I/O staging, and buffered host bounce.
#[derive(Debug, Default)]
pub struct DirectStorageQueue {
    /// Would-be NVMe→device DMA (probe-gated; no GPU buffer binding yet).
    enqueued: AtomicU64,
    bytes: AtomicU64,
    /// Buffered host-staged fills (no Direct I/O).
    staged: AtomicU64,
    staged_bytes: AtomicU64,
    /// Aligned Direct I/O into pinned-shaped host memory (H2D source).
    pinned: AtomicU64,
    pinned_bytes: AtomicU64,
    /// NVMe→host→VRAM fills with a bound device destination.
    gpu_dma: AtomicU64,
    gpu_dma_bytes: AtomicU64,
}

impl DirectStorageQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueued(&self) -> u64 {
        self.enqueued.load(Ordering::Relaxed)
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn staged(&self) -> u64 {
        self.staged.load(Ordering::Relaxed)
    }

    pub fn staged_bytes(&self) -> u64 {
        self.staged_bytes.load(Ordering::Relaxed)
    }

    pub fn pinned(&self) -> u64 {
        self.pinned.load(Ordering::Relaxed)
    }

    pub fn pinned_bytes(&self) -> u64 {
        self.pinned_bytes.load(Ordering::Relaxed)
    }

    pub fn gpu_dma(&self) -> u64 {
        self.gpu_dma.load(Ordering::Relaxed)
    }

    pub fn gpu_dma_bytes(&self) -> u64 {
        self.gpu_dma_bytes.load(Ordering::Relaxed)
    }

    /// Record a would-be DMA of `bytes` from `path` @ `offset` into device memory.
    /// Returns false when DirectStorage is unavailable (caller should host-stage).
    pub fn try_enqueue(
        &self,
        probe: &DirectStorageProbe,
        path: &Path,
        offset: u64,
        bytes: usize,
    ) -> bool {
        let _ = (path, offset);
        if !probe.available || bytes == 0 {
            return false;
        }
        self.enqueued.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        true
    }

    /// Host-staged NVMe→RAM→VRAM fill (no Direct I/O / no DS DMA).
    pub fn record_staged_h2d(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.staged.fetch_add(1, Ordering::Relaxed);
        self.staged_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Aligned Direct I/O completed into pinned-shaped host memory (H2D source).
    pub fn record_pinned_h2d(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.pinned.fetch_add(1, Ordering::Relaxed);
        self.pinned_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Prefer DMA intent when probe is hot; otherwise record staged.
    pub fn enqueue_or_stage(
        &self,
        probe: &DirectStorageProbe,
        path: &Path,
        offset: u64,
        bytes: usize,
    ) {
        if !self.try_enqueue(probe, path, offset, bytes) {
            self.record_staged_h2d(bytes);
        }
    }

    /// Deepest real host path: Direct I/O (or buffered fallback) into an aligned
    /// buffer. Does **not** bind a GPU destination — records `pinned` or `staged`.
    pub fn stage_aligned_read(
        &self,
        path: &Path,
        offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>, AllpacaError> {
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let aligned = align_up(bytes, DIRECT_ALIGN);
        let mut buf = AlignedHostBuf::new(aligned, DIRECT_ALIGN)?;
        let direct = read_aligned_at(path, offset, buf.as_mut())?;
        let out = buf.as_mut()[..bytes].to_vec();
        if direct {
            self.record_pinned_h2d(aligned);
        } else {
            self.record_staged_h2d(aligned);
        }
        Ok(out)
    }

    /// Deepest feasible path: aligned NVMe/host read into pinned memory, then
    /// H2D into a bound expert ring slot. Records `gpu_dma_*` when successful.
    pub fn fill_to_device(
        &self,
        probe: &DirectStorageProbe,
        path: &Path,
        offset: u64,
        bytes: usize,
        copy_h2d: impl FnOnce(&[u8]) -> Result<(), AllpacaError>,
    ) -> Result<bool, AllpacaError> {
        if !probe.available || bytes == 0 {
            return Ok(false);
        }
        let _ = self.try_enqueue(probe, path, offset, bytes);
        let host = self.stage_aligned_read(path, offset, bytes)?;
        copy_h2d(&host)?;
        self.gpu_dma.fetch_add(1, Ordering::Relaxed);
        self.gpu_dma_bytes
            .fetch_add(host.len() as u64, Ordering::Relaxed);
        Ok(true)
    }
}

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

struct AlignedHostBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedHostBuf {
    fn new(size: usize, align: usize) -> Result<Self, AllpacaError> {
        let size = size.max(align);
        let layout = std::alloc::Layout::from_size_align(size, align).map_err(|_| {
            AllpacaError::Fatal(FatalError::InvalidConfig(
                "aligned DirectStorage staging layout".into(),
            ))
        })?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(AllpacaError::Fatal(FatalError::OutOfMemory(format!(
                "aligned DirectStorage staging ({size} bytes)"
            ))));
        }
        Ok(Self { ptr, layout })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedHostBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

fn read_aligned_at(path: &Path, offset: u64, dest: &mut [u8]) -> Result<bool, AllpacaError> {
    if let Ok(mut file) = open_direct_reader(path) {
        if read_fully(&mut file, offset, dest).is_ok() {
            return Ok(true);
        }
    }
    let mut file = File::open(path)?;
    read_fully(&mut file, offset, dest)?;
    Ok(false)
}

fn read_fully(file: &mut File, offset: u64, dest: &mut [u8]) -> Result<(), AllpacaError> {
    file.seek(SeekFrom::Start(offset))?;
    let mut filled = 0usize;
    while filled < dest.len() {
        let n = file.read(&mut dest[filled..])?;
        if n == 0 {
            return Err(AllpacaError::Fatal(FatalError::ModelLoad(format!(
                "DirectStorage staging short read at offset {offset}"
            ))));
        }
        filled += n;
    }
    Ok(())
}

#[cfg(windows)]
fn open_direct_reader(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING)
        .open(path)
}

#[cfg(not(windows))]
fn open_direct_reader(path: &Path) -> std::io::Result<File> {
    // O_DIRECT is platform-specific; fall through to buffered open on failure sites.
    File::open(path)
}

/// Probe the host for a DirectStorage-capable stack.
pub fn probe_direct_storage() -> DirectStorageProbe {
    #[cfg(windows)]
    {
        return probe_windows();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        return probe_linux_gds();
    }
    #[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
    {
        DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "platform has no DirectStorage/GDS probe".into(),
            factory_resolved: false,
        }
    }
}

#[cfg(windows)]
fn probe_windows() -> DirectStorageProbe {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
        fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
        fn GetProcAddress(
            module: *mut std::ffi::c_void,
            name: *const u8,
        ) -> *mut std::ffi::c_void;
    }
    let wide: Vec<u16> = "dstorage.dll"
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
    if handle.is_null() {
        return DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "dstorage.dll not found; keep host-staged NVMe".into(),
            factory_resolved: false,
        };
    }
    let factory = unsafe { GetProcAddress(handle, b"DStorageGetFactory\0".as_ptr()) };
    let factory_resolved = !factory.is_null();
    unsafe {
        let _ = FreeLibrary(handle);
    }
    let detail = if factory_resolved {
        "dstorage.dll loadable; DStorageGetFactory resolved (no COM/NuGet)"
    } else {
        "dstorage.dll loadable; DStorageGetFactory missing"
    };
    DirectStorageProbe {
        available: true,
        backend: DirectStorageBackend::WindowsDirectStorage,
        detail: detail.into(),
        factory_resolved,
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn probe_linux_gds() -> DirectStorageProbe {
    let markers = [
        "/dev/nvidia-fs0",
        "/usr/local/cuda/gds",
        "/etc/cufile.json",
    ];
    if markers.iter().any(|p| Path::new(p).exists()) {
        DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::LinuxGds,
            detail: "GDS marker present".into(),
            factory_resolved: false,
        }
    } else {
        DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "no GDS marker; keep host-staged NVMe".into(),
            factory_resolved: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn probe_returns_structured_backend() {
        let p = probe_direct_storage();
        assert!(!p.detail.is_empty(), "{p:?}");
        match p.backend {
            DirectStorageBackend::HostStaging => {
                assert!(!p.available);
                assert!(!p.factory_resolved);
            }
            DirectStorageBackend::WindowsDirectStorage | DirectStorageBackend::LinuxGds => {
                assert!(p.available);
            }
        }
    }

    #[test]
    fn enqueue_records_only_when_available() {
        let q = DirectStorageQueue::new();
        let off = DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "off".into(),
            factory_resolved: false,
        };
        assert!(!q.try_enqueue(&off, Path::new("x.bin"), 0, 4096));
        assert_eq!(q.enqueued(), 0);

        let on = DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::WindowsDirectStorage,
            detail: "on".into(),
            factory_resolved: true,
        };
        assert!(q.try_enqueue(&on, &PathBuf::from("experts.bin"), 0, 8192));
        assert_eq!(q.enqueued(), 1);
        assert_eq!(q.bytes(), 8192);
    }

    #[test]
    fn enqueue_or_stage_falls_back_to_staged() {
        let q = DirectStorageQueue::new();
        let off = DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "off".into(),
            factory_resolved: false,
        };
        q.enqueue_or_stage(&off, Path::new("experts.bin"), 4096, 8192);
        assert_eq!(q.enqueued(), 0);
        assert_eq!(q.staged(), 1);
        assert_eq!(q.staged_bytes(), 8192);
    }

    #[test]
    fn stage_aligned_read_works_without_dstorage_dll() {
        let path = std::env::temp_dir().join(format!(
            "allpaca-ds-stage-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let payload = vec![0xABu8; DIRECT_ALIGN];
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        let q = DirectStorageQueue::new();
        let got = q.stage_aligned_read(&path, 0, DIRECT_ALIGN).unwrap();
        assert_eq!(got, payload);
        // Direct I/O or buffered — either records a host-side transfer.
        assert!(
            q.pinned() + q.staged() >= 1,
            "pinned={} staged={}",
            q.pinned(),
            q.staged()
        );
        assert!(q.pinned_bytes() + q.staged_bytes() >= DIRECT_ALIGN as u64);
        assert_eq!(q.enqueued(), 0, "no DMA intent without probe");
        let _ = std::fs::remove_file(&path);
    }
}
