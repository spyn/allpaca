//! DirectStorage / GPU-direct NVMe path (GEM).
//!
//! Today expert and KV cold tiers bounce NVMe → host RAM → PCIe → VRAM.
//! DirectStorage (Windows) / GDS (Linux) can DMA into device memory and skip
//! the host bounce for large expert fills.
//!
//! Vertical: capability probe + file-coordinate enqueue from packed expert NVMe
//! reads (`ds_enqueued` / `ds_bytes`). Real NVMe→VRAM DMA still needs the SDK;
//! when the probe is hot we record DMA intents with path/offset/length.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Result of probing host DirectStorage / GDS availability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirectStorageProbe {
    pub available: bool,
    pub backend: DirectStorageBackend,
    pub detail: String,
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

/// Records intended NVMe→device transfers (probe-gated) without issuing DMA yet.
#[derive(Debug, Default)]
pub struct DirectStorageQueue {
    enqueued: AtomicU64,
    bytes: AtomicU64,
    /// Host-staged fills counted when DS is unavailable (or after H2D).
    staged: AtomicU64,
    staged_bytes: AtomicU64,
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

    /// Host-staged NVMe→RAM→VRAM fill (no DS DMA).
    pub fn record_staged_h2d(&self, bytes: usize) {
        if bytes == 0 {
            return;
        }
        self.staged.fetch_add(1, Ordering::Relaxed);
        self.staged_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
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
        }
    }
}

#[cfg(windows)]
fn probe_windows() -> DirectStorageProbe {
    // LoadLibraryW("dstorage.dll") without linking the SDK.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
        fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
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
        };
    }
    unsafe {
        let _ = FreeLibrary(handle);
    }
    DirectStorageProbe {
        available: true,
        backend: DirectStorageBackend::WindowsDirectStorage,
        detail: "dstorage.dll loadable".into(),
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn probe_linux_gds() -> DirectStorageProbe {
    // GDS typically exposes nvidia-fs / cufile; we only look for a marker path.
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
        }
    } else {
        DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "no GDS marker; keep host-staged NVMe".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_structured_backend() {
        let p = probe_direct_storage();
        assert!(!p.detail.is_empty(), "{p:?}");
        match p.backend {
            DirectStorageBackend::HostStaging => assert!(!p.available),
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
        };
        assert!(!q.try_enqueue(&off, Path::new("x.bin"), 0, 4096));
        assert_eq!(q.enqueued(), 0);

        let on = DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::WindowsDirectStorage,
            detail: "on".into(),
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
        };
        q.enqueue_or_stage(&off, Path::new("experts.bin"), 4096, 8192);
        assert_eq!(q.enqueued(), 0);
        assert_eq!(q.staged(), 1);
        assert_eq!(q.staged_bytes(), 8192);
    }
}
