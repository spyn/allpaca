//! DirectStorage / GPU-direct NVMe path (GEM).
//!
//! Today expert and KV cold tiers bounce NVMe → host RAM → PCIe → VRAM.
//! DirectStorage (Windows) / GDS (Linux) can DMA into device memory and skip
//! the host bounce for large expert fills.
//!
//! Vertical:
//! 1. Probe `dstorage.dll` and resolve `DStorageGetFactory`.
//! 2. Allocate expert weights as a shareable D3D12 buffer imported into HIP.
//! 3. Submit DirectStorage file→buffer requests for NVMe expert fills.
//! 4. Fall back to aligned Direct I/O (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`) + H2D
//!    when any GPU-DMA layer is missing. Linux `O_DIRECT` is best-effort: tmpfs
//!    and some network FS return EINVAL and we reopen buffered.

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
    /// Windows: `DStorageGetFactory` export resolved.
    pub factory_resolved: bool,
    /// Windows: `DStorageGetFactory` COM call returned a factory pointer.
    pub factory_bound: bool,
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
    /// DirectStorage file-coordinate requests accepted by the probe.
    enqueued: AtomicU64,
    bytes: AtomicU64,
    /// Buffered host-staged fills (no Direct I/O).
    staged: AtomicU64,
    staged_bytes: AtomicU64,
    /// Aligned Direct I/O into pinned-shaped host memory (H2D source).
    pinned: AtomicU64,
    pinned_bytes: AtomicU64,
    /// Completed NVMe→D3D12/HIP fills with no host byte buffer.
    gpu_dma: AtomicU64,
    gpu_dma_bytes: AtomicU64,
    /// Successful COM factory binds (`DStorageGetFactory`).
    com_binds: AtomicU64,
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

    pub fn com_binds(&self) -> u64 {
        self.com_binds.load(Ordering::Relaxed)
    }

    pub fn record_com_bind(&self) {
        self.com_binds.fetch_add(1, Ordering::Relaxed);
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
        if probe.factory_bound {
            self.record_com_bind();
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

    /// Prefer DirectStorage into a D3D12/HIP allocation, then truthfully fall
    /// back to aligned NVMe/host read plus H2D.
    pub fn fill_to_device(
        &self,
        probe: &DirectStorageProbe,
        path: &Path,
        offset: u64,
        bytes: usize,
        direct_dma: impl FnOnce() -> Result<bool, AllpacaError>,
        copy_h2d: impl FnOnce(&[u8]) -> Result<(), AllpacaError>,
    ) -> Result<bool, AllpacaError> {
        if !probe.available || bytes == 0 {
            return Ok(false);
        }
        let _ = self.try_enqueue(probe, path, offset, bytes);
        if matches!(direct_dma(), Ok(true)) {
            self.gpu_dma.fetch_add(1, Ordering::Relaxed);
            self.gpu_dma_bytes
                .fetch_add(bytes as u64, Ordering::Relaxed);
            return Ok(true);
        }
        let host = self.stage_aligned_read(path, offset, bytes)?;
        copy_h2d(&host)?;
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
/// Linux `O_DIRECT` on x86_64/aarch64. Declared locally to avoid a libc crate.
#[cfg(target_os = "linux")]
const LINUX_O_DIRECT: i32 = 0o40000;

fn read_aligned_at(path: &Path, offset: u64, dest: &mut [u8]) -> Result<bool, AllpacaError> {
    if let Ok(mut file) = open_direct(path, false) {
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

/// Aligned Direct I/O open (Windows no-buffering / Linux `O_DIRECT`).
/// Offset, length, and the caller's buffer must be [`DIRECT_ALIGN`]-aligned.
pub(crate) fn open_direct(path: &Path, write: bool) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        let mut opts = OpenOptions::new();
        opts.read(true).custom_flags(FILE_FLAG_NO_BUFFERING);
        if write {
            opts.write(true);
        }
        return opts.open(path);
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut opts = OpenOptions::new();
        opts.read(true).custom_flags(LINUX_O_DIRECT);
        if write {
            opts.write(true);
        }
        return opts.open(path);
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = (path, write);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Direct I/O is not available on this platform",
        ))
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
            factory_resolved: false,
            factory_bound: false,
        }
    }
}

/// Public GUID for `IDStorageFactory` (dstorage.h).
#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

/// From stable Microsoft DirectStorage 1.3.0 `dstorage.h`: IDStorageFactory.
const IID_IDSTORAGE_FACTORY: Guid = Guid {
    data1: 0x69_24_ea_0c,
    data2: 0xc3_cd,
    data3: 0x48_26,
    data4: [0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1],
};

#[cfg(windows)]
fn probe_windows() -> DirectStorageProbe {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
        fn FreeLibrary(module: *mut std::ffi::c_void) -> i32;
        fn GetProcAddress(module: *mut std::ffi::c_void, name: *const u8) -> *mut std::ffi::c_void;
    }
    type DStorageGetFactoryFn =
        unsafe extern "system" fn(riid: *const Guid, ppv: *mut *mut std::ffi::c_void) -> i32;

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
            factory_bound: false,
        };
    }
    let factory = unsafe { GetProcAddress(handle, b"DStorageGetFactory\0".as_ptr()) };
    let factory_resolved = !factory.is_null();
    let mut factory_bound = false;
    if factory_resolved {
        let get_factory: DStorageGetFactoryFn = unsafe { std::mem::transmute(factory) };
        let mut ppv: *mut std::ffi::c_void = std::ptr::null_mut();
        let hr = unsafe { get_factory(&IID_IDSTORAGE_FACTORY, &mut ppv) };
        if hr >= 0 && !ppv.is_null() {
            factory_bound = true;
            // IUnknown::Release at vtable[2]
            unsafe {
                let vtable = *(ppv as *const *const usize);
                let release: unsafe extern "system" fn(*mut std::ffi::c_void) -> u32 =
                    std::mem::transmute(*vtable.add(2));
                let _ = release(ppv);
            }
        }
    }
    unsafe {
        let _ = FreeLibrary(handle);
    }
    let (available, backend, detail) = match (factory_resolved, factory_bound) {
        (true, true) => (
            true,
            DirectStorageBackend::WindowsDirectStorage,
            "DirectStorage 1.3 factory bound; D3D12/HIP DMA attempted per allocation",
        ),
        (true, false) => (
            false,
            DirectStorageBackend::HostStaging,
            "dstorage.dll export resolved but factory bind failed; keep host staging",
        ),
        _ => (
            false,
            DirectStorageBackend::HostStaging,
            "dstorage.dll lacks DStorageGetFactory; keep host staging",
        ),
    };
    DirectStorageProbe {
        available,
        backend,
        detail: detail.into(),
        factory_resolved,
        factory_bound,
    }
}

#[cfg(all(unix, not(target_os = "macos")))]
fn probe_linux_gds() -> DirectStorageProbe {
    let markers = ["/dev/nvidia-fs0", "/usr/local/cuda/gds", "/etc/cufile.json"];
    if markers.iter().any(|p| Path::new(p).exists()) {
        DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::LinuxGds,
            detail: "GDS marker present".into(),
            factory_resolved: false,
            factory_bound: false,
        }
    } else {
        DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "no GDS marker; keep host-staged NVMe".into(),
            factory_resolved: false,
            factory_bound: false,
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
                assert!(!p.factory_bound);
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
            factory_bound: false,
        };
        assert!(!q.try_enqueue(&off, Path::new("x.bin"), 0, 4096));
        assert_eq!(q.enqueued(), 0);

        let on = DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::WindowsDirectStorage,
            detail: "on".into(),
            factory_resolved: true,
            factory_bound: true,
        };
        assert!(q.try_enqueue(&on, &PathBuf::from("experts.bin"), 0, 8192));
        assert_eq!(q.enqueued(), 1);
        assert_eq!(q.bytes(), 8192);
        assert_eq!(q.com_binds(), 1);
    }

    #[test]
    fn enqueue_or_stage_falls_back_to_staged() {
        let q = DirectStorageQueue::new();
        let off = DirectStorageProbe {
            available: false,
            backend: DirectStorageBackend::HostStaging,
            detail: "off".into(),
            factory_resolved: false,
            factory_bound: false,
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

    #[cfg(target_os = "linux")]
    fn linux_fs_type(path: &Path) -> Option<String> {
        let out = std::process::Command::new("stat")
            .args(["-f", "-c", "%T"])
            .arg(path)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        Some(
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .to_ascii_lowercase(),
        )
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn stage_aligned_read_uses_o_direct_on_block_fs() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/allpaca-odirect");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "ds-odirect-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let payload = vec![0x11u8; DIRECT_ALIGN];
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        let q = DirectStorageQueue::new();
        let got = q.stage_aligned_read(&path, 0, DIRECT_ALIGN).unwrap();
        assert_eq!(got, payload);
        let fs = linux_fs_type(&path).unwrap_or_default();
        let expects_direct = matches!(
            fs.as_str(),
            "ext2" | "ext3" | "ext4" | "ext2/ext3" | "xfs" | "btrfs" | "f2fs"
        );
        if expects_direct {
            assert_eq!(
                q.pinned(),
                1,
                "O_DIRECT must pin-stage on {fs}; staged={}",
                q.staged()
            );
            assert_eq!(q.staged(), 0);
        } else {
            eprintln!(
                "skip O_DIRECT pin assert on fs={fs:?} path={}",
                path.display()
            );
            assert!(q.pinned() + q.staged() >= 1);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fill_to_device_falls_back_to_h2d_when_direct_dma_is_unavailable() {
        let path = std::env::temp_dir().join(format!(
            "allpaca-ds-fill-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let payload = vec![0xCDu8; DIRECT_ALIGN];
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        let q = DirectStorageQueue::new();
        let probe = DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::WindowsDirectStorage,
            detail: "test".into(),
            factory_resolved: true,
            factory_bound: true,
        };
        let mut dest = vec![0u8; DIRECT_ALIGN];
        let ok = q
            .fill_to_device(
                &probe,
                &path,
                0,
                DIRECT_ALIGN,
                || Ok(false),
                |src| {
                    dest.copy_from_slice(src);
                    Ok(())
                },
            )
            .unwrap();
        assert!(ok);
        assert_eq!(dest, payload);
        assert_eq!(q.gpu_dma(), 0);
        assert_eq!(q.gpu_dma_bytes(), 0);
        assert!(q.pinned() + q.staged() >= 1);
        assert!(q.enqueued() >= 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fill_to_device_prefers_direct_dma_and_skips_host_staging() {
        let q = DirectStorageQueue::new();
        let probe = DirectStorageProbe {
            available: true,
            backend: DirectStorageBackend::WindowsDirectStorage,
            detail: "test".into(),
            factory_resolved: true,
            factory_bound: true,
        };
        let h2d_called = std::cell::Cell::new(false);
        let ok = q
            .fill_to_device(
                &probe,
                Path::new("experts.bin"),
                4096,
                DIRECT_ALIGN,
                || Ok(true),
                |_| {
                    h2d_called.set(true);
                    Ok(())
                },
            )
            .unwrap();
        assert!(ok);
        assert!(!h2d_called.get());
        assert_eq!(q.gpu_dma(), 1);
        assert_eq!(q.gpu_dma_bytes(), DIRECT_ALIGN as u64);
        assert_eq!(q.pinned() + q.staged(), 0);
    }
}
