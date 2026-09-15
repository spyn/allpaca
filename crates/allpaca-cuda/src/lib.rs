//! CUDA kernel FFI. Feature-gated like `allpaca-hip`.
//!
//! `--features cuda` enables the path when a toolkit is found at build time
//! (`CUDA_PATH` / `nvcc`). Without a driver or GPU, `device_count()` is 0 and
//! device GEMM returns unavailable — the crate still compiles and links.
//!
//! Symbols are `allpaca_cuda_*` so HIP and CUDA ABIs do not collide (ADR 0001).
//!
//! GPU ISA helpers live in [`cuda_arch`] (also linked from `build.rs`).

mod cuda_arch;

use allpaca_types::{AllpacaError, DegradedError, DeviceInfo, DeviceKind};
use std::sync::atomic::{AtomicU64, Ordering};

pub const ALLPACA_OK: i32 = 0;
pub const ALLPACA_ERR_NOT_IMPLEMENTED: i32 = 1;
pub const ALLPACA_ERR_INVALID: i32 = 2;
pub const ALLPACA_ERR_DEVICE: i32 = 3;

pub const MEMCPY_H2D: i32 = 0;
pub const MEMCPY_D2H: i32 = 1;
pub const MEMCPY_D2D: i32 = 2;
pub const MEMCPY_DEFAULT: i32 = 3;
pub const MEMCPY_D2H_ASYNC: i32 = 4;

#[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_char;

    extern "C" {
        pub fn allpaca_cuda_device_count(count: *mut i32) -> i32;
        pub fn allpaca_cuda_device_props(
            id: i32,
            name: *mut c_char,
            name_len: i32,
            memory_bytes: *mut u64,
            bandwidth_gbps: *mut f64,
        ) -> i32;
        pub fn allpaca_cuda_device_arch(
            id: i32,
            arch: *mut c_char,
            arch_len: i32,
            l2_bytes: *mut i32,
        ) -> i32;
        pub fn allpaca_cuda_malloc(ptr: *mut *mut c_void, bytes: usize, device: i32) -> i32;
        pub fn allpaca_cuda_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_cuda_free_async(ptr: *mut c_void, stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_host_malloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn allpaca_cuda_host_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_cuda_memcpy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_set_device(device: i32) -> i32;
        pub fn allpaca_cuda_mem_info(
            device: i32,
            free_bytes: *mut u64,
            total_bytes: *mut u64,
        ) -> i32;
        pub fn allpaca_cuda_stream_synchronize(stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_stream_create(stream: *mut *mut c_void) -> i32;
        pub fn allpaca_cuda_stream_destroy(stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_stream_wait_event(stream: *mut c_void, event: *mut c_void) -> i32;
        pub fn allpaca_cuda_event_create(event: *mut *mut c_void) -> i32;
        pub fn allpaca_cuda_event_destroy(event: *mut c_void) -> i32;
        pub fn allpaca_cuda_event_record(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_gemm_nt(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_residual_add(
            x: *const f32,
            y: *const f32,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_rmsnorm(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            n: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_silu_mul(
            gate: *const f32,
            up: *const f32,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_embed_gather(
            table: *const f32,
            token: i32,
            out: *mut f32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_rope(
            x: *mut f32,
            n_heads: i32,
            head_dim: i32,
            pos: i32,
            theta: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_attention(
            q: *const f32,
            k: *const f32,
            v: *const f32,
            out: *mut f32,
            n_q: i32,
            n_kv: i32,
            n_heads: i32,
            n_kv_heads: i32,
            head_dim: i32,
            causal: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn status(rc: i32, op: &str) -> Result<(), AllpacaError> {
    if rc == ALLPACA_OK {
        Ok(())
    } else {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            format!("CUDA {op} failed rc={rc}"),
        )))
    }
}

/// True when this build linked a CUDA-compiled kernel library.
pub fn runtime_linked() -> bool {
    cfg!(all(feature = "cuda", allpaca_cuda_runtime))
}

pub fn device_count() -> i32 {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        unsafe {
            let mut count = 0i32;
            if ffi::allpaca_cuda_device_count(&mut count) != ALLPACA_OK {
                return 0;
            }
            count.max(0)
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        0
    }
}

/// Enumerate CUDA devices. Empty when feature off, no toolkit, no driver, or no GPU.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let count = device_count();
    if count <= 0 {
        return Vec::new();
    }
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut out = Vec::with_capacity(count as usize);
        for id in 0..count {
            let mut name_buf = [0i8; 256];
            let mut mem = 0u64;
            let mut bw = 0.0f64;
            let rc = unsafe {
                ffi::allpaca_cuda_device_props(id, name_buf.as_mut_ptr(), 256, &mut mem, &mut bw)
            };
            if rc != ALLPACA_OK {
                continue;
            }
            let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            out.push(DeviceInfo {
                id,
                kind: DeviceKind::CudaGpu,
                name,
                memory_bytes: mem,
                bandwidth_gbps: bw,
                peers: Vec::new(),
            });
        }
        out
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = count;
        Vec::new()
    }
}

/// Live compute capability (`sm_86`, `sm_120`, …) and L2 cache size in bytes.
pub fn device_sm_arch(id: i32) -> Option<(String, usize)> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut buf = [0i8; 32];
        let mut l2 = 0i32;
        let rc = unsafe { ffi::allpaca_cuda_device_arch(id, buf.as_mut_ptr(), 32, &mut l2) };
        if rc != ALLPACA_OK {
            return None;
        }
        let arch = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if arch.is_empty() || !arch.starts_with("sm_") {
            return None;
        }
        Some((arch, l2.max(0) as usize))
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = id;
        None
    }
}

pub fn set_device(device: i32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_set_device(device) };
        status(rc, "set_device")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA set_device unavailable".into(),
        )))
    }
}

/// Driver-measured `(free, total)` VRAM bytes for `device`.
///
/// Like HIP [`allpaca_hip::device_memory_info`]: board-wide (other processes
/// included). Sample off the decode hot path.
pub fn device_memory_info(device: i32) -> Result<(u64, u64), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut free = 0u64;
        let mut total = 0u64;
        let rc = unsafe { ffi::allpaca_cuda_mem_info(device, &mut free, &mut total) };
        status(rc, "mem_info")?;
        Ok((free, total))
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA mem_info unavailable".into(),
        )))
    }
}

pub fn device_malloc(bytes: usize, device: i32) -> Result<*mut std::ffi::c_void, AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_cuda_malloc(&mut ptr, bytes, device) };
        status(rc, "malloc")?;
        if ptr.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA malloc returned null".into(),
            )));
        }
        Ok(ptr)
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (bytes, device);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA malloc unavailable".into(),
        )))
    }
}

pub fn device_free(ptr: *mut std::ffi::c_void) {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if !ptr.is_null() {
            unsafe {
                let _ = ffi::allpaca_cuda_free(ptr);
            }
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = ptr;
    }
}

/// Stream-ordered device free. Null `stream` falls back to synchronous [`device_free`].
pub fn device_free_async(ptr: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if !ptr.is_null() {
            unsafe {
                let _ = ffi::allpaca_cuda_free_async(ptr, stream);
            }
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (ptr, stream);
    }
}

/// Page-locked host allocation for faster H2D/D2H (mirrors HIP `host_malloc`).
pub fn host_malloc(bytes: usize) -> Result<*mut u8, AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_cuda_host_malloc(&mut ptr, bytes) };
        status(rc, "host_malloc")?;
        if ptr.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA host_malloc returned null".into(),
            )));
        }
        Ok(ptr.cast())
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = bytes;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA host_malloc unavailable".into(),
        )))
    }
}

pub fn host_free(ptr: *mut u8) {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if !ptr.is_null() {
            unsafe {
                let _ = ffi::allpaca_cuda_host_free(ptr.cast());
            }
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = ptr;
    }
}

pub fn memcpy(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
    kind: i32,
) -> Result<(), AllpacaError> {
    memcpy_on(dst, src, bytes, kind, std::ptr::null_mut())
}

/// Copy on `stream` (null = default / sync path for non-async kinds).
pub fn memcpy_on(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
    kind: i32,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_memcpy(dst, src, bytes, kind, stream) };
        status(rc, "memcpy")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (dst, src, bytes, kind, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA memcpy unavailable".into(),
        )))
    }
}

/// Wait for queued work on the default (null) stream / device.
pub fn stream_synchronize() -> Result<(), AllpacaError> {
    // SAFETY: null stream is the documented default-device sync path.
    unsafe { stream_synchronize_on(std::ptr::null_mut()) }
}

/// Wait for queued work on `stream` (null synchronizes the device).
///
/// # Safety
/// `stream` must be null or a stream this process still owns.
pub unsafe fn stream_synchronize_on(stream: *mut std::ffi::c_void) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_stream_synchronize(stream) };
        status(rc, "stream_synchronize")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = stream;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA stream_synchronize unavailable".into(),
        )))
    }
}

fn stream_create_raw() -> Result<*mut std::ffi::c_void, AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut stream = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_cuda_stream_create(&mut stream) };
        status(rc, "stream_create")?;
        Ok(stream)
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA stream_create unavailable".into(),
        )))
    }
}

fn stream_destroy_raw(stream: *mut std::ffi::c_void) {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if !stream.is_null() {
            unsafe {
                let _ = ffi::allpaca_cuda_stream_destroy(stream);
            }
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = stream;
    }
}

fn event_create_raw() -> Result<*mut std::ffi::c_void, AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut event = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_cuda_event_create(&mut event) };
        status(rc, "event_create")?;
        Ok(event)
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA event_create unavailable".into(),
        )))
    }
}

/// Owned non-blocking CUDA stream (compute or H2D side stream).
pub struct CudaStream {
    stream: *mut std::ffi::c_void,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self, AllpacaError> {
        Ok(Self {
            stream: stream_create_raw()?,
        })
    }

    pub fn raw(&self) -> *mut std::ffi::c_void {
        self.stream
    }

    pub fn synchronize(&self) -> Result<(), AllpacaError> {
        unsafe { stream_synchronize_on(self.stream) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        stream_destroy_raw(self.stream);
        self.stream = std::ptr::null_mut();
    }
}

/// Side stream for overlapping H2D with compute; join via event (HIP `H2dStream`).
pub struct H2dStream {
    stream: *mut std::ffi::c_void,
    event: *mut std::ffi::c_void,
}

unsafe impl Send for H2dStream {}
unsafe impl Sync for H2dStream {}

impl H2dStream {
    pub fn new() -> Result<Self, AllpacaError> {
        let stream = stream_create_raw()?;
        let event = match event_create_raw() {
            Ok(e) => e,
            Err(e) => {
                stream_destroy_raw(stream);
                return Err(e);
            }
        };
        Ok(Self { stream, event })
    }

    pub fn raw(&self) -> *mut std::ffi::c_void {
        self.stream
    }

    /// Record on this stream, then make the null compute stream wait (host does not block).
    pub fn join_compute(&self) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            let rc = unsafe { ffi::allpaca_cuda_event_record(self.event, self.stream) };
            status(rc, "h2d_event_record")?;
            let rc =
                unsafe { ffi::allpaca_cuda_stream_wait_event(std::ptr::null_mut(), self.event) };
            status(rc, "h2d_stream_wait_event")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA stream_wait_event unavailable".into(),
            )))
        }
    }

    pub fn synchronize(&self) -> Result<(), AllpacaError> {
        unsafe { stream_synchronize_on(self.stream) }
    }
}

impl Drop for H2dStream {
    fn drop(&mut self) {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        unsafe {
            let _ = ffi::allpaca_cuda_event_destroy(self.event);
            let _ = ffi::allpaca_cuda_stream_destroy(self.stream);
        }
        self.event = std::ptr::null_mut();
        self.stream = std::ptr::null_mut();
    }
}

/// Page-locked host buffer for staging DMA.
pub struct PinnedHostBuffer {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    pub fn new(bytes: usize) -> Result<Self, AllpacaError> {
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty pinned host buffer".into(),
            )));
        }
        Ok(Self {
            ptr: host_malloc(bytes)?,
            bytes,
        })
    }

    /// Allocate and copy `src` into pinned memory (CPU memcpy).
    pub fn from_slice(src: &[u8]) -> Result<Self, AllpacaError> {
        let buf = Self::new(src.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr, src.len());
        }
        Ok(buf)
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.bytes) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.bytes) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        host_free(self.ptr);
        self.ptr = std::ptr::null_mut();
    }
}

/// RAII device allocation. Optional stream-ordered free avoids device-wide sync on drop.
pub struct DeviceBuffer {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
    free_stream: *mut std::ffi::c_void,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn alloc(bytes: usize, device: i32) -> Result<Self, AllpacaError> {
        Ok(Self {
            ptr: device_malloc(bytes, device)?,
            bytes,
            free_stream: std::ptr::null_mut(),
        })
    }

    /// Free via `cudaFreeAsync` on `stream` when dropped (null = sync free).
    pub fn with_async_free(mut self, stream: *mut std::ffi::c_void) -> Self {
        self.free_stream = stream;
        self
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn as_f32(&self) -> *mut f32 {
        self.ptr.cast()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Release ownership without freeing (caller must free).
    pub fn into_raw(mut self) -> *mut std::ffi::c_void {
        let p = self.ptr;
        self.ptr = std::ptr::null_mut();
        p
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if self.free_stream.is_null() {
            device_free(self.ptr);
        } else {
            device_free_async(self.ptr, self.free_stream);
        }
        self.ptr = std::ptr::null_mut();
    }
}

pub fn gemm_nt_device(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_gemm_nt(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt")?;
        // Standalone / RPC contract: complete before return.
        let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
        status(sync, "stream_synchronize")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA gemm_nt unavailable".into(),
        )))
    }
}

/// Host-pointer F32 NT GEMM via CUDA when a device exists.
pub fn gemm_nt(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    if device_count() <= 0 {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA gemm_nt: no device".into(),
        )));
    }
    if m <= 0 || n <= 0 || k <= 0 {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "invalid GEMM shape".into(),
        )));
    }
    let a_n = (m as usize) * (k as usize);
    let b_n = (n as usize) * (k as usize);
    let c_n = (m as usize) * (n as usize);
    if a.len() < a_n || b.len() < b_n || c.len() < c_n {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "GEMM buffer size mismatch".into(),
        )));
    }
    let da = device_malloc(a_n * 4, 0)?;
    let db = device_malloc(b_n * 4, 0)?;
    let dc = device_malloc(c_n * 4, 0)?;
    set_device(0)?;
    let result = (|| {
        memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
        memcpy(db, b.as_ptr().cast(), b_n * 4, MEMCPY_H2D)?;
        gemm_nt_device(da.cast(), db.cast(), dc.cast(), m, n, k)?;
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
    })();
    device_free(da);
    device_free(db);
    device_free(dc);
    result
}

/// Minimal CUDA workspace: scratch buffer + owned compute stream + F32 GEMM.
/// Kernels enqueue on the workspace stream; call [`CudaWorkspace::synchronize`]
/// (or a host roundtrip) to batch-wait — no per-op device sync.
pub struct CudaWorkspace {
    pub device: i32,
    scratch: *mut std::ffi::c_void,
    scratch_bytes: usize,
    stream: *mut std::ffi::c_void,
    activation_h2d: AtomicU64,
    activation_d2h: AtomicU64,
}

unsafe impl Send for CudaWorkspace {}
unsafe impl Sync for CudaWorkspace {}

impl CudaWorkspace {
    pub fn new(device: i32, scratch_bytes: usize) -> Result<Self, AllpacaError> {
        if device_count() <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA workspace: no device".into(),
            )));
        }
        let scratch_bytes = scratch_bytes.max(4096);
        set_device(device)?;
        let stream = stream_create_raw()?;
        let scratch = match device_malloc(scratch_bytes, device) {
            Ok(p) => p,
            Err(e) => {
                stream_destroy_raw(stream);
                return Err(e);
            }
        };
        Ok(Self {
            device,
            scratch,
            scratch_bytes,
            stream,
            activation_h2d: AtomicU64::new(0),
            activation_d2h: AtomicU64::new(0),
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.scratch_bytes as u64
    }

    pub fn activation_h2d_bytes(&self) -> u64 {
        self.activation_h2d.load(Ordering::Relaxed)
    }

    pub fn activation_d2h_bytes(&self) -> u64 {
        self.activation_d2h.load(Ordering::Relaxed)
    }

    /// Raw compute stream handle for enqueueing work.
    pub fn raw_stream(&self) -> *mut std::ffi::c_void {
        self.stream
    }

    /// Wait for all work previously enqueued on this workspace stream.
    pub fn synchronize(&self) -> Result<(), AllpacaError> {
        unsafe { stream_synchronize_on(self.stream) }
    }

    pub fn gemm_nt_dev(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_gemm_nt(a, b, c, m, n, k, self.stream) };
            status(rc, "gemm_nt")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (a, b, c, m, n, k);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gemm_nt unavailable".into(),
            )))
        }
    }

    pub fn residual_add_dev(
        &self,
        x: *const f32,
        y: *const f32,
        out: *mut f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_residual_add(x, y, out, n, self.stream) };
            status(rc, "residual_add")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, y, out, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA residual_add unavailable".into(),
            )))
        }
    }

    pub fn rmsnorm_dev(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_rmsnorm(x, w, out, n, eps, self.stream) };
            status(rc, "rmsnorm")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, w, out, n, eps);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA rmsnorm unavailable".into(),
            )))
        }
    }

    pub fn silu_mul_dev(
        &self,
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_silu_mul(gate, up, out, n, self.stream) };
            status(rc, "silu_mul")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (gate, up, out, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA silu_mul unavailable".into(),
            )))
        }
    }

    pub fn embed_gather_dev(
        &self,
        table: *const f32,
        token: i32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if table.is_null() || out.is_null() || dim <= 0 || token < 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA embed_gather".into(),
                )));
            }
            set_device(self.device)?;
            let rc =
                unsafe { ffi::allpaca_cuda_embed_gather(table, token, out, dim, self.stream) };
            status(rc, "embed_gather")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (table, token, out, dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA embed_gather unavailable".into(),
            )))
        }
    }

    pub fn rope_dev(
        &self,
        x: *mut f32,
        n_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if x.is_null() || n_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA rope".into(),
                )));
            }
            set_device(self.device)?;
            let rc =
                unsafe { ffi::allpaca_cuda_rope(x, n_heads, head_dim, pos, theta, self.stream) };
            status(rc, "rope")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, n_heads, head_dim, pos, theta);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA rope unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_dev(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if q.is_null()
                || k.is_null()
                || v.is_null()
                || out.is_null()
                || n_q <= 0
                || n_kv <= 0
                || n_heads <= 0
                || n_kv_heads <= 0
                || head_dim <= 0
                || n_heads % n_kv_heads != 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA attention".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_attention(
                    q,
                    k,
                    v,
                    out,
                    n_q,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    i32::from(causal),
                    self.stream,
                )
            };
            status(rc, "attention")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA attention unavailable".into(),
            )))
        }
    }

    /// Host roundtrip: gather one F32 embed row via device.
    pub fn embed_gather(
        &self,
        table: &[f32],
        token: i32,
        out: &mut [f32],
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if dim <= 0 || token < 0 || out.len() < dim as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed_gather buffers".into(),
            )));
        }
        let need = (token as usize + 1).saturating_mul(dim as usize);
        if table.len() < need {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed table too small".into(),
            )));
        }
        let d_table = self.upload_f32(table)?;
        let d_out = device_malloc((dim as usize) * 4, self.device)?;
        let result = (|| {
            self.embed_gather_dev(d_table, token, d_out.cast(), dim)?;
            memcpy_on(
                out.as_mut_ptr().cast(),
                d_out,
                (dim as usize) * 4,
                MEMCPY_D2H,
                self.stream,
            )?;
            self.synchronize()?;
            self.activation_d2h
                .fetch_add((dim as usize * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_table);
        self.free_device(d_out);
        result
    }

    /// Host roundtrip: in-place RoPE via device.
    pub fn rope(
        &self,
        x: &mut [f32],
        n_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        let n = (n_heads as usize).saturating_mul(head_dim as usize);
        if n == 0 || x.len() < n || head_dim % 2 != 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rope shape".into(),
            )));
        }
        let d_x = self.upload_f32(&x[..n])?;
        let result = (|| {
            self.rope_dev(d_x, n_heads, head_dim, pos, theta)?;
            memcpy_on(
                x.as_mut_ptr().cast(),
                d_x.cast(),
                n * 4,
                MEMCPY_D2H,
                self.stream,
            )?;
            self.synchronize()?;
            self.activation_d2h
                .fetch_add((n * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_x);
        result
    }

    /// Host roundtrip: attention via device.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        let kv_n = (n_kv as usize) * (n_kv_heads as usize) * (head_dim as usize);
        if q.len() < q_n || k.len() < kv_n || v.len() < kv_n || out.len() < q_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention buffer size mismatch".into(),
            )));
        }
        let d_q = self.upload_f32(&q[..q_n])?;
        let d_k = self.upload_f32(&k[..kv_n])?;
        let d_v = self.upload_f32(&v[..kv_n])?;
        let d_out = device_malloc(q_n * 4, self.device)?;
        let result = (|| {
            self.attention_dev(
                d_q, d_k, d_v, d_out.cast(), n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            )?;
            memcpy_on(out.as_mut_ptr().cast(), d_out, q_n * 4, MEMCPY_D2H, self.stream)?;
            self.synchronize()?;
            self.activation_d2h
                .fetch_add((q_n * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_q);
        self.free_ptr(d_k);
        self.free_ptr(d_v);
        self.free_device(d_out);
        result
    }

    /// Host Q + resident device K/V (no KV D2H). Used by CUDA decode pages.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_host_q_dev_kv(
        &self,
        q: &[f32],
        k_dev: *const f32,
        v_dev: *const f32,
        out: &mut [f32],
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        if q.len() < q_n || out.len() < q_n || k_dev.is_null() || v_dev.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention_host_q_dev_kv buffer mismatch".into(),
            )));
        }
        let d_q = self.upload_f32(&q[..q_n])?;
        let d_out = device_malloc(q_n * 4, self.device)?;
        let result = (|| {
            self.attention_dev(
                d_q, k_dev, v_dev, d_out.cast(), n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            )?;
            memcpy_on(out.as_mut_ptr().cast(), d_out, q_n * 4, MEMCPY_D2H, self.stream)?;
            self.synchronize()?;
            self.activation_d2h
                .fetch_add((q_n * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_q);
        self.free_device(d_out);
        result
    }

    /// Upload host F32 into a device blob (activation H2D accounting).
    /// Enqueues on the workspace stream without host-waiting; pair with kernels
    /// and a single [`synchronize`] / [`download_f32`] at the boundary.
    pub fn upload_f32(&self, host: &[f32]) -> Result<*mut f32, AllpacaError> {
        let bytes = host.len().saturating_mul(4);
        set_device(self.device)?;
        let ptr = device_malloc(bytes, self.device)?;
        if bytes >= 4096 {
            let pinned = PinnedHostBuffer::from_slice(unsafe {
                std::slice::from_raw_parts(host.as_ptr().cast(), bytes)
            })?;
            memcpy_on(
                ptr,
                pinned.as_ptr().cast(),
                bytes,
                MEMCPY_H2D,
                self.stream,
            )?;
            // Pinned lives until H2D completes — sync this staging only.
            self.synchronize()?;
        } else {
            memcpy_on(ptr, host.as_ptr().cast(), bytes, MEMCPY_H2D, self.stream)?;
        }
        self.activation_h2d
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(ptr.cast())
    }

    /// Download device F32 into host (activation D2H accounting).
    pub fn download_f32(&self, src: *const f32, dst: &mut [f32]) -> Result<(), AllpacaError> {
        if dst.is_empty() {
            return Ok(());
        }
        if src.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA D2H download".into(),
            )));
        }
        set_device(self.device)?;
        let bytes = dst.len().saturating_mul(4);
        if bytes >= 4096 {
            let pinned = PinnedHostBuffer::new(bytes)?;
            memcpy_on(
                pinned.as_mut_ptr().cast(),
                src.cast(),
                bytes,
                MEMCPY_D2H,
                self.stream,
            )?;
            self.synchronize()?;
            dst.copy_from_slice(unsafe {
                std::slice::from_raw_parts(pinned.as_ptr().cast::<f32>(), dst.len())
            });
        } else {
            memcpy_on(
                dst.as_mut_ptr().cast(),
                src.cast(),
                bytes,
                MEMCPY_D2H,
                self.stream,
            )?;
            self.synchronize()?;
        }
        self.activation_d2h
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    /// GEMM with host activation A and device-resident weight B → host C.
    pub fn gemm_nt_host_a_dev_b(
        &self,
        a: &[f32],
        b_dev: *const f32,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if b_dev.is_null() || m <= 0 || n <= 0 || k <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA gemm_nt_host_a_dev_b".into(),
            )));
        }
        let a_n = (m as usize).saturating_mul(k as usize);
        let c_n = (m as usize).saturating_mul(n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "GEMM buffer size mismatch".into(),
            )));
        }
        let da = self.upload_f32(&a[..a_n])?;
        let dc = device_malloc(c_n * 4, self.device)?;
        let result = (|| {
            self.gemm_nt_dev(da, b_dev, dc.cast(), m, n, k)?;
            self.download_f32(dc.cast(), &mut c[..c_n])
        })();
        self.free_ptr(da);
        self.free_device(dc);
        result
    }

    /// RMSNorm with host x and device-resident weight → host out.
    pub fn rmsnorm_host_x_dev_w(
        &self,
        x: &[f32],
        w_dev: *const f32,
        out: &mut [f32],
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if w_dev.is_null() || n <= 0 || x.len() < n as usize || out.len() < n as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA rmsnorm_host_x_dev_w".into(),
            )));
        }
        let dx = self.upload_f32(&x[..n as usize])?;
        let dout = device_malloc((n as usize) * 4, self.device)?;
        let result = (|| {
            self.rmsnorm_dev(dx, w_dev, dout.cast(), n, eps)?;
            self.download_f32(dout.cast(), &mut out[..n as usize])
        })();
        self.free_ptr(dx);
        self.free_device(dout);
        result
    }

    /// Embed gather from a device-resident table → host out (no table re-upload).
    pub fn embed_gather_dev_table(
        &self,
        table_dev: *const f32,
        token: i32,
        out: &mut [f32],
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if table_dev.is_null() || dim <= 0 || token < 0 || out.len() < dim as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA embed_gather_dev_table".into(),
            )));
        }
        let dout = device_malloc((dim as usize) * 4, self.device)?;
        let result = (|| {
            self.embed_gather_dev(table_dev, token, dout.cast(), dim)?;
            self.download_f32(dout.cast(), &mut out[..dim as usize])
        })();
        self.free_device(dout);
        result
    }

    /// Copy host F32 into an existing device buffer (vision soft embeds / soft_x).
    /// Mirrors [`allpaca_hip::HipWorkspace::copy_h2d_f32`]. Completes before return.
    pub fn copy_h2d_f32(&self, dst: *mut f32, src: &[f32]) -> Result<(), AllpacaError> {
        if src.is_empty() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA H2D copy".into(),
            )));
        }
        set_device(self.device)?;
        let bytes = src.len().saturating_mul(4);
        if bytes >= 4096 {
            let pinned = PinnedHostBuffer::from_slice(unsafe {
                std::slice::from_raw_parts(src.as_ptr().cast(), bytes)
            })?;
            memcpy_on(
                dst.cast(),
                pinned.as_ptr().cast(),
                bytes,
                MEMCPY_H2D,
                self.stream,
            )?;
        } else {
            memcpy_on(
                dst.cast(),
                src.as_ptr().cast(),
                bytes,
                MEMCPY_H2D,
                self.stream,
            )?;
        }
        self.synchronize()?;
        self.activation_h2d
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Stream-ordered free of a device pointer allocated for this workspace.
    pub fn free_ptr(&self, ptr: *mut f32) {
        self.free_device(ptr.cast());
    }

    pub fn free_device(&self, ptr: *mut std::ffi::c_void) {
        device_free_async(ptr, self.stream);
    }
}

impl Drop for CudaWorkspace {
    fn drop(&mut self) {
        let _ = unsafe { stream_synchronize_on(self.stream) };
        device_free(self.scratch);
        self.scratch = std::ptr::null_mut();
        stream_destroy_raw(self.stream);
        self.stream = std::ptr::null_mut();
    }
}

/// Packed F32 weight tensors uploaded once via H2D (not activation traffic).
pub struct CudaDeviceWeights {
    #[allow(dead_code)]
    device: i32,
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for CudaDeviceWeights {}
unsafe impl Sync for CudaDeviceWeights {}

impl CudaDeviceWeights {
    /// Concatenate host F32 slices into one device blob. Returns (weights, h2d bytes).
    pub fn upload(device: i32, slices: &[&[f32]]) -> Result<(Self, u64), AllpacaError> {
        let floats: usize = slices.iter().map(|s| s.len()).sum();
        let bytes = floats.saturating_mul(4);
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty CUDA weight upload".into(),
            )));
        }
        set_device(device)?;
        let ptr = device_malloc(bytes, device)?;
        let mut off = 0usize;
        // Stage through pinned host memory when the packed blob is large.
        if bytes >= 4096 {
            let mut pinned = PinnedHostBuffer::new(bytes)?;
            let dst = pinned.as_mut_slice();
            for slice in slices {
                if slice.is_empty() {
                    continue;
                }
                let n = slice.len() * 4;
                dst[off..off + n].copy_from_slice(unsafe {
                    std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), n)
                });
                off += n;
            }
            memcpy(ptr, pinned.as_ptr().cast(), bytes, MEMCPY_H2D)?;
        } else {
            for slice in slices {
                if slice.is_empty() {
                    continue;
                }
                let n = slice.len() * 4;
                memcpy(
                    unsafe { ptr.add(off) },
                    slice.as_ptr().cast(),
                    n,
                    MEMCPY_H2D,
                )?;
                off += n;
            }
        }
        Ok((
            Self {
                device,
                ptr: ptr.cast(),
                bytes,
            },
            bytes as u64,
        ))
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn ptr_f32(&self, byte_off: usize) -> Result<*mut f32, AllpacaError> {
        if byte_off >= self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA weight ptr out of range".into(),
            )));
        }
        Ok(unsafe { self.ptr.add(byte_off) }.cast())
    }
}

impl Drop for CudaDeviceWeights {
    fn drop(&mut self) {
        device_free(self.ptr.cast());
        self.ptr = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_weights_upload_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0];
        let (w, h2d) = CudaDeviceWeights::upload(0, &[&a, &b]).expect("upload");
        assert_eq!(h2d, 24);
        assert_eq!(w.bytes(), 24);
        let p0 = w.ptr_f32(0).expect("ptr");
        let mut back = [0.0f32; 4];
        memcpy(
            back.as_mut_ptr().cast(),
            p0.cast(),
            16,
            MEMCPY_D2H,
        )
        .expect("d2h");
        assert_eq!(back, a);
    }

    #[test]
    fn stub_or_no_device_reports_zero_when_no_gpu() {
        // This workstation has a toolkit but no NVIDIA driver; count must be 0.
        // Builds without the runtime cfg also return 0.
        assert!(device_count() >= 0);
        if !runtime_linked() || device_count() == 0 {
            assert!(enumerate_devices().is_empty());
            let mut c = [0.0f32; 4];
            let err = gemm_nt(&[1.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0, 1.0], &mut c, 2, 2, 2);
            assert!(err.is_err());
            assert!(CudaWorkspace::new(0, 4096).is_err());
        }
    }

    #[test]
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    fn runtime_linked_when_toolkit_builds() {
        assert!(runtime_linked());
    }

    #[test]
    fn copy_h2d_f32_roundtrip_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let src: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
        let d = device_malloc(src.len() * 4, 0).expect("malloc");
        ws.copy_h2d_f32(d.cast(), &src).expect("h2d");
        let mut back = vec![0.0f32; src.len()];
        memcpy(
            back.as_mut_ptr().cast(),
            d,
            src.len() * 4,
            MEMCPY_D2H,
        )
        .expect("d2h");
        device_free(d);
        assert_eq!(back, src);
        assert!(ws.activation_h2d_bytes() >= (src.len() * 4) as u64);
    }

    #[test]
    fn gemm_nt_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let m = 2i32;
        let n = 3i32;
        let k = 4i32;
        let a: Vec<f32> = (0..(m * k)).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..(n * k)).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let mut hip = vec![0.0f32; (m * n) as usize];
        let mut cpu = vec![0.0f32; (m * n) as usize];
        gemm_nt(&a, &b, &mut hip, m, n, k).expect("cuda gemm");
        allpaca_cpu::matmul_nt(&a, &b, &mut cpu, m as usize, n as usize, k as usize)
            .expect("cpu gemm");
        for i in 0..cpu.len() {
            assert!(
                (hip[i] - cpu[i]).abs() < 1e-4,
                "mismatch at {i}: cuda={} cpu={}",
                hip[i],
                cpu[i]
            );
        }
        let ws = CudaWorkspace::new(0, 8192).expect("workspace");
        assert!(ws.allocated_bytes() >= 8192);
    }

    #[test]
    fn device_pointer_gemm_nt_matches_host_path_when_present() {
        if device_count() <= 0 {
            return;
        }
        let m = 2i32;
        let n = 2i32;
        let k = 2i32;
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        let mut host_c = [0.0f32; 4];
        gemm_nt(&a, &b, &mut host_c, m, n, k).expect("host gemm");

        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let da = ws.upload_f32(&a).expect("upload a");
        let db = ws.upload_f32(&b).expect("upload b");
        let dc = device_malloc(16, 0).expect("alloc c");
        ws.gemm_nt_dev(da.cast(), db.cast(), dc.cast(), m, n, k)
            .expect("device gemm");
        ws.synchronize().expect("sync");
        let mut device_c = [0.0f32; 4];
        memcpy(
            device_c.as_mut_ptr().cast(),
            dc,
            16,
            MEMCPY_D2H,
        )
        .expect("d2h");
        ws.free_ptr(da);
        ws.free_ptr(db);
        ws.free_device(dc);
        let _ = ws.synchronize();
        for i in 0..4 {
            assert!(
                (host_c[i] - device_c[i]).abs() < 1e-5,
                "i={i} host={} device={}",
                host_c[i],
                device_c[i]
            );
        }
    }

    #[test]
    fn pinned_host_and_device_buffer_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let mut pinned = PinnedHostBuffer::new(64).expect("pinned");
        pinned.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(&pinned.as_slice()[..4], &[1, 2, 3, 4]);

        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        assert!(!ws.raw_stream().is_null());
        let buf = DeviceBuffer::alloc(64, 0)
            .expect("device buf")
            .with_async_free(ws.raw_stream());
        ws.copy_h2d_f32(buf.as_f32(), &[1.0, 2.0, 3.0, 4.0])
            .expect("h2d");
        let mut back = [0.0f32; 4];
        ws.download_f32(buf.as_f32(), &mut back).expect("d2h");
        assert_eq!(back, [1.0, 2.0, 3.0, 4.0]);
        drop(buf);
        ws.synchronize().expect("drain async free");
    }

    #[test]
    fn fused_residual_silu_batches_on_stream_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 8i32;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.05).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 + 2.0) * 0.02).collect();
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let dx = ws.upload_f32(&x).expect("x");
        let dy = ws.upload_f32(&y).expect("y");
        let dup = ws.upload_f32(&up).expect("up");
        let d_tmp = device_malloc((n as usize) * 4, 0).expect("tmp");
        let d_out = device_malloc((n as usize) * 4, 0).expect("out");
        // Two kernels, one sync — no per-op device synchronize.
        ws.residual_add_dev(dx, dy, d_tmp.cast(), n).expect("residual");
        ws.silu_mul_dev(d_tmp.cast(), dup, d_out.cast(), n)
            .expect("silu");
        let mut out = vec![0.0f32; n as usize];
        ws.download_f32(d_out.cast(), &mut out).expect("d2h");
        for i in 0..n as usize {
            let r = x[i] + y[i];
            let gate = r / (1.0 + (-r).exp());
            let expect = gate * up[i];
            assert!(
                (out[i] - expect).abs() < 1e-4,
                "i={i} out={} expect={}",
                out[i],
                expect
            );
        }
        ws.free_ptr(dx);
        ws.free_ptr(dy);
        ws.free_ptr(dup);
        ws.free_device(d_tmp);
        ws.free_device(d_out);
        ws.synchronize().expect("drain");
    }

    #[test]
    fn cuda_stream_create_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let s = CudaStream::new().expect("stream");
        assert!(!s.raw().is_null());
        s.synchronize().expect("sync");
        let h2d = H2dStream::new().expect("h2d stream");
        assert!(!h2d.raw().is_null());
        h2d.synchronize().expect("h2d sync");
    }

    #[test]
    fn mem_info_unavailable_without_device() {
        if device_count() > 0 {
            return;
        }
        assert!(device_memory_info(0).is_err());
    }

    #[test]
    fn mem_info_reports_board_free_that_shrinks_on_allocation() {
        if device_count() <= 0 {
            return;
        }
        let (free_before, total) = device_memory_info(0).expect("mem_info");
        assert!(total > 0, "total VRAM should be positive");
        assert!(free_before <= total);
        let chunk = 16 * 1024 * 1024;
        let ptr = device_malloc(chunk, 0).expect("alloc");
        let (free_after, total_after) = device_memory_info(0).expect("mem_info after");
        assert_eq!(total_after, total);
        assert!(
            free_after + (chunk as u64) / 2 <= free_before,
            "free_before={free_before} free_after={free_after} chunk={chunk}"
        );
        device_free(ptr);
        let (free_now, _) = device_memory_info(0).expect("mem_info after free");
        assert!(free_now >= free_after);
    }

    #[test]
    fn device_sm_arch_none_without_device() {
        if device_count() > 0 {
            return;
        }
        assert!(device_sm_arch(0).is_none());
    }

    #[test]
    fn device_sm_arch_is_sm_prefix_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let (arch, l2) = device_sm_arch(0).expect("sm arch");
        assert!(
            arch.starts_with("sm_") && arch.len() >= 5,
            "unexpected arch {arch}"
        );
        // Consumer Ampere/Ada/Blackwell laptops all have an L2 cache.
        assert!(l2 > 0, "l2={l2}");
        let digits: String = arch.trim_start_matches("sm_").to_string();
        assert!(
            digits.chars().all(|c| c.is_ascii_digit()) && !digits.is_empty(),
            "arch digits {digits}"
        );
    }

    #[test]
    fn embed_gather_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let dim = 16i32;
        let vocab = 4usize;
        let table: Vec<f32> = (0..(vocab * dim as usize))
            .map(|i| (i as f32) * 0.01)
            .collect();
        let token = 2i32;
        let mut cuda_out = vec![0.0f32; dim as usize];
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        ws.embed_gather(&table, token, &mut cuda_out, dim)
            .expect("cuda embed_gather");
        let start = (token as usize) * (dim as usize);
        let cpu = &table[start..start + dim as usize];
        for i in 0..dim as usize {
            assert!(
                (cuda_out[i] - cpu[i]).abs() < 1e-6,
                "embed mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn rope_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_heads = 2i32;
        let head_dim = 8i32;
        let pos = 3i32;
        let theta = 10000.0f32;
        let n = (n_heads * head_dim) as usize;
        let mut cuda_x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut cpu_x = cuda_x.clone();
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        ws.rope(&mut cuda_x, n_heads, head_dim, pos, theta)
            .expect("cuda rope");
        for h in 0..n_heads as usize {
            let off = h * head_dim as usize;
            allpaca_cpu::rope(&mut cpu_x[off..off + head_dim as usize], pos as usize, theta)
                .expect("cpu rope");
        }
        for i in 0..n {
            assert!(
                (cuda_x[i] - cpu_x[i]).abs() < 1e-4,
                "rope mismatch at {i}: cuda={} cpu={}",
                cuda_x[i],
                cpu_x[i]
            );
        }
    }

    #[test]
    fn attention_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_q = 1i32;
        let n_kv = 4i32;
        let n_heads = 4i32;
        let n_kv_heads = 2i32;
        let head_dim = 8i32;
        let q_n = (n_q * n_heads * head_dim) as usize;
        let kv_n = (n_kv * n_kv_heads * head_dim) as usize;
        let q: Vec<f32> = (0..q_n).map(|i| (i as f32) * 0.02).collect();
        let k: Vec<f32> = (0..kv_n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let v: Vec<f32> = (0..kv_n).map(|i| (i as f32 + 2.0) * 0.015).collect();
        let mut cuda_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.attention(
            &q,
            &k,
            &v,
            &mut cuda_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            true,
        )
        .expect("cuda attention");
        allpaca_cpu::attention(
            &q,
            &k,
            &v,
            &mut cpu_out,
            n_q as usize,
            n_kv as usize,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
            true,
        )
        .expect("cpu attention");
        for i in 0..q_n {
            assert!(
                (cuda_out[i] - cpu_out[i]).abs() < 1e-4,
                "attention mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu_out[i]
            );
        }
    }
}
