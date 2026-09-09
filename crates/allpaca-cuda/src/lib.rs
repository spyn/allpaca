//! CUDA kernel FFI. Feature-gated like `allpaca-hip`.
//!
//! `--features cuda` enables the path when a toolkit is found at build time
//! (`CUDA_PATH` / `nvcc`). Without a driver or GPU, `device_count()` is 0 and
//! device GEMM returns unavailable — the crate still compiles and links.
//!
//! Symbols are `allpaca_cuda_*` so HIP and CUDA ABIs do not collide (ADR 0001).

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
        pub fn allpaca_cuda_malloc(ptr: *mut *mut c_void, bytes: usize, device: i32) -> i32;
        pub fn allpaca_cuda_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_cuda_memcpy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_set_device(device: i32) -> i32;
        pub fn allpaca_cuda_stream_synchronize(stream: *mut c_void) -> i32;
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

pub fn memcpy(
    dst: *mut std::ffi::c_void,
    src: *const std::ffi::c_void,
    bytes: usize,
    kind: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_memcpy(dst, src, bytes, kind, std::ptr::null_mut()) };
        status(rc, "memcpy")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (dst, src, bytes, kind);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA memcpy unavailable".into(),
        )))
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

/// Minimal CUDA workspace: scratch buffer + F32 GEMM. Enough to open when a
/// device appears; full forward parity grows behind this type (like HipWorkspace).
pub struct CudaWorkspace {
    pub device: i32,
    scratch: *mut std::ffi::c_void,
    scratch_bytes: usize,
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
        let scratch = device_malloc(scratch_bytes, device)?;
        Ok(Self {
            device,
            scratch,
            scratch_bytes,
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

    pub fn gemm_nt_dev(
        &self,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        set_device(self.device)?;
        gemm_nt_device(a, b, c, m, n, k)
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
            let rc =
                unsafe { ffi::allpaca_cuda_residual_add(x, y, out, n, std::ptr::null_mut()) };
            status(rc, "residual_add")?;
            let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
            status(sync, "stream_synchronize")
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
            let rc =
                unsafe { ffi::allpaca_cuda_rmsnorm(x, w, out, n, eps, std::ptr::null_mut()) };
            status(rc, "rmsnorm")?;
            let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
            status(sync, "stream_synchronize")
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
            let rc =
                unsafe { ffi::allpaca_cuda_silu_mul(gate, up, out, n, std::ptr::null_mut()) };
            status(rc, "silu_mul")?;
            let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
            status(sync, "stream_synchronize")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (gate, up, out, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA silu_mul unavailable".into(),
            )))
        }
    }

    /// Upload host F32 weights into a device blob (H2D). Returns device pointer.
    pub fn upload_f32(&self, host: &[f32]) -> Result<*mut f32, AllpacaError> {
        let bytes = host.len().saturating_mul(4);
        set_device(self.device)?;
        let ptr = device_malloc(bytes, self.device)?;
        memcpy(ptr, host.as_ptr().cast(), bytes, MEMCPY_H2D)?;
        self.activation_h2d
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(ptr.cast())
    }

    pub fn free_ptr(&self, ptr: *mut f32) {
        device_free(ptr.cast());
    }
}

impl Drop for CudaWorkspace {
    fn drop(&mut self) {
        device_free(self.scratch);
        self.scratch = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
