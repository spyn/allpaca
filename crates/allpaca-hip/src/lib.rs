//! HIP kernel FFI. Default build is a stub so the workspace compiles without ROCm.
//!
//! `--features hip` enables the HIP path when a SDK is found at build time
//! (`HIP_PATH` / `ROCM_PATH` / common install locations). Without an SDK the
//! crate still compiles; [`device_count`] returns 0 and GPU tests skip.

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use allpaca_types::{AllpacaError, BandwidthBenchmark, DegradedError, DeviceInfo};
#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use allpaca_types::DeviceKind;

pub const ALLPACA_OK: i32 = 0;
pub const ALLPACA_ERR_NOT_IMPLEMENTED: i32 = 1;
pub const ALLPACA_ERR_INVALID: i32 = 2;
pub const ALLPACA_ERR_DEVICE: i32 = 3;

pub const MEMCPY_H2D: i32 = 0;
pub const MEMCPY_D2H: i32 = 1;
pub const MEMCPY_D2D: i32 = 2;
pub const MEMCPY_DEFAULT: i32 = 3;

static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
mod ffi {
    use std::os::raw::c_void;

    extern "C" {
        pub fn allpaca_hip_device_count(count: *mut i32) -> i32;
        pub fn allpaca_hip_device_props(
            id: i32,
            name: *mut i8,
            name_len: i32,
            memory_bytes: *mut u64,
            bandwidth_gbps: *mut f64,
        ) -> i32;
        pub fn allpaca_hip_can_p2p(src: i32, dst: i32, can: *mut i32) -> i32;
        pub fn allpaca_hip_enable_peer_access(src: i32, dst: i32) -> i32;
        pub fn allpaca_hip_malloc(ptr: *mut *mut c_void, bytes: usize, device: i32) -> i32;
        pub fn allpaca_hip_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_hip_host_malloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn allpaca_hip_host_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_hip_memcpy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_hip_memcpy_peer(
            dst: *mut c_void,
            dst_dev: i32,
            src: *const c_void,
            src_dev: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn allpaca_hip_benchmark(
            src: i32,
            dst: i32,
            bytes: usize,
            p2p_gbps: *mut f64,
            host_gbps: *mut f64,
        ) -> i32;
        pub fn allpaca_hip_mem_used(device: i32, used_bytes: *mut u64) -> i32;
        pub fn allpaca_hip_set_device(device: i32) -> i32;
        pub fn allpaca_hip_stream_synchronize(stream: *mut c_void) -> i32;
        pub fn allpaca_gemm(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q8_0(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_attention(
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
        pub fn allpaca_copy_async(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rmsnorm(
            x: *const f32,
            weight: *const f32,
            out: *mut f32,
            n: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rope(
            x: *mut f32,
            n_heads: i32,
            head_dim: i32,
            pos: i32,
            theta: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_silu_mul(
            gate: *mut f32,
            up: *const f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather(
            table: *const f32,
            token: i32,
            out: *mut f32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_residual_add(
            x: *mut f32,
            residual: *const f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_dequant_q8_0(
            blocks: *const u8,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather_q8_0(
            table: *const u8,
            token: i32,
            out: *mut f32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// True when this build linked a HIP-compiled `allpaca_kernels` library.
pub fn runtime_linked() -> bool {
    cfg!(all(feature = "hip", allpaca_hip_runtime))
}

pub fn device_count() -> i32 {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        unsafe {
            let mut count = 0i32;
            if ffi::allpaca_hip_device_count(&mut count) != ALLPACA_OK {
                return 0;
            }
            count.max(0)
        }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        0
    }
}

/// Enumerate HIP devices. Empty when the `hip` feature is off, no SDK, or no GPU.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let count = device_count();
    if count <= 0 {
        return Vec::new();
    }
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut out = Vec::with_capacity(count as usize);
        for id in 0..count {
            let mut name_buf = [0i8; 256];
            let mut mem = 0u64;
            let mut bw = 0.0f64;
            let rc = unsafe {
                ffi::allpaca_hip_device_props(id, name_buf.as_mut_ptr(), 256, &mut mem, &mut bw)
            };
            if rc != ALLPACA_OK {
                continue;
            }
            let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let mut peers = Vec::new();
            for other in 0..count {
                if other != id && probe_peer(id, other).capable {
                    peers.push(other);
                }
            }
            out.push(DeviceInfo {
                id,
                kind: DeviceKind::HipGpu,
                name,
                memory_bytes: mem,
                bandwidth_gbps: bw,
                peers,
            });
        }
        out
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = count;
        Vec::new()
    }
}

pub fn can_p2p(src: i32, dst: i32) -> bool {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        unsafe {
            let mut can = 0;
            ffi::allpaca_hip_can_p2p(src, dst, &mut can) == ALLPACA_OK && can != 0
        }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (src, dst);
        false
    }
}

/// Result of query → enable → re-query for one direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerAccess {
    pub capable: bool,
    pub enabled: bool,
}

/// Query, enable `src → dst`, then re-query. Policy stays in Rust.
pub fn enable_peer_access(src: i32, dst: i32) -> bool {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        unsafe { ffi::allpaca_hip_enable_peer_access(src, dst) == ALLPACA_OK }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (src, dst);
        false
    }
}

/// Query, attempt enable, re-query. `capable` is the post-enable answer.
pub fn probe_peer(src: i32, dst: i32) -> PeerAccess {
    let queried = can_p2p(src, dst);
    let enabled = enable_peer_access(src, dst);
    let capable = can_p2p(src, dst);
    tracing::debug!(src, dst, queried, enabled, capable, "P2P probe query-enable-requery");
    PeerAccess { capable, enabled }
}

pub fn memcpy_peer(
    dst: *mut u8,
    dst_dev: i32,
    src: *const u8,
    src_dev: i32,
    bytes: usize,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_hip_memcpy_peer(
                dst.cast(),
                dst_dev,
                src.cast(),
                src_dev,
                bytes,
                std::ptr::null_mut(),
            )
        };
        status(rc, "memcpy_peer")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, dst_dev, src, src_dev, bytes);
        Err(AllpacaError::Degraded(DegradedError::HostStagingFallback(
            "HIP peer copy unavailable".into(),
        )))
    }
}

/// Runtime enable + peer-copy vs host-staging measurement. Never assumes P2P is faster.
pub fn benchmark_transfer(src: i32, dst: i32, bytes: u64) -> Result<BandwidthBenchmark, AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        use std::time::Instant;

        let n = bytes as usize;
        if n == 0 {
            return Err(AllpacaError::Degraded(DegradedError::HostStagingFallback(
                "transfer benchmark bytes must be > 0".into(),
            )));
        }
        let d_src = device_malloc(n, src)?;
        let d_dst = match device_malloc(n, dst) {
            Ok(p) => p,
            Err(e) => {
                device_free(d_src, n);
                return Err(e);
            }
        };
        let h_pin = match host_malloc(n) {
            Ok(p) => p,
            Err(e) => {
                device_free(d_src, n);
                device_free(d_dst, n);
                return Err(e);
            }
        };

        let t = Instant::now();
        let ab = probe_peer(src, dst);
        let ba = probe_peer(dst, src);
        let copy_ok = ab.enabled
            && ba.enabled
            && ab.capable
            && ba.capable
            && memcpy_peer(d_dst, dst, d_src, src, n).is_ok();
        let p2p_elapsed_us = t.elapsed().as_secs_f64() * 1.0e6;
        let (p2p_gbps, p2p_us, p2p_copy_ok) = if copy_ok && p2p_elapsed_us > 0.0 {
            let gbps = (bytes as f64 * 8.0 / 1.0e9) / (p2p_elapsed_us / 1.0e6);
            (gbps, p2p_elapsed_us, true)
        } else {
            (0.0, 0.0, false)
        };

        let t = Instant::now();
        let host_ok = memcpy(h_pin, d_src, n, MEMCPY_D2H).is_ok()
            && memcpy(d_dst, h_pin, n, MEMCPY_H2D).is_ok();
        let host_elapsed_us = t.elapsed().as_secs_f64() * 1.0e6;
        let (host_staging_gbps, host_staging_us) = if host_ok && host_elapsed_us > 0.0 {
            let gbps = (bytes as f64 * 8.0 / 1.0e9) / (host_elapsed_us / 1.0e6);
            (gbps, host_elapsed_us)
        } else {
            (0.0, 0.0)
        };

        host_free(h_pin);
        device_free(d_src, n);
        device_free(d_dst, n);

        if !host_ok && !p2p_copy_ok {
            return Err(AllpacaError::Degraded(DegradedError::HostStagingFallback(
                "HIP transfer benchmark failed for both P2P and host staging".into(),
            )));
        }
        Ok(BandwidthBenchmark {
            bytes,
            p2p_gbps,
            host_staging_gbps,
            p2p_us,
            host_staging_us,
            p2p_copy_ok,
        })
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (src, dst, bytes);
        Err(AllpacaError::Degraded(DegradedError::HostStagingFallback(
            "HIP feature disabled or SDK missing; P2P not measured".into(),
        )))
    }
}

pub fn device_malloc(bytes: usize, device: i32) -> Result<*mut u8, AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_hip_malloc(&mut ptr, bytes, device) };
        if rc != ALLPACA_OK || ptr.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("hipMalloc({bytes}) on device {device} failed rc={rc}"),
            )));
        }
        ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(ptr.cast())
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (bytes, device);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP malloc unavailable".into(),
        )))
    }
}

pub fn device_free(ptr: *mut u8, bytes: usize) {
    if ptr.is_null() {
        return;
    }
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    unsafe {
        let _ = ffi::allpaca_hip_free(ptr.cast());
        ALLOCATED_BYTES.fetch_sub(bytes as u64, Ordering::Relaxed);
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = bytes;
    }
}

pub fn host_malloc(bytes: usize) -> Result<*mut u8, AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_hip_host_malloc(&mut ptr, bytes) };
        if rc != ALLPACA_OK || ptr.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("hipHostMalloc({bytes}) failed rc={rc}"),
            )));
        }
        Ok(ptr.cast())
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = bytes;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP host malloc unavailable".into(),
        )))
    }
}

pub fn host_free(ptr: *mut u8) {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    if !ptr.is_null() {
        unsafe {
            let _ = ffi::allpaca_hip_host_free(ptr.cast());
        }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = ptr;
    }
}

pub fn memcpy(dst: *mut u8, src: *const u8, bytes: usize, kind: i32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_hip_memcpy(dst.cast(), src.cast(), bytes, kind, std::ptr::null_mut())
        };
        status(rc, "memcpy")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, src, bytes, kind);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP memcpy unavailable".into(),
        )))
    }
}

/// Wait for queued kernels and copies on the default stream.
pub fn stream_synchronize() -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_stream_synchronize(std::ptr::null_mut()) };
        status(rc, "stream_synchronize")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP stream_synchronize unavailable".into(),
        )))
    }
}

/// Bind subsequent kernels / memcpy to `device`.
pub fn set_device(device: i32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_set_device(device) };
        status(rc, "set_device")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP set_device unavailable".into(),
        )))
    }
}

/// Bytes currently tracked from [`device_malloc`].
pub fn allocated_bytes() -> u64 {
    ALLOCATED_BYTES.load(Ordering::Relaxed)
}

pub fn device_memory_used(device: i32) -> Result<u64, AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut used = 0u64;
        let rc = unsafe { ffi::allpaca_hip_mem_used(device, &mut used) };
        status(rc, "mem_used").map(|()| used)
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP mem_used unavailable".into(),
        )))
    }
}

/// Host-pointer GEMM: copies to device, runs the kernel, copies back.
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_kind(a, b, c, m, n, k, false, 0)
}

/// Host-pointer NT GEMM: `C[m,n] = A[m,k] * B[n,k]^T`.
pub fn gemm_nt(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_kind(a, b, c, m, n, k, true, 0)
}

#[allow(clippy::too_many_arguments)]
fn gemm_kind(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
    nt: bool,
    device: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid GEMM shape".into(),
            )));
        }
        let a_bytes = (m as usize) * (k as usize) * 4;
        let b_bytes = if nt {
            (n as usize) * (k as usize) * 4
        } else {
            (k as usize) * (n as usize) * 4
        };
        let c_bytes = (m as usize) * (n as usize) * 4;
        if a.len() * 4 < a_bytes || b.len() * 4 < b_bytes || c.len() * 4 < c_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_bytes, device)?;
        let db = device_malloc(b_bytes, device)?;
        let dc = device_malloc(c_bytes, device)?;
        set_device(device)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_bytes, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr().cast(), b_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                if nt {
                    ffi::allpaca_gemm_nt(
                        da.cast(),
                        db.cast(),
                        dc.cast(),
                        m,
                        n,
                        k,
                        std::ptr::null_mut(),
                    )
                } else {
                    ffi::allpaca_gemm(
                        da.cast(),
                        db.cast(),
                        dc.cast(),
                        m,
                        n,
                        k,
                        std::ptr::null_mut(),
                    )
                }
            };
            status(rc, if nt { "gemm_nt" } else { "gemm" })?;
            memcpy(c.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H)
        })();
        device_free(da, a_bytes);
        device_free(db, b_bytes);
        device_free(dc, c_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k, nt, device);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm unavailable; use CPU fallback".into(),
        )))
    }
}

/// Device-pointer GEMM NT (no host copies). `a`, `b`, `c` are hipMalloc pointers.
pub fn gemm_nt_device(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with GGUF Q8_0 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_q8_0_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_q8_0(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q8_0")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q8_0 unavailable".into(),
        )))
    }
}

/// Host-pointer Q8_0 NT GEMM: copies to device, runs the kernel, copies back.
pub fn gemm_nt_q8_0(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid GEMM shape".into(),
            )));
        }
        let a_n = (m as usize) * (k as usize);
        let c_n = (m as usize) * (n as usize);
        let nblocks = (k as usize).div_ceil(32);
        let b_bytes = (n as usize) * nblocks * 34;
        if a.len() < a_n || b.len() < b_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q8_0 GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(b_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), b_bytes, MEMCPY_H2D)?;
            gemm_nt_q8_0_device(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, b_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q8_0 unavailable".into(),
        )))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
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
    attention_on(
        0, q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
    )
}

#[allow(clippy::too_many_arguments)]
fn attention_on(
    device: i32,
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
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let q_bytes = q.len() * 4;
        let k_bytes = k.len() * 4;
        let v_bytes = v.len() * 4;
        let o_bytes = out.len() * 4;
        let dq = device_malloc(q_bytes, device)?;
        let dk = device_malloc(k_bytes, device)?;
        let dv = device_malloc(v_bytes, device)?;
        let dout = device_malloc(o_bytes, device)?;
        set_device(device)?;
        let result = (|| {
            memcpy(dq, q.as_ptr().cast(), q_bytes, MEMCPY_H2D)?;
            memcpy(dk, k.as_ptr().cast(), k_bytes, MEMCPY_H2D)?;
            memcpy(dv, v.as_ptr().cast(), v_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_attention(
                    dq.cast(),
                    dk.cast(),
                    dv.cast(),
                    dout.cast(),
                    n_q,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    i32::from(causal),
                    std::ptr::null_mut(),
                )
            };
            status(rc, "attention")?;
            memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)
        })();
        device_free(dq, q_bytes);
        device_free(dk, k_bytes);
        device_free(dv, v_bytes);
        device_free(dout, o_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (
            device, q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
        );
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP attention unavailable; use CPU fallback".into(),
        )))
    }
}

pub fn copy_async(dst: &mut [u8], src: &[u8]) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_copy_async(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                src.len(),
                std::ptr::null_mut(),
            )
        };
        status(rc, "copy_async")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, src);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP copy_async unavailable; use CPU fallback".into(),
        )))
    }
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn status(rc: i32, op: &str) -> Result<(), AllpacaError> {
    if rc == ALLPACA_OK {
        Ok(())
    } else {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            format!("HIP {op} returned {rc}; CPU fallback"),
        )))
    }
}

/// Packed weight tensors uploaded once via H2D. GEMM must use [`Self::ptr`] — no per-op copy.
pub struct DeviceWeights {
    device: i32,
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for DeviceWeights {}
unsafe impl Sync for DeviceWeights {}

impl DeviceWeights {
    /// Allocate one device blob and copy `tensors` in order. Records no Engine metrics.
    pub fn upload(device: i32, tensors: &[&[f32]]) -> Result<Self, AllpacaError> {
        let bytes: usize = tensors.iter().map(|t| t.len() * 4).sum();
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty weight upload".into(),
            )));
        }
        let ptr = device_malloc(bytes, device)?;
        set_device(device)?;
        let mut off = 0usize;
        for t in tensors {
            let n = t.len() * 4;
            if let Err(e) = memcpy(unsafe { ptr.add(off) }, t.as_ptr().cast(), n, MEMCPY_H2D) {
                device_free(ptr, bytes);
                return Err(e);
            }
            off += n;
        }
        Ok(Self { device, ptr, bytes })
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn bytes(&self) -> u64 {
        self.bytes as u64
    }

    pub fn ptr_at(&self, byte_offset: usize) -> *const f32 {
        unsafe { self.ptr.add(byte_offset).cast() }
    }

    pub fn byte_ptr_at(&self, byte_offset: usize) -> *const u8 {
        unsafe { self.ptr.add(byte_offset) }
    }

    /// Packed native-quant / F32 byte slices uploaded once via H2D.
    pub fn upload_bytes(device: i32, parts: &[&[u8]]) -> Result<Self, AllpacaError> {
        let bytes: usize = parts.iter().map(|t| t.len()).sum();
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty weight upload".into(),
            )));
        }
        let ptr = device_malloc(bytes, device)?;
        set_device(device)?;
        let mut off = 0usize;
        for t in parts {
            if let Err(e) = memcpy(unsafe { ptr.add(off) }, t.as_ptr(), t.len(), MEMCPY_H2D) {
                device_free(ptr, bytes);
                return Err(e);
            }
            off += t.len();
        }
        Ok(Self { device, ptr, bytes })
    }
}

impl Drop for DeviceWeights {
    fn drop(&mut self) {
        device_free(self.ptr, self.bytes);
    }
}

/// Named hipMalloc view used for one sequential `forward_token` on device.
#[derive(Clone, Copy)]
pub struct DeviceDecode {
    pub x: *mut f32,
    pub normed: *mut f32,
    pub q: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub attn: *mut f32,
    pub proj: *mut f32,
    pub gate: *mut f32,
    pub up: *mut f32,
    pub logits: *mut f32,
    pub packed_k: *mut f32,
    pub packed_v: *mut f32,
    pub max_kv: usize,
}

struct DecodeAlloc {
    ptr: *mut u8,
    bytes: usize,
    dim: usize,
    kv_dim: usize,
    ff_dim: usize,
    vocab: usize,
    max_kv: usize,
}

unsafe impl Send for DecodeAlloc {}
unsafe impl Sync for DecodeAlloc {}

impl DecodeAlloc {
    fn ptrs(&self) -> DeviceDecode {
        let mut off = 0usize;
        let mut next = |n: usize| {
            let p = unsafe { self.ptr.add(off * 4).cast::<f32>() };
            off += n;
            p
        };
        DeviceDecode {
            x: next(self.dim),
            normed: next(self.dim),
            q: next(self.dim),
            k: next(self.kv_dim),
            v: next(self.kv_dim),
            attn: next(self.dim),
            proj: next(self.dim),
            gate: next(self.ff_dim),
            up: next(self.ff_dim),
            logits: next(self.vocab),
            packed_k: next(self.max_kv * self.kv_dim),
            packed_v: next(self.max_kv * self.kv_dim),
            max_kv: self.max_kv,
        }
    }
}

fn decode_floats(dim: usize, kv_dim: usize, ff_dim: usize, vocab: usize, max_kv: usize) -> usize {
    dim * 5 + kv_dim * 2 + ff_dim * 2 + vocab + max_kv * kv_dim * 2
}

/// Persistent device workspace used by the Engine for GEMM / attention / logits.
struct DequantAlloc {
    ptr: *mut u8,
    bytes: usize,
    n_floats: usize,
}

unsafe impl Send for DequantAlloc {}
unsafe impl Sync for DequantAlloc {}

pub struct HipWorkspace {
    pub device: i32,
    scratch: *mut u8,
    scratch_bytes: usize,
    pinned_logits: *mut u8,
    pinned_logits_bytes: usize,
    decode: Mutex<Option<DecodeAlloc>>,
    dequant: Mutex<Option<DequantAlloc>>,
    activation_h2d: AtomicU64,
    activation_d2h: AtomicU64,
}

unsafe impl Send for HipWorkspace {}
unsafe impl Sync for HipWorkspace {}

impl HipWorkspace {
    pub fn new(device: i32, scratch_bytes: usize, vocab: usize) -> Result<Self, AllpacaError> {
        let scratch_bytes = scratch_bytes.max(4096);
        let scratch = device_malloc(scratch_bytes, device)?;
        let pinned_logits_bytes = vocab.max(1) * 4;
        let pinned_logits = match host_malloc(pinned_logits_bytes) {
            Ok(p) => p,
            Err(_) => std::ptr::null_mut(),
        };
        Ok(Self {
            device,
            scratch,
            scratch_bytes,
            pinned_logits,
            pinned_logits_bytes,
            decode: Mutex::new(None),
            dequant: Mutex::new(None),
            activation_h2d: AtomicU64::new(0),
            activation_d2h: AtomicU64::new(0),
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        let mut n = self.scratch_bytes as u64;
        if !self.pinned_logits.is_null() {
            n += self.pinned_logits_bytes as u64;
        }
        if let Ok(g) = self.decode.lock() {
            if let Some(d) = g.as_ref() {
                n += d.bytes as u64;
            }
        }
        if let Ok(g) = self.dequant.lock() {
            if let Some(d) = g.as_ref() {
                n += d.bytes as u64;
            }
        }
        n
    }

    pub fn activation_h2d_bytes(&self) -> u64 {
        self.activation_h2d.load(Ordering::Relaxed)
    }

    pub fn activation_d2h_bytes(&self) -> u64 {
        self.activation_d2h.load(Ordering::Relaxed)
    }

    fn add_h2d(&self, bytes: usize) {
        self.activation_h2d
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn add_d2h(&self, bytes: usize) {
        self.activation_d2h
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }

    pub fn bind(&self) -> Result<(), AllpacaError> {
        set_device(self.device)
    }

    /// Persistent hidden / QKV / packed-KV buffers. Grows if the model or seq is larger.
    pub fn ensure_decode(
        &self,
        dim: usize,
        kv_dim: usize,
        ff_dim: usize,
        vocab: usize,
        max_kv: usize,
    ) -> Result<DeviceDecode, AllpacaError> {
        let max_kv = max_kv.max(1);
        let mut guard = self
            .decode
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let grow = match guard.as_ref() {
            None => true,
            Some(a) => {
                a.dim < dim
                    || a.kv_dim < kv_dim
                    || a.ff_dim < ff_dim
                    || a.vocab < vocab
                    || a.max_kv < max_kv
            }
        };
        if grow {
            if let Some(old) = guard.take() {
                device_free(old.ptr, old.bytes);
            }
            let dim = dim.max(1);
            let kv_dim = kv_dim.max(1);
            let ff_dim = ff_dim.max(1);
            let vocab = vocab.max(1);
            let floats = decode_floats(dim, kv_dim, ff_dim, vocab, max_kv);
            let bytes = floats * 4;
            let ptr = device_malloc(bytes, self.device)?;
            *guard = Some(DecodeAlloc {
                ptr,
                bytes,
                dim,
                kv_dim,
                ff_dim,
                vocab,
                max_kv,
            });
        }
        Ok(guard.as_ref().expect("decode alloc").ptrs())
    }

    /// Reusable F32 workspace for on-device Q8_0 dequant before GEMM.
    pub fn ensure_dequant(&self, n_floats: usize) -> Result<*mut f32, AllpacaError> {
        let n_floats = n_floats.max(1);
        let mut guard = self.dequant.lock().unwrap_or_else(|e| e.into_inner());
        let grow = match guard.as_ref() {
            None => true,
            Some(a) => a.n_floats < n_floats,
        };
        if grow {
            if let Some(old) = guard.take() {
                device_free(old.ptr, old.bytes);
            }
            let bytes = n_floats * 4;
            let ptr = device_malloc(bytes, self.device)?;
            *guard = Some(DequantAlloc {
                ptr,
                bytes,
                n_floats,
            });
        }
        Ok(guard.as_ref().expect("dequant alloc").ptr.cast())
    }

    pub fn dequant_q8_0(&self, blocks: *const u8, n: i32, out: *mut f32) -> Result<(), AllpacaError> {
        if blocks.is_null() || out.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid q8_0 dequant".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_dequant_q8_0(blocks, out, n, std::ptr::null_mut()) };
            status(rc, "dequant_q8_0")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP dequant_q8_0 unavailable".into(),
            )))
        }
    }

    pub fn embed_gather_q8_0(
        &self,
        table: *const u8,
        token: u32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid q8_0 embed gather".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_embed_gather_q8_0(table, token as i32, out, dim, std::ptr::null_mut())
            };
            status(rc, "embed_gather_q8_0")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = token;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_q8_0 unavailable".into(),
            )))
        }
    }

    /// GEMM NT using the persistent scratch buffer (H2D / kernel / D2H).
    pub fn gemm_nt(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.gemm_via_scratch(a, b, c, m, n, k, true)
    }

    /// GEMM NT with a **device** weight pointer `b`. Only activations (`a`) and `c` are copied.
    pub fn gemm_nt_dev_b(
        &self,
        a: &[f32],
        b: *const f32,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 || b.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid resident GEMM".into(),
            )));
        }
        let a_n = (m as usize) * (k as usize);
        let c_n = (m as usize) * (n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "resident GEMM buffer size mismatch".into(),
            )));
        }
        let need = (a_n + c_n) * 4;
        if need > self.scratch_bytes {
            let da = device_malloc(a_n * 4, self.device)?;
            let dc = match device_malloc(c_n * 4, self.device) {
                Ok(p) => p,
                Err(e) => {
                    device_free(da, a_n * 4);
                    return Err(e);
                }
            };
            let result = (|| {
                self.bind()?;
                memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
                self.add_h2d(a_n * 4);
                gemm_nt_device(da.cast(), b, dc.cast(), m, n, k)?;
                memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)?;
                self.add_d2h(c_n * 4);
                Ok(())
            })();
            device_free(da, a_n * 4);
            device_free(dc, c_n * 4);
            return result;
        }
        self.bind()?;
        let da = self.scratch;
        let dc = unsafe { self.scratch.add(a_n * 4) };
        memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
        self.add_h2d(a_n * 4);
        gemm_nt_device(da.cast(), b, dc.cast(), m, n, k)?;
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)?;
        self.add_d2h(c_n * 4);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_via_scratch(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
        nt: bool,
    ) -> Result<(), AllpacaError> {
        let a_n = (m as usize) * (k as usize);
        let b_n = (n as usize) * (k as usize);
        let c_n = (m as usize) * (n as usize);
        let need = (a_n + b_n + c_n) * 4;
        if need > self.scratch_bytes {
            return gemm_kind(a, b, c, m, n, k, nt, self.device);
        }
        self.bind()?;
        let da = self.scratch;
        let db = unsafe { self.scratch.add(a_n * 4) };
        let dc = unsafe { self.scratch.add((a_n + b_n) * 4) };
        memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
        self.add_h2d(a_n * 4);
        memcpy(db, b.as_ptr().cast(), b_n * 4, MEMCPY_H2D)?;
        if nt {
            gemm_nt_device(da.cast(), db.cast(), dc.cast(), m, n, k)?;
        } else {
            #[cfg(all(feature = "hip", allpaca_hip_runtime))]
            {
                let rc = unsafe {
                    ffi::allpaca_gemm(da.cast(), db.cast(), dc.cast(), m, n, k, std::ptr::null_mut())
                };
                status(rc, "gemm")?;
            }
            #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
            {
                return gemm(a, b, c, m, n, k);
            }
        }
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)?;
        self.add_d2h(c_n * 4);
        Ok(())
    }

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
        let need = (q.len() + k.len() + v.len() + out.len()) * 4;
        if need > self.scratch_bytes {
            return attention_on(
                self.device,
                q,
                k,
                v,
                out,
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
            );
        }
        self.bind()?;
        let dq = self.scratch;
        let dk = unsafe { self.scratch.add(q.len() * 4) };
        let dv = unsafe { self.scratch.add((q.len() + k.len()) * 4) };
        let dout = unsafe { self.scratch.add((q.len() + k.len() + v.len()) * 4) };
        memcpy(dq, q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D)?;
        memcpy(dk, k.as_ptr().cast(), k.len() * 4, MEMCPY_H2D)?;
        memcpy(dv, v.as_ptr().cast(), v.len() * 4, MEMCPY_H2D)?;
        self.add_h2d((q.len() + k.len() + v.len()) * 4);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_attention(
                    dq.cast(),
                    dk.cast(),
                    dv.cast(),
                    dout.cast(),
                    n_q,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    i32::from(causal),
                    std::ptr::null_mut(),
                )
            };
            status(rc, "attention")?;
            memcpy(out.as_mut_ptr().cast(), dout, out.len() * 4, MEMCPY_D2H)?;
            self.add_d2h(out.len() * 4);
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = dout;
            attention_on(
                self.device,
                q,
                k,
                v,
                out,
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
            )
        }
    }

    pub fn rmsnorm(
        &self,
        x: &[f32],
        weight: &[f32],
        weight_dev: Option<*const f32>,
        eps: f32,
        out: &mut [f32],
    ) -> Result<(), AllpacaError> {
        if x.len() != out.len() || x.is_empty() || (weight_dev.is_none() && weight.len() != x.len())
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm length mismatch".into(),
            )));
        }
        let n = x.len();
        let need = n * 4 * if weight_dev.is_some() { 2 } else { 3 };
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm exceeds scratch".into(),
            )));
        }
        self.bind()?;
        let dx = self.scratch;
        let dw = unsafe { self.scratch.add(n * 4) };
        let dout = if weight_dev.is_some() {
            dw
        } else {
            unsafe { self.scratch.add(n * 8) }
        };
        memcpy(dx, x.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        self.add_h2d(n * 4);
        let wptr = if let Some(w) = weight_dev {
            w
        } else {
            memcpy(dw, weight.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
            dw.cast()
        };
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_rmsnorm(dx.cast(), wptr, dout.cast(), n as i32, eps, std::ptr::null_mut())
            };
            status(rc, "rmsnorm")?;
            memcpy(out.as_mut_ptr().cast(), dout, n * 4, MEMCPY_D2H)?;
            self.add_d2h(n * 4);
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (wptr, dout, eps);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rmsnorm unavailable".into(),
            )))
        }
    }

    pub fn rope(
        &self,
        x: &mut [f32],
        n_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        let n = x.len();
        if n * 4 > self.scratch_bytes || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rope exceeds scratch".into(),
            )));
        }
        self.bind()?;
        memcpy(self.scratch, x.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        self.add_h2d(n * 4);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_rope(
                    self.scratch.cast(),
                    n_heads,
                    head_dim,
                    pos,
                    theta,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "rope")?;
            memcpy(x.as_mut_ptr().cast(), self.scratch, n * 4, MEMCPY_D2H)?;
            self.add_d2h(n * 4);
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (pos, theta);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rope unavailable".into(),
            )))
        }
    }

    pub fn silu_mul(&self, gate: &mut [f32], up: &[f32]) -> Result<(), AllpacaError> {
        if gate.len() != up.len() || gate.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "silu_mul length mismatch".into(),
            )));
        }
        let n = gate.len();
        if n * 8 > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "silu_mul exceeds scratch".into(),
            )));
        }
        self.bind()?;
        let dg = self.scratch;
        let du = unsafe { self.scratch.add(n * 4) };
        memcpy(dg, gate.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        memcpy(du, up.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        self.add_h2d(n * 8);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_silu_mul(dg.cast(), du.cast(), n as i32, std::ptr::null_mut()) };
            status(rc, "silu_mul")?;
            memcpy(gate.as_mut_ptr().cast(), dg, n * 4, MEMCPY_D2H)?;
            self.add_d2h(n * 4);
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP silu_mul unavailable".into(),
            )))
        }
    }

    /// Copy device logits into the pinned host ring slot when one exists.
    pub fn publish_logits(&self, device_logits: *const f32, host: &mut [f32]) -> Result<(), AllpacaError> {
        let bytes = host.len() * 4;
        if !self.pinned_logits.is_null() && self.pinned_logits_bytes >= bytes {
            memcpy(self.pinned_logits, device_logits.cast(), bytes, MEMCPY_D2H)?;
            self.add_d2h(bytes);
            unsafe {
                std::ptr::copy_nonoverlapping(self.pinned_logits.cast::<f32>(), host.as_mut_ptr(), host.len());
            }
            Ok(())
        } else {
            memcpy(host.as_mut_ptr().cast(), device_logits.cast(), bytes, MEMCPY_D2H)?;
            self.add_d2h(bytes);
            Ok(())
        }
    }

    pub fn embed_gather(
        &self,
        table: *const f32,
        token: u32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed gather".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_embed_gather(table, token as i32, out, dim, std::ptr::null_mut())
            };
            status(rc, "embed_gather")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = token;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather unavailable".into(),
            )))
        }
    }

    pub fn residual_add(&self, x: *mut f32, residual: *const f32, n: i32) -> Result<(), AllpacaError> {
        if x.is_null() || residual.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid residual add".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_residual_add(x, residual, n, std::ptr::null_mut()) };
            status(rc, "residual_add")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP residual_add unavailable".into(),
            )))
        }
    }

    pub fn rmsnorm_device(
        &self,
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || weight.is_null() || out.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device rmsnorm".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_rmsnorm(x, weight, out, n, eps, std::ptr::null_mut()) };
            status(rc, "rmsnorm")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = eps;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rmsnorm unavailable".into(),
            )))
        }
    }

    pub fn rope_device(
        &self,
        x: *mut f32,
        n_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device rope".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_rope(x, n_heads, head_dim, pos, theta, std::ptr::null_mut()) };
            status(rc, "rope")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (pos, theta);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rope unavailable".into(),
            )))
        }
    }

    pub fn silu_mul_device(&self, gate: *mut f32, up: *const f32, n: i32) -> Result<(), AllpacaError> {
        if gate.is_null() || up.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device silu_mul".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_silu_mul(gate, up, n, std::ptr::null_mut()) };
            status(rc, "silu_mul")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP silu_mul unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_device(
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
        if q.is_null() || k.is_null() || v.is_null() || out.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device attention".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_attention(
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
                    std::ptr::null_mut(),
                )
            };
            status(rc, "attention")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (n_q, n_kv, n_heads, n_kv_heads, head_dim, causal);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP attention unavailable".into(),
            )))
        }
    }
}

impl Drop for HipWorkspace {
    fn drop(&mut self) {
        device_free(self.scratch, self.scratch_bytes);
        host_free(self.pinned_logits);
        if let Ok(mut g) = self.decode.lock() {
            if let Some(d) = g.take() {
                device_free(d.ptr, d.bytes);
            }
        }
        if let Ok(mut g) = self.dequant.lock() {
            if let Some(d) = g.take() {
                device_free(d.ptr, d.bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_build_has_no_hip_devices() {
        if cfg!(not(feature = "hip")) {
            assert!(enumerate_devices().is_empty());
            assert!(!can_p2p(0, 1));
            let access = probe_peer(0, 1);
            assert!(!access.capable);
            assert!(!access.enabled);
            assert_eq!(device_count(), 0);
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_feature_skips_cleanly_without_gpu() {
        if device_count() == 0 {
            assert!(enumerate_devices().is_empty());
            return;
        }
        let devices = enumerate_devices();
        assert!(!devices.is_empty());
        assert!(devices.iter().all(|d| d.id >= 0 && d.kind == allpaca_types::DeviceKind::HipGpu));
    }

    #[cfg(feature = "hip")]
    fn cpu_gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for r in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[r * k + p] * b[j * k + p];
                }
                c[r * n + j] = acc;
            }
        }
    }

    #[cfg(feature = "hip")]
    fn gemm_close(got: f32, exp: f32) -> bool {
        let d = (got - exp).abs();
        d <= 1e-4 || d <= 1e-3 * got.abs().max(exp.abs())
    }

    /// Public `gemm_nt` must match CPU `C[m,n] = A[m,k] * B[n,k]^T` on small random shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_matches_cpu_on_small_random_matrices() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (1, 8, 4),
            (2, 3, 5),
            (1, 64, 32),
            (7, 17, 13),
            (4, 32, 16),
            (1, 128, 64),
            (3, 65, 17),
            (16, 64, 32),
            (32, 32, 32),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let mut a = vec![0.0f32; m * k];
            let mut b = vec![0.0f32; n * k];
            for (i, v) in a.iter_mut().enumerate() {
                *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in b.iter_mut().enumerate() {
                *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
            }
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b, &mut cpu, m, n, k);
            gemm_nt(&a, &b, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    #[cfg(feature = "hip")]
    fn f32_to_f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mant = bits & 0x7f_ffff;
        if exp == 255 {
            return sign | 0x7c00 | ((mant >> 13) as u16);
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return sign | 0x7c00;
        }
        if new_exp <= 0 {
            return sign;
        }
        sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
    }

    #[cfg(feature = "hip")]
    fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
        let n_blocks = data.len().div_ceil(32);
        let mut out = vec![0u8; n_blocks * 34];
        for b in 0..n_blocks {
            let start = b * 32;
            let end = (start + 32).min(data.len());
            let mut amax = 0.0f32;
            for v in &data[start..end] {
                amax = amax.max(v.abs());
            }
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
            let off = b * 34;
            out[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
            for i in 0..32 {
                let src = start + i;
                let q = if src < data.len() {
                    (data[src] * inv).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                };
                out[off + 2 + i] = q as u8;
            }
        }
        out
    }

    /// Public `gemm_nt_q8_0` must match CPU dequant + NT GEMM on small random shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q8_0_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (1, 8, 32),
            (2, 5, 64),
            (1, 17, 32),
            (3, 9, 96),
            (1, 1, 33),
            (1, 4, 33),
            (2, 7, 33),
            (1, 33, 64),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let mut a = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; n * k];
            for (i, v) in a.iter_mut().enumerate() {
                *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in b_f.iter_mut().enumerate() {
                *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
            }
            let mut b_q8 = Vec::new();
            for row in 0..n {
                b_q8.extend(quantize_q8_0(&b_f[row * k..(row + 1) * k]));
            }
            let mut b_deq = vec![0.0f32; n * k];
            let nblocks = k.div_ceil(32);
            for row in 0..n {
                let blk = &b_q8[row * nblocks * 34..(row + 1) * nblocks * 34];
                for bi in 0..nblocks {
                    let off = bi * 34;
                    let d = {
                        let h = u16::from_le_bytes([blk[off], blk[off + 1]]);
                        let sign = u32::from(h & 0x8000) << 16;
                        let exp = (h >> 10) & 0x1f;
                        let mant = u32::from(h & 0x3ff);
                        let bits = if exp == 0 {
                            sign
                        } else if exp == 31 {
                            sign | 0x7f80_0000 | (mant << 13)
                        } else {
                            sign | ((u32::from(exp) + (127 - 15)) << 23) | (mant << 13)
                        };
                        f32::from_bits(bits)
                    };
                    for j in 0..32 {
                        let idx = bi * 32 + j;
                        if idx < k {
                            b_deq[row * k + idx] = d * (blk[off + 2 + j] as i8 as f32);
                        }
                    }
                }
            }
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_q8_0(&a, &b_q8, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_q8_0 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q8_0 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Queued Q8 GEMV + residual must match CPU after a single D2H (no per-op host wait).
    #[cfg(feature = "hip")]
    #[test]
    fn queued_q8_gemm_and_residual_match_cpu() {
        if device_count() == 0 {
            return;
        }
        let m = 1usize;
        let n = 7usize;
        let k = 64usize;
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        let mut residual = vec![0.0f32; m * n];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 11) % 40) as f32 * 0.03 - 0.4;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 19) % 40) as f32 * 0.03 - 0.4;
        }
        for (i, v) in residual.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 0.3;
        }
        let mut b_q8 = Vec::new();
        for row in 0..n {
            b_q8.extend(quantize_q8_0(&b_f[row * k..(row + 1) * k]));
        }
        let mut b_deq = vec![0.0f32; n * k];
        let nblocks = k.div_ceil(32);
        for row in 0..n {
            let blk = &b_q8[row * nblocks * 34..(row + 1) * nblocks * 34];
            for bi in 0..nblocks {
                let off = bi * 34;
                let h = u16::from_le_bytes([blk[off], blk[off + 1]]);
                let sign = u32::from(h & 0x8000) << 16;
                let exp = (h >> 10) & 0x1f;
                let mant = u32::from(h & 0x3ff);
                let bits = if exp == 0 {
                    sign
                } else if exp == 31 {
                    sign | 0x7f80_0000 | (mant << 13)
                } else {
                    sign | ((u32::from(exp) + (127 - 15)) << 23) | (mant << 13)
                };
                let d = f32::from_bits(bits);
                for j in 0..32 {
                    let idx = bi * 32 + j;
                    if idx < k {
                        b_deq[row * k + idx] = d * (blk[off + 2 + j] as i8 as f32);
                    }
                }
            }
        }
        let mut cpu = vec![0.0f32; m * n];
        cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
        for i in 0..m * n {
            cpu[i] += residual[i];
        }

        let a_bytes = a.len() * 4;
        let b_bytes = b_q8.len();
        let c_bytes = cpu.len() * 4;
        let da = device_malloc(a_bytes, 0).expect("da");
        let db = device_malloc(b_bytes, 0).expect("db");
        let dc = device_malloc(c_bytes, 0).expect("dc");
        let dr = device_malloc(c_bytes, 0).expect("dr");
        set_device(0).expect("bind");
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_bytes, MEMCPY_H2D)?;
            memcpy(db, b_q8.as_ptr(), b_bytes, MEMCPY_H2D)?;
            memcpy(dr, residual.as_ptr().cast(), c_bytes, MEMCPY_H2D)?;
            gemm_nt_q8_0_device(da.cast(), db, dc.cast(), m as i32, n as i32, k as i32)?;
            let hip = HipWorkspace::new(0, 4096, 8)?;
            hip.residual_add(dc.cast(), dr.cast(), (m * n) as i32)?;
            let mut got = vec![0.0f32; m * n];
            memcpy(got.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H)?;
            for i in 0..m * n {
                assert!(
                    gemm_close(got[i], cpu[i]),
                    "queued q8+residual mismatch idx={i} hip={} cpu={}",
                    got[i],
                    cpu[i]
                );
            }
            Ok::<(), AllpacaError>(())
        })();
        device_free(da, a_bytes);
        device_free(db, b_bytes);
        device_free(dc, c_bytes);
        device_free(dr, c_bytes);
        result.expect("queued device Q8 GEMV + residual must match CPU");
    }

    #[cfg(feature = "hip")]
    fn cpu_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        n_q: usize,
        n_kv: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let reps = n_heads / n_kv_heads;
        let mut scores = vec![0.0f32; n_kv];
        for qi in 0..n_q {
            for h in 0..n_heads {
                let kv_h = h / reps;
                let q_off = (qi * n_heads + h) * head_dim;
                for t in 0..n_kv {
                    if causal && t > qi + n_kv.saturating_sub(n_q) {
                        scores[t] = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_off = (t * n_kv_heads + kv_h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_off + d] * k[k_off + d];
                    }
                    scores[t] = dot * scale;
                }
                let mut max = f32::NEG_INFINITY;
                for s in &scores {
                    if *s > max {
                        max = *s;
                    }
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                if sum == 0.0 {
                    sum = 1.0;
                }
                let o_off = q_off;
                for d in 0..head_dim {
                    out[o_off + d] = 0.0;
                }
                for t in 0..n_kv {
                    let w = scores[t] / sum;
                    let v_off = (t * n_kv_heads + kv_h) * head_dim;
                    for d in 0..head_dim {
                        out[o_off + d] += w * v[v_off + d];
                    }
                }
            }
        }
    }

    /// Public `attention` must match CPU causal / GQA softmax on decode and short prefill.
    #[cfg(feature = "hip")]
    #[test]
    fn attention_matches_cpu_on_decode_and_prefill() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (1, 1, 1, 1, 2, true),
            (1, 8, 4, 1, 4, true),
            (1, 16, 2, 2, 8, true),
            (1, 32, 4, 2, 16, true),
            (1, 8, 1, 1, 128, true),
            (4, 4, 2, 2, 4, true),
            (3, 8, 4, 1, 8, true),
            (2, 2, 1, 1, 4, false),
        ];
        for (si, (n_q, n_kv, n_heads, n_kv_heads, head_dim, causal)) in
            shapes.iter().copied().enumerate()
        {
            let qn = n_q * n_heads * head_dim;
            let kn = n_kv * n_kv_heads * head_dim;
            let mut q = vec![0.0f32; qn];
            let mut k = vec![0.0f32; kn];
            let mut v = vec![0.0f32; kn];
            for (i, x) in q.iter_mut().enumerate() {
                *x = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, x) in k.iter_mut().enumerate() {
                *x = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, x) in v.iter_mut().enumerate() {
                *x = ((i * 11 + si * 5) % 50) as f32 * 0.02 - 0.5;
            }
            let mut cpu = vec![0.0f32; qn];
            let mut hip = vec![0.0f32; qn];
            cpu_attention(
                &q, &k, &v, &mut cpu, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            );
            attention(
                &q,
                &k,
                &v,
                &mut hip,
                n_q as i32,
                n_kv as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                causal,
            )
            .expect("HIP attention must run when a device is present");
            for i in 0..qn {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "attention mismatch n_q={n_q} n_kv={n_kv} heads={n_heads}/{n_kv_heads} d={head_dim} causal={causal} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }
}
