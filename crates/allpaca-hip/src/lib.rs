//! HIP kernel FFI. Default build is a stub so the workspace compiles without ROCm.
//!
//! `--features hip` enables the HIP path when a SDK is found at build time
//! (`HIP_PATH` / `ROCM_PATH` / common install locations). Without an SDK the
//! crate still compiles; [`device_count`] returns 0 and GPU tests skip.

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Persistent device workspace used by the Engine for GEMM / attention / logits.
pub struct HipWorkspace {
    pub device: i32,
    scratch: *mut u8,
    scratch_bytes: usize,
    pinned_logits: *mut u8,
    pinned_logits_bytes: usize,
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
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        let mut n = self.scratch_bytes as u64;
        if !self.pinned_logits.is_null() {
            n += self.pinned_logits_bytes as u64;
        }
        n
    }

    pub fn bind(&self) -> Result<(), AllpacaError> {
        set_device(self.device)
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
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
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
            memcpy(out.as_mut_ptr().cast(), dout, out.len() * 4, MEMCPY_D2H)
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

    /// Copy device logits into the pinned host ring slot when one exists.
    pub fn publish_logits(&self, device_logits: *const f32, host: &mut [f32]) -> Result<(), AllpacaError> {
        let bytes = host.len() * 4;
        if !self.pinned_logits.is_null() && self.pinned_logits_bytes >= bytes {
            memcpy(self.pinned_logits, device_logits.cast(), bytes, MEMCPY_D2H)?;
            unsafe {
                std::ptr::copy_nonoverlapping(self.pinned_logits.cast::<f32>(), host.as_mut_ptr(), host.len());
            }
            Ok(())
        } else {
            memcpy(host.as_mut_ptr().cast(), device_logits.cast(), bytes, MEMCPY_D2H)
        }
    }
}

impl Drop for HipWorkspace {
    fn drop(&mut self) {
        device_free(self.scratch, self.scratch_bytes);
        host_free(self.pinned_logits);
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
}
