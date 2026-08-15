//! HIP kernel FFI. Default build is a stub so the workspace compiles without ROCm.
//!
//! Enable `--features hip` and link a HIP-built `allpaca_kernels` library to use
//! the C ABI (`allpaca_gemm`, `allpaca_attention`, `allpaca_copy_async`).

use allpaca_types::{AllpacaError, BandwidthBenchmark, DegradedError, DeviceInfo};
#[cfg(feature = "hip")]
use allpaca_types::DeviceKind;

pub const ALLPACA_OK: i32 = 0;
pub const ALLPACA_ERR_NOT_IMPLEMENTED: i32 = 1;
pub const ALLPACA_ERR_INVALID: i32 = 2;
pub const ALLPACA_ERR_DEVICE: i32 = 3;

#[cfg(feature = "hip")]
mod ffi {
    use std::os::raw::{c_int, c_void};

    #[link(name = "amdhip64", kind = "dylib")]
    extern "C" {
        pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
        pub fn hipDeviceCanAccessPeer(can: *mut c_int, src: c_int, dst: c_int) -> c_int;
        pub fn hipDeviceGetName(name: *mut i8, len: c_int, device: c_int) -> c_int;
        pub fn hipDeviceTotalMem(bytes: *mut usize, device: c_int) -> c_int;
    }

    #[link(name = "allpaca_kernels", kind = "static")]
    extern "C" {
        pub fn allpaca_gemm(
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

/// Enumerate HIP devices. Empty when the `hip` feature is off or no GPU is present.
pub fn enumerate_devices() -> Vec<DeviceInfo> {
    #[cfg(not(feature = "hip"))]
    {
        Vec::new()
    }
    #[cfg(feature = "hip")]
    {
        unsafe {
            let mut count: i32 = 0;
            if ffi::hipGetDeviceCount(&mut count) != 0 || count <= 0 {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(count as usize);
            for id in 0..count {
                let mut name_buf = [0i8; 256];
                let _ = ffi::hipDeviceGetName(name_buf.as_mut_ptr(), 256, id);
                let name = std::ffi::CStr::from_ptr(name_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                let mut mem = 0usize;
                let _ = ffi::hipDeviceTotalMem(&mut mem, id);
                let mut peers = Vec::new();
                for other in 0..count {
                    if other == id {
                        continue;
                    }
                    let mut can = 0;
                    if ffi::hipDeviceCanAccessPeer(&mut can, id, other) == 0 && can != 0 {
                        peers.push(other);
                    }
                }
                out.push(DeviceInfo {
                    id,
                    kind: DeviceKind::HipGpu,
                    name,
                    memory_bytes: mem as u64,
                    bandwidth_gbps: 0.0,
                    peers,
                });
            }
            out
        }
    }
}

pub fn can_p2p(src: i32, dst: i32) -> bool {
    #[cfg(not(feature = "hip"))]
    {
        let _ = (src, dst);
        false
    }
    #[cfg(feature = "hip")]
    {
        unsafe {
            let mut can = 0;
            ffi::hipDeviceCanAccessPeer(&mut can, src, dst) == 0 && can != 0
        }
    }
}

/// Runtime P2P vs host-staging measurement. Never assumes P2P is faster.
pub fn benchmark_transfer(_src: i32, _dst: i32, bytes: u64) -> Result<BandwidthBenchmark, AllpacaError> {
    #[cfg(not(feature = "hip"))]
    {
        let _ = bytes;
        Err(AllpacaError::Degraded(DegradedError::HostStagingFallback(
            "HIP feature disabled; P2P not measured".into(),
        )))
    }
    #[cfg(feature = "hip")]
    {
        // TODO: timed hipMemcpyPeer vs host-staged hipMemcpy when a real kernel lib is linked.
        let _ = bytes;
        Err(AllpacaError::Degraded(DegradedError::HostStagingFallback(
            "HIP transfer benchmark not implemented; host staging selected".into(),
        )))
    }
}

pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(not(feature = "hip"))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm unavailable; use CPU fallback".into(),
        )))
    }
    #[cfg(feature = "hip")]
    {
        let rc = unsafe {
            ffi::allpaca_gemm(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                std::ptr::null_mut(),
            )
        };
        status(rc, "gemm")
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
    #[cfg(not(feature = "hip"))]
    {
        let _ = (
            q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
        );
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP attention unavailable; use CPU fallback".into(),
        )))
    }
    #[cfg(feature = "hip")]
    {
        let rc = unsafe {
            ffi::allpaca_attention(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                out.as_mut_ptr(),
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
}

pub fn copy_async(dst: &mut [u8], src: &[u8]) -> Result<(), AllpacaError> {
    #[cfg(not(feature = "hip"))]
    {
        let _ = (dst, src);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP copy_async unavailable; use CPU fallback".into(),
        )))
    }
    #[cfg(feature = "hip")]
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
}

#[cfg(feature = "hip")]
fn status(rc: i32, op: &str) -> Result<(), AllpacaError> {
    if rc == ALLPACA_OK {
        Ok(())
    } else {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            format!("HIP {op} returned {rc}; CPU fallback"),
        )))
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
        }
    }
}
