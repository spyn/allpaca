//! HIP kernel FFI. Default build is a stub so the workspace compiles without ROCm.
//!
//! `--features hip` enables the HIP path when a SDK is found at build time
//! (`HIP_PATH` / `ROCM_PATH` / common install locations). Without an SDK the
//! crate still compiles; [`device_count`] returns 0 and GPU tests skip.

mod gemv_tile;

pub use gemv_tile::{
    gemv_tile_working_bytes, infinity_cache_from_l2, profile_gemv_tile_cols,
    profiled_gemv_tile_cols, select_gemv_tile_cols,
};

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use allpaca_types::{AllpacaError, BandwidthBenchmark, DegradedError, DeviceInfo};
#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use allpaca_types::DeviceKind;

/// GGUF Q8_0 packed block (f16 scale + 32 qs).
pub const Q8_PACKED_BLOCK: usize = 34;
/// Device-staged Q8_0: f32 scale at 0, qs at 4, 36-byte stride (4-byte aligned).
pub const Q8_STAGED_BLOCK: usize = 36;
pub const Q8_STAGED_QS: usize = 4;
/// GGUF Q4_K super-block (f16 d + f16 dmin + 12 scale bytes + 128 qs).
pub const Q4_K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// GGUF Q6_K super-block (128 ql + 64 qh + 16 scales + f16 d).
pub const Q6_K_BLOCK_ELEMS: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// GGUF Q5_K super-block (f16 d + f16 dmin + 12 scales + 32 qh + 128 qs).
pub const Q5_K_BLOCK_ELEMS: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 176;

fn f16_bits_to_f32(h: u16) -> f32 {
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
}

/// Repack GGUF 34-byte Q8_0 blocks into 36-byte device blocks with an F32 scale.
pub fn stage_q8_0_blocks(packed: &[u8]) -> Result<Vec<u8>, AllpacaError> {
    if packed.len() % Q8_PACKED_BLOCK != 0 {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "Q8_0 packed length is not a multiple of 34".into(),
        )));
    }
    let n = packed.len() / Q8_PACKED_BLOCK;
    let mut out = vec![0u8; n * Q8_STAGED_BLOCK];
    for i in 0..n {
        let src = i * Q8_PACKED_BLOCK;
        let dst = i * Q8_STAGED_BLOCK;
        let h = u16::from_le_bytes([packed[src], packed[src + 1]]);
        out[dst..dst + 4].copy_from_slice(&f16_bits_to_f32(h).to_le_bytes());
        out[dst + Q8_STAGED_QS..dst + Q8_STAGED_QS + 32]
            .copy_from_slice(&packed[src + 2..src + Q8_PACKED_BLOCK]);
    }
    Ok(out)
}

/// Repack GGUF Q8_0 into 36-byte column-major GEMV blocks (F32 scale + aligned qs).
pub fn stage_q8_0_gemv(packed: &[u8], n: usize, k: usize) -> Result<Vec<u8>, AllpacaError> {
    if n == 0 || k == 0 {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "Q8_0 GEMV stage needs n>0 and k>0".into(),
        )));
    }
    let nblocks = k.div_ceil(32);
    let packed_need = n.saturating_mul(nblocks).saturating_mul(Q8_PACKED_BLOCK);
    if packed.len() < packed_need {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "Q8_0 GEMV packed length is short for n,k".into(),
        )));
    }
    stage_q8_0_blocks(&packed[..packed_need])
}

pub const ALLPACA_OK: i32 = 0;
pub const ALLPACA_ERR_NOT_IMPLEMENTED: i32 = 1;
pub const ALLPACA_ERR_INVALID: i32 = 2;
pub const ALLPACA_ERR_DEVICE: i32 = 3;

pub const MEMCPY_H2D: i32 = 0;
pub const MEMCPY_D2H: i32 = 1;
pub const MEMCPY_D2D: i32 = 2;
pub const MEMCPY_DEFAULT: i32 = 3;
pub const MEMCPY_D2H_ASYNC: i32 = 4;

static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

/// HIP decode timing buckets (microseconds) for the last/EMA target forward.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DecodeProfile {
    pub q8_gemm_us: f64,
    pub attention_us: f64,
    pub lm_head_us: f64,
    pub other_gpu_us: f64,
    pub host_us: f64,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum DecodeOp {
    Q8Gemm = 0,
    Attention = 1,
    LmHead = 2,
    OtherGpu = 3,
}

struct ProfileState {
    #[allow(dead_code)]
    pairs: Vec<[*mut std::ffi::c_void; 2]>,
    ops: Vec<u8>,
    used: usize,
    full: bool,
    wall_start: Option<std::time::Instant>,
    ema_ns: [u64; 5],
}

impl Default for ProfileState {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            ops: Vec::new(),
            used: 0,
            full: false,
            wall_start: None,
            ema_ns: [0; 5],
        }
    }
}

impl Drop for ProfileState {
    fn drop(&mut self) {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            for pair in &self.pairs {
                unsafe {
                    ffi::allpaca_hip_event_destroy(pair[0]);
                    ffi::allpaca_hip_event_destroy(pair[1]);
                }
            }
        }
    }
}

unsafe impl Send for ProfileState {}

fn ema_ns(prev: u64, sample: u64) -> u64 {
    if prev == 0 {
        sample
    } else {
        (prev * 8 + sample * 2) / 10
    }
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_create() -> Result<*mut std::ffi::c_void, AllpacaError> {
    let mut ev = std::ptr::null_mut();
    let rc = unsafe { ffi::allpaca_hip_event_create(&mut ev) };
    status(rc, "event_create")?;
    if ev.is_null() {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP event_create returned null".into(),
        )));
    }
    Ok(ev)
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_record(ev: *mut std::ffi::c_void) -> Result<(), AllpacaError> {
    let rc = unsafe { ffi::allpaca_hip_event_record(ev, std::ptr::null_mut()) };
    status(rc, "event_record")
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_elapsed_ns(start: *mut std::ffi::c_void, stop: *mut std::ffi::c_void) -> u64 {
    let mut us = 0.0f64;
    let rc = unsafe { ffi::allpaca_hip_event_elapsed_us(start, stop, &mut us) };
    if rc != ALLPACA_OK || us <= 0.0 {
        0
    } else {
        (us * 1_000.0).round() as u64
    }
}

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
        pub fn allpaca_hip_event_create(event: *mut *mut c_void) -> i32;
        pub fn allpaca_hip_event_destroy(event: *mut c_void) -> i32;
        pub fn allpaca_hip_event_record(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn allpaca_hip_event_elapsed_us(
            start: *mut c_void,
            stop: *mut c_void,
            us: *mut f64,
        ) -> i32;
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
        pub fn allpaca_gemm_nt_untiled(
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
        pub fn allpaca_gemm_nt_q8_0_intdot(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q4_k(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q4_k_untiled(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q6_k(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q6_k_cols(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            cols: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q6_k_untiled(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q5_k(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q5_k_cols(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            cols: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q5_k_untiled(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_f16(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_f16_untiled(
            a: *const f32,
            b: *const u16,
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
        pub fn allpaca_attention_masked(
            q: *const f32,
            k: *const f32,
            v: *const f32,
            out: *mut f32,
            n_q: i32,
            n_kv: i32,
            n_heads: i32,
            n_kv_heads: i32,
            head_dim: i32,
            mask: *const u8,
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
        pub fn allpaca_rmsnorm_rows(
            x: *const f32,
            weight: *const f32,
            out: *mut f32,
            rows: i32,
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
        pub fn allpaca_embed_gather_rows(
            table: *const f32,
            tokens: *const i32,
            out: *mut f32,
            n_tok: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather_f16(
            table: *const u16,
            token: i32,
            out: *mut f32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather_f16_rows(
            table: *const u16,
            tokens: *const i32,
            out: *mut f32,
            n_tok: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_residual_add(
            x: *mut f32,
            residual: *const f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_scale_f32(
            x: *mut f32,
            n: i32,
            scale: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_residual_rmsnorm(
            x: *mut f32,
            residual: *const f32,
            weight: *const f32,
            out: *mut f32,
            n: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        #[allow(dead_code)]
        pub fn allpaca_rope_qk(
            q: *mut f32,
            k: *mut f32,
            n_q_heads: i32,
            n_k_heads: i32,
            head_dim: i32,
            pos: i32,
            theta: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rope_qk_rows(
            q: *mut f32,
            k: *mut f32,
            n_tok: i32,
            n_q_heads: i32,
            n_k_heads: i32,
            head_dim: i32,
            start_pos: i32,
            theta: f32,
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
        pub fn allpaca_embed_gather_q8_0_rows(
            table: *const u8,
            tokens: *const i32,
            out: *mut f32,
            n_tok: i32,
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

/// Expert-slot H2D probe. Returns bytes/s, or 0 when HIP is stubbed.
pub fn benchmark_h2d_bytes_per_sec(device: i32, bytes: usize) -> Result<f64, AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        use std::time::Instant;
        if bytes == 0 {
            return Ok(0.0);
        }
        let blob = DeviceWeights::alloc(device, bytes)?;
        let src = vec![1u8; bytes];
        blob.copy_h2d(0, &src)?;
        let iters = 4usize;
        let t = Instant::now();
        for _ in 0..iters {
            blob.copy_h2d(0, &src)?;
        }
        let secs = t.elapsed().as_secs_f64().max(1e-9);
        Ok((bytes as f64 * iters as f64) / secs)
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (device, bytes);
        Ok(0.0)
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
        #[cfg(allpaca_hipblaslt)]
        if rc == ALLPACA_OK {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::info!("F32 gemm_nt: hipBLASLt linked; first successful launch");
            });
        }
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

/// Untiled F32 GEMV baseline (§12 before/after).
pub fn gemm_nt_untiled_device(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_untiled unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with GGUF Q8_0 `b` (`C = A * dequant(B)^T`).
/// Decode uses float A × Q8; integer A×Q8 is [`gemm_nt_q8_0_intdot_device`].
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

/// Device-pointer integer A×Q8 GEMV. Not used by decode (float kernel is faster).
pub fn gemm_nt_q8_0_intdot_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_q8_0_intdot(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q8_0_intdot")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q8_0_intdot unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with packed IEEE F16 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_f16_device(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_f16(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_f16")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_f16 unavailable".into(),
        )))
    }
}

/// Untiled F16 GEMV baseline (§12 before/after).
pub fn gemm_nt_f16_untiled_device(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc =
            unsafe { ffi::allpaca_gemm_nt_f16_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_f16_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_f16_untiled unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native GGUF Q4_K `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_q4_k_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_q4_k(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q4_k")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q4_k unavailable".into(),
        )))
    }
}

/// Untiled Q4_K GEMV baseline (§12 before/after).
pub fn gemm_nt_q4_k_untiled_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc =
            unsafe { ffi::allpaca_gemm_nt_q4_k_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q4_k_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q4_k_untiled unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native GGUF Q6_K `b` (`C = A * dequant(B)^T`).
/// Picks §12 tile cols from Infinity Cache budget.
pub fn gemm_nt_q6_k_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let cols = select_gemv_tile_cols(n, k, infinity_cache_budget(), Q6_K_BLOCK_BYTES);
        record_gemv_tile_cols(cols);
        let rc = unsafe {
            ffi::allpaca_gemm_nt_q6_k_cols(a, b, c, m, n, k, cols, std::ptr::null_mut())
        };
        status(rc, "gemm_nt_q6_k")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q6_k unavailable".into(),
        )))
    }
}

/// Untiled Q6_K GEMV baseline (§12 before/after).
pub fn gemm_nt_q6_k_untiled_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc =
            unsafe { ffi::allpaca_gemm_nt_q6_k_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q6_k_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q6_k_untiled unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native GGUF Q5_K `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_q5_k_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let cols = select_gemv_tile_cols(n, k, infinity_cache_budget(), Q5_K_BLOCK_BYTES);
        record_gemv_tile_cols(cols);
        let rc = unsafe {
            ffi::allpaca_gemm_nt_q5_k_cols(a, b, c, m, n, k, cols, std::ptr::null_mut())
        };
        status(rc, "gemm_nt_q5_k")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q5_k unavailable".into(),
        )))
    }
}

/// Untiled Q5_K GEMV baseline (§12 before/after).
pub fn gemm_nt_q5_k_untiled_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc =
            unsafe { ffi::allpaca_gemm_nt_q5_k_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q5_k_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q5_k_untiled unavailable".into(),
        )))
    }
}

static LAST_GEMV_TILE_COLS: AtomicU64 = AtomicU64::new(64);
static INFINITY_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);

#[allow(dead_code)] // used from HIP-runtime gemm paths; stub builds keep picker API live
fn record_gemv_tile_cols(cols: i32) {
    LAST_GEMV_TILE_COLS.store(cols.max(0) as u64, Ordering::Relaxed);
}

/// Last §12 GEMV column tile chosen by [`select_gemv_tile_cols`].
pub fn last_gemv_tile_cols() -> u64 {
    LAST_GEMV_TILE_COLS.load(Ordering::Relaxed)
}

#[allow(dead_code)] // HIP-runtime gemm paths; also exercised via set_infinity_cache_budget
fn infinity_cache_budget() -> usize {
    let cached = INFINITY_CACHE_BYTES.load(Ordering::Relaxed) as usize;
    if cached > 0 {
        return cached;
    }
    if let Ok(raw) = std::env::var("ALLPACA_INFINITY_CACHE_MB") {
        if let Ok(mb) = raw.parse::<usize>() {
            let bytes = mb.saturating_mul(1 << 20);
            INFINITY_CACHE_BYTES.store(bytes as u64, Ordering::Relaxed);
            return bytes;
        }
    }
    // Default RDNA4-class budget when a GPU is present; 0 keeps COLS=64 via picker.
    let bytes = if device_count() > 0 { 64usize << 20 } else { 0 };
    INFINITY_CACHE_BYTES.store(bytes as u64, Ordering::Relaxed);
    bytes
}

/// Override Infinity Cache budget used by the §12 tile picker (tests / ops).
pub fn set_infinity_cache_budget(bytes: usize) {
    INFINITY_CACHE_BYTES.store(bytes as u64, Ordering::Relaxed);
}

/// Host-pointer F16 NT GEMM: copies packed IEEE F16 `b`, runs the kernel, copies back.
pub fn gemm_nt_f16(
    a: &[f32],
    b: &[u16],
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
        let b_n = (n as usize) * (k as usize);
        let c_n = (m as usize) * (n as usize);
        if a.len() < a_n || b.len() < b_n || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "F16 GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(b_n * 2, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr().cast(), b_n * 2, MEMCPY_H2D)?;
            gemm_nt_f16_device(da.cast(), db.cast(), dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, b_n * 2);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_f16 unavailable".into(),
        )))
    }
}

/// Host-pointer Q4_K NT GEMM: copies native GGUF blocks, runs the kernel, copies back.
pub fn gemm_nt_q4_k(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_q4_k_host(a, b, c, m, n, k, gemm_nt_q4_k_device)
}

/// Untiled Q4_K host GEMV for §12 before/after tracers.
pub fn gemm_nt_q4_k_untiled(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_q4_k_host(a, b, c, m, n, k, gemm_nt_q4_k_untiled_device)
}

/// Host-pointer Q6_K NT GEMM: copies native GGUF blocks, runs the fused kernel, copies back.
pub fn gemm_nt_q6_k(
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
        let nblocks = (k as usize).div_ceil(Q6_K_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * Q6_K_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q6_K GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            gemm_nt_q6_k_device(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, packed_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q6_k unavailable".into(),
        )))
    }
}

/// Untiled Q6_K host GEMV for §12 before/after tracers.
pub fn gemm_nt_q6_k_untiled(
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
        let nblocks = (k as usize).div_ceil(Q6_K_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * Q6_K_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q6_K GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            gemm_nt_q6_k_untiled_device(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, packed_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q6_k_untiled unavailable".into(),
        )))
    }
}

/// Host-pointer Q5_K NT GEMM: copies native GGUF blocks, runs the fused kernel, copies back.
pub fn gemm_nt_q5_k(
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
        let nblocks = (k as usize).div_ceil(Q5_K_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * Q5_K_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q5_K GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            gemm_nt_q5_k_device(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, packed_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q5_k unavailable".into(),
        )))
    }
}

/// Untiled Q5_K host GEMV for §12 before/after tracers.
pub fn gemm_nt_q5_k_untiled(
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
        let nblocks = (k as usize).div_ceil(Q5_K_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * Q5_K_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q5_K GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            gemm_nt_q5_k_untiled_device(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, packed_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q5_k_untiled unavailable".into(),
        )))
    }
}

fn gemm_nt_q4_k_host(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
    kernel: fn(*const f32, *const u8, *mut f32, i32, i32, i32) -> Result<(), AllpacaError>,
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
        let nblocks = (k as usize).div_ceil(Q4_K_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * Q4_K_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q4_K GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            kernel(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, packed_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k, kernel);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q4_k unavailable".into(),
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
    gemm_nt_q8_0_host(a, b, c, m, n, k, gemm_nt_q8_0_device)
}

/// Host-pointer integer A×Q8 GEMV (same staging as [`gemm_nt_q8_0`]).
pub fn gemm_nt_q8_0_intdot(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_q8_0_host(a, b, c, m, n, k, gemm_nt_q8_0_intdot_device)
}

fn gemm_nt_q8_0_host(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
    kernel: fn(*const f32, *const u8, *mut f32, i32, i32, i32) -> Result<(), AllpacaError>,
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
        let packed_bytes = (n as usize) * nblocks * Q8_PACKED_BLOCK;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q8_0 GEMM buffer size mismatch".into(),
            )));
        }
        let staged = stage_q8_0_gemv(&b[..packed_bytes], n as usize, k as usize)?;
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(staged.len(), 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, staged.as_ptr(), staged.len(), MEMCPY_H2D)?;
            kernel(da.cast(), db, dc.cast(), m, n, k)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(db, staged.len());
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k, kernel);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q8_0 unavailable".into(),
        )))
    }
}

/// Host-pointer batched RMSNorm: `out[rows, n] = RMSNorm(x[rows, n], weight[n])`.
pub fn rmsnorm_rows(
    x: &[f32],
    weight: &[f32],
    out: &mut [f32],
    rows: i32,
    n: i32,
    eps: f32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if rows <= 0 || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rmsnorm_rows shape".into(),
            )));
        }
        let rows_n = (rows as usize).saturating_mul(n as usize);
        if x.len() < rows_n || out.len() < rows_n || weight.len() < n as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm_rows buffer size mismatch".into(),
            )));
        }
        let x_bytes = rows_n * 4;
        let w_bytes = (n as usize) * 4;
        let dx = device_malloc(x_bytes, 0)?;
        let dw = device_malloc(w_bytes, 0)?;
        let dout = device_malloc(x_bytes, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(dx, x.as_ptr().cast(), x_bytes, MEMCPY_H2D)?;
            memcpy(dw, weight.as_ptr().cast(), w_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_rmsnorm_rows(
                    dx.cast(),
                    dw.cast(),
                    dout.cast(),
                    rows,
                    n,
                    eps,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "rmsnorm_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, x_bytes, MEMCPY_D2H)
        })();
        device_free(dx, x_bytes);
        device_free(dw, w_bytes);
        device_free(dout, x_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, weight, out, rows, n, eps);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP rmsnorm_rows unavailable".into(),
        )))
    }
}

/// Host-pointer batched Q/K RoPE: in-place on `q[n_tok, n_q*head_dim]` and `k[n_tok, n_k*head_dim]`.
pub fn rope_qk_rows(
    q: &mut [f32],
    k: &mut [f32],
    n_tok: i32,
    n_q_heads: i32,
    n_k_heads: i32,
    head_dim: i32,
    start_pos: i32,
    theta: f32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if n_tok <= 0 || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rope_qk_rows shape".into(),
            )));
        }
        let q_n = (n_tok as usize)
            .saturating_mul(n_q_heads as usize)
            .saturating_mul(head_dim as usize);
        let k_n = (n_tok as usize)
            .saturating_mul(n_k_heads as usize)
            .saturating_mul(head_dim as usize);
        if q.len() < q_n || k.len() < k_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rope_qk_rows buffer size mismatch".into(),
            )));
        }
        let q_bytes = q_n * 4;
        let k_bytes = k_n * 4;
        let dq = device_malloc(q_bytes, 0)?;
        let dk = device_malloc(k_bytes, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(dq, q.as_ptr().cast(), q_bytes, MEMCPY_H2D)?;
            memcpy(dk, k.as_ptr().cast(), k_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_rope_qk_rows(
                    dq.cast(),
                    dk.cast(),
                    n_tok,
                    n_q_heads,
                    n_k_heads,
                    head_dim,
                    start_pos,
                    theta,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "rope_qk_rows")?;
            memcpy(q.as_mut_ptr().cast(), dq, q_bytes, MEMCPY_D2H)?;
            memcpy(k.as_mut_ptr().cast(), dk, k_bytes, MEMCPY_D2H)
        })();
        device_free(dq, q_bytes);
        device_free(dk, k_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (q, k, n_tok, n_q_heads, n_k_heads, head_dim, start_pos, theta);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP rope_qk_rows unavailable".into(),
        )))
    }
}

/// Host-pointer batched F32 embed gather: `out[n_tok, dim] = table[tokens[i]]`.
pub fn embed_gather_rows(
    table: &[f32],
    tokens: &[u32],
    out: &mut [f32],
    dim: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if dim <= 0 || tokens.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed_gather_rows shape".into(),
            )));
        }
        let n_tok = tokens.len();
        let dim_u = dim as usize;
        let max_tok = *tokens.iter().max().unwrap_or(&0) as usize;
        if table.len() < (max_tok + 1).saturating_mul(dim_u) || out.len() < n_tok * dim_u {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed_gather_rows buffer size mismatch".into(),
            )));
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let t_bytes = table.len() * 4;
        let id_bytes = ids.len() * 4;
        let o_bytes = n_tok * dim_u * 4;
        let dt = device_malloc(t_bytes, 0)?;
        let di = device_malloc(id_bytes, 0)?;
        let dout = device_malloc(o_bytes, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(dt, table.as_ptr().cast(), t_bytes, MEMCPY_H2D)?;
            memcpy(di, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_embed_gather_rows(
                    dt.cast(),
                    di.cast(),
                    dout.cast(),
                    n_tok as i32,
                    dim,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "embed_gather_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)
        })();
        device_free(dt, t_bytes);
        device_free(di, id_bytes);
        device_free(dout, o_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (table, tokens, out, dim);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP embed_gather_rows unavailable".into(),
        )))
    }
}

/// Host-pointer batched F16 embed gather. `table` is packed IEEE F16 rows.
pub fn embed_gather_f16_rows(
    table: &[u16],
    tokens: &[u32],
    out: &mut [f32],
    dim: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if dim <= 0 || tokens.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed_gather_f16_rows shape".into(),
            )));
        }
        let n_tok = tokens.len();
        let dim_u = dim as usize;
        let max_tok = *tokens.iter().max().unwrap_or(&0) as usize;
        if table.len() < (max_tok + 1).saturating_mul(dim_u) || out.len() < n_tok * dim_u {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed_gather_f16_rows buffer size mismatch".into(),
            )));
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let t_bytes = table.len() * 2;
        let id_bytes = ids.len() * 4;
        let o_bytes = n_tok * dim_u * 4;
        let dt = device_malloc(t_bytes, 0)?;
        let di = device_malloc(id_bytes, 0)?;
        let dout = device_malloc(o_bytes, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(dt, table.as_ptr().cast(), t_bytes, MEMCPY_H2D)?;
            memcpy(di, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_embed_gather_f16_rows(
                    dt.cast(),
                    di.cast(),
                    dout.cast(),
                    n_tok as i32,
                    dim,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "embed_gather_f16_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)
        })();
        device_free(dt, t_bytes);
        device_free(di, id_bytes);
        device_free(dout, o_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (table, tokens, out, dim);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP embed_gather_f16_rows unavailable".into(),
        )))
    }
}

/// Host-pointer batched Q8 embed gather. `table` is packed 34-byte GGUF Q8_0 rows.
pub fn embed_gather_q8_0_rows(
    table: &[u8],
    tokens: &[u32],
    out: &mut [f32],
    dim: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if dim <= 0 || tokens.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed_gather_q8_0_rows shape".into(),
            )));
        }
        let n_tok = tokens.len();
        let dim_u = dim as usize;
        let max_tok = *tokens.iter().max().unwrap_or(&0) as usize;
        let nblocks = dim_u.div_ceil(32);
        let packed_need = (max_tok + 1).saturating_mul(nblocks).saturating_mul(Q8_PACKED_BLOCK);
        if table.len() < packed_need || out.len() < n_tok * dim_u {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed_gather_q8_0_rows buffer size mismatch".into(),
            )));
        }
        let staged = stage_q8_0_blocks(&table[..packed_need])?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let id_bytes = ids.len() * 4;
        let o_bytes = n_tok * dim_u * 4;
        let dt = device_malloc(staged.len(), 0)?;
        let di = device_malloc(id_bytes, 0)?;
        let dout = device_malloc(o_bytes, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(dt, staged.as_ptr(), staged.len(), MEMCPY_H2D)?;
            memcpy(di, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_embed_gather_q8_0_rows(
                    dt,
                    di.cast(),
                    dout.cast(),
                    n_tok as i32,
                    dim,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "embed_gather_q8_0_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)
        })();
        device_free(dt, staged.len());
        device_free(di, id_bytes);
        device_free(dout, o_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (table, tokens, out, dim);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP embed_gather_q8_0_rows unavailable".into(),
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

pub fn memcpy_async(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    kind: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_hip_memcpy(
                dst.cast(),
                src.cast(),
                bytes,
                kind,
                std::ptr::null_mut(),
            )
        };
        status(rc, "memcpy_async")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, src, bytes, kind);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP memcpy_async unavailable; use CPU fallback".into(),
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
    pub fn alloc(device: i32, bytes: usize) -> Result<Self, AllpacaError> {
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty weight upload".into(),
            )));
        }
        let ptr = device_malloc(bytes, device)?;
        Ok(Self { device, ptr, bytes })
    }

    /// Copy host bytes into this blob at `offset`. Caller must keep the region unused by GEMM.
    pub fn copy_h2d(&self, offset: usize, src: &[u8]) -> Result<(), AllpacaError> {
        if src.is_empty() {
            return Ok(());
        }
        if offset.saturating_add(src.len()) > self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "weight H2D overflow: off={offset} n={} cap={}",
                    src.len(),
                    self.bytes
                ),
            )));
        }
        set_device(self.device)?;
        memcpy(
            unsafe { self.ptr.add(offset) },
            src.as_ptr(),
            src.len(),
            MEMCPY_H2D,
        )
    }

    /// Queue H2D on the default stream (overlap with GPU kernels on same stream).
    pub fn copy_h2d_async(&self, offset: usize, src: &[u8]) -> Result<(), AllpacaError> {
        if src.is_empty() {
            return Ok(());
        }
        if offset.saturating_add(src.len()) > self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "weight async H2D overflow: off={offset} n={} cap={}",
                    src.len(),
                    self.bytes
                ),
            )));
        }
        set_device(self.device)?;
        memcpy_async(
            unsafe { self.ptr.add(offset) },
            src.as_ptr(),
            src.len(),
            MEMCPY_H2D,
        )
    }

    /// Allocate one device blob and copy `tensors` in order. Records no Engine metrics.
    pub fn upload(device: i32, tensors: &[&[f32]]) -> Result<Self, AllpacaError> {
        Self::upload_f32_with_tail(device, tensors, 0)
    }

    /// Like [`upload`], with extra uninitialized bytes at the end (FFN ring slots).
    pub fn upload_f32_with_tail(
        device: i32,
        tensors: &[&[f32]],
        tail_bytes: usize,
    ) -> Result<Self, AllpacaError> {
        let packed: usize = tensors.iter().map(|t| t.len() * 4).sum();
        let this = Self::alloc(device, packed.saturating_add(tail_bytes))?;
        let mut off = 0usize;
        for t in tensors {
            let n = t.len() * 4;
            let bytes = unsafe { std::slice::from_raw_parts(t.as_ptr().cast::<u8>(), n) };
            this.copy_h2d(off, bytes)?;
            off += n;
        }
        Ok(this)
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

impl DeviceDecode {
    pub fn x_row(&self, i: usize, dim: usize) -> *mut f32 {
        unsafe { self.x.add(i * dim) }
    }

    pub fn q_row(&self, i: usize, dim: usize) -> *mut f32 {
        unsafe { self.q.add(i * dim) }
    }

    pub fn k_row(&self, i: usize, kv_dim: usize) -> *mut f32 {
        unsafe { self.k.add(i * kv_dim) }
    }

    pub fn v_row(&self, i: usize, kv_dim: usize) -> *mut f32 {
        unsafe { self.v.add(i * kv_dim) }
    }
}

struct DecodeAlloc {
    ptr: *mut u8,
    bytes: usize,
    dim: usize,
    kv_dim: usize,
    ff_dim: usize,
    vocab: usize,
    max_kv: usize,
    batch: usize,
    logits_rows: usize,
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
        let b = self.batch.max(1);
        let logits_rows = self.logits_rows.max(1);
        DeviceDecode {
            x: next(self.dim * b),
            normed: next(self.dim * b),
            q: next(self.dim * b),
            k: next(self.kv_dim * b),
            v: next(self.kv_dim * b),
            attn: next(self.dim * b),
            proj: next(self.dim * b),
            gate: next(self.ff_dim * b),
            up: next(self.ff_dim * b),
            logits: next(self.vocab * logits_rows),
            packed_k: next(self.max_kv * self.kv_dim),
            packed_v: next(self.max_kv * self.kv_dim),
            max_kv: self.max_kv,
        }
    }
}

fn decode_floats(
    dim: usize,
    kv_dim: usize,
    ff_dim: usize,
    vocab: usize,
    max_kv: usize,
    batch: usize,
    logits_rows: usize,
) -> usize {
    let b = batch.max(1);
    let rows = logits_rows.max(1);
    b * (dim * 5 + kv_dim * 2 + ff_dim * 2) + vocab * rows + max_kv * kv_dim * 2
}

/// Persistent device workspace used by the Engine for GEMM / attention / logits.
pub struct LmHeadScope<'a> {
    flag: &'a AtomicU64,
}

impl Drop for LmHeadScope<'_> {
    fn drop(&mut self) {
        self.flag.store(0, Ordering::Relaxed);
    }
}

struct DequantAlloc {
    ptr: *mut u8,
    bytes: usize,
    n_floats: usize,
}

unsafe impl Send for DequantAlloc {}
unsafe impl Sync for DequantAlloc {}

struct PinnedLogits {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for PinnedLogits {}

pub struct HipWorkspace {
    pub device: i32,
    scratch: *mut u8,
    scratch_bytes: usize,
    pinned: Mutex<PinnedLogits>,
    decode: Mutex<Option<DecodeAlloc>>,
    dequant: Mutex<Option<DequantAlloc>>,
    activation_h2d: AtomicU64,
    activation_d2h: AtomicU64,
    profiling: AtomicU64,
    lm_head: AtomicU64,
    profile_span: AtomicU64,
    logits_inflight: AtomicU64,
    profile: Mutex<ProfileState>,
}

unsafe impl Send for HipWorkspace {}
unsafe impl Sync for HipWorkspace {}

impl HipWorkspace {
    pub fn new(device: i32, scratch_bytes: usize, vocab: usize) -> Result<Self, AllpacaError> {
        let scratch_bytes = scratch_bytes.max(4096);
        let scratch = device_malloc(scratch_bytes, device)?;
        let pinned_bytes = vocab.max(1) * 4;
        let pinned_ptr = match host_malloc(pinned_bytes) {
            Ok(p) => p,
            Err(_) => std::ptr::null_mut(),
        };
        Ok(Self {
            device,
            scratch,
            scratch_bytes,
            pinned: Mutex::new(PinnedLogits {
                ptr: pinned_ptr,
                bytes: if pinned_ptr.is_null() { 0 } else { pinned_bytes },
            }),
            decode: Mutex::new(None),
            dequant: Mutex::new(None),
            activation_h2d: AtomicU64::new(0),
            activation_d2h: AtomicU64::new(0),
            profiling: AtomicU64::new(0),
            lm_head: AtomicU64::new(0),
            profile_span: AtomicU64::new(0),
            logits_inflight: AtomicU64::new(0),
            profile: Mutex::new(ProfileState::default()),
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        let mut n = self.scratch_bytes as u64;
        if let Ok(g) = self.pinned.lock() {
            if !g.ptr.is_null() {
                n += g.bytes as u64;
            }
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

    /// Start a HIP decode sample. Kernel ABI calls until [`end_decode_profile`] are timed.
    pub fn begin_decode_profile(&self) {
        self.profiling.store(1, Ordering::Relaxed);
        self.lm_head.store(0, Ordering::Relaxed);
        self.profile_span.store(0, Ordering::Relaxed);
        let mut g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
        g.used = 0;
        g.ops.clear();
        g.full = true;
        g.wall_start = Some(std::time::Instant::now());
    }

    /// Classify the next F32/F16 GEMM ABI calls as lm_head until the guard drops.
    pub fn lm_head_scope(&self) -> LmHeadScope<'_> {
        self.lm_head.store(1, Ordering::Relaxed);
        LmHeadScope { flag: &self.lm_head }
    }

    /// Time consecutive Q8 GEMVs as one profile span (fewer HIP event queries).
    pub fn profile_q8<T>(
        &self,
        f: impl FnOnce() -> Result<T, AllpacaError>,
    ) -> Result<T, AllpacaError> {
        if self.profiling.load(Ordering::Relaxed) == 0 {
            return f();
        }
        self.profile_span.store(1, Ordering::Relaxed);
        let result = self.time_op(DecodeOp::Q8Gemm, f);
        self.profile_span.store(0, Ordering::Relaxed);
        result
    }

    /// Last/EMA decode buckets in microseconds.
    pub fn decode_profile(&self) -> DecodeProfile {
        let g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
        DecodeProfile {
            q8_gemm_us: g.ema_ns[0] as f64 / 1_000.0,
            attention_us: g.ema_ns[1] as f64 / 1_000.0,
            lm_head_us: g.ema_ns[2] as f64 / 1_000.0,
            other_gpu_us: g.ema_ns[3] as f64 / 1_000.0,
            host_us: g.ema_ns[4] as f64 / 1_000.0,
        }
    }

    /// Finish the sample after the last kernel so HIP events are complete.
    pub fn end_decode_profile(&self) {
        self.profiling.store(0, Ordering::Relaxed);
        self.lm_head.store(0, Ordering::Relaxed);
        self.profile_span.store(0, Ordering::Relaxed);
        let mut g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
        let wall_ns = g
            .wall_start
            .take()
            .map(|t| t.elapsed().as_nanos() as u64)
            .unwrap_or(0);
        if g.full {
            let mut buckets = [0u64; 5];
            #[cfg(all(feature = "hip", allpaca_hip_runtime))]
            {
                let _ = stream_synchronize();
                for i in 0..g.used.min(g.ops.len()).min(g.pairs.len()) {
                    let op = g.ops[i] as usize;
                    if op >= 4 {
                        continue;
                    }
                    buckets[op] += event_elapsed_ns(g.pairs[i][0], g.pairs[i][1]);
                }
            }
            let gpu_ns = buckets[0] + buckets[1] + buckets[2] + buckets[3];
            buckets[4] = wall_ns.saturating_sub(gpu_ns);
            for i in 0..5 {
                g.ema_ns[i] = ema_ns(g.ema_ns[i], buckets[i]);
            }
        } else {
            let gpu_ns = g.ema_ns[0] + g.ema_ns[1] + g.ema_ns[2] + g.ema_ns[3];
            g.ema_ns[4] = ema_ns(g.ema_ns[4], wall_ns.saturating_sub(gpu_ns));
        }
        g.used = 0;
        g.ops.clear();
        g.full = false;
    }

    fn timed_gpu<T>(
        &self,
        op: DecodeOp,
        f: impl FnOnce() -> Result<T, AllpacaError>,
    ) -> Result<T, AllpacaError> {
        if self.profile_span.load(Ordering::Relaxed) != 0 {
            return f();
        }
        self.time_op(op, f)
    }

    fn time_op<T>(
        &self,
        op: DecodeOp,
        f: impl FnOnce() -> Result<T, AllpacaError>,
    ) -> Result<T, AllpacaError> {
        if self.profiling.load(Ordering::Relaxed) == 0 {
            return f();
        }
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let pair = {
                let mut g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
                if g.used >= g.pairs.len() {
                    match (event_create(), event_create()) {
                        (Ok(a), Ok(b)) => g.pairs.push([a, b]),
                        (Ok(a), Err(e)) => {
                            unsafe { ffi::allpaca_hip_event_destroy(a) };
                            return Err(e);
                        }
                        (Err(e), Ok(b)) => {
                            unsafe { ffi::allpaca_hip_event_destroy(b) };
                            return Err(e);
                        }
                        (Err(e), Err(_)) => return Err(e),
                    }
                }
                g.pairs[g.used]
            };
            let _ = event_record(pair[0]);
            let result = f();
            let _ = event_record(pair[1]);
            {
                let mut g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
                g.ops.push(op as u8);
                g.used += 1;
            }
            result
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = op;
            f()
        }
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
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            DecodeOp::OtherGpu
        };
        self.timed_gpu(op, || gemm_nt_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_q8_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::Q8Gemm, || gemm_nt_q8_0_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_q4_k_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::OtherGpu, || gemm_nt_q4_k_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_q6_k_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::OtherGpu, || gemm_nt_q6_k_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_q5_k_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::OtherGpu, || gemm_nt_q5_k_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_f16_dev(
        &self,
        a: *const f32,
        b: *const u16,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            DecodeOp::OtherGpu
        };
        self.timed_gpu(op, || gemm_nt_f16_device(a, b, c, m, n, k))
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
        self.ensure_decode_batch(dim, kv_dim, ff_dim, vocab, max_kv, 1, 1)
    }

    /// Same workspace as [`ensure_decode`], sized for `batch` token rows (prefill M=N).
    /// `logits_rows` is 1 for last-token prefill logits, or `batch` for packed linear verify.
    pub fn ensure_decode_batch(
        &self,
        dim: usize,
        kv_dim: usize,
        ff_dim: usize,
        vocab: usize,
        max_kv: usize,
        batch: usize,
        logits_rows: usize,
    ) -> Result<DeviceDecode, AllpacaError> {
        let max_kv = max_kv.max(1);
        let batch = batch.max(1);
        let logits_rows = logits_rows.max(1);
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
                    || a.batch < batch
                    || a.logits_rows < logits_rows
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
            let floats = decode_floats(dim, kv_dim, ff_dim, vocab, max_kv, batch, logits_rows);
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
                batch,
                logits_rows,
            });
        }
        let vocab = guard.as_ref().map(|a| a.vocab).unwrap_or(vocab);
        let pinned_rows = guard.as_ref().map(|a| a.logits_rows).unwrap_or(logits_rows);
        drop(guard);
        let _ = self.ensure_pinned_logits(vocab.saturating_mul(pinned_rows));
        Ok(self
            .decode
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .expect("decode alloc")
            .ptrs())
    }

    fn ensure_pinned_logits(&self, n_floats: usize) -> Result<*mut u8, AllpacaError> {
        let need = n_floats.max(1) * 4;
        let mut g = self.pinned.lock().unwrap_or_else(|e| e.into_inner());
        if !g.ptr.is_null() && g.bytes >= need {
            return Ok(g.ptr);
        }
        if !g.ptr.is_null() {
            host_free(g.ptr);
            g.ptr = std::ptr::null_mut();
            g.bytes = 0;
        }
        let ptr = host_malloc(need)?;
        g.ptr = ptr;
        g.bytes = need;
        Ok(ptr)
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_dequant_q8_0(blocks, out, n, std::ptr::null_mut()) };
                status(rc, "dequant_q8_0")
            })
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_q8_0(table, token as i32, out, dim, std::ptr::null_mut())
                };
                status(rc, "embed_gather_q8_0")
            })
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
        self.start_logits_d2h(device_logits, host.len())?;
        self.wait_logits_d2h(host)
    }

    /// Queue logits D2H into pinned host memory (no stream wait).
    pub fn start_logits_d2h(&self, device_logits: *const f32, n: usize) -> Result<(), AllpacaError> {
        if device_logits.is_null() || n == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid logits D2H".into(),
            )));
        }
        let bytes = n * 4;
        let pinned = self.ensure_pinned_logits(n)?;
        memcpy(pinned, device_logits.cast(), bytes, MEMCPY_D2H_ASYNC)?;
        self.logits_inflight.store(n as u64, Ordering::Relaxed);
        self.add_d2h(bytes);
        Ok(())
    }

    /// Wait for a queued logits D2H and copy pinned bytes into `host`.
    pub fn wait_logits_d2h(&self, host: &mut [f32]) -> Result<(), AllpacaError> {
        let n = self.logits_inflight.swap(0, Ordering::Relaxed) as usize;
        if n == 0 {
            return Ok(());
        }
        stream_synchronize()?;
        let g = self.pinned.lock().unwrap_or_else(|e| e.into_inner());
        if g.ptr.is_null() || g.bytes < n * 4 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "pinned logits missing after D2H".into(),
            )));
        }
        let copy_n = n.min(host.len());
        unsafe {
            std::ptr::copy_nonoverlapping(g.ptr.cast::<f32>(), host.as_mut_ptr(), copy_n);
        }
        Ok(())
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather(table, token as i32, out, dim, std::ptr::null_mut())
                };
                status(rc, "embed_gather")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = token;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather unavailable".into(),
            )))
        }
    }

    pub fn embed_gather_f16(
        &self,
        table: *const u16,
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_f16(table, token as i32, out, dim, std::ptr::null_mut())
                };
                status(rc, "embed_gather_f16")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = token;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_f16 unavailable".into(),
            )))
        }
    }

    pub fn embed_gather_rows(
        &self,
        table: *const f32,
        tokens: &[u32],
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed gather".into(),
            )));
        }
        let id_bytes = tokens.len() * 4;
        if id_bytes > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed gather tokens exceed scratch".into(),
            )));
        }
        self.bind()?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        memcpy(self.scratch, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_rows(
                        table,
                        self.scratch.cast(),
                        out,
                        tokens.len() as i32,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather unavailable".into(),
            )))
        }
    }

    pub fn embed_gather_f16_rows(
        &self,
        table: *const u16,
        tokens: &[u32],
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed gather".into(),
            )));
        }
        let id_bytes = tokens.len() * 4;
        if id_bytes > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed gather tokens exceed scratch".into(),
            )));
        }
        self.bind()?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        memcpy(self.scratch, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_f16_rows(
                        table,
                        self.scratch.cast(),
                        out,
                        tokens.len() as i32,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_f16_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_f16 unavailable".into(),
            )))
        }
    }

    pub fn embed_gather_q8_0_rows(
        &self,
        table: *const u8,
        tokens: &[u32],
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if tokens.is_empty() {
            return Ok(());
        }
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid q8_0 embed gather".into(),
            )));
        }
        let id_bytes = tokens.len() * 4;
        if id_bytes > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed gather tokens exceed scratch".into(),
            )));
        }
        self.bind()?;
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        memcpy(self.scratch, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_q8_0_rows(
                        table,
                        self.scratch.cast(),
                        out,
                        tokens.len() as i32,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_q8_0_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_q8_0 unavailable".into(),
            )))
        }
    }

    pub fn copy_d2h_f32(&self, src: *const f32, dst: &mut [f32]) -> Result<(), AllpacaError> {
        if dst.is_empty() {
            return Ok(());
        }
        if src.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid D2H copy".into(),
            )));
        }
        self.bind()?;
        let bytes = dst.len() * 4;
        memcpy(dst.as_mut_ptr().cast(), src.cast(), bytes, MEMCPY_D2H)?;
        self.add_d2h(bytes);
        Ok(())
    }

    pub fn copy_h2d_f32(&self, dst: *mut f32, src: &[f32]) -> Result<(), AllpacaError> {
        if src.is_empty() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid H2D copy".into(),
            )));
        }
        self.bind()?;
        let bytes = src.len() * 4;
        memcpy(dst.cast(), src.as_ptr().cast(), bytes, MEMCPY_H2D)?;
        self.add_h2d(bytes);
        Ok(())
    }

    pub fn copy_d2d_f32(&self, dst: *mut f32, src: *const f32, n: usize) -> Result<(), AllpacaError> {
        if n == 0 {
            return Ok(());
        }
        if dst.is_null() || src.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid D2D copy".into(),
            )));
        }
        self.bind()?;
        memcpy(dst.cast(), src.cast(), n * 4, MEMCPY_D2D)
    }

    pub fn scale_f32(&self, ptr: *mut f32, n: usize, scale: f32) -> Result<(), AllpacaError> {
        if n == 0 || ptr.is_null() {
            return Ok(());
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_scale_f32(ptr, n as i32, scale, std::ptr::null_mut())
                };
                status(rc, "scale_f32")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = scale;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP scale_f32 unavailable".into(),
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_residual_add(x, residual, n, std::ptr::null_mut()) };
                status(rc, "residual_add")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP residual_add unavailable".into(),
            )))
        }
    }

    pub fn residual_rmsnorm(
        &self,
        x: *mut f32,
        residual: *const f32,
        weight: *const f32,
        out: *mut f32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || residual.is_null() || weight.is_null() || out.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid residual_rmsnorm".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_residual_rmsnorm(
                        x,
                        residual,
                        weight,
                        out,
                        n,
                        eps,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "residual_rmsnorm")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = eps;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP residual_rmsnorm unavailable".into(),
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
        self.rmsnorm_device_rows(x, weight, out, 1, n, eps)
    }

    pub fn rmsnorm_device_rows(
        &self,
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if rows == 0 {
            return Ok(());
        }
        if x.is_null() || weight.is_null() || out.is_null() || rows < 0 || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device rmsnorm".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_rmsnorm_rows(x, weight, out, rows, n, eps, std::ptr::null_mut())
                };
                status(rc, "rmsnorm_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = eps;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rmsnorm unavailable".into(),
            )))
        }
    }

    pub fn rope_qk_device(
        &self,
        q: *mut f32,
        k: *mut f32,
        n_q_heads: i32,
        n_k_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        self.rope_qk_device_rows(q, k, 1, n_q_heads, n_k_heads, head_dim, pos, theta)
    }

    pub fn rope_qk_device_rows(
        &self,
        q: *mut f32,
        k: *mut f32,
        n_tok: i32,
        n_q_heads: i32,
        n_k_heads: i32,
        head_dim: i32,
        start_pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 {
            return Ok(());
        }
        if q.is_null() || k.is_null() || n_tok < 0 || n_q_heads <= 0 || n_k_heads <= 0 || head_dim <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device rope_qk".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_rope_qk_rows(
                        q,
                        k,
                        n_tok,
                        n_q_heads,
                        n_k_heads,
                        head_dim,
                        start_pos,
                        theta,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "rope_qk_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (start_pos, theta);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rope_qk unavailable".into(),
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_rope(x, n_heads, head_dim, pos, theta, std::ptr::null_mut()) };
                status(rc, "rope")
            })
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
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_silu_mul(gate, up, n, std::ptr::null_mut()) };
                status(rc, "silu_mul")
            })
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
            self.timed_gpu(DecodeOp::Attention, || {
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
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (n_q, n_kv, n_heads, n_kv_heads, head_dim, causal);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP attention unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_device_masked(
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
        mask: *const u8,
    ) -> Result<(), AllpacaError> {
        if q.is_null() || k.is_null() || v.is_null() || out.is_null() || mask.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device masked attention".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::Attention, || {
                let rc = unsafe {
                    ffi::allpaca_attention_masked(
                        q,
                        k,
                        v,
                        out,
                        n_q,
                        n_kv,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        mask,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "attention_masked")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (n_q, n_kv, n_heads, n_kv_heads, head_dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP attention_masked unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_masked(
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
        mask: &[u8],
    ) -> Result<(), AllpacaError> {
        let mask_n = (n_q as usize).saturating_mul(n_kv as usize);
        if mask.len() < mask_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention mask size mismatch".into(),
            )));
        }
        let mask_bytes = mask_n;
        let need = q.len() * 4 + k.len() * 4 + v.len() * 4 + out.len() * 4 + mask_bytes;
        if need > self.scratch_bytes {
            let dq = device_malloc(q.len() * 4, self.device)?;
            let dk = match device_malloc(k.len() * 4, self.device) {
                Ok(p) => p,
                Err(e) => {
                    device_free(dq, q.len() * 4);
                    return Err(e);
                }
            };
            let dv = match device_malloc(v.len() * 4, self.device) {
                Ok(p) => p,
                Err(e) => {
                    device_free(dq, q.len() * 4);
                    device_free(dk, k.len() * 4);
                    return Err(e);
                }
            };
            let dout = match device_malloc(out.len() * 4, self.device) {
                Ok(p) => p,
                Err(e) => {
                    device_free(dq, q.len() * 4);
                    device_free(dk, k.len() * 4);
                    device_free(dv, v.len() * 4);
                    return Err(e);
                }
            };
            let dm = match device_malloc(mask_bytes, self.device) {
                Ok(p) => p,
                Err(e) => {
                    device_free(dq, q.len() * 4);
                    device_free(dk, k.len() * 4);
                    device_free(dv, v.len() * 4);
                    device_free(dout, out.len() * 4);
                    return Err(e);
                }
            };
            let result = (|| {
                self.bind()?;
                memcpy(dq, q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D)?;
                memcpy(dk, k.as_ptr().cast(), k.len() * 4, MEMCPY_H2D)?;
                memcpy(dv, v.as_ptr().cast(), v.len() * 4, MEMCPY_H2D)?;
                memcpy(dm, mask.as_ptr(), mask_bytes, MEMCPY_H2D)?;
                self.add_h2d(q.len() * 4 + k.len() * 4 + v.len() * 4 + mask_bytes);
                self.attention_device_masked(
                    dq.cast(),
                    dk.cast(),
                    dv.cast(),
                    dout.cast(),
                    n_q,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    dm,
                )?;
                memcpy(out.as_mut_ptr().cast(), dout, out.len() * 4, MEMCPY_D2H)?;
                self.add_d2h(out.len() * 4);
                Ok(())
            })();
            device_free(dq, q.len() * 4);
            device_free(dk, k.len() * 4);
            device_free(dv, v.len() * 4);
            device_free(dout, out.len() * 4);
            device_free(dm, mask_bytes);
            return result;
        }
        self.bind()?;
        let dq = self.scratch;
        let dk = unsafe { self.scratch.add(q.len() * 4) };
        let dv = unsafe { self.scratch.add((q.len() + k.len()) * 4) };
        let dout = unsafe { self.scratch.add((q.len() + k.len() + v.len()) * 4) };
        let dm = unsafe { self.scratch.add((q.len() + k.len() + v.len() + out.len()) * 4) };
        memcpy(dq, q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D)?;
        memcpy(dk, k.as_ptr().cast(), k.len() * 4, MEMCPY_H2D)?;
        memcpy(dv, v.as_ptr().cast(), v.len() * 4, MEMCPY_H2D)?;
        memcpy(dm, mask.as_ptr(), mask_bytes, MEMCPY_H2D)?;
        self.add_h2d(q.len() * 4 + k.len() * 4 + v.len() * 4 + mask_bytes);
        self.attention_device_masked(
            dq.cast(),
            dk.cast(),
            dv.cast(),
            dout.cast(),
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            dm,
        )?;
        memcpy(out.as_mut_ptr().cast(), dout, out.len() * 4, MEMCPY_D2H)?;
        self.add_d2h(out.len() * 4);
        Ok(())
    }
}

impl Drop for HipWorkspace {
    fn drop(&mut self) {
        device_free(self.scratch, self.scratch_bytes);
        if let Ok(mut g) = self.pinned.lock() {
            host_free(g.ptr);
            g.ptr = std::ptr::null_mut();
            g.bytes = 0;
        }
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
    fn stage_q8_0_blocks_keeps_scale_and_qs() {
        let mut packed = vec![0u8; Q8_PACKED_BLOCK];
        packed[0] = 0x00;
        packed[1] = 0x3c; // f16 1.0
        packed[2] = 5u8;
        packed[33] = 0xfb; // -5 as i8
        let staged = stage_q8_0_blocks(&packed).unwrap();
        assert_eq!(staged.len(), Q8_STAGED_BLOCK);
        let scale = f32::from_le_bytes(staged[0..4].try_into().unwrap());
        assert!((scale - 1.0).abs() < 1e-3, "staged scale {scale}");
        assert_eq!(staged[Q8_STAGED_QS], 5);
        assert_eq!(staged[Q8_STAGED_QS + 31], 0xfb);
    }

    fn pack_q8_block(scale_f16: u16, qs0: u8) -> Vec<u8> {
        let mut packed = vec![0u8; Q8_PACKED_BLOCK];
        packed[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        packed[2] = qs0;
        packed
    }

    /// GEMV staging is 36-byte column-major: F32 scale then 32 qs, N rows of K-blocks.
    #[test]
    fn stage_q8_0_gemv_uses_36_byte_column_major() {
        let mut packed = Vec::new();
        for col in 0..8u8 {
            let mut blk = pack_q8_block(0x3c00, col + 1);
            blk[33] = 0x80 + col;
            packed.extend_from_slice(&blk);
        }
        let staged = stage_q8_0_gemv(&packed, 8, 32).unwrap();
        assert_eq!(staged.len(), 8 * Q8_STAGED_BLOCK);
        for c in 0..8 {
            let dst = c * Q8_STAGED_BLOCK;
            let scale = f32::from_le_bytes(staged[dst..dst + 4].try_into().unwrap());
            assert!((scale - 1.0).abs() < 1e-3, "scale col {c} = {scale}");
            assert_eq!(staged[dst + Q8_STAGED_QS], c as u8 + 1);
            assert_eq!(staged[dst + Q8_STAGED_QS + 31], 0x80 + c as u8);
        }
    }

    /// One column's K-blocks stay consecutive 36-byte strides (not interleaved with other cols).
    #[test]
    fn stage_q8_0_gemv_keeps_k_blocks_consecutive_per_column() {
        let n = 9usize;
        let k = 64usize;
        let nblocks = 2usize;
        let mut packed = vec![0u8; n * nblocks * Q8_PACKED_BLOCK];
        for col in 0..n {
            for blk in 0..nblocks {
                let src = (col * nblocks + blk) * Q8_PACKED_BLOCK;
                packed[src..src + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                packed[src + 2] = (col as u8) + 1;
                packed[src + 3] = blk as u8 + 10;
            }
        }
        let staged = stage_q8_0_gemv(&packed, n, k).unwrap();
        assert_eq!(staged.len(), n * nblocks * Q8_STAGED_BLOCK);
        for blk in 0..nblocks {
            let dst = blk * Q8_STAGED_BLOCK;
            assert_eq!(staged[dst + Q8_STAGED_QS], 1);
            assert_eq!(staged[dst + Q8_STAGED_QS + 1], blk as u8 + 10);
        }
        let col8 = 8 * nblocks * Q8_STAGED_BLOCK;
        assert_eq!(staged[col8 + Q8_STAGED_QS], 9);
        assert_eq!(staged[col8 + Q8_STAGED_QS + 1], 10);
    }

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
            (64, 128, 64),
            (1, 129, 33),
            (1, 1024, 64),
            (1, 2048, 2048),
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
            if m == 64 && n == 128 && k == 64 {
                let t0 = std::time::Instant::now();
                for _ in 0..20 {
                    gemm_nt(&a, &b, &mut hip, m as i32, n as i32, k as i32)
                        .expect("HIP gemm_nt 64x128x64 timing");
                }
                eprintln!("gemm_nt 64x128x64 x20 {:?}", t0.elapsed());
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

    #[cfg(feature = "hip")]
    fn packed_q8_scale(blk: &[u8]) -> f32 {
        let h = u16::from_le_bytes([blk[0], blk[1]]);
        super::f16_bits_to_f32(h)
    }

    /// CPU Q8_0 integer GEMV: per-32 A tile → int8+scale, int32 MAC with weight qs.
    #[cfg(feature = "hip")]
    fn cpu_q8_intdot_gemm(a: &[f32], b_packed: &[u8], m: usize, n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(32);
        let mut c = vec![0.0f32; m * n];
        for row in 0..m {
            let arow = &a[row * k..(row + 1) * k];
            for col in 0..n {
                let mut acc = 0.0f32;
                let brow = &b_packed[col * nblocks * 34..(col + 1) * nblocks * 34];
                for b in 0..nblocks {
                    let start = b * 32;
                    let mut amax = 0.0f32;
                    for i in 0..32 {
                        let v = if start + i < k { arow[start + i] } else { 0.0 };
                        amax = amax.max(v.abs());
                    }
                    let da = if amax == 0.0 { 0.0 } else { amax / 127.0 };
                    let inv = if da == 0.0 { 0.0 } else { 1.0 / da };
                    let off = b * 34;
                    let dw = packed_q8_scale(&brow[off..]);
                    let mut sumi = 0i32;
                    for i in 0..32 {
                        let av = if start + i < k { arow[start + i] } else { 0.0 };
                        let qa = (av * inv).round().clamp(-127.0, 127.0) as i8;
                        let qw = brow[off + 2 + i] as i8;
                        sumi += i32::from(qa) * i32::from(qw);
                    }
                    acc += da * dw * (sumi as f32);
                }
                c[row * n + col] = acc;
            }
        }
        c
    }

    #[cfg(feature = "hip")]
    fn dequant_q8_matrix(b_packed: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(32);
        let mut b_deq = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_packed[row * nblocks * 34..(row + 1) * nblocks * 34];
            for bi in 0..nblocks {
                let off = bi * 34;
                let d = packed_q8_scale(&blk[off..]);
                for j in 0..32 {
                    let idx = bi * 32 + j;
                    if idx < k {
                        b_deq[row * k + idx] = d * (blk[off + 2 + j] as i8 as f32);
                    }
                }
            }
        }
        b_deq
    }

    #[cfg(feature = "hip")]
    fn q8_intdot_shapes() -> [(usize, usize, usize); 12] {
        [
            (1, 8, 32),
            (2, 5, 64),
            (1, 17, 32),
            (3, 9, 96),
            (1, 1, 33),
            (1, 4, 33),
            (2, 7, 33),
            (1, 33, 64),
            (1, 64, 128),
            (1, 128, 256),
            (1, 96, 33),
            (2, 40, 64),
        ]
    }

    #[cfg(feature = "hip")]
    fn fill_q8_gemm_case(si: usize, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
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
        (a, b_q8, b_f)
    }

    /// Integer A×Q8 GEMV kernel must match CPU int-dot on small random shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q8_0_matches_cpu_q8_intdot() {
        if device_count() == 0 {
            return;
        }
        for (si, (m, n, k)) in q8_intdot_shapes().iter().copied().enumerate() {
            let (a, b_q8, _) = fill_q8_gemm_case(si, m, n, k);
            let cpu = cpu_q8_intdot_gemm(&a, &b_q8, m, n, k);
            let mut hip = vec![0.0f32; m * n];
            gemm_nt_q8_0_intdot(&a, &b_q8, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_q8_0_intdot must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "q8 intdot mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    #[cfg(feature = "hip")]
    fn fill_q4k_gemm_case(si: usize, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
        }
        let mut b_q = Vec::new();
        for row in 0..n {
            b_q.extend(allpaca_cpu::quantize_q4_k(&b_f[row * k..(row + 1) * k]));
        }
        (a, b_q, b_f)
    }

    #[cfg(feature = "hip")]
    fn dequant_q4k_matrix(b_q: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(Q4_K_BLOCK_ELEMS);
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_q[row * nblocks * Q4_K_BLOCK_BYTES..(row + 1) * nblocks * Q4_K_BLOCK_BYTES];
            let mut tmp = vec![0.0f32; nblocks * Q4_K_BLOCK_ELEMS];
            allpaca_cpu::dequant_q4_k(blk, nblocks * Q4_K_BLOCK_ELEMS, &mut tmp).unwrap();
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// Public `gemm_nt_q4_k` must match CPU dequant + NT GEMM on small Q4_K shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q4_k_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 256), (1, 8, 256), (2, 3, 256), (1, 4, 512)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q, _) = fill_q4k_gemm_case(si, m, n, k);
            let b_deq = dequant_q4k_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_q4_k(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_q4_k must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q4_k mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    #[cfg(feature = "hip")]
    fn fill_q6k_gemm_case(si: usize, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
        }
        let mut b_q = Vec::new();
        for row in 0..n {
            b_q.extend(allpaca_cpu::quantize_q6_k(&b_f[row * k..(row + 1) * k]));
        }
        (a, b_q, b_f)
    }

    #[cfg(feature = "hip")]
    fn dequant_q6k_matrix(b_q: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(Q6_K_BLOCK_ELEMS);
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_q[row * nblocks * Q6_K_BLOCK_BYTES..(row + 1) * nblocks * Q6_K_BLOCK_BYTES];
            let mut tmp = vec![0.0f32; nblocks * Q6_K_BLOCK_ELEMS];
            allpaca_cpu::dequant_q6_k(blk, nblocks * Q6_K_BLOCK_ELEMS, &mut tmp).unwrap();
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// GEM mixed-quant: fused Q6_K GEMV must match CPU dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q6_k_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 256), (1, 8, 256), (2, 3, 256), (1, 4, 512)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q, _) = fill_q6k_gemm_case(si, m, n, k);
            let b_deq = dequant_q6k_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_q6_k(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_q6_k must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q6_k mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    #[cfg(feature = "hip")]
    fn fill_q5k_gemm_case(si: usize, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
        }
        let mut b_q = Vec::new();
        for row in 0..n {
            b_q.extend(allpaca_cpu::quantize_q5_k(&b_f[row * k..(row + 1) * k]));
        }
        (a, b_q, b_f)
    }

    #[cfg(feature = "hip")]
    fn dequant_q5k_matrix(b_q: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(Q5_K_BLOCK_ELEMS);
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_q[row * nblocks * Q5_K_BLOCK_BYTES..(row + 1) * nblocks * Q5_K_BLOCK_BYTES];
            let mut tmp = vec![0.0f32; nblocks * Q5_K_BLOCK_ELEMS];
            allpaca_cpu::dequant_q5_k(blk, nblocks * Q5_K_BLOCK_ELEMS, &mut tmp).unwrap();
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// GEM mixed-quant: fused Q5_K GEMV must match CPU dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q5_k_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 256), (1, 8, 256), (2, 3, 256), (1, 4, 512)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q, _) = fill_q5k_gemm_case(si, m, n, k);
            let b_deq = dequant_q5k_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_q5_k(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_q5_k must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q5_k mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// §12: tiled Q4_K GEMV matches untiled baseline; before/after wall times recorded.
    #[cfg(feature = "hip")]
    #[test]
    fn q4_k_tiled_gemv_matches_untiled_and_records_before_after() {
        if device_count() == 0 {
            return;
        }
        let m = 1usize;
        let n = 128usize;
        let k = 4096usize;
        let (a, b_q, _) = fill_q4k_gemm_case(42, m, n, k);
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_q4_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32)
            .expect("tiled Q4_K");
        gemm_nt_q4_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32)
            .expect("untiled Q4_K");
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "tiled vs untiled mismatch idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
        let warm = 3usize;
        let reps = 20usize;
        for _ in 0..warm {
            gemm_nt_q4_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32).unwrap();
            gemm_nt_q4_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_q4_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32).unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_q4_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).unwrap();
        }
        let after_us = t1.elapsed().as_micros();
        eprintln!(
            "§12 Q4_K GEMV 1x{n}x{k} x{reps}: untiled(before)={before_us}us tiled(after)={after_us}us"
        );
        assert!(
            before_us > 0 && after_us > 0,
            "before/after tracers must record positive times"
        );
    }

    /// §12: tiled Q5_K GEMV matches untiled; tile picker records cols.
    #[cfg(feature = "hip")]
    #[test]
    fn q5_k_tiled_gemv_matches_untiled_and_records_tile_cols() {
        if device_count() == 0 {
            return;
        }
        set_infinity_cache_budget(64 << 20);
        let m = 1usize;
        let n = 128usize;
        let k = 4096usize;
        let (a, b_q, _) = fill_q5k_gemm_case(7, m, n, k);
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_q5_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).expect("tiled Q5_K");
        gemm_nt_q5_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32)
            .expect("untiled Q5_K");
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "Q5_K tiled vs untiled idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
        assert!(
            last_gemv_tile_cols() >= 32,
            "§12 picker must record a tile; cols={}",
            last_gemv_tile_cols()
        );
        let warm = 2usize;
        let reps = 10usize;
        for _ in 0..warm {
            gemm_nt_q5_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32).unwrap();
            gemm_nt_q5_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_q5_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32).unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_q5_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).unwrap();
        }
        let after_us = t1.elapsed().as_micros();
        eprintln!(
            "§12 Q5_K GEMV 1x{n}x{k} x{reps} cols={}: untiled={before_us}us tiled={after_us}us",
            last_gemv_tile_cols()
        );
        assert!(before_us > 0 && after_us > 0);
    }

    /// §12: tiled Q6_K GEMV matches untiled baseline.
    #[cfg(feature = "hip")]
    #[test]
    fn q6_k_tiled_gemv_matches_untiled_and_records_before_after() {
        if device_count() == 0 {
            return;
        }
        set_infinity_cache_budget(64 << 20);
        let m = 1usize;
        let n = 128usize;
        let k = 4096usize;
        let (a, b_q, _) = fill_q6k_gemm_case(11, m, n, k);
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_q6_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).expect("tiled Q6_K");
        gemm_nt_q6_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32)
            .expect("untiled Q6_K");
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "Q6_K tiled vs untiled idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
        let warm = 2usize;
        let reps = 10usize;
        for _ in 0..warm {
            gemm_nt_q6_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32).unwrap();
            gemm_nt_q6_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_q6_k_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32).unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_q6_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).unwrap();
        }
        let after_us = t1.elapsed().as_micros();
        eprintln!(
            "§12 Q6_K GEMV 1x{n}x{k} x{reps} cols={}: untiled={before_us}us tiled={after_us}us",
            last_gemv_tile_cols()
        );
        assert!(before_us > 0 && after_us > 0);
    }

    /// §12: tiled F16 GEMV matches untiled; before/after wall times recorded.
    #[cfg(feature = "hip")]
    #[test]
    fn f16_tiled_gemv_matches_untiled_and_records_before_after() {
        if device_count() == 0 {
            return;
        }
        let m = 1usize;
        let n = 128usize;
        let k = 4096usize;
        let a = vec![0.1f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.01;
        }
        let b_f16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_f16).collect();
        let a_bytes = a.len() * 4;
        let b_bytes = b_f16.len() * 2;
        let c_bytes = m * n * 4;
        let da = device_malloc(a_bytes, 0).unwrap();
        let db = device_malloc(b_bytes, 0).unwrap();
        let dc = device_malloc(c_bytes, 0).unwrap();
        set_device(0).unwrap();
        memcpy(da, a.as_ptr().cast(), a_bytes, MEMCPY_H2D).unwrap();
        memcpy(db, b_f16.as_ptr().cast(), b_bytes, MEMCPY_H2D).unwrap();
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_f16_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        memcpy(tiled.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H).unwrap();
        gemm_nt_f16_untiled_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
            .unwrap();
        memcpy(untiled.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H).unwrap();
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "F16 tiled vs untiled idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
        let warm = 3usize;
        let reps = 20usize;
        for _ in 0..warm {
            gemm_nt_f16_untiled_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
            gemm_nt_f16_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_f16_untiled_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_f16_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
        }
        let after_us = t1.elapsed().as_micros();
        eprintln!(
            "§12 F16 GEMV 1x{n}x{k} x{reps}: untiled(before)={before_us}us tiled(after)={after_us}us"
        );
        device_free(da, a_bytes);
        device_free(db, b_bytes);
        device_free(dc, c_bytes);
        assert!(before_us > 0 && after_us > 0);
    }

    /// §12: tiled F32 lm_head GEMV matches untiled; before/after wall times recorded.
    #[cfg(feature = "hip")]
    #[test]
    fn f32_lm_head_tiled_gemv_matches_untiled_and_records_before_after() {
        if device_count() == 0 {
            return;
        }
        let m = 1usize;
        let n = 4096usize;
        let k = 4096usize;
        let a = vec![0.1f32; m * k];
        let mut b = vec![0.0f32; n * k];
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 19) as f32) * 0.01;
        }
        let a_bytes = a.len() * 4;
        let b_bytes = b.len() * 4;
        let c_bytes = m * n * 4;
        let da = device_malloc(a_bytes, 0).unwrap();
        let db = device_malloc(b_bytes, 0).unwrap();
        let dc = device_malloc(c_bytes, 0).unwrap();
        set_device(0).unwrap();
        memcpy(da, a.as_ptr().cast(), a_bytes, MEMCPY_H2D).unwrap();
        memcpy(db, b.as_ptr().cast(), b_bytes, MEMCPY_H2D).unwrap();
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        memcpy(tiled.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H).unwrap();
        gemm_nt_untiled_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
            .unwrap();
        memcpy(untiled.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H).unwrap();
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "F32 tiled vs untiled idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
        let warm = 3usize;
        let reps = 20usize;
        for _ in 0..warm {
            gemm_nt_untiled_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
            gemm_nt_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_untiled_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_device(da.cast(), db.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        }
        let after_us = t1.elapsed().as_micros();
        eprintln!(
            "§12 F32 lm_head GEMV 1x{n}x{k} x{reps}: untiled(before)={before_us}us tiled(after)={after_us}us"
        );
        device_free(da, a_bytes);
        device_free(db, b_bytes);
        device_free(dc, c_bytes);
        assert!(before_us > 0 && after_us > 0);
    }

    /// Public `gemm_nt_q8_0` must match CPU dequant + NT GEMM on small random shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q8_0_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        for (si, (m, n, k)) in q8_intdot_shapes().iter().copied().enumerate() {
            let (a, b_q8, _) = fill_q8_gemm_case(si, m, n, k);
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
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

    /// Public `gemm_nt_f16` must match CPU F16-dequant + NT GEMM on decode shapes (M≤8).
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_f16_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (1, 8, 4),
            (1, 64, 32),
            (1, 129, 33),
            (2, 3, 5),
            (7, 17, 13),
            (8, 32, 16),
            (1, 128, 64),
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
            let b_f16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_f16).collect();
            let b_deq: Vec<f32> = b_f16.iter().copied().map(super::f16_bits_to_f32).collect();
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_f16(&a, &b_f16, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_f16 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_f16 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Prefill-wide F16 NT GEMM (M>8) must match CPU F16-dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_f16_wide_m_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(16, 64, 128), (9, 17, 64), (32, 33, 96)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let mut a = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; n * k];
            for (i, v) in a.iter_mut().enumerate() {
                *v = ((i * 17 + si * 43) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in b_f.iter_mut().enumerate() {
                *v = ((i * 13 + si * 11) % 50) as f32 * 0.02 - 0.5;
            }
            let b_f16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_f16).collect();
            let b_deq: Vec<f32> = b_f16.iter().copied().map(super::f16_bits_to_f32).collect();
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_f16(&a, &b_f16, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_f16 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_f16 wide mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// F16 decode GEMV at lm_head-ish n must stay in the same ballpark as F32 GEMV.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_f16_wide_n_decode_is_not_orders_slower_than_f32() {
        if device_count() == 0 {
            return;
        }
        let m = 1usize;
        let n = 32_768usize;
        let k = 2_048usize;
        let a = vec![0.1f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i % 17) as f32) * 0.01;
        }
        let b_f16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_f16).collect();
        let a_b = a.len() * 4;
        let b32_b = b_f.len() * 4;
        let b16_b = b_f16.len() * 2;
        let c_b = n * 4;
        set_device(0).unwrap();
        let da = device_malloc(a_b, 0).unwrap();
        let db32 = device_malloc(b32_b, 0).unwrap();
        let db16 = device_malloc(b16_b, 0).unwrap();
        let dc = device_malloc(c_b, 0).unwrap();
        memcpy(da, a.as_ptr().cast(), a_b, MEMCPY_H2D).unwrap();
        memcpy(db32, b_f.as_ptr().cast(), b32_b, MEMCPY_H2D).unwrap();
        memcpy(db16, b_f16.as_ptr().cast(), b16_b, MEMCPY_H2D).unwrap();
        gemm_nt_device(da.cast(), db32.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        gemm_nt_f16_device(da.cast(), db16.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        stream_synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            gemm_nt_device(da.cast(), db32.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        }
        stream_synchronize().unwrap();
        let f32_ms = t0.elapsed().as_secs_f64() * 1_000.0 / 10.0;
        let t1 = std::time::Instant::now();
        for _ in 0..10 {
            gemm_nt_f16_device(da.cast(), db16.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
        }
        stream_synchronize().unwrap();
        let f16_ms = t1.elapsed().as_secs_f64() * 1_000.0 / 10.0;
        device_free(da, a_b);
        device_free(db32, b32_b);
        device_free(db16, b16_b);
        device_free(dc, c_b);
        eprintln!("device GEMV 1x{n}x{k} f32={f32_ms:.2}ms f16={f16_ms:.2}ms");
        assert!(
            f16_ms < f32_ms * 8.0 + 2.0,
            "F16 GEMV {f16_ms:.2}ms vs F32 {f32_ms:.2}ms on 1x{n}x{k}"
        );
    }

    /// Draft GPU is device 1. F16 GEMV must run there, not only on the 9070.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_f16_runs_on_every_hip_device() {
        let n = device_count();
        if n == 0 {
            return;
        }
        let m = 1i32;
        let cols = 64i32;
        let k = 128i32;
        let a = vec![0.25f32; k as usize];
        let b: Vec<u16> = (0..cols * k)
            .map(|i| allpaca_cpu::f32_to_f16(((i % 9) as f32) * 0.05))
            .collect();
        let mut c = vec![0.0f32; cols as usize];
        for id in 0..n {
            let t0 = std::time::Instant::now();
            set_device(id).expect("set_device");
            let da = device_malloc(a.len() * 4, id).expect("A");
            let db = device_malloc(b.len() * 2, id).expect("B");
            let dc = device_malloc(c.len() * 4, id).expect("C");
            memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).unwrap();
            memcpy(db, b.as_ptr().cast(), b.len() * 2, MEMCPY_H2D).unwrap();
            gemm_nt_f16_device(da.cast(), db.cast(), dc.cast(), m, cols, k)
                .unwrap_or_else(|e| panic!("F16 GEMV on device {id}: {e}"));
            stream_synchronize().unwrap_or_else(|e| panic!("sync device {id}: {e}"));
            memcpy(c.as_mut_ptr().cast(), dc, c.len() * 4, MEMCPY_D2H).unwrap();
            device_free(da, a.len() * 4);
            device_free(db, b.len() * 2);
            device_free(dc, c.len() * 4);
            eprintln!(
                "f16 gemv device {id} ok in {:?}",
                t0.elapsed()
            );
        }
    }

    /// Prefill-wide Q8 NT GEMM (M>8) must match CPU dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q8_0_wide_m_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(16, 64, 128), (9, 17, 64), (32, 33, 96)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q8, _) = fill_q8_gemm_case(si + 40, m, n, k);
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_q8_0(&a, &b_q8, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_q8_0 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q8_0 wide mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Int-dot GEMV vs float GEMV: small max abs error on random vectors.
    #[cfg(feature = "hip")]
    #[test]
    fn q8_intdot_gemv_matches_float_gemv_within_max_abs() {
        let shapes = [(1, 8, 32), (1, 64, 128), (1, 128, 256), (1, 96, 4096)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q8, _) = fill_q8_gemm_case(si + 11, m, n, k);
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
            let mut cpu_f = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu_f, m, n, k);
            let cpu_i = cpu_q8_intdot_gemm(&a, &b_q8, m, n, k);
            let mut max_abs = 0.0f32;
            for i in 0..m * n {
                max_abs = max_abs.max((cpu_i[i] - cpu_f[i]).abs());
            }
            if device_count() > 0 {
                let mut hip = vec![0.0f32; m * n];
                gemm_nt_q8_0_intdot(&a, &b_q8, &mut hip, m as i32, n as i32, k as i32)
                    .expect("HIP gemm_nt_q8_0_intdot must run when a device is present");
                for i in 0..m * n {
                    max_abs = max_abs.max((hip[i] - cpu_f[i]).abs());
                    assert!(
                        gemm_close(hip[i], cpu_i[i]),
                        "HIP int-dot vs CPU int-dot shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                        hip[i],
                        cpu_i[i]
                    );
                }
            }
            let limit = 0.02 * (1.0 + (k as f32 / 32.0).sqrt());
            assert!(
                max_abs <= limit,
                "int-dot vs float GEMV max abs {max_abs} exceeds {limit} on {m}x{n}x{k}"
            );
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
        let mut b_q8_packed = Vec::new();
        for row in 0..n {
            b_q8_packed.extend(quantize_q8_0(&b_f[row * k..(row + 1) * k]));
        }
        let b_q8 = stage_q8_0_gemv(&b_q8_packed, n, k).expect("stage q8 gemv");
        let b_deq = dequant_q8_matrix(&b_q8_packed, n, k);
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

    /// Prefill-shaped batched RMSNorm must match per-row CPU RMSNorm.
    #[cfg(feature = "hip")]
    #[test]
    fn rmsnorm_rows_matches_cpu_per_row() {
        if device_count() == 0 {
            return;
        }
        let eps = 1e-6f32;
        let shapes = [(16, 32), (9, 8), (1, 64)];
        for (si, (rows, n)) in shapes.iter().copied().enumerate() {
            let mut x = vec![0.0f32; rows * n];
            let mut weight = vec![0.0f32; n];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in weight.iter_mut().enumerate() {
                *v = 0.5 + ((i * 13 + si) % 20) as f32 * 0.05;
            }
            let mut cpu = vec![0.0f32; rows * n];
            for r in 0..rows {
                allpaca_cpu::rmsnorm(
                    &x[r * n..(r + 1) * n],
                    &weight,
                    eps,
                    &mut cpu[r * n..(r + 1) * n],
                )
                .expect("CPU rmsnorm");
            }
            let mut hip = vec![0.0f32; rows * n];
            rmsnorm_rows(&x, &weight, &mut hip, rows as i32, n as i32, eps)
                .expect("HIP rmsnorm_rows must run when a device is present");
            for i in 0..rows * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "rmsnorm_rows mismatch rows={rows} n={n} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Prefill-shaped batched Q/K RoPE must match per-token CPU RoPE.
    #[cfg(feature = "hip")]
    #[test]
    fn rope_qk_rows_matches_cpu_per_token() {
        if device_count() == 0 {
            return;
        }
        let theta = 10_000.0f32;
        let shapes = [(16, 4, 2, 8, 3), (9, 2, 2, 4, 0), (1, 4, 1, 16, 7)];
        for (si, (n_tok, n_q, n_k, head_dim, start_pos)) in shapes.iter().copied().enumerate() {
            let q_stride = n_q * head_dim;
            let k_stride = n_k * head_dim;
            let mut q = vec![0.0f32; n_tok * q_stride];
            let mut k = vec![0.0f32; n_tok * k_stride];
            for (i, v) in q.iter_mut().enumerate() {
                *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in k.iter_mut().enumerate() {
                *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
            }
            let mut cpu_q = q.clone();
            let mut cpu_k = k.clone();
            for t in 0..n_tok {
                let pos = start_pos + t;
                for h in 0..n_q {
                    let s = t * q_stride + h * head_dim;
                    allpaca_cpu::rope(&mut cpu_q[s..s + head_dim], pos, theta).expect("CPU q rope");
                }
                for h in 0..n_k {
                    let s = t * k_stride + h * head_dim;
                    allpaca_cpu::rope(&mut cpu_k[s..s + head_dim], pos, theta).expect("CPU k rope");
                }
            }
            rope_qk_rows(
                &mut q,
                &mut k,
                n_tok as i32,
                n_q as i32,
                n_k as i32,
                head_dim as i32,
                start_pos as i32,
                theta,
            )
            .expect("HIP rope_qk_rows must run when a device is present");
            for i in 0..q.len() {
                assert!(
                    gemm_close(q[i], cpu_q[i]),
                    "rope_qk_rows Q mismatch n_tok={n_tok} idx={i} hip={} cpu={}",
                    q[i],
                    cpu_q[i]
                );
            }
            for i in 0..k.len() {
                assert!(
                    gemm_close(k[i], cpu_k[i]),
                    "rope_qk_rows K mismatch n_tok={n_tok} idx={i} hip={} cpu={}",
                    k[i],
                    cpu_k[i]
                );
            }
        }
    }

    /// Prefill-shaped batched F32 embed gather must match table rows.
    #[cfg(feature = "hip")]
    #[test]
    fn embed_gather_rows_matches_table_rows() {
        if device_count() == 0 {
            return;
        }
        let vocab = 8usize;
        let dim = 16usize;
        let tokens = [0u32, 3, 7, 1, 3, 2];
        let mut table = vec![0.0f32; vocab * dim];
        for (i, v) in table.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        let mut cpu = vec![0.0f32; tokens.len() * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            let s = tok as usize * dim;
            cpu[r * dim..(r + 1) * dim].copy_from_slice(&table[s..s + dim]);
        }
        let mut hip = vec![0.0f32; tokens.len() * dim];
        embed_gather_rows(&table, &tokens, &mut hip, dim as i32)
            .expect("HIP embed_gather_rows must run when a device is present");
        for i in 0..cpu.len() {
            assert!(
                gemm_close(hip[i], cpu[i]),
                "embed_gather_rows mismatch idx={i} hip={} cpu={}",
                hip[i],
                cpu[i]
            );
        }
    }

    /// Prefill-shaped batched F16 embed gather must match dequantized table rows.
    #[cfg(feature = "hip")]
    #[test]
    fn embed_gather_f16_rows_matches_table_rows() {
        if device_count() == 0 {
            return;
        }
        let vocab = 8usize;
        let dim = 16usize;
        let tokens = [0u32, 3, 7, 1, 3, 2];
        let mut table_f = vec![0.0f32; vocab * dim];
        for (i, v) in table_f.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        let table: Vec<u16> = table_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_f16)
            .collect();
        let mut cpu = vec![0.0f32; tokens.len() * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            let s = tok as usize * dim;
            for i in 0..dim {
                cpu[r * dim + i] = super::f16_bits_to_f32(table[s + i]);
            }
        }
        let mut hip = vec![0.0f32; tokens.len() * dim];
        embed_gather_f16_rows(&table, &tokens, &mut hip, dim as i32)
            .expect("HIP embed_gather_f16_rows must run when a device is present");
        for i in 0..cpu.len() {
            assert!(
                gemm_close(hip[i], cpu[i]),
                "embed_gather_f16_rows mismatch idx={i} hip={} cpu={}",
                hip[i],
                cpu[i]
            );
        }
    }

    /// Prefill-shaped batched Q8 embed gather must match dequantized table rows.
    #[cfg(feature = "hip")]
    #[test]
    fn embed_gather_q8_0_rows_matches_dequant_rows() {
        if device_count() == 0 {
            return;
        }
        let vocab = 8usize;
        let dim = 32usize;
        let tokens = [0u32, 3, 7, 1, 3, 2];
        let mut table = vec![0.0f32; vocab * dim];
        for (i, v) in table.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        let mut packed = Vec::new();
        for row in 0..vocab {
            packed.extend(quantize_q8_0(&table[row * dim..(row + 1) * dim]));
        }
        let b_deq = dequant_q8_matrix(&packed, vocab, dim);
        let mut cpu = vec![0.0f32; tokens.len() * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            let s = tok as usize * dim;
            cpu[r * dim..(r + 1) * dim].copy_from_slice(&b_deq[s..s + dim]);
        }
        let mut hip = vec![0.0f32; tokens.len() * dim];
        embed_gather_q8_0_rows(&packed, &tokens, &mut hip, dim as i32)
            .expect("HIP embed_gather_q8_0_rows must run when a device is present");
        for i in 0..cpu.len() {
            assert!(
                gemm_close(hip[i], cpu[i]),
                "embed_gather_q8_0_rows mismatch idx={i} hip={} cpu={}",
                hip[i],
                cpu[i]
            );
        }
    }
}
