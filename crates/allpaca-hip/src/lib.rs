//! HIP kernel FFI. Default build is a stub so the workspace compiles without ROCm.
//!
//! `--features hip` enables the HIP path when a SDK is found at build time
//! (`HIP_PATH` / `ROCM_PATH` / common install locations). Without an SDK the
//! crate still compiles; [`device_count`] returns 0 and GPU tests skip.

mod gemv_tile;

pub use gemv_tile::{
    gemv_tile_working_bytes, infinity_cache_from_l2, profile_gemv_tile_cols,
    profiled_gemv_tile_cols, select_gemv_tile_cols, select_gemv_tile_cols_heuristic,
};

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
use allpaca_types::DeviceKind;
use allpaca_types::{AllpacaError, BandwidthBenchmark, DegradedError, DeviceInfo};

/// GGUF Q8_0 packed block (f16 scale + 32 qs).
pub const Q8_PACKED_BLOCK: usize = 34;
/// Device-staged Q8_0: f32 scale at 0, qs at 4, 36-byte stride (4-byte aligned).
pub const Q8_STAGED_BLOCK: usize = 36;
pub const Q8_STAGED_QS: usize = 4;
/// GGUF Q4_K super-block (f16 d + f16 dmin + 12 scale bytes + 128 qs).
pub const Q4_K_BLOCK_ELEMS: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// GGUF TQ2_0 ternary block (qs[64] + f16 d).
pub const TQ2_0_BLOCK_ELEMS: usize = 256;
pub const TQ2_0_BLOCK_BYTES: usize = 66;
/// GGUF TQ1_0 ternary block (qs[48] + qh[4] + f16 d).
pub const TQ1_0_BLOCK_ELEMS: usize = 256;
pub const TQ1_0_BLOCK_BYTES: usize = 54;
/// GGUF Q6_K super-block (128 ql + 64 qh + 16 scales + f16 d).
pub const Q6_K_BLOCK_ELEMS: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// GGUF Q5_K super-block (f16 d + f16 dmin + 12 scales + 32 qh + 128 qs).
pub const Q3_K_BLOCK_ELEMS: usize = 256;
pub const Q3_K_BLOCK_BYTES: usize = 110;
pub const Q5_K_BLOCK_ELEMS: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// ROCmFP4 block (ggml_type_id=100): 16 packed nibble bytes + 2 UE4M3 scales.
pub const ROCMFP4_BLOCK_ELEMS: usize = 32;
pub const ROCMFP4_BLOCK_BYTES: usize = 18;
/// ggml MXFP4 (ggml_type_id=39): E8M0 + 16 packed nibbles / 32 elems.
pub const MXFP4_BLOCK_ELEMS: usize = 32;
pub const MXFP4_BLOCK_BYTES: usize = 17;

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
    /// Decode forwards seen; every [`PROFILE_EVERY`]th is event-timed.
    forwards: u64,
    /// lm_head time of this sampled forward, measured between stream syncs.
    lm_head_ns: u64,
}

/// Event-time one decode forward in this many. Timing every forward cost two
/// `hipEventRecord`s per kernel (~2.6k per token on a 64-layer model) plus a
/// full sync and ~1.3k elapsed-time queries at token end; on Windows each
/// marker can also split the command buffer. Buckets are EMAs, so a sample
/// every few tokens tracks them just as well. The workspace's first forward is
/// never sampled: it pays one-time costs (module load, buffer growth) whose
/// event times exceed its wall time and would seed the EMAs wrong.
const PROFILE_EVERY: u64 = 16;

impl Default for ProfileState {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            ops: Vec::new(),
            used: 0,
            full: false,
            wall_start: None,
            ema_ns: [0; 5],
            forwards: 0,
            lm_head_ns: 0,
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
fn event_ring<const N: usize>() -> Result<[*mut std::ffi::c_void; N], AllpacaError> {
    let mut out = [std::ptr::null_mut(); N];
    for i in 0..N {
        match event_create() {
            Ok(e) => out[i] = e,
            Err(err) => {
                for ev in out.iter().take(i) {
                    unsafe {
                        let _ = ffi::allpaca_hip_event_destroy(*ev);
                    }
                }
                return Err(err);
            }
        }
    }
    Ok(out)
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_record(ev: *mut std::ffi::c_void) -> Result<(), AllpacaError> {
    event_record_on(ev, std::ptr::null_mut())
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_record_on(
    ev: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    let rc = unsafe { ffi::allpaca_hip_event_record(ev, stream) };
    status(rc, "event_record")
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_synchronize(ev: *mut std::ffi::c_void) -> Result<(), AllpacaError> {
    let rc = unsafe { ffi::allpaca_hip_event_synchronize(ev) };
    status(rc, "event_synchronize")
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn stream_wait_event(
    stream: *mut std::ffi::c_void,
    ev: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    let rc = unsafe { ffi::allpaca_hip_stream_wait_event(stream, ev) };
    status(rc, "stream_wait_event")
}

const SWIGLU_STREAMS: usize = 4;
const SWIGLU_EVENTS: usize = 3;

fn make_swiglu_lanes(
    device: i32,
) -> (
    [*mut std::ffi::c_void; SWIGLU_STREAMS],
    [*mut std::ffi::c_void; SWIGLU_EVENTS],
) {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let nulls = (
            [std::ptr::null_mut(); SWIGLU_STREAMS],
            [std::ptr::null_mut(); SWIGLU_EVENTS],
        );
        if set_device(device).is_err() {
            return nulls;
        }
        let mut streams = [std::ptr::null_mut(); SWIGLU_STREAMS];
        let mut events = [std::ptr::null_mut(); SWIGLU_EVENTS];
        let mut n = 0usize;
        for stream in &mut streams {
            let mut s = std::ptr::null_mut();
            let rc = unsafe { ffi::allpaca_hip_stream_create(&mut s) };
            if rc != 0 || s.is_null() {
                break;
            }
            *stream = s;
            n += 1;
        }
        if n < 2 {
            destroy_swiglu_pair(streams, events);
            return nulls;
        }
        for event in &mut events {
            match event_create() {
                Ok(e) => *event = e,
                Err(_) => {
                    destroy_swiglu_pair(streams, events);
                    return nulls;
                }
            }
        }
        (streams, events)
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = device;
        (
            [std::ptr::null_mut(); SWIGLU_STREAMS],
            [std::ptr::null_mut(); SWIGLU_EVENTS],
        )
    }
}

fn destroy_swiglu_pair(
    streams: [*mut std::ffi::c_void; SWIGLU_STREAMS],
    events: [*mut std::ffi::c_void; SWIGLU_EVENTS],
) {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    unsafe {
        for ev in events {
            if !ev.is_null() {
                let _ = ffi::allpaca_hip_event_destroy(ev);
            }
        }
        for stream in streams {
            if !stream.is_null() {
                let _ = ffi::allpaca_hip_stream_destroy(stream);
            }
        }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (streams, events);
    }
}

#[cfg(all(feature = "hip", allpaca_hip_runtime))]
fn event_elapsed_ns(start: *mut std::ffi::c_void, stop: *mut std::ffi::c_void) -> u64 {
    let mut us = 0.0f64;
    let rc = unsafe { ffi::allpaca_hip_event_elapsed_us(start, stop, &mut us) };
    if rc != ALLPACA_OK {
        0
    } else {
        // Short kernels read back as zero or slightly negative when the GPU is
        // shared (the two markers land in one submission): the op still ran,
        // so it keeps its bucket at the timer's floor.
        ((us * 1_000.0).round() as i64).max(1) as u64
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
        pub fn allpaca_hip_device_arch(
            id: i32,
            arch: *mut i8,
            arch_len: i32,
            l2_bytes: *mut i32,
        ) -> i32;
        pub fn allpaca_hip_can_p2p(src: i32, dst: i32, can: *mut i32) -> i32;
        pub fn allpaca_hip_enable_peer_access(src: i32, dst: i32) -> i32;
        pub fn allpaca_hip_malloc(ptr: *mut *mut c_void, bytes: usize, device: i32) -> i32;
        pub fn allpaca_hip_memset(
            ptr: *mut c_void,
            value: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_hip_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_hip_free_async(ptr: *mut c_void, stream: *mut c_void) -> i32;
        pub fn allpaca_hip_host_malloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        pub fn allpaca_hip_host_free(ptr: *mut c_void) -> i32;
        pub fn allpaca_hip_host_register(ptr: *mut c_void, bytes: usize, flags: u32) -> i32;
        pub fn allpaca_hip_host_unregister(ptr: *mut c_void) -> i32;
        pub fn allpaca_dstorage_alloc(
            handle: *mut *mut c_void,
            hip_ptr: *mut *mut c_void,
            bytes: usize,
            device: i32,
        ) -> i32;
        pub fn allpaca_dstorage_read(
            handle: *mut c_void,
            path: *const u16,
            file_offset: u64,
            buffer_offset: u64,
            bytes: usize,
        ) -> i32;
        pub fn allpaca_dstorage_free(handle: *mut c_void) -> i32;
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
        pub fn allpaca_hip_touch_resident(ptr: *mut c_void, bytes: usize, device: i32) -> i32;
        #[allow(dead_code)]
        pub fn allpaca_hip_benchmark(
            src: i32,
            dst: i32,
            bytes: usize,
            p2p_gbps: *mut f64,
            host_gbps: *mut f64,
        ) -> i32;
        pub fn allpaca_hip_mem_used(device: i32, used_bytes: *mut u64) -> i32;
        pub fn allpaca_hip_local_headroom(device: i32, bytes: *mut u64) -> i32;
        pub fn allpaca_hip_mem_info(
            device: i32,
            free_bytes: *mut u64,
            total_bytes: *mut u64,
        ) -> i32;
        pub fn allpaca_hip_set_device(device: i32) -> i32;
        pub fn allpaca_hip_blas_warmup(device: i32) -> i32;
        pub fn allpaca_hip_memcpy_batch(
            dsts: *mut *mut c_void,
            srcs: *mut *mut c_void,
            sizes: *mut usize,
            count: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_hip_has_memcpy_batch() -> i32;
        pub fn allpaca_hip_stream_synchronize(stream: *mut c_void) -> i32;
        pub fn allpaca_hip_stream_create(stream: *mut *mut c_void) -> i32;
        pub fn allpaca_hip_stream_destroy(stream: *mut c_void) -> i32;
        pub fn allpaca_hip_stream_wait_event(stream: *mut c_void, event: *mut c_void) -> i32;
        pub fn allpaca_hip_event_create(event: *mut *mut c_void) -> i32;
        pub fn allpaca_hip_event_destroy(event: *mut c_void) -> i32;
        pub fn allpaca_hip_event_record(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn allpaca_hip_event_synchronize(event: *mut c_void) -> i32;
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
        pub fn allpaca_gemm_nt_tq2_0(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_tq1_0(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_rocmfp4(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_rocmfp4_untiled(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_rocmfp4_fast(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_p48fp4(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_rocmfp4_f32(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            fast: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_mxfp4(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_mxfp4_untiled(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_mxfp4_swiglu_gate_up(
            a: *const f32,
            bg: *const u8,
            bu: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            limit: f32,
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
        pub fn allpaca_gemm_nt_q3_k(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q2_0(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_q2_0_moe_ffn(
            x: *const f32,
            pool: *const u8,
            slots: *const i32,
            weights: *const f32,
            n_pick: i32,
            h: *mut f32,
            out: *mut f32,
            dim: i32,
            ff: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_q4x_routed_ffn(
            x: *const f32,
            scores: *const f32,
            pool: *const u8,
            n_experts: i32,
            n_pick: i32,
            scale: f32,
            picks: *mut u8,
            h: *mut f32,
            out: *mut f32,
            dim: i32,
            ff: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_silu_scale(x: *mut f32, n: i32, s: f32, stream: *mut c_void) -> i32;
        pub fn allpaca_q4x_stream_norm(
            x: *const f32,
            gains: *const f32,
            xn: *mut f32,
            rows: i32,
            hc: i32,
            dim: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_q4x_gated_mean(
            xn: *const f32,
            gate: *const f32,
            mixed: *mut f32,
            rows: i32,
            hc: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_q4x_combine(
            streams: *mut f32,
            out: *const f32,
            logits: *const f32,
            rows: i32,
            hc: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_q4x_ffn_sum(
            routed: *const f32,
            shared: *const f32,
            gate: *const f32,
            out: *mut f32,
            rows: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_moe_ffn_scratch_bytes(rows: i32, dim: i32, ff: i32, pairs: i32) -> u64;
        pub fn allpaca_det_silu_mul(
            g: *const f32,
            u: *const f32,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_hip_set_exact_rows(on: i32);
        pub fn allpaca_copy_f32_kernel(dst: *mut f32, src: *const f32, n: i32, stream: *mut c_void) -> i32;
        pub fn allpaca_moe_route_softmax_topk(
            logits: *const f32,
            rows: i32,
            n_exp: i32,
            k: i32,
            x: *const f32,
            shexp_gate: *const f32,
            dim: i32,
            w_out: *mut f32,
            row_out: *mut i32,
            exp_out: *mut i32,
            row_start: *mut i32,
            shexp_scale: *mut f32,
            disabled: *const u8,
            counts: *mut u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_scale_rows(x: *mut f32, scale: *const f32, rows: i32, dim: i32, stream: *mut c_void) -> i32;
        pub fn allpaca_topk_chunks(
            x: *const f32,
            rows: i32,
            n: i32,
            chunk: i32,
            k: i32,
            out_v: *mut f32,
            out_i: *mut i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rope_neox_rows(
            x: *mut f32,
            n_tok: i32,
            n_heads: i32,
            head_dim: i32,
            start_pos: i32,
            theta: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_dflash_conv(
            h: *const f32,
            dynamic: *const f32,
            base: *const f32,
            out: *mut f32,
            rows: i32,
            block: i32,
            dim: i32,
            kernel: i32,
            group: i32,
            side: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_moe_ffn_kquant(
            x: *const f32,
            xq: *const i8,
            xd: *const f32,
            rows: i32,
            dim: i32,
            ff: i32,
            pairs: i32,
            pair_row: *const i32,
            pair_expert: *const i32,
            pair_w: *const f32,
            row_start: *const i32,
            groups: i32,
            group_start: *const i32,
            row_pairs: *const i32,
            max_group: i32,
            gate: *const u8,
            gate_qt: i32,
            gate_stride: u64,
            up: *const u8,
            up_qt: i32,
            up_stride: u64,
            down: *const u8,
            down_qt: i32,
            down_stride: u64,
            scratch: *mut c_void,
            out: *mut f32,
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
        pub fn allpaca_gemm_nt_bf16(
            a: *const f32,
            b: *const u16,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_bf16_untiled(
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
        pub fn allpaca_attention_sinks(
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
            sinks: *const f32,
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
        pub fn allpaca_layernorm_rows(
            x: *const f32,
            weight: *const f32,
            bias: *const f32,
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
        pub fn allpaca_silu_mul(gate: *mut f32, up: *const f32, n: i32, stream: *mut c_void)
            -> i32;
        pub fn allpaca_gelu_mul(gate: *mut f32, up: *const f32, n: i32, stream: *mut c_void)
            -> i32;
        pub fn allpaca_relu_sqr(x: *mut f32, n: i32, stream: *mut c_void) -> i32;
        pub fn allpaca_deinterleave_q_gate(
            q_full: *const f32,
            q: *mut f32,
            gate: *mut f32,
            n_heads: i32,
            head_dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_sigmoid_mul(
            x: *mut f32,
            gate: *const f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_scale_heads_sigmoid(
            attn: *mut f32,
            scores: *const f32,
            n_tok: i32,
            n_heads: i32,
            head_dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rmsnorm_heads(
            x: *mut f32,
            weight: *const f32,
            n_heads: i32,
            head_dim: i32,
            eps: f32,
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
        pub fn allpaca_bias_act_rows(
            x: *mut f32,
            bias: *const f32,
            rows: i32,
            n: i32,
            act: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_vision_rope_rows(
            x: *mut f32,
            pos_row: *const i32,
            pos_col: *const i32,
            n_tok: i32,
            n_heads: i32,
            head_dim: i32,
            theta: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_scatter_add_scale_rows(
            src: *const f32,
            idx: *const i32,
            w: *const f32,
            acc: *mut f32,
            m: i32,
            dim: i32,
            n_tok: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_hc_mix_pre(
            packed: *const f32,
            pre: *const f32,
            curs: *mut f32,
            n_tok: i32,
            hc: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_hc_post(
            x: *const f32,
            residual: *const f32,
            post: *const f32,
            comb: *const f32,
            out: *mut f32,
            n_tok: i32,
            hc: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_scale_f32(x: *mut f32, n: i32, scale: f32, stream: *mut c_void) -> i32;
        pub fn allpaca_residual_rmsnorm(
            x: *mut f32,
            residual: *const f32,
            weight: *const f32,
            out: *mut f32,
            n: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gemm_nt_q8_0_pair(
            a: *const f32,
            b1: *const u8,
            c1: *mut f32,
            b2: *const u8,
            c2: *mut f32,
            m: i32,
            n1: i32,
            n2: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_scale_add_copy(
            x: *mut f32,
            y: *mut f32,
            s: f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gdn_norm_gate(
            y: *const f32,
            z: *const f32,
            weight: *const f32,
            out: *mut f32,
            n_heads: i32,
            head_dim: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gdn_step_fused(
            qkv: *const f32,
            conv_w: *const f32,
            conv_state: *mut f32,
            conv_out: *mut f32,
            alpha: *const f32,
            dt_bias: *const f32,
            a: *const f32,
            decay: *mut f32,
            beta: *mut f32,
            state: *mut f32,
            out: *mut f32,
            n_v: i32,
            n_k: i32,
            head_v: i32,
            head_k: i32,
            conv_kernel: i32,
            eps: f32,
            interleave: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gated_delta_step(
            q: *mut f32,
            k: *mut f32,
            v: *const f32,
            decay: *const f32,
            beta: *const f32,
            state: *mut f32,
            out: *mut f32,
            n_v_heads: i32,
            head_v: i32,
            head_k: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gdn_conv1d_step(
            x: *const f32,
            kernel: *const f32,
            conv_state: *mut f32,
            out: *mut f32,
            qkv_dim: i32,
            conv_kernel: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_expand_k_heads(
            src: *const f32,
            dst: *mut f32,
            n_k: i32,
            n_v: i32,
            head_k: i32,
            interleave: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_gdn_decay_beta(
            alpha: *const f32,
            dt_bias: *const f32,
            a: *const f32,
            decay: *mut f32,
            beta: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_mamba2_conv_step(
            xbc: *const f32,
            kernel: *const f32,
            bias: *const f32,
            conv_state: *mut f32,
            out: *mut f32,
            conv_dim: i32,
            conv_kernel: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_mamba2_ssd_step(
            z: *const f32,
            xbc: *const f32,
            dt: *const f32,
            a: *const f32,
            d: *const f32,
            dt_bias: *const f32,
            norm: *const f32,
            state: *mut f32,
            out: *mut f32,
            d_inner: i32,
            d_state: i32,
            n_heads: i32,
            n_groups: i32,
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
        pub fn allpaca_rope_offset_rows(
            x: *mut f32,
            n_tok: i32,
            n_heads: i32,
            head_dim: i32,
            n_rot: i32,
            offset: i32,
            start_pos: i32,
            theta: f32,
            back: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rope_yarn_rows(
            x: *mut f32,
            n_tok: i32,
            n_heads: i32,
            head_dim: i32,
            n_rot: i32,
            offset: i32,
            start_pos: i32,
            theta: f32,
            scale: f32,
            low: f32,
            high: f32,
            back: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_rope_imrope_qk(
            q: *mut f32,
            k: *mut f32,
            n_q_heads: i32,
            n_k_heads: i32,
            head_dim: i32,
            n_rot: i32,
            pos: *const i32,
            theta: f32,
            sections: *const i32,
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
        pub fn allpaca_embed_gather_rocmfp4(
            table: *const u8,
            token: i32,
            out: *mut f32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather_rocmfp4_rows(
            table: *const u8,
            tokens: *const i32,
            out: *mut f32,
            n_tok: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather_rocmfp4_fast(
            table: *const u8,
            token: i32,
            out: *mut f32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_embed_gather_rocmfp4_fast_rows(
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

/// ISA name from HIP (`gfx906:sramecc+:xnack-`) and L2 size in bytes.
pub fn device_gcn_arch(id: i32) -> Option<(String, usize)> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut buf = [0i8; 128];
        let mut l2 = 0i32;
        let rc = unsafe { ffi::allpaca_hip_device_arch(id, buf.as_mut_ptr(), 128, &mut l2) };
        if rc != ALLPACA_OK {
            return None;
        }
        let arch = unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if arch.is_empty() {
            return None;
        }
        Some((arch, l2.max(0) as usize))
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = id;
        None
    }
}

fn arch_has_infinity_cache(arch: &str) -> bool {
    let base = arch.split(':').next().unwrap_or(arch);
    base.starts_with("gfx10") || base.starts_with("gfx11") || base.starts_with("gfx12")
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

/// ROCm 6.2 + gfx906 often reports `hipDeviceCanAccessPeer=0` and
/// `hipDeviceEnablePeerAccess` fails (`invalidDevice`), while `hipMemcpyPeer`
/// still copies. Treat a successful peer memcpy as capability.
fn peer_copy_reachable(src: i32, dst: i32) -> bool {
    const N: usize = 64 * 1024;
    let Ok(s) = device_malloc(N, src) else {
        return false;
    };
    let Ok(d) = device_malloc(N, dst) else {
        device_free(s, N);
        return false;
    };
    let ok = memcpy_peer(d, dst, s, src, N).is_ok();
    device_free(s, N);
    device_free(d, N);
    ok
}

/// Query, attempt enable, re-query. `capable` is post-enable *or* a working peer copy.
pub fn probe_peer(src: i32, dst: i32) -> PeerAccess {
    let queried = can_p2p(src, dst);
    let enabled = enable_peer_access(src, dst);
    let mut capable = can_p2p(src, dst);
    if !capable {
        capable = peer_copy_reachable(src, dst);
        if capable {
            tracing::info!(
                src,
                dst,
                queried,
                enabled,
                "P2P memcpy works without hipDeviceCanAccessPeer (typical gfx906 / no PCI_P2PDMA)"
            );
        }
    }
    tracing::debug!(
        src,
        dst,
        queried,
        enabled,
        capable,
        "P2P probe query-enable-requery"
    );
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

/// Copy `n` f32s between HIP devices. Peer DMA first; `stage` is the host bounce.
pub fn copy_dev_f32(
    dst: *mut f32,
    dst_dev: i32,
    src: *const f32,
    src_dev: i32,
    n: usize,
    stage: &mut [f32],
) -> Result<(), AllpacaError> {
    if n == 0 {
        return Ok(());
    }
    if dst.is_null() || src.is_null() {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "invalid device activation copy".into(),
        )));
    }
    if stage.len() < n {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            format!("activation staging too small: need {n} have {}", stage.len()),
        )));
    }
    let bytes = n * 4;
    if src_dev == dst_dev {
        set_device(dst_dev)?;
        return memcpy(dst.cast(), src.cast(), bytes, MEMCPY_D2D);
    }
    set_device(src_dev)?;
    stream_synchronize()?;
    if memcpy_peer(dst.cast(), dst_dev, src.cast(), src_dev, bytes).is_ok() {
        set_device(dst_dev)?;
        return Ok(());
    }
    set_device(src_dev)?;
    memcpy(stage.as_mut_ptr().cast(), src.cast(), bytes, MEMCPY_D2H)?;
    set_device(dst_dev)?;
    memcpy(dst.cast(), stage.as_ptr().cast(), bytes, MEMCPY_H2D)
}

/// Fault-in `bytes` of device memory without a host-side stream sync.
pub fn touch_resident(ptr: *mut u8, bytes: usize, device: i32) -> Result<(), AllpacaError> {
    if ptr.is_null() || bytes == 0 {
        return Ok(());
    }
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_touch_resident(ptr.cast(), bytes, device) };
        status(rc, "touch_resident")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (ptr, bytes, device);
        Ok(())
    }
}

/// Runtime enable + peer-copy vs host-staging measurement. Never assumes P2P is faster.
pub fn benchmark_transfer(
    src: i32,
    dst: i32,
    bytes: u64,
) -> Result<BandwidthBenchmark, AllpacaError> {
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

        // Formal enable is best-effort. gfx906 often copies without it.
        let _ = probe_peer(src, dst);
        let _ = probe_peer(dst, src);
        let _ = memcpy_peer(d_dst, dst, d_src, src, n);
        let t = Instant::now();
        let copy_ok = memcpy_peer(d_dst, dst, d_src, src, n).is_ok();
        let p2p_elapsed_us = t.elapsed().as_secs_f64() * 1.0e6;
        let (p2p_gbps, p2p_us, p2p_copy_ok) = if copy_ok && p2p_elapsed_us > 0.0 {
            let gbps = (bytes as f64 * 8.0 / 1.0e9) / (p2p_elapsed_us / 1.0e6);
            (gbps, p2p_elapsed_us, true)
        } else {
            (0.0, 0.0, false)
        };

        let _ = memcpy(h_pin, d_src, n, MEMCPY_D2H);
        let _ = memcpy(d_dst, h_pin, n, MEMCPY_H2D);
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

/// Stream-ordered free. Null `stream` falls back to synchronous [`device_free`].
pub fn device_free_async(ptr: *mut u8, bytes: usize, stream: *mut std::ffi::c_void) {
    if ptr.is_null() {
        return;
    }
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    unsafe {
        let _ = ffi::allpaca_hip_free_async(ptr.cast(), stream);
        ALLOCATED_BYTES.fetch_sub(bytes as u64, Ordering::Relaxed);
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (bytes, stream);
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

/// Pin existing host memory (including GGUF mmap) for async H2D.
pub fn host_register(ptr: *mut u8, bytes: usize, flags: u32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if ptr.is_null() || bytes == 0 {
            return Ok(());
        }
        let rc = unsafe { ffi::allpaca_hip_host_register(ptr.cast(), bytes, flags) };
        if rc != ALLPACA_OK {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("hipHostRegister({bytes}) failed rc={rc}"),
            )));
        }
        Ok(())
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (ptr, bytes, flags);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP host register unavailable".into(),
        )))
    }
}

pub fn host_unregister(ptr: *mut u8) {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    if !ptr.is_null() {
        unsafe {
            let _ = ffi::allpaca_hip_host_unregister(ptr.cast());
        }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = ptr;
    }
}

pub fn memcpy(dst: *mut u8, src: *const u8, bytes: usize, kind: i32) -> Result<(), AllpacaError> {
    memcpy_on(dst, src, bytes, kind, std::ptr::null_mut())
}

/// Same as [`memcpy`], queued on `stream` (null is the default stream).
pub fn memcpy_on(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    kind: i32,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_memcpy(dst.cast(), src.cast(), bytes, kind, stream) };
        status(rc, "memcpy")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, src, bytes, kind, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP memcpy unavailable".into(),
        )))
    }
}

/// Fill `bytes` of device memory with `value` (low 8 bits). Null stream is synchronous.
pub fn device_memset(dst: *mut u8, value: u8, bytes: usize) -> Result<(), AllpacaError> {
    memset_on(dst, value, bytes, std::ptr::null_mut())
}

/// Fill on `stream` (`hipMemsetAsync`). Null stream uses synchronous `hipMemset`.
pub fn memset_on(
    dst: *mut u8,
    value: u8,
    bytes: usize,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_memset(dst.cast(), i32::from(value), bytes, stream) };
        status(rc, "memset")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, value, bytes, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP memset unavailable".into(),
        )))
    }
}

/// One batched DMA for `copies` (dst, src, bytes) of a single `kind` on `stream`.
///
/// Collapses N FFI round-trips into one and lets ROCm 7.1+ issue them as a
/// single `hipMemcpyBatchAsync`. Ordering matches the per-copy loop it replaces.
///
/// # Safety
/// Every `dst` must be a live device allocation of at least `bytes`, and every
/// `src` a live host allocation of at least `bytes`, until the stream drains.
pub unsafe fn memcpy_batch_on(
    copies: &[(*mut u8, *const u8, usize)],
    kind: i32,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    if copies.is_empty() {
        return Ok(());
    }
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut dsts: Vec<*mut std::ffi::c_void> =
            copies.iter().map(|(d, _, _)| d.cast()).collect();
        let mut srcs: Vec<*mut std::ffi::c_void> =
            copies.iter().map(|(_, s, _)| s.cast_mut().cast()).collect();
        let mut sizes: Vec<usize> = copies.iter().map(|(_, _, n)| *n).collect();
        let rc = unsafe {
            ffi::allpaca_hip_memcpy_batch(
                dsts.as_mut_ptr(),
                srcs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                copies.len(),
                kind,
                stream,
            )
        };
        status(rc, "memcpy_batch")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (copies, kind, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP memcpy_batch unavailable".into(),
        )))
    }
}

/// True when the loaded runtime exports `hipMemcpyBatchAsync` (ROCm 7.1+).
/// A false here still batches at the ABI, just via per-copy async underneath.
pub fn has_memcpy_batch() -> bool {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        unsafe { ffi::allpaca_hip_has_memcpy_batch() != 0 }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        false
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

/// Wait for queued work on `stream` (null is the default compute stream).
///
/// # Safety
/// `stream` must be null or a stream this process still owns.
pub unsafe fn stream_synchronize_on(stream: *mut std::ffi::c_void) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_stream_synchronize(stream) };
        status(rc, "stream_synchronize_on")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = stream;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP stream_synchronize unavailable".into(),
        )))
    }
}

/// A non-blocking side stream for expert H2D, so weight copies overlap compute
/// on the null stream instead of serializing behind it. Joined with an event,
/// never a host-side sync (ADR 0001: scheduling stays in Rust, waits stay on GPU).
/// In-flight miss-ready events so bounce can record expert N+k before GEMM N waits.
/// Host-syncing the ring on wrap is a driver round-trip even when the event is
/// already signaled; a deeper ring keeps DMA queued across a ChatML miss wave.
pub const H2D_READY_RING: usize = 64;

pub struct H2dStream {
    stream: *mut std::ffi::c_void,
    #[allow(dead_code)] // Only read under the HIP runtime cfg.
    event: *mut std::ffi::c_void,
    /// Per-bounce-buffer completion (host-visible). Index 0/1.
    #[allow(dead_code)]
    slot_ev: [*mut std::ffi::c_void; 2],
    /// Per-expert DMA-complete events for miss GEMM (`record_ready` / `wait_ready_compute`).
    #[allow(dead_code)]
    ready_ev: [*mut std::ffi::c_void; H2D_READY_RING],
}

// The handles are only touched through the C ABI, which is internally synchronized.
unsafe impl Send for H2dStream {}
unsafe impl Sync for H2dStream {}

impl H2dStream {
    pub fn new() -> Result<Self, AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
            let rc = unsafe { ffi::allpaca_hip_stream_create(&mut stream) };
            status(rc, "stream_create")?;
            let event = match event_create() {
                Ok(e) => e,
                Err(e) => {
                    unsafe { ffi::allpaca_hip_stream_destroy(stream) };
                    return Err(e);
                }
            };
            let ev0 = match event_create() {
                Ok(e) => e,
                Err(e) => {
                    unsafe {
                        ffi::allpaca_hip_event_destroy(event);
                        ffi::allpaca_hip_stream_destroy(stream);
                    }
                    return Err(e);
                }
            };
            let ev1 = match event_create() {
                Ok(e) => e,
                Err(e) => {
                    unsafe {
                        ffi::allpaca_hip_event_destroy(ev0);
                        ffi::allpaca_hip_event_destroy(event);
                        ffi::allpaca_hip_stream_destroy(stream);
                    }
                    return Err(e);
                }
            };
            let ready_ev = match event_ring::<H2D_READY_RING>() {
                Ok(ring) => ring,
                Err(e) => {
                    unsafe {
                        ffi::allpaca_hip_event_destroy(ev1);
                        ffi::allpaca_hip_event_destroy(ev0);
                        ffi::allpaca_hip_event_destroy(event);
                        ffi::allpaca_hip_stream_destroy(stream);
                    }
                    return Err(e);
                }
            };
            Ok(Self {
                stream,
                event,
                slot_ev: [ev0, ev1],
                ready_ev,
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP stream_create unavailable".into(),
            )))
        }
    }

    pub fn raw(&self) -> *mut std::ffi::c_void {
        self.stream
    }

    /// Record the copy event on this stream, then make the compute stream wait
    /// for it. The host does not block.
    pub fn join_compute(&self) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_hip_event_record(self.event, self.stream) };
            status(rc, "h2d_event_record")?;
            let rc =
                unsafe { ffi::allpaca_hip_stream_wait_event(std::ptr::null_mut(), self.event) };
            status(rc, "h2d_stream_wait_event")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP stream_wait_event unavailable".into(),
            )))
        }
    }

    /// Block the host until every queued copy on this stream has landed.
    pub fn synchronize(&self) -> Result<(), AllpacaError> {
        unsafe { stream_synchronize_on(self.stream) }
    }

    /// Record completion of the next copy onto bounce-buffer `slot` (0 or 1).
    pub fn record_slot(&self, slot: usize) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let ev = *self.slot_ev.get(slot).ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "H2D bounce slot out of range".into(),
                ))
            })?;
            let rc = unsafe { ffi::allpaca_hip_event_record(ev, self.stream) };
            status(rc, "h2d_slot_record")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = slot;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP event record unavailable".into(),
            )))
        }
    }

    /// Host-wait until bounce-buffer `slot` is safe to overwrite.
    pub fn synchronize_slot(&self, slot: usize) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let ev = *self.slot_ev.get(slot).ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "H2D bounce slot out of range".into(),
                ))
            })?;
            let rc = unsafe { ffi::allpaca_hip_event_synchronize(ev) };
            status(rc, "h2d_slot_sync")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = slot;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP event synchronize unavailable".into(),
            )))
        }
    }

    /// Record DMA-complete for miss GEMM slot `idx` (`0..H2D_READY_RING`).
    pub fn record_ready(&self, idx: usize) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let ev = *self.ready_ev.get(idx).ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "H2D ready ring out of range".into(),
                ))
            })?;
            let rc = unsafe { ffi::allpaca_hip_event_record(ev, self.stream) };
            status(rc, "h2d_ready_record")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = idx;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP event record unavailable".into(),
            )))
        }
    }

    /// Host-wait until ready-ring `idx` can be recorded again.
    pub fn synchronize_ready(&self, idx: usize) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let ev = *self.ready_ev.get(idx).ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "H2D ready ring out of range".into(),
                ))
            })?;
            let rc = unsafe { ffi::allpaca_hip_event_synchronize(ev) };
            status(rc, "h2d_ready_sync")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = idx;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP event synchronize unavailable".into(),
            )))
        }
    }

    /// Make the current device default stream wait for ready-ring `idx` (host does not block).
    pub fn wait_ready_compute(&self, idx: usize) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let ev = *self.ready_ev.get(idx).ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "H2D ready ring out of range".into(),
                ))
            })?;
            stream_wait_event(std::ptr::null_mut(), ev)
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = idx;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP stream_wait_event unavailable".into(),
            )))
        }
    }
}

impl Drop for H2dStream {
    fn drop(&mut self) {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        unsafe {
            let _ = ffi::allpaca_hip_event_destroy(self.slot_ev[0]);
            let _ = ffi::allpaca_hip_event_destroy(self.slot_ev[1]);
            for ev in self.ready_ev {
                let _ = ffi::allpaca_hip_event_destroy(ev);
            }
            let _ = ffi::allpaca_hip_event_destroy(self.event);
            let _ = ffi::allpaca_hip_stream_destroy(self.stream);
        }
    }
}

/// Bind subsequent kernels / memcpy to `device`.
/// Runs one small rocBLAS GEMM on `device` and waits for it. The first GEMM a
/// process runs on a device pays the library's start-up there (seconds on
/// some GPUs), so a caller can pay it at load instead of in the first request.
/// Leaves `device` current on the calling thread.
pub fn warm_up_blas(device: i32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_hip_blas_warmup(device) };
        status(rc, "blas_warmup")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP blas_warmup unavailable".into(),
        )))
    }
}

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

/// Driver-measured `(free, total)` VRAM bytes for `device`.
///
/// [`allocated_bytes`] only counts what this process asked for; this counts the
/// whole board, so the scheduler sees pressure from the compositor and other
/// apps before it grows the expert pool. Cheap, but it is a driver call: sample
/// it at a few Hz off the decode path, never per token.
pub fn device_memory_info(device: i32) -> Result<(u64, u64), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut free = 0u64;
        let mut total = 0u64;
        let rc = unsafe { ffi::allpaca_hip_mem_info(device, &mut free, &mut total) };
        status(rc, "mem_info")?;
        Ok((free, total))
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP mem_info unavailable".into(),
        )))
    }
}

/// Dedicated VRAM on `device` that no process holds, when the OS reports it
/// (Windows PDH). `None` elsewhere. Use it with [`device_memory_info`] before
/// sizing large placements: on Windows `hipMemGetInfo` ignores the desktop's
/// share, and WDDM silently demotes over-budget allocations to system RAM.
pub fn device_local_headroom(device: i32) -> Option<u64> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let mut bytes = 0u64;
        let rc = unsafe { ffi::allpaca_hip_local_headroom(device, &mut bytes) };
        (rc == 0).then_some(bytes)
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = device;
        None
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
        let rc =
            unsafe { ffi::allpaca_gemm_nt_q8_0_intdot(a, b, c, m, n, k, std::ptr::null_mut()) };
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

/// Device-pointer NT GEMM with packed bfloat16 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_bf16_device(
    a: *const f32,
    b: *const u16,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_bf16(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_bf16")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_bf16 unavailable".into(),
        )))
    }
}

/// Untiled BF16 GEMV baseline.
pub fn gemm_nt_bf16_untiled_device(
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
            unsafe { ffi::allpaca_gemm_nt_bf16_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_bf16_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_bf16_untiled unavailable".into(),
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

/// Device-pointer NT GEMM with native GGUF TQ2_0 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_tq2_0_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_tq2_0(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_tq2_0")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_tq2_0 unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native GGUF TQ1_0 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_tq1_0_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_tq1_0(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_tq1_0")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_tq1_0 unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native ROCmFP4 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_rocmfp4_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_rocmfp4(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_rocmfp4")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_rocmfp4 unavailable".into(),
        )))
    }
}

/// As [`gemm_nt_rocmfp4_device`] for ROCmFP4 FAST `b` (ggml type 101, 17-byte blocks).
pub fn gemm_nt_rocmfp4_fast_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_rocmfp4_fast(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_rocmfp4_fast")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_rocmfp4_fast unavailable".into(),
        )))
    }
}

/// `C[m,n] = A[m,k] * W^T` for MoESQ paired-4:8 FP4 weights (tensor type
/// 200, see `allpaca_cpu::dequant_p48fp4`), all in device memory.
pub fn gemm_nt_p48fp4_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_p48fp4(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_p48fp4")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_p48fp4 unavailable".into(),
        )))
    }
}

/// Float-activation ROCmFP4 GEMV (the kernel the int8 path replaced), for
/// A/B and as the exact-float reference. `fast`: 17-byte type-101 blocks.
pub fn gemm_nt_rocmfp4_f32_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    fast: bool,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_gemm_nt_rocmfp4_f32(a, b, c, m, n, k, i32::from(fast), std::ptr::null_mut())
        };
        status(rc, "gemm_nt_rocmfp4_f32")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k, fast);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_rocmfp4_f32 unavailable".into(),
        )))
    }
}

/// Untiled ROCmFP4 GEMV baseline (§12 before/after).
pub fn gemm_nt_rocmfp4_untiled_device(
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
            unsafe { ffi::allpaca_gemm_nt_rocmfp4_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_rocmfp4_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_rocmfp4_untiled unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native ggml MXFP4 `b` (`C = A * dequant(B)^T`).
pub fn gemm_nt_mxfp4_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_mxfp4_device_on(a, b, c, m, n, k, std::ptr::null_mut())
}

/// Same as [`gemm_nt_mxfp4_device`], queued on `stream` (null is the default stream).
pub fn gemm_nt_mxfp4_device_on(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_mxfp4(a, b, c, m, n, k, stream) };
        status(rc, "gemm_nt_mxfp4")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_mxfp4 unavailable".into(),
        )))
    }
}

/// Device-pointer fused MXFP4 gate+up+SiLU: `C = silu(A * dequant(Bg)^T) * (A * dequant(Bu)^T)`.
/// With `limit > 0` the gate is capped at `limit` before SiLU and the up
/// projection to `±limit`.
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_swiglu_gate_up_device(
    a: *const f32,
    bg: *const u8,
    bu: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    limit: f32,
) -> Result<(), AllpacaError> {
    mxfp4_swiglu_gate_up_device_on(a, bg, bu, c, m, n, k, limit, std::ptr::null_mut())
}

/// Same as [`mxfp4_swiglu_gate_up_device`], queued on `stream` (null is the default stream).
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_swiglu_gate_up_device_on(
    a: *const f32,
    bg: *const u8,
    bu: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    limit: f32,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_mxfp4_swiglu_gate_up(a, bg, bu, c, m, n, k, limit, stream) };
        status(rc, "mxfp4_swiglu_gate_up")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, bg, bu, c, m, n, k, limit, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP mxfp4_swiglu_gate_up unavailable".into(),
        )))
    }
}

/// Untiled MXFP4 GEMV baseline.
pub fn gemm_nt_mxfp4_untiled_device(
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
            unsafe { ffi::allpaca_gemm_nt_mxfp4_untiled(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_mxfp4_untiled")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_mxfp4_untiled unavailable".into(),
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
        let rc =
            unsafe { ffi::allpaca_gemm_nt_q6_k_cols(a, b, c, m, n, k, cols, std::ptr::null_mut()) };
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

/// Device-pointer NT GEMM with native GGUF Q3_K `b` (`C = A * dequant(B)^T`);
/// `k` must be whole 256-value super-blocks.
pub fn gemm_nt_q3_k_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_q3_k(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q3_k")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q3_k unavailable".into(),
        )))
    }
}

/// Device-pointer NT GEMM with native GGUF Q5_K `b` (`C = A * dequant(B)^T`).
/// Grow-only pinned host buffer. hipHostMalloc memory is device-mapped, so
/// kernels read and write it in place: small uploads skip a blocking
/// `hipMemcpy` (~60-80 us each on Windows).
#[derive(Debug, Default)]
pub struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
}

unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// At least `bytes`, contents undefined after a grow.
    pub fn ensure(&mut self, bytes: usize) -> Result<*mut u8, AllpacaError> {
        if self.ptr.is_null() || self.bytes < bytes {
            host_free(std::mem::replace(&mut self.ptr, std::ptr::null_mut()));
            let want = bytes.max(1 << 16);
            self.ptr = host_malloc(want)?;
            self.bytes = want;
        }
        Ok(self.ptr)
    }
}

/// A pair table built on the device by [`moe_route_softmax_topk_device`]:
/// `rows * k` pairs, row-sorted, in the layout [`moe_ffn_kquant_device`] reads.
#[derive(Debug, Clone, Copy)]
pub struct MoeDeviceRoutes {
    pub w: *mut f32,
    pub row: *mut i32,
    pub expert: *mut i32,
    pub row_start: *mut i32,
    /// Per-row shared-expert gate `sigmoid(x . g)`.
    pub shexp_scale: *mut f32,
    pub rows: usize,
    pub k: usize,
}

impl MoeDeviceRoutes {
    /// Carve the table out of device memory at `base`; returns it and the
    /// bytes used (256-byte aligned sections).
    pub fn at(base: *mut u8, rows: usize, k: usize) -> (Self, usize) {
        let pairs = rows * k;
        let a = |n: usize| (n * 4).next_multiple_of(256);
        let mut off = 0usize;
        let mut take = |n: usize| {
            let p = unsafe { base.add(off) };
            off += a(n);
            p
        };
        let t = Self {
            w: take(pairs).cast(),
            row: take(pairs).cast(),
            expert: take(pairs).cast(),
            row_start: take(rows + 1).cast(),
            shexp_scale: take(rows).cast(),
            rows,
            k,
        };
        (t, off)
    }
}

/// Softmax + top-k + renormalize routing for `rows` rows of router logits,
/// on the device (see `allpaca_moe_route_softmax_topk`). With `shexp_gate`,
/// also `sigmoid(x_r . gate)` per row into `routes.shexp_scale`. Optional
/// device `disabled` (n_exp bytes, nonzero = never picked) and `counts`
/// (n_exp u32, +1 per pick); null for none.
#[allow(clippy::too_many_arguments)]
pub fn moe_route_softmax_topk_device(
    logits: *const f32,
    n_exp: usize,
    x: *const f32,
    shexp_gate: *const f32,
    dim: usize,
    routes: &MoeDeviceRoutes,
    disabled: *const u8,
    counts: *mut u32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_moe_route_softmax_topk(
                logits,
                routes.rows as i32,
                n_exp as i32,
                routes.k as i32,
                x,
                shexp_gate,
                dim as i32,
                routes.w,
                routes.row,
                routes.expert,
                routes.row_start,
                routes.shexp_scale,
                disabled,
                counts,
                std::ptr::null_mut(),
            )
        };
        status(rc, "moe_route_softmax_topk")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (logits, n_exp, x, shexp_gate, dim, routes, disabled, counts);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP moe_route_softmax_topk unavailable".into(),
        )))
    }
}

/// Largest `k` of every `chunk` (<= 2048) of each row of `x[rows][n]`, largest
/// first, ties to the lower index: `out_v` / `out_i` are `[rows][ceil(n/chunk)][k]`.
#[allow(clippy::too_many_arguments)]
pub fn topk_chunks_device(
    x: *const f32,
    rows: usize,
    n: usize,
    chunk: usize,
    k: usize,
    out_v: *mut f32,
    out_i: *mut i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_topk_chunks(
                x,
                rows as i32,
                n as i32,
                chunk as i32,
                k as i32,
                out_v,
                out_i,
                std::ptr::null_mut(),
            )
        };
        status(rc, "topk_chunks")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, rows, n, chunk, k, out_v, out_i);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP topk_chunks unavailable".into(),
        )))
    }
}

/// NeoX RoPE (pairs `i`, `i + head_dim/2`) over `x[n_tok][n_heads][head_dim]`,
/// row `t` at position `start_pos + t`.
pub fn rope_neox_rows_device(
    x: *mut f32,
    n_tok: usize,
    n_heads: usize,
    head_dim: usize,
    start_pos: usize,
    theta: f32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_rope_neox_rows(
                x,
                n_tok as i32,
                n_heads as i32,
                head_dim as i32,
                start_pos as i32,
                theta,
                std::ptr::null_mut(),
            )
        };
        status(rc, "rope_neox_rows")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, n_tok, n_heads, head_dim, start_pos, theta);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP rope_neox_rows unavailable".into(),
        )))
    }
}

/// DFlash2 two-tap dynamic convolution along each `block` of rows (device
/// pointers; `out` must not alias `h`). See `dflash_conv_kernel`.
#[allow(clippy::too_many_arguments)]
pub fn dflash_conv_device(
    h: *const f32,
    dynamic: *const f32,
    base: *const f32,
    out: *mut f32,
    rows: usize,
    block: usize,
    dim: usize,
    kernel: usize,
    group: usize,
    side: usize,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_dflash_conv(
                h,
                dynamic,
                base,
                out,
                rows as i32,
                block as i32,
                dim as i32,
                kernel as i32,
                group as i32,
                side as i32,
                std::ptr::null_mut(),
            )
        };
        status(rc, "dflash_conv")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (h, dynamic, base, out, rows, block, dim, kernel, group, side);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP dflash_conv unavailable".into(),
        )))
    }
}

/// `x[r][:] *= scale[r]` with `scale` in device memory.
pub fn scale_rows_device(
    x: *mut f32,
    scale: *const f32,
    rows: usize,
    dim: usize,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_scale_rows(x, scale, rows as i32, dim as i32, std::ptr::null_mut())
        };
        status(rc, "scale_rows")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, scale, rows, dim);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP scale_rows unavailable".into(),
        )))
    }
}

/// Copy `n` floats with a kernel (queued, not waited on). Either side may be
/// pinned host memory from [`host_malloc`]: small copies skip the ~80 us
/// DMA setup a `hipMemcpy` pays on Windows.
pub fn copy_f32_kernel(dst: *mut f32, src: *const f32, n: usize) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_copy_f32_kernel(dst, src, n as i32, std::ptr::null_mut()) };
        status(rc, "copy_f32_kernel")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (dst, src, n);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP copy_f32_kernel unavailable".into(),
        )))
    }
}

/// One stacked `[n_experts][n][k]` K-quant or staged Q8_0 tensor for
/// [`moe_ffn_kquant_device`].
#[derive(Debug, Clone, Copy)]
pub struct MoeExperts {
    pub ptr: *const u8,
    /// ggml type id: 12 (Q4_K), 13 (Q5_K), 14 (Q6_K), 8 (Q8_0 in the
    /// 36-byte staged layout, see [`stage_q8_0_blocks`]) or 200 (MoESQ
    /// paired-4:8 FP4).
    pub ggml_type: u32,
    /// Bytes per expert.
    pub stride: u64,
}

impl MoeExperts {
    fn qt(&self) -> Option<i32> {
        match self.ggml_type {
            12 => Some(4),
            13 => Some(5),
            14 => Some(6),
            8 => Some(8),
            200 => Some(48),
            _ => None,
        }
    }

    /// The fused MoE kernels read this tensor.
    pub fn fused_ok(&self) -> bool {
        self.qt().is_some()
    }

    /// The fused MoE kernels run this `[gate, up, down]` set: gate and up
    /// share a launch, which exists for K-quant pairs, Q8_0 pairs and
    /// paired-4:8 pairs.
    pub fn fused_set_ok(set: &[Self; 3]) -> bool {
        // K-quants mix with each other; the other two only pair with themselves.
        let family = |e: &Self| match e.ggml_type {
            8 | 200 => e.ggml_type,
            _ => 0,
        };
        set.iter().all(Self::fused_ok) && family(&set[0]) == family(&set[1])
    }
}

/// Activation rows for [`moe_ffn_kquant_device`], in device memory.
#[derive(Debug, Clone, Copy)]
pub enum MoeInput {
    F32(*const f32),
    /// Already int8-quantized per 32 values (`quants`, then one scale each).
    Q8 { q: *const i8, d: *const f32 },
}

/// `(row, expert, weight)` pairs for [`moe_ffn_kquant_device`], grouped by
/// expert (at most [`MOE_MAX_GROUP`] rows per group) so the kernels read each
/// expert's weights once for all the rows that picked it. Packed with an
/// optional prefix (the worker's `x`) for a single H2D. Offsets are in floats
/// from the start of `host`, each 256-byte aligned; integers are stored as
/// their bit patterns.
#[derive(Debug, Clone, Default)]
pub struct MoePairTable {
    pub host: Vec<f32>,
    pub pairs: usize,
    pub groups: usize,
    pub max_group: usize,
    /// Per pair (group order): routing weight and token row.
    pub w: usize,
    pub row: usize,
    /// Per group: expert id, then `groups + 1` starts into the pairs.
    pub expert: usize,
    pub group_start: usize,
    /// `rows + 1` starts into `row_pairs`, which lists each row's pairs in
    /// routing order (the weighted sum's order).
    pub row_start: usize,
    pub row_pairs: usize,
    /// First float past the table: callers may place output or scratch here.
    pub end: usize,
}

/// Rows per expert group in [`MoePairTable`]; the kernels' LDS holds this many.
pub const MOE_MAX_GROUP: usize = 8;

impl MoePairTable {
    pub fn new(rows: usize, pairs: &[(u32, u32, f32)], prefix: &[f32]) -> Self {
        // Group order: by expert, then row. Stable, so a row's pairs keep
        // their routing order within the row index lists below.
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        order.sort_by_key(|&i| (pairs[i].1, pairs[i].0));
        let mut group_expert: Vec<u32> = Vec::new();
        let mut group_start: Vec<i32> = Vec::new();
        for (pos, &i) in order.iter().enumerate() {
            let e = pairs[i].1;
            let open = group_expert.last() == Some(&e)
                && pos - (*group_start.last().expect("group") as usize) < MOE_MAX_GROUP;
            if !open {
                group_expert.push(e);
                group_start.push(pos as i32);
            }
        }
        group_start.push(order.len() as i32);
        let max_group = group_start.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(1);
        // Position of each input pair in group order.
        let mut pos_of = vec![0usize; pairs.len()];
        for (pos, &i) in order.iter().enumerate() {
            pos_of[i] = pos;
        }
        let mut row_start = vec![0i32; rows + 1];
        for &(row, _, _) in pairs {
            row_start[row as usize + 1] += 1;
        }
        for r in 0..rows {
            row_start[r + 1] += row_start[r];
        }
        let mut fill = row_start.clone();
        let mut row_pairs = vec![0i32; pairs.len()];
        for (i, &(row, _, _)) in pairs.iter().enumerate() {
            let slot = &mut fill[row as usize];
            row_pairs[*slot as usize] = pos_of[i] as i32;
            *slot += 1;
        }
        let pad = |v: &mut Vec<f32>| v.resize(v.len().next_multiple_of(64), 0.0);
        let bits = |v: i32| f32::from_bits(v as u32);
        let mut host = prefix.to_vec();
        pad(&mut host);
        let w = host.len();
        host.extend(order.iter().map(|&i| pairs[i].2));
        pad(&mut host);
        let row = host.len();
        host.extend(order.iter().map(|&i| f32::from_bits(pairs[i].0)));
        pad(&mut host);
        let expert = host.len();
        host.extend(group_expert.iter().map(|&e| f32::from_bits(e)));
        pad(&mut host);
        let gs = host.len();
        host.extend(group_start.iter().map(|&v| bits(v)));
        pad(&mut host);
        let rs = host.len();
        host.extend(row_start.iter().map(|&v| bits(v)));
        pad(&mut host);
        let rp = host.len();
        host.extend(row_pairs.iter().map(|&v| bits(v)));
        pad(&mut host);
        Self {
            end: host.len(),
            host,
            pairs: pairs.len(),
            groups: group_expert.len(),
            max_group,
            w,
            row,
            expert,
            group_start: gs,
            row_start: rs,
            row_pairs: rp,
        }
    }

    /// [`moe_ffn_kquant_device`] with this table uploaded at `dev`.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        dev: *mut f32,
        x: MoeInput,
        rows: usize,
        dim: usize,
        ff: usize,
        experts: [MoeExperts; 3],
        scratch: *mut u8,
        out: *mut f32,
    ) -> Result<(), AllpacaError> {
        let at = |i: usize| unsafe { dev.add(i) };
        moe_ffn_kquant_device(
            x,
            rows,
            dim,
            ff,
            self.pairs,
            at(self.row).cast(),
            at(self.expert).cast(),
            at(self.w),
            at(self.row_start).cast(),
            MoeGroups {
                groups: self.groups,
                group_start: at(self.group_start).cast(),
                row_pairs: at(self.row_pairs).cast(),
                max_group: self.max_group,
            },
            experts[0],
            experts[1],
            experts[2],
            scratch,
            out,
        )
    }
}

/// How [`moe_ffn_kquant_device`]'s pairs are grouped by expert.
#[derive(Debug, Clone, Copy)]
pub struct MoeGroups {
    pub groups: usize,
    /// `groups + 1` starts into the pairs; null = one pair per group.
    pub group_start: *const i32,
    /// Each row's pairs in routing order; null = pairs are already row-sorted.
    pub row_pairs: *const i32,
    pub max_group: usize,
}

impl MoeGroups {
    /// One pair per group, pairs row-sorted (as the GPU router writes them).
    pub fn per_pair(pairs: usize) -> Self {
        Self {
            groups: pairs,
            group_start: std::ptr::null(),
            row_pairs: std::ptr::null(),
            max_group: 1,
        }
    }
}

/// Row-exact GEMMs on this thread while the guard lives: every row of an
/// M-row GEMM equals that row computed alone (wide M runs as row chunks on
/// the small-M GEMV kernels instead of the tiled / BLAS GEMMs). For batched
/// forwards over several sequences, whose rows must match their solo runs.
pub struct ExactRows(());

impl ExactRows {
    pub fn enable() -> Self {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        unsafe {
            ffi::allpaca_hip_set_exact_rows(1);
        }
        ExactRows(())
    }
}

impl Drop for ExactRows {
    fn drop(&mut self) {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        unsafe {
            ffi::allpaca_hip_set_exact_rows(0);
        }
    }
}

/// Device scratch [`moe_ffn_kquant_device`] needs (beyond its inputs).
/// `silu(g) * u` with the MoE kernels' deterministic exp, on the device
/// (test hook: it must equal `allpaca_cpu::det_silu_mul` bit for bit).
pub fn det_silu_mul(g: &[f32], u: &[f32]) -> Result<Vec<f32>, AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let n = g.len().min(u.len());
        let (dg, du, dout) = (device_malloc(n * 4, 0)?, device_malloc(n * 4, 0)?, device_malloc(n * 4, 0)?);
        let result = (|| {
            memcpy(dg, g[..n].as_ptr().cast(), n * 4, MEMCPY_H2D)?;
            memcpy(du, u[..n].as_ptr().cast(), n * 4, MEMCPY_H2D)?;
            status(
                unsafe {
                    ffi::allpaca_det_silu_mul(dg.cast(), du.cast(), dout.cast(), n as i32, std::ptr::null_mut())
                },
                "det_silu_mul",
            )?;
            let mut out = vec![0.0f32; n];
            stream_synchronize()?;
            memcpy(out.as_mut_ptr().cast(), dout, n * 4, MEMCPY_D2H)?;
            Ok(out)
        })();
        for p in [dg, du, dout] {
            device_free(p, n * 4);
        }
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (g, u);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled("HIP unavailable".into())))
    }
}

pub fn moe_ffn_scratch_bytes(rows: usize, dim: usize, ff: usize, pairs: usize) -> usize {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        unsafe {
            ffi::allpaca_moe_ffn_scratch_bytes(rows as i32, dim as i32, ff as i32, pairs as i32)
                as usize
        }
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (rows, dim, ff, pairs);
        0
    }
}

/// Expert-parallel MoE FFN in seven launches: for each row `r`, the sum over
/// its `(expert, weight)` pairs of `w * down_e(silu(gate_e x_r) * up_e x_r)`.
/// Pairs are sorted by row and `row_start` has `rows + 1` entries; every
/// pointer is device memory. Needs `dim` and `ff` in whole 256-value blocks.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_kquant_device(
    x: MoeInput,
    rows: usize,
    dim: usize,
    ff: usize,
    pairs: usize,
    pair_row: *const i32,
    group_expert: *const i32,
    pair_w: *const f32,
    row_start: *const i32,
    groups: MoeGroups,
    gate: MoeExperts,
    up: MoeExperts,
    down: MoeExperts,
    scratch: *mut u8,
    out: *mut f32,
) -> Result<(), AllpacaError> {
    let (Some(gq), Some(uq), Some(dq)) = (gate.qt(), up.qt(), down.qt()) else {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "fused MoE needs Q4_K/Q5_K/Q6_K, Q8_0 or paired-4:8 FP4 experts".into(),
        )));
    };
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let (xf, xq, xd) = match x {
            MoeInput::F32(p) => (p, std::ptr::null(), std::ptr::null()),
            MoeInput::Q8 { q, d } => (std::ptr::null(), q, d),
        };
        let rc = unsafe {
            ffi::allpaca_moe_ffn_kquant(
                xf,
                xq,
                xd,
                rows as i32,
                dim as i32,
                ff as i32,
                pairs as i32,
                pair_row,
                group_expert,
                pair_w,
                row_start,
                groups.groups as i32,
                groups.group_start,
                groups.row_pairs,
                groups.max_group as i32,
                gate.ptr,
                gq,
                gate.stride,
                up.ptr,
                uq,
                up.stride,
                down.ptr,
                dq,
                down.stride,
                scratch.cast(),
                out,
                std::ptr::null_mut(),
            )
        };
        status(rc, "moe_ffn_kquant")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, rows, dim, ff, pairs, pair_row, group_expert, pair_w, row_start, groups);
        let _ = (gq, uq, dq, scratch, out);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP moe_ffn_kquant unavailable".into(),
        )))
    }
}

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
        let rc =
            unsafe { ffi::allpaca_gemm_nt_q5_k_cols(a, b, c, m, n, k, cols, std::ptr::null_mut()) };
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
    // RDNA Infinity Cache is much larger than L2. GCN/CDNA (gfx9) has L2 only.
    let bytes = match device_gcn_arch(0) {
        Some((arch, l2)) if arch_has_infinity_cache(&arch) => {
            infinity_cache_from_l2(l2).max(16 << 20)
        }
        Some((_, l2)) if l2 > 0 => l2,
        Some((_, _)) => 8 << 20,
        None if device_count() > 0 => 64usize << 20,
        None => 0,
    };
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

/// Host-pointer BF16 NT GEMM: copies packed bfloat16 `b`, runs the kernel, copies back.
pub fn gemm_nt_bf16(
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
                "BF16 GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(b_n * 2, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr().cast(), b_n * 2, MEMCPY_H2D)?;
            gemm_nt_bf16_device(da.cast(), db.cast(), dc.cast(), m, n, k)?;
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
            "HIP gemm_nt_bf16 unavailable".into(),
        )))
    }
}

/// Untiled host BF16 GEMV.
pub fn gemm_nt_bf16_untiled(
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
                "BF16 GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(b_n * 2, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr().cast(), b_n * 2, MEMCPY_H2D)?;
            gemm_nt_bf16_untiled_device(da.cast(), db.cast(), dc.cast(), m, n, k)?;
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
            "HIP gemm_nt_bf16_untiled unavailable".into(),
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

/// Host-pointer TQ2_0 NT GEMM: copies native GGUF blocks, runs the kernel, copies back.
pub fn gemm_nt_tq2_0(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_tq_host(
        a,
        b,
        c,
        m,
        n,
        k,
        TQ2_0_BLOCK_ELEMS,
        TQ2_0_BLOCK_BYTES,
        gemm_nt_tq2_0_device,
    )
}

/// Host-pointer TQ1_0 NT GEMM: copies native GGUF blocks, runs the kernel, copies back.
pub fn gemm_nt_tq1_0(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_tq_host(
        a,
        b,
        c,
        m,
        n,
        k,
        TQ1_0_BLOCK_ELEMS,
        TQ1_0_BLOCK_BYTES,
        gemm_nt_tq1_0_device,
    )
}

/// Host-pointer ROCmFP4 NT GEMM: copies native blocks, runs the fused kernel, copies back.
pub fn gemm_nt_rocmfp4(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_rocmfp4_host(a, b, c, m, n, k, gemm_nt_rocmfp4_device)
}

/// Untiled ROCmFP4 host GEMV for §12 before/after tracers.
pub fn gemm_nt_rocmfp4_untiled(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_rocmfp4_host(a, b, c, m, n, k, gemm_nt_rocmfp4_untiled_device)
}

fn gemm_nt_rocmfp4_host(
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
        let nblocks = (k as usize).div_ceil(ROCMFP4_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * ROCMFP4_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "ROCmFP4 GEMM buffer size mismatch".into(),
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
            "HIP gemm_nt_rocmfp4 unavailable".into(),
        )))
    }
}

/// Host-pointer MXFP4 NT GEMM: copies native ggml blocks, runs the kernel, copies back.
pub fn gemm_nt_mxfp4(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_mxfp4_host(a, b, c, m, n, k, gemm_nt_mxfp4_device)
}

/// Host-pointer fused MXFP4 gate+up+SiLU.
#[allow(clippy::too_many_arguments)]
pub fn mxfp4_swiglu_gate_up(
    a: &[f32],
    bg: &[u8],
    bu: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
    limit: f32,
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
        let nblocks = (k as usize).div_ceil(MXFP4_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * MXFP4_BLOCK_BYTES;
        if a.len() < a_n || bg.len() < packed_bytes || bu.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU gate-up buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let dbg = device_malloc(packed_bytes, 0)?;
        let dbu = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(dbg, bg.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            memcpy(dbu, bu.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            mxfp4_swiglu_gate_up_device(da.cast(), dbg, dbu, dc.cast(), m, n, k, limit)?;
            memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
        })();
        device_free(da, a_n * 4);
        device_free(dbg, packed_bytes);
        device_free(dbu, packed_bytes);
        device_free(dc, c_n * 4);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, bg, bu, c, m, n, k, limit);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP mxfp4_swiglu_gate_up unavailable".into(),
        )))
    }
}

/// Untiled MXFP4 host GEMV.
pub fn gemm_nt_mxfp4_untiled(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    gemm_nt_mxfp4_host(a, b, c, m, n, k, gemm_nt_mxfp4_untiled_device)
}

fn gemm_nt_mxfp4_host(
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
        let nblocks = (k as usize).div_ceil(MXFP4_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * MXFP4_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 GEMM buffer size mismatch".into(),
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
            "HIP gemm_nt_mxfp4 unavailable".into(),
        )))
    }
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

/// Device-pointer NT GEMM with native ggml Q2_0 `b` (`C = A * dequant(B)^T`);
/// `k` must be whole 64-value blocks.
pub fn gemm_nt_q2_0_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_gemm_nt_q2_0(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q2_0")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_q2_0 unavailable".into(),
        )))
    }
}

/// ggml Q2_0: 64 weights in 18 bytes.
const Q2_0_BLOCK_ELEMS: usize = 64;
const Q2_0_BLOCK_BYTES: usize = 18;

/// Bytes of one expert slot of ggml Q2_0 blocks: gate `[ff, dim]`, up
/// `[ff, dim]` and down `[dim, ff]`, each row-major.
pub fn q2_0_expert_slot_bytes(dim: usize, ff: usize) -> usize {
    3 * ff * dim / Q2_0_BLOCK_ELEMS * Q2_0_BLOCK_BYTES
}

/// Routed experts from a pool of Q2_0 slots, all on the device:
/// `out[dim] = sum_p weights[p] * down_p * (silu(gate_p * x) * (up_p * x))`.
/// `h` is scratch of `slots.len() * ff` floats; at most 32 picks.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_moe_ffn_device(
    x: *const f32,
    pool: *const u8,
    slots: &[i32],
    weights: &[f32],
    h: *mut f32,
    out: *mut f32,
    dim: i32,
    ff: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if slots.len() != weights.len() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q2_0 MoE picks and weights differ in length".into(),
            )));
        }
        let rc = unsafe {
            ffi::allpaca_q2_0_moe_ffn(
                x,
                pool,
                slots.as_ptr(),
                weights.as_ptr(),
                slots.len() as i32,
                h,
                out,
                dim,
                ff,
                std::ptr::null_mut(),
            )
        };
        status(rc, "q2_0_moe_ffn")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, pool, slots, weights, h, out, dim, ff);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP q2_0_moe_ffn unavailable".into(),
        )))
    }
}

/// `x[i] = silu(x[i] * s)` on the device.
pub fn silu_scale_device(x: *mut f32, n: i32, s: f32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        let rc = unsafe { ffi::allpaca_silu_scale(x, n, s, std::ptr::null_mut()) };
        status(rc, "silu_scale")
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, n, s);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP silu_scale unavailable".into(),
        )))
    }
}

/// The device ops of a `qwen4exp` hyper-connection over `rows` tokens of
/// `hc` streams of `dim` floats, `x[(r * hc + s) * dim + d]`. All pointers
/// are device memory of the bound device.
pub mod q4x {
    use super::*;

    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    fn unavailable() -> Result<(), AllpacaError> {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP qwen4exp kernels unavailable".into(),
        )))
    }

    /// `xn[r, s] = x[r, s] / rms(x[r, s]) * gains[s]`.
    pub fn stream_norm(
        x: *const f32,
        gains: *const f32,
        xn: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_q4x_stream_norm(x, gains, xn, rows, hc, dim, eps, std::ptr::null_mut()) };
            status(rc, "q4x_stream_norm")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, gains, xn, rows, hc, dim, eps);
            unavailable()
        }
    }

    /// `mixed[r, d] = mean over streams of xn[r, s, d] * sigmoid(gate[r, s, d])`.
    pub fn gated_mean(
        xn: *const f32,
        gate: *const f32,
        mixed: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_q4x_gated_mean(xn, gate, mixed, rows, hc, dim, std::ptr::null_mut()) };
            status(rc, "q4x_gated_mean")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (xn, gate, mixed, rows, hc, dim);
            unavailable()
        }
    }

    /// `streams[r, s, d] += out[r, d] * 2 * sigmoid(logits[r, s] / hc)`.
    pub fn combine(
        streams: *mut f32,
        out: *const f32,
        logits: *const f32,
        rows: i32,
        hc: i32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_q4x_combine(streams, out, logits, rows, hc, dim, std::ptr::null_mut()) };
            status(rc, "q4x_combine")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (streams, out, logits, rows, hc, dim);
            unavailable()
        }
    }

    /// Bytes of the `picks` scratch of [`routed_experts`]: 32 slots, then
    /// their 32 weights.
    pub const PICKS_BYTES: usize = 32 * 8;

    /// One token's routed experts with the routing on the device: the
    /// `n_pick` best of `n_experts` router `scores` by softmax weight (ties
    /// to the lower expert), weights renormalised over the picks and times
    /// `scale`, then the experts from `pool` (Q2_0 slots, as
    /// [`q2_0_moe_ffn_device`]) into `out`. `picks` is device scratch of
    /// [`PICKS_BYTES`] and keeps the picks; `h` is scratch of `n_pick * ff`
    /// floats. At most 1024 experts and 32 picks.
    #[allow(clippy::too_many_arguments)]
    pub fn routed_experts(
        x: *const f32,
        scores: *const f32,
        pool: *const u8,
        n_experts: i32,
        n_pick: i32,
        scale: f32,
        picks: *mut u8,
        h: *mut f32,
        out: *mut f32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_q4x_routed_ffn(
                    x,
                    scores,
                    pool,
                    n_experts,
                    n_pick,
                    scale,
                    picks,
                    h,
                    out,
                    dim,
                    ff,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "q4x_routed_ffn")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, scores, pool, n_experts, n_pick, scale, picks, h, out, dim, ff);
            unavailable()
        }
    }

    /// `out[r, d] = routed[r, d] + sigmoid(gate[r]) * shared[r, d]`;
    /// `routed` or `shared` (with `gate`) may be null.
    pub fn ffn_sum(
        routed: *const f32,
        shared: *const f32,
        gate: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_q4x_ffn_sum(routed, shared, gate, out, rows, dim, std::ptr::null_mut()) };
            status(rc, "q4x_ffn_sum")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (routed, shared, gate, out, rows, dim);
            unavailable()
        }
    }
}

/// Host-pointer Q2_0 NT GEMM: copies native ggml blocks, runs the kernel, copies back.
pub fn gemm_nt_q2_0(
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
        let packed_bytes =
            (n as usize) * (k as usize).div_ceil(Q2_0_BLOCK_ELEMS) * Q2_0_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q2_0 GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            gemm_nt_q2_0_device(da.cast(), db, dc.cast(), m, n, k)?;
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
            "HIP gemm_nt_q2_0 unavailable".into(),
        )))
    }
}

/// Host-pointer Q3_K NT GEMM: copies native GGUF blocks, runs the fused kernel, copies back.
pub fn gemm_nt_q3_k(
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
        let nblocks = (k as usize).div_ceil(Q3_K_BLOCK_ELEMS);
        let packed_bytes = (n as usize) * nblocks * Q3_K_BLOCK_BYTES;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q3_K GEMM buffer size mismatch".into(),
            )));
        }
        let da = device_malloc(a_n * 4, 0)?;
        let db = device_malloc(packed_bytes, 0)?;
        let dc = device_malloc(c_n * 4, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D)?;
            memcpy(db, b.as_ptr(), packed_bytes, MEMCPY_H2D)?;
            gemm_nt_q3_k_device(da.cast(), db, dc.cast(), m, n, k)?;
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
            "HIP gemm_nt_q3_k unavailable".into(),
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

fn gemm_nt_tq_host(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
    block_elems: usize,
    block_bytes: usize,
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
        let nblocks = (k as usize).div_ceil(block_elems);
        let packed_bytes = (n as usize) * nblocks * block_bytes;
        if a.len() < a_n || b.len() < packed_bytes || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "TQ GEMM buffer size mismatch".into(),
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
        let _ = (a, b, c, m, n, k, block_elems, block_bytes, kernel);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP gemm_nt_tq unavailable".into(),
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

/// Host-pointer batched mean/var LayerNorm: `out[rows, n] = LN(x[rows, n], weight[n], bias[n])`.
pub fn layernorm_rows(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    out: &mut [f32],
    rows: i32,
    n: i32,
    eps: f32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if rows <= 0 || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid layernorm_rows shape".into(),
            )));
        }
        let rows_n = (rows as usize).saturating_mul(n as usize);
        if x.len() < rows_n
            || out.len() < rows_n
            || weight.len() < n as usize
            || bias.len() < n as usize
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "layernorm_rows buffer size mismatch".into(),
            )));
        }
        let x_bytes = rows_n * 4;
        let w_bytes = (n as usize) * 4;
        let dx = device_malloc(x_bytes, 0)?;
        let dw = device_malloc(w_bytes, 0)?;
        let db = match device_malloc(w_bytes, 0) {
            Ok(p) => p,
            Err(e) => {
                device_free(dx, x_bytes);
                device_free(dw, w_bytes);
                return Err(e);
            }
        };
        let dout = match device_malloc(x_bytes, 0) {
            Ok(p) => p,
            Err(e) => {
                device_free(dx, x_bytes);
                device_free(dw, w_bytes);
                device_free(db, w_bytes);
                return Err(e);
            }
        };
        set_device(0)?;
        let result = (|| {
            memcpy(dx, x.as_ptr().cast(), x_bytes, MEMCPY_H2D)?;
            memcpy(dw, weight.as_ptr().cast(), w_bytes, MEMCPY_H2D)?;
            memcpy(db, bias.as_ptr().cast(), w_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_layernorm_rows(
                    dx.cast(),
                    dw.cast(),
                    db.cast(),
                    dout.cast(),
                    rows,
                    n,
                    eps,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "layernorm_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, x_bytes, MEMCPY_D2H)
        })();
        device_free(dx, x_bytes);
        device_free(dw, w_bytes);
        device_free(db, w_bytes);
        device_free(dout, x_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (x, weight, bias, out, rows, n, eps);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP layernorm_rows unavailable".into(),
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
        let _ = (
            q, k, n_tok, n_q_heads, n_k_heads, head_dim, start_pos, theta,
        );
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
        let packed_need = (max_tok + 1)
            .saturating_mul(nblocks)
            .saturating_mul(Q8_PACKED_BLOCK);
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

/// Host-pointer batched ROCmFP4 embed gather. `table` is native 18-byte blocks.
pub fn embed_gather_rocmfp4_rows(
    table: &[u8],
    tokens: &[u32],
    out: &mut [f32],
    dim: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    {
        if dim <= 0 || tokens.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid embed_gather_rocmfp4_rows shape".into(),
            )));
        }
        let n_tok = tokens.len();
        let dim_u = dim as usize;
        let max_tok = *tokens.iter().max().unwrap_or(&0) as usize;
        let nblocks = dim_u.div_ceil(32);
        let packed_need = (max_tok + 1)
            .saturating_mul(nblocks)
            .saturating_mul(ROCMFP4_BLOCK_BYTES);
        if table.len() < packed_need || out.len() < n_tok * dim_u {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "embed_gather_rocmfp4_rows buffer size mismatch".into(),
            )));
        }
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let id_bytes = ids.len() * 4;
        let o_bytes = n_tok * dim_u * 4;
        let dt = device_malloc(packed_need, 0)?;
        let di = device_malloc(id_bytes, 0)?;
        let dout = device_malloc(o_bytes, 0)?;
        set_device(0)?;
        let result = (|| {
            memcpy(dt, table.as_ptr(), packed_need, MEMCPY_H2D)?;
            memcpy(di, ids.as_ptr().cast(), id_bytes, MEMCPY_H2D)?;
            let rc = unsafe {
                ffi::allpaca_embed_gather_rocmfp4_rows(
                    dt,
                    di.cast(),
                    dout.cast(),
                    n_tok as i32,
                    dim,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "embed_gather_rocmfp4_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)
        })();
        device_free(dt, packed_need);
        device_free(di, id_bytes);
        device_free(dout, o_bytes);
        result
    }
    #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
    {
        let _ = (table, tokens, out, dim);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "HIP embed_gather_rocmfp4_rows unavailable".into(),
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
            ffi::allpaca_hip_memcpy(dst.cast(), src.cast(), bytes, kind, std::ptr::null_mut())
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

/// Shareable D3D12 buffer imported into HIP and writable by DirectStorage.
///
/// Construction is available only when the Windows HIP runtime and the stable
/// Microsoft DirectStorage 1.3 SDK were found at build time.
pub struct DirectStorageBuffer {
    #[allow(dead_code)] // Used only by the Windows DirectStorage build.
    handle: *mut std::ffi::c_void,
    ptr: *mut u8,
    device: i32,
    bytes: usize,
}

unsafe impl Send for DirectStorageBuffer {}
unsafe impl Sync for DirectStorageBuffer {}

impl DirectStorageBuffer {
    pub fn alloc(device: i32, bytes: usize) -> Result<Self, AllpacaError> {
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "DirectStorage buffer must not be empty".into(),
            )));
        }
        #[cfg(all(windows, feature = "hip", allpaca_hip_runtime, allpaca_direct_storage))]
        {
            let mut handle = std::ptr::null_mut();
            let mut ptr = std::ptr::null_mut();
            let rc = unsafe { ffi::allpaca_dstorage_alloc(&mut handle, &mut ptr, bytes, device) };
            if rc != ALLPACA_OK || handle.is_null() || ptr.is_null() {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    format!(
                        "D3D12/HIP external-memory allocation failed on device {device} rc={rc}"
                    ),
                )));
            }
            ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
            Ok(Self {
                handle,
                ptr: ptr.cast(),
                device,
                bytes,
            })
        }
        #[cfg(not(all(windows, feature = "hip", allpaca_hip_runtime, allpaca_direct_storage)))]
        {
            let _ = device;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "DirectStorage D3D12/HIP bridge unavailable; use host staging".into(),
            )))
        }
    }

    pub fn read_file(
        &self,
        path: &std::path::Path,
        file_offset: u64,
        buffer_offset: usize,
        bytes: usize,
    ) -> Result<(), AllpacaError> {
        if bytes == 0 {
            return Ok(());
        }
        if buffer_offset.saturating_add(bytes) > self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "DirectStorage buffer overflow: off={buffer_offset} n={bytes} cap={}",
                    self.bytes
                ),
            )));
        }
        #[cfg(all(windows, feature = "hip", allpaca_hip_runtime, allpaca_direct_storage))]
        {
            use std::os::windows::ffi::OsStrExt;
            let wide: Vec<u16> = path
                .as_os_str()
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let rc = unsafe {
                ffi::allpaca_dstorage_read(
                    self.handle,
                    wide.as_ptr(),
                    file_offset,
                    buffer_offset as u64,
                    bytes,
                )
            };
            status(rc, "DirectStorage file-to-D3D12 DMA")
        }
        #[cfg(not(all(windows, feature = "hip", allpaca_hip_runtime, allpaca_direct_storage)))]
        {
            let _ = (path, file_offset);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "DirectStorage D3D12/HIP bridge unavailable; use host staging".into(),
            )))
        }
    }

    pub fn copy_d2h(&self, offset: usize, dest: &mut [u8]) -> Result<(), AllpacaError> {
        if offset.saturating_add(dest.len()) > self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "DirectStorage D2H readback overflow".into(),
            )));
        }
        set_device(self.device)?;
        memcpy(
            dest.as_mut_ptr(),
            unsafe { self.ptr.add(offset) },
            dest.len(),
            MEMCPY_D2H,
        )
    }

    pub fn byte_ptr_at(&self, offset: usize) -> *const u8 {
        unsafe { self.ptr.add(offset) }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for DirectStorageBuffer {
    fn drop(&mut self) {
        #[cfg(all(windows, feature = "hip", allpaca_hip_runtime, allpaca_direct_storage))]
        unsafe {
            let _ = ffi::allpaca_dstorage_free(self.handle);
            ALLOCATED_BYTES.fetch_sub(self.bytes as u64, Ordering::Relaxed);
        }
    }
}

/// Packed weight tensors uploaded once via H2D. GEMM must use [`Self::ptr`] — no per-op copy.
pub struct DeviceWeights {
    device: i32,
    ptr: *mut u8,
    bytes: usize,
    direct_storage: Option<DirectStorageBuffer>,
}

unsafe impl Send for DeviceWeights {}
unsafe impl Sync for DeviceWeights {}

impl DeviceWeights {
    /// Ordinary HIP device heap for weight blobs (H2D memcpy / GEMM).
    ///
    /// DirectStorage D3D12-imported memory is opt-in via [`Self::alloc_direct_storage`];
    /// using it for every upload breaks large H2D packs (hipMemcpy rc=3 on imported ptrs).
    pub fn alloc(device: i32, bytes: usize) -> Result<Self, AllpacaError> {
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty weight upload".into(),
            )));
        }
        let ptr = device_malloc(bytes, device)?;
        Ok(Self {
            device,
            ptr,
            bytes,
            direct_storage: None,
        })
    }

    /// D3D12 resource imported into HIP for DirectStorage NVMe→GPU DMA fills.
    pub fn alloc_direct_storage(device: i32, bytes: usize) -> Result<Self, AllpacaError> {
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty weight upload".into(),
            )));
        }
        let direct_storage = DirectStorageBuffer::alloc(device, bytes)?;
        let ptr = direct_storage.byte_ptr_at(0) as *mut u8;
        Ok(Self {
            device,
            ptr,
            bytes,
            direct_storage: Some(direct_storage),
        })
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

    /// Blocking D2H of `dest.len()` bytes from `offset`.
    pub fn copy_d2h(&self, offset: usize, dest: &mut [u8]) -> Result<(), AllpacaError> {
        if dest.is_empty() {
            return Ok(());
        }
        if offset.saturating_add(dest.len()) > self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(format!(
                "weight D2H overflow: off={offset} n={} cap={}",
                dest.len(),
                self.bytes
            ))));
        }
        set_device(self.device)?;
        memcpy(
            dest.as_mut_ptr(),
            unsafe { self.ptr.add(offset) },
            dest.len(),
            MEMCPY_D2H,
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

    /// One batched H2D for several disjoint `(offset, bytes)` regions of this
    /// blob. Replaces a per-slot `copy_h2d_async` loop when expert ring slots
    /// are not adjacent. `stream` null runs on the default compute stream.
    ///
    /// # Safety
    /// `chunks` must stay alive and unmodified until `stream` drains.
    pub unsafe fn copy_h2d_batch_on(
        &self,
        chunks: &[(usize, &[u8])],
        stream: *mut std::ffi::c_void,
    ) -> Result<(), AllpacaError> {
        if chunks.is_empty() {
            return Ok(());
        }
        for (offset, src) in chunks {
            if offset.saturating_add(src.len()) > self.bytes {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    format!(
                        "weight batch H2D overflow: off={offset} n={} cap={}",
                        src.len(),
                        self.bytes
                    ),
                )));
            }
        }
        set_device(self.device)?;
        let copies: Vec<(*mut u8, *const u8, usize)> = chunks
            .iter()
            .filter(|(_, src)| !src.is_empty())
            .map(|(offset, src)| (unsafe { self.ptr.add(*offset) }, src.as_ptr(), src.len()))
            .collect();
        unsafe { memcpy_batch_on(&copies, MEMCPY_H2D, stream) }
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

    /// Queue a read-only touch so WDDM pages this blob in. Does not host-sync.
    pub fn touch(&self) -> Result<(), AllpacaError> {
        touch_resident(self.ptr, self.bytes, self.device)
    }

    /// Fill a region directly from a file when this allocation is backed by an
    /// imported D3D12 resource. Returns `false` for ordinary `hipMalloc`.
    pub fn read_file_direct(
        &self,
        path: &std::path::Path,
        file_offset: u64,
        byte_offset: usize,
        bytes: usize,
    ) -> Result<bool, AllpacaError> {
        let Some(buffer) = &self.direct_storage else {
            return Ok(false);
        };
        buffer.read_file(path, file_offset, byte_offset, bytes)?;
        Ok(true)
    }

    /// Packed native-quant / F32 byte slices uploaded once via H2D.
    pub fn upload_bytes(device: i32, parts: &[&[u8]]) -> Result<Self, AllpacaError> {
        let bytes: usize = parts.iter().map(|t| t.len()).sum();
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty weight upload".into(),
            )));
        }
        let this = Self::alloc(device, bytes)?;
        let mut off = 0usize;
        for t in parts {
            this.copy_h2d(off, t)?;
            off += t.len();
        }
        Ok(this)
    }
}

impl Drop for DeviceWeights {
    fn drop(&mut self) {
        if self.direct_storage.is_none() {
            device_free(self.ptr, self.bytes);
        }
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
        // SAFETY: caller sizes `x` as at least `(i + 1) * dim` floats.
        unsafe { self.x.add(i * dim) }
    }

    pub fn normed_row(&self, i: usize, dim: usize) -> *mut f32 {
        // SAFETY: caller sizes `normed` as at least `(i + 1) * dim` floats.
        unsafe { self.normed.add(i * dim) }
    }

    pub fn proj_row(&self, i: usize, dim: usize) -> *mut f32 {
        // SAFETY: caller sizes `proj` as at least `(i + 1) * dim` floats.
        unsafe { self.proj.add(i * dim) }
    }

    pub fn q_row(&self, i: usize, dim: usize) -> *mut f32 {
        // SAFETY: caller sizes `q` as at least `(i + 1) * dim` floats.
        unsafe { self.q.add(i * dim) }
    }

    pub fn k_row(&self, i: usize, kv_dim: usize) -> *mut f32 {
        // SAFETY: caller sizes `k` as at least `(i + 1) * kv_dim` floats.
        unsafe { self.k.add(i * kv_dim) }
    }

    pub fn v_row(&self, i: usize, kv_dim: usize) -> *mut f32 {
        // SAFETY: caller sizes `v` as at least `(i + 1) * kv_dim` floats.
        unsafe { self.v.add(i * kv_dim) }
    }

    pub fn attn_row(&self, i: usize, dim: usize) -> *mut f32 {
        // SAFETY: caller sizes `attn` as at least `(i + 1) * dim` floats.
        unsafe { self.attn.add(i * dim) }
    }

    pub fn gate_row(&self, i: usize, stride: usize) -> *mut f32 {
        // SAFETY: caller sizes `gate` as at least `(i + 1) * stride` floats.
        unsafe { self.gate.add(i * stride) }
    }

    pub fn up_row(&self, i: usize, stride: usize) -> *mut f32 {
        // SAFETY: caller sizes `up` as at least `(i + 1) * stride` floats.
        unsafe { self.up.add(i * stride) }
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
    /// Scratch width for `q`/`attn` (SSM in_proj / GDN qkv). ≥ `dim`.
    q_dim: usize,
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
        let scratch = self.q_dim.max(self.dim);
        DeviceDecode {
            x: next(self.dim * b),
            normed: next(self.dim * b),
            q: next(scratch * b),
            k: next(self.kv_dim * b),
            v: next(self.kv_dim * b),
            attn: next(scratch * b),
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
    q_dim: usize,
) -> usize {
    let b = batch.max(1);
    let rows = logits_rows.max(1);
    let scratch = q_dim.max(dim);
    b * (dim * 3 + scratch * 2 + kv_dim * 2 + ff_dim * 2) + vocab * rows + max_kv * kv_dim * 2
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

/// Per-request / per-layer GDN recurrent state kept on device between tokens.
struct GdnLayerResident {
    state: DeviceWeights,
    state_n: usize,
    conv: Option<DeviceWeights>,
    conv_n: usize,
    seeded: bool,
    dirty: bool,
    ssm: bool,
}

/// GDN step inputs recorded while a request's state is cloned for verify.
///
/// Each row is `[qkv | alpha | beta]` as the step saw it (pre-conv qkv, beta
/// before its in-place sigmoid). Rewinding a partial accept restores the clone
/// and re-runs only the recurrent kernels over the accepted rows, instead of a
/// second full forward through every weight.
struct GdnJournal {
    rows_buf: DeviceWeights,
    cap: usize,
    rows: usize,
    overflow: bool,
    qkv_dim: usize,
    n_v: usize,
    n_k: usize,
    head_v: usize,
    head_k: usize,
    conv_kernel: usize,
    eps: f32,
    k_interleave: bool,
    conv_w: *const f32,
    dt_bias: *const f32,
    a: *const f32,
}

/// Rows a journal holds before it gives up (verify spans are max_depth + 1).
const GDN_JOURNAL_ROWS: usize = 16;

/// Per-layer GDN extras (conv / dt / a / norm) uploaded once on the Target workspace.
struct GdnWeightResident {
    conv: DeviceWeights,
    dt: DeviceWeights,
    a: DeviceWeights,
    norm: DeviceWeights,
}

struct Mamba2WeightResident {
    conv: DeviceWeights,
    bias: Option<DeviceWeights>,
    dt: DeviceWeights,
    a: DeviceWeights,
    d: DeviceWeights,
    norm: DeviceWeights,
}

/// Per-GPU packed MoE activations: full `xs` plus scatter-add accumulator.
struct MoeStage {
    x: *mut u8,
    acc: *mut u8,
    meta: *mut u8,
    x_bytes: usize,
    acc_bytes: usize,
    meta_bytes: usize,
    n_tok: usize,
    dim: usize,
    active: bool,
}

unsafe impl Send for MoeStage {}
unsafe impl Sync for MoeStage {}

fn free_moe_stage(stage: MoeStage) {
    device_free(stage.x, stage.x_bytes);
    device_free(stage.acc, stage.acc_bytes);
    device_free(stage.meta, stage.meta_bytes);
}

pub struct HipWorkspace {
    pub device: i32,
    scratch: *mut u8,
    scratch_bytes: usize,
    /// Side streams for overlapping MXFP4 gate/up (and two-expert SwiGLU).
    swiglu_s: [*mut std::ffi::c_void; SWIGLU_STREAMS],
    /// 0 = H2D complete on the default stream; 1/2 = gate/up complete.
    swiglu_ev: [*mut std::ffi::c_void; SWIGLU_EVENTS],
    pinned: Mutex<PinnedLogits>,
    decode: Mutex<Option<DecodeAlloc>>,
    dequant: Mutex<Option<DequantAlloc>>,
    gdn: Mutex<std::collections::HashMap<(u64, usize), GdnLayerResident>>,
    gdn_snap: Mutex<std::collections::HashMap<(u64, usize), GdnLayerResident>>,
    /// Requests whose `gdn_snap` slots hold a live clone. Dropping a clone only
    /// leaves this set, so the next verify reuses the slots without hipMalloc.
    gdn_clone_live: Mutex<std::collections::HashSet<u64>>,
    gdn_journal: Mutex<std::collections::HashMap<(u64, usize), GdnJournal>>,
    gdn_w: Mutex<std::collections::HashMap<usize, GdnWeightResident>>,
    mamba2_w: Mutex<std::collections::HashMap<usize, Mamba2WeightResident>>,
    moe: Mutex<Option<MoeStage>>,
    activation_h2d: AtomicU64,
    activation_d2h: AtomicU64,
    profiling: AtomicU64,
    lm_head: AtomicU64,
    /// Device ranges `[start, end)` of dense weights staged as Q8_0 blocks,
    /// sorted. See [`Self::register_q8_dense`].
    q8_dense: std::sync::RwLock<Vec<(usize, usize)>>,
    /// Bits of the clamp the MXFP4 SwiGLU kernels apply (0.0 = none).
    mxfp4_swiglu_limit: std::sync::atomic::AtomicU32,
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
        let (swiglu_s, swiglu_ev) = make_swiglu_lanes(device);
        let pinned_bytes = vocab.max(1) * 4;
        let pinned_ptr = match host_malloc(pinned_bytes) {
            Ok(p) => p,
            Err(_) => std::ptr::null_mut(),
        };
        Ok(Self {
            device,
            scratch,
            scratch_bytes,
            swiglu_s,
            swiglu_ev,
            pinned: Mutex::new(PinnedLogits {
                ptr: pinned_ptr,
                bytes: if pinned_ptr.is_null() {
                    0
                } else {
                    pinned_bytes
                },
            }),
            decode: Mutex::new(None),
            dequant: Mutex::new(None),
            gdn: Mutex::new(std::collections::HashMap::new()),
            gdn_snap: Mutex::new(std::collections::HashMap::new()),
            gdn_clone_live: Mutex::new(std::collections::HashSet::new()),
            gdn_journal: Mutex::new(std::collections::HashMap::new()),
            gdn_w: Mutex::new(std::collections::HashMap::new()),
            mamba2_w: Mutex::new(std::collections::HashMap::new()),
            moe: Mutex::new(None),
            activation_h2d: AtomicU64::new(0),
            activation_d2h: AtomicU64::new(0),
            profiling: AtomicU64::new(0),
            lm_head: AtomicU64::new(0),
            q8_dense: std::sync::RwLock::new(Vec::new()),
            mxfp4_swiglu_limit: std::sync::atomic::AtomicU32::new(0),
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
        if let Ok(g) = self.gdn.lock() {
            for layer in g.values() {
                n += layer.state.bytes();
                if let Some(c) = layer.conv.as_ref() {
                    n += c.bytes();
                }
            }
        }
        if let Ok(g) = self.gdn_snap.lock() {
            for layer in g.values() {
                n += layer.state.bytes();
                if let Some(c) = layer.conv.as_ref() {
                    n += c.bytes();
                }
            }
        }
        if let Ok(g) = self.gdn_w.lock() {
            for w in g.values() {
                n += w.conv.bytes() + w.dt.bytes() + w.a.bytes() + w.norm.bytes();
            }
        }
        if let Ok(g) = self.mamba2_w.lock() {
            for w in g.values() {
                n += w.conv.bytes() + w.dt.bytes() + w.a.bytes() + w.d.bytes() + w.norm.bytes();
                if let Some(bias) = w.bias.as_ref() {
                    n += bias.bytes();
                }
            }
        }
        if let Ok(g) = self.moe.lock() {
            if let Some(s) = g.as_ref() {
                n += s.x_bytes as u64 + s.acc_bytes as u64 + s.meta_bytes as u64;
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

    /// Start a HIP decode sample. On a sampled forward (one in [`PROFILE_EVERY`])
    /// kernel ABI calls until [`end_decode_profile`] are event-timed; the rest
    /// only record wall time.
    pub fn begin_decode_profile(&self) {
        self.lm_head.store(0, Ordering::Relaxed);
        self.profile_span.store(0, Ordering::Relaxed);
        let mut g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
        g.used = 0;
        g.ops.clear();
        g.lm_head_ns = 0;
        g.full = g.forwards % PROFILE_EVERY == 1
            || std::env::var("ALLPACA_PROFILE_ALL").is_ok_and(|v| v == "1" || v == "2");
        g.forwards = g.forwards.wrapping_add(1);
        self.profiling.store(u64::from(g.full), Ordering::Relaxed);
        g.wall_start = Some(std::time::Instant::now());
    }

    /// Classify the next F32/F16 GEMM ABI calls as lm_head until the guard drops.
    pub fn lm_head_scope(&self) -> LmHeadScope<'_> {
        self.lm_head.store(1, Ordering::Relaxed);
        LmHeadScope {
            flag: &self.lm_head,
        }
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
                // The thread may be bound to another GPU by now (layer split):
                // wait on this workspace's last marker, or the last pairs
                // (lm_head) read back as not ready and count zero.
                let _ = self.bind();
                let _ = stream_synchronize();
                let last = g.used.min(g.pairs.len());
                if last > 0 {
                    let _ = event_synchronize(g.pairs[last - 1][1]);
                }
                let dump = std::env::var("ALLPACA_PROFILE_ALL").is_ok_and(|v| v == "2");
                let mut line = String::new();
                for i in 0..g.used.min(g.ops.len()).min(g.pairs.len()) {
                    let op = g.ops[i] as usize;
                    if op >= 4 {
                        continue;
                    }
                    let ns = event_elapsed_ns(g.pairs[i][0], g.pairs[i][1]);
                    buckets[op] += ns;
                    if dump {
                        line.push_str(&format!("{op}:{} ", ns / 1000));
                    }
                }
                if dump {
                    eprintln!("OPS dev={} {line}", self.device);
                }
            }
            buckets[DecodeOp::LmHead as usize] += g.lm_head_ns;
            let gpu_ns = buckets[0] + buckets[1] + buckets[2] + buckets[3];
            buckets[4] = wall_ns.saturating_sub(gpu_ns);
            if std::env::var("ALLPACA_PROFILE_ALL").is_ok_and(|v| v == "1") {
                eprintln!(
                    "PROF dev={} ops={} q8={:.2} attn={:.2} lm={:.2} other={:.2} host={:.2} ms",
                    self.device,
                    g.used,
                    buckets[0] as f64 / 1e6,
                    buckets[1] as f64 / 1e6,
                    buckets[2] as f64 / 1e6,
                    buckets[3] as f64 / 1e6,
                    buckets[4] as f64 / 1e6
                );
            }
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
        self.bind()?;
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
            if matches!(op, DecodeOp::LmHead) {
                // The vocab GEMV is the forward's last kernel, so its two
                // markers share a submission and their event delta is noise.
                // One GEMV per sampled forward: time it between stream syncs.
                let _ = stream_synchronize();
                let started = std::time::Instant::now();
                let result = f();
                let _ = stream_synchronize();
                let ns = started.elapsed().as_nanos() as u64;
                let mut g = self.profile.lock().unwrap_or_else(|e| e.into_inner());
                g.lm_head_ns += ns.max(1);
                return result;
            }
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
        // Vocab GEMV under lm_head_scope must land in lm_head_us, not q8_gemm_us.
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            DecodeOp::Q8Gemm
        };
        self.timed_gpu(op, || gemm_nt_q8_0_device(a, b, c, m, n, k))
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
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            DecodeOp::OtherGpu
        };
        self.timed_gpu(op, || gemm_nt_q4_k_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_tq2_0_dev(
        &self,
        a: *const f32,
        b: *const u8,
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
        self.timed_gpu(op, || gemm_nt_tq2_0_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_tq1_0_dev(
        &self,
        a: *const f32,
        b: *const u8,
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
        self.timed_gpu(op, || gemm_nt_tq1_0_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_rocmfp4_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            // Treat packed ROCmFP4 body GEMVs like Q8 for §12 decode accounting.
            DecodeOp::Q8Gemm
        };
        self.timed_gpu(op, || gemm_nt_rocmfp4_device(a, b, c, m, n, k))
    }

    /// Paired-4:8 FP4 (tensor type 200) GEMV; accounted like [`Self::gemm_nt_rocmfp4_dev`].
    pub fn gemm_nt_p48fp4_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            DecodeOp::Q8Gemm
        };
        self.timed_gpu(op, || gemm_nt_p48fp4_device(a, b, c, m, n, k))
    }

    /// ROCmFP4 FAST (ggml type 101) GEMV; accounted like [`Self::gemm_nt_rocmfp4_dev`].
    pub fn gemm_nt_rocmfp4_fast_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        let op = if self.lm_head.load(Ordering::Relaxed) != 0 {
            DecodeOp::LmHead
        } else {
            // Treat packed ROCmFP4 body GEMVs like Q8 for §12 decode accounting.
            DecodeOp::Q8Gemm
        };
        self.timed_gpu(op, || gemm_nt_rocmfp4_fast_device(a, b, c, m, n, k))
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

    pub fn gemm_nt_q3_k_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::OtherGpu, || gemm_nt_q3_k_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_q2_0_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::OtherGpu, || gemm_nt_q2_0_device(a, b, c, m, n, k))
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

    /// Dense weights of a model whose "BF16" tensors are Q8_0 in the file:
    /// `bytes` at `ptr` hold 36-byte staged Q8_0 blocks
    /// ([`stage_q8_0_gemv`]). Every dense GEMM of this workspace whose
    /// weight pointer falls in a registered range runs the Q8_0 kernel, so
    /// the callers' BF16 paths serve both formats.
    pub fn register_q8_dense(&self, ptr: *const u8, bytes: usize) {
        let mut ranges = self.q8_dense.write().unwrap_or_else(|e| e.into_inner());
        ranges.push((ptr as usize, ptr as usize + bytes));
        ranges.sort_unstable();
    }

    fn is_q8_dense(&self, ptr: usize) -> bool {
        let ranges = self.q8_dense.read().unwrap_or_else(|e| e.into_inner());
        if ranges.is_empty() {
            return false;
        }
        let i = ranges.partition_point(|r| r.0 <= ptr);
        i > 0 && ptr < ranges[i - 1].1
    }

    pub fn gemm_nt_bf16_dev(
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
        if self.is_q8_dense(b as usize) {
            return self.timed_gpu(op, || gemm_nt_q8_0_device(a, b.cast(), c, m, n, k));
        }
        self.timed_gpu(op, || gemm_nt_bf16_device(a, b, c, m, n, k))
    }

    pub fn gemm_nt_mxfp4_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        self.timed_gpu(DecodeOp::Q8Gemm, || gemm_nt_mxfp4_device(a, b, c, m, n, k))
    }

    /// True when two non-blocking SwiGLU streams were created on this device.
    pub fn has_swiglu_pair(&self) -> bool {
        self.swiglu_pipeline_slots() >= 2
    }

    /// Live side streams for pipelined expert SwiGLU (2..=4).
    pub fn swiglu_pipeline_slots(&self) -> usize {
        self.swiglu_s.iter().take_while(|s| !s.is_null()).count()
    }

    /// Bytes available in one pipelined SwiGLU slot (`scratch / n_streams`).
    pub fn swiglu_slot_bytes(&self) -> usize {
        self.scratch_bytes / self.swiglu_pipeline_slots().max(1)
    }

    /// Device workspace for packed MXFP4 SwiGLU: `x | gate | up | y`.
    pub fn swiglu_workspace_bytes(m: usize, dim: usize, ff: usize) -> usize {
        m.saturating_mul(
            dim.saturating_add(ff)
                .saturating_add(ff)
                .saturating_add(dim),
        )
        .saturating_mul(4)
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
        self.ensure_decode_batch(dim, kv_dim, ff_dim, vocab, max_kv, 1, 1, dim)
    }

    /// Same workspace as [`ensure_decode`], sized for `batch` token rows (prefill M=N).
    /// `logits_rows` is 1 for last-token prefill logits, or `batch` for packed linear verify.
    /// `q_dim` is the scratch width for `q`/`attn` (SSM in_proj); use `dim` for dense prefill.
    pub fn ensure_decode_batch(
        &self,
        dim: usize,
        kv_dim: usize,
        ff_dim: usize,
        vocab: usize,
        max_kv: usize,
        batch: usize,
        logits_rows: usize,
        q_dim: usize,
    ) -> Result<DeviceDecode, AllpacaError> {
        let max_kv = max_kv.max(1);
        let batch = batch.max(1);
        let logits_rows = logits_rows.max(1);
        let q_dim = q_dim.max(dim).max(1);
        let mut guard = self.decode.lock().unwrap_or_else(|e| e.into_inner());
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
                    || a.q_dim < q_dim
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
            let floats = decode_floats(
                dim,
                kv_dim,
                ff_dim,
                vocab,
                max_kv,
                batch,
                logits_rows,
                q_dim,
            );
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
                q_dim,
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

    pub fn dequant_q8_0(
        &self,
        blocks: *const u8,
        n: i32,
        out: *mut f32,
    ) -> Result<(), AllpacaError> {
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
                    ffi::allpaca_embed_gather_q8_0(
                        table,
                        token as i32,
                        out,
                        dim,
                        std::ptr::null_mut(),
                    )
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

    pub fn embed_gather_rocmfp4(
        &self,
        table: *const u8,
        token: u32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rocmfp4 embed gather".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_rocmfp4(
                        table,
                        token as i32,
                        out,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_rocmfp4")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = token;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_rocmfp4 unavailable".into(),
            )))
        }
    }

    /// One token's row from a ROCmFP4 FAST (17-byte block) embedding table.
    pub fn embed_gather_rocmfp4_fast(
        &self,
        table: *const u8,
        token: u32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if table.is_null() || out.is_null() || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rocmfp4 embed gather".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_embed_gather_rocmfp4_fast(
                        table,
                        token as i32,
                        out,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_rocmfp4_fast")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = token;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_rocmfp4 unavailable".into(),
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

    /// Packed BF16 `b` already on device; activations H2D / D2H via scratch.
    pub fn gemm_nt_bf16_dev_b(
        &self,
        a: &[f32],
        b: *const u16,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 || b.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid resident BF16 GEMM".into(),
            )));
        }
        let a_n = (m as usize) * (k as usize);
        let c_n = (m as usize) * (n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "resident BF16 GEMM buffer size mismatch".into(),
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
                self.gemm_nt_bf16_dev(da.cast(), b, dc.cast(), m, n, k)?;
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
        self.gemm_nt_bf16_dev(da.cast(), b, dc.cast(), m, n, k)?;
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)?;
        self.add_d2h(c_n * 4);
        Ok(())
    }

    /// Packed BF16 `b` on device; `a` is already a device `[m,k]` pointer (no A H2D).
    pub fn gemm_nt_bf16_from_device_a(
        &self,
        a: *const f32,
        b: *const u16,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device-A BF16 GEMM".into(),
            )));
        }
        let c_n = (m as usize) * (n as usize);
        if c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "device-A BF16 GEMM C size mismatch".into(),
            )));
        }
        let c_bytes = c_n * 4;
        if c_bytes > self.scratch_bytes {
            let dc = device_malloc(c_bytes, self.device)?;
            let result = (|| {
                self.bind()?;
                self.gemm_nt_bf16_dev(a, b, dc.cast(), m, n, k)?;
                memcpy(c.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H)?;
                self.add_d2h(c_bytes);
                Ok(())
            })();
            device_free(dc, c_bytes);
            return result;
        }
        self.bind()?;
        let dc = self.scratch;
        self.gemm_nt_bf16_dev(a, b, dc.cast(), m, n, k)?;
        memcpy(c.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H)?;
        self.add_d2h(c_bytes);
        Ok(())
    }

    /// Packed BF16 `b` on device; `a` and `c` are device pointers (no activation copies).
    pub fn gemm_nt_bf16_from_device_a_into(
        &self,
        a: *const f32,
        b: *const u16,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 || a.is_null() || b.is_null() || c.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device-A/C BF16 GEMM".into(),
            )));
        }
        self.bind()?;
        self.gemm_nt_bf16_dev(a, b, c, m, n, k)
    }

    /// Packed MXFP4 `b` already on device; activations H2D / D2H via scratch.
    pub fn gemm_nt_mxfp4_dev_b(
        &self,
        a: &[f32],
        b: *const u8,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 || b.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid resident MXFP4 GEMM".into(),
            )));
        }
        let a_n = (m as usize) * (k as usize);
        let c_n = (m as usize) * (n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "resident MXFP4 GEMM buffer size mismatch".into(),
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
                self.gemm_nt_mxfp4_dev(da.cast(), b, dc.cast(), m, n, k)?;
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
        self.gemm_nt_mxfp4_dev(da.cast(), b, dc.cast(), m, n, k)?;
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)?;
        self.add_d2h(c_n * 4);
        Ok(())
    }

    fn with_swiglu_workspace<R>(
        &self,
        m: usize,
        d: usize,
        h: usize,
        f: impl FnOnce(*mut u8, *mut u8, *mut u8, *mut u8) -> Result<R, AllpacaError>,
    ) -> Result<R, AllpacaError> {
        let need = m
            .saturating_mul(d.saturating_add(h).saturating_add(h).saturating_add(d))
            .saturating_mul(4);
        if need == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid SwiGLU workspace".into(),
            )));
        }
        let run = |base: *mut u8| {
            let dx = base;
            let dg = unsafe { base.add(m * d * 4) };
            let du = unsafe { base.add(m * (d + h) * 4) };
            let dy = unsafe { base.add(m * (d + h + h) * 4) };
            f(dx, dg, du, dy)
        };
        if need <= self.scratch_bytes {
            self.bind()?;
            run(self.scratch)
        } else {
            let p = device_malloc(need, self.device)?;
            let result = (|| {
                self.bind()?;
                run(p)
            })();
            // Wait only the default-stream kernels that used `p`. A full
            // `stream_synchronize` waits every stream and deadlocks bounce H2D
            // (legacy null stream vs host slot-event waits).
            let _ = self.join_default_event();
            device_free(p, need);
            result
        }
    }

    fn join_default_event(&self) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let ev = self.swiglu_ev[0];
            if ev.is_null() {
                return stream_synchronize();
            }
            event_record(ev)?;
            event_synchronize(ev)
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Ok(())
        }
    }

    /// Packed MXFP4 SwiGLU: H2D `x` once (`M` rows), gate/up GEMM, device SiLU, down GEMM, D2H `y`.
    #[allow(clippy::too_many_arguments)]
    /// Clamp for the MXFP4 SwiGLU this workspace launches from now on: the
    /// gate is capped at `limit` before SiLU and the up projection to
    /// `±limit`; 0 = none. DeepSeek-V4 sets its layer's value before each
    /// layer's experts.
    pub fn set_mxfp4_swiglu_limit(&self, limit: f32) {
        self.mxfp4_swiglu_limit
            .store(limit.max(0.0).to_bits(), std::sync::atomic::Ordering::Relaxed);
    }

    fn mxfp4_limit(&self) -> f32 {
        f32::from_bits(self.mxfp4_swiglu_limit.load(std::sync::atomic::Ordering::Relaxed))
    }

    pub fn mxfp4_swiglu_gemm(
        &self,
        x: &[f32],
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        y: &mut [f32],
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || dim <= 0 || ff <= 0 || gate_w.is_null() || up_w.is_null() || down_w.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid MXFP4 SwiGLU GEMM".into(),
            )));
        }
        let rows = m as usize;
        let d = dim as usize;
        let h = ff as usize;
        if x.len() < rows * d || y.len() < rows * d {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU buffer size mismatch".into(),
            )));
        }
        let silu_n = m.saturating_mul(ff);
        self.with_swiglu_workspace(rows, d, h, |dx, dg, du, dy| {
            let bytes = rows * d * 4;
            memcpy(dx, x.as_ptr().cast(), bytes, MEMCPY_H2D)?;
            self.add_h2d(bytes);
            self.mxfp4_swiglu_compute(dx, dg, du, dy, gate_w, up_w, down_w, m, dim, ff, silu_n)?;
            memcpy(y.as_mut_ptr().cast(), dy, bytes, MEMCPY_D2H)?;
            self.add_d2h(bytes);
            Ok(())
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn mxfp4_swiglu_compute(
        &self,
        dx: *mut u8,
        dg: *mut u8,
        du: *mut u8,
        dy: *mut u8,
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        m: i32,
        dim: i32,
        ff: i32,
        silu_n: i32,
    ) -> Result<(), AllpacaError> {
        let limit = self.mxfp4_limit();
        match self.timed_gpu(DecodeOp::Q8Gemm, || {
            mxfp4_swiglu_gate_up_device_on(
                dx.cast(),
                gate_w,
                up_w,
                dg.cast(),
                m,
                ff,
                dim,
                limit,
                std::ptr::null_mut(),
            )
        }) {
            Ok(()) => return self.gemm_nt_mxfp4_dev(dg.cast(), down_w, dy.cast(), m, dim, ff),
            // Only the fused kernel clamps; the split path below does not.
            Err(e) if limit > 0.0 => return Err(e),
            Err(_) => {}
        }
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let s0 = self.swiglu_s[0];
            let s1 = self.swiglu_s[1];
            let ev_h2d = self.swiglu_ev[0];
            let ev_g = self.swiglu_ev[1];
            let ev_u = self.swiglu_ev[2];
            // M=1 decode grids underfill gfx906 (~32 blocks for ff=2048). Dual
            // launch helps prefill (M>1) but the event join shows up on `"hi"`.
            if m > 1
                && !s0.is_null()
                && !s1.is_null()
                && !ev_h2d.is_null()
                && !ev_g.is_null()
                && !ev_u.is_null()
            {
                event_record_on(ev_h2d, std::ptr::null_mut())?;
                stream_wait_event(s0, ev_h2d)?;
                stream_wait_event(s1, ev_h2d)?;
                self.timed_gpu(DecodeOp::Q8Gemm, || {
                    gemm_nt_mxfp4_device_on(dx.cast(), gate_w, dg.cast(), m, ff, dim, s0)?;
                    gemm_nt_mxfp4_device_on(dx.cast(), up_w, du.cast(), m, ff, dim, s1)?;
                    event_record_on(ev_g, s0)?;
                    event_record_on(ev_u, s1)?;
                    stream_wait_event(std::ptr::null_mut(), ev_g)?;
                    stream_wait_event(std::ptr::null_mut(), ev_u)?;
                    Ok(())
                })?;
                self.silu_mul_device(dg.cast(), du.cast(), silu_n)?;
                self.gemm_nt_mxfp4_dev(dg.cast(), down_w, dy.cast(), m, dim, ff)?;
                return Ok(());
            }
        }
        self.gemm_nt_mxfp4_dev(dx.cast(), gate_w, dg.cast(), m, ff, dim)?;
        self.gemm_nt_mxfp4_dev(dx.cast(), up_w, du.cast(), m, ff, dim)?;
        self.silu_mul_device(dg.cast(), du.cast(), silu_n)?;
        self.gemm_nt_mxfp4_dev(dg.cast(), down_w, dy.cast(), m, dim, ff)
    }

    /// Queue packed MXFP4 SwiGLU on ping-pong slot `0` or `1`. `x` must stay live until
    /// [`Self::mxfp4_swiglu_collect`].
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_swiglu_enqueue(
        &self,
        slot: usize,
        x: &[f32],
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if gate_w.is_null() || up_w.is_null() || down_w.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid MXFP4 SwiGLU enqueue".into(),
            )));
        }
        let (stream, dx, dg, du, dy) = self.mxfp4_swiglu_slot(slot, m, dim, ff)?;
        let rows = m as usize;
        let d = dim as usize;
        if x.len() < rows.saturating_mul(d) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU enqueue buffer size mismatch".into(),
            )));
        }
        let bytes = rows * d * 4;
        memcpy_on(dx, x.as_ptr().cast(), bytes, MEMCPY_H2D, stream)?;
        self.add_h2d(bytes);
        self.mxfp4_swiglu_launch_on(dx, dg, du, dy, gate_w, up_w, down_w, m, dim, ff, stream)
    }

    /// Gather `row_idx` from device `xs` (see [`Self::moe_begin`]) and SwiGLU on `slot`.
    /// `row_idx` / `row_w` stay on the slot until [`Self::mxfp4_swiglu_accumulate_slot`].
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_swiglu_enqueue_from_rows(
        &self,
        slot: usize,
        row_idx: &[i32],
        row_w: &[f32],
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if gate_w.is_null() || up_w.is_null() || down_w.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid MXFP4 SwiGLU enqueue".into(),
            )));
        }
        let rows = m as usize;
        if row_idx.len() < rows || row_w.len() < rows {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU row meta size mismatch".into(),
            )));
        }
        let (stream, dx, dg, du, dy) = self.mxfp4_swiglu_slot(slot, m, dim, ff)?;
        let (table, idx_dev, w_dev, n_tok) = self.moe_slot_meta(slot)?;
        if rows > n_tok {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU rows exceed moe n_tok".into(),
            )));
        }
        memcpy_on(
            idx_dev.cast(),
            row_idx.as_ptr().cast(),
            rows * 4,
            MEMCPY_H2D,
            stream,
        )?;
        memcpy_on(
            w_dev.cast(),
            row_w.as_ptr().cast(),
            rows * 4,
            MEMCPY_H2D,
            stream,
        )?;
        self.embed_gather_rows_dev(table, idx_dev, dx.cast(), m, dim, stream)?;
        self.mxfp4_swiglu_launch_on(dx, dg, du, dy, gate_w, up_w, down_w, m, dim, ff, stream)
    }

    /// Wait for [`Self::mxfp4_swiglu_enqueue`] on `slot` and copy `y` to the host.
    pub fn mxfp4_swiglu_collect(
        &self,
        slot: usize,
        y: &mut [f32],
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if slot >= self.swiglu_pipeline_slots() || m <= 0 || dim <= 0 || ff <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid MXFP4 SwiGLU collect".into(),
            )));
        }
        let stream = self.swiglu_s[slot];
        if stream.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU pair streams missing".into(),
            )));
        }
        let rows = m as usize;
        let d = dim as usize;
        let h = ff as usize;
        if y.len() < rows.saturating_mul(d) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU collect buffer size mismatch".into(),
            )));
        }
        let (_, _, _, dy) = self.swiglu_slot_ptrs(slot, rows, d, h);
        let bytes = rows * d * 4;
        memcpy_on(y.as_mut_ptr().cast(), dy, bytes, MEMCPY_D2H, stream)?;
        self.add_d2h(bytes);
        Ok(())
    }

    /// Scatter-add the slot's `y` into device `moe_acc`. Same-stream after enqueue, no D2H.
    pub fn mxfp4_swiglu_accumulate_slot(
        &self,
        slot: usize,
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        let (stream, _, _, _, dy) = self.mxfp4_swiglu_slot(slot, m, dim, ff)?;
        let (_, idx_dev, w_dev, n_tok) = self.moe_slot_meta(slot)?;
        let acc = self.moe_acc_ptr()?;
        self.scatter_add_scale_rows_dev(
            dy.cast(),
            idx_dev,
            w_dev,
            acc,
            m,
            dim,
            n_tok as i32,
            stream,
        )
    }

    /// Serial device-side SwiGLU + scatter-add on the default stream (joins before return).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_swiglu_accumulate_gemm(
        &self,
        row_idx: &[i32],
        row_w: &[f32],
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || dim <= 0 || ff <= 0 || gate_w.is_null() || up_w.is_null() || down_w.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid MXFP4 SwiGLU accumulate GEMM".into(),
            )));
        }
        let rows = m as usize;
        let d = dim as usize;
        let h = ff as usize;
        if row_idx.len() < rows || row_w.len() < rows {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU accumulate meta size mismatch".into(),
            )));
        }
        let silu_n = m.saturating_mul(ff);
        let (table, idx_dev, w_dev, n_tok) = self.moe_slot_meta(0)?;
        if rows > n_tok {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU rows exceed moe n_tok".into(),
            )));
        }
        let acc = self.moe_acc_ptr()?;
        self.with_swiglu_workspace(rows, d, h, |dx, dg, du, dy| {
            memcpy(
                idx_dev.cast(),
                row_idx.as_ptr().cast(),
                rows * 4,
                MEMCPY_H2D,
            )?;
            memcpy(w_dev.cast(), row_w.as_ptr().cast(), rows * 4, MEMCPY_H2D)?;
            self.embed_gather_rows_dev(table, idx_dev, dx.cast(), m, dim, std::ptr::null_mut())?;
            self.mxfp4_swiglu_compute(dx, dg, du, dy, gate_w, up_w, down_w, m, dim, ff, silu_n)?;
            self.scatter_add_scale_rows_dev(
                dy.cast(),
                idx_dev,
                w_dev,
                acc,
                m,
                dim,
                n_tok as i32,
                std::ptr::null_mut(),
            )
        })?;
        self.join_default_event()
    }

    /// Upload packed `xs` `[n_tok, dim]` once and zero the expert accumulator.
    pub fn moe_begin(&self, xs: &[f32], n_tok: usize, dim: usize) -> Result<(), AllpacaError> {
        if n_tok == 0 || dim == 0 {
            return Ok(());
        }
        let floats = n_tok.saturating_mul(dim);
        if xs.len() < floats || floats > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid moe xs shape".into(),
            )));
        }
        self.ensure_moe(n_tok, dim)?;
        let (x, acc, bytes) = {
            let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
            let s = g.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
            })?;
            (s.x, s.acc, floats * 4)
        };
        memcpy(x, xs.as_ptr().cast(), bytes, MEMCPY_H2D)?;
        self.add_h2d(bytes);
        memset_on(acc, 0, bytes, std::ptr::null_mut())?;
        stream_synchronize()?;
        if let Some(s) = self.moe.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            s.active = true;
        }
        Ok(())
    }

    /// Device `xs` `[n_tok, dim]` already resident; D2D into `moe_x` (no activation H2D).
    pub fn moe_begin_from_device(
        &self,
        xs: *const f32,
        n_tok: usize,
        dim: usize,
    ) -> Result<(), AllpacaError> {
        if n_tok == 0 || dim == 0 {
            return Ok(());
        }
        let floats = n_tok.saturating_mul(dim);
        if xs.is_null() || floats > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device moe xs".into(),
            )));
        }
        self.ensure_moe(n_tok, dim)?;
        let (x, acc, bytes) = {
            let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
            let s = g.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
            })?;
            (s.x, s.acc, floats * 4)
        };
        if x as *const f32 != xs {
            self.copy_d2d_f32(x.cast(), xs, floats)?;
        }
        memset_on(acc, 0, bytes, std::ptr::null_mut())?;
        stream_synchronize()?;
        if let Some(s) = self.moe.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            s.active = true;
        }
        Ok(())
    }

    /// Device `xs` on `src_dev`; same-GPU D2D or peer into `moe_x` (no activation H2D).
    pub fn moe_begin_from_peer(
        &self,
        xs: *const f32,
        src_dev: i32,
        n_tok: usize,
        dim: usize,
    ) -> Result<(), AllpacaError> {
        if src_dev == self.device {
            return self.moe_begin_from_device(xs, n_tok, dim);
        }
        if n_tok == 0 || dim == 0 {
            return Ok(());
        }
        let floats = n_tok.saturating_mul(dim);
        if xs.is_null() || floats > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid peer moe xs".into(),
            )));
        }
        self.ensure_moe(n_tok, dim)?;
        let (x, acc, bytes) = {
            let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
            let s = g.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
            })?;
            (s.x, s.acc, floats * 4)
        };
        memcpy_peer(x, self.device, xs.cast(), src_dev, bytes)?;
        self.bind()?;
        memset_on(acc, 0, bytes, std::ptr::null_mut())?;
        stream_synchronize()?;
        if let Some(s) = self.moe.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            s.active = true;
        }
        Ok(())
    }

    pub fn moe_active(&self) -> bool {
        self.moe
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|s| s.active)
    }

    /// Device pointer to packed `xs` from [`Self::moe_begin`].
    pub fn moe_x_device(&self) -> Result<*const f32, AllpacaError> {
        self.moe_x_ptr()
    }

    /// Device pointer to the expert/shexp accumulator.
    pub fn moe_acc_device(&self) -> Result<*mut f32, AllpacaError> {
        self.moe_acc_ptr()
    }

    /// Packed `n_tok * dim` floats in the active moe stage.
    pub fn moe_floats(&self) -> Result<usize, AllpacaError> {
        let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
        let s = g.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
        })?;
        if !s.active {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe stage inactive".into(),
            )));
        }
        Ok(s.n_tok.saturating_mul(s.dim))
    }

    /// Device `moe_x` used as H2D staging for [`Self::moe_add_from_host`].
    pub fn moe_staging_device(&self) -> Result<*mut u8, AllpacaError> {
        let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
        let s = g.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
        })?;
        if !s.active {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe stage inactive".into(),
            )));
        }
        Ok(s.x)
    }

    /// `acc += add` after H2D into `moe_x` (callers must no longer need `xs`).
    pub fn moe_add_from_host(&self, add: &[f32]) -> Result<(), AllpacaError> {
        let (x, acc, n) = {
            let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
            let s = g.as_ref().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
            })?;
            if !s.active {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "moe stage inactive".into(),
                )));
            }
            let n = s.n_tok.saturating_mul(s.dim);
            (s.x, s.acc, n)
        };
        if add.len() < n || n > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe_add_from_host size mismatch".into(),
            )));
        }
        let bytes = n * 4;
        self.bind()?;
        memcpy(x, add.as_ptr().cast(), bytes, MEMCPY_H2D)?;
        self.add_h2d(bytes);
        self.residual_add(acc.cast(), x.cast(), n as i32)
    }

    /// Drop the moe stage without downloading `moe_acc`.
    pub fn moe_release(&self) {
        if let Some(s) = self.moe.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            s.active = false;
        }
    }

    /// Join expert SwiGLU streams without downloading `moe_acc`.
    pub fn moe_join_compute(&self) -> Result<(), AllpacaError> {
        self.bind()?;
        self.join_swiglu_streams()?;
        stream_synchronize()
    }

    /// Join expert streams and copy `moe_acc` `[n_tok, dim]` to the host.
    pub fn moe_finish(&self, out: &mut [f32]) -> Result<(), AllpacaError> {
        let (acc, bytes, n_tok, dim) = {
            let mut g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
            let s = g.as_mut().ok_or_else(|| {
                AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
            })?;
            if !s.active {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "moe stage inactive".into(),
                )));
            }
            s.active = false;
            (
                s.acc,
                s.n_tok.saturating_mul(s.dim).saturating_mul(4),
                s.n_tok,
                s.dim,
            )
        };
        if out.len() < n_tok.saturating_mul(dim) {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe_finish buffer size mismatch".into(),
            )));
        }
        self.bind()?;
        self.join_swiglu_streams()?;
        stream_synchronize()?;
        memcpy(out.as_mut_ptr().cast(), acc, bytes, MEMCPY_D2H)?;
        self.add_d2h(bytes);
        Ok(())
    }

    fn ensure_moe(&self, n_tok: usize, dim: usize) -> Result<(), AllpacaError> {
        let need = n_tok.saturating_mul(dim).saturating_mul(4);
        let meta_need = SWIGLU_STREAMS
            .saturating_mul(n_tok)
            .saturating_mul(8)
            .max(8);
        if need == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid moe stage shape".into(),
            )));
        }
        self.bind()?;
        let mut g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
        if g.as_ref()
            .is_some_and(|s| s.x_bytes >= need && s.acc_bytes >= need && s.meta_bytes >= meta_need)
        {
            if let Some(s) = g.as_mut() {
                s.n_tok = n_tok;
                s.dim = dim;
                s.active = false;
            }
            return Ok(());
        }
        let x = device_malloc(need, self.device)?;
        let acc = match device_malloc(need, self.device) {
            Ok(p) => p,
            Err(e) => {
                device_free(x, need);
                return Err(e);
            }
        };
        let meta = match device_malloc(meta_need, self.device) {
            Ok(p) => p,
            Err(e) => {
                device_free(x, need);
                device_free(acc, need);
                return Err(e);
            }
        };
        if let Some(old) = g.take() {
            free_moe_stage(old);
        }
        *g = Some(MoeStage {
            x,
            acc,
            meta,
            x_bytes: need,
            acc_bytes: need,
            meta_bytes: meta_need,
            n_tok,
            dim,
            active: false,
        });
        Ok(())
    }

    fn moe_slot_meta(
        &self,
        slot: usize,
    ) -> Result<(*const f32, *mut i32, *mut f32, usize), AllpacaError> {
        let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
        let s = g.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
        })?;
        if !s.active || slot >= SWIGLU_STREAMS || s.n_tok == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe meta slot unavailable".into(),
            )));
        }
        let stride = s.n_tok.saturating_mul(8);
        // SAFETY: `ensure_moe` sized `meta` to `SWIGLU_STREAMS * n_tok * 8`.
        let base = unsafe { s.meta.add(slot.saturating_mul(stride)) };
        let w = unsafe { base.add(s.n_tok.saturating_mul(4)) };
        Ok((s.x.cast(), base.cast(), w.cast(), s.n_tok))
    }

    fn moe_x_ptr(&self) -> Result<*const f32, AllpacaError> {
        let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
        let s = g.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
        })?;
        if !s.active {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe stage inactive".into(),
            )));
        }
        Ok(s.x.cast())
    }

    fn moe_acc_ptr(&self) -> Result<*mut f32, AllpacaError> {
        let g = self.moe.lock().unwrap_or_else(|e| e.into_inner());
        let s = g.as_ref().ok_or_else(|| {
            AllpacaError::Degraded(DegradedError::FeatureDisabled("moe stage missing".into()))
        })?;
        if !s.active {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "moe stage inactive".into(),
            )));
        }
        Ok(s.acc.cast())
    }

    fn join_swiglu_streams(&self) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            for s in self.swiglu_s {
                if !s.is_null() {
                    unsafe {
                        stream_synchronize_on(s)?;
                    }
                }
            }
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Ok(())
        }
    }

    fn mxfp4_swiglu_slot(
        &self,
        slot: usize,
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(*mut std::ffi::c_void, *mut u8, *mut u8, *mut u8, *mut u8), AllpacaError> {
        if slot >= self.swiglu_pipeline_slots() || m <= 0 || dim <= 0 || ff <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid MXFP4 SwiGLU slot".into(),
            )));
        }
        let stream = self.swiglu_s[slot];
        if stream.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU pair streams missing".into(),
            )));
        }
        let rows = m as usize;
        let d = dim as usize;
        let h = ff as usize;
        let need = Self::swiglu_workspace_bytes(rows, d, h);
        let half = self.swiglu_slot_bytes();
        if need == 0 || need > half {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "MXFP4 SwiGLU slot does not fit scratch half".into(),
            )));
        }
        self.bind()?;
        let (dx, dg, du, dy) = self.swiglu_slot_ptrs(slot, rows, d, h);
        Ok((stream, dx, dg, du, dy))
    }

    fn mxfp4_swiglu_launch_on(
        &self,
        dx: *mut u8,
        dg: *mut u8,
        du: *mut u8,
        dy: *mut u8,
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        m: i32,
        dim: i32,
        ff: i32,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), AllpacaError> {
        let silu_n = m.saturating_mul(ff);
        let limit = self.mxfp4_limit();
        match mxfp4_swiglu_gate_up_device_on(
            dx.cast(),
            gate_w,
            up_w,
            dg.cast(),
            m,
            ff,
            dim,
            limit,
            stream,
        ) {
            Ok(()) => {
                return gemm_nt_mxfp4_device_on(dg.cast(), down_w, dy.cast(), m, dim, ff, stream)
            }
            // Only the fused kernel clamps; the split path below does not.
            Err(e) if limit > 0.0 => return Err(e),
            Err(_) => {}
        }
        gemm_nt_mxfp4_device_on(dx.cast(), gate_w, dg.cast(), m, ff, dim, stream)?;
        gemm_nt_mxfp4_device_on(dx.cast(), up_w, du.cast(), m, ff, dim, stream)?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_silu_mul(dg.cast(), du.cast(), silu_n, stream) };
            status(rc, "silu_mul")?;
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = silu_n;
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP silu_mul unavailable".into(),
            )));
        }
        gemm_nt_mxfp4_device_on(dg.cast(), down_w, dy.cast(), m, dim, ff, stream)
    }

    fn embed_gather_rows_dev(
        &self,
        table: *const f32,
        tokens: *const i32,
        out: *mut f32,
        n_tok: i32,
        dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), AllpacaError> {
        if table.is_null() || tokens.is_null() || out.is_null() || n_tok <= 0 || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device gather".into(),
            )));
        }
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc =
                unsafe { ffi::allpaca_embed_gather_rows(table, tokens, out, n_tok, dim, stream) };
            status(rc, "embed_gather_rows")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = stream;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather unavailable".into(),
            )))
        }
    }

    fn scatter_add_scale_rows_dev(
        &self,
        src: *const f32,
        idx: *const i32,
        w: *const f32,
        acc: *mut f32,
        m: i32,
        dim: i32,
        n_tok: i32,
        stream: *mut std::ffi::c_void,
    ) -> Result<(), AllpacaError> {
        if src.is_null() || idx.is_null() || w.is_null() || acc.is_null() || m <= 0 || dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid scatter-add".into(),
            )));
        }
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_scatter_add_scale_rows(src, idx, w, acc, m, dim, n_tok, stream)
            };
            status(rc, "scatter_add_scale_rows")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = stream;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP scatter_add_scale_rows unavailable".into(),
            )))
        }
    }

    fn swiglu_slot_ptrs(
        &self,
        slot: usize,
        m: usize,
        d: usize,
        h: usize,
    ) -> (*mut u8, *mut u8, *mut u8, *mut u8) {
        let half = self.swiglu_slot_bytes();
        // SAFETY: callers keep `slot < swiglu_pipeline_slots()` and
        // `swiglu_workspace_bytes(m,d,h) <= swiglu_slot_bytes()`, so the four
        // sub-buffers sit inside this GPU scratch slot.
        let base = unsafe { self.scratch.add(slot.saturating_mul(half)) };
        let dx = base;
        let dg = unsafe { base.add(m * d * 4) };
        let du = unsafe { base.add(m * (d + h) * 4) };
        let dy = unsafe { base.add(m * (d + h + h) * 4) };
        (dx, dg, du, dy)
    }

    /// Packed MXFP4 SwiGLU expert: H2D `x` once, gate/up GEMV, device SiLU, down GEMV, D2H `y`.
    pub fn mxfp4_swiglu_gemv(
        &self,
        x: &[f32],
        gate_w: *const u8,
        up_w: *const u8,
        down_w: *const u8,
        y: &mut [f32],
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        self.mxfp4_swiglu_gemm(x, gate_w, up_w, down_w, y, 1, dim, ff)
    }

    /// Packed BF16 SwiGLU: H2D `x` once (`M` rows), gate/up GEMM, device SiLU, down GEMM, D2H `y`.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_swiglu_gemm(
        &self,
        x: &[f32],
        gate_w: *const u16,
        up_w: *const u16,
        down_w: *const u16,
        y: &mut [f32],
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || dim <= 0 || ff <= 0 || gate_w.is_null() || up_w.is_null() || down_w.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid BF16 SwiGLU GEMM".into(),
            )));
        }
        let rows = m as usize;
        let d = dim as usize;
        let h = ff as usize;
        if x.len() < rows * d || y.len() < rows * d {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "BF16 SwiGLU buffer size mismatch".into(),
            )));
        }
        let silu_n = m.saturating_mul(ff);
        self.with_swiglu_workspace(rows, d, h, |dx, dg, du, dy| {
            let bytes = rows * d * 4;
            memcpy(dx, x.as_ptr().cast(), bytes, MEMCPY_H2D)?;
            self.add_h2d(bytes);
            self.gemm_nt_bf16_dev(dx.cast(), gate_w, dg.cast(), m, ff, dim)?;
            self.gemm_nt_bf16_dev(dx.cast(), up_w, du.cast(), m, ff, dim)?;
            self.silu_mul_device(dg.cast(), du.cast(), silu_n)?;
            self.gemm_nt_bf16_dev(dg.cast(), down_w, dy.cast(), m, dim, ff)?;
            memcpy(y.as_mut_ptr().cast(), dy, bytes, MEMCPY_D2H)?;
            self.add_d2h(bytes);
            Ok(())
        })
    }

    /// Packed BF16 SwiGLU from device `moe_x` into `moe_acc` (no activation H2D/D2H).
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_swiglu_accumulate(
        &self,
        gate_w: *const u16,
        up_w: *const u16,
        down_w: *const u16,
        m: i32,
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || dim <= 0 || ff <= 0 || gate_w.is_null() || up_w.is_null() || down_w.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid BF16 SwiGLU accumulate".into(),
            )));
        }
        let rows = m as usize;
        let d = dim as usize;
        let h = ff as usize;
        let n = rows.saturating_mul(d);
        if n > i32::MAX as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "BF16 SwiGLU accumulate shape exceeds i32".into(),
            )));
        }
        let dx_in = self.moe_x_ptr()?;
        let acc = self.moe_acc_ptr()?;
        let silu_n = m.saturating_mul(ff);
        self.with_swiglu_workspace(rows, d, h, |_dx, dg, du, dy| {
            self.gemm_nt_bf16_dev(dx_in, gate_w, dg.cast(), m, ff, dim)?;
            self.gemm_nt_bf16_dev(dx_in, up_w, du.cast(), m, ff, dim)?;
            self.silu_mul_device(dg.cast(), du.cast(), silu_n)?;
            self.gemm_nt_bf16_dev(dg.cast(), down_w, dy.cast(), m, dim, ff)?;
            self.residual_add(acc, dy.cast(), n as i32)
        })?;
        self.join_default_event()
    }

    /// Packed BF16 SwiGLU: H2D `x` once, gate/up GEMV, device SiLU, down GEMV, D2H `y`.
    pub fn bf16_swiglu_gemv(
        &self,
        x: &[f32],
        gate_w: *const u16,
        up_w: *const u16,
        down_w: *const u16,
        y: &mut [f32],
        dim: i32,
        ff: i32,
    ) -> Result<(), AllpacaError> {
        self.bf16_swiglu_gemm(x, gate_w, up_w, down_w, y, 1, dim, ff)
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
                    ffi::allpaca_gemm(
                        da.cast(),
                        db.cast(),
                        dc.cast(),
                        m,
                        n,
                        k,
                        std::ptr::null_mut(),
                    )
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

    /// Host Q/K/V attention with optional per-head sink logits (virtual token, no V).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_sinks(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        out: &mut [f32],
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        if sinks.is_empty() {
            return self.attention(
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            );
        }
        let q_bytes = q.len() * 4;
        let k_bytes = k.len() * 4;
        let v_bytes = v.len() * 4;
        let o_bytes = out.len() * 4;
        let s_bytes = sinks.len() * 4;
        let need = q_bytes
            .saturating_add(k_bytes)
            .saturating_add(v_bytes)
            .saturating_add(o_bytes)
            .saturating_add(s_bytes);
        let mut launch = |dq: *mut u8,
                          dk: *mut u8,
                          dv: *mut u8,
                          dout: *mut u8,
                          ds: *mut u8|
         -> Result<(), AllpacaError> {
            memcpy(dq, q.as_ptr().cast(), q_bytes, MEMCPY_H2D)?;
            memcpy(dk, k.as_ptr().cast(), k_bytes, MEMCPY_H2D)?;
            memcpy(dv, v.as_ptr().cast(), v_bytes, MEMCPY_H2D)?;
            memcpy(ds, sinks.as_ptr().cast(), s_bytes, MEMCPY_H2D)?;
            self.add_h2d(q_bytes + k_bytes + v_bytes + s_bytes);
            #[cfg(all(feature = "hip", allpaca_hip_runtime))]
            {
                let rc = unsafe {
                    ffi::allpaca_attention_sinks(
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
                        ds.cast(),
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "attention_sinks")?;
                memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)?;
                self.add_d2h(o_bytes);
                Ok(())
            }
            #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
            {
                let _ = (dout, ds, causal);
                Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "HIP attention_sinks unavailable".into(),
                )))
            }
        };
        if need > 0 && need <= self.scratch_bytes {
            self.bind()?;
            let dq = self.scratch;
            let dk = unsafe { self.scratch.add(q_bytes) };
            let dv = unsafe { self.scratch.add(q_bytes + k_bytes) };
            let dout = unsafe { self.scratch.add(q_bytes + k_bytes + v_bytes) };
            let ds = unsafe { self.scratch.add(q_bytes + k_bytes + v_bytes + o_bytes) };
            return launch(dq, dk, dv, dout, ds);
        }
        let blob = device_malloc(need.max(4), self.device)?;
        let result = (|| {
            self.bind()?;
            let dq = blob;
            let dk = unsafe { blob.add(q_bytes) };
            let dv = unsafe { blob.add(q_bytes + k_bytes) };
            let dout = unsafe { blob.add(q_bytes + k_bytes + v_bytes) };
            let ds = unsafe { blob.add(q_bytes + k_bytes + v_bytes + o_bytes) };
            launch(dq, dk, dv, dout, ds)
        })();
        device_free(blob, need.max(4));
        result
    }

    /// Device `q`; H2D `k`/`v`/`sinks` only and D2H `out`.
    pub fn attention_sinks_from_device_q(
        &self,
        q: *const f32,
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        out: &mut [f32],
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        if q.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device-Q attention".into(),
            )));
        }
        if sinks.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "device-Q attention requires sinks".into(),
            )));
        }
        let k_bytes = k.len() * 4;
        let v_bytes = v.len() * 4;
        let o_bytes = out.len() * 4;
        let s_bytes = sinks.len() * 4;
        let need = k_bytes
            .saturating_add(v_bytes)
            .saturating_add(o_bytes)
            .saturating_add(s_bytes);
        let mut launch =
            |dk: *mut u8, dv: *mut u8, dout: *mut u8, ds: *mut u8| -> Result<(), AllpacaError> {
                memcpy(dk, k.as_ptr().cast(), k_bytes, MEMCPY_H2D)?;
                memcpy(dv, v.as_ptr().cast(), v_bytes, MEMCPY_H2D)?;
                memcpy(ds, sinks.as_ptr().cast(), s_bytes, MEMCPY_H2D)?;
                self.add_h2d(k_bytes + v_bytes + s_bytes);
                #[cfg(all(feature = "hip", allpaca_hip_runtime))]
                {
                    // SAFETY: `q` is a live device buffer of `n_q*n_heads*head_dim`
                    // floats; `dk`/`dv`/`dout`/`ds` were allocated for `need` bytes.
                    let rc = unsafe {
                        ffi::allpaca_attention_sinks(
                            q.cast_mut().cast(),
                            dk.cast(),
                            dv.cast(),
                            dout.cast(),
                            n_q,
                            n_kv,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            i32::from(causal),
                            ds.cast(),
                            std::ptr::null_mut(),
                        )
                    };
                    status(rc, "attention_sinks")?;
                    memcpy(out.as_mut_ptr().cast(), dout, o_bytes, MEMCPY_D2H)?;
                    self.add_d2h(o_bytes);
                    Ok(())
                }
                #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
                {
                    let _ = (dout, ds, causal, n_q, n_kv, n_heads, n_kv_heads, head_dim);
                    Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "HIP attention_sinks unavailable".into(),
                    )))
                }
            };
        if need > 0 && need <= self.scratch_bytes {
            self.bind()?;
            let dk = self.scratch;
            // SAFETY: `need` fits in scratch; layout is k | v | out | sinks.
            let dv = unsafe { self.scratch.add(k_bytes) };
            let dout = unsafe { self.scratch.add(k_bytes + v_bytes) };
            let ds = unsafe { self.scratch.add(k_bytes + v_bytes + o_bytes) };
            return launch(dk, dv, dout, ds);
        }
        let blob = device_malloc(need.max(4), self.device)?;
        let result = (|| {
            self.bind()?;
            let dk = blob;
            // SAFETY: `device_malloc` returned `need` bytes; layout is k | v | out | sinks.
            let dv = unsafe { blob.add(k_bytes) };
            let dout = unsafe { blob.add(k_bytes + v_bytes) };
            let ds = unsafe { blob.add(k_bytes + v_bytes + o_bytes) };
            launch(dk, dv, dout, ds)
        })();
        device_free(blob, need.max(4));
        result
    }

    /// Device `q` and device `out`; H2D `k`/`v`/`sinks` only (no out D2H).
    pub fn attention_sinks_from_device_q_into(
        &self,
        q: *const f32,
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        dout: *mut f32,
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        if q.is_null() || dout.is_null() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device-Q/out attention".into(),
            )));
        }
        if sinks.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "device-Q attention requires sinks".into(),
            )));
        }
        let k_bytes = k.len() * 4;
        let v_bytes = v.len() * 4;
        let s_bytes = sinks.len() * 4;
        let need = k_bytes.saturating_add(v_bytes).saturating_add(s_bytes);
        let launch = |dk: *mut u8, dv: *mut u8, ds: *mut u8| -> Result<(), AllpacaError> {
            memcpy(dk, k.as_ptr().cast(), k_bytes, MEMCPY_H2D)?;
            memcpy(dv, v.as_ptr().cast(), v_bytes, MEMCPY_H2D)?;
            memcpy(ds, sinks.as_ptr().cast(), s_bytes, MEMCPY_H2D)?;
            self.add_h2d(k_bytes + v_bytes + s_bytes);
            #[cfg(all(feature = "hip", allpaca_hip_runtime))]
            {
                // SAFETY: `q` and `dout` are live device buffers; `dk`/`dv`/`ds`
                // were allocated for `need` bytes.
                let rc = unsafe {
                    ffi::allpaca_attention_sinks(
                        q.cast_mut().cast(),
                        dk.cast(),
                        dv.cast(),
                        dout.cast(),
                        n_q,
                        n_kv,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        i32::from(causal),
                        ds.cast(),
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "attention_sinks")
            }
            #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
            {
                let _ = (causal, n_q, n_kv, n_heads, n_kv_heads, head_dim);
                Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "HIP attention_sinks unavailable".into(),
                )))
            }
        };
        if need > 0 && need <= self.scratch_bytes {
            self.bind()?;
            let dk = self.scratch;
            // SAFETY: `need` fits in scratch; layout is k | v | sinks.
            let dv = unsafe { self.scratch.add(k_bytes) };
            let ds = unsafe { self.scratch.add(k_bytes + v_bytes) };
            return launch(dk, dv, ds);
        }
        let blob = device_malloc(need.max(4), self.device)?;
        let result = (|| {
            self.bind()?;
            let dk = blob;
            // SAFETY: `device_malloc` returned `need` bytes; layout is k | v | sinks.
            let dv = unsafe { blob.add(k_bytes) };
            let ds = unsafe { blob.add(k_bytes + v_bytes) };
            launch(dk, dv, ds)
        })();
        device_free(blob, need.max(4));
        result
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
                ffi::allpaca_rmsnorm(
                    dx.cast(),
                    wptr,
                    dout.cast(),
                    n as i32,
                    eps,
                    std::ptr::null_mut(),
                )
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

    /// Host-pointer batched RMSNorm: `out[rows, n] = RMSNorm(x[rows, n], weight[n])`.
    pub fn rmsnorm_rows(
        &self,
        x: &[f32],
        weight: &[f32],
        out: &mut [f32],
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        self.rmsnorm_rows_w(x, weight, None, out, rows, n, eps)
    }

    fn rmsnorm_rows_w(
        &self,
        x: &[f32],
        weight: &[f32],
        weight_dev: Option<*const f32>,
        out: &mut [f32],
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows < 0 || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rmsnorm_rows shape".into(),
            )));
        }
        let rows_n = (rows as usize).saturating_mul(n as usize);
        if x.len() < rows_n || out.len() < rows_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm_rows buffer size mismatch".into(),
            )));
        }
        if weight_dev.is_none() && weight.len() < n as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm_rows weight size mismatch".into(),
            )));
        }
        let x_bytes = rows_n.saturating_mul(4);
        let w_bytes = (n as usize).saturating_mul(4);
        let need = if weight_dev.is_some() {
            x_bytes.saturating_mul(2)
        } else {
            x_bytes.saturating_mul(2).saturating_add(w_bytes)
        };
        let mut launch = |dx: *mut u8, dw: *mut u8, dout: *mut u8| -> Result<(), AllpacaError> {
            memcpy(dx, x.as_ptr().cast(), x_bytes, MEMCPY_H2D)?;
            self.add_h2d(x_bytes);
            let wptr = if let Some(w) = weight_dev {
                w
            } else {
                memcpy(dw, weight.as_ptr().cast(), w_bytes, MEMCPY_H2D)?;
                self.add_h2d(w_bytes);
                dw.cast()
            };
            #[cfg(all(feature = "hip", allpaca_hip_runtime))]
            {
                let rc = unsafe {
                    ffi::allpaca_rmsnorm_rows(
                        dx.cast(),
                        wptr,
                        dout.cast(),
                        rows,
                        n,
                        eps,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "rmsnorm_rows")?;
                memcpy(out.as_mut_ptr().cast(), dout, x_bytes, MEMCPY_D2H)?;
                self.add_d2h(x_bytes);
                Ok(())
            }
            #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
            {
                let _ = (wptr, dout, eps);
                Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "HIP rmsnorm_rows unavailable".into(),
                )))
            }
        };
        if need > 0 && need <= self.scratch_bytes {
            self.bind()?;
            let dx = self.scratch;
            let dw = unsafe { self.scratch.add(x_bytes) };
            let dout = if weight_dev.is_some() {
                dw
            } else {
                unsafe { self.scratch.add(x_bytes.saturating_add(w_bytes)) }
            };
            return launch(dx, dw, dout);
        }
        let dx = device_malloc(x_bytes, self.device)?;
        let dout = match device_malloc(x_bytes, self.device) {
            Ok(p) => p,
            Err(e) => {
                device_free(dx, x_bytes);
                return Err(e);
            }
        };
        let dw = if weight_dev.is_some() {
            std::ptr::null_mut()
        } else {
            match device_malloc(w_bytes, self.device) {
                Ok(p) => p,
                Err(e) => {
                    device_free(dx, x_bytes);
                    device_free(dout, x_bytes);
                    return Err(e);
                }
            }
        };
        let result = (|| {
            self.bind()?;
            launch(dx, dw, dout)
        })();
        device_free(dx, x_bytes);
        device_free(dout, x_bytes);
        if weight_dev.is_none() {
            device_free(dw, w_bytes);
        }
        result
    }

    /// Host-pointer mean/var LayerNorm over `rows` of length `n` (ViT / CLIP; not RMSNorm).
    pub fn layernorm_rows(
        &self,
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        out: &mut [f32],
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if rows <= 0 || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid layernorm_rows shape".into(),
            )));
        }
        let rows_n = (rows as usize).saturating_mul(n as usize);
        if x.len() < rows_n
            || out.len() < rows_n
            || weight.len() < n as usize
            || bias.len() < n as usize
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "layernorm_rows buffer size mismatch".into(),
            )));
        }
        let x_bytes = rows_n * 4;
        let w_bytes = (n as usize) * 4;
        let need = x_bytes
            .saturating_mul(2)
            .saturating_add(w_bytes.saturating_mul(2));
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "layernorm exceeds scratch".into(),
            )));
        }
        self.bind()?;
        let dx = self.scratch;
        let dw = unsafe { self.scratch.add(x_bytes) };
        let db = unsafe { self.scratch.add(x_bytes + w_bytes) };
        let dout = unsafe { self.scratch.add(x_bytes + w_bytes * 2) };
        memcpy(dx, x.as_ptr().cast(), x_bytes, MEMCPY_H2D)?;
        memcpy(dw, weight.as_ptr().cast(), w_bytes, MEMCPY_H2D)?;
        memcpy(db, bias.as_ptr().cast(), w_bytes, MEMCPY_H2D)?;
        self.add_h2d(x_bytes + w_bytes * 2);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_layernorm_rows(
                    dx.cast(),
                    dw.cast(),
                    db.cast(),
                    dout.cast(),
                    rows,
                    n,
                    eps,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "layernorm_rows")?;
            memcpy(out.as_mut_ptr().cast(), dout, x_bytes, MEMCPY_D2H)?;
            self.add_d2h(x_bytes);
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (dx, dw, db, dout, eps);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP layernorm unavailable".into(),
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
            let rc = unsafe {
                ffi::allpaca_silu_mul(dg.cast(), du.cast(), n as i32, std::ptr::null_mut())
            };
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
    pub fn publish_logits(
        &self,
        device_logits: *const f32,
        host: &mut [f32],
    ) -> Result<(), AllpacaError> {
        self.start_logits_d2h(device_logits, host.len())?;
        self.wait_logits_d2h(host)
    }

    /// Queue logits D2H into pinned host memory (no stream wait).
    pub fn start_logits_d2h(
        &self,
        device_logits: *const f32,
        n: usize,
    ) -> Result<(), AllpacaError> {
        if device_logits.is_null() || n == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid logits D2H".into(),
            )));
        }
        self.bind()?;
        let bytes = n * 4;
        let pinned = self.ensure_pinned_logits(n)?;
        memcpy(pinned, device_logits.cast(), bytes, MEMCPY_D2H_ASYNC)?;
        self.logits_inflight.store(n as u64, Ordering::Relaxed);
        self.add_d2h(bytes);
        Ok(())
    }

    /// Wait for a queued logits D2H and copy pinned bytes into `host`.
    pub fn wait_logits_d2h(&self, host: &mut [f32]) -> Result<(), AllpacaError> {
        let n = self.logits_inflight.load(Ordering::Relaxed) as usize;
        if n == 0 {
            return Ok(());
        }
        self.bind()?;
        stream_synchronize()?;
        self.logits_inflight.store(0, Ordering::Relaxed);
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
                    ffi::allpaca_embed_gather_f16(
                        table,
                        token as i32,
                        out,
                        dim,
                        std::ptr::null_mut(),
                    )
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

    pub fn embed_gather_rocmfp4_rows(
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
                "invalid rocmfp4 embed gather".into(),
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
                    ffi::allpaca_embed_gather_rocmfp4_rows(
                        table,
                        self.scratch.cast(),
                        out,
                        tokens.len() as i32,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_rocmfp4_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_rocmfp4 unavailable".into(),
            )))
        }
    }

    /// Rows of a ROCmFP4 FAST (17-byte block) embedding table.
    pub fn embed_gather_rocmfp4_fast_rows(
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
                "invalid rocmfp4 embed gather".into(),
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
                    ffi::allpaca_embed_gather_rocmfp4_fast_rows(
                        table,
                        self.scratch.cast(),
                        out,
                        tokens.len() as i32,
                        dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "embed_gather_rocmfp4_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP embed_gather_rocmfp4 unavailable".into(),
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

    pub fn copy_d2d_f32(
        &self,
        dst: *mut f32,
        src: *const f32,
        n: usize,
    ) -> Result<(), AllpacaError> {
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

    /// Two Q8_0 GEMVs sharing one input row in one launch (a shared
    /// expert's gate and up; a GDN layer's qkv and z), `b1` with `n1` rows and
    /// `b2` with `n2`. `Ok(false)` = not covered (m != 1): run them
    /// separately. Each output is bit-identical to its own GEMV.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_nt_q8_pair_dev(
        &self,
        a: *const f32,
        b1: *const u8,
        c1: *mut f32,
        b2: *const u8,
        c2: *mut f32,
        m: i32,
        n1: i32,
        n2: i32,
        k: i32,
    ) -> Result<bool, AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.bind()?;
            let rc = unsafe {
                ffi::allpaca_gemm_nt_q8_0_pair(a, b1, c1, b2, c2, m, n1, n2, k, std::ptr::null_mut())
            };
            if rc == ALLPACA_ERR_INVALID {
                return Ok(false);
            }
            status(rc, "gemm_nt_q8_0_pair")?;
            Ok(true)
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (a, b1, c1, b2, c2, m, n1, n2, k);
            Ok(false)
        }
    }

    /// `x += y * s; y = x` over `n` floats in one launch: the bits of
    /// [`Self::scale_f32`] + [`Self::residual_add`] + a copy back.
    pub fn scale_add_copy(&self, x: *mut f32, y: *mut f32, s: f32, n: usize) -> Result<(), AllpacaError> {
        if n == 0 {
            return Ok(());
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe { ffi::allpaca_scale_add_copy(x, y, s, n as i32, std::ptr::null_mut()) };
            status(rc, "scale_add_copy")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, y, s);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP scale_add_copy unavailable".into(),
            )))
        }
    }

    pub fn scale_f32(&self, ptr: *mut f32, n: usize, scale: f32) -> Result<(), AllpacaError> {
        if n == 0 || ptr.is_null() {
            return Ok(());
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc =
                    unsafe { ffi::allpaca_scale_f32(ptr, n as i32, scale, std::ptr::null_mut()) };
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

    pub fn residual_add(
        &self,
        x: *mut f32,
        residual: *const f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
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

    /// `x[r][c] += bias[c]` on device rows, then tanh GELU when `gelu`.
    pub fn bias_act_rows_dev(
        &self,
        x: *mut f32,
        bias: *const f32,
        rows: i32,
        n: i32,
        gelu: bool,
    ) -> Result<(), AllpacaError> {
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_bias_act_rows(x, bias, rows, n, i32::from(gelu), std::ptr::null_mut())
            };
            status(rc, "bias_act_rows")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, bias, rows, n, gelu);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP bias_act_rows unavailable".into(),
            )))
        }
    }

    /// Qwen-VL vision 2D RoPE in place on device `[n_tok][n_heads * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_rope_rows_dev(
        &self,
        x: *mut f32,
        pos_row: *const i32,
        pos_col: *const i32,
        n_tok: i32,
        n_heads: i32,
        head_dim: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_vision_rope_rows(
                    x,
                    pos_row,
                    pos_col,
                    n_tok,
                    n_heads,
                    head_dim,
                    theta,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "vision_rope_rows")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, pos_row, pos_col, n_tok, n_heads, head_dim, theta);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP vision_rope_rows unavailable".into(),
            )))
        }
    }

    /// Mean/variance LayerNorm of device rows: `out = (x - mean) / sqrt(var + eps) * w + b`.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_rows_dev(
        &self,
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_layernorm_rows(x, weight, bias, out, rows, n, eps, std::ptr::null_mut())
            };
            status(rc, "layernorm_rows")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, weight, bias, out, rows, n, eps);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP layernorm_rows unavailable".into(),
            )))
        }
    }

    pub fn hc_mix_pre_device(
        &self,
        packed: *const f32,
        pre: *const f32,
        curs: *mut f32,
        n_tok: i32,
        hc: i32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if packed.is_null() || pre.is_null() || curs.is_null() || n_tok <= 0 || hc <= 0 || dim <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid hc_mix_pre shape".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_hc_mix_pre(packed, pre, curs, n_tok, hc, dim, std::ptr::null_mut())
            };
            status(rc, "hc_mix_pre")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (packed, pre, curs, n_tok, hc, dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP hc_mix_pre unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post_device(
        &self,
        x: *const f32,
        residual: *const f32,
        post: *const f32,
        comb: *const f32,
        out: *mut f32,
        n_tok: i32,
        hc: i32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if x.is_null()
            || residual.is_null()
            || post.is_null()
            || comb.is_null()
            || out.is_null()
            || n_tok <= 0
            || hc <= 0
            || dim <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid hc_post shape".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let rc = unsafe {
                ffi::allpaca_hc_post(
                    x,
                    residual,
                    post,
                    comb,
                    out,
                    n_tok,
                    hc,
                    dim,
                    std::ptr::null_mut(),
                )
            };
            status(rc, "hc_post")
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (x, residual, post, comb, out, n_tok, hc, dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP hc_post unavailable".into(),
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

    /// Qwen3.5 Gated DeltaNet one-token step on device (matches host `deltanet`).
    pub fn gated_delta_step(
        &self,
        q: *mut f32,
        k: *mut f32,
        v: *const f32,
        decay: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        n_v_heads: i32,
        head_v: i32,
        head_k: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if q.is_null()
            || k.is_null()
            || v.is_null()
            || decay.is_null()
            || beta.is_null()
            || state.is_null()
            || out.is_null()
            || n_v_heads <= 0
            || head_v <= 0
            || head_k <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid gated_delta_step".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_gated_delta_step(
                        q,
                        k,
                        v,
                        decay,
                        beta,
                        state,
                        out,
                        n_v_heads,
                        head_v,
                        head_k,
                        eps,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "gated_delta_step")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = eps;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP gated_delta_step unavailable".into(),
            )))
        }
    }

    /// Host-buffer GDN step: H2D into scratch, device kernel, D2H. Same math as
    /// [`Self::gated_delta_step`] / host `deltanet::gated_delta_step_eps`.
    pub fn gated_delta_step_host(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        state: &mut [f32],
        out: &mut [f32],
        n_v_heads: i32,
        head_v: i32,
        head_k: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        let n_v = n_v_heads as usize;
        let hv = head_v as usize;
        let hk = head_k as usize;
        if n_v == 0
            || hv == 0
            || hk == 0
            || q.len() != n_v * hk
            || k.len() != n_v * hk
            || v.len() != n_v * hv
            || decay.len() != n_v
            || beta.len() != n_v
            || state.len() != n_v * hv * hk
            || out.len() != n_v * hv
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gated_delta_step_host length mismatch".into(),
            )));
        }
        let need =
            (q.len() + k.len() + v.len() + decay.len() + beta.len() + state.len() + out.len()) * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "gated_delta_step_host need {need} scratch={}",
                    self.scratch_bytes
                ),
            )));
        }
        self.bind()?;
        let mut off = 0usize;
        let dq = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dq.cast(), q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D)?;
        off += q.len() * 4;
        let dk = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dk.cast(), k.as_ptr().cast(), k.len() * 4, MEMCPY_H2D)?;
        off += k.len() * 4;
        let dv = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dv.cast(), v.as_ptr().cast(), v.len() * 4, MEMCPY_H2D)?;
        off += v.len() * 4;
        let ddecay = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            ddecay.cast(),
            decay.as_ptr().cast(),
            decay.len() * 4,
            MEMCPY_H2D,
        )?;
        off += decay.len() * 4;
        let dbeta = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            dbeta.cast(),
            beta.as_ptr().cast(),
            beta.len() * 4,
            MEMCPY_H2D,
        )?;
        off += beta.len() * 4;
        let dstate = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            dstate.cast(),
            state.as_ptr().cast(),
            state.len() * 4,
            MEMCPY_H2D,
        )?;
        off += state.len() * 4;
        let dout = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dout.cast(), out.as_ptr().cast(), out.len() * 4, MEMCPY_H2D)?;
        self.add_h2d(need);
        self.gated_delta_step(
            dq, dk, dv, ddecay, dbeta, dstate, dout, n_v_heads, head_v, head_k, eps,
        )?;
        memcpy(q.as_mut_ptr().cast(), dq.cast(), q.len() * 4, MEMCPY_D2H)?;
        memcpy(k.as_mut_ptr().cast(), dk.cast(), k.len() * 4, MEMCPY_D2H)?;
        memcpy(
            state.as_mut_ptr().cast(),
            dstate.cast(),
            state.len() * 4,
            MEMCPY_D2H,
        )?;
        memcpy(
            out.as_mut_ptr().cast(),
            dout.cast(),
            out.len() * 4,
            MEMCPY_D2H,
        )?;
        self.add_d2h((q.len() + k.len() + state.len() + out.len()) * 4);
        Ok(())
    }

    /// True when this request already has seeded device GDN slots (source of truth).
    pub fn gdn_request_resident(&self, request_id: u64) -> bool {
        self.gdn
            .lock()
            .ok()
            .map(|g| {
                g.iter()
                    .any(|((rid, _), slot)| *rid == request_id && slot.seeded)
            })
            .unwrap_or(false)
    }

    /// Drop all resident GDN state for a finished / restored request.
    pub fn gdn_clear_request(&self, request_id: u64) {
        if let Ok(mut g) = self.gdn.lock() {
            g.retain(|&(rid, _), _| rid != request_id);
        }
        self.gdn_drop_clone(request_id);
        if let Ok(mut g) = self.gdn_snap.lock() {
            g.retain(|&(rid, _), _| rid != request_id);
        }
        if let Ok(mut j) = self.gdn_journal.lock() {
            j.retain(|&(rid, _), _| rid != request_id);
        }
    }

    fn gdn_clone_is_live(&self, request_id: u64) -> bool {
        self.gdn_clone_live
            .lock()
            .is_ok_and(|l| l.contains(&request_id))
    }

    fn gdn_journal_reset(&self, request_id: u64) {
        if let Ok(mut j) = self.gdn_journal.lock() {
            for ((rid, _), journal) in j.iter_mut() {
                if *rid == request_id {
                    journal.rows = 0;
                    journal.overflow = false;
                }
            }
        }
    }

    /// Append one step's inputs while `key` has a clone (a verify is in flight).
    #[allow(clippy::too_many_arguments)]
    fn gdn_journal_record(
        &self,
        key: (u64, usize),
        qkv: *const f32,
        alpha: *const f32,
        beta: *const f32,
        conv_w: *const f32,
        dt_bias: *const f32,
        a: *const f32,
        dims: [usize; 7],
        eps: f32,
    ) -> Result<(), AllpacaError> {
        let cloned = self.gdn_clone_is_live(key.0)
            && self
                .gdn_snap
                .lock()
                .map_err(|_| Self::gdn_mutex_err())?
                .contains_key(&key);
        if !cloned {
            return Ok(());
        }
        let [qkv_dim, n_v, n_k, head_v, head_k, conv_kernel, interleave] = dims;
        let row_floats = qkv_dim + 2 * n_v;
        let mut j = self.gdn_journal.lock().map_err(|_| Self::gdn_mutex_err())?;
        let fits = j
            .get(&key)
            .is_some_and(|jr| jr.qkv_dim == qkv_dim && jr.n_v == n_v);
        if !fits {
            j.insert(
                key,
                GdnJournal {
                    rows_buf: DeviceWeights::alloc(
                        self.device,
                        GDN_JOURNAL_ROWS * row_floats * 4,
                    )?,
                    cap: GDN_JOURNAL_ROWS,
                    rows: 0,
                    overflow: false,
                    qkv_dim,
                    n_v,
                    n_k,
                    head_v,
                    head_k,
                    conv_kernel,
                    eps,
                    k_interleave: interleave != 0,
                    conv_w,
                    dt_bias,
                    a,
                },
            );
        }
        let jr = j.get_mut(&key).expect("journal present");
        if jr.rows >= jr.cap {
            jr.overflow = true;
            return Ok(());
        }
        let row = unsafe { (jr.rows_buf.ptr_at(0) as *mut f32).add(jr.rows * row_floats) };
        self.copy_d2d_f32(row, qkv, qkv_dim)?;
        self.copy_d2d_f32(unsafe { row.add(qkv_dim) }, alpha, n_v)?;
        self.copy_d2d_f32(unsafe { row.add(qkv_dim + n_v) }, beta, n_v)?;
        jr.rows += 1;
        Ok(())
    }

    /// Every cloned GDN slot of `request_id` can be rolled forward `keep`
    /// steps from its journal (no Mamba-2 slots, no overflowed journal).
    pub fn gdn_can_replay(&self, request_id: u64, keep: usize) -> bool {
        if !self.gdn_clone_is_live(request_id) {
            return true;
        }
        let (Ok(snap), Ok(j)) = (self.gdn_snap.lock(), self.gdn_journal.lock()) else {
            return false;
        };
        snap.iter()
            .filter(|((rid, _), _)| *rid == request_id)
            .all(|(key, slot)| {
                !slot.ssm
                    && match j.get(key) {
                        Some(jr) => !jr.overflow && jr.rows >= keep,
                        None => keep == 0,
                    }
            })
    }

    /// Restore the clone, then replay the first `keep` journaled steps of every
    /// GDN layer: recurrent kernels only, no projection GEMMs.
    pub fn gdn_rewind_request_to(&self, request_id: u64, keep: usize) -> Result<(), AllpacaError> {
        self.gdn_rewind_request(request_id)?;
        // (slot, journal rows, dims, eps, [conv_w, dt_bias, a])
        type Replay = ((u64, usize), *const f32, [usize; 7], f32, [*const f32; 3]);
        let journals: Vec<Replay> = {
            let j = self.gdn_journal.lock().map_err(|_| Self::gdn_mutex_err())?;
            j.iter()
                .filter(|((rid, _), jr)| *rid == request_id && jr.rows >= keep && keep > 0)
                .map(|(key, jr)| {
                    (
                        *key,
                        jr.rows_buf.ptr_at(0),
                        [
                            jr.qkv_dim,
                            jr.n_v,
                            jr.n_k,
                            jr.head_v,
                            jr.head_k,
                            jr.conv_kernel,
                            usize::from(jr.k_interleave),
                        ],
                        jr.eps,
                        [jr.conv_w, jr.dt_bias, jr.a],
                    )
                })
                .collect()
        };
        for (key, rows, dims, eps, [conv_w, dt_bias, a]) in journals {
            let [qkv_dim, n_v, n_k, head_v, head_k, conv_kernel, interleave] = dims;
            let (dstate, dconv) = {
                let g = self.gdn.lock().map_err(|_| Self::gdn_mutex_err())?;
                let slot = g.get(&key).ok_or_else(|| {
                    AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "gdn replay missing live slot".into(),
                    ))
                })?;
                (
                    slot.state.ptr_at(0) as *mut f32,
                    slot.conv
                        .as_ref()
                        .map(|c| c.ptr_at(0) as *mut f32)
                        .unwrap_or(std::ptr::null_mut()),
                )
            };
            let value_dim = n_v * head_v;
            let q_exp_n = n_v * head_k;
            let layer_scratch = qkv_dim + 2 * q_exp_n + n_v;
            if (layer_scratch + value_dim + n_v) * 4 > self.scratch_bytes {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "gdn replay scratch too small".into(),
                )));
            }
            let y = unsafe { self.scratch.add(layer_scratch * 4).cast::<f32>() };
            let beta = unsafe { self.scratch.add((layer_scratch + value_dim) * 4).cast::<f32>() };
            let row_floats = qkv_dim + 2 * n_v;
            for t in 0..keep {
                let row = unsafe { rows.add(t * row_floats) };
                // The step sigmoids beta in place; replay from a scratch copy.
                self.copy_d2d_f32(beta, unsafe { row.add(qkv_dim + n_v) }, n_v)?;
                self.gdn_layer_step(
                    row,
                    unsafe { row.add(qkv_dim) },
                    beta,
                    conv_w,
                    dconv,
                    dt_bias,
                    a,
                    dstate,
                    y,
                    n_v as i32,
                    n_k as i32,
                    head_v as i32,
                    head_k as i32,
                    conv_kernel as i32,
                    eps,
                    interleave != 0,
                )?;
            }
        }
        self.gdn_journal_reset(request_id);
        Ok(())
    }

    /// Snapshot device GDN state+conv for `request_id` (D2D). Returns layers cloned.
    pub fn gdn_clone_request(&self, request_id: u64) -> Result<usize, AllpacaError> {
        let layers = self.gdn_layer_sizes(request_id, false)?;
        if layers.is_empty() {
            return Ok(0);
        }
        self.gdn_ensure_slots(request_id, &layers, true)?;
        self.gdn_copy_slots(request_id, &layers, false)?;
        self.gdn_journal_reset(request_id);
        self.gdn_clone_live
            .lock()
            .map_err(|_| Self::gdn_mutex_err())?
            .insert(request_id);
        Ok(layers.len())
    }

    /// Restore live device GDN slots from the last [`Self::gdn_clone_request`].
    pub fn gdn_rewind_request(&self, request_id: u64) -> Result<(), AllpacaError> {
        if !self.gdn_clone_is_live(request_id) {
            return Ok(());
        }
        let layers = self.gdn_layer_sizes(request_id, true)?;
        if layers.is_empty() {
            return Ok(());
        }
        self.gdn_ensure_slots(request_id, &layers, false)?;
        self.gdn_copy_slots(request_id, &layers, true)?;
        let mut g = self.gdn.lock().map_err(|_| {
            AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                "gdn mutex poisoned".into(),
            ))
        })?;
        for &(layer, ..) in &layers {
            if let Some(slot) = g.get_mut(&(request_id, layer)) {
                slot.seeded = true;
                slot.dirty = true;
            }
        }
        Ok(())
    }

    fn gdn_mutex_err() -> AllpacaError {
        AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
            "gdn mutex poisoned".into(),
        ))
    }

    fn gdn_layer_sizes(
        &self,
        request_id: u64,
        from_snap: bool,
    ) -> Result<Vec<(usize, usize, usize, bool, bool)>, AllpacaError> {
        let g = if from_snap {
            self.gdn_snap.lock().map_err(|_| Self::gdn_mutex_err())?
        } else {
            self.gdn.lock().map_err(|_| Self::gdn_mutex_err())?
        };
        Ok(g.iter()
            .filter(|((rid, _), _)| *rid == request_id)
            .map(|((_, layer), slot)| {
                (
                    *layer,
                    slot.state_n,
                    slot.conv_n,
                    slot.conv.is_some(),
                    slot.ssm,
                )
            })
            .collect())
    }

    fn gdn_ensure_slots(
        &self,
        request_id: u64,
        layers: &[(usize, usize, usize, bool, bool)],
        into_snap: bool,
    ) -> Result<(), AllpacaError> {
        let mut g = if into_snap {
            self.gdn_snap.lock().map_err(|_| Self::gdn_mutex_err())?
        } else {
            self.gdn.lock().map_err(|_| Self::gdn_mutex_err())?
        };
        for &(layer, state_n, conv_n, has_conv, is_ssm) in layers {
            let key = (request_id, layer);
            let reuse = g.get(&key).is_some_and(|s| {
                s.state_n == state_n && s.conv_n == conv_n && s.conv.is_some() == has_conv
            });
            if reuse {
                continue;
            }
            let state = DeviceWeights::alloc(self.device, state_n.max(1) * 4)?;
            let conv = if has_conv && conv_n > 0 {
                Some(DeviceWeights::alloc(self.device, conv_n * 4)?)
            } else {
                None
            };
            g.insert(
                key,
                GdnLayerResident {
                    state,
                    state_n,
                    conv,
                    conv_n,
                    seeded: true,
                    dirty: false,
                    ssm: is_ssm,
                },
            );
        }
        Ok(())
    }

    fn gdn_copy_slots(
        &self,
        request_id: u64,
        layers: &[(usize, usize, usize, bool, bool)],
        snap_to_live: bool,
    ) -> Result<(), AllpacaError> {
        self.bind()?;
        for &(layer, state_n, conv_n, _, _) in layers {
            let (src_s, src_c, dst_s, dst_c) = {
                let live = self.gdn.lock().map_err(|_| Self::gdn_mutex_err())?;
                let snap = self.gdn_snap.lock().map_err(|_| Self::gdn_mutex_err())?;
                let (src, dst) = if snap_to_live {
                    (
                        snap.get(&(request_id, layer)),
                        live.get(&(request_id, layer)),
                    )
                } else {
                    (
                        live.get(&(request_id, layer)),
                        snap.get(&(request_id, layer)),
                    )
                };
                let src = src.ok_or_else(|| {
                    AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "gdn copy missing source slot".into(),
                    ))
                })?;
                let dst = dst.ok_or_else(|| {
                    AllpacaError::Degraded(DegradedError::FeatureDisabled(
                        "gdn copy missing dest slot".into(),
                    ))
                })?;
                (
                    src.state.ptr_at(0) as *const f32,
                    src.conv.as_ref().map(|c| c.ptr_at(0) as *const f32),
                    dst.state.ptr_at(0) as *mut f32,
                    dst.conv.as_ref().map(|c| c.ptr_at(0) as *mut f32),
                )
            };
            self.copy_d2d_f32(dst_s, src_s, state_n)?;
            if conv_n > 0 {
                if let (Some(d), Some(s)) = (dst_c, src_c) {
                    self.copy_d2d_f32(d, s, conv_n)?;
                }
            }
        }
        // D2D is queued async; the next GDN kernel must see the restored slots.
        stream_synchronize()?;
        Ok(())
    }

    /// End `request_id`'s clone. The snapshot slots stay allocated for the
    /// next verify; [`Self::gdn_clear_request`] frees them.
    pub fn gdn_drop_clone(&self, request_id: u64) {
        if let Ok(mut l) = self.gdn_clone_live.lock() {
            l.remove(&request_id);
        }
        self.gdn_journal_reset(request_id);
    }

    /// Force re-seed from host on next step (after recurrent restore).
    pub fn gdn_invalidate_request(&self, request_id: u64) {
        if let Ok(mut g) = self.gdn.lock() {
            for ((rid, _), layer) in g.iter_mut() {
                if *rid == request_id {
                    layer.seeded = false;
                    layer.dirty = false;
                }
            }
        }
    }

    /// Pull dirty per-layer GDN state into host `states[layer]` (anchor/radix capture).
    pub fn gdn_sync_request_to_host(
        &self,
        request_id: u64,
        states: &mut [Vec<f32>],
    ) -> Result<(), AllpacaError> {
        let mut g = self.gdn.lock().map_err(|_| {
            AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                "gdn mutex poisoned".into(),
            ))
        })?;
        self.bind()?;
        for ((rid, layer), slot) in g.iter_mut() {
            if *rid != request_id || slot.ssm || !slot.dirty {
                continue;
            }
            if *layer >= states.len() {
                continue;
            }
            let host = &mut states[*layer];
            if host.len() != slot.state_n {
                *host = vec![0.0; slot.state_n];
            }
            memcpy(
                host.as_mut_ptr().cast(),
                slot.state.ptr_at(0).cast(),
                slot.state_n * 4,
                MEMCPY_D2H,
            )?;
            self.add_d2h(slot.state_n * 4);
            if let Some(conv) = slot.conv.as_ref() {
                if slot.conv_n > 0 {
                    // Host conv is restored on the next seed after invalidate; keep it current
                    // for packed-verify rewind / radix capture.
                    let _ = conv;
                }
            }
            slot.dirty = false;
        }
        Ok(())
    }

    /// Pull dirty per-layer GDN conv history into host `convs[layer]`.
    pub fn gdn_sync_conv_to_host(
        &self,
        request_id: u64,
        convs: &mut [Vec<f32>],
    ) -> Result<(), AllpacaError> {
        let mut g = self.gdn.lock().map_err(|_| {
            AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                "gdn mutex poisoned".into(),
            ))
        })?;
        self.bind()?;
        for ((rid, layer), slot) in g.iter_mut() {
            if *rid != request_id || slot.ssm {
                continue;
            }
            let Some(conv) = slot.conv.as_ref() else {
                continue;
            };
            if *layer >= convs.len() || slot.conv_n == 0 {
                continue;
            }
            let host = &mut convs[*layer];
            if host.len() != slot.conv_n {
                *host = vec![0.0; slot.conv_n];
            }
            memcpy(
                host.as_mut_ptr().cast(),
                conv.ptr_at(0).cast(),
                slot.conv_n * 4,
                MEMCPY_D2H,
            )?;
            self.add_d2h(slot.conv_n * 4);
        }
        Ok(())
    }

    pub fn mamba2_sync_request_to_host(
        &self,
        request_id: u64,
        states: &mut [Vec<f32>],
        convs: &mut [Vec<f32>],
    ) -> Result<(), AllpacaError> {
        let mut g = self.gdn.lock().map_err(|_| Self::gdn_mutex_err())?;
        self.bind()?;
        for ((rid, layer), slot) in g.iter_mut() {
            if *rid != request_id || !slot.ssm || !slot.dirty {
                continue;
            }
            if *layer < states.len() {
                let host = &mut states[*layer];
                if host.len() != slot.state_n {
                    *host = vec![0.0; slot.state_n];
                }
                memcpy(
                    host.as_mut_ptr().cast(),
                    slot.state.ptr_at(0).cast(),
                    slot.state_n * 4,
                    MEMCPY_D2H,
                )?;
                self.add_d2h(slot.state_n * 4);
            }
            if let Some(conv) = slot.conv.as_ref() {
                if *layer < convs.len() && slot.conv_n > 0 {
                    let host = &mut convs[*layer];
                    if host.len() != slot.conv_n {
                        *host = vec![0.0; slot.conv_n];
                    }
                    memcpy(
                        host.as_mut_ptr().cast(),
                        conv.ptr_at(0).cast(),
                        slot.conv_n * 4,
                        MEMCPY_D2H,
                    )?;
                    self.add_d2h(slot.conv_n * 4);
                }
            }
            slot.dirty = false;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_mamba2_weights(
        &self,
        layer: usize,
        conv: &[f32],
        bias: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        d: &[f32],
        norm: &[f32],
    ) -> Result<
        (
            *const f32,
            *const f32,
            *const f32,
            *const f32,
            *const f32,
            *const f32,
        ),
        AllpacaError,
    > {
        if conv.is_empty() || dt_bias.is_empty() || a.is_empty() || d.is_empty() || norm.is_empty()
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty Mamba-2 extras".into(),
            )));
        }
        self.bind()?;
        {
            let g = self.mamba2_w.lock().map_err(|_| Self::gdn_mutex_err())?;
            if let Some(w) = g.get(&layer) {
                return Ok((
                    w.conv.ptr_at(0),
                    w.bias
                        .as_ref()
                        .map(|v| v.ptr_at(0))
                        .unwrap_or(std::ptr::null()),
                    w.dt.ptr_at(0),
                    w.a.ptr_at(0),
                    w.d.ptr_at(0),
                    w.norm.ptr_at(0),
                ));
            }
        }
        let upload = |values: &[f32]| -> Result<DeviceWeights, AllpacaError> {
            let allocation = DeviceWeights::alloc(self.device, values.len().max(1) * 4)?;
            self.copy_h2d_f32(allocation.ptr_at(0) as *mut f32, values)?;
            Ok(allocation)
        };
        let cw = upload(conv)?;
        let bb = if bias.is_empty() {
            None
        } else {
            Some(upload(bias)?)
        };
        let dt = upload(dt_bias)?;
        let aa = upload(a)?;
        let dd = upload(d)?;
        let nn = upload(norm)?;
        let ptrs = (
            cw.ptr_at(0),
            bb.as_ref().map(|v| v.ptr_at(0)).unwrap_or(std::ptr::null()),
            dt.ptr_at(0),
            aa.ptr_at(0),
            dd.ptr_at(0),
            nn.ptr_at(0),
        );
        self.mamba2_w
            .lock()
            .map_err(|_| Self::gdn_mutex_err())?
            .insert(
                layer,
                Mamba2WeightResident {
                    conv: cw,
                    bias: bb,
                    dt,
                    a: aa,
                    d: dd,
                    norm: nn,
                },
            );
        Ok(ptrs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_decode_step_resident(
        &self,
        request_id: u64,
        layer: usize,
        host_state: &mut [f32],
        host_conv: &mut [f32],
        projected: *const f32,
        conv_w: *const f32,
        conv_bias: *const f32,
        dt_bias: *const f32,
        a: *const f32,
        d: *const f32,
        norm: *const f32,
        out: *mut f32,
        d_inner: i32,
        d_state: i32,
        n_heads: i32,
        n_groups: i32,
        conv_kernel: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if projected.is_null()
            || conv_w.is_null()
            || dt_bias.is_null()
            || a.is_null()
            || d.is_null()
            || norm.is_null()
            || out.is_null()
            || d_inner <= 0
            || d_state <= 0
            || n_heads <= 0
            || n_groups <= 0
            || conv_kernel <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid mamba2_decode_step_resident".into(),
            )));
        }
        let inner = d_inner as usize;
        let state_size = d_state as usize;
        let heads = n_heads as usize;
        let groups = n_groups as usize;
        let conv_dim = inner + 2 * groups * state_size;
        if host_state.len() != inner * state_size
            || host_conv.len() != (conv_kernel as usize - 1) * conv_dim
            || inner % heads != 0
            || heads % groups != 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Mamba-2 state/layout mismatch".into(),
            )));
        }
        self.bind()?;
        let key = (request_id, layer);
        let (device_state, device_conv, seed) = {
            let mut slots = self.gdn.lock().map_err(|_| Self::gdn_mutex_err())?;
            if slots.get(&key).is_none_or(|slot| {
                !slot.ssm || slot.state_n != host_state.len() || slot.conv_n != host_conv.len()
            }) {
                slots.insert(
                    key,
                    GdnLayerResident {
                        state: DeviceWeights::alloc(self.device, host_state.len().max(1) * 4)?,
                        state_n: host_state.len(),
                        conv: Some(DeviceWeights::alloc(
                            self.device,
                            host_conv.len().max(1) * 4,
                        )?),
                        conv_n: host_conv.len(),
                        seeded: false,
                        dirty: false,
                        ssm: true,
                    },
                );
            }
            let slot = slots.get_mut(&key).expect("Mamba-2 slot present");
            (
                slot.state.ptr_at(0) as *mut f32,
                slot.conv.as_ref().expect("Mamba-2 conv present").ptr_at(0) as *mut f32,
                !slot.seeded,
            )
        };
        if seed {
            self.copy_h2d_f32(device_state, host_state)?;
            self.copy_h2d_f32(device_conv, host_conv)?;
        }
        let need = conv_dim * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "Mamba-2 scratch need={need} available={}",
                    self.scratch_bytes
                ),
            )));
        }
        let xbc_raw = unsafe { projected.add(inner) };
        let dt = unsafe { projected.add(inner + conv_dim) };
        let xbc = self.scratch.cast::<f32>();
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                status(
                    unsafe {
                        ffi::allpaca_mamba2_conv_step(
                            xbc_raw,
                            conv_w,
                            conv_bias,
                            device_conv,
                            xbc,
                            conv_dim as i32,
                            conv_kernel,
                            std::ptr::null_mut(),
                        )
                    },
                    "mamba2_conv_step",
                )
            })?;
            self.timed_gpu(DecodeOp::OtherGpu, || {
                status(
                    unsafe {
                        ffi::allpaca_mamba2_ssd_step(
                            projected,
                            xbc,
                            dt,
                            a,
                            d,
                            dt_bias,
                            norm,
                            device_state,
                            out,
                            d_inner,
                            d_state,
                            n_heads,
                            n_groups,
                            eps,
                            std::ptr::null_mut(),
                        )
                    },
                    "mamba2_ssd_step",
                )
            })?;
            if let Ok(mut slots) = self.gdn.lock() {
                if let Some(slot) = slots.get_mut(&key) {
                    slot.seeded = true;
                    slot.dirty = true;
                }
            }
            Ok(())
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (conv_bias, eps, xbc_raw, dt, xbc);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP Mamba-2 unavailable".into(),
            )))
        }
    }

    /// Upload GDN extras once per layer on this workspace (shared across requests).
    pub fn ensure_gdn_weights(
        &self,
        layer: usize,
        conv: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        norm: &[f32],
    ) -> Result<(*const f32, *const f32, *const f32, *const f32), AllpacaError> {
        if conv.is_empty() || dt_bias.is_empty() || a.is_empty() || norm.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty GDN extras".into(),
            )));
        }
        self.bind()?;
        {
            let g = self.gdn_w.lock().map_err(|_| {
                AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                    "gdn weight mutex poisoned".into(),
                ))
            })?;
            if let Some(w) = g.get(&layer) {
                return Ok((
                    w.conv.ptr_at(0),
                    w.dt.ptr_at(0),
                    w.a.ptr_at(0),
                    w.norm.ptr_at(0),
                ));
            }
        }
        let cw = DeviceWeights::alloc(self.device, conv.len() * 4)?;
        self.copy_h2d_f32(cw.ptr_at(0) as *mut f32, conv)?;
        let dt = DeviceWeights::alloc(self.device, dt_bias.len() * 4)?;
        self.copy_h2d_f32(dt.ptr_at(0) as *mut f32, dt_bias)?;
        let aa = DeviceWeights::alloc(self.device, a.len() * 4)?;
        self.copy_h2d_f32(aa.ptr_at(0) as *mut f32, a)?;
        let nn = DeviceWeights::alloc(self.device, norm.len() * 4)?;
        self.copy_h2d_f32(nn.ptr_at(0) as *mut f32, norm)?;
        let ptrs = (cw.ptr_at(0), dt.ptr_at(0), aa.ptr_at(0), nn.ptr_at(0));
        let mut g = self.gdn_w.lock().map_err(|_| {
            AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                "gdn weight mutex poisoned".into(),
            ))
        })?;
        g.insert(
            layer,
            GdnWeightResident {
                conv: cw,
                dt,
                a: aa,
                norm: nn,
            },
        );
        Ok(ptrs)
    }

    /// GDN step with request-local device-resident state (no per-token state H2D/D2H).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_step_resident(
        &self,
        request_id: u64,
        layer: usize,
        host_state: &mut [f32],
        q: &mut [f32],
        k: &mut [f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        out: &mut [f32],
        n_v_heads: i32,
        head_v: i32,
        head_k: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        let n_v = n_v_heads as usize;
        let hv = head_v as usize;
        let hk = head_k as usize;
        if n_v == 0
            || hv == 0
            || hk == 0
            || q.len() != n_v * hk
            || k.len() != n_v * hk
            || v.len() != n_v * hv
            || decay.len() != n_v
            || beta.len() != n_v
            || host_state.len() != n_v * hv * hk
            || out.len() != n_v * hv
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gated_delta_step_resident length mismatch".into(),
            )));
        }
        let need = (q.len() + k.len() + v.len() + decay.len() + beta.len() + out.len()) * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "gated_delta_step_resident need {need} scratch={}",
                    self.scratch_bytes
                ),
            )));
        }
        self.bind()?;
        let key = (request_id, layer);
        let (dstate, need_seed) = {
            let mut g = self.gdn.lock().map_err(|_| {
                AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                    "gdn mutex poisoned".into(),
                ))
            })?;
            if g.get(&key).is_none_or(|s| s.state_n != host_state.len()) {
                let blob = DeviceWeights::alloc(self.device, host_state.len() * 4)?;
                g.insert(
                    key,
                    GdnLayerResident {
                        state: blob,
                        state_n: host_state.len(),
                        conv: None,
                        conv_n: 0,
                        seeded: false,
                        dirty: false,
                        ssm: false,
                    },
                );
            }
            let slot = g.get_mut(&key).expect("gdn layer present");
            (slot.state.ptr_at(0) as *mut f32, !slot.seeded)
        };
        if need_seed {
            self.copy_h2d_f32(dstate, host_state)?;
            if let Ok(mut g) = self.gdn.lock() {
                if let Some(slot) = g.get_mut(&key) {
                    slot.seeded = true;
                }
            }
        }

        let mut off = 0usize;
        let dq = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dq.cast(), q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D)?;
        off += q.len() * 4;
        let dk = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dk.cast(), k.as_ptr().cast(), k.len() * 4, MEMCPY_H2D)?;
        off += k.len() * 4;
        let dv = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dv.cast(), v.as_ptr().cast(), v.len() * 4, MEMCPY_H2D)?;
        off += v.len() * 4;
        let ddecay = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            ddecay.cast(),
            decay.as_ptr().cast(),
            decay.len() * 4,
            MEMCPY_H2D,
        )?;
        off += decay.len() * 4;
        let dbeta = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            dbeta.cast(),
            beta.as_ptr().cast(),
            beta.len() * 4,
            MEMCPY_H2D,
        )?;
        off += beta.len() * 4;
        let dout = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dout.cast(), out.as_ptr().cast(), out.len() * 4, MEMCPY_H2D)?;
        self.add_h2d(need);

        self.gated_delta_step(
            dq, dk, dv, ddecay, dbeta, dstate, dout, n_v_heads, head_v, head_k, eps,
        )?;

        memcpy(q.as_mut_ptr().cast(), dq.cast(), q.len() * 4, MEMCPY_D2H)?;
        memcpy(k.as_mut_ptr().cast(), dk.cast(), k.len() * 4, MEMCPY_D2H)?;
        memcpy(
            out.as_mut_ptr().cast(),
            dout.cast(),
            out.len() * 4,
            MEMCPY_D2H,
        )?;
        self.add_d2h((q.len() + k.len() + out.len()) * 4);

        if let Ok(mut g) = self.gdn.lock() {
            if let Some(slot) = g.get_mut(&(request_id, layer)) {
                slot.dirty = true;
            }
        }
        Ok(())
    }

    /// Full one-token GDN decode on device activations (conv→expand→decay→delta→silu/RMSNorm).
    /// `qkv` / `z` / `α` / `β` stay on device. Recurrent state + conv history are request-local.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_decode_step_resident(
        &self,
        request_id: u64,
        layer: usize,
        host_state: &mut [f32],
        host_conv: &mut [f32],
        qkv: *const f32,
        z: *const f32,
        alpha: *const f32,
        beta: *mut f32,
        conv_w: *const f32,
        dt_bias: *const f32,
        a: *const f32,
        norm: *const f32,
        out: *mut f32,
        n_v_heads: i32,
        n_k_heads: i32,
        head_v: i32,
        head_k: i32,
        conv_kernel: i32,
        eps: f32,
        k_interleave: bool,
    ) -> Result<(), AllpacaError> {
        if qkv.is_null()
            || z.is_null()
            || alpha.is_null()
            || beta.is_null()
            || conv_w.is_null()
            || dt_bias.is_null()
            || a.is_null()
            || norm.is_null()
            || out.is_null()
            || n_v_heads <= 0
            || n_k_heads <= 0
            || head_v <= 0
            || head_k <= 0
            || conv_kernel <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid gdn_decode_step_resident".into(),
            )));
        }
        let n_v = n_v_heads as usize;
        let hv = head_v as usize;
        let hk = head_k as usize;
        let expect_state = n_v * hv * hk;
        if host_state.len() != expect_state {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gdn_decode_step_resident state length mismatch".into(),
            )));
        }
        self.bind()?;
        let key = (request_id, layer);
        let (dstate, dconv, need_seed_state, need_seed_conv) = {
            let mut g = self.gdn.lock().map_err(|_| {
                AllpacaError::Fatal(allpaca_types::FatalError::InvalidConfig(
                    "gdn mutex poisoned".into(),
                ))
            })?;
            if g.get(&key)
                .is_none_or(|s| s.state_n != host_state.len() || s.conv_n != host_conv.len())
            {
                let state = DeviceWeights::alloc(self.device, host_state.len().max(1) * 4)?;
                let conv = if host_conv.is_empty() {
                    None
                } else {
                    Some(DeviceWeights::alloc(self.device, host_conv.len() * 4)?)
                };
                g.insert(
                    key,
                    GdnLayerResident {
                        state,
                        state_n: host_state.len(),
                        conv,
                        conv_n: host_conv.len(),
                        seeded: false,
                        dirty: false,
                        ssm: false,
                    },
                );
            } else if let Some(slot) = g.get_mut(&key) {
                if slot.conv.is_none() && !host_conv.is_empty() {
                    slot.conv = Some(DeviceWeights::alloc(self.device, host_conv.len() * 4)?);
                    slot.conv_n = host_conv.len();
                }
            }
            let slot = g.get_mut(&key).expect("gdn layer present");
            let dstate = slot.state.ptr_at(0) as *mut f32;
            let dconv = slot
                .conv
                .as_ref()
                .map(|c| c.ptr_at(0) as *mut f32)
                .unwrap_or(std::ptr::null_mut());
            let need_conv = slot.conv.is_some() && (!slot.seeded || slot.conv_n == host_conv.len());
            (dstate, dconv, !slot.seeded, need_conv && !slot.seeded)
        };
        if need_seed_state {
            self.copy_h2d_f32(dstate, host_state)?;
        }
        if need_seed_conv && !host_conv.is_empty() && !dconv.is_null() {
            self.copy_h2d_f32(dconv, host_conv)?;
        }
        if need_seed_state || need_seed_conv {
            if let Ok(mut g) = self.gdn.lock() {
                if let Some(slot) = g.get_mut(&key) {
                    slot.seeded = true;
                }
            }
        }
        if dconv.is_null() && !host_conv.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gdn_decode_step_resident missing conv state".into(),
            )));
        }
        let n_k = n_k_heads as usize;
        let key_dim = n_k * hk;
        let value_dim = n_v * hv;
        let qkv_dim = key_dim.saturating_mul(2).saturating_add(value_dim);
        let q_exp_n = n_v * hk;
        let layer_scratch = qkv_dim + q_exp_n + q_exp_n + n_v;
        let need = (layer_scratch + value_dim) * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "gdn_decode_step_resident need {need} scratch={}",
                    self.scratch_bytes
                ),
            )));
        }
        let y = unsafe { self.scratch.add(layer_scratch * 4).cast::<f32>() };
        self.gdn_journal_record(
            key,
            qkv,
            alpha,
            beta,
            conv_w,
            dt_bias,
            a,
            [qkv_dim, n_v, n_k, hv, hk, conv_kernel as usize, usize::from(k_interleave)],
            eps,
        )?;
        self.gdn_layer_step(
            qkv,
            alpha,
            beta,
            conv_w,
            dconv,
            dt_bias,
            a,
            dstate,
            y,
            n_v_heads,
            n_k_heads,
            head_v,
            head_k,
            conv_kernel,
            eps,
            k_interleave,
        )?;
        self.gdn_rmsnorm_silu_gate(y, z, norm, out, n_v_heads, head_v, eps)?;
        if let Ok(mut g) = self.gdn.lock() {
            if let Some(slot) = g.get_mut(&key) {
                slot.dirty = true;
            }
        }
        Ok(())
    }

    /// Depthwise causal GDN conv1d + SiLU on device pointers (matches host `deltanet::conv1d_step`).
    pub fn gdn_conv1d_step(
        &self,
        x: *const f32,
        kernel: *const f32,
        conv_state: *mut f32,
        out: *mut f32,
        qkv_dim: i32,
        conv_kernel: i32,
    ) -> Result<(), AllpacaError> {
        if x.is_null()
            || kernel.is_null()
            || conv_state.is_null()
            || out.is_null()
            || qkv_dim <= 0
            || conv_kernel <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid gdn_conv1d_step".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_gdn_conv1d_step(
                        x,
                        kernel,
                        conv_state,
                        out,
                        qkv_dim,
                        conv_kernel,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "gdn_conv1d_step")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP gdn_conv1d unavailable".into(),
            )))
        }
    }

    pub fn gdn_conv1d_step_host(
        &self,
        x: &[f32],
        kernel: &[f32],
        conv_state: &mut [f32],
        out: &mut [f32],
        conv_kernel: i32,
    ) -> Result<(), AllpacaError> {
        let qkv = x.len();
        let k = conv_kernel as usize;
        if k == 0
            || out.len() != qkv
            || kernel.len() != k.saturating_mul(qkv)
            || conv_state.len() != k.saturating_sub(1).saturating_mul(qkv)
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gdn_conv1d_step_host length mismatch".into(),
            )));
        }
        let need = (qkv + kernel.len() + conv_state.len() + out.len()) * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("gdn_conv1d need {need} scratch={}", self.scratch_bytes),
            )));
        }
        self.bind()?;
        let mut off = 0usize;
        let dx = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dx.cast(), x.as_ptr().cast(), qkv * 4, MEMCPY_H2D)?;
        off += qkv * 4;
        let dk = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            dk.cast(),
            kernel.as_ptr().cast(),
            kernel.len() * 4,
            MEMCPY_H2D,
        )?;
        off += kernel.len() * 4;
        let ds = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(
            ds.cast(),
            conv_state.as_ptr().cast(),
            conv_state.len() * 4,
            MEMCPY_H2D,
        )?;
        off += conv_state.len() * 4;
        let dout = unsafe { self.scratch.add(off).cast::<f32>() };
        self.add_h2d((qkv + kernel.len() + conv_state.len()) * 4);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_gdn_conv1d_step(
                        dx,
                        dk,
                        ds,
                        dout,
                        qkv as i32,
                        conv_kernel,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "gdn_conv1d_step")
            })?;
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP gdn_conv1d unavailable".into(),
            )));
        }
        memcpy(out.as_mut_ptr().cast(), dout.cast(), qkv * 4, MEMCPY_D2H)?;
        memcpy(
            conv_state.as_mut_ptr().cast(),
            ds.cast(),
            conv_state.len() * 4,
            MEMCPY_D2H,
        )?;
        self.add_d2h((qkv + conv_state.len()) * 4);
        Ok(())
    }

    /// Tile K-heads to V-heads on device pointers (ggml `repeat` along heads).
    pub fn expand_k_heads(
        &self,
        src: *const f32,
        dst: *mut f32,
        n_k: i32,
        n_v: i32,
        head_k: i32,
        interleave: bool,
    ) -> Result<(), AllpacaError> {
        if src.is_null() || dst.is_null() || n_k <= 0 || n_v <= 0 || head_k <= 0 || n_v % n_k != 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid expand_k_heads".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_expand_k_heads(src, dst, n_k, n_v, head_k, i32::from(interleave), std::ptr::null_mut())
                };
                status(rc, "expand_k_heads")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP expand_k unavailable".into(),
            )))
        }
    }

    pub fn expand_k_heads_host(
        &self,
        src: &[f32],
        dst: &mut [f32],
        n_k: i32,
        n_v: i32,
        head_k: i32,
        interleave: bool,
    ) -> Result<(), AllpacaError> {
        let nk = n_k as usize;
        let nv = n_v as usize;
        let hk = head_k as usize;
        if nk == 0
            || nv == 0
            || hk == 0
            || nv % nk != 0
            || src.len() != nk * hk
            || dst.len() != nv * hk
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "expand_k_heads_host length mismatch".into(),
            )));
        }
        let need = (src.len() + dst.len()) * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("expand_k need {need}"),
            )));
        }
        self.bind()?;
        let ds = self.scratch.cast::<f32>();
        let dd = unsafe { self.scratch.add(src.len() * 4).cast::<f32>() };
        memcpy(ds.cast(), src.as_ptr().cast(), src.len() * 4, MEMCPY_H2D)?;
        self.add_h2d(src.len() * 4);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_expand_k_heads(ds, dd, n_k, n_v, head_k, i32::from(interleave), std::ptr::null_mut())
                };
                status(rc, "expand_k_heads")
            })?;
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP expand_k unavailable".into(),
            )));
        }
        memcpy(
            dst.as_mut_ptr().cast(),
            dd.cast(),
            dst.len() * 4,
            MEMCPY_D2H,
        )?;
        self.add_d2h(dst.len() * 4);
        Ok(())
    }

    /// `decay = exp(softplus(α+dt)*a)` and `β = sigmoid(β)` on device pointers.
    pub fn gdn_decay_beta(
        &self,
        alpha: *const f32,
        dt_bias: *const f32,
        a: *const f32,
        decay: *mut f32,
        beta: *mut f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if alpha.is_null()
            || dt_bias.is_null()
            || a.is_null()
            || decay.is_null()
            || beta.is_null()
            || n <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid gdn_decay_beta".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_gdn_decay_beta(
                        alpha,
                        dt_bias,
                        a,
                        decay,
                        beta,
                        n,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "gdn_decay_beta")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP gdn_decay unavailable".into(),
            )))
        }
    }

    /// One-token GDN layer on device: conv1d → expand-k → decay/β → gated delta.
    /// Activations `qkv` / `α` / `β` stay on device (no D2H). `β` is sigmoid'd in place.
    /// Three launches when the fused kernels cover the shape (`head_k` a
    /// multiple of 32 up to 256), else [`Self::gdn_layer_step_unfused`];
    /// both give the same bits.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_layer_step(
        &self,
        qkv: *const f32,
        alpha: *const f32,
        beta: *mut f32,
        conv_w: *const f32,
        conv_state: *mut f32,
        dt_bias: *const f32,
        a: *const f32,
        state: *mut f32,
        y: *mut f32,
        n_v_heads: i32,
        n_k_heads: i32,
        head_v: i32,
        head_k: i32,
        conv_kernel: i32,
        eps: f32,
        k_interleave: bool,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let (n_v, n_k, hv, hk) = (
                n_v_heads.max(0) as usize,
                n_k_heads.max(0) as usize,
                head_v.max(0) as usize,
                head_k.max(0) as usize,
            );
            let qkv_dim = 2 * n_k * hk + n_v * hv;
            // Same scratch layout as the unfused step: conv_out first, decay
            // after the two (unused here) expanded q/k slots.
            let decay_off = qkv_dim + 2 * n_v * hk;
            if (decay_off + n_v) * 4 <= self.scratch_bytes && !conv_w.is_null() {
                self.bind()?;
                let conv_out = self.scratch.cast::<f32>();
                let decay = unsafe { self.scratch.add(decay_off * 4).cast::<f32>() };
                let rc = unsafe {
                    ffi::allpaca_gdn_step_fused(
                        qkv,
                        conv_w,
                        conv_state,
                        conv_out,
                        alpha,
                        dt_bias,
                        a,
                        decay,
                        beta,
                        state,
                        y,
                        n_v_heads,
                        n_k_heads,
                        head_v,
                        head_k,
                        conv_kernel,
                        eps,
                        i32::from(k_interleave),
                        std::ptr::null_mut(),
                    )
                };
                if rc != ALLPACA_ERR_INVALID {
                    return status(rc, "gdn_step_fused");
                }
            }
        }
        self.gdn_layer_step_unfused(
            qkv,
            alpha,
            beta,
            conv_w,
            conv_state,
            dt_bias,
            a,
            state,
            y,
            n_v_heads,
            n_k_heads,
            head_v,
            head_k,
            conv_kernel,
            eps,
            k_interleave,
        )
    }

    /// [`Self::gdn_layer_step`] as separate launches (conv, shift, 2x expand,
    /// decay/β, L2 norm, delta step): the fallback, and the reference the
    /// fused path is tested against.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_layer_step_unfused(
        &self,
        qkv: *const f32,
        alpha: *const f32,
        beta: *mut f32,
        conv_w: *const f32,
        conv_state: *mut f32,
        dt_bias: *const f32,
        a: *const f32,
        state: *mut f32,
        y: *mut f32,
        n_v_heads: i32,
        n_k_heads: i32,
        head_v: i32,
        head_k: i32,
        conv_kernel: i32,
        eps: f32,
        k_interleave: bool,
    ) -> Result<(), AllpacaError> {
        let n_v = n_v_heads as usize;
        let n_k = n_k_heads as usize;
        let hv = head_v as usize;
        let hk = head_k as usize;
        let ck = conv_kernel as usize;
        if n_v == 0
            || n_k == 0
            || hv == 0
            || hk == 0
            || ck == 0
            || n_v % n_k != 0
            || qkv.is_null()
            || alpha.is_null()
            || beta.is_null()
            || conv_w.is_null()
            || conv_state.is_null()
            || dt_bias.is_null()
            || a.is_null()
            || state.is_null()
            || y.is_null()
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid gdn_layer_step".into(),
            )));
        }
        let key_dim = n_k * hk;
        let value_dim = n_v * hv;
        let qkv_dim = key_dim.saturating_mul(2).saturating_add(value_dim);
        let q_exp_n = n_v * hk;
        let scratch_n = qkv_dim + q_exp_n + q_exp_n + n_v;
        if scratch_n * 4 > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!(
                    "gdn_layer_step need {} scratch={}",
                    scratch_n * 4,
                    self.scratch_bytes
                ),
            )));
        }
        self.bind()?;
        let conv_out = self.scratch.cast::<f32>();
        let q_exp = unsafe { self.scratch.add(qkv_dim * 4).cast::<f32>() };
        let k_exp = unsafe { self.scratch.add((qkv_dim + q_exp_n) * 4).cast::<f32>() };
        let decay = unsafe {
            self.scratch
                .add((qkv_dim + q_exp_n + q_exp_n) * 4)
                .cast::<f32>()
        };
        self.gdn_conv1d_step(
            qkv,
            conv_w,
            conv_state,
            conv_out,
            qkv_dim as i32,
            conv_kernel,
        )?;
        let q_raw = conv_out as *const f32;
        let k_raw = unsafe { conv_out.add(key_dim) as *const f32 };
        let v_raw = unsafe { conv_out.add(key_dim * 2) as *const f32 };
        self.expand_k_heads(q_raw, q_exp, n_k_heads, n_v_heads, head_k, k_interleave)?;
        self.expand_k_heads(k_raw, k_exp, n_k_heads, n_v_heads, head_k, k_interleave)?;
        self.gdn_decay_beta(alpha, dt_bias, a, decay, beta, n_v_heads)?;
        self.gated_delta_step(
            q_exp, k_exp, v_raw, decay, beta, state, y, n_v_heads, head_v, head_k, eps,
        )
    }

    /// `out = silu(z) * rmsnorm(y)` per V-head (matches host `deltanet::rmsnorm_silu_gate`).
    /// One launch; [`Self::gdn_rmsnorm_silu_gate_unfused`] gives the same bits.
    pub fn gdn_rmsnorm_silu_gate(
        &self,
        y: *const f32,
        z: *const f32,
        weight: *const f32,
        out: *mut f32,
        n_heads: i32,
        head_v: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            if !(y.is_null() || z.is_null() || weight.is_null() || out.is_null())
                && n_heads > 0
                && head_v > 0
            {
                self.bind()?;
                let rc = unsafe {
                    ffi::allpaca_gdn_norm_gate(
                        y,
                        z,
                        weight,
                        out,
                        n_heads,
                        head_v,
                        eps,
                        std::ptr::null_mut(),
                    )
                };
                return status(rc, "gdn_norm_gate");
            }
        }
        self.gdn_rmsnorm_silu_gate_unfused(y, z, weight, out, n_heads, head_v, eps)
    }

    /// [`Self::gdn_rmsnorm_silu_gate`] as copies + `rmsnorm_heads` + `silu_mul`:
    /// the reference the fused kernel is tested against.
    pub fn gdn_rmsnorm_silu_gate_unfused(
        &self,
        y: *const f32,
        z: *const f32,
        weight: *const f32,
        out: *mut f32,
        n_heads: i32,
        head_v: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if y.is_null()
            || z.is_null()
            || weight.is_null()
            || out.is_null()
            || n_heads <= 0
            || head_v <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid gdn_rmsnorm_silu_gate".into(),
            )));
        }
        let n = (n_heads as usize).saturating_mul(head_v as usize);
        if n * 4 > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("gdn_rmsnorm_silu_gate need {} scratch", n * 4),
            )));
        }
        self.bind()?;
        // Copy z first so `out == z` (decode writes gated into the z buffer) is safe.
        let z_silu = self.scratch.cast::<f32>();
        if z_silu as usize != z as usize {
            self.copy_d2d_f32(z_silu, z, n)?;
        }
        if out as usize != y as usize {
            self.copy_d2d_f32(out, y, n)?;
        }
        self.rmsnorm_heads_device(out, weight, n_heads, head_v, eps)?;
        self.silu_mul_device(z_silu, out, n as i32)?;
        if out as usize != z_silu as usize {
            self.copy_d2d_f32(out, z_silu, n)?;
        }
        Ok(())
    }

    pub fn gdn_decay_beta_host(
        &self,
        alpha: &[f32],
        dt_bias: &[f32],
        a: &[f32],
        decay: &mut [f32],
        beta: &mut [f32],
    ) -> Result<(), AllpacaError> {
        let n = alpha.len();
        if n == 0 || dt_bias.len() != n || a.len() != n || decay.len() != n || beta.len() != n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gdn_decay_beta_host length mismatch".into(),
            )));
        }
        let need = n * 5 * 4;
        if need > self.scratch_bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                format!("gdn_decay need {need}"),
            )));
        }
        self.bind()?;
        let mut off = 0usize;
        let dalpha = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dalpha.cast(), alpha.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        off += n * 4;
        let ddt = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(ddt.cast(), dt_bias.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        off += n * 4;
        let da = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(da.cast(), a.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        off += n * 4;
        let ddecay = unsafe { self.scratch.add(off).cast::<f32>() };
        off += n * 4;
        let dbeta = unsafe { self.scratch.add(off).cast::<f32>() };
        memcpy(dbeta.cast(), beta.as_ptr().cast(), n * 4, MEMCPY_H2D)?;
        self.add_h2d(n * 4 * 4);
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_gdn_decay_beta(
                        dalpha,
                        ddt,
                        da,
                        ddecay,
                        dbeta,
                        n as i32,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "gdn_decay_beta")
            })?;
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP gdn_decay unavailable".into(),
            )));
        }
        memcpy(decay.as_mut_ptr().cast(), ddecay.cast(), n * 4, MEMCPY_D2H)?;
        memcpy(beta.as_mut_ptr().cast(), dbeta.cast(), n * 4, MEMCPY_D2H)?;
        self.add_d2h(n * 8);
        Ok(())
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

    pub fn layernorm_device_rows(
        &self,
        x: *const f32,
        weight: *const f32,
        bias: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if rows == 0 {
            return Ok(());
        }
        if x.is_null() || weight.is_null() || bias.is_null() || out.is_null() || rows < 0 || n <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device layernorm".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_layernorm_rows(
                        x,
                        weight,
                        bias,
                        out,
                        rows,
                        n,
                        eps,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "layernorm_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = eps;
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP layernorm unavailable".into(),
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

    pub fn rope_imrope_qk_device(
        &self,
        q: *mut f32,
        k: *mut f32,
        n_q_heads: i32,
        n_k_heads: i32,
        head_dim: i32,
        n_rot: i32,
        pos: [i32; 3],
        theta: f32,
        sections: [i32; 4],
    ) -> Result<(), AllpacaError> {
        if q.is_null()
            || k.is_null()
            || n_q_heads <= 0
            || n_k_heads <= 0
            || head_dim <= 0
            || n_rot <= 0
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device imrope".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_rope_imrope_qk(
                        q,
                        k,
                        n_q_heads,
                        n_k_heads,
                        head_dim,
                        n_rot,
                        pos.as_ptr(),
                        theta,
                        sections.as_ptr(),
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "rope_imrope_qk")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (pos, theta, sections);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rope_imrope_qk unavailable".into(),
            )))
        }
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
        if q.is_null()
            || k.is_null()
            || n_tok < 0
            || n_q_heads <= 0
            || n_k_heads <= 0
            || head_dim <= 0
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

    /// Partial RoPE on device `x[n_tok, n_heads, head_dim]` (`[offset, offset+n_rot)`).
    /// `n_rot == 0` is full-head RoPE. `back` is ggml `rope_ext_back`.
    pub fn rope_offset_device_rows(
        &self,
        x: *mut f32,
        n_tok: i32,
        n_heads: i32,
        head_dim: i32,
        n_rot: i32,
        offset: i32,
        start_pos: i32,
        theta: f32,
        back: bool,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || n_tok <= 0 || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device rope_offset".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                // SAFETY: `x` is a device `[n_tok, n_heads, head_dim]` buffer;
                // the kernel only writes `[offset, offset+n_rot)` per head.
                let rc = unsafe {
                    ffi::allpaca_rope_offset_rows(
                        x,
                        n_tok,
                        n_heads,
                        head_dim,
                        n_rot,
                        offset,
                        start_pos,
                        theta,
                        i32::from(back),
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "rope_offset_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (n_rot, offset, start_pos, theta, back);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rope_offset unavailable".into(),
            )))
        }
    }

    /// [`Self::rope_offset_device_rows`] with YaRN frequencies: `yarn` is
    /// the scale of the slow pairs and the pair indices the blend runs between.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_device_rows(
        &self,
        x: *mut f32,
        n_tok: i32,
        n_heads: i32,
        head_dim: i32,
        n_rot: i32,
        offset: i32,
        start_pos: i32,
        theta: f32,
        yarn: [f32; 3],
        back: bool,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || n_tok <= 0 || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device rope_yarn".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                // SAFETY: `x` is a device `[n_tok, n_heads, head_dim]` buffer;
                // the kernel only writes `[offset, offset+n_rot)` per head.
                let rc = unsafe {
                    ffi::allpaca_rope_yarn_rows(
                        x,
                        n_tok,
                        n_heads,
                        head_dim,
                        n_rot,
                        offset,
                        start_pos,
                        theta,
                        yarn[0],
                        yarn[1],
                        yarn[2],
                        i32::from(back),
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "rope_yarn_rows")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (n_rot, offset, start_pos, theta, yarn, back);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rope_yarn unavailable".into(),
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
                let rc = unsafe {
                    ffi::allpaca_rope(x, n_heads, head_dim, pos, theta, std::ptr::null_mut())
                };
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

    pub fn silu_mul_device(
        &self,
        gate: *mut f32,
        up: *const f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
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

    pub fn gelu_mul_device(
        &self,
        gate: *mut f32,
        up: *const f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if gate.is_null() || up.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device gelu_mul".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_gelu_mul(gate, up, n, std::ptr::null_mut()) };
                status(rc, "gelu_mul")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP gelu_mul unavailable".into(),
            )))
        }
    }

    pub fn relu_sqr_device(&self, x: *mut f32, n: i32) -> Result<(), AllpacaError> {
        if x.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid device relu_sqr".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_relu_sqr(x, n, std::ptr::null_mut()) };
                status(rc, "relu_sqr")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP relu_sqr unavailable".into(),
            )))
        }
    }

    pub fn scale_heads_sigmoid_device(
        &self,
        attn: *mut f32,
        scores: *const f32,
        n_tok: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), AllpacaError> {
        if attn.is_null() || scores.is_null() || n_tok <= 0 || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid scale_heads_sigmoid".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_scale_heads_sigmoid(
                        attn,
                        scores,
                        n_tok,
                        n_heads,
                        head_dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "scale_heads_sigmoid")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP scale_heads_sigmoid unavailable".into(),
            )))
        }
    }

    /// Device Q/K/V attention with a host-side [n_q, n_kv] uint8 mask (1 = attend).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_device_masked_upload(
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
        mask: &[u8],
    ) -> Result<(), AllpacaError> {
        let mask_n = (n_q as usize).saturating_mul(n_kv as usize);
        if mask.len() < mask_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention mask size mismatch".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let dm = device_malloc(mask_n, self.device)?;
            let result = (|| {
                memcpy(dm, mask.as_ptr(), mask_n, MEMCPY_H2D)?;
                self.add_h2d(mask_n);
                self.attention_device_masked(
                    q,
                    k,
                    v,
                    out,
                    n_q,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    dm.cast(),
                )
            })();
            device_free(dm, mask_n);
            result
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            let _ = (q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP attention_masked unavailable".into(),
            )))
        }
    }

    pub fn deinterleave_q_gate_device(
        &self,
        q_full: *const f32,
        q: *mut f32,
        gate: *mut f32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), AllpacaError> {
        if q_full.is_null() || q.is_null() || gate.is_null() || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid deinterleave_q_gate".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_deinterleave_q_gate(
                        q_full,
                        q,
                        gate,
                        n_heads,
                        head_dim,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "deinterleave_q_gate")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP deinterleave_q_gate unavailable".into(),
            )))
        }
    }

    pub fn sigmoid_mul_device(
        &self,
        x: *mut f32,
        gate: *const f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || gate.is_null() || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid sigmoid_mul".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe { ffi::allpaca_sigmoid_mul(x, gate, n, std::ptr::null_mut()) };
                status(rc, "sigmoid_mul")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP sigmoid_mul unavailable".into(),
            )))
        }
    }

    pub fn rmsnorm_heads_device(
        &self,
        x: *mut f32,
        weight: *const f32,
        n_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if x.is_null() || weight.is_null() || n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rmsnorm_heads".into(),
            )));
        }
        self.bind()?;
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            self.timed_gpu(DecodeOp::OtherGpu, || {
                let rc = unsafe {
                    ffi::allpaca_rmsnorm_heads(
                        x,
                        weight,
                        n_heads,
                        head_dim,
                        eps,
                        std::ptr::null_mut(),
                    )
                };
                status(rc, "rmsnorm_heads")
            })
        }
        #[cfg(not(all(feature = "hip", allpaca_hip_runtime)))]
        {
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "HIP rmsnorm_heads unavailable".into(),
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
        let dm = unsafe {
            self.scratch
                .add((q.len() + k.len() + v.len() + out.len()) * 4)
        };
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
        destroy_swiglu_pair(self.swiglu_s, self.swiglu_ev);
        self.swiglu_s = [std::ptr::null_mut(); SWIGLU_STREAMS];
        self.swiglu_ev = [std::ptr::null_mut(); SWIGLU_EVENTS];
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
        if let Ok(mut g) = self.moe.lock() {
            if let Some(stage) = g.take() {
                free_moe_stage(stage);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small host<->device copy latency: pageable `hipMemcpy` vs a pinned
    /// staging buffer. `cargo test --release -p allpaca-hip --features hip --lib bench_small_copies -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_small_copies() {
        if device_count() == 0 {
            return;
        }
        let n = 2048usize;
        let bytes = n * 4;
        let dev = device_malloc(bytes, 0).unwrap();
        let pinned = host_malloc(bytes).unwrap();
        let mut host = vec![1.0f32; n];
        let time = |name: &str, f: &mut dyn FnMut()| {
            for _ in 0..50 {
                f();
            }
            let mut v: Vec<f64> = (0..400)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e6
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("COPY {name}: median {:.1} us p90 {:.1} us", v[200], v[360]);
        };
        time("sync only", &mut || stream_synchronize().unwrap());
        time("h2d pageable", &mut || {
            memcpy(dev, host.as_ptr().cast(), bytes, MEMCPY_H2D).unwrap();
            stream_synchronize().unwrap();
        });
        time("h2d pinned", &mut || {
            memcpy(dev, pinned, bytes, MEMCPY_H2D).unwrap();
            stream_synchronize().unwrap();
        });
        time("d2h pageable", &mut || {
            memcpy(host.as_mut_ptr().cast(), dev, bytes, MEMCPY_D2H).unwrap();
        });
        time("d2h pinned", &mut || {
            memcpy(pinned, dev, bytes, MEMCPY_D2H).unwrap();
        });
        time("d2h 2 KB pinned", &mut || {
            memcpy(pinned, dev, 2048, MEMCPY_D2H).unwrap();
        });
        time("h2d by kernel from pinned", &mut || {
            copy_f32_kernel(dev.cast(), pinned.cast(), n).unwrap();
            stream_synchronize().unwrap();
        });
        time("d2h by kernel to pinned", &mut || {
            copy_f32_kernel(pinned.cast(), dev.cast(), n).unwrap();
            stream_synchronize().unwrap();
        });
        // The value must actually land in host memory.
        let probe = device_malloc(bytes, 0).unwrap();
        let ones = vec![3.5f32; n];
        memcpy(probe, ones.as_ptr().cast(), bytes, MEMCPY_H2D).unwrap();
        copy_f32_kernel(pinned.cast(), probe.cast(), n).unwrap();
        stream_synchronize().unwrap();
        let got = unsafe { std::slice::from_raw_parts(pinned.cast::<f32>(), n) };
        assert!(got.iter().all(|v| *v == 3.5), "kernel D2H into pinned memory");
        device_free(probe, bytes);
        // CPU cost of issuing work, no waits.
        let t = std::time::Instant::now();
        for _ in 0..2000 {
            copy_f32_kernel(dev.cast(), probe_src(dev), n).unwrap();
        }
        let launch_us = t.elapsed().as_secs_f64() * 1e6 / 2000.0;
        let t = std::time::Instant::now();
        stream_synchronize().unwrap();
        eprintln!("COPY launch-only cpu cost: {launch_us:.2} us/launch; drain {:.0} us for 2000", t.elapsed().as_secs_f64() * 1e6);
        let t = std::time::Instant::now();
        for _ in 0..2000 {
            set_device(0).unwrap();
        }
        eprintln!("COPY set_device: {:.2} us/call", t.elapsed().as_secs_f64() * 1e6 / 2000.0);
        host_free(pinned);
        device_free(dev, bytes);
    }

    /// GPU cost of the Qwen3-Next GDN decode GEMVs, pipelined (1000 launches
    /// then one sync). `cargo test --release -p allpaca-hip --features hip --lib bench_gdn_gemv -- --ignored --nocapture`
    /// A shared expert's gate + up as one paired Q8_0 launch gives the bits of
    /// two separate GEMVs; scale_add_copy the bits of scale_f32 +
    /// residual_add + copy.
    #[test]
    fn q8_pair_and_scale_add_copy_match_separate_launches() {
        if device_count() == 0 {
            return;
        }
        let Ok(hip) = HipWorkspace::new(0, 1 << 20, 8) else { return };
        let (n, k) = (512usize, 2048usize);
        let q8 = |salt: usize| -> Vec<u8> {
            let packed: Vec<u8> = (0..n * k / 32 * 34)
                .map(|i| match i % 34 {
                    0 => ((i / 34 + salt) % 7) as u8 * 16,
                    1 => 0x1c,
                    _ => ((i * 7 + salt * 13) % 251) as u8,
                })
                .collect();
            stage_q8_0_blocks(&packed).unwrap()
        };
        let (g, u) = (q8(1), q8(2));
        let x: Vec<f32> = (0..k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013).collect();
        let dev = |bytes: usize| device_malloc(bytes, 0).unwrap();
        let (gd, ud, xd) = (dev(g.len()), dev(u.len()), dev(k * 4));
        memcpy(gd, g.as_ptr(), g.len(), MEMCPY_H2D).unwrap();
        memcpy(ud, u.as_ptr(), u.len(), MEMCPY_H2D).unwrap();
        memcpy(xd, x.as_ptr().cast(), k * 4, MEMCPY_H2D).unwrap();
        let outs: Vec<*mut u8> = (0..4).map(|_| dev(n * 4)).collect();
        hip.gemm_nt_q8_dev(xd.cast(), gd, outs[0].cast(), 1, n as i32, k as i32).unwrap();
        hip.gemm_nt_q8_dev(xd.cast(), ud, outs[1].cast(), 1, n as i32, k as i32).unwrap();
        let paired = hip
            .gemm_nt_q8_pair_dev(xd.cast(), gd, outs[2].cast(), ud, outs[3].cast(), 1, n as i32, n as i32, k as i32)
            .unwrap();
        let read = |p: *mut u8| {
            let mut v = vec![0.0f32; n];
            hip.copy_d2h_f32(p.cast(), &mut v).unwrap();
            v
        };
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert!(read(outs[0]).iter().any(|v| *v != 0.0));
        // Small weights stay on the f32 GEMV, so the pair launch applies.
        assert!(paired, "m == 1 must take the paired launch");
        assert_eq!(bits(&read(outs[0])), bits(&read(outs[2])), "paired gate differs");
        assert_eq!(bits(&read(outs[1])), bits(&read(outs[3])), "paired up differs");

        // scale_add_copy vs scale_f32 + residual_add + copy_d2d.
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).cos() * 3.0).collect();
        let s = 0.731_234_5f32;
        let put = |p: *mut u8, v: &[f32]| hip.copy_h2d_f32(p.cast(), v).unwrap();
        put(outs[0], &a);
        put(outs[1], &b);
        hip.scale_f32(outs[1].cast(), n, s).unwrap();
        hip.residual_add(outs[0].cast(), outs[1].cast(), n as i32).unwrap();
        hip.copy_d2d_f32(outs[1].cast(), outs[0].cast(), n).unwrap();
        put(outs[2], &a);
        put(outs[3], &b);
        hip.scale_add_copy(outs[2].cast(), outs[3].cast(), s, n).unwrap();
        assert_eq!(bits(&read(outs[0])), bits(&read(outs[2])), "scale_add_copy x differs");
        assert_eq!(bits(&read(outs[1])), bits(&read(outs[3])), "scale_add_copy y differs");
        for p in outs {
            device_free(p, n * 4);
        }
        device_free(gd, g.len());
        device_free(ud, u.len());
        device_free(xd, k * 4);
    }

    /// One fused GDN decode step at Qwen3.8-27B dims (48 v-heads, 16 k-heads,
    /// 128x128 state): `BENCH_DEV=0|1 cargo test --release -p allpaca-hip
    /// --features hip --lib bench_gdn_step -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_gdn_step() {
        if device_count() == 0 {
            return;
        }
        let dev = std::env::var("BENCH_DEV").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let ws = HipWorkspace::new(dev, 8 << 20, 16).unwrap();
        let (n_v, n_k, hv, hk, ck) = (48usize, 16usize, 128usize, 128usize, 4usize);
        let qkv_dim = 2 * n_k * hk + n_v * hv;
        let up = |v: &[f32]| {
            let p = device_malloc(v.len() * 4, dev).unwrap();
            memcpy(p, v.as_ptr().cast(), v.len() * 4, MEMCPY_H2D).unwrap();
            p.cast::<f32>()
        };
        let qkv = up(&vec![0.01; qkv_dim]);
        let alpha = up(&vec![0.1; n_v]);
        let beta = up(&vec![0.2; n_v]);
        let conv_w = up(&vec![0.25; qkv_dim * ck]);
        let conv_state = up(&vec![0.0; qkv_dim * (ck - 1)]);
        let dt = up(&vec![0.0; n_v]);
        let a = up(&vec![-1.0; n_v]);
        // Several states so they stream from VRAM like 48 layers do.
        let states: Vec<*mut f32> = (0..16).map(|_| up(&vec![0.0; n_v * hv * hk])).collect();
        let y = up(&vec![0.0; n_v * hv]);
        let next = std::cell::Cell::new(0usize);
        let step = || {
            let st = states[next.get() % states.len()];
            next.set(next.get() + 1);
            ws.gdn_layer_step(
                qkv, alpha, beta, conv_w, conv_state, dt, a, st, y, n_v as i32, n_k as i32,
                hv as i32, hk as i32, ck as i32, 1e-6, false,
            )
            .unwrap();
        };
        for _ in 0..20 {
            step();
        }
        stream_synchronize().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..500 {
            step();
        }
        stream_synchronize().unwrap();
        let us = t.elapsed().as_secs_f64() * 1e6 / 500.0;
        let bytes = 2 * n_v * hv * hk * 4;
        eprintln!("dev{dev} gdn_step_fused: {us:.1} us ({:.0} GB/s on state r+w)", bytes as f64 / us / 1e3);
    }

    /// Time of one Q4_K and one Q8_0 product for m = 1..=16 rows (decode and verify
    /// windows), to see where a row count costs more than its neighbours.
    #[test]
    #[ignore]
    fn bench_kquant_rows() {
        if device_count() == 0 {
            return;
        }
        let dev = std::env::var("BENCH_DEV").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        set_device(dev).unwrap();
        for (n, k) in [(5120usize, 5120usize), (17408, 5120), (5120, 17408)] {
            let blocks = n * k / 256;
            let packed: Vec<u8> = (0..blocks * 144)
                .map(|i| match i % 144 {
                    0 | 2 => 0x00,
                    1 | 3 => 0x1c,
                    j => (i * 7 % 251 + j) as u8,
                })
                .collect();
            let w = device_malloc(packed.len(), dev).unwrap();
            memcpy(w, packed.as_ptr(), packed.len(), MEMCPY_H2D).unwrap();
            let x = device_malloc(16 * k * 4, dev).unwrap();
            let y = device_malloc(16 * n * 4, dev).unwrap();
            let xs: Vec<f32> = (0..16 * k).map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.01).collect();
            memcpy(x, xs.as_ptr().cast(), 16 * k * 4, MEMCPY_H2D).unwrap();
            let mut line = format!("dev{dev} q4_k n={n} k={k} us:");
            for m in 1..=16i32 {
                let f = || gemm_nt_q4_k_device(x.cast(), w, y.cast(), m, n as i32, k as i32).unwrap();
                for _ in 0..30 {
                    f();
                }
                stream_synchronize().unwrap();
                let t = std::time::Instant::now();
                for _ in 0..300 {
                    f();
                }
                stream_synchronize().unwrap();
                line += &format!(" m{m}={:.0}", t.elapsed().as_secs_f64() * 1e6 / 300.0);
            }
            eprintln!("{line}");
            let b8 = n * k / 32;
            let p8: Vec<u8> = (0..b8 * 34)
                .map(|i| match i % 34 {
                    0 => 0x00,
                    1 => 0x1c,
                    _ => (i * 7 % 251) as u8,
                })
                .collect();
            let staged = stage_q8_0_blocks(&p8).unwrap();
            let w8 = device_malloc(staged.len(), dev).unwrap();
            memcpy(w8, staged.as_ptr(), staged.len(), MEMCPY_H2D).unwrap();
            let mut line = format!("dev{dev} q8_0 n={n} k={k} us:");
            for m in 1..=16i32 {
                let f = || gemm_nt_q8_0_device(x.cast(), w8, y.cast(), m, n as i32, k as i32).unwrap();
                for _ in 0..30 {
                    f();
                }
                stream_synchronize().unwrap();
                let t = std::time::Instant::now();
                for _ in 0..300 {
                    f();
                }
                stream_synchronize().unwrap();
                line += &format!(" m{m}={:.0}", t.elapsed().as_secs_f64() * 1e6 / 300.0);
            }
            eprintln!("{line}");
            device_free(w8, staged.len());
            device_free(x, 16 * k * 4);
            device_free(y, 16 * n * 4);
            device_free(w, packed.len());
        }
    }

    #[test]
    #[ignore]
    fn bench_gdn_gemv() {
        if device_count() == 0 {
            return;
        }
        let dev = std::env::var("BENCH_DEV").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        set_device(dev).unwrap();
        let k = 2048usize;
        let x = device_malloc(8192 * 4, dev).unwrap();
        let y = device_malloc(16384 * 4, dev).unwrap();
        memcpy(x, vec![0.01f32; 8192].as_ptr().cast(), 8192 * 4, MEMCPY_H2D).unwrap();
        let run = |name: &str, bytes: usize, f: &dyn Fn()| {
            for _ in 0..50 {
                f();
            }
            stream_synchronize().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..1000 {
                f();
            }
            stream_synchronize().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / 1000.0;
            eprintln!("GEMV {name}: {us:.1} us ({:.0} GB/s)", bytes as f64 / us / 1e3);
        };
        for (n, kk) in [(8192usize, k), (4096, k), (2048, 4096), (512, k), (2048, 512)] {
            let blocks = n * kk / 32;
            let packed: Vec<u8> = (0..blocks * 34).map(|i| if i % 34 < 2 { if i % 34 == 0 { 0x00 } else { 0x1c } } else { (i * 7 % 251) as u8 }).collect();
            let staged = stage_q8_0_blocks(&packed).unwrap();
            let w = device_malloc(staged.len(), dev).unwrap();
            memcpy(w, staged.as_ptr(), staged.len(), MEMCPY_H2D).unwrap();
            run(&format!("q8_0 n={n} k={kk}"), staged.len(), &|| {
                gemm_nt_q8_0_device(x.cast(), w, y.cast(), 1, n as i32, kk as i32).unwrap()
            });
            device_free(w, staged.len());
        }
        let wf = device_malloc(64 * k * 4, dev).unwrap();
        run("f32 n=32 k=2048", 32 * k * 4, &|| {
            gemm_nt_device(x.cast(), wf.cast(), y.cast(), 1, 32, k as i32).unwrap()
        });
        run("f32 n=64 k=2048", 64 * k * 4, &|| {
            gemm_nt_device(x.cast(), wf.cast(), y.cast(), 1, 64, k as i32).unwrap()
        });
        let wr = device_malloc(512 * k * 4, dev).unwrap();
        run("f32 n=512 k=2048 (router)", 512 * k * 4, &|| {
            gemm_nt_device(x.cast(), wr.cast(), y.cast(), 1, 512, k as i32).unwrap()
        });
        device_free(wr, 512 * k * 4);
        run("d2d hipMemcpy 2048 floats", 8192, &|| {
            memcpy(y, x, 8192, MEMCPY_D2D).unwrap()
        });
        run("kernel + d2d hipMemcpy + kernel", 8192, &|| {
            copy_f32_kernel(y.cast(), x.cast(), 2048).unwrap();
            memcpy(y, x, 8192, MEMCPY_D2D).unwrap();
            copy_f32_kernel(y.cast(), x.cast(), 2048).unwrap();
        });
        run("kernel x3 (same work by kernel)", 8192, &|| {
            for _ in 0..3 {
                copy_f32_kernel(y.cast(), x.cast(), 2048).unwrap();
            }
        });
        {
            // One GEMV after an idle gap, timed launch -> sync (minus an empty kernel's).
            let packed: Vec<u8> = (0..8192 * 2048 / 32 * 34).map(|i| if i % 34 < 2 { if i % 34 == 0 { 0 } else { 0x1c } } else { (i * 7 % 251) as u8 }).collect();
            let staged = stage_q8_0_blocks(&packed).unwrap();
            let w = device_malloc(staged.len(), dev).unwrap();
            memcpy(w, staged.as_ptr(), staged.len(), MEMCPY_H2D).unwrap();
            for gap_us in [0u64, 200, 500, 1000, 3000] {
                let mut tot = 0.0;
                let mut base = 0.0;
                for it in 0..300 {
                    let until = std::time::Instant::now() + std::time::Duration::from_micros(gap_us);
                    while std::time::Instant::now() < until {}
                    let t = std::time::Instant::now();
                    if it % 2 == 0 {
                        gemm_nt_q8_0_device(x.cast(), w, y.cast(), 1, 8192, 2048).unwrap();
                    } else {
                        copy_f32_kernel(y.cast(), x.cast(), 16).unwrap();
                    }
                    stream_synchronize().unwrap();
                    let us = t.elapsed().as_secs_f64() * 1e6;
                    if it % 2 == 0 { tot += us } else { base += us }
                }
                eprintln!("GEMV idle gap {gap_us}us: q8 n=8192 launch->sync {:.1} us, empty kernel {:.1} us", tot / 150.0, base / 150.0);
            }
            // A decode layer body: many small kernels after the RPC wait. Burst
            // (host enqueues back to back) vs trickle (host work between
            // launches), timed from the first launch to sync.
            for (n, host_us) in [(1usize, 0u64), (8, 0), (25, 0), (50, 0), (25, 5), (25, 15)] {
                let mut tot = 0.0;
                for _ in 0..100 {
                    let until = std::time::Instant::now() + std::time::Duration::from_micros(500);
                    while std::time::Instant::now() < until {}
                    let t = std::time::Instant::now();
                    for _ in 0..n {
                        copy_f32_kernel(y.cast(), x.cast(), 2048).unwrap();
                        let until = std::time::Instant::now() + std::time::Duration::from_micros(host_us);
                        while std::time::Instant::now() < until {}
                    }
                    stream_synchronize().unwrap();
                    tot += t.elapsed().as_secs_f64() * 1e6;
                }
                eprintln!("GEMV layer-like {n} kernels, {host_us}us host between: {:.1} us to sync ({:.1} us/kernel)", tot / 100.0, tot / 100.0 / n as f64);
            }
            device_free(w, staged.len());
        }
        run("copy 2048 (launch floor)", 8192, &|| {
            copy_f32_kernel(y.cast(), x.cast(), 2048).unwrap()
        });
        device_free(wf, 64 * k * 4);
        device_free(x, 8192 * 4);
        device_free(y, 16384 * 4);
    }

    fn probe_src(dev: *mut u8) -> *const f32 {
        dev.cast::<f32>().cast_const()
    }

    #[cfg(all(windows, feature = "hip", allpaca_hip_runtime))]
    #[test]
    fn direct_storage_reads_file_into_hip_visible_d3d12_buffer() {
        use std::io::Write;

        const BYTES: usize = 64 * 1024;
        let Some(device) = enumerate_devices().first().map(|d| d.id) else {
            return;
        };
        let path = std::env::temp_dir().join(format!(
            "allpaca-dstorage-dma-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let payload: Vec<u8> = (0..BYTES).map(|i| (i % 251) as u8).collect();
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&payload).unwrap();
            file.sync_all().unwrap();
        }

        let buffer =
            DirectStorageBuffer::alloc(device, BYTES).expect("D3D12 resource imported into HIP");
        buffer
            .read_file(&path, 0, 0, BYTES)
            .expect("DirectStorage NVMe-to-D3D12 request");
        let mut readback = vec![0u8; BYTES];
        buffer
            .copy_d2h(0, &mut readback)
            .expect("HIP must read the imported D3D12 resource");
        assert_eq!(readback, payload);

        let _ = std::fs::remove_file(path);
    }

    #[cfg(all(windows, feature = "hip", allpaca_hip_runtime))]
    #[test]
    fn device_weights_accepts_direct_storage_file_fill() {
        use std::io::Write;

        const BYTES: usize = 64 * 1024;
        let Some(device) = enumerate_devices().first().map(|d| d.id) else {
            return;
        };
        let path = std::env::temp_dir().join(format!(
            "allpaca-weight-dma-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let payload = vec![0xA5; BYTES];
        {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&payload).unwrap();
            file.sync_all().unwrap();
        }

        let weights = DeviceWeights::alloc_direct_storage(device, BYTES)
            .expect("D3D12-imported device weights");
        assert!(
            weights
                .read_file_direct(&path, 0, 0, BYTES)
                .expect("DirectStorage weight fill"),
            "Windows HIP weights should use the D3D12-imported allocation"
        );
        let mut readback = vec![0u8; BYTES];
        memcpy(
            readback.as_mut_ptr(),
            weights.byte_ptr_at(0),
            BYTES,
            MEMCPY_D2H,
        )
        .unwrap();
        assert_eq!(readback, payload);

        let _ = std::fs::remove_file(path);
    }

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

    #[test]
    fn memset_empty_is_ok() {
        device_memset(std::ptr::null_mut(), 0xAB, 0).expect("empty memset");
    }

    #[test]
    fn memset_rejects_null_payload() {
        assert!(device_memset(std::ptr::null_mut(), 0xAB, 64).is_err());
    }

    #[cfg(feature = "hip")]
    #[test]
    fn memset_fills_device_buffer_when_present() {
        if device_count() == 0 {
            return;
        }
        let n = 256usize;
        let d = device_malloc(n, 0).expect("malloc");
        device_memset(d, 0xAB, n).expect("memset");
        let mut back = vec![0u8; n];
        memcpy(back.as_mut_ptr(), d, n, MEMCPY_D2H).expect("d2h");
        device_free(d, n);
        assert_eq!(back, vec![0xAB; n]);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn memset_on_side_stream_when_device_present() {
        if device_count() == 0 {
            return;
        }
        let n = 128usize;
        let d = device_malloc(n, 0).expect("malloc");
        let h2d = H2dStream::new().expect("side stream");
        memset_on(d, 0x3C, n, h2d.raw()).expect("memset_on");
        h2d.synchronize().expect("sync");
        let mut back = vec![0u8; n];
        memcpy(back.as_mut_ptr(), d, n, MEMCPY_D2H).expect("d2h");
        device_free(d, n);
        assert_eq!(back, vec![0x3C; n]);
    }

    #[cfg(feature = "hip")]
    #[test]
    fn memcpy_peer_is_capable_even_when_access_api_denies() {
        if device_count() < 2 {
            return;
        }
        let access = probe_peer(0, 1);
        assert!(
            access.capable,
            "hipMemcpyPeer must count as P2P capable (gfx906 CanAccessPeer is often 0)"
        );
        let bench = benchmark_transfer(0, 1, 4 << 20).expect("transfer bench");
        assert!(bench.p2p_copy_ok, "peer copy must be measured: {bench:?}");
        assert!(
            bench.p2p_gbps > 0.0,
            "peer copy must report bandwidth: {bench:?}"
        );
        eprintln!(
            "P2P bench: copy_ok={} p2p_gbps={:.2} host_gbps={:.2} can_p2p={} enabled={}",
            bench.p2p_copy_ok,
            bench.p2p_gbps,
            bench.host_staging_gbps,
            can_p2p(0, 1),
            access.enabled
        );
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
        assert!(devices
            .iter()
            .all(|d| d.id >= 0 && d.kind == allpaca_types::DeviceKind::HipGpu));
    }

    #[cfg(feature = "hip")]
    /// A as the K-quant GEMV sees it: each 32-float block quantized to int8
    /// with scale amax/127 (llama.cpp q8_1), then dequantized.
    fn q8_activations(a: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; a.len()];
        for (src, dst) in a.chunks(32).zip(out.chunks_mut(32)) {
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
            let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
            for (s, o) in src.iter().zip(dst.iter_mut()) {
                *o = d * (s * inv).round().clamp(-127.0, 127.0);
            }
        }
        out
    }

    /// Every row of a K-quant GEMV batch equals that row run alone, bit for
    /// bit: speculative verify rows must reproduce decode exactly.
    fn assert_rows_match_single(
        name: &str,
        run: impl Fn(&[f32], &mut [f32], i32),
        a: &[f32],
        batched: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) {
        for r in 0..m {
            let mut one = vec![0.0f32; n];
            run(&a[r * k..(r + 1) * k], &mut one, 1);
            assert_eq!(
                &batched[r * n..(r + 1) * n],
                &one[..],
                "{name} row {r} of {m}x{n}x{k} differs from the single-row GEMV"
            );
        }
    }

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

    /// The activations a Q8_0 GEMV of `m` rows over an `n x k` weight
    /// actually multiplies: int8-quantized on the wide-row GEMV path
    /// (m <= 16, k >= 5120, whole 32-value blocks), as given otherwise or
    /// with ALLPACA_Q8_F32_GEMV=1.
    #[cfg(feature = "hip")]
    fn q8_0_gemv_activations(a: &[f32], m: usize, _n: usize, k: usize) -> Vec<f32> {
        let f32_gemv = std::env::var("ALLPACA_Q8_F32_GEMV").is_ok_and(|v| v == "1");
        if m <= 16 && k >= 5120 && k % 32 == 0 && !f32_gemv {
            q8_activations(a)
        } else {
            a.to_vec()
        }
    }

    #[cfg(feature = "hip")]
    fn gemm_close(got: f32, exp: f32) -> bool {
        let d = (got - exp).abs();
        d <= 1e-4 || d <= 1e-3 * got.abs().max(exp.abs())
    }

    /// Tolerance for GEMVs on int8-quantized activations against an exact
    /// float result: 2% of the output's scale (measured error is ~0.5%).
    fn i8_gemv_close(got: f32, exp: f32, scale: f32) -> bool {
        (got - exp).abs() <= 1e-4 + 0.02 * scale
    }

    fn abs_max(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
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
    fn fill_q4k_gemm_case(
        si: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
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
            let blk =
                &b_q[row * nblocks * Q4_K_BLOCK_BYTES..(row + 1) * nblocks * Q4_K_BLOCK_BYTES];
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
        // Covers every warp-kernel row batch (1, 2, <=4, 8, >8) and n % 8 != 0.
        let shapes = [
            (1, 4, 256),
            (1, 8, 256),
            (2, 3, 256),
            (1, 4, 512),
            (3, 13, 512),
            (8, 9, 256),
            (11, 5, 768),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q, _) = fill_q4k_gemm_case(si, m, n, k);
            let b_deq = dequant_q4k_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&q8_activations(&a), &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q4_k(a, &b_q, c, rows, n as i32, k as i32)
                    .expect("HIP gemm_nt_q4_k must run when a device is present")
            };
            run(&a, &mut hip, m as i32);
            assert_rows_match_single("q4_k", run, &a, &hip, m, n, k);
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
    fn encode_tq2_0(values: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(values.len(), 256);
        let mut out = vec![0u8; TQ2_0_BLOCK_BYTES];
        for group in 0..2 {
            for m in 0..32 {
                let mut packed = 0u8;
                for digit in 0..4 {
                    let trit = values[group * 128 + digit * 32 + m] + 1;
                    assert!((0..=2).contains(&trit));
                    packed |= (trit as u8) << (digit * 2);
                }
                out[group * 32 + m] = packed;
            }
        }
        out[64..66].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        out
    }

    #[cfg(feature = "hip")]
    fn encode_tq1_0(values: &[i8], scale: f32) -> Vec<u8> {
        assert_eq!(values.len(), 256);
        let mut out = vec![0u8; TQ1_0_BLOCK_BYTES];
        let encode_digits = |digits: &[i8]| {
            let mut q = 0u16;
            for &value in digits {
                assert!((-1..=1).contains(&value));
                q = q * 3 + (value + 1) as u16;
            }
            ((q * 256 + 242) / 243) as u8
        };

        for m in 0..32 {
            let digits: [i8; 5] = std::array::from_fn(|digit| values[digit * 32 + m]);
            out[m] = encode_digits(&digits);
        }
        for m in 0..16 {
            let digits: [i8; 5] = std::array::from_fn(|digit| values[160 + digit * 16 + m]);
            out[32 + m] = encode_digits(&digits);
        }
        for m in 0..4 {
            let mut digits = [0i8; 5];
            for digit in 0..4 {
                digits[digit] = values[240 + digit * 4 + m];
            }
            digits[4] = -1;
            out[48 + m] = encode_digits(&digits);
        }
        out[52..54].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
        out
    }

    #[cfg(feature = "hip")]
    fn fill_tq_gemm_case(
        si: usize,
        m: usize,
        n: usize,
        k: usize,
        tq2: bool,
    ) -> (Vec<f32>, Vec<u8>) {
        let mut a = vec![0.0f32; m * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
        }
        let scale = 0.25f32;
        let mut b_q = Vec::new();
        let nblocks = k.div_ceil(256);
        for row in 0..n {
            for ib in 0..nblocks {
                let mut trits = [0i8; 256];
                for j in 0..256 {
                    let idx = row * k + ib * 256 + j;
                    trits[j] = (((idx * 13 + si * 7) % 3) as i8) - 1;
                }
                if tq2 {
                    b_q.extend(encode_tq2_0(&trits, scale));
                } else {
                    b_q.extend(encode_tq1_0(&trits, scale));
                }
            }
        }
        (a, b_q)
    }

    #[cfg(feature = "hip")]
    fn dequant_tq_matrix(b_q: &[u8], n: usize, k: usize, tq2: bool) -> Vec<f32> {
        let (block_elems, block_bytes) = if tq2 {
            (TQ2_0_BLOCK_ELEMS, TQ2_0_BLOCK_BYTES)
        } else {
            (TQ1_0_BLOCK_ELEMS, TQ1_0_BLOCK_BYTES)
        };
        let nblocks = k.div_ceil(block_elems);
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_q[row * nblocks * block_bytes..(row + 1) * nblocks * block_bytes];
            let mut tmp = vec![0.0f32; nblocks * block_elems];
            if tq2 {
                allpaca_cpu::dequant_tq2_0(blk, nblocks * block_elems, &mut tmp).unwrap();
            } else {
                allpaca_cpu::dequant_tq1_0(blk, nblocks * block_elems, &mut tmp).unwrap();
            }
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// Public `gemm_nt_tq2_0` must match CPU dequant + NT GEMM on small TQ2_0 shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_tq2_0_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 256), (1, 8, 256), (2, 3, 256), (1, 4, 512)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q) = fill_tq_gemm_case(si, m, n, k, true);
            let b_deq = dequant_tq_matrix(&b_q, n, k, true);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_tq2_0(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_tq2_0 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_tq2_0 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Public `gemm_nt_tq1_0` must match CPU dequant + NT GEMM on small TQ1_0 shapes.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_tq1_0_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 256), (1, 8, 256), (2, 3, 256), (1, 4, 512)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q) = fill_tq_gemm_case(si, m, n, k, false);
            let b_deq = dequant_tq_matrix(&b_q, n, k, false);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_tq1_0(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_tq1_0 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_tq1_0 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Scattered slot writes through the batch ABI must land exactly where the
    /// per-copy path would put them, on the default stream and on a side stream.
    #[cfg(feature = "hip")]
    #[test]
    fn memcpy_batch_writes_scattered_slots_on_either_stream() {
        if device_count() == 0 {
            return;
        }
        const SLOT: usize = 512;
        const SLOTS: usize = 6;
        let blob = DeviceWeights::alloc(0, SLOT * SLOTS).expect("device blob");
        // Zero the blob so an overrun into an untouched slot is visible.
        blob.copy_h2d(0, &vec![0u8; SLOT * SLOTS])
            .expect("zero blob");
        // Non-adjacent destinations: slots 4, 1, 3 in that order.
        let order = [4usize, 1, 3];
        let chunks: Vec<Vec<u8>> = order
            .iter()
            .enumerate()
            .map(|(i, slot)| vec![(0xA0 + i * 7 + slot) as u8; SLOT])
            .collect();
        let regions: Vec<(usize, &[u8])> = order
            .iter()
            .zip(chunks.iter())
            .map(|(slot, c)| (slot * SLOT, c.as_slice()))
            .collect();

        let h2d = H2dStream::new().expect("side stream");
        for stream in [std::ptr::null_mut(), h2d.raw()] {
            unsafe { blob.copy_h2d_batch_on(&regions, stream) }.expect("batched H2D");
            if stream.is_null() {
                stream_synchronize().expect("default stream sync");
            } else {
                h2d.synchronize().expect("side stream sync");
            }
            let mut back = vec![0u8; SLOT * SLOTS];
            memcpy(
                back.as_mut_ptr(),
                blob.byte_ptr_at(0),
                back.len(),
                MEMCPY_D2H,
            )
            .expect("read back");
            for (i, slot) in order.iter().enumerate() {
                let got = &back[slot * SLOT..(slot + 1) * SLOT];
                assert_eq!(
                    got,
                    chunks[i].as_slice(),
                    "slot {slot} content differs (side_stream={})",
                    !stream.is_null()
                );
            }
            // Untouched slots must stay zero: the batch must not overrun.
            for slot in 0..SLOTS {
                if order.contains(&slot) {
                    continue;
                }
                assert!(
                    back[slot * SLOT..(slot + 1) * SLOT].iter().all(|b| *b == 0),
                    "batch overran into slot {slot}"
                );
            }
        }
        eprintln!(
            "memcpy_batch: hipMemcpyBatchAsync available = {}",
            has_memcpy_batch()
        );
    }

    /// Callers of the batched H2D copy reuse or drop their source as soon as
    /// the call returns (bounce buffers, RAM-tier slots), so the copy must
    /// have read it by then. `hipMemcpyBatchAsync` on the ROCm 10 runtime
    /// reads the source only when the stream runs.
    #[cfg(feature = "hip")]
    #[test]
    fn blas_warm_up_can_run_twice_and_leaves_gemm_correct() {
        if device_count() == 0 {
            return;
        }
        // Not every build links rocBLAS; the GEMM below must be right either way.
        let first = warm_up_blas(0);
        assert_eq!(first.is_ok(), warm_up_blas(0).is_ok());
        let (m, n, k) = (16usize, 8usize, 4usize);
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.5).collect();
        let mut c = vec![0.0f32; m * n];
        gemm_nt(&a, &b, &mut c, m as i32, n as i32, k as i32).unwrap();
        for r in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|x| a[r * k + x] * b[j * k + x]).sum();
                assert!((c[r * n + j] - want).abs() < 1e-4, "row {r} col {j}");
            }
        }
    }

    #[test]
    fn batched_h2d_copy_has_read_its_source_when_it_returns() {
        if device_count() == 0 {
            return;
        }
        const PART: usize = 4 << 20;
        const PARTS: usize = 3;
        let blob = DeviceWeights::alloc(0, PART * PARTS).expect("device blob");
        let want: Vec<u8> = (0..PART * PARTS)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let h2d = H2dStream::new().expect("side stream");
        let mut src = want.clone();
        {
            let regions: Vec<(usize, &[u8])> = (0..PARTS)
                .map(|p| (p * PART, &src[p * PART..(p + 1) * PART]))
                .collect();
            unsafe { blob.copy_h2d_batch_on(&regions, h2d.raw()) }.expect("batched H2D");
        }
        src.fill(0x11);
        h2d.synchronize().expect("side stream sync");
        let mut back = vec![0u8; PART * PARTS];
        memcpy(
            back.as_mut_ptr(),
            blob.byte_ptr_at(0),
            back.len(),
            MEMCPY_D2H,
        )
        .expect("read back");
        let wrong = back.iter().zip(&want).filter(|(a, b)| a != b).count();
        assert_eq!(wrong, 0, "{wrong} bytes came from the source after it was reused");
    }

    /// Driver VRAM must be self-consistent and must react to a real allocation.
    /// This is the signal `rebalance_vram` grows the expert pool against, so a
    /// stale or process-local number would be worse than none.
    #[cfg(feature = "hip")]
    #[test]
    fn mem_info_reports_board_free_that_shrinks_on_allocation() {
        if device_count() == 0 {
            return;
        }
        let (free_before, total) = device_memory_info(0).expect("mem_info");
        assert!(total > 0, "total VRAM must be known");
        assert!(free_before <= total, "free {free_before} > total {total}");

        const CHUNK: usize = 256 << 20;
        if free_before as usize <= CHUNK * 2 {
            return; // Board is already too full to measure a delta safely.
        }
        // Board-wide readings: other tests in this binary (and other
        // processes) allocate and free meanwhile, so take the best of a few.
        let dropped = (0..5).any(|_| {
            let (before, _) = device_memory_info(0).expect("mem_info");
            let hog = DeviceWeights::alloc(0, CHUNK).expect("device alloc");
            let (after, total_after) = device_memory_info(0).expect("mem_info after alloc");
            assert_eq!(total_after, total, "total VRAM must not move");
            drop(hog);
            after < before
        });
        assert!(dropped, "free VRAM never dropped after a {CHUNK}B alloc");

        // `used` is the same reading; the two entry points must not disagree.
        let (used, total_now, skew) = (0..5)
            .map(|_| {
                let used = device_memory_used(0).expect("mem_used");
                let (free_now, total_now) = device_memory_info(0).expect("mem_info");
                (used, total_now, (total_now - free_now).abs_diff(used))
            })
            .min_by_key(|r| r.2)
            .unwrap();
        assert_eq!(total_now, total_now.max(used), "used must fit in total");
        assert!(skew <= 64 << 20, "mem_used {used} and total-free disagree by {skew}");
    }

    /// The sampler runs on threads that are about to launch kernels, so the
    /// driver read has to be cheap once its context exists. Prints the first
    /// call separately because that one pays for context creation.
    #[cfg(feature = "hip")]
    #[test]
    fn mem_info_is_cheap_after_the_first_call() {
        if device_count() == 0 {
            return;
        }
        for dev in 0..device_count() {
            let t0 = std::time::Instant::now();
            let ok = device_memory_info(dev).is_ok();
            let first = t0.elapsed();
            let t1 = std::time::Instant::now();
            for _ in 0..100 {
                let _ = device_memory_info(dev);
            }
            let per_call = t1.elapsed() / 100;
            eprintln!("mem_info dev{dev}: ok={ok} first={first:?} steady={per_call:?}");
            assert!(
                per_call < std::time::Duration::from_millis(1),
                "dev{dev} steady-state mem_info {per_call:?} is too slow to sample at 10 Hz"
            );
        }
    }

    /// An out-of-range region must be refused before any copy is issued.
    #[cfg(feature = "hip")]
    #[test]
    fn memcpy_batch_rejects_region_past_end_of_blob() {
        if device_count() == 0 {
            return;
        }
        let blob = DeviceWeights::alloc(0, 256).expect("device blob");
        let src = vec![1u8; 128];
        let bad = [(0usize, src.as_slice()), (200usize, src.as_slice())];
        let err = unsafe { blob.copy_h2d_batch_on(&bad, std::ptr::null_mut()) }
            .expect_err("region past end must be refused");
        assert!(err.to_string().contains("overflow"), "got {err}");
    }

    #[cfg(feature = "hip")]
    fn fill_rocmfp4_gemm_case(si: usize, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>) {
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
            b_q.extend(allpaca_cpu::quantize_rocmfp4(&b_f[row * k..(row + 1) * k]));
        }
        (a, b_q)
    }

    #[cfg(feature = "hip")]
    fn dequant_rocmfp4_matrix(b_q: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(ROCMFP4_BLOCK_ELEMS);
        let row_bytes = nblocks * ROCMFP4_BLOCK_BYTES;
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_q[row * row_bytes..(row + 1) * row_bytes];
            let mut tmp = vec![0.0f32; nblocks * ROCMFP4_BLOCK_ELEMS];
            allpaca_cpu::dequant_rocmfp4(blk, nblocks * ROCMFP4_BLOCK_ELEMS, &mut tmp).unwrap();
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// The fused ROCmFP4 GEMV must match CPU dequant + NT GEMM. Shapes cross the
    /// 64-column tile and the 256-float LDS slab so both kernel edges are hit.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_rocmfp4_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (1, 4, 32),
            (1, 8, 256),
            (2, 3, 128),
            (1, 70, 512),
            (1, 4, 288),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q) = fill_rocmfp4_gemm_case(si, m, n, k);
            let b_deq = dequant_rocmfp4_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_rocmfp4(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_rocmfp4 must run when a device is present");
            let scale = abs_max(&cpu);
            for i in 0..m * n {
                assert!(
                    i8_gemv_close(hip[i], cpu[i], scale),
                    "gemm_nt_rocmfp4 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// ROCmFP4 FAST (type 101) GEMV: bit-identical to the dual-scale kernel on
    /// the same weights with the scale byte duplicated, and close to the CPU
    /// dequant + GEMM. Shapes cover one row, several rows, a K that is not a
    /// whole number of 256-float slabs, and many columns.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_p48fp4_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        for (si, (m, n, k)) in [(1, 4, 32), (1, 8, 256), (3, 5, 128), (1, 70, 512), (2, 9, 288), (9, 33, 768)]
            .iter()
            .copied()
            .enumerate()
        {
            let a: Vec<f32> = (0..m * k).map(|i| ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5).collect();
            let b_f: Vec<f32> = (0..n * k).map(|i| ((i * 13 + si * 7) % 61) as f32 * 0.02 - 0.6).collect();
            let packed = allpaca_cpu::quantize_p48fp4(&b_f);
            assert_eq!(packed.len(), n * k / 32 * allpaca_cpu::P48FP4_BLOCK_BYTES);
            let mut b_deq = vec![0.0f32; n * k];
            allpaca_cpu::dequant_p48fp4(&packed, n * k, &mut b_deq).unwrap();
            // Half of every 8 weights are pruned, so the result differs from dense.
            assert!(b_deq.chunks(8).all(|c| c.iter().filter(|v| **v == 0.0).count() >= 4));
            let mut cpu = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            let da = device_malloc(a.len() * 4, 0).unwrap();
            let db = device_malloc(packed.len(), 0).unwrap();
            let dc = device_malloc(m * n * 4, 0).unwrap();
            set_device(0).unwrap();
            memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).unwrap();
            memcpy(db, packed.as_ptr(), packed.len(), MEMCPY_H2D).unwrap();
            gemm_nt_p48fp4_device(da.cast(), db, dc.cast(), m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_p48fp4 must run when a device is present");
            let mut got = vec![0.0f32; m * n];
            memcpy(got.as_mut_ptr().cast(), dc, m * n * 4, MEMCPY_D2H).unwrap();
            // A row computed alone equals the same row computed in the batch.
            let mut solo = vec![0.0f32; n];
            gemm_nt_p48fp4_device(
                unsafe { da.cast::<f32>().add((m - 1) * k) },
                db,
                dc.cast(),
                1,
                n as i32,
                k as i32,
            )
            .unwrap();
            memcpy(solo.as_mut_ptr().cast(), dc, n * 4, MEMCPY_D2H).unwrap();
            device_free(da, a.len() * 4);
            device_free(db, packed.len());
            device_free(dc, m * n * 4);
            for i in 0..m * n {
                assert!(
                    gemm_close(got[i], cpu[i]),
                    "gemm_nt_p48fp4 vs cpu, shape {m}x{n}x{k} idx={i}: {} vs {}",
                    got[i],
                    cpu[i]
                );
            }
            for j in 0..n {
                assert_eq!(got[(m - 1) * n + j].to_bits(), solo[j].to_bits(), "row alone differs, col {j}");
            }
        }
        // A row length that is not whole blocks is refused.
        assert!(gemm_nt_p48fp4_device(std::ptr::NonNull::dangling().as_ptr(), std::ptr::NonNull::dangling().as_ptr(), std::ptr::NonNull::dangling().as_ptr(), 1, 4, 40).is_err());
    }

    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_rocmfp4_fast_matches_dual_scale_kernel_and_cpu() {
        if device_count() == 0 {
            return;
        }
        for (si, (m, n, k)) in [(1, 4, 32), (1, 8, 256), (3, 5, 128), (1, 70, 512), (2, 9, 288)]
            .iter()
            .copied()
            .enumerate()
        {
            let mut a = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; n * k];
            for (i, v) in a.iter_mut().enumerate() {
                *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in b_f.iter_mut().enumerate() {
                *v = ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5;
            }
            let nblocks = k.div_ceil(ROCMFP4_BLOCK_ELEMS);
            let (mut fast, mut dual) = (Vec::new(), Vec::new());
            let mut b_deq = vec![0.0f32; n * k];
            for row in 0..n {
                let q = allpaca_cpu::quantize_rocmfp4_fast(&b_f[row * k..(row + 1) * k]);
                assert_eq!(q.len(), nblocks * allpaca_cpu::ROCMFP4_FAST_BLOCK_BYTES);
                let mut tmp = vec![0.0f32; nblocks * ROCMFP4_BLOCK_ELEMS];
                allpaca_cpu::dequant_rocmfp4_fast(&q, tmp.len(), &mut tmp).unwrap();
                b_deq[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
                for blk in q.chunks_exact(allpaca_cpu::ROCMFP4_FAST_BLOCK_BYTES) {
                    dual.extend_from_slice(blk);
                    dual.push(blk[16]);
                }
                fast.extend(q);
            }
            let mut cpu = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            let mut want = vec![0.0f32; m * n];
            gemm_nt_rocmfp4(&a, &dual, &mut want, m as i32, n as i32, k as i32).unwrap();

            let da = device_malloc(a.len() * 4, 0).unwrap();
            let db = device_malloc(fast.len(), 0).unwrap();
            let dc = device_malloc(m * n * 4, 0).unwrap();
            set_device(0).unwrap();
            memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).unwrap();
            memcpy(db, fast.as_ptr(), fast.len(), MEMCPY_H2D).unwrap();
            gemm_nt_rocmfp4_fast_device(da.cast(), db, dc.cast(), m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_rocmfp4_fast must run when a device is present");
            let mut got = vec![0.0f32; m * n];
            memcpy(got.as_mut_ptr().cast(), dc, m * n * 4, MEMCPY_D2H).unwrap();
            device_free(da, a.len() * 4);
            device_free(db, fast.len());
            device_free(dc, m * n * 4);

            for i in 0..m * n {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "FAST vs dual-scale kernel differ, shape {m}x{n}x{k} idx={i}: {} vs {}",
                    got[i],
                    want[i]
                );
                assert!(
                    i8_gemv_close(got[i], cpu[i], abs_max(&cpu)),
                    "gemm_nt_rocmfp4_fast vs cpu, shape {m}x{n}x{k} idx={i}: {} vs {}",
                    got[i],
                    cpu[i]
                );
            }
        }
    }

    /// ROCmFP4 FAST embed rows (single token and batched) equal the CPU
    /// dequant of those rows exactly, for a dim that is not a multiple of 32.
    #[cfg(feature = "hip")]
    #[test]
    fn embed_gather_rocmfp4_fast_matches_dequant_rows() {
        if device_count() == 0 {
            return;
        }
        let (vocab, dim) = (9usize, 80usize);
        let row_bytes = dim.div_ceil(32) * allpaca_cpu::ROCMFP4_FAST_BLOCK_BYTES;
        let tokens = [0u32, 3, 8, 1, 3, 2];
        let mut packed = Vec::new();
        for row in 0..vocab {
            let vals: Vec<f32> =
                (0..dim).map(|i| (((row * dim + i) * 17) % 50) as f32 * 0.02 - 0.5).collect();
            packed.extend(allpaca_cpu::quantize_rocmfp4_fast(&vals));
        }
        assert_eq!(packed.len(), vocab * row_bytes);
        let mut cpu = vec![0.0f32; tokens.len() * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            let off = tok as usize * row_bytes;
            let mut tmp = vec![0.0f32; dim.div_ceil(32) * 32];
            allpaca_cpu::dequant_rocmfp4_fast(&packed[off..off + row_bytes], tmp.len(), &mut tmp)
                .unwrap();
            cpu[r * dim..(r + 1) * dim].copy_from_slice(&tmp[..dim]);
        }
        let ws = HipWorkspace::new(0, 1 << 20, 64).unwrap();
        let dt = device_malloc(packed.len(), 0).unwrap();
        let dout = device_malloc(cpu.len() * 4, 0).unwrap();
        set_device(0).unwrap();
        memcpy(dt, packed.as_ptr(), packed.len(), MEMCPY_H2D).unwrap();
        ws.embed_gather_rocmfp4_fast_rows(dt, &tokens, dout.cast(), dim as i32)
            .expect("HIP embed_gather_rocmfp4_fast_rows must run when a device is present");
        let mut rows = vec![0.0f32; cpu.len()];
        memcpy(rows.as_mut_ptr().cast(), dout, cpu.len() * 4, MEMCPY_D2H).unwrap();
        assert_eq!(rows, cpu, "batched FAST embed rows");
        ws.embed_gather_rocmfp4_fast(dt, 8, dout.cast(), dim as i32).unwrap();
        let mut one = vec![0.0f32; dim];
        memcpy(one.as_mut_ptr().cast(), dout, dim * 4, MEMCPY_D2H).unwrap();
        assert_eq!(one, cpu[2 * dim..3 * dim], "single FAST embed row");
        device_free(dt, packed.len());
        device_free(dout, cpu.len() * 4);
    }

    /// The int8-activation ROCmFP4 GEMV (types 100 and 101) at a Qwen3.8
    /// shape: within int8 activation error of the float kernel, every row of
    /// an m-row call bit-identical to that row run alone (so verify and
    /// batched rows reproduce decode), and faster than the float kernel, far
    /// more so for several rows.
    #[cfg(feature = "hip")]
    #[test]
    fn rocmfp4_i8_gemv_matches_float_kernel_and_is_row_exact() {
        if device_count() == 0 {
            return;
        }
        let (n, k) = (2048usize, 5120usize);
        for fast in [false, true] {
            let bb = if fast { 17 } else { ROCMFP4_BLOCK_BYTES };
            let max_m = 8usize;
            let mut a = vec![0.0f32; max_m * k];
            let mut seed = 0x2545_F491_4F6C_DD1Du64;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            };
            for v in a.iter_mut() {
                *v = rnd() * 2.0;
            }
            let mut b_q = Vec::with_capacity(n * (k / 32) * bb);
            let mut row = vec![0.0f32; k];
            for _ in 0..n {
                for v in row.iter_mut() {
                    *v = rnd() * 0.1;
                }
                if fast {
                    b_q.extend(allpaca_cpu::quantize_rocmfp4_fast(&row));
                } else {
                    b_q.extend(allpaca_cpu::quantize_rocmfp4(&row));
                }
            }
            let da = device_malloc(a.len() * 4, 0).unwrap();
            let db = device_malloc(b_q.len(), 0).unwrap();
            let dc = device_malloc(max_m * n * 4, 0).unwrap();
            set_device(0).unwrap();
            memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).unwrap();
            memcpy(db, b_q.as_ptr(), b_q.len(), MEMCPY_H2D).unwrap();
            let i8k = |a: *const f32, c: *mut f32, m: usize| {
                if fast {
                    gemm_nt_rocmfp4_fast_device(a, db, c, m as i32, n as i32, k as i32)
                } else {
                    gemm_nt_rocmfp4_device(a, db, c, m as i32, n as i32, k as i32)
                }
                .unwrap()
            };
            let f32k = |a: *const f32, c: *mut f32, m: usize| {
                gemm_nt_rocmfp4_f32_device(a, db, c, m as i32, n as i32, k as i32, fast).unwrap()
            };
            let read = |m: usize| {
                let mut v = vec![0.0f32; m * n];
                memcpy(v.as_mut_ptr().cast(), dc, m * n * 4, MEMCPY_D2H).unwrap();
                v
            };
            // Each row alone (m == 1) is the reference for every batch size.
            let mut single = Vec::new();
            for r in 0..max_m {
                i8k(unsafe { da.cast::<f32>().add(r * k) }, dc.cast(), 1);
                single.extend(read(1));
            }
            for m in [2usize, 3, 4, 5, 8] {
                i8k(da.cast(), dc.cast(), m);
                let got = read(m);
                for i in 0..m * n {
                    assert_eq!(
                        got[i].to_bits(),
                        single[i].to_bits(),
                        "fast={fast} m={m}: row {} col {} differs from its m=1 result",
                        i / n,
                        i % n
                    );
                }
            }
            f32k(da.cast(), dc.cast(), max_m);
            let float = read(max_m);
            let scale = float.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            let worst = single
                .iter()
                .zip(&float)
                .fold(0.0f32, |w, (a, b)| w.max((a - b).abs()));
            assert!(
                worst <= 0.02 * scale,
                "fast={fast}: int8 GEMV off by {worst} against the float kernel (output scale {scale})"
            );
            // Best of five: other tests share the GPU, and a single window
            // can land on their kernels.
            let time = |f: &dyn Fn(*const f32, *mut f32, usize), m: usize| {
                f(da.cast(), dc.cast(), m);
                stream_synchronize().unwrap();
                (0..5)
                    .map(|_| {
                        let t = std::time::Instant::now();
                        for _ in 0..20 {
                            f(da.cast(), dc.cast(), m);
                        }
                        stream_synchronize().unwrap();
                        t.elapsed().as_micros() / 20
                    })
                    .min()
                    .unwrap()
            };
            let (i1, f1) = (time(&i8k, 1), time(&f32k, 1));
            let (i4, f4) = (time(&i8k, 4), time(&f32k, 4));
            eprintln!(
                "rocmfp4 gemv n={n} k={k} fast={fast}: m=1 int8 {i1}us float {f1}us; m=4 int8 {i4}us float {f4}us; max err {:.3}% of scale",
                100.0 * worst / scale
            );
            // Typically 2-4x; the margin is loose because WDDM timing on a shared
            // GPU swings by 2x between windows.
            assert!(i4 < f4, "fast={fast}: 4-row int8 GEMV {i4}us not faster than float {f4}us");
            device_free(da, a.len() * 4);
            device_free(db, b_q.len());
            device_free(dc, max_m * n * 4);
        }
    }

    /// K that is not a whole number of 32-weight blocks cannot use the int8
    /// path: the decode entry point falls back to the float kernel and
    /// matches the CPU dequant exactly-float.
    #[cfg(feature = "hip")]
    #[test]
    fn rocmfp4_partial_block_k_uses_float_kernel() {
        if device_count() == 0 {
            return;
        }
        let (m, n, k) = (2usize, 9usize, 300usize);
        let (a, b_q) = fill_rocmfp4_gemm_case(3, m, n, k);
        let b_deq = dequant_rocmfp4_matrix(&b_q, n, k);
        let mut cpu = vec![0.0f32; m * n];
        let mut hip = vec![0.0f32; m * n];
        cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
        gemm_nt_rocmfp4(&a, &b_q, &mut hip, m as i32, n as i32, k as i32).unwrap();
        for i in 0..m * n {
            assert!(gemm_close(hip[i], cpu[i]), "idx={i} hip={} cpu={}", hip[i], cpu[i]);
        }
    }

    /// The §12 tiled ROCmFP4 GEMV must agree with the untiled baseline.
    #[cfg(feature = "hip")]
    #[test]
    fn rocmfp4_tiled_gemv_matches_untiled() {
        if device_count() == 0 {
            return;
        }
        let (m, n, k) = (1usize, 96usize, 512usize);
        let (a, b_q) = fill_rocmfp4_gemm_case(5, m, n, k);
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_rocmfp4(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).expect("tiled ROCmFP4");
        gemm_nt_rocmfp4_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32)
            .expect("untiled ROCmFP4");
        for i in 0..m * n {
            assert!(
                i8_gemv_close(tiled[i], untiled[i], abs_max(&untiled)),
                "ROCmFP4 tiled/untiled mismatch idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
    }

    /// FFN-sized decode GEMV must match untiled and beat it.
    #[cfg(feature = "hip")]
    #[test]
    fn rocmfp4_decode_gemv_matches_and_beats_untiled() {
        if device_count() == 0 {
            return;
        }
        let (m, n, k) = (1usize, 4096usize, 5120usize);
        let (a, b_q) = fill_rocmfp4_gemm_case(9, m, n, k);
        let mut got = vec![0.0f32; n];
        let mut expect = vec![0.0f32; n];
        gemm_nt_rocmfp4(&a, &b_q, &mut got, 1, n as i32, k as i32).expect("lds");
        gemm_nt_rocmfp4_untiled(&a, &b_q, &mut expect, 1, n as i32, k as i32).expect("untiled");
        for i in 0..n {
            assert!(
                i8_gemv_close(got[i], expect[i], abs_max(&expect)),
                "lds mismatch idx={i} lds={} untiled={}",
                got[i],
                expect[i]
            );
        }
        #[cfg(all(feature = "hip", allpaca_hip_runtime))]
        {
            let a_n = m * k;
            let nblocks = k.div_ceil(32);
            let packed = n * nblocks * 18;
            let da = device_malloc(a_n * 4, 0).unwrap();
            let db = device_malloc(packed, 0).unwrap();
            let dc = device_malloc(n * 4, 0).unwrap();
            set_device(0).unwrap();
            memcpy(da, a.as_ptr().cast(), a_n * 4, MEMCPY_H2D).unwrap();
            memcpy(db, b_q.as_ptr(), packed, MEMCPY_H2D).unwrap();
            let time = |kernel: fn(*const f32, *const u8, *mut f32, i32, i32, i32) -> Result<(), AllpacaError>| {
                kernel(da.cast(), db, dc.cast(), 1, n as i32, k as i32).unwrap();
                stream_synchronize().unwrap();
                let t = std::time::Instant::now();
                for _ in 0..20 {
                    kernel(da.cast(), db, dc.cast(), 1, n as i32, k as i32).unwrap();
                }
                stream_synchronize().unwrap();
                t.elapsed().as_micros() / 20
            };
            let lds_us = time(gemm_nt_rocmfp4_device);
            let old_us = time(gemm_nt_rocmfp4_untiled_device);
            device_free(da, a_n * 4);
            device_free(db, packed);
            device_free(dc, n * 4);
            eprintln!("rocmfp4 gemv n={n} k={k} decode_us={lds_us} untiled_us={old_us}");
            assert!(
                lds_us * 10 < old_us * 8,
                "decode GEMV {lds_us}us not faster than untiled {old_us}us"
            );
        }
    }

    #[cfg(feature = "hip")]
    fn fill_mxfp4_gemm_case(si: usize, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<u8>) {
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
            b_q.extend(allpaca_cpu::quantize_mxfp4(&b_f[row * k..(row + 1) * k]));
        }
        (a, b_q)
    }

    #[cfg(feature = "hip")]
    fn dequant_mxfp4_matrix(b_q: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(MXFP4_BLOCK_ELEMS);
        let row_bytes = nblocks * MXFP4_BLOCK_BYTES;
        let mut out = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_q[row * row_bytes..(row + 1) * row_bytes];
            let mut tmp = vec![0.0f32; nblocks * MXFP4_BLOCK_ELEMS];
            allpaca_cpu::dequant_mxfp4(blk, nblocks * MXFP4_BLOCK_ELEMS, &mut tmp).unwrap();
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// The fused MXFP4 GEMV must match CPU dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_mxfp4_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (1, 4, 32),
            (1, 8, 256),
            (2, 3, 128),
            (1, 70, 512),
            (1, 4, 288),
            (24, 96, 512),
            (40, 70, 288),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q) = fill_mxfp4_gemm_case(si, m, n, k);
            let b_deq = dequant_mxfp4_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_mxfp4(&a, &b_q, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_mxfp4 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_mxfp4 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// The tiled MXFP4 GEMV must agree with the untiled baseline.
    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_tiled_gemv_matches_untiled() {
        if device_count() == 0 {
            return;
        }
        let (m, n, k) = (1usize, 96usize, 512usize);
        let (a, b_q) = fill_mxfp4_gemm_case(5, m, n, k);
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_mxfp4(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).expect("tiled MXFP4");
        gemm_nt_mxfp4_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32)
            .expect("untiled MXFP4");
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "MXFP4 tiled/untiled mismatch idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
    }

    /// M>1 tiled GEMM reuses MXFP4 B across rows and must match the untiled baseline.
    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_tiled_gemm_m_matches_untiled() {
        if device_count() == 0 {
            return;
        }
        for (m, n, k, seed) in [
            (8usize, 96usize, 512usize, 6usize),
            (3usize, 64usize, 256usize, 7usize),
            (9usize, 96usize, 512usize, 8usize),
            (24usize, 96usize, 512usize, 9usize),
            (32usize, 64usize, 256usize, 10usize),
            (40usize, 70usize, 288usize, 11usize),
        ] {
            let (a, b_q) = fill_mxfp4_gemm_case(seed, m, n, k);
            let mut tiled = vec![0.0f32; m * n];
            let mut untiled = vec![0.0f32; m * n];
            gemm_nt_mxfp4(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).expect("tiled M");
            gemm_nt_mxfp4_untiled(&a, &b_q, &mut untiled, m as i32, n as i32, k as i32)
                .expect("untiled M");
            for i in 0..m * n {
                assert!(
                    gemm_close(tiled[i], untiled[i]),
                    "MXFP4 tiled-M/untiled mismatch m={m} idx={i} tiled={} untiled={}",
                    tiled[i],
                    untiled[i]
                );
            }
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_swiglu_gate_up_matches_two_gemms() {
        if device_count() == 0 {
            return;
        }
        for (m, n, k, seed) in [
            (2usize, 64usize, 128usize, 3usize),
            (24usize, 96usize, 256usize, 13usize),
            (40usize, 70usize, 288usize, 14usize),
        ] {
            let (a, bg) = fill_mxfp4_gemm_case(seed, m, n, k);
            let (_, bu) = fill_mxfp4_gemm_case(seed + 6, m, n, k);
            let bg_deq = dequant_mxfp4_matrix(&bg, n, k);
            let bu_deq = dequant_mxfp4_matrix(&bu, n, k);
            let mut gate = vec![0.0f32; m * n];
            let mut up = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &bg_deq, &mut gate, m, n, k);
            cpu_gemm_nt(&a, &bu_deq, &mut up, m, n, k);
            let mut expect = vec![0.0f32; m * n];
            for i in 0..m * n {
                let g = gate[i];
                expect[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            let mut got = vec![0.0f32; m * n];
            super::mxfp4_swiglu_gate_up(&a, &bg, &bu, &mut got, m as i32, n as i32, k as i32, 0.0)
                .expect("fused MXFP4 gate-up");
            for i in 0..m * n {
                assert!(
                    gemm_close(got[i], expect[i]),
                    "fused gate-up mismatch m={m} idx={i} hip={} cpu={}",
                    got[i],
                    expect[i]
                );
            }
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_swiglu_caps_gate_and_up_at_the_limit() {
        if device_count() == 0 {
            return;
        }
        let limit = 0.25f32;
        // One row (the wide kernel), a few rows and many rows (the tiled ones).
        for (m, n, k, seed) in [
            (1usize, 64usize, 256usize, 5usize),
            (2, 64, 128, 3),
            (24, 96, 256, 13),
        ] {
            let (a, bg) = fill_mxfp4_gemm_case(seed, m, n, k);
            let (_, bu) = fill_mxfp4_gemm_case(seed + 6, m, n, k);
            let mut gate = vec![0.0f32; m * n];
            let mut up = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &dequant_mxfp4_matrix(&bg, n, k), &mut gate, m, n, k);
            cpu_gemm_nt(&a, &dequant_mxfp4_matrix(&bu, n, k), &mut up, m, n, k);
            let capped = (0..m * n).filter(|&i| gate[i] > limit || up[i].abs() > limit).count();
            assert!(capped > 0, "case m={m} never reaches the limit");
            let mut got = vec![0.0f32; m * n];
            super::mxfp4_swiglu_gate_up(&a, &bg, &bu, &mut got, m as i32, n as i32, k as i32, limit)
                .expect("fused MXFP4 gate-up");
            for i in 0..m * n {
                let g = gate[i].min(limit);
                let want = (g / (1.0 + (-g).exp())) * up[i].clamp(-limit, limit);
                assert!(
                    gemm_close(got[i], want),
                    "capped gate-up mismatch m={m} idx={i} hip={} cpu={want}",
                    got[i]
                );
            }
        }
        // A workspace applies its limit to the whole expert, as the CPU kernel does.
        let (m, dim, ff) = (2usize, 64usize, 96usize);
        let (x, gate_q) = fill_mxfp4_gemm_case(4, m, ff, dim);
        let (_, up_q) = fill_mxfp4_gemm_case(8, m, ff, dim);
        let (_, down_q) = fill_mxfp4_gemm_case(12, m, dim, ff);
        let expert = allpaca_cpu::Mxfp4Expert {
            gate: &gate_q,
            up: &up_q,
            down: &down_q,
            x: &x,
            m,
        };
        let want = allpaca_cpu::mxfp4_swiglu_experts(&[expert], dim, ff, limit, 1).remove(0);
        let free = allpaca_cpu::mxfp4_swiglu_experts(
            &[allpaca_cpu::Mxfp4Expert {
                gate: &gate_q,
                up: &up_q,
                down: &down_q,
                x: &x,
                m,
            }],
            dim,
            ff,
            0.0,
            1,
        )
        .remove(0);
        assert!(want.iter().zip(&free).any(|(a, b)| (a - b).abs() > 1e-3), "limit never reached");
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("hip");
        hip.set_mxfp4_swiglu_limit(limit);
        let gb = device_malloc(gate_q.len(), 0).expect("gate");
        let ub = device_malloc(up_q.len(), 0).expect("up");
        let db = device_malloc(down_q.len(), 0).expect("down");
        set_device(0).expect("bind");
        memcpy(gb, gate_q.as_ptr(), gate_q.len(), MEMCPY_H2D).expect("h2d gate");
        memcpy(ub, up_q.as_ptr(), up_q.len(), MEMCPY_H2D).expect("h2d up");
        memcpy(db, down_q.as_ptr(), down_q.len(), MEMCPY_H2D).expect("h2d down");
        let mut got = vec![0.0f32; m * dim];
        hip.mxfp4_swiglu_gemm(&x, gb, ub, db, &mut got, m as i32, dim as i32, ff as i32)
            .expect("swiglu gemm");
        device_free(gb, gate_q.len());
        device_free(ub, up_q.len());
        device_free(db, down_q.len());
        for i in 0..m * dim {
            assert!(
                gemm_close(got[i], want[i]),
                "capped swiglu mismatch idx={i} hip={} cpu={}",
                got[i],
                want[i]
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_swiglu_gemm_matches_three_gemms() {
        if device_count() == 0 {
            return;
        }
        let m = 2usize;
        let dim = 64usize;
        let ff = 96usize;
        let (x, gate_q) = fill_mxfp4_gemm_case(4, m, ff, dim);
        let (_, up_q) = fill_mxfp4_gemm_case(8, m, ff, dim);
        let (_, down_q) = fill_mxfp4_gemm_case(12, m, dim, ff);
        let mut gate = vec![0.0f32; m * ff];
        let mut up = vec![0.0f32; m * ff];
        gemm_nt_mxfp4(&x, &gate_q, &mut gate, m as i32, ff as i32, dim as i32).expect("gate");
        gemm_nt_mxfp4(&x, &up_q, &mut up, m as i32, ff as i32, dim as i32).expect("up");
        for i in 0..m * ff {
            let g = gate[i];
            gate[i] = (g / (1.0 + (-g).exp())) * up[i];
        }
        let mut expect = vec![0.0f32; m * dim];
        gemm_nt_mxfp4(&gate, &down_q, &mut expect, m as i32, dim as i32, ff as i32).expect("down");
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("hip");
        let gb = device_malloc(gate_q.len(), 0).expect("gate");
        let ub = device_malloc(up_q.len(), 0).expect("up");
        let db = device_malloc(down_q.len(), 0).expect("down");
        set_device(0).expect("bind");
        memcpy(gb, gate_q.as_ptr(), gate_q.len(), MEMCPY_H2D).expect("h2d gate");
        memcpy(ub, up_q.as_ptr(), up_q.len(), MEMCPY_H2D).expect("h2d up");
        memcpy(db, down_q.as_ptr(), down_q.len(), MEMCPY_H2D).expect("h2d down");
        let mut got = vec![0.0f32; m * dim];
        hip.mxfp4_swiglu_gemm(&x, gb, ub, db, &mut got, m as i32, dim as i32, ff as i32)
            .expect("swiglu gemm");
        device_free(gb, gate_q.len());
        device_free(ub, up_q.len());
        device_free(db, down_q.len());
        for i in 0..m * dim {
            assert!(
                gemm_close(got[i], expect[i]),
                "mxfp4 swiglu mismatch idx={i} hip={} three-gemm={}",
                got[i],
                expect[i]
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_swiglu_pipelined_matches_serial() {
        if device_count() == 0 {
            return;
        }
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("hip");
        let n = hip.swiglu_pipeline_slots();
        if n < 2 {
            return;
        }
        let m = 1usize;
        let dim = 64usize;
        let ff = 96usize;
        set_device(0).expect("bind");
        let mut xs = Vec::new();
        let mut packed = Vec::new();
        let mut dev = Vec::new();
        let mut serial = Vec::new();
        for i in 0..n {
            let (x, g) = fill_mxfp4_gemm_case(i * 3 + 1, m, ff, dim);
            let (_, u) = fill_mxfp4_gemm_case(i * 3 + 2, m, ff, dim);
            let (_, d) = fill_mxfp4_gemm_case(i * 3 + 3, m, dim, ff);
            let gd = device_malloc(g.len(), 0).expect("g");
            let ud = device_malloc(u.len(), 0).expect("u");
            let dd = device_malloc(d.len(), 0).expect("d");
            memcpy(gd, g.as_ptr(), g.len(), MEMCPY_H2D).expect("h2d g");
            memcpy(ud, u.as_ptr(), u.len(), MEMCPY_H2D).expect("h2d u");
            memcpy(dd, d.as_ptr(), d.len(), MEMCPY_H2D).expect("h2d d");
            let mut y = vec![0.0f32; m * dim];
            hip.mxfp4_swiglu_gemm(&x, gd, ud, dd, &mut y, m as i32, dim as i32, ff as i32)
                .expect("serial");
            xs.push(x);
            packed.push((g, u, d));
            dev.push((gd, ud, dd));
            serial.push(y);
        }
        for i in 0..n {
            hip.mxfp4_swiglu_enqueue(
                i, &xs[i], dev[i].0, dev[i].1, dev[i].2, m as i32, dim as i32, ff as i32,
            )
            .expect("enqueue");
        }
        for i in 0..n {
            let mut pipe = vec![0.0f32; m * dim];
            hip.mxfp4_swiglu_collect(i, &mut pipe, m as i32, dim as i32, ff as i32)
                .expect("collect");
            for j in 0..m * dim {
                assert!(
                    gemm_close(pipe[j], serial[i][j]),
                    "pipeline {i} idx={j} pipe={} serial={}",
                    pipe[j],
                    serial[i][j]
                );
            }
        }
        for i in 0..n {
            device_free(dev[i].0, packed[i].0.len());
            device_free(dev[i].1, packed[i].1.len());
            device_free(dev[i].2, packed[i].2.len());
        }
    }

    /// Overlapping token rows must atomic-add the scaled packed source into acc.
    #[cfg(all(feature = "hip", allpaca_hip_runtime))]
    #[test]
    fn scatter_add_scale_rows_matches_cpu() {
        if device_count() == 0 {
            return;
        }
        let n_tok = 3usize;
        let dim = 32usize;
        let m = 3usize;
        let mut src = vec![0.0f32; m * dim];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 0.2;
        }
        let idx = [1i32, 1, 0];
        let w = [0.5f32, 0.25, 2.0];
        let mut expect = vec![0.0f32; n_tok * dim];
        for row in 0..m {
            let t = idx[row] as usize;
            for j in 0..dim {
                expect[t * dim + j] += src[row * dim + j] * w[row];
            }
        }
        set_device(0).expect("bind");
        let src_d = device_malloc(src.len() * 4, 0).expect("src");
        let idx_d = device_malloc(idx.len() * 4, 0).expect("idx");
        let w_d = device_malloc(w.len() * 4, 0).expect("w");
        let acc_d = device_malloc(expect.len() * 4, 0).expect("acc");
        memcpy(src_d, src.as_ptr().cast(), src.len() * 4, MEMCPY_H2D).expect("h2d src");
        memcpy(idx_d, idx.as_ptr().cast(), idx.len() * 4, MEMCPY_H2D).expect("h2d idx");
        memcpy(w_d, w.as_ptr().cast(), w.len() * 4, MEMCPY_H2D).expect("h2d w");
        super::memset_on(acc_d, 0, expect.len() * 4, std::ptr::null_mut()).expect("zero");
        stream_synchronize().expect("sync zero");
        let rc = unsafe {
            super::ffi::allpaca_scatter_add_scale_rows(
                src_d.cast(),
                idx_d.cast(),
                w_d.cast(),
                acc_d.cast(),
                m as i32,
                dim as i32,
                n_tok as i32,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, 0, "scatter_add_scale_rows rc={rc}");
        let mut got = vec![0.0f32; expect.len()];
        memcpy(got.as_mut_ptr().cast(), acc_d, expect.len() * 4, MEMCPY_D2H).expect("d2h");
        device_free(src_d, src.len() * 4);
        device_free(idx_d, idx.len() * 4);
        device_free(w_d, w.len() * 4);
        device_free(acc_d, expect.len() * 4);
        for i in 0..expect.len() {
            assert!(
                gemm_close(got[i], expect[i]),
                "scatter-add mismatch idx={i} hip={} cpu={}",
                got[i],
                expect[i]
            );
        }
    }

    /// Device gather + SwiGLU + scatter-add must match host pack/scatter of serial GEMMs.
    #[cfg(feature = "hip")]
    #[test]
    fn mxfp4_swiglu_device_acc_matches_host_scatter() {
        if device_count() == 0 {
            return;
        }
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("hip");
        if hip.swiglu_pipeline_slots() < 2 {
            return;
        }
        let n_tok = 4usize;
        let dim = 64usize;
        let ff = 96usize;
        set_device(0).expect("bind");
        let mut xs = vec![0.0f32; n_tok * dim];
        for (i, v) in xs.iter_mut().enumerate() {
            *v = ((i * 19) % 50) as f32 * 0.02 - 0.4;
        }
        let groups: [Vec<(usize, f32)>; 2] = [
            vec![(0, 0.5), (2, 0.25)],
            vec![(1, 1.0), (2, 0.1), (3, 0.2)],
        ];
        let mut packed = Vec::new();
        let mut dev = Vec::new();
        let mut expect = vec![0.0f32; n_tok * dim];
        for (i, group) in groups.iter().enumerate() {
            let m = group.len();
            let (_, g) = fill_mxfp4_gemm_case(i * 3 + 1, m, ff, dim);
            let (_, u) = fill_mxfp4_gemm_case(i * 3 + 2, m, ff, dim);
            let (_, d) = fill_mxfp4_gemm_case(i * 3 + 3, m, dim, ff);
            let mut gx = vec![0.0f32; m * dim];
            for (r, &(t, _)) in group.iter().enumerate() {
                gx[r * dim..(r + 1) * dim].copy_from_slice(&xs[t * dim..(t + 1) * dim]);
            }
            let gd = device_malloc(g.len(), 0).expect("g");
            let ud = device_malloc(u.len(), 0).expect("u");
            let dd = device_malloc(d.len(), 0).expect("d");
            memcpy(gd, g.as_ptr(), g.len(), MEMCPY_H2D).expect("h2d g");
            memcpy(ud, u.as_ptr(), u.len(), MEMCPY_H2D).expect("h2d u");
            memcpy(dd, d.as_ptr(), d.len(), MEMCPY_H2D).expect("h2d d");
            let mut y = vec![0.0f32; m * dim];
            hip.mxfp4_swiglu_gemm(&gx, gd, ud, dd, &mut y, m as i32, dim as i32, ff as i32)
                .expect("serial");
            for (r, &(_, wt)) in group.iter().enumerate() {
                let t = group[r].0;
                for j in 0..dim {
                    expect[t * dim + j] += y[r * dim + j] * wt;
                }
            }
            packed.push((g, u, d));
            dev.push((gd, ud, dd));
        }
        hip.moe_begin(&xs, n_tok, dim).expect("moe_begin");
        for (i, group) in groups.iter().enumerate() {
            let m = group.len();
            let idx: Vec<i32> = group.iter().map(|&(t, _)| t as i32).collect();
            let w: Vec<f32> = group.iter().map(|&(_, wt)| wt).collect();
            hip.mxfp4_swiglu_enqueue_from_rows(
                i, &idx, &w, dev[i].0, dev[i].1, dev[i].2, m as i32, dim as i32, ff as i32,
            )
            .expect("enqueue rows");
        }
        for (i, group) in groups.iter().enumerate() {
            hip.mxfp4_swiglu_accumulate_slot(i, group.len() as i32, dim as i32, ff as i32)
                .expect("accumulate");
        }
        let mut got = vec![0.0f32; n_tok * dim];
        hip.moe_finish(&mut got).expect("moe_finish");
        for i in 0..n_tok * dim {
            assert!(
                gemm_close(got[i], expect[i]),
                "device acc mismatch idx={i} hip={} host={}",
                got[i],
                expect[i]
            );
        }
        for i in 0..dev.len() {
            device_free(dev[i].0, packed[i].0.len());
            device_free(dev[i].1, packed[i].1.len());
            device_free(dev[i].2, packed[i].2.len());
        }
    }

    /// Device-resident `xs` seeds `moe_x` without activation H2D.
    #[cfg(feature = "hip")]
    #[test]
    fn moe_begin_from_device_copies_xs_without_activation_h2d() {
        if device_count() == 0 {
            return;
        }
        let n_tok = 3usize;
        let dim = 8usize;
        let xs: Vec<f32> = (0..n_tok * dim).map(|i| 0.03 * i as f32 - 0.1).collect();
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        hip.bind().expect("bind");
        let dx = device_malloc(xs.len() * 4, 0).expect("device xs");
        memcpy(dx, xs.as_ptr().cast(), xs.len() * 4, MEMCPY_H2D).expect("seed xs");
        let h2d_before = hip.activation_h2d_bytes();
        hip.moe_begin_from_device(dx.cast(), n_tok, dim)
            .expect("moe_begin_from_device");
        assert_eq!(
            hip.activation_h2d_bytes(),
            h2d_before,
            "moe_begin_from_device must not H2D xs"
        );
        let mut got = vec![0.0f32; xs.len()];
        hip.copy_d2h_f32(hip.moe_x_device().expect("moe_x"), &mut got)
            .expect("d2h moe_x");
        for i in 0..xs.len() {
            let diff = (got[i] - xs[i]).abs();
            assert!(
                diff < 1e-6,
                "moe_x[{i}] got={} xs={} diff={diff}",
                got[i],
                xs[i]
            );
        }
        device_free(dx, xs.len() * 4);
    }

    /// GPU0 `xs` seeds GPU1 `moe_x` via peer D2D (no activation H2D).
    #[cfg(feature = "hip")]
    #[test]
    fn moe_begin_from_peer_device_skips_activation_h2d() {
        if device_count() < 2 {
            return;
        }
        let n_tok = 3usize;
        let dim = 8usize;
        let xs: Vec<f32> = (0..n_tok * dim).map(|i| 0.03 * i as f32 - 0.1).collect();
        let src = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("gpu0");
        let dst = HipWorkspace::new(1, 8 * 1024 * 1024, 64).expect("gpu1");
        src.bind().expect("bind gpu0");
        let dx = device_malloc(xs.len() * 4, 0).expect("gpu0 xs");
        memcpy(dx, xs.as_ptr().cast(), xs.len() * 4, MEMCPY_H2D).expect("seed gpu0 xs");
        let h2d_before = dst.activation_h2d_bytes();
        dst.moe_begin_from_peer(dx.cast(), src.device, n_tok, dim)
            .expect("moe_begin_from_peer");
        assert_eq!(
            dst.activation_h2d_bytes(),
            h2d_before,
            "peer moe_begin must not H2D xs"
        );
        dst.bind().expect("bind gpu1");
        let mut got = vec![0.0f32; xs.len()];
        dst.copy_d2h_f32(dst.moe_x_device().expect("moe_x"), &mut got)
            .expect("d2h gpu1 moe_x");
        for i in 0..xs.len() {
            let diff = (got[i] - xs[i]).abs();
            assert!(
                diff < 1e-6,
                "gpu1 moe_x[{i}] got={} xs={} diff={diff}",
                got[i],
                xs[i]
            );
        }
        src.bind().expect("bind gpu0 free");
        device_free(dx, xs.len() * 4);
    }

    /// Packed BF16 SwiGLU into `moe_acc` must match host GEMM plus residual add.
    #[cfg(feature = "hip")]
    #[test]
    fn bf16_swiglu_accumulate_matches_host_gemm() {
        if device_count() == 0 {
            return;
        }
        let m = 4usize;
        let dim = 32usize;
        let ff = 64usize;
        let mut x = vec![0.0f32; m * dim];
        let mut gate_f = vec![0.0f32; ff * dim];
        let mut up_f = vec![0.0f32; ff * dim];
        let mut down_f = vec![0.0f32; dim * ff];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in gate_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in up_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in down_f.iter_mut().enumerate() {
            *v = ((i * 7) % 50) as f32 * 0.02 - 0.5;
        }
        let gate: Vec<u16> = gate_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_bf16)
            .collect();
        let up: Vec<u16> = up_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let down: Vec<u16> = down_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_bf16)
            .collect();
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("hip");
        let gb = device_malloc(gate.len() * 2, 0).expect("gate");
        let ub = device_malloc(up.len() * 2, 0).expect("up");
        let db = device_malloc(down.len() * 2, 0).expect("down");
        set_device(0).expect("bind");
        memcpy(gb, gate.as_ptr().cast(), gate.len() * 2, MEMCPY_H2D).expect("h2d gate");
        memcpy(ub, up.as_ptr().cast(), up.len() * 2, MEMCPY_H2D).expect("h2d up");
        memcpy(db, down.as_ptr().cast(), down.len() * 2, MEMCPY_H2D).expect("h2d down");
        let mut host = vec![0.0f32; m * dim];
        hip.bf16_swiglu_gemm(
            &x,
            gb.cast(),
            ub.cast(),
            db.cast(),
            &mut host,
            m as i32,
            dim as i32,
            ff as i32,
        )
        .expect("host gemm");
        hip.moe_begin(&x, m, dim).expect("moe_begin");
        hip.bf16_swiglu_accumulate(
            gb.cast(),
            ub.cast(),
            db.cast(),
            m as i32,
            dim as i32,
            ff as i32,
        )
        .expect("accumulate");
        let mut got = vec![0.0f32; m * dim];
        hip.moe_finish(&mut got).expect("moe_finish");
        device_free(gb, gate.len() * 2);
        device_free(ub, up.len() * 2);
        device_free(db, down.len() * 2);
        for i in 0..m * dim {
            assert!(
                gemm_close(got[i], host[i]),
                "shexp acc mismatch idx={i} hip={} host={}",
                got[i],
                host[i]
            );
        }
    }

    /// Overflow SwiGLU malloc must drain kernels before `device_free` (ChatML shexp).
    #[cfg(feature = "hip")]
    #[test]
    fn bf16_swiglu_accumulate_overflow_scratch_matches_host() {
        if device_count() == 0 {
            return;
        }
        let m = 8usize;
        let dim = 64usize;
        let ff = 96usize;
        let need = HipWorkspace::swiglu_workspace_bytes(m, dim, ff);
        assert!(need > 4096, "fixture must miss the 4KiB scratch");
        let mut x = vec![0.0f32; m * dim];
        let mut gate_f = vec![0.0f32; ff * dim];
        let mut up_f = vec![0.0f32; ff * dim];
        let mut down_f = vec![0.0f32; dim * ff];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in gate_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in up_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in down_f.iter_mut().enumerate() {
            *v = ((i * 7) % 50) as f32 * 0.02 - 0.5;
        }
        let gate: Vec<u16> = gate_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_bf16)
            .collect();
        let up: Vec<u16> = up_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let down: Vec<u16> = down_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_bf16)
            .collect();
        let hip = HipWorkspace::new(0, 4096, 8).expect("tiny scratch");
        let gb = device_malloc(gate.len() * 2, 0).expect("gate");
        let ub = device_malloc(up.len() * 2, 0).expect("up");
        let db = device_malloc(down.len() * 2, 0).expect("down");
        set_device(0).expect("bind");
        memcpy(gb, gate.as_ptr().cast(), gate.len() * 2, MEMCPY_H2D).expect("h2d gate");
        memcpy(ub, up.as_ptr().cast(), up.len() * 2, MEMCPY_H2D).expect("h2d up");
        memcpy(db, down.as_ptr().cast(), down.len() * 2, MEMCPY_H2D).expect("h2d down");
        let mut host = vec![0.0f32; m * dim];
        hip.bf16_swiglu_gemm(
            &x,
            gb.cast(),
            ub.cast(),
            db.cast(),
            &mut host,
            m as i32,
            dim as i32,
            ff as i32,
        )
        .expect("host gemm");
        hip.moe_begin(&x, m, dim).expect("moe_begin");
        hip.bf16_swiglu_accumulate(
            gb.cast(),
            ub.cast(),
            db.cast(),
            m as i32,
            dim as i32,
            ff as i32,
        )
        .expect("overflow accumulate");
        let mut got = vec![0.0f32; m * dim];
        hip.moe_finish(&mut got).expect("moe_finish");
        device_free(gb, gate.len() * 2);
        device_free(ub, up.len() * 2);
        device_free(db, down.len() * 2);
        for i in 0..m * dim {
            assert!(
                gemm_close(got[i], host[i]),
                "overflow shexp acc mismatch idx={i} hip={} host={}",
                got[i],
                host[i]
            );
        }
    }

    #[cfg(feature = "hip")]
    fn fill_q6k_gemm_case(
        si: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
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
            let blk =
                &b_q[row * nblocks * Q6_K_BLOCK_BYTES..(row + 1) * nblocks * Q6_K_BLOCK_BYTES];
            let mut tmp = vec![0.0f32; nblocks * Q6_K_BLOCK_ELEMS];
            allpaca_cpu::dequant_q6_k(blk, nblocks * Q6_K_BLOCK_ELEMS, &mut tmp).unwrap();
            out[row * k..(row + 1) * k].copy_from_slice(&tmp[..k]);
        }
        out
    }

    /// Pseudo-random Q3_K rows (every hmask/qs/scale bit pattern is valid).
    #[cfg(feature = "hip")]
    fn fill_q3k_blocks(si: usize, n: usize, k: usize) -> Vec<u8> {
        let nblocks = k / Q3_K_BLOCK_ELEMS;
        let mut b = vec![0u8; n * nblocks * Q3_K_BLOCK_BYTES];
        let mut x = 0x9e37_79b9u32 ^ si as u32;
        for (i, v) in b.iter_mut().enumerate() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *v = if i % Q3_K_BLOCK_BYTES >= 108 { 0 } else { x as u8 };
        }
        for blk in b.chunks_mut(Q3_K_BLOCK_BYTES) {
            let d = 0.01 + (blk[0] as f32) * 1e-4;
            blk[108..110].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
        }
        b
    }

    /// ggml Q2_0 runs natively: the kernel must match CPU dequant + NT GEMM
    /// for every code, positive and negative block scales, and batch rows
    /// must equal single rows.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q2_0_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 64), (1, 70, 192), (2, 3, 320), (3, 65, 640), (5, 9, 2560), (8, 130, 448)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, _, _) = fill_q5k_gemm_case(si, m, n, k);
            let nblocks = n * k / allpaca_cpu::Q2_0_BLOCK_ELEMS;
            let mut b_q = vec![0u8; nblocks * allpaca_cpu::Q2_0_BLOCK_BYTES];
            let mut seed = 0x9e37_79b9u32.wrapping_mul(si as u32 + 1);
            for (bi, block) in b_q.chunks_exact_mut(allpaca_cpu::Q2_0_BLOCK_BYTES).enumerate() {
                for byte in block[2..].iter_mut() {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *byte = (seed >> 24) as u8;
                }
                // Every third block has a negative scale.
                let d = (0.01 + 0.003 * (bi % 17) as f32) * if bi % 3 == 0 { -1.0 } else { 1.0 };
                block[..2].copy_from_slice(&allpaca_cpu::f32_to_f16(d).to_le_bytes());
            }
            let mut b_deq = vec![0.0f32; n * k];
            allpaca_cpu::dequant_q2_0(&b_q, n * k, &mut b_deq).unwrap();
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q2_0(a, &b_q, c, rows, n as i32, k as i32)
                    .expect("HIP gemm_nt_q2_0 must run when a device is present")
            };
            run(&a, &mut hip, m as i32);
            assert_rows_match_single("q2_0", run, &a, &hip, m, n, k);
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q2_0 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
        // A row that is not whole blocks is refused, not misread.
        assert!(gemm_nt_q2_0(&[0.0; 65], &[0u8; 36], &mut [0.0; 1], 1, 1, 65).is_err());
    }

    /// Routed experts from a pool of Q2_0 slots must match the CPU: dequant
    /// each picked slot, SwiGLU, weighted sum. `silu_scale` is checked on the way.
    #[cfg(feature = "hip")]
    #[test]
    fn q2_0_moe_ffn_matches_cpu_swiglu_of_the_picked_slots() {
        if device_count() == 0 {
            return;
        }
        for (dim, ff, n_slots, slots, weights) in [
            (128usize, 64usize, 3usize, vec![2i32], vec![1.0f32]),
            (320, 192, 6, vec![5, 0, 3], vec![0.5, 0.3, 0.2]),
            (2560, 640, 4, vec![1, 3, 0, 2], vec![0.4, 0.3, 0.2, 0.1]),
        ] {
            let slot_bytes = q2_0_expert_slot_bytes(dim, ff);
            let mut pool = vec![0u8; n_slots * slot_bytes];
            let mut seed = 0x51ed_270bu32 ^ dim as u32;
            for (bi, block) in pool.chunks_exact_mut(allpaca_cpu::Q2_0_BLOCK_BYTES).enumerate() {
                for byte in block[2..].iter_mut() {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *byte = (seed >> 24) as u8;
                }
                let d = (0.05 + 0.01 * (bi % 13) as f32) * if bi % 5 == 0 { -1.0 } else { 1.0 };
                block[..2].copy_from_slice(&allpaca_cpu::f32_to_f16(d).to_le_bytes());
            }
            let x: Vec<f32> = (0..dim).map(|i| ((i * 37 % 101) as f32 / 50.0 - 1.0) * 0.8).collect();
            // CPU reference.
            let mut expect = vec![0.0f32; dim];
            for (slot, w) in slots.iter().zip(&weights) {
                let bytes = &pool[*slot as usize * slot_bytes..][..slot_bytes];
                let mut m = vec![0.0f32; 3 * ff * dim];
                allpaca_cpu::dequant_q2_0(bytes, 3 * ff * dim, &mut m).unwrap();
                let (gate, rest) = m.split_at(ff * dim);
                let (up, down) = rest.split_at(ff * dim);
                let (mut g, mut u) = (vec![0.0f32; ff], vec![0.0f32; ff]);
                cpu_gemm_nt(&x, gate, &mut g, 1, ff, dim);
                cpu_gemm_nt(&x, up, &mut u, 1, ff, dim);
                let h: Vec<f32> = g.iter().zip(&u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect();
                let mut y = vec![0.0f32; dim];
                cpu_gemm_nt(&h, down, &mut y, 1, dim, ff);
                for (e, y) in expect.iter_mut().zip(&y) {
                    *e += w * y;
                }
            }
            set_device(0).unwrap();
            let dpool = DeviceWeights::upload_bytes(0, &[&pool]).unwrap();
            let floats = dim + slots.len() * ff + dim;
            let dev = device_malloc(floats * 4, 0).unwrap();
            let (dx, dh, dout) = unsafe {
                let p = dev.cast::<f32>();
                (p, p.add(dim), p.add(dim + slots.len() * ff))
            };
            let mut got = vec![0.0f32; dim];
            let mut silu = x.clone();
            let run = (|| {
                memcpy(dx.cast(), x.as_ptr().cast(), dim * 4, MEMCPY_H2D)?;
                q2_0_moe_ffn_device(dx, dpool.byte_ptr_at(0), &slots, &weights, dh, dout, dim as i32, ff as i32)?;
                memcpy(got.as_mut_ptr().cast(), dout.cast(), dim * 4, MEMCPY_D2H)?;
                silu_scale_device(dx, dim as i32, 0.25)?;
                memcpy(silu.as_mut_ptr().cast(), dx.cast(), dim * 4, MEMCPY_D2H)
            })();
            device_free(dev, floats * 4);
            run.expect("Q2_0 MoE kernels must run when a device is present");
            for i in 0..dim {
                assert!(
                    gemm_close(got[i], expect[i]),
                    "q2_0_moe_ffn dim {dim} ff {ff} idx {i}: hip {} cpu {}",
                    got[i],
                    expect[i]
                );
                let v = x[i] * 0.25;
                assert!((silu[i] - v / (1.0 + (-v).exp())).abs() < 1e-6, "silu_scale idx {i}");
            }
            assert!(expect.iter().any(|v| v.abs() > 1e-3), "reference output is all zero");
        }
    }

    /// Routing on the device picks the experts the host picks from the same
    /// scores (ties to the lower expert) and gives the routed-expert output
    /// of those picks.
    #[cfg(feature = "hip")]
    #[test]
    fn q4x_routed_experts_pick_and_weigh_like_the_host() {
        if device_count() == 0 {
            return;
        }
        for (dim, ff, n_exp, k, scale) in
            [(128usize, 64usize, 7usize, 1usize, 1.0f32), (320, 192, 40, 5, 1.0), (2560, 640, 512, 10, 2.5)]
        {
            let slot_bytes = q2_0_expert_slot_bytes(dim, ff);
            let mut pool = vec![0u8; n_exp * slot_bytes];
            let mut seed = 0x9e37_79b9u32 ^ dim as u32;
            for (bi, block) in pool.chunks_exact_mut(allpaca_cpu::Q2_0_BLOCK_BYTES).enumerate() {
                for byte in block[2..].iter_mut() {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *byte = (seed >> 24) as u8;
                }
                let d = (0.05 + 0.01 * (bi % 13) as f32) * if bi % 5 == 0 { -1.0 } else { 1.0 };
                block[..2].copy_from_slice(&allpaca_cpu::f32_to_f16(d).to_le_bytes());
            }
            let x: Vec<f32> = (0..dim).map(|i| ((i * 37 % 101) as f32 / 50.0 - 1.0) * 0.8).collect();
            // Scores with ties among the best.
            let mut scores: Vec<f32> = (0..n_exp).map(|e| ((e * 53 % 97) as f32 / 20.0 - 2.0).round() * 0.5).collect();
            scores[n_exp - 1] = 3.0;
            // Host routing: softmax, the k best (ties to the lower expert), renormalise, scale.
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut w: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = w.iter().sum();
            w.iter_mut().for_each(|v| *v /= sum);
            let mut order: Vec<usize> = (0..n_exp).collect();
            order.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap().then(a.cmp(&b)));
            order.truncate(k);
            let part: f32 = order.iter().map(|&e| w[e]).sum();
            let slots: Vec<i32> = order.iter().map(|&e| e as i32).collect();
            let weights: Vec<f32> = order.iter().map(|&e| w[e] / part * scale).collect();
            set_device(0).unwrap();
            let dpool = DeviceWeights::upload_bytes(0, &[&pool]).unwrap();
            let floats = dim + n_exp + k * ff + 2 * dim + q4x::PICKS_BYTES / 4;
            let dev = device_malloc(floats * 4, 0).unwrap();
            let (dx, dscores, dh, dout, dref, dpicks) = unsafe {
                let p = dev.cast::<f32>();
                let h = p.add(dim + n_exp);
                (p, p.add(dim), h, h.add(k * ff), h.add(k * ff + dim), h.add(k * ff + 2 * dim))
            };
            let (mut got, mut expect) = (vec![0.0f32; dim], vec![0.0f32; dim]);
            let mut picks = vec![0u8; q4x::PICKS_BYTES];
            let run = (|| {
                memcpy(dx.cast(), x.as_ptr().cast(), dim * 4, MEMCPY_H2D)?;
                memcpy(dscores.cast(), scores.as_ptr().cast(), n_exp * 4, MEMCPY_H2D)?;
                q2_0_moe_ffn_device(dx, dpool.byte_ptr_at(0), &slots, &weights, dh, dref, dim as i32, ff as i32)?;
                q4x::routed_experts(
                    dx,
                    dscores,
                    dpool.byte_ptr_at(0),
                    n_exp as i32,
                    k as i32,
                    scale,
                    dpicks.cast(),
                    dh,
                    dout,
                    dim as i32,
                    ff as i32,
                )?;
                memcpy(got.as_mut_ptr().cast(), dout.cast(), dim * 4, MEMCPY_D2H)?;
                memcpy(expect.as_mut_ptr().cast(), dref.cast(), dim * 4, MEMCPY_D2H)?;
                memcpy(picks.as_mut_ptr().cast(), dpicks.cast(), q4x::PICKS_BYTES, MEMCPY_D2H)
            })();
            device_free(dev, floats * 4);
            run.expect("device routing must run when a device is present");
            let picked: Vec<i32> = picks[..k * 4].chunks(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
            assert_eq!(picked, slots, "picks for {n_exp} experts");
            for (p, want) in picks[128..][..k * 4].chunks(4).zip(&weights) {
                let got = f32::from_le_bytes(p.try_into().unwrap());
                assert!((got - want).abs() <= 1e-6 * want.abs().max(1e-3), "weight {got} vs {want}");
            }
            for i in 0..dim {
                assert!(gemm_close(got[i], expect[i]), "routed dim {dim} idx {i}: {} vs {}", got[i], expect[i]);
            }
            assert!(expect.iter().any(|v| v.abs() > 1e-3), "reference output is all zero");
        }
    }

    /// Opt-in: what one small kernel launch costs, queued and with a sync each.
    #[cfg(feature = "hip")]
    #[test]
    #[ignore]
    fn bench_kernel_launch() {
        if device_count() == 0 {
            return;
        }
        let dev: i32 = std::env::var("BENCH_DEV").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        set_device(dev).unwrap();
        let buf = device_malloc(4096 * 4, dev).unwrap().cast::<f32>();
        let mut host = [0.0f32; 16];
        for _ in 0..200 {
            silu_scale_device(buf, 64, 1.0).unwrap();
        }
        stream_synchronize().unwrap();
        let n = 2000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            silu_scale_device(buf, 64, 1.0).unwrap();
        }
        let queued = start.elapsed();
        stream_synchronize().unwrap();
        let done = start.elapsed();
        let start = std::time::Instant::now();
        for _ in 0..n {
            silu_scale_device(buf, 64, 1.0).unwrap();
            memcpy(host.as_mut_ptr().cast(), buf.cast(), 64, MEMCPY_D2H).unwrap();
        }
        let synced = start.elapsed();
        device_free(buf.cast(), 4096 * 4);
        eprintln!(
            "device {dev}: launch {:.1} us to queue, {:.1} us to run; launch + 64-byte download {:.1} us",
            queued.as_secs_f64() * 1e6 / n as f64,
            done.as_secs_f64() * 1e6 / n as f64,
            synced.as_secs_f64() * 1e6 / n as f64
        );
    }

    /// Opt-in: what the kernels of one Qwen3.8-Flash-Next block cost on a
    /// device (`ALLPACA_BENCH_DEVICE`, default 0), one token, zeroed
    /// weights: the two stream mixers, a Gated DeltaNet block's matrices,
    /// the router with the shared expert, and ten routed experts picked on
    /// the card. The recurrent step and the small elementwise kernels of
    /// the block are left out.
    /// `cargo test --release -p allpaca-hip --features hip --lib q4x_block_cost -- --ignored --nocapture`
    #[cfg(feature = "hip")]
    #[test]
    #[ignore]
    fn q4x_block_cost() {
        let device: i32 = std::env::var("ALLPACA_BENCH_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if device_count() <= device {
            return;
        }
        let (hc, dim, low, ff, n_exp, picks) = (4usize, 2560usize, 320usize, 640usize, 512usize, 10usize);
        let k = hc * dim;
        let q3k = |n: usize, k: usize| n * k / 256 * 110;
        let (value, qkv) = (6144usize, 10240usize);
        set_device(device).unwrap();
        let alloc = |bytes: usize| {
            let ptr = device_malloc(bytes, device).unwrap();
            device_memset(ptr, 0, bytes).unwrap();
            ptr
        };
        let slot_bytes = q2_0_expert_slot_bytes(dim, ff);
        let pool = alloc(n_exp * slot_bytes);
        let (w_down, w_up, w_inject) = (alloc(low * k * 2), alloc(k * low * 2), alloc(hc * k * 2));
        let (w_qkv, w_gate, w_out) = (alloc(q3k(qkv, dim)), alloc(q3k(value, dim)), alloc(q3k(dim, value)));
        let (w_alpha, w_router) = (alloc(48 * dim * 2), alloc(n_exp * dim * 2));
        let (w_shared_in, w_shared_down) = (alloc(q3k(ff, dim)), alloc(slot_bytes / 3));
        let act = alloc(64 * 12288 * 4).cast::<f32>();
        let at = |i: usize| unsafe { act.add(i * 12288) };
        let (r, h, d) = (1i32, hc as i32, dim as i32);
        let mix = || -> Result<(), AllpacaError> {
            q4x::stream_norm(at(0), at(1), at(2), r, h, d, 1e-6)?;
            gemm_nt_bf16_device(at(2), w_down.cast(), at(3), 1, low as i32, k as i32)?;
            silu_scale_device(at(3), low as i32, 0.25)?;
            gemm_nt_bf16_device(at(3), w_up.cast(), at(4), 1, k as i32, low as i32)?;
            q4x::gated_mean(at(2), at(4), at(5), r, h, d)?;
            gemm_nt_bf16_device(at(2), w_inject.cast(), at(6), 1, h, k as i32)
        };
        let gdn = || -> Result<(), AllpacaError> {
            gemm_nt_q3_k_device(at(5), w_qkv, at(7), 1, qkv as i32, d)?;
            gemm_nt_q3_k_device(at(5), w_gate, at(8), 1, value as i32, d)?;
            gemm_nt_bf16_device(at(5), w_alpha.cast(), at(9), 1, 48, d)?;
            gemm_nt_bf16_device(at(5), w_alpha.cast(), at(10), 1, 48, d)?;
            gemm_nt_q3_k_device(at(8), w_out, at(11), 1, d, value as i32)
        };
        let route = || -> Result<(), AllpacaError> {
            gemm_nt_bf16_device(at(5), w_router.cast(), at(12), 1, n_exp as i32, d)?;
            gemm_nt_q3_k_device(at(5), w_shared_in, at(13), 1, ff as i32, d)?;
            gemm_nt_q3_k_device(at(5), w_shared_in, at(14), 1, ff as i32, d)?;
            gemm_nt_q2_0_device(at(13), w_shared_down, at(15), 1, d, ff as i32)?;
            gemm_nt_bf16_device(at(5), w_alpha.cast(), at(16), 1, 1, d)
        };
        let experts = || -> Result<(), AllpacaError> {
            q4x::routed_experts(at(5), at(12), pool, n_exp as i32, picks as i32, 1.0, at(17).cast(), at(18), at(19), d, ff as i32)?;
            q4x::ffn_sum(at(19), at(15), at(16), at(11), r, d)?;
            q4x::combine(at(0), at(11), at(6), r, h, d)
        };
        let time = |name: &str, f: &dyn Fn() -> Result<(), AllpacaError>| {
            for _ in 0..20 {
                f().unwrap();
            }
            stream_synchronize().unwrap();
            let n = 200;
            let clock = std::time::Instant::now();
            for _ in 0..n {
                f().unwrap();
            }
            stream_synchronize().unwrap();
            let us = clock.elapsed().as_secs_f64() * 1e6 / n as f64;
            eprintln!("BLOCK device {device} {name}: {us:.1} us");
            us
        };
        let total = 2.0 * time("mixer", &mix) + time("deltanet matrices", &gdn) + time("router and shared expert", &route)
            + time("routed experts, sum, combine", &experts);
        let whole = time("whole block", &|| {
            mix()?;
            gdn()?;
            mix()?;
            route()?;
            experts()
        });
        eprintln!("BLOCK device {device} sum of parts {total:.1} us, whole block {whole:.1} us");
    }

    /// The qwen4exp stream kernels against their definitions, on two rows.
    #[cfg(feature = "hip")]
    #[test]
    fn q4x_stream_kernels_match_their_definitions() {
        if device_count() == 0 {
            return;
        }
        let (rows, hc, dim, eps) = (2usize, 4usize, 2560usize, 1e-6f32);
        let (per_row, n) = (hc * dim, rows * hc * dim);
        let wave = |i: usize, a: usize, b: usize| ((i * a % b) as f32 / b as f32 - 0.5) * 3.0;
        let x: Vec<f32> = (0..n).map(|i| wave(i, 31, 997)).collect();
        let gains: Vec<f32> = (0..per_row).map(|i| 1.0 + 0.2 * wave(i, 17, 251)).collect();
        let gate: Vec<f32> = (0..n).map(|i| wave(i, 13, 409)).collect();
        let out: Vec<f32> = (0..rows * dim).map(|i| wave(i, 7, 311)).collect();
        let logits = [0.0f32, 1.5, -2.0, 0.3, 0.7, -0.4, 2.5, -1.1];
        let open = [0.9f32, -1.3];
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        // Layout: x, gate, xn (n each), gains, out, mixed, sum, logits, open.
        let offs = [0, n, 2 * n, 3 * n, 3 * n + per_row, 3 * n + per_row + rows * dim];
        let (o_sum, o_logits) = (offs[5] + rows * dim, offs[5] + 2 * rows * dim);
        let floats = o_logits + rows * hc + rows;
        set_device(0).unwrap();
        let dev = device_malloc(floats * 4, 0).unwrap().cast::<f32>();
        let at = |off: usize| unsafe { dev.add(off) };
        let (dx, dgate, dxn, dgains, dout, dmixed, dsum, dlogits, dopen) =
            (at(0), at(offs[1]), at(offs[2]), at(offs[3]), at(offs[4]), at(offs[5]), at(o_sum), at(o_logits), at(o_logits + rows * hc));
        let up = |dst: *mut f32, src: &[f32]| memcpy(dst.cast(), src.as_ptr().cast(), src.len() * 4, MEMCPY_H2D);
        let down = |src: *const f32, len: usize| {
            let mut v = vec![0.0f32; len];
            memcpy(v.as_mut_ptr().cast(), src.cast(), len * 4, MEMCPY_D2H).map(|()| v)
        };
        let (r, h, d) = (rows as i32, hc as i32, dim as i32);
        let run = (|| {
            up(dx, &x)?;
            up(dgains, &gains)?;
            up(dgate, &gate)?;
            up(dout, &out)?;
            up(dlogits, &logits)?;
            up(dopen, &open)?;
            q4x::stream_norm(dx, dgains, dxn, r, h, d, eps)?;
            q4x::gated_mean(dxn, dgate, dmixed, r, h, d)?;
            let (xn, mixed) = (down(dxn, n)?, down(dmixed, rows * dim)?);
            q4x::ffn_sum(dout, dmixed, dopen, dsum, r, d)?;
            let both = down(dsum, rows * dim)?;
            q4x::ffn_sum(std::ptr::null(), dmixed, dopen, dsum, r, d)?;
            let shared_only = down(dsum, rows * dim)?;
            q4x::ffn_sum(dout, std::ptr::null(), std::ptr::null(), dsum, r, d)?;
            let routed_only = down(dsum, rows * dim)?;
            q4x::combine(dx, dout, dlogits, r, h, d)?;
            Ok::<_, AllpacaError>((xn, mixed, both, shared_only, routed_only, down(dx, n)?))
        })();
        device_free(dev.cast(), floats * 4);
        let (xn, mixed, both, shared_only, routed_only, combined) = run.expect("q4x kernels must run when a device is present");
        let close = |a: f32, b: f32| (a - b).abs() <= 1e-5 + 1e-4 * b.abs();
        let mut want_xn = vec![0.0f32; n];
        for stream in 0..rows * hc {
            let row = &x[stream * dim..(stream + 1) * dim];
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
            for d in 0..dim {
                want_xn[stream * dim + d] = row[d] * inv * gains[(stream % hc) * dim + d];
                assert!(close(xn[stream * dim + d], want_xn[stream * dim + d]), "stream_norm {stream},{d}");
            }
        }
        for row in 0..rows {
            for d in 0..dim {
                let i = row * dim + d;
                let at = |s: usize| (row * hc + s) * dim + d;
                let want: f32 = (0..hc).map(|s| want_xn[at(s)] * sig(gate[at(s)])).sum::<f32>() / hc as f32;
                assert!(close(mixed[i], want), "gated_mean {row},{d}: {} vs {want}", mixed[i]);
                let shared = sig(open[row]) * want;
                assert!(close(both[i], out[i] + shared), "ffn_sum both {row},{d}");
                assert!(close(shared_only[i], shared), "ffn_sum shared {row},{d}");
                assert_eq!(routed_only[i], out[i], "ffn_sum routed {row},{d}");
                for s in 0..hc {
                    let want = x[at(s)] + out[i] * 2.0 * sig(logits[row * hc + s] / hc as f32);
                    assert!(close(combined[at(s)], want), "combine {row},{s},{d}");
                }
            }
        }
    }

    /// The draft lm_head's Q3_K runs natively: fused GEMV must match CPU
    /// dequant x q8 activations, and batch rows must equal single rows.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q3_k_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [(1, 4, 256), (1, 33, 512), (2, 3, 256), (3, 13, 512), (4, 9, 768), (8, 9, 256), (11, 5, 768)];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, _, _) = fill_q5k_gemm_case(si, m, n, k);
            let b_q = fill_q3k_blocks(si, n, k);
            let mut b_deq = vec![0.0f32; n * k];
            allpaca_cpu::dequant_q3_k(&b_q, n * k, &mut b_deq).unwrap();
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&q8_activations(&a), &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q3_k(a, &b_q, c, rows, n as i32, k as i32)
                    .expect("HIP gemm_nt_q3_k must run when a device is present")
            };
            run(&a, &mut hip, m as i32);
            assert_rows_match_single("q3_k", run, &a, &hip, m, n, k);
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_q3_k mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// GEM mixed-quant: fused Q6_K GEMV must match CPU dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q6_k_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        // Covers every warp-kernel row batch (1, 2, <=4, 8, >8) and n % 8 != 0.
        let shapes = [
            (1, 4, 256),
            (1, 8, 256),
            (2, 3, 256),
            (1, 4, 512),
            (3, 13, 512),
            (8, 9, 256),
            (11, 5, 768),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q, _) = fill_q6k_gemm_case(si, m, n, k);
            let b_deq = dequant_q6k_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&q8_activations(&a), &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q6_k(a, &b_q, c, rows, n as i32, k as i32)
                    .expect("HIP gemm_nt_q6_k must run when a device is present")
            };
            run(&a, &mut hip, m as i32);
            assert_rows_match_single("q6_k", run, &a, &hip, m, n, k);
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
    fn fill_q5k_gemm_case(
        si: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
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
            let blk =
                &b_q[row * nblocks * Q5_K_BLOCK_BYTES..(row + 1) * nblocks * Q5_K_BLOCK_BYTES];
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
        // Covers every warp-kernel row batch (1, 2, <=4, 8, >8) and n % 8 != 0.
        let shapes = [
            (1, 4, 256),
            (1, 8, 256),
            (2, 3, 256),
            (1, 4, 512),
            (3, 13, 512),
            (8, 9, 256),
            (11, 5, 768),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let (a, b_q, _) = fill_q5k_gemm_case(si, m, n, k);
            let b_deq = dequant_q5k_matrix(&b_q, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&q8_activations(&a), &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q5_k(a, &b_q, c, rows, n as i32, k as i32)
                    .expect("HIP gemm_nt_q5_k must run when a device is present")
            };
            run(&a, &mut hip, m as i32);
            assert_rows_match_single("q5_k", run, &a, &hip, m, n, k);
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
        gemm_nt_q4_k(&a, &b_q, &mut tiled, m as i32, n as i32, k as i32).expect("tiled Q4_K");
        gemm_nt_q4_k_untiled(&q8_activations(&a), &b_q, &mut untiled, m as i32, n as i32, k as i32)
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
        gemm_nt_q5_k_untiled(&q8_activations(&a), &b_q, &mut untiled, m as i32, n as i32, k as i32)
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
        gemm_nt_q6_k_untiled(&q8_activations(&a), &b_q, &mut untiled, m as i32, n as i32, k as i32)
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
        gemm_nt_f16_device(
            da.cast(),
            db.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
        .unwrap();
        memcpy(tiled.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H).unwrap();
        gemm_nt_f16_untiled_device(
            da.cast(),
            db.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
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
            gemm_nt_f16_untiled_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
            gemm_nt_f16_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_f16_untiled_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_f16_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
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
        gemm_nt_device(
            da.cast(),
            db.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
        .unwrap();
        memcpy(tiled.as_mut_ptr().cast(), dc, c_bytes, MEMCPY_D2H).unwrap();
        gemm_nt_untiled_device(
            da.cast(),
            db.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
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
            gemm_nt_untiled_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
            gemm_nt_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_untiled_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
        }
        let before_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            gemm_nt_device(
                da.cast(),
                db.cast(),
                dc.cast(),
                m as i32,
                n as i32,
                k as i32,
            )
            .unwrap();
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
        // Multi-row GEMV batches (verify spans): 4-row and 8-row blocks, n and
        // k off the tile grid.
        let extra = [(4, 130, 96), (5, 20, 96), (8, 33, 64), (7, 129, 33), (9, 21, 64), (16, 9, 96)];
        for (si, (m, n, k)) in q8_intdot_shapes().iter().chain(extra.iter()).copied().enumerate() {
            let (a, b_q8, _) = fill_q8_gemm_case(si, m, n, k);
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&q8_0_gemv_activations(&a, m, n, k), &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q8_0(a, &b_q8, c, rows, n as i32, k as i32)
                    .expect("HIP gemm_nt_q8_0 must run when a device is present")
            };
            run(&a, &mut hip, m as i32);
            if m <= 16 {
                // Decode and verify spans (up to max_depth 8 + a deferred
                // correction) must agree with a single row bit for bit.
                assert_rows_match_single("q8_0", run, &a, &hip, m, n, k);
            }
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

    /// Public `gemm_nt_bf16` must match CPU BF16-dequant + NT GEMM on decode shapes (M≤8).
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_bf16_matches_cpu_dequant_gemm() {
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
            let b_bf16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
            let b_deq: Vec<f32> = b_bf16
                .iter()
                .copied()
                .map(allpaca_cpu::bf16_to_f32)
                .collect();
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_bf16(&a, &b_bf16, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_bf16 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_bf16 mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Prefill-wide BF16 NT GEMM (M>8) must match CPU BF16-dequant + NT GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_bf16_wide_m_matches_cpu_dequant_gemm() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (16, 64, 128),
            (9, 17, 64),
            (32, 33, 96),
            (40, 65, 67),
            (48, 64, 32),
        ];
        for (si, (m, n, k)) in shapes.iter().copied().enumerate() {
            let mut a = vec![0.0f32; m * k];
            let mut b_f = vec![0.0f32; n * k];
            for (i, v) in a.iter_mut().enumerate() {
                *v = ((i * 17 + si * 43) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in b_f.iter_mut().enumerate() {
                *v = ((i * 13 + si * 11) % 50) as f32 * 0.02 - 0.5;
            }
            let b_bf16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
            let b_deq: Vec<f32> = b_bf16
                .iter()
                .copied()
                .map(allpaca_cpu::bf16_to_f32)
                .collect();
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_bf16(&a, &b_bf16, &mut hip, m as i32, n as i32, k as i32)
                .expect("HIP gemm_nt_bf16 wide must run when a device is present");
            for i in 0..m * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "gemm_nt_bf16 wide mismatch shape {m}x{n}x{k} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
        }
    }

    /// Device-resident A skips activation H2D and matches host-A BF16 GEMM.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_bf16_from_device_a_matches_host_a_without_h2d() {
        if device_count() == 0 {
            return;
        }
        let m = 3usize;
        let n = 16usize;
        let k = 32usize;
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        let b_bf16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let b_deq: Vec<f32> = b_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let mut cpu = vec![0.0f32; m * n];
        cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        hip.bind().expect("bind");
        let da = device_malloc(a.len() * 4, 0).expect("device a");
        let db = device_malloc(b_bf16.len() * 2, 0).expect("device b");
        memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).expect("h2d a");
        memcpy(db, b_bf16.as_ptr().cast(), b_bf16.len() * 2, MEMCPY_H2D).expect("h2d b");
        let mut host_a = vec![0.0f32; m * n];
        hip.gemm_nt_bf16_dev_b(&a, db.cast(), &mut host_a, m as i32, n as i32, k as i32)
            .expect("host-A gemm");
        let h2d_before = hip.activation_h2d_bytes();
        let mut device_a = vec![0.0f32; m * n];
        hip.gemm_nt_bf16_from_device_a(
            da.cast(),
            db.cast(),
            &mut device_a,
            m as i32,
            n as i32,
            k as i32,
        )
        .expect("device-A gemm");
        assert_eq!(
            hip.activation_h2d_bytes(),
            h2d_before,
            "device-A BF16 GEMM must not H2D activations"
        );
        for i in 0..m * n {
            assert!(
                gemm_close(device_a[i], cpu[i]),
                "device-A mismatch idx={i} hip={} cpu={}",
                device_a[i],
                cpu[i]
            );
            assert!(
                gemm_close(device_a[i], host_a[i]),
                "device-A vs host-A idx={i} {} vs {}",
                device_a[i],
                host_a[i]
            );
        }
        device_free(da, a.len() * 4);
        device_free(db, b_bf16.len() * 2);
    }

    /// Device C from device A skips activation H2D and D2H.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_bf16_from_device_a_into_device_c_skips_activation_copies() {
        if device_count() == 0 {
            return;
        }
        let m = 3usize;
        let n = 16usize;
        let k = 32usize;
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        let b_bf16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let b_deq: Vec<f32> = b_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let mut cpu = vec![0.0f32; m * n];
        cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        hip.bind().expect("bind");
        let da = device_malloc(a.len() * 4, 0).expect("device a");
        let db = device_malloc(b_bf16.len() * 2, 0).expect("device b");
        let dc = device_malloc(cpu.len() * 4, 0).expect("device c");
        memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).expect("h2d a");
        memcpy(db, b_bf16.as_ptr().cast(), b_bf16.len() * 2, MEMCPY_H2D).expect("h2d b");
        let h2d_before = hip.activation_h2d_bytes();
        let d2h_before = hip.activation_d2h_bytes();
        hip.gemm_nt_bf16_from_device_a_into(
            da.cast(),
            db.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
        .expect("device A into C");
        assert_eq!(hip.activation_h2d_bytes(), h2d_before);
        assert_eq!(hip.activation_d2h_bytes(), d2h_before);
        let mut got = vec![0.0f32; cpu.len()];
        memcpy(got.as_mut_ptr().cast(), dc, got.len() * 4, MEMCPY_D2H).expect("d2h c");
        for i in 0..cpu.len() {
            assert!(
                gemm_close(got[i], cpu[i]),
                "device-C mismatch idx={i} hip={} cpu={}",
                got[i],
                cpu[i]
            );
        }
        device_free(da, a.len() * 4);
        device_free(db, b_bf16.len() * 2);
        device_free(dc, cpu.len() * 4);
    }

    /// MLA Q-style q_a → rmsnorm → q_b on device must not H2D intermediate qr.
    #[cfg(feature = "hip")]
    #[test]
    fn mla_q_chain_from_device_skips_intermediate_qr_h2d() {
        if device_count() == 0 {
            return;
        }
        let m = 3usize;
        let dim = 32usize;
        let q_lora = 16usize;
        let q_out = 24usize;
        let eps = 1e-5f32;
        let mut a = vec![0.0f32; m * dim];
        let mut qa_f = vec![0.0f32; q_lora * dim];
        let mut qb_f = vec![0.0f32; q_out * q_lora];
        let mut nw = vec![0.0f32; q_lora];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in qa_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in qb_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in nw.iter_mut().enumerate() {
            *v = 0.8 + 0.01 * i as f32;
        }
        let qa_bf16: Vec<u16> = qa_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let qb_bf16: Vec<u16> = qb_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let qa_deq: Vec<f32> = qa_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let qb_deq: Vec<f32> = qb_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let mut qr = vec![0.0f32; m * q_lora];
        cpu_gemm_nt(&a, &qa_deq, &mut qr, m, q_lora, dim);
        let mut qr_n = vec![0.0f32; m * q_lora];
        for t in 0..m {
            allpaca_cpu::rmsnorm(
                &qr[t * q_lora..(t + 1) * q_lora],
                &nw,
                eps,
                &mut qr_n[t * q_lora..(t + 1) * q_lora],
            )
            .expect("cpu rmsnorm");
        }
        let mut cpu_q = vec![0.0f32; m * q_out];
        cpu_gemm_nt(&qr_n, &qb_deq, &mut cpu_q, m, q_out, q_lora);
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        hip.bind().expect("bind");
        let da = device_malloc(a.len() * 4, 0).expect("da");
        let dqa = device_malloc(qa_bf16.len() * 2, 0).expect("dqa");
        let dqb = device_malloc(qb_bf16.len() * 2, 0).expect("dqb");
        let dqr = device_malloc(qr.len() * 4, 0).expect("dqr");
        let dq = device_malloc(cpu_q.len() * 4, 0).expect("dq");
        let dnw = device_malloc(nw.len() * 4, 0).expect("dnw");
        memcpy(da, a.as_ptr().cast(), a.len() * 4, MEMCPY_H2D).expect("h2d a");
        memcpy(dqa, qa_bf16.as_ptr().cast(), qa_bf16.len() * 2, MEMCPY_H2D).expect("h2d qa");
        memcpy(dqb, qb_bf16.as_ptr().cast(), qb_bf16.len() * 2, MEMCPY_H2D).expect("h2d qb");
        memcpy(dnw, nw.as_ptr().cast(), nw.len() * 4, MEMCPY_H2D).expect("h2d nw");
        let h2d_before = hip.activation_h2d_bytes();
        let d2h_before = hip.activation_d2h_bytes();
        hip.gemm_nt_bf16_from_device_a_into(
            da.cast(),
            dqa.cast(),
            dqr.cast(),
            m as i32,
            q_lora as i32,
            dim as i32,
        )
        .expect("q_a");
        hip.rmsnorm_device_rows(
            dqr.cast(),
            dnw.cast(),
            dqr.cast(),
            m as i32,
            q_lora as i32,
            eps,
        )
        .expect("q_a rmsnorm");
        hip.gemm_nt_bf16_from_device_a_into(
            dqr.cast(),
            dqb.cast(),
            dq.cast(),
            m as i32,
            q_out as i32,
            q_lora as i32,
        )
        .expect("q_b");
        assert_eq!(
            hip.activation_h2d_bytes(),
            h2d_before,
            "MLA Q chain must not H2D qr"
        );
        assert_eq!(
            hip.activation_d2h_bytes(),
            d2h_before,
            "MLA Q chain must not D2H qr"
        );
        let mut got = vec![0.0f32; cpu_q.len()];
        memcpy(got.as_mut_ptr().cast(), dq, got.len() * 4, MEMCPY_D2H).expect("d2h q");
        for i in 0..cpu_q.len() {
            assert!(
                gemm_close(got[i], cpu_q[i]),
                "MLA Q chain mismatch idx={i} hip={} cpu={}",
                got[i],
                cpu_q[i]
            );
        }
        device_free(da, a.len() * 4);
        device_free(dqa, qa_bf16.len() * 2);
        device_free(dqb, qb_bf16.len() * 2);
        device_free(dqr, qr.len() * 4);
        device_free(dq, cpu_q.len() * 4);
        device_free(dnw, nw.len() * 4);
    }

    /// Grouped o_a → device oa → o_b must not H2D/D2H the oa intermediate.
    #[cfg(feature = "hip")]
    #[test]
    fn mla_o_chain_from_device_skips_intermediate_oa_h2d() {
        if device_count() == 0 {
            return;
        }
        let m = 3usize;
        let groups = 2usize;
        let o_group_dim = 8usize;
        let per = 4usize;
        let dim = 16usize;
        let oa_out = per * groups;
        let q_out = groups * o_group_dim;
        let mut attn = vec![0.0f32; m * q_out];
        let mut oa_f = vec![0.0f32; groups * per * o_group_dim];
        let mut ob_f = vec![0.0f32; dim * oa_out];
        for (i, v) in attn.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in oa_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in ob_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        let oa_bf16: Vec<u16> = oa_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let ob_bf16: Vec<u16> = ob_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let oa_deq: Vec<f32> = oa_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let ob_deq: Vec<f32> = ob_bf16
            .iter()
            .copied()
            .map(allpaca_cpu::bf16_to_f32)
            .collect();
        let mut cpu_oa = vec![0.0f32; m * oa_out];
        for g in 0..groups {
            let mut xs = vec![0.0f32; m * o_group_dim];
            for t in 0..m {
                let src = t * q_out + g * o_group_dim;
                let dst = t * o_group_dim;
                xs[dst..dst + o_group_dim].copy_from_slice(&attn[src..src + o_group_dim]);
            }
            let w = &oa_deq[g * per * o_group_dim..(g + 1) * per * o_group_dim];
            let mut ys = vec![0.0f32; m * per];
            cpu_gemm_nt(&xs, w, &mut ys, m, per, o_group_dim);
            for t in 0..m {
                let src = t * per;
                let dst = t * oa_out + g * per;
                cpu_oa[dst..dst + per].copy_from_slice(&ys[src..src + per]);
            }
        }
        let mut cpu_out = vec![0.0f32; m * dim];
        cpu_gemm_nt(&cpu_oa, &ob_deq, &mut cpu_out, m, dim, oa_out);
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        hip.bind().expect("bind");
        let dxs: Vec<*mut u8> = (0..groups)
            .map(|_| device_malloc(m * o_group_dim * 4, 0).expect("dxs"))
            .collect();
        let doa_w: Vec<*mut u8> = (0..groups)
            .map(|_| device_malloc(per * o_group_dim * 2, 0).expect("doa_w"))
            .collect();
        let dob = device_malloc(ob_bf16.len() * 2, 0).expect("dob");
        let dys = device_malloc(m * per * 4, 0).expect("dys");
        let doa = device_malloc(m * oa_out * 4, 0).expect("doa");
        let dout = device_malloc(m * dim * 4, 0).expect("dout");
        memcpy(dob, ob_bf16.as_ptr().cast(), ob_bf16.len() * 2, MEMCPY_H2D).expect("h2d ob");
        for g in 0..groups {
            let mut xs = vec![0.0f32; m * o_group_dim];
            for t in 0..m {
                let src = t * q_out + g * o_group_dim;
                let dst = t * o_group_dim;
                xs[dst..dst + o_group_dim].copy_from_slice(&attn[src..src + o_group_dim]);
            }
            memcpy(dxs[g], xs.as_ptr().cast(), xs.len() * 4, MEMCPY_H2D).expect("h2d xs");
            let w_off = g * per * o_group_dim;
            memcpy(
                doa_w[g],
                oa_bf16[w_off..].as_ptr().cast(),
                per * o_group_dim * 2,
                MEMCPY_H2D,
            )
            .expect("h2d oa_w");
        }
        let h2d_before = hip.activation_h2d_bytes();
        let d2h_before = hip.activation_d2h_bytes();
        for g in 0..groups {
            hip.gemm_nt_bf16_from_device_a_into(
                dxs[g].cast(),
                doa_w[g].cast(),
                dys.cast(),
                m as i32,
                per as i32,
                o_group_dim as i32,
            )
            .expect("o_a group");
            for t in 0..m {
                // SAFETY: `doa` is `[m, oa_out]`; `dys` is `[m, per]`; both live.
                let dst = unsafe { doa.cast::<f32>().add(t * oa_out + g * per) };
                let src = unsafe { dys.cast::<f32>().add(t * per) };
                hip.copy_d2d_f32(dst, src, per).expect("scatter oa");
            }
        }
        hip.gemm_nt_bf16_from_device_a_into(
            doa.cast(),
            dob.cast(),
            dout.cast(),
            m as i32,
            dim as i32,
            oa_out as i32,
        )
        .expect("o_b");
        assert_eq!(
            hip.activation_h2d_bytes(),
            h2d_before,
            "MLA O chain must not H2D oa"
        );
        assert_eq!(
            hip.activation_d2h_bytes(),
            d2h_before,
            "MLA O chain must not D2H oa"
        );
        let mut got = vec![0.0f32; cpu_out.len()];
        memcpy(got.as_mut_ptr().cast(), dout, got.len() * 4, MEMCPY_D2H).expect("d2h out");
        for i in 0..cpu_out.len() {
            assert!(
                gemm_close(got[i], cpu_out[i]),
                "MLA O chain mismatch idx={i} hip={} cpu={}",
                got[i],
                cpu_out[i]
            );
        }
        for p in dxs {
            device_free(p, m * o_group_dim * 4);
        }
        for p in doa_w {
            device_free(p, per * o_group_dim * 2);
        }
        device_free(dob, ob_bf16.len() * 2);
        device_free(dys, m * per * 4);
        device_free(doa, m * oa_out * 4);
        device_free(dout, m * dim * 4);
    }

    /// Token-parallel BF16 SwiGLU (`M>1`) must match repeated `M=1` GEMVs.
    #[cfg(feature = "hip")]
    #[test]
    fn bf16_swiglu_gemm_rows_match_gemv() {
        if device_count() == 0 {
            return;
        }
        let m = 4usize;
        let dim = 32usize;
        let ff = 64usize;
        let mut x = vec![0.0f32; m * dim];
        let mut gate_f = vec![0.0f32; ff * dim];
        let mut up_f = vec![0.0f32; ff * dim];
        let mut down_f = vec![0.0f32; dim * ff];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in gate_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in up_f.iter_mut().enumerate() {
            *v = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in down_f.iter_mut().enumerate() {
            *v = ((i * 7) % 50) as f32 * 0.02 - 0.5;
        }
        let gate: Vec<u16> = gate_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_bf16)
            .collect();
        let up: Vec<u16> = up_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let down: Vec<u16> = down_f
            .iter()
            .copied()
            .map(allpaca_cpu::f32_to_bf16)
            .collect();
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 8).expect("hip");
        let gb = device_malloc(gate.len() * 2, 0).expect("gate");
        let ub = device_malloc(up.len() * 2, 0).expect("up");
        let db = device_malloc(down.len() * 2, 0).expect("down");
        set_device(0).expect("bind");
        memcpy(gb, gate.as_ptr().cast(), gate.len() * 2, MEMCPY_H2D).expect("h2d gate");
        memcpy(ub, up.as_ptr().cast(), up.len() * 2, MEMCPY_H2D).expect("h2d up");
        memcpy(db, down.as_ptr().cast(), down.len() * 2, MEMCPY_H2D).expect("h2d down");
        let mut rows = vec![0.0f32; m * dim];
        hip.bf16_swiglu_gemm(
            &x,
            gb.cast(),
            ub.cast(),
            db.cast(),
            &mut rows,
            m as i32,
            dim as i32,
            ff as i32,
        )
        .expect("swiglu gemm");
        let mut looped = vec![0.0f32; m * dim];
        for t in 0..m {
            hip.bf16_swiglu_gemv(
                &x[t * dim..(t + 1) * dim],
                gb.cast(),
                ub.cast(),
                db.cast(),
                &mut looped[t * dim..(t + 1) * dim],
                dim as i32,
                ff as i32,
            )
            .expect("swiglu gemv");
        }
        device_free(gb, gate.len() * 2);
        device_free(ub, up.len() * 2);
        device_free(db, down.len() * 2);
        for i in 0..m * dim {
            assert!(
                gemm_close(rows[i], looped[i]),
                "swiglu M={m} idx={i} rows={} gemv={}",
                rows[i],
                looped[i]
            );
        }
    }

    /// The §12 tiled BF16 GEMV must agree with the untiled baseline.
    #[cfg(feature = "hip")]
    #[test]
    fn bf16_tiled_gemv_matches_untiled() {
        if device_count() == 0 {
            return;
        }
        let (m, n, k) = (1usize, 96usize, 512usize);
        let mut a = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; n * k];
        for (i, v) in a.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        let b_bf16: Vec<u16> = b_f.iter().copied().map(allpaca_cpu::f32_to_bf16).collect();
        let mut tiled = vec![0.0f32; m * n];
        let mut untiled = vec![0.0f32; m * n];
        gemm_nt_bf16(&a, &b_bf16, &mut tiled, m as i32, n as i32, k as i32).expect("tiled BF16");
        gemm_nt_bf16_untiled(&a, &b_bf16, &mut untiled, m as i32, n as i32, k as i32)
            .expect("untiled BF16");
        for i in 0..m * n {
            assert!(
                gemm_close(tiled[i], untiled[i]),
                "BF16 tiled/untiled mismatch idx={i} tiled={} untiled={}",
                tiled[i],
                untiled[i]
            );
        }
    }
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
        gemm_nt_device(
            da.cast(),
            db32.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
        .unwrap();
        gemm_nt_f16_device(
            da.cast(),
            db16.cast(),
            dc.cast(),
            m as i32,
            n as i32,
            k as i32,
        )
        .unwrap();
        stream_synchronize().unwrap();
        // Best of five windows: other tests share the GPU, and one window can
        // land on their kernels.
        let best = |f: &dyn Fn()| -> f64 {
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    for _ in 0..10 {
                        f();
                    }
                    stream_synchronize().unwrap();
                    t.elapsed().as_secs_f64() * 1_000.0 / 10.0
                })
                .fold(f64::INFINITY, f64::min)
        };
        let f32_ms = best(&|| {
            gemm_nt_device(da.cast(), db32.cast(), dc.cast(), m as i32, n as i32, k as i32).unwrap();
        });
        let f16_ms = best(&|| {
            gemm_nt_f16_device(da.cast(), db16.cast(), dc.cast(), m as i32, n as i32, k as i32)
                .unwrap();
        });
        device_free(da, a_b);
        device_free(db32, b32_b);
        device_free(db16, b16_b);
        device_free(dc, c_b);
        eprintln!("device GEMV 1x{n}x{k} f32={f32_ms:.2}ms f16={f16_ms:.2}ms");
        // gfx9 (MI50 / Vega) converts F16 in software in this kernel; RDNA stays near F32.
        let max_ratio = match device_gcn_arch(0) {
            Some((arch, _)) if !arch_has_infinity_cache(&arch) => 20.0,
            _ => 8.0,
        };
        assert!(
            f16_ms < f32_ms * max_ratio + 2.0,
            "F16 GEMV {f16_ms:.2}ms vs F32 {f32_ms:.2}ms on 1x{n}x{k} (ratio cap {max_ratio})"
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
            eprintln!("f16 gemv device {id} ok in {:?}", t0.elapsed());
        }
    }

    /// The int8 kernels pick how many rows share a block from the row count
    /// (1, 2, 4..=8, and several blocks above 8). Every count gives each row
    /// the bits it gets alone.
    #[cfg(feature = "hip")]
    #[test]
    fn int8_gemv_rows_match_single_rows_for_every_row_count() {
        if device_count() == 0 || std::env::var("ALLPACA_Q8_F32_GEMV").is_ok_and(|v| v == "1") {
            return;
        }
        for m in 1usize..=17 {
            let (n, k) = (37usize, 512usize);
            let (a, b, _) = fill_q4k_gemm_case(m, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q4_k(a, &b, c, rows, n as i32, k as i32).expect("gemm_nt_q4_k")
            };
            let mut out = vec![0.0f32; m * n];
            run(&a, &mut out, m as i32);
            assert_rows_match_single("q4_k", run, &a, &out, m, n, k);

            let (a, b, _) = fill_q5k_gemm_case(m, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q5_k(a, &b, c, rows, n as i32, k as i32).expect("gemm_nt_q5_k")
            };
            let mut out = vec![0.0f32; m * n];
            run(&a, &mut out, m as i32);
            assert_rows_match_single("q5_k", run, &a, &out, m, n, k);

            let (a, b, _) = fill_q6k_gemm_case(m, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q6_k(a, &b, c, rows, n as i32, k as i32).expect("gemm_nt_q6_k")
            };
            let mut out = vec![0.0f32; m * n];
            run(&a, &mut out, m as i32);
            assert_rows_match_single("q6_k", run, &a, &out, m, n, k);

            if m <= 16 {
                let (n, k) = (37usize, 5120usize);
                let (a, b, _) = fill_q8_gemm_case(m + 90, m, n, k);
                let run = |a: &[f32], c: &mut [f32], rows: i32| {
                    gemm_nt_q8_0(a, &b, c, rows, n as i32, k as i32).expect("gemm_nt_q8_0")
                };
                let mut out = vec![0.0f32; m * n];
                run(&a, &mut out, m as i32);
                assert_rows_match_single("q8_0", run, &a, &out, m, n, k);
            }
        }
    }

    /// Wide-row Q8_0 weights (k >= 5120) take the int8-activation GEMV: every
    /// row batch matches its single-row result bit for bit and the CPU GEMM on
    /// int8-quantized activations; the f32 pair launch declines them.
    #[cfg(feature = "hip")]
    #[test]
    fn gemm_nt_q8_0_wide_rows_take_the_int8_gemv() {
        if device_count() == 0 || std::env::var("ALLPACA_Q8_F32_GEMV").is_ok_and(|v| v == "1") {
            return;
        }
        let (n, k) = (1000usize, 5120usize);
        for (si, m) in [1usize, 2, 3, 5, 9].into_iter().enumerate() {
            let (a, b_q8, _) = fill_q8_gemm_case(si + 70, m, n, k);
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut hip = vec![0.0f32; m * n];
            cpu_gemm_nt(&q8_activations(&a), &b_deq, &mut cpu, m, n, k);
            let run = |a: &[f32], c: &mut [f32], rows: i32| {
                gemm_nt_q8_0(a, &b_q8, c, rows, n as i32, k as i32).expect("gemm_nt_q8_0")
            };
            run(&a, &mut hip, m as i32);
            assert_rows_match_single("q8_0 large", run, &a, &hip, m, n, k);
            let scale = cpu.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            for i in 0..m * n {
                assert!(
                    (hip[i] - cpu[i]).abs() <= 1e-4 + 1e-4 * scale,
                    "q8_0 large m={m} idx={i} hip={} cpu={}",
                    hip[i],
                    cpu[i]
                );
            }
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
            cpu_gemm_nt(&q8_0_gemv_activations(&a, m, n, k), &b_deq, &mut cpu, m, n, k);
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
        cpu_gemm_nt(&q8_0_gemv_activations(&a, m, n, k), &b_deq, &mut cpu, m, n, k);
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

    #[cfg(feature = "hip")]
    fn cpu_attention_sinks(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        sinks: &[f32],
        out: &mut [f32],
        n_q: usize,
        n_kv: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_dim = n_kv_heads * head_dim;
        let prefix = n_kv.saturating_sub(n_q);
        out.fill(0.0);
        let mut scores = vec![0.0f32; n_kv];
        for qi in 0..n_q {
            let t_end = (qi + prefix + 1).min(n_kv);
            for h in 0..n_heads {
                let q_off = (qi * n_heads + h) * head_dim;
                let qh = &q[q_off..q_off + head_dim];
                let kv_h = (h * n_kv_heads) / n_heads.max(1);
                let sink = sinks.get(h).copied().unwrap_or(0.0);
                let mut max = sink;
                for t in 0..t_end {
                    let k_off = t * kv_dim + kv_h * head_dim;
                    let mut s = 0.0f32;
                    for i in 0..head_dim {
                        s += qh[i] * k[k_off + i];
                    }
                    s *= scale;
                    scores[t] = s;
                    if s > max {
                        max = s;
                    }
                }
                let mut acc = (sink - max).exp();
                for s in &mut scores[..t_end] {
                    *s = (*s - max).exp();
                    acc += *s;
                }
                let inv = 1.0 / acc.max(1e-9);
                for t in 0..t_end {
                    let w = scores[t] * inv;
                    let v_off = t * kv_dim + kv_h * head_dim;
                    for i in 0..head_dim {
                        out[q_off + i] += w * v[v_off + i];
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
            // Long rows: chunked softmax folded across 256-key chunks.
            (1, 3000, 4, 2, 128, true),
            (2, 1300, 2, 1, 64, true),
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

    /// The MoE SiLU's exp is the same function as the CPU reference (and the
    /// CUDA kernel's), bit for bit, so expert layers give the same values on
    /// AMD and NVIDIA workers.
    #[cfg(feature = "hip")]
    #[test]
    fn det_silu_mul_matches_cpu_reference_bitwise() {
        if device_count() == 0 {
            return;
        }
        // Every regime: tiny, ordinary, both overflow clamps, signed zero.
        let mut g: Vec<f32> = (0..40_000).map(|i| ((i as f32) - 20_000.0) * 0.0061).collect();
        g.extend([0.0, -0.0, 1e-30, -1e-30, 88.4, -88.4, 120.0, -120.0, 1e4, -1e4]);
        let u: Vec<f32> = (0..g.len()).map(|i| ((i % 13) as f32 - 6.0) * 0.37).collect();
        let want: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| allpaca_cpu::det_silu_mul(a, b)).collect();
        let got = det_silu_mul(&g, &u).unwrap();
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "g={} u={}: {a} vs {b}", g[i], u[i]);
        }
        // And it is exp to within 2 ulp where exp is representable.
        for x in [-80.0f32, -10.0, -1.0, -0.01, 0.0, 0.3, 1.0, 5.0, 40.0, 88.0] {
            let (d, e) = (allpaca_cpu::det_expf(x), x.exp());
            assert!((d - e).abs() <= 2.0 * f32::EPSILON * e.abs().max(f32::MIN_POSITIVE), "exp({x}) {d} vs {e}");
        }
    }

    /// Several query rows (the Q-tiled kernel, keys staged in LDS) give each
    /// row bit for bit what a one-row call over the keys it sees gives, so
    /// prefill, verify and decode rows agree.
    #[cfg(feature = "hip")]
    #[test]
    fn attention_rows_equal_single_row_calls() {
        if device_count() == 0 {
            return;
        }
        let shapes = [
            (13, 40, 4, 2, 128, true),
            (9, 9, 8, 8, 64, true),
            (20, 300, 6, 2, 128, true),
            (3, 5, 2, 1, 4, true),
            (17, 50, 2, 2, 96, true),
            (11, 70, 4, 4, 256, true),
            (6, 33, 4, 2, 128, false),
            // Several 256-key chunks: one-row calls fold them in parallel
            // (split + merge), multi-row calls fold them in the Q-tiled kernel.
            (5, 1000, 8, 2, 128, true),
            (3, 2049, 4, 4, 64, true),
            (2, 700, 4, 1, 128, false),
            (9, 513, 2, 2, 96, true),
        ];
        for (si, (n_q, n_kv, n_heads, n_kv_heads, head_dim, causal)) in
            shapes.iter().copied().enumerate()
        {
            let q_row = n_heads * head_dim;
            let kv_row = n_kv_heads * head_dim;
            let fill = |n: usize, a: usize, b: usize| -> Vec<f32> {
                (0..n).map(|i| ((i * a + si * b) % 97) as f32 * 0.021 - 1.0).collect()
            };
            let q = fill(n_q * q_row, 17, 3);
            let k = fill(n_kv * kv_row, 13, 7);
            let v = fill(n_kv * kv_row, 11, 5);
            let mut rows = vec![0.0f32; n_q * q_row];
            attention(
                &q, &k, &v, &mut rows, n_q as i32, n_kv as i32, n_heads as i32,
                n_kv_heads as i32, head_dim as i32, causal,
            )
            .unwrap();
            for i in 0..n_q {
                let seen = if causal { n_kv - n_q + i + 1 } else { n_kv };
                let mut one = vec![0.0f32; q_row];
                attention(
                    &q[i * q_row..(i + 1) * q_row],
                    &k[..seen * kv_row],
                    &v[..seen * kv_row],
                    &mut one,
                    1,
                    seen as i32,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    causal,
                )
                .unwrap();
                let got = &rows[i * q_row..(i + 1) * q_row];
                assert!(
                    got.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "row {i} of n_q={n_q} n_kv={n_kv} heads={n_heads}/{n_kv_heads} d={head_dim} causal={causal} differs from its one-row call"
                );
            }
        }
    }

    /// Batched causal `attention_sinks` (scratch and malloc) must match CPU sinks.
    #[cfg(feature = "hip")]
    #[test]
    fn attention_sinks_batched_matches_cpu_and_malloc() {
        if device_count() == 0 {
            return;
        }
        let n_q = 16usize;
        let n_kv = 24usize;
        let n_heads = 8usize;
        let n_kv_heads = 2usize;
        let head_dim = 64usize;
        let qn = n_q * n_heads * head_dim;
        let kn = n_kv * n_kv_heads * head_dim;
        let mut q = vec![0.0f32; qn];
        let mut k = vec![0.0f32; kn];
        let mut v = vec![0.0f32; kn];
        let mut sinks = vec![0.0f32; n_heads];
        for (i, x) in q.iter_mut().enumerate() {
            *x = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in k.iter_mut().enumerate() {
            *x = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in sinks.iter_mut().enumerate() {
            *x = (i as f32) * 0.07 - 0.3;
        }
        let mut cpu = vec![0.0f32; qn];
        cpu_attention_sinks(
            &q, &k, &v, &sinks, &mut cpu, n_q, n_kv, n_heads, n_kv_heads, head_dim,
        );
        let scratch = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("8MiB scratch");
        let mut hip_scratch = vec![0.0f32; qn];
        scratch
            .attention_sinks(
                &q,
                &k,
                &v,
                &sinks,
                &mut hip_scratch,
                n_q as i32,
                n_kv as i32,
                n_heads as i32,
                n_kv_heads as i32,
                head_dim as i32,
                true,
            )
            .expect("scratch attention_sinks");
        let tiny = HipWorkspace::new(0, 4096, 64).expect("4KiB scratch");
        let mut hip_malloc = vec![0.0f32; qn];
        tiny.attention_sinks(
            &q,
            &k,
            &v,
            &sinks,
            &mut hip_malloc,
            n_q as i32,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            true,
        )
        .expect("malloc attention_sinks");
        for i in 0..qn {
            assert!(
                gemm_close(hip_scratch[i], cpu[i]),
                "scratch sinks mismatch idx={i} hip={} cpu={}",
                hip_scratch[i],
                cpu[i]
            );
            assert!(
                gemm_close(hip_malloc[i], cpu[i]),
                "malloc sinks mismatch idx={i} hip={} cpu={}",
                hip_malloc[i],
                cpu[i]
            );
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

    /// Batched mean/var LayerNorm must match per-row CPU LN (ViT / CLIP; not RMSNorm).
    #[cfg(feature = "hip")]
    #[test]
    fn layernorm_rows_matches_cpu_per_row() {
        if device_count() == 0 {
            return;
        }
        let eps = 1e-5f32;
        let shapes = [(16, 32), (9, 8), (1, 64)];
        for (si, (rows, n)) in shapes.iter().copied().enumerate() {
            let mut x = vec![0.0f32; rows * n];
            let mut weight = vec![0.0f32; n];
            let mut bias = vec![0.0f32; n];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5;
            }
            for (i, v) in weight.iter_mut().enumerate() {
                *v = 0.5 + ((i * 13 + si) % 20) as f32 * 0.05;
            }
            for (i, v) in bias.iter_mut().enumerate() {
                *v = ((i * 7 + si) % 11) as f32 * 0.03 - 0.15;
            }
            let mut cpu = vec![0.0f32; rows * n];
            for r in 0..rows {
                let row = &x[r * n..(r + 1) * n];
                let dst = &mut cpu[r * n..(r + 1) * n];
                let nf = n as f32;
                let mean = row.iter().sum::<f32>() / nf;
                let mut var = 0.0f32;
                for &v in row {
                    let d = v - mean;
                    var += d * d;
                }
                var /= nf;
                let inv = 1.0 / (var + eps).sqrt();
                for i in 0..n {
                    dst[i] = (row[i] - mean) * inv * weight[i] + bias[i];
                }
            }
            let mut hip = vec![0.0f32; rows * n];
            layernorm_rows(&x, &weight, &bias, &mut hip, rows as i32, n as i32, eps)
                .expect("HIP layernorm_rows must run when a device is present");
            for i in 0..rows * n {
                assert!(
                    gemm_close(hip[i], cpu[i]),
                    "layernorm_rows mismatch rows={rows} n={n} idx={i} hip={} cpu={}",
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

    /// Partial RoPE (`rope_offset`) on device must match CPU per head, including invert.
    #[cfg(feature = "hip")]
    #[test]
    fn rope_offset_device_rows_matches_cpu_and_inverts() {
        if device_count() == 0 {
            return;
        }
        let theta = 10_000.0f32;
        let n_tok = 3usize;
        let n_heads = 4usize;
        let head_dim = 8usize;
        let n_rot = 4usize;
        let offset = 2usize;
        let start_pos = 5usize;
        let mut x = vec![0.0f32; n_tok * n_heads * head_dim];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        let mut cpu = x.clone();
        for t in 0..n_tok {
            let pos = start_pos + t;
            for h in 0..n_heads {
                let s = (t * n_heads + h) * head_dim;
                allpaca_cpu::rope_offset(&mut cpu[s..s + head_dim], pos, theta, n_rot, offset)
                    .expect("cpu rope_offset");
            }
        }
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        hip.bind().expect("bind");
        let dx = device_malloc(x.len() * 4, 0).expect("dx");
        memcpy(dx, x.as_ptr().cast(), x.len() * 4, MEMCPY_H2D).expect("h2d x");
        hip.rope_offset_device_rows(
            dx.cast(),
            n_tok as i32,
            n_heads as i32,
            head_dim as i32,
            n_rot as i32,
            offset as i32,
            start_pos as i32,
            theta,
            false,
        )
        .expect("device rope_offset");
        let mut got = vec![0.0f32; x.len()];
        memcpy(got.as_mut_ptr().cast(), dx, got.len() * 4, MEMCPY_D2H).expect("d2h");
        for i in 0..cpu.len() {
            assert!(
                gemm_close(got[i], cpu[i]),
                "rope_offset mismatch idx={i} hip={} cpu={}",
                got[i],
                cpu[i]
            );
        }
        hip.rope_offset_device_rows(
            dx.cast(),
            n_tok as i32,
            n_heads as i32,
            head_dim as i32,
            n_rot as i32,
            offset as i32,
            start_pos as i32,
            theta,
            true,
        )
        .expect("device rope_offset_back");
        memcpy(got.as_mut_ptr().cast(), dx, got.len() * 4, MEMCPY_D2H).expect("d2h back");
        for i in 0..x.len() {
            assert!(
                gemm_close(got[i], x[i]),
                "rope_offset_back must invert idx={i} hip={} orig={}",
                got[i],
                x[i]
            );
        }
        device_free(dx, x.len() * 4);
    }

    /// Device-Q attention must match host-Q attention without H2D of Q.
    #[cfg(feature = "hip")]
    #[test]
    fn attention_sinks_from_device_q_skips_q_h2d() {
        if device_count() == 0 {
            return;
        }
        let n_q = 3usize;
        let n_kv = 5usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 8usize;
        let qn = n_q * n_heads * head_dim;
        let kn = n_kv * n_kv_heads * head_dim;
        let mut q = vec![0.0f32; qn];
        let mut k = vec![0.0f32; kn];
        let mut v = vec![0.0f32; kn];
        let sinks = vec![0.1f32, -0.2];
        for (i, x) in q.iter_mut().enumerate() {
            *x = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in k.iter_mut().enumerate() {
            *x = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        let mut host = vec![0.0f32; qn];
        hip.attention_sinks(
            &q,
            &k,
            &v,
            &sinks,
            &mut host,
            n_q as i32,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            true,
        )
        .expect("host-Q attention");
        hip.bind().expect("bind");
        let dq = device_malloc(q.len() * 4, 0).expect("dq");
        memcpy(dq, q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D).expect("seed q");
        let h2d_before = hip.activation_h2d_bytes();
        let mut got = vec![0.0f32; qn];
        hip.attention_sinks_from_device_q(
            dq.cast(),
            &k,
            &v,
            &sinks,
            &mut got,
            n_q as i32,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            true,
        )
        .expect("device-Q attention");
        let h2d_delta = hip.activation_h2d_bytes().saturating_sub(h2d_before);
        let kv_sink_bytes = ((k.len() + v.len() + sinks.len()) * 4) as u64;
        let q_bytes = (q.len() * 4) as u64;
        assert_eq!(
            h2d_delta, kv_sink_bytes,
            "device-Q attention H2D should be k+v+sinks only (delta={h2d_delta} q={q_bytes})"
        );
        for i in 0..host.len() {
            assert!(
                gemm_close(got[i], host[i]),
                "device-Q attention mismatch idx={i} hip={} host={}",
                got[i],
                host[i]
            );
        }
        device_free(dq, q.len() * 4);
    }

    /// Device-Q attention into a device out buffer must skip out D2H.
    #[cfg(feature = "hip")]
    #[test]
    fn attention_sinks_from_device_q_into_skips_out_d2h() {
        if device_count() == 0 {
            return;
        }
        let n_q = 3usize;
        let n_kv = 5usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 8usize;
        let qn = n_q * n_heads * head_dim;
        let kn = n_kv * n_kv_heads * head_dim;
        let mut q = vec![0.0f32; qn];
        let mut k = vec![0.0f32; kn];
        let mut v = vec![0.0f32; kn];
        let sinks = vec![0.1f32, -0.2];
        for (i, x) in q.iter_mut().enumerate() {
            *x = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in k.iter_mut().enumerate() {
            *x = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = ((i * 11) % 50) as f32 * 0.02 - 0.5;
        }
        let hip = HipWorkspace::new(0, 8 * 1024 * 1024, 64).expect("hip workspace");
        let mut host = vec![0.0f32; qn];
        hip.attention_sinks(
            &q,
            &k,
            &v,
            &sinks,
            &mut host,
            n_q as i32,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            true,
        )
        .expect("host-Q attention");
        hip.bind().expect("bind");
        let dq = device_malloc(q.len() * 4, 0).expect("dq");
        let dout = device_malloc(host.len() * 4, 0).expect("dout");
        memcpy(dq, q.as_ptr().cast(), q.len() * 4, MEMCPY_H2D).expect("seed q");
        let h2d_before = hip.activation_h2d_bytes();
        let d2h_before = hip.activation_d2h_bytes();
        hip.attention_sinks_from_device_q_into(
            dq.cast(),
            &k,
            &v,
            &sinks,
            dout.cast(),
            n_q as i32,
            n_kv as i32,
            n_heads as i32,
            n_kv_heads as i32,
            head_dim as i32,
            true,
        )
        .expect("device-Q into device out");
        let h2d_delta = hip.activation_h2d_bytes().saturating_sub(h2d_before);
        let d2h_delta = hip.activation_d2h_bytes().saturating_sub(d2h_before);
        assert_eq!(
            h2d_delta,
            ((k.len() + v.len() + sinks.len()) * 4) as u64,
            "device-out attention H2D should be k+v+sinks only"
        );
        assert_eq!(d2h_delta, 0, "device-out attention must not D2H out");
        let mut got = vec![0.0f32; host.len()];
        memcpy(got.as_mut_ptr().cast(), dout, got.len() * 4, MEMCPY_D2H).expect("d2h check");
        for i in 0..host.len() {
            assert!(
                gemm_close(got[i], host[i]),
                "device-out attention mismatch idx={i} hip={} host={}",
                got[i],
                host[i]
            );
        }
        device_free(dq, q.len() * 4);
        device_free(dout, host.len() * 4);
    }

    /// Device IMRoPE must match CPU `rope_imrope_neox_heads` (partial + sections).
    #[cfg(feature = "hip")]
    #[test]
    fn rope_imrope_qk_matches_cpu() {
        if device_count() == 0 {
            return;
        }
        let n_q = 4usize;
        let n_k = 2usize;
        let head_dim = 64usize;
        let n_rot = 32usize;
        // Image-token (t, row, column) positions.
        let pos = [7usize, 3, 11];
        let theta = 10_000.0f32;
        let sections = [5i32, 5, 6, 0];
        let mut q = vec![0.0f32; n_q * head_dim];
        let mut k = vec![0.0f32; n_k * head_dim];
        for (i, v) in q.iter_mut().enumerate() {
            *v = ((i * 13) % 50) as f32 * 0.02 - 0.5;
        }
        for (i, v) in k.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.4;
        }
        let mut cpu_q = q.clone();
        let mut cpu_k = k.clone();
        allpaca_cpu::rope_imrope_neox_heads_at(&mut cpu_q, n_q, head_dim, pos, theta, n_rot, sections)
            .unwrap();
        allpaca_cpu::rope_imrope_neox_heads_at(&mut cpu_k, n_k, head_dim, pos, theta, n_rot, sections)
            .unwrap();
        let hip = HipWorkspace::new(0, (n_q + n_k) * head_dim * 4 + 4096, 8).expect("hip");
        let q_bytes = q.len() * 4;
        let k_bytes = k.len() * 4;
        let dq = device_malloc(q_bytes, 0).unwrap();
        let dk = device_malloc(k_bytes, 0).unwrap();
        memcpy(dq, q.as_ptr().cast(), q_bytes, MEMCPY_H2D).unwrap();
        memcpy(dk, k.as_ptr().cast(), k_bytes, MEMCPY_H2D).unwrap();
        hip.rope_imrope_qk_device(
            dq.cast(),
            dk.cast(),
            n_q as i32,
            n_k as i32,
            head_dim as i32,
            n_rot as i32,
            pos.map(|p| p as i32),
            theta,
            sections,
        )
        .expect("imrope device");
        memcpy(q.as_mut_ptr().cast(), dq, q_bytes, MEMCPY_D2H).unwrap();
        memcpy(k.as_mut_ptr().cast(), dk, k_bytes, MEMCPY_D2H).unwrap();
        device_free(dq, q_bytes);
        device_free(dk, k_bytes);
        for i in 0..q.len() {
            assert!(
                gemm_close(q[i], cpu_q[i]),
                "imrope Q mismatch idx={i} hip={} cpu={}",
                q[i],
                cpu_q[i]
            );
        }
        for i in 0..k.len() {
            assert!(
                gemm_close(k[i], cpu_k[i]),
                "imrope K mismatch idx={i} hip={} cpu={}",
                k[i],
                cpu_k[i]
            );
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

    /// Prefill-shaped batched ROCmFP4 embed gather must match CPU dequant rows.
    #[cfg(feature = "hip")]
    #[test]
    fn embed_gather_rocmfp4_rows_matches_dequant_rows() {
        if device_count() == 0 {
            return;
        }
        let vocab = 8usize;
        let dim = 64usize;
        let tokens = [0u32, 3, 7, 1, 3, 2];
        let mut table = vec![0.0f32; vocab * dim];
        for (i, v) in table.iter_mut().enumerate() {
            *v = ((i * 17) % 50) as f32 * 0.02 - 0.5;
        }
        let mut packed = Vec::new();
        for row in 0..vocab {
            packed.extend(allpaca_cpu::quantize_rocmfp4(
                &table[row * dim..(row + 1) * dim],
            ));
        }
        let mut cpu = vec![0.0f32; tokens.len() * dim];
        for (r, &tok) in tokens.iter().enumerate() {
            let row_off = tok as usize * dim.div_ceil(32) * ROCMFP4_BLOCK_BYTES;
            allpaca_cpu::dequant_rocmfp4(
                &packed[row_off..row_off + dim.div_ceil(32) * ROCMFP4_BLOCK_BYTES],
                dim,
                &mut cpu[r * dim..(r + 1) * dim],
            )
            .unwrap();
        }
        let mut hip = vec![0.0f32; tokens.len() * dim];
        embed_gather_rocmfp4_rows(&packed, &tokens, &mut hip, dim as i32)
            .expect("HIP embed_gather_rocmfp4_rows must run when a device is present");
        for i in 0..cpu.len() {
            assert!(
                gemm_close(hip[i], cpu[i]),
                "embed_gather_rocmfp4_rows mismatch idx={i} hip={} cpu={}",
                hip[i],
                cpu[i]
            );
        }
    }
}
