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
use std::sync::Mutex;
use std::time::Instant;

pub const ALLPACA_OK: i32 = 0;
pub const ALLPACA_ERR_NOT_IMPLEMENTED: i32 = 1;
pub const ALLPACA_ERR_INVALID: i32 = 2;
pub const ALLPACA_ERR_DEVICE: i32 = 3;

pub const MEMCPY_H2D: i32 = 0;
pub const MEMCPY_D2H: i32 = 1;
pub const MEMCPY_D2D: i32 = 2;
pub const MEMCPY_DEFAULT: i32 = 3;
pub const MEMCPY_D2H_ASYNC: i32 = 4;

/// GGUF Q8_0 packed block (f16 scale + 32 qs).
pub const Q8_PACKED_BLOCK: usize = 34;
/// Device-staged Q8_0: f32 scale at 0, qs at 4, 36-byte stride.
pub const Q8_STAGED_BLOCK: usize = 36;
pub const Q8_STAGED_QS: usize = 4;

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

/// Device scratch [`CudaWorkspace::moe_ffn_kquant_dev`] needs past its inputs.
pub fn moe_ffn_scratch_bytes(rows: usize, dim: usize, ff: usize, pairs: usize) -> u64 {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        unsafe { ffi::allpaca_cuda_moe_ffn_scratch_bytes(rows as i32, dim as i32, ff as i32, pairs as i32) }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (rows, dim, ff, pairs);
        0
    }
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
        pub fn allpaca_cuda_host_register(ptr: *mut c_void, bytes: usize, flags: u32) -> i32;
        pub fn allpaca_cuda_host_unregister(ptr: *mut c_void) -> i32;
        pub fn allpaca_cuda_memcpy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_memcpy_batch(
            dsts: *mut *mut c_void,
            srcs: *mut *mut c_void,
            sizes: *mut usize,
            count: usize,
            kind: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_memset(
            dst: *mut c_void,
            value: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_set_device(device: i32) -> i32;
        pub fn allpaca_cuda_can_p2p(src: i32, dst: i32, can: *mut i32) -> i32;
        pub fn allpaca_cuda_enable_peer_access(src: i32, dst: i32) -> i32;
        pub fn allpaca_cuda_memcpy_peer(
            dst: *mut c_void,
            dst_dev: i32,
            src: *const c_void,
            src_dev: i32,
            bytes: usize,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_mem_info(
            device: i32,
            free_bytes: *mut u64,
            total_bytes: *mut u64,
        ) -> i32;
        pub fn allpaca_cuda_mem_used(device: i32, used_bytes: *mut u64) -> i32;
        pub fn allpaca_cuda_stream_synchronize(stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_stream_create(stream: *mut *mut c_void) -> i32;
        pub fn allpaca_cuda_stream_destroy(stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_stream_wait_event(stream: *mut c_void, event: *mut c_void) -> i32;
        pub fn allpaca_cuda_event_create(event: *mut *mut c_void) -> i32;
        pub fn allpaca_cuda_event_create_timing(event: *mut *mut c_void) -> i32;
        pub fn allpaca_cuda_event_destroy(event: *mut c_void) -> i32;
        pub fn allpaca_cuda_event_record(event: *mut c_void, stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_event_elapsed_us(
            start: *mut c_void,
            stop: *mut c_void,
            us: *mut f64,
        ) -> i32;
        pub fn allpaca_cuda_gemm_nt(
            a: *const f32,
            b: *const f32,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_gemm_nt_kquant(
            qt: i32,
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_moe_ffn_scratch_bytes(rows: i32, dim: i32, ff: i32, pairs: i32) -> u64;
        pub fn allpaca_cuda_det_silu_mul(
            g: *const f32,
            u: *const f32,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_moe_ffn_kquant(
            x: *const f32,
            xq: *const i8,
            xd: *const f32,
            rows: i32,
            dim: i32,
            ff: i32,
            pairs: i32,
            pair_row: *const i32,
            expert: *const i32,
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
        pub fn allpaca_cuda_gemm_nt_q8_0(
            a: *const f32,
            b: *const u8,
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
        pub fn allpaca_cuda_rmsnorm_rows(
            x: *const f32,
            w: *const f32,
            out: *mut f32,
            rows: i32,
            n: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_rmsnorm_heads(
            x: *mut f32,
            weight: *const f32,
            n_heads: i32,
            head_dim: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_deinterleave_q_gate(
            q_full: *const f32,
            q: *mut f32,
            gate: *mut f32,
            n_heads: i32,
            head_dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_sigmoid_mul(
            x: *mut f32,
            gate: *const f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_silu_mul(
            gate: *const f32,
            up: *const f32,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_gelu_mul(
            gate: *const f32,
            up: *const f32,
            out: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_relu_sqr(x: *mut f32, n: i32, stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_silu(x: *mut f32, n: i32, stream: *mut c_void) -> i32;
        pub fn allpaca_cuda_scale_f32(
            x: *mut f32,
            n: i32,
            scale: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_expand_k_heads(
            src: *const f32,
            dst: *mut f32,
            n_k: i32,
            n_v: i32,
            head_k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_gdn_decay_beta(
            alpha: *const f32,
            dt_bias: *const f32,
            a: *const f32,
            decay: *mut f32,
            beta: *mut f32,
            n: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_gdn_conv1d_step(
            x: *const f32,
            kernel: *const f32,
            conv_state: *mut f32,
            out: *mut f32,
            qkv_dim: i32,
            conv_kernel: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_gated_delta_step(
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
        pub fn allpaca_cuda_mamba2_conv_step(
            xbc: *const f32,
            kernel: *const f32,
            bias: *const f32,
            conv_state: *mut f32,
            out: *mut f32,
            conv_dim: i32,
            conv_kernel: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_mamba2_ssd_step(
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
        pub fn allpaca_cuda_scale_heads_sigmoid(
            attn: *mut f32,
            scores: *const f32,
            n_tok: i32,
            n_heads: i32,
            head_dim: i32,
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
        pub fn allpaca_cuda_rope_qk_rows(
            q: *mut f32,
            k: *mut f32,
            n_tok: i32,
            n_q_heads: i32,
            n_k_heads: i32,
            head_dim: i32,
            q_stride: i32,
            k_stride: i32,
            start_pos: i32,
            theta: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_rope_imrope_qk(
            q: *mut f32,
            k: *mut f32,
            n_q_heads: i32,
            n_k_heads: i32,
            head_dim: i32,
            n_rot: i32,
            pos: i32,
            theta: f32,
            sections: *const i32,
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
        pub fn allpaca_cuda_attention_sinks(
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
        pub fn allpaca_cuda_attention_masked(
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
        pub fn allpaca_cuda_attention_softcap(
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
            softcap: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_attention_masked_softcap(
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
            softcap: f32,
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

pub fn can_p2p(src: i32, dst: i32) -> bool {
    if src == dst {
        return false;
    }
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        unsafe {
            let mut can = 0;
            ffi::allpaca_cuda_can_p2p(src, dst, &mut can) == ALLPACA_OK && can != 0
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (src, dst);
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerAccess {
    pub capable: bool,
    pub enabled: bool,
}

pub fn enable_peer_access(src: i32, dst: i32) -> bool {
    if src == dst {
        return false;
    }
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        unsafe { ffi::allpaca_cuda_enable_peer_access(src, dst) == ALLPACA_OK }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (src, dst);
        false
    }
}

pub fn memcpy_peer(
    dst: *mut std::ffi::c_void,
    dst_dev: i32,
    src: *const std::ffi::c_void,
    src_dev: i32,
    bytes: usize,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_cuda_memcpy_peer(dst, dst_dev, src, src_dev, bytes, std::ptr::null_mut())
        };
        status(rc, "memcpy_peer")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (dst, dst_dev, src, src_dev, bytes);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA memcpy_peer unavailable".into(),
        )))
    }
}

fn peer_copy_reachable(src: i32, dst: i32) -> bool {
    const N: usize = 64 * 1024;
    let Ok(s) = device_malloc(N, src) else {
        return false;
    };
    let Ok(d) = device_malloc(N, dst) else {
        device_free(s);
        return false;
    };
    let ok = memcpy_peer(d, dst, s, src, N).is_ok();
    device_free(s);
    device_free(d);
    ok
}

/// Query, attempt enable, re-query. `capable` is post-enable *or* a working peer copy.
pub fn probe_peer(src: i32, dst: i32) -> PeerAccess {
    if src == dst {
        return PeerAccess {
            capable: false,
            enabled: false,
        };
    }
    let queried = can_p2p(src, dst);
    let enabled = enable_peer_access(src, dst);
    let mut capable = can_p2p(src, dst);
    if !capable {
        capable = peer_copy_reachable(src, dst);
    }
    PeerAccess {
        capable: capable || queried,
        enabled,
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
            let mut peers = Vec::new();
            for other in 0..count {
                if other != id && probe_peer(id, other).capable {
                    peers.push(other);
                }
            }
            out.push(DeviceInfo {
                id,
                kind: DeviceKind::CudaGpu,
                name,
                memory_bytes: mem,
                bandwidth_gbps: bw,
                peers,
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

/// Driver-measured used VRAM (`total - free`) for `device`.
pub fn device_memory_used(device: i32) -> Result<u64, AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut used = 0u64;
        let rc = unsafe { ffi::allpaca_cuda_mem_used(device, &mut used) };
        status(rc, "mem_used")?;
        Ok(used)
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = device;
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA mem_used unavailable".into(),
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

/// Pin existing host memory for async DMA (mirrors HIP `host_register`).
pub fn host_register(ptr: *mut u8, bytes: usize, flags: u32) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if ptr.is_null() || bytes == 0 {
            return Ok(());
        }
        let rc = unsafe { ffi::allpaca_cuda_host_register(ptr.cast(), bytes, flags) };
        status(rc, "host_register")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (ptr, bytes, flags);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA host_register unavailable".into(),
        )))
    }
}

pub fn host_unregister(ptr: *mut u8) {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if !ptr.is_null() {
            unsafe {
                let _ = ffi::allpaca_cuda_host_unregister(ptr.cast());
            }
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = ptr;
    }
}

/// RAII pin of a host byte range. Unregisters on drop.
pub struct HostRegistration {
    ptr: *mut u8,
}

unsafe impl Send for HostRegistration {}
unsafe impl Sync for HostRegistration {}

impl HostRegistration {
    pub fn new(ptr: *mut u8, bytes: usize) -> Result<Self, AllpacaError> {
        host_register(ptr, bytes, 0)?;
        Ok(Self { ptr })
    }
}

impl Drop for HostRegistration {
    fn drop(&mut self) {
        host_unregister(self.ptr);
        self.ptr = std::ptr::null_mut();
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

/// Fill `bytes` of device memory with `value` (low 8 bits). Null stream is synchronous.
pub fn device_memset(
    dst: *mut std::ffi::c_void,
    value: u8,
    bytes: usize,
) -> Result<(), AllpacaError> {
    memset_on(dst, value, bytes, std::ptr::null_mut())
}

/// Fill on `stream` (`cudaMemsetAsync`). Null stream uses synchronous `cudaMemset`.
pub fn memset_on(
    dst: *mut std::ffi::c_void,
    value: u8,
    bytes: usize,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    if bytes == 0 {
        return Ok(());
    }
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_memset(dst, i32::from(value), bytes, stream) };
        status(rc, "memset")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (dst, value, bytes, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA memset unavailable".into(),
        )))
    }
}

/// Batch of 1D copies on one stream. Empty `copies` is a no-op.
///
/// # Safety
/// Each `dst`/`src` must stay live for `bytes` until the stream drains.
pub unsafe fn memcpy_batch_on(
    copies: &[(*mut u8, *const u8, usize)],
    kind: i32,
    stream: *mut std::ffi::c_void,
) -> Result<(), AllpacaError> {
    if copies.is_empty() {
        return Ok(());
    }
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut dsts: Vec<*mut std::ffi::c_void> =
            copies.iter().map(|(d, _, _)| d.cast()).collect();
        let mut srcs: Vec<*mut std::ffi::c_void> =
            copies.iter().map(|(_, s, _)| s.cast_mut().cast()).collect();
        let mut sizes: Vec<usize> = copies.iter().map(|(_, _, n)| *n).collect();
        let rc = unsafe {
            ffi::allpaca_cuda_memcpy_batch(
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
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (copies, kind, stream);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA memcpy_batch unavailable".into(),
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

fn event_create_timing_raw() -> Result<*mut std::ffi::c_void, AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let mut event = std::ptr::null_mut();
        let rc = unsafe { ffi::allpaca_cuda_event_create_timing(&mut event) };
        status(rc, "event_create_timing")?;
        Ok(event)
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA event_create_timing unavailable".into(),
        )))
    }
}

fn event_destroy_raw(event: *mut std::ffi::c_void) {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        if !event.is_null() {
            unsafe {
                let _ = ffi::allpaca_cuda_event_destroy(event);
            }
        }
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = event;
    }
}

/// Last decode sample: GPU kernel time vs host wall remainder (microseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeProfile {
    pub gpu_us: f64,
    pub host_us: f64,
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

/// Device-pointer float A × staged Q8_0 B NT GEMM (HIP `gemm_nt_q8_0_device`).
pub fn gemm_nt_q8_0_device(
    a: *const f32,
    b: *const u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe { ffi::allpaca_cuda_gemm_nt_q8_0(a, b, c, m, n, k, std::ptr::null_mut()) };
        status(rc, "gemm_nt_q8_0")?;
        let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
        status(sync, "gemm_nt_q8_0 sync")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA gemm_nt_q8_0 unavailable".into(),
        )))
    }
}

/// Host-pointer Q8_0 NT GEMM: stages GGUF blocks, copies to device, runs kernel, copies back.
pub fn gemm_nt_q8_0(
    a: &[f32],
    b: &[u8],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    if device_count() <= 0 {
        return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA gemm_nt_q8_0: no device".into(),
        )));
    }
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
        memcpy(db, staged.as_ptr().cast(), staged.len(), MEMCPY_H2D)?;
        gemm_nt_q8_0_device(da.cast(), db.cast(), dc.cast(), m, n, k)?;
        memcpy(c.as_mut_ptr().cast(), dc, c_n * 4, MEMCPY_D2H)
    })();
    device_free(da);
    device_free(db);
    device_free(dc);
    result
}
/// Kernels enqueue on the workspace stream; call [`CudaWorkspace::synchronize`]
/// (or a host roundtrip) to batch-wait — no per-op device sync.

/// Multi-token activation region for CUDA chunked prefill (M=N GEMM / attention).
#[derive(Clone, Copy)]
pub struct CudaPrefillActs {
    pub x: *mut f32,
    pub norm: *mut f32,
    pub q: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub attn: *mut f32,
    pub proj: *mut f32,
    pub gate: *mut f32,
    pub up: *mut f32,
    pub bytes: usize,
    pub n_tok: usize,
    pub dim: usize,
    pub kv_dim: usize,
    pub ff_dim: usize,
}

pub struct CudaWorkspace {
    pub device: i32,
    scratch: *mut std::ffi::c_void,
    scratch_bytes: usize,
    stream: *mut std::ffi::c_void,
    t0: *mut std::ffi::c_void,
    t1: *mut std::ffi::c_void,
    profile_wall: Mutex<Option<Instant>>,
    ema_gpu_us: AtomicU64,
    ema_host_us: AtomicU64,
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
        let t0 = match event_create_timing_raw() {
            Ok(e) => e,
            Err(e) => {
                device_free(scratch);
                stream_destroy_raw(stream);
                return Err(e);
            }
        };
        let t1 = match event_create_timing_raw() {
            Ok(e) => e,
            Err(e) => {
                event_destroy_raw(t0);
                device_free(scratch);
                stream_destroy_raw(stream);
                return Err(e);
            }
        };
        Ok(Self {
            device,
            scratch,
            scratch_bytes,
            stream,
            t0,
            t1,
            profile_wall: Mutex::new(None),
            ema_gpu_us: AtomicU64::new(0),
            ema_host_us: AtomicU64::new(0),
            activation_h2d: AtomicU64::new(0),
            activation_d2h: AtomicU64::new(0),
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.scratch_bytes as u64
    }

    /// Bytes for dense prefill acts: x/norm/q/attn/proj (`n_tok*dim`), k/v (`n_tok*kv_dim`),
    /// gate/up (`n_tok*ff_dim`). Packed KV uses the remainder via `kv_scratch_pair_after`.
    pub fn prefill_act_bytes(n_tok: usize, dim: usize, kv_dim: usize, ff_dim: usize) -> usize {
        n_tok
            .saturating_mul(
                5usize
                    .saturating_mul(dim)
                    .saturating_add(2usize.saturating_mul(kv_dim))
                    .saturating_add(2usize.saturating_mul(ff_dim)),
            )
            .saturating_mul(4)
    }

    /// Multi-token activation region for CUDA M=N prefill.
    pub fn prefill_acts(
        &self,
        n_tok: usize,
        dim: usize,
        kv_dim: usize,
        ff_dim: usize,
    ) -> Result<CudaPrefillActs, AllpacaError> {
        let bytes = Self::prefill_act_bytes(n_tok, dim, kv_dim, ff_dim);
        if n_tok == 0 || dim == 0 || kv_dim == 0 || bytes == 0 {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                "CUDA prefill acts need n_tok>0, dim>0, kv_dim>0".into(),
            )));
        }
        if bytes > self.scratch_bytes {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                format!(
                    "CUDA prefill acts ({bytes} bytes for n_tok={n_tok}) exceed scratch {}",
                    self.scratch_bytes
                ),
            )));
        }
        let base = self.scratch.cast::<f32>();
        let row_x = n_tok * dim;
        let row_kv = n_tok * kv_dim;
        let row_ff = n_tok * ff_dim;
        Ok(CudaPrefillActs {
            x: base,
            norm: unsafe { base.add(row_x) },
            q: unsafe { base.add(2 * row_x) },
            k: unsafe { base.add(3 * row_x) },
            v: unsafe { base.add(3 * row_x + row_kv) },
            attn: unsafe { base.add(3 * row_x + 2 * row_kv) },
            proj: unsafe { base.add(4 * row_x + 2 * row_kv) },
            gate: unsafe { base.add(5 * row_x + 2 * row_kv) },
            up: unsafe { base.add(5 * row_x + 2 * row_kv + row_ff) },
            bytes,
            n_tok,
            dim,
            kv_dim,
            ff_dim,
        })
    }

    /// Scratch K/V pair after a byte prefix (prefill acts live in `[0, prefix)`).
    pub fn kv_scratch_pair_after(
        &self,
        prefix_bytes: usize,
        n_kv: usize,
        kv_width: usize,
    ) -> Result<(*mut f32, *mut f32), AllpacaError> {
        let k_n = n_kv.saturating_mul(kv_width);
        let need = k_n.saturating_mul(8);
        if k_n == 0 {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                "CUDA KV scratch needs n_kv>0".into(),
            )));
        }
        if prefix_bytes % 4 != 0 {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                "CUDA KV scratch prefix must be 4-byte aligned".into(),
            )));
        }
        let total = prefix_bytes.saturating_add(need);
        if total > self.scratch_bytes {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                format!(
                    "packed CUDA KV ({need} bytes after {prefix_bytes}) exceeds scratch {}",
                    self.scratch_bytes
                ),
            )));
        }
        let k = unsafe { self.scratch.add(prefix_bytes).cast::<f32>() };
        let v = unsafe { self.scratch.add(prefix_bytes + k_n * 4).cast::<f32>() };
        Ok((k, v))
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

    /// Record a GPU start event and host wall clock for this decode step.
    pub fn begin_decode_profile(&self) {
        *self.profile_wall.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        unsafe {
            let _ = ffi::allpaca_cuda_event_record(self.t0, self.stream);
        }
    }

    /// Finish the sample: GPU elapsed from events, host remainder from wall.
    pub fn end_decode_profile(&self) {
        let wall = self
            .profile_wall
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .map(|t| t.elapsed().as_secs_f64() * 1_000_000.0)
            .unwrap_or(0.0);
        let mut gpu_us = 0.0f64;
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            unsafe {
                let _ = ffi::allpaca_cuda_event_record(self.t1, self.stream);
            }
            let _ = self.synchronize();
            let rc = unsafe { ffi::allpaca_cuda_event_elapsed_us(self.t0, self.t1, &mut gpu_us) };
            if rc != ALLPACA_OK {
                gpu_us = 0.0;
            }
        }
        let host_us = (wall - gpu_us).max(0.0);
        // EMA: 7/8 previous + 1/8 sample, stored as microsecond integers.
        let blend = |prev: &AtomicU64, sample: f64| {
            let old = prev.load(Ordering::Relaxed) as f64;
            let next = if old == 0.0 {
                sample
            } else {
                old * 0.875 + sample * 0.125
            };
            prev.store(next.round() as u64, Ordering::Relaxed);
        };
        blend(&self.ema_gpu_us, gpu_us);
        blend(&self.ema_host_us, host_us);
    }

    pub fn decode_profile(&self) -> DecodeProfile {
        DecodeProfile {
            gpu_us: self.ema_gpu_us.load(Ordering::Relaxed) as f64,
            host_us: self.ema_host_us.load(Ordering::Relaxed) as f64,
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

    /// Float A × K-quant B (ggml Q4_K / Q5_K / Q6_K as `qt` 4 / 5 / 6, GGUF
    /// layout) on this workspace's stream: A is quantized to int8 per 32
    /// values and each row is a dp4a dot, the HIP int8 K-quant scheme.
    /// `k` must be whole 256-weight super-blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_nt_kquant_dev(
        &self,
        qt: i32,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_gemm_nt_kquant(qt, a, b, c, m, n, k, self.stream) };
            status(rc, "gemm_nt_kquant")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (qt, a, b, c, m, n, k);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gemm_nt_kquant unavailable".into(),
            )))
        }
    }

    /// `silu(g) * u` with the MoE kernels' deterministic exp (test hook: it
    /// must equal `allpaca_cpu::det_silu_mul` bit for bit).
    pub fn det_silu_mul(&self, g: &[f32], u: &[f32]) -> Result<Vec<f32>, AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            let n = g.len().min(u.len());
            let (dg, du) = (self.upload_f32(&g[..n])?, self.upload_f32(&u[..n])?);
            let dout = device_malloc(n * 4, self.device)?;
            let result = (|| {
                set_device(self.device)?;
                status(
                    unsafe { ffi::allpaca_cuda_det_silu_mul(dg, du, dout.cast(), n as i32, self.stream) },
                    "det_silu_mul",
                )?;
                let mut out = vec![0.0f32; n];
                memcpy_on(out.as_mut_ptr().cast(), dout, n * 4, MEMCPY_D2H, self.stream)?;
                self.synchronize()?;
                Ok(out)
            })();
            self.free_ptr(dg);
            self.free_ptr(du);
            device_free(dout);
            result
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (g, u);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled("CUDA unavailable".into())))
        }
    }

    /// Fused expert-parallel MoE FFN over stacked K-quant experts on this
    /// workspace's stream (same contract and pointer table as HIP
    /// `moe_ffn_kquant_device`). `x` is F32 rows, or `xq`/`xd` int8 rows.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_ffn_kquant_dev(
        &self,
        x: *const f32,
        xq: *const i8,
        xd: *const f32,
        rows: usize,
        dim: usize,
        ff: usize,
        pairs: usize,
        pair_row: *const i32,
        expert: *const i32,
        pair_w: *const f32,
        row_start: *const i32,
        groups: usize,
        group_start: *const i32,
        row_pairs: *const i32,
        max_group: usize,
        experts: [(*const u8, i32, u64); 3],
        scratch: *mut u8,
        out: *mut f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let [(g, gq, gs), (u, uq, us), (d, dq, ds)] = experts;
            let rc = unsafe {
                ffi::allpaca_cuda_moe_ffn_kquant(
                    x,
                    xq,
                    xd,
                    rows as i32,
                    dim as i32,
                    ff as i32,
                    pairs as i32,
                    pair_row,
                    expert,
                    pair_w,
                    row_start,
                    groups as i32,
                    group_start,
                    row_pairs,
                    max_group as i32,
                    g,
                    gq,
                    gs,
                    u,
                    uq,
                    us,
                    d,
                    dq,
                    ds,
                    scratch.cast(),
                    out,
                    self.stream,
                )
            };
            status(rc, "moe_ffn_kquant")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, xq, xd, rows, dim, ff, pairs, pair_row, expert, pair_w, row_start);
            let _ = (groups, group_start, row_pairs, max_group, experts, scratch, out);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA moe_ffn_kquant unavailable".into(),
            )))
        }
    }

    /// Float A × staged Q8_0 B ([`stage_q8_0_blocks`]) on this workspace's
    /// stream, ordered with its other kernels.
    pub fn gemm_nt_q8_0_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_gemm_nt_q8_0(a, b, c, m, n, k, self.stream) };
            status(rc, "gemm_nt_q8_0")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (a, b, c, m, n, k);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gemm_nt_q8_0 unavailable".into(),
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

    pub fn rmsnorm_dev_rows(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        rows: i32,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if rows == 0 {
            return Ok(());
        }
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if x.is_null() || w.is_null() || out.is_null() || rows < 0 || n <= 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA rmsnorm_rows".into(),
                )));
            }
            set_device(self.device)?;
            let rc =
                unsafe { ffi::allpaca_cuda_rmsnorm_rows(x, w, out, rows, n, eps, self.stream) };
            status(rc, "rmsnorm_rows")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, w, out, rows, n, eps);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA rmsnorm unavailable".into(),
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
        if rows <= 0 || n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA rmsnorm_rows shape".into(),
            )));
        }
        let rows_n = (rows as usize).saturating_mul(n as usize);
        if x.len() < rows_n || out.len() < rows_n || weight.len() < n as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm_rows buffer size mismatch".into(),
            )));
        }
        let dx = self.upload_f32(&x[..rows_n])?;
        let dw = self.upload_f32(&weight[..n as usize])?;
        let dout = device_malloc(rows_n * 4, self.device)?;
        let result = (|| {
            self.rmsnorm_dev_rows(dx, dw, dout.cast(), rows, n, eps)?;
            self.download_f32(dout.cast(), &mut out[..rows_n])
        })();
        self.free_ptr(dx);
        self.free_ptr(dw);
        self.free_device(dout);
        result
    }

    pub fn rmsnorm_heads_dev(
        &self,
        x: *mut f32,
        weight: *const f32,
        n_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA rmsnorm_heads".into(),
            )));
        }
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_rmsnorm_heads(x, weight, n_heads, head_dim, eps, self.stream)
            };
            status(rc, "rmsnorm_heads")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, weight, n_heads, head_dim, eps);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA rmsnorm_heads unavailable".into(),
            )))
        }
    }

    /// Host roundtrip: in-place per-head RMSNorm via device (HIP `rmsnorm_heads_device` parity).
    pub fn rmsnorm_heads(
        &self,
        x: &mut [f32],
        weight: &[f32],
        n_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA rmsnorm_heads shape".into(),
            )));
        }
        let n = (n_heads as usize).saturating_mul(head_dim as usize);
        if x.len() < n || weight.len() < head_dim as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "rmsnorm_heads buffer size mismatch".into(),
            )));
        }
        let dx = self.upload_f32(&x[..n])?;
        let dw = self.upload_f32(&weight[..head_dim as usize])?;
        let result = (|| {
            self.rmsnorm_heads_dev(dx, dw, n_heads, head_dim, eps)?;
            self.download_f32(dx, &mut x[..n])
        })();
        self.free_ptr(dx);
        self.free_ptr(dw);
        result
    }

    pub fn deinterleave_q_gate_dev(
        &self,
        q_full: *const f32,
        q: *mut f32,
        gate: *mut f32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), AllpacaError> {
        if n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA deinterleave_q_gate".into(),
            )));
        }
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_deinterleave_q_gate(
                    q_full,
                    q,
                    gate,
                    n_heads,
                    head_dim,
                    self.stream,
                )
            };
            status(rc, "deinterleave_q_gate")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (q_full, q, gate, n_heads, head_dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA deinterleave_q_gate unavailable".into(),
            )))
        }
    }

    /// Host roundtrip: deinterleave gated Q `[Q|gate]` → separate buffers.
    pub fn deinterleave_q_gate(
        &self,
        q_full: &[f32],
        q: &mut [f32],
        gate: &mut [f32],
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), AllpacaError> {
        if n_heads <= 0 || head_dim <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA deinterleave_q_gate shape".into(),
            )));
        }
        let attn = (n_heads as usize).saturating_mul(head_dim as usize);
        let full = attn.saturating_mul(2);
        if q_full.len() < full || q.len() < attn || gate.len() < attn {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "deinterleave_q_gate buffer size mismatch".into(),
            )));
        }
        let d_full = self.upload_f32(&q_full[..full])?;
        let d_q = device_malloc(attn * 4, self.device)?;
        let d_gate = device_malloc(attn * 4, self.device)?;
        let result = (|| {
            self.deinterleave_q_gate_dev(d_full, d_q.cast(), d_gate.cast(), n_heads, head_dim)?;
            self.download_f32(d_q.cast(), &mut q[..attn])?;
            self.download_f32(d_gate.cast(), &mut gate[..attn])
        })();
        self.free_ptr(d_full);
        self.free_device(d_q);
        self.free_device(d_gate);
        result
    }

    pub fn sigmoid_mul_dev(
        &self,
        x: *mut f32,
        gate: *const f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA sigmoid_mul".into(),
            )));
        }
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_sigmoid_mul(x, gate, n, self.stream) };
            status(rc, "sigmoid_mul")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, gate, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA sigmoid_mul unavailable".into(),
            )))
        }
    }

    /// Host roundtrip: in-place `x[i] *= sigmoid(gate[i])`.
    pub fn sigmoid_mul(&self, x: &mut [f32], gate: &[f32]) -> Result<(), AllpacaError> {
        let n = x.len().min(gate.len());
        if n == 0 {
            return Ok(());
        }
        let dx = self.upload_f32(&x[..n])?;
        let dg = self.upload_f32(&gate[..n])?;
        let result = (|| {
            self.sigmoid_mul_dev(dx, dg, n as i32)?;
            self.download_f32(dx, &mut x[..n])
        })();
        self.free_ptr(dx);
        self.free_ptr(dg);
        result
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

    pub fn gelu_mul_dev(
        &self,
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if gate.is_null() || up.is_null() || out.is_null() || n <= 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA gelu_mul".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_gelu_mul(gate, up, out, n, self.stream) };
            status(rc, "gelu_mul")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (gate, up, out, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gelu_mul unavailable".into(),
            )))
        }
    }

    pub fn relu_sqr_dev(&self, x: *mut f32, n: i32) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if x.is_null() || n <= 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA relu_sqr".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_relu_sqr(x, n, self.stream) };
            status(rc, "relu_sqr")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA relu_sqr unavailable".into(),
            )))
        }
    }

    pub fn silu_dev(&self, x: *mut f32, n: i32) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if x.is_null() || n <= 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA silu".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_silu(x, n, self.stream) };
            status(rc, "silu")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA silu unavailable".into(),
            )))
        }
    }

    pub fn scale_f32_dev(&self, x: *mut f32, n: i32, scale: f32) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if x.is_null() || n <= 0 {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA scale_f32".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe { ffi::allpaca_cuda_scale_f32(x, n, scale, self.stream) };
            status(rc, "scale_f32")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, n, scale);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA scale_f32 unavailable".into(),
            )))
        }
    }

    pub fn expand_k_heads_dev(
        &self,
        src: *const f32,
        dst: *mut f32,
        n_k: i32,
        n_v: i32,
        head_k: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if src.is_null()
                || dst.is_null()
                || n_k <= 0
                || n_v <= 0
                || head_k <= 0
                || n_v % n_k != 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA expand_k_heads".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_expand_k_heads(src, dst, n_k, n_v, head_k, self.stream)
            };
            status(rc, "expand_k_heads")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (src, dst, n_k, n_v, head_k);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA expand_k_heads unavailable".into(),
            )))
        }
    }

    pub fn gdn_decay_beta_dev(
        &self,
        alpha: *const f32,
        dt_bias: *const f32,
        a: *const f32,
        decay: *mut f32,
        beta: *mut f32,
        n: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if alpha.is_null()
                || dt_bias.is_null()
                || a.is_null()
                || decay.is_null()
                || beta.is_null()
                || n <= 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA gdn_decay_beta".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_gdn_decay_beta(
                    alpha,
                    dt_bias,
                    a,
                    decay,
                    beta,
                    n,
                    self.stream,
                )
            };
            status(rc, "gdn_decay_beta")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (alpha, dt_bias, a, decay, beta, n);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gdn_decay_beta unavailable".into(),
            )))
        }
    }

    pub fn gdn_conv1d_step_dev(
        &self,
        x: *const f32,
        kernel: *const f32,
        conv_state: *mut f32,
        out: *mut f32,
        qkv_dim: i32,
        conv_kernel: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if x.is_null()
                || kernel.is_null()
                || conv_state.is_null()
                || out.is_null()
                || qkv_dim <= 0
                || conv_kernel <= 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA gdn_conv1d_step".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_gdn_conv1d_step(
                    x,
                    kernel,
                    conv_state,
                    out,
                    qkv_dim,
                    conv_kernel,
                    self.stream,
                )
            };
            status(rc, "gdn_conv1d_step")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (x, kernel, conv_state, out, qkv_dim, conv_kernel);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gdn_conv1d_step unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_step_dev(
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
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
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
                    "invalid CUDA gated_delta_step".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_gated_delta_step(
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
                    self.stream,
                )
            };
            status(rc, "gated_delta_step")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                q, k, v, decay, beta, state, out, n_v_heads, head_v, head_k, eps,
            );
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA gated_delta_step unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_conv_step_dev(
        &self,
        xbc: *const f32,
        kernel: *const f32,
        bias: *const f32,
        conv_state: *mut f32,
        out: *mut f32,
        conv_dim: i32,
        conv_kernel: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if xbc.is_null()
                || kernel.is_null()
                || conv_state.is_null()
                || out.is_null()
                || conv_dim <= 0
                || conv_kernel <= 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA mamba2_conv_step".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_mamba2_conv_step(
                    xbc,
                    kernel,
                    bias,
                    conv_state,
                    out,
                    conv_dim,
                    conv_kernel,
                    self.stream,
                )
            };
            status(rc, "mamba2_conv_step")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (xbc, kernel, bias, conv_state, out, conv_dim, conv_kernel);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA mamba2_conv_step unavailable".into(),
            )))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_ssd_step_dev(
        &self,
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
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if z.is_null()
                || xbc.is_null()
                || dt.is_null()
                || a.is_null()
                || d.is_null()
                || dt_bias.is_null()
                || norm.is_null()
                || state.is_null()
                || out.is_null()
                || d_inner <= 0
                || d_state <= 0
                || n_heads <= 0
                || n_groups <= 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA mamba2_ssd_step".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_mamba2_ssd_step(
                    z,
                    xbc,
                    dt,
                    a,
                    d,
                    dt_bias,
                    norm,
                    state,
                    out,
                    d_inner,
                    d_state,
                    n_heads,
                    n_groups,
                    eps,
                    self.stream,
                )
            };
            status(rc, "mamba2_ssd_step")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                z, xbc, dt, a, d, dt_bias, norm, state, out, d_inner, d_state, n_heads, n_groups,
                eps,
            );
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA mamba2_ssd_step unavailable".into(),
            )))
        }
    }

    pub fn scale_heads_sigmoid_dev(
        &self,
        attn: *mut f32,
        scores: *const f32,
        n_tok: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if attn.is_null()
                || scores.is_null()
                || n_tok <= 0
                || n_heads <= 0
                || head_dim <= 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA scale_heads_sigmoid".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_scale_heads_sigmoid(
                    attn,
                    scores,
                    n_tok,
                    n_heads,
                    head_dim,
                    self.stream,
                )
            };
            status(rc, "scale_heads_sigmoid")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (attn, scores, n_tok, n_heads, head_dim);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA scale_heads_sigmoid unavailable".into(),
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

    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_dev_rows(
        &self,
        q: *mut f32,
        k: *mut f32,
        n_tok: i32,
        n_q_heads: i32,
        n_k_heads: i32,
        head_dim: i32,
        q_stride: i32,
        k_stride: i32,
        start_pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if q.is_null()
                || k.is_null()
                || n_tok <= 0
                || n_q_heads <= 0
                || n_k_heads <= 0
                || head_dim <= 0
                || q_stride <= 0
                || k_stride <= 0
                || head_dim % 2 != 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA rope_qk_rows".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_rope_qk_rows(
                    q,
                    k,
                    n_tok,
                    n_q_heads,
                    n_k_heads,
                    head_dim,
                    q_stride,
                    k_stride,
                    start_pos,
                    theta,
                    self.stream,
                )
            };
            status(rc, "rope_qk_rows")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                q,
                k,
                n_tok,
                n_q_heads,
                n_k_heads,
                head_dim,
                q_stride,
                k_stride,
                start_pos,
                theta,
            );
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA rope_qk_rows unavailable".into(),
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
    pub fn rope_imrope_qk_dev(
        &self,
        q: *mut f32,
        k: *mut f32,
        n_q_heads: i32,
        n_k_heads: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        theta: f32,
        sections: [i32; 4],
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if q.is_null()
                || k.is_null()
                || n_q_heads <= 0
                || n_k_heads <= 0
                || head_dim <= 0
                || n_rot <= 0
                || n_rot > head_dim
                || n_rot % 2 != 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA rope_imrope_qk".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_rope_imrope_qk(
                    q,
                    k,
                    n_q_heads,
                    n_k_heads,
                    head_dim,
                    n_rot,
                    pos,
                    theta,
                    sections.as_ptr(),
                    self.stream,
                )
            };
            status(rc, "rope_imrope_qk")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (q, k, n_q_heads, n_k_heads, head_dim, n_rot, pos, theta, sections);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA rope_imrope_qk unavailable".into(),
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

    #[allow(clippy::too_many_arguments)]
    pub fn attention_dev_sinks(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        sinks: *const f32,
        out: *mut f32,
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        if sinks.is_null() {
            return self.attention_dev(
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            );
        }
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
                    "invalid CUDA attention_sinks".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_attention_sinks(
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
                    sinks,
                    self.stream,
                )
            };
            status(rc, "attention_sinks")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                q, k, v, sinks, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            );
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA attention_sinks unavailable".into(),
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

    /// Host roundtrip: fused Q/K IMRoPE / partial NeoX (HIP rope_imrope_qk parity).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_imrope_qk(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        n_q_heads: i32,
        n_k_heads: i32,
        head_dim: i32,
        n_rot: i32,
        pos: i32,
        theta: f32,
        sections: [i32; 4],
    ) -> Result<(), AllpacaError> {
        let q_n = (n_q_heads as usize).saturating_mul(head_dim as usize);
        let k_n = (n_k_heads as usize).saturating_mul(head_dim as usize);
        if q_n == 0 || k_n == 0 || q.len() < q_n || k.len() < k_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid rope_imrope_qk shape".into(),
            )));
        }
        let d_q = self.upload_f32(&q[..q_n])?;
        let d_k = self.upload_f32(&k[..k_n])?;
        let result = (|| {
            self.rope_imrope_qk_dev(
                d_q,
                d_k,
                n_q_heads,
                n_k_heads,
                head_dim,
                n_rot,
                pos,
                theta,
                sections,
            )?;
            memcpy_on(
                q.as_mut_ptr().cast(),
                d_q.cast(),
                q_n * 4,
                MEMCPY_D2H,
                self.stream,
            )?;
            memcpy_on(
                k.as_mut_ptr().cast(),
                d_k.cast(),
                k_n * 4,
                MEMCPY_D2H,
                self.stream,
            )?;
            self.synchronize()?;
            self.activation_d2h
                .fetch_add(((q_n + k_n) * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_q);
        self.free_ptr(d_k);
        result
    }

    /// Host roundtrip: in-place `x += y` via device residual_add.
    pub fn residual_add_host(&self, x: &mut [f32], y: &[f32]) -> Result<(), AllpacaError> {
        let n = x.len().min(y.len());
        if n == 0 {
            return Ok(());
        }
        let dx = self.upload_f32(&x[..n])?;
        let dy = self.upload_f32(&y[..n])?;
        let result = (|| {
            self.residual_add_dev(dx, dy, dx, n as i32)?;
            self.download_f32(dx, &mut x[..n])
        })();
        self.free_ptr(dx);
        self.free_ptr(dy);
        result
    }

    /// Host roundtrip: in-place `gate = silu(gate) * up` via device.
    pub fn silu_mul_host(&self, gate: &mut [f32], up: &[f32]) -> Result<(), AllpacaError> {
        let n = gate.len().min(up.len());
        if n == 0 {
            return Ok(());
        }
        let dg = self.upload_f32(&gate[..n])?;
        let du = self.upload_f32(&up[..n])?;
        let result = (|| {
            self.silu_mul_dev(dg, du, dg, n as i32)?;
            self.download_f32(dg, &mut gate[..n])
        })();
        self.free_ptr(dg);
        self.free_ptr(du);
        result
    }

    /// Host roundtrip: in-place `gate = gelu(gate) * up` via device (erf GELU).
    pub fn gelu_mul_host(&self, gate: &mut [f32], up: &[f32]) -> Result<(), AllpacaError> {
        let n = gate.len().min(up.len());
        if n == 0 {
            return Ok(());
        }
        let dg = self.upload_f32(&gate[..n])?;
        let du = self.upload_f32(&up[..n])?;
        let result = (|| {
            self.gelu_mul_dev(dg, du, dg, n as i32)?;
            self.download_f32(dg, &mut gate[..n])
        })();
        self.free_ptr(dg);
        self.free_ptr(du);
        result
    }

    /// Host roundtrip: in-place ReLU² via device.
    pub fn relu_sqr_host(&self, x: &mut [f32]) -> Result<(), AllpacaError> {
        if x.is_empty() {
            return Ok(());
        }
        let dx = self.upload_f32(x)?;
        let result = (|| {
            self.relu_sqr_dev(dx, x.len() as i32)?;
            self.download_f32(dx, x)
        })();
        self.free_ptr(dx);
        result
    }

    /// Host roundtrip: in-place SiLU via device.
    pub fn silu_host(&self, x: &mut [f32]) -> Result<(), AllpacaError> {
        if x.is_empty() {
            return Ok(());
        }
        let dx = self.upload_f32(x)?;
        let result = (|| {
            self.silu_dev(dx, x.len() as i32)?;
            self.download_f32(dx, x)
        })();
        self.free_ptr(dx);
        result
    }

    /// Host roundtrip: in-place `x[i] *= scale`.
    pub fn scale_f32_host(&self, x: &mut [f32], scale: f32) -> Result<(), AllpacaError> {
        if x.is_empty() {
            return Ok(());
        }
        let dx = self.upload_f32(x)?;
        let result = (|| {
            self.scale_f32_dev(dx, x.len() as i32, scale)?;
            self.download_f32(dx, x)
        })();
        self.free_ptr(dx);
        result
    }

    /// Host roundtrip: tile K-heads → V-heads (GDN).
    pub fn expand_k_heads_host(
        &self,
        src: &[f32],
        dst: &mut [f32],
        n_k: i32,
        n_v: i32,
        head_k: i32,
    ) -> Result<(), AllpacaError> {
        let src_n = (n_k as usize).saturating_mul(head_k as usize);
        let dst_n = (n_v as usize).saturating_mul(head_k as usize);
        if src_n == 0 || dst_n == 0 || src.len() < src_n || dst.len() < dst_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid expand_k_heads shape".into(),
            )));
        }
        let ds = self.upload_f32(&src[..src_n])?;
        let dd = device_malloc(dst_n * 4, self.device)?;
        let result = (|| {
            self.expand_k_heads_dev(ds, dd.cast(), n_k, n_v, head_k)?;
            self.download_f32(dd.cast(), &mut dst[..dst_n])
        })();
        self.free_ptr(ds);
        self.free_device(dd);
        result
    }

    /// Host roundtrip: GDN decay + β sigmoid (HIP gdn_decay_beta parity).
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
                "gdn_decay_beta length mismatch".into(),
            )));
        }
        let dalpha = self.upload_f32(alpha)?;
        let ddt = self.upload_f32(dt_bias)?;
        let da = self.upload_f32(a)?;
        let ddecay = device_malloc(n * 4, self.device)?;
        let dbeta = self.upload_f32(beta)?;
        let result = (|| {
            self.gdn_decay_beta_dev(
                dalpha,
                ddt,
                da,
                ddecay.cast(),
                dbeta,
                n as i32,
            )?;
            self.download_f32(ddecay.cast(), decay)?;
            self.download_f32(dbeta, beta)
        })();
        self.free_ptr(dalpha);
        self.free_ptr(ddt);
        self.free_ptr(da);
        self.free_device(ddecay);
        self.free_ptr(dbeta);
        result
    }

    /// Host roundtrip: GDN depthwise conv1d + SiLU (updates `conv_state` in place).
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
        let hist = k.saturating_sub(1);
        if k == 0
            || out.len() != qkv
            || kernel.len() != k.saturating_mul(qkv)
            || conv_state.len() != hist.saturating_mul(qkv)
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gdn_conv1d_step length mismatch".into(),
            )));
        }
        let dx = self.upload_f32(x)?;
        let dk = self.upload_f32(kernel)?;
        let (ds, free_ds) = if hist == 0 {
            (device_malloc(4, self.device)?, true)
        } else {
            (self.upload_f32(conv_state)?.cast(), false)
        };
        let dout = device_malloc(qkv * 4, self.device)?;
        let result = (|| {
            self.gdn_conv1d_step_dev(
                dx,
                dk,
                ds.cast(),
                dout.cast(),
                qkv as i32,
                conv_kernel,
            )?;
            self.download_f32(dout.cast(), out)?;
            if hist > 0 {
                self.download_f32(ds.cast(), conv_state)?;
            }
            Ok(())
        })();
        self.free_ptr(dx);
        self.free_ptr(dk);
        if free_ds {
            self.free_device(ds);
        } else {
            self.free_ptr(ds.cast());
        }
        self.free_device(dout);
        result
    }

    /// Host roundtrip: one-token Gated DeltaNet (mutates q, k, and state).
    #[allow(clippy::too_many_arguments)]
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
        let nv = n_v_heads as usize;
        let hv = head_v as usize;
        let hk = head_k as usize;
        let qk_n = nv.saturating_mul(hk);
        let v_n = nv.saturating_mul(hv);
        let st_n = nv.saturating_mul(hv).saturating_mul(hk);
        if nv == 0
            || hv == 0
            || hk == 0
            || q.len() < qk_n
            || k.len() < qk_n
            || v.len() < v_n
            || decay.len() < nv
            || beta.len() < nv
            || state.len() < st_n
            || out.len() < v_n
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gated_delta_step length mismatch".into(),
            )));
        }
        let dq = self.upload_f32(&q[..qk_n])?;
        let dk = self.upload_f32(&k[..qk_n])?;
        let dv = self.upload_f32(&v[..v_n])?;
        let dd = self.upload_f32(&decay[..nv])?;
        let db = self.upload_f32(&beta[..nv])?;
        let ds = self.upload_f32(&state[..st_n])?;
        let dout = device_malloc(v_n * 4, self.device)?;
        let result = (|| {
            self.gated_delta_step_dev(
                dq, dk, dv, dd, db, ds, dout.cast(), n_v_heads, head_v, head_k, eps,
            )?;
            self.download_f32(dout.cast(), &mut out[..v_n])?;
            self.download_f32(dq, &mut q[..qk_n])?;
            self.download_f32(dk, &mut k[..qk_n])?;
            self.download_f32(ds, &mut state[..st_n])
        })();
        self.free_ptr(dq);
        self.free_ptr(dk);
        self.free_ptr(dv);
        self.free_ptr(dd);
        self.free_ptr(db);
        self.free_ptr(ds);
        self.free_device(dout);
        result
    }

    /// Host roundtrip: Mamba-2 depthwise conv + SiLU (mutates conv_state).
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_conv_step_host(
        &self,
        xbc: &[f32],
        kernel: &[f32],
        bias: &[f32],
        conv_state: &mut [f32],
        out: &mut [f32],
        conv_dim: i32,
        conv_kernel: i32,
    ) -> Result<(), AllpacaError> {
        let cd = conv_dim as usize;
        let ck = conv_kernel as usize;
        let hist = ck.saturating_sub(1);
        if cd == 0
            || ck == 0
            || xbc.len() < cd
            || kernel.len() < cd.saturating_mul(ck)
            || out.len() < cd
            || (hist > 0 && conv_state.len() < hist.saturating_mul(cd))
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "mamba2_conv_step length mismatch".into(),
            )));
        }
        let dx = self.upload_f32(&xbc[..cd])?;
        let dk = self.upload_f32(&kernel[..cd * ck])?;
        let (ds, free_ds) = if hist == 0 {
            (device_malloc(4, self.device)?, true)
        } else {
            (self.upload_f32(&conv_state[..hist * cd])?.cast(), false)
        };
        let db = if bias.len() >= cd {
            Some(self.upload_f32(&bias[..cd])?)
        } else {
            None
        };
        let dout = device_malloc(cd * 4, self.device)?;
        let result = (|| {
            self.mamba2_conv_step_dev(
                dx,
                dk,
                db.map(|p| p as *const f32).unwrap_or(std::ptr::null()),
                ds.cast(),
                dout.cast(),
                conv_dim,
                conv_kernel,
            )?;
            self.download_f32(dout.cast(), &mut out[..cd])?;
            if hist > 0 {
                self.download_f32(ds.cast(), &mut conv_state[..hist * cd])?;
            }
            Ok(())
        })();
        self.free_ptr(dx);
        self.free_ptr(dk);
        if let Some(p) = db {
            self.free_ptr(p);
        }
        if free_ds {
            self.free_device(ds);
        } else {
            self.free_ptr(ds.cast());
        }
        self.free_device(dout);
        result
    }

    /// Host roundtrip: Mamba-2 SSD + gate + grouped RMSNorm (mutates state).
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_ssd_step_host(
        &self,
        z: &[f32],
        xbc: &[f32],
        dt: &[f32],
        a: &[f32],
        d: &[f32],
        dt_bias: &[f32],
        norm: &[f32],
        state: &mut [f32],
        out: &mut [f32],
        d_inner: i32,
        d_state: i32,
        n_heads: i32,
        n_groups: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        let inner = d_inner as usize;
        let ds = d_state as usize;
        let nh = n_heads as usize;
        let ng = n_groups as usize;
        let xbc_n = inner + 2 * ng * ds;
        let st_n = inner.saturating_mul(ds);
        if inner == 0
            || ds == 0
            || nh == 0
            || ng == 0
            || z.len() < inner
            || xbc.len() < xbc_n
            || dt.len() < nh
            || a.len() < nh
            || d.len() < nh
            || dt_bias.len() < nh
            || norm.len() < inner
            || state.len() < st_n
            || out.len() < inner
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "mamba2_ssd_step length mismatch".into(),
            )));
        }
        let dz = self.upload_f32(&z[..inner])?;
        let dxbc = self.upload_f32(&xbc[..xbc_n])?;
        let ddt = self.upload_f32(&dt[..nh])?;
        let da = self.upload_f32(&a[..nh])?;
        let dd = self.upload_f32(&d[..nh])?;
        let ddtb = self.upload_f32(&dt_bias[..nh])?;
        let dn = self.upload_f32(&norm[..inner])?;
        let dst = self.upload_f32(&state[..st_n])?;
        let dout = device_malloc(inner * 4, self.device)?;
        let result = (|| {
            self.mamba2_ssd_step_dev(
                dz,
                dxbc,
                ddt,
                da,
                dd,
                ddtb,
                dn,
                dst,
                dout.cast(),
                d_inner,
                d_state,
                n_heads,
                n_groups,
                eps,
            )?;
            self.download_f32(dout.cast(), &mut out[..inner])?;
            self.download_f32(dst, &mut state[..st_n])
        })();
        self.free_ptr(dz);
        self.free_ptr(dxbc);
        self.free_ptr(ddt);
        self.free_ptr(da);
        self.free_ptr(dd);
        self.free_ptr(ddtb);
        self.free_ptr(dn);
        self.free_ptr(dst);
        self.free_device(dout);
        result
    }

    /// Host roundtrip: `out = silu(z) * rmsnorm(y)` per V-head (HIP `gdn_rmsnorm_silu_gate`).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_rmsnorm_silu_gate_host(
        &self,
        y: &[f32],
        z: &[f32],
        weight: &[f32],
        out: &mut [f32],
        n_heads: i32,
        head_v: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        let nh = n_heads as usize;
        let hv = head_v as usize;
        let n = nh.saturating_mul(hv);
        if nh == 0
            || hv == 0
            || y.len() < n
            || z.len() < n
            || weight.len() < hv
            || out.len() < n
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "gdn_rmsnorm_silu_gate length mismatch".into(),
            )));
        }
        let dy = self.upload_f32(&y[..n])?;
        let dz = self.upload_f32(&z[..n])?;
        let dw = self.upload_f32(&weight[..hv])?;
        let dout = device_malloc(n * 4, self.device)?;
        let result = (|| {
            self.rmsnorm_heads_dev(dy, dw, n_heads, head_v, eps)?;
            self.silu_mul_dev(dz, dy, dout.cast(), n as i32)?;
            self.download_f32(dout.cast(), &mut out[..n])
        })();
        self.free_ptr(dy);
        self.free_ptr(dz);
        self.free_ptr(dw);
        self.free_device(dout);
        result
    }

    /// Host roundtrip: `attn[tok,h,d] *= sigmoid(scores[tok,h])`.
    pub fn scale_heads_sigmoid(
        &self,
        attn: &mut [f32],
        scores: &[f32],
        n_tok: i32,
        n_heads: i32,
        head_dim: i32,
    ) -> Result<(), AllpacaError> {
        let attn_n = (n_tok as usize)
            .saturating_mul(n_heads as usize)
            .saturating_mul(head_dim as usize);
        let score_n = (n_tok as usize).saturating_mul(n_heads as usize);
        if attn_n == 0 || attn.len() < attn_n || scores.len() < score_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid scale_heads_sigmoid shape".into(),
            )));
        }
        let da = self.upload_f32(&attn[..attn_n])?;
        let ds = self.upload_f32(&scores[..score_n])?;
        let result = (|| {
            self.scale_heads_sigmoid_dev(da, ds, n_tok, n_heads, head_dim)?;
            self.download_f32(da, &mut attn[..attn_n])
        })();
        self.free_ptr(da);
        self.free_ptr(ds);
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
        if sinks.len() < n_heads as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention_sinks length mismatch".into(),
            )));
        }
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
        let d_s = self.upload_f32(&sinks[..n_heads as usize])?;
        let d_out = device_malloc(q_n * 4, self.device)?;
        let result = (|| {
            self.attention_dev_sinks(
                d_q,
                d_k,
                d_v,
                d_s,
                d_out.cast(),
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
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
        self.free_ptr(d_s);
        self.free_device(d_out);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_dev_masked(
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
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if q.is_null()
                || k.is_null()
                || v.is_null()
                || out.is_null()
                || mask.is_null()
                || n_q <= 0
                || n_kv <= 0
                || n_heads <= 0
                || n_kv_heads <= 0
                || head_dim <= 0
                || n_heads % n_kv_heads != 0
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA attention_masked".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_attention_masked(
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
                    self.stream,
                )
            };
            status(rc, "attention_masked")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, mask);
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA attention_masked unavailable".into(),
            )))
        }
    }

    /// Host Q/K/V attention with an explicit [n_q, n_kv] uint8 mask (nonzero = attend).
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
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        let kv_n = (n_kv as usize) * (n_kv_heads as usize) * (head_dim as usize);
        let mask_n = (n_q as usize).saturating_mul(n_kv as usize);
        if q.len() < q_n || k.len() < kv_n || v.len() < kv_n || out.len() < q_n || mask.len() < mask_n
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention mask size mismatch".into(),
            )));
        }
        let d_q = self.upload_f32(&q[..q_n])?;
        let d_k = self.upload_f32(&k[..kv_n])?;
        let d_v = self.upload_f32(&v[..kv_n])?;
        let d_m = device_malloc(mask_n, self.device)?;
        let d_out = device_malloc(q_n * 4, self.device)?;
        let result = (|| {
            set_device(self.device)?;
            memcpy_on(d_m, mask.as_ptr().cast(), mask_n, MEMCPY_H2D, self.stream)?;
            self.attention_dev_masked(
                d_q,
                d_k,
                d_v,
                d_out.cast(),
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                d_m.cast(),
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
        self.free_device(d_m);
        self.free_device(d_out);
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_dev_softcap(
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
        softcap: f32,
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
                || !(softcap > 0.0)
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA attention_softcap".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_attention_softcap(
                    q,
                    k,
                    v,
                    out,
                    n_q,
                    n_kv,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    1,
                    softcap,
                    self.stream,
                )
            };
            status(rc, "attention_softcap")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, softcap,
            );
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA attention_softcap unavailable".into(),
            )))
        }
    }

    /// Host Q/K/V causal attention with Gemma-style logit softcap.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_softcap(
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
        softcap: f32,
    ) -> Result<(), AllpacaError> {
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        let kv_n = (n_kv as usize) * (n_kv_heads as usize) * (head_dim as usize);
        if q.len() < q_n || k.len() < kv_n || v.len() < kv_n || out.len() < q_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention softcap size mismatch".into(),
            )));
        }
        let d_q = self.upload_f32(&q[..q_n])?;
        let d_k = self.upload_f32(&k[..kv_n])?;
        let d_v = self.upload_f32(&v[..kv_n])?;
        let d_out = device_malloc(q_n * 4, self.device)?;
        let result = (|| {
            self.attention_dev_softcap(
                d_q,
                d_k,
                d_v,
                d_out.cast(),
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                softcap,
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

    #[allow(clippy::too_many_arguments)]
    pub fn attention_dev_masked_softcap(
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
        softcap: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
        {
            if q.is_null()
                || k.is_null()
                || v.is_null()
                || out.is_null()
                || mask.is_null()
                || n_q <= 0
                || n_kv <= 0
                || n_heads <= 0
                || n_kv_heads <= 0
                || head_dim <= 0
                || n_heads % n_kv_heads != 0
                || !(softcap > 0.0)
            {
                return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                    "invalid CUDA attention_masked_softcap".into(),
                )));
            }
            set_device(self.device)?;
            let rc = unsafe {
                ffi::allpaca_cuda_attention_masked_softcap(
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
                    softcap,
                    self.stream,
                )
            };
            status(rc, "attention_masked_softcap")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, mask, softcap,
            );
            Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA attention_masked_softcap unavailable".into(),
            )))
        }
    }

    /// Host Q/K/V attention with mask + Gemma-style logit softcap.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_masked_softcap(
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
        softcap: f32,
    ) -> Result<(), AllpacaError> {
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        let kv_n = (n_kv as usize) * (n_kv_heads as usize) * (head_dim as usize);
        let mask_n = (n_q as usize).saturating_mul(n_kv as usize);
        if q.len() < q_n || k.len() < kv_n || v.len() < kv_n || out.len() < q_n || mask.len() < mask_n
        {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "attention masked softcap size mismatch".into(),
            )));
        }
        let d_q = self.upload_f32(&q[..q_n])?;
        let d_k = self.upload_f32(&k[..kv_n])?;
        let d_v = self.upload_f32(&v[..kv_n])?;
        let d_m = device_malloc(mask_n, self.device)?;
        let d_out = device_malloc(q_n * 4, self.device)?;
        let result = (|| {
            set_device(self.device)?;
            memcpy_on(d_m, mask.as_ptr().cast(), mask_n, MEMCPY_H2D, self.stream)?;
            self.attention_dev_masked_softcap(
                d_q,
                d_k,
                d_v,
                d_out.cast(),
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                d_m.cast(),
                softcap,
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
        self.free_device(d_m);
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
            // Pin the caller's pages when the driver allows it; otherwise copy
            // through a dedicated pinned staging buffer.
            let host_ptr = host.as_ptr() as *mut u8;
            if let Ok(_pin) = HostRegistration::new(host_ptr, bytes) {
                memcpy_on(ptr, host.as_ptr().cast(), bytes, MEMCPY_H2D, self.stream)?;
                self.synchronize()?;
            } else {
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
                self.synchronize()?;
            }
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
            let dst_ptr = dst.as_mut_ptr() as *mut u8;
            if let Ok(_pin) = HostRegistration::new(dst_ptr, bytes) {
                memcpy_on(
                    dst.as_mut_ptr().cast(),
                    src.cast(),
                    bytes,
                    MEMCPY_D2H,
                    self.stream,
                )?;
                self.synchronize()?;
            } else {
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
            }
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

    /// GEMM with host A and device-resident staged Q8_0 B → host C.
    pub fn gemm_nt_host_a_dev_b_q8(
        &self,
        a: &[f32],
        b_dev: *const u8,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if b_dev.is_null() || m <= 0 || n <= 0 || k <= 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA gemm_nt_host_a_dev_b_q8".into(),
            )));
        }
        let a_n = (m as usize).saturating_mul(k as usize);
        let c_n = (m as usize).saturating_mul(n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "Q8 GEMM buffer size mismatch".into(),
            )));
        }
        let da = self.upload_f32(&a[..a_n])?;
        let dc = device_malloc(c_n * 4, self.device)?;
        let result = (|| {
            gemm_nt_q8_0_device(da.cast(), b_dev, dc.cast(), m, n, k)?;
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
        event_destroy_raw(self.t0);
        event_destroy_raw(self.t1);
        self.t0 = std::ptr::null_mut();
        self.t1 = std::ptr::null_mut();
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
        let mut pack = Vec::with_capacity(bytes);
        for slice in slices {
            for v in *slice {
                pack.extend_from_slice(&v.to_le_bytes());
            }
        }
        Self::upload_bytes(device, &pack)
    }

    /// Upload a pre-packed host byte blob (F32 and/or staged Q8_0).
    pub fn upload_bytes(device: i32, bytes: &[u8]) -> Result<(Self, u64), AllpacaError> {
        if bytes.is_empty() {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty CUDA weight upload".into(),
            )));
        }
        set_device(device)?;
        let ptr = device_malloc(bytes.len(), device)?;
        if bytes.len() >= 4096 {
            let mut pinned = PinnedHostBuffer::new(bytes.len())?;
            pinned.as_mut_slice().copy_from_slice(bytes);
            memcpy(ptr, pinned.as_ptr().cast(), bytes.len(), MEMCPY_H2D)?;
        } else {
            memcpy(ptr, bytes.as_ptr().cast(), bytes.len(), MEMCPY_H2D)?;
        }
        Ok((
            Self {
                device,
                ptr: ptr.cast(),
                bytes: bytes.len(),
            },
            bytes.len() as u64,
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

    pub fn byte_ptr_at(&self, byte_off: usize) -> Result<*mut u8, AllpacaError> {
        if byte_off >= self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA weight byte ptr out of range".into(),
            )));
        }
        Ok(unsafe { self.ptr.add(byte_off) })
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
    fn memcpy_batch_empty_is_ok() {
        unsafe {
            memcpy_batch_on(&[], MEMCPY_H2D, std::ptr::null_mut()).expect("empty batch");
        }
    }

    #[test]
    fn memcpy_batch_two_h2d_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0];
        let da = device_malloc(16, 0).expect("a");
        let db = device_malloc(8, 0).expect("b");
        let copies = [
            (da.cast(), a.as_ptr().cast(), 16usize),
            (db.cast(), b.as_ptr().cast(), 8usize),
        ];
        unsafe {
            memcpy_batch_on(&copies, MEMCPY_H2D, std::ptr::null_mut()).expect("batch h2d");
        }
        let mut a_back = [0.0f32; 4];
        let mut b_back = [0.0f32; 2];
        memcpy(a_back.as_mut_ptr().cast(), da, 16, MEMCPY_D2H).expect("d2h a");
        memcpy(b_back.as_mut_ptr().cast(), db, 8, MEMCPY_D2H).expect("d2h b");
        device_free(da);
        device_free(db);
        assert_eq!(a_back, a);
        assert_eq!(b_back, b);
    }

    #[test]
    fn host_register_empty_is_ok() {
        host_register(std::ptr::null_mut(), 0, 0).expect("empty register");
    }

    #[test]
    fn host_register_roundtrip_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let mut host: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
        let bytes = host.len() * 4;
        let pin = HostRegistration::new(host.as_mut_ptr().cast(), bytes).expect("register");
        let d = device_malloc(bytes, 0).expect("malloc");
        memcpy(d, host.as_ptr().cast(), bytes, MEMCPY_H2D).expect("h2d");
        host.fill(0.0);
        memcpy(host.as_mut_ptr().cast(), d, bytes, MEMCPY_D2H).expect("d2h");
        device_free(d);
        drop(pin);
        for i in 0..host.len() {
            assert!((host[i] - i as f32 * 0.5).abs() < 1e-6, "i={i}");
        }
        // Unregister on drop must allow a second pin of the same range.
        let pin2 = HostRegistration::new(host.as_mut_ptr().cast(), bytes).expect("re-register");
        drop(pin2);
    }

    #[test]
    fn upload_f32_large_uses_register_or_pinned_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let host: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let d = ws.upload_f32(&host).expect("upload");
        let mut back = vec![0.0f32; host.len()];
        ws.download_f32(d, &mut back).expect("download");
        ws.free_ptr(d);
        assert_eq!(back, host);
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
    fn decode_profile_records_gpu_time_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let n = 256i32;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let y: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.01).collect();
        let dx = ws.upload_f32(&x).expect("x");
        let dy = ws.upload_f32(&y).expect("y");
        let dout = device_malloc((n as usize) * 4, 0).expect("out");
        ws.begin_decode_profile();
        ws.residual_add_dev(dx, dy, dout.cast(), n).expect("add");
        ws.end_decode_profile();
        let p = ws.decode_profile();
        assert!(
            p.gpu_us > 0.0,
            "expected GPU event time after residual, got {p:?}"
        );
        assert!(p.host_us >= 0.0);
        ws.free_ptr(dx);
        ws.free_ptr(dy);
        ws.free_device(dout);
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
    fn mem_used_unavailable_without_device() {
        if device_count() > 0 {
            return;
        }
        assert!(device_memory_used(0).is_err());
    }

    #[test]
    fn mem_used_matches_total_minus_free_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let (free, total) = device_memory_info(0).expect("mem_info");
        let used = device_memory_used(0).expect("mem_used");
        let derived = total.saturating_sub(free);
        let skew = used.abs_diff(derived);
        assert!(
            skew < 16 * 1024 * 1024,
            "mem_used {used} and total-free {derived} disagree by {skew}"
        );
        let chunk = 16 * 1024 * 1024;
        let ptr = device_malloc(chunk, 0).expect("alloc");
        let used_after = device_memory_used(0).expect("mem_used after");
        assert!(
            used_after >= used + (chunk as u64) / 2,
            "used={used} used_after={used_after} chunk={chunk}"
        );
        device_free(ptr);
    }

    #[test]
    fn can_p2p_same_device_is_false() {
        assert!(!can_p2p(0, 0));
        let access = probe_peer(0, 0);
        assert!(!access.capable);
        assert!(!access.enabled);
        assert!(!enable_peer_access(0, 0));
    }

    #[test]
    fn can_p2p_unavailable_without_second_device() {
        if device_count() >= 2 {
            return;
        }
        assert!(!can_p2p(0, 1));
        assert!(!enable_peer_access(0, 1));
        let access = probe_peer(0, 1);
        assert!(!access.capable);
        assert!(!access.enabled);
    }

    #[test]
    fn memcpy_peer_rejects_null_payload() {
        assert!(memcpy_peer(std::ptr::null_mut(), 0, std::ptr::null(), 0, 64).is_err());
    }

    #[test]
    fn enumerate_devices_single_gpu_has_empty_peers() {
        if device_count() != 1 {
            return;
        }
        let devices = enumerate_devices();
        assert_eq!(devices.len(), 1);
        assert!(devices[0].peers.is_empty());
    }

    #[test]
    fn enumerate_devices_reports_theoretical_bandwidth_when_present() {
        if device_count() <= 0 {
            return;
        }
        let devices = enumerate_devices();
        assert!(!devices.is_empty());
        for d in &devices {
            assert!(
                d.bandwidth_gbps > 10.0 && d.bandwidth_gbps < 50_000.0,
                "id={} name={} bandwidth_gbps={} (clock*bus*2, GB/s)",
                d.id,
                d.name,
                d.bandwidth_gbps
            );
            assert!(d.memory_bytes > 0, "id={} memory_bytes=0", d.id);
        }
    }

    #[test]
    fn memcpy_peer_same_device_roundtrip_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 256usize;
        let src_host: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let d_src = device_malloc(n, 0).expect("src");
        let d_dst = device_malloc(n, 0).expect("dst");
        memcpy(d_src, src_host.as_ptr().cast(), n, MEMCPY_H2D).expect("h2d");
        memcpy_peer(d_dst, 0, d_src, 0, n).expect("peer same device");
        let mut back = vec![0u8; n];
        memcpy(back.as_mut_ptr().cast(), d_dst, n, MEMCPY_D2H).expect("d2h");
        device_free(d_src);
        device_free(d_dst);
        assert_eq!(back, src_host);
    }

    #[test]
    fn memcpy_peer_two_gpus_when_present() {
        if device_count() < 2 {
            return;
        }
        let access = probe_peer(0, 1);
        assert!(
            access.capable,
            "cudaMemcpyPeer must count as P2P capable"
        );
        let n = 64 * 1024;
        let src_host: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let d_src = device_malloc(n, 0).expect("src");
        let d_dst = device_malloc(n, 1).expect("dst");
        memcpy(d_src, src_host.as_ptr().cast(), n, MEMCPY_H2D).expect("h2d");
        memcpy_peer(d_dst, 1, d_src, 0, n).expect("p2p");
        let mut back = vec![0u8; n];
        memcpy(back.as_mut_ptr().cast(), d_dst, n, MEMCPY_D2H).expect("d2h");
        device_free(d_src);
        device_free(d_dst);
        assert_eq!(back, src_host);
        let devices = enumerate_devices();
        assert!(
            devices.iter().any(|d| !d.peers.is_empty()),
            "enumerate_devices must record at least one peer pair"
        );
    }

    #[test]
    fn memset_empty_is_ok() {
        device_memset(std::ptr::null_mut(), 0xAB, 0).expect("empty memset");
    }

    #[test]
    fn memset_rejects_null_payload() {
        assert!(device_memset(std::ptr::null_mut(), 0xAB, 64).is_err());
    }

    #[test]
    fn memset_fills_device_buffer_when_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 256usize;
        let d = device_malloc(n, 0).expect("malloc");
        device_memset(d, 0xAB, n).expect("memset");
        let mut back = vec![0u8; n];
        memcpy(back.as_mut_ptr().cast(), d, n, MEMCPY_D2H).expect("d2h");
        device_free(d);
        assert_eq!(back, vec![0xAB; n]);
    }

    #[test]
    fn memset_on_workspace_stream_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 128usize;
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let d = device_malloc(n, 0).expect("malloc");
        memset_on(d, 0x3C, n, ws.raw_stream()).expect("memset_on");
        ws.synchronize().expect("sync");
        let mut back = vec![0u8; n];
        memcpy(back.as_mut_ptr().cast(), d, n, MEMCPY_D2H).expect("d2h");
        device_free(d);
        assert_eq!(back, vec![0x3C; n]);
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

    /// Device IMRoPE / partial NeoX must match CPU rope_imrope_neox_heads.
    #[test]
    fn rope_imrope_qk_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_q = 4i32;
        let n_k = 2i32;
        let head_dim = 64i32;
        let n_rot = 32i32;
        let pos = 7i32;
        let theta = 10_000.0f32;
        let sections = [5i32, 5, 6, 0];
        let q_n = (n_q * head_dim) as usize;
        let k_n = (n_k * head_dim) as usize;
        let mut q: Vec<f32> = (0..q_n)
            .map(|i| ((i * 13) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let mut k: Vec<f32> = (0..k_n)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.4)
            .collect();
        let mut cpu_q = q.clone();
        let mut cpu_k = k.clone();
        allpaca_cpu::rope_imrope_neox_heads(
            &mut cpu_q,
            n_q as usize,
            head_dim as usize,
            pos as usize,
            theta,
            n_rot as usize,
            sections,
        )
        .expect("cpu imrope q");
        allpaca_cpu::rope_imrope_neox_heads(
            &mut cpu_k,
            n_k as usize,
            head_dim as usize,
            pos as usize,
            theta,
            n_rot as usize,
            sections,
        )
        .expect("cpu imrope k");
        let ws = CudaWorkspace::new(0, (q_n + k_n) * 4 + 4096).expect("workspace");
        ws.rope_imrope_qk(
            &mut q,
            &mut k,
            n_q,
            n_k,
            head_dim,
            n_rot,
            pos,
            theta,
            sections,
        )
        .expect("cuda rope_imrope_qk");
        for i in 0..q_n {
            assert!(
                (q[i] - cpu_q[i]).abs() < 1e-4,
                "imrope Q mismatch at {i}: cuda={} cpu={}",
                q[i],
                cpu_q[i]
            );
        }
        for i in 0..k_n {
            assert!(
                (k[i] - cpu_k[i]).abs() < 1e-4,
                "imrope K mismatch at {i}: cuda={} cpu={}",
                k[i],
                cpu_k[i]
            );
        }
        // Partial-only (sections zero) path.
        let sections0 = [0i32; 4];
        let mut q2: Vec<f32> = (0..q_n)
            .map(|i| ((i * 11) % 50) as f32 * 0.02 - 0.3)
            .collect();
        let mut k2: Vec<f32> = (0..k_n)
            .map(|i| ((i * 19) % 50) as f32 * 0.02 - 0.2)
            .collect();
        let mut cpu_q2 = q2.clone();
        let mut cpu_k2 = k2.clone();
        allpaca_cpu::rope_imrope_neox_heads(
            &mut cpu_q2,
            n_q as usize,
            head_dim as usize,
            pos as usize,
            theta,
            n_rot as usize,
            sections0,
        )
        .unwrap();
        allpaca_cpu::rope_imrope_neox_heads(
            &mut cpu_k2,
            n_k as usize,
            head_dim as usize,
            pos as usize,
            theta,
            n_rot as usize,
            sections0,
        )
        .unwrap();
        ws.rope_imrope_qk(
            &mut q2,
            &mut k2,
            n_q,
            n_k,
            head_dim,
            n_rot,
            pos,
            theta,
            sections0,
        )
        .expect("cuda partial rope_imrope_qk");
        for i in 0..q_n {
            assert!(
                (q2[i] - cpu_q2[i]).abs() < 1e-4,
                "partial Q mismatch at {i}: cuda={} cpu={}",
                q2[i],
                cpu_q2[i]
            );
        }
        for i in 0..k_n {
            assert!(
                (k2[i] - cpu_k2[i]).abs() < 1e-4,
                "partial K mismatch at {i}: cuda={} cpu={}",
                k2[i],
                cpu_k2[i]
            );
        }
    }

    #[test]
    fn residual_add_and_silu_mul_host_match_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let mut x = [1.0f32, 2.0, -0.5, 0.25];
        let y = [0.5f32, 0.25, 1.0, -0.25];
        let mut cpu_x = x;
        ws.residual_add_host(&mut x, &y).expect("residual");
        for i in 0..4 {
            cpu_x[i] += y[i];
            assert!((x[i] - cpu_x[i]).abs() < 1e-5, "residual i={i}");
        }
        let mut gate = [1.0f32, -1.0, 0.5, 2.0];
        let up = [2.0f32, 0.5, 1.0, 0.25];
        let mut cpu_g = gate;
        ws.silu_mul_host(&mut gate, &up).expect("silu");
        for i in 0..4 {
            let s = cpu_g[i] / (1.0 + (-cpu_g[i]).exp());
            cpu_g[i] = s * up[i];
            assert!((gate[i] - cpu_g[i]).abs() < 1e-4, "silu i={i}");
        }
    }

    #[test]
    fn gelu_mul_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let mut gate = [1.0f32, -1.0, 0.5, 2.0, -0.25, 3.0];
        let up = [2.0f32, 0.5, 1.0, 0.25, 1.5, -0.5];
        let mut cpu_g = gate;
        ws.gelu_mul_host(&mut gate, &up).expect("cuda gelu_mul");
        for i in 0..gate.len() {
            cpu_g[i] = allpaca_cpu::gelu(cpu_g[i]) * up[i];
            assert!(
                (gate[i] - cpu_g[i]).abs() < 1e-3,
                "gelu_mul mismatch at {i}: cuda={} cpu={}",
                gate[i],
                cpu_g[i]
            );
        }
    }

    #[test]
    fn relu_sqr_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let mut x = [1.0f32, -2.0, 0.5, -0.25, 3.0, 0.0];
        let cpu: Vec<f32> = x
            .iter()
            .map(|&v| if v > 0.0 { v * v } else { 0.0 })
            .collect();
        ws.relu_sqr_host(&mut x).expect("cuda relu_sqr");
        for i in 0..x.len() {
            assert!(
                (x[i] - cpu[i]).abs() < 1e-6,
                "relu_sqr mismatch at {i}: cuda={} cpu={}",
                x[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn silu_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let mut x = [1.0f32, -1.0, 0.5, 2.0, -0.25, 0.0];
        let cpu: Vec<f32> = x
            .iter()
            .map(|&v| v / (1.0 + (-v).exp()))
            .collect();
        ws.silu_host(&mut x).expect("cuda silu");
        for i in 0..x.len() {
            assert!(
                (x[i] - cpu[i]).abs() < 1e-5,
                "silu mismatch at {i}: cuda={} cpu={}",
                x[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn scale_f32_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        let mut x = [1.0f32, -2.0, 0.5, 3.0];
        let scale = 2.5f32;
        let cpu: Vec<f32> = x.iter().map(|&v| v * scale).collect();
        ws.scale_f32_host(&mut x, scale).expect("cuda scale_f32");
        for i in 0..x.len() {
            assert!(
                (x[i] - cpu[i]).abs() < 1e-6,
                "scale_f32 mismatch at {i}: cuda={} cpu={}",
                x[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn expand_k_heads_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_k = 2i32;
        let n_v = 4i32;
        let head_k = 2i32;
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let mut dst = [0.0f32; 8];
        let mut cpu = [0.0f32; 8];
        // ggml_repeat tiles whole block: [h0,h1, h0,h1]
        for vh in 0..n_v as usize {
            let kh = vh % n_k as usize;
            for j in 0..head_k as usize {
                cpu[vh * head_k as usize + j] = src[kh * head_k as usize + j];
            }
        }
        let ws = CudaWorkspace::new(0, 4096).expect("workspace");
        ws.expand_k_heads_host(&src, &mut dst, n_k, n_v, head_k)
            .expect("cuda expand_k_heads");
        for i in 0..8 {
            assert!(
                (dst[i] - cpu[i]).abs() < 1e-6,
                "expand_k_heads mismatch at {i}: cuda={} cpu={}",
                dst[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn gdn_decay_beta_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let alpha = [0.5f32, -1.0, 2.0];
        let dt_bias = [0.1f32, 0.0, -0.25];
        let a = [-0.5f32, -1.0, -0.25];
        let beta0 = [0.0f32, 2.0, -2.0];
        let mut decay_h = [0.0f32; 3];
        // softplus(x) = ln(1+e^x) with x>20 clamp
        for i in 0..3 {
            let x = alpha[i] + dt_bias[i];
            let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            decay_h[i] = (sp * a[i]).exp();
        }
        let beta_h = [
            1.0 / (1.0 + (-beta0[0]).exp()),
            1.0 / (1.0 + (-beta0[1]).exp()),
            1.0 / (1.0 + (-beta0[2]).exp()),
        ];
        let mut decay = [0.0f32; 3];
        let mut beta = beta0;
        let ws = CudaWorkspace::new(0, 8192).expect("workspace");
        ws.gdn_decay_beta_host(&alpha, &dt_bias, &a, &mut decay, &mut beta)
            .expect("cuda gdn_decay_beta");
        for i in 0..3 {
            assert!(
                (decay[i] - decay_h[i]).abs() < 1e-5,
                "decay mismatch at {i}: cuda={} cpu={}",
                decay[i],
                decay_h[i]
            );
            assert!(
                (beta[i] - beta_h[i]).abs() < 1e-5,
                "beta mismatch at {i}: cuda={} cpu={}",
                beta[i],
                beta_h[i]
            );
        }
    }

    #[test]
    fn gdn_conv1d_step_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let qkv = 4usize;
        let k = 3usize;
        let hist = k - 1;
        let x = [0.5f32, -1.0, 0.25, 2.0];
        let mut kernel = vec![0.0f32; k * qkv];
        for c in 0..qkv {
            for i in 0..k {
                kernel[i + c * k] = ((c * 3 + i) as f32) * 0.1 - 0.2;
            }
        }
        let mut state = vec![0.0f32; hist * qkv];
        for i in 0..hist * qkv {
            state[i] = ((i % 7) as f32) * 0.05;
        }
        let mut cpu_state = state.clone();
        let mut cpu_out = [0.0f32; 4];
        for c in 0..qkv {
            let mut acc = 0.0f32;
            for i in 0..hist {
                acc += kernel[i + c * k] * cpu_state[i * qkv + c];
            }
            acc += kernel[(k - 1) + c * k] * x[c];
            cpu_out[c] = acc / (1.0 + (-acc).exp());
        }
        for i in 0..hist.saturating_sub(1) {
            let dst = i * qkv;
            let src = (i + 1) * qkv;
            cpu_state.copy_within(src..src + qkv, dst);
        }
        let last = (hist - 1) * qkv;
        cpu_state[last..last + qkv].copy_from_slice(&x);
        let mut cuda_out = [0.0f32; 4];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.gdn_conv1d_step_host(&x, &kernel, &mut state, &mut cuda_out, k as i32)
            .expect("cuda gdn_conv1d");
        for i in 0..qkv {
            assert!(
                (cuda_out[i] - cpu_out[i]).abs() < 1e-5,
                "conv1d out mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu_out[i]
            );
        }
        for i in 0..hist * qkv {
            assert!(
                (state[i] - cpu_state[i]).abs() < 1e-5,
                "conv1d state mismatch at {i}: cuda={} cpu={}",
                state[i],
                cpu_state[i]
            );
        }
    }

    #[test]
    fn gated_delta_step_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_v = 1i32;
        let head_v = 2i32;
        let head_k = 2i32;
        let eps = 1e-6f32;
        let mut q = [1.0f32, 0.5];
        let mut k = [0.25f32, -0.75];
        let v = [0.4f32, -0.2];
        let decay = [0.8f32];
        let beta = [0.5f32];
        let mut state = [0.1f32, 0.2, -0.1, 0.3];
        let mut cpu_q = q;
        let mut cpu_k = k;
        let mut cpu_state = state;
        let mut cpu_out = [0.0f32; 2];
        let q_scale = 1.0 / (head_k as f32).sqrt();
        let ss_q: f32 = cpu_q.iter().map(|x| x * x).sum();
        let ss_k: f32 = cpu_k.iter().map(|x| x * x).sum();
        let inv_q = 1.0 / (ss_q + eps).sqrt();
        let inv_k = 1.0 / (ss_k + eps).sqrt();
        for x in &mut cpu_q {
            *x *= inv_q * q_scale;
        }
        for x in &mut cpu_k {
            *x *= inv_k;
        }
        let g = decay[0];
        let b = beta[0];
        for i in 0..head_v as usize {
            let row = &mut cpu_state[i * 2..(i + 1) * 2];
            let mut acc = 0.0f32;
            for j in 0..2 {
                row[j] *= g;
                acc += row[j] * cpu_k[j];
            }
            let delta = b * (v[i] - acc);
            let mut y = 0.0f32;
            for j in 0..2 {
                row[j] += delta * cpu_k[j];
                y += row[j] * cpu_q[j];
            }
            cpu_out[i] = y;
        }
        let mut out = [0.0f32; 2];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.gated_delta_step_host(
            &mut q,
            &mut k,
            &v,
            &decay,
            &beta,
            &mut state,
            &mut out,
            n_v,
            head_v,
            head_k,
            eps,
        )
        .expect("cuda gated_delta");
        for i in 0..2 {
            assert!(
                (out[i] - cpu_out[i]).abs() < 1e-5,
                "gated_delta out mismatch at {i}: cuda={} cpu={}",
                out[i],
                cpu_out[i]
            );
            assert!(
                (q[i] - cpu_q[i]).abs() < 1e-5,
                "gated_delta q mismatch at {i}: cuda={} cpu={}",
                q[i],
                cpu_q[i]
            );
            assert!(
                (k[i] - cpu_k[i]).abs() < 1e-5,
                "gated_delta k mismatch at {i}: cuda={} cpu={}",
                k[i],
                cpu_k[i]
            );
        }
        for i in 0..4 {
            assert!(
                (state[i] - cpu_state[i]).abs() < 1e-5,
                "gated_delta state mismatch at {i}: cuda={} cpu={}",
                state[i],
                cpu_state[i]
            );
        }
    }

    #[test]
    fn gdn_rmsnorm_silu_gate_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_heads = 2i32;
        let head_v = 2i32;
        let eps = 1e-6f32;
        let y = [1.0f32, 2.0, -1.0, 0.5];
        let z = [0.5f32, -0.5, 1.5, 0.0];
        let weight = [1.0f32, 0.5];
        let mut cpu_out = [0.0f32; 4];
        for h in 0..n_heads as usize {
            let y_h = &y[h * 2..(h + 1) * 2];
            let z_h = &z[h * 2..(h + 1) * 2];
            let mut ss = 0.0f32;
            for &v in y_h {
                ss += v * v;
            }
            let inv = 1.0 / (ss / head_v as f32 + eps).sqrt();
            for i in 0..2 {
                let silu = z_h[i] / (1.0 + (-z_h[i]).exp());
                cpu_out[h * 2 + i] = silu * (y_h[i] * inv * weight[i]);
            }
        }
        let mut out = [0.0f32; 4];
        let ws = CudaWorkspace::new(0, 8192).expect("workspace");
        ws.gdn_rmsnorm_silu_gate_host(&y, &z, &weight, &mut out, n_heads, head_v, eps)
            .expect("cuda gdn_rmsnorm_silu_gate");
        for i in 0..4 {
            assert!(
                (out[i] - cpu_out[i]).abs() < 1e-4,
                "gdn_rmsnorm_silu_gate mismatch at {i}: cuda={} cpu={}",
                out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn mamba2_conv_step_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let conv_dim = 4usize;
        let conv_kernel = 2usize;
        let hist = conv_kernel - 1;
        let xbc = [1.0f32, -0.5, 0.5, -0.5];
        let kernel = [0.5f32, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0];
        let bias = [0.0f32; 4];
        let mut state = [0.25f32, 0.1, -0.1, 0.2];
        let mut cpu_state = state;
        let mut cpu_out = [0.0f32; 4];
        for c in 0..conv_dim {
            let mut acc = bias[c];
            for k in 0..hist {
                acc += kernel[k + c * conv_kernel] * cpu_state[k * conv_dim + c];
            }
            acc += kernel[hist + c * conv_kernel] * xbc[c];
            cpu_out[c] = acc / (1.0 + (-acc).exp());
        }
        for k in 0..hist.saturating_sub(1) {
            let src = (k + 1) * conv_dim;
            cpu_state.copy_within(src..src + conv_dim, k * conv_dim);
        }
        let last = (hist - 1) * conv_dim;
        cpu_state[last..last + conv_dim].copy_from_slice(&xbc);
        let mut out = [0.0f32; 4];
        let ws = CudaWorkspace::new(0, 8192).expect("workspace");
        ws.mamba2_conv_step_host(
            &xbc,
            &kernel,
            &bias,
            &mut state,
            &mut out,
            conv_dim as i32,
            conv_kernel as i32,
        )
        .expect("cuda mamba2_conv");
        for i in 0..4 {
            assert!(
                (out[i] - cpu_out[i]).abs() < 1e-5,
                "mamba2_conv out mismatch at {i}: cuda={} cpu={}",
                out[i],
                cpu_out[i]
            );
            assert!(
                (state[i] - cpu_state[i]).abs() < 1e-5,
                "mamba2_conv state mismatch at {i}: cuda={} cpu={}",
                state[i],
                cpu_state[i]
            );
        }
    }

    #[test]
    fn mamba2_ssd_step_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        // d_inner=2, d_state=1, n_heads=1, n_groups=1 → xbc = x|B|C len 4
        let d_inner = 2i32;
        let d_state = 1i32;
        let n_heads = 1i32;
        let n_groups = 1i32;
        let eps = 1e-5f32;
        let z = [0.5f32, -0.25];
        let xbc = [1.0f32, -0.5, 0.5, -0.25];
        let dt = [0.1f32];
        let a = [-1.0f32];
        let d = [0.25f32];
        let dt_bias = [0.0f32];
        let norm = [1.0f32, 1.0];
        let mut state = [0.1f32, -0.2];
        let softplus = |x: f32| {
            if x > 20.0 {
                x
            } else {
                (1.0 + x.exp()).ln()
            }
        };
        let mut cpu_state = state;
        let mut cpu_out = [0.0f32; 2];
        let delta = softplus(dt[0] + dt_bias[0]);
        let decay = (delta * a[0]).exp();
        let b = xbc[2];
        let c_val = xbc[3];
        for channel in 0..2 {
            let value = decay * cpu_state[channel] + delta * b * xbc[channel];
            cpu_state[channel] = value;
            let acc = c_val * value;
            let silu = z[channel] / (1.0 + (-z[channel]).exp());
            cpu_out[channel] = (acc + d[0] * xbc[channel]) * silu;
        }
        let ss: f32 = cpu_out.iter().map(|v| v * v).sum();
        let inv = 1.0 / (ss / 2.0 + eps).sqrt();
        for i in 0..2 {
            cpu_out[i] *= inv * norm[i];
        }
        let mut out = [0.0f32; 2];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.mamba2_ssd_step_host(
            &z,
            &xbc,
            &dt,
            &a,
            &d,
            &dt_bias,
            &norm,
            &mut state,
            &mut out,
            d_inner,
            d_state,
            n_heads,
            n_groups,
            eps,
        )
        .expect("cuda mamba2_ssd");
        for i in 0..2 {
            assert!(
                (out[i] - cpu_out[i]).abs() < 1e-4,
                "mamba2_ssd out mismatch at {i}: cuda={} cpu={}",
                out[i],
                cpu_out[i]
            );
            assert!(
                (state[i] - cpu_state[i]).abs() < 1e-4,
                "mamba2_ssd state mismatch at {i}: cuda={} cpu={}",
                state[i],
                cpu_state[i]
            );
        }
    }

    #[test]
    fn scale_heads_sigmoid_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_tok = 2i32;
        let n_heads = 3i32;
        let head_dim = 4i32;
        let attn_n = (n_tok * n_heads * head_dim) as usize;
        let score_n = (n_tok * n_heads) as usize;
        let mut attn: Vec<f32> = (0..attn_n).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let scores: Vec<f32> = (0..score_n)
            .map(|i| ((i as f32) - 2.0) * 0.5)
            .collect();
        let mut cpu = attn.clone();
        for tok in 0..n_tok as usize {
            for h in 0..n_heads as usize {
                let g = 1.0 / (1.0 + (-scores[tok * n_heads as usize + h]).exp());
                let base = (tok * n_heads as usize + h) * head_dim as usize;
                for d in 0..head_dim as usize {
                    cpu[base + d] *= g;
                }
            }
        }
        let ws = CudaWorkspace::new(0, 8192).expect("workspace");
        ws.scale_heads_sigmoid(&mut attn, &scores, n_tok, n_heads, head_dim)
            .expect("cuda scale_heads_sigmoid");
        for i in 0..attn_n {
            assert!(
                (attn[i] - cpu[i]).abs() < 1e-5,
                "scale_heads_sigmoid mismatch at {i}: cuda={} cpu={}",
                attn[i],
                cpu[i]
            );
        }
    }

    fn attn_close(got: f32, exp: f32) -> bool {
        let d = (got - exp).abs();
        d <= 1e-4 || d <= 1e-3 * got.abs().max(exp.abs())
    }

    /// The MoE SiLU's exp equals the CPU reference (and the HIP kernel's) bit
    /// for bit.
    #[test]
    fn det_silu_mul_matches_cpu_reference_bitwise() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        // Every regime: tiny, ordinary, both overflow clamps, signed zero.
        let mut g: Vec<f32> = (0..40_000).map(|i| ((i as f32) - 20_000.0) * 0.0061).collect();
        g.extend([0.0, -0.0, 1e-30, -1e-30, 88.4, -88.4, 120.0, -120.0, 1e4, -1e4]);
        let u: Vec<f32> = (0..g.len()).map(|i| ((i % 13) as f32 - 6.0) * 0.37).collect();
        let want: Vec<f32> = g.iter().zip(&u).map(|(&a, &b)| allpaca_cpu::det_silu_mul(a, b)).collect();
        let got = ws.det_silu_mul(&g, &u).unwrap();
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "g={} u={}: {a} vs {b}", g[i], u[i]);
        }
    }

    /// The K-quant activation scratch outlives workspaces: a GEMV on a new
    /// workspace after the previous one (and its stream) was dropped must
    /// still run. It segfaulted in libcuda when an RPC client reconnected.
    #[test]
    fn kquant_gemv_survives_a_dropped_workspace() {
        if device_count() <= 0 {
            return;
        }
        let (n, k) = (64usize, 256usize);
        let w: Vec<f32> = (0..n * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
        let q = allpaca_cpu::quantize_q4_k(&w);
        let a: Vec<f32> = (0..k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let run = |ws: &CudaWorkspace| -> Vec<f32> {
            let db = device_malloc(q.len(), 0).expect("weights");
            memcpy_on(db, q.as_ptr().cast(), q.len(), MEMCPY_H2D, ws.raw_stream()).unwrap();
            let da = ws.upload_f32(&a).expect("a");
            let dc = device_malloc(n * 4, 0).expect("c");
            ws.gemm_nt_kquant_dev(4, da, db.cast_const().cast(), dc.cast(), 1, n as i32, k as i32)
                .expect("kquant gemv");
            let mut out = vec![0.0f32; n];
            memcpy_on(out.as_mut_ptr().cast(), dc, n * 4, MEMCPY_D2H, ws.raw_stream()).unwrap();
            ws.synchronize().unwrap();
            ws.free_ptr(da);
            device_free(dc);
            device_free(db);
            out
        };
        // The second workspace exists before the first goes, so its stream
        // is a different one, and the scratch last ran on a destroyed stream.
        let first = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        let want = run(&first);
        let second = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        drop(first);
        assert_eq!(run(&second), want);
    }

    /// Int8-activation K-quant GEMV (Q4_K / Q5_K / Q6_K) against the CPU
    /// dequantized weights, for 1..8 rows and a column count that does not
    /// fill the last block; each row equals its one-row call bit for bit.
    #[test]
    fn kquant_gemv_matches_cpu_and_is_row_exact() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        let (n, k) = (77usize, 512usize);
        let w: Vec<f32> = (0..n * k).map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.004).collect();
        let a: Vec<f32> = (0..8 * k).map(|i| (((i * 13) % 59) as f32 - 29.0) * 0.03).collect();
        for qt in [4i32, 5, 6] {
            let (q, mut wd) = match qt {
                4 => (allpaca_cpu::quantize_q4_k(&w), vec![0.0f32; n * k]),
                5 => (allpaca_cpu::quantize_q5_k(&w), vec![0.0f32; n * k]),
                _ => (allpaca_cpu::quantize_q6_k(&w), vec![0.0f32; n * k]),
            };
            match qt {
                4 => allpaca_cpu::dequant_q4_k(&q, n * k, &mut wd).unwrap(),
                5 => allpaca_cpu::dequant_q5_k(&q, n * k, &mut wd).unwrap(),
                _ => allpaca_cpu::dequant_q6_k(&q, n * k, &mut wd).unwrap(),
            }
            let db = device_malloc(q.len(), 0).expect("weights");
            memcpy_on(db, q.as_ptr().cast(), q.len(), MEMCPY_H2D, ws.raw_stream()).unwrap();
            let run = |rows: &[f32], m: usize| -> Vec<f32> {
                let da = ws.upload_f32(rows).expect("a");
                let dc = device_malloc(m * n * 4, 0).expect("c");
                ws.gemm_nt_kquant_dev(qt, da, db.cast_const().cast(), dc.cast(), m as i32, n as i32, k as i32)
                    .expect("kquant gemv");
                let mut out = vec![0.0f32; m * n];
                memcpy_on(out.as_mut_ptr().cast(), dc, m * n * 4, MEMCPY_D2H, ws.raw_stream()).unwrap();
                ws.synchronize().unwrap();
                ws.free_ptr(da);
                device_free(dc);
                out
            };
            let all = run(&a, 8);
            let mut want = vec![0.0f32; 8 * n];
            allpaca_cpu::matmul_nt(&a, &wd, &mut want, 8, n, k).unwrap();
            let scale = want.iter().fold(0.0f32, |s, v| s.max(v.abs()));
            for (i, (g, e)) in all.iter().zip(&want).enumerate() {
                assert!((g - e).abs() <= 0.02 * scale, "qt={qt} out[{i}] {g} vs {e} (scale {scale})");
            }
            for m in [1usize, 2, 3] {
                let some = run(&a[..m * k], m);
                assert!(
                    some.iter().zip(&all[..m * n]).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "qt={qt} m={m} rows differ from the 8-row call"
                );
            }
            ws.synchronize().unwrap();
            device_free(db);
        }
    }

    /// Several rows (the Q-tiled kernel) give each row bit for bit what a
    /// one-row call over the keys it sees gives, as on HIP.
    #[test]
    fn attention_rows_equal_single_row_calls() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        let shapes = [
            (13usize, 40usize, 4usize, 2usize, 128usize, true),
            (9, 9, 8, 8, 64, true),
            (20, 300, 6, 2, 128, true),
            (3, 5, 2, 1, 4, true),
            (17, 50, 2, 2, 96, true),
            (6, 33, 4, 2, 128, false),
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
            let call = |q: &[f32], n_q: usize, seen: usize, out: &mut [f32]| {
                ws.attention(
                    q,
                    &k[..seen * kv_row],
                    &v[..seen * kv_row],
                    out,
                    n_q as i32,
                    seen as i32,
                    n_heads as i32,
                    n_kv_heads as i32,
                    head_dim as i32,
                    causal,
                )
                .expect("cuda attention");
            };
            let mut rows = vec![0.0f32; n_q * q_row];
            call(&q, n_q, n_kv, &mut rows);
            for i in 0..n_q {
                let seen = if causal { n_kv - n_q + i + 1 } else { n_kv };
                let mut one = vec![0.0f32; q_row];
                call(&q[i * q_row..(i + 1) * q_row], 1, seen, &mut one);
                let got = &rows[i * q_row..(i + 1) * q_row];
                assert!(
                    got.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "row {i} of n_q={n_q} n_kv={n_kv} heads={n_heads}/{n_kv_heads} d={head_dim} causal={causal} differs from its one-row call"
                );
            }
        }
    }

    #[test]
    fn attention_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let shapes = [
            (1i32, 4i32, 4i32, 2i32, 8i32, true),
            (1, 1, 1, 1, 2, true),
            (1, 8, 4, 1, 4, true),
            (1, 16, 2, 2, 8, true),
            (1, 32, 4, 2, 16, true),
            (1, 8, 1, 1, 128, true),
            (4, 4, 2, 2, 4, true),
            (3, 8, 4, 1, 8, true),
            (2, 2, 1, 1, 4, false),
        ];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        for (si, (n_q, n_kv, n_heads, n_kv_heads, head_dim, causal)) in
            shapes.iter().copied().enumerate()
        {
            let q_n = (n_q * n_heads * head_dim) as usize;
            let kv_n = (n_kv * n_kv_heads * head_dim) as usize;
            let q: Vec<f32> = (0..q_n)
                .map(|i| ((i * 17 + si * 3) % 50) as f32 * 0.02 - 0.5)
                .collect();
            let k: Vec<f32> = (0..kv_n)
                .map(|i| ((i * 13 + si * 7) % 50) as f32 * 0.02 - 0.5)
                .collect();
            let v: Vec<f32> = (0..kv_n)
                .map(|i| ((i * 11 + si * 5) % 50) as f32 * 0.02 - 0.5)
                .collect();
            let mut cuda_out = vec![0.0f32; q_n];
            let mut cpu_out = vec![0.0f32; q_n];
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
                causal,
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
                causal,
            )
            .expect("cpu attention");
            for i in 0..q_n {
                assert!(
                    attn_close(cuda_out[i], cpu_out[i]),
                    "attention mismatch n_q={n_q} n_kv={n_kv} heads={n_heads}/{n_kv_heads} d={head_dim} causal={causal} idx={i}: cuda={} cpu={}",
                    cuda_out[i],
                    cpu_out[i]
                );
            }
        }
    }

    #[test]
    fn rmsnorm_rows_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let eps = 1e-6f32;
        let shapes = [(16usize, 32usize), (9, 8), (1, 64)];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
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
            let mut cuda_out = vec![0.0f32; rows * n];
            ws.rmsnorm_rows(&x, &weight, &mut cuda_out, rows as i32, n as i32, eps)
                .expect("cuda rmsnorm_rows");
            for i in 0..rows * n {
                assert!(
                    attn_close(cuda_out[i], cpu[i]),
                    "rmsnorm_rows mismatch rows={rows} n={n} idx={i}: cuda={} cpu={}",
                    cuda_out[i],
                    cpu[i]
                );
            }
        }
    }

    #[test]
    fn rmsnorm_heads_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let eps = 1e-5f32;
        let shapes = [(4usize, 8usize), (8, 16), (2, 32)];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        for (si, (n_heads, head_dim)) in shapes.iter().copied().enumerate() {
            let mut x = vec![0.0f32; n_heads * head_dim];
            let mut weight = vec![0.0f32; head_dim];
            for (i, v) in x.iter_mut().enumerate() {
                *v = ((i * 19 + si * 5) % 47) as f32 * 0.03 - 0.6;
            }
            for (i, v) in weight.iter_mut().enumerate() {
                *v = 0.4 + ((i * 11 + si) % 17) as f32 * 0.06;
            }
            let mut cpu = x.clone();
            allpaca_core_rmsnorm_heads_ref(&mut cpu, &weight, head_dim, eps);
            ws.rmsnorm_heads(&mut x, &weight, n_heads as i32, head_dim as i32, eps)
                .expect("cuda rmsnorm_heads");
            for i in 0..n_heads * head_dim {
                assert!(
                    attn_close(x[i], cpu[i]),
                    "rmsnorm_heads mismatch heads={n_heads} d={head_dim} idx={i}: cuda={} cpu={}",
                    x[i],
                    cpu[i]
                );
            }
        }
    }

    #[test]
    fn deinterleave_q_gate_matches_host_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        let n_heads = 4usize;
        let head_dim = 8usize;
        let mut full = vec![0.0f32; n_heads * 2 * head_dim];
        for (i, v) in full.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 1.0;
        }
        let mut q_cpu = vec![0.0f32; n_heads * head_dim];
        let mut gate_cpu = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let src = h * 2 * head_dim;
            let dst = h * head_dim;
            q_cpu[dst..dst + head_dim].copy_from_slice(&full[src..src + head_dim]);
            gate_cpu[dst..dst + head_dim]
                .copy_from_slice(&full[src + head_dim..src + 2 * head_dim]);
        }
        let mut q = vec![0.0f32; n_heads * head_dim];
        let mut gate = vec![0.0f32; n_heads * head_dim];
        ws.deinterleave_q_gate(&full, &mut q, &mut gate, n_heads as i32, head_dim as i32)
            .expect("cuda deinterleave");
        for i in 0..n_heads * head_dim {
            assert!(
                attn_close(q[i], q_cpu[i]),
                "q mismatch idx={i}: cuda={} cpu={}",
                q[i],
                q_cpu[i]
            );
            assert!(
                attn_close(gate[i], gate_cpu[i]),
                "gate mismatch idx={i}: cuda={} cpu={}",
                gate[i],
                gate_cpu[i]
            );
        }
    }

    #[test]
    fn sigmoid_mul_matches_host_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        let n = 64usize;
        let mut x = vec![0.0f32; n];
        let mut gate = vec![0.0f32; n];
        for i in 0..n {
            x[i] = (i as f32) * 0.05 - 1.5;
            gate[i] = ((i * 3) % 20) as f32 * 0.2 - 2.0;
        }
        let mut cpu = x.clone();
        for i in 0..n {
            cpu[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
        ws.sigmoid_mul(&mut x, &gate).expect("cuda sigmoid_mul");
        for i in 0..n {
            assert!(
                attn_close(x[i], cpu[i]),
                "sigmoid_mul mismatch idx={i}: cuda={} cpu={}",
                x[i],
                cpu[i]
            );
        }
    }

    /// Local CPU reference matching `deltanet::rmsnorm_heads` (avoid core dep cycle).
    fn allpaca_core_rmsnorm_heads_ref(x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        assert_eq!(weight.len(), head_dim);
        assert_eq!(x.len() % head_dim, 0);
        let n_heads = x.len() / head_dim;
        for h in 0..n_heads {
            let row = &mut x[h * head_dim..(h + 1) * head_dim];
            let mut ss = 0.0f32;
            for &v in row.iter() {
                ss += v * v;
            }
            let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
            for i in 0..head_dim {
                row[i] = row[i] * inv * weight[i];
            }
        }
    }

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
        causal: bool,
    ) {
        let scale = 1.0 / (head_dim as f32).sqrt();
        let reps = n_heads / n_kv_heads;
        out.fill(0.0);
        for qi in 0..n_q {
            for h in 0..n_heads {
                let kv_h = h / reps;
                let q_off = (qi * n_heads + h) * head_dim;
                let qh = &q[q_off..q_off + head_dim];
                let sink = sinks[h];
                let mut scores = vec![0.0f32; n_kv];
                let mut max = sink;
                for t in 0..n_kv {
                    if causal && t > qi + n_kv.saturating_sub(n_q) {
                        scores[t] = f32::NEG_INFINITY;
                        continue;
                    }
                    let k_off = (t * n_kv_heads + kv_h) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += qh[d] * k[k_off + d];
                    }
                    scores[t] = dot * scale;
                    if scores[t] > max {
                        max = scores[t];
                    }
                }
                let mut acc = (sink - max).exp();
                for s in &mut scores {
                    *s = (*s - max).exp();
                    acc += *s;
                }
                let inv = 1.0 / acc.max(1e-9);
                let o_off = (qi * n_heads + h) * head_dim;
                for t in 0..n_kv {
                    let w = scores[t] * inv;
                    if w == 0.0 {
                        continue;
                    }
                    let v_off = (t * n_kv_heads + kv_h) * head_dim;
                    for d in 0..head_dim {
                        out[o_off + d] += w * v[v_off + d];
                    }
                }
            }
        }
    }

    /// Virtual per-head sink logits soak softmax mass without contributing V.
    #[test]
    fn attention_sinks_match_cpu_when_device_present() {
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
        let q: Vec<f32> = (0..q_n)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let k: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 13) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let v: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 11) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let sinks: Vec<f32> = (0..n_heads as usize)
            .map(|h| 0.25 + h as f32 * 0.5)
            .collect();
        let mut cuda_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.attention_sinks(
            &q,
            &k,
            &v,
            &sinks,
            &mut cuda_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            true,
        )
        .expect("cuda attention_sinks");
        cpu_attention_sinks(
            &q,
            &k,
            &v,
            &sinks,
            &mut cpu_out,
            n_q as usize,
            n_kv as usize,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
            true,
        );
        for i in 0..q_n {
            assert!(
                attn_close(cuda_out[i], cpu_out[i]),
                "attention_sinks mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu_out[i]
            );
        }
    }

    /// Explicit [n_q, n_kv] uint8 mask (nonzero = attend) must match CPU attention_with_mask.
    #[test]
    fn attention_masked_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_q = 2i32;
        let n_kv = 4i32;
        let n_heads = 2i32;
        let n_kv_heads = 1i32;
        let head_dim = 8i32;
        let q_n = (n_q * n_heads * head_dim) as usize;
        let kv_n = (n_kv * n_kv_heads * head_dim) as usize;
        let q: Vec<f32> = (0..q_n)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let k: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 13) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let v: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 11) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let mut mask = vec![0u8; (n_q * n_kv) as usize];
        mask[0] = 1;
        mask[2] = 1;
        mask[4] = 1;
        mask[5] = 1;
        mask[7] = 1;
        let bool_mask: Vec<bool> = mask.iter().map(|&b| b != 0).collect();
        let mut cuda_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.attention_masked(
            &q,
            &k,
            &v,
            &mut cuda_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            &mask,
        )
        .expect("cuda attention_masked");
        allpaca_cpu::attention_with_mask(
            &q,
            &k,
            &v,
            &mut cpu_out,
            n_q as usize,
            n_kv as usize,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
            &bool_mask,
        )
        .expect("cpu attention_with_mask");
        for i in 0..q_n {
            assert!(
                attn_close(cuda_out[i], cpu_out[i]),
                "attention_masked mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu_out[i]
            );
        }
    }

    /// Gemma logit softcap after scale must match CPU attention_softcap.
    #[test]
    fn attention_softcap_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_q = 2i32;
        let n_kv = 4i32;
        let n_heads = 2i32;
        let n_kv_heads = 1i32;
        let head_dim = 8i32;
        let softcap = 2.0f32;
        let q_n = (n_q * n_heads * head_dim) as usize;
        let kv_n = (n_kv * n_kv_heads * head_dim) as usize;
        let q: Vec<f32> = (0..q_n)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let k: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 13) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let v: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 11) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let mut cuda_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.attention_softcap(
            &q,
            &k,
            &v,
            &mut cuda_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            softcap,
        )
        .expect("cuda attention_softcap");
        allpaca_cpu::attention_softcap(
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
            softcap,
        )
        .expect("cpu attention_softcap");
        for i in 0..q_n {
            assert!(
                attn_close(cuda_out[i], cpu_out[i]),
                "attention_softcap mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu_out[i]
            );
        }
    }

    /// Masked + softcap (SWA+Gemma) must match CPU attention_with_mask_softcap.
    #[test]
    fn attention_masked_softcap_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n_q = 2i32;
        let n_kv = 4i32;
        let n_heads = 2i32;
        let n_kv_heads = 1i32;
        let head_dim = 8i32;
        let softcap = 1.5f32;
        let q_n = (n_q * n_heads * head_dim) as usize;
        let kv_n = (n_kv * n_kv_heads * head_dim) as usize;
        let q: Vec<f32> = (0..q_n)
            .map(|i| ((i * 17) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let k: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 13) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let v: Vec<f32> = (0..kv_n)
            .map(|i| ((i * 11) % 50) as f32 * 0.02 - 0.5)
            .collect();
        let mut mask = vec![0u8; (n_q * n_kv) as usize];
        mask[0] = 1;
        mask[2] = 1;
        mask[4] = 1;
        mask[5] = 1;
        mask[7] = 1;
        let bool_mask: Vec<bool> = mask.iter().map(|&b| b != 0).collect();
        let mut cuda_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = CudaWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.attention_masked_softcap(
            &q,
            &k,
            &v,
            &mut cuda_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            &mask,
            softcap,
        )
        .expect("cuda attention_masked_softcap");
        allpaca_cpu::attention_with_mask_softcap(
            &q,
            &k,
            &v,
            &mut cpu_out,
            n_q as usize,
            n_kv as usize,
            n_heads as usize,
            n_kv_heads as usize,
            head_dim as usize,
            &bool_mask,
            softcap,
        )
        .expect("cpu attention_with_mask_softcap");
        for i in 0..q_n {
            assert!(
                attn_close(cuda_out[i], cpu_out[i]),
                "attention_masked_softcap mismatch at {i}: cuda={} cpu={}",
                cuda_out[i],
                cpu_out[i]
            );
        }
    }

    fn gemm_close(got: f32, exp: f32) -> bool {
        let d = (got - exp).abs();
        d <= 1e-4 || d <= 1e-3 * got.abs().max(exp.abs())
    }

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

    fn dequant_q8_matrix(b_packed: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(32);
        let mut b_deq = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_packed[row * nblocks * 34..(row + 1) * nblocks * 34];
            let mut row_f = vec![0.0f32; nblocks * 32];
            allpaca_cpu::dequant_q8_0(blk, nblocks * 32, &mut row_f).expect("dequant");
            for j in 0..k {
                b_deq[row * k + j] = row_f[j];
            }
        }
        b_deq
    }

    /// Public `gemm_nt_q8_0` must match CPU dequant + NT GEMM on decode-shaped matrices.
    #[test]
    fn gemm_nt_q8_0_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let shapes = [
            (1usize, 8usize, 32usize),
            (1, 17, 32),
            (2, 5, 64),
            (1, 4, 33),
            (1, 64, 128),
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
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut cuda_out = vec![0.0f32; m * n];
            allpaca_cpu::matmul_nt(&a, &b_deq, &mut cpu, m, n, k).expect("cpu gemm");
            gemm_nt_q8_0(&a, &b_q8, &mut cuda_out, m as i32, n as i32, k as i32)
                .expect("cuda gemm_nt_q8_0");
            for i in 0..m * n {
                assert!(
                    gemm_close(cuda_out[i], cpu[i]),
                    "gemm_nt_q8_0 mismatch shape {m}x{n}x{k} idx={i}: cuda={} cpu={}",
                    cuda_out[i],
                    cpu[i]
                );
            }
        }
    }
}
