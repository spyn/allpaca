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

/// GGUF Q8_0 packed block (f16 scale + 32 qs).
pub const Q8_PACKED_BLOCK: usize = 34;
/// Device-staged Q8_0: f32 scale at 0, qs at 4, 36-byte stride (4-byte aligned).
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
        pub fn allpaca_cuda_gemm_nt_q8_0(
            a: *const f32,
            b: *const u8,
            c: *mut f32,
            m: i32,
            n: i32,
            k: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn allpaca_cuda_embed_gather_q8_0(
            table: *const u8,
            token: i32,
            out: *mut f32,
            dim: i32,
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
        status(sync, "stream_synchronize")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (a, b, c, m, n, k);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA gemm_nt_q8_0 unavailable".into(),
        )))
    }
}

/// Host-pointer Q8_0 NT GEMM: stages packed 34-byte GGUF blocks, runs device GEMV, copies back.
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
            "invalid Q8 GEMM shape".into(),
        )));
    }
    let a_n = (m as usize) * (k as usize);
    let c_n = (m as usize) * (n as usize);
    let packed_bytes = (n as usize)
        .saturating_mul((k as usize).div_ceil(32))
        .saturating_mul(Q8_PACKED_BLOCK);
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

pub fn embed_gather_q8_0_device(
    table: *const u8,
    token: i32,
    out: *mut f32,
    dim: i32,
) -> Result<(), AllpacaError> {
    #[cfg(all(feature = "cuda", allpaca_cuda_runtime))]
    {
        let rc = unsafe {
            ffi::allpaca_cuda_embed_gather_q8_0(table, token, out, dim, std::ptr::null_mut())
        };
        status(rc, "embed_gather_q8_0")?;
        let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
        status(sync, "stream_synchronize")
    }
    #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
    {
        let _ = (table, token, out, dim);
        Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
            "CUDA embed_gather_q8_0 unavailable".into(),
        )))
    }
}

/// Fixed activation region at the front of CUDA scratch for in-layer QKV fusion.
/// Packed KV uses the remainder (`kv_scratch_pair_after`).
#[derive(Clone, Copy)]
pub struct CudaDecodeActs {
    pub x: *mut f32,
    pub norm: *mut f32,
    pub q: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub attn: *mut f32,
    pub proj: *mut f32,
    pub bytes: usize,
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

    /// Bytes for `decode_acts`: x, norm, q, attn, proj (`dim`) plus k, v (`kv_dim`).
    pub fn decode_act_bytes(dim: usize, kv_dim: usize) -> usize {
        (5usize
            .saturating_mul(dim)
            .saturating_add(2usize.saturating_mul(kv_dim)))
        .saturating_mul(4)
    }

    /// Device Q/K/V/attn slots at the front of scratch (no host QKV roundtrip).
    pub fn decode_acts(&self, dim: usize, kv_dim: usize) -> Result<CudaDecodeActs, AllpacaError> {
        let bytes = Self::decode_act_bytes(dim, kv_dim);
        if dim == 0 || kv_dim == 0 || bytes == 0 {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                "CUDA decode acts need dim>0 and kv_dim>0".into(),
            )));
        }
        if bytes > self.scratch_bytes {
            return Err(AllpacaError::Fatal(allpaca_types::FatalError::OutOfMemory(
                format!(
                    "CUDA decode acts ({bytes} bytes) exceed scratch {}",
                    self.scratch_bytes
                ),
            )));
        }
        let base = self.scratch.cast::<f32>();
        Ok(CudaDecodeActs {
            x: base,
            norm: unsafe { base.add(dim) },
            q: unsafe { base.add(2 * dim) },
            k: unsafe { base.add(3 * dim) },
            v: unsafe { base.add(3 * dim + kv_dim) },
            attn: unsafe { base.add(3 * dim + 2 * kv_dim) },
            proj: unsafe { base.add(4 * dim + 2 * kv_dim) },
            bytes,
        })
    }

    /// Scratch K/V pair after a byte prefix (QKV acts live in `[0, prefix)`).
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

    /// Scratch K/V pair for packing paged KV into one contiguous device view.
    pub fn kv_scratch_pair(
        &self,
        n_kv: usize,
        kv_width: usize,
    ) -> Result<(*mut f32, *mut f32), AllpacaError> {
        self.kv_scratch_pair_after(0, n_kv, kv_width)
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

    pub fn gemm_nt_q8_dev(
        &self,
        a: *const f32,
        b: *const u8,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        set_device(self.device)?;
        gemm_nt_q8_0_device(a, b, c, m, n, k)
    }

    pub fn embed_gather_q8_dev(
        &self,
        table: *const u8,
        token: i32,
        out: *mut f32,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        set_device(self.device)?;
        embed_gather_q8_0_device(table, token, out, dim)
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
            let rc = unsafe { ffi::allpaca_cuda_residual_add(x, y, out, n, std::ptr::null_mut()) };
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
            let rc = unsafe { ffi::allpaca_cuda_rmsnorm(x, w, out, n, eps, std::ptr::null_mut()) };
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
            let rc = unsafe { ffi::allpaca_cuda_silu_mul(gate, up, out, n, std::ptr::null_mut()) };
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
            let rc = unsafe {
                ffi::allpaca_cuda_embed_gather(table, token, out, dim, std::ptr::null_mut())
            };
            status(rc, "embed_gather")?;
            let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
            status(sync, "stream_synchronize")
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
            let rc = unsafe {
                ffi::allpaca_cuda_rope(x, n_heads, head_dim, pos, theta, std::ptr::null_mut())
            };
            status(rc, "rope")?;
            let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
            status(sync, "stream_synchronize")
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
                    std::ptr::null_mut(),
                )
            };
            status(rc, "attention")?;
            let sync = unsafe { ffi::allpaca_cuda_stream_synchronize(std::ptr::null_mut()) };
            status(sync, "stream_synchronize")
        }
        #[cfg(not(all(feature = "cuda", allpaca_cuda_runtime)))]
        {
            let _ = (
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            );
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
            memcpy(
                out.as_mut_ptr().cast(),
                d_out,
                (dim as usize) * 4,
                MEMCPY_D2H,
            )?;
            self.activation_d2h
                .fetch_add((dim as usize * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_table);
        device_free(d_out);
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
            memcpy(x.as_mut_ptr().cast(), d_x.cast(), n * 4, MEMCPY_D2H)?;
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
                d_q,
                d_k,
                d_v,
                d_out.cast(),
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
            )?;
            memcpy(out.as_mut_ptr().cast(), d_out, q_n * 4, MEMCPY_D2H)?;
            self.activation_d2h
                .fetch_add((q_n * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_q);
        self.free_ptr(d_k);
        self.free_ptr(d_v);
        device_free(d_out);
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
                d_q,
                k_dev,
                v_dev,
                d_out.cast(),
                n_q,
                n_kv,
                n_heads,
                n_kv_heads,
                head_dim,
                causal,
            )?;
            memcpy(out.as_mut_ptr().cast(), d_out, q_n * 4, MEMCPY_D2H)?;
            self.activation_d2h
                .fetch_add((q_n * 4) as u64, Ordering::Relaxed);
            Ok(())
        })();
        self.free_ptr(d_q);
        device_free(d_out);
        result
    }

    /// Upload host F32 into a device blob (activation H2D accounting).
    pub fn upload_f32(&self, host: &[f32]) -> Result<*mut f32, AllpacaError> {
        let bytes = host.len().saturating_mul(4);
        set_device(self.device)?;
        let ptr = device_malloc(bytes, self.device)?;
        memcpy(ptr, host.as_ptr().cast(), bytes, MEMCPY_H2D)?;
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
        memcpy(dst.as_mut_ptr().cast(), src.cast(), bytes, MEMCPY_D2H)?;
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
        device_free(dc);
        result
    }

    /// Q8 GEMM with host activation A and device-resident staged Q8 B → host C.
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
            self.gemm_nt_q8_dev(da, b_dev, dc.cast(), m, n, k)?;
            self.download_f32(dc.cast(), &mut c[..c_n])
        })();
        self.free_ptr(da);
        device_free(dc);
        result
    }

    /// Embed gather from a device-resident staged Q8 table → host out.
    pub fn embed_gather_q8_dev_table(
        &self,
        table_dev: *const u8,
        token: i32,
        out: &mut [f32],
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if table_dev.is_null() || dim <= 0 || token < 0 || out.len() < dim as usize {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "invalid CUDA embed_gather_q8_dev_table".into(),
            )));
        }
        let dout = device_malloc((dim as usize) * 4, self.device)?;
        let result = (|| {
            self.embed_gather_q8_dev(table_dev, token, dout.cast(), dim)?;
            self.download_f32(dout.cast(), &mut out[..dim as usize])
        })();
        device_free(dout);
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
        device_free(dout);
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
        device_free(dout);
        result
    }

    /// Copy host F32 into an existing device buffer (vision soft embeds / soft_x).
    /// Mirrors [`allpaca_hip::HipWorkspace::copy_h2d_f32`].
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
        memcpy(dst.cast(), src.as_ptr().cast(), bytes, MEMCPY_H2D)?;
        self.activation_h2d
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(())
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
        Ok((
            Self {
                device,
                ptr: ptr.cast(),
                bytes,
            },
            bytes as u64,
        ))
    }

    /// Concatenate packed byte slices (F32 and/or staged Q8) into one device blob.
    pub fn upload_bytes(device: i32, parts: &[&[u8]]) -> Result<(Self, u64), AllpacaError> {
        let bytes: usize = parts.iter().map(|p| p.len()).sum();
        if bytes == 0 {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "empty CUDA weight upload".into(),
            )));
        }
        set_device(device)?;
        let ptr = device_malloc(bytes, device)?;
        let mut off = 0usize;
        for part in parts {
            if part.is_empty() {
                continue;
            }
            memcpy(
                unsafe { ptr.add(off) },
                part.as_ptr().cast(),
                part.len(),
                MEMCPY_H2D,
            )?;
            off += part.len();
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
        Ok(self.byte_ptr_at(byte_off)?.cast())
    }

    pub fn byte_ptr_at(&self, byte_off: usize) -> Result<*mut u8, AllpacaError> {
        if byte_off >= self.bytes {
            return Err(AllpacaError::Degraded(DegradedError::FeatureDisabled(
                "CUDA weight ptr out of range".into(),
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
        memcpy(back.as_mut_ptr().cast(), p0.cast(), 16, MEMCPY_D2H).expect("d2h");
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
            let err = gemm_nt(
                &[1.0, 0.0, 0.0, 1.0],
                &[1.0, 0.0, 0.0, 1.0],
                &mut c,
                2,
                2,
                2,
            );
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
    fn decode_acts_kv_scratch_do_not_overlap_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let ws = CudaWorkspace::new(0, 8 * 1024 * 1024).expect("workspace");
        let dim = 32usize;
        let kv_dim = 16usize;
        let acts = ws.decode_acts(dim, kv_dim).expect("acts");
        let marker = vec![7.0f32; dim];
        ws.copy_h2d_f32(acts.q, &marker).expect("q h2d");
        let n_kv = 4usize;
        let (dk, dv) = ws
            .kv_scratch_pair_after(acts.bytes, n_kv, kv_dim)
            .expect("kv scratch");
        let packed = vec![9.0f32; n_kv * kv_dim];
        ws.copy_h2d_f32(dk, &packed).expect("k pack");
        ws.copy_h2d_f32(dv, &packed).expect("v pack");
        let mut back = vec![0.0f32; dim];
        ws.download_f32(acts.q, &mut back).expect("q d2h");
        assert_eq!(
            back, marker,
            "KV pack must not clobber QKV activation slots"
        );
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
        memcpy(back.as_mut_ptr().cast(), d, src.len() * 4, MEMCPY_D2H).expect("d2h");
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
            allpaca_cpu::rope(
                &mut cpu_x[off..off + head_dim as usize],
                pos as usize,
                theta,
            )
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

    #[test]
    fn stage_q8_0_gemv_uses_36_byte_column_major() {
        let mut packed = Vec::new();
        for col in 0..8u8 {
            let mut blk = vec![0u8; Q8_PACKED_BLOCK];
            blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes());
            blk[2] = col + 1;
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

    fn cpu_gemm_nt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a[row * k + t] * b[col * k + t];
                }
                c[row * n + col] = acc;
            }
        }
    }

    fn dequant_q8_matrix(b_packed: &[u8], n: usize, k: usize) -> Vec<f32> {
        let nblocks = k.div_ceil(32);
        let mut b_deq = vec![0.0f32; n * k];
        for row in 0..n {
            let blk = &b_packed[row * nblocks * 34..(row + 1) * nblocks * 34];
            let mut tmp = vec![0.0f32; nblocks * 32];
            allpaca_cpu::dequant_q8_0(blk, nblocks * 32, &mut tmp).expect("dequant");
            for j in 0..k {
                b_deq[row * k + j] = tmp[j];
            }
        }
        b_deq
    }

    /// Public `gemm_nt_q8_0` must match CPU dequant + NT GEMM on decode shapes.
    #[test]
    fn gemm_nt_q8_0_matches_cpu_dequant_gemm() {
        if device_count() <= 0 {
            return;
        }
        let shapes = [
            (1usize, 8usize, 32usize),
            (1, 17, 32),
            (2, 5, 64),
            (1, 64, 128),
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
                b_q8.extend(allpaca_cpu::quantize_q8_0(&b_f[row * k..(row + 1) * k]));
            }
            let b_deq = dequant_q8_matrix(&b_q8, n, k);
            let mut cpu = vec![0.0f32; m * n];
            let mut cuda = vec![0.0f32; m * n];
            cpu_gemm_nt(&a, &b_deq, &mut cpu, m, n, k);
            gemm_nt_q8_0(&a, &b_q8, &mut cuda, m as i32, n as i32, k as i32)
                .expect("CUDA gemm_nt_q8_0 must run when a device is present");
            for i in 0..m * n {
                assert!(
                    (cuda[i] - cpu[i]).abs() < 2e-3,
                    "gemm_nt_q8_0 mismatch shape {m}x{n}x{k} idx={i} cuda={} cpu={}",
                    cuda[i],
                    cpu[i]
                );
            }
        }
    }
}
