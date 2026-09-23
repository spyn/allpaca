//! Vulkan/`ash` workspace for rust-gpu kernels. Feature-gated like `allpaca-cuda`.
//!
//! `--features vulkan` enables the path when rust-gpu successfully emits SPIR-V
//! (`allpaca_vulkan_runtime`). Without shaders or a device, `device_count()` is 0
//! and the crate still compiles.

use allpaca_types::{AllpacaError, DegradedError, DeviceInfo};
use std::sync::atomic::{AtomicU64, Ordering};

fn degraded(msg: impl Into<String>) -> AllpacaError {
    AllpacaError::Degraded(DegradedError::FeatureDisabled(msg.into()))
}

#[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
mod runtime;

/// Byte range inside a device buffer (storage-buffer bind).
#[derive(Clone, Copy, Debug)]
pub struct DeviceSlice {
    pub offset: u64,
    pub bytes: u64,
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    buffer: ash::vk::Buffer,
}

#[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
impl DeviceSlice {
    fn dummy(offset: u64, bytes: u64) -> Self {
        Self { offset, bytes }
    }
}

/// True when this build embedded a rust-gpu SPIR-V module.
pub fn runtime_linked() -> bool {
    cfg!(all(feature = "vulkan", allpaca_vulkan_runtime))
}

/// True when this build embedded the glslang native-f16 GEMV (`gemm_f16.comp`).
pub fn glsl_f16_linked() -> bool {
    cfg!(allpaca_vulkan_glsl_f16)
}

pub fn device_count() -> i32 {
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    {
        runtime::device_count()
    }
    #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
    {
        0
    }
}

pub fn enumerate_devices() -> Vec<DeviceInfo> {
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    {
        runtime::enumerate_devices()
    }
    #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
    {
        Vec::new()
    }
}

/// F32 Vulkan workspace: scratch + GEMM/elementwise/attention matching `CudaWorkspace`.
pub struct VulkanWorkspace {
    pub device: i32,
    scratch_bytes: usize,
    activation_h2d: AtomicU64,
    activation_d2h: AtomicU64,
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    inner: runtime::Inner,
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    #[allow(dead_code)]
    scratch: runtime::DeviceBuf,
}

// SAFETY: `Inner` serializes queue submit and allocator access with Mutex;
// ash handles are not used from overlapping threads without that lock.
unsafe impl Send for VulkanWorkspace {}
unsafe impl Sync for VulkanWorkspace {}

impl VulkanWorkspace {
    pub fn new(device: i32, scratch_bytes: usize) -> Result<Self, AllpacaError> {
        if device_count() <= 0 {
            return Err(degraded("Vulkan workspace: no device"));
        }
        let scratch_bytes = scratch_bytes.max(4096);
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let inner = runtime::Inner::new(device)?;
            let scratch = inner.alloc(scratch_bytes as u64, "scratch")?;
            Ok(Self {
                device,
                scratch_bytes,
                activation_h2d: AtomicU64::new(0),
                activation_d2h: AtomicU64::new(0),
                inner,
                scratch,
            })
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (device, scratch_bytes);
            Err(degraded("Vulkan workspace unavailable"))
        }
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

    /// True when dispatch uses the glslang `float16_t` GEMV instead of rust-gpu unpack.
    pub fn uses_glsl_f16_gemm(&self) -> bool {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.uses_glsl_f16_gemm()
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            false
        }
    }

    #[cfg_attr(not(all(feature = "vulkan", allpaca_vulkan_runtime)), allow(dead_code))]
    fn account_h2d(&self, bytes: u64) {
        self.activation_h2d.fetch_add(bytes, Ordering::Relaxed);
    }

    #[cfg_attr(not(all(feature = "vulkan", allpaca_vulkan_runtime)), allow(dead_code))]
    fn account_d2h(&self, bytes: u64) {
        self.activation_d2h.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn gemm_nt_dev(
        &self,
        a: DeviceSlice,
        b: DeviceSlice,
        c: DeviceSlice,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.gemm_nt(a, b, c, m, n, k)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (a, b, c, m, n, k);
            Err(degraded("Vulkan gemm unavailable"))
        }
    }

    pub fn gemm_nt_host_a_dev_b(
        &self,
        a: &[f32],
        b: DeviceSlice,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(degraded("invalid Vulkan gemm_nt_host_a_dev_b"));
        }
        let a_n = (m as usize).saturating_mul(k as usize);
        let c_n = (m as usize).saturating_mul(n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(degraded("GEMM buffer size mismatch"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let da = self.inner.upload_f32(&a[..a_n])?;
            self.account_h2d((a_n * 4) as u64);
            let dc = self.inner.alloc((c_n * 4) as u64, "gemm_c")?;
            let sa = runtime::Inner::slice_of(&da, 0, (a_n * 4) as u64);
            let sc = runtime::Inner::slice_of(&dc, 0, (c_n * 4) as u64);
            self.inner.gemm_nt(sa, b, sc, m, n, k)?;
            self.inner.download_f32(&dc, &mut c[..c_n])?;
            self.account_d2h((c_n * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (a, b, c, m, n, k);
            Err(degraded("Vulkan gemm unavailable"))
        }
    }

    pub fn gemm_nt_f16_host_a_dev_b(
        &self,
        a: &[f32],
        b: DeviceSlice,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(degraded("invalid Vulkan gemm_nt_f16_host_a_dev_b"));
        }
        let a_n = (m as usize).saturating_mul(k as usize);
        let c_n = (m as usize).saturating_mul(n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(degraded("GEMM buffer size mismatch"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let da = self.inner.upload_f32(&a[..a_n])?;
            self.account_h2d((a_n * 4) as u64);
            let dc = self.inner.alloc((c_n * 4) as u64, "gemm_c")?;
            let sa = runtime::Inner::slice_of(&da, 0, (a_n * 4) as u64);
            let sc = runtime::Inner::slice_of(&dc, 0, (c_n * 4) as u64);
            self.inner.gemm_nt_f16(sa, b, sc, m, n, k)?;
            self.inner.download_f32(&dc, &mut c[..c_n])?;
            self.account_d2h((c_n * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (a, b, c, m, n, k);
            Err(degraded("Vulkan gemm unavailable"))
        }
    }

    pub fn gemm_nt_q8_host_a_dev_b(
        &self,
        a: &[f32],
        b: DeviceSlice,
        c: &mut [f32],
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<(), AllpacaError> {
        if m <= 0 || n <= 0 || k <= 0 {
            return Err(degraded("invalid Vulkan gemm_nt_q8_host_a_dev_b"));
        }
        let a_n = (m as usize).saturating_mul(k as usize);
        let c_n = (m as usize).saturating_mul(n as usize);
        if a.len() < a_n || c.len() < c_n {
            return Err(degraded("GEMM buffer size mismatch"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let da = self.inner.upload_f32(&a[..a_n])?;
            self.account_h2d((a_n * 4) as u64);
            let dc = self.inner.alloc((c_n * 4) as u64, "gemm_c")?;
            let sa = runtime::Inner::slice_of(&da, 0, (a_n * 4) as u64);
            let sc = runtime::Inner::slice_of(&dc, 0, (c_n * 4) as u64);
            self.inner.gemm_nt_q8(sa, b, sc, m, n, k)?;
            self.inner.download_f32(&dc, &mut c[..c_n])?;
            self.account_d2h((c_n * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (a, b, c, m, n, k);
            Err(degraded("Vulkan gemm unavailable"))
        }
    }

    pub fn residual_add_dev(
        &self,
        x: DeviceSlice,
        y: DeviceSlice,
        out: DeviceSlice,
        n: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.residual_add(x, y, out, n)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (x, y, out, n);
            Err(degraded("Vulkan residual_add unavailable"))
        }
    }

    pub fn residual_add(
        &self,
        x: &[f32],
        y: &[f32],
        out: &mut [f32],
        n: i32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 || x.len() < n as usize || y.len() < n as usize || out.len() < n as usize {
            return Err(degraded("invalid Vulkan residual_add"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dx = self.inner.upload_f32(&x[..n as usize])?;
            let dy = self.inner.upload_f32(&y[..n as usize])?;
            self.account_h2d((n as usize * 8) as u64);
            let dout = self.inner.alloc((n as usize * 4) as u64, "residual_out")?;
            let sx = runtime::Inner::slice_of(&dx, 0, (n as usize * 4) as u64);
            let sy = runtime::Inner::slice_of(&dy, 0, (n as usize * 4) as u64);
            let so = runtime::Inner::slice_of(&dout, 0, (n as usize * 4) as u64);
            self.inner.residual_add(sx, sy, so, n)?;
            self.inner.download_f32(&dout, &mut out[..n as usize])?;
            self.account_d2h((n as usize * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (x, y, out, n);
            Err(degraded("Vulkan residual_add unavailable"))
        }
    }

    pub fn rmsnorm_dev(
        &self,
        x: DeviceSlice,
        w: DeviceSlice,
        out: DeviceSlice,
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.rmsnorm(x, w, out, n, eps)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (x, w, out, n, eps);
            Err(degraded("Vulkan rmsnorm unavailable"))
        }
    }

    pub fn rmsnorm_host_x_dev_w(
        &self,
        x: &[f32],
        w: DeviceSlice,
        out: &mut [f32],
        n: i32,
        eps: f32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 || x.len() < n as usize || out.len() < n as usize {
            return Err(degraded("invalid Vulkan rmsnorm_host_x_dev_w"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dx = self.inner.upload_f32(&x[..n as usize])?;
            self.account_h2d((n as usize * 4) as u64);
            let dout = self.inner.alloc((n as usize * 4) as u64, "rms_out")?;
            let sx = runtime::Inner::slice_of(&dx, 0, (n as usize * 4) as u64);
            let so = runtime::Inner::slice_of(&dout, 0, (n as usize * 4) as u64);
            self.inner.rmsnorm(sx, w, so, n, eps)?;
            self.inner.download_f32(&dout, &mut out[..n as usize])?;
            self.account_d2h((n as usize * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (x, w, out, n, eps);
            Err(degraded("Vulkan rmsnorm unavailable"))
        }
    }

    pub fn silu_mul_dev(
        &self,
        gate: DeviceSlice,
        up: DeviceSlice,
        out: DeviceSlice,
        n: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.silu_mul(gate, up, out, n)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (gate, up, out, n);
            Err(degraded("Vulkan silu_mul unavailable"))
        }
    }

    pub fn silu_mul(
        &self,
        gate: &[f32],
        up: &[f32],
        out: &mut [f32],
        n: i32,
    ) -> Result<(), AllpacaError> {
        if n <= 0 || gate.len() < n as usize || up.len() < n as usize || out.len() < n as usize {
            return Err(degraded("invalid Vulkan silu_mul"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dg = self.inner.upload_f32(&gate[..n as usize])?;
            let du = self.inner.upload_f32(&up[..n as usize])?;
            self.account_h2d((n as usize * 8) as u64);
            let dout = self.inner.alloc((n as usize * 4) as u64, "silu_out")?;
            let sg = runtime::Inner::slice_of(&dg, 0, (n as usize * 4) as u64);
            let su = runtime::Inner::slice_of(&du, 0, (n as usize * 4) as u64);
            let so = runtime::Inner::slice_of(&dout, 0, (n as usize * 4) as u64);
            self.inner.silu_mul(sg, su, so, n)?;
            self.inner.download_f32(&dout, &mut out[..n as usize])?;
            self.account_d2h((n as usize * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (gate, up, out, n);
            Err(degraded("Vulkan silu_mul unavailable"))
        }
    }

    pub fn embed_gather_dev(
        &self,
        table: DeviceSlice,
        token: i32,
        out: DeviceSlice,
        dim: i32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.embed_gather(table, token, out, dim)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (table, token, out, dim);
            Err(degraded("Vulkan embed_gather unavailable"))
        }
    }

    pub fn embed_gather(
        &self,
        table: &[f32],
        token: i32,
        out: &mut [f32],
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if dim <= 0 || token < 0 || out.len() < dim as usize {
            return Err(degraded("invalid embed_gather buffers"));
        }
        let need = (token as usize + 1).saturating_mul(dim as usize);
        if table.len() < need {
            return Err(degraded("embed table too small"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dt = self.inner.upload_f32(&table[..need])?;
            self.account_h2d((need * 4) as u64);
            let dout = self.inner.alloc((dim as usize * 4) as u64, "embed_out")?;
            let st = runtime::Inner::slice_of(&dt, 0, (need * 4) as u64);
            let so = runtime::Inner::slice_of(&dout, 0, (dim as usize * 4) as u64);
            self.inner.embed_gather(st, token, so, dim)?;
            self.inner.download_f32(&dout, &mut out[..dim as usize])?;
            self.account_d2h((dim as usize * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            Err(degraded("Vulkan embed_gather unavailable"))
        }
    }

    pub fn embed_gather_dev_table(
        &self,
        table: DeviceSlice,
        token: i32,
        out: &mut [f32],
        dim: i32,
    ) -> Result<(), AllpacaError> {
        if dim <= 0 || token < 0 || out.len() < dim as usize {
            return Err(degraded("invalid Vulkan embed_gather_dev_table"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dout = self.inner.alloc((dim as usize * 4) as u64, "embed_out")?;
            let so = runtime::Inner::slice_of(&dout, 0, (dim as usize * 4) as u64);
            self.inner.embed_gather(table, token, so, dim)?;
            self.inner.download_f32(&dout, &mut out[..dim as usize])?;
            self.account_d2h((dim as usize * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = table;
            Err(degraded("Vulkan embed_gather unavailable"))
        }
    }

    pub fn rope_dev(
        &self,
        x: DeviceSlice,
        n_heads: i32,
        head_dim: i32,
        pos: i32,
        theta: f32,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.rope(x, n_heads, head_dim, pos, theta)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (x, n_heads, head_dim, pos, theta);
            Err(degraded("Vulkan rope unavailable"))
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
        let n = (n_heads as usize).saturating_mul(head_dim as usize);
        if n == 0 || x.len() < n || head_dim % 2 != 0 {
            return Err(degraded("invalid rope shape"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dx = self.inner.upload_f32(&x[..n])?;
            self.account_h2d((n * 4) as u64);
            let sx = runtime::Inner::slice_of(&dx, 0, (n * 4) as u64);
            self.inner.rope(sx, n_heads, head_dim, pos, theta)?;
            self.inner.download_f32(&dx, &mut x[..n])?;
            self.account_d2h((n * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (x, n_heads, head_dim, pos, theta);
            Err(degraded("Vulkan rope unavailable"))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_dev(
        &self,
        q: DeviceSlice,
        k: DeviceSlice,
        v: DeviceSlice,
        out: DeviceSlice,
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.attention(
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            )
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (
                q, k, v, out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            );
            Err(degraded("Vulkan attention unavailable"))
        }
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
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        let kv_n = (n_kv as usize) * (n_kv_heads as usize) * (head_dim as usize);
        if q.len() < q_n || k.len() < kv_n || v.len() < kv_n || out.len() < q_n {
            return Err(degraded("attention buffer size mismatch"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dq = self.inner.upload_f32(&q[..q_n])?;
            let dk = self.inner.upload_f32(&k[..kv_n])?;
            let dv = self.inner.upload_f32(&v[..kv_n])?;
            self.account_h2d(((q_n + kv_n + kv_n) * 4) as u64);
            let dout = self.inner.alloc((q_n * 4) as u64, "attn_out")?;
            let sq = runtime::Inner::slice_of(&dq, 0, (q_n * 4) as u64);
            let sk = runtime::Inner::slice_of(&dk, 0, (kv_n * 4) as u64);
            let sv = runtime::Inner::slice_of(&dv, 0, (kv_n * 4) as u64);
            let so = runtime::Inner::slice_of(&dout, 0, (q_n * 4) as u64);
            self.inner.attention(
                sq, sk, sv, so, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            )?;
            self.inner.download_f32(&dout, &mut out[..q_n])?;
            self.account_d2h((q_n * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = causal;
            Err(degraded("Vulkan attention unavailable"))
        }
    }

    /// Host Q + resident device K/V (no KV D2H). Used by Vulkan decode pages.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_host_q_dev_kv(
        &self,
        q: &[f32],
        k: DeviceSlice,
        v: DeviceSlice,
        out: &mut [f32],
        n_q: i32,
        n_kv: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        causal: bool,
    ) -> Result<(), AllpacaError> {
        let q_n = (n_q as usize) * (n_heads as usize) * (head_dim as usize);
        if q.len() < q_n || out.len() < q_n {
            return Err(degraded("attention_host_q_dev_kv buffer mismatch"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let dq = self.inner.upload_f32(&q[..q_n])?;
            self.account_h2d((q_n * 4) as u64);
            let dout = self.inner.alloc((q_n * 4) as u64, "attn_out")?;
            let sq = runtime::Inner::slice_of(&dq, 0, (q_n * 4) as u64);
            let so = runtime::Inner::slice_of(&dout, 0, (q_n * 4) as u64);
            self.inner.attention(
                sq, k, v, so, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
            )?;
            self.inner.download_f32(&dout, &mut out[..q_n])?;
            self.account_d2h((q_n * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (k, v, n_kv, n_kv_heads, causal);
            Err(degraded("Vulkan attention unavailable"))
        }
    }

    pub fn copy_h2d_f32(
        &self,
        dst: &VulkanDeviceWeights,
        byte_off: usize,
        src: &[f32],
    ) -> Result<(), AllpacaError> {
        if src.is_empty() {
            return Ok(());
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner.write_into(&dst.buf, byte_off as u64, src)?;
            self.account_h2d((src.len() * 4) as u64);
            Ok(())
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (dst, byte_off, src);
            Err(degraded("Vulkan H2D copy unavailable"))
        }
    }

    pub fn alloc_kv_page(&self, bytes: usize) -> Result<VulkanKvPage, AllpacaError> {
        let bytes = bytes.max(256);
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let buf = self.inner.alloc(bytes as u64, "kv")?;
            Ok(VulkanKvPage { buf, bytes })
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = bytes;
            Err(degraded("Vulkan KV alloc unavailable"))
        }
    }

    pub fn kv_write_f32(
        &self,
        page: &VulkanKvPage,
        float_off: usize,
        src: &[f32],
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner
                .write_into(&page.buf, (float_off * 4) as u64, src)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (page, float_off, src);
            Err(degraded("Vulkan KV write unavailable"))
        }
    }

    pub fn kv_read_f32(
        &self,
        page: &VulkanKvPage,
        float_off: usize,
        dst: &mut [f32],
    ) -> Result<(), AllpacaError> {
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            self.inner
                .read_into(&page.buf, (float_off * 4) as u64, dst)
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = (page, float_off, dst);
            Err(degraded("Vulkan KV read unavailable"))
        }
    }

    pub fn kv_slice(
        &self,
        page: &VulkanKvPage,
        float_off: usize,
        n: usize,
    ) -> Result<DeviceSlice, AllpacaError> {
        let bytes = n.saturating_mul(4);
        let start = float_off.saturating_mul(4);
        if start.saturating_add(bytes) > page.bytes {
            return Err(degraded("Vulkan KV slice out of range"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            Ok(runtime::Inner::slice_of(
                &page.buf,
                start as u64,
                bytes as u64,
            ))
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            Ok(DeviceSlice::dummy(start as u64, bytes as u64))
        }
    }
}

/// Device-resident KV page (host-visible storage buffer).
pub struct VulkanKvPage {
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    buf: runtime::DeviceBuf,
    bytes: usize,
}

// SAFETY: `DeviceBuf` is only touched under the workspace submit/allocator mutexes.
unsafe impl Send for VulkanKvPage {}
unsafe impl Sync for VulkanKvPage {}

impl VulkanKvPage {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Packed F32 weights uploaded once (not activation traffic).
pub struct VulkanDeviceWeights {
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    buf: runtime::DeviceBuf,
    bytes: usize,
}

// SAFETY: `DeviceBuf` is only touched under the workspace submit/allocator mutexes.
unsafe impl Send for VulkanDeviceWeights {}
unsafe impl Sync for VulkanDeviceWeights {}

/// Host bytes for a packed Vulkan weight upload (F32 or packed F16/Q8).
pub enum WeightBytes<'a> {
    F32(&'a [f32]),
    Packed(&'a [u8]),
}

impl VulkanDeviceWeights {
    /// Concatenate host F32 slices with 256-byte padding between tensors.
    pub fn upload(ws: &VulkanWorkspace, slices: &[&[f32]]) -> Result<(Self, u64), AllpacaError> {
        let parts: Vec<WeightBytes<'_>> = slices.iter().copied().map(WeightBytes::F32).collect();
        Self::upload_parts(ws, &parts)
    }

    /// Concatenate F32 and packed (F16/Q8) tensors with 256-byte padding.
    pub fn upload_parts(
        ws: &VulkanWorkspace,
        parts: &[WeightBytes<'_>],
    ) -> Result<(Self, u64), AllpacaError> {
        let mut packed: Vec<u8> = Vec::new();
        for part in parts {
            match part {
                // SAFETY: `f32` is POD; the byte length is `len * 4`.
                WeightBytes::F32(slice) => packed.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), slice.len() * 4)
                }),
                WeightBytes::Packed(bytes) => packed.extend_from_slice(bytes),
            }
            while packed.len() % 256 != 0 {
                packed.push(0);
            }
        }
        if packed.is_empty() {
            return Err(degraded("empty Vulkan weight upload"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            let buf = ws.inner.alloc(packed.len() as u64, "weights")?;
            ws.inner.write_bytes(&buf, 0, &packed)?;
            let bytes = packed.len();
            Ok((Self { buf, bytes }, bytes as u64))
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            let _ = ws;
            Err(degraded("Vulkan weight upload unavailable"))
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn slice_f32(&self, byte_off: usize, n: usize) -> Result<DeviceSlice, AllpacaError> {
        let bytes = n.saturating_mul(4);
        if byte_off.saturating_add(bytes) > self.bytes {
            return Err(degraded("Vulkan weight slice out of range"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            Ok(runtime::Inner::slice_of(
                &self.buf,
                byte_off as u64,
                bytes as u64,
            ))
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            Ok(DeviceSlice::dummy(byte_off as u64, bytes as u64))
        }
    }

    pub fn slice_bytes(&self, byte_off: usize, bytes: usize) -> Result<DeviceSlice, AllpacaError> {
        if byte_off.saturating_add(bytes) > self.bytes {
            return Err(degraded("Vulkan weight slice out of range"));
        }
        #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
        {
            Ok(runtime::Inner::slice_of(
                &self.buf,
                byte_off as u64,
                bytes as u64,
            ))
        }
        #[cfg(not(all(feature = "vulkan", allpaca_vulkan_runtime)))]
        {
            Ok(DeviceSlice::dummy(byte_off as u64, bytes as u64))
        }
    }
}

pub fn gemm_nt(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), AllpacaError> {
    let ws = VulkanWorkspace::new(0, 4096)?;
    let (weights, _) = VulkanDeviceWeights::upload(&ws, &[b])?;
    let b_n = (n as usize).saturating_mul(k as usize);
    let bs = weights.slice_f32(0, b_n)?;
    ws.gemm_nt_host_a_dev_b(a, bs, c, m, n, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_or_no_device_reports_zero_when_no_gpu() {
        assert!(device_count() >= 0);
        if !runtime_linked() || device_count() == 0 {
            assert!(enumerate_devices().is_empty() || !runtime_linked());
            if device_count() == 0 {
                assert!(VulkanWorkspace::new(0, 4096).is_err());
            }
        }
    }

    #[test]
    #[cfg(all(feature = "vulkan", allpaca_vulkan_runtime))]
    fn runtime_linked_when_shaders_build() {
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
        let mut gpu = vec![0.0f32; (m * n) as usize];
        let mut cpu = vec![0.0f32; (m * n) as usize];
        gemm_nt(&a, &b, &mut gpu, m, n, k).expect("vulkan gemm");
        allpaca_cpu::matmul_nt(&a, &b, &mut cpu, m as usize, n as usize, k as usize)
            .expect("cpu gemm");
        for i in 0..cpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "mismatch at {i}: vulkan={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
        let ws = VulkanWorkspace::new(0, 8192).expect("workspace");
        assert!(ws.allocated_bytes() >= 8192);
    }

    fn pack_f16_bytes(b: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(b.len() * 2 + 4);
        for v in b {
            out.extend_from_slice(&allpaca_cpu::f32_to_f16(*v).to_le_bytes());
        }
        while out.len() % 4 != 0 {
            out.push(0);
        }
        out
    }

    #[test]
    fn glsl_f16_link_status_is_stable() {
        let linked = glsl_f16_linked();
        if linked {
            assert!(
                runtime_linked(),
                "glslang f16 SPIR-V is only embedded when rust-gpu runtime linked"
            );
        }
    }

    #[test]
    fn gemm_nt_f16_unpack_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let m = 1i32;
        let n = 8i32;
        let k = 16i32;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..(n * k)).map(|i| (i as f32 + 1.0) * 0.02).collect();
        let packed = pack_f16_bytes(&b);
        let mut gpu = vec![0.0f32; (m * n) as usize];
        let mut cpu = vec![0.0f32; (m * n) as usize];
        let mut b_deq = vec![0.0f32; b.len()];
        for (i, v) in b.iter().enumerate() {
            b_deq[i] = allpaca_cpu::f16_to_f32(allpaca_cpu::f32_to_f16(*v));
        }
        allpaca_cpu::matmul_nt(&a, &b_deq, &mut cpu, m as usize, n as usize, k as usize)
            .expect("cpu gemm");
        let ws = VulkanWorkspace::new(0, 64 * 1024).expect("workspace");
        let (weights, _) =
            VulkanDeviceWeights::upload_parts(&ws, &[WeightBytes::Packed(&packed)]).expect("upload");
        let bs = weights.slice_bytes(0, packed.len()).expect("slice");
        ws.gemm_nt_f16_host_a_dev_b(&a, bs, &mut gpu, m, n, k)
            .expect("vulkan f16 gemm");
        let _ = (glsl_f16_linked(), ws.uses_glsl_f16_gemm());
        for i in 0..cpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 2e-3,
                "f16 unpack mismatch at {i}: vulkan={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn gemm_nt_q8_unpack_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let m = 1i32;
        let n = 4i32;
        let k = 32i32;
        let a: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.04).collect();
        let b: Vec<f32> = (0..(n * k)).map(|i| ((i as f32) - 8.0) * 0.03).collect();
        let packed = allpaca_cpu::quantize_q8_0(&b);
        let mut b_deq = vec![0.0f32; b.len()];
        allpaca_cpu::dequant_q8_0(&packed, b.len(), &mut b_deq).expect("dequant");
        let mut cpu = vec![0.0f32; (m * n) as usize];
        allpaca_cpu::matmul_nt(&a, &b_deq, &mut cpu, m as usize, n as usize, k as usize)
            .expect("cpu gemm");
        let mut gpu = vec![0.0f32; (m * n) as usize];
        let mut bytes = packed.clone();
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        let ws = VulkanWorkspace::new(0, 64 * 1024).expect("workspace");
        let (weights, _) =
            VulkanDeviceWeights::upload_parts(&ws, &[WeightBytes::Packed(&bytes)]).expect("upload");
        let bs = weights.slice_bytes(0, bytes.len()).expect("slice");
        ws.gemm_nt_q8_host_a_dev_b(&a, bs, &mut gpu, m, n, k)
            .expect("vulkan q8 gemm");
        for i in 0..cpu.len() {
            assert!(
                (gpu[i] - cpu[i]).abs() < 2e-3,
                "q8 unpack mismatch at {i}: vulkan={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn residual_add_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 8i32;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let y: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut gpu = vec![0.0f32; n as usize];
        let ws = VulkanWorkspace::new(0, 4096).expect("workspace");
        ws.residual_add(&x, &y, &mut gpu, n)
            .expect("vulkan residual_add");
        for i in 0..n as usize {
            let want = x[i] + y[i];
            assert!(
                (gpu[i] - want).abs() < 1e-5,
                "residual mismatch at {i}: vulkan={} cpu={}",
                gpu[i],
                want
            );
        }
    }

    #[test]
    fn rmsnorm_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 8usize;
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let w: Vec<f32> = vec![1.0; n];
        let mut gpu = vec![0.0f32; n];
        let mut cpu = vec![0.0f32; n];
        let ws = VulkanWorkspace::new(0, 4096).expect("workspace");
        let (ww, _) = VulkanDeviceWeights::upload(&ws, &[&w]).expect("upload w");
        ws.rmsnorm_host_x_dev_w(
            &x,
            ww.slice_f32(0, n).expect("w slice"),
            &mut gpu,
            n as i32,
            1e-5,
        )
        .expect("vulkan rmsnorm");
        allpaca_cpu::rmsnorm(&x, &w, 1e-5, &mut cpu).expect("cpu rmsnorm");
        for i in 0..n {
            assert!(
                (gpu[i] - cpu[i]).abs() < 1e-4,
                "rmsnorm mismatch at {i}: vulkan={} cpu={}",
                gpu[i],
                cpu[i]
            );
        }
    }

    #[test]
    fn silu_mul_matches_cpu_when_device_present() {
        if device_count() <= 0 {
            return;
        }
        let n = 8i32;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.2 - 0.4).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.15 + 0.5).collect();
        let mut gpu = vec![0.0f32; n as usize];
        let ws = VulkanWorkspace::new(0, 4096).expect("workspace");
        ws.silu_mul(&gate, &up, &mut gpu, n)
            .expect("vulkan silu_mul");
        for i in 0..n as usize {
            let want = allpaca_cpu::silu(gate[i]) * up[i];
            assert!(
                (gpu[i] - want).abs() < 1e-4,
                "silu_mul mismatch at {i}: vulkan={} cpu={}",
                gpu[i],
                want
            );
        }
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
        let mut gpu_out = vec![0.0f32; dim as usize];
        let ws = VulkanWorkspace::new(0, 4096).expect("workspace");
        ws.embed_gather(&table, token, &mut gpu_out, dim)
            .expect("vulkan embed_gather");
        let start = (token as usize) * (dim as usize);
        let cpu = &table[start..start + dim as usize];
        for i in 0..dim as usize {
            assert!(
                (gpu_out[i] - cpu[i]).abs() < 1e-6,
                "embed mismatch at {i}: vulkan={} cpu={}",
                gpu_out[i],
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
        let mut gpu_x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let mut cpu_x = gpu_x.clone();
        let ws = VulkanWorkspace::new(0, 4096).expect("workspace");
        ws.rope(&mut gpu_x, n_heads, head_dim, pos, theta)
            .expect("vulkan rope");
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
                (gpu_x[i] - cpu_x[i]).abs() < 1e-4,
                "rope mismatch at {i}: vulkan={} cpu={}",
                gpu_x[i],
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
        let mut gpu_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = VulkanWorkspace::new(0, 64 * 1024).expect("workspace");
        ws.attention(
            &q,
            &k,
            &v,
            &mut gpu_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            true,
        )
        .expect("vulkan attention");
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
                (gpu_out[i] - cpu_out[i]).abs() < 1e-4,
                "attention mismatch at {i}: vulkan={} cpu={}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn attention_host_q_dev_kv_matches_cpu_when_device_present() {
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
        let mut gpu_out = vec![0.0f32; q_n];
        let mut cpu_out = vec![0.0f32; q_n];
        let ws = VulkanWorkspace::new(0, 64 * 1024).expect("workspace");
        let k_page = ws.alloc_kv_page(kv_n * 4).expect("k page");
        let v_page = ws.alloc_kv_page(kv_n * 4).expect("v page");
        ws.kv_write_f32(&k_page, 0, &k).expect("write k");
        ws.kv_write_f32(&v_page, 0, &v).expect("write v");
        let k_dev = ws.kv_slice(&k_page, 0, kv_n).expect("k slice");
        let v_dev = ws.kv_slice(&v_page, 0, kv_n).expect("v slice");
        ws.attention_host_q_dev_kv(
            &q,
            k_dev,
            v_dev,
            &mut gpu_out,
            n_q,
            n_kv,
            n_heads,
            n_kv_heads,
            head_dim,
            true,
        )
        .expect("vulkan resident-kv attention");
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
                (gpu_out[i] - cpu_out[i]).abs() < 1e-4,
                "resident-kv attention mismatch at {i}: vulkan={} cpu={}",
                gpu_out[i],
                cpu_out[i]
            );
        }
    }
}
