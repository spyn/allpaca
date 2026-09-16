//! First-party Vulkan compute kernels (rust-gpu).
//!
//! Compiled to SPIR-V by `allpaca-vulkan`'s `build.rs`. The same crate also
//! builds as ordinary Rust so host tests can share push-constant layouts.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(clippy::too_many_arguments)]

use spirv_std::glam::UVec3;
use spirv_std::num_traits::Float;
use spirv_std::spirv;

pub const GEMM_TILE: u32 = 16;
pub const ELEM_THREADS: u32 = 256;
pub const REDUCE_THREADS: u32 = 64;
pub const ATTN_THREADS: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GemmPush {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct LenPush {
    pub n: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RmsPush {
    pub n: u32,
    pub eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmbedPush {
    pub token: u32,
    pub dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RopePush {
    pub n_heads: u32,
    pub head_dim: u32,
    pub pos: u32,
    pub theta: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnPush {
    pub n_q: u32,
    pub n_kv: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub causal: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// `C = A (m×k) * B^T (k×n)` with B stored as n×k (same as CPU `matmul_nt`).
#[spirv(compute(threads(16, 16)))]
pub fn gemm_f32(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] c: &mut [f32],
    #[spirv(push_constant)] dims: &GemmPush,
    #[spirv(workgroup)] tile_a: &mut [f32; 256],
    #[spirv(workgroup)] tile_b: &mut [f32; 256],
) {
    let row = gid.y;
    let col = gid.x;
    let tx = lid.x;
    let ty = lid.y;
    let m = dims.m;
    let n = dims.n;
    let k = dims.k;
    let mut acc = 0.0f32;
    let n_tiles = (k + GEMM_TILE - 1) / GEMM_TILE;
    let mut t = 0u32;
    while t < n_tiles {
        let a_col = t * GEMM_TILE + tx;
        let a_idx = (ty * GEMM_TILE + tx) as usize;
        tile_a[a_idx] = if row < m && a_col < k {
            a[(row * k + a_col) as usize]
        } else {
            0.0
        };
        let b_k = t * GEMM_TILE + ty;
        let b_idx = (tx * GEMM_TILE + ty) as usize;
        tile_b[b_idx] = if col < n && b_k < k {
            b[(col * k + b_k) as usize]
        } else {
            0.0
        };
        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
        let mut kk = 0u32;
        while kk < GEMM_TILE {
            acc += tile_a[(ty * GEMM_TILE + kk) as usize] * tile_b[(tx * GEMM_TILE + kk) as usize];
            kk += 1;
        }
        spirv_std::arch::workgroup_memory_barrier_with_group_sync();
        t += 1;
    }
    if row < m && col < n {
        c[(row * n + col) as usize] = acc;
    }
}

#[spirv(compute(threads(256), entry_point_name = "residual_add"))]
pub fn residual_add(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] y: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [f32],
    #[spirv(push_constant)] dims: &LenPush,
) {
    let i = gid.x;
    if i < dims.n {
        let idx = i as usize;
        out[idx] = x[idx] + y[idx];
    }
    // Keep a SPIR-V memory barrier so rust-gpu does not drop this entry point.
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
}

#[spirv(compute(threads(64)))]
pub fn rmsnorm(
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] w: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [f32],
    #[spirv(push_constant)] dims: &RmsPush,
    #[spirv(workgroup)] scratch: &mut [f32; 64],
) {
    let t = lid.x;
    let n = dims.n;
    let mut sumsq = 0.0f32;
    let mut i = t;
    while i < n {
        let v = x[i as usize];
        sumsq += v * v;
        i += REDUCE_THREADS;
    }
    scratch[t as usize] = sumsq;
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    if t == 0 {
        let mut total = 0.0f32;
        let mut s = 0u32;
        while s < REDUCE_THREADS {
            total += scratch[s as usize];
            s += 1;
        }
        let mean = total / (n as f32);
        scratch[0] = libm_rsqrt(mean + dims.eps);
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    let scale = scratch[0];
    let mut j = t;
    while j < n {
        let idx = j as usize;
        out[idx] = x[idx] * scale * w[idx];
        j += REDUCE_THREADS;
    }
}

fn libm_rsqrt(x: f32) -> f32 {
    1.0 / x.sqrt()
}

fn from_bits_neg_inf() -> f32 {
    f32::from_bits(0xff80_0000)
}

#[spirv(compute(threads(256)))]
pub fn silu_mul(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] gate: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] up: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] out: &mut [f32],
    #[spirv(push_constant)] dims: &LenPush,
) {
    let i = gid.x;
    if i < dims.n {
        let idx = i as usize;
        let g = gate[idx];
        let sig = 1.0 / (1.0 + (-g).exp());
        out[idx] = g * sig * up[idx];
    }
}

#[spirv(compute(threads(256), entry_point_name = "embed_gather"))]
pub fn embed_gather(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] table: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] out: &mut [f32],
    #[spirv(push_constant)] dims: &EmbedPush,
) {
    let i = gid.x;
    if i < dims.dim {
        let idx = i as usize;
        out[idx] = table[(dims.token * dims.dim) as usize + idx];
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
}

#[spirv(compute(threads(256)))]
pub fn rope(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] x: &mut [f32],
    #[spirv(push_constant)] dims: &RopePush,
) {
    let pair = gid.x;
    let n_pairs = dims.n_heads * (dims.head_dim / 2);
    if pair >= n_pairs {
        return;
    }
    let head = pair / (dims.head_dim / 2);
    let i = pair % (dims.head_dim / 2);
    let half = (dims.head_dim / 2) as f32;
    let freq = 1.0 / dims.theta.powf(i as f32 / half);
    let angle = (dims.pos as f32) * freq;
    let (sin, cos) = (angle.sin(), angle.cos());
    let base = (head * dims.head_dim + i * 2) as usize;
    let a = x[base];
    let b = x[base + 1];
    x[base] = a * cos - b * sin;
    x[base + 1] = a * sin + b * cos;
}

/// One workgroup per `(query, head)`. Threads stride the KV axis, then reduce.
#[spirv(compute(threads(64)))]
pub fn attention(
    #[spirv(workgroup_id)] wid: UVec3,
    #[spirv(local_invocation_id)] lid: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] q: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] k: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] v: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] out: &mut [f32],
    #[spirv(push_constant)] dims: &AttnPush,
    #[spirv(workgroup)] scratch: &mut [f32; 64],
) {
    let t = lid.x;
    let n_heads = dims.n_heads;
    let n_kv_heads = dims.n_kv_heads;
    let n_q = dims.n_q;
    let n_kv = dims.n_kv;
    let head_dim = dims.head_dim;
    let idx = wid.x;
    if n_heads == 0 || idx >= n_q * n_heads {
        return;
    }
    let qi = idx / n_heads;
    let h = idx % n_heads;
    let kv_den = if n_kv_heads == 0 { 1 } else { n_kv_heads };
    let reps = n_heads / kv_den;
    let rep_den = if reps == 0 { 1 } else { reps };
    let kv_h = h / rep_den;
    let q_off = (qi * n_heads + h) * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let causal_base = if n_kv > n_q { n_kv - n_q } else { 0 };

    let mut local_max = from_bits_neg_inf();
    let mut kv_i = t;
    while kv_i < n_kv {
        let masked = dims.causal != 0 && kv_i > qi + causal_base;
        if !masked {
            let k_off = (kv_i * n_kv_heads + kv_h) * head_dim;
            let mut dot = 0.0f32;
            let mut d = 0u32;
            while d < head_dim {
                dot += q[(q_off + d) as usize] * k[(k_off + d) as usize];
                d += 1;
            }
            let score = dot * scale;
            if score > local_max {
                local_max = score;
            }
        }
        kv_i += ATTN_THREADS;
    }
    scratch[t as usize] = local_max;
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    if t == 0 {
        let mut m = scratch[0];
        let mut s = 1u32;
        while s < ATTN_THREADS {
            if scratch[s as usize] > m {
                m = scratch[s as usize];
            }
            s += 1;
        }
        scratch[0] = m;
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    let gmax = scratch[0];

    let mut local_sum = 0.0f32;
    kv_i = t;
    while kv_i < n_kv {
        let masked = dims.causal != 0 && kv_i > qi + causal_base;
        if !masked {
            let k_off = (kv_i * n_kv_heads + kv_h) * head_dim;
            let mut dot = 0.0f32;
            let mut d = 0u32;
            while d < head_dim {
                dot += q[(q_off + d) as usize] * k[(k_off + d) as usize];
                d += 1;
            }
            local_sum += (dot * scale - gmax).exp();
        }
        kv_i += ATTN_THREADS;
    }
    scratch[t as usize] = local_sum;
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    if t == 0 {
        let mut ssum = 0.0f32;
        let mut s = 0u32;
        while s < ATTN_THREADS {
            ssum += scratch[s as usize];
            s += 1;
        }
        scratch[0] = if ssum > 0.0 { ssum } else { 1.0 };
    }
    spirv_std::arch::workgroup_memory_barrier_with_group_sync();
    let denom = scratch[0];

    // Thread 0 writes the output (head_dim is runtime; shared V-reduction would
    // need a compile-time cap). Scores are recomputed; TinyModel n_kv is small.
    if t == 0 {
        let o_off = (qi * n_heads + h) * head_dim;
        let mut d = 0u32;
        while d < head_dim {
            out[(o_off + d) as usize] = 0.0;
            d += 1;
        }
        let mut kv_t = 0u32;
        while kv_t < n_kv {
            let masked = dims.causal != 0 && kv_t > qi + causal_base;
            if !masked {
                let k_off = (kv_t * n_kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                d = 0;
                while d < head_dim {
                    dot += q[(q_off + d) as usize] * k[(k_off + d) as usize];
                    d += 1;
                }
                let w = (dot * scale - gmax).exp() / denom;
                let v_off = (kv_t * n_kv_heads + kv_h) * head_dim;
                d = 0;
                while d < head_dim {
                    out[(o_off + d) as usize] += w * v[(v_off + d) as usize];
                    d += 1;
                }
            }
            kv_t += 1;
        }
    }
}
