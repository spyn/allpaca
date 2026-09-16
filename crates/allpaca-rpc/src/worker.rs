use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;

use allpaca_cpu::{attention, matmul_nt, rmsnorm, rope, silu};

use crate::protocol::{
    decode_hello_req, decode_launch_list, encode_hello_reply, fnv1a64, read_request, tokens_match,
    write_reply, write_u32, write_u64, write_u8, BufId, Cmd, HelloReply, KernelOp, LaunchList,
    RpcBackendKind, RpcError, DTYPE_F32, KERNEL_ABI_REV, PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH,
    STATUS_ERR, STATUS_OK,
};

struct Slot {
    gen: u32,
    data: Vec<u8>,
}

/// In-process CPU worker: `Vec<u8>` arenas + allpaca-cpu kernels.
pub struct CpuWorker {
    next_id: u32,
    slots: HashMap<u32, Slot>,
    hash_cache: HashMap<u64, Vec<u8>>,
    memory_bytes: u64,
    used: u64,
}

impl CpuWorker {
    pub fn new(memory_bytes: u64) -> Self {
        Self {
            next_id: 1,
            slots: HashMap::new(),
            hash_cache: HashMap::new(),
            memory_bytes,
            used: 0,
        }
    }

    fn hello(&self) -> HelloReply {
        HelloReply {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            kernel_abi_rev: KERNEL_ABI_REV,
            backend: RpcBackendKind::Cpu,
            dtype_mask: DTYPE_F32,
            device_name: "cpu".into(),
            memory_bytes: self.memory_bytes.saturating_sub(self.used),
        }
    }

    fn slot(&self, id: BufId) -> Result<&Slot, RpcError> {
        let s = self
            .slots
            .get(&id.id)
            .ok_or_else(|| RpcError::Worker(format!("unknown buf {}", id.id)))?;
        if s.gen != id.gen {
            return Err(RpcError::Worker(format!(
                "stale buf {} gen {} != {}",
                id.id, id.gen, s.gen
            )));
        }
        Ok(s)
    }

    fn slot_mut(&mut self, id: BufId) -> Result<&mut Slot, RpcError> {
        let s = self
            .slots
            .get_mut(&id.id)
            .ok_or_else(|| RpcError::Worker(format!("unknown buf {}", id.id)))?;
        if s.gen != id.gen {
            return Err(RpcError::Worker(format!(
                "stale buf {} gen {} != {}",
                id.id, id.gen, s.gen
            )));
        }
        Ok(s)
    }

    fn alloc(&mut self, bytes: u64) -> Result<BufId, RpcError> {
        if self.used.saturating_add(bytes) > self.memory_bytes && self.memory_bytes > 0 {
            return Err(RpcError::Worker("out of worker memory".into()));
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let gen = 1;
        self.slots.insert(
            id,
            Slot {
                gen,
                data: vec![0u8; bytes as usize],
            },
        );
        self.used = self.used.saturating_add(bytes);
        Ok(BufId { id, gen })
    }

    fn free(&mut self, id: BufId) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        let n = s.data.len() as u64;
        self.slots.remove(&id.id);
        self.used = self.used.saturating_sub(n);
        Ok(())
    }

    fn clear(&mut self, id: BufId, value: u8) -> Result<(), RpcError> {
        let s = self.slot_mut(id)?;
        s.data.fill(value);
        Ok(())
    }

    fn h2d(&mut self, id: BufId, offset: u64, data: &[u8]) -> Result<(), RpcError> {
        let s = self.slot_mut(id)?;
        let start = offset as usize;
        let end = start
            .checked_add(data.len())
            .ok_or_else(|| RpcError::Worker("h2d overflow".into()))?;
        if end > s.data.len() {
            return Err(RpcError::Worker("h2d out of range".into()));
        }
        s.data[start..end].copy_from_slice(data);
        if data.len() > 10 * 1024 * 1024 {
            self.hash_cache.insert(fnv1a64(data), data.to_vec());
        }
        Ok(())
    }

    fn h2d_hash(&mut self, id: BufId, offset: u64, hash: u64) -> Result<bool, RpcError> {
        if let Some(cached) = self.hash_cache.get(&hash).cloned() {
            self.h2d(id, offset, &cached)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn d2h(&self, id: BufId, offset: u64, size: u64) -> Result<Vec<u8>, RpcError> {
        let s = self.slot(id)?;
        let start = offset as usize;
        let end = start
            .checked_add(size as usize)
            .ok_or_else(|| RpcError::Worker("d2h overflow".into()))?;
        if end > s.data.len() {
            return Err(RpcError::Worker("d2h out of range".into()));
        }
        Ok(s.data[start..end].to_vec())
    }

    fn copy(&mut self, src: BufId, dst: BufId) -> Result<(), RpcError> {
        let src_data = self.slot(src)?.data.clone();
        let dst_slot = self.slot_mut(dst)?;
        let n = src_data.len().min(dst_slot.data.len());
        dst_slot.data[..n].copy_from_slice(&src_data[..n]);
        Ok(())
    }

    fn launch(&mut self, list: &LaunchList) -> Result<(), RpcError> {
        for item in &list.items {
            self.launch_one(item.op, &item.bufs, &item.i32s, &item.f32s)?;
        }
        Ok(())
    }

    fn sync(&self) -> Result<(), RpcError> {
        Ok(())
    }

    fn launch_one(
        &mut self,
        op: KernelOp,
        bufs: &[BufId],
        i32s: &[i32],
        f32s: &[f32],
    ) -> Result<(), RpcError> {
        match op {
            KernelOp::GemmNt => {
                let m = *i32s
                    .first()
                    .ok_or_else(|| RpcError::Worker("gemm m".into()))?
                    as usize;
                let n = *i32s
                    .get(1)
                    .ok_or_else(|| RpcError::Worker("gemm n".into()))?
                    as usize;
                let k = *i32s
                    .get(2)
                    .ok_or_else(|| RpcError::Worker("gemm k".into()))?
                    as usize;
                let a = f32_buf(self.slot(need(bufs, 0)?)?)?;
                let b = f32_buf(self.slot(need(bufs, 1)?)?)?;
                let mut c = f32_buf(self.slot(need(bufs, 2)?)?)?.to_vec();
                matmul_nt(a, b, &mut c, m, n, k).map_err(map_kernel)?;
                write_f32(self.slot_mut(need(bufs, 2)?)?, &c)?;
            }
            KernelOp::Attention => {
                let n_q = *i32s.first().unwrap_or(&1) as usize;
                let n_kv = *i32s
                    .get(1)
                    .ok_or_else(|| RpcError::Worker("attn n_kv".into()))?
                    as usize;
                let n_heads = *i32s
                    .get(2)
                    .ok_or_else(|| RpcError::Worker("attn heads".into()))?
                    as usize;
                let n_kv_heads = *i32s.get(3).unwrap_or(&(n_heads as i32)) as usize;
                let head_dim = *i32s
                    .get(4)
                    .ok_or_else(|| RpcError::Worker("attn dim".into()))?
                    as usize;
                let causal = *i32s.get(5).unwrap_or(&1) != 0;
                let q = f32_buf(self.slot(need(bufs, 0)?)?)?;
                let k = f32_buf(self.slot(need(bufs, 1)?)?)?;
                let v = f32_buf(self.slot(need(bufs, 2)?)?)?;
                let mut out = f32_buf(self.slot(need(bufs, 3)?)?)?.to_vec();
                attention(
                    q, k, v, &mut out, n_q, n_kv, n_heads, n_kv_heads, head_dim, causal,
                )
                .map_err(map_kernel)?;
                write_f32(self.slot_mut(need(bufs, 3)?)?, &out)?;
            }
            KernelOp::RmsNorm => {
                let eps = *f32s.first().unwrap_or(&1e-5);
                let x = f32_buf(self.slot(need(bufs, 0)?)?)?;
                let w = f32_buf(self.slot(need(bufs, 1)?)?)?;
                let mut out = f32_buf(self.slot(need(bufs, 2)?)?)?.to_vec();
                let n = x.len().min(w.len()).min(out.len());
                rmsnorm(&x[..n], &w[..n], eps, &mut out[..n]).map_err(map_kernel)?;
                write_f32(self.slot_mut(need(bufs, 2)?)?, &out)?;
            }
            KernelOp::Rope => {
                let n_heads = *i32s.first().unwrap_or(&1) as usize;
                let head_dim = *i32s
                    .get(1)
                    .ok_or_else(|| RpcError::Worker("rope dim".into()))?
                    as usize;
                let pos = *i32s.get(2).unwrap_or(&0) as usize;
                let theta = *f32s.first().unwrap_or(&10_000.0);
                let mut x = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                for h in 0..n_heads {
                    let start = h * head_dim;
                    let end = start + head_dim;
                    if end > x.len() {
                        return Err(RpcError::Worker("rope range".into()));
                    }
                    rope(&mut x[start..end], pos, theta).map_err(map_kernel)?;
                }
                write_f32(self.slot_mut(need(bufs, 0)?)?, &x)?;
            }
            KernelOp::SiluMul => {
                let mut gate = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let up = f32_buf(self.slot(need(bufs, 1)?)?)?;
                let n = gate.len().min(up.len());
                for i in 0..n {
                    gate[i] = silu(gate[i]) * up[i];
                }
                write_f32(self.slot_mut(need(bufs, 0)?)?, &gate)?;
            }
            KernelOp::ResidualAdd => {
                let mut dst = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let src = f32_buf(self.slot(need(bufs, 1)?)?)?;
                let n = dst.len().min(src.len());
                for i in 0..n {
                    dst[i] += src[i];
                }
                write_f32(self.slot_mut(need(bufs, 0)?)?, &dst)?;
            }
            KernelOp::EmbedGather => {
                let dim = *i32s
                    .first()
                    .ok_or_else(|| RpcError::Worker("embed dim".into()))?
                    as usize;
                let table = f32_buf(self.slot(need(bufs, 0)?)?)?;
                let ids = f32_buf(self.slot(need(bufs, 1)?)?)?;
                // ids stored as f32 bit patterns of u32, or as u32 bytes in the buffer.
                let id_bytes = &self.slot(need(bufs, 1)?)?.data;
                let n_tok = if i32s.len() > 1 {
                    i32s[1] as usize
                } else {
                    id_bytes.len() / 4
                };
                let mut out = vec![0.0f32; n_tok * dim];
                for t in 0..n_tok {
                    let tok = u32::from_le_bytes(
                        id_bytes[t * 4..t * 4 + 4]
                            .try_into()
                            .map_err(|_| RpcError::Worker("embed id".into()))?,
                    ) as usize;
                    let src = tok * dim;
                    if src + dim > table.len() {
                        return Err(RpcError::Worker("embed oob".into()));
                    }
                    out[t * dim..t * dim + dim].copy_from_slice(&table[src..src + dim]);
                }
                let _ = ids;
                write_f32(self.slot_mut(need(bufs, 2)?)?, &out)?;
            }
            KernelOp::Memcpy => {
                let src = self.slot(need(bufs, 1)?)?.data.clone();
                let dst = self.slot_mut(need(bufs, 0)?)?;
                let n = src.len().min(dst.data.len());
                dst.data[..n].copy_from_slice(&src[..n]);
            }
        }
        Ok(())
    }
}

fn need(bufs: &[BufId], i: usize) -> Result<BufId, RpcError> {
    bufs.get(i)
        .copied()
        .ok_or_else(|| RpcError::Worker(format!("missing buf {i}")))
}

fn f32_buf(slot: &Slot) -> Result<&[f32], RpcError> {
    if slot.data.len() % 4 != 0 {
        return Err(RpcError::Worker("buf not f32 aligned".into()));
    }
    let n = slot.data.len() / 4;
    if slot.data.as_ptr() as usize % 4 != 0 {
        return Err(RpcError::Worker("buf misaligned".into()));
    }
    Ok(unsafe { std::slice::from_raw_parts(slot.data.as_ptr().cast::<f32>(), n) })
}

fn write_f32(slot: &mut Slot, src: &[f32]) -> Result<(), RpcError> {
    let bytes = f32_slice_bytes(src);
    let n = bytes.len().min(slot.data.len());
    slot.data[..n].copy_from_slice(&bytes[..n]);
    Ok(())
}

fn f32_slice_bytes(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 4);
    for v in src {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn map_kernel(e: allpaca_types::AllpacaError) -> RpcError {
    RpcError::Worker(e.to_string())
}

/// HIP worker: device arenas + HIP kernel ABI (feature-gated).
#[cfg(feature = "hip")]
pub struct HipWorker {
    hip: allpaca_hip::HipWorkspace,
    device: i32,
    next_id: u32,
    slots: HashMap<u32, HipSlot>,
    hash_cache: HashMap<u64, Vec<u8>>,
    memory_bytes: u64,
    used: u64,
}

#[cfg(feature = "hip")]
struct HipSlot {
    gen: u32,
    ptr: *mut u8,
    bytes: usize,
}

#[cfg(feature = "hip")]
unsafe impl Send for HipWorker {}

#[cfg(feature = "hip")]
impl HipWorker {
    pub fn new(device: i32) -> Result<Self, RpcError> {
        let hip = allpaca_hip::HipWorkspace::new(device, 1 << 20, 64)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        let (free, total) = allpaca_hip::device_memory_info(device).unwrap_or((0, 0));
        let _ = free;
        Ok(Self {
            hip,
            device,
            next_id: 1,
            slots: HashMap::new(),
            hash_cache: HashMap::new(),
            memory_bytes: total,
            used: 0,
        })
    }

    fn hello(&self) -> HelloReply {
        HelloReply {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            kernel_abi_rev: KERNEL_ABI_REV,
            backend: RpcBackendKind::Hip,
            dtype_mask: DTYPE_F32,
            device_name: format!("hip:{}", self.device),
            memory_bytes: self.memory_bytes.saturating_sub(self.used),
        }
    }

    fn slot(&self, id: BufId) -> Result<&HipSlot, RpcError> {
        let s = self
            .slots
            .get(&id.id)
            .ok_or_else(|| RpcError::Worker(format!("unknown buf {}", id.id)))?;
        if s.gen != id.gen {
            return Err(RpcError::Worker("stale hip buf".into()));
        }
        Ok(s)
    }

    fn alloc(&mut self, bytes: u64) -> Result<BufId, RpcError> {
        let ptr = allpaca_hip::device_malloc(bytes as usize, self.device)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.slots.insert(
            id,
            HipSlot {
                gen: 1,
                ptr,
                bytes: bytes as usize,
            },
        );
        self.used = self.used.saturating_add(bytes);
        Ok(BufId { id, gen: 1 })
    }

    fn free(&mut self, id: BufId) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        allpaca_hip::device_free(s.ptr, s.bytes);
        let n = s.bytes as u64;
        self.slots.remove(&id.id);
        self.used = self.used.saturating_sub(n);
        Ok(())
    }

    fn h2d(&mut self, id: BufId, offset: u64, data: &[u8]) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        if offset as usize + data.len() > s.bytes {
            return Err(RpcError::Worker("hip h2d oob".into()));
        }
        let dst = unsafe { s.ptr.add(offset as usize) };
        allpaca_hip::memcpy(dst, data.as_ptr(), data.len(), allpaca_hip::MEMCPY_H2D)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        if data.len() > 10 * 1024 * 1024 {
            self.hash_cache.insert(fnv1a64(data), data.to_vec());
        }
        Ok(())
    }

    fn h2d_hash(&mut self, id: BufId, offset: u64, hash: u64) -> Result<bool, RpcError> {
        if let Some(cached) = self.hash_cache.get(&hash).cloned() {
            self.h2d(id, offset, &cached)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn d2h(&self, id: BufId, offset: u64, size: u64) -> Result<Vec<u8>, RpcError> {
        let s = self.slot(id)?;
        if offset as usize + size as usize > s.bytes {
            return Err(RpcError::Worker("hip d2h oob".into()));
        }
        let mut out = vec![0u8; size as usize];
        let src = unsafe { s.ptr.add(offset as usize) };
        allpaca_hip::memcpy(
            out.as_mut_ptr(),
            src,
            size as usize,
            allpaca_hip::MEMCPY_D2H,
        )
        .map_err(|e| RpcError::Worker(e.to_string()))?;
        Ok(out)
    }

    fn copy(&mut self, src: BufId, dst: BufId) -> Result<(), RpcError> {
        let src_s = self.slot(src)?;
        let dst_s = self.slot(dst)?;
        let n = src_s.bytes.min(dst_s.bytes);
        allpaca_hip::memcpy(dst_s.ptr, src_s.ptr, n, allpaca_hip::MEMCPY_D2D)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        Ok(())
    }

    fn clear(&mut self, id: BufId, value: u8) -> Result<(), RpcError> {
        let zeros = vec![value; self.slot(id)?.bytes];
        self.h2d(id, 0, &zeros)
    }

    fn launch(&mut self, list: &LaunchList) -> Result<(), RpcError> {
        self.hip
            .bind()
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        for item in &list.items {
            self.launch_one(item.op, &item.bufs, &item.i32s, &item.f32s)?;
        }
        Ok(())
    }

    fn sync(&self) -> Result<(), RpcError> {
        allpaca_hip::stream_synchronize().map_err(|e| RpcError::Worker(e.to_string()))
    }

    fn ptr_f32(&self, id: BufId) -> Result<*mut f32, RpcError> {
        Ok(self.slot(id)?.ptr.cast())
    }

    fn launch_one(
        &mut self,
        op: KernelOp,
        bufs: &[BufId],
        i32s: &[i32],
        f32s: &[f32],
    ) -> Result<(), RpcError> {
        match op {
            KernelOp::GemmNt => {
                let m = i32s[0];
                let n = i32s[1];
                let k = i32s[2];
                let a = self.ptr_f32(need(bufs, 0)?)?;
                let b = self.ptr_f32(need(bufs, 1)?)?;
                let c = self.ptr_f32(need(bufs, 2)?)?;
                allpaca_hip::gemm_nt_device(a, b, c, m, n, k)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Attention => {
                let n_q = i32s[0];
                let n_kv = i32s[1];
                let n_heads = i32s[2];
                let n_kv_heads = *i32s.get(3).unwrap_or(&n_heads);
                let head_dim = i32s[4];
                let causal = *i32s.get(5).unwrap_or(&1) != 0;
                self.hip
                    .attention_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        self.ptr_f32(need(bufs, 2)?)?,
                        self.ptr_f32(need(bufs, 3)?)?,
                        n_q,
                        n_kv,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::RmsNorm => {
                let eps = *f32s.first().unwrap_or(&1e-5);
                let x = self.ptr_f32(need(bufs, 0)?)?;
                let w = self.ptr_f32(need(bufs, 1)?)?;
                let out = self.ptr_f32(need(bufs, 2)?)?;
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                self.hip
                    .rmsnorm_device(x, w, out, n, eps)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Rope => {
                let n_heads = i32s[0];
                let head_dim = i32s[1];
                let pos = i32s[2];
                let theta = *f32s.first().unwrap_or(&10_000.0);
                let x = self.ptr_f32(need(bufs, 0)?)?;
                self.hip
                    .rope_device(x, n_heads, head_dim, pos, theta)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::SiluMul => {
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                self.hip
                    .silu_mul_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::ResidualAdd => {
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                self.hip
                    .residual_add(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::EmbedGather | KernelOp::Memcpy => {
                // Fall back to host bounce for these less common ops.
                let src = self.d2h(need(bufs, 1)?, 0, self.slot(need(bufs, 1)?)?.bytes as u64)?;
                let mut cpu = CpuWorker::new(1 << 30);
                let a = cpu.alloc(self.slot(need(bufs, 0)?)?.bytes as u64)?;
                let b = cpu.alloc(src.len() as u64)?;
                cpu.h2d(b, 0, &src)?;
                if op == KernelOp::Memcpy {
                    cpu.launch_one(op, &[need(bufs, 0)?, need(bufs, 1)?], i32s, f32s)?;
                    // remap: copy into a from b after swapping ids — simpler host copy:
                    let tmp = cpu.d2h(b, 0, src.len() as u64)?;
                    self.h2d(need(bufs, 0)?, 0, &tmp)?;
                    let _ = a;
                } else {
                    let table =
                        self.d2h(need(bufs, 0)?, 0, self.slot(need(bufs, 0)?)?.bytes as u64)?;
                    let ids =
                        self.d2h(need(bufs, 1)?, 0, self.slot(need(bufs, 1)?)?.bytes as u64)?;
                    let out_n = self.slot(need(bufs, 2)?)?.bytes as u64;
                    let t = cpu.alloc(table.len() as u64)?;
                    let i = cpu.alloc(ids.len() as u64)?;
                    let o = cpu.alloc(out_n)?;
                    cpu.h2d(t, 0, &table)?;
                    cpu.h2d(i, 0, &ids)?;
                    cpu.launch_one(op, &[t, i, o], i32s, f32s)?;
                    let out = cpu.d2h(o, 0, out_n)?;
                    self.h2d(need(bufs, 2)?, 0, &out)?;
                    let _ = a;
                }
            }
        }
        Ok(())
    }
}

/// CUDA worker: device arenas + CUDA kernel ABI (feature-gated).
///
/// Same `KernelOp` wire layout as [`HipWorker`] so coordinators treat AMD and
/// NVIDIA RPC workers interchangeably after HELLO.
#[cfg(feature = "cuda")]
pub struct CudaWorker {
    cuda: allpaca_cuda::CudaWorkspace,
    device: i32,
    next_id: u32,
    slots: HashMap<u32, CudaSlot>,
    hash_cache: HashMap<u64, Vec<u8>>,
    memory_bytes: u64,
    used: u64,
}

#[cfg(feature = "cuda")]
struct CudaSlot {
    gen: u32,
    ptr: *mut std::ffi::c_void,
    bytes: usize,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaWorker {}

#[cfg(feature = "cuda")]
impl CudaWorker {
    pub fn new(device: i32) -> Result<Self, RpcError> {
        if allpaca_cuda::device_count() <= 0 {
            return Err(RpcError::Worker("no CUDA device".into()));
        }
        let cuda = allpaca_cuda::CudaWorkspace::new(device, 1 << 20)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        let memory_bytes = allpaca_cuda::enumerate_devices()
            .into_iter()
            .find(|d| d.id == device)
            .map(|d| d.memory_bytes)
            .unwrap_or(0);
        Ok(Self {
            cuda,
            device,
            next_id: 1,
            slots: HashMap::new(),
            hash_cache: HashMap::new(),
            memory_bytes,
            used: 0,
        })
    }

    fn hello(&self) -> HelloReply {
        HelloReply {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            kernel_abi_rev: KERNEL_ABI_REV,
            backend: RpcBackendKind::Cuda,
            dtype_mask: DTYPE_F32,
            device_name: match allpaca_cuda::device_sm_arch(self.device) {
                Some((arch, _)) => format!("cuda:{}:{arch}", self.device),
                None => format!("cuda:{}", self.device),
            },
            memory_bytes: self.memory_bytes.saturating_sub(self.used),
        }
    }

    fn slot(&self, id: BufId) -> Result<&CudaSlot, RpcError> {
        let s = self
            .slots
            .get(&id.id)
            .ok_or_else(|| RpcError::Worker(format!("unknown buf {}", id.id)))?;
        if s.gen != id.gen {
            return Err(RpcError::Worker("stale cuda buf".into()));
        }
        Ok(s)
    }

    fn alloc(&mut self, bytes: u64) -> Result<BufId, RpcError> {
        let ptr = allpaca_cuda::device_malloc(bytes as usize, self.device)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.slots.insert(
            id,
            CudaSlot {
                gen: 1,
                ptr,
                bytes: bytes as usize,
            },
        );
        self.used = self.used.saturating_add(bytes);
        Ok(BufId { id, gen: 1 })
    }

    fn free(&mut self, id: BufId) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        allpaca_cuda::device_free(s.ptr);
        let n = s.bytes as u64;
        self.slots.remove(&id.id);
        self.used = self.used.saturating_sub(n);
        Ok(())
    }

    fn h2d(&mut self, id: BufId, offset: u64, data: &[u8]) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        if offset as usize + data.len() > s.bytes {
            return Err(RpcError::Worker("cuda h2d oob".into()));
        }
        let dst = unsafe { s.ptr.cast::<u8>().add(offset as usize) }.cast();
        allpaca_cuda::memcpy_on(
            dst,
            data.as_ptr().cast(),
            data.len(),
            allpaca_cuda::MEMCPY_H2D,
            self.cuda.raw_stream(),
        )
        .map_err(|e| RpcError::Worker(e.to_string()))?;
        if data.len() > 10 * 1024 * 1024 {
            self.hash_cache.insert(fnv1a64(data), data.to_vec());
        }
        Ok(())
    }

    fn h2d_hash(&mut self, id: BufId, offset: u64, hash: u64) -> Result<bool, RpcError> {
        if let Some(cached) = self.hash_cache.get(&hash).cloned() {
            self.h2d(id, offset, &cached)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn d2h(&self, id: BufId, offset: u64, size: u64) -> Result<Vec<u8>, RpcError> {
        let s = self.slot(id)?;
        if offset as usize + size as usize > s.bytes {
            return Err(RpcError::Worker("cuda d2h oob".into()));
        }
        let mut out = vec![0u8; size as usize];
        let src = unsafe { s.ptr.cast::<u8>().add(offset as usize) }.cast();
        allpaca_cuda::memcpy_on(
            out.as_mut_ptr().cast(),
            src,
            size as usize,
            allpaca_cuda::MEMCPY_D2H,
            self.cuda.raw_stream(),
        )
        .map_err(|e| RpcError::Worker(e.to_string()))?;
        self.cuda
            .synchronize()
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        Ok(out)
    }

    fn copy(&mut self, src: BufId, dst: BufId) -> Result<(), RpcError> {
        let src_s = self.slot(src)?;
        let dst_s = self.slot(dst)?;
        let n = src_s.bytes.min(dst_s.bytes);
        allpaca_cuda::memcpy_on(
            dst_s.ptr,
            src_s.ptr,
            n,
            allpaca_cuda::MEMCPY_D2D,
            self.cuda.raw_stream(),
        )
        .map_err(|e| RpcError::Worker(e.to_string()))?;
        Ok(())
    }

    fn sync(&self) -> Result<(), RpcError> {
        self.cuda
            .synchronize()
            .map_err(|e| RpcError::Worker(e.to_string()))
    }

    fn clear(&mut self, id: BufId, value: u8) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        allpaca_cuda::memset_on(s.ptr, value, s.bytes, self.cuda.raw_stream())
            .map_err(|e| RpcError::Worker(e.to_string()))
    }

    fn launch(&mut self, list: &LaunchList) -> Result<(), RpcError> {
        allpaca_cuda::set_device(self.device).map_err(|e| RpcError::Worker(e.to_string()))?;
        for item in &list.items {
            self.launch_one(item.op, &item.bufs, &item.i32s, &item.f32s)?;
        }
        Ok(())
    }

    fn ptr_f32(&self, id: BufId) -> Result<*mut f32, RpcError> {
        Ok(self.slot(id)?.ptr.cast())
    }

    fn launch_one(
        &mut self,
        op: KernelOp,
        bufs: &[BufId],
        i32s: &[i32],
        f32s: &[f32],
    ) -> Result<(), RpcError> {
        match op {
            KernelOp::GemmNt => {
                let m = i32s[0];
                let n = i32s[1];
                let k = i32s[2];
                let a = self.ptr_f32(need(bufs, 0)?)?;
                let b = self.ptr_f32(need(bufs, 1)?)?;
                let c = self.ptr_f32(need(bufs, 2)?)?;
                self.cuda
                    .gemm_nt_dev(a, b, c, m, n, k)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Attention => {
                let n_q = i32s[0];
                let n_kv = i32s[1];
                let n_heads = i32s[2];
                let n_kv_heads = *i32s.get(3).unwrap_or(&n_heads);
                let head_dim = i32s[4];
                let causal = *i32s.get(5).unwrap_or(&1) != 0;
                self.cuda
                    .attention_dev(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        self.ptr_f32(need(bufs, 2)?)?,
                        self.ptr_f32(need(bufs, 3)?)?,
                        n_q,
                        n_kv,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        causal,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::RmsNorm => {
                let eps = *f32s.first().unwrap_or(&1e-5);
                let x = self.ptr_f32(need(bufs, 0)?)?;
                let w = self.ptr_f32(need(bufs, 1)?)?;
                let out = self.ptr_f32(need(bufs, 2)?)?;
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                self.cuda
                    .rmsnorm_dev(x, w, out, n, eps)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Rope => {
                let n_heads = i32s[0];
                let head_dim = i32s[1];
                let pos = i32s[2];
                let theta = *f32s.first().unwrap_or(&10_000.0);
                let x = self.ptr_f32(need(bufs, 0)?)?;
                self.cuda
                    .rope_dev(x, n_heads, head_dim, pos, theta)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::SiluMul => {
                // Wire ABI matches HIP: in-place silu(gate)*up into buf0.
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                let gate = self.ptr_f32(need(bufs, 0)?)?;
                let up = self.ptr_f32(need(bufs, 1)?)?;
                self.cuda
                    .silu_mul_dev(gate, up, gate, n)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::ResidualAdd => {
                // Wire ABI matches HIP: in-place x += residual into buf0.
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                let x = self.ptr_f32(need(bufs, 0)?)?;
                let y = self.ptr_f32(need(bufs, 1)?)?;
                self.cuda
                    .residual_add_dev(x, y, x, n)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Memcpy => {
                self.copy(need(bufs, 1)?, need(bufs, 0)?)?;
            }
            KernelOp::EmbedGather => {
                let dim = *i32s
                    .first()
                    .ok_or_else(|| RpcError::Worker("embed dim".into()))?;
                let n_tok = *i32s.get(1).unwrap_or(&1);
                if n_tok != 1 {
                    return Err(RpcError::Worker(
                        "cuda embed_gather supports n_tok=1".into(),
                    ));
                }
                let table = self.ptr_f32(need(bufs, 0)?)?;
                let ids = self.slot(need(bufs, 1)?)?;
                if ids.bytes < 4 {
                    return Err(RpcError::Worker("embed id".into()));
                }
                let mut tok = 0i32;
                allpaca_cuda::memcpy_on(
                    (&mut tok as *mut i32).cast(),
                    ids.ptr,
                    4,
                    allpaca_cuda::MEMCPY_D2H,
                    self.cuda.raw_stream(),
                )
                .map_err(|e| RpcError::Worker(e.to_string()))?;
                self.cuda
                    .synchronize()
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
                let out = self.ptr_f32(need(bufs, 2)?)?;
                self.cuda
                    .embed_gather_dev(table, tok, out, dim)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
        }
        Ok(())
    }
}

pub enum WorkerBackend {
    Cpu(CpuWorker),
    #[cfg(feature = "hip")]
    Hip(HipWorker),
    #[cfg(feature = "cuda")]
    Cuda(CudaWorker),
}

impl WorkerBackend {
    fn hello(&self) -> HelloReply {
        match self {
            Self::Cpu(w) => w.hello(),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.hello(),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.hello(),
        }
    }

    fn alloc(&mut self, bytes: u64) -> Result<BufId, RpcError> {
        match self {
            Self::Cpu(w) => w.alloc(bytes),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.alloc(bytes),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.alloc(bytes),
        }
    }

    fn free(&mut self, id: BufId) -> Result<(), RpcError> {
        match self {
            Self::Cpu(w) => w.free(id),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.free(id),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.free(id),
        }
    }

    fn clear(&mut self, id: BufId, value: u8) -> Result<(), RpcError> {
        match self {
            Self::Cpu(w) => w.clear(id, value),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.clear(id, value),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.clear(id, value),
        }
    }

    fn h2d(&mut self, id: BufId, offset: u64, data: &[u8]) -> Result<(), RpcError> {
        match self {
            Self::Cpu(w) => w.h2d(id, offset, data),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.h2d(id, offset, data),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.h2d(id, offset, data),
        }
    }

    fn h2d_hash(&mut self, id: BufId, offset: u64, hash: u64) -> Result<bool, RpcError> {
        match self {
            Self::Cpu(w) => w.h2d_hash(id, offset, hash),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.h2d_hash(id, offset, hash),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.h2d_hash(id, offset, hash),
        }
    }

    fn d2h(&self, id: BufId, offset: u64, size: u64) -> Result<Vec<u8>, RpcError> {
        match self {
            Self::Cpu(w) => w.d2h(id, offset, size),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.d2h(id, offset, size),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.d2h(id, offset, size),
        }
    }

    fn copy(&mut self, src: BufId, dst: BufId) -> Result<(), RpcError> {
        match self {
            Self::Cpu(w) => w.copy(src, dst),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.copy(src, dst),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.copy(src, dst),
        }
    }

    fn launch(&mut self, list: &LaunchList) -> Result<(), RpcError> {
        match self {
            Self::Cpu(w) => w.launch(list),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.launch(list),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.launch(list),
        }
    }

    fn sync(&self) -> Result<(), RpcError> {
        match self {
            Self::Cpu(w) => w.sync(),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.sync(),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.sync(),
        }
    }

    fn mem_info(&self) -> (u64, u64) {
        match self {
            Self::Cpu(w) => (w.memory_bytes.saturating_sub(w.used), w.memory_bytes),
            #[cfg(feature = "hip")]
            Self::Hip(w) => {
                allpaca_hip::device_memory_info(w.device).unwrap_or_else(|_| {
                    (w.memory_bytes.saturating_sub(w.used), w.memory_bytes)
                })
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => {
                allpaca_cuda::device_memory_info(w.device).unwrap_or_else(|_| {
                    (w.memory_bytes.saturating_sub(w.used), w.memory_bytes)
                })
            }
        }
    }
}

/// Serve one client on `stream` until disconnect. HELLO must be first.
pub fn serve_connection<S: Read + Write>(
    stream: S,
    backend: WorkerBackend,
) -> Result<(), RpcError> {
    serve_connection_with_token(stream, backend, None)
}

/// Same as [`serve_connection`], checking an optional HELLO shared secret.
pub fn serve_connection_with_token<S: Read + Write>(
    mut stream: S,
    mut backend: WorkerBackend,
    expected_token: Option<&str>,
) -> Result<(), RpcError> {
    let mut saw_hello = false;
    loop {
        let (cmd, payload) = match read_request(&mut stream) {
            Ok(v) => v,
            Err(RpcError::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        if !saw_hello && cmd != Cmd::Hello {
            write_reply(&mut stream, STATUS_ERR, b"HELLO required")?;
            return Err(RpcError::Protocol("first command must be HELLO".into()));
        }
        let result = handle(&mut backend, cmd, &payload, &mut saw_hello, expected_token);
        match result {
            Ok(body) => write_reply(&mut stream, STATUS_OK, &body)?,
            Err(e) => {
                let msg = e.to_string();
                write_reply(&mut stream, STATUS_ERR, msg.as_bytes())?;
                if matches!(e, RpcError::Unauthorized) {
                    return Err(e);
                }
            }
        }
    }
}

/// Accept one client (llama.cpp: one client at a time) on an existing listener.
pub fn serve_listener(
    listener: &std::net::TcpListener,
    backend: WorkerBackend,
) -> Result<(), RpcError> {
    serve_listener_with_token(listener, backend, None)
}

/// Same as [`serve_listener`], checking an optional HELLO shared secret.
pub fn serve_listener_with_token(
    listener: &std::net::TcpListener,
    backend: WorkerBackend,
    expected_token: Option<&str>,
) -> Result<(), RpcError> {
    let (stream, addr) = listener.accept()?;
    tracing::info!(%addr, "rpc client connected");
    stream.set_nodelay(true)?;
    serve_connection_with_token(stream, backend, expected_token)
}

pub fn serve_tcp(stream: TcpStream, backend: WorkerBackend) -> Result<(), RpcError> {
    stream.set_nodelay(true)?;
    serve_connection(stream, backend)
}

fn handle(
    backend: &mut WorkerBackend,
    cmd: Cmd,
    payload: &[u8],
    saw_hello: &mut bool,
    expected_token: Option<&str>,
) -> Result<Vec<u8>, RpcError> {
    use crate::protocol::Cursor;
    match cmd {
        Cmd::Hello => {
            let req = decode_hello_req(payload)?;
            if req.major != PROTO_MAJOR {
                return Err(RpcError::Version {
                    major: req.major,
                    minor: req.minor,
                    patch: req.patch,
                });
            }
            if let Some(expected) = expected_token.filter(|t| !t.is_empty()) {
                if !tokens_match(expected, &req.token) {
                    return Err(RpcError::Unauthorized);
                }
            }
            *saw_hello = true;
            Ok(encode_hello_reply(&backend.hello()))
        }
        Cmd::Alloc => {
            let mut c = Cursor::new(payload);
            let bytes = c.u64()?;
            let id = backend.alloc(bytes)?;
            let mut out = Vec::new();
            crate::protocol::write_buf_id(&mut out, id);
            Ok(out)
        }
        Cmd::Free => {
            let mut c = Cursor::new(payload);
            backend.free(c.buf_id()?)?;
            Ok(Vec::new())
        }
        Cmd::Clear => {
            let mut c = Cursor::new(payload);
            let id = c.buf_id()?;
            let value = c.u8()?;
            backend.clear(id, value)?;
            Ok(Vec::new())
        }
        Cmd::H2D => {
            let mut c = Cursor::new(payload);
            let id = c.buf_id()?;
            let offset = c.u64()?;
            let data = c.bytes()?;
            backend.h2d(id, offset, data)?;
            Ok(Vec::new())
        }
        Cmd::H2DHash => {
            let mut c = Cursor::new(payload);
            let id = c.buf_id()?;
            let offset = c.u64()?;
            let hash = c.u64()?;
            let hit = backend.h2d_hash(id, offset, hash)?;
            let mut out = Vec::new();
            write_u8(&mut out, u8::from(hit));
            Ok(out)
        }
        Cmd::D2H => {
            let mut c = Cursor::new(payload);
            let id = c.buf_id()?;
            let offset = c.u64()?;
            let size = c.u64()?;
            backend.d2h(id, offset, size)
        }
        Cmd::Copy => {
            let mut c = Cursor::new(payload);
            let src = c.buf_id()?;
            let dst = c.buf_id()?;
            backend.copy(src, dst)?;
            Ok(Vec::new())
        }
        Cmd::LaunchList => {
            let list = decode_launch_list(payload)?;
            backend.launch(&list)?;
            Ok(Vec::new())
        }
        Cmd::MemInfo => {
            let (free, total) = backend.mem_info();
            let mut out = Vec::new();
            write_u64(&mut out, free);
            write_u64(&mut out, total);
            Ok(out)
        }
        Cmd::DeviceCount => {
            let mut out = Vec::new();
            write_u32(&mut out, 1);
            Ok(out)
        }
        Cmd::Sync => {
            backend.sync()?;
            Ok(Vec::new())
        }
    }
}
