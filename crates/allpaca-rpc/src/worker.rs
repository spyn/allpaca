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
    /// ggml type id of a `PutTensor` blob. `0` is raw F32.
    ggml_type: u32,
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
            devices: Vec::new(),
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
                ggml_type: 0,
                data: vec![0u8; bytes as usize],
            },
        );
        self.used = self.used.saturating_add(bytes);
        Ok(BufId { id, gen })
    }

    fn set_ggml_type(&mut self, id: BufId, ggml_type: u32) -> Result<(), RpcError> {
        self.slot_mut(id)?.ggml_type = ggml_type;
        Ok(())
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
            KernelOp::GemmNtQuant => {
                let m = *i32s.first().ok_or_else(|| RpcError::Worker("gemm m".into()))? as usize;
                let n = *i32s.get(1).ok_or_else(|| RpcError::Worker("gemm n".into()))? as usize;
                let k = *i32s.get(2).ok_or_else(|| RpcError::Worker("gemm k".into()))? as usize;
                let a = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let ty = self.slot(need(bufs, 1)?)?.ggml_type;
                let mut w = vec![0.0f32; n.saturating_mul(k)];
                dequant_weight(ty, &self.slot(need(bufs, 1)?)?.data, n * k, &mut w)?;
                let mut y = vec![0.0f32; m.saturating_mul(n)];
                matmul_nt(&a, &w, &mut y, m, n, k).map_err(map_kernel)?;
                write_f32(self.slot_mut(need(bufs, 2)?)?, &y)?;
            }
            KernelOp::DeinterleaveQGate => {
                let n_heads = *i32s.first().ok_or_else(|| RpcError::Worker("deinterleave heads".into()))? as usize;
                let head_dim = *i32s.get(1).ok_or_else(|| RpcError::Worker("deinterleave head".into()))? as usize;
                let full = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let mut q = vec![0.0f32; n_heads * head_dim];
                let mut gate = vec![0.0f32; n_heads * head_dim];
                deinterleave_q_gate(&full, &mut q, &mut gate, head_dim)?;
                write_f32(self.slot_mut(need(bufs, 1)?)?, &q)?;
                write_f32(self.slot_mut(need(bufs, 2)?)?, &gate)?;
            }
            KernelOp::RmsNormHeads => {
                let n_heads = *i32s.first().ok_or_else(|| RpcError::Worker("rmsnorm heads".into()))? as usize;
                let head_dim = *i32s.get(1).ok_or_else(|| RpcError::Worker("rmsnorm head".into()))? as usize;
                let eps = *f32s.first().unwrap_or(&1e-5);
                let mut x = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let w = f32_buf(self.slot(need(bufs, 1)?)?)?.to_vec();
                rmsnorm_heads(&mut x, &w, n_heads, head_dim, eps)?;
                write_f32(self.slot_mut(need(bufs, 0)?)?, &x)?;
            }
            KernelOp::SigmoidMul => {
                let n = *i32s.first().ok_or_else(|| RpcError::Worker("sigmoid n".into()))? as usize;
                let mut x = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let gate = f32_buf(self.slot(need(bufs, 1)?)?)?;
                let n = n.min(x.len()).min(gate.len());
                for i in 0..n {
                    let s = 1.0 / (1.0 + (-gate[i]).exp());
                    x[i] *= s;
                }
                write_f32(self.slot_mut(need(bufs, 0)?)?, &x)?;
            }
            KernelOp::RopePartial => {
                let n_heads = *i32s.first().ok_or_else(|| RpcError::Worker("rope heads".into()))? as usize;
                let n_kv = *i32s.get(1).ok_or_else(|| RpcError::Worker("rope kv heads".into()))? as usize;
                let head_dim = *i32s.get(2).ok_or_else(|| RpcError::Worker("rope head".into()))? as usize;
                let pos = *i32s.get(3).ok_or_else(|| RpcError::Worker("rope pos".into()))? as usize;
                let n_rot = *i32s.get(4).unwrap_or(&(head_dim as i32)) as usize;
                let theta = *f32s.first().unwrap_or(&10_000.0);
                let sections = [
                    *f32s.get(1).unwrap_or(&0.0) as i32,
                    *f32s.get(2).unwrap_or(&0.0) as i32,
                    *f32s.get(3).unwrap_or(&0.0) as i32,
                    *f32s.get(4).unwrap_or(&0.0) as i32,
                ];
                let mut q = f32_buf(self.slot(need(bufs, 0)?)?)?.to_vec();
                let mut k = f32_buf(self.slot(need(bufs, 1)?)?)?.to_vec();
                allpaca_cpu::rope_imrope_neox_heads(&mut q, n_heads, head_dim, pos, theta, n_rot, sections)
                    .map_err(map_kernel)?;
                allpaca_cpu::rope_imrope_neox_heads(&mut k, n_kv, head_dim, pos, theta, n_rot, sections)
                    .map_err(map_kernel)?;
                write_f32(self.slot_mut(need(bufs, 0)?)?, &q)?;
                write_f32(self.slot_mut(need(bufs, 1)?)?, &k)?;
            }
            KernelOp::Memcpy => {
                // i32s empty → full buffer copy; [dst_off, nbytes] → partial into dst.
                let src = self.slot(need(bufs, 1)?)?.data.clone();
                let dst = self.slot_mut(need(bufs, 0)?)?;
                if i32s.len() >= 2 {
                    let dst_off = i32s[0] as usize;
                    let nbytes = i32s[1] as usize;
                    if dst_off.saturating_add(nbytes) > dst.data.len() || nbytes > src.len() {
                        return Err(RpcError::Worker("memcpy offset/oob".into()));
                    }
                    dst.data[dst_off..dst_off + nbytes].copy_from_slice(&src[..nbytes]);
                } else {
                    let n = src.len().min(dst.data.len());
                    dst.data[..n].copy_from_slice(&src[..n]);
                }
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

fn deinterleave_q_gate(q_full: &[f32], q: &mut [f32], gate: &mut [f32], head_dim: usize) -> Result<(), RpcError> {
    if head_dim == 0 || q.len() != gate.len() || q_full.len() != q.len().saturating_mul(2) || q.len() % head_dim != 0 {
        return Err(RpcError::Worker("deinterleave shape".into()));
    }
    let n_heads = q.len() / head_dim;
    for h in 0..n_heads {
        let src = h * 2 * head_dim;
        let dst = h * head_dim;
        q[dst..dst + head_dim].copy_from_slice(&q_full[src..src + head_dim]);
        gate[dst..dst + head_dim].copy_from_slice(&q_full[src + head_dim..src + 2 * head_dim]);
    }
    Ok(())
}

fn rmsnorm_heads(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) -> Result<(), RpcError> {
    if head_dim == 0 || weight.len() != head_dim || x.len() != n_heads.saturating_mul(head_dim) {
        return Err(RpcError::Worker("rmsnorm heads shape".into()));
    }
    for h in 0..n_heads {
        let row = &mut x[h * head_dim..(h + 1) * head_dim];
        let mut ss = 0.0f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            row[i] *= inv * weight[i];
        }
    }
    Ok(())
}

/// Expand a packed GGUF weight to F32 for the CPU worker. HIP uses device GEMM instead.
/// `(row, weight)` lists per expert, experts ascending.
fn moe_groups(pairs: &[(u32, u32, f32)]) -> std::collections::BTreeMap<u32, Vec<(usize, f32)>> {
    let mut groups: std::collections::BTreeMap<u32, Vec<(usize, f32)>> = Default::default();
    for &(row, expert, weight) in pairs {
        groups.entry(expert).or_default().push((row as usize, weight));
    }
    groups
}

impl CpuWorker {
    fn moe_ffn(
        &mut self,
        req: &crate::protocol::MoeFfnRequest,
        x: &[f32],
        pairs: &[(u32, u32, f32)],
    ) -> Result<Vec<f32>, RpcError> {
        let (dim, ff, n_exp) = (req.dim as usize, req.ff as usize, req.n_experts as usize);
        let mut out = vec![0.0f32; x.len()];
        let expert = |slot: &Slot, e: usize, n: usize| -> Result<Vec<f32>, RpcError> {
            let stride = slot.data.len() / n_exp;
            let mut w = vec![0.0f32; n];
            dequant_weight(slot.ggml_type, &slot.data[e * stride..(e + 1) * stride], n, &mut w)?;
            Ok(w)
        };
        for (e, rows) in moe_groups(pairs) {
            let e = e as usize;
            let wg = expert(self.slot(req.gate)?, e, ff * dim)?;
            let wu = expert(self.slot(req.up)?, e, ff * dim)?;
            let wd = expert(self.slot(req.down)?, e, dim * ff)?;
            for (row, weight) in rows {
                let xr = &x[row * dim..(row + 1) * dim];
                let (mut g, mut u, mut o) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; dim]);
                matmul_nt(xr, &wg, &mut g, 1, ff, dim).map_err(map_kernel)?;
                matmul_nt(xr, &wu, &mut u, 1, ff, dim).map_err(map_kernel)?;
                let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
                matmul_nt(&h, &wd, &mut o, 1, dim, ff).map_err(map_kernel)?;
                for (acc, v) in out[row * dim..(row + 1) * dim].iter_mut().zip(&o) {
                    *acc += weight * v;
                }
            }
        }
        Ok(out)
    }
}

fn dequant_weight(ty: u32, bytes: &[u8], n: usize, out: &mut [f32]) -> Result<(), RpcError> {
    if out.len() < n {
        return Err(RpcError::Worker("dequant output too small".into()));
    }
    let bad = |e: allpaca_types::AllpacaError| RpcError::Worker(e.to_string());
    match ty {
        0 => {
            if bytes.len() < n.saturating_mul(4) {
                return Err(RpcError::Worker("f32 weight short".into()));
            }
            for i in 0..n {
                let b: [u8; 4] = bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .map_err(|_| RpcError::Worker("f32 weight".into()))?;
                out[i] = f32::from_le_bytes(b);
            }
            Ok(())
        }
        1 | 30 => {
            if bytes.len() < n.saturating_mul(2) {
                return Err(RpcError::Worker("f16 weight short".into()));
            }
            for i in 0..n {
                let b: [u8; 2] = bytes[i * 2..i * 2 + 2]
                    .try_into()
                    .map_err(|_| RpcError::Worker("f16 weight".into()))?;
                let h = u16::from_le_bytes(b);
                out[i] = if ty == 1 {
                    allpaca_cpu::f16_to_f32(h)
                } else {
                    allpaca_cpu::bf16_to_f32(h)
                };
            }
            Ok(())
        }
        2 => allpaca_cpu::dequant_q4_0(bytes, n, out).map_err(bad),
        3 => allpaca_cpu::dequant_q4_1(bytes, n, out).map_err(bad),
        6 => allpaca_cpu::dequant_q5_0(bytes, n, out).map_err(bad),
        7 => allpaca_cpu::dequant_q5_1(bytes, n, out).map_err(bad),
        8 => allpaca_cpu::dequant_q8_0(bytes, n, out).map_err(bad),
        9 => allpaca_cpu::dequant_q8_1(bytes, n, out).map_err(bad),
        10 => allpaca_cpu::dequant_q2_k(bytes, n, out).map_err(bad),
        11 => allpaca_cpu::dequant_q3_k(bytes, n, out).map_err(bad),
        12 => allpaca_cpu::dequant_q4_k(bytes, n, out).map_err(bad),
        13 => allpaca_cpu::dequant_q5_k(bytes, n, out).map_err(bad),
        14 => allpaca_cpu::dequant_q6_k(bytes, n, out).map_err(bad),
        20 => allpaca_cpu::dequant_iq4_nl(bytes, n, out).map_err(bad),
        23 => allpaca_cpu::dequant_iq4_xs(bytes, n, out).map_err(bad),
        34 => allpaca_cpu::dequant_tq1_0(bytes, n, out).map_err(bad),
        35 => allpaca_cpu::dequant_tq2_0(bytes, n, out).map_err(bad),
        39 => allpaca_cpu::dequant_mxfp4(bytes, n, out).map_err(bad),
        100 => allpaca_cpu::dequant_rocmfp4(bytes, n, out).map_err(bad),
        101 => allpaca_cpu::dequant_rocmfp4_fast(bytes, n, out).map_err(bad),
        other => Err(RpcError::Worker(format!(
            "ggml type {other} has no cpu dequant"
        ))),
    }
}

/// One HIP device inside a pooled worker.
#[cfg(feature = "hip")]
struct HipGpu {
    device: i32,
    hip: allpaca_hip::HipWorkspace,
    total: u64,
    used: u64,
    /// Vega 20. Decode GEMV uses the vectorized kernel, not the scalar tile.
    gfx906: bool,
    /// Grow-only scratch for `MoeFfn`.
    moe_scratch: Option<allpaca_hip::DeviceWeights>,
}

/// HIP worker: every visible device, or one pinned ordinal.
#[cfg(feature = "hip")]
pub struct HipWorker {
    gpus: Vec<HipGpu>,
    next_id: u32,
    slots: HashMap<u32, HipSlot>,
    hash_cache: HashMap<u64, Vec<u8>>,
    /// Device used for allocs that do not name one, so a layer stays together.
    sticky: usize,
}

/// A session's buffers die with its backend: a coordinator that reconnects
/// would otherwise find the GPUs still full from its last run.
#[cfg(feature = "hip")]
impl Drop for HipWorker {
    fn drop(&mut self) {
        for (_, s) in self.slots.drain() {
            if let Some(g) = self.gpus.get(s.gpu) {
                let _ = g.hip.bind();
            }
            allpaca_hip::device_free(s.ptr, s.bytes);
        }
    }
}

#[cfg(feature = "hip")]
struct HipSlot {
    gen: u32,
    /// ggml type id of a `PutTensor` blob. `0` is raw F32.
    ggml_type: u32,
    ptr: *mut u8,
    bytes: usize,
    gpu: usize,
}

#[cfg(feature = "hip")]
impl HipWorker {
    fn moe_ffn(
        &mut self,
        req: &crate::protocol::MoeFfnRequest,
        x_in: &crate::protocol::MoeX,
        pairs: &[(u32, u32, f32)],
    ) -> Result<Vec<f32>, RpcError> {
        let bad = |e: allpaca_types::AllpacaError| RpcError::Worker(e.to_string());
        let (dim, ff, n_exp) = (req.dim as usize, req.ff as usize, req.n_experts as usize);
        let rows = x_in.len() / dim.max(1);
        let (g, u, d) = (self.slot(req.gate)?, self.slot(req.up)?, self.slot(req.down)?);
        if g.gpu != u.gpu || g.gpu != d.gpu {
            return Err(RpcError::Worker("moe ffn experts span GPUs".into()));
        }
        let gpu = g.gpu;
        let (gs, us, ds) = (g.bytes / n_exp, u.bytes / n_exp, d.bytes / n_exp);
        let (gp, up, dp) = (g.ptr, u.ptr, d.ptr);
        let (gt, ut, dt) = (g.ggml_type, u.ggml_type, d.ggml_type);
        let experts = |ptr: *mut u8, ggml_type: u32, stride: usize| allpaca_hip::MoeExperts {
            ptr,
            ggml_type,
            stride: stride as u64,
        };
        let fused = [experts(gp, gt, gs), experts(up, ut, us), experts(dp, dt, ds)];
        if dim % 256 == 0 && ff % 256 == 0 && fused.iter().all(allpaca_hip::MoeExperts::fused_ok) {
            return self.moe_ffn_fused(gpu, dim, ff, x_in, pairs, fused);
        }
        let x_f32 = x_in.to_f32();
        let x = x_f32.as_slice();
        // Scratch: x, gathered rows, gate, up, expert out, accumulator.
        let floats = 4 * rows * dim + 2 * rows * ff;
        let scratch = self.moe_scratch(gpu, floats * 4)?;
        let f = |i: usize| unsafe { scratch.add(i) };
        let (xs, xe, out_e, acc) = (f(0), f(rows * dim), f(2 * rows * dim), f(3 * rows * dim));
        let (gate, upb) = (f(4 * rows * dim), f(4 * rows * dim + rows * ff));
        let hip = &self.gpus[gpu].hip;
        hip.copy_h2d_f32(xs, x).map_err(bad)?;
        hip.copy_h2d_f32(acc, &vec![0.0f32; rows * dim]).map_err(bad)?;
        for (e, group) in moe_groups(pairs) {
            let e = e as usize;
            for (i, &(row, _)) in group.iter().enumerate() {
                hip.copy_d2d_f32(unsafe { xe.add(i * dim) }, unsafe { xs.add(row * dim) }, dim)
                    .map_err(bad)?;
            }
            let m = group.len() as i32;
            let (dimi, ffi) = (dim as i32, ff as i32);
            hip_quant_gemm(gt, xe, unsafe { gp.add(e * gs) }, gate, m, ffi, dimi)?;
            hip_quant_gemm(ut, xe, unsafe { up.add(e * us) }, upb, m, ffi, dimi)?;
            hip.silu_mul_device(gate, upb, m * ffi).map_err(bad)?;
            hip_quant_gemm(dt, gate, unsafe { dp.add(e * ds) }, out_e, m, dimi, ffi)?;
            for (i, &(row, weight)) in group.iter().enumerate() {
                let src = unsafe { out_e.add(i * dim) };
                hip.scale_f32(src, dim, weight).map_err(bad)?;
                hip.residual_add(unsafe { acc.add(row * dim) }, src, dimi).map_err(bad)?;
            }
        }
        let mut out = vec![0.0f32; rows * dim];
        hip.copy_d2h_f32(acc, &mut out).map_err(bad)?;
        Ok(out)
    }

    /// K-quant experts: one H2D of `x` plus the row-sorted pair table, the
    /// seven-launch fused MoE kernels, one D2H.
    fn moe_ffn_fused(
        &mut self,
        gpu: usize,
        dim: usize,
        ff: usize,
        x: &crate::protocol::MoeX,
        pairs: &[(u32, u32, f32)],
        [gate, up, down]: [allpaca_hip::MoeExperts; 3],
    ) -> Result<Vec<f32>, RpcError> {
        let bad = |e: allpaca_types::AllpacaError| RpcError::Worker(e.to_string());
        let rows = x.len() / dim.max(1);
        // One upload: [x | pair table], then output and kernel scratch. Q8
        // rows go as [int8 quants (4 per float) | scales].
        let (prefix, x_q8) = match x {
            crate::protocol::MoeX::F32(v) => (v.clone(), None),
            crate::protocol::MoeX::Q8 { q, d } => {
                let mut v: Vec<f32> = q
                    .chunks_exact(4)
                    .map(|b| f32::from_bits(u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8])))
                    .collect();
                v.resize(v.len().next_multiple_of(64), 0.0);
                let d_off = v.len();
                v.extend_from_slice(d);
                (v, Some(d_off))
            }
        };
        let table = allpaca_hip::MoePairTable::new(rows, pairs, &prefix);
        let out_off = table.end;
        let work_off = (out_off + rows * dim).next_multiple_of(64);
        let bytes = work_off * 4 + allpaca_hip::moe_ffn_scratch_bytes(rows, dim, ff, table.pairs);
        let base = self.moe_scratch(gpu, bytes)?;
        let at = |i: usize| unsafe { base.add(i) };
        let hip = &self.gpus[gpu].hip;
        hip.copy_h2d_f32(base, &table.host).map_err(bad)?;
        let input = match x_q8 {
            None => allpaca_hip::MoeInput::F32(base),
            Some(d_off) => allpaca_hip::MoeInput::Q8 { q: base.cast(), d: at(d_off) },
        };
        table
            .run(base, input, rows, dim, ff, [gate, up, down], at(work_off).cast(), at(out_off))
            .map_err(bad)?;
        let mut out = vec![0.0f32; rows * dim];
        hip.copy_d2h_f32(at(out_off), &mut out).map_err(bad)?;
        Ok(out)
    }

    /// Grow-only per-GPU scratch for [`Self::moe_ffn`].
    fn moe_scratch(&mut self, gpu: usize, bytes: usize) -> Result<*mut f32, RpcError> {
        let g = &mut self.gpus[gpu];
        if g.moe_scratch.as_ref().is_none_or(|s| s.bytes() < bytes as u64) {
            g.moe_scratch = None;
            let buf = allpaca_hip::DeviceWeights::alloc(g.device, bytes.max(1 << 20))
                .map_err(|e| RpcError::Worker(e.to_string()))?;
            g.moe_scratch = Some(buf);
        }
        Ok(g.moe_scratch.as_ref().expect("moe scratch").ptr_at(0) as *mut f32)
    }
}

#[cfg(feature = "hip")]
fn hip_quant_gemm(
    ty: u32,
    a: *const f32,
    b: *mut u8,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
) -> Result<(), RpcError> {
    let bad = |e: allpaca_types::AllpacaError| RpcError::Worker(e.to_string());
    match ty {
        0 => allpaca_hip::gemm_nt_device(a, b.cast(), c, m, n, k).map_err(bad),
        1 => allpaca_hip::gemm_nt_f16_device(a, b.cast(), c, m, n, k).map_err(bad),
        8 => allpaca_hip::gemm_nt_q8_0_device(a, b, c, m, n, k).map_err(bad),
        12 => allpaca_hip::gemm_nt_q4_k_device(a, b, c, m, n, k).map_err(bad),
        13 => allpaca_hip::gemm_nt_q5_k_device(a, b, c, m, n, k).map_err(bad),
        14 => allpaca_hip::gemm_nt_q6_k_device(a, b, c, m, n, k).map_err(bad),
        30 => allpaca_hip::gemm_nt_bf16_device(a, b.cast(), c, m, n, k).map_err(bad),
        34 => allpaca_hip::gemm_nt_tq1_0_device(a, b, c, m, n, k).map_err(bad),
        35 => allpaca_hip::gemm_nt_tq2_0_device(a, b, c, m, n, k).map_err(bad),
        39 => allpaca_hip::gemm_nt_mxfp4_device(a, b, c, m, n, k).map_err(bad),
        100 => allpaca_hip::gemm_nt_rocmfp4_device(a, b, c, m, n, k).map_err(bad),
        101 => allpaca_hip::gemm_nt_rocmfp4_fast_device(a, b, c, m, n, k).map_err(bad),
        other => Err(RpcError::Worker(format!(
            "hip worker has no gemm for ggml type {other}"
        ))),
    }
}

/// Pick a device for `bytes`. A hint must fit. Otherwise stay on `sticky` until it
/// cannot, then the device with the most free space.
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
pub(crate) fn place_sticky(
    free: &[u64],
    sticky: usize,
    bytes: u64,
    hint: Option<usize>,
) -> Result<usize, &'static str> {
    if free.is_empty() {
        return Err("rpc worker has no device");
    }
    if let Some(h) = hint {
        if h >= free.len() {
            return Err("rpc device index");
        }
        if free[h] < bytes {
            return Err("rpc device full");
        }
        return Ok(h);
    }
    let sticky = sticky.min(free.len() - 1);
    if free[sticky] >= bytes {
        return Ok(sticky);
    }
    let mut best: Option<(usize, u64)> = None;
    for (i, &room) in free.iter().enumerate() {
        if room >= bytes && best.map(|(_, b)| room > b).unwrap_or(true) {
            best = Some((i, room));
        }
    }
    best.map(|(i, _)| i).ok_or("rpc worker out of memory")
}

/// True for Vega 20. Its ROCmFP4 GEMV is one thread per column; F32 GEMV is vectorized.
pub(crate) fn gfx906_expands_rocmfp4(arch: &str) -> bool {
    arch.split(':').next() == Some("gfx906")
}

/// Whole-block ROCmFP4 bytes (`fast`: 17-byte type-101 blocks, else 18-byte
/// type 100) to little-endian F32. `None` is not returned; a truncated blob
/// is an error.
pub(crate) fn expand_rocmfp4_f32(bytes: &[u8], fast: bool) -> Result<Vec<u8>, RpcError> {
    let block_bytes = if fast {
        allpaca_cpu::ROCMFP4_FAST_BLOCK_BYTES
    } else {
        allpaca_cpu::ROCMFP4_BLOCK_BYTES
    };
    if bytes.is_empty() || bytes.len() % block_bytes != 0 {
        return Err(RpcError::Worker("rocmfp4 blob is not whole blocks".into()));
    }
    let n = bytes.len() / block_bytes * allpaca_cpu::ROCMFP4_BLOCK_ELEMS;
    let mut weights = vec![0.0f32; n];
    if fast {
        allpaca_cpu::dequant_rocmfp4_fast(bytes, n, &mut weights)
    } else {
        allpaca_cpu::dequant_rocmfp4(bytes, n, &mut weights)
    }
    .map_err(|e| RpcError::Worker(e.to_string()))?;
    let mut out = Vec::with_capacity(n * 4);
    for v in weights {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

#[cfg(feature = "hip")]
unsafe impl Send for HipWorker {}

#[cfg(feature = "hip")]
impl HipGpu {
    fn open(device: i32) -> Result<Self, RpcError> {
        let hip = allpaca_hip::HipWorkspace::new(device, 1 << 20, 64)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        let (_free, total) = allpaca_hip::device_memory_info(device).unwrap_or((0, 0));
        let gfx906 = allpaca_hip::device_gcn_arch(device)
            .is_some_and(|(arch, _)| gfx906_expands_rocmfp4(&arch));
        Ok(Self {
            device,
            hip,
            total,
            used: 0,
            gfx906,
            moe_scratch: None,
        })
    }

    fn free_bytes(&self) -> u64 {
        self.total.saturating_sub(self.used)
    }
}

#[cfg(feature = "hip")]
impl HipWorker {
    pub fn new(device: i32) -> Result<Self, RpcError> {
        Self::from_gpus(vec![HipGpu::open(device)?])
    }

    /// One worker, every visible HIP device.
    pub fn open_all() -> Result<Self, RpcError> {
        let n = allpaca_hip::device_count();
        if n <= 0 {
            return Err(RpcError::Worker("no HIP device".into()));
        }
        let mut gpus = Vec::with_capacity(n as usize);
        for id in 0..n {
            gpus.push(HipGpu::open(id)?);
        }
        Self::from_gpus(gpus)
    }

    fn from_gpus(gpus: Vec<HipGpu>) -> Result<Self, RpcError> {
        if gpus.is_empty() {
            return Err(RpcError::Worker("no HIP device".into()));
        }
        Ok(Self {
            gpus,
            next_id: 1,
            slots: HashMap::new(),
            hash_cache: HashMap::new(),
            sticky: 0,
        })
    }

    fn hello(&self) -> HelloReply {
        let devices: Vec<u64> = self.gpus.iter().map(HipGpu::free_bytes).collect();
        let memory_bytes = devices.iter().copied().sum();
        let device_name = self
            .gpus
            .iter()
            .map(|g| format!("hip:{}", g.device))
            .collect::<Vec<_>>()
            .join("+");
        HelloReply {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            kernel_abi_rev: KERNEL_ABI_REV,
            backend: RpcBackendKind::Hip,
            dtype_mask: DTYPE_F32,
            device_name,
            memory_bytes,
            devices,
        }
    }

    fn mem_info(&self) -> (u64, u64) {
        let free = self.gpus.iter().map(HipGpu::free_bytes).sum();
        let total = self.gpus.iter().map(|g| g.total).sum();
        (free, total)
    }

    fn device_count(&self) -> u32 {
        self.gpus.len() as u32
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

    fn alloc(&mut self, bytes: u64, hint: Option<usize>) -> Result<BufId, RpcError> {
        let free: Vec<u64> = self.gpus.iter().map(HipGpu::free_bytes).collect();
        let gpu = place_sticky(&free, self.sticky, bytes, hint)
            .map_err(|e| RpcError::Worker(e.into()))?;
        let device = self.gpus[gpu].device;
        let ptr = allpaca_hip::device_malloc(bytes as usize, device)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.slots.insert(
            id,
            HipSlot {
                gen: 1,
                ggml_type: 0,
                ptr,
                bytes: bytes as usize,
                gpu,
            },
        );
        self.gpus[gpu].used = self.gpus[gpu].used.saturating_add(bytes);
        self.sticky = gpu;
        Ok(BufId { id, gen: 1 })
    }

    fn set_ggml_type(&mut self, id: BufId, ggml_type: u32) -> Result<(), RpcError> {
        let s = self
            .slots
            .get_mut(&id.id)
            .ok_or_else(|| RpcError::Worker(format!("unknown buf {}", id.id)))?;
        if s.gen != id.gen {
            return Err(RpcError::Worker("stale hip buf".into()));
        }
        s.ggml_type = ggml_type;
        Ok(())
    }

    fn bind_gpu(&self, gpu: usize) -> Result<(), RpcError> {
        self.gpus
            .get(gpu)
            .ok_or_else(|| RpcError::Worker("rpc device index".into()))?
            .hip
            .bind()
            .map_err(|e| RpcError::Worker(e.to_string()))
    }

    fn free(&mut self, id: BufId) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        let gpu = s.gpu;
        let ptr = s.ptr;
        let bytes = s.bytes;
        self.bind_gpu(gpu)?;
        allpaca_hip::device_free(ptr, bytes);
        self.slots.remove(&id.id);
        self.gpus[gpu].used = self.gpus[gpu].used.saturating_sub(bytes as u64);
        Ok(())
    }

    fn h2d(&mut self, id: BufId, offset: u64, data: &[u8]) -> Result<(), RpcError> {
        let gpu = self.slot(id)?.gpu;
        self.bind_gpu(gpu)?;
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
        let gpu = self.slot(id)?.gpu;
        self.bind_gpu(gpu)?;
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
        if src_s.gpu != dst_s.gpu {
            return Err(RpcError::Worker(
                "rpc copy across HIP devices is not supported".into(),
            ));
        }
        let gpu = src_s.gpu;
        let n = src_s.bytes.min(dst_s.bytes);
        let dst_ptr = dst_s.ptr;
        let src_ptr = src_s.ptr;
        self.bind_gpu(gpu)?;
        allpaca_hip::memcpy(dst_ptr, src_ptr, n, allpaca_hip::MEMCPY_D2D)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        Ok(())
    }

    /// Device-resident partial copy: `src[0..nbytes]` → `dst[dst_off..]`.
    fn copy_at(
        &mut self,
        src: BufId,
        dst: BufId,
        dst_off: u64,
        nbytes: u64,
    ) -> Result<(), RpcError> {
        let src_s = self.slot(src)?;
        let dst_s = self.slot(dst)?;
        let n = nbytes as usize;
        let off = dst_off as usize;
        if n > src_s.bytes || off.saturating_add(n) > dst_s.bytes {
            return Err(RpcError::Worker("hip memcpy_at oob".into()));
        }
        let dst_ptr = unsafe { dst_s.ptr.add(off) };
        allpaca_hip::memcpy(dst_ptr, src_s.ptr, n, allpaca_hip::MEMCPY_D2D)
            .map_err(|e| RpcError::Worker(e.to_string()))?;
        Ok(())
    }

    fn clear(&mut self, id: BufId, value: u8) -> Result<(), RpcError> {
        let s = self.slot(id)?;
        allpaca_hip::device_memset(s.ptr, value, s.bytes)
            .map_err(|e| RpcError::Worker(e.to_string()))
    }

    fn launch(&mut self, list: &LaunchList) -> Result<(), RpcError> {
        for item in &list.items {
            let gpu = self.launch_gpu(&item.bufs)?;
            self.bind_gpu(gpu)?;
            self.launch_one(gpu, item.op, &item.bufs, &item.i32s, &item.f32s)?;
        }
        Ok(())
    }

    fn launch_gpu(&self, bufs: &[BufId]) -> Result<usize, RpcError> {
        let first = need(bufs, 0)?;
        let gpu = self.slot(first)?.gpu;
        for id in bufs {
            if self.slot(*id)?.gpu != gpu {
                return Err(RpcError::Worker(
                    "rpc launch buffers span HIP devices".into(),
                ));
            }
        }
        Ok(gpu)
    }

    fn sync(&self) -> Result<(), RpcError> {
        allpaca_hip::stream_synchronize().map_err(|e| RpcError::Worker(e.to_string()))
    }

    fn ptr_f32(&self, id: BufId) -> Result<*mut f32, RpcError> {
        Ok(self.slot(id)?.ptr.cast())
    }

    fn launch_one(
        &mut self,
        gpu: usize,
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
            KernelOp::GemmNtQuant => {
                let m = *i32s.first().ok_or_else(|| RpcError::Worker("gemm m".into()))?;
                let n = *i32s.get(1).ok_or_else(|| RpcError::Worker("gemm n".into()))?;
                let k = *i32s.get(2).ok_or_else(|| RpcError::Worker("gemm k".into()))?;
                let a = self.ptr_f32(need(bufs, 0)?)?;
                let b_id = need(bufs, 1)?;
                let ty = self.slot(b_id)?.ggml_type;
                let b = self.slot(b_id)?.ptr;
                let c = self.ptr_f32(need(bufs, 2)?)?;
                if self.gpus[gpu].gfx906 && ty == 0 && m <= 8 {
                    allpaca_hip::gemm_nt_untiled_device(a, b.cast(), c, m, n, k)
                        .map_err(|e| RpcError::Worker(e.to_string()))?;
                } else {
                    hip_quant_gemm(ty, a, b, c, m, n, k)?;
                }
            }
            KernelOp::DeinterleaveQGate => {
                let n_heads = *i32s.first().ok_or_else(|| RpcError::Worker("deinterleave heads".into()))?;
                let head_dim = *i32s.get(1).ok_or_else(|| RpcError::Worker("deinterleave head".into()))?;
                self.gpus[gpu]
                    .hip
                    .deinterleave_q_gate_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        self.ptr_f32(need(bufs, 2)?)?,
                        n_heads,
                        head_dim,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::RmsNormHeads => {
                let n_heads = *i32s.first().ok_or_else(|| RpcError::Worker("rmsnorm heads".into()))?;
                let head_dim = *i32s.get(1).ok_or_else(|| RpcError::Worker("rmsnorm head".into()))?;
                let eps = *f32s.first().unwrap_or(&1e-5);
                self.gpus[gpu]
                    .hip
                    .rmsnorm_heads_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n_heads,
                        head_dim,
                        eps,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::SigmoidMul => {
                let n = *i32s.first().ok_or_else(|| RpcError::Worker("sigmoid n".into()))?;
                self.gpus[gpu]
                    .hip
                    .sigmoid_mul_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::RopePartial => {
                let n_heads = *i32s.first().ok_or_else(|| RpcError::Worker("rope heads".into()))?;
                let n_kv = *i32s.get(1).ok_or_else(|| RpcError::Worker("rope kv heads".into()))?;
                let head_dim = *i32s.get(2).ok_or_else(|| RpcError::Worker("rope head".into()))?;
                let pos = *i32s.get(3).ok_or_else(|| RpcError::Worker("rope pos".into()))?;
                let n_rot = *i32s.get(4).unwrap_or(&head_dim);
                let theta = *f32s.first().unwrap_or(&10_000.0);
                let sections = [
                    *f32s.get(1).unwrap_or(&0.0) as i32,
                    *f32s.get(2).unwrap_or(&0.0) as i32,
                    *f32s.get(3).unwrap_or(&0.0) as i32,
                    *f32s.get(4).unwrap_or(&0.0) as i32,
                ];
                self.gpus[gpu]
                    .hip
                    .rope_imrope_qk_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n_heads,
                        n_kv,
                        head_dim,
                        n_rot,
                        [pos; 3],
                        theta,
                        sections,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Attention => {
                let n_q = i32s[0];
                let n_kv = i32s[1];
                let n_heads = i32s[2];
                let n_kv_heads = *i32s.get(3).unwrap_or(&n_heads);
                let head_dim = i32s[4];
                let causal = *i32s.get(5).unwrap_or(&1) != 0;
                self.gpus[gpu]
                    .hip
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
                self.gpus[gpu]
                    .hip
                    .rmsnorm_device(x, w, out, n, eps)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::Rope => {
                let n_heads = i32s[0];
                let head_dim = i32s[1];
                let pos = i32s[2];
                let theta = *f32s.first().unwrap_or(&10_000.0);
                let x = self.ptr_f32(need(bufs, 0)?)?;
                self.gpus[gpu]
                    .hip
                    .rope_device(x, n_heads, head_dim, pos, theta)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::SiluMul => {
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                self.gpus[gpu]
                    .hip
                    .silu_mul_device(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::ResidualAdd => {
                let n = (self.slot(need(bufs, 0)?)?.bytes / 4) as i32;
                self.gpus[gpu]
                    .hip
                    .residual_add(
                        self.ptr_f32(need(bufs, 0)?)?,
                        self.ptr_f32(need(bufs, 1)?)?,
                        n,
                    )
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
            }
            KernelOp::EmbedGather | KernelOp::Memcpy => {
                if op == KernelOp::Memcpy {
                    if i32s.len() >= 2 {
                        self.copy_at(
                            need(bufs, 1)?,
                            need(bufs, 0)?,
                            i32s[0] as u64,
                            i32s[1] as u64,
                        )?;
                    } else {
                        self.copy(need(bufs, 1)?, need(bufs, 0)?)?;
                    }
                    return Ok(());
                }
                let dim = *i32s
                    .first()
                    .ok_or_else(|| RpcError::Worker("embed dim".into()))?;
                let n_tok = *i32s.get(1).unwrap_or(&1);
                if n_tok != 1 {
                    return Err(RpcError::Worker("hip embed_gather supports n_tok=1".into()));
                }
                let table = self.ptr_f32(need(bufs, 0)?)?;
                let ids = self.slot(need(bufs, 1)?)?;
                if ids.bytes < 4 {
                    return Err(RpcError::Worker("embed id".into()));
                }
                let mut tok = 0i32;
                allpaca_hip::memcpy(
                    (&mut tok as *mut i32).cast(),
                    ids.ptr,
                    4,
                    allpaca_hip::MEMCPY_D2H,
                )
                .map_err(|e| RpcError::Worker(e.to_string()))?;
                let out = self.ptr_f32(need(bufs, 2)?)?;
                self.gpus[gpu]
                    .hip
                    .embed_gather(table, tok as u32, out, dim)
                    .map_err(|e| RpcError::Worker(e.to_string()))?;
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
impl Drop for CudaWorker {
    fn drop(&mut self) {
        for (_, s) in self.slots.drain() {
            allpaca_cuda::device_free(s.ptr);
        }
    }
}

#[cfg(feature = "cuda")]
struct CudaSlot {
    gen: u32,
    /// ggml type id of a `PutTensor` blob. `0` is raw F32.
    ggml_type: u32,
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
            devices: Vec::new(),
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
                ggml_type: 0,
                ptr,
                bytes: bytes as usize,
            },
        );
        self.used = self.used.saturating_add(bytes);
        Ok(BufId { id, gen: 1 })
    }

    fn set_ggml_type(&mut self, id: BufId, ggml_type: u32) -> Result<(), RpcError> {
        let s = self
            .slots
            .get_mut(&id.id)
            .ok_or_else(|| RpcError::Worker(format!("unknown buf {}", id.id)))?;
        if s.gen != id.gen {
            return Err(RpcError::Worker("stale cuda buf".into()));
        }
        s.ggml_type = ggml_type;
        Ok(())
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

    /// Device-resident partial copy: `src[0..nbytes]` → `dst[dst_off..]`.
    fn copy_at(
        &mut self,
        src: BufId,
        dst: BufId,
        dst_off: u64,
        nbytes: u64,
    ) -> Result<(), RpcError> {
        let src_s = self.slot(src)?;
        let dst_s = self.slot(dst)?;
        let n = nbytes as usize;
        let off = dst_off as usize;
        if n > src_s.bytes || off.saturating_add(n) > dst_s.bytes {
            return Err(RpcError::Worker("cuda memcpy_at oob".into()));
        }
        let dst_ptr = unsafe { dst_s.ptr.cast::<u8>().add(off) }.cast();
        allpaca_cuda::memcpy_on(
            dst_ptr,
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
            KernelOp::GemmNtQuant
            | KernelOp::DeinterleaveQGate
            | KernelOp::RmsNormHeads
            | KernelOp::SigmoidMul
            | KernelOp::RopePartial => {
                return Err(RpcError::Worker(
                    "quant gemm and gated attention are implemented on CPU and HIP workers".into(),
                ));
            }
            KernelOp::Memcpy => {
                if i32s.len() >= 2 {
                    self.copy_at(
                        need(bufs, 1)?,
                        need(bufs, 0)?,
                        i32s[0] as u64,
                        i32s[1] as u64,
                    )?;
                } else {
                    self.copy(need(bufs, 1)?, need(bufs, 0)?)?;
                }
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
    fn moe_ffn(
        &mut self,
        req: &crate::protocol::MoeFfnRequest,
        x: &crate::protocol::MoeX,
        pairs: &[(u32, u32, f32)],
    ) -> Result<Vec<f32>, RpcError> {
        match self {
            Self::Cpu(w) => w.moe_ffn(req, &x.to_f32(), pairs),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.moe_ffn(req, x, pairs),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => Err(RpcError::Worker("CUDA worker has no MoeFfn yet".into())),
        }
    }

    fn hello(&self) -> HelloReply {
        match self {
            Self::Cpu(w) => w.hello(),
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.hello(),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.hello(),
        }
    }

    fn put_tensor(
        &mut self,
        ggml_type: u32,
        bytes: &[u8],
        device: Option<u32>,
    ) -> Result<BufId, RpcError> {
        let repacked = self.repack_rocmfp4(ggml_type, bytes, device)?;
        let staged = self.stage_q8_0(ggml_type, bytes)?;
        let (payload, stored_ty) = match (repacked.as_deref(), staged.as_deref()) {
            (Some(f32_bytes), _) => (f32_bytes, 0),
            (None, Some(q8)) => (q8, ggml_type),
            (None, None) => (bytes, ggml_type),
        };
        let id = self.alloc(payload.len() as u64, device)?;
        self.h2d(id, 0, payload)?;
        match self {
            Self::Cpu(w) => w.set_ggml_type(id, stored_ty)?,
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.set_ggml_type(id, stored_ty)?,
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => w.set_ggml_type(id, stored_ty)?,
        }
        Ok(id)
    }

    /// HIP Q8_0 kernels read 36-byte staged blocks (F32 scale, aligned qs), not
    /// GGUF's 34-byte blocks. Block-wise, so stacked expert tensors stage too.
    fn stage_q8_0(&self, ggml_type: u32, bytes: &[u8]) -> Result<Option<Vec<u8>>, RpcError> {
        #[cfg(feature = "hip")]
        if ggml_type == 8 && matches!(self, Self::Hip(_)) {
            return allpaca_hip::stage_q8_0_blocks(bytes)
                .map(Some)
                .map_err(|e| RpcError::Worker(e.to_string()));
        }
        let _ = (ggml_type, bytes);
        Ok(None)
    }

    /// gfx906 runs the scalar ROCmFP4 GEMV. Expand those weights to F32 once
    /// so decode uses the vectorized GEMV. Other arches keep the packed blob.
    fn repack_rocmfp4(
        &self,
        ggml_type: u32,
        bytes: &[u8],
        device: Option<u32>,
    ) -> Result<Option<Vec<u8>>, RpcError> {
        if !matches!(ggml_type, 100 | 101) {
            return Ok(None);
        }
        #[cfg(feature = "hip")]
        if let Self::Hip(w) = self {
            let idx = device.unwrap_or(w.sticky as u32) as usize;
            let dev = w
                .gpus
                .get(idx)
                .or_else(|| w.gpus.first())
                .map(|g| g.device)
                .unwrap_or(0);
            let expand = allpaca_hip::device_gcn_arch(dev)
                .is_some_and(|(arch, _)| gfx906_expands_rocmfp4(&arch));
            if expand {
                return expand_rocmfp4_f32(bytes, ggml_type == 101).map(Some);
            }
        }
        let _ = (bytes, device);
        Ok(None)
    }

    fn alloc(&mut self, bytes: u64, gpu: Option<u32>) -> Result<BufId, RpcError> {
        match self {
            Self::Cpu(w) => {
                if gpu.unwrap_or(0) != 0 {
                    return Err(RpcError::Worker("cpu worker has one device".into()));
                }
                w.alloc(bytes)
            }
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.alloc(bytes, gpu.map(|g| g as usize)),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => {
                if gpu.unwrap_or(0) != 0 {
                    return Err(RpcError::Worker("cuda worker has one device".into()));
                }
                w.alloc(bytes)
            }
        }
    }

    /// Per-device stats. HIP workers on Linux read amdgpu sysfs (device-wide
    /// VRAM, busy percent, temperature, power); elsewhere memory comes from
    /// the runtime and the rest is unreported.
    fn device_stats(&self) -> Vec<crate::protocol::DeviceStat> {
        use crate::protocol::DeviceStat;
        match self {
            Self::Cpu(w) => vec![DeviceStat {
                name: "cpu".into(),
                mem_used: w.used,
                mem_total: w.memory_bytes,
                util_pct: -1,
                temp_c: -1,
                power_w: -1,
            }],
            #[cfg(feature = "hip")]
            Self::Hip(w) => {
                let cards = crate::gpu_sysfs::amd_cards();
                w.gpus
                    .iter()
                    .map(|g| {
                        let (free, total) =
                            allpaca_hip::device_memory_info(g.device).unwrap_or((0, g.total));
                        let mut stat = DeviceStat {
                            name: format!("hip:{}", g.device),
                            mem_used: total.saturating_sub(free),
                            mem_total: total,
                            util_pct: -1,
                            temp_c: -1,
                            power_w: -1,
                        };
                        // HIP enumerates AMD cards in PCI order, as sysfs does.
                        if let Some(card) = usize::try_from(g.device).ok().and_then(|i| cards.get(i)) {
                            card.fill(&mut stat);
                        }
                        stat
                    })
                    .collect()
            }
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => {
                let (free, total) = allpaca_cuda::device_memory_info(w.device)
                    .unwrap_or((w.memory_bytes.saturating_sub(w.used), w.memory_bytes));
                vec![DeviceStat {
                    name: format!("cuda:{}", w.device),
                    mem_used: total.saturating_sub(free),
                    mem_total: total,
                    util_pct: -1,
                    temp_c: -1,
                    power_w: -1,
                }]
            }
        }
    }

    fn device_count(&self) -> u32 {
        match self {
            Self::Cpu(_) => 1,
            #[cfg(feature = "hip")]
            Self::Hip(w) => w.device_count(),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => 1,
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
            Self::Hip(w) => w.mem_info(),
            #[cfg(feature = "cuda")]
            Self::Cuda(w) => allpaca_cuda::device_memory_info(w.device)
                .unwrap_or_else(|_| (w.memory_bytes.saturating_sub(w.used), w.memory_bytes)),
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
    // A coordinator that dies without closing the socket (killed, network
    // drop) would otherwise leave the worker blocked on it forever, never
    // accepting the next client.
    let keepalive = socket2::TcpKeepalive::new()
        .with_time(std::time::Duration::from_secs(10))
        .with_interval(std::time::Duration::from_secs(5));
    if let Err(e) = socket2::SockRef::from(&stream).set_tcp_keepalive(&keepalive) {
        tracing::warn!(error = %e, "rpc keepalive not set");
    }
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
            let gpu = if c.has_remaining() {
                Some(c.u32()?)
            } else {
                None
            };
            let id = backend.alloc(bytes, gpu)?;
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
            write_u32(&mut out, backend.device_count());
            Ok(out)
        }
        Cmd::DeviceStats => Ok(crate::protocol::encode_device_stats(&backend.device_stats())),
        Cmd::Sync => {
            backend.sync()?;
            Ok(Vec::new())
        }
        Cmd::RunSpan => run_span(backend, payload),
        Cmd::MoeFfn => {
            let (req, x, pairs) = crate::protocol::decode_moe_ffn(payload)?;
            let out = backend.moe_ffn(&req, &x, &pairs)?;
            Ok(crate::protocol::encode_moe_reply(&out, req.f16_out))
        }
        Cmd::PutTensor => {
            let mut c = Cursor::new(payload);
            let ggml_type = c.u32()?;
            let n_dims = c.u32()?;
            if n_dims > 4 {
                return Err(RpcError::Protocol("put tensor rank".into()));
            }
            for _ in 0..n_dims {
                let _ = c.u32()?;
            }
            let data = c.bytes()?;
            let device = if c.has_remaining() {
                Some(c.u32()?)
            } else {
                None
            };
            let id = backend.put_tensor(ggml_type, data, device)?;
            let mut out = Vec::new();
            crate::protocol::write_buf_id(&mut out, id);
            Ok(out)
        }
    }
}

/// H2D the activation, run each launch list, D2H the last activation. One reply.
fn run_span(backend: &mut WorkerBackend, payload: &[u8]) -> Result<Vec<u8>, RpcError> {
    let t0 = std::time::Instant::now();
    let (steps, activation) = crate::protocol::decode_run_span(payload)?;
    let nbytes = activation.len() as u64;
    let first = steps[0].0;
    backend.h2d(first, 0, &activation)?;
    let t_h2d = t0.elapsed();
    backend.launch(&steps[0].1)?;
    for i in 1..steps.len() {
        let src = steps[i - 1].0;
        let dst = steps[i].0;
        if backend.copy(src, dst).is_err() {
            let mid = backend.d2h(src, 0, nbytes)?;
            backend.h2d(dst, 0, &mid)?;
        }
        backend.launch(&steps[i].1)?;
    }
    let t_launch = t0.elapsed();
    backend.sync()?;
    let t_sync = t0.elapsed();
    let out = backend.d2h(steps[steps.len() - 1].0, 0, nbytes)?;
    tracing::debug!(
        steps = steps.len(),
        h2d_us = t_h2d.as_micros() as u64,
        launch_us = t_launch.saturating_sub(t_h2d).as_micros() as u64,
        sync_us = t_sync.saturating_sub(t_launch).as_micros() as u64,
        d2h_us = t0.elapsed().saturating_sub(t_sync).as_micros() as u64,
        "run_span"
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::place_sticky;

    #[test]
    fn sticky_stays_until_the_device_cannot_fit() {
        assert_eq!(place_sticky(&[100, 100], 0, 40, None).unwrap(), 0);
        assert_eq!(place_sticky(&[30, 80], 0, 40, None).unwrap(), 1);
        assert_eq!(place_sticky(&[10, 10], 1, 4, Some(0)).unwrap(), 0);
        assert!(place_sticky(&[10, 80], 0, 40, Some(0)).is_err());
        assert!(place_sticky(&[10, 10], 0, 40, None).is_err());
    }

    #[test]
    fn gfx906_repacks_rocmfp4_other_arch_keeps_it() {
        use super::{expand_rocmfp4_f32, gfx906_expands_rocmfp4};

        assert!(gfx906_expands_rocmfp4("gfx906"));
        assert!(gfx906_expands_rocmfp4("gfx906:sramecc+:xnack-"));
        assert!(!gfx906_expands_rocmfp4("gfx1201"));
        assert!(!gfx906_expands_rocmfp4("gfx908"));
        let mut block = vec![0u8; allpaca_cpu::ROCMFP4_BLOCK_BYTES];
        block[0] = 0x21;
        block[16] = 0x38;
        block[17] = 0x38;
        // FAST (type 101) blocks expand through their own 17-byte layout.
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 - 20.0) * 0.07).collect();
        let fast = allpaca_cpu::quantize_rocmfp4_fast(&vals);
        let got = expand_rocmfp4_f32(&fast, true).unwrap();
        let mut want = vec![0.0f32; 64];
        allpaca_cpu::dequant_rocmfp4_fast(&fast, 64, &mut want).unwrap();
        let want: Vec<u8> = want.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(got, want);
        assert!(expand_rocmfp4_f32(&fast, false).is_err(), "34 bytes is not whole 18-byte blocks");
        let f32_bytes = expand_rocmfp4_f32(&block, false).unwrap();
        assert_eq!(
            f32_bytes.len(),
            allpaca_cpu::ROCMFP4_BLOCK_ELEMS * 4
        );
    }

    /// Fused `MoeFfn` at Qwen3-Next shapes on one GPU: 512 experts, dim 2048,
    /// ff 512, Q5_K gate/up and Q6_K down, 10 experts per token.
    /// `cargo test --release -p allpaca-rpc --features hip --lib bench_fused_moe -- --ignored --nocapture`
    #[cfg(feature = "hip")]
    #[test]
    #[ignore]
    fn bench_fused_moe_qwen3next_shapes() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let (n_exp, dim, ff) = (512usize, 2048usize, 512usize);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut noise = |bytes: &mut [u8]| {
            for b in bytes.iter_mut() {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *b = (seed >> 56) as u8;
            }
        };
        // Random quants with small, finite f16 scales (0x1400 ~ 1.2e-3).
        let tensor = |noise: &mut dyn FnMut(&mut [u8]), ty: u32, n: usize| -> Vec<u8> {
            let (bb, scales): (usize, &[usize]) = if ty == 13 { (176, &[0, 2]) } else { (210, &[208]) };
            let mut v = vec![0u8; n / 256 * bb];
            noise(&mut v);
            for blk in v.chunks_exact_mut(bb) {
                for &o in scales {
                    blk[o..o + 2].copy_from_slice(&0x1400u16.to_le_bytes());
                }
            }
            v
        };
        let mut backend = super::WorkerBackend::Hip(super::HipWorker::new(0).expect("HipWorker"));
        let gate = backend.put_tensor(13, &tensor(&mut noise, 13, n_exp * ff * dim), Some(0)).unwrap();
        let up = backend.put_tensor(13, &tensor(&mut noise, 13, n_exp * ff * dim), Some(0)).unwrap();
        let down = backend.put_tensor(14, &tensor(&mut noise, 14, n_exp * dim * ff), Some(0)).unwrap();
        let req = crate::protocol::MoeFfnRequest {
            gate,
            up,
            down,
            n_experts: n_exp as u32,
            dim: dim as u32,
            ff: ff as u32,
            f16_out: false,
        };
        // Verify-span shape: 4 rows sharing 3 of their 10 experts (~78% of
        // pairs unique, as measured in decode) vs the same pairs all distinct.
        for shared in [true, false] {
            let rows = 4usize;
            let xf: Vec<f32> = (0..rows * dim).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
            let (q, d) = crate::protocol::quantize_q8_rows(&xf);
            let x = crate::protocol::MoeX::Q8 { q, d };
            let pairs_for = |it: usize| -> Vec<(u32, u32, f32)> {
                (0..rows * 10)
                    .map(|i| {
                        let (r, j) = (i / 10, i % 10);
                        let e = if shared && j < 3 { j * 101 + it * 7 } else { (r * 10 + j) * 11 + it * 7 };
                        (r as u32, (e % n_exp) as u32, 0.1)
                    })
                    .collect()
            };
            for it in 0..50 {
                backend.moe_ffn(&req, &x, &pairs_for(it)).unwrap();
            }
            let t = std::time::Instant::now();
            for it in 0..300 {
                backend.moe_ffn(&req, &x, &pairs_for(it + 50)).unwrap();
            }
            eprintln!("BENCH rows=4 shared3={shared} moe_ffn {:.1} us", t.elapsed().as_secs_f64() * 1e6 / 300.0);
        }
        // (rows, fresh experts every call, idle gap between calls in us)
        for (rows, vary, gap_us) in [(1usize, true, 0u64), (1, true, 250), (1, true, 500), (1, true, 1000), (1, true, 2000), (4, true, 0)] {
            let xf: Vec<f32> = (0..rows * dim).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
            // Rows arrive pre-quantized, as the coordinator sends them.
            let (q, d) = crate::protocol::quantize_q8_rows(&xf);
            let x = crate::protocol::MoeX::Q8 { q, d };
            let pairs_for = |it: usize| -> Vec<(u32, u32, f32)> {
                let salt = if vary { it * 131 } else { 0 };
                (0..rows * 10)
                    .map(|i| ((i / 10) as u32, ((i * 53 + 7 + salt) % n_exp) as u32, 0.1))
                    .collect()
            };
            for it in 0..50 {
                backend.moe_ffn(&req, &x, &pairs_for(it)).unwrap();
            }
            let iters = 300;
            let mut busy = 0.0f64;
            for it in 0..iters {
                let p = pairs_for(it + 50);
                if gap_us > 0 {
                    let until = std::time::Instant::now() + std::time::Duration::from_micros(gap_us);
                    while std::time::Instant::now() < until {}
                }
                let t = std::time::Instant::now();
                backend.moe_ffn(&req, &x, &p).unwrap();
                busy += t.elapsed().as_secs_f64();
            }
            let us = busy * 1e6 / iters as f64;
            let mb = (rows * 10) as f64 * (2.0 * 176.0 + 210.0) / 256.0 * (ff * dim) as f64 / 1e6;
            eprintln!("BENCH rows={rows} vary={vary} gap={gap_us}us moe_ffn {us:.1} us ({:.0} GB/s)", mb / us * 1e3);
        }
    }
}
