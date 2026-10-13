use std::io::{self, Read, Write};

use thiserror::Error;

/// Protocol major; must match on both sides (llama.cpp HELLO rule).
pub const PROTO_MAJOR: u16 = 1;
pub const PROTO_MINOR: u16 = 0;
pub const PROTO_PATCH: u16 = 0;
/// Bump when the named kernel set or C ABI revision changes.
/// 2 adds `PutTensor` and `GemmNtQuant` (packed GGUF weights).
/// 3 adds `MoeFfn` (expert-parallel MoE FFN).
/// 6 adds `DeviceStats` (per-GPU memory, utilisation, temperature).
/// 7 adds multi-row ops for batched prefill (`RmsNormRows`, `RopeRows`,
/// `RopePartialRows`), an explicit element count on `SiluMul` /
/// `ResidualAdd`, and `RunSpan` steps that share one activation buffer.
pub const KERNEL_ABI_REV: u32 = 7;

pub const STATUS_OK: u8 = 0;
pub const STATUS_ERR: u8 = 1;

pub const DTYPE_F32: u32 = 1;

/// Generation-counted remote buffer handle. Never a raw device pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufId {
    pub id: u32,
    pub gen: u32,
}

impl BufId {
    pub fn new(id: u32, gen: u32) -> Self {
        Self { id, gen }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Cmd {
    Hello = 0,
    Alloc = 1,
    Free = 2,
    Clear = 3,
    H2D = 4,
    D2H = 5,
    Copy = 6,
    H2DHash = 7,
    LaunchList = 8,
    MemInfo = 9,
    DeviceCount = 10,
    Sync = 11,
    /// One activation upload, one or more launch lists, one activation download.
    RunSpan = 12,
    /// Store one packed GGUF tensor (ggml type + dims + raw bytes). Reply is a BufId.
    PutTensor = 13,
    /// Expert-parallel MoE FFN over stacked expert tensors; see [`MoeFfnRequest`].
    MoeFfn = 14,
    /// Live per-device stats; reply is [`DeviceStat`] per device.
    DeviceStats = 15,
    /// A call to a worker extension: its name, a head, then the bulk payload.
    /// Only sent to workers whose HELLO names the extension.
    Ext = 16,
}

impl Cmd {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Hello,
            1 => Self::Alloc,
            2 => Self::Free,
            3 => Self::Clear,
            4 => Self::H2D,
            5 => Self::D2H,
            6 => Self::Copy,
            7 => Self::H2DHash,
            8 => Self::LaunchList,
            9 => Self::MemInfo,
            10 => Self::DeviceCount,
            11 => Self::Sync,
            12 => Self::RunSpan,
            13 => Self::PutTensor,
            14 => Self::MoeFfn,
            15 => Self::DeviceStats,
            16 => Self::Ext,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RpcBackendKind {
    Cpu = 0,
    Hip = 1,
    Cuda = 2,
}

impl RpcBackendKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::Cpu,
            1 => Self::Hip,
            2 => Self::Cuda,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KernelOp {
    GemmNt = 1,
    Attention = 2,
    RmsNorm = 3,
    Rope = 4,
    SiluMul = 5,
    ResidualAdd = 6,
    EmbedGather = 7,
    Memcpy = 8,
    /// `C = A @ B^T` where `B` is the packed tensor from `PutTensor`.
    /// `i32s` are `[m, n, k]`. The ggml type lives on the weight buffer.
    GemmNtQuant = 9,
    /// Split per-head `[Q|gate]` into separate buffers. `i32s`: `[n_heads, head_dim]`.
    DeinterleaveQGate = 10,
    /// Per-head RMSNorm. `i32s`: `[n_heads, head_dim]`. `f32s`: `[eps]`.
    RmsNormHeads = 11,
    /// `x *= sigmoid(gate)`. `i32s`: `[n]`.
    SigmoidMul = 12,
    /// Partial or sectioned RoPE on Q and K. `i32s`: `[n_heads, n_kv_heads, head_dim, pos, n_rot]`.
    /// `f32s`: `[theta, section0, section1, section2, section3]`.
    RopePartial = 13,
    /// RMSNorm of `rows` rows. `bufs`: `[x, w, out]`. `i32s`: `[rows, dim]`.
    /// `f32s`: `[eps]`.
    RmsNormRows = 14,
    /// Full-head RoPE on Q and K for `rows` consecutive positions. `bufs`:
    /// `[q, k]`. `i32s`: `[rows, n_heads, n_kv_heads, head_dim, start_pos]`.
    /// `f32s`: `[theta]`.
    RopeRows = 15,
    /// [`Self::RopePartial`] for `rows` consecutive positions. `i32s`:
    /// `[rows, n_heads, n_kv_heads, head_dim, start_pos, n_rot]`.
    RopePartialRows = 16,
}

impl KernelOp {
    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => Self::GemmNt,
            2 => Self::Attention,
            3 => Self::RmsNorm,
            4 => Self::Rope,
            5 => Self::SiluMul,
            6 => Self::ResidualAdd,
            7 => Self::EmbedGather,
            8 => Self::Memcpy,
            9 => Self::GemmNtQuant,
            10 => Self::DeinterleaveQGate,
            11 => Self::RmsNormHeads,
            12 => Self::SigmoidMul,
            13 => Self::RopePartial,
            14 => Self::RmsNormRows,
            15 => Self::RopeRows,
            16 => Self::RopePartialRows,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HelloRequest {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub caps: u32,
    /// Optional shared secret. Empty when the worker is loopback-only.
    pub token: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HelloReply {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub kernel_abi_rev: u32,
    pub backend: RpcBackendKind,
    pub dtype_mask: u32,
    pub device_name: String,
    /// Sum of [`Self::devices`] when that list is non-empty.
    pub memory_bytes: u64,
    /// Free bytes per local device. Empty means one device of `memory_bytes`
    /// (older workers that do not append the list).
    pub devices: Vec<u64>,
}

/// One worker device as the operator sees it. `-1` = not reported by this
/// platform.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceStat {
    pub name: String,
    /// Device-wide memory in use (every process), bytes.
    pub mem_used: u64,
    pub mem_total: u64,
    /// GPU busy percent, 0..=100.
    pub util_pct: i32,
    pub temp_c: i32,
    pub power_w: i32,
}

pub fn encode_device_stats(stats: &[DeviceStat]) -> Vec<u8> {
    let mut w = Vec::new();
    write_u32(&mut w, stats.len() as u32);
    for s in stats {
        write_str(&mut w, &s.name);
        write_u64(&mut w, s.mem_used);
        write_u64(&mut w, s.mem_total);
        write_i32(&mut w, s.util_pct);
        write_i32(&mut w, s.temp_c);
        write_i32(&mut w, s.power_w);
    }
    w
}

pub fn decode_device_stats(data: &[u8]) -> Result<Vec<DeviceStat>, RpcError> {
    let mut c = Cursor::new(data);
    let n = c.u32()? as usize;
    if n > 64 {
        return Err(RpcError::Protocol("too many rpc devices".into()));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(DeviceStat {
            name: c.str()?,
            mem_used: c.u64()?,
            mem_total: c.u64()?,
            util_pct: c.i32()?,
            temp_c: c.i32()?,
            power_w: c.i32()?,
        });
    }
    Ok(out)
}

impl HelloReply {
    /// One entry per device this worker can place buffers on.
    pub fn device_memories(&self) -> Vec<u64> {
        if self.devices.is_empty() {
            vec![self.memory_bytes]
        } else {
            self.devices.clone()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchItem {
    pub op: KernelOp,
    pub bufs: Vec<BufId>,
    pub i32s: Vec<i32>,
    pub f32s: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LaunchList {
    pub items: Vec<LaunchItem>,
}

#[derive(Debug, Error)]
pub enum RpcError {
    #[error("rpc io: {0}")]
    Io(#[from] io::Error),
    #[error("rpc protocol: {0}")]
    Protocol(String),
    #[error("rpc version mismatch: server {major}.{minor}.{patch}")]
    Version { major: u16, minor: u16, patch: u16 },
    #[error("rpc worker: {0}")]
    Worker(String),
    #[error("rpc unauthorized")]
    Unauthorized,
}

pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

pub fn write_u8(w: &mut Vec<u8>, v: u8) {
    w.push(v);
}

pub fn write_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub fn write_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub fn write_i32(w: &mut Vec<u8>, v: i32) {
    w.extend_from_slice(&v.to_le_bytes());
}

pub fn write_f32(w: &mut Vec<u8>, v: f32) {
    w.extend_from_slice(&v.to_le_bytes());
}

/// One MoE layer's experts on a worker: `gate` / `up` are stacked
/// `[n_experts][ff][dim]` and `down` `[n_experts][dim][ff]` packed GGUF tensors
/// (`PutTensor`), each expert an equal byte slice. When all three name one
/// buffer it holds a slot per expert, gate | up | down: the HIP worker runs
/// Q2_0 slots (ggml type 42) with its routed-expert kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeFfnRequest {
    pub gate: BufId,
    pub up: BufId,
    pub down: BufId,
    pub n_experts: u32,
    pub dim: u32,
    pub ff: u32,
    /// Reply as F16 instead of F32: half the bytes back over a slow link, at
    /// ~5e-4 relative rounding of the MoE output.
    pub f16_out: bool,
}

/// `MoeFfn` format flags (one u32 in the request).
const MOE_FLAG_Q8_IN: u32 = 1;
const MOE_FLAG_F16_OUT: u32 = 2;

/// F32 to F16 bits, round to nearest even, with subnormals and overflow to
/// infinity (unlike `allpaca_cpu::f32_to_f16`, which truncates).
pub fn f32_to_f16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        // Inf stays inf; NaN stays a (quiet) NaN.
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: shift the full significand, rounding half to even.
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rest = m & ((1u32 << shift) - 1);
        let mut out = m >> shift;
        if rest > half || (rest == half && out & 1 == 1) {
            out += 1;
        }
        return sign | out as u16;
    }
    let mut out = ((e as u32) << 10) | (mant >> 13);
    let rest = mant & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && out & 1 == 1) {
        out += 1; // may carry into the exponent, and up to inf: both correct
    }
    sign | out as u16
}

/// `MoeFfn` reply payload: `out` as F32 or F16 little-endian values.
pub fn encode_moe_reply(out: &[f32], f16_out: bool) -> Vec<u8> {
    if f16_out {
        out.iter().flat_map(|&v| f32_to_f16_rne(v).to_le_bytes()).collect()
    } else {
        out.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// Inverse of [`encode_moe_reply`]; `None` when `body` is not `n` values.
pub fn decode_moe_reply(body: &[u8], n: usize, f16_out: bool) -> Option<Vec<f32>> {
    if f16_out {
        (body.len() == n * 2).then(|| {
            body.chunks_exact(2)
                .map(|b| allpaca_cpu::f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        })
    } else {
        (body.len() == n * 4).then(|| {
            body.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        })
    }
}

/// Quantize activation rows the way the worker's int8 K-quant GEMVs do: per
/// 32 values, `d = amax / 127` and `q = round(x / d)`, bit for bit what the
/// HIP `quantize_rows_q8_kernel` computes. `x.len()` must be whole blocks.
pub fn quantize_q8_rows(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let mut q = Vec::with_capacity(x.len());
    let mut d = Vec::with_capacity(x.len() / 32);
    for blk in x.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        let inv = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        q.extend(blk.iter().map(|v| (v * inv).round().clamp(-127.0, 127.0) as i8));
        d.push(scale);
    }
    (q, d)
}

/// `MoeFfn` activation rows: F32, or pre-quantized int8 blocks of 32 (a
/// quarter of the bytes on the wire, and exactly what the fused kernels read).
#[derive(Debug, Clone, PartialEq)]
pub enum MoeX {
    F32(Vec<f32>),
    Q8 { q: Vec<i8>, d: Vec<f32> },
}

impl MoeX {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(x) => x.len(),
            Self::Q8 { q, .. } => q.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// F32 rows; dequantized for Q8.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Self::F32(x) => x.clone(),
            Self::Q8 { q, d } => q
                .iter()
                .enumerate()
                .map(|(i, &v)| f32::from(v) * d[i / 32])
                .collect(),
        }
    }
}

/// `MoeFfn` payload: request, rows `x` (`[rows][dim]`, F32 or int8 blocks
/// when `q8`), then `(row, expert, weight)` pairs. The reply is `[rows][dim]`
/// F32 of `sum weight * down_e(silu(gate_e x) * up_e x)` per row.
pub fn encode_moe_ffn(
    req: &MoeFfnRequest,
    x: &[f32],
    q8: bool,
    pairs: &[(u32, u32, f32)],
) -> Vec<u8> {
    let q8 = q8 && x.len() % 32 == 0;
    let mut w = Vec::with_capacity(64 + x.len() * 4 + pairs.len() * 12);
    for id in [req.gate, req.up, req.down] {
        write_buf_id(&mut w, id);
    }
    let flags = if q8 { MOE_FLAG_Q8_IN } else { 0 } | if req.f16_out { MOE_FLAG_F16_OUT } else { 0 };
    for v in [req.n_experts, req.dim, req.ff, x.len() as u32, pairs.len() as u32, flags] {
        w.extend_from_slice(&v.to_le_bytes());
    }
    if q8 {
        let (q, d) = quantize_q8_rows(x);
        w.extend(q.iter().map(|&v| v as u8));
        for v in d {
            w.extend_from_slice(&v.to_le_bytes());
        }
    } else {
        for v in x {
            w.extend_from_slice(&v.to_le_bytes());
        }
    }
    for &(row, expert, weight) in pairs {
        w.extend_from_slice(&row.to_le_bytes());
        w.extend_from_slice(&expert.to_le_bytes());
        w.extend_from_slice(&weight.to_le_bytes());
    }
    w
}

#[allow(clippy::type_complexity)]
pub fn decode_moe_ffn(data: &[u8]) -> Result<(MoeFfnRequest, MoeX, Vec<(u32, u32, f32)>), RpcError> {
    let mut c = Cursor::new(data);
    let (gate, up, down) = (c.buf_id()?, c.buf_id()?, c.buf_id()?);
    let (n_experts, dim, ff, n_x, n_pairs) = (c.u32()?, c.u32()?, c.u32()?, c.u32()?, c.u32()?);
    let flags = c.u32()?;
    let (q8, f16_out) = (flags & MOE_FLAG_Q8_IN != 0, flags & MOE_FLAG_F16_OUT != 0);
    if dim == 0 || n_x % dim != 0 || n_experts == 0 || (q8 && n_x % 32 != 0) {
        return Err(RpcError::Protocol("moe ffn shape".into()));
    }
    let rows = n_x / dim;
    let x = if q8 {
        let q = (0..n_x).map(|_| c.u8().map(|b| b as i8)).collect::<Result<Vec<_>, _>>()?;
        let d = (0..n_x / 32).map(|_| c.f32()).collect::<Result<Vec<_>, _>>()?;
        MoeX::Q8 { q, d }
    } else {
        MoeX::F32((0..n_x).map(|_| c.f32()).collect::<Result<Vec<_>, _>>()?)
    };
    let mut pairs = Vec::with_capacity(n_pairs as usize);
    for _ in 0..n_pairs {
        let (row, expert, weight) = (c.u32()?, c.u32()?, c.f32()?);
        if row >= rows || expert >= n_experts {
            return Err(RpcError::Protocol("moe ffn pair out of range".into()));
        }
        pairs.push((row, expert, weight));
    }
    let req = MoeFfnRequest { gate, up, down, n_experts, dim, ff, f16_out };
    Ok((req, x, pairs))
}

pub fn write_buf_id(w: &mut Vec<u8>, id: BufId) {
    write_u32(w, id.id);
    write_u32(w, id.gen);
}

pub fn write_bytes(w: &mut Vec<u8>, data: &[u8]) {
    write_u64(w, data.len() as u64);
    w.extend_from_slice(data);
}

pub fn write_str(w: &mut Vec<u8>, s: &str) {
    write_bytes(w, s.as_bytes());
}

pub struct Cursor<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, off: 0 }
    }

    pub fn has_remaining(&self) -> bool {
        self.off < self.data.len()
    }

    pub fn u8(&mut self) -> Result<u8, RpcError> {
        if self.off >= self.data.len() {
            return Err(RpcError::Protocol("truncated u8".into()));
        }
        let v = self.data[self.off];
        self.off += 1;
        Ok(v)
    }

    pub fn u16(&mut self) -> Result<u16, RpcError> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    pub fn u32(&mut self) -> Result<u32, RpcError> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn u64(&mut self) -> Result<u64, RpcError> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn i32(&mut self) -> Result<i32, RpcError> {
        Ok(self.u32()? as i32)
    }

    pub fn f32(&mut self) -> Result<f32, RpcError> {
        Ok(f32::from_le_bytes(self.u32()?.to_le_bytes()))
    }

    pub fn buf_id(&mut self) -> Result<BufId, RpcError> {
        Ok(BufId {
            id: self.u32()?,
            gen: self.u32()?,
        })
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], RpcError> {
        let n = self.u64()? as usize;
        if self.off + n > self.data.len() {
            return Err(RpcError::Protocol("truncated bytes".into()));
        }
        let s = &self.data[self.off..self.off + n];
        self.off += n;
        Ok(s)
    }

    pub fn str(&mut self) -> Result<String, RpcError> {
        let b = self.bytes()?;
        String::from_utf8(b.to_vec()).map_err(|_| RpcError::Protocol("invalid utf-8".into()))
    }

    fn read_exact(&mut self, dest: &mut [u8]) -> Result<(), RpcError> {
        if self.off + dest.len() > self.data.len() {
            return Err(RpcError::Protocol("truncated".into()));
        }
        dest.copy_from_slice(&self.data[self.off..self.off + dest.len()]);
        self.off += dest.len();
        Ok(())
    }
}

pub fn encode_hello_req(req: &HelloRequest) -> Vec<u8> {
    let mut w = Vec::new();
    write_u16(&mut w, req.major);
    write_u16(&mut w, req.minor);
    write_u16(&mut w, req.patch);
    write_u32(&mut w, req.caps);
    if !req.token.is_empty() {
        write_str(&mut w, &req.token);
    }
    w
}

pub fn decode_hello_req(data: &[u8]) -> Result<HelloRequest, RpcError> {
    let mut c = Cursor::new(data);
    let major = c.u16()?;
    let minor = c.u16()?;
    let patch = c.u16()?;
    let caps = c.u32()?;
    let token = if c.has_remaining() {
        c.str()?
    } else {
        String::new()
    };
    Ok(HelloRequest {
        major,
        minor,
        patch,
        caps,
        token,
    })
}

/// Constant-time compare for HELLO tokens (length mismatch is not constant-time).
pub fn tokens_match(expected: &str, given: &str) -> bool {
    if expected.len() != given.len() {
        return false;
    }
    expected
        .bytes()
        .zip(given.bytes())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

/// Loopback bind hosts that do not need `--insecure-bind` / `--token`.
pub fn is_loopback_bind_host(host: &str) -> bool {
    let h = host.trim().trim_matches(|c| c == '[' || c == ']');
    h.eq_ignore_ascii_case("localhost")
        || h == "::1"
        || h == "0:0:0:0:0:0:0:1"
        || h.parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Non-loopback RPC binds require an explicit insecure flag and a shared secret.
pub fn check_rpc_bind(host: &str, insecure_bind: bool, token: &str) -> Result<(), RpcError> {
    if is_loopback_bind_host(host) {
        return Ok(());
    }
    if !insecure_bind {
        return Err(RpcError::Protocol(
            "refusing non-loopback bind without --insecure-bind (RPC has no TLS)".into(),
        ));
    }
    if token.is_empty() {
        return Err(RpcError::Protocol(
            "refusing non-loopback bind without --token / ALLPACA_RPC_TOKEN".into(),
        ));
    }
    Ok(())
}

pub fn encode_hello_reply(r: &HelloReply) -> Vec<u8> {
    let mut w = Vec::new();
    write_u16(&mut w, r.major);
    write_u16(&mut w, r.minor);
    write_u16(&mut w, r.patch);
    write_u32(&mut w, r.kernel_abi_rev);
    write_u8(&mut w, r.backend as u8);
    write_u32(&mut w, r.dtype_mask);
    write_str(&mut w, &r.device_name);
    write_u64(&mut w, r.memory_bytes);
    write_u32(&mut w, r.devices.len() as u32);
    for bytes in &r.devices {
        write_u64(&mut w, *bytes);
    }
    w
}

/// Append the names of a worker's extensions to its encoded HELLO reply.
pub fn append_hello_extensions(reply: &mut Vec<u8>, names: &[&str]) {
    write_u32(reply, names.len() as u32);
    for name in names {
        write_str(reply, name);
    }
}

/// The extension names after a HELLO reply's device list (none from a
/// worker that does not append them).
pub fn decode_hello_extensions(data: &[u8]) -> Result<Vec<String>, RpcError> {
    let mut c = Cursor::new(data);
    let _ = (c.u16()?, c.u16()?, c.u16()?, c.u32()?, c.u8()?, c.u32()?);
    c.str()?;
    c.u64()?;
    if !c.has_remaining() {
        return Ok(Vec::new());
    }
    for _ in 0..c.u32()? {
        c.u64()?;
    }
    if !c.has_remaining() {
        return Ok(Vec::new());
    }
    let n = c.u32()? as usize;
    if n > 64 {
        return Err(RpcError::Protocol("too many rpc extensions".into()));
    }
    (0..n).map(|_| c.str()).collect()
}

/// The parts of an extension call's payload: name, head, bulk data.
pub fn decode_ext(payload: &[u8]) -> Result<(String, &[u8], &[u8]), RpcError> {
    let mut c = Cursor::new(payload);
    let name = c.str()?;
    let head = c.bytes()?;
    Ok((name, head, &payload[c.off..]))
}

/// Send an extension call without copying its bulk `data`.
pub fn write_ext_request<W: Write>(w: &mut W, name: &str, head: &[u8], data: &[u8]) -> Result<(), RpcError> {
    let mut front = vec![Cmd::Ext as u8];
    write_u64(&mut front, (16 + name.len() + head.len() + data.len()) as u64);
    write_str(&mut front, name);
    write_bytes(&mut front, head);
    if data.len() <= ONE_WRITE_FRAME {
        front.extend_from_slice(data);
        w.write_all(&front)?;
    } else {
        w.write_all(&front)?;
        w.write_all(data)?;
    }
    w.flush()?;
    Ok(())
}

pub fn decode_hello_reply(data: &[u8]) -> Result<HelloReply, RpcError> {
    let mut c = Cursor::new(data);
    let major = c.u16()?;
    let minor = c.u16()?;
    let patch = c.u16()?;
    let kernel_abi_rev = c.u32()?;
    let backend =
        RpcBackendKind::from_u8(c.u8()?).ok_or_else(|| RpcError::Protocol("bad backend".into()))?;
    let dtype_mask = c.u32()?;
    let device_name = c.str()?;
    let memory_bytes = c.u64()?;
    let mut devices = Vec::new();
    if c.has_remaining() {
        let n = c.u32()? as usize;
        if n > 64 {
            return Err(RpcError::Protocol("too many rpc devices".into()));
        }
        devices.reserve(n);
        for _ in 0..n {
            devices.push(c.u64()?);
        }
    }
    Ok(HelloReply {
        major,
        minor,
        patch,
        kernel_abi_rev,
        backend,
        dtype_mask,
        device_name,
        memory_bytes,
        devices,
    })
}

pub fn encode_launch_list(list: &LaunchList) -> Vec<u8> {
    let mut w = Vec::new();
    write_u32(&mut w, list.items.len() as u32);
    for item in &list.items {
        write_u8(&mut w, item.op as u8);
        write_u8(&mut w, item.bufs.len() as u8);
        for b in &item.bufs {
            write_buf_id(&mut w, *b);
        }
        write_u8(&mut w, item.i32s.len() as u8);
        for v in &item.i32s {
            write_i32(&mut w, *v);
        }
        write_u8(&mut w, item.f32s.len() as u8);
        for v in &item.f32s {
            write_f32(&mut w, *v);
        }
    }
    w
}

/// `xs[i]` is the activation buffer for `lists[i]`. `activation` is written into `xs[0]`.
pub fn encode_run_span(
    xs: &[BufId],
    lists: &[LaunchList],
    activation: &[u8],
) -> Result<Vec<u8>, RpcError> {
    if xs.is_empty() || xs.len() != lists.len() {
        return Err(RpcError::Protocol("run span length".into()));
    }
    let mut w = Vec::new();
    write_u32(&mut w, xs.len() as u32);
    for (x, list) in xs.iter().zip(lists) {
        write_buf_id(&mut w, *x);
        write_bytes(&mut w, &encode_launch_list(list));
    }
    write_bytes(&mut w, activation);
    Ok(w)
}

pub fn decode_run_span(data: &[u8]) -> Result<(Vec<(BufId, LaunchList)>, Vec<u8>), RpcError> {
    let mut c = Cursor::new(data);
    let n = c.u32()? as usize;
    if n == 0 {
        return Err(RpcError::Protocol("empty run span".into()));
    }
    let mut steps = Vec::with_capacity(n);
    for _ in 0..n {
        let x = c.buf_id()?;
        let encoded = c.bytes()?;
        steps.push((x, decode_launch_list(encoded)?));
    }
    let activation = c.bytes()?.to_vec();
    Ok((steps, activation))
}

pub fn decode_launch_list(data: &[u8]) -> Result<LaunchList, RpcError> {
    let mut c = Cursor::new(data);
    let n = c.u32()? as usize;
    let mut items = Vec::with_capacity(n);
    for _ in 0..n {
        let op =
            KernelOp::from_u8(c.u8()?).ok_or_else(|| RpcError::Protocol("bad kernel op".into()))?;
        let nb = c.u8()? as usize;
        let mut bufs = Vec::with_capacity(nb);
        for _ in 0..nb {
            bufs.push(c.buf_id()?);
        }
        let ni = c.u8()? as usize;
        let mut i32s = Vec::with_capacity(ni);
        for _ in 0..ni {
            i32s.push(c.i32()?);
        }
        let nf = c.u8()? as usize;
        let mut f32s = Vec::with_capacity(nf);
        for _ in 0..nf {
            f32s.push(c.f32()?);
        }
        items.push(LaunchItem {
            op,
            bufs,
            i32s,
            f32s,
        });
    }
    Ok(LaunchList { items })
}

/// Length-prefixed request: cmd u8 | size u64 | payload.
pub fn write_request<W: Write>(w: &mut W, cmd: Cmd, payload: &[u8]) -> Result<(), RpcError> {
    write_frame(w, cmd as u8, payload)
}

/// Frames up to this size leave in a single write.
const ONE_WRITE_FRAME: usize = 64 * 1024;

/// Tag u8 | size u64 | payload. The sockets run with TCP_NODELAY, so every
/// write is its own segment: writing the tag, the size and the payload
/// separately put three segments on the wire per message. A frame that fits
/// is copied into one buffer and sent with one write; a large one (a tensor
/// upload) sends the 9-byte head and then the payload, without the copy.
fn write_frame<W: Write>(w: &mut W, tag: u8, payload: &[u8]) -> Result<(), RpcError> {
    let mut head = [0u8; 9];
    head[0] = tag;
    head[1..].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    if payload.len() <= ONE_WRITE_FRAME {
        let mut frame = Vec::with_capacity(9 + payload.len());
        frame.extend_from_slice(&head);
        frame.extend_from_slice(payload);
        w.write_all(&frame)?;
    } else {
        w.write_all(&head)?;
        w.write_all(payload)?;
    }
    w.flush()?;
    Ok(())
}

pub fn read_request<R: Read>(r: &mut R) -> Result<(Cmd, Vec<u8>), RpcError> {
    let mut head = [0u8; 9];
    r.read_exact(&mut head)?;
    let cmd = Cmd::from_u8(head[0]).ok_or_else(|| RpcError::Protocol("bad cmd".into()))?;
    let size_bytes: [u8; 8] = head[1..9]
        .try_into()
        .map_err(|_| RpcError::Protocol("bad size".into()))?;
    let size = u64::from_le_bytes(size_bytes) as usize;
    let mut payload = vec![0u8; size];
    if size > 0 {
        r.read_exact(&mut payload)?;
    }
    Ok((cmd, payload))
}

/// Reply: status u8 | size u64 | payload.
pub fn write_reply<W: Write>(w: &mut W, status: u8, payload: &[u8]) -> Result<(), RpcError> {
    write_frame(w, status, payload)
}

pub fn read_reply<R: Read>(r: &mut R) -> Result<(u8, Vec<u8>), RpcError> {
    let mut head = [0u8; 9];
    r.read_exact(&mut head)?;
    let status = head[0];
    let size_bytes: [u8; 8] = head[1..9]
        .try_into()
        .map_err(|_| RpcError::Protocol("bad size".into()))?;
    let size = u64::from_le_bytes(size_bytes) as usize;
    let mut payload = vec![0u8; size];
    if size > 0 {
        r.read_exact(&mut payload)?;
    }
    Ok((status, payload))
}

pub fn check_hello(reply: &HelloReply) -> Result<(), RpcError> {
    if reply.major != PROTO_MAJOR {
        return Err(RpcError::Version {
            major: reply.major,
            minor: reply.minor,
            patch: reply.patch,
        });
    }
    if reply.minor > PROTO_MINOR {
        return Err(RpcError::Version {
            major: reply.major,
            minor: reply.minor,
            patch: reply.patch,
        });
    }
    if reply.kernel_abi_rev != KERNEL_ABI_REV {
        return Err(RpcError::Protocol(format!(
            "worker kernel abi {} != client {}",
            reply.kernel_abi_rev, KERNEL_ABI_REV
        )));
    }
    Ok(())
}

#[cfg(test)]
mod frame_tests {
    use super::*;

    /// Records every write call, as a TCP_NODELAY socket would send them.
    #[derive(Default)]
    struct Segments(Vec<Vec<u8>>);

    impl Write for Segments {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.push(buf.to_vec());
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_request_and_its_reply_each_leave_in_one_write() {
        let payload: Vec<u8> = (0..8300u32).map(|i| i as u8).collect();
        let mut wire = Segments::default();
        write_request(&mut wire, Cmd::Hello, &payload).unwrap();
        assert_eq!(wire.0.len(), 1, "request writes");
        let (cmd, body) = read_request(&mut wire.0[0].as_slice()).unwrap();
        assert!(matches!(cmd, Cmd::Hello));
        assert_eq!(body, payload);

        let mut wire = Segments::default();
        write_reply(&mut wire, STATUS_OK, &payload).unwrap();
        assert_eq!(wire.0.len(), 1, "reply writes");
        let (status, body) = read_reply(&mut wire.0[0].as_slice()).unwrap();
        assert_eq!(status, STATUS_OK);
        assert_eq!(body, payload);
    }

    #[test]
    fn a_large_frame_is_not_copied_and_still_round_trips() {
        let payload = vec![7u8; 300 * 1024];
        let mut wire = Segments::default();
        write_request(&mut wire, Cmd::Hello, &payload).unwrap();
        assert_eq!(wire.0.len(), 2, "head, then the payload as given");
        let bytes = wire.0.concat();
        let (_, body) = read_request(&mut bytes.as_slice()).unwrap();
        assert_eq!(body, payload);
    }
}
