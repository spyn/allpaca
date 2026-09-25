use std::io::{self, Read, Write};

use thiserror::Error;

/// Protocol major; must match on both sides (llama.cpp HELLO rule).
pub const PROTO_MAJOR: u16 = 1;
pub const PROTO_MINOR: u16 = 0;
pub const PROTO_PATCH: u16 = 0;
/// Bump when the named kernel set or C ABI revision changes.
/// 2 adds `PutTensor` and `GemmNtQuant` (packed GGUF weights).
pub const KERNEL_ABI_REV: u32 = 2;

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
    w.write_all(&[cmd as u8])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
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
    w.write_all(&[status])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()?;
    Ok(())
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
