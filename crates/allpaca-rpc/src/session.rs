use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::protocol::{
    check_hello, decode_hello_reply, encode_hello_req, encode_launch_list, read_reply,
    write_request, BufId, Cmd, HelloReply, HelloRequest, LaunchList, RpcError, KERNEL_ABI_REV,
    PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH, STATUS_OK,
};

/// Calls to one worker since connect, readable while the session is busy.
/// `wall_us` is coordinator wall time per call: network plus worker work.
#[derive(Debug, Default)]
pub struct CallCounters {
    pub calls: AtomicU64,
    pub wall_us: AtomicU64,
    pub bytes_out: AtomicU64,
    pub bytes_in: AtomicU64,
}

impl CallCounters {
    fn add(&self, started: std::time::Instant, bytes_out: usize, bytes_in: usize) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.wall_us
            .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.bytes_out.fetch_add(bytes_out as u64, Ordering::Relaxed);
        self.bytes_in.fetch_add(bytes_in as u64, Ordering::Relaxed);
    }
}

/// Coordinator-side TCP session to one `rpc-server`.
pub struct RpcSession {
    stream: TcpStream,
    hello: HelloReply,
    endpoint: String,
    token: String,
    /// Extensions the worker announced in its HELLO.
    extensions: Vec<String>,
    /// The worker's [`crate::BUILD_ID`], when it sent one.
    build_id: Option<String>,
    /// Request/reply commands (MoeFfn, RunSpan, uploads, ...).
    calls: Arc<CallCounters>,
    /// Extension calls and raw streams: `wall_us` counts only the wait for
    /// each reply, since pipelined calls overlap.
    ext_calls: Arc<CallCounters>,
}

impl RpcSession {
    pub fn connect(addr: impl ToSocketAddrs + std::fmt::Display) -> Result<Self, RpcError> {
        let endpoint = addr.to_string();
        let addrs: Vec<_> = addr.to_socket_addrs().map_err(RpcError::from)?.collect();
        if addrs.is_empty() {
            return Err(RpcError::Protocol(format!("no addresses for {endpoint}")));
        }
        Self::connect_addr(&addrs[0].to_string())
    }

    pub fn connect_addr(endpoint: &str) -> Result<Self, RpcError> {
        Self::connect_addr_with_token(endpoint, "")
    }

    pub fn connect_addr_with_token(endpoint: &str, token: &str) -> Result<Self, RpcError> {
        let stream = TcpStream::connect(endpoint)?;
        stream.set_nodelay(true)?;
        crate::protocol::widen_socket_buffers(&stream);
        let mut session = Self {
            stream,
            hello: HelloReply {
                major: 0,
                minor: 0,
                patch: 0,
                kernel_abi_rev: 0,
                backend: crate::RpcBackendKind::Cpu,
                dtype_mask: 0,
                device_name: String::new(),
                memory_bytes: 0,
                devices: Vec::new(),
            },
            endpoint: endpoint.to_string(),
            token: token.to_string(),
            extensions: Vec::new(),
            build_id: None,
            calls: Arc::default(),
            ext_calls: Arc::default(),
        };
        session.handshake()?;
        Ok(session)
    }

    fn handshake(&mut self) -> Result<(), RpcError> {
        let req = HelloRequest {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            caps: 0,
            token: self.token.clone(),
        };
        let payload = encode_hello_req(&req);
        write_request(&mut self.stream, Cmd::Hello, &payload)?;
        let (status, body) = read_reply(&mut self.stream)?;
        if status != STATUS_OK {
            return Err(RpcError::Worker(
                String::from_utf8_lossy(&body).into_owned(),
            ));
        }
        let hello = decode_hello_reply(&body)?;
        (self.extensions, self.build_id) = crate::protocol::decode_hello_tail(&body)?;
        check_hello(&hello)?;
        if hello.kernel_abi_rev == 0 {
            return Err(RpcError::Protocol("worker kernel abi is 0".into()));
        }
        let _ = KERNEL_ABI_REV;
        self.hello = hello;
        Ok(())
    }

    /// The fingerprint of the sources the worker was built from; `None`
    /// from a worker too old to send it.
    pub fn build_id(&self) -> Option<&str> {
        self.build_id.as_deref()
    }

    /// The worker was built from the sources this side was built from.
    pub fn same_build(&self) -> bool {
        self.build_id() == Some(crate::BUILD_ID)
    }

    /// The worker announced the extension `name` in its HELLO.
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|e| e == name)
    }

    /// Call the worker's extension `name` with a small `head` and a bulk
    /// `data` payload.
    pub fn ext(&mut self, name: &str, head: &[u8], data: &[u8]) -> Result<Vec<u8>, RpcError> {
        self.ext_send(name, head, data)?;
        self.ext_recv()
    }

    /// Send a call to the extension `name` without waiting for its reply.
    /// The worker answers calls in the order they were sent; each one's
    /// reply is read with [`Self::ext_recv`], and nothing else may use the
    /// session in between. Keep what is in flight well under the socket
    /// buffers ([`SOCKET_BUFFER`]).
    pub fn ext_send(&mut self, name: &str, head: &[u8], data: &[u8]) -> Result<(), RpcError> {
        if !self.has_extension(name) {
            return Err(RpcError::Protocol(format!("extension {name} is not served")));
        }
        self.ext_calls
            .bytes_out
            .fetch_add((head.len() + data.len()) as u64, Ordering::Relaxed);
        crate::protocol::write_ext_request(&mut self.stream, name, head, data)
    }

    /// The reply to the oldest call sent with [`Self::ext_send`].
    pub fn ext_recv(&mut self) -> Result<Vec<u8>, RpcError> {
        let started = std::time::Instant::now();
        let (status, body) = read_reply(&mut self.stream)?;
        self.ext_calls.add(started, 0, body.len());
        if status != STATUS_OK {
            return Err(RpcError::Worker(String::from_utf8_lossy(&body).into_owned()));
        }
        Ok(body)
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn hello(&self) -> &HelloReply {
        &self.hello
    }

    /// Counters for this session: `(request/reply commands, extension calls
    /// and raw streams)`. The handles stay valid while the session is busy.
    pub fn counters(&self) -> (Arc<CallCounters>, Arc<CallCounters>) {
        (Arc::clone(&self.calls), Arc::clone(&self.ext_calls))
    }

    fn rpc(&mut self, cmd: Cmd, payload: &[u8]) -> Result<Vec<u8>, RpcError> {
        let started = std::time::Instant::now();
        write_request(&mut self.stream, cmd, payload)?;
        let (status, body) = read_reply(&mut self.stream)?;
        self.calls.add(started, payload.len(), body.len());
        if status != STATUS_OK {
            return Err(RpcError::Worker(
                String::from_utf8_lossy(&body).into_owned(),
            ));
        }
        Ok(body)
    }

    pub fn alloc(&mut self, bytes: u64) -> Result<BufId, RpcError> {
        self.alloc_on(bytes, None)
    }

    /// `device` selects a GPU on a multi-device worker. `None` lets the worker place it.
    pub fn alloc_on(&mut self, bytes: u64, device: Option<u32>) -> Result<BufId, RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_u64(&mut p, bytes);
        if let Some(device) = device {
            crate::protocol::write_u32(&mut p, device);
        }
        let body = self.rpc(Cmd::Alloc, &p)?;
        let mut c = crate::protocol::Cursor::new(&body);
        c.buf_id()
    }

    pub fn free(&mut self, id: BufId) -> Result<(), RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_buf_id(&mut p, id);
        self.rpc(Cmd::Free, &p)?;
        Ok(())
    }

    pub fn clear(&mut self, id: BufId, value: u8) -> Result<(), RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_buf_id(&mut p, id);
        crate::protocol::write_u8(&mut p, value);
        self.rpc(Cmd::Clear, &p)?;
        Ok(())
    }

    pub fn h2d(&mut self, id: BufId, offset: u64, data: &[u8]) -> Result<(), RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_buf_id(&mut p, id);
        crate::protocol::write_u64(&mut p, offset);
        crate::protocol::write_bytes(&mut p, data);
        self.rpc(Cmd::H2D, &p)?;
        Ok(())
    }

    pub fn h2d_hash(&mut self, id: BufId, offset: u64, hash: u64) -> Result<bool, RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_buf_id(&mut p, id);
        crate::protocol::write_u64(&mut p, offset);
        crate::protocol::write_u64(&mut p, hash);
        let body = self.rpc(Cmd::H2DHash, &p)?;
        Ok(!body.is_empty() && body[0] != 0)
    }

    pub fn d2h(&mut self, id: BufId, offset: u64, size: u64) -> Result<Vec<u8>, RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_buf_id(&mut p, id);
        crate::protocol::write_u64(&mut p, offset);
        crate::protocol::write_u64(&mut p, size);
        self.rpc(Cmd::D2H, &p)
    }

    pub fn copy(&mut self, src: BufId, dst: BufId) -> Result<(), RpcError> {
        let mut p = Vec::new();
        crate::protocol::write_buf_id(&mut p, src);
        crate::protocol::write_buf_id(&mut p, dst);
        self.rpc(Cmd::Copy, &p)?;
        Ok(())
    }

    pub fn launch(&mut self, list: &LaunchList) -> Result<(), RpcError> {
        let payload = encode_launch_list(list);
        self.rpc(Cmd::LaunchList, &payload)?;
        Ok(())
    }

    /// Upload packed GGUF bytes. The worker records `ggml_type` on the new buffer.
    pub fn put_tensor(
        &mut self,
        ggml_type: u32,
        dims: &[u32],
        bytes: &[u8],
        device: Option<u32>,
    ) -> Result<BufId, RpcError> {
        if dims.len() > 4 {
            return Err(RpcError::Protocol("put tensor rank".into()));
        }
        let mut p = Vec::new();
        crate::protocol::write_u32(&mut p, ggml_type);
        crate::protocol::write_u32(&mut p, dims.len() as u32);
        for d in dims {
            crate::protocol::write_u32(&mut p, *d);
        }
        crate::protocol::write_bytes(&mut p, bytes);
        if let Some(device) = device {
            crate::protocol::write_u32(&mut p, device);
        }
        let body = self.rpc(Cmd::PutTensor, &p)?;
        let mut c = crate::protocol::Cursor::new(&body);
        c.buf_id()
    }

    /// Upload `activation` into the first buffer, run every list, return the last buffer.
    pub fn run_span(
        &mut self,
        xs: &[BufId],
        lists: &[LaunchList],
        activation: &[u8],
    ) -> Result<Vec<u8>, RpcError> {
        let payload = crate::protocol::encode_run_span(xs, lists, activation)?;
        self.rpc(Cmd::RunSpan, &payload)
    }

    /// Routing-weighted SwiGLU of the chosen experts per row (one round trip).
    /// `q8` sends `x` as the worker's int8 blocks (see
    /// [`crate::protocol::quantize_q8_rows`]): lossless for K-quant experts,
    /// whose fused kernels quantize to exactly that.
    pub fn moe_ffn(
        &mut self,
        req: crate::protocol::MoeFfnRequest,
        x: &[f32],
        q8: bool,
        pairs: &[(u32, u32, f32)],
    ) -> Result<Vec<f32>, RpcError> {
        let body = self.rpc(Cmd::MoeFfn, &crate::protocol::encode_moe_ffn(&req, x, q8, pairs))?;
        crate::protocol::decode_moe_reply(&body, x.len(), req.f16_out)
            .ok_or_else(|| RpcError::Protocol("moe ffn reply size".into()))
    }

    pub fn mem_info(&mut self) -> Result<(u64, u64), RpcError> {
        let body = self.rpc(Cmd::MemInfo, &[])?;
        let mut c = crate::protocol::Cursor::new(&body);
        let free = c.u64()?;
        let total = c.u64()?;
        Ok((free, total))
    }

    pub fn device_count(&mut self) -> Result<u32, RpcError> {
        let body = self.rpc(Cmd::DeviceCount, &[])?;
        let mut c = crate::protocol::Cursor::new(&body);
        c.u32()
    }

    /// Live stats for each of the worker's devices.
    pub fn device_stats(&mut self) -> Result<Vec<crate::protocol::DeviceStat>, RpcError> {
        let body = self.rpc(Cmd::DeviceStats, &[])?;
        crate::protocol::decode_device_stats(&body)
    }

    pub fn sync(&mut self) -> Result<(), RpcError> {
        self.rpc(Cmd::Sync, &[])?;
        Ok(())
    }
}

impl Read for RpcSession {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.stream.read(buf)?;
        self.ext_calls.bytes_in.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

impl Write for RpcSession {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.stream.write(buf)?;
        self.ext_calls.bytes_out.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}
