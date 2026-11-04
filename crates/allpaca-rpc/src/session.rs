use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

/// Calls to one worker per operation since connect: commands by name
/// (`MoeFfn`, `RunSpan`, ...), extension calls as `<extension>:<op>` where
/// `<op>` is the first word of the head (the variant of a JSON enum).
/// Readable while the session is busy (the lock is held only to add).
#[derive(Debug, Default)]
pub struct OpCounters(Mutex<BTreeMap<String, OpCount>>);

/// One operation's calls: `calls` waited for their reply, `posted` did not
/// ([`RpcSession::ext_post`]). `wait_us` is the coordinator's wait for the
/// replies of `calls`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpCount {
    pub op: String,
    pub calls: u64,
    pub posted: u64,
    pub wait_us: u64,
}

impl OpCounters {
    fn add(&self, op: &str, calls: u64, posted: u64, wait_us: u64) {
        let Ok(mut map) = self.0.lock() else {
            return;
        };
        let entry = map.entry(op.to_string()).or_insert_with(|| OpCount {
            op: op.to_string(),
            ..OpCount::default()
        });
        entry.calls += calls;
        entry.posted += posted;
        entry.wait_us += wait_us;
    }

    /// Every operation seen so far, by name.
    pub fn snapshot(&self) -> Vec<OpCount> {
        self.0
            .lock()
            .map(|map| map.values().cloned().collect())
            .unwrap_or_default()
    }
}

/// The operation an extension call names: the first word of its head (a
/// JSON enum's variant), else its first byte.
fn ext_op(name: &str, head: &[u8]) -> String {
    let text = head
        .iter()
        .position(|&b| b != b'{' && b != b'"')
        .map(|start| &head[start..])
        .unwrap_or(&[]);
    let end = text
        .iter()
        .position(|b| !(b.is_ascii_alphanumeric() || *b == b'_'))
        .unwrap_or(text.len());
    match (end, head.first()) {
        (0, Some(b)) => format!("{name}:op{b}"),
        (0, None) => format!("{name}:"),
        _ => format!("{name}:{}", String::from_utf8_lossy(&text[..end])),
    }
}

/// What a call sent with [`RpcSession::ext_post`] came to. Known once a
/// later call on the session has read its reply: every call that waits
/// reads the posted replies before its own (the worker answers in order).
#[derive(Debug, Clone, Default)]
pub struct Posted(Arc<Mutex<Option<Result<(), String>>>>);

impl Posted {
    /// A handle for a call that has already been answered.
    pub fn settled(outcome: Result<(), String>) -> Self {
        Self(Arc::new(Mutex::new(Some(outcome))))
    }

    /// `None` while the reply is unread.
    pub fn outcome(&self) -> Option<Result<(), String>> {
        self.0.lock().ok().and_then(|o| o.clone())
    }

    /// The worker's error, once the reply is read and it was one.
    pub fn failed(&self) -> Option<String> {
        self.outcome().and_then(Result::err)
    }

    fn settle(&self, outcome: Result<(), String>) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(outcome);
        }
    }
}

/// A reply the worker still owes, in the order the calls went out.
#[derive(Debug)]
enum Due {
    /// [`RpcSession::ext_send`]: read by [`RpcSession::ext_recv`].
    Wanted(String),
    /// [`RpcSession::ext_post`]: read by whatever waits next.
    Posted(String, Posted),
}

/// Posted replies left unread before a post reads them itself, so a
/// session that only posts cannot fill the socket buffers.
const MAX_POSTED: usize = 256;

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
    /// Calls per operation.
    ops: Arc<OpCounters>,
    /// Replies still owed for calls sent ahead, oldest first.
    due: VecDeque<Due>,
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
            ops: Arc::default(),
            due: VecDeque::new(),
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
        self.write_ext(name, head, data)?;
        let op = ext_op(name, head);
        self.ops.add(&op, 1, 0, 0);
        self.due.push_back(Due::Wanted(op));
        Ok(())
    }

    /// Send a call to the extension `name` whose reply nobody waits for.
    /// The worker runs it before any call sent after it; the next call
    /// that waits reads its reply first, and from then on the returned
    /// handle tells how it went. For calls whose reply the next step does
    /// not use (copying or dropping a sequence's state): whatever depends
    /// on the call checks the handle after its own next reply.
    pub fn ext_post(&mut self, name: &str, head: &[u8], data: &[u8]) -> Result<Posted, RpcError> {
        if self.due.len() >= MAX_POSTED {
            self.read_posted()?;
        }
        self.write_ext(name, head, data)?;
        let op = ext_op(name, head);
        self.ops.add(&op, 0, 1, 0);
        let posted = Posted::default();
        self.due.push_back(Due::Posted(op, posted.clone()));
        Ok(posted)
    }

    fn write_ext(&mut self, name: &str, head: &[u8], data: &[u8]) -> Result<(), RpcError> {
        if !self.has_extension(name) {
            return Err(RpcError::Protocol(format!("extension {name} is not served")));
        }
        self.ext_calls
            .bytes_out
            .fetch_add((head.len() + data.len()) as u64, Ordering::Relaxed);
        crate::protocol::write_ext_request(&mut self.stream, name, head, data)
    }

    /// Read the replies of posted calls at the front of the queue and
    /// settle their handles; stop at a reply someone waits for. Errors
    /// only on I/O: a failed call is reported on its handle.
    fn read_posted(&mut self) -> Result<(), RpcError> {
        while matches!(self.due.front(), Some(Due::Posted(..))) {
            let Some(Due::Posted(op, posted)) = self.due.pop_front() else {
                break;
            };
            let (status, body) = read_reply(&mut self.stream)?;
            self.ext_calls.calls.fetch_add(1, Ordering::Relaxed);
            self.ext_calls.bytes_in.fetch_add(body.len() as u64, Ordering::Relaxed);
            if status == STATUS_OK {
                posted.settle(Ok(()));
            } else {
                let error = String::from_utf8_lossy(&body).into_owned();
                if Arc::strong_count(&posted.0) == 1 {
                    tracing::warn!(op, error, endpoint = %self.endpoint, "a posted rpc call failed");
                }
                posted.settle(Err(error));
            }
        }
        Ok(())
    }

    /// Read every posted reply the worker owes ahead of any reply someone
    /// waits for. Errors only on I/O.
    pub fn settle_posted(&mut self) -> Result<(), RpcError> {
        self.read_posted()
    }

    /// The reply to the oldest call sent with [`Self::ext_send`]. Replies
    /// of calls posted before it are read first.
    pub fn ext_recv(&mut self) -> Result<Vec<u8>, RpcError> {
        let started = std::time::Instant::now();
        self.read_posted()?;
        let Some(Due::Wanted(op)) = self.due.pop_front() else {
            return Err(RpcError::Protocol("no extension reply is due".into()));
        };
        let (status, body) = read_reply(&mut self.stream)?;
        self.ext_calls.add(started, 0, body.len());
        self.ops.add(&op, 0, 0, started.elapsed().as_micros() as u64);
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

    /// Calls per operation ([`OpCounters`]); the handle stays valid while
    /// the session is busy.
    pub fn op_counters(&self) -> Arc<OpCounters> {
        Arc::clone(&self.ops)
    }

    fn rpc(&mut self, cmd: Cmd, payload: &[u8]) -> Result<Vec<u8>, RpcError> {
        if self.due.iter().any(|d| matches!(d, Due::Wanted(_))) {
            return Err(RpcError::Protocol(format!(
                "{cmd:?} sent while a pipelined extension reply is unread"
            )));
        }
        let started = std::time::Instant::now();
        write_request(&mut self.stream, cmd, payload)?;
        // The worker answers in order: posted calls' replies come first.
        self.read_posted()?;
        let (status, body) = read_reply(&mut self.stream)?;
        self.calls.add(started, payload.len(), body.len());
        self.ops.add(&format!("{cmd:?}"), 1, 0, started.elapsed().as_micros() as u64);
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

impl Drop for RpcSession {
    /// Read the replies still owed before closing: a socket closed with
    /// unread data resets the connection, and the worker would see an
    /// error instead of the end of the session.
    fn drop(&mut self) {
        if self.due.is_empty() {
            return;
        }
        let _ = self.stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
        while self.due.pop_front().is_some() {
            if read_reply(&mut self.stream).is_err() {
                break;
            }
        }
    }
}

impl Read for RpcSession {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // A raw stream comes after the replies the worker already owes.
        self.read_posted().map_err(std::io::Error::other)?;
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
