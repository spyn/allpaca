use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};

use crate::protocol::{
    check_hello, decode_hello_reply, encode_hello_req, encode_launch_list, read_reply,
    write_request, BufId, Cmd, HelloReply, HelloRequest, LaunchList, RpcError, KERNEL_ABI_REV,
    PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH, STATUS_OK,
};

/// Coordinator-side TCP session to one `rpc-server`.
pub struct RpcSession {
    stream: TcpStream,
    hello: HelloReply,
    endpoint: String,
    token: String,
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
        check_hello(&hello)?;
        if hello.kernel_abi_rev == 0 {
            return Err(RpcError::Protocol("worker kernel abi is 0".into()));
        }
        let _ = KERNEL_ABI_REV;
        self.hello = hello;
        Ok(())
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn hello(&self) -> &HelloReply {
        &self.hello
    }

    fn rpc(&mut self, cmd: Cmd, payload: &[u8]) -> Result<Vec<u8>, RpcError> {
        write_request(&mut self.stream, cmd, payload)?;
        let (status, body) = read_reply(&mut self.stream)?;
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

    pub fn sync(&mut self) -> Result<(), RpcError> {
        self.rpc(Cmd::Sync, &[])?;
        Ok(())
    }
}

impl Read for RpcSession {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.stream.read(buf)
    }
}

impl Write for RpcSession {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}
