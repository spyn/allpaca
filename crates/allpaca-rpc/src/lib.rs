//! Kernel-ABI RPC: remote buffers plus batched named launches.
//!
//! This is the llama.cpp RPC analogue without a ggml graph. The coordinator
//! owns scheduling; a worker only allocates buffers and runs the existing
//! CPU/HIP kernel ABI.

mod protocol;
mod session;
mod worker;

pub use protocol::{
    encode_hello_reply, fnv1a64, write_reply, BufId, Cmd, HelloReply, HelloRequest, KernelOp,
    LaunchItem, LaunchList, RpcBackendKind, KERNEL_ABI_REV, PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH,
    STATUS_ERR, STATUS_OK,
};
pub use session::RpcSession;
pub use worker::{serve_connection, serve_listener, serve_tcp, CpuWorker, WorkerBackend};

#[cfg(feature = "hip")]
pub use worker::HipWorker;

pub use protocol::RpcError;

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use super::*;
    use crate::protocol::{decode_launch_list, encode_hello_req, encode_launch_list};

    #[test]
    fn codec_hello_and_launch_roundtrip() {
        let req = HelloRequest {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            caps: 3,
        };
        let bytes = encode_hello_req(&req);
        let back = crate::protocol::decode_hello_req(&bytes).unwrap();
        assert_eq!(req, back);

        let list = LaunchList {
            items: vec![LaunchItem {
                op: KernelOp::GemmNt,
                bufs: vec![BufId::new(1, 1), BufId::new(2, 1), BufId::new(3, 1)],
                i32s: vec![2, 3, 4],
                f32s: vec![],
            }],
        };
        let enc = encode_launch_list(&list);
        let dec = decode_launch_list(&enc).unwrap();
        assert_eq!(list, dec);
    }

    #[test]
    fn fnv_stable() {
        assert_ne!(fnv1a64(b"abc"), fnv1a64(b"abd"));
    }

    #[test]
    fn loopback_gemm_nt() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        assert_eq!(client.hello().major, PROTO_MAJOR);
        let (free, total) = client.mem_info().unwrap();
        assert!(total >= free);

        // C = A (2x2) * B^T (2x2 stored as 2x2 NT) → identity-ish
        let m = 2i32;
        let n = 2i32;
        let k = 2i32;
        let a = client.alloc(8 * 4).unwrap();
        let b = client.alloc(8 * 4).unwrap();
        let c = client.alloc(8 * 4).unwrap();
        let a_data: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_data = a_data.clone();
        client.h2d(a, 0, &a_data).unwrap();
        client.h2d(b, 0, &b_data).unwrap();
        client
            .launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::GemmNt,
                    bufs: vec![a, b, c],
                    i32s: vec![m, n, k],
                    f32s: vec![],
                }],
            })
            .unwrap();
        client.sync().unwrap();
        let out = client.d2h(c, 0, 16).unwrap();
        let c00 = f32::from_le_bytes(out[0..4].try_into().unwrap());
        assert!((c00 - 1.0).abs() < 1e-5, "c00={c00}");
        client.free(a).unwrap();
        client.free(b).unwrap();
        client.free(c).unwrap();
        drop(client);
        worker.join().unwrap();
    }
}
