//! Kernel-ABI RPC: remote buffers plus batched named launches.
//!
//! This is the llama.cpp RPC analogue without a ggml graph. The coordinator
//! owns scheduling; a worker only allocates buffers and runs the existing
//! CPU/HIP/CUDA kernel ABI. HIP and CUDA workers speak the same `KernelOp`
//! wire protocol so an AMD coordinator can drive an NVIDIA worker (and vice
//! versa) as long as HELLO majors match.

mod protocol;
mod session;
mod worker;

pub use protocol::{
    check_rpc_bind, encode_hello_reply, fnv1a64, is_loopback_bind_host, write_reply, BufId, Cmd,
    HelloReply, HelloRequest, KernelOp, LaunchItem, LaunchList, RpcBackendKind, KERNEL_ABI_REV,
    PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH, STATUS_ERR, STATUS_OK,
};
pub use session::RpcSession;
pub use worker::{
    serve_connection, serve_connection_with_token, serve_listener, serve_listener_with_token,
    serve_tcp, CpuWorker, WorkerBackend,
};

#[cfg(feature = "hip")]
pub use worker::HipWorker;

#[cfg(feature = "cuda")]
pub use worker::CudaWorker;

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
            token: String::new(),
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

    #[test]
    fn hello_token_roundtrip() {
        let req = HelloRequest {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            caps: 0,
            token: "s3cret".into(),
        };
        let back = crate::protocol::decode_hello_req(&encode_hello_req(&req)).unwrap();
        assert_eq!(req, back);
    }

    #[test]
    fn loopback_bind_hosts() {
        assert!(is_loopback_bind_host("127.0.0.1"));
        assert!(is_loopback_bind_host("localhost"));
        assert!(is_loopback_bind_host("::1"));
        assert!(is_loopback_bind_host("[::1]"));
        assert!(!is_loopback_bind_host("0.0.0.0"));
        assert!(!is_loopback_bind_host("10.0.0.2"));
        assert!(check_rpc_bind("127.0.0.1", false, "").is_ok());
        assert!(check_rpc_bind("0.0.0.0", false, "tok").is_err());
        assert!(check_rpc_bind("0.0.0.0", true, "").is_err());
        assert!(check_rpc_bind("0.0.0.0", true, "tok").is_ok());
    }

    #[test]
    fn hello_token_mismatch_rejects() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection_with_token(
                stream,
                WorkerBackend::Cpu(CpuWorker::new(8 << 20)),
                Some("expected"),
            )
        });
        let err = match RpcSession::connect_addr_with_token(&addr.to_string(), "wrong") {
            Ok(_) => panic!("expected unauthorized HELLO"),
            Err(e) => e,
        };
        assert!(
            matches!(err, RpcError::Worker(_)) || matches!(err, RpcError::Unauthorized),
            "{err}"
        );
        let _ = worker.join();
    }

    #[test]
    fn hello_token_match_accepts() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection_with_token(
                stream,
                WorkerBackend::Cpu(CpuWorker::new(8 << 20)),
                Some("shared"),
            )
            .unwrap();
        });
        let client = RpcSession::connect_addr_with_token(&addr.to_string(), "shared").unwrap();
        assert_eq!(client.hello().major, PROTO_MAJOR);
        drop(client);
        worker.join().unwrap();
    }

    #[test]
    fn rpc_backend_kinds_wire_compat_cpu_hip_cuda() {
        assert_eq!(RpcBackendKind::from_u8(0), Some(RpcBackendKind::Cpu));
        assert_eq!(RpcBackendKind::from_u8(1), Some(RpcBackendKind::Hip));
        assert_eq!(RpcBackendKind::from_u8(2), Some(RpcBackendKind::Cuda));
        let cuda_hello = HelloReply {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            kernel_abi_rev: KERNEL_ABI_REV,
            backend: RpcBackendKind::Cuda,
            dtype_mask: 1,
            device_name: "cuda:0".into(),
            memory_bytes: 12 << 30,
        };
        let hip_hello = HelloReply {
            backend: RpcBackendKind::Hip,
            device_name: "hip:0".into(),
            ..cuda_hello.clone()
        };
        let cuda_bytes = encode_hello_reply(&cuda_hello);
        let hip_bytes = encode_hello_reply(&hip_hello);
        let cuda_back = crate::protocol::decode_hello_reply(&cuda_bytes).unwrap();
        let hip_back = crate::protocol::decode_hello_reply(&hip_bytes).unwrap();
        assert_eq!(cuda_back.backend, RpcBackendKind::Cuda);
        assert_eq!(hip_back.backend, RpcBackendKind::Hip);
        assert_eq!(cuda_back.device_name, "cuda:0");
        assert_eq!(hip_back.device_name, "hip:0");
        // Same KernelOp set for both GPU backends — ABI revision shared.
        assert_eq!(cuda_back.kernel_abi_rev, hip_back.kernel_abi_rev);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_worker_hello_advertises_cuda_when_device_present() {
        if allpaca_cuda::device_count() <= 0 {
            eprintln!("skip cuda_worker_hello: no CUDA device");
            return;
        }
        let worker = CudaWorker::new(0).expect("CudaWorker");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let th = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cuda(worker)).unwrap();
        });
        let client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let reply = client.hello().clone();
        drop(client);
        th.join().unwrap();
        assert_eq!(reply.backend, RpcBackendKind::Cuda);
        assert!(reply.device_name.starts_with("cuda:"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_gemm_nt_when_device_present() {
        if allpaca_cuda::device_count() <= 0 {
            eprintln!("skip loopback_cuda_gemm_nt: no CUDA device");
            return;
        }
        let worker = CudaWorker::new(0).expect("CudaWorker");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let th = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cuda(worker)).unwrap();
        });

        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        assert_eq!(client.hello().backend, RpcBackendKind::Cuda);

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
        th.join().unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_residual_add_matches_host_when_device_present() {
        if allpaca_cuda::device_count() <= 0 {
            eprintln!("skip loopback_cuda_residual: no CUDA device");
            return;
        }
        let worker = CudaWorker::new(0).expect("CudaWorker");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let th = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cuda(worker)).unwrap();
        });

        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let n = 4usize;
        let x = client.alloc((n * 4) as u64).unwrap();
        let y = client.alloc((n * 4) as u64).unwrap();
        let x_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let y_host: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
        let x_bytes: Vec<u8> = x_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let y_bytes: Vec<u8> = y_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        client.h2d(x, 0, &x_bytes).unwrap();
        client.h2d(y, 0, &y_bytes).unwrap();
        client
            .launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::ResidualAdd,
                    bufs: vec![x, y],
                    i32s: vec![],
                    f32s: vec![],
                }],
            })
            .unwrap();
        client.sync().unwrap();
        let out = client.d2h(x, 0, (n * 4) as u64).unwrap();
        for i in 0..n {
            let got = f32::from_le_bytes(out[i * 4..(i + 1) * 4].try_into().unwrap());
            let want = x_host[i] + y_host[i];
            assert!((got - want).abs() < 1e-5, "i={i} got={got} want={want}");
        }
        client.free(x).unwrap();
        client.free(y).unwrap();
        drop(client);
        th.join().unwrap();
    }
}
