//! Kernel-ABI RPC: remote buffers plus batched named launches.
//!
//! This is the llama.cpp RPC analogue without a ggml graph. The coordinator
//! owns scheduling; a worker only allocates buffers and runs the existing
//! CPU/HIP/CUDA kernel ABI. Weights may be packed GGUF blobs (`PutTensor`);
//! `GemmNtQuant` keeps that ggml type. HIP and CUDA workers speak the same `KernelOp`
//! wire protocol so an AMD coordinator can drive an NVIDIA worker (and vice
//! versa) as long as HELLO majors match.

mod protocol;
mod session;
mod worker;

pub use protocol::{
    f32_to_f16_rne, quantize_q8_rows, MoeFfnRequest, MoeX,
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

        let reply = HelloReply {
            major: PROTO_MAJOR,
            minor: PROTO_MINOR,
            patch: PROTO_PATCH,
            kernel_abi_rev: KERNEL_ABI_REV,
            backend: RpcBackendKind::Hip,
            dtype_mask: 1,
            device_name: "hip:0+hip:1".into(),
            memory_bytes: 24,
            devices: vec![8, 16],
        };
        let enc = encode_hello_reply(&reply);
        let dec = crate::protocol::decode_hello_reply(&enc).unwrap();
        assert_eq!(dec, reply);
        assert_eq!(dec.device_memories(), vec![8, 16]);
        let legacy = &enc[..enc.len() - 4 - 16];
        let old = crate::protocol::decode_hello_reply(legacy).unwrap();
        assert!(old.devices.is_empty());
        assert_eq!(old.memory_bytes, 24);
        assert_eq!(old.device_memories(), vec![24]);
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
    fn loopback_quant_gemm_q8_0() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });

        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let m = 1i32;
        let n = 32i32;
        let k = 32i32;
        let weight: Vec<f32> = (0..n * k)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let packed = allpaca_cpu::quantize_q8_0(&weight);
        let a_f: Vec<f32> = (0..k).map(|i| (i as f32) * 0.05).collect();
        let a = client.alloc((k as usize * 4) as u64).unwrap();
        let c = client.alloc((n as usize * 4) as u64).unwrap();
        let b = client
            .put_tensor(8, &[k as u32, n as u32], &packed, None)
            .unwrap();
        let a_bytes: Vec<u8> = a_f.iter().flat_map(|v| v.to_le_bytes()).collect();
        client.h2d(a, 0, &a_bytes).unwrap();
        client
            .launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::GemmNtQuant,
                    bufs: vec![a, b, c],
                    i32s: vec![m, n, k],
                    f32s: vec![],
                }],
            })
            .unwrap();
        let raw = client.d2h(c, 0, (n as u64) * 4).unwrap();
        let mut got = vec![0.0f32; n as usize];
        for i in 0..n as usize {
            got[i] = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let mut w = vec![0.0f32; (n * k) as usize];
        allpaca_cpu::dequant_q8_0(&packed, w.len(), &mut w).unwrap();
        let mut expect = vec![0.0f32; n as usize];
        allpaca_cpu::matmul_nt(&a_f, &w, &mut expect, 1, n as usize, k as usize).unwrap();
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
        drop(client);
        worker.join().unwrap();
    }

    /// Expert-parallel MoE: the worker runs each selected expert's SwiGLU on
    /// its rows and returns the routing-weighted sum per row.
    #[test]
    fn loopback_moe_ffn_sums_weighted_experts() {
        check_moe_ffn(WorkerBackend::Cpu(CpuWorker::new(64 << 20)), MOE_Q8);
        check_moe_ffn(WorkerBackend::Cpu(CpuWorker::new(64 << 20)), MOE_KQUANT);
    }

    /// Q8_0 experts take the per-expert GEMM path on HIP.
    #[cfg(feature = "hip")]
    #[test]
    fn loopback_hip_moe_ffn_sums_weighted_experts() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        check_moe_ffn(WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker")), MOE_Q8);
    }

    /// Q5_K gate/up with Q6_K down (Qwen3-Next) take the fused MoE kernels.
    #[cfg(feature = "hip")]
    #[test]
    fn loopback_hip_fused_kquant_moe_ffn_matches_reference() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        check_moe_ffn(WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker")), MOE_KQUANT_FUSED);
    }

    /// Rows that share experts run as expert groups (each expert's weights
    /// read once for all its rows); the result must equal each row on its own.
    #[cfg(feature = "hip")]
    #[test]
    fn loopback_hip_grouped_experts_match_single_rows() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Hip(HipWorker::new(0).unwrap())).unwrap();
        });
        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let (n_exp, dim, ff) = (3usize, 256usize, 256usize);
        let mat = |seed: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| (((i * 7 + seed * 13) % 23) as f32) * 0.02 - 0.2).collect()
        };
        let mut put = |ty: u32, seed: usize, rows: usize, cols: usize| {
            let bytes: Vec<u8> = (0..n_exp).flat_map(|e| quantize(ty, &mat(seed + e, rows * cols))).collect();
            client.put_tensor(ty, &[cols as u32, rows as u32, n_exp as u32], &bytes, None).unwrap()
        };
        let (gate, up, down) = (put(13, 1, ff, dim), put(13, 5, ff, dim), put(14, 9, dim, ff));
        let req = MoeFfnRequest { gate, up, down, n_experts: n_exp as u32, dim: dim as u32, ff: ff as u32, f16_out: false };
        let rows = 4usize;
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i % 61) as f32) * 0.03 - 0.9).collect();
        let routes: [&[(u32, f32)]; 4] = [
            &[(1, 0.6), (0, 0.4)],
            &[(1, 0.5), (2, 0.5)],
            &[(0, 0.7), (1, 0.3)],
            &[(1, 1.0)],
        ];
        let pairs: Vec<(u32, u32, f32)> = routes
            .iter()
            .enumerate()
            .flat_map(|(r, rt)| rt.iter().map(move |&(e, w)| (r as u32, e, w)))
            .collect();
        for q8 in [false, true] {
            let all = client.moe_ffn(req, &x, q8, &pairs).unwrap();
            for (r, rt) in routes.iter().enumerate() {
                let one: Vec<(u32, u32, f32)> = rt.iter().map(|&(e, w)| (0, e, w)).collect();
                let single = client.moe_ffn(req, &x[r * dim..(r + 1) * dim], q8, &one).unwrap();
                assert!(
                    single.iter().zip(&all[r * dim..(r + 1) * dim]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "row {r} (q8 {q8}) differs between grouped and single-row runs"
                );
            }
        }
        drop(client);
        worker.join().unwrap();
    }

    /// Expert tensor types, sizes, and the allowed error relative to the
    /// largest output (the fused kernels quantize activations to int8).
    struct MoeCase {
        gate_up: u32,
        down: u32,
        dim: usize,
        ff: usize,
        tol: f32,
        /// Also send int8 rows and require the identical result (fused path).
        q8_exact: bool,
    }
    const MOE_Q8: MoeCase =
        MoeCase { gate_up: 8, down: 8, dim: 32, ff: 64, tol: 2e-3, q8_exact: false };
    const MOE_KQUANT: MoeCase =
        MoeCase { gate_up: 13, down: 14, dim: 256, ff: 256, tol: 3e-2, q8_exact: false };
    const MOE_KQUANT_FUSED: MoeCase = MoeCase { q8_exact: true, ..MOE_KQUANT };

    fn quantize(ty: u32, m: &[f32]) -> Vec<u8> {
        match ty {
            8 => allpaca_cpu::quantize_q8_0(m),
            13 => allpaca_cpu::quantize_q5_k(m),
            14 => allpaca_cpu::quantize_q6_k(m),
            _ => unreachable!("test type {ty}"),
        }
    }

    /// The weights the worker actually holds after quantization.
    fn dequantize(ty: u32, m: &[f32]) -> Vec<f32> {
        let q = quantize(ty, m);
        let mut w = vec![0.0f32; m.len()];
        match ty {
            8 => allpaca_cpu::dequant_q8_0(&q, w.len(), &mut w).unwrap(),
            13 => allpaca_cpu::dequant_q5_k(&q, w.len(), &mut w).unwrap(),
            14 => allpaca_cpu::dequant_q6_k(&q, w.len(), &mut w).unwrap(),
            _ => unreachable!(),
        }
        w
    }

    fn check_moe_ffn(backend: WorkerBackend, case: MoeCase) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, backend).unwrap();
        });
        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let (n_exp, dim, ff, rows) = (3usize, case.dim, case.ff, 3usize);
        let mat = |seed: usize, rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|i| (((i * 7 + seed * 13) % 23) as f32) * 0.02 - 0.2).collect()
        };
        // Stacked like GGUF `ffn_*_exps`: expert e's matrix follows expert e-1's.
        let stack = |seed: usize, rows: usize, cols: usize| -> Vec<Vec<f32>> {
            (0..n_exp).map(|e| mat(seed + e, rows, cols)).collect()
        };
        let (gates, ups, downs) = (stack(1, ff, dim), stack(5, ff, dim), stack(9, dim, ff));
        let put = |c: &mut RpcSession, ty: u32, m: &[Vec<f32>], rows: usize, cols: usize| {
            let bytes: Vec<u8> = m.iter().flat_map(|w| quantize(ty, w)).collect();
            c.put_tensor(ty, &[cols as u32, rows as u32, n_exp as u32], &bytes, None).unwrap()
        };
        let gb = put(&mut client, case.gate_up, &gates, ff, dim);
        let ub = put(&mut client, case.gate_up, &ups, ff, dim);
        let db = put(&mut client, case.down, &downs, dim, ff);
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i % 61) as f32) * 0.03 - 0.9).collect();
        // Unsorted, and row 2 routes to no expert.
        let pairs = [(1u32, 1u32, 1.0f32), (0, 0, 0.7), (0, 2, 0.3)];
        let req = MoeFfnRequest {
            gate: gb,
            up: ub,
            down: db,
            n_experts: n_exp as u32,
            dim: dim as u32,
            ff: ff as u32,
            f16_out: false,
        };
        let got = client.moe_ffn(req, &x, false, &pairs).unwrap();
        // F16 replies: the same values, rounded to nearest-even F16.
        let half = client.moe_ffn(MoeFfnRequest { f16_out: true, ..req }, &x, false, &pairs).unwrap();
        for (h, g) in half.iter().zip(&got) {
            assert_eq!(*h, allpaca_cpu::f16_to_f32(f32_to_f16_rne(*g)), "f16 reply of {g}");
        }
        if case.q8_exact {
            // Pre-quantized rows are what the fused kernels compute anyway.
            let q8 = client.moe_ffn(req, &x, true, &pairs).unwrap();
            assert!(
                q8.iter().zip(&got).all(|(a, b)| a.to_bits() == b.to_bits()),
                "q8 rows must match f32 rows bit for bit"
            );
        }
        let mut want = vec![0.0f32; rows * dim];
        for &(row, e, wgt) in &pairs {
            let (r, e) = (row as usize, e as usize);
            let xr = &x[r * dim..(r + 1) * dim];
            let (mut g, mut u, mut o) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; dim]);
            allpaca_cpu::matmul_nt(xr, &dequantize(case.gate_up, &gates[e]), &mut g, 1, ff, dim)
                .unwrap();
            allpaca_cpu::matmul_nt(xr, &dequantize(case.gate_up, &ups[e]), &mut u, 1, ff, dim)
                .unwrap();
            let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
            allpaca_cpu::matmul_nt(&h, &dequantize(case.down, &downs[e]), &mut o, 1, dim, ff)
                .unwrap();
            for i in 0..dim {
                want[r * dim + i] += wgt * o[i];
            }
        }
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-3);
        let tol = if case.tol < 1e-2 { case.tol } else { case.tol * scale };
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < tol, "out[{i}] {g} vs {w} (tol {tol})");
        }
        assert!(got[2 * dim..].iter().all(|v| *v == 0.0), "unrouted row must be zero");
        drop(client);
        worker.join().unwrap();
    }

    #[test]
    fn f16_rne_rounds_to_nearest_even() {
        for v in [0.0f32, -0.0, 1.0, -2.5, 65504.0, 1.0e-5, -6.0e-8, 0.333_333, 1234.567] {
            let h = f32_to_f16_rne(v);
            let back = allpaca_cpu::f16_to_f32(h);
            // Spacing: 2^-10 relative for normals, a fixed 2^-24 below 2^-14.
            let ulp = (back.abs() / 1024.0).max(1.0 / 16_777_216.0);
            assert!((back - v).abs() <= ulp * 0.5 + f32::EPSILON * v.abs(), "{v} -> {back}");
        }
        // Ties go to even: 1 + 2^-11 sits halfway between 1 and 1 + 2^-10.
        assert_eq!(f32_to_f16_rne(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16_rne(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_f16_rne(70000.0), 0x7c00);
        assert!(allpaca_cpu::f16_to_f32(f32_to_f16_rne(f32::NAN)).is_nan());
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
            devices: Vec::new(),
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
        if let Some((arch, _)) = allpaca_cuda::device_sm_arch(0) {
            assert!(
                reply.device_name.contains(&arch),
                "hello should advertise {}, got {}",
                arch,
                reply.device_name
            );
        }
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

    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_fused_residual_silu_on_one_stream() {
        if allpaca_cuda::device_count() <= 0 {
            eprintln!("skip loopback_cuda_fused: no CUDA device");
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
        let gate = client.alloc((n * 4) as u64).unwrap();
        let residual = client.alloc((n * 4) as u64).unwrap();
        let up = client.alloc((n * 4) as u64).unwrap();
        let g_host = [1.0f32, 2.0, -1.0, 0.5];
        let r_host = [0.5f32, 0.5, 0.5, 0.5];
        let u_host = [2.0f32, 1.0, 1.0, 3.0];
        let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        client.h2d(gate, 0, &to_bytes(&g_host)).unwrap();
        client.h2d(residual, 0, &to_bytes(&r_host)).unwrap();
        client.h2d(up, 0, &to_bytes(&u_host)).unwrap();
        client
            .launch(&LaunchList {
                items: vec![
                    LaunchItem {
                        op: KernelOp::ResidualAdd,
                        bufs: vec![gate, residual],
                        i32s: vec![],
                        f32s: vec![],
                    },
                    LaunchItem {
                        op: KernelOp::SiluMul,
                        bufs: vec![gate, up],
                        i32s: vec![],
                        f32s: vec![],
                    },
                ],
            })
            .unwrap();
        client.sync().unwrap();
        let out = client.d2h(gate, 0, (n * 4) as u64).unwrap();
        for i in 0..n {
            let got = f32::from_le_bytes(out[i * 4..(i + 1) * 4].try_into().unwrap());
            let r = g_host[i] + r_host[i];
            let silu = r / (1.0 + (-r).exp());
            let want = silu * u_host[i];
            assert!((got - want).abs() < 1e-4, "i={i} got={got} want={want}");
        }
        client.free(gate).unwrap();
        client.free(residual).unwrap();
        client.free(up).unwrap();
        drop(client);
        th.join().unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_embed_gather_when_device_present() {
        if allpaca_cuda::device_count() <= 0 {
            eprintln!("skip loopback_cuda_embed: no CUDA device");
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
        let dim = 4i32;
        let table_host = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let table = client.alloc((table_host.len() * 4) as u64).unwrap();
        let ids = client.alloc(4).unwrap();
        let out = client.alloc((dim * 4) as u64).unwrap();
        let table_bytes: Vec<u8> = table_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        client.h2d(table, 0, &table_bytes).unwrap();
        client.h2d(ids, 0, &1u32.to_le_bytes()).unwrap();
        client
            .launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::EmbedGather,
                    bufs: vec![table, ids, out],
                    i32s: vec![dim, 1],
                    f32s: vec![],
                }],
            })
            .unwrap();
        client.sync().unwrap();
        let got = client.d2h(out, 0, (dim * 4) as u64).unwrap();
        for i in 0..dim as usize {
            let v = f32::from_le_bytes(got[i * 4..(i + 1) * 4].try_into().unwrap());
            assert!((v - table_host[4 + i]).abs() < 1e-5, "i={i} v={v}");
        }
        client.free(table).unwrap();
        client.free(ids).unwrap();
        client.free(out).unwrap();
        drop(client);
        th.join().unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_clear_fills_device_buffer_when_present() {
        if allpaca_cuda::device_count() <= 0 {
            eprintln!("skip loopback_cuda_clear: no CUDA device");
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
        let buf = client.alloc(16).unwrap();
        client.h2d(buf, 0, &[1u8; 16]).unwrap();
        client.clear(buf, 0xAB).unwrap();
        client.sync().unwrap();
        let out = client.d2h(buf, 0, 16).unwrap();
        assert_eq!(out, vec![0xAB; 16]);
        client.free(buf).unwrap();
        drop(client);
        th.join().unwrap();
    }

    #[cfg(feature = "hip")]
    #[test]
    fn loopback_hip_clear_fills_device_buffer_when_present() {
        if allpaca_hip::device_count() <= 0 {
            eprintln!("skip loopback_hip_clear: no HIP device");
            return;
        }
        let worker = HipWorker::new(0).expect("HipWorker");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let th = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Hip(worker)).unwrap();
        });

        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let buf = client.alloc(16).unwrap();
        client.h2d(buf, 0, &[1u8; 16]).unwrap();
        client.clear(buf, 0xAB).unwrap();
        client.sync().unwrap();
        let out = client.d2h(buf, 0, 16).unwrap();
        assert_eq!(out, vec![0xAB; 16]);
        client.free(buf).unwrap();
        drop(client);
        th.join().unwrap();
    }

    #[cfg(feature = "hip")]
    #[test]
    fn loopback_hip_memcpy_and_embed_gather_when_present() {
        if allpaca_hip::device_count() <= 0 {
            eprintln!("skip loopback_hip_memcpy_embed: no HIP device");
            return;
        }
        let worker = HipWorker::new(0).expect("HipWorker");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let th = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Hip(worker)).unwrap();
        });

        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let src = client.alloc(8).unwrap();
        let dst = client.alloc(8).unwrap();
        client.h2d(src, 0, &[1u8, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        client
            .launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::Memcpy,
                    bufs: vec![dst, src],
                    i32s: vec![],
                    f32s: vec![],
                }],
            })
            .unwrap();
        client.sync().unwrap();
        assert_eq!(
            client.d2h(dst, 0, 8).unwrap(),
            vec![1u8, 2, 3, 4, 5, 6, 7, 8]
        );

        let dim = 4i32;
        let table_host = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let table = client.alloc((table_host.len() * 4) as u64).unwrap();
        let ids = client.alloc(4).unwrap();
        let out = client.alloc((dim * 4) as u64).unwrap();
        let table_bytes: Vec<u8> = table_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        client.h2d(table, 0, &table_bytes).unwrap();
        client.h2d(ids, 0, &1u32.to_le_bytes()).unwrap();
        client
            .launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::EmbedGather,
                    bufs: vec![table, ids, out],
                    i32s: vec![dim, 1],
                    f32s: vec![],
                }],
            })
            .unwrap();
        client.sync().unwrap();
        let got = client.d2h(out, 0, (dim * 4) as u64).unwrap();
        for i in 0..dim as usize {
            let v = f32::from_le_bytes(got[i * 4..(i + 1) * 4].try_into().unwrap());
            assert!((v - table_host[4 + i]).abs() < 1e-5, "i={i} v={v}");
        }
        client.free(src).unwrap();
        client.free(dst).unwrap();
        client.free(table).unwrap();
        client.free(ids).unwrap();
        client.free(out).unwrap();
        drop(client);
        th.join().unwrap();
    }
}
