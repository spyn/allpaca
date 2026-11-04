//! Kernel-ABI RPC: remote buffers plus batched named launches.
//!
//! This is the llama.cpp RPC analogue without a ggml graph. The coordinator
//! owns scheduling; a worker only allocates buffers and runs the existing
//! CPU/HIP/CUDA kernel ABI. Weights may be packed GGUF blobs (`PutTensor`);
//! `GemmNtQuant` keeps that ggml type. HIP and CUDA workers speak the same `KernelOp`
//! wire protocol so an AMD coordinator can drive an NVIDIA worker (and vice
//! versa) as long as HELLO majors match.

mod gpu_sysfs;
pub mod link;
mod protocol;
mod session;
mod worker;

pub use protocol::{
    f32_to_f16_rne, quantize_q8_rows, MoeFfnRequest, MoeX,
    check_rpc_bind, encode_hello_reply, fnv1a64, is_loopback_bind_host, write_reply, BufId, Cmd,
    DeviceStat,
    HelloReply, HelloRequest, KernelOp, LaunchItem, LaunchList, RpcBackendKind, BUILD_ID, KERNEL_ABI_REV,
    SOCKET_BUFFER,
    PROTO_MAJOR, PROTO_MINOR, PROTO_PATCH, STATUS_ERR, STATUS_OK,
};
pub use session::{CallCounters, OpCount, OpCounters, Posted, RpcSession};
pub use worker::{
    serve_connection, serve_connection_ext, serve_connection_with_token, serve_listener,
    serve_listener_ext, serve_listener_with_token, serve_tcp, CpuWorker, Extension, WorkerBackend,
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
    fn session_counts_calls_and_bytes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(64 << 20))).unwrap();
        });
        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let (calls, ext) = client.counters();
        // The HELLO handshake is not a counted call.
        assert_eq!(calls.calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        let buf = client.alloc(1024).unwrap();
        client.h2d(buf, 0, &[7u8; 1000]).unwrap();
        let back = client.d2h(buf, 0, 1000).unwrap();
        assert_eq!(back, vec![7u8; 1000]);
        let load = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(load(&calls.calls), 3);
        assert!(load(&calls.bytes_out) >= 1000, "the upload counts as sent");
        assert!(load(&calls.bytes_in) >= 1000, "the download counts as received");
        assert_eq!(load(&ext.calls), 0);
        drop(client);
        worker.join().unwrap();
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
        check_grouped_match_single(WorkerBackend::Hip(HipWorker::new(0).unwrap()));
    }

    /// A worker serves the extensions it was started with and names them in
    /// its HELLO; a call carries a small head and a bulk payload.
    #[test]
    fn loopback_extension_is_announced_and_called() {
        struct Echo(u32);
        impl Extension for Echo {
            fn name(&self) -> &str {
                "echo"
            }
            fn call(&mut self, head: &[u8], data: &[u8]) -> Result<Vec<u8>, String> {
                if head == b"fail" {
                    return Err("asked to fail".into());
                }
                self.0 += 1;
                let mut out = vec![self.0 as u8];
                out.extend(head.iter().rev());
                out.extend((data.len() as u32).to_le_bytes());
                Ok(out)
            }
        }
        let serve = |exts: Vec<Box<dyn Extension>>| {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let worker = thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                serve_connection_ext(stream, WorkerBackend::Cpu(CpuWorker::new(1 << 20)), None, exts).unwrap();
            });
            (RpcSession::connect_addr(&addr.to_string()).unwrap(), worker)
        };
        let (mut client, worker) = serve(vec![Box::new(Echo(0))]);
        assert!(client.has_extension("echo") && !client.has_extension("other"));
        assert_eq!(client.build_id(), Some(BUILD_ID), "a worker of this build");
        assert!(client.same_build() && BUILD_ID.len() == 16);
        let big = vec![7u8; 200_000];
        assert_eq!(client.ext("echo", b"abc", &big).unwrap(), [&[1u8][..], b"cba", &200_000u32.to_le_bytes()].concat());
        assert_eq!(client.ext("echo", b"", &[]).unwrap(), [2, 0, 0, 0, 0]);
        let err = client.ext("echo", b"fail", &[]).unwrap_err().to_string();
        assert!(err.contains("asked to fail"), "{err}");
        // The session survives a failed call and an unknown extension.
        assert!(client.ext("other", b"", &[]).is_err());
        assert_eq!(client.ext("echo", b"z", &[]).unwrap()[0], 3);
        // Calls sent ahead are answered in order.
        client.ext_send("echo", b"a", &[]).unwrap();
        client.ext_send("echo", b"fail", &[]).unwrap();
        client.ext_send("echo", b"b", &big).unwrap();
        assert_eq!(client.ext_recv().unwrap()[..2], [4, b'a']);
        assert!(client.ext_recv().is_err());
        assert_eq!(client.ext_recv().unwrap()[..2], [5, b'b']);
        drop(client);
        worker.join().unwrap();

        let (client, worker) = serve(Vec::new());
        assert!(!client.has_extension("echo"));
        assert!(client.same_build(), "the build id does not need an extension");
        // A reply that ends at the device list is from a worker without either.
        let old = encode_hello_reply(client.hello());
        assert_eq!(crate::protocol::decode_hello_tail(&old).unwrap(), (Vec::new(), None));
        drop(client);
        worker.join().unwrap();
    }

    /// An extension that logs every call's head and sleeps on `sleep`.
    struct Logger(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
    impl Extension for Logger {
        fn name(&self) -> &str {
            "log"
        }
        fn call(&mut self, head: &[u8], _data: &[u8]) -> Result<Vec<u8>, String> {
            let head = String::from_utf8_lossy(head).into_owned();
            self.0.lock().unwrap().push(head.clone());
            match head.as_str() {
                "sleep" => std::thread::sleep(std::time::Duration::from_millis(300)),
                "fail" => return Err("asked to fail".into()),
                _ => {}
            }
            Ok(head.into_bytes())
        }
    }

    fn serve_logger() -> (RpcSession, std::sync::Arc<std::sync::Mutex<Vec<String>>>, thread::JoinHandle<()>) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let ext = Logger(log.clone());
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection_ext(stream, WorkerBackend::Cpu(CpuWorker::new(1 << 20)), None, vec![Box::new(ext)]).unwrap();
        });
        (RpcSession::connect_addr(&addr.to_string()).unwrap(), log, worker)
    }

    /// A call whose reply nobody needs goes out without waiting for it; the
    /// worker still runs calls in the order they were sent, and the next
    /// call that waits reads the posted replies first.
    #[test]
    fn posted_calls_do_not_wait_and_keep_their_order() {
        let (mut client, log, worker) = serve_logger();
        let started = std::time::Instant::now();
        let slow = client.ext_post("log", b"sleep", &[]).unwrap();
        let quick = client.ext_post("log", b"a", &[]).unwrap();
        assert!(
            started.elapsed() < std::time::Duration::from_millis(200),
            "posting must not wait for the worker ({:?})",
            started.elapsed()
        );
        assert_eq!(slow.outcome(), None, "nothing has read the reply yet");
        assert_eq!(client.ext("log", b"b", &[]).unwrap(), b"b");
        assert_eq!(slow.outcome(), Some(Ok(())));
        assert_eq!(quick.outcome(), Some(Ok(())));
        assert_eq!(*log.lock().unwrap(), ["sleep", "a", "b"]);
        // A plain command reads posted replies too.
        let posted = client.ext_post("log", b"c", &[]).unwrap();
        client.mem_info().unwrap();
        assert_eq!(posted.outcome(), Some(Ok(())));
        // So does the next pipelined reply.
        let posted = client.ext_post("log", b"d", &[]).unwrap();
        client.ext_send("log", b"e", &[]).unwrap();
        assert_eq!(client.ext_recv().unwrap(), b"e");
        assert_eq!(posted.outcome(), Some(Ok(())));
        // A session closed with posted replies unread still ends cleanly
        // for the worker (no connection reset).
        drop(client.ext_post("log", b"last", &[]).unwrap());
        drop(client);
        worker.join().unwrap();
        assert_eq!(log.lock().unwrap().last().map(String::as_str), Some("last"));
    }

    /// A posted call that fails reports it on its handle once the next call
    /// that waits has read the reply; that call's own result is unaffected.
    #[test]
    fn a_failed_posted_call_is_reported_at_the_next_reply() {
        let (mut client, _log, worker) = serve_logger();
        let failed = client.ext_post("log", b"fail", &[]).unwrap();
        assert_eq!(failed.failed(), None, "not known before a reply is read");
        assert_eq!(client.ext("log", b"x", &[]).unwrap(), b"x");
        let error = failed.failed().expect("the failure is known after the next reply");
        assert!(error.contains("asked to fail"), "{error}");
        // A dropped handle's failure only logs; the session goes on.
        drop(client.ext_post("log", b"fail", &[]).unwrap());
        assert_eq!(client.ext("log", b"y", &[]).unwrap(), b"y");
        drop(client);
        worker.join().unwrap();
    }

    /// Calls are counted per operation: commands by name, extension calls by
    /// extension and the first word of their head; posted calls apart.
    #[test]
    fn session_counts_calls_per_operation() {
        let (mut client, _log, worker) = serve_logger();
        let ops = client.op_counters();
        client.mem_info().unwrap();
        client.ext("log", br#"{"Run":{"seq":1}}"#, &[]).unwrap();
        client.ext("log", br#"{"Run":{"seq":2}}"#, &[]).unwrap();
        drop(client.ext_post("log", br#"{"CloneSeq":{"src":1,"dst":2}}"#, &[]).unwrap());
        drop(client.ext_post("log", br#""Load""#, &[]).unwrap());
        client.sync().unwrap();
        let snap: std::collections::BTreeMap<String, (u64, u64)> =
            ops.snapshot().into_iter().map(|o| (o.op, (o.calls, o.posted))).collect();
        assert_eq!(snap["MemInfo"], (1, 0));
        assert_eq!(snap["Sync"], (1, 0));
        assert_eq!(snap["log:Run"], (2, 0));
        assert_eq!(snap["log:CloneSeq"], (0, 1));
        assert_eq!(snap["log:Load"], (0, 1));
        drop(client);
        worker.join().unwrap();
    }

    /// Q2_0 experts as one buffer of slots (gate | up | down per expert, the
    /// request naming it three times) run through the routed-expert kernel.
    #[cfg(feature = "hip")]
    #[test]
    fn loopback_hip_q2_0_expert_slots_match_reference() {
        if allpaca_hip::device_count() == 0 {
            return;
        }
        let backend = WorkerBackend::Hip(HipWorker::new(0).expect("HipWorker"));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, backend).unwrap();
        });
        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let (n_exp, dim, ff, rows) = (4usize, 128usize, 192usize, 3usize);
        let mat = |seed: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| (((i * 7 + seed * 13) % 23) as f32) * 0.02 - 0.2).collect()
        };
        let parts = |e: usize| [mat(1 + e, ff * dim), mat(5 + e, ff * dim), mat(9 + e, dim * ff)];
        let bytes: Vec<u8> = (0..n_exp)
            .flat_map(|e| parts(e).into_iter().flat_map(|m| allpaca_cpu::quantize_q2_0(&m)).collect::<Vec<u8>>())
            .collect();
        let slot = bytes.len() / n_exp;
        let pool = client.put_tensor(42, &[slot as u32, n_exp as u32], &bytes, None).unwrap();
        let req = MoeFfnRequest {
            gate: pool,
            up: pool,
            down: pool,
            n_experts: n_exp as u32,
            dim: dim as u32,
            ff: ff as u32,
            f16_out: false,
        };
        let x: Vec<f32> = (0..rows * dim).map(|i| ((i % 61) as f32) * 0.03 - 0.9).collect();
        // Row 2 routes to no expert.
        let pairs = [(1u32, 3u32, 1.0f32), (0, 0, 0.7), (1, 1, 0.5), (0, 2, 0.3)];
        let got = client.moe_ffn(req, &x, false, &pairs).unwrap();

        let held = |m: &[f32]| {
            let mut w = vec![0.0f32; m.len()];
            allpaca_cpu::dequant_q2_0(&allpaca_cpu::quantize_q2_0(m), w.len(), &mut w).unwrap();
            w
        };
        // The float SwiGLU, and the CPU's on rows rounded to 8 bits per block (as the kernels run).
        let (mut want, mut want8) = (vec![0.0f32; rows * dim], vec![0.0f32; rows * dim]);
        for &(row, e, wgt) in &pairs {
            let (r, [gate, up, down]) = (row as usize, parts(e as usize));
            let xr = &x[r * dim..(r + 1) * dim];
            let (mut g, mut u, mut o) = (vec![0.0f32; ff], vec![0.0f32; ff], vec![0.0f32; dim]);
            allpaca_cpu::matmul_nt(xr, &held(&gate), &mut g, 1, ff, dim).unwrap();
            allpaca_cpu::matmul_nt(xr, &held(&up), &mut u, 1, ff, dim).unwrap();
            let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
            allpaca_cpu::matmul_nt(&h, &held(&down), &mut o, 1, dim, ff).unwrap();
            let input = allpaca_cpu::Q2_0Input::new(xr).unwrap();
            let q = |m: &[f32]| allpaca_cpu::quantize_q2_0(m);
            allpaca_cpu::gemv_q2_0_i8(&q(&gate), &input, &mut g).unwrap();
            allpaca_cpu::gemv_q2_0_i8(&q(&up), &input, &mut u).unwrap();
            let h: Vec<f32> = g.iter().zip(&u).map(|(a, b)| a / (1.0 + (-a).exp()) * b).collect();
            let mut o8 = vec![0.0f32; dim];
            allpaca_cpu::gemv_q2_0_i8(&q(&down), &allpaca_cpu::Q2_0Input::new(&h).unwrap(), &mut o8).unwrap();
            for i in 0..dim {
                want[r * dim + i] += wgt * o[i];
                want8[r * dim + i] += wgt * o8[i];
            }
        }
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.1, "reference is near zero ({scale})");
        for (i, ((g, w), w8)) in got.iter().zip(&want).zip(&want8).enumerate() {
            assert!((g - w8).abs() <= 1e-4 * scale, "out[{i}] {g} vs 8-bit {w8}");
            assert!((g - w).abs() <= 0.03 * scale, "out[{i}] {g} vs {w}");
        }
        assert!(got[2 * dim..].iter().all(|v| *v == 0.0), "unrouted row must be zero");
        // A row alone gives the bits it gives among others.
        let one = client.moe_ffn(req, &x[dim..2 * dim], false, &[(0, 3, 1.0), (0, 1, 0.5)]).unwrap();
        assert!(one.iter().zip(&got[dim..2 * dim]).all(|(a, b)| a.to_bits() == b.to_bits()), "row 1 alone differs");
        // Int8 rows are the same rows rounded to 8 bits (on the wire, then again per block).
        let q8 = client.moe_ffn(req, &x, true, &pairs).unwrap();
        for (i, (g, w)) in q8.iter().zip(&want8).enumerate() {
            assert!((g - w).abs() <= 2e-2 * scale, "q8 out[{i}] {g} vs {w}");
        }
        drop(client);
        worker.join().unwrap();
    }

    /// The CUDA worker's fused K-quant MoE (port of the HIP kernels): Q5_K
    /// gate/up with Q6_K down as in Qwen3-Next, and all-Q4_K.
    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_fused_kquant_moe_ffn_matches_reference() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        check_moe_ffn(WorkerBackend::Cuda(CudaWorker::new(0).expect("CudaWorker")), MOE_KQUANT_FUSED);
        check_moe_ffn(WorkerBackend::Cuda(CudaWorker::new(0).expect("CudaWorker")), MOE_Q4K_FUSED);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn loopback_cuda_grouped_experts_match_single_rows() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        check_grouped_match_single(WorkerBackend::Cuda(CudaWorker::new(0).unwrap()));
    }

    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn check_grouped_match_single(backend: WorkerBackend) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, backend).unwrap();
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

    /// Cost of one fused MoeFfn call against the number of (row, expert)
    /// pairs, at Qwen3-Next's expert shape (512 experts, dim 2048, ff 512,
    /// Q5_K gate/up, Q6_K down), int8 rows in and F16 out as the coordinator
    /// sends them. `ALLPACA_BENCH_RPC=host:port` (token in `ALLPACA_RPC_TOKEN`)
    /// measures a remote worker; unset, a loopback HIP worker on this machine.
    /// `cargo test --release -p allpaca-rpc --features hip --lib moe_cost_curve -- --ignored --nocapture`
    #[cfg(feature = "hip")]
    #[test]
    #[ignore]
    fn moe_cost_curve() {
        let remote = std::env::var("ALLPACA_BENCH_RPC").ok();
        let mut worker = None;
        let mut client = match &remote {
            Some(ep) => {
                let token = std::env::var("ALLPACA_RPC_TOKEN").unwrap_or_default();
                RpcSession::connect_addr_with_token(ep, &token).unwrap()
            }
            None => {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let addr = listener.local_addr().unwrap();
                worker = Some(thread::spawn(move || {
                    let (stream, _) = listener.accept().unwrap();
                    serve_connection(stream, WorkerBackend::Hip(HipWorker::new(0).unwrap())).unwrap();
                }));
                RpcSession::connect_addr(&addr.to_string()).unwrap()
            }
        };
        let (n_exp, dim, ff) = (512usize, 2048usize, 512usize);
        // Weights do not change the cost: one quantized expert, repeated.
        let one = |ty: u32, rows: usize, cols: usize| -> Vec<u8> {
            let m: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919 % 1000) as f32 - 500.0) * 1e-4).collect();
            let q = quantize(ty, &m);
            (0..n_exp).flat_map(|_| q.iter().copied()).collect()
        };
        let t = std::time::Instant::now();
        let gate = client.put_tensor(13, &[dim as u32, ff as u32, n_exp as u32], &one(13, ff, dim), None).unwrap();
        let up = client.put_tensor(13, &[dim as u32, ff as u32, n_exp as u32], &one(13, ff, dim), None).unwrap();
        let down = client.put_tensor(14, &[ff as u32, dim as u32, n_exp as u32], &one(14, dim, ff), None).unwrap();
        eprintln!("upload {:.1} s to {}", t.elapsed().as_secs_f64(), remote.as_deref().unwrap_or("loopback hip"));
        let req = MoeFfnRequest { gate, up, down, n_experts: n_exp as u32, dim: dim as u32, ff: ff as u32, f16_out: true };
        let x: Vec<f32> = (0..8 * dim).map(|i| ((i % 97) as f32 - 48.0) * 0.02).collect();
        for rows in [1usize, 4] {
            for k in [1usize, 2, 4, 6, 8, 10] {
                let pairs: Vec<(u32, u32, f32)> = (0..rows as u32)
                    .flat_map(|r| (0..k as u32).map(move |j| (r, (r * 37 + j * 51) % 512, 0.1)))
                    .collect();
                for _ in 0..5 {
                    client.moe_ffn(req, &x[..rows * dim], true, &pairs).unwrap();
                }
                let n = 40;
                let t = std::time::Instant::now();
                for _ in 0..n {
                    client.moe_ffn(req, &x[..rows * dim], true, &pairs).unwrap();
                }
                let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
                eprintln!("COST rows={rows} experts_per_row={k} pairs={} us={us:.0}", pairs.len());
            }
        }
        drop(client);
        if let Some(w) = worker {
            w.join().unwrap();
        }
    }

    /// Cost of one `MoeFfn` call on Q2_0 expert slots at Qwen3.8-Flash-Next's
    /// expert shape (dim 2560, ff 640, ten experts a row), int8 and float
    /// rows in, F16 out. `ALLPACA_BENCH_RPC=host:port` (token in
    /// `ALLPACA_RPC_TOKEN`) measures a remote worker; unset, a loopback HIP
    /// worker on this machine.
    /// `cargo test --release -p allpaca-rpc --features hip --lib q2_0_slots_cost -- --ignored --nocapture`
    #[cfg(feature = "hip")]
    #[test]
    #[ignore]
    fn q2_0_slots_cost() {
        let remote = std::env::var("ALLPACA_BENCH_RPC").ok();
        let mut worker = None;
        let mut client = match &remote {
            Some(ep) => {
                let token = std::env::var("ALLPACA_RPC_TOKEN").unwrap_or_default();
                RpcSession::connect_addr_with_token(ep, &token).unwrap()
            }
            None => {
                let listener = TcpListener::bind("127.0.0.1:0").unwrap();
                let addr = listener.local_addr().unwrap();
                worker = Some(thread::spawn(move || {
                    let (stream, _) = listener.accept().unwrap();
                    serve_connection(stream, WorkerBackend::Hip(HipWorker::new(0).unwrap())).unwrap();
                }));
                RpcSession::connect_addr(&addr.to_string()).unwrap()
            }
        };
        let (n_exp, dim, ff) = (64usize, 2560usize, 640usize);
        let m: Vec<f32> = (0..3 * ff * dim).map(|i| ((i * 7919 % 1000) as f32 - 500.0) * 1e-4).collect();
        let slot = allpaca_cpu::quantize_q2_0(&m);
        let bytes: Vec<u8> = (0..n_exp).flat_map(|_| slot.iter().copied()).collect();
        let pool = client.put_tensor(42, &[slot.len() as u32, n_exp as u32], &bytes, None).unwrap();
        let req = MoeFfnRequest { gate: pool, up: pool, down: pool, n_experts: n_exp as u32, dim: dim as u32, ff: ff as u32, f16_out: true };
        let x: Vec<f32> = (0..16 * dim).map(|i| ((i % 97) as f32 - 48.0) * 0.02).collect();
        for q8 in [true, false] {
            for (rows, k) in [(1usize, 1usize), (1, 10), (4, 10), (16, 10)] {
                let pairs: Vec<(u32, u32, f32)> = (0..rows as u32)
                    .flat_map(|r| (0..k as u32).map(move |j| (r, (r * 37 + j * 51) % n_exp as u32, 0.1)))
                    .collect();
                for _ in 0..5 {
                    client.moe_ffn(req, &x[..rows * dim], q8, &pairs).unwrap();
                }
                // `ALLPACA_BENCH_GAP_US`: idle between calls, as a forward leaves.
                let gap = std::env::var("ALLPACA_BENCH_GAP_US").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
                let (n, mut busy) = (40, std::time::Duration::ZERO);
                for _ in 0..n {
                    std::thread::sleep(std::time::Duration::from_micros(gap));
                    let t = std::time::Instant::now();
                    client.moe_ffn(req, &x[..rows * dim], q8, &pairs).unwrap();
                    busy += t.elapsed();
                }
                let us = busy.as_secs_f64() * 1e6 / n as f64;
                eprintln!("COST q8={q8} rows={rows} experts_per_row={k} gap={gap}us us={us:.0}");
            }
        }
        drop(client);
        if let Some(w) = worker {
            w.join().unwrap();
        }
    }

    /// Prints digests of K-quant GEMV and fused MoE outputs for fixed inputs,
    /// to compare HIP and CUDA workers across machines bit for bit:
    /// `cargo test --release -p allpaca-rpc --features <hip|cuda> --lib kquant_digest -- --ignored --nocapture`
    #[cfg(any(feature = "hip", feature = "cuda"))]
    #[test]
    #[ignore]
    fn kquant_digest() {
        #[cfg(feature = "hip")]
        let backend = WorkerBackend::Hip(HipWorker::new(0).unwrap());
        #[cfg(all(feature = "cuda", not(feature = "hip")))]
        let backend = WorkerBackend::Cuda(CudaWorker::new(0).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, backend).unwrap();
        });
        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let digest = |v: &[f32]| v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        });
        // Qwen3-Next's expert shape; 1-row calls (decode) and grouped rows.
        let (dim, ff, n_exp) = (2048usize, 512usize, 4usize);
        let mat = |seed: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((((i * 2654435761usize) ^ (seed * 40503)) % 1000) as f32 - 500.0) * 0.0007).collect()
        };
        let x: Vec<f32> = mat(77, 4 * dim).iter().map(|v| v * 9.0).collect();
        for (gu, dn) in [(12u32, 12u32), (13, 14)] {
            let mut put = |ty: u32, seed: usize, rows: usize, cols: usize| {
                let bytes: Vec<u8> = (0..n_exp).flat_map(|e| quantize(ty, &mat(seed + e, rows * cols))).collect();
                client.put_tensor(ty, &[cols as u32, rows as u32, n_exp as u32], &bytes, None).unwrap()
            };
            let (gate, up, down) = (put(gu, 1, ff, dim), put(gu, 5, ff, dim), put(dn, 9, dim, ff));
            let req = MoeFfnRequest { gate, up, down, n_experts: n_exp as u32, dim: dim as u32, ff: ff as u32, f16_out: false };
            let pairs = [(0u32, 1u32, 0.6f32), (0, 3, 0.4), (1, 1, 0.5), (1, 2, 0.5), (2, 0, 1.0), (3, 1, 0.3), (3, 3, 0.7)];
            for q8 in [false, true] {
                let y = client.moe_ffn(req, &x, q8, &pairs).unwrap();
                eprintln!("DIGEST moe gate_up={gu} down={dn} q8={q8}: {:016x} first={:?}", digest(&y), &y[..3]);
                let one: Vec<(u32, u32, f32)> = (0..n_exp as u32).map(|e| (0, e, 0.25)).collect();
                let y1 = client.moe_ffn(req, &x[..dim], q8, &one).unwrap();
                let half = client.moe_ffn(MoeFfnRequest { f16_out: true, ..req }, &x[..dim], q8, &one).unwrap();
                eprintln!("DIGEST moe1 gate_up={gu} down={dn} q8={q8}: {:016x} f16 {:016x}", digest(&y1), digest(&half));
                // Prefill-sized: 20 rows, 2 experts each, groups of up to 8 rows.
                let xs: Vec<f32> = mat(91, 20 * dim).iter().map(|v| v * 7.0).collect();
                let many: Vec<(u32, u32, f32)> = (0..20u32)
                    .flat_map(|r| [(r, r % 4, 0.7f32), (r, (r * 3 + 1) % 4, 0.3)])
                    .collect();
                let yp = client.moe_ffn(MoeFfnRequest { f16_out: true, ..req }, &xs, q8, &many).unwrap();
                eprintln!("DIGEST moe20 gate_up={gu} down={dn} q8={q8}: {:016x}", digest(&yp));
            }
        }
        // Real routing shape: 21 rows x 10 distinct experts of 64 (groups of
        // 1-3 rows, the MR=4 template with short groups).
        let n_exp = 64usize;
        for (gu, dn) in [(13u32, 14u32)] {
            let mut put = |ty: u32, seed: usize, rows: usize, cols: usize| {
                let bytes: Vec<u8> = (0..n_exp).flat_map(|e| quantize(ty, &mat(seed + e, rows * cols))).collect();
                client.put_tensor(ty, &[cols as u32, rows as u32, n_exp as u32], &bytes, None).unwrap()
            };
            let (gate, up, down) = (put(gu, 1, ff, dim), put(gu, 5, ff, dim), put(dn, 9, dim, ff));
            let req = MoeFfnRequest { gate, up, down, n_experts: n_exp as u32, dim: dim as u32, ff: ff as u32, f16_out: true };
            let xs: Vec<f32> = mat(93, 21 * dim).iter().map(|v| v * 7.0).collect();
            let many: Vec<(u32, u32, f32)> = (0..21u32)
                .flat_map(|r| (0..10u32).map(move |j| (r, (r * 7 + j * 13 + j * j) % 64, 0.1)))
                .collect();
            for q8 in [false, true] {
                let yp = client.moe_ffn(req, &xs, q8, &many).unwrap();
                eprintln!("DIGEST moe21x10 q8={q8}: {:016x}", digest(&yp));
            }
        }
        drop(client);
        worker.join().unwrap();
    }

    /// One line per dense-layer case, fixed inputs: Llama 3.2 3B's shape and
    /// the shapes the attention kernels treat differently. Compares workers
    /// across machines bit for bit, like `kquant_digest`.
    #[cfg(any(feature = "hip", feature = "cuda"))]
    fn dense_digests() -> Vec<String> {
        #[cfg(feature = "hip")]
        let backend = WorkerBackend::Hip(HipWorker::new(0).unwrap());
        #[cfg(all(feature = "cuda", not(feature = "hip")))]
        let backend = WorkerBackend::Cuda(CudaWorker::new(0).unwrap());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, backend).unwrap();
        });
        let mut c = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let mut lines = Vec::new();
        let digest = |v: &[f32]| v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
            (h ^ u64::from(x.to_bits())).wrapping_mul(0x100_0000_01b3)
        });
        let mat = |seed: usize, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((((i * 2654435761usize) ^ (seed * 40503)) % 1000) as f32 - 500.0) * 0.002).collect()
        };
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let (heads, hd) = (24usize, 128usize);
        let dim = heads * hd;
        // Attention: decode (one row), prefill (many rows), both sides of a
        // chunk, GQA, and head dims off the float4 path and past shared memory.
        let shapes = [(24usize, 24usize, 128usize), (8, 2, 64), (12, 12, 32), (6, 6, 256), (4, 4, 30), (2, 2, 512)];
        let lengths = [(1usize, 1usize), (1, 2), (1, 8), (1, 15), (1, 16), (1, 17), (1, 21), (1, 127), (1, 128), (1, 129), (1, 257), (1, 300), (2, 2), (9, 9), (21, 21), (21, 300), (130, 130)];
        for (a_heads, a_kv, hd_a) in shapes {
            let (q_row, k_row) = (a_heads * hd_a, a_kv * hd_a);
            for (rows, n_kv) in lengths {
                let (q, k, v, out) = (
                    c.alloc((rows * q_row * 4) as u64).unwrap(),
                    c.alloc((n_kv * k_row * 4) as u64).unwrap(),
                    c.alloc((n_kv * k_row * 4) as u64).unwrap(),
                    c.alloc((rows * q_row * 4) as u64).unwrap(),
                );
                c.h2d(q, 0, &bytes(&mat(1, rows * q_row))).unwrap();
                c.h2d(k, 0, &bytes(&mat(2, n_kv * k_row))).unwrap();
                c.h2d(v, 0, &bytes(&mat(3, n_kv * k_row))).unwrap();
                c.launch(&LaunchList {
                    items: vec![LaunchItem {
                        op: KernelOp::Attention,
                        bufs: vec![q, k, v, out],
                        i32s: vec![rows as i32, n_kv as i32, a_heads as i32, a_kv as i32, hd_a as i32, 1],
                        f32s: vec![],
                    }],
                })
                .unwrap();
                c.sync().unwrap();
                let y: Vec<f32> = c
                    .d2h(out, 0, (rows * q_row * 4) as u64)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                lines.push(format!("attn heads={a_heads}/{a_kv}x{hd_a} rows={rows} n_kv={n_kv}: {:016x} first={:?}", digest(&y), &y[..2]));
                for id in [q, k, v, out] {
                    c.free(id).unwrap();
                }
            }
        }
        for &(rows, start) in &[(1usize, 0usize), (1, 1), (1, 5), (21, 0), (21, 100), (128, 128)] {
            let (q, k) = (c.alloc((rows * dim * 4) as u64).unwrap(), c.alloc((rows * dim * 4) as u64).unwrap());
            let (mq, mk) = (mat(4, rows * dim), mat(5, rows * dim));
            c.h2d(q, 0, &bytes(&mq)).unwrap();
            c.h2d(k, 0, &bytes(&mk)).unwrap();
            c.launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::RopeRows,
                    bufs: vec![q, k],
                    i32s: vec![rows as i32, heads as i32, heads as i32, hd as i32, start as i32],
                    f32s: vec![10_000.0],
                }],
            })
            .unwrap();
            c.sync().unwrap();
            let y: Vec<f32> = c
                .d2h(q, 0, (rows * dim * 4) as u64)
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            lines.push(format!("rope rows={rows} start={start}: {:016x} first={:?}", digest(&y), &y[..3]));
        }
        // The rest of a dense layer: norm, SiLU, residual.
        for rows in [1usize, 2, 21] {
            let (x, w, o) = (
                c.alloc((rows * dim * 4) as u64).unwrap(),
                c.alloc((dim * 4) as u64).unwrap(),
                c.alloc((rows * dim * 4) as u64).unwrap(),
            );
            c.h2d(x, 0, &bytes(&mat(7, rows * dim))).unwrap();
            c.h2d(w, 0, &bytes(&mat(8, dim))).unwrap();
            let read = |c: &mut RpcSession, id, n: usize| -> Vec<f32> {
                c.sync().unwrap();
                c.d2h(id, 0, (n * 4) as u64)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            };
            c.launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::RmsNormRows,
                    bufs: vec![x, w, o],
                    i32s: vec![rows as i32, dim as i32],
                    f32s: vec![1e-5],
                }],
            })
            .unwrap();
            let y = read(&mut c, o, rows * dim);
            lines.push(format!("rms rows={rows}: {:016x} first={:?}", digest(&y), &y[..3]));
            c.h2d(o, 0, &bytes(&mat(9, rows * dim))).unwrap();
            c.launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::SiluMul,
                    bufs: vec![o, x],
                    i32s: vec![(rows * dim) as i32],
                    f32s: vec![],
                }],
            })
            .unwrap();
            let y = read(&mut c, o, rows * dim);
            lines.push(format!("silu rows={rows}: {:016x} first={:?}", digest(&y), &y[..3]));
            c.launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::ResidualAdd,
                    bufs: vec![o, x],
                    i32s: vec![(rows * dim) as i32],
                    f32s: vec![],
                }],
            })
            .unwrap();
            let y = read(&mut c, o, rows * dim);
            lines.push(format!("residual rows={rows}: {:016x}", digest(&y)));
        }
        // Gated layers: per-head norm and sigmoid.
        {
            let (x, w, g) = (
                c.alloc((dim * 4) as u64).unwrap(),
                c.alloc((hd * 4) as u64).unwrap(),
                c.alloc((dim * 4) as u64).unwrap(),
            );
            c.h2d(x, 0, &bytes(&mat(11, dim))).unwrap();
            c.h2d(w, 0, &bytes(&mat(12, hd))).unwrap();
            c.h2d(g, 0, &bytes(&mat(13, dim))).unwrap();
            c.launch(&LaunchList {
                items: vec![
                    LaunchItem {
                        op: KernelOp::RmsNormHeads,
                        bufs: vec![x, w],
                        i32s: vec![heads as i32, hd as i32],
                        f32s: vec![1e-5],
                    },
                    LaunchItem {
                        op: KernelOp::SigmoidMul,
                        bufs: vec![x, g],
                        i32s: vec![dim as i32],
                        f32s: vec![],
                    },
                ],
            })
            .unwrap();
            c.sync().unwrap();
            let y: Vec<f32> = c
                .d2h(x, 0, (dim * 4) as u64)
                .unwrap()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            lines.push(format!("gated: {:016x} first={:?}", digest(&y), &y[..2]));
        }
        // Dense GEMV: `rows` of each width, row digests comparable with the
        // same row run alone. Type 0 is F32, 8 is Q8_0, 12..14 the K-quants;
        // narrow outputs and wide rows take other paths on some cards.
        for (ty, n, k) in [
            (12u32, 3072usize, 3072usize),
            (13, 3072, 3072),
            (14, 3072, 3072),
            (12, 1024, 3072),
            (14, 1024, 3072),
            (14, 8192, 3072),
            (8, 3072, 3072),
            (8, 1024, 5120),
            (0, 3072, 3072),
        ] {
            let m = mat(10 + ty as usize, n * k);
            let packed = if ty == 0 { bytes(&m) } else { quantize(ty, &m) };
            let w = c.put_tensor(ty, &[k as u32, n as u32], &packed, None).unwrap();
            for rows in [1usize, 2, 3, 4, 5, 8, 9, 12, 21] {
                let (a, y) = (c.alloc((rows * k * 4) as u64).unwrap(), c.alloc((rows * n * 4) as u64).unwrap());
                c.h2d(a, 0, &bytes(&mat(6, rows * k))).unwrap();
                c.launch(&LaunchList {
                    items: vec![LaunchItem {
                        op: KernelOp::GemmNtQuant,
                        bufs: vec![a, w, y],
                        i32s: vec![rows as i32, n as i32, k as i32],
                        f32s: vec![],
                    }],
                })
                .unwrap();
                c.sync().unwrap();
                let v: Vec<f32> = c
                    .d2h(y, 0, (rows * n * 4) as u64)
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                lines.push(format!("gemv ty={ty} {n}x{k} rows={rows}: {:016x} first={:?}", digest(&v), &v[..2]));
                c.free(a).unwrap();
                c.free(y).unwrap();
            }
            c.free(w).unwrap();
        }
        drop(c);
        worker.join().unwrap();
        lines
    }

    /// Prints [`dense_digests`], to find which case differs on a new card:
    /// `cargo test --release -p allpaca-rpc --features <hip|cuda> --lib dense_digest -- --ignored --nocapture`
    #[cfg(any(feature = "hip", feature = "cuda"))]
    #[test]
    #[ignore]
    fn dense_digest() {
        for line in dense_digests() {
            eprintln!("DIGEST {line}");
        }
    }

    /// Every dense-layer op gives these bits on every backend: RX 9070, RX
    /// 9060 XT, MI50 and a K80 (sm_37). A last-bit difference is not harmless:
    /// the int8 activations in the K-quant GEMV turn it into a 1% difference
    /// within a few layers (S1). A change that moves this number has to move
    /// it on HIP and CUDA alike; re-pin it after checking both agree.
    #[cfg(any(feature = "hip", feature = "cuda"))]
    #[test]
    fn dense_ops_give_the_same_bits_on_every_backend() {
        #[cfg(feature = "cuda")]
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        // F32 and Q8_0 GEMV still differ by card, and no pinned model's layer
        // weights use them: they stay in `dense_digest` to see, out of the pin.
        let digests: String = dense_digests()
            .iter()
            .filter(|l| !l.starts_with("gemv ty=0 ") && !l.starts_with("gemv ty=8 "))
            .map(|l| l.split(": ").nth(1).and_then(|d| d.split_whitespace().next()).unwrap_or(""))
            .collect();
        let all = digests.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        assert_eq!(all, 0x1bb6_909a_8b8e_f82b, "dense-layer bits changed: {all:#018x}");
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
    #[cfg(feature = "cuda")]
    const MOE_Q4K_FUSED: MoeCase = MoeCase { gate_up: 12, down: 12, ..MOE_KQUANT_FUSED };

    /// A coordinator connects, sets up (stats, allocations, packed weights)
    /// and only then launches. No setup call may leave a CUDA error pending
    /// for the first kernel to pick up: on a K80 (sm_37) worker the first
    /// request after connect failed with "CUDA rmsnorm failed rc=3" (S1).
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_workers_first_kernel_runs_after_each_setup_call() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        type Setup = fn(&mut RpcSession);
        let setups: [(&str, Setup); 11] = [
            ("put_tensor q6_k", |c| {
                let m: Vec<f32> = (0..256 * 4).map(|i| (i % 17) as f32 * 0.01).collect();
                drop(
                    c.put_tensor(14, &[256, 4], &quantize(14, &m), None)
                        .unwrap(),
                );
            }),
            ("put_tensor q8_0", |c| {
                let m: Vec<f32> = (0..256 * 4).map(|i| (i % 17) as f32 * 0.01).collect();
                drop(c.put_tensor(8, &[256, 4], &quantize(8, &m), None).unwrap());
            }),
            ("put_tensor f32", |c| {
                let m: Vec<u8> = (0..256).flat_map(|i| (i as f32).to_le_bytes()).collect();
                drop(c.put_tensor(0, &[256], &m, None).unwrap());
            }),
            ("alloc 2 GiB", |c| drop(c.alloc(2 << 30).unwrap())),
            ("none", |_| {}),
            ("mem_info", |c| drop(c.mem_info().unwrap())),
            ("device_count", |c| drop(c.device_count().unwrap())),
            ("device_stats", |c| drop(c.device_stats().unwrap())),
            ("alloc+clear", |c| {
                let b = c.alloc(4096).unwrap();
                c.clear(b, 0).unwrap();
            }),
            ("put_tensor q4_k", |c| {
                let m: Vec<f32> = (0..256 * 4).map(|i| (i % 17) as f32 * 0.01).collect();
                drop(
                    c.put_tensor(12, &[256, 4], &quantize(12, &m), None)
                        .unwrap(),
                );
            }),
            ("sync", |c| c.sync().unwrap()),
        ];
        let mut failed = Vec::new();
        for (name, setup) in setups {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            let worker = thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                let backend = WorkerBackend::Cuda(CudaWorker::new(0).unwrap());
                serve_connection(stream, backend).unwrap();
            });
            let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
            setup(&mut client);
            let n = 64usize;
            let ones: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
            let (x, w, out) = (
                client.alloc(n as u64 * 4).unwrap(),
                client.alloc(n as u64 * 4).unwrap(),
                client.alloc(n as u64 * 4).unwrap(),
            );
            client.h2d(x, 0, &ones).unwrap();
            client.h2d(w, 0, &ones).unwrap();
            let first = client.launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::RmsNorm,
                    bufs: vec![x, w, out],
                    i32s: vec![],
                    f32s: vec![1e-5],
                }],
            });
            if let Err(e) = first {
                failed.push(format!("{name}: {e}"));
            }
            drop(client);
            worker.join().unwrap();
        }
        assert!(failed.is_empty(), "first kernel failed after: {failed:#?}");
    }

    /// `allpaca-rpc-server` serves coordinators one after another on one
    /// thread, opening a new backend for each and dropping the old one. The
    /// teardown of the last coordinator's buffers must not leave a CUDA error
    /// for the next coordinator's first kernel (K80 symptom: HTTP 500, "CUDA
    /// rmsnorm failed rc=3", on the first request after every restart).
    #[cfg(feature = "cuda")]
    #[test]
    fn a_second_coordinator_on_the_same_cuda_worker_thread_runs_its_first_kernel() {
        if allpaca_cuda::device_count() <= 0 {
            return;
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            for _ in 0..2 {
                let (stream, _) = listener.accept().unwrap();
                let backend = WorkerBackend::Cuda(CudaWorker::new(0).unwrap());
                serve_connection(stream, backend).unwrap();
            }
        });
        let n = 64usize;
        let ones: Vec<u8> = (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        let mut results = Vec::new();
        for coordinator in 0..2 {
            let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
            let m: Vec<f32> = (0..256 * 4).map(|i| (i % 17) as f32 * 0.01).collect();
            client
                .put_tensor(12, &[256, 4], &quantize(12, &m), None)
                .unwrap();
            let (x, w, out) = (
                client.alloc(n as u64 * 4).unwrap(),
                client.alloc(n as u64 * 4).unwrap(),
                client.alloc(n as u64 * 4).unwrap(),
            );
            client.h2d(x, 0, &ones).unwrap();
            client.h2d(w, 0, &ones).unwrap();
            let first = client.launch(&LaunchList {
                items: vec![LaunchItem {
                    op: KernelOp::RmsNorm,
                    bufs: vec![x, w, out],
                    i32s: vec![],
                    f32s: vec![1e-5],
                }],
            });
            results.push(format!("coordinator {coordinator}: {first:?}"));
            // Hang up without freeing, as a killed coordinator does.
            drop(client);
        }
        worker.join().unwrap();
        assert!(
            results.iter().all(|r| r.ends_with("Ok(())")),
            "{results:#?}"
        );
    }

    fn quantize(ty: u32, m: &[f32]) -> Vec<u8> {
        match ty {
            8 => allpaca_cpu::quantize_q8_0(m),
            12 => allpaca_cpu::quantize_q4_k(m),
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
            12 => allpaca_cpu::dequant_q4_k(&q, w.len(), &mut w).unwrap(),
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

    /// `DeviceStats` over a real session: one entry per worker device, with
    /// memory in use tracking allocations.
    #[test]
    fn device_stats_roundtrip_over_loopback() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, WorkerBackend::Cpu(CpuWorker::new(8 << 20))).unwrap();
        });
        let mut client = RpcSession::connect_addr(&addr.to_string()).unwrap();
        let before = client.device_stats().unwrap();
        assert_eq!(before.len(), 1);
        assert_eq!(before[0].name, "cpu");
        assert_eq!((before[0].mem_used, before[0].mem_total), (0, 8 << 20));
        assert_eq!((before[0].util_pct, before[0].temp_c, before[0].power_w), (-1, -1, -1));
        let _buf = client.alloc(4096).unwrap();
        assert_eq!(client.device_stats().unwrap()[0].mem_used, 4096);
        drop(client);
        worker.join().unwrap();
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
