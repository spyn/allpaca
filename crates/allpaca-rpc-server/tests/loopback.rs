use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use allpaca_core::{Engine, ModelConfig};
use allpaca_types::{Config, GenerationParams};

fn spawn_rpc_server(port: u16) -> Child {
    let bin = env!("CARGO_BIN_EXE_allpaca-rpc-server");
    Command::new(bin)
        .args([
            "--cpu-only",
            "-H",
            "127.0.0.1",
            "-p",
            &port.to_string(),
            "-m",
            "256",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn allpaca-rpc-server")
}

fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

#[test]
fn two_process_synthetic_generate() {
    let port = free_port();
    let mut child = spawn_rpc_server(port);
    thread::sleep(Duration::from_millis(200));

    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = true;
    cfg.engine.cpu_only = true;
    cfg.speculative.enabled = false;
    cfg.rpc.endpoints = vec![format!("127.0.0.1:{port}")];
    cfg.rpc.max_seq = 64;
    cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);

    let result = (|| {
        let engine = Engine::new(cfg)?;
        engine.generate(
            "hello",
            GenerationParams {
                max_tokens: 4,
                temperature: 0.0,
                seed: Some(2),
                ..GenerationParams::default()
            },
        )
    })();

    let _ = child.kill();
    let _ = child.wait();
    let out = result.expect("rpc generate");
    assert!(!out.token_ids.is_empty());
}

#[test]
fn gated_hip_two_process() {
    if std::env::var("ALLPACA_RPC_HIP").ok().as_deref() != Some("1") {
        eprintln!("skip gated_hip_two_process: set ALLPACA_RPC_HIP=1");
        return;
    }
    #[cfg(not(feature = "hip"))]
    panic!("ALLPACA_RPC_HIP=1 requires cargo test -p allpaca-rpc-server --features hip");

    #[cfg(feature = "hip")]
    {
        let port = free_port();
        let bin = env!("CARGO_BIN_EXE_allpaca-rpc-server");
        let mut child = Command::new(bin)
            .args([
                "--device",
                "hip:0",
                "-H",
                "127.0.0.1",
                "-p",
                &port.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn allpaca-rpc-server");
        thread::sleep(Duration::from_millis(200));
        if let Ok(Some(status)) = child.try_wait() {
            eprintln!("skip gated_hip_two_process: hip worker exited {status} (need a HIP device)");
            return;
        }

        let mut cfg = Config::compiled_defaults();
        cfg.model.synthetic = true;
        cfg.engine.cpu_only = true;
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![format!("127.0.0.1:{port}")];
        cfg.rpc.max_seq = 64;
        cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);

        let result = (|| {
            let engine = Engine::new(cfg)?;
            let out = engine.generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )?;
            Ok::<_, allpaca_types::AllpacaError>((out, engine.stats().rpc_roundtrips))
        })();

        let _ = child.kill();
        let _ = child.wait();
        let (out, roundtrips) = result.expect("hip rpc generate");
        assert!(!out.token_ids.is_empty());
        assert!(
            roundtrips > 0,
            "HIP Rpc Server must run remote layers, rpc_roundtrips={roundtrips}"
        );
    }
}

#[test]
fn gated_hip_pools_every_device() {
    if std::env::var("ALLPACA_RPC_HIP").ok().as_deref() != Some("1") {
        eprintln!("skip gated_hip_pools_every_device: set ALLPACA_RPC_HIP=1");
        return;
    }
    #[cfg(not(feature = "hip"))]
    panic!("ALLPACA_RPC_HIP=1 requires cargo test -p allpaca-rpc-server --features hip");

    #[cfg(feature = "hip")]
    {
        let expected = allpaca_hip::device_count();
        assert!(expected > 0, "HIP device_count");
        let port = free_port();
        let bin = env!("CARGO_BIN_EXE_allpaca-rpc-server");
        let mut child = Command::new(bin)
            .args([
                "--device",
                "hip",
                "-H",
                "127.0.0.1",
                "-p",
                &port.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn allpaca-rpc-server");
        thread::sleep(Duration::from_millis(500));
        if let Ok(Some(status)) = child.try_wait() {
            panic!("pooled hip worker exited {status}");
        }

        let mut cfg = Config::compiled_defaults();
        cfg.model.synthetic = true;
        cfg.engine.cpu_only = true;
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![format!("127.0.0.1:{port}")];
        cfg.rpc.max_seq = 64;
        cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);

        let result = (|| {
            let engine = Engine::new(cfg)?;
            let remote: Vec<_> = engine
                .inventory()
                .devices
                .iter()
                .filter(|d| d.rpc_endpoint().is_some())
                .map(|d| (d.name.clone(), d.memory_bytes))
                .collect();
            let out = engine.generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )?;
            Ok::<_, allpaca_types::AllpacaError>((out, remote, engine.stats().rpc_roundtrips))
        })();

        let _ = child.kill();
        let _ = child.wait();
        let (out, remote, roundtrips) = result.expect("pooled hip rpc generate");
        assert_eq!(
            remote.len(),
            expected as usize,
            "one endpoint must advertise every HIP device: {remote:?}"
        );
        assert!(remote.iter().all(|(_, bytes)| *bytes > 0));
        assert!(!out.token_ids.is_empty());
        assert!(roundtrips > 0, "rpc_roundtrips={roundtrips}");
    }
}

#[test]
fn gated_cuda_two_process() {
    if std::env::var("ALLPACA_RPC_CUDA").ok().as_deref() != Some("1") {
        eprintln!("skip gated_cuda_two_process: set ALLPACA_RPC_CUDA=1");
        return;
    }
    #[cfg(not(feature = "cuda"))]
    panic!("ALLPACA_RPC_CUDA=1 requires cargo test -p allpaca-rpc-server --features cuda");

    #[cfg(feature = "cuda")]
    {
        let port = free_port();
        let bin = env!("CARGO_BIN_EXE_allpaca-rpc-server");
        let mut child = Command::new(bin)
            .args([
                "--device",
                "cuda:0",
                "-H",
                "127.0.0.1",
                "-p",
                &port.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn allpaca-rpc-server");
        thread::sleep(Duration::from_millis(200));
        if let Ok(Some(status)) = child.try_wait() {
            eprintln!(
                "skip gated_cuda_two_process: cuda worker exited {status} (need a CUDA device)"
            );
            return;
        }

        let mut cfg = Config::compiled_defaults();
        cfg.model.synthetic = true;
        cfg.engine.cpu_only = true;
        cfg.speculative.enabled = false;
        cfg.rpc.endpoints = vec![format!("127.0.0.1:{port}")];
        cfg.rpc.max_seq = 64;
        cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);

        let result = (|| {
            let engine = Engine::new(cfg)?;
            let out = engine.generate(
                "hello",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(2),
                    ..GenerationParams::default()
                },
            )?;
            Ok::<_, allpaca_types::AllpacaError>((out, engine.stats().rpc_roundtrips))
        })();

        let _ = child.kill();
        let _ = child.wait();
        let (out, roundtrips) = result.expect("cuda rpc generate");
        assert!(!out.token_ids.is_empty());
        assert!(
            roundtrips > 0,
            "CUDA Rpc Server must run remote layers, rpc_roundtrips={roundtrips}"
        );
    }
}

/// Live worker. Skips unless `ALLPACA_RPC_REMOTE=host:port`.
/// Non-loopback workers also need `ALLPACA_RPC_TOKEN`.
#[test]
fn gated_remote_synthetic_generate() {
    let Ok(endpoint) = std::env::var("ALLPACA_RPC_REMOTE") else {
        eprintln!("skip gated_remote_synthetic_generate: set ALLPACA_RPC_REMOTE=host:port");
        return;
    };
    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = true;
    cfg.engine.cpu_only = true;
    cfg.speculative.enabled = false;
    cfg.rpc.endpoints = vec![endpoint];
    cfg.rpc.max_seq = 64;
    cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
    if let Ok(token) = std::env::var("ALLPACA_RPC_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            cfg.rpc.token = Some(token);
        }
    }

    let params = GenerationParams {
        max_tokens: 4,
        temperature: 0.0,
        seed: Some(2),
        ..GenerationParams::default()
    };
    let mut wide = ModelConfig::tiny();
    wide.n_layers = 4;

    let mut cpu_cfg = Config::compiled_defaults();
    cpu_cfg.model.synthetic = false;
    cpu_cfg.engine.cpu_only = true;
    cpu_cfg.speculative.enabled = false;
    let cpu = Engine::new(cpu_cfg).expect("cpu engine");
    cpu.load_synthetic(wide.clone()).expect("cpu synthetic");
    let cpu_ids = cpu
        .generate("hello", params.clone())
        .expect("cpu generate")
        .token_ids;

    let engine = Engine::new(cfg).expect("remote rpc connect");
    engine
        .load_synthetic(wide)
        .expect("remote synthetic span");
    let out = engine
        .generate("hello", params)
        .expect("remote rpc generate");
    let stats = engine.stats();
    assert_eq!(out.token_ids, cpu_ids);
    assert_eq!(
        stats.rpc_roundtrips, 6,
        "two contiguous remote layers must be one RunSpan per forward, got {}",
        stats.rpc_roundtrips
    );
    assert_eq!(stats.rpc_activation_bytes, 1536);
    eprintln!(
        "remote rpc ok tokens={} roundtrips={} activation_bytes={}",
        out.token_ids.len(),
        stats.rpc_roundtrips,
        stats.rpc_activation_bytes
    );
}

fn tiny_q8_gguf() -> std::path::PathBuf {
    use std::collections::HashMap;

    use allpaca_gguf::{
        quantize_q8_0, write_gguf_quant, GgmlType, GgufValue, GgufWriteQuantTensor, GgufWriteTensor,
    };

    let dim = 32usize;
    let vocab = 8usize;
    let ff = 64usize;
    let n_kv = 16usize;
    let mut meta = HashMap::new();
    meta.insert(
        "general.architecture".into(),
        GgufValue::String("llama".into()),
    );
    meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
    meta.insert(
        "llama.embedding_length".into(),
        GgufValue::Uint32(dim as u32),
    );
    meta.insert("llama.block_count".into(), GgufValue::Uint32(2));
    meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
    meta.insert(
        "llama.attention.head_count_kv".into(),
        GgufValue::Uint32(2),
    );
    meta.insert(
        "llama.feed_forward_length".into(),
        GgufValue::Uint32(ff as u32),
    );
    meta.insert(
        "tokenizer.ggml.model".into(),
        GgufValue::String("wordlevel".into()),
    );
    meta.insert(
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(
            ["<unk>", "<s>", "</s>", "hello", "a", "b", "c", "d"]
                .into_iter()
                .map(|s| GgufValue::String(s.into()))
                .collect(),
        ),
    );
    meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
    meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));

    let mut quant = Vec::new();
    let mut f32s: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    let embd = vec![0.02f32; vocab * dim];
    quant.push((
        "token_embd.weight".to_string(),
        vec![vocab as u64, dim as u64],
        quantize_q8_0(&embd),
    ));
    for layer in 0..2 {
        let q = vec![0.01f32; dim * dim];
        let kv = vec![0.01f32; n_kv * dim];
        let ffn_up = vec![0.01f32; ff * dim];
        let ffn_down = vec![0.01f32; dim * ff];
        let mut push = |suffix: &str, dims: Vec<u64>, data: &[f32]| {
            quant.push((
                format!("blk.{layer}.{suffix}"),
                dims,
                quantize_q8_0(data),
            ));
        };
        push("attn_q.weight", vec![dim as u64, dim as u64], &q);
        push("attn_k.weight", vec![dim as u64, n_kv as u64], &kv);
        push("attn_v.weight", vec![dim as u64, n_kv as u64], &kv);
        push("attn_output.weight", vec![dim as u64, dim as u64], &q);
        push("ffn_gate.weight", vec![dim as u64, ff as u64], &ffn_up);
        push("ffn_up.weight", vec![dim as u64, ff as u64], &ffn_up);
        push("ffn_down.weight", vec![ff as u64, dim as u64], &ffn_down);
        f32s.push((
            format!("blk.{layer}.attn_norm.weight"),
            vec![dim as u64],
            vec![1.0; dim],
        ));
        f32s.push((
            format!("blk.{layer}.ffn_norm.weight"),
            vec![dim as u64],
            vec![1.0; dim],
        ));
    }
    f32s.push((
        "output_norm.weight".into(),
        vec![dim as u64],
        vec![1.0; dim],
    ));
    let head = vec![0.02f32; vocab * dim];
    quant.push((
        "output.weight".into(),
        vec![vocab as u64, dim as u64],
        quantize_q8_0(&head),
    ));
    let f32_tensors: Vec<GgufWriteTensor> = f32s
        .iter()
        .map(|(name, dims, data)| GgufWriteTensor {
            name: name.clone(),
            dims: dims.clone(),
            data: data.as_slice(),
        })
        .collect();
    let quant_tensors: Vec<GgufWriteQuantTensor> = quant
        .iter()
        .map(|(name, dims, bytes)| GgufWriteQuantTensor {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: GgmlType::Q8_0,
            bytes: bytes.as_slice(),
        })
        .collect();
    let path = std::env::temp_dir().join("allpaca_rpc_packed_q8_remote.gguf");
    write_gguf_quant(&path, &meta, &f32_tensors, &quant_tensors).expect("write tiny q8 gguf");
    path
}

/// Packed Q8_0 layer on a live worker. Skips unless `ALLPACA_RPC_REMOTE=host:port`.
#[test]
fn gated_remote_packed_q8_generate() {
    let Ok(endpoint) = std::env::var("ALLPACA_RPC_REMOTE") else {
        eprintln!("skip gated_remote_packed_q8_generate: set ALLPACA_RPC_REMOTE=host:port");
        return;
    };
    let path = tiny_q8_gguf();
    let params = GenerationParams {
        max_tokens: 4,
        temperature: 0.0,
        seed: Some(2),
        ..GenerationParams::default()
    };

    let mut cpu_cfg = Config::compiled_defaults();
    cpu_cfg.model.synthetic = false;
    cpu_cfg.model.path = Some(path.clone());
    cpu_cfg.engine.cpu_only = true;
    cpu_cfg.speculative.enabled = false;
    cpu_cfg.kv.page_tokens = 8;
    let cpu = Engine::new(cpu_cfg).expect("cpu engine");
    let cpu_ids = cpu
        .generate("hello", params.clone())
        .expect("cpu generate")
        .token_ids;

    let mut cfg = Config::compiled_defaults();
    cfg.model.synthetic = false;
    cfg.model.path = Some(path);
    cfg.engine.cpu_only = true;
    cfg.speculative.enabled = false;
    cfg.kv.page_tokens = 8;
    cfg.rpc.endpoints = vec![endpoint];
    cfg.rpc.max_seq = 32;
    cfg.rpc.tensor_split = Some(vec![1.0, 1.0]);
    if let Ok(token) = std::env::var("ALLPACA_RPC_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            cfg.rpc.token = Some(token);
        }
    }

    let engine = Engine::new(cfg).expect("remote packed connect");
    let out = engine
        .generate("hello", params)
        .expect("remote packed generate");
    let stats = engine.stats();
    assert!(
        stats.rpc_roundtrips > 0,
        "packed layer must run, trips={}",
        stats.rpc_roundtrips
    );
    assert!(
        stats.rpc_h2d_bytes > 0 && stats.rpc_h2d_bytes < 20_000,
        "packed Q8 upload should stay under the F32 size, got {}",
        stats.rpc_h2d_bytes
    );
    assert_eq!(out.token_ids, cpu_ids);
    eprintln!(
        "remote packed q8 ok tokens={:?} roundtrips={} h2d_bytes={}",
        out.token_ids, stats.rpc_roundtrips, stats.rpc_h2d_bytes
    );
}
