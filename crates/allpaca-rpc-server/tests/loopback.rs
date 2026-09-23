use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

use allpaca_core::Engine;
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
