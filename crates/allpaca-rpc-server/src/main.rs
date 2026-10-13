use std::net::TcpListener;

use allpaca_rpc::{CpuWorker, WorkerBackend};
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "allpaca-rpc-server",
    about = "Allpaca RPC worker (llama.cpp rpc-server analogue)"
)]
struct Cli {
    /// Bind address. Default localhost; never expose on an open network.
    #[arg(short = 'H', long = "host", default_value = "127.0.0.1")]
    host: String,
    /// Listen port.
    #[arg(short = 'p', long = "port", default_value_t = 50052)]
    port: u16,
    /// Device selector: `cpu`, `hip` (every HIP device), `hip:N`, or `cuda:N`.
    #[arg(short = 'd', long = "device", default_value = "cpu")]
    device: String,
    /// Force the CPU kernel backend.
    #[arg(long)]
    cpu_only: bool,
    /// Advertised worker memory in MiB (CPU backend). 0 = 8 GiB default.
    #[arg(short = 'm', long = "mem", default_value_t = 0)]
    mem_mib: u64,
    /// Required to bind a non-loopback address. Also requires `--token`.
    #[arg(long)]
    insecure_bind: bool,
    /// Shared HELLO secret. Required for non-loopback binds. Env: `ALLPACA_RPC_TOKEN`.
    #[arg(long)]
    token: Option<String>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("allpaca=info".parse()?))
        .init();

    let cli = Cli::parse();
    let token = rpc_token(&cli);
    allpaca_rpc::check_rpc_bind(&cli.host, cli.insecure_bind, &token)?;
    if !allpaca_rpc::is_loopback_bind_host(&cli.host) {
        tracing::warn!(
            host = %cli.host,
            "rpc worker bound off-loopback; HELLO token required, still no TLS"
        );
    }

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = TcpListener::bind(&addr)?;
    tracing::info!(%addr, "allpaca-rpc-server listening");
    let token = if token.is_empty() { None } else { Some(token) };
    loop {
        let backend = open_backend(&cli)?;
        // A coordinator that dies mid-session resets the socket. Its buffers
        // go with the backend; keep serving the next client.
        // HIP workers also host whole Qwen3.8-Flash-Next (qwen4exp) blocks.
        let extensions = match backend {
            #[cfg(feature = "hip")]
            WorkerBackend::Hip(_) => vec![allpaca_core::qwen4_remote::worker_extension()],
            _ => Vec::new(),
        };
        match allpaca_rpc::serve_listener_ext(&listener, backend, token.as_deref(), extensions) {
            Ok(()) => tracing::info!("rpc client disconnected; waiting for next"),
            Err(e) => tracing::warn!(error = %e, "rpc client dropped; waiting for next"),
        }
    }
}

fn rpc_token(cli: &Cli) -> String {
    if let Some(t) = cli
        .token
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return t.to_string();
    }
    std::env::var("ALLPACA_RPC_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default()
}

fn open_backend(cli: &Cli) -> Result<WorkerBackend, Box<dyn std::error::Error>> {
    if cli.cpu_only || cli.device.eq_ignore_ascii_case("cpu") {
        let bytes = if cli.mem_mib == 0 {
            8 << 30
        } else {
            cli.mem_mib.saturating_mul(1024 * 1024)
        };
        return Ok(WorkerBackend::Cpu(CpuWorker::new(bytes)));
    }
    if cli.device.eq_ignore_ascii_case("hip") || cli.device.eq_ignore_ascii_case("all") {
        #[cfg(feature = "hip")]
        {
            return Ok(WorkerBackend::Hip(allpaca_rpc::HipWorker::open_all()?));
        }
        #[cfg(not(feature = "hip"))]
        {
            return Err("rebuild with --features hip to pool HIP devices".into());
        }
    }
    if let Some(rest) = cli
        .device
        .strip_prefix("hip:")
        .or_else(|| cli.device.strip_prefix("HIP:"))
    {
        let id: i32 = rest.parse()?;
        #[cfg(feature = "hip")]
        {
            return Ok(WorkerBackend::Hip(allpaca_rpc::HipWorker::new(id)?));
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = id;
            return Err("rebuild with --features hip for hip:N workers".into());
        }
    }
    if let Some(rest) = cli
        .device
        .strip_prefix("cuda:")
        .or_else(|| cli.device.strip_prefix("CUDA:"))
    {
        let id: i32 = rest.parse()?;
        #[cfg(feature = "cuda")]
        {
            return Ok(WorkerBackend::Cuda(allpaca_rpc::CudaWorker::new(id)?));
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = id;
            return Err("rebuild with --features cuda for cuda:N workers".into());
        }
    }
    Err(format!(
        "unknown --device {} (expected cpu, hip, hip:N, or cuda:N)",
        cli.device
    )
    .into())
}
