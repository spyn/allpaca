use std::net::TcpListener;

use allpaca_rpc::{serve_listener, CpuWorker, WorkerBackend};
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
    /// Device selector: `cpu`, `hip:N`. CUDA workers land after engine CUDA parity.
    #[arg(short = 'd', long = "device", default_value = "cpu")]
    device: String,
    /// Force the CPU kernel backend.
    #[arg(long)]
    cpu_only: bool,
    /// Advertised worker memory in MiB (CPU backend). 0 = 8 GiB default.
    #[arg(short = 'm', long = "mem", default_value_t = 0)]
    mem_mib: u64,
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
    if cli.host != "127.0.0.1" && cli.host != "localhost" && cli.host != "::1" {
        eprintln!(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
             WARNING: allpaca-rpc-server has no authentication.\n\
             Bound to {} — anyone who can reach this port can run\n\
             kernels and read worker memory. Do not expose it on an\n\
             open network. Use a private LAN or a VPN.\n\
             !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",
            cli.host
        );
    }

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = TcpListener::bind(&addr)?;
    tracing::info!(%addr, "allpaca-rpc-server listening");
    loop {
        let backend = open_backend(&cli)?;
        serve_listener(&listener, backend)?;
        tracing::info!("rpc client disconnected; waiting for next");
    }
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
    if cli.device.starts_with("cuda:") || cli.device.starts_with("CUDA:") {
        return Err(
            "CUDA RPC workers require engine CUDA parity; use --cpu-only or hip:N for v1".into(),
        );
    }
    Err(format!("unknown --device {} (expected cpu or hip:N)", cli.device).into())
}
