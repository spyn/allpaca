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
    /// Keep the weights of hosted model blocks in this directory and take
    /// them from there the next time (HIP workers). Env: `ALLPACA_WORKER_CACHE`.
    #[arg(long)]
    cache_dir: Option<std::path::PathBuf>,
    /// Most disk the weight cache uses, in GiB; the weights used longest ago go first.
    #[arg(long, default_value_t = 64)]
    cache_gib: u64,
    /// Keep coordinators' prompt-cache checkpoints in this much RAM, in GiB
    /// (their `[prompt_cache] worker = true`), and as much again of the
    /// state of the Flash-Next blocks it hosts. 0 = not served.
    #[arg(long, default_value_t = 0)]
    prompt_cache_gib: u64,
    /// Print the fingerprint of the sources this worker was built from and exit.
    #[arg(long)]
    build_id: bool,
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
    if cli.build_id {
        println!("{}", allpaca_rpc::BUILD_ID);
        return Ok(());
    }
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
    tracing::info!(%addr, build = allpaca_rpc::BUILD_ID, "allpaca-rpc-server listening");
    let token = if token.is_empty() { None } else { Some(token) };
    // One store for every connection: it outlives a coordinator's restart.
    let prompts = (cli.prompt_cache_gib > 0).then(|| {
        std::sync::Arc::new(std::sync::Mutex::new(allpaca_core::prompt_store::WorkerStore::new(
            (cli.prompt_cache_gib << 30) as usize,
        )))
    });
    loop {
        let backend = open_backend(&cli)?;
        // A coordinator that dies mid-session resets the socket. Its buffers
        // go with the backend; keep serving the next client.
        // HIP workers also host whole Qwen3.8-Flash-Next (qwen4exp) blocks.
        let mut extensions = match backend {
            #[cfg(feature = "hip")]
            WorkerBackend::Hip(_) => {
                let cache = cli
                    .cache_dir
                    .clone()
                    .or_else(|| std::env::var_os("ALLPACA_WORKER_CACHE").map(Into::into))
                    .map(|dir| allpaca_core::qwen4_remote::WeightCache::new(dir, cli.cache_gib << 30));
                // Coordinators' prompt caches keep Flash-Next blocks' state here too.
                vec![allpaca_core::qwen4_remote::worker_extension_saving(
                    cache,
                    (cli.prompt_cache_gib << 30) as usize,
                )]
            }
            _ => Vec::new(),
        };
        if let Some(store) = &prompts {
            extensions.push(allpaca_core::prompt_store::worker_extension(store.clone()));
        }
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
