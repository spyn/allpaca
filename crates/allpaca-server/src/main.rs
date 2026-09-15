mod api;
mod auth;
mod cli;
mod host;
mod logs;
mod otlp_push;
mod tune;
mod ui;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use allpaca_core::Engine;
use allpaca_types::Config;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

use crate::cli::Cli;

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let log_ring = logs::LogRing::new(2048);
    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env().add_directive("allpaca=info".parse()?))
        .with(tracing_subscriber::fmt::layer())
        .with(log_ring.layer())
        .init();

    let cli = Cli::parse_args();
    let mut config = if let Some(path) = &cli.config {
        Config::from_toml_file(path)?
    } else {
        Config::compiled_defaults()
    };
    config.merge_cli(&cli.to_overrides());
    config.validate()?;

    if cli.tune {
        let config_path = cli.config.clone();
        return crate::tune::run(cli.model.clone(), config, config_path);
    }

    let password = match config
        .http
        .password
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        Some(p) => p.to_string(),
        None => {
            let generated = auth::generate_password();
            tracing::info!("webui password: {generated}");
            generated
        }
    };

    let engine = Engine::new(config.clone())?;
    let engine = Arc::new(engine);
    let state = api::AppState::new(
        Arc::clone(&engine),
        config.clone(),
        auth::AuthState::from_password(Some(password)),
        log_ring,
    );
    let _otlp_push = otlp_push::maybe_spawn_otlp_push(
        engine,
        config.telemetry.otlp_endpoint.clone(),
        Duration::from_secs(15),
    );
    let app = api::router(state);
    let addr = SocketAddr::new(config.http.host.parse()?, config.http.port);
    tracing::info!(%addr, "allpaca-server listening (CPU-only={})", config.engine.cpu_only);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
