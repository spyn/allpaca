mod api;
mod api_access;
mod auth;
mod cli;
mod config_api;
mod engine_slot;
mod host;
mod image_url;
mod logs;
mod model_api;
mod model_scan;
mod model_settings;
mod otlp_push;
mod security;
mod startup;
mod systemone;
mod tune;
mod ui;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use allpaca_core::Engine;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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
    let loaded = startup::load_startup_config(cli.config.as_deref(), &std::env::current_dir()?)?;
    let mut config = loaded.config;
    config.merge_cli(&cli.to_overrides());
    config.validate()?;

    if cli.tune {
        let config_path = loaded.path.clone();
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
            eprintln!("webui password: {generated}");
            tracing::info!("webui password generated (printed to stderr, not captured in logs)");
            generated
        }
    };

    // The model's saved settings: context size before the engine is built,
    // disabled experts once it is up.
    // `base` (without the saved settings) is what a model switch starts from.
    let base = config.clone();
    let saved = model_settings::SettingsStore::open(&config).get();
    saved.apply_to_config(&mut config);
    let engine = Engine::new(config.clone())?;
    saved.apply_to_engine(&engine);
    let mut state = api::AppState::new(
        Arc::new(engine),
        base,
        auth::AuthState::from_password(Some(password)),
        log_ring,
    );
    state.config_path = loaded.path.clone();
    // Saved API settings (on/off, lockdown, keys, listen address). A
    // `--host` / `--port` on the command line wins for this run.
    let api_store =
        api_access::ApiStore::open(api_access::ApiStore::path_for(loaded.path.as_deref()));
    let saved_api = api_store.get();
    state.api = Arc::new(api_store);
    let host = match (&cli.host, saved_api.host) {
        (None, Some(saved)) => saved,
        _ => config.http.host.clone(),
    };
    let port = match (cli.port, saved_api.port) {
        (None, Some(saved)) => saved,
        _ => config.http.port,
    };
    let addr = SocketAddr::new(host.parse()?, port);
    let (listen, mut rebind_rx) = api_access::ListenCtl::new(addr);
    state.listen = Some(Arc::new(listen));
    let _otlp_push = otlp_push::maybe_spawn_otlp_push(
        Arc::clone(&state.engine),
        config.telemetry.otlp_endpoint.clone(),
        Duration::from_secs(15),
    );
    let app = api::router(state);
    let mut addr = addr;
    let mut listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "allpaca-server listening (CPU-only={})", config.engine.cpu_only);
    // Serve until the API page asks for another address, then move there.
    // Connections already open on the old address finish on their own.
    loop {
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let mut stop = Some(stop);
        let service = app
            .clone()
            .into_make_service_with_connect_info::<SocketAddr>();
        let mut server = tokio::spawn(async move {
            axum::serve(listener, service)
                .with_graceful_shutdown(async move {
                    let _ = stopped.await;
                })
                .await
        });
        listener = loop {
            tokio::select! {
                done = &mut server => {
                    done??;
                    return Ok(());
                }
                req = rebind_rx.recv() => {
                    let Some((want, reply)) = req else {
                        continue;
                    };
                    match tokio::net::TcpListener::bind(want).await {
                        Ok(next) => {
                            if let Some(stop) = stop.take() {
                                let _ = stop.send(());
                            }
                            addr = want;
                            let _ = reply.send(Ok(()));
                            break next;
                        }
                        // The old listener may hold the port (same port,
                        // wider or narrower host): let go of it first.
                        Err(first) if want.port() == addr.port() => {
                            if let Some(stop) = stop.take() {
                                let _ = stop.send(());
                            }
                            match bind_retry(want).await {
                                Ok(next) => {
                                    addr = want;
                                    let _ = reply.send(Ok(()));
                                    break next;
                                }
                                Err(_) => {
                                    let back = bind_retry(addr).await?;
                                    let _ = reply.send(Err(first.to_string()));
                                    break back;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = reply.send(Err(e.to_string()));
                        }
                    }
                }
            }
        };
    }
}

/// Bind once the previous listener on the port has closed (it closes as its
/// serve task notices the shutdown signal).
async fn bind_retry(addr: SocketAddr) -> std::io::Result<tokio::net::TcpListener> {
    let mut last = None;
    for _ in 0..40 {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => return Ok(l),
            Err(e) => last = Some(e),
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Err(last.unwrap_or_else(|| std::io::Error::other("bind failed")))
}
