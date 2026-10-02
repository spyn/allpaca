//! The served engine behind a swappable slot, so the model can be reloaded
//! in place (a new context size changes the KV budget, which the GPU layer
//! split is planned around, so it needs a fresh `Engine`).

use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use allpaca_core::Engine;
use allpaca_types::{AllpacaError, Config};
use serde::Serialize;

/// What the slot is doing, for `/v1/status` and the WebUI.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EngineStatus {
    /// `ready`, `loading` or `failed`.
    pub state: &'static str,
    pub message: Option<String>,
    /// Unix ms when this state began.
    pub since_ms: u64,
}

impl EngineStatus {
    fn now(state: &'static str, message: Option<String>) -> Self {
        let since_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            state,
            message,
            since_ms,
        }
    }
}

pub struct EngineSlot {
    engine: RwLock<Option<Arc<Engine>>>,
    /// The config the current engine was built from.
    config: RwLock<Config>,
    status: Mutex<EngineStatus>,
}

/// How long a reload waits for in-flight requests to let go of the engine.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(120);

impl EngineSlot {
    pub fn new(engine: Arc<Engine>, config: Config) -> Arc<Self> {
        Arc::new(Self {
            engine: RwLock::new(Some(engine)),
            config: RwLock::new(config),
            status: Mutex::new(EngineStatus::now("ready", None)),
        })
    }

    /// The engine, or `None` while a reload is in progress.
    pub fn get(&self) -> Option<Arc<Engine>> {
        self.engine.read().ok().and_then(|g| g.clone())
    }

    pub fn status(&self) -> EngineStatus {
        self.status
            .lock()
            .map(|s| s.clone())
            .unwrap_or_else(|_| EngineStatus::now("failed", Some("status lock".into())))
    }

    pub fn config(&self) -> Config {
        self.config
            .read()
            .map(|c| c.clone())
            .unwrap_or_else(|_| Config::compiled_defaults())
    }

    fn set_status(&self, state: &'static str, message: Option<String>) {
        if let Ok(mut s) = self.status.lock() {
            *s = EngineStatus::now(state, message);
        }
    }

    /// Rebuild the engine from `config`. Returns once the reload has started;
    /// progress shows in [`Self::status`]. The old engine is dropped (its
    /// VRAM freed) before the new one loads; if the new one fails, the old
    /// config is loaded again. `after` runs on each engine that comes up,
    /// with the config it was built from.
    pub fn reload(
        self: &Arc<Self>,
        config: Config,
        after: impl Fn(&Engine, &Config) + Send + Sync + 'static,
    ) -> Result<(), String> {
        {
            let mut s = self.status.lock().map_err(|_| "status lock".to_string())?;
            if s.state == "loading" {
                return Err("a reload is already in progress".into());
            }
            *s = EngineStatus::now("loading", Some("waiting for running requests".into()));
        }
        let slot = Arc::clone(self);
        std::thread::Builder::new()
            .name("allpaca-reload".into())
            .spawn(move || slot.reload_blocking(config, after))
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    fn reload_blocking(&self, config: Config, after: impl Fn(&Engine, &Config)) {
        let old = self.engine.write().ok().and_then(|mut g| g.take());
        if let Some(old) = old {
            // In-flight requests hold their own Arc; the VRAM is only freed
            // once they are done.
            let start = Instant::now();
            while Arc::strong_count(&old) > 1 {
                if start.elapsed() > DRAIN_TIMEOUT {
                    tracing::warn!("reload: requests still running after {DRAIN_TIMEOUT:?}; keeping the old engine");
                    if let Ok(mut g) = self.engine.write() {
                        *g = Some(old);
                    }
                    self.set_status(
                        "ready",
                        Some("reload cancelled: requests were still running".into()),
                    );
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            self.set_status("loading", Some("releasing the old model".into()));
            drop(old);
        }
        let previous = self.config();
        self.set_status("loading", Some("loading the model".into()));
        tracing::info!(kv_vram_bytes = config.kv.vram_bytes, "reload: building engine");
        match build(&config) {
            Ok(engine) => {
                after(&engine, &config);
                self.install(engine, config);
                self.set_status("ready", Some("reloaded".into()));
                tracing::info!("reload: engine ready");
            }
            Err(e) => {
                let msg = e.to_string();
                tracing::warn!(error = %msg, "reload failed; restoring the previous settings");
                self.set_status(
                    "loading",
                    Some(format!("reload failed ({msg}); restoring the previous settings")),
                );
                match build(&previous) {
                    Ok(engine) => {
                        after(&engine, &previous);
                        self.install(engine, previous);
                        self.set_status("failed", Some(format!("reload failed: {msg}; previous settings restored")));
                    }
                    Err(e2) => {
                        self.set_status(
                            "failed",
                            Some(format!("reload failed: {msg}; restoring also failed: {e2}")),
                        );
                    }
                }
            }
        }
    }

    fn install(&self, engine: Engine, config: Config) {
        if let Ok(mut g) = self.engine.write() {
            *g = Some(Arc::new(engine));
        }
        if let Ok(mut c) = self.config.write() {
            *c = config;
        }
    }
}

fn build(config: &Config) -> Result<Engine, AllpacaError> {
    Engine::new(config.clone())
}
