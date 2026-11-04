//! Who may call the HTTP API, and where it listens. Runtime settings the
//! operator changes from the WebUI's API page, kept in `allpaca-api.json`
//! next to the config file:
//!
//! - `enabled`: off refuses every bearer credential; only signed-in WebUI
//!   sessions (the cookie) keep working.
//! - `require_keys` ("lockdown"): on, a client needs an API key; off, the
//!   client endpoints answer without one.
//! - `host` / `port`: the listen address, changed without restarting.
//! - `keys`: generated API keys, stored as SHA-256 hashes.
//!
//! The operator (cookie session, or the password as a bearer token) reaches
//! every endpoint. An API key reaches the client endpoints only
//! ([`is_client_route`]): inference and read-only telemetry, never model
//! management, configuration, logs or the keys themselves.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{mpsc, oneshot};

use crate::api::AppState;

const FILE_NAME: &str = "allpaca-api.json";
const SETTINGS_VERSION: u32 = 1;
const KEY_PREFIX: &str = "alp_";
const MAX_KEYS: usize = 64;
const MAX_DESCRIPTION_CHARS: usize = 80;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiKey {
    pub id: String,
    pub description: String,
    /// The first characters of the key, to tell keys apart in the list.
    pub prefix: String,
    /// SHA-256 of the key, hex. The key itself is shown once and not kept.
    pub hash: String,
    pub created_ms: u64,
    #[serde(default)]
    pub last_used_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiSettings {
    pub version: u32,
    pub enabled: bool,
    pub require_keys: bool,
    /// Listen address; unset keeps the config file's `http.host`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<ApiKey>,
}

impl Default for ApiSettings {
    /// What the server did before these settings existed: API on, a
    /// credential required.
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            enabled: true,
            require_keys: true,
            host: None,
            port: None,
            keys: Vec::new(),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hash_key(key: &str) -> String {
    hex(&Sha256::digest(key.as_bytes()))
}

fn bytes_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The settings and their file. In memory only without a path (tests).
pub struct ApiStore {
    path: Option<PathBuf>,
    inner: Mutex<ApiSettings>,
}

impl ApiStore {
    pub fn in_memory() -> Self {
        Self {
            path: None,
            inner: Mutex::new(ApiSettings::default()),
        }
    }

    /// Next to the config file, else in the working directory.
    pub fn path_for(config_path: Option<&Path>) -> PathBuf {
        config_path
            .and_then(Path::parent)
            .map(|dir| dir.join(FILE_NAME))
            .unwrap_or_else(|| PathBuf::from(FILE_NAME))
    }

    /// A missing file is the defaults; an unreadable one is logged and the
    /// defaults (credential required) are used until the next save.
    pub fn open(path: PathBuf) -> Self {
        let settings = match std::fs::read_to_string(&path) {
            Ok(src) => match serde_json::from_str::<ApiSettings>(&src) {
                Ok(s) => {
                    tracing::info!(path = %path.display(), keys = s.keys.len(), "api settings loaded");
                    s
                }
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "api settings unreadable; using defaults");
                    ApiSettings::default()
                }
            },
            Err(_) => ApiSettings::default(),
        };
        Self {
            path: Some(path),
            inner: Mutex::new(settings),
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn get(&self) -> ApiSettings {
        self.inner.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Change and persist (temp file, then rename). Nothing changes when the
    /// write fails.
    pub fn update(&self, change: impl FnOnce(&mut ApiSettings)) -> Result<ApiSettings, String> {
        let mut g = self.inner.lock().map_err(|_| "api settings lock".to_string())?;
        let mut next = g.clone();
        change(&mut next);
        next.version = SETTINGS_VERSION;
        if let Some(path) = &self.path {
            let json = serde_json::to_string_pretty(&next).map_err(|e| e.to_string())?;
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
            std::fs::rename(&tmp, path)
                .map_err(|e| format!("rename to {}: {e}", path.display()))?;
        }
        *g = next.clone();
        Ok(next)
    }

    /// A new key: the secret (returned once) and its stored entry.
    pub fn create_key(&self, description: &str) -> Result<(String, ApiKey), String> {
        let description = description.trim();
        if description.is_empty() {
            return Err("a key needs a description".into());
        }
        if description.chars().count() > MAX_DESCRIPTION_CHARS {
            return Err(format!("description is longer than {MAX_DESCRIPTION_CHARS} characters"));
        }
        if self.get().keys.len() >= MAX_KEYS {
            return Err(format!("at most {MAX_KEYS} keys; revoke one first"));
        }
        let mut raw = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut raw);
        let secret = format!("{KEY_PREFIX}{}", hex(&raw));
        let mut id = [0u8; 6];
        rand::thread_rng().fill_bytes(&mut id);
        let entry = ApiKey {
            id: hex(&id),
            description: description.to_string(),
            prefix: secret.chars().take(KEY_PREFIX.len() + 6).collect(),
            hash: hash_key(&secret),
            created_ms: now_ms(),
            last_used_ms: 0,
        };
        let stored = entry.clone();
        self.update(move |s| s.keys.push(stored))?;
        Ok((secret, entry))
    }

    pub fn revoke(&self, id: &str) -> Result<bool, String> {
        if !self.get().keys.iter().any(|k| k.id == id) {
            return Ok(false);
        }
        self.update(|s| s.keys.retain(|k| k.id != id))?;
        Ok(true)
    }

    /// Whether `token` is a live key. Notes the use in memory; it reaches
    /// the file with the next change.
    pub fn key_matches(&self, token: &str) -> bool {
        if !token.starts_with(KEY_PREFIX) {
            return false;
        }
        let hash = hash_key(token);
        let Ok(mut g) = self.inner.lock() else {
            return false;
        };
        let mut hit = false;
        for key in &mut g.keys {
            if bytes_eq(key.hash.as_bytes(), hash.as_bytes()) {
                key.last_used_ms = now_ms();
                hit = true;
            }
        }
        hit
    }
}

/// Endpoints an API key (or, with lockdown off, anyone) may call.
pub fn is_client_route(method: &Method, path: &str) -> bool {
    is_client_route_str(method.as_str(), path)
}

/// [`is_client_route`] for the endpoint table (a `const` context).
pub const fn is_client_route_str(method: &str, path: &str) -> bool {
    const fn eq(a: &str, b: &str) -> bool {
        let (a, b) = (a.as_bytes(), b.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }
    const CLIENT: [(&str, &str); 9] = [
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/responses"),
        ("POST", "/v1/systemone"),
        ("GET", "/v1/models"),
        ("GET", "/v1/status"),
        ("GET", "/stats"),
        ("GET", "/metrics"),
        ("GET", "/v1/metrics/otlp"),
    ];
    let mut i = 0;
    while i < CLIENT.len() {
        if eq(method, CLIENT[i].0) && eq(path, CLIENT[i].1) {
            return true;
        }
        i += 1;
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// No valid credential (401).
    Unauthorized,
    /// A valid credential, but the API is turned off (403).
    Disabled,
}

/// The access decision for one request to a protected route.
pub fn check(state: &AppState, method: &Method, path: &str, headers: &HeaderMap) -> Result<(), Denied> {
    // No password at all (tests, embedders): nothing to guard with.
    if !state.auth.is_enabled() || state.auth.session_valid(headers) {
        return Ok(());
    }
    let api = state.api.get();
    let allow = |ok: bool| if ok { Ok(()) } else { Err(Denied::Disabled) };
    match crate::auth::bearer_token(headers) {
        Some(token) if state.auth.password_matches(token) => allow(api.enabled),
        Some(token) if is_client_route(method, path) && state.api.key_matches(token) => {
            allow(api.enabled)
        }
        Some(_) => Err(Denied::Unauthorized),
        None if api.enabled && !api.require_keys && is_client_route(method, path) => Ok(()),
        None => Err(Denied::Unauthorized),
    }
}

/// Asks the serve loop to listen somewhere else.
pub struct ListenCtl {
    tx: mpsc::Sender<(SocketAddr, oneshot::Sender<Result<(), String>>)>,
    current: Mutex<SocketAddr>,
}

pub type RebindRx = mpsc::Receiver<(SocketAddr, oneshot::Sender<Result<(), String>>)>;

impl ListenCtl {
    pub fn new(current: SocketAddr) -> (Self, RebindRx) {
        let (tx, rx) = mpsc::channel(1);
        (
            Self {
                tx,
                current: Mutex::new(current),
            },
            rx,
        )
    }

    pub fn current(&self) -> SocketAddr {
        self.current
            .lock()
            .map(|a| *a)
            .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 0)))
    }

    /// Returns once the server listens on `addr` (the old address is closed),
    /// or with the reason it could not, in which case it still listens on
    /// the old one.
    pub async fn rebind(&self, addr: SocketAddr) -> Result<(), String> {
        if addr == self.current() {
            return Ok(());
        }
        let (reply, done) = oneshot::channel();
        self.tx
            .send((addr, reply))
            .await
            .map_err(|_| "the server is shutting down".to_string())?;
        done.await.map_err(|_| "the server is shutting down".to_string())??;
        if let Ok(mut cur) = self.current.lock() {
            *cur = addr;
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct KeyView {
    id: String,
    description: String,
    prefix: String,
    created_ms: u64,
    last_used_ms: u64,
}

#[derive(Serialize)]
struct Interface {
    name: String,
    addr: String,
}

#[derive(Serialize)]
pub struct ApiBody {
    enabled: bool,
    require_keys: bool,
    /// Where the server listens now.
    host: String,
    port: u16,
    /// The config file's (or command line's) address, used when no
    /// override is saved.
    config_host: String,
    config_port: u16,
    /// A change of address applies at once; false means on the next start.
    live_rebind: bool,
    settings_path: Option<String>,
    /// This machine's IPv4 addresses, as choices for the listen address.
    interfaces: Vec<Interface>,
    keys: Vec<KeyView>,
}

fn interfaces() -> Vec<Interface> {
    let nets = sysinfo::Networks::new_with_refreshed_list();
    let mut out: Vec<Interface> = nets
        .iter()
        .flat_map(|(name, data)| {
            data.ip_networks().iter().filter_map(move |net| match net.addr {
                IpAddr::V4(v4) if !v4.is_loopback() && !v4.is_link_local() => Some(Interface {
                    name: name.clone(),
                    addr: v4.to_string(),
                }),
                _ => None,
            })
        })
        .collect();
    out.sort_by(|a, b| a.addr.cmp(&b.addr));
    out
}

fn body(state: &AppState) -> ApiBody {
    let s = state.api.get();
    let (config_host, config_port) = (state.config.http.host.clone(), state.config.http.port);
    let listening = state.listen.as_ref().map(|l| l.current());
    ApiBody {
        enabled: s.enabled,
        require_keys: s.require_keys,
        host: listening.map_or_else(|| s.host.clone().unwrap_or_else(|| config_host.clone()), |a| a.ip().to_string()),
        port: listening.map_or_else(|| s.port.unwrap_or(config_port), |a| a.port()),
        config_host,
        config_port,
        live_rebind: state.listen.is_some(),
        settings_path: state.api.path().map(|p| p.to_string_lossy().into_owned()),
        interfaces: interfaces(),
        keys: s
            .keys
            .into_iter()
            .map(|k| KeyView {
                id: k.id,
                description: k.description,
                prefix: k.prefix,
                created_ms: k.created_ms,
                last_used_ms: k.last_used_ms,
            })
            .collect(),
    }
}

fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message.into(), "type": kind}})),
    )
        .into_response()
}

pub async fn get_api(State(state): State<AppState>) -> Json<ApiBody> {
    Json(body(&state))
}

#[derive(Debug, Default, Deserialize)]
pub struct ApiUpdate {
    pub enabled: Option<bool>,
    pub require_keys: Option<bool>,
    /// An IP address; empty goes back to the config file's host.
    pub host: Option<String>,
    /// `0` goes back to the config file's port.
    pub port: Option<u16>,
}

pub async fn put_api(State(state): State<AppState>, Json(update): Json<ApiUpdate>) -> Response {
    let saved = state.api.get();
    let host = match update.host.as_deref().map(str::trim) {
        None => saved.host.clone(),
        Some("") => None,
        Some(h) => match h.parse::<IpAddr>() {
            Ok(ip) => Some(ip.to_string()),
            Err(_) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request",
                    format!("'{h}' is not an IP address (use 127.0.0.1, 0.0.0.0 or one of this machine's addresses)"),
                )
            }
        },
    };
    let port = match update.port {
        None => saved.port,
        Some(0) => None,
        Some(p) => Some(p),
    };
    // The config file's own address is "no override": a later edit of the
    // file then takes effect.
    let host = host.filter(|h| *h != state.config.http.host);
    let port = port.filter(|p| *p != state.config.http.port);
    if host != saved.host || port != saved.port {
        if let Some(listen) = &state.listen {
            let ip: IpAddr = host
                .as_deref()
                .unwrap_or(&state.config.http.host)
                .parse()
                .unwrap_or(IpAddr::from([127, 0, 0, 1]));
            let addr = SocketAddr::new(ip, port.unwrap_or(state.config.http.port));
            if let Err(e) = listen.rebind(addr).await {
                return error(
                    StatusCode::CONFLICT,
                    "conflict",
                    format!("cannot listen on {addr}: {e}; still on {}", listen.current()),
                );
            }
            tracing::info!(%addr, "api: listen address changed");
        }
    }
    let result = state.api.update(|s| {
        if let Some(v) = update.enabled {
            s.enabled = v;
        }
        if let Some(v) = update.require_keys {
            s.require_keys = v;
        }
        s.host = host;
        s.port = port;
    });
    match result {
        Ok(s) => {
            tracing::info!(enabled = s.enabled, require_keys = s.require_keys, "api: settings saved");
            Json(body(&state)).into_response()
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "fatal", format!("saving api settings: {e}")),
    }
}

#[derive(Debug, Deserialize)]
pub struct NewKey {
    pub description: String,
}

pub async fn create_key(State(state): State<AppState>, Json(req): Json<NewKey>) -> Response {
    match state.api.create_key(&req.description) {
        Ok((secret, entry)) => {
            tracing::info!(id = %entry.id, description = %entry.description, "api: key created");
            Json(serde_json::json!({
                "key": secret,
                "id": entry.id,
                "description": entry.description,
                "prefix": entry.prefix,
                "created_ms": entry.created_ms,
            }))
            .into_response()
        }
        Err(e) => error(StatusCode::BAD_REQUEST, "invalid_request", e),
    }
}

pub async fn revoke_key(State(state): State<AppState>, UrlPath(id): UrlPath<String>) -> Response {
    match state.api.revoke(&id) {
        Ok(true) => {
            tracing::info!(%id, "api: key revoked");
            Json(serde_json::json!({"ok": true})).into_response()
        }
        Ok(false) => error(StatusCode::NOT_FOUND, "not_found", "no such key"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "fatal", format!("saving api settings: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_match_only_while_they_exist_and_are_stored_hashed() {
        let dir = std::env::temp_dir().join(format!("allpaca-api-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(FILE_NAME);
        let _ = std::fs::remove_file(&path);
        let store = ApiStore::open(path.clone());
        assert!(store.create_key("   ").is_err(), "a description is required");
        let (secret, entry) = store.create_key("  ci runner ").unwrap();
        assert!(secret.starts_with(KEY_PREFIX) && secret.len() == KEY_PREFIX.len() + 48);
        assert_eq!(entry.description, "ci runner");
        assert!(store.key_matches(&secret));
        assert!(!store.key_matches(&format!("{secret}0")));
        assert!(!store.key_matches("not-a-key"));
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert!(!on_disk.contains(&secret), "the key itself must not be stored");
        assert!(on_disk.contains(&entry.hash));
        // A restart reads the same keys back.
        let reopened = ApiStore::open(path.clone());
        assert!(reopened.key_matches(&secret));
        assert!(reopened.revoke(&entry.id).unwrap());
        assert!(!reopened.revoke(&entry.id).unwrap());
        assert!(!reopened.key_matches(&secret));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn defaults_keep_the_api_on_and_locked() {
        let s = ApiSettings::default();
        assert!(s.enabled && s.require_keys && s.host.is_none() && s.keys.is_empty());
        let parsed: ApiSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed, s);
    }

    #[test]
    fn client_routes_exclude_management() {
        assert!(is_client_route(&Method::POST, "/v1/chat/completions"));
        assert!(is_client_route(&Method::POST, "/v1/messages"));
        assert!(is_client_route(&Method::POST, "/v1/responses"));
        assert!(is_client_route(&Method::POST, "/v1/systemone"));
        assert!(is_client_route(&Method::GET, "/v1/models"));
        for (m, p) in [
            (Method::POST, "/v1/model/load"),
            (Method::PUT, "/v1/model/settings"),
            (Method::GET, "/v1/config"),
            (Method::GET, "/v1/api"),
            (Method::POST, "/v1/api/keys"),
            (Method::GET, "/v1/logs"),
            (Method::GET, "/v1/chat/completions"),
        ] {
            assert!(!is_client_route(&m, p), "{m} {p}");
        }
    }
}
