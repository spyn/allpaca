use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, FromRequestParts, Request, State};
use axum::http::request::Parts;
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rand::RngCore;
use serde::Deserialize;

use crate::api::AppState;

pub const COOKIE_NAME: &str = "allpaca_session";
const SESSION_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);
const LOGIN_WINDOW: Duration = Duration::from_secs(60);
const LOGIN_MAX_FAILS: usize = 5;

#[derive(Clone)]
pub struct AuthState {
    password: Option<String>,
    sessions: Arc<Mutex<HashMap<String, Instant>>>,
    login_fails: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl AuthState {
    pub fn from_password(password: Option<String>) -> Self {
        let password = password.and_then(|p| {
            let t = p.trim().to_string();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        });
        Self {
            password,
            sessions: Arc::new(Mutex::new(HashMap::new())),
            login_fails: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.password.is_some()
    }

    pub fn login(&self, password: &str) -> Option<String> {
        let expected = self.password.as_deref()?;
        if !passwords_match(expected, password) {
            return None;
        }
        let token = random_token();
        if let Ok(mut map) = self.sessions.lock() {
            map.retain(|_, at| at.elapsed() < SESSION_TTL);
            map.insert(token.clone(), Instant::now());
        }
        Some(token)
    }

    pub fn logout(&self, token: &str) {
        if let Ok(mut map) = self.sessions.lock() {
            map.remove(token);
        }
    }

    /// The request carries a live WebUI session cookie.
    pub fn session_valid(&self, headers: &HeaderMap) -> bool {
        let Some(cookie) = cookie_value(headers, COOKIE_NAME) else {
            return false;
        };
        let Ok(mut map) = self.sessions.lock() else {
            return false;
        };
        map.retain(|_, at| at.elapsed() < SESSION_TTL);
        map.contains_key(&cookie)
    }

    /// `token` is the operator password.
    pub fn password_matches(&self, token: &str) -> bool {
        self.password
            .as_deref()
            .is_some_and(|expected| passwords_match(expected, token))
    }

    fn client_key(addr: Option<SocketAddr>) -> String {
        addr.map(|a| a.ip().to_string())
            .unwrap_or_else(|| "unknown".into())
    }

    fn login_allowed(&self, key: &str) -> bool {
        let Ok(mut map) = self.login_fails.lock() else {
            return false;
        };
        let now = Instant::now();
        map.retain(|_, fails| {
            fails.retain(|at| now.saturating_duration_since(*at) < LOGIN_WINDOW);
            !fails.is_empty()
        });
        map.get(key)
            .is_none_or(|fails| fails.len() < LOGIN_MAX_FAILS)
    }

    fn record_login_fail(&self, key: &str) {
        let Ok(mut map) = self.login_fails.lock() else {
            return;
        };
        let now = Instant::now();
        let fails = map.entry(key.to_string()).or_default();
        fails.retain(|at| now.saturating_duration_since(*at) < LOGIN_WINDOW);
        fails.push(now);
    }

    fn clear_login_fails(&self, key: &str) {
        if let Ok(mut map) = self.login_fails.lock() {
            map.remove(key);
        }
    }
}

pub fn generate_password() -> String {
    let mut bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex_encode(&bytes)
}

pub fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    hex_encode(&bytes)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

fn passwords_match(expected: &str, given: &str) -> bool {
    if expected.len() != given.len() {
        return false;
    }
    expected
        .bytes()
        .zip(given.bytes())
        .fold(0u8, |acc, (a, b)| acc | (a ^ b))
        == 0
}

pub fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let raw = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    raw.strip_prefix("Bearer ")
        .or_else(|| raw.strip_prefix("bearer "))
}

pub fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<String> {
    let raw = headers.get(header::COOKIE)?.to_str().ok()?;
    for part in raw.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            if k == name {
                return Some(v.to_string());
            }
        }
    }
    None
}

pub fn session_cookie(token: &str) -> String {
    format!(
        "{COOKIE_NAME}={token}; HttpOnly; SameSite=Lax; Path=/; Max-Age={}",
        SESSION_TTL.as_secs()
    )
}

pub fn clear_session_cookie() -> String {
    format!("{COOKIE_NAME}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
}

#[derive(Debug, Deserialize)]
pub struct LoginBody {
    pub password: String,
}

pub struct ClientAddr(Option<SocketAddr>);

impl<S> FromRequestParts<S> for ClientAddr
where
    S: Send + Sync,
{
    type Rejection = Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        Ok(Self(
            parts
                .extensions
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ConnectInfo(addr)| *addr),
        ))
    }
}

pub async fn login(
    State(state): State<AppState>,
    ClientAddr(addr): ClientAddr,
    Json(body): Json<LoginBody>,
) -> Response {
    if !state.auth.is_enabled() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"ok": true, "auth": false})),
        )
            .into_response();
    }
    let key = AuthState::client_key(addr);
    if !state.auth.login_allowed(&key) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "60")],
            Json(serde_json::json!({"error": {"message": "too many login attempts", "type": "fatal"}})),
        )
            .into_response();
    }
    match state.auth.login(&body.password) {
        Some(token) => {
            state.auth.clear_login_fails(&key);
            (
                StatusCode::OK,
                [(header::SET_COOKIE, session_cookie(&token))],
                Json(serde_json::json!({"ok": true})),
            )
                .into_response()
        }
        None => {
            state.auth.record_login_fail(&key);
            (
                StatusCode::UNAUTHORIZED,
                Json(
                    serde_json::json!({"error": {"message": "invalid password", "type": "fatal"}}),
                ),
            )
                .into_response()
        }
    }
}

pub async fn logout(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Some(token) = cookie_value(&headers, COOKIE_NAME) {
        state.auth.logout(&token);
    }
    (
        StatusCode::OK,
        [(header::SET_COOKIE, clear_session_cookie())],
        Json(serde_json::json!({"ok": true})),
    )
        .into_response()
}

pub async fn require_auth(State(state): State<AppState>, request: Request, next: Next) -> Response {
    use crate::api_access::{check, Denied};
    match check(&state, request.method(), request.uri().path(), request.headers()) {
        Ok(()) => next.run(request).await,
        Err(Denied::Unauthorized) => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "unauthorized", "type": "fatal"}})),
        )
            .into_response(),
        Err(Denied::Disabled) => (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error": {"message": "the API is disabled on this server", "type": "forbidden"}})),
        )
            .into_response(),
    }
}
