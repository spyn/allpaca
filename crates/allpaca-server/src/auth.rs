use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rand::RngCore;
use serde::Deserialize;

use crate::api::AppState;

pub const COOKIE_NAME: &str = "allpaca_session";
const SESSION_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);

#[derive(Clone)]
pub struct AuthState {
    password: Option<String>,
    sessions: Arc<Mutex<HashMap<String, Instant>>>,
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

    pub fn authorized(&self, headers: &HeaderMap) -> bool {
        let Some(expected) = self.password.as_deref() else {
            return true;
        };
        if let Some(bearer) = bearer_token(headers) {
            if passwords_match(expected, bearer) {
                return true;
            }
        }
        if let Some(cookie) = cookie_value(headers, COOKIE_NAME) {
            if let Ok(mut map) = self.sessions.lock() {
                map.retain(|_, at| at.elapsed() < SESSION_TTL);
                if let Some(at) = map.get(&cookie) {
                    if at.elapsed() < SESSION_TTL {
                        return true;
                    }
                }
            }
        }
        false
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

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let raw = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    raw.strip_prefix("Bearer ").or_else(|| raw.strip_prefix("bearer "))
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

pub async fn login(
    State(state): State<AppState>,
    Json(body): Json<LoginBody>,
) -> Response {
    if !state.auth.is_enabled() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({"ok": true, "auth": false})),
        )
            .into_response();
    }
    match state.auth.login(&body.password) {
        Some(token) => (
            StatusCode::OK,
            [(header::SET_COOKIE, session_cookie(&token))],
            Json(serde_json::json!({"ok": true})),
        )
            .into_response(),
        None => (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({"error": {"message": "invalid password", "type": "fatal"}})),
        )
            .into_response(),
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

pub async fn require_auth(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    if state.auth.authorized(request.headers()) {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({"error": {"message": "unauthorized", "type": "fatal"}})),
    )
        .into_response()
}
