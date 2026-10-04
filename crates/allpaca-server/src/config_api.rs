//! Read-only view of the running configuration and of the HTTP endpoints,
//! for the WebUI Settings page (`GET /v1/config`).

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use serde_json::Value;

use crate::api::AppState;

/// One HTTP endpoint, as the Settings page lists it and builds curl
/// commands from. Keep in step with [`crate::api::router`].
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Endpoint {
    pub method: &'static str,
    pub path: &'static str,
    /// Needs a credential: the session cookie, the operator password as a
    /// bearer token or, on client endpoints, an API key.
    pub auth: bool,
    /// An API key may call it (see `api_access::is_client_route`).
    pub client: bool,
    pub summary: &'static str,
}

const fn ep(method: &'static str, path: &'static str, auth: bool, summary: &'static str) -> Endpoint {
    Endpoint {
        method,
        path,
        auth,
        client: crate::api_access::is_client_route_str(method, path),
        summary,
    }
}

pub const ENDPOINTS: &[Endpoint] = &[
    ep("POST", "/v1/chat/completions", true, "OpenAI-compatible chat completion; `stream: true` answers as server-sent events"),
    ep("GET", "/v1/models", true, "The served model, in OpenAI list format"),
    ep("GET", "/v1/status", true, "Engine stats, devices, remote GPUs, host load and reload state"),
    ep("GET", "/stats", true, "Engine counters only"),
    ep("GET", "/metrics", true, "Engine counters in Prometheus text format"),
    ep("GET", "/v1/metrics/otlp", true, "Engine counters as OTLP/JSON"),
    ep("GET", "/v1/logs", true, "Snapshot of the server's log ring"),
    ep("GET", "/v1/logs/stream", true, "Live log lines as server-sent events"),
    ep("GET", "/v1/config", true, "The running configuration (secrets redacted) and this endpoint list"),
    ep("GET", "/v1/api", true, "API switches, listen address and the API keys (without their secrets)"),
    ep("PUT", "/v1/api", true, "Turn the API or key lockdown on or off, or move the listen address"),
    ep("POST", "/v1/api/keys", true, "Create an API key; the key is in the reply and shown only once"),
    ep("DELETE", "/v1/api/keys/{id}", true, "Revoke an API key"),
    ep("GET", "/v1/model/settings", true, "The loaded model's saved settings, KV cost and layer placement"),
    ep("PUT", "/v1/model/settings", true, "Save context size, disabled experts or expert names for the loaded model"),
    ep("GET", "/v1/model/available", true, "Models found in the models folder"),
    ep("POST", "/v1/model/load", true, "Switch to a model from the models folder (unloads the current one)"),
    ep("POST", "/v1/model/reload", true, "Reload the current model with its saved settings"),
    ep("GET", "/v1/model/experts/usage", true, "Routing hit counts per layer and expert"),
    ep("POST", "/v1/model/experts/usage/reset", true, "Zero the routing hit counts"),
    ep("POST", "/v1/model/experts/recording", true, "Turn routing hit recording on or off"),
    ep("POST", "/v1/auth/login", false, "Exchange the password for a session cookie"),
    ep("POST", "/v1/auth/logout", false, "End the session"),
];

#[derive(Serialize)]
pub struct ConfigBody {
    version: &'static str,
    /// The TOML file the server was started with, if any.
    config_path: Option<String>,
    auth_required: bool,
    /// What the running engine was built from: file, CLI flags, the WebUI's
    /// model choice and the model's saved settings.
    config: Value,
    /// Compiled defaults, so the page can mark what was changed.
    defaults: Value,
    endpoints: &'static [Endpoint],
}

const REDACTED: &str = "(set, hidden)";

/// Replaces every secret-looking string with a marker, at any depth.
fn redact(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for (key, value) in map.iter_mut() {
                let secret = ["password", "token", "secret", "api_key"]
                    .iter()
                    .any(|s| key.to_ascii_lowercase().contains(s));
                if secret && value.as_str().is_some_and(|s| !s.is_empty()) {
                    *value = Value::String(REDACTED.into());
                } else {
                    redact(value);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(redact),
        _ => {}
    }
}

pub async fn get_config(State(state): State<AppState>) -> Json<ConfigBody> {
    let mut config = serde_json::to_value(state.engine.config()).unwrap_or(Value::Null);
    redact(&mut config);
    let defaults =
        serde_json::to_value(allpaca_types::Config::compiled_defaults()).unwrap_or(Value::Null);
    Json(ConfigBody {
        version: env!("CARGO_PKG_VERSION"),
        config_path: state
            .config_path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned()),
        auth_required: state.auth.is_enabled(),
        config,
        defaults,
        endpoints: ENDPOINTS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_hides_secrets_at_any_depth_and_keeps_the_rest() {
        let mut v = serde_json::json!({
            "http": {"host": "0.0.0.0", "password": "hunter2", "max_tokens": 4096},
            "rpc": {"token": "abc", "endpoints": ["a:1"]},
            "nested": [{"api_key": "k"}],
            "unset": {"password": null, "token": ""},
        });
        redact(&mut v);
        assert_eq!(v["http"]["password"], REDACTED);
        assert_eq!(v["rpc"]["token"], REDACTED);
        assert_eq!(v["nested"][0]["api_key"], REDACTED);
        assert_eq!(v["http"]["host"], "0.0.0.0");
        assert_eq!(v["http"]["max_tokens"], 4096);
        assert_eq!(v["rpc"]["endpoints"][0], "a:1");
        assert!(v["unset"]["password"].is_null());
        assert_eq!(v["unset"]["token"], "");
    }
}
