//! `/v1/model/*`: per-model settings (context size, disabled experts, expert
//! names), routing-hit counts, and in-place reload.

use std::collections::BTreeMap;
use std::sync::Arc;

use allpaca_core::KvEstimate;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::{engine_unavailable, AppState};
use crate::engine_slot::EngineStatus;
use crate::model_settings::ModelSettings;

/// Context sizes the API accepts (tokens).
const MIN_CONTEXT: u64 = 256;
const MAX_CONTEXT: u64 = 1 << 24;
const MAX_NAME_CHARS: usize = 80;

#[derive(Serialize)]
struct ExpertInfo {
    n_layers: usize,
    n_experts: usize,
    n_active: usize,
    recording: bool,
    disabled_total: usize,
}

#[derive(Serialize)]
struct KvInfo {
    #[serde(flatten)]
    estimate: KvEstimate,
    /// Context the engine's current KV budget holds.
    context_tokens: u64,
}

#[derive(Serialize)]
struct SettingsBody {
    model: Option<String>,
    path: Option<String>,
    persisted: bool,
    settings: ModelSettings,
    experts: Option<ExpertInfo>,
    kv: Option<KvInfo>,
    /// The saved context differs from what the running engine was built with.
    reload_pending: bool,
    engine: EngineStatus,
}

fn bad_request(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({"error": {"message": msg.into(), "type": "invalid_request"}})),
    )
        .into_response()
}

fn settings_body(state: &AppState) -> SettingsBody {
    let settings = state.settings.get();
    let engine = state.engine();
    let experts = engine.as_ref().and_then(|e| e.expert_controls()).and_then(|ctl| {
        let shape = ctl.shape();
        (shape.n_experts > 0).then(|| ExpertInfo {
            n_layers: shape.n_layers,
            n_experts: shape.n_experts,
            n_active: shape.n_active,
            recording: ctl.recording(),
            disabled_total: ctl.disabled().iter().map(Vec::len).sum(),
        })
    });
    let kv = engine.as_ref().and_then(|e| e.kv_estimate()).map(|estimate| KvInfo {
        context_tokens: estimate.tokens_for(estimate.budget_bytes),
        estimate,
    });
    let reload_pending = match (&kv, settings.kv_vram_bytes) {
        (Some(kv), Some(want)) => want != kv.estimate.budget_bytes,
        _ => false,
    };
    SettingsBody {
        model: engine.as_ref().map(|e| e.model_id()),
        path: state.settings.path().map(|p| p.display().to_string()),
        persisted: state.settings.path().is_some(),
        settings,
        experts,
        kv,
        reload_pending,
        engine: state.engine.status(),
    }
}

pub async fn get_settings(State(state): State<AppState>) -> Response {
    Json(settings_body(&state)).into_response()
}

#[derive(Debug, Deserialize)]
pub struct SettingsUpdate {
    pub context_tokens: Option<u64>,
    /// Replaces the whole disabled set (layer -> expert indices).
    pub disabled_experts: Option<BTreeMap<u32, Vec<u32>>>,
    /// Replaces all names.
    pub expert_names: Option<BTreeMap<String, String>>,
}

pub async fn put_settings(
    State(state): State<AppState>,
    Json(update): Json<SettingsUpdate>,
) -> Response {
    let Some(engine) = state.engine() else {
        return engine_unavailable(&state);
    };
    let mut next = state.settings.get();
    if let Some(ctx) = update.context_tokens {
        if !(MIN_CONTEXT..=MAX_CONTEXT).contains(&ctx) {
            return bad_request(format!("context_tokens must be {MIN_CONTEXT}..={MAX_CONTEXT}"));
        }
        let Some(est) = engine.kv_estimate() else {
            return bad_request("no model loaded to size the context for");
        };
        next.context_tokens = Some(ctx);
        next.kv_vram_bytes = Some(est.bytes_for(ctx).max(1 << 20));
    }
    if let Some(names) = update.expert_names {
        next.expert_names = names
            .into_iter()
            .map(|(k, v)| (k, v.trim().chars().take(MAX_NAME_CHARS).collect::<String>()))
            .filter(|(_, v)| !v.is_empty())
            .collect();
    }
    if let Some(disabled) = update.disabled_experts {
        next.disabled_experts = disabled.into_iter().filter(|(_, ids)| !ids.is_empty()).collect();
        // Validate and apply live before saving: nothing is written if the
        // engine rejects it.
        let Some(ctl) = engine.expert_controls() else {
            return bad_request("no model loaded");
        };
        let dense = match next.disabled_dense(ctl.shape().n_layers) {
            Ok(d) => d,
            Err(e) => return bad_request(e),
        };
        if let Err(e) = ctl.set_disabled(&dense) {
            return bad_request(e.to_string());
        }
    }
    if let Err(e) = state.settings.set(next) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": format!("saving settings: {e}"), "type": "fatal"}})),
        )
            .into_response();
    }
    Json(settings_body(&state)).into_response()
}

#[derive(Serialize)]
struct UsageBody {
    n_layers: usize,
    n_experts: usize,
    recording: bool,
    /// Layer-major hit counts, `n_layers * n_experts`.
    counts: Vec<u64>,
}

pub async fn get_usage(State(state): State<AppState>) -> Response {
    let Some(engine) = state.engine() else {
        return engine_unavailable(&state);
    };
    let res = tokio::task::spawn_blocking(move || {
        let ctl = engine.expert_controls();
        let counts = engine.expert_usage();
        (ctl, counts)
    })
    .await;
    match res {
        Ok((Some(ctl), Ok(counts))) => {
            let shape = ctl.shape();
            Json(UsageBody {
                n_layers: shape.n_layers,
                n_experts: shape.n_experts,
                recording: ctl.recording(),
                counts,
            })
            .into_response()
        }
        Ok((None, _)) => bad_request("no model loaded"),
        Ok((_, Err(e))) => bad_request(e.to_string()),
        Err(e) => bad_request(e.to_string()),
    }
}

#[derive(Deserialize)]
pub struct RecordingBody {
    pub on: bool,
}

pub async fn set_recording(
    State(state): State<AppState>,
    Json(body): Json<RecordingBody>,
) -> Response {
    let Some(ctl) = state.engine().and_then(|e| e.expert_controls()) else {
        return engine_unavailable(&state);
    };
    ctl.set_recording(body.on);
    Json(serde_json::json!({"recording": ctl.recording()})).into_response()
}

pub async fn reset_usage(State(state): State<AppState>) -> Response {
    let Some(ctl) = state.engine().and_then(|e| e.expert_controls()) else {
        return engine_unavailable(&state);
    };
    ctl.reset_usage();
    StatusCode::NO_CONTENT.into_response()
}

/// Rebuild the engine with the saved settings (a new context size needs it).
pub async fn reload(State(state): State<AppState>) -> Response {
    let mut config = state.engine.config();
    state.settings.get().apply_to_config(&mut config);
    let store = Arc::clone(&state.settings);
    match state
        .engine
        .reload(config, move |engine| store.get().apply_to_engine(engine))
    {
        Ok(()) => (StatusCode::ACCEPTED, Json(state.engine.status())).into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": {"message": e, "type": "conflict"}})),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::router;
    use crate::auth::AuthState;
    use crate::logs::LogRing;
    use allpaca_core::Engine;
    use allpaca_types::{CliOverrides, Config};
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    /// One-layer Mixtral-style GGUF: 4 experts, top-2.
    fn write_tiny_moe_gguf(path: &std::path::Path) {
        use allpaca_gguf::{write_gguf, GgufValue, GgufWriteTensor};
        use std::collections::HashMap;
        let (dim, vocab, ff, n_exp) = (8usize, 4usize, 16usize, 4usize);
        let mut meta = HashMap::new();
        meta.insert("general.architecture".into(), GgufValue::String("llama".into()));
        meta.insert("llama.vocab_size".into(), GgufValue::Uint32(vocab as u32));
        meta.insert("llama.embedding_length".into(), GgufValue::Uint32(dim as u32));
        meta.insert("llama.block_count".into(), GgufValue::Uint32(1));
        meta.insert("llama.attention.head_count".into(), GgufValue::Uint32(4));
        meta.insert("llama.attention.head_count_kv".into(), GgufValue::Uint32(1));
        meta.insert("llama.feed_forward_length".into(), GgufValue::Uint32(ff as u32));
        meta.insert("llama.expert_count".into(), GgufValue::Uint32(n_exp as u32));
        meta.insert("llama.expert_used_count".into(), GgufValue::Uint32(2));
        meta.insert("tokenizer.ggml.model".into(), GgufValue::String("wordlevel".into()));
        meta.insert(
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                ["<unk>", "<s>", "</s>", "hello"]
                    .iter()
                    .map(|t| GgufValue::String((*t).into()))
                    .collect(),
            ),
        );
        meta.insert("tokenizer.ggml.bos_token_id".into(), GgufValue::Uint32(1));
        meta.insert("tokenizer.ggml.eos_token_id".into(), GgufValue::Uint32(2));
        let w = 0.01f32;
        let d = dim as u64;
        let owned: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight", vec![vocab as u64, d], vec![w; vocab * dim]),
            ("blk.0.attn_norm.weight", vec![d], vec![1.0; dim]),
            ("blk.0.attn_q.weight", vec![d, d], vec![w; dim * dim]),
            ("blk.0.attn_k.weight", vec![2, d], vec![w; 2 * dim]),
            ("blk.0.attn_v.weight", vec![2, d], vec![w; 2 * dim]),
            ("blk.0.attn_output.weight", vec![d, d], vec![w; dim * dim]),
            ("blk.0.ffn_norm.weight", vec![d], vec![1.0; dim]),
            ("blk.0.ffn_gate_inp.weight", vec![n_exp as u64, d], vec![w; n_exp * dim]),
            ("blk.0.ffn_gate_exps.weight", vec![n_exp as u64, ff as u64, d], vec![w; n_exp * ff * dim]),
            ("blk.0.ffn_up_exps.weight", vec![n_exp as u64, ff as u64, d], vec![w; n_exp * ff * dim]),
            ("blk.0.ffn_down_exps.weight", vec![n_exp as u64, d, ff as u64], vec![w; n_exp * dim * ff]),
            ("output_norm.weight", vec![d], vec![1.0; dim]),
        ];
        let views: Vec<GgufWriteTensor<'_>> = owned
            .iter()
            .map(|(name, dims, data)| GgufWriteTensor {
                name: (*name).to_string(),
                dims: dims.clone(),
                data,
            })
            .collect();
        write_gguf(path, &meta, &views).unwrap();
    }

    fn moe_state(dir: &std::path::Path) -> (AppState, std::path::PathBuf) {
        let model = dir.join("tiny-moe.gguf");
        write_tiny_moe_gguf(&model);
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            cpu_only: true,
            seed: Some(1),
            ..CliOverrides::default()
        });
        cfg.model.path = Some(model.clone());
        let state = AppState::new(
            Arc::new(Engine::new(cfg.clone()).unwrap()),
            cfg,
            AuthState::from_password(None),
            LogRing::new(64),
        );
        (state, model)
    }

    async fn call(
        app: &axum::Router,
        method: &str,
        uri: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder().method(method).uri(uri);
        let req = match body {
            Some(b) => req
                .header("content-type", "application/json")
                .body(Body::from(b.to_string())),
            None => req.body(Body::empty()),
        }
        .unwrap();
        let res = app.clone().oneshot(req).await.unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("allpaca-model-api-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Disabling experts applies live, is validated, and is saved next to the
    /// model; names are trimmed and saved; usage counts while recording.
    #[tokio::test]
    async fn experts_disable_rename_and_count_through_the_api() {
        let dir = tmp_dir("experts");
        let (state, model) = moe_state(&dir);
        let app = router(state.clone());

        let (st, body) = call(&app, "GET", "/v1/model/settings", None).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(body["experts"]["n_experts"], 4);
        assert_eq!(body["experts"]["n_active"], 2);
        assert_eq!(body["persisted"], true);

        // Three of four disabled would leave fewer than top-2: rejected, not saved.
        let (st, _) = call(
            &app,
            "PUT",
            "/v1/model/settings",
            Some(serde_json::json!({"disabled_experts": {"0": [0, 1, 2]}})),
        )
        .await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        let file = model.with_file_name("tiny-moe.gguf.allpaca.json");
        assert!(!file.exists());

        let (st, body) = call(
            &app,
            "PUT",
            "/v1/model/settings",
            Some(serde_json::json!({
                "disabled_experts": {"0": [1]},
                "expert_names": {"0:1": "  greetings  ", "0:2": "   "}
            })),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{body}");
        assert_eq!(body["experts"]["disabled_total"], 1);
        let saved: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert_eq!(saved["disabled_experts"]["0"], serde_json::json!([1]));
        assert_eq!(saved["expert_names"], serde_json::json!({"0:1": "greetings"}));

        let (st, _) = call(&app, "POST", "/v1/model/experts/recording", Some(serde_json::json!({"on": true}))).await;
        assert_eq!(st, StatusCode::OK);
        let engine = state.engine().unwrap();
        tokio::task::spawn_blocking(move || {
            engine
                .generate(
                    "hello",
                    allpaca_types::GenerationParams {
                        max_tokens: 4,
                        temperature: 0.0,
                        ..Default::default()
                    },
                )
                .unwrap()
        })
        .await
        .unwrap();
        let (st, usage) = call(&app, "GET", "/v1/model/experts/usage", None).await;
        assert_eq!(st, StatusCode::OK);
        let counts: Vec<u64> = serde_json::from_value(usage["counts"].clone()).unwrap();
        assert_eq!(counts.len(), 4);
        assert_eq!(counts[1], 0, "disabled expert routed: {counts:?}");
        assert!(counts.iter().sum::<u64>() > 0);
        let (st, _) = call(&app, "POST", "/v1/model/experts/usage/reset", None).await;
        assert_eq!(st, StatusCode::NO_CONTENT);
        let (_, usage) = call(&app, "GET", "/v1/model/experts/usage", None).await;
        assert!(usage["counts"].as_array().unwrap().iter().all(|c| c == 0));

        // A fresh server on the same model picks the settings up.
        let (again, _) = moe_state(&dir);
        let saved = again.settings.get();
        saved.apply_to_engine(&again.engine().unwrap());
        let ctl = again.engine().unwrap().expert_controls().unwrap();
        assert_eq!(ctl.disabled(), vec![vec![1]]);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A new context size is saved, marks a reload pending, and the reload
    /// rebuilds the engine with that KV budget and keeps disabled experts.
    #[tokio::test]
    async fn context_change_reloads_the_engine_in_place() {
        let dir = tmp_dir("reload");
        let (state, _) = moe_state(&dir);
        let app = router(state.clone());
        let (_, before) = call(&app, "GET", "/v1/model/settings", None).await;
        let per_token = before["kv"]["full_bytes_per_token"].as_u64().unwrap();
        assert!(per_token > 0);

        let ctx = 40_000u64;
        let (st, body) = call(
            &app,
            "PUT",
            "/v1/model/settings",
            Some(serde_json::json!({"context_tokens": ctx, "disabled_experts": {"0": [3]}})),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{body}");
        assert_eq!(body["reload_pending"], true);
        let want = body["settings"]["kv_vram_bytes"].as_u64().unwrap();
        assert_eq!(want, (ctx * per_token).max(1 << 20));

        let old = state.engine().unwrap();
        let (st, _) = call(&app, "POST", "/v1/model/reload", None).await;
        assert_eq!(st, StatusCode::ACCEPTED);
        // The old engine is only released once nothing holds it.
        drop(old);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let body = loop {
            let (_, body) = call(&app, "GET", "/v1/model/settings", None).await;
            if body["engine"]["state"] == "ready" && body["kv"].is_object() {
                break body;
            }
            assert!(std::time::Instant::now() < deadline, "reload did not finish: {body}");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        };
        assert_eq!(body["engine"]["message"], "reloaded", "{body}");
        assert_eq!(body["kv"]["budget_bytes"].as_u64().unwrap(), want);
        assert_eq!(body["reload_pending"], false);
        assert_eq!(body["experts"]["disabled_total"], 1);
        let (st, _) = call(&app, "GET", "/v1/models", None).await;
        assert_eq!(st, StatusCode::OK);
        let _ = std::fs::remove_dir_all(dir);
    }
}
