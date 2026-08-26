use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use allpaca_core::Engine;
use allpaca_types::{Config, FinishReason, GenerationParams};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub config: Config,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/stats", get(stats))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct JsonSchemaSpec {
    #[serde(default)]
    pub name: Option<String>,
    pub schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub json_schema: Option<JsonSchemaSpec>,
    #[serde(default)]
    pub grammar: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn prompt_from_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&msg.role);
        out.push_str(": ");
        out.push_str(&msg.content);
        out.push('\n');
    }
    out.push_str("assistant:");
    out
}

fn params_from_request(req: &ChatCompletionRequest, config: &Config) -> GenerationParams {
    let format = req.response_format.as_ref();
    let json_mode = format
        .map(|f| f.kind == "json_object")
        .unwrap_or(config.structured_output.json_mode);
    let json_schema = format.and_then(|f| {
        if f.kind == "json_schema" {
            f.json_schema
                .as_ref()
                .and_then(|spec| serde_json::to_string(&spec.schema).ok())
        } else {
            None
        }
    });
    let grammar = format.and_then(|f| f.grammar.clone());
    GenerationParams {
        max_tokens: req.max_tokens.unwrap_or(32),
        temperature: req.temperature.unwrap_or(config.sampling.temperature),
        top_k: req.top_k.unwrap_or(config.sampling.top_k),
        top_p: req.top_p.unwrap_or(config.sampling.top_p),
        json_mode,
        json_schema,
        grammar,
        seed: req.seed.or(config.engine.determinism_seed),
        stop_token_ids: Vec::new(),
    }
}

fn finish_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if !state.engine.has_model() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": {"message": "no model loaded", "type": "fatal"}})),
        )
            .into_response();
    }
    let prompt = prompt_from_messages(&req.messages);
    let params = params_from_request(&req, &state.config);
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.engine.model_id());
    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();

    if req.stream {
        return stream_completion(state, prompt, params, model, id, created).into_response();
    }

    if state.config.engine.max_active_requests > 1 {
        let mut stream = Box::pin(state.engine.generate_stream(prompt.clone(), params));
        let mut text = String::new();
        let mut completion_tokens = 0usize;
        let mut finish = FinishReason::Stop;
        while let Some(item) = stream.next().await {
            match item {
                Ok(ev) => {
                    text.push_str(&ev.text);
                    completion_tokens += 1;
                    if let Some(reason) = ev.finish_reason {
                        finish = reason;
                        break;
                    }
                }
                Err(e) => return error_response(e),
            }
        }
        return Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant",
                    content: text,
                },
                finish_reason: Some(finish_name(finish).into()),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens,
                total_tokens: completion_tokens,
            },
        })
        .into_response();
    }

    let engine = Arc::clone(&state.engine);
    match tokio::task::spawn_blocking(move || engine.generate(&prompt, params)).await {
        Ok(Ok(out)) => Json(ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessageOut {
                    role: "assistant",
                    content: out.text,
                },
                finish_reason: Some(finish_name(out.finish_reason).into()),
            }],
            usage: Usage {
                prompt_tokens: out.prompt_tokens,
                completion_tokens: out.completion_tokens,
                total_tokens: out.prompt_tokens + out.completion_tokens,
            },
        })
        .into_response(),
        Ok(Err(e)) => error_response(e),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": {"message": e.to_string(), "type": "fatal"}})),
        )
            .into_response(),
    }
}

fn stream_completion(
    state: AppState,
    prompt: String,
    params: GenerationParams,
    model: String,
    id: String,
    created: u64,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(16);
    let use_async_poll = state.config.engine.max_active_requests > 1;
    let engine = Arc::clone(&state.engine);
    if use_async_poll {
        tokio::spawn(async move {
            let mut stream = Box::pin(engine.generate_stream(prompt, params));
            let first = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
            });
            let _ = tx
                .send(Ok(Event::default().data(first.to_string())))
                .await;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(ev) => {
                        let chunk = serde_json::json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": ev.text},
                                "finish_reason": ev.finish_reason.map(finish_name)
                            }]
                        });
                        if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let err = serde_json::json!({"error": {"message": e.to_string()}});
                        let _ = tx.send(Ok(Event::default().data(err.to_string()))).await;
                        break;
                    }
                }
            }
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        });
    } else {
        let rt = tokio::runtime::Handle::current();
        tokio::task::spawn_blocking(move || {
            let mut stream = Box::pin(state.engine.generate_stream(prompt, params));
            rt.block_on(async {
                let first = serde_json::json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
                });
                let _ = tx
                    .send(Ok(Event::default().data(first.to_string())))
                    .await;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(ev) => {
                            let chunk = serde_json::json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": ev.text},
                                    "finish_reason": ev.finish_reason.map(finish_name)
                                }]
                            });
                            if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let err = serde_json::json!({"error": {"message": e.to_string()}});
                            let _ = tx.send(Ok(Event::default().data(err.to_string()))).await;
                            break;
                        }
                    }
                }
                let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
            });
        });
    }
    Sse::new(async_stream::stream! {
        let mut rx = rx;
        while let Some(item) = rx.recv().await {
            yield item;
        }
    })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let id = state.engine.model_id();
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id,
            object: "model",
            created: now_secs(),
            owned_by: "allpaca",
        }],
    })
}

async fn stats(State(state): State<AppState>) -> Json<allpaca_types::StatsSnapshot> {
    Json(state.engine.stats())
}

fn error_response(e: allpaca_types::AllpacaError) -> Response {
    let status = match e.class() {
        allpaca_types::ErrorClass::Fatal => StatusCode::INTERNAL_SERVER_ERROR,
        allpaca_types::ErrorClass::Degraded => StatusCode::OK,
        allpaca_types::ErrorClass::Transient => StatusCode::TOO_MANY_REQUESTS,
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": e.to_string(),
                "type": e.class().to_string(),
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use allpaca_types::CliOverrides;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn test_state() -> AppState {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(1),
            ..CliOverrides::default()
        });
        AppState {
            engine: Arc::new(Engine::new(cfg.clone()).unwrap()),
            config: cfg,
        }
    }

    #[tokio::test]
    async fn models_and_stats() {
        let app = router(test_state());
        let res = app
            .clone()
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let models: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(models["object"], "list");
        assert_eq!(models["data"][0]["id"], "synthetic");
        assert_eq!(models["data"][0]["object"], "model");

        let res = app
            .oneshot(Request::builder().uri("/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json.get("pipeline_mode").is_some());
        assert!(json.get("page_migrations").is_some());
        assert_eq!(json["model_id"], "synthetic");
    }

    fn multi_request_state() -> AppState {
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(1),
            ..CliOverrides::default()
        });
        cfg.engine.max_active_requests = 2;
        AppState {
            engine: Arc::new(Engine::new(cfg.clone()).unwrap()),
            config: cfg,
        }
    }

    #[tokio::test]
    async fn overlapping_chat_completions_drive_poll_when_max_active_two() {
        let state = multi_request_state();
        let app = router(state.clone());
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "seed": 1
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let (res_a, res_b) = tokio::join!(
            app.clone().oneshot(make_req()),
            app.oneshot(make_req()),
        );
        assert_eq!(res_a.expect("a").status(), StatusCode::OK);
        assert_eq!(res_b.expect("b").status(), StatusCode::OK);
        let stats = state.engine.stats();
        assert!(
            stats.poll_ticks > 0,
            "HTTP chat with max_active_requests=2 must use Engine::poll; stats={stats:?}"
        );
    }

    #[tokio::test]
    async fn stats_exposes_continuous_batching_fields_after_overlapping_chat() {
        let state = multi_request_state();
        let app = router(state);
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "seed": 1
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let (res_a, res_b) = tokio::join!(
            app.clone().oneshot(make_req()),
            app.clone().oneshot(make_req()),
        );
        assert_eq!(res_a.expect("a").status(), StatusCode::OK);
        assert_eq!(res_b.expect("b").status(), StatusCode::OK);

        let res = app
            .oneshot(Request::builder().uri("/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            json["poll_ticks"].as_u64().unwrap_or(0) > 0,
            "GET /stats must report poll_ticks after multi-request chat; json={json}"
        );
        assert!(
            json["poll_decode_picks"].as_u64().unwrap_or(0) > 0,
            "GET /stats must report decode picks; json={json}"
        );
        assert!(
            json.get("poll_prefill_picks").is_some(),
            "GET /stats must expose poll_prefill_picks; json={json}"
        );
        assert!(
            json.get("poll_wait_ms_ema").is_some(),
            "GET /stats must expose poll_wait_ms_ema; json={json}"
        );
        assert!(
            json.get("active_requests").is_some(),
            "GET /stats must expose queue depth as active_requests; json={json}"
        );
    }

    #[tokio::test]
    async fn overlapping_chat_completions_do_not_500() {
        let app = router(multi_request_state());
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "seed": 1
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let (res_a, res_b) = tokio::join!(
            app.clone().oneshot(make_req()),
            app.oneshot(make_req()),
        );
        let res_a = res_a.expect("request a");
        let res_b = res_b.expect("request b");
        assert_ne!(
            res_a.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "overlapping chat must not 500"
        );
        assert_ne!(
            res_b.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "overlapping chat must not 500"
        );
        assert_eq!(res_a.status(), StatusCode::OK);
        assert_eq!(res_b.status(), StatusCode::OK);
        let bytes_a = res_a.into_body().collect().await.unwrap().to_bytes();
        let bytes_b = res_b.into_body().collect().await.unwrap().to_bytes();
        let json_a: serde_json::Value = serde_json::from_slice(&bytes_a).unwrap();
        let json_b: serde_json::Value = serde_json::from_slice(&bytes_b).unwrap();
        assert_eq!(json_a["object"], "chat.completion");
        assert_eq!(json_b["object"], "chat.completion");
        assert!(json_a["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
        assert!(json_b["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn overlapping_stream_completions_drive_poll_when_max_active_two() {
        let state = multi_request_state();
        let app = router(state.clone());
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 4,
            "temperature": 0.0,
            "seed": 1
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let (res_a, res_b) = tokio::join!(
            app.clone().oneshot(make_req()),
            app.oneshot(make_req()),
        );
        let res_a = res_a.expect("a");
        let res_b = res_b.expect("b");
        assert_eq!(res_a.status(), StatusCode::OK);
        assert_eq!(res_b.status(), StatusCode::OK);
        let body_a = res_a.into_body().collect().await.unwrap().to_bytes();
        let body_b = res_b.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&body_a).contains("[DONE]"));
        assert!(String::from_utf8_lossy(&body_b).contains("[DONE]"));
        let stats = state.engine.stats();
        assert!(
            stats.poll_ticks > 0,
            "overlapping SSE must use fair poll; stats={stats:?}"
        );
    }

    #[tokio::test]
    async fn stats_exposes_draft_depth_field() {
        let app = router(test_state());
        let res = app
            .oneshot(Request::builder().uri("/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            json.get("draft_depth").is_some(),
            "GET /stats must expose draft_depth; json={json}"
        );
    }

    #[tokio::test]
    async fn chat_completion_non_stream() {
        let app = router(test_state());
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "seed": 1
        });
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"][0]["message"]["content"].is_string());
        assert!(json["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn chat_completion_stream() {
        let app = router(test_state());
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "max_tokens": 4,
            "temperature": 0.0,
            "seed": 1
        });
        let res = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            text.contains("chat.completion.chunk"),
            "stream must emit OpenAI chunks: {text}"
        );
        assert!(text.contains("[DONE]"), "stream must terminate with [DONE]: {text}");
        assert!(
            text.contains("\"content\"") || text.contains("assistant"),
            "stream must carry token or role deltas: {text}"
        );
    }
}
