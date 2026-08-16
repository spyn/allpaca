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
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
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
    GenerationParams {
        max_tokens: req.max_tokens.unwrap_or(32),
        temperature: req.temperature.unwrap_or(config.sampling.temperature),
        top_k: req.top_k.unwrap_or(config.sampling.top_k),
        top_p: req.top_p.unwrap_or(config.sampling.top_p),
        json_mode: req
            .response_format
            .as_ref()
            .map(|f| f.kind == "json_object")
            .unwrap_or(config.structured_output.json_mode),
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
