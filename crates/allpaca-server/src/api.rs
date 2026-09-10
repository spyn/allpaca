use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use allpaca_core::{Engine, ImageInput};
use allpaca_types::{AllpacaError, Config, FatalError, FinishReason, GenerationParams};
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
        .route("/metrics", get(prometheus_metrics))
        .route("/v1/metrics/otlp", get(otlp_metrics))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlSpec },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImageUrlSpec {
    pub url: String,
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

fn messages_to_prompt_and_images(
    messages: &[ChatMessage],
) -> Result<(String, Vec<ImageInput>), AllpacaError> {
    let mut out = String::new();
    let mut images = Vec::new();
    for msg in messages {
        out.push_str(&msg.role);
        out.push_str(": ");
        match &msg.content {
            MessageContent::Text(text) => out.push_str(text),
            MessageContent::Parts(parts) => {
                let mut texts = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => texts.push(text.as_str()),
                        ContentPart::ImageUrl { image_url } => {
                            images.push(load_image_url(&image_url.url)?);
                        }
                    }
                }
                out.push_str(&texts.join(" "));
            }
        }
        out.push('\n');
    }
    out.push_str("assistant:");
    Ok((out, images))
}

fn load_image_url(url: &str) -> Result<ImageInput, AllpacaError> {
    let raw = url.trim();
    if raw.starts_with("data:") {
        return load_data_uri(raw);
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        return fetch_http_image(raw);
    }
    let path = resolve_image_path(raw)?;
    let bytes = std::fs::read(&path).map_err(|e| {
        AllpacaError::Fatal(FatalError::Io(format!(
            "read image {}: {e}",
            path.display()
        )))
    })?;
    let mime = mime_from_path(&path);
    Ok(ImageInput::new(bytes, mime))
}

fn fetch_http_image(url: &str) -> Result<ImageInput, AllpacaError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent("allpaca/0.1")
        .build()
        .map_err(|e| {
            AllpacaError::Fatal(FatalError::Io(format!("http client init: {e}")))
        })?;
    let resp = client.get(url).send().map_err(|e| {
        AllpacaError::Fatal(FatalError::Io(format!("fetch image_url: {e}")))
    })?;
    if !resp.status().is_success() {
        return Err(AllpacaError::Fatal(FatalError::Io(format!(
            "fetch image_url {}: HTTP {}",
            url,
            resp.status()
        ))));
    }
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.split(';').next().unwrap_or(s).trim().to_ascii_lowercase())
        .filter(|s| s.starts_with("image/"))
        .unwrap_or_else(|| mime_from_url_path(url));
    let bytes = resp.bytes().map_err(|e| {
        AllpacaError::Fatal(FatalError::Io(format!("read image_url body: {e}")))
    })?;
    Ok(ImageInput::new(bytes.to_vec(), mime))
}

fn mime_from_url_path(url: &str) -> String {
    let path = url.split('?').next().unwrap_or(url);
    mime_from_path(std::path::Path::new(path))
}

fn load_data_uri(url: &str) -> Result<ImageInput, AllpacaError> {
    // data:[<mime>][;base64],<data>
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| AllpacaError::Fatal(FatalError::InvalidConfig("invalid data URI".into())))?;
    let (meta, payload) = rest.split_once(',').ok_or_else(|| {
        AllpacaError::Fatal(FatalError::InvalidConfig(
            "data URI missing comma separator".into(),
        ))
    })?;
    let mut mime = "application/octet-stream".to_string();
    let mut is_b64 = false;
    for part in meta.split(';').filter(|p| !p.is_empty()) {
        if part.eq_ignore_ascii_case("base64") {
            is_b64 = true;
        } else if part.contains('/') {
            mime = part.to_ascii_lowercase();
        }
    }
    let bytes = if is_b64 {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(payload.trim().as_bytes())
            .map_err(|e| {
                AllpacaError::Fatal(FatalError::InvalidConfig(format!(
                    "invalid data URI base64: {e}"
                )))
            })?
    } else {
        // Percent-decoded raw payload is uncommon for images; keep bytes as UTF-8 text.
        payload.as_bytes().to_vec()
    };
    Ok(ImageInput::new(bytes, mime))
}

fn resolve_image_path(url: &str) -> Result<std::path::PathBuf, AllpacaError> {
    let raw = url.trim();
    let path = if let Some(rest) = raw.strip_prefix("file://") {
        // file:///C:/... or file://localhost/C:/... or file:/C:/...
        let rest = rest
            .strip_prefix("//")
            .unwrap_or(rest)
            .strip_prefix("localhost/")
            .unwrap_or(rest);
        #[cfg(windows)]
        let rest = rest.strip_prefix('/').unwrap_or(rest);
        std::path::PathBuf::from(rest)
    } else if raw.starts_with("http://") || raw.starts_with("https://") {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "http(s) image_url must be handled by load_image_url".into(),
        )));
    } else if raw.starts_with("data:") {
        return Err(AllpacaError::Fatal(FatalError::InvalidConfig(
            "data URI must be handled by load_image_url".into(),
        )));
    } else {
        std::path::PathBuf::from(raw)
    };
    if !path.is_file() {
        return Err(AllpacaError::Fatal(FatalError::Io(format!(
            "image_url path not found: {}",
            path.display()
        ))));
    }
    Ok(path)
}

fn mime_from_path(path: &std::path::Path) -> String {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png".into(),
        Some("gif") => "image/gif".into(),
        Some("webp") => "image/webp".into(),
        Some("avif") => "image/avif".into(),
        Some("jpg") | Some("jpeg") => "image/jpeg".into(),
        _ => "application/octet-stream".into(),
    }
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
    let messages = req.messages.clone();
    let (prompt, images) = match tokio::task::spawn_blocking(move || {
        messages_to_prompt_and_images(&messages)
    })
    .await
    {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return error_response(e),
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": {"message": e.to_string(), "type": "fatal"}})),
            )
                .into_response();
        }
    };
    let params = params_from_request(&req, &state.config);
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| state.engine.model_id());
    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();

    if req.stream {
        return stream_completion(state, prompt, images, params, model, id, created).into_response();
    }

    if state.config.engine.max_active_requests > 1 {
        let mut stream = Box::pin(
            state
                .engine
                .generate_stream_multimodal(prompt.clone(), images, params),
        );
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
    match tokio::task::spawn_blocking(move || engine.generate_multimodal(&prompt, &images, params))
        .await
    {
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
    images: Vec<ImageInput>,
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
            let mut stream = Box::pin(engine.generate_stream_multimodal(prompt, images, params));
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
            let mut stream =
                Box::pin(state.engine.generate_stream_multimodal(prompt, images, params));
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

/// Prometheus text exposition (subset of Engine `/stats`).
async fn prometheus_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let body = render_prometheus_text(&state.engine.stats());
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

/// OTLP-ish JSON metrics (`allpaca_*` names). Pull endpoint; push uses `telemetry.otlp_endpoint`.
async fn otlp_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let snap = state.engine.stats();
    let body = state.engine.metrics().export_otlp_json(&snap);
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body,
    )
}

fn render_prometheus_text(s: &allpaca_types::StatsSnapshot) -> String {
    let mut out = String::with_capacity(2048);
    macro_rules! counter {
        ($name:literal, $help:literal, $val:expr) => {{
            out.push_str(concat!("# HELP ", $name, " ", $help, "\n"));
            out.push_str(concat!("# TYPE ", $name, " counter\n"));
            out.push_str(&format!("{} {}\n", $name, $val));
        }};
    }
    macro_rules! gauge {
        ($name:literal, $help:literal, $val:expr) => {{
            out.push_str(concat!("# HELP ", $name, " ", $help, "\n"));
            out.push_str(concat!("# TYPE ", $name, " gauge\n"));
            out.push_str(&format!("{} {}\n", $name, $val));
        }};
    }
    counter!(
        "allpaca_requests_total",
        "Total admitted generation requests",
        s.requests_total
    );
    gauge!(
        "allpaca_active_requests",
        "Currently active requests",
        s.active_requests
    );
    counter!(
        "allpaca_prompt_tokens_total",
        "Prompt tokens processed",
        s.prompt_tokens
    );
    counter!(
        "allpaca_tokens_generated_total",
        "Tokens generated",
        s.tokens_generated
    );
    counter!(
        "allpaca_draft_proposed_total",
        "Draft tokens proposed",
        s.draft_proposed
    );
    counter!(
        "allpaca_draft_accepted_total",
        "Draft tokens accepted",
        s.draft_accepted
    );
    gauge!(
        "allpaca_acceptance_rate",
        "Draft acceptance rate",
        s.acceptance_rate
    );
    counter!(
        "allpaca_poll_ticks_total",
        "Engine poll ticks",
        s.poll_ticks
    );
    counter!(
        "allpaca_batch_gemm_plans_total",
        "Packed cross-request GEMM plans recorded",
        s.batch_gemm_plans
    );
    counter!(
        "allpaca_batch_gemm_executes_total",
        "Soft packed GEMM co-steps that advanced plan peers",
        s.batch_gemm_executes
    );
    counter!(
        "allpaca_batch_gemm_hip_kernels_total",
        "Packed host-to-HIP GEMM launches from batch soft execute",
        s.batch_gemm_hip_kernels
    );
    counter!(
        "allpaca_micro_batch_plans_total",
        "Cross-request micro-batch stagger plans",
        s.micro_batch_plans
    );
    gauge!(
        "allpaca_vram_pages",
        "KV pages in VRAM",
        s.vram_pages
    );
    gauge!(
        "allpaca_dram_pages",
        "KV pages in pinned DRAM",
        s.dram_pages
    );
    gauge!(
        "allpaca_ttft_ms_last",
        "Last time-to-first-token in ms",
        s.ttft_ms_last
    );
    gauge!(
        "allpaca_itl_ms_ema",
        "Inter-token latency EMA in ms",
        s.itl_ms_ema
    );
    gauge!(
        "allpaca_gpu_util",
        "Target GPU busy fraction from last sample",
        s.gpu_util
    );
    counter!(
        "allpaca_vision_bytes_on_device_total",
        "Vision soft-token embedding bytes staged H2D onto HIP",
        s.vision_bytes_on_device
    );
    gauge!(
        "allpaca_vision_weight_bytes_on_device",
        "Vision encoder weight bytes resident on HIP after install",
        s.vision_weight_bytes_on_device
    );
    out
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
    async fn prometheus_metrics_exposes_allpaca_prefix() {
        let app = router(test_state());
        let res = app
            .oneshot(Request::builder().uri("/metrics").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(
            body.contains("allpaca_requests_total"),
            "missing allpaca_requests_total in {body}"
        );
        assert!(body.contains("allpaca_tokens_generated_total"));
        assert!(body.contains("allpaca_poll_ticks_total"));
        assert!(body.contains("allpaca_batch_gemm_plans_total"));
        assert!(body.contains("allpaca_batch_gemm_executes_total"));
        assert!(body.contains("allpaca_batch_gemm_hip_kernels_total"));
    }

    #[tokio::test]
    async fn otlp_metrics_exposes_allpaca_json() {
        let app = router(test_state());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/metrics/otlp")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(body.contains("resourceMetrics"));
        assert!(body.contains("allpaca_requests_total"));
        assert!(body.contains("allpaca_tokens_generated_total"));
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

    #[test]
    fn image_url_fetches_https_avif_yeelight() {
        let url = "https://spyn.dev/_astro/yeelight.B4dLTFRe_2tsYNB.avif";
        let img = load_image_url(url).expect("https fetch must succeed");
        assert!(
            img.mime == "image/avif" || img.mime.starts_with("image/"),
            "mime={}",
            img.mime
        );
        assert!(
            img.bytes.len() > 10_000,
            "expected avif body, got {} bytes",
            img.bytes.len()
        );
        // AVIF magic / ftyp brand
        assert!(
            img.bytes.windows(4).any(|w| w == b"ftyp") || img.bytes[4..8] == *b"ftyp",
            "missing ftyp brand in avif"
        );
    }

    #[test]
    fn image_url_loads_data_uri_base64() {
        // Minimal JPEG SOI marker payload, base64-encoded.
        let raw = [0xFFu8, 0xD8, 0xFF, 0xD9];
        let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, raw);
        let url = format!("data:image/jpeg;base64,{b64}");
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "look".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlSpec { url },
                },
            ]),
        }];
        let (prompt, images) = messages_to_prompt_and_images(&messages).unwrap();
        assert!(prompt.contains("look"));
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime, "image/jpeg");
        assert_eq!(images[0].bytes, raw);
    }

    #[test]
    fn image_url_loads_local_jpg_into_image_input() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/images.jpg");
        assert!(
            path.is_file(),
            "expected fixture at {}",
            path.display()
        );
        let url = path.to_string_lossy().replace('\\', "/");
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "what is this?".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrlSpec { url: url.clone() },
                },
            ]),
        }];
        let (prompt, images) = messages_to_prompt_and_images(&messages).unwrap();
        assert!(prompt.contains("what is this?"));
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].mime, "image/jpeg");
        assert!(
            images[0].bytes.len() > 1000,
            "expected jpeg payload, got {} bytes",
            images[0].bytes.len()
        );
    }

    #[tokio::test]
    async fn chat_completion_local_image_url_degraded_without_encoder() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/images.jpg");
        let url = path.to_string_lossy().replace('\\', "/");
        let app = router(test_state());
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": url}}
                ]
            }],
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
        assert_eq!(json["error"]["type"], "degraded");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("vision encoder"),
            "json={json}"
        );
    }

    #[tokio::test]
    async fn chat_completion_local_image_url_with_stub_encoder() {
        use allpaca_core::{EncoderPlacement, VisionEncoder, VisionTokens};

        struct StubEnc;
        impl EncoderPlacement for StubEnc {
            fn device(&self) -> Option<i32> {
                Some(1)
            }
        }
        impl VisionEncoder for StubEnc {
            fn encode(
                &self,
                images: &[ImageInput],
            ) -> Result<VisionTokens, AllpacaError> {
                assert!(!images.is_empty());
                Ok(VisionTokens {
                    embeddings: vec![0.1; 2 * 32],
                    n_tokens: 2,
                    device: Some(1),
                })
            }
        }

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/images.jpg");
        let url = path.to_string_lossy().replace('\\', "/");
        let state = test_state();
        state
            .engine
            .set_vision_encoder(Arc::new(StubEnc))
            .unwrap();
        let text_only = state
            .engine
            .generate(
                "user: describe\nassistant:",
                GenerationParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    seed: Some(1),
                    ..GenerationParams::default()
                },
            )
            .unwrap();
        let app = router(state);
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": url}}
                ]
            }],
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
        assert!(json.get("error").is_none(), "json={json}");
        let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
        assert_eq!(
            prompt_tokens,
            text_only.prompt_tokens as u64 + 2,
            "soft vision tokens must increase prompt_tokens"
        );
    }

    #[tokio::test]
    async fn chat_completion_https_avif_with_vision_weights() {
        let vision = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/vision_patch_dim32.safetensors");
        if !vision.is_file() {
            let enc = allpaca_core::PatchVisionEncoder::synthetic(None, 32, 42).unwrap();
            enc.write_safetensors(&vision).unwrap();
        }
        let mut cfg = Config::compiled_defaults();
        cfg.merge_cli(&CliOverrides {
            synthetic: true,
            cpu_only: true,
            seed: Some(1),
            ..CliOverrides::default()
        });
        cfg.multimodal.vision_path = Some(vision);
        let state = AppState {
            engine: Arc::new(Engine::new(cfg.clone()).unwrap()),
            config: cfg,
        };
        let app = router(state);
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what product is this?"},
                    {"type": "image_url", "image_url": {
                        "url": "https://spyn.dev/_astro/yeelight.B4dLTFRe_2tsYNB.avif"
                    }}
                ]
            }],
            "max_tokens": 8,
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
        assert!(json.get("error").is_none(), "json={json}");
        assert!(json["choices"][0]["message"]["content"].is_string());
        // text-only "what product..." is short; +16 vision patches for 64/16
        assert!(
            json["usage"]["prompt_tokens"].as_u64().unwrap() >= 16,
            "prompt_tokens should include vision soft tokens; json={json}"
        );
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

    #[tokio::test]
    async fn chat_completion_stream_with_data_uri_image() {
        use allpaca_core::{EncoderPlacement, VisionEncoder, VisionTokens};
        use base64::Engine as _;

        struct StubEnc;
        impl EncoderPlacement for StubEnc {
            fn device(&self) -> Option<i32> {
                Some(1)
            }
        }
        impl VisionEncoder for StubEnc {
            fn encode(
                &self,
                images: &[ImageInput],
            ) -> Result<VisionTokens, AllpacaError> {
                assert!(!images.is_empty());
                assert_eq!(images[0].mime, "image/jpeg");
                Ok(VisionTokens {
                    embeddings: vec![0.2; 2 * 32],
                    n_tokens: 2,
                    device: Some(1),
                })
            }
        }

        let raw = std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/images.jpg"),
        )
        .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
        let url = format!("data:image/jpeg;base64,{b64}");
        let state = test_state();
        state
            .engine
            .set_vision_encoder(Arc::new(StubEnc))
            .unwrap();
        let app = router(state);
        let body = serde_json::json!({
            "model": "synthetic",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image_url", "image_url": {"url": url}}
                ]
            }],
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
            !text.contains("streaming multimodal"),
            "stream multimodal must not be rejected: {text}"
        );
        assert!(
            text.contains("chat.completion.chunk"),
            "stream must emit OpenAI chunks: {text}"
        );
        assert!(text.contains("[DONE]"), "stream must terminate with [DONE]: {text}");
        assert!(
            !text.contains("\"error\""),
            "stub encoder stream must succeed: {text}"
        );
    }
}
