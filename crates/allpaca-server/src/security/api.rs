use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use super::{GuardrailError, GuardrailStatus, ToolCall, inspect_payload};
use crate::api::AppState;

#[derive(Debug, Deserialize)]
pub struct ToolCallRequest {
    pub model: Option<String>,
    pub session_id: Option<String>,
    pub user_prompt: String,
    pub call: ToolCall,
}

#[derive(Debug, Deserialize)]
pub struct ConfirmationRequest {
    pub allow: bool,
}

pub async fn create_tool_call(
    State(state): State<AppState>,
    Json(request): Json<ToolCallRequest>,
) -> Response {
    let model = request
        .model
        .unwrap_or_else(|| state.config.model.id.clone());
    let session = request.session_id.unwrap_or_else(|| "anonymous".into());
    let detection = inspect_payload(&request.call);
    match detection {
        Some(detection) if state.config.security.enable_guardrails => {
            let case = state
                .security
                .create(&model, &session, request.call, detection);
            let verifier_state = state.clone();
            let pending = case.clone();
            tokio::spawn(async move {
                let verification = verifier_state
                    .intent_verifier
                    .verify(&request.user_prompt, &pending.call)
                    .await;
                let _ = verifier_state
                    .security
                    .record_verification(&pending.id, verification);
            });
            (StatusCode::ACCEPTED, Json(case)).into_response()
        }
        Some(detection) => match state
            .security
            .execute_bypassed(&model, &session, request.call, detection)
            .await
        {
            Ok(case) => (StatusCode::OK, Json(case)).into_response(),
            Err(error) => guardrail_error(error),
        },
        None => match state.security.execute(&request.call).await {
            Ok(result) => (
                StatusCode::OK,
                Json(serde_json::json!({
                    "status": "allowed",
                    "explanation": "No guarded command pattern detected.",
                    "execution_result": result,
                })),
            )
                .into_response(),
            Err(error) => guardrail_error(error),
        },
    }
}

pub async fn get_guardrail_case(State(state): State<AppState>, Path(id): Path<String>) -> Response {
    match state.security.get(&id) {
        Some(case) => (StatusCode::OK, Json(case)).into_response(),
        None => guardrail_error(GuardrailError::NotFound),
    }
}

pub async fn confirm_guardrail_case(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<ConfirmationRequest>,
) -> Response {
    match state.security.confirm(&id, request.allow).await {
        Ok(case) if case.status == GuardrailStatus::Denied => (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({
                "error": {"message": case.explanation, "type": "security_rejected"},
                "case": case,
            })),
        )
            .into_response(),
        Ok(case) => (StatusCode::OK, Json(case)).into_response(),
        Err(error) => guardrail_error(error),
    }
}

fn guardrail_error(error: GuardrailError) -> Response {
    let status = match error {
        GuardrailError::NotFound => StatusCode::NOT_FOUND,
        GuardrailError::Conflict(_) => StatusCode::CONFLICT,
        GuardrailError::Expired => StatusCode::GONE,
        GuardrailError::Execution(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        Json(
            serde_json::json!({"error": {"message": error.to_string(), "type": "guardrail_error"}}),
        ),
    )
        .into_response()
}
