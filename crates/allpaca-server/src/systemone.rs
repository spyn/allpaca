//! `POST /v1/systemone`: typed questions about a `state`, answered by a
//! decision model in one forward pass each. Same request and response as
//! llama.cpp's endpoint of that name.

use crate::api::{engine_unavailable, AppState};
use allpaca_core::decision::{DecisionError, Json as OrderedJson};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use allpaca_types::{AllpacaError, TransientError};
use axum::Json;

/// How long a question waits for a free request slot before the request
/// fails as busy.
const SLOT_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

fn error(status: StatusCode, kind: &str, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": kind}})),
    )
        .into_response()
}

pub(crate) async fn systemone(State(state): State<AppState>, body: String) -> Response {
    let Some(engine) = state.engine().filter(|e| e.has_model()) else {
        return engine_unavailable(&state);
    };
    let Some(model) = engine.decision() else {
        return error(
            StatusCode::NOT_IMPLEMENTED,
            "not_supported",
            "the loaded model is not a decision model".into(),
        );
    };
    let request = match model.parse(&body) {
        Ok(request) => request,
        Err(DecisionError::Invalid(message)) => {
            return error(StatusCode::BAD_REQUEST, "invalid_request", message)
        }
        Err(DecisionError::Unsupported(message)) => {
            return error(StatusCode::NOT_IMPLEMENTED, "not_supported", message)
        }
    };
    let model_id = engine.model_id();
    tracing::info!(
        "systemone model={model_id} questions={}",
        request.questions.len()
    );
    // Questions are independent: one forward pass each, in request order.
    let answered = tokio::task::spawn_blocking(move || {
        let mut answers = Vec::with_capacity(request.questions.len());
        let mut input_tokens = 0usize;
        for question in &request.questions {
            let prompt = model.render(&request.state, question);
            let read = model.read(question);
            // A question is one short forward pass: wait for a free request
            // slot instead of answering "busy".
            let waiting = std::time::Instant::now();
            let (scores, tokens) = loop {
                match engine.decision_scores(&prompt, &read) {
                    Err(AllpacaError::Transient(TransientError::RequestBusy))
                        if waiting.elapsed() < SLOT_WAIT =>
                    {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    other => break other?,
                }
            };
            input_tokens += tokens;
            answers.push((question.id.clone(), model.answer(question, &scores)?));
        }
        Ok::<_, AllpacaError>((answers, input_tokens))
    })
    .await;
    let (answers, input_tokens) = match answered {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return crate::api::error_response(e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, "fatal", e.to_string()),
    };
    let body = OrderedJson::Obj(vec![
        ("model".into(), OrderedJson::Str(model_id)),
        ("answers".into(), OrderedJson::Obj(answers)),
        (
            "usage".into(),
            OrderedJson::Obj(vec![
                ("input_tokens".into(), OrderedJson::Num(input_tokens.into())),
                ("output_tokens".into(), OrderedJson::Num(0u64.into())),
            ]),
        ),
    ]);
    Json(body).into_response()
}
