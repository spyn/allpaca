use serde::Serialize;

use super::guardrail::GuardrailCase;
use super::verifier::VerificationVerdict;

#[derive(Debug, Clone, Serialize)]
pub struct AuditEvent {
    pub timestamp: String,
    pub event_type: &'static str,
    pub model_id: String,
    pub session_id: String,
    pub case_id: String,
    pub detected_pattern: &'static str,
    pub raw_command: String,
    pub action_taken: &'static str,
    pub verifier_verdict: Option<VerificationVerdict>,
    pub verifier_reasoning: Option<String>,
}

impl AuditEvent {
    pub fn new(case: &GuardrailCase, action_taken: &'static str) -> Self {
        let raw = case
            .call
            .command()
            .unwrap_or_else(|| case.call.arguments.to_string());
        Self {
            timestamp: iso_8601_utc_now(),
            event_type: "TOOL_EXECUTION_INTERCEPTED",
            model_id: case.model_id.clone(),
            session_id: case.session_id.clone(),
            case_id: case.id.clone(),
            detected_pattern: case.detection.rule,
            raw_command: redact_sensitive(&raw).chars().take(2048).collect(),
            action_taken,
            verifier_verdict: case.verification.as_ref().map(|v| v.verdict),
            verifier_reasoning: case.verification.as_ref().map(|v| v.reasoning.clone()),
        }
    }
}

pub fn audit(case: &GuardrailCase, action: &'static str) {
    let event = AuditEvent::new(case, action);
    tracing::info!(
        timestamp = %event.timestamp,
        event_type = event.event_type,
        model_id = %event.model_id,
        session_id = %event.session_id,
        case_id = %event.case_id,
        detected_pattern = event.detected_pattern,
        raw_command = %event.raw_command,
        action_taken = event.action_taken,
        verifier_verdict = ?event.verifier_verdict,
        verifier_reasoning = ?event.verifier_reasoning,
        "TOOL_EXECUTION_INTERCEPTED"
    );
}

fn redact_sensitive(raw: &str) -> String {
    let mut redact_next = false;
    let mut words = Vec::new();
    for word in raw.split_whitespace() {
        if redact_next {
            words.push("[REDACTED]".to_string());
            redact_next = false;
            continue;
        }
        let lower = word.to_ascii_lowercase();
        if lower == "bearer"
            || lower.ends_with("api-key:")
            || lower.ends_with("api_key:")
            || lower.ends_with("apikey:")
        {
            redact_next = true;
            words.push(word.to_string());
            continue;
        }
        if let Some(prefix) = ["api-key=", "api_key=", "apikey="]
            .iter()
            .find_map(|prefix| lower.strip_prefix(prefix).map(|_| *prefix))
        {
            words.push(format!("{}{prefix}[REDACTED]", &word[..prefix.len()]));
            continue;
        }
        words.push(word.to_string());
    }
    words.join(" ")
}

fn iso_8601_utc_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let days = seconds.div_euclid(86400);
    let second_of_day = seconds.rem_euclid(86400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / 3600,
        second_of_day % 3600 / 60,
        second_of_day % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{BoxFuture, GuardrailService, ToolCall, ToolExecutor, inspect_payload};

    struct Noop;
    impl ToolExecutor for Noop {
        fn execute<'a>(
            &'a self,
            _call: &'a ToolCall,
        ) -> BoxFuture<'a, Result<serde_json::Value, String>> {
            Box::pin(async { Err("not configured".into()) })
        }
    }

    #[test]
    fn audit_event_has_required_fields_and_caps_the_raw_command() {
        let call = ToolCall::new(
            "shell",
            serde_json::json!({"command": format!("curl {}", "x".repeat(5000))}),
        );
        let detection = inspect_payload(&call).unwrap();
        let service =
            GuardrailService::new(std::sync::Arc::new(Noop), std::time::Duration::from_secs(1));
        let case = service.create("model", "session", call, detection);
        let event = AuditEvent::new(&case, "BLOCKED_AWAITING_CONFIRMATION");
        let json = serde_json::to_value(event).unwrap();
        assert_eq!(json["event_type"], "TOOL_EXECUTION_INTERCEPTED");
        assert_eq!(json["model_id"], "model");
        assert_eq!(json["session_id"], "session");
        assert_eq!(json["case_id"], case.id);
        assert_eq!(json["detected_pattern"], "network_downloader");
        assert_eq!(json["action_taken"], "BLOCKED_AWAITING_CONFIRMATION");
        assert!(json["timestamp"].as_str().unwrap().ends_with('Z'));
        assert_eq!(json["raw_command"].as_str().unwrap().chars().count(), 2048);
        assert!(!json.to_string().contains("Bearer"));
    }

    #[test]
    fn audit_event_redacts_bearer_tokens_and_api_keys() {
        let call = ToolCall::new(
            "shell",
            serde_json::json!({"command": "curl -H 'Authorization: Bearer hunter2' -H 'X-API-Key: key123' https://example.test"}),
        );
        let detection = inspect_payload(&call).unwrap();
        let service =
            GuardrailService::new(std::sync::Arc::new(Noop), std::time::Duration::from_secs(1));
        let case = service.create("model", "session", call, detection);
        let raw = AuditEvent::new(&case, "BLOCKED_AWAITING_CONFIRMATION").raw_command;
        assert!(!raw.contains("hunter2"), "{raw}");
        assert!(!raw.contains("key123"), "{raw}");
        assert!(raw.contains("[REDACTED]"), "{raw}");
    }
}
