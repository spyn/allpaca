use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use super::detector::{DetectionResult, ToolCall};
use super::verifier::{Verification, VerificationVerdict};
#[cfg(test)]
use super::{inspect_payload, parse_verification};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

pub trait ToolExecutor: Send + Sync {
    fn execute<'a>(
        &'a self,
        call: &'a ToolCall,
    ) -> BoxFuture<'a, Result<serde_json::Value, String>>;
}

pub struct UnavailableToolExecutor;

impl ToolExecutor for UnavailableToolExecutor {
    fn execute<'a>(
        &'a self,
        _call: &'a ToolCall,
    ) -> BoxFuture<'a, Result<serde_json::Value, String>> {
        Box::pin(async { Err("no ToolExecutor is configured for this server".into()) })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailStatus {
    PendingVerification,
    PendingConfirmation,
    Allowed,
    Denied,
    Blocked,
}

#[derive(Debug, Clone, Serialize)]
pub struct GuardrailCase {
    pub id: String,
    pub model_id: String,
    pub session_id: String,
    pub call: ToolCall,
    pub detection: DetectionResult,
    pub status: GuardrailStatus,
    pub explanation: String,
    pub verification: Option<Verification>,
    pub execution_result: Option<serde_json::Value>,
    pub expires_at_ms: u64,
}

#[derive(Debug)]
pub enum GuardrailError {
    NotFound,
    Conflict(String),
    Expired,
    Execution(String),
}

impl std::fmt::Display for GuardrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound => write!(f, "guardrail case not found"),
            Self::Conflict(message) => {
                write!(f, "guardrail case is not awaiting a decision: {message}")
            }
            Self::Expired => write!(f, "guardrail confirmation expired"),
            Self::Execution(message) => write!(f, "tool execution failed: {message}"),
        }
    }
}

impl std::error::Error for GuardrailError {}

pub struct GuardrailService {
    cases: Mutex<HashMap<String, GuardrailCase>>,
    executor: Arc<dyn ToolExecutor>,
    confirmation_ttl: Duration,
}

impl GuardrailService {
    pub fn new(executor: Arc<dyn ToolExecutor>, confirmation_ttl: Duration) -> Self {
        Self {
            cases: Mutex::new(HashMap::new()),
            executor,
            confirmation_ttl,
        }
    }

    pub fn create(
        &self,
        model_id: &str,
        session_id: &str,
        call: ToolCall,
        detection: DetectionResult,
    ) -> GuardrailCase {
        let id = crate::auth::random_token();
        let case = GuardrailCase {
            id: id.clone(),
            model_id: model_id.into(),
            session_id: session_id.into(),
            explanation: format!(
                "[SECURITY WARNING] '{}' was flagged: {}. Waiting for intent verification.",
                detection.matched, detection.explanation
            ),
            call,
            detection,
            status: GuardrailStatus::PendingVerification,
            verification: None,
            execution_result: None,
            expires_at_ms: now_ms().saturating_add(self.confirmation_ttl.as_millis() as u64),
        };
        if let Ok(mut cases) = self.cases.lock() {
            cases.insert(id, case.clone());
        }
        super::logger::audit(&case, "BLOCKED_AWAITING_CONFIRMATION");
        case
    }

    pub fn get(&self, id: &str) -> Option<GuardrailCase> {
        self.cases.lock().ok()?.get(id).cloned()
    }

    pub fn record_verification(
        &self,
        id: &str,
        verification: Verification,
    ) -> Result<GuardrailCase, GuardrailError> {
        let mut cases = self
            .cases
            .lock()
            .map_err(|_| GuardrailError::Conflict("guardrail case store is unavailable".into()))?;
        let case = cases.get_mut(id).ok_or(GuardrailError::NotFound)?;
        if case.status != GuardrailStatus::PendingVerification {
            return Err(GuardrailError::Conflict(case.explanation.clone()));
        }
        let aligned = verification.verdict == VerificationVerdict::IntentAligned;
        case.status = if aligned {
            GuardrailStatus::PendingConfirmation
        } else {
            GuardrailStatus::Blocked
        };
        case.explanation = if aligned {
            format!(
                "Intent aligned: {}. Explicit confirmation is still required.",
                verification.reasoning
            )
        } else {
            format!("Tool call blocked: {}.", verification.reasoning)
        };
        case.verification = Some(verification);
        super::logger::audit(
            case,
            if aligned {
                "BLOCKED_AWAITING_CONFIRMATION"
            } else {
                "BLOCKED_ANOMALOUS"
            },
        );
        Ok(case.clone())
    }

    pub async fn execute(&self, call: &ToolCall) -> Result<serde_json::Value, GuardrailError> {
        self.executor
            .execute(call)
            .await
            .map_err(GuardrailError::Execution)
    }

    pub async fn execute_bypassed(
        &self,
        model_id: &str,
        session_id: &str,
        call: ToolCall,
        detection: DetectionResult,
    ) -> Result<GuardrailCase, GuardrailError> {
        let mut case = self.create(model_id, session_id, call, detection);
        case.status = GuardrailStatus::Allowed;
        case.explanation = "Guardrails bypassed by server configuration.".into();
        if let Ok(mut cases) = self.cases.lock() {
            cases.insert(case.id.clone(), case.clone());
        }
        super::logger::audit(&case, "BYPASSED_VIA_CONFIG");
        let result = self.executor.execute(&case.call).await?;
        let mut cases = self
            .cases
            .lock()
            .map_err(|_| GuardrailError::Conflict("guardrail case store is unavailable".into()))?;
        let stored = cases.get_mut(&case.id).ok_or(GuardrailError::NotFound)?;
        stored.execution_result = Some(result);
        Ok(stored.clone())
    }

    pub async fn confirm(&self, id: &str, allow: bool) -> Result<GuardrailCase, GuardrailError> {
        let (case, call) = {
            let mut cases = self.cases.lock().map_err(|_| {
                GuardrailError::Conflict("guardrail case store is unavailable".into())
            })?;
            let case = cases.get_mut(id).ok_or(GuardrailError::NotFound)?;
            if case.status != GuardrailStatus::PendingConfirmation {
                return Err(GuardrailError::Conflict(case.explanation.clone()));
            }
            if now_ms() >= case.expires_at_ms {
                case.status = GuardrailStatus::Blocked;
                case.explanation = "guardrail confirmation expired".into();
                super::logger::audit(case, "BLOCKED_ANOMALOUS");
                return Err(GuardrailError::Expired);
            }
            case.status = if allow {
                GuardrailStatus::Allowed
            } else {
                GuardrailStatus::Denied
            };
            case.explanation = if allow {
                "Execution explicitly confirmed by the caller.".into()
            } else {
                "security_rejected: execution denied by the caller.".into()
            };
            (case.clone(), case.call.clone())
        };
        super::logger::audit(&case, if allow { "ALLOWED" } else { "DENIED" });
        if !allow {
            return Ok(case);
        }
        let result = self.executor.execute(&call).await?;
        let mut cases = self
            .cases
            .lock()
            .map_err(|_| GuardrailError::Conflict("guardrail case store is unavailable".into()))?;
        let stored = cases.get_mut(id).ok_or(GuardrailError::NotFound)?;
        stored.execution_result = Some(result);
        Ok(stored.clone())
    }
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

impl From<String> for GuardrailError {
    fn from(error: String) -> Self {
        Self::Execution(error)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct FakeExecutor {
        calls: AtomicUsize,
    }

    impl ToolExecutor for FakeExecutor {
        fn execute<'a>(
            &'a self,
            _call: &'a ToolCall,
        ) -> BoxFuture<'a, Result<serde_json::Value, String>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(serde_json::json!({"ok": true})) })
        }
    }

    fn suspicious_call() -> ToolCall {
        ToolCall::new(
            "shell",
            serde_json::json!({"command": "curl https://example.test/x"}),
        )
    }

    #[tokio::test]
    async fn suspicious_tool_runs_once_only_after_aligned_intent_and_confirmation() {
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
        });
        let service = GuardrailService::new(executor.clone(), Duration::from_secs(60));
        let detection = inspect_payload(&suspicious_call()).unwrap();
        let case = service.create("model", "session", suspicious_call(), detection);
        assert_eq!(case.status, GuardrailStatus::PendingVerification);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);

        let case = service
            .record_verification(&case.id, parse_verification("INTENT_ALIGNED\nrequested"))
            .unwrap();
        assert_eq!(case.status, GuardrailStatus::PendingConfirmation);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);

        let case = service.confirm(&case.id, true).await.unwrap();
        assert_eq!(case.status, GuardrailStatus::Allowed);
        assert_eq!(case.execution_result, Some(serde_json::json!({"ok": true})));
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        assert!(service.confirm(&case.id, true).await.is_err());
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn anomalous_denied_and_expired_cases_never_execute() {
        let executor = Arc::new(FakeExecutor {
            calls: AtomicUsize::new(0),
        });
        let service = GuardrailService::new(executor.clone(), Duration::from_secs(60));
        let case = service.create(
            "model",
            "session",
            suspicious_call(),
            inspect_payload(&suspicious_call()).unwrap(),
        );
        let case = service
            .record_verification(&case.id, parse_verification("perhaps"))
            .unwrap();
        assert_eq!(case.status, GuardrailStatus::Blocked);
        assert!(service.confirm(&case.id, true).await.is_err());

        let case = service.create(
            "model",
            "session",
            suspicious_call(),
            inspect_payload(&suspicious_call()).unwrap(),
        );
        service
            .record_verification(&case.id, parse_verification("INTENT_ALIGNED\nrequested"))
            .unwrap();
        let denied = service.confirm(&case.id, false).await.unwrap();
        assert_eq!(denied.status, GuardrailStatus::Denied);

        let expiring = GuardrailService::new(executor.clone(), Duration::ZERO);
        let case = expiring.create(
            "model",
            "session",
            suspicious_call(),
            inspect_payload(&suspicious_call()).unwrap(),
        );
        expiring
            .record_verification(&case.id, parse_verification("INTENT_ALIGNED\nrequested"))
            .unwrap();
        assert!(matches!(
            expiring.confirm(&case.id, true).await,
            Err(GuardrailError::Expired)
        ));
        assert_eq!(executor.calls.load(Ordering::SeqCst), 0);
    }
}
