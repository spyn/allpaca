use std::fmt;

/// Error class used by the control plane. Recoverable failures never panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Fatal,
    Degraded,
    Transient,
}

#[derive(Debug, thiserror::Error)]
pub enum AllpacaError {
    #[error(transparent)]
    Fatal(#[from] FatalError),
    #[error(transparent)]
    Degraded(#[from] DegradedError),
    #[error(transparent)]
    Transient(#[from] TransientError),
}

impl AllpacaError {
    pub fn class(&self) -> ErrorClass {
        match self {
            Self::Fatal(_) => ErrorClass::Fatal,
            Self::Degraded(_) => ErrorClass::Degraded,
            Self::Transient(_) => ErrorClass::Transient,
        }
    }

    pub fn is_fatal(&self) -> bool {
        matches!(self, Self::Fatal(_))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FatalError {
    #[error("model load failed: {0}")]
    ModelLoad(String),
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("corrupt GGUF: {0}")]
    CorruptGguf(String),
    #[error("invalid UTF-8")]
    InvalidUtf8,
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("no model loaded")]
    NoModel,
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for FatalError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

impl From<std::io::Error> for AllpacaError {
    fn from(value: std::io::Error) -> Self {
        Self::Fatal(FatalError::from(value))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DegradedError {
    #[error("draft GPU unavailable; speculation disabled: {0}")]
    DraftDisabled(String),
    #[error("P2P unavailable or slower; using host staging: {0}")]
    HostStagingFallback(String),
    #[error("JSON constraint produced an empty mask; allowing all tokens")]
    EmptyConstraintMask,
    #[error("feature disabled: {0}")]
    FeatureDisabled(String),
}

#[derive(Debug, thiserror::Error)]
pub enum TransientError {
    #[error("another request is already active")]
    RequestBusy,
    #[error("page migration contention: {0}")]
    PageMigration(String),
    #[error("kernel timeout: {0}")]
    KernelTimeout(String),
    #[error("backpressure: {0}")]
    Backpressure(String),
}

impl fmt::Display for ErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fatal => write!(f, "fatal"),
            Self::Degraded => write!(f, "degraded"),
            Self::Transient => write!(f, "transient"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_matches_variant() {
        let fatal = AllpacaError::from(FatalError::InvalidUtf8);
        assert_eq!(fatal.class(), ErrorClass::Fatal);
        let degraded = AllpacaError::from(DegradedError::EmptyConstraintMask);
        assert_eq!(degraded.class(), ErrorClass::Degraded);
        let transient = AllpacaError::from(TransientError::RequestBusy);
        assert_eq!(transient.class(), ErrorClass::Transient);
    }
}
