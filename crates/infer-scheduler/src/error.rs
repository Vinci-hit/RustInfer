//! Scheduler error types.

use std::time::Duration;

/// Top-level scheduler error.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("cache exhausted: needed {needed} blocks, {available} available")]
    CacheExhausted { needed: usize, available: usize },

    #[error("sequence too long: {length} exceeds max_model_len {max}")]
    SequenceTooLong { length: usize, max: usize },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("worker error: {0}")]
    WorkerError(String),

    #[error("preemption failed: {0}")]
    PreemptionFailed(String),

    #[error("codec error: {0}")]
    Codec(String),

    #[error("internal: {0}")]
    Internal(String),

    #[error("not implemented: {0}")]
    NotImplemented(String),

    #[error("shutdown")]
    Shutdown,
}

/// Transport-layer errors.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    #[error("send failed: {0}")]
    SendFailed(String),

    #[error("receive failed: {0}")]
    ReceiveFailed(String),

    #[error("timeout after {0:?}")]
    Timeout(Duration),

    #[error("serialization: {0}")]
    Serialization(String),
}

/// Convenience type alias.
pub type Result<T> = std::result::Result<T, SchedulerError>;
