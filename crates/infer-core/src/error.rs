use crate::types::Shape;

/// Operator-level error.
#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("not contiguous: shape={0:?}")]
    NotContiguous(Shape),
    #[error("unsupported op '{op}' on backend '{backend}'")]
    Unsupported {
        backend: &'static str,
        op: &'static str,
    },
    #[error("kernel failed: {0}")]
    Kernel(String),
    /// The device/context is poisoned and cannot be reused (e.g. an illegal
    /// memory access or launch failure observed at a sync point). The worker
    /// must abort the process rather than retry the affected sequences — every
    /// later CUDA call would re-observe the same sticky error.
    #[error("fatal device error: {0}")]
    Fatal(String),
    #[error("shutdown requested")]
    Shutdown,
}

impl OpError {
    pub fn unsupported(backend: &'static str, op: &'static str) -> Self {
        Self::Unsupported { backend, op }
    }

    /// True for errors that poison the device/context and require the worker
    /// process to exit, as opposed to per-sequence recoverable failures.
    pub fn is_fatal(&self) -> bool {
        matches!(self, OpError::Fatal(_))
    }
}

pub type OpResult<T> = std::result::Result<T, OpError>;
