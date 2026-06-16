use crate::domain::types::Shape;

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
    #[error("shutdown requested")]
    Shutdown,
}

impl OpError {
    pub fn unsupported(backend: &'static str, op: &'static str) -> Self {
        Self::Unsupported { backend, op }
    }
}

pub type OpResult<T> = std::result::Result<T, OpError>;
