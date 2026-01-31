use std::alloc::LayoutError;
pub use anyhow::Result;
// 使用 thiserror 库可以非常方便地定义错误类型
use thiserror::Error;
#[cfg(feature = "cuda")]
use crate::cuda;
use super::DeviceType;
#[derive(Error, Debug)]
pub enum Error {
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Device mismatch: expected {expected:?}, got {actual:?}")]
    DeviceMismatch {
        expected: DeviceType,
        actual: DeviceType,
        in_method: String,
    },

    #[cfg(feature = "cuda")]
    #[error("CUDA FFI call failed: {0}")]
    CudaError(#[from] cuda::error::CudaError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Layout error:{0}")]
    LayoutError(#[from] LayoutError),
    
    #[error("Index out of bounds:{0}")]
    IndexOutOfBounds(String),
    #[error("Internal error: {0}")]
    InternalError(String),

    #[error("Serialization/Deserialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Unimplemented feature: {0}")]
    Unimplemented(String),
}

// pub type Result<T> = std::result::Result<T, Error>;