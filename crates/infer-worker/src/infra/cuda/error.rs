//! CUDA error type + cuda_check! macro.
use std::ffi::CStr;
use super::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub ffi::cudaError_t);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = unsafe { ffi::cudaGetErrorName(self.0) };
        let desc = unsafe { ffi::cudaGetErrorString(self.0) };
        if name.is_null() || desc.is_null() {
            return write!(f, "CUDA error code {:?}", self.0);
        }
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        let desc = unsafe { CStr::from_ptr(desc) }.to_string_lossy();
        write!(f, "CUDA Error ({}): {}", name, desc)
    }
}
impl std::error::Error for CudaError {}

/// Check a CUDA FFI call. Returns `Err(OpError::Kernel(...))` on failure.
macro_rules! cuda_check {
    ($expr:expr) => {{
        let code = $expr;
        if code != crate::infra::cuda::ffi::cudaError_cudaSuccess {
            return Err(crate::domain::ports::OpError::Kernel(
                format!("{}", crate::infra::cuda::CudaError(code))
            ));
        }
    }};
}
pub(crate) use cuda_check;
