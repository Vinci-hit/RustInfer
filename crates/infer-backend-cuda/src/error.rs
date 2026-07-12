//! CUDA error type + cuda_check! macro.
use super::ffi;
use std::ffi::CStr;

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

/// Classify a CUDA error observed at a **synchronization point** (i.e. a prior
/// async kernel faulted). For every code except a bare allocation failure this
/// poisons the context — all subsequent CUDA calls re-observe the same sticky
/// error — so it is reported as [`OpError::Fatal`] to drive a clean worker exit
/// instead of a per-sequence retry that can never succeed. A plain
/// out-of-memory is recoverable (the context survives) and stays non-fatal.
fn classify_sync_error(code: ffi::cudaError_t, context: &str) -> infer_core::ports::OpError {
    let msg = format!("{}: {}", context, CudaError(code));
    if code == ffi::cudaError_cudaErrorMemoryAllocation {
        infer_core::ports::OpError::Kernel(msg)
    } else {
        infer_core::ports::OpError::Fatal(msg)
    }
}

pub fn check_last_error(context: &str) -> infer_core::ports::OpResult<()> {
    let code = unsafe { ffi::cudaGetLastError() };
    if code != ffi::cudaError_cudaSuccess {
        return Err(classify_sync_error(code, context));
    }
    Ok(())
}

/// Check a CUDA FFI call. Returns `Err(OpError::Kernel(...))` on failure.
macro_rules! cuda_check {
    ($expr:expr) => {{
        let code = $expr;
        if code != crate::ffi::cudaError_cudaSuccess {
            return Err(infer_core::ports::OpError::Kernel(format!(
                "{}",
                crate::CudaError(code)
            )));
        }
    }};
}
pub(crate) use cuda_check;
