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
pub(crate) fn classify_sync_error(
    code: ffi::cudaError_t,
    context: &str,
) -> infer_core::ports::OpError {
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

/// Consume an allocation API's reported failure without leaving a recoverable
/// OOM in CUDA's last-error slot. A different pending device fault takes
/// precedence, so an OOM retry never hides an asynchronous kernel failure.
pub(crate) fn allocation_error(
    code: ffi::cudaError_t,
    context: &str,
) -> infer_core::ports::OpError {
    debug_assert_ne!(code, ffi::cudaError_cudaSuccess);
    let pending = unsafe { ffi::cudaGetLastError() };
    classify_allocation_error(code, pending, context)
}

fn classify_allocation_error(
    code: ffi::cudaError_t,
    pending: ffi::cudaError_t,
    context: &str,
) -> infer_core::ports::OpError {
    if pending != ffi::cudaError_cudaSuccess && pending != code {
        let error = classify_sync_error(pending, context);
        if error.is_fatal() {
            return error;
        }
    }
    match code {
        ffi::cudaError_cudaErrorInvalidValue | ffi::cudaError_cudaErrorNotSupported => {
            infer_core::ports::OpError::Kernel(format!("{context}: {}", CudaError(code)))
        }
        _ => classify_sync_error(code, context),
    }
}

pub(crate) fn classify_capture_error(
    code: ffi::cudaError_t,
    context: &str,
) -> infer_core::ports::OpError {
    match code {
        ffi::cudaError_cudaErrorInvalidValue
        | ffi::cudaError_cudaErrorNotSupported
        | ffi::cudaError_cudaErrorStreamCaptureUnsupported
        | ffi::cudaError_cudaErrorStreamCaptureInvalidated => {
            infer_core::ports::OpError::Kernel(format!("{context}: {}", CudaError(code)))
        }
        _ => classify_sync_error(code, context),
    }
}

/// EndCapture reports invalidation even when it has successfully restored the
/// stream. Consume only the known recoverable capture errors so a subsequent
/// launch does not mistake them for an asynchronous device fault. Any other
/// pending error keeps the normal classification, including fatal GPU faults.
pub(crate) fn check_capture_cleanup_error(
    reported_error: Option<ffi::cudaError_t>,
) -> infer_core::ports::OpResult<()> {
    let code = unsafe { ffi::cudaGetLastError() };
    // The caller is already returning this synchronous API failure. Consume
    // only known non-fatal codes; a matching device fault must still be fatal.
    if reported_error == Some(code)
        && matches!(
            code,
            ffi::cudaError_cudaErrorMemoryAllocation
                | ffi::cudaError_cudaErrorInvalidValue
                | ffi::cudaError_cudaErrorNotSupported
        )
    {
        return Ok(());
    }
    match code {
        ffi::cudaError_cudaSuccess
        | ffi::cudaError_cudaErrorStreamCaptureUnsupported
        | ffi::cudaError_cudaErrorStreamCaptureInvalidated => Ok(()),
        _ => Err(classify_sync_error(code, "CUDA graph capture cleanup")),
    }
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

#[cfg(test)]
mod allocation_tests {
    use super::*;

    #[test]
    fn pool_oom_never_hides_a_pending_device_fault() {
        let oom = ffi::cudaError_cudaErrorMemoryAllocation;
        let fault = ffi::cudaError_cudaErrorIllegalAddress;
        let invalid = ffi::cudaError_cudaErrorInvalidValue;
        let success = ffi::cudaError_cudaSuccess;
        for (reported, pending, fatal) in [
            (oom, success, false),
            (oom, oom, false),
            (oom, fault, true),
            (fault, oom, true),
            (fault, fault, true),
            (invalid, invalid, false),
            (invalid, fault, true),
        ] {
            let error = classify_allocation_error(reported, pending, "pool test");
            assert_eq!(error.is_fatal(), fatal, "{reported}/{pending}: {error}");
        }
    }
}
