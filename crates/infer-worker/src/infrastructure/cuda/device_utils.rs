//! CUDA device utility functions.
use super::ffi;
use crate::domain::ports::OpError;

pub fn current_device() -> Result<i32, OpError> {
    let mut id: i32 = -1;
    unsafe {
        let code = ffi::cudaGetDevice(&mut id);
        if code != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!("cudaGetDevice failed: {:?}", code)));
        }
    }
    Ok(id)
}

pub fn set_current_device(device_id: i32) -> Result<(), OpError> {
    unsafe {
        let code = ffi::cudaSetDevice(device_id);
        if code != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!(
                "cudaSetDevice({}) failed: {:?}",
                device_id, code
            )));
        }
    }
    Ok(())
}

/// Query free and total bytes of global memory on the current device.
/// Returns `(free_bytes, total_bytes)`.
pub fn mem_get_info() -> Result<(usize, usize), OpError> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let code = ffi::cudaMemGetInfo(&mut free, &mut total);
        if code != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!(
                "cudaMemGetInfo failed: {:?}",
                code
            )));
        }
    }
    Ok((free, total))
}
