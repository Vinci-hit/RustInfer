use super::ffi;
use std::ffi::CStr;
use thiserror::Error;

// 定义一个 CudaError 枚举，包装了 cudaError_t
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaError(pub ffi::cudaError_t);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_name_ptr = unsafe { ffi::cudaGetErrorName(self.0) };
        let err_str_ptr = unsafe { ffi::cudaGetErrorString(self.0) };

        if err_name_ptr.is_null() || err_str_ptr.is_null() {
            return write!(f, "Unknown CUDA Error Code: {}", self.0);
        }

        let err_name = unsafe { CStr::from_ptr(err_name_ptr) }.to_string_lossy();
        let err_str = unsafe { CStr::from_ptr(err_str_ptr) }.to_string_lossy();
        
        write!(f, "CUDA Error ({}): {}", err_name, err_str)
    }
}

/// 一个帮助宏，用于检查 FFI 调用的返回值
/// 如果不是 cudaSuccess，就返回一个 CudaError
#[macro_export]
macro_rules! cuda_check {
    ($expr:expr) => {
        {
            let result = $expr;
            if result != $crate::cuda::ffi::cudaError_cudaSuccess {
                Err($crate::base::error::Error::CudaError(crate::cuda::error::CudaError(result)))
            } else {
                // 成功时返回 Ok(())
                Ok(())
            }
        }
    };
}