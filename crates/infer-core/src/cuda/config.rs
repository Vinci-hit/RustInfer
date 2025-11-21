use crate::base::error::Result;

use super::ffi;

/// CudaConfig 包含了执行 CUDA 内核所需的上下文信息。
/// 例如 CUDA stream, cuBLAS handle, cuDNN handle 等。
/// 这个结构体的生命周期由上层（如计算图执行器）管理。
#[derive(Debug)]
pub struct CudaConfig {
    /// CUDA stream for asynchronous execution.
    pub stream: ffi::cudaStream_t,
    
    // pub cublas_handle: cublasHandle_t, // (未来)
    // pub cudnn_handle: cudnnHandle_t,   // (未来)
}

impl CudaConfig {
    /// 创建一个新的 CudaConfig，通常在初始化计算图时调用。
    pub fn new() -> Result<Self> {
        let mut stream: ffi::cudaStream_t = std::ptr::null_mut();
        
        // 创建一个新的 CUDA stream
        unsafe { crate::cuda_check!(ffi::cudaStreamCreate(&mut stream))? };

        Ok(Self { stream })
    }
}

// 实现 Drop trait 来自动销毁 CUDA 资源 (RAII)
impl Drop for CudaConfig {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                // 在 drop 中忽略错误，或添加日志
                let _ = ffi::cudaStreamDestroy(self.stream);
            }
        }
    }
}