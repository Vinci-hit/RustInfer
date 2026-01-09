use std::os::raw::c_void;

use crate::base::error::Result;

use super::ffi;

/// CudaConfig 包含了执行 CUDA 内核所需的上下文信息。
/// 例如 CUDA stream, cuBLAS handle, cuDNN handle 等。
/// 这个结构体的生命周期由上层（如计算图执行器）管理。
#[derive(Debug)]
pub struct CudaConfig {
    /// CUDA stream for asynchronous execution.
    pub stream: ffi::cudaStream_t,
    pub cublaslt_handle: ffi::cublasLtHandle_t,
    pub cublas_handle_v2: ffi::cublasHandle_t,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    pub cuda_graph: Option<CudaGraph>,
    // pub cudnn_handle: cudnnHandle_t,   // (未来)
}
#[derive(Debug)]
pub struct CudaGraph {
    graph: ffi::cudaGraph_t,
    exec: ffi::cudaGraphExec_t,
}
impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe {
            ffi::cudaGraphExecDestroy(self.exec);
            ffi::cudaGraphDestroy(self.graph);
        }
    }
}

impl CudaConfig {
    /// 创建一个新的 CudaConfig，通常在初始化计算图时调用。
    pub fn new() -> Result<Self> {
        let mut stream: ffi::cudaStream_t = std::ptr::null_mut();
        
        // 创建一个新的 CUDA stream
        unsafe { crate::cuda_check!(ffi::cudaStreamCreate(&mut stream))? };
        
        // 创建 cuBLAS handle
        let mut cublaslt_handle: ffi::cublasLtHandle_t = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cublasLtCreate(&mut cublaslt_handle))? };
        // 创建 cuBLAS handle v2
        let mut cublas_handle_v2: ffi::cublasHandle_t = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cublasCreate_v2(&mut cublas_handle_v2))? };
        
        // 创建 workspace
        let mut workspace: *mut c_void = std::ptr::null_mut();
        let workspace_size = 32 * 1024 * 1024; // 32MB
        unsafe { crate::cuda_check!(ffi::cudaMalloc(&mut workspace, workspace_size))? };

        let cuda_graph = None;

        Ok(Self { stream, cublaslt_handle, cublas_handle_v2, workspace, workspace_size, cuda_graph })
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
        if !self.cublaslt_handle.is_null() {
            unsafe {
                // 在 drop 中忽略错误，或添加日志
                let _ = ffi::cublasLtDestroy(self.cublaslt_handle);
            }
        }
        if !self.cublas_handle_v2.is_null() {
            unsafe {
                // 在 drop 中忽略错误，或添加日志
                let _ = ffi::cublasDestroy_v2(self.cublas_handle_v2);
            }
        }
        if !self.workspace.is_null() {
            unsafe {
                // 在 drop 中忽略错误，或添加日志
                let _ = ffi::cudaFree(self.workspace);
            }
        }
        // cuda_graph 会自动在其 Drop 中释放
    }
}
// CUDA resources are internally thread-safe, so we can safely implement Send/Sync
// Raw pointers prevent automatic Send/Sync, but CUDA handles synchronization
unsafe impl Send for CudaConfig {}
unsafe impl Sync for CudaConfig {}

impl CudaConfig {
    pub fn capture_graph_begin(&mut self) -> Result<()> {
        unsafe {
            crate::cuda_check!(ffi::cudaStreamBeginCapture(self.stream, 0))?;
        }
        Ok(())
    }

    pub fn capture_graph_end(&mut self) -> Result<()> {
        unsafe {
            let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
            crate::cuda_check!(ffi::cudaStreamEndCapture(self.stream, &mut graph))?;

            let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
            crate::cuda_check!(ffi::cudaGraphInstantiate(&mut exec, graph, 0))?;

            self.cuda_graph = Some(CudaGraph { graph, exec });
        }
        Ok(())
    }

    pub fn launch_graph(&self) -> Result<()> {
        if let Some(cuda_graph) = &self.cuda_graph {
            unsafe {
                crate::cuda_check!(ffi::cudaGraphLaunch(cuda_graph.exec, self.stream))?;
            }
            Ok(())
        } else {
            Err(crate::base::error::Error::InvalidArgument("CUDA graph not captured".to_string()).into())
        }
    }
}