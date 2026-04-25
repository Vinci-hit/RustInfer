use std::os::raw::c_void;

use crate::base::error::Result;

use super::ffi;

/// cuBLASLt 默认 workspace 大小（字节）。
///
/// 32 MB 是 NVIDIA 建议的最低 workspace（GEMM 规模小时 heuristic 能选到足够的算法）。
/// 对 600B 级别大模型、大 batch 场景，当前版本**暂未**提供扩容接口，
/// 需要时可在 [`CudaConfig::new`] 里直接把这个常量调大重编译。
const DEFAULT_GEMM_WORKSPACE_SIZE: usize = 32 * 1024 * 1024;

/// Split-K flash-decoding 用的 N_SPLIT 常量。
/// 必须与 `.cu` 内 `N_SPLIT` / `N_SPLIT_1B` 保持一致；改常量需三处同步。
pub const FLASH_DECODE_N_SPLIT: usize = 8;

/// CudaConfig 包含了执行 CUDA 内核所需的上下文信息。
/// 例如 CUDA stream, cuBLAS handle, cuDNN handle 等。
/// 这个结构体的生命周期由上层（如计算图执行器）管理。
///
/// # Workspace 约定
///
/// - [`Self::workspace`] (32 MB) 给 cuBLASLt 用，[`Self::new`] 里默认分配。
///
/// - [`Self::flash_decode_workspace`] 给 split-K flash-decoding (pass-1 → pass-2) 用。
///   **默认不分配** (null)；在要跑 bf16 decode 的模型 init 时显式链式调用
///   [`Self::with_flash_decode`] 按实际 `(num_q_heads, head_dim)` 一次性分配：
///
///   ```ignore
///   let cfg = CudaConfig::new()?.with_flash_decode(head_num, head_size)?;
///   ```
///
///   若 decode 路径 kernel 被调用时发现该字段为 null，dispatcher 会返回明确错误
///   而非静默写坏显存。单测中构造的 `CudaConfig::new()` 实例永远不会命中 bf16
///   decode 分支，所以不需要调 `with_flash_decode`。
#[derive(Debug)]
pub struct CudaConfig {
    /// CUDA stream for asynchronous execution.
    pub stream: ffi::cudaStream_t,
    pub cublaslt_handle: ffi::cublasLtHandle_t,
    pub cublas_handle_v2: ffi::cublasHandle_t,
    /// cuBLASLt 算法选择 workspace（32 MB，构造时分配）。
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    /// Split-K flash-decoding (pass-1 → pass-2) 所需的 fp32 scratch。
    /// 默认 null；由 [`Self::with_flash_decode`] 按模型实际 shape 分配。
    pub flash_decode_workspace: *mut c_void,
    pub flash_decode_workspace_size: usize,
    pub cuda_graph: Option<CudaGraph>,
    /// cuDNN handle，用于 Conv2d 等卷积操作。构造时创建并绑定到 stream。
    pub cudnn_handle: ffi::cudnnHandle_t,
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
    /// 创建一个 CudaConfig。分配 stream、cuBLAS/cuBLASLt handle、默认 32 MB
    /// cuBLASLt workspace。**不**分配 flash-decode workspace；如需要请链式调
    /// [`Self::with_flash_decode`]。
    pub fn new() -> Result<Self> {
        let mut stream: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cudaStreamCreate(&mut stream))? };

        let mut cublaslt_handle: ffi::cublasLtHandle_t = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cublasLtCreate(&mut cublaslt_handle))? };
        let mut cublas_handle_v2: ffi::cublasHandle_t = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cublasCreate_v2(&mut cublas_handle_v2))? };

        let mut workspace: *mut c_void = std::ptr::null_mut();
        let workspace_size = DEFAULT_GEMM_WORKSPACE_SIZE;
        unsafe { crate::cuda_check!(ffi::cudaMalloc(&mut workspace, workspace_size))? };

        // cuDNN handle
        let mut cudnn_handle: ffi::cudnnHandle_t = std::ptr::null_mut();
        unsafe {
            let status = ffi::cudnnCreate(&mut cudnn_handle);
            if status != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(crate::base::error::Error::InvalidArgument(
                    format!("cudnnCreate failed: {:?}", status)
                ).into());
            }
            let status = ffi::cudnnSetStream(cudnn_handle, stream);
            if status != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(crate::base::error::Error::InvalidArgument(
                    format!("cudnnSetStream failed: {:?}", status)
                ).into());
            }
        }

        Ok(Self {
            stream,
            cublaslt_handle,
            cublas_handle_v2,
            workspace,
            workspace_size,
            flash_decode_workspace: std::ptr::null_mut(),
            flash_decode_workspace_size: 0,
            cuda_graph: None,
            cudnn_handle,
        })
    }

    /// 一次性分配 split-K flash-decoding 所需的 fp32 scratch，按实际模型形状。
    ///
    /// 总 fp32 数 = `num_q_heads × FLASH_DECODE_N_SPLIT × (2 + head_dim)`。
    /// 典型 Qwen3 (32, 128) ≈ 130 KB，Llama-3.2-1B (32, 64) ≈ 66 KB。
    ///
    /// **只应调用一次，且必须在任何 `capture_graph_begin` 之前**。重复调用或在
    /// graph 捕获期间调用会 panic（前者是误用，后者会破坏 graph）。
    pub fn with_flash_decode(
        mut self,
        num_q_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        assert!(
            self.flash_decode_workspace.is_null(),
            "with_flash_decode called twice on the same CudaConfig"
        );

        let elems = num_q_heads * FLASH_DECODE_N_SPLIT * (2 + head_dim);
        let bytes = elems * std::mem::size_of::<f32>();
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe { crate::cuda_check!(ffi::cudaMalloc(&mut ptr, bytes))? };

        self.flash_decode_workspace = ptr;
        self.flash_decode_workspace_size = bytes;
        Ok(self)
    }
}

// 实现 Drop trait 来自动销毁 CUDA 资源 (RAII)
impl Drop for CudaConfig {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                let _ = ffi::cudaStreamDestroy(self.stream);
            }
        }
        if !self.cublaslt_handle.is_null() {
            unsafe {
                let _ = ffi::cublasLtDestroy(self.cublaslt_handle);
            }
        }
        if !self.cublas_handle_v2.is_null() {
            unsafe {
                let _ = ffi::cublasDestroy_v2(self.cublas_handle_v2);
            }
        }
        if !self.workspace.is_null() {
            unsafe {
                let _ = ffi::cudaFree(self.workspace);
            }
        }
        if !self.flash_decode_workspace.is_null() {
            unsafe {
                let _ = ffi::cudaFree(self.flash_decode_workspace);
            }
        }
        if !self.cudnn_handle.is_null() {
            unsafe {
                let _ = ffi::cudnnDestroy(self.cudnn_handle);
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
    /// 从 `Option<&CudaConfig>` 中获取 stream。
    /// - `Some(config)` → 用 config.stream
    /// - `None` → fallback 到 thread-local current stream（仿 PyTorch 的 `at::cuda::getCurrentCUDAStream()`）
    #[inline]
    pub fn resolve_stream(cuda_config: Option<&CudaConfig>) -> super::ffi::cudaStream_t {
        match cuda_config {
            Some(config) => config.stream,
            None => super::thread_stream::get_current_cuda_stream(),
        }
    }
}

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

    pub fn sync_stream(&self) -> Result<()> {
        unsafe {
            crate::cuda_check!(ffi::cudaStreamSynchronize(self.stream))?;
        }
        Ok(())
    }
}
