use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Mutex;

use crate::base::error::Result;

use super::ffi;

/// cuBLASLt 默认 workspace 大小（字节）。
const DEFAULT_GEMM_WORKSPACE_SIZE: usize = 128 * 1024 * 1024;

/// CUDA Graph 的用途分桶。一个 [`CudaConfig`] 可以同时 cache 多张不同用途/不同 batch
/// 形状的 graph；同一 key 里只保存一张 graph（后 capture 的覆盖前一张）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphSlot {
    LlmDecode {
        batch: usize,
        buffer_id: usize,
        slot_signature: u64,
    },
    LlmMixedPreAttn(usize),
    LlmMixedPostAttn(usize),
    Denoise {
        latent_h: usize,
        latent_w: usize,
        cap_padded_len: usize,
        steps: usize,
    },
}

/// CudaConfig 包含了执行 CUDA 内核所需的上下文信息。
/// 例如 CUDA stream, cuBLAS handle, cuDNN handle 等。
/// 这个结构体的生命周期由上层（如计算图执行器）管理。
///
/// # Workspace 约定
///
/// - [`Self::workspace`] (128 MB) 给 cuBLASLt 用，[`Self::new`] 里默认分配。
///
/// Attention workspaces are now owned by the caller (see
/// `FlashAttnDecodeBatch::workspace_bytes`), not by `CudaConfig`.
#[derive(Debug)]
pub struct CudaConfig {
    /// CUDA stream for asynchronous execution.
    pub stream: ffi::cudaStream_t,
    pub cublaslt_handle: ffi::cublasLtHandle_t,
    pub cublas_handle_v2: ffi::cublasHandle_t,
    /// cuBLASLt 算法选择 workspace（128 MB，构造时分配）。
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    /// 按用途/shape 分桶 cache 的 CUDA Graph。key 见 [`GraphSlot`]。
    pub graphs: HashMap<GraphSlot, CudaGraph>,
    /// cuDNN handle，用于 Conv2d 等卷积操作。构造时创建并绑定到 stream。
    pub cudnn_handle: ffi::cudnnHandle_t,
    /// Descriptor / algorithm / workspace cache shared across every
    /// `conv2d_cudnn` invocation that runs against this handle.
    pub conv2d_cache: Mutex<crate::op::kernels::cuda::Conv2dCache>,
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
            graphs: HashMap::new(),
            cudnn_handle,
            conv2d_cache: Mutex::new(
                crate::op::kernels::cuda::Conv2dCache::default(),
            ),
        })
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
    // ─────────────────── Graph capture / replay API ───────────────────
    //
    // 所有调用方（LLM decode / LLM batch decode / diffusion denoise 等）都走这套；
    // 用 `GraphSlot` 区分用途与 shape（对 LlmDecode 还带 batch_size 细分）。
    //
    // 使用范式（首次 capture + 之后 replay）：
    //     let slot = GraphSlot::LlmDecode { batch: batch_size, buffer_id };
    //     if !cfg.graph_ready(slot) {
    //         cfg.capture_begin()?;              // 开 stream capture
    //         run_forward_once_in_capture(...)?; // capture 当前 forward
    //         cfg.capture_end(slot)?;            // 结束 + instantiate，塞进 graphs[slot]
    //     } else {
    //         cfg.launch(slot)?;                 // 直接 replay
    //     }

    /// 判断某 slot 是否已有 capture 好的 graph。
    pub fn graph_ready(&self, slot: GraphSlot) -> bool {
        self.graphs.contains_key(&slot)
    }

    /// 开始 stream capture。`&self` 即可（capture state 由 CUDA driver 内部管理，
    /// 实际把 graph 写入 HashMap 的动作在 `capture_end` 里，需 `&mut self`）。
    ///
    /// 默认走 `cudaStreamCaptureModeGlobal`（最严格）。如果 capture 期间有
    /// 任何 op 在 legacy default stream 上隐式 launch（被 cuBLAS / cuDNN
    /// 等第三方库踩到），会触发 `cudaErrorStreamCaptureImplicit` —— 那就改用
    /// [`Self::capture_begin_relaxed`]，CUDA 会放过这种隐式依赖。
    pub fn capture_begin(&self) -> Result<()> {
        unsafe { crate::cuda_check!(ffi::cudaStreamBeginCapture(self.stream, 0))?; }
        Ok(())
    }

    /// `cudaStreamCaptureModeRelaxed` 版的 capture 启动 —— 允许 capture 期间
    /// 有跨 stream / legacy default stream 的隐式依赖（cuBLAS handle 在第三方
    /// 库内部偶尔会踩到）。LLM decode graph 路径用这个；纯自有 kernel 路径
    /// （e.g. z-image denoise）继续用严格的 [`Self::capture_begin`]。
    pub fn capture_begin_relaxed(&self) -> Result<()> {
        const RELAXED: ffi::cudaStreamCaptureMode = 2;
        unsafe {
            crate::cuda_check!(ffi::cudaStreamBeginCapture(self.stream, RELAXED))?;
        }
        Ok(())
    }

    /// 结束 capture，instantiate 成可 replay 的 exec graph，并放到 `graphs[slot]`。
    /// 如果该 slot 已存在旧 graph，会被覆盖（旧 graph 的 Drop 会 destroy）。
    pub fn capture_end(&mut self, slot: GraphSlot) -> Result<()> {
        unsafe {
            let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
            crate::cuda_check!(ffi::cudaStreamEndCapture(self.stream, &mut graph))?;
            let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
            crate::cuda_check!(ffi::cudaGraphInstantiate(&mut exec, graph, 0))?;
            self.graphs.insert(slot, CudaGraph { graph, exec });
        }
        Ok(())
    }

    /// Replay slot 里的 graph；如未 capture 返回错误。
    pub fn launch(&self, slot: GraphSlot) -> Result<()> {
        let g = self.graphs.get(&slot).ok_or_else(|| {
            crate::base::error::Error::InvalidArgument(
                format!("CUDA graph not captured for slot {:?}", slot)
            )
        })?;
        unsafe { crate::cuda_check!(ffi::cudaGraphLaunch(g.exec, self.stream))?; }
        Ok(())
    }

    pub fn sync_stream(&self) -> Result<()> {
        unsafe { crate::cuda_check!(ffi::cudaStreamSynchronize(self.stream))?; }
        Ok(())
    }
}
