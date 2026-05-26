//! CudaConfig — CUDA execution context (stream + handles + workspace).

use std::collections::HashMap;
use std::os::raw::c_void;
use super::ffi;
use super::error::cuda_check;
use crate::domain::ports::{OpError, OpResult};

const DEFAULT_GEMM_WORKSPACE_SIZE: usize = 128 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphSlot {
    LlmDecode { batch: usize, buffer_id: usize, slot_signature: u64 },
    LlmMixedPreAttn(usize),
    LlmMixedPostAttn(usize),
    Denoise { latent_h: usize, latent_w: usize, cap_padded_len: usize, steps: usize },
}

#[derive(Debug)]
pub struct CudaGraph {
    graph: ffi::cudaGraph_t,
    exec: ffi::cudaGraphExec_t,
}
impl Drop for CudaGraph {
    fn drop(&mut self) {
        unsafe { ffi::cudaGraphExecDestroy(self.exec); ffi::cudaGraphDestroy(self.graph); }
    }
}

#[derive(Debug)]
pub struct CudaConfig {
    pub stream: ffi::cudaStream_t,
    pub cublaslt_handle: ffi::cublasLtHandle_t,
    pub cublas_handle_v2: ffi::cublasHandle_t,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    pub graphs: HashMap<GraphSlot, CudaGraph>,
    pub cudnn_handle: ffi::cudnnHandle_t,
}

impl CudaConfig {
    pub fn new() -> OpResult<Self> {
        let mut stream: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe { cuda_check!(ffi::cudaStreamCreate(&mut stream)); }
        let mut cublaslt_handle: ffi::cublasLtHandle_t = std::ptr::null_mut();
        unsafe { cuda_check!(ffi::cublasLtCreate(&mut cublaslt_handle)); }
        let mut cublas_handle_v2: ffi::cublasHandle_t = std::ptr::null_mut();
        unsafe { cuda_check!(ffi::cublasCreate_v2(&mut cublas_handle_v2)); }
        let mut workspace: *mut c_void = std::ptr::null_mut();
        unsafe { cuda_check!(ffi::cudaMalloc(&mut workspace, DEFAULT_GEMM_WORKSPACE_SIZE)); }
        let mut cudnn_handle: ffi::cudnnHandle_t = std::ptr::null_mut();
        unsafe {
            let s = ffi::cudnnCreate(&mut cudnn_handle);
            if s != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(OpError::Kernel(format!("cudnnCreate failed: {:?}", s)));
            }
            let s = ffi::cudnnSetStream(cudnn_handle, stream);
            if s != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(OpError::Kernel(format!("cudnnSetStream failed: {:?}", s)));
            }
        }
        Ok(Self { stream, cublaslt_handle, cublas_handle_v2, workspace, workspace_size: DEFAULT_GEMM_WORKSPACE_SIZE, graphs: HashMap::new(), cudnn_handle })
    }

    pub fn graph_ready(&self, slot: GraphSlot) -> bool { self.graphs.contains_key(&slot) }

    pub fn capture_begin_relaxed(&self) -> OpResult<()> {
        unsafe { cuda_check!(ffi::cudaStreamBeginCapture(self.stream, 2)); }
        Ok(())
    }

    pub fn capture_end(&mut self, slot: GraphSlot) -> OpResult<()> {
        let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
        unsafe { cuda_check!(ffi::cudaStreamEndCapture(self.stream, &mut graph)); }
        let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
        unsafe { cuda_check!(ffi::cudaGraphInstantiate(&mut exec, graph, 0)); }
        self.graphs.insert(slot, CudaGraph { graph, exec });
        Ok(())
    }

    pub fn launch(&self, slot: GraphSlot) -> OpResult<()> {
        let g = self.graphs.get(&slot).ok_or_else(|| OpError::Kernel(format!("graph {:?} not found", slot)))?;
        unsafe { cuda_check!(ffi::cudaGraphLaunch(g.exec, self.stream)); }
        Ok(())
    }

    pub fn invalidate_all_graphs(&mut self) { self.graphs.clear(); }

    pub fn synchronize(&self) -> OpResult<()> {
        unsafe { cuda_check!(ffi::cudaStreamSynchronize(self.stream)); }
        Ok(())
    }
}

impl Drop for CudaConfig {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() { ffi::cudaStreamDestroy(self.stream); }
            if !self.cublaslt_handle.is_null() { ffi::cublasLtDestroy(self.cublaslt_handle); }
            if !self.cublas_handle_v2.is_null() { ffi::cublasDestroy_v2(self.cublas_handle_v2); }
            if !self.workspace.is_null() { ffi::cudaFree(self.workspace); }
            if !self.cudnn_handle.is_null() { ffi::cudnnDestroy(self.cudnn_handle); }
        }
    }
}
unsafe impl Send for CudaConfig {}
unsafe impl Sync for CudaConfig {}
