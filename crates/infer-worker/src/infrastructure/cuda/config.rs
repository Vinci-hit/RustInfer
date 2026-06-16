//! CudaConfig — CUDA execution context (stream + handles + workspace).

use super::error::cuda_check;
use super::ffi;
use crate::domain::ports::{OpError, OpResult};
use std::collections::HashMap;
use std::os::raw::c_void;

const DEFAULT_GEMM_WORKSPACE_SIZE: usize = 4usize * 1024 * 1024 * 1024;

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

#[derive(Debug)]
pub struct CudaConfig {
    pub stream: ffi::cudaStream_t,
    pub cublaslt_handle: ffi::cublasLtHandle_t,
    pub cublas_handle_v2: ffi::cublasHandle_t,
    pub workspace: *mut c_void,
    pub workspace_size: usize,
    /// Captured CUDA graphs, keyed by slot. Behind a Mutex so the runner
    /// can capture from `&CudaConfig` without an outer `&mut`.
    pub graphs: std::sync::Mutex<HashMap<GraphSlot, CudaGraph>>,
    pub cudnn_handle: ffi::cudnnHandle_t,

    // ─── Bubble-free decode pipeline (copy streams + events) ─────────
    //
    // The compute stream (`stream`) runs forward graph + argmax and then
    // merges C/B back into A. Two auxiliary copy streams overlap transfers
    // with that compute:
    //   - `copy_in_stream` (Si): uploads B (new tokens) + src selector.
    //   - `copy_out_stream` (So): downloads stable A tokens for the
    //     scheduler/client, concurrently with the next forward reading A.
    // Two events order them against the compute stream:
    //   - `ev_in`:  recorded on Si after the B/src upload; compute waits
    //               on it before the merge kernel reads B/src.
    //   - `ev_a`:   recorded on compute after merge refreshed A; So waits
    //               on it before downloading A.
    //   - `ev_out`: recorded on So after the A download; the next merge
    //               waits on it before overwriting A (WAR guard).
    pub copy_in_stream: ffi::cudaStream_t,
    pub copy_out_stream: ffi::cudaStream_t,
    pub ev_in: ffi::cudaEvent_t,
    pub ev_a: ffi::cudaEvent_t,
    pub ev_out: ffi::cudaEvent_t,
}

impl CudaConfig {
    pub fn new() -> OpResult<Self> {
        let mut stream: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaStreamCreate(&mut stream));
        }
        let mut cublaslt_handle: ffi::cublasLtHandle_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cublasLtCreate(&mut cublaslt_handle));
        }
        let mut cublas_handle_v2: ffi::cublasHandle_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cublasCreate_v2(&mut cublas_handle_v2));
        }
        let mut workspace: *mut c_void = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaMalloc(&mut workspace, DEFAULT_GEMM_WORKSPACE_SIZE));
        }
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
        // Bubble-free decode pipeline: two copy streams + two ordering events.
        let mut copy_in_stream: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaStreamCreate(&mut copy_in_stream));
        }
        let mut copy_out_stream: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaStreamCreate(&mut copy_out_stream));
        }
        let mut ev_in: ffi::cudaEvent_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaEventCreate(&mut ev_in));
        }
        let mut ev_a: ffi::cudaEvent_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaEventCreate(&mut ev_a));
        }
        let mut ev_out: ffi::cudaEvent_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaEventCreate(&mut ev_out));
        }
        Ok(Self {
            stream,
            cublaslt_handle,
            cublas_handle_v2,
            workspace,
            workspace_size: DEFAULT_GEMM_WORKSPACE_SIZE,
            graphs: std::sync::Mutex::new(HashMap::new()),
            cudnn_handle,
            copy_in_stream,
            copy_out_stream,
            ev_in,
            ev_a,
            ev_out,
        })
    }

    pub fn graph_ready(&self, slot: GraphSlot) -> bool {
        self.graphs.lock().unwrap().contains_key(&slot)
    }

    pub fn capture_begin_relaxed(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaStreamBeginCapture(self.stream, 1));
        }
        Ok(())
    }

    pub fn capture_end(&self, slot: GraphSlot) -> OpResult<()> {
        let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaStreamEndCapture(self.stream, &mut graph));
        }
        let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaGraphInstantiate(&mut exec, graph, 0));
        }
        self.graphs
            .lock()
            .unwrap()
            .insert(slot, CudaGraph { graph, exec });
        Ok(())
    }

    pub fn launch(&self, slot: GraphSlot) -> OpResult<()> {
        let guard = self.graphs.lock().unwrap();
        let g = guard
            .get(&slot)
            .ok_or_else(|| OpError::Kernel(format!("graph {:?} not found", slot)))?;
        unsafe {
            cuda_check!(ffi::cudaGraphLaunch(g.exec, self.stream));
        }
        super::error::check_last_error("cuda graph launch observed prior kernel error")?;
        Ok(())
    }

    pub fn invalidate_all_graphs(&self) {
        self.graphs.lock().unwrap().clear();
    }

    pub fn synchronize(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaStreamSynchronize(self.stream));
        }
        super::error::check_last_error("cuda stream sync observed prior kernel error")?;
        Ok(())
    }

    // ─── Bubble-free decode pipeline ordering primitives ─────────────

    /// Record `ev_in` on the copy-in stream (Si), after the B/src upload
    /// has been enqueued there. The compute stream must wait on this event
    /// before the merge kernel reads B/src.
    pub fn record_copy_in(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaEventRecord(self.ev_in, self.copy_in_stream));
        }
        Ok(())
    }

    /// Make the compute stream wait for `ev_in` (the B/src upload on Si).
    /// Issued before the merge kernel launch.
    pub fn compute_wait_copy_in(&self) -> OpResult<()> {
        // flags=0 (cudaEventWaitDefault).
        unsafe {
            cuda_check!(ffi::cudaStreamWaitEvent(self.stream, self.ev_in, 0));
        }
        Ok(())
    }

    /// Record `ev_a` on the compute stream after A contains the committed
    /// output token. The copy-out stream waits on this before downloading A.
    pub fn record_compute_a(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaEventRecord(self.ev_a, self.stream));
        }
        Ok(())
    }

    /// Make the copy-out stream (So) wait for `ev_a`. Issued before the
    /// stable A download.
    pub fn copy_out_wait_compute_a(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaStreamWaitEvent(self.copy_out_stream, self.ev_a, 0));
        }
        Ok(())
    }

    /// Record `ev_out` on the copy-out stream after the A download has been
    /// enqueued. The next merge must wait on this before overwriting A.
    pub fn record_copy_out(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaEventRecord(self.ev_out, self.copy_out_stream));
        }
        Ok(())
    }

    /// Make the compute stream wait for `ev_out` (the previous step's A
    /// download on So) before overwriting A. Issued before the merge.
    pub fn compute_wait_copy_out(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaStreamWaitEvent(self.stream, self.ev_out, 0));
        }
        Ok(())
    }

    /// Sync only the copy-out stream (So) — used to ensure the scheduler's
    /// A download has landed on the host before reading it.
    pub fn synchronize_copy_out(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaStreamSynchronize(self.copy_out_stream));
        }
        super::error::check_last_error("cuda copy-out sync observed prior kernel error")?;
        Ok(())
    }

    /// Sync the copy-in stream (Si).
    pub fn synchronize_copy_in(&self) -> OpResult<()> {
        unsafe {
            cuda_check!(ffi::cudaStreamSynchronize(self.copy_in_stream));
        }
        super::error::check_last_error("cuda copy-in sync observed prior kernel error")?;
        Ok(())
    }

    /// Async H2D copy on the copy-in stream (Si). Used to upload B
    /// (`new_token_dev`) + src (`src_map_dev`) while the compute stream is
    /// busy. The host buffer must stay alive until Si consumes the copy
    /// (the workspace owns its host staging for the runner's lifetime).
    ///
    /// # Safety
    /// `dst` is a device pointer with ≥ `size` bytes, `src` a host pointer
    /// with ≥ `size` bytes.
    pub unsafe fn upload_h2d_copy_in(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> OpResult<()> {
        if size == 0 {
            return Ok(());
        }
        unsafe {
            cuda_check!(ffi::cudaMemcpyAsync(
                dst,
                src,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.copy_in_stream,
            ));
        }
        Ok(())
    }

    /// Async D2H copy on the copy-out stream (So). Used to download A
    /// (next-step input_ids) for the scheduler concurrently with the next
    /// forward. Caller must sync So (via `synchronize_copy_out`) before
    /// reading the host destination.
    ///
    /// # Safety
    /// `dst` is a host pointer with ≥ `size` bytes, `src` a device pointer
    /// with ≥ `size` bytes.
    pub unsafe fn download_d2h_copy_out(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
    ) -> OpResult<()> {
        if size == 0 {
            return Ok(());
        }
        unsafe {
            cuda_check!(ffi::cudaMemcpyAsync(
                dst,
                src,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.copy_out_stream,
            ));
        }
        Ok(())
    }
}

impl Drop for CudaConfig {
    fn drop(&mut self) {
        unsafe {
            if !self.ev_in.is_null() {
                ffi::cudaEventDestroy(self.ev_in);
            }
            if !self.ev_a.is_null() {
                ffi::cudaEventDestroy(self.ev_a);
            }
            if !self.ev_out.is_null() {
                ffi::cudaEventDestroy(self.ev_out);
            }
            if !self.copy_in_stream.is_null() {
                ffi::cudaStreamDestroy(self.copy_in_stream);
            }
            if !self.copy_out_stream.is_null() {
                ffi::cudaStreamDestroy(self.copy_out_stream);
            }
            if !self.stream.is_null() {
                ffi::cudaStreamDestroy(self.stream);
            }
            if !self.cublaslt_handle.is_null() {
                ffi::cublasLtDestroy(self.cublaslt_handle);
            }
            if !self.cublas_handle_v2.is_null() {
                ffi::cublasDestroy_v2(self.cublas_handle_v2);
            }
            if !self.workspace.is_null() {
                ffi::cudaFree(self.workspace);
            }
            if !self.cudnn_handle.is_null() {
                ffi::cudnnDestroy(self.cudnn_handle);
            }
        }
    }
}
unsafe impl Send for CudaConfig {}
unsafe impl Sync for CudaConfig {}
