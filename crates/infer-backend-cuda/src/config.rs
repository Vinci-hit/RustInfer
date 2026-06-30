//! CudaConfig — CUDA execution context (stream + handles + workspace).

use super::error::cuda_check;
use super::ffi;
use infer_core::ports::{OpError, OpResult};
use std::collections::HashMap;
use std::os::raw::c_void;

const DEFAULT_GEMM_WORKSPACE_SIZE: usize = 4usize * 1024 * 1024 * 1024;

/// Size of the graph-capture scratch arena (1 GiB). Bounds the peak per-step
/// decode scratch; for a 1B–8B dense model at batch ≤ 256 the real peak is a
/// few hundred MiB, so this leaves comfortable headroom.
const GRAPH_ARENA_SIZE: usize = 1usize * 1024 * 1024 * 1024;

/// Upper bound on bytes the recycling pool retains across the free lists. The
/// hot per-forward scratch is served from `ForwardScratch` (not the pool), so
/// the pool only recycles residual/transient allocations; this cap stops a
/// pathological mix of distinct size classes (e.g. ragged prompt lengths) from
/// growing the free list without bound. Over budget, freed blocks are
/// `cudaFree`d instead of retained (off the hot path, never mid-capture).
const POOL_RETAIN_BUDGET: usize = 512usize * 1024 * 1024;

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

/// Size-keyed free-list of recycled device blocks. Eliminates the per-forward
/// `cudaMalloc`/`cudaFree` storm: every transient scratch `Tensor` returns its
/// block here on drop (keyed by 256B-rounded size) and the next same-size alloc
/// pops it, so an eager forward issues ~0 `cudaMalloc`/`cudaFree` in steady
/// state. Disjoint from the graph-capture arena — arena pointers are filtered
/// out (via `arena_contains`) before they can reach the pool, so a pooled block
/// can never alias a captured graph's baked-in scratch addresses.
#[derive(Debug, Default)]
struct PoolState {
    /// key = `round_up_256(size)` → stack of reusable device pointers.
    free: HashMap<usize, Vec<*mut c_void>>,
    /// Bytes currently handed out (popped or cold-malloc'd) and not yet returned.
    live_bytes: usize,
    /// Bytes retained across the free lists, available for reuse.
    pooled_bytes: usize,
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

    // ─── Graph-capture scratch arena ─────────────────────────────────
    //
    // CUDA stream capture forbids `cudaMalloc`. The decode forward allocates
    // ~190 transient scratch tensors per step, so during capture (and replay)
    // those allocations are served from this pre-reserved bump arena instead.
    // Sizes/order are deterministic for a fixed decode batch, so the arena
    // hands out identical addresses every step — exactly what graph replay
    // requires. `free` of an arena pointer is a no-op; the arena is reset to
    // offset 0 at the start of each capture.
    pub arena_base: std::sync::atomic::AtomicPtr<c_void>,
    pub arena_off: std::sync::atomic::AtomicUsize,
    pub arena_enabled: std::sync::atomic::AtomicBool,

    // ─── Recycling scratch allocator (eager forward path) ────────────
    //
    // Outside graph capture, transient scratch tensors are recycled through
    // this size-keyed free list instead of round-tripping `cudaMalloc`/
    // `cudaFree` (the latter device-synchronizes). See `PoolState`.
    pool: std::sync::Mutex<PoolState>,
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
        // Graph-capture scratch arena (best-effort: if the reservation fails,
        // `arena_base` stays null and `supports_graphs()` reports false, so the
        // worker transparently falls back to eager decode).
        let mut arena_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            if ffi::cudaMalloc(&mut arena_ptr, GRAPH_ARENA_SIZE) != ffi::cudaError_cudaSuccess {
                arena_ptr = std::ptr::null_mut();
            }
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
            arena_base: std::sync::atomic::AtomicPtr::new(arena_ptr),
            arena_off: std::sync::atomic::AtomicUsize::new(0),
            arena_enabled: std::sync::atomic::AtomicBool::new(false),
            pool: std::sync::Mutex::new(PoolState::default()),
        })
    }

    // ─── Graph-capture scratch arena ─────────────────────────────────

    /// True if the arena reservation succeeded (graphs are usable).
    pub fn arena_available(&self) -> bool {
        !self
            .arena_base
            .load(std::sync::atomic::Ordering::Acquire)
            .is_null()
    }

    /// Reset the bump offset and route subsequent `alloc_bytes` through the
    /// arena. Called just before `cudaStreamBeginCapture`.
    pub fn arena_begin(&self) {
        self.arena_off
            .store(0, std::sync::atomic::Ordering::Release);
        self.arena_enabled
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Stop routing allocations through the arena (eager `cudaMalloc` resumes).
    pub fn arena_end(&self) {
        self.arena_enabled
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Serve `size` bytes from the arena, or `None` if the arena is disabled,
    /// unavailable, or exhausted (caller then falls back to `cudaMalloc`).
    /// Zero-initializes asynchronously on the compute stream (capture-safe).
    pub fn arena_alloc(&self, size: usize) -> Option<*mut c_void> {
        use std::sync::atomic::Ordering;
        if !self.arena_enabled.load(Ordering::Acquire) {
            return None;
        }
        let base = self.arena_base.load(Ordering::Acquire);
        if base.is_null() {
            return None;
        }
        let n = self.round_up_256(size);
        let off = self.arena_off.fetch_add(n, Ordering::AcqRel);
        if off + n > GRAPH_ARENA_SIZE {
            self.arena_off.fetch_sub(n, Ordering::AcqRel);
            return None;
        }
        let ptr = unsafe { (base as *mut u8).add(off) as *mut c_void };
        // Zero-initialize on the compute stream (capture-safe: cudaMemsetAsync
        // is a recordable stream op).
        unsafe {
            ffi::cudaMemsetAsync(ptr, 0, n, self.stream);
        }
        Some(ptr)
    }

    /// Whether `ptr` lies inside the arena (its `free` must be a no-op).
    pub fn arena_contains(&self, ptr: *mut c_void) -> bool {
        let base = self.arena_base.load(std::sync::atomic::Ordering::Acquire);
        if base.is_null() {
            return false;
        }
        let p = ptr as usize;
        let b = base as usize;
        p >= b && p < b + GRAPH_ARENA_SIZE
    }

    // ─── Recycling scratch allocator ─────────────────────────────────

    /// Size class for the free list: round up to 256 B so an alloc and its
    /// later free hash to the same bucket. Shared with the arena so both
    /// allocators agree on block sizes.
    #[inline]
    pub fn round_up_256(&self, n: usize) -> usize {
        (n.max(1) + 255) & !255usize
    }

    /// Pop a previously-freed block of the given rounded size, if any.
    /// No CUDA calls under the lock — the caller zeros the block afterward.
    pub fn pool_pop(&self, n: usize) -> Option<*mut c_void> {
        let mut g = self.pool.lock().unwrap();
        let p = g.free.get_mut(&n).and_then(|v| v.pop());
        if p.is_some() {
            g.pooled_bytes = g.pooled_bytes.saturating_sub(n);
            g.live_bytes += n;
        }
        p
    }

    /// Record a cold `cudaMalloc` so `live_bytes` stays consistent (diagnostic).
    pub fn pool_note_cold_alloc(&self, n: usize) {
        self.pool.lock().unwrap().live_bytes += n;
    }

    /// Return a block to the free list for later reuse (no `cudaFree`), unless
    /// retaining it would push the pool over `POOL_RETAIN_BUDGET` — then the
    /// block is `cudaFree`d (outside the lock; legal here since arena/capture
    /// pointers were already filtered out by `free_bytes`).
    pub fn pool_push(&self, n: usize, ptr: *mut c_void) {
        let retained = {
            let mut g = self.pool.lock().unwrap();
            g.live_bytes = g.live_bytes.saturating_sub(n);
            if g.pooled_bytes + n > POOL_RETAIN_BUDGET {
                false
            } else {
                debug_assert!(
                    g.free.get(&n).map_or(true, |v| !v.contains(&ptr)),
                    "double-free into cuda pool: ptr={:?} size-class={}",
                    ptr,
                    n
                );
                g.free.entry(n).or_default().push(ptr);
                g.pooled_bytes += n;
                true
            }
        };
        if !retained {
            // SAFETY: ptr came from cudaMalloc and is not arena/capture-owned.
            unsafe {
                ffi::cudaFree(ptr);
            }
        }
    }

    /// `cudaFree` every retained block and clear the free lists. Called on
    /// device teardown (Drop). Must run while the CUDA context is still valid.
    pub fn pool_drain(&self) {
        let mut g = self.pool.lock().unwrap();
        for (_n, v) in g.free.drain() {
            for ptr in v {
                unsafe {
                    ffi::cudaFree(ptr);
                }
            }
        }
        g.pooled_bytes = 0;
    }

    pub fn graph_ready(&self, slot: GraphSlot) -> bool {
        self.graphs.lock().unwrap().contains_key(&slot)
    }

    /// Diagnostic: returns "active", "invalidated", "none", keyed by the
    /// current capture state of the compute stream.
    pub fn capture_state(&self) -> &'static str {
        let mut st: ffi::cudaStreamCaptureStatus =
            ffi::cudaStreamCaptureStatus_cudaStreamCaptureStatusNone;
        unsafe {
            ffi::cudaStreamIsCapturing(self.stream, &mut st);
        }
        match st {
            ffi::cudaStreamCaptureStatus_cudaStreamCaptureStatusActive => "active",
            ffi::cudaStreamCaptureStatus_cudaStreamCaptureStatusInvalidated => "invalidated",
            _ => "none",
        }
    }

    pub fn capture_begin_relaxed(&self) -> OpResult<()> {
        // Mode 2 = cudaStreamCaptureModeRelaxed. Relaxed (not ThreadLocal=1)
        // is required so the potentially-unsafe API calls that cuBLASLt / cuDNN
        // make internally while enqueuing a matmul/attention are tolerated
        // during capture instead of returning an error (e.g. cuBLASLt
        // EXECUTION_FAILED / status=13 under ThreadLocal capture).
        unsafe {
            cuda_check!(ffi::cudaStreamBeginCapture(self.stream, 2));
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
        // The scratch pool is intentionally left intact: pooled blocks are a
        // disjoint cudaMalloc region, never baked into any captured graph, so
        // dropping graphs requires no pool drain.
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

    /// Page-lock (pin) a host `i32` buffer in place so `cudaMemcpyAsync` on the
    /// copy-in/copy-out streams runs truly asynchronously (pageable host memory
    /// makes those copies host-synchronous, which serializes the decode
    /// pipeline). Call once per buffer — re-registering an already-pinned
    /// region errors. Not unregistered: the ABC staging lives for the process
    /// lifetime, so the OS reclaims it at exit.
    pub fn pin_host_i32(&self, buf: &[i32]) -> OpResult<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::size_of_val(buf);
        // flags = cudaHostRegisterDefault (0): page-lock in place.
        unsafe {
            cuda_check!(ffi::cudaHostRegister(buf.as_ptr() as *mut c_void, bytes, 0));
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
            let arena = self.arena_base.load(std::sync::atomic::Ordering::Acquire);
            if !arena.is_null() {
                ffi::cudaFree(arena);
            }
        }
        // Release every recycled scratch block (disjoint from the arena).
        self.pool_drain();
    }
}
unsafe impl Send for CudaConfig {}
unsafe impl Sync for CudaConfig {}
