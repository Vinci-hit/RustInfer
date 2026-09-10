//! CudaConfig — CUDA execution context (stream + handles + workspace).

use super::error::cuda_check;
use super::ffi;
pub use super::pool::CudaPoolStats;
use super::pool::{PoolBlock, PoolState};
use infer_core::ports::{OpError, OpResult};
use std::collections::HashMap;
use std::os::raw::c_void;

#[cfg(test)]
static CUDA_TEST_IN_USE: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
#[cfg(test)]
static CUDA_TEST_AVAILABLE: std::sync::Condvar = std::sync::Condvar::new();

/// Test-only process-wide lease. A default `CudaConfig` reserves 256 MiB
/// eagerly and another 256 MiB on first graph capture, so
/// constructing one per default test-runner thread exhausts even large GPUs.
/// The lease is Send-safe and remains held through `CudaConfig::drop`.
#[cfg(test)]
#[derive(Debug)]
struct CudaTestLease;

#[cfg(test)]
impl CudaTestLease {
    fn acquire() -> Self {
        let mut in_use = CUDA_TEST_IN_USE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while *in_use {
            in_use = CUDA_TEST_AVAILABLE
                .wait(in_use)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        *in_use = true;
        Self
    }
}

#[cfg(test)]
impl Drop for CudaTestLease {
    fn drop(&mut self) {
        let mut in_use = CUDA_TEST_IN_USE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *in_use = false;
        CUDA_TEST_AVAILABLE.notify_one();
    }
}

const MIB: usize = 1024 * 1024;

fn pool_allocation_size(size: usize) -> OpResult<usize> {
    size.max(1)
        .checked_add(255)
        .map(|n| n & !255usize)
        .ok_or_else(|| OpError::Kernel("CUDA memory pool allocation size overflow".into()))
}

/// Central CUDA scratch-memory policy.
///
/// This is the single place that defines fixed CUDA scratch regions and cache
/// limits. Add or remove a region here instead of scattering size constants and
/// raw pointer/length pairs through `CudaConfig`. Production callers pass the
/// values read from the shared launch configuration to `Cuda::with_memory_plan`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryPlan {
    pub kernel_workspace_bytes: usize,
    pub graph_arena_bytes: usize,
    pub pool_retain_bytes: usize,
}

impl Default for CudaMemoryPlan {
    fn default() -> Self {
        Self {
            // Qwen3-4B FP8 at the default 4096-token cap needs about 40 MiB
            // for dynamic activations plus a comparatively small CUTLASS
            // scheduler workspace. Keep headroom for cuBLASLt/cuDNN users
            // without reserving the previous unconditional 4 GiB.
            kernel_workspace_bytes: 256 * MIB,
            // The graph arena covers all capture-time transient tensors, not
            // merely one kernel's scratch, so it keeps separate headroom.
            graph_arena_bytes: 256 * MIB,
            // This is only a retention ceiling; it is not preallocated.
            pool_retain_bytes: 256 * MIB,
        }
    }
}

impl CudaMemoryPlan {
    pub fn with_kernel_workspace_bytes(mut self, bytes: usize) -> Self {
        self.kernel_workspace_bytes = bytes;
        self
    }

    pub fn with_graph_arena_bytes(mut self, bytes: usize) -> Self {
        self.graph_arena_bytes = bytes;
        self
    }

    pub fn with_pool_retain_bytes(mut self, bytes: usize) -> Self {
        self.pool_retain_bytes = bytes;
        self
    }
}

/// Immutable capabilities of one CUDA device.
///
/// Query these once when the backend is constructed. Kernel launchers consume
/// this value instead of issuing `cudaGetDevice*` calls in their hot paths, and
/// device initializers use the same source of truth when deciding which kernel
/// variants the hardware can support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaDeviceInfo {
    pub(crate) device_id: i32,
    pub(crate) sm_count: i32,
    pub(crate) max_dynamic_shared_memory: i32,
}

impl CudaDeviceInfo {
    fn query(device_id: i32) -> OpResult<Self> {
        let mut sm_count = 0;
        let mut max_dynamic_shared_memory = 0;
        unsafe {
            cuda_check!(ffi::cudaDeviceGetAttribute(
                &mut sm_count,
                ffi::cudaDeviceAttr_cudaDevAttrMultiProcessorCount,
                device_id,
            ));
            cuda_check!(ffi::cudaDeviceGetAttribute(
                &mut max_dynamic_shared_memory,
                ffi::cudaDeviceAttr_cudaDevAttrMaxSharedMemoryPerBlockOptin,
                device_id,
            ));
        }
        if sm_count <= 0 || max_dynamic_shared_memory <= 0 {
            return Err(OpError::Kernel(format!(
                "invalid CUDA device capabilities for cuda:{device_id}: sm_count={sm_count}, \
                 max_dynamic_shared_memory={max_dynamic_shared_memory}"
            )));
        }
        Ok(Self {
            device_id,
            sm_count,
            max_dynamic_shared_memory,
        })
    }
}

#[derive(Debug)]
struct DeviceRegion {
    ptr: *mut c_void,
    size: usize,
}

impl DeviceRegion {
    fn allocate(size: usize) -> OpResult<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                size: 0,
            });
        }
        let mut ptr = std::ptr::null_mut();
        unsafe {
            cuda_check!(ffi::cudaMalloc(&mut ptr, size));
        }
        Ok(Self { ptr, size })
    }
}

impl Drop for DeviceRegion {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::cudaFree(self.ptr);
            }
        }
    }
}

/// Borrowed view of one address-stable CUDA scratch region. Keeping pointer and
/// capacity together prevents callers from accidentally pairing different
/// regions when crossing an FFI boundary.
#[derive(Debug, Clone, Copy)]
pub struct CudaWorkspace {
    ptr: *mut c_void,
    size: usize,
}

impl CudaWorkspace {
    pub fn ptr(self) -> *mut c_void {
        self.ptr
    }

    pub fn size(self) -> usize {
        self.size
    }
}

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

/// Release all handles from a discarded capture, even if one destroy fails.
/// Always consume the runtime's last error and preserve fatal device faults
/// ahead of recoverable failures. These handles came directly from CUDA.
fn discard_capture_graph(
    graph: ffi::cudaGraph_t,
    exec: ffi::cudaGraphExec_t,
    reported_error: Option<ffi::cudaError_t>,
) -> OpResult<()> {
    // A later successful destroy may replace the runtime's last-error value;
    // retain a fatal original API result independently of that thread-local slot.
    let mut error = reported_error
        .filter(|&code| code != ffi::cudaError_cudaSuccess)
        .map(|code| super::error::classify_capture_error(code, "CUDA graph capture cleanup"))
        .filter(OpError::is_fatal);
    let results = unsafe {
        [
            (
                "destroy discarded graph exec",
                if exec.is_null() {
                    ffi::cudaError_cudaSuccess
                } else {
                    ffi::cudaGraphExecDestroy(exec)
                },
            ),
            (
                "destroy discarded graph",
                if graph.is_null() {
                    ffi::cudaError_cudaSuccess
                } else {
                    ffi::cudaGraphDestroy(graph)
                },
            ),
        ]
    };
    for (context, code) in results {
        if code != ffi::cudaError_cudaSuccess {
            let next = super::error::classify_sync_error(code, context);
            if error.is_none() || next.is_fatal() {
                error = Some(next);
            }
        }
    }
    if let Err(next) = super::error::check_capture_cleanup_error(reported_error)
        && (error.is_none() || next.is_fatal())
    {
        error = Some(next);
    }
    error.map_or(Ok(()), Err)
}

#[derive(Debug)]
struct CudaDeviceRestore {
    previous: i32,
}

impl Drop for CudaDeviceRestore {
    fn drop(&mut self) {
        if self.previous >= 0 {
            unsafe {
                ffi::cudaSetDevice(self.previous);
            }
        }
    }
}

#[derive(Debug)]
pub struct CudaConfig {
    /// Device on which every stream, handle, graph and allocation below was
    /// created. Multi-GPU teardown must reactivate it before destroying them.
    device_id: i32,
    device_info: CudaDeviceInfo,
    pub stream: ffi::cudaStream_t,
    pub cublaslt_handle: ffi::cublasLtHandle_t,
    pub cublas_handle_v2: ffi::cublasHandle_t,
    memory_plan: CudaMemoryPlan,
    kernel_workspace: DeviceRegion,
    pub(crate) bulk_upload: std::sync::Mutex<super::upload::BulkUpload>,
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
    // Captured transient tensors need addresses reserved across graph replays.
    // The ordinary pool can recycle dropped tensors, so capture allocations
    // must stay in this arena. `free` of an arena pointer is a no-op; captures
    // share the backing region and execute serially on the compute stream.
    // Reset the offset before each capture, never while capture is active.
    graph_arena: std::sync::Mutex<Option<DeviceRegion>>,
    arena_base: std::sync::atomic::AtomicPtr<c_void>,
    arena_failed: std::sync::atomic::AtomicBool,
    pub arena_off: std::sync::atomic::AtomicUsize,
    pub arena_enabled: std::sync::atomic::AtomicBool,
    /// Track capture even for callers using the low-level capture API without
    /// an arena. Such captures must never allocate from the recycling pool.
    capture_active: std::sync::atomic::AtomicBool,
    /// A local allocation error need not invalidate CUDA's capture. Remember
    /// it so even a caller that tries to finish cannot publish a partial graph.
    capture_failed: std::sync::atomic::AtomicBool,

    // ─── Recycling scratch allocator (eager forward path) ────────────
    //
    // Outside graph capture, transient scratch tensors are recycled through
    // this size-keyed free list instead of round-tripping `cudaMalloc`/
    // `cudaFree` (the latter device-synchronizes). See `PoolState`.
    pool: std::sync::Mutex<PoolState>,
    /// Drop cannot return an error. A failed CUDA free poisons further pool
    /// operations instead of silently handing out an uncertain allocation.
    pool_error: std::sync::atomic::AtomicU32,
    /// Armed at the beginning of `Drop`, then dropped after every device-owned
    /// field so the caller's previously active CUDA device is restored last.
    restore_device: CudaDeviceRestore,
    #[cfg(test)]
    _test_lease: CudaTestLease,
}

impl CudaConfig {
    pub fn new() -> OpResult<Self> {
        Self::with_memory_plan(CudaMemoryPlan::default())
    }

    pub fn with_memory_plan(memory_plan: CudaMemoryPlan) -> OpResult<Self> {
        #[cfg(test)]
        let test_lease = CudaTestLease::acquire();
        let mut device_id = -1;
        unsafe {
            cuda_check!(ffi::cudaGetDevice(&mut device_id));
        }
        let device_info = CudaDeviceInfo::query(device_id)?;
        let bulk_upload = super::upload::BulkUpload::from_env()?;
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
        let kernel_workspace = DeviceRegion::allocate(memory_plan.kernel_workspace_bytes)?;
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
            device_id,
            device_info,
            stream,
            cublaslt_handle,
            cublas_handle_v2,
            memory_plan,
            kernel_workspace,
            bulk_upload: std::sync::Mutex::new(bulk_upload),
            graphs: std::sync::Mutex::new(HashMap::new()),
            cudnn_handle,
            copy_in_stream,
            copy_out_stream,
            ev_in,
            ev_a,
            ev_out,
            graph_arena: std::sync::Mutex::new(None),
            arena_base: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            arena_failed: std::sync::atomic::AtomicBool::new(false),
            arena_off: std::sync::atomic::AtomicUsize::new(0),
            arena_enabled: std::sync::atomic::AtomicBool::new(false),
            capture_active: std::sync::atomic::AtomicBool::new(false),
            capture_failed: std::sync::atomic::AtomicBool::new(false),
            pool: std::sync::Mutex::new(PoolState::default()),
            pool_error: std::sync::atomic::AtomicU32::new(ffi::cudaError_cudaSuccess),
            restore_device: CudaDeviceRestore { previous: -1 },
            #[cfg(test)]
            _test_lease: test_lease,
        })
    }

    pub fn memory_plan(&self) -> CudaMemoryPlan {
        self.memory_plan
    }

    pub(crate) fn device_info(&self) -> CudaDeviceInfo {
        self.device_info
    }

    pub fn kernel_workspace(&self) -> CudaWorkspace {
        CudaWorkspace {
            ptr: self.kernel_workspace.ptr,
            size: self.kernel_workspace.size,
        }
    }

    // ─── Graph-capture scratch arena ─────────────────────────────────

    /// True while graph capture is configured and its lazy arena allocation has
    /// not failed.
    pub fn arena_available(&self) -> bool {
        self.memory_plan.graph_arena_bytes > 0
            && !self.arena_failed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Lazily allocate the graph arena, reset its bump offset, and route
    /// subsequent `alloc_bytes` through it. Called before stream capture, where
    /// `cudaMalloc` would no longer be legal.
    pub fn arena_begin(&self) -> OpResult<()> {
        use std::sync::atomic::Ordering;
        if self.capture_active.load(Ordering::Acquire) {
            return Err(OpError::Kernel(
                "cannot reset CUDA graph arena during capture".into(),
            ));
        }
        if self.memory_plan.graph_arena_bytes == 0 {
            return Err(OpError::Kernel("CUDA graph arena is disabled".into()));
        }
        if self.arena_base.load(Ordering::Acquire).is_null() {
            let mut arena = self.graph_arena.lock().unwrap();
            if arena.is_none() {
                match DeviceRegion::allocate(self.memory_plan.graph_arena_bytes) {
                    Ok(region) => {
                        self.arena_base.store(region.ptr, Ordering::Release);
                        *arena = Some(region);
                    }
                    Err(error) => {
                        self.arena_failed.store(true, Ordering::Release);
                        return Err(error);
                    }
                }
            }
        }
        self.arena_off.store(0, Ordering::Release);
        self.arena_enabled.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop an eager arena session. Capture keeps its routing until end/abort,
    /// so a nested pipeline cleanup cannot expose the ordinary pool to capture.
    pub fn arena_end(&self) {
        if self
            .capture_active
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return;
        }
        self.arena_enabled
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Serve `size` bytes from the arena. Only eager execution may fall back to
    /// the recycling pool. Capture failures poison this capture until abort/end;
    /// a pooled address must never be baked into a graph and then recycled.
    /// Zero-initializes asynchronously on the compute stream (capture-safe).
    pub fn arena_alloc(&self, size: usize) -> OpResult<Option<*mut c_void>> {
        use std::sync::atomic::Ordering;
        let capturing = self.capture_active.load(Ordering::Acquire);
        let result = self.try_arena_alloc(size, capturing);
        if capturing && result.is_err() {
            self.capture_failed.store(true, Ordering::Release);
        }
        result
    }

    fn try_arena_alloc(&self, size: usize, capturing: bool) -> OpResult<Option<*mut c_void>> {
        use std::sync::atomic::Ordering;
        if !self.arena_enabled.load(Ordering::Acquire) {
            return if capturing {
                Err(OpError::Kernel(
                    "CUDA capture allocation requires an active graph arena".into(),
                ))
            } else {
                Ok(None)
            };
        }
        let base = self.arena_base.load(Ordering::Acquire);
        if base.is_null() {
            return Err(OpError::Kernel(
                "active CUDA graph arena has no storage".into(),
            ));
        }
        let n = size
            .max(1)
            .checked_add(255)
            .map(|n| n & !255usize)
            .ok_or_else(|| OpError::Kernel("CUDA graph arena allocation size overflow".into()))?;
        // Reserve `n` bytes atomically without ever transiently over-committing
        // the bump offset. `fetch_add`+rollback could momentarily publish an
        // offset past the arena end to a concurrent allocator; `fetch_update`
        // only commits when the new offset fits, so a losing race simply retries
        // or returns `None`. (Capture is single-threaded today, so this is
        // belt-and-suspenders, but it makes the invariant local and robust.)
        let off = self
            .arena_off
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let end = cur.checked_add(n)?;
                (end <= self.memory_plan.graph_arena_bytes).then_some(end)
            });
        let off = match off {
            Ok(prev) => prev, // prev offset; our region is [prev, prev+n)
            Err(used) if capturing => {
                return Err(OpError::Kernel(format!(
                    "CUDA graph arena exhausted: requested {n} bytes, used {used} bytes, capacity {} bytes",
                    self.memory_plan.graph_arena_bytes,
                )));
            }
            Err(_) => return Ok(None),
        };
        let ptr = unsafe { (base as *mut u8).add(off) as *mut c_void };
        // Zero-initialize on the compute stream (capture-safe: cudaMemsetAsync
        // is a recordable stream op).
        unsafe {
            cuda_check!(ffi::cudaMemsetAsync(ptr, 0, n, self.stream));
        }
        Ok(Some(ptr))
    }

    /// Whether `ptr` lies inside the arena (its `free` must be a no-op).
    pub fn arena_contains(&self, ptr: *mut c_void) -> bool {
        let base = self.arena_base.load(std::sync::atomic::Ordering::Acquire);
        if base.is_null() {
            return false;
        }
        let p = ptr as usize;
        let b = base as usize;
        p >= b && p < b + self.memory_plan.graph_arena_bytes
    }

    // ─── Recycling scratch allocator ─────────────────────────────────

    /// Eager-pool counters and capacities. Fixed kernel workspaces and graph
    /// arenas are separate allocations and are not included in this snapshot.
    pub fn pool_stats(&self) -> CudaPoolStats {
        self.pool.lock().unwrap().stats()
    }

    fn check_pool_error(&self) -> OpResult<()> {
        let code = self.pool_error.load(std::sync::atomic::Ordering::Acquire);
        if code == ffi::cudaError_cudaSuccess {
            Ok(())
        } else {
            Err(OpError::Fatal(format!(
                "CUDA memory pool cannot be reused after a deallocation failure: {}",
                super::error::CudaError(code),
            )))
        }
    }

    /// Allocate on the compute stream. A reused block may be slightly larger
    /// than `size`; only the requested, 256B-rounded range needs clearing.
    pub(crate) fn pool_alloc(&self, size: usize) -> OpResult<*mut c_void> {
        self.pool_alloc_with(size, Self::pool_malloc, |ptr, n| {
            // All eager scratch uses this compute stream. Prior consumers
            // precede initialization of the next checkout on that stream.
            unsafe { ffi::cudaMemsetAsync(ptr, 0, n, self.stream) }
        })
    }

    // Keep driver calls injectable for small fault-path tests: an OOM must be
    // testable without filling the GPU, and a failed zero must not poison it.
    fn pool_alloc_with(
        &self,
        size: usize,
        mut malloc: impl FnMut(usize) -> Result<*mut c_void, ffi::cudaError_t>,
        initialize: impl FnOnce(*mut c_void, usize) -> ffi::cudaError_t,
    ) -> OpResult<*mut c_void> {
        self.check_pool_error()?;
        let n = pool_allocation_size(size)?;
        let cached = self.pool.lock().unwrap().take(n);
        let block = match cached {
            Some(block) => block,
            None => {
                let ptr = self.pool_cold_alloc(n, &mut malloc)?;
                self.pool.lock().unwrap().record_cold_allocation(n);
                PoolBlock { ptr, capacity: n }
            }
        };
        let code = initialize(block.ptr, n);
        if code != ffi::cudaError_cudaSuccess {
            let error = super::error::allocation_error(code, "CUDA pool initialization");
            let block = self.pool.lock().unwrap().release(block.ptr, n);
            // A failed initialization must neither escape as Tensor::zeros nor
            // leave capacity charged to a live Tensor that was never created.
            if let Err(cleanup) = self.free_pool_block(block)
                && !error.is_fatal()
            {
                return Err(cleanup);
            }
            return Err(error);
        }
        Ok(block.ptr)
    }

    fn pool_malloc(n: usize) -> Result<*mut c_void, ffi::cudaError_t> {
        let mut ptr = std::ptr::null_mut();
        let code = unsafe { ffi::cudaMalloc(&mut ptr, n) };
        if code == ffi::cudaError_cudaSuccess {
            Ok(ptr)
        } else {
            Err(code)
        }
    }

    fn pool_cold_alloc(
        &self,
        n: usize,
        malloc: &mut impl FnMut(usize) -> Result<*mut c_void, ffi::cudaError_t>,
    ) -> OpResult<*mut c_void> {
        self.pool.lock().unwrap().note_malloc_attempt();
        match malloc(n) {
            Ok(ptr) => Ok(ptr),
            Err(code) => {
                let error = super::error::allocation_error(code, "CUDA pool allocation");
                // A different pending device failure takes precedence over an
                // OOM, and cannot be repaired by freeing cached allocations.
                if code != ffi::cudaError_cudaErrorMemoryAllocation || error.is_fatal() {
                    return Err(error);
                }
                if self.trim_pool(0)? == 0 {
                    return Err(error);
                }
                {
                    let mut pool = self.pool.lock().unwrap();
                    pool.note_retry();
                    pool.note_malloc_attempt();
                }
                malloc(n)
                    .map_err(|code| super::error::allocation_error(code, "CUDA pool OOM retry"))
            }
        }
    }

    /// Return an eager allocation using its original request size. Metadata
    /// recovers the physical capacity when a larger block supplied that request.
    pub(crate) fn pool_release(&self, ptr: *mut c_void, size: usize) {
        use std::sync::atomic::Ordering;
        let n = pool_allocation_size(size).expect("free must match a successful allocation");
        let capturing = self.capture_active.load(Ordering::Acquire);
        let poisoned = self.pool_error.load(Ordering::Acquire) != ffi::cudaError_cudaSuccess;
        // Restore capacity and return the block under one lock, just as on the
        // original exact-size fast path. CUDA frees stay outside the lock.
        let evicted = {
            let mut pool = self.pool.lock().unwrap();
            let block = pool.release(ptr, n);
            if capturing || poisoned {
                // An eager pointer may already be recorded as a graph input.
                // Keep it isolated until the recording has been discarded.
                pool.defer(block);
                if capturing {
                    self.capture_failed.store(true, Ordering::Release);
                }
                return;
            }
            pool.retain_or_evict(block, self.memory_plan.pool_retain_bytes)
        };
        if let Err(error) = self.free_pool_blocks(evicted) {
            tracing::error!(?error, "CUDA pool release failed");
        }
    }

    fn retain_pool_block(&self, block: PoolBlock) -> OpResult<()> {
        let evicted = self
            .pool
            .lock()
            .unwrap()
            .retain_or_evict(block, self.memory_plan.pool_retain_bytes);
        self.free_pool_blocks(evicted).map(|_| ())
    }

    fn free_pool_block(&self, block: PoolBlock) -> OpResult<()> {
        // No driver calls while holding the pool mutex. Only complete
        // cudaMalloc allocations are released; the pool never splits blocks.
        let code = unsafe { ffi::cudaFree(block.ptr) };
        if code == ffi::cudaError_cudaSuccess {
            self.pool.lock().unwrap().record_free(block.capacity);
            Ok(())
        } else {
            self.pool_error
                .store(code, std::sync::atomic::Ordering::Release);
            // CUDA may have reported an asynchronous error. Quarantine this
            // address rather than guessing whether the free actually occurred.
            self.pool.lock().unwrap().defer(block);
            self.check_pool_error()
        }
    }

    fn flush_deferred_pool(&self) -> OpResult<()> {
        self.check_pool_error()?;
        let blocks = self.pool.lock().unwrap().take_deferred();
        let mut error = None;
        for block in blocks {
            if let Err(next) = self.check_pool_error() {
                self.pool.lock().unwrap().defer(block);
                error.get_or_insert(next);
            } else if let Err(next) = self.retain_pool_block(block) {
                error = Some(next);
            }
        }
        error.map_or(Ok(()), Err)
    }

    /// Release idle blocks until cached capacity is at most `target_bytes`.
    /// Returns bytes actually released. Live tensors, fixed workspaces and
    /// graph arenas are untouched. This can synchronize CUDA and must be used
    /// outside capture, with the owning device active.
    pub fn trim_pool(&self, target_bytes: usize) -> OpResult<usize> {
        if self
            .capture_active
            .load(std::sync::atomic::Ordering::Acquire)
        {
            return Err(OpError::Kernel(
                "cannot trim CUDA memory pool during capture".into(),
            ));
        }
        self.check_pool_error()?;
        let blocks = self.pool.lock().unwrap().drain_to(target_bytes);
        self.free_pool_blocks(blocks)
    }

    fn free_pool_blocks(&self, blocks: Vec<PoolBlock>) -> OpResult<usize> {
        let mut released = 0;
        let mut error = None;
        for block in blocks {
            if let Err(next) = self.check_pool_error() {
                self.pool.lock().unwrap().defer(block);
                error.get_or_insert(next);
            } else {
                match self.free_pool_block(block) {
                    Ok(()) => released += block.capacity,
                    Err(next) => error = Some(next),
                }
            }
        }
        error.map_or(Ok(released), Err)
    }

    /// Release all idle eager blocks; live allocations remain owned by tensors.
    pub fn pool_drain(&self) -> OpResult<usize> {
        self.trim_pool(0)
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
        use std::sync::atomic::Ordering;
        self.check_pool_error()?;
        if self
            .capture_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(OpError::Kernel(
                "CUDA graph capture is already active".into(),
            ));
        }
        self.capture_failed.store(false, Ordering::Release);
        // Mode 2 = cudaStreamCaptureModeRelaxed. Relaxed (not ThreadLocal=1)
        // is required so the potentially-unsafe API calls that cuBLASLt / cuDNN
        // make internally while enqueuing a matmul/attention are tolerated
        // during capture instead of returning an error (e.g. cuBLASLt
        // EXECUTION_FAILED / status=13 under ThreadLocal capture).
        let code = unsafe { ffi::cudaStreamBeginCapture(self.stream, 2) };
        if code != ffi::cudaError_cudaSuccess {
            self.capture_active.store(false, Ordering::Release);
            self.arena_end();
            discard_capture_graph(std::ptr::null_mut(), std::ptr::null_mut(), Some(code))?;
            return Err(super::error::classify_capture_error(
                code,
                "cudaStreamBeginCapture",
            ));
        }
        Ok(())
    }

    /// Discard the in-progress capture, including one CUDA has invalidated.
    /// EndCapture is required to restore the stream even in that case. Never
    /// instantiate or publish the discarded graph, and leave existing slots
    /// intact so a failed replacement does not destroy a previously valid graph.
    pub fn capture_abort(&self) -> OpResult<()> {
        use std::sync::atomic::Ordering;
        if !self.capture_active.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut graph = std::ptr::null_mut();
        let code = unsafe { ffi::cudaStreamEndCapture(self.stream, &mut graph) };
        self.capture_active.store(false, Ordering::Release);
        self.capture_failed.store(false, Ordering::Release);
        self.arena_end();
        discard_capture_graph(graph, std::ptr::null_mut(), Some(code))?;
        // CUDA reports invalidation while successfully ending that capture.
        if code != ffi::cudaError_cudaSuccess
            && code != ffi::cudaError_cudaErrorStreamCaptureInvalidated
        {
            return Err(super::error::classify_capture_error(
                code,
                "cudaStreamEndCapture while aborting",
            ));
        }
        self.flush_deferred_pool()
    }

    pub fn capture_end(&self, slot: GraphSlot) -> OpResult<()> {
        use std::sync::atomic::Ordering;
        if !self.capture_active.load(Ordering::Acquire) {
            return Err(OpError::Kernel("no CUDA graph capture is active".into()));
        }
        if self.capture_failed.load(Ordering::Acquire) {
            self.capture_abort()?;
            return Err(OpError::Kernel(
                "CUDA graph capture discarded after an allocation or ownership failure".into(),
            ));
        }
        let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
        let code = unsafe { ffi::cudaStreamEndCapture(self.stream, &mut graph) };
        self.capture_active.store(false, Ordering::Release);
        self.capture_failed.store(false, Ordering::Release);
        self.arena_end();
        if code != ffi::cudaError_cudaSuccess {
            discard_capture_graph(graph, std::ptr::null_mut(), Some(code))?;
            return Err(super::error::classify_capture_error(
                code,
                "cudaStreamEndCapture",
            ));
        }
        if graph.is_null() {
            return Err(OpError::Kernel(
                "cudaStreamEndCapture returned a null graph".into(),
            ));
        }
        let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
        let code = unsafe { ffi::cudaGraphInstantiate(&mut exec, graph, 0) };
        if code != ffi::cudaError_cudaSuccess {
            discard_capture_graph(graph, exec, Some(code))?;
            return Err(super::error::classify_capture_error(
                code,
                "cudaGraphInstantiate",
            ));
        }
        self.graphs
            .lock()
            .unwrap()
            .insert(slot, CudaGraph { graph, exec });
        Ok(())
    }

    pub fn launch(&self, slot: GraphSlot) -> OpResult<()> {
        self.check_pool_error()?;
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
        self.check_pool_error()?;
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
            // Fields such as graphs and DeviceRegion are dropped after this
            // body. Arm the final restore field, then leave the owning device
            // active until all of those device-local resources are gone.
            let mut previous = -1;
            if ffi::cudaGetDevice(&mut previous) == ffi::cudaError_cudaSuccess {
                self.restore_device.previous = previous;
            }
            let set_device = ffi::cudaSetDevice(self.device_id);
            if set_device != ffi::cudaError_cudaSuccess {
                tracing::error!(
                    device = self.device_id,
                    error = ?set_device,
                    "cudaSetDevice during CudaConfig teardown failed"
                );
            }
            if let Err(error) = self.capture_abort() {
                tracing::error!(?error, "abort CUDA capture during teardown failed");
            }
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
            if !self.cudnn_handle.is_null() {
                ffi::cudnnDestroy(self.cudnn_handle);
            }
        }
        // Release every recycled scratch block (disjoint from the arena).
        if let Err(error) = self.pool_drain() {
            tracing::error!(?error, "drain CUDA pool during teardown failed");
        }
    }
}
unsafe impl Send for CudaConfig {}
unsafe impl Sync for CudaConfig {}

#[cfg(test)]
mod memory_plan_tests {
    use super::*;

    #[test]
    fn pool_alignment_checks_overflow() {
        assert_eq!(pool_allocation_size(0).unwrap(), 256);
        assert_eq!(pool_allocation_size(255).unwrap(), 256);
        assert_eq!(pool_allocation_size(256).unwrap(), 256);
        assert_eq!(pool_allocation_size(257).unwrap(), 512);
        assert_eq!(
            pool_allocation_size(usize::MAX - 255).unwrap(),
            usize::MAX - 255
        );
        assert!(pool_allocation_size(usize::MAX).is_err());
    }

    #[test]
    fn default_plan_uses_compact_fixed_regions() {
        let plan = CudaMemoryPlan::default();
        assert_eq!(plan.kernel_workspace_bytes, 256 * MIB);
        assert_eq!(plan.graph_arena_bytes, 256 * MIB);
        assert_eq!(plan.pool_retain_bytes, 256 * MIB);
    }

    #[test]
    fn builder_changes_regions_independently() {
        let plan = CudaMemoryPlan::default()
            .with_kernel_workspace_bytes(64)
            .with_graph_arena_bytes(128)
            .with_pool_retain_bytes(32);
        assert_eq!(
            plan,
            CudaMemoryPlan {
                kernel_workspace_bytes: 64,
                graph_arena_bytes: 128,
                pool_retain_bytes: 32,
            }
        );
    }
}

#[cfg(test)]
mod pool_failure_tests {
    use super::*;

    struct Allocation<'a> {
        config: &'a CudaConfig,
        ptr: *mut c_void,
        requested_bytes: usize,
    }

    impl Drop for Allocation<'_> {
        fn drop(&mut self) {
            self.config.pool_release(self.ptr, self.requested_bytes);
        }
    }

    fn config() -> CudaConfig {
        let code = unsafe { ffi::cudaSetDevice(0) };
        assert_eq!(code, ffi::cudaError_cudaSuccess, "select logical GPU 0");
        CudaConfig::with_memory_plan(CudaMemoryPlan {
            kernel_workspace_bytes: 0,
            graph_arena_bytes: 0,
            pool_retain_bytes: MIB,
        })
        .expect("create CUDA config for injected pool failures")
    }

    fn allocation(config: &CudaConfig, bytes: usize) -> Allocation<'_> {
        Allocation {
            config,
            ptr: config.pool_alloc(bytes).expect("allocate small CUDA block"),
            requested_bytes: bytes,
        }
    }

    fn warm(config: &CudaConfig, bytes: usize) -> *mut c_void {
        let block = allocation(config, bytes);
        let code = unsafe { ffi::cudaMemsetAsync(block.ptr, 71, bytes, config.stream) };
        assert_eq!(code, ffi::cudaError_cudaSuccess, "fill cached allocation");
        config.synchronize().expect("finish pool warmup");
        let ptr = block.ptr;
        drop(block);
        ptr
    }

    fn assert_zeroed(block: &Allocation<'_>) {
        let mut host = vec![255u8; block.requested_bytes];
        // SAFETY: the allocation owns the requested byte range, and the host
        // destination remains alive until the stream synchronization below.
        let code = unsafe {
            ffi::cudaMemcpyAsync(
                host.as_mut_ptr().cast(),
                block.ptr,
                block.requested_bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                block.config.stream,
            )
        };
        assert_eq!(
            code,
            ffi::cudaError_cudaSuccess,
            "download initialized block"
        );
        block
            .config
            .synchronize()
            .expect("finish initialization check");
        assert!(host.iter().all(|&byte| byte == 0));
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn oom_reclaims_idle_capacity_and_retries_once_successfully() {
        let config = config();
        warm(&config, 1024);
        let before = config.pool_stats();
        let mut calls = 0;
        let ptr = config
            .pool_alloc_with(
                2048,
                |n| {
                    assert_eq!(n, 2048);
                    calls += 1;
                    if calls == 1 {
                        Err(ffi::cudaError_cudaErrorMemoryAllocation)
                    } else {
                        CudaConfig::pool_malloc(n)
                    }
                },
                |ptr, n| unsafe { ffi::cudaMemsetAsync(ptr, 0, n, config.stream) },
            )
            .expect("retry succeeds after injected OOM");
        let block = Allocation {
            config: &config,
            ptr,
            requested_bytes: 2048,
        };
        assert_eq!(calls, 2);
        assert_zeroed(&block);
        let after = config.pool_stats();
        assert_eq!(after.allocation_requests, before.allocation_requests + 1);
        assert_eq!(after.cuda_malloc_calls, before.cuda_malloc_calls + 2);
        assert_eq!(after.cold_allocations, before.cold_allocations + 1);
        assert_eq!(after.allocation_retries, before.allocation_retries + 1);
        assert_eq!(after.cuda_frees, before.cuda_frees + 1);
        assert_eq!(after.cache_hits, before.cache_hits);
        assert_eq!(after.live_bytes, 2048);
        assert_eq!(after.pooled_bytes, 0);
        assert_eq!(after.reserved_bytes, 2048);
        assert_eq!(after.pending_bytes, 0);
        drop(block);
        assert_eq!(config.trim_pool(0).unwrap(), 2048);
        assert_eq!(config.pool_stats().reserved_bytes, 0);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn persistent_oom_stops_after_one_retry_and_never_initializes() {
        let config = config();
        warm(&config, 1024);
        let before = config.pool_stats();
        let mut calls = 0;
        let result = config.pool_alloc_with(
            2048,
            |n| {
                assert_eq!(n, 2048);
                calls += 1;
                Err(ffi::cudaError_cudaErrorMemoryAllocation)
            },
            |_, _| panic!("failed allocation must not initialize"),
        );
        let error = result.expect_err("both injected allocation attempts fail");
        assert!(
            !error.is_fatal(),
            "injected OOM remains recoverable: {error}"
        );
        assert_eq!(calls, 2);
        let after = config.pool_stats();
        assert_eq!(after.allocation_requests, before.allocation_requests + 1);
        assert_eq!(after.cuda_malloc_calls, before.cuda_malloc_calls + 2);
        assert_eq!(after.cold_allocations, before.cold_allocations);
        assert_eq!(after.allocation_retries, before.allocation_retries + 1);
        assert_eq!(after.cuda_frees, before.cuda_frees + 1);
        assert_eq!(after.live_bytes, 0);
        assert_eq!(after.pooled_bytes, 0);
        assert_eq!(after.reserved_bytes, 0);
        assert_eq!(after.pending_bytes, 0);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn oom_without_idle_capacity_does_not_retry_or_initialize() {
        let config = config();
        let before = config.pool_stats();
        let mut calls = 0;
        let result = config.pool_alloc_with(
            2048,
            |_| {
                calls += 1;
                Err(ffi::cudaError_cudaErrorMemoryAllocation)
            },
            |_, _| panic!("failed allocation must not initialize"),
        );
        assert!(result.is_err());
        assert_eq!(calls, 1);
        let after = config.pool_stats();
        assert_eq!(after.cuda_malloc_calls, before.cuda_malloc_calls + 1);
        assert_eq!(after.cold_allocations, before.cold_allocations);
        assert_eq!(after.allocation_retries, before.allocation_retries);
        assert_eq!(after.cuda_frees, before.cuda_frees);
        assert_eq!(after.live_bytes, 0);
        assert_eq!(after.pooled_bytes, 0);
        assert_eq!(after.reserved_bytes, 0);
        assert_eq!(after.pending_bytes, 0);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn non_oom_allocation_error_preserves_idle_blocks_without_retry() {
        let config = config();
        let address = warm(&config, 1024);
        let before = config.pool_stats();
        let mut calls = 0;
        let result = config.pool_alloc_with(
            2048,
            |_| {
                calls += 1;
                Err(ffi::cudaError_cudaErrorInvalidValue)
            },
            |_, _| panic!("failed allocation must not initialize"),
        );
        let error = result.expect_err("injected invalid-value allocation failure");
        assert!(!error.is_fatal(), "synchronous API failure: {error}");
        assert_eq!(calls, 1);
        let after = config.pool_stats();
        assert_eq!(after.cuda_malloc_calls, before.cuda_malloc_calls + 1);
        assert_eq!(after.cold_allocations, before.cold_allocations);
        assert_eq!(after.allocation_retries, before.allocation_retries);
        assert_eq!(after.cuda_frees, before.cuda_frees);
        assert_eq!(after.live_bytes, 0);
        assert_eq!(after.pooled_bytes, 1024);
        assert_eq!(after.reserved_bytes, 1024);
        assert_eq!(after.pending_bytes, 0);
        let recovered = allocation(&config, 1024);
        assert_eq!(recovered.ptr, address);
        assert_zeroed(&recovered);
        drop(recovered);
        assert_eq!(config.trim_pool(0).unwrap(), 1024);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn failed_initialization_releases_full_reused_capacity_and_recovers() {
        let config = config();
        let address = warm(&config, 8192);
        let before = config.pool_stats();
        let result = config.pool_alloc_with(
            7680,
            |_| panic!("larger cached block must avoid malloc"),
            |ptr, n| {
                assert_eq!(ptr, address);
                assert_eq!(n, 7680);
                ffi::cudaError_cudaErrorInvalidValue
            },
        );
        let error = result.expect_err("injected initialization failure");
        assert!(
            !error.is_fatal(),
            "synchronous initialization failure: {error}"
        );
        let after = config.pool_stats();
        assert_eq!(after.allocation_requests, before.allocation_requests + 1);
        assert_eq!(after.cache_hits, before.cache_hits + 1);
        assert_eq!(after.larger_reuses, before.larger_reuses + 1);
        assert_eq!(after.cuda_malloc_calls, before.cuda_malloc_calls);
        assert_eq!(after.cold_allocations, before.cold_allocations);
        assert_eq!(after.allocation_retries, before.allocation_retries);
        assert_eq!(after.cuda_frees, before.cuda_frees + 1);
        assert_eq!(after.live_bytes, 0);
        assert_eq!(after.pooled_bytes, 0);
        assert_eq!(
            after.reserved_bytes, 0,
            "the complete 8192-byte block was freed"
        );
        assert_eq!(after.pending_bytes, 0);
        let recovered = allocation(&config, 8192);
        assert_zeroed(&recovered);
        let recovered_stats = config.pool_stats();
        assert_eq!(recovered_stats.live_bytes, 8192);
        assert_eq!(recovered_stats.reserved_bytes, 8192);
        assert_eq!(
            recovered_stats.cuda_malloc_calls,
            before.cuda_malloc_calls + 1
        );
        assert_eq!(
            recovered_stats.cold_allocations,
            before.cold_allocations + 1
        );
        drop(recovered);
        assert_eq!(config.pool_stats().live_bytes, 0);
        assert_eq!(config.trim_pool(0).unwrap(), 8192);
        assert_eq!(config.pool_stats().reserved_bytes, 0);
    }
}
