//! FlashInfer Rust Bindings
//!
//! This module provides Rust bindings to FlashInfer's BatchDecode kernels
//! for efficient paged attention in LLM inference.
//!
//! ## Architecture
//!
//! FlashInfer uses a Plan-Run pattern:
//! 1. **Plan**: Analyze the batch and compute execution metadata (workspace offsets, grid sizes)
//! 2. **Run**: Execute the actual attention computation using the plan

use crate::base::error::{Error, Result};
use crate::cuda::error::CudaError;
use crate::cuda::ffi::cudaStream_t;
use std::ffi::c_void;
use std::ptr;

// ============================================================================
// FFI Declarations
// ============================================================================

#[repr(C)]
pub struct FlashInferContext {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn flashinfer_batch_decode_create() -> *mut FlashInferContext;
    fn flashinfer_batch_decode_destroy(ctx: *mut FlashInferContext);

    fn flashinfer_batch_decode_plan(
        ctx: *mut FlashInferContext,
        float_workspace: *mut c_void,
        float_ws_size: usize,
        int_workspace: *mut c_void,
        int_ws_size: usize,
        pinned_workspace: *mut c_void,
        pinned_ws_size: usize,
        indptr_h: *const i32,
        batch_size: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
        enable_cuda_graph: bool,
        stream: cudaStream_t,
    ) -> i32;

    fn flashinfer_batch_decode_run_bf16(
        ctx: *mut FlashInferContext,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        o: *mut c_void,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        last_page_len: *const i32,
        batch_size: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn flashinfer_batch_decode_run_fp16(
        ctx: *mut FlashInferContext,
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        o: *mut c_void,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        last_page_len: *const i32,
        batch_size: u32,
        stream: cudaStream_t,
    ) -> i32;

    fn flashinfer_get_workspace_sizes(
        max_batch_size: u32,
        max_seq_len: u32,
        num_qo_heads: u32,
        head_dim: u32,
        float_workspace_size: *mut usize,
        int_workspace_size: *mut usize,
        pinned_workspace_size: *mut usize,
    );

    fn flashinfer_is_plan_valid(ctx: *mut FlashInferContext) -> bool;

    fn flashinfer_get_plan_info(
        ctx: *mut FlashInferContext,
        padded_batch_size: *mut i64,
        split_kv: *mut bool,
        enable_cuda_graph: *mut bool,
    );
}

// ============================================================================
// Workspace Configuration
// ============================================================================

/// Recommended workspace sizes for FlashInfer
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceSizes {
    /// Size of float workspace (device memory) in bytes
    pub float_workspace: usize,
    /// Size of int workspace (device memory) in bytes
    pub int_workspace: usize,
    /// Size of pinned workspace (host memory) in bytes
    pub pinned_workspace: usize,
}

impl WorkspaceSizes {
    /// Calculate recommended workspace sizes for given configuration
    pub fn calculate(
        max_batch_size: u32,
        max_seq_len: u32,
        num_qo_heads: u32,
        head_dim: u32,
    ) -> Self {
        let mut float_ws = 0usize;
        let mut int_ws = 0usize;
        let mut pinned_ws = 0usize;

        unsafe {
            flashinfer_get_workspace_sizes(
                max_batch_size,
                max_seq_len,
                num_qo_heads,
                head_dim,
                &mut float_ws,
                &mut int_ws,
                &mut pinned_ws,
            );
        }

        Self {
            float_workspace: float_ws,
            int_workspace: int_ws,
            pinned_workspace: pinned_ws,
        }
    }

    /// Default workspace sizes for typical LLM configurations
    /// (batch_size=64, seq_len=4096, 32 heads, head_dim=128)
    pub fn default_llm() -> Self {
        Self::calculate(64, 4096, 32, 128)
    }

    /// Total device memory required
    pub fn total_device_memory(&self) -> usize {
        self.float_workspace + self.int_workspace
    }
}

// ============================================================================
// Plan Information
// ============================================================================

/// Information about the current execution plan
#[derive(Debug, Clone, Copy)]
pub struct PlanInfo {
    /// Padded batch size used for kernel launch
    pub padded_batch_size: i64,
    /// Whether KV cache is split across multiple kernel launches
    pub split_kv: bool,
    /// Whether CUDA graph compatibility is enabled
    pub enable_cuda_graph: bool,
}

// ============================================================================
// Batch Decode Handler
// ============================================================================

/// Handler for FlashInfer batch decode operations with paged KV cache
///
/// This struct manages the lifecycle of FlashInfer's batch decode:
/// - Workspace memory allocation
/// - Plan computation
/// - Kernel execution
pub struct BatchDecodeHandler {
    ctx: *mut FlashInferContext,
    // Workspace buffers (device memory)
    float_workspace: *mut c_void,
    float_workspace_size: usize,
    int_workspace: *mut c_void,
    int_workspace_size: usize,
    // Pinned memory buffer (host memory)
    pinned_workspace: *mut c_void,
    pinned_workspace_size: usize,
    // Track ownership
    owns_workspaces: bool,
}

// Safety: The context and workspaces can be sent between threads
unsafe impl Send for BatchDecodeHandler {}

impl BatchDecodeHandler {
    /// Create a new batch decode handler with default workspace sizes
    pub fn new() -> Result<Self> {
        Self::with_workspace_sizes(WorkspaceSizes::default_llm())
    }

    /// Create a new batch decode handler with specified workspace sizes
    pub fn with_workspace_sizes(sizes: WorkspaceSizes) -> Result<Self> {
        let ctx = unsafe { flashinfer_batch_decode_create() };
        if ctx.is_null() {
            return Err(Error::CudaError(CudaError(crate::cuda::ffi::cudaError_cudaErrorMemoryAllocation)).into());
        }

        // Allocate device memory for workspaces
        let mut float_workspace: *mut c_void = ptr::null_mut();
        let mut int_workspace: *mut c_void = ptr::null_mut();

        unsafe {
            let status = crate::cuda::ffi::cudaMalloc(&mut float_workspace, sizes.float_workspace);
            if status != crate::cuda::ffi::cudaError_cudaSuccess {
                flashinfer_batch_decode_destroy(ctx);
                return Err(Error::CudaError(CudaError(status)).into());
            }

            let status = crate::cuda::ffi::cudaMalloc(&mut int_workspace, sizes.int_workspace);
            if status != crate::cuda::ffi::cudaError_cudaSuccess {
                crate::cuda::ffi::cudaFree(float_workspace);
                flashinfer_batch_decode_destroy(ctx);
                return Err(Error::CudaError(CudaError(status)).into());
            }
        }

        // Allocate pinned memory
        let mut pinned_workspace: *mut c_void = ptr::null_mut();
        unsafe {
            let status =
                crate::cuda::ffi::cudaMallocHost(&mut pinned_workspace, sizes.pinned_workspace);
            if status != crate::cuda::ffi::cudaError_cudaSuccess {
                crate::cuda::ffi::cudaFree(float_workspace);
                crate::cuda::ffi::cudaFree(int_workspace);
                flashinfer_batch_decode_destroy(ctx);
                return Err(Error::CudaError(CudaError(status)).into());
            }
        }

        Ok(Self {
            ctx,
            float_workspace,
            float_workspace_size: sizes.float_workspace,
            int_workspace,
            int_workspace_size: sizes.int_workspace,
            pinned_workspace,
            pinned_workspace_size: sizes.pinned_workspace,
            owns_workspaces: true,
        })
    }

    /// Create a handler with externally managed workspace buffers
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All pointers are valid and point to allocated memory
    /// - Device pointers (float_workspace, int_workspace) point to device memory
    /// - pinned_workspace points to page-locked host memory
    /// - The memory remains valid for the lifetime of this handler
    pub unsafe fn with_external_workspaces(
        float_workspace: *mut c_void,
        float_workspace_size: usize,
        int_workspace: *mut c_void,
        int_workspace_size: usize,
        pinned_workspace: *mut c_void,
        pinned_workspace_size: usize,
    ) -> Result<Self> {
        let ctx = flashinfer_batch_decode_create();
        if ctx.is_null() {
            return Err(Error::CudaError(CudaError(crate::cuda::ffi::cudaError_cudaErrorMemoryAllocation)).into());
        }

        Ok(Self {
            ctx,
            float_workspace,
            float_workspace_size,
            int_workspace,
            int_workspace_size,
            pinned_workspace,
            pinned_workspace_size,
            owns_workspaces: false,
        })
    }

    /// Plan the batch decode execution
    ///
    /// This must be called before `run_*` methods. The plan can be reused
    /// for multiple runs as long as the batch composition (indptr) doesn't change.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &mut self,
        indptr_host: &[i32],
        batch_size: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        page_size: u32,
        head_dim: u32,
        enable_cuda_graph: bool,
        stream: cudaStream_t,
    ) -> Result<()> {
        if indptr_host.len() != (batch_size + 1) as usize {
            return Err(Error::InvalidArgument(format!(
                "indptr_host length {} doesn't match batch_size + 1 = {}",
                indptr_host.len(),
                batch_size + 1
            )).into());
        }

        let status = unsafe {
            flashinfer_batch_decode_plan(
                self.ctx,
                self.float_workspace,
                self.float_workspace_size,
                self.int_workspace,
                self.int_workspace_size,
                self.pinned_workspace,
                self.pinned_workspace_size,
                indptr_host.as_ptr(),
                batch_size,
                num_qo_heads,
                num_kv_heads,
                page_size,
                head_dim,
                enable_cuda_graph,
                stream,
            )
        };

        if status != 0 {
            return Err(Error::CudaError(CudaError(status as crate::cuda::ffi::cudaError_t)).into());
        }

        Ok(())
    }

    /// Execute batch decode with BF16 tensors
    ///
    /// # Safety
    ///
    /// All pointers must be valid device pointers with correct sizes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_bf16(
        &self,
        q: *const half::bf16,
        k_cache: *const half::bf16,
        v_cache: *const half::bf16,
        o: *mut half::bf16,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        last_page_len: *const i32,
        batch_size: u32,
        stream: cudaStream_t,
    ) -> Result<()> {
        let status = flashinfer_batch_decode_run_bf16(
            self.ctx,
            q as *const c_void,
            k_cache as *const c_void,
            v_cache as *const c_void,
            o as *mut c_void,
            kv_indices,
            kv_indptr,
            last_page_len,
            batch_size,
            stream,
        );

        if status != 0 {
            return Err(Error::CudaError(CudaError(status as crate::cuda::ffi::cudaError_t)).into());
        }

        Ok(())
    }

    /// Execute batch decode with FP16 tensors
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_fp16(
        &self,
        q: *const half::f16,
        k_cache: *const half::f16,
        v_cache: *const half::f16,
        o: *mut half::f16,
        kv_indices: *const i32,
        kv_indptr: *const i32,
        last_page_len: *const i32,
        batch_size: u32,
        stream: cudaStream_t,
    ) -> Result<()> {
        let status = flashinfer_batch_decode_run_fp16(
            self.ctx,
            q as *const c_void,
            k_cache as *const c_void,
            v_cache as *const c_void,
            o as *mut c_void,
            kv_indices,
            kv_indptr,
            last_page_len,
            batch_size,
            stream,
        );

        if status != 0 {
            return Err(Error::CudaError(CudaError(status as crate::cuda::ffi::cudaError_t)).into());
        }

        Ok(())
    }

    /// Check if the current plan is valid
    pub fn is_plan_valid(&self) -> bool {
        unsafe { flashinfer_is_plan_valid(self.ctx) }
    }

    /// Get information about the current execution plan
    pub fn get_plan_info(&self) -> Option<PlanInfo> {
        if !self.is_plan_valid() {
            return None;
        }

        let mut padded_batch_size = 0i64;
        let mut split_kv = false;
        let mut enable_cuda_graph = false;

        unsafe {
            flashinfer_get_plan_info(
                self.ctx,
                &mut padded_batch_size,
                &mut split_kv,
                &mut enable_cuda_graph,
            );
        }

        Some(PlanInfo {
            padded_batch_size,
            split_kv,
            enable_cuda_graph,
        })
    }
}

impl Drop for BatchDecodeHandler {
    fn drop(&mut self) {
        unsafe {
            flashinfer_batch_decode_destroy(self.ctx);

            if self.owns_workspaces {
                if !self.float_workspace.is_null() {
                    crate::cuda::ffi::cudaFree(self.float_workspace);
                }
                if !self.int_workspace.is_null() {
                    crate::cuda::ffi::cudaFree(self.int_workspace);
                }
                if !self.pinned_workspace.is_null() {
                    crate::cuda::ffi::cudaFreeHost(self.pinned_workspace);
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Workspace size tests (pure computation, no GPU needed)
    // ========================================================================

    #[test]
    fn test_workspace_sizes_calculation() {
        let sizes = WorkspaceSizes::calculate(32, 2048, 32, 128);

        assert!(sizes.float_workspace > 0, "Float workspace should be > 0");
        assert!(sizes.int_workspace > 0, "Int workspace should be > 0");
        assert!(sizes.pinned_workspace > 0, "Pinned workspace should be > 0");

        assert_eq!(
            sizes.total_device_memory(),
            sizes.float_workspace + sizes.int_workspace
        );

        println!("Workspace sizes for batch=32, seq=2048, heads=32, dim=128:");
        println!("  Float: {:.2} MB, Int: {:.2} MB, Pinned: {:.2} MB",
            sizes.float_workspace as f64 / 1024.0 / 1024.0,
            sizes.int_workspace as f64 / 1024.0 / 1024.0,
            sizes.pinned_workspace as f64 / 1024.0 / 1024.0);
    }

    #[test]
    fn test_default_workspace_sizes() {
        let sizes = WorkspaceSizes::default_llm();
        assert!(sizes.float_workspace >= 1024 * 1024, "Float workspace should be at least 1MB");
        assert!(sizes.int_workspace >= 1024, "Int workspace should be at least 1KB");
    }

    #[test]
    fn test_workspace_sizes_scaling() {
        let small = WorkspaceSizes::calculate(8, 1024, 32, 128);
        let large = WorkspaceSizes::calculate(64, 4096, 32, 128);

        assert!(large.float_workspace >= small.float_workspace);
        assert!(large.int_workspace >= small.int_workspace);
    }

    // ========================================================================
    // GPU tests: handler lifecycle, plan, and run
    // ========================================================================

    /// Helper: allocate device memory and copy host data to it.
    /// Returns device pointer. Caller must cudaFree.
    unsafe fn to_device<T>(host: &[T]) -> *mut c_void {
        let bytes = host.len() * std::mem::size_of::<T>();
        let mut d_ptr: *mut c_void = ptr::null_mut();
        let status = unsafe { crate::cuda::ffi::cudaMalloc(&mut d_ptr, bytes) };
        assert_eq!(status, crate::cuda::ffi::cudaError_cudaSuccess, "cudaMalloc failed");
        let status = unsafe {
            crate::cuda::ffi::cudaMemcpy(
                d_ptr,
                host.as_ptr() as *const c_void,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        assert_eq!(status, crate::cuda::ffi::cudaError_cudaSuccess, "cudaMemcpy H2D failed");
        d_ptr
    }

    /// Helper: copy device data back to a host Vec<T>.
    unsafe fn to_host<T: Clone + Default>(d_ptr: *const c_void, count: usize) -> Vec<T> {
        let bytes = count * std::mem::size_of::<T>();
        let mut host = vec![T::default(); count];
        let status = unsafe {
            crate::cuda::ffi::cudaMemcpy(
                host.as_mut_ptr() as *mut c_void,
                d_ptr,
                bytes,
                crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        assert_eq!(status, crate::cuda::ffi::cudaError_cudaSuccess, "cudaMemcpy D2H failed");
        host
    }

    #[test]
    fn test_handler_create_and_drop() {
        let handler = BatchDecodeHandler::new();
        assert!(handler.is_ok(), "Handler creation should succeed on GPU");
        // Drop frees CUDA resources via RAII
    }

    #[test]
    fn test_handler_with_custom_sizes() {
        let sizes = WorkspaceSizes::calculate(16, 2048, 32, 128);
        let handler = BatchDecodeHandler::with_workspace_sizes(sizes);
        assert!(handler.is_ok());
    }

    #[test]
    fn test_plan_lifecycle() -> Result<()> {
        let mut handler = BatchDecodeHandler::new()?;

        // Before plan: invalid
        assert!(!handler.is_plan_valid());
        assert!(handler.get_plan_info().is_none());

        // Plan: batch_size=2, each request has 1 page
        let indptr = vec![0i32, 1, 2];
        let stream: cudaStream_t = ptr::null_mut();
        handler.plan(&indptr, 2, 32, 8, 16, 128, false, stream)?;

        // After plan: valid
        assert!(handler.is_plan_valid());
        let info = handler.get_plan_info().unwrap();
        assert_eq!(info.padded_batch_size, 2);
        assert!(!info.split_kv);
        assert!(!info.enable_cuda_graph);

        Ok(())
    }

    #[test]
    fn test_plan_with_cuda_graph_flag() -> Result<()> {
        let mut handler = BatchDecodeHandler::new()?;
        let indptr = vec![0i32, 2, 5];
        let stream: cudaStream_t = ptr::null_mut();
        handler.plan(&indptr, 2, 32, 8, 16, 128, true, stream)?;

        let info = handler.get_plan_info().unwrap();
        assert!(info.enable_cuda_graph);
        assert_eq!(info.padded_batch_size, 2);
        Ok(())
    }

    #[test]
    fn test_plan_invalid_indptr_length() {
        let mut handler = BatchDecodeHandler::new().unwrap();
        let indptr = vec![0i32, 1]; // length 2, but batch_size=2 expects length 3
        let stream: cudaStream_t = ptr::null_mut();

        let result = handler.plan(&indptr, 2, 32, 8, 16, 128, false, stream);
        assert!(result.is_err(), "Should reject mismatched indptr length");
    }

    #[test]
    fn test_plan_replan() -> Result<()> {
        let mut handler = BatchDecodeHandler::new()?;
        let stream: cudaStream_t = ptr::null_mut();

        // First plan: batch_size=1
        handler.plan(&[0i32, 3], 1, 32, 8, 16, 128, false, stream)?;
        let info1 = handler.get_plan_info().unwrap();
        assert_eq!(info1.padded_batch_size, 1);

        // Re-plan: batch_size=4
        handler.plan(&[0i32, 1, 2, 3, 4], 4, 32, 8, 16, 128, false, stream)?;
        let info2 = handler.get_plan_info().unwrap();
        assert_eq!(info2.padded_batch_size, 4);

        Ok(())
    }

    #[test]
    fn test_run_bf16_copies_q_to_o() -> Result<()> {
        let batch_size: u32 = 2;
        let num_qo_heads: u32 = 4;
        let num_kv_heads: u32 = 4;
        let head_dim: u32 = 128;
        let page_size: u32 = 16;
        let q_len = (batch_size * num_qo_heads * head_dim) as usize;
        let total_pages: u32 = 2;
        let kv_len = (total_pages * page_size * num_kv_heads * head_dim) as usize;

        // Prepare host data
        let q_host: Vec<half::bf16> = (0..q_len)
            .map(|i| half::bf16::from_f32((i as f32) * 0.01))
            .collect();
        let kv_zeros: Vec<half::bf16> = vec![half::bf16::ZERO; kv_len];
        let kv_indices: Vec<i32> = vec![0, 1];
        let kv_indptr: Vec<i32> = vec![0, 1, 2];
        let last_page_len: Vec<i32> = vec![page_size as i32; batch_size as usize];
        let o_zeros: Vec<half::bf16> = vec![half::bf16::ZERO; q_len];

        // Plan
        let mut handler = BatchDecodeHandler::new()?;
        let stream: cudaStream_t = ptr::null_mut();
        let plan_indptr: Vec<i32> = vec![0, 1, 2];
        handler.plan(
            &plan_indptr, batch_size, num_qo_heads, num_kv_heads,
            page_size, head_dim, false, stream,
        )?;

        unsafe {
            // Allocate and copy to device
            let q_d = to_device(&q_host);
            let k_d = to_device(&kv_zeros);
            let v_d = to_device(&kv_zeros);
            let o_d = to_device(&o_zeros);
            let idx_d = to_device(&kv_indices);
            let indptr_d = to_device(&kv_indptr);
            let lpl_d = to_device(&last_page_len);

            // Run kernel (stub copies q -> o)
            handler.run_bf16(
                q_d as *const half::bf16,
                k_d as *const half::bf16,
                v_d as *const half::bf16,
                o_d as *mut half::bf16,
                idx_d as *const i32,
                indptr_d as *const i32,
                lpl_d as *const i32,
                batch_size,
                stream,
            )?;

            crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?;

            // Verify: output should equal input q
            let o_host: Vec<half::bf16> = to_host(o_d, q_len);
            for i in 0..q_len {
                assert_eq!(
                    o_host[i].to_f32(), q_host[i].to_f32(),
                    "BF16 mismatch at index {}: got {}, expected {}",
                    i, o_host[i], q_host[i]
                );
            }

            // Cleanup
            crate::cuda::ffi::cudaFree(q_d);
            crate::cuda::ffi::cudaFree(k_d);
            crate::cuda::ffi::cudaFree(v_d);
            crate::cuda::ffi::cudaFree(o_d);
            crate::cuda::ffi::cudaFree(idx_d);
            crate::cuda::ffi::cudaFree(indptr_d);
            crate::cuda::ffi::cudaFree(lpl_d);
        }

        println!("BF16 batch decode stub: {} elements verified OK", q_len);
        Ok(())
    }

    #[test]
    fn test_run_fp16_copies_q_to_o() -> Result<()> {
        let batch_size: u32 = 2;
        let num_qo_heads: u32 = 4;
        let num_kv_heads: u32 = 4;
        let head_dim: u32 = 128;
        let page_size: u32 = 16;
        let q_len = (batch_size * num_qo_heads * head_dim) as usize;
        let total_pages: u32 = 2;
        let kv_len = (total_pages * page_size * num_kv_heads * head_dim) as usize;

        let q_host: Vec<half::f16> = (0..q_len)
            .map(|i| half::f16::from_f32((i as f32) * 0.01))
            .collect();
        let kv_zeros: Vec<half::f16> = vec![half::f16::ZERO; kv_len];
        let kv_indices: Vec<i32> = vec![0, 1];
        let kv_indptr: Vec<i32> = vec![0, 1, 2];
        let last_page_len: Vec<i32> = vec![page_size as i32; batch_size as usize];
        let o_zeros: Vec<half::f16> = vec![half::f16::ZERO; q_len];

        let mut handler = BatchDecodeHandler::new()?;
        let stream: cudaStream_t = ptr::null_mut();
        handler.plan(
            &[0i32, 1, 2], batch_size, num_qo_heads, num_kv_heads,
            page_size, head_dim, false, stream,
        )?;

        unsafe {
            let q_d = to_device(&q_host);
            let k_d = to_device(&kv_zeros);
            let v_d = to_device(&kv_zeros);
            let o_d = to_device(&o_zeros);
            let idx_d = to_device(&kv_indices);
            let indptr_d = to_device(&kv_indptr);
            let lpl_d = to_device(&last_page_len);

            handler.run_fp16(
                q_d as *const half::f16,
                k_d as *const half::f16,
                v_d as *const half::f16,
                o_d as *mut half::f16,
                idx_d as *const i32,
                indptr_d as *const i32,
                lpl_d as *const i32,
                batch_size,
                stream,
            )?;

            crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?;

            let o_host: Vec<half::f16> = to_host(o_d, q_len);
            for i in 0..q_len {
                assert_eq!(
                    o_host[i].to_f32(), q_host[i].to_f32(),
                    "FP16 mismatch at index {}: got {}, expected {}",
                    i, o_host[i], q_host[i]
                );
            }

            crate::cuda::ffi::cudaFree(q_d);
            crate::cuda::ffi::cudaFree(k_d);
            crate::cuda::ffi::cudaFree(v_d);
            crate::cuda::ffi::cudaFree(o_d);
            crate::cuda::ffi::cudaFree(idx_d);
            crate::cuda::ffi::cudaFree(indptr_d);
            crate::cuda::ffi::cudaFree(lpl_d);
        }

        println!("FP16 batch decode stub: {} elements verified OK", q_len);
        Ok(())
    }

    #[test]
    fn test_run_bf16_larger_batch() -> Result<()> {
        let batch_size: u32 = 8;
        let num_qo_heads: u32 = 32;
        let num_kv_heads: u32 = 8;
        let head_dim: u32 = 128;
        let page_size: u32 = 16;
        let q_len = (batch_size * num_qo_heads * head_dim) as usize;
        let total_pages: u32 = 8;
        let kv_len = (total_pages * page_size * num_kv_heads * head_dim) as usize;

        let q_host: Vec<half::bf16> = (0..q_len)
            .map(|i| half::bf16::from_f32(((i % 1000) as f32) * 0.001))
            .collect();
        let kv_zeros: Vec<half::bf16> = vec![half::bf16::ZERO; kv_len];
        // Each request uses 1 page
        let kv_indices: Vec<i32> = (0..batch_size as i32).collect();
        let kv_indptr: Vec<i32> = (0..=batch_size as i32).collect();
        let last_page_len: Vec<i32> = vec![page_size as i32; batch_size as usize];
        let o_zeros: Vec<half::bf16> = vec![half::bf16::ZERO; q_len];

        let mut handler = BatchDecodeHandler::new()?;
        let stream: cudaStream_t = ptr::null_mut();
        handler.plan(
            &kv_indptr, batch_size, num_qo_heads, num_kv_heads,
            page_size, head_dim, false, stream,
        )?;

        unsafe {
            let q_d = to_device(&q_host);
            let k_d = to_device(&kv_zeros);
            let v_d = to_device(&kv_zeros);
            let o_d = to_device(&o_zeros);
            let idx_d = to_device(&kv_indices);
            let indptr_d = to_device(&kv_indptr);
            let lpl_d = to_device(&last_page_len);

            handler.run_bf16(
                q_d as *const half::bf16,
                k_d as *const half::bf16,
                v_d as *const half::bf16,
                o_d as *mut half::bf16,
                idx_d as *const i32,
                indptr_d as *const i32,
                lpl_d as *const i32,
                batch_size,
                stream,
            )?;

            crate::cuda_check!(crate::cuda::ffi::cudaDeviceSynchronize())?;

            let o_host: Vec<half::bf16> = to_host(o_d, q_len);
            for i in 0..q_len {
                assert_eq!(
                    o_host[i].to_f32(), q_host[i].to_f32(),
                    "Batch=8 BF16 mismatch at index {}", i
                );
            }

            crate::cuda::ffi::cudaFree(q_d);
            crate::cuda::ffi::cudaFree(k_d);
            crate::cuda::ffi::cudaFree(v_d);
            crate::cuda::ffi::cudaFree(o_d);
            crate::cuda::ffi::cudaFree(idx_d);
            crate::cuda::ffi::cudaFree(indptr_d);
            crate::cuda::ffi::cudaFree(lpl_d);
        }

        println!("BF16 batch=8 heads=32/8 decode stub: {} elements verified OK", q_len);
        Ok(())
    }
}
