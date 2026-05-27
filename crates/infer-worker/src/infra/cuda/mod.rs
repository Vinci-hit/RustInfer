//! CUDA infrastructure adapter.
//!
//! Implements `Device`, `MemoryPort`, and `OpBackend` for `Cuda`.
//! Contains: FFI bindings, CudaConfig (handles), thread-local stream,
//! and kernel dispatch wrappers.

pub mod ffi;
pub mod error;
pub mod device_utils;
pub mod thread_stream;
pub mod config;
pub mod kernels;

pub use config::{CudaConfig, GraphSlot};
pub use error::CudaError;
pub use thread_stream::{get_current_cuda_stream, with_cuda_stream};

use std::ptr::NonNull;
use std::sync::Arc;

use crate::domain::ports::{Device, MemoryPort, OpBackend, OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};

/// CUDA device — carries device_id + shared CudaConfig (handles, stream).
#[derive(Debug, Clone)]
pub struct Cuda {
    pub device_id: i32,
    pub config: Arc<CudaConfig>,
}

impl Device for Cuda {
    type ExecCtx = CudaConfig;
    fn exec_ctx(&self) -> &CudaConfig { &self.config }
    fn name(&self) -> &'static str { "cuda" }
}

impl Cuda {
    /// Create a new Cuda device (allocates stream + handles).
    pub fn new(device_id: i32) -> Result<Self, OpError> {
        device_utils::set_current_device(device_id)
            .map_err(|e| OpError::Kernel(format!("set device failed: {}", e)))?;
        let config = Arc::new(CudaConfig::new()
            .map_err(|e| OpError::Kernel(format!("CudaConfig::new failed: {}", e)))?);
        Ok(Self { device_id, config })
    }
}

// ─── Cuda MemoryPort ─────────────────────────────────────────────────────────

impl MemoryPort for Cuda {
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>> {
        let n = size.max(1);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cudaMalloc/cudaMemset are safe to call with valid args.
        unsafe {
            let code = ffi::cudaMalloc(&mut ptr, n);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaMalloc({}) failed: {:?}", n, code)));
            }
            let code = ffi::cudaMemset(ptr, 0, n);
            if code != ffi::cudaError_cudaSuccess {
                ffi::cudaFree(ptr);
                return Err(OpError::Kernel(format!("cudaMemset failed: {:?}", code)));
            }
        }
        NonNull::new(ptr as *mut u8)
            .ok_or_else(|| OpError::Kernel("cudaMalloc returned null".into()))
    }

    unsafe fn free_bytes(&self, ptr: NonNull<u8>, _size: usize) {
        // SAFETY: ptr came from cudaMalloc.
        unsafe { ffi::cudaFree(ptr.as_ptr() as *mut std::ffi::c_void); }
    }

    unsafe fn upload(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        if size == 0 { return Ok(()); }
        let stream = self.config.stream;
        // SAFETY: caller asserts dst is a device ptr with `size` bytes,
        // src is a host ptr with `size` bytes.
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst.as_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaMemcpyAsync H2D failed: {:?}", code)));
            }
            // Sync so the host buffer can be freed/reused safely after this returns.
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaStreamSynchronize failed: {:?}", code)));
            }
        }
        Ok(())
    }

    unsafe fn upload_async(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        if size == 0 { return Ok(()); }
        let stream = self.config.stream;
        // SAFETY: caller asserts the host pointer remains valid until the
        // device stream consumes the copy (workspaces own host staging
        // buffers for their entire lifetime, so this is upheld).
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst.as_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaMemcpyAsync H2D async failed: {:?}", code)));
            }
            // NO cudaStreamSynchronize — graph capture friendly.
        }
        Ok(())
    }

    unsafe fn download(&self, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()> {
        if size == 0 { return Ok(()); }
        let stream = self.config.stream;
        // SAFETY: caller asserts ptrs and size.
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaMemcpyAsync D2H failed: {:?}", code)));
            }
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaStreamSynchronize failed: {:?}", code)));
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> OpResult<()> {
        let stream = self.config.stream;
        // SAFETY: stream is owned by self.config.
        let code = unsafe { ffi::cudaStreamSynchronize(stream) };
        if code != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!("cudaStreamSynchronize failed: {:?}", code)));
        }
        Ok(())
    }
}

/// OpBackend for Cuda — dispatches to CUDA kernels.
/// Each method fetches the stream from the tensor's device context.
impl OpBackend for Cuda {
    // alloc_tensor uses the default impl (Tensor::zeros via MemoryPort).

    fn add<T: Dtype>(a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::add::add(a, b, dst)
    }
    fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()> {
        kernels::add::add_inplace(dst, src)
    }
    fn rmsnorm<T: Dtype>(input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>, eps: f32) -> OpResult<()> {
        kernels::rmsnorm::rmsnorm(input, weight, output, eps)
    }
    fn rmsnorm_inplace<T: Dtype>(x: &mut Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32) -> OpResult<()> {
        kernels::rmsnorm::rmsnorm_inplace(x, weight, eps)
    }
    fn fused_add_rmsnorm<T: Dtype>(
        output: &mut Tensor<T, Self>, residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32,
    ) -> OpResult<()> {
        kernels::fused_add_rmsnorm::fused_add_rmsnorm(output, residual, input, weight, eps)
    }
    fn matmul<T: Dtype>(input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::matmul::matmul(input, weight, output)
    }
    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        input: &Tensor<A, Self>, weight: &Tensor<W, Self>, output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>, zeros: Option<&Tensor<W, Self>>, group_size: usize,
    ) -> OpResult<()> {
        kernels::matmul::matmul_quant(input, weight, output, scales, zeros, group_size)
    }
    fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::activation::silu_inplace(x)
    }
    fn swiglu_inplace<T: Dtype>(x: &mut Tensor<T, Self>, gate: &Tensor<T, Self>) -> OpResult<()> {
        kernels::activation::swiglu_inplace(x, gate)
    }
    fn softmax<T: Dtype>(input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::softmax::softmax(input, output)
    }
    fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()> {
        kernels::scalar::scalar_mul_inplace(x, scalar)
    }
    fn embedding<T: Dtype>(table: &Tensor<T, Self>, indices: &Tensor<i32, Self>, output: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::embedding::embedding(table, indices, output)
    }
    fn rope_inplace<T: Dtype>(
        q: &mut Tensor<T, Self>, k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>, cos: &Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize,
    ) -> OpResult<()> {
        let num_tokens = q.shape().as_slice()[0] as i32;
        kernels::rope::rope_inplace(
            q, k, sin, cos,
            positions.data_ptr(),
            num_tokens, head_num as i32, kv_head_num as i32, head_dim as i32,
        )
    }
    fn attention_paged<T: Dtype>(
        q: &Tensor<T, Self>,
        k_pool: &Tensor<T, Self>,
        v_pool: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        plan: &crate::domain::batch::BatchPlan<Self>,
        workspace: &mut Tensor<f32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()> {
        kernels::attention_paged::attention_paged(
            q, k_pool, v_pool, output, plan, workspace,
            head_num, kv_head_num, head_dim, scale,
        )
    }

    fn split_qkv<T: Dtype>(
        qkv: &Tensor<T, Self>,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        v: &mut Tensor<T, Self>,
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        let total_cols = (q_dim + 2 * kv_dim) as i32;
        let rows = num_tokens as i32;
        kernels::split_cols::split_cols(qkv, q, rows, total_cols, 0, q_dim as i32)?;
        kernels::split_cols::split_cols(qkv, k, rows, total_cols, q_dim as i32, kv_dim as i32)?;
        kernels::split_cols::split_cols(qkv, v, rows, total_cols, (q_dim + kv_dim) as i32, kv_dim as i32)?;
        Ok(())
    }

    fn scatter_kv_paged<T: Dtype>(
        k_src: &Tensor<T, Self>,
        v_src: &Tensor<T, Self>,
        k_pool: &mut Tensor<T, Self>,
        v_pool: &mut Tensor<T, Self>,
        block_tables: &Tensor<i32, Self>,
        seq_positions: &Tensor<i32, Self>,
        cu_q_lens: &Tensor<i32, Self>,
        seq_lens_step: &Tensor<i32, Self>,
        max_blocks_per_seq: usize,
        block_size: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        kernels::scatter_kv_paged::scatter_kv_paged(
            k_src, v_src, k_pool, v_pool,
            block_tables, seq_positions, cu_q_lens, seq_lens_step,
            max_blocks_per_seq, block_size, kv_dim,
        )
    }

    fn argmax_batched<T: Dtype>(
        logits: &Tensor<T, Self>,
        cu_q_lens: &Tensor<i32, Self>,
        batch: usize,
    ) -> OpResult<Vec<i32>> {
        kernels::argmax_batched::argmax_batched(logits, cu_q_lens, batch)
    }

    // ═══════════════════════════════════════════════════════════════════
    // Diffusion ops — CUDA dispatch
    // ═══════════════════════════════════════════════════════════════════

    fn conv2d<T: Dtype>(
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>,
        bias: Option<&Tensor<T, Self>>, output: &mut Tensor<T, Self>,
        stride: usize, padding: usize,
    ) -> OpResult<()> {
        kernels::conv2d::conv2d(input, weight, bias, output, stride, padding)
    }

    fn groupnorm<T: Dtype>(
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>, num_groups: usize, eps: f32,
    ) -> OpResult<()> {
        kernels::groupnorm::groupnorm(input, weight, bias, output, num_groups, eps)
    }

    fn groupnorm_silu<T: Dtype>(
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>, num_groups: usize, eps: f32,
    ) -> OpResult<()> {
        kernels::groupnorm::groupnorm_silu(input, weight, bias, output, num_groups, eps)
    }

    fn layernorm<T: Dtype>(
        input: &Tensor<T, Self>, weight: &Tensor<T, Self>, bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>, eps: f32,
    ) -> OpResult<()> {
        kernels::layernorm::layernorm(input, weight, bias, output, eps)
    }

    fn upsample_nearest_2x<T: Dtype>(
        input: &Tensor<T, Self>, output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::upsample::upsample_nearest_2x(input, output)
    }

    fn broadcast_mul_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>, scale: &Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::broadcast_mul::broadcast_mul_inplace(x, scale)
    }

    fn ewise_mul<T: Dtype>(
        a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::ewise_mul::ewise_mul(a, b, dst)
    }

    fn sdpa<T: Dtype>(
        _q: &Tensor<T, Self>, _k: &Tensor<T, Self>, _v: &Tensor<T, Self>,
        _output: &mut Tensor<T, Self>,
        _num_heads: usize, _num_kv_heads: usize, _head_dim: usize, _scale: f32,
    ) -> OpResult<()> {
        // SDPA is only used by diffusion; LLM path goes through attention_paged.
        Err(OpError::Kernel("sdpa: not supported in paged-only build".into()))
    }
}

