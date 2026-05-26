//! CUDA infrastructure adapter.
//!
//! Implements `Device` and `OpBackend` for `Cuda`.
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

use std::marker::PhantomData;
use std::sync::Arc;
use crate::domain::ports::{Device, OpBackend, OpResult, OpError};
use crate::domain::types::{Dtype, Shape};
use crate::domain::tensor::Tensor;

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
    pub fn new(device_id: i32) -> Result<Self, crate::domain::ports::OpError> {
        device_utils::set_current_device(device_id)
            .map_err(|e| OpError::Kernel(format!("set device failed: {}", e)))?;
        let config = Arc::new(CudaConfig::new()
            .map_err(|e| OpError::Kernel(format!("CudaConfig::new failed: {}", e)))?);
        Ok(Self { device_id, config })
    }
}

/// Allocate a zeroed CUDA tensor via cudaMalloc + cudaMemset.
impl<T: Dtype> Tensor<T, Cuda> {
    pub fn zeros_cuda(shape: impl Into<Shape>, device: &Cuda) -> OpResult<Self> {
        let shape = shape.into();
        let numel = shape.numel();
        let size_bytes = numel * T::SIZE_BYTES;
        let size = size_bytes.max(1);

        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let code = ffi::cudaMalloc(&mut ptr, size);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("cudaMalloc({} bytes) failed: {:?}", size, code)));
            }
            let code = ffi::cudaMemset(ptr, 0, size);
            if code != ffi::cudaError_cudaSuccess {
                ffi::cudaFree(ptr);
                return Err(OpError::Kernel(format!("cudaMemset failed: {:?}", code)));
            }
        }

        let strides = shape.contiguous_strides();
        Ok(Tensor {
            shape, strides, offset_elems: 0, numel,
            is_contiguous: true,
            storage_ptr: ptr as *mut u8, storage_len: size_bytes,
            device: device.clone(), _marker: PhantomData,
        })
    }
}

/// OpBackend for Cuda — dispatches to CUDA kernels.
/// Each method fetches the stream from the tensor's device context.
impl OpBackend for Cuda {
    fn alloc_tensor<T: Dtype>(shape: Shape, device: &Self) -> OpResult<Tensor<T, Self>> {
        Tensor::<T, Cuda>::zeros_cuda(shape, device)
    }

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
    fn attention<T: Dtype>(
        q: &Tensor<T, Self>, k: &Tensor<T, Self>, v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        _seq_starts: &Tensor<i32, Self>,
        head_num: usize, kv_head_num: usize, head_dim: usize, scale: f32,
    ) -> OpResult<()> {
        kernels::attention::attention_prefill(
            q, k, v, output,
            head_num as i32, kv_head_num as i32, head_dim as i32, scale,
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
        // Split Q: columns [0, q_dim)
        kernels::split_cols::split_cols(qkv, q, rows, total_cols, 0, q_dim as i32)?;
        // Split K: columns [q_dim, q_dim + kv_dim)
        kernels::split_cols::split_cols(qkv, k, rows, total_cols, q_dim as i32, kv_dim as i32)?;
        // Split V: columns [q_dim + kv_dim, q_dim + 2*kv_dim)
        kernels::split_cols::split_cols(qkv, v, rows, total_cols, (q_dim + kv_dim) as i32, kv_dim as i32)?;
        Ok(())
    }
    fn scatter_kv<T: Dtype>(
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        k_cache: &mut Tensor<T, Self>,
        v_cache: &mut Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        kv_dim: usize,
    ) -> OpResult<()> {
        kernels::scatter_kv::scatter_kv(k, v, k_cache, v_cache, positions, kv_dim)
    }
    fn argmax<T: Dtype>(logits: &Tensor<T, Self>, num_rows: usize) -> OpResult<i32> {
        // Use the sampler kernel on the last row
        let vocab = logits.numel() / num_rows;
        let last_row_offset = (num_rows - 1) * vocab;

        // Create a single-element i32 output tensor on device
        let device = logits.device();
        let mut output = Tensor::<i32, Cuda>::zeros_cuda([1], device)?;

        // Point to last row of logits
        let last_row_ptr = unsafe { (logits.data_ptr() as *const u8).add(last_row_offset * T::SIZE_BYTES) };
        let stream = device.config.stream;

        use crate::domain::types::DataType;
        unsafe extern "C" {
            fn argmax_kernel_bf16(output: *mut i32, input: *const half::bf16, vocab_size: i32, stream: ffi::cudaStream_t);
            fn argmax_kernel_fp16(output: *mut i32, input: *const half::f16, vocab_size: i32, stream: ffi::cudaStream_t);
            fn argmax_kernel_fp32(output: *mut i32, input: *const f32, vocab_size: i32, stream: ffi::cudaStream_t);
        }
        unsafe {
            match T::DATA_TYPE {
                DataType::BF16 => argmax_kernel_bf16(output.data_ptr_mut(), last_row_ptr as _, vocab as i32, stream),
                DataType::F16 => argmax_kernel_fp16(output.data_ptr_mut(), last_row_ptr as _, vocab as i32, stream),
                DataType::F32 => argmax_kernel_fp32(output.data_ptr_mut(), last_row_ptr as _, vocab as i32, stream),
                _ => return Err(OpError::Kernel(format!("argmax: {:?}", T::DATA_TYPE))),
            }
        }

        // Copy result back to host
        let mut result: i32 = 0;
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                &mut result as *mut i32 as *mut std::ffi::c_void,
                output.data_ptr() as *const std::ffi::c_void,
                4,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("argmax D2H copy failed: {:?}", code)));
            }
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("argmax sync failed: {:?}", code)));
            }
        }
        Ok(result)
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
        q: &Tensor<T, Self>, k: &Tensor<T, Self>, v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_heads: usize, num_kv_heads: usize, head_dim: usize, scale: f32,
    ) -> OpResult<()> {
        kernels::attention::attention_prefill(
            q, k, v, output,
            num_heads as i32, num_kv_heads as i32, head_dim as i32, scale,
        )
    }
}

