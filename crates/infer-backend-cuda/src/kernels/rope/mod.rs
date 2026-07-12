//! RoPE (Rotary Position Embedding) CUDA kernel wrapper.
//!
//! Dispatch is an attribute of the element type: [`RopeKernel`] is implemented
//! once per supported dtype and names that dtype's `extern "C"` entry point, so
//! [`rope_inplace`] is generic with no runtime `match`. Adding a dtype is one
//! `impl`; an unsupported dtype fails to compile.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use crate::kernels::dtype_kernel::CudaFloat;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

unsafe extern "C" {
    pub fn rope_kernel_cu(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut f32,
        k: *mut f32,
        positions: *const i32,
        seq_len: i32,
        sin_cache: *const f32,
        cos_cache: *const f32,
        stream: cudaStream_t,
    );
    pub fn rope_kernel_cu_bf16(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut half::bf16,
        k: *mut half::bf16,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const half::bf16,
        cos_cache: *const half::bf16,
        stream: cudaStream_t,
    );
    pub fn rope_kernel_cu_fp16(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut half::f16,
        k: *mut half::f16,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const half::f16,
        cos_cache: *const half::f16,
        stream: cudaStream_t,
    );
}

/// Element types with a RoPE CUDA kernel. The method forwards to this dtype's
/// `extern` entry; the wrapper below is generic over this trait, so the
/// dtype→kernel mapping lives here as a type attribute.
///
/// The f32 entry point predates the strided-view support and ignores the
/// `q_row_stride`/`k_row_stride` arguments; the bf16/f16 entries honor them.
///
/// # Safety
/// Implementors' pointers must be valid device pointers for the given token/head
/// layout on `stream`; this just names the FFI entry and performs no checks.
pub trait RopeKernel: CudaFloat {
    /// Apply RoPE in-place to `q`/`k` using `sin`/`cos` caches and `positions`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn rope(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut Self,
        k: *mut Self,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const Self,
        cos_cache: *const Self,
        stream: cudaStream_t,
    );
}

impl RopeKernel for f32 {
    #[inline]
    unsafe fn rope(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut Self,
        k: *mut Self,
        positions: *const i32,
        seq_len: i32,
        _q_row_stride: i32,
        _k_row_stride: i32,
        sin_cache: *const Self,
        cos_cache: *const Self,
        stream: cudaStream_t,
    ) {
        unsafe {
            rope_kernel_cu(
                dim, kv_dim, head_size, q, k, positions, seq_len, sin_cache, cos_cache, stream,
            )
        }
    }
}

impl RopeKernel for half::bf16 {
    #[inline]
    unsafe fn rope(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut Self,
        k: *mut Self,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const Self,
        cos_cache: *const Self,
        stream: cudaStream_t,
    ) {
        unsafe {
            rope_kernel_cu_bf16(
                dim,
                kv_dim,
                head_size,
                q,
                k,
                positions,
                seq_len,
                q_row_stride,
                k_row_stride,
                sin_cache,
                cos_cache,
                stream,
            )
        }
    }
}

impl RopeKernel for half::f16 {
    #[inline]
    unsafe fn rope(
        dim: i32,
        kv_dim: i32,
        head_size: i32,
        q: *mut Self,
        k: *mut Self,
        positions: *const i32,
        seq_len: i32,
        q_row_stride: i32,
        k_row_stride: i32,
        sin_cache: *const Self,
        cos_cache: *const Self,
        stream: cudaStream_t,
    ) {
        unsafe {
            rope_kernel_cu_fp16(
                dim,
                kv_dim,
                head_size,
                q,
                k,
                positions,
                seq_len,
                q_row_stride,
                k_row_stride,
                sin_cache,
                cos_cache,
                stream,
            )
        }
    }
}

/// Apply RoPE in-place to Q and K tensors.
/// q: [num_tokens, q_dim] = [num_tokens, head_num*head_dim]
/// k: [num_tokens, kv_dim] = [num_tokens, kv_head_num*head_dim]
/// q/k may be **strided views** (e.g. zero-copy slices of a fused QKV
/// buffer) — we read each tensor's row stride directly, so the kernel
/// works whether or not q/k are contiguous along the row dimension.
/// sin/cos: [max_seq_len, head_dim/2]
/// positions: device pointer to [num_tokens] i32
pub fn rope_inplace<T: RopeKernel>(
    stream: cudaStream_t,
    q: &mut Tensor<T, Cuda>,
    k: &mut Tensor<T, Cuda>,
    sin: &Tensor<T, Cuda>,
    cos: &Tensor<T, Cuda>,
    positions_dev: *const i32,
    num_tokens: i32,
    head_num: i32,
    kv_head_num: i32,
    head_dim: i32,
) -> OpResult<()> {
    let q_dim = head_num * head_dim;
    let kv_dim = kv_head_num * head_dim;
    // Row stride from tensor: stride[0] for a 2D [rows, cols] view.
    let q_row_stride = q.strides().as_slice()[0] as i32;
    let k_row_stride = k.strides().as_slice()[0] as i32;
    unsafe {
        T::rope(
            q_dim,
            kv_dim,
            head_dim,
            q.data_ptr_mut(),
            k.data_ptr_mut(),
            positions_dev,
            num_tokens,
            q_row_stride,
            k_row_stride,
            sin.data_ptr(),
            cos.data_ptr(),
            stream,
        );
    }
    Ok(())
}
