//! Device-agnostic tensor utilities used by diffusion models.
//!
//! All ops here run natively on both CPU and CUDA. For CUDA we call
//! the dedicated kernels we ship in `op/kernels/cuda/{permute,ewise_mul,rope_interleaved,cast_fill}`.

use crate::base::error::{Error, Result};
use crate::base::{DataType, DeviceType};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

// ─────────────────── FFI declarations ───────────────────

#[cfg(feature = "cuda")]
unsafe extern "C" {
    // ewise_mul
    fn ewise_mul_f32_forward(dst: *mut f32, a: *const f32, b: *const f32, n: i32, stream: cudaStream_t);
    fn ewise_mul_bf16_forward(dst: *mut half::bf16, a: *const half::bf16, b: *const half::bf16, n: i32, stream: cudaStream_t);
    fn ewise_mul_f16_forward(dst: *mut half::f16, a: *const half::f16, b: *const half::f16, n: i32, stream: cudaStream_t);
    fn ewise_mul_inplace_f32_forward(a: *mut f32, b: *const f32, n: i32, stream: cudaStream_t);
    fn ewise_mul_inplace_bf16_forward(a: *mut half::bf16, b: *const half::bf16, n: i32, stream: cudaStream_t);
    fn ewise_mul_inplace_f16_forward(a: *mut half::f16, b: *const half::f16, n: i32, stream: cudaStream_t);

    // rope interleaved
    fn rope_interleaved_f32_forward(x: *mut f32,
        cos: *const f32, sin: *const f32,
        seq: i32, n_heads: i32, head_dim: i32, stream: cudaStream_t);
    fn rope_interleaved_bf16_forward(x: *mut half::bf16,
        cos: *const f32, sin: *const f32,
        seq: i32, n_heads: i32, head_dim: i32, stream: cudaStream_t);

    // dtype cast
    fn cast_f32_to_bf16_forward(dst: *mut half::bf16, src: *const f32, n: i32, stream: cudaStream_t);
    fn cast_bf16_to_f32_forward(dst: *mut f32, src: *const half::bf16, n: i32, stream: cudaStream_t);
    fn cast_f32_to_f16_forward(dst: *mut half::f16, src: *const f32, n: i32, stream: cudaStream_t);
    fn cast_f16_to_f32_forward(dst: *mut f32, src: *const half::f16, n: i32, stream: cudaStream_t);

    // broadcast row
    fn broadcast_row_bf16_forward(dst: *mut half::bf16, row: *const half::bf16, num_rows: i32, d: i32, stream: cudaStream_t);
    fn broadcast_row_f32_forward(dst: *mut f32, row: *const f32, num_rows: i32, d: i32, stream: cudaStream_t);

    // fill repeat last
    fn fill_repeat_last_row_bf16_forward(dst: *mut half::bf16, n_src: i32, target_len: i32, d: i32, stream: cudaStream_t);
    fn fill_repeat_last_row_f32_forward(dst: *mut f32, n_src: i32, target_len: i32, d: i32, stream: cudaStream_t);
}

// memcpy D2D helpers for concat/pad — use bindgen-provided ffi.
#[cfg(feature = "cuda")]
#[inline]
unsafe fn d2d_memcpy_async(dst: *mut core::ffi::c_void, src: *const core::ffi::c_void, count: usize, stream: cudaStream_t) -> Result<()> {
    unsafe {
        let rc = crate::cuda::ffi::cudaMemcpyAsync(
            dst, src, count,
            crate::cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            stream);
        if rc != crate::cuda::ffi::cudaError_cudaSuccess {
            return Err(Error::InternalError(format!(
                "cudaMemcpyAsync D2D failed: {}", rc)).into());
        }
    }
    Ok(())
}

// ─────────────────── Public API ───────────────────

/// Deep-copy a tensor. Handles any dtype/device.
pub fn clone_tensor(t: &Tensor) -> Result<Tensor> {
    let mut out = Tensor::new(t.shape(), t.dtype(), t.device())?;
    out.copy_from(t)?;
    Ok(out)
}

/// Materialize a possibly non-contiguous view into a contiguous tensor.
pub fn materialize(t: &Tensor) -> Result<Tensor> {
    let mut out = Tensor::new(t.shape(), t.dtype(), t.device())?;
    out.copy_from(t)?;
    Ok(out)
}

/// ND permute — 委托给 `Tensor::permute`，它会自动分发到 CPU 逻辑或 CUDA kernel。
///
/// New shape: new_shape[j] = src_shape[perm[j]]
///
/// 这是分配路径的便利包装；零分配 hot path 请用 [`permute`]（dst-write）。
pub fn permute_nd(src: &Tensor, perm: &[usize]) -> Result<Tensor> {
    src.permute(perm)
}

/// Dst-write permute: `dst = src.permute(perm)`，不分配。
///
/// `dst` 的 shape 必须提前准备为
/// `[src.shape[perm[0]], ..., src.shape[perm[ndim-1]]]`；dtype / device 必须
/// 与 `src` 相同。底层直接转调 [`Tensor::permute_into`]。
pub fn permute(src: &Tensor, perm: &[usize], dst: &mut Tensor) -> Result<()> {
    src.permute_into(perm, dst)
}

/// Element-wise multiply: dst = a * b (same shape). Device-agnostic.
pub fn ewise_mul(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != dst.shape() {
        return Err(Error::InvalidArgument(format!(
            "ewise_mul: shape mismatch a={:?} b={:?} dst={:?}",
            a.shape(), b.shape(), dst.shape())).into());
    }
    let n = a.num_elements() as i32;
    match a.device() {
        DeviceType::Cpu => ewise_mul_cpu(a, b, dst),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            match a.dtype() {
                DataType::F32 => {
                    unsafe {
                        ewise_mul_f32_forward(
                            dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                            a.as_f32()?.buffer().as_ptr() as *const f32,
                            b.as_f32()?.buffer().as_ptr() as *const f32,
                            n, stream);
                    }
                }
                DataType::BF16 => {
                    unsafe {
                        ewise_mul_bf16_forward(
                            dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                            a.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                            b.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                            n, stream);
                    }
                }
                DataType::F16 => {
                    unsafe {
                        ewise_mul_f16_forward(
                            dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16,
                            a.as_f16()?.buffer().as_ptr() as *const half::f16,
                            b.as_f16()?.buffer().as_ptr() as *const half::f16,
                            n, stream);
                    }
                }
                other => return Err(Error::InvalidArgument(format!(
                    "ewise_mul CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

/// In-place element-wise multiply: a *= b (same shape). Device-agnostic.
pub fn ewise_mul_inplace(a: &mut Tensor, b: &Tensor) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(Error::InvalidArgument(format!(
            "ewise_mul_inplace: shape mismatch a={:?} b={:?}",
            a.shape(), b.shape())).into());
    }
    let n = a.num_elements() as i32;
    match a.device() {
        DeviceType::Cpu => ewise_mul_inplace_cpu(a, b),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            match a.dtype() {
                DataType::F32 => {
                    unsafe {
                        ewise_mul_inplace_f32_forward(
                            a.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                            b.as_f32()?.buffer().as_ptr() as *const f32,
                            n, stream);
                    }
                }
                DataType::BF16 => {
                    unsafe {
                        ewise_mul_inplace_bf16_forward(
                            a.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                            b.as_bf16()?.buffer().as_ptr() as *const half::bf16,
                            n, stream);
                    }
                }
                DataType::F16 => {
                    unsafe {
                        ewise_mul_inplace_f16_forward(
                            a.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16,
                            b.as_f16()?.buffer().as_ptr() as *const half::f16,
                            n, stream);
                    }
                }
                other => return Err(Error::InvalidArgument(format!(
                    "ewise_mul_inplace CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

fn ewise_mul_cpu(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    match a.dtype() {
        DataType::F32 => {
            let a_s = a.as_f32()?.as_slice()?;
            let b_s = b.as_f32()?.as_slice()?;
            let d_s = dst.as_f32_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() { d_s[i] = a_s[i] * b_s[i]; }
        }
        DataType::BF16 => {
            let a_s = a.as_bf16()?.as_slice()?;
            let b_s = b.as_bf16()?.as_slice()?;
            let d_s = dst.as_bf16_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() {
                d_s[i] = half::bf16::from_f32(a_s[i].to_f32() * b_s[i].to_f32());
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "ewise_mul CPU: unsupported dtype {:?}", a.dtype())).into()),
    }
    Ok(())
}

fn ewise_mul_inplace_cpu(a: &mut Tensor, b: &Tensor) -> Result<()> {
    match a.dtype() {
        DataType::F32 => {
            let b_s_vec: Vec<f32> = b.as_f32()?.as_slice()?.to_vec();
            let a_s = a.as_f32_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() { a_s[i] *= b_s_vec[i]; }
        }
        DataType::BF16 => {
            let b_s_vec: Vec<half::bf16> = b.as_bf16()?.as_slice()?.to_vec();
            let a_s = a.as_bf16_mut()?.as_slice_mut()?;
            for i in 0..a_s.len() {
                a_s[i] = half::bf16::from_f32(a_s[i].to_f32() * b_s_vec[i].to_f32());
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "ewise_mul_inplace CPU: unsupported dtype {:?}", a.dtype())).into()),
    }
    Ok(())
}

/// Apply interleaved 3D RoPE. Device-agnostic.
///
/// - `x`: [seq, n_heads, head_dim]
/// - `cos`, `sin`: [seq, head_dim/2] (F32 always)
pub fn apply_rope_interleaved_dev(
    x: &mut Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
) -> Result<()> {
    let shape = x.shape().to_vec();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: expected [seq, n_heads, head_dim], got {:?}", shape
        )).into());
    }
    let (seq, n_heads, d) = (shape[0], shape[1], shape[2]);
    if d != head_dim {
        return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: head_dim mismatch: shape={}, arg={}", d, head_dim)).into());
    }
    let half = head_dim / 2;
    let cos_shape = cos.shape();
    if cos_shape.len() != 2 || cos_shape[0] != seq || cos_shape[1] != half {
        return Err(Error::InvalidArgument(format!(
            "apply_rope_interleaved: cos shape mismatch: {:?} vs expected [{}, {}]",
            cos_shape, seq, half)).into());
    }
    if cos.dtype() != DataType::F32 || sin.dtype() != DataType::F32 {
        return Err(Error::InvalidArgument("apply_rope_interleaved: cos/sin must be F32".into()).into());
    }

    match x.device() {
        DeviceType::Cpu => {
            crate::model::diffusion::z_image::rope_embedder_3d::apply_rope_interleaved(
                x, cos, sin, head_dim)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            let cos_p = cos.as_f32()?.buffer().as_ptr() as *const f32;
            let sin_p = sin.as_f32()?.buffer().as_ptr() as *const f32;
            match x.dtype() {
                DataType::F32 => {
                    let xp = x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
                    unsafe {
                        rope_interleaved_f32_forward(xp, cos_p, sin_p,
                            seq as i32, n_heads as i32, head_dim as i32, stream);
                    }
                }
                DataType::BF16 => {
                    let xp = x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
                    unsafe {
                        rope_interleaved_bf16_forward(xp, cos_p, sin_p,
                            seq as i32, n_heads as i32, head_dim as i32, stream);
                    }
                }
                other => return Err(Error::InvalidArgument(format!(
                    "apply_rope_interleaved CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

/// Cast a tensor to a new dtype. Device-agnostic.
///
/// 分配路径的便利包装；零分配 hot path 请用 [`cast_dtype_into`]（dst-write）。
pub fn cast_dtype(src: &Tensor, new_dtype: DataType) -> Result<Tensor> {
    if src.dtype() == new_dtype { return clone_tensor(src); }
    let mut dst = Tensor::new(src.shape(), new_dtype, src.device())?;
    cast_dtype_into(src, &mut dst)?;
    Ok(dst)
}

/// Dst-write dtype cast: `dst = src.cast(dst.dtype())`，不分配。
///
/// `dst` 必须预分配为与 `src` 相同 shape、相同 device、目标 dtype 的张量。
/// 若 `src.dtype() == dst.dtype()`，退化为 `dst.copy_from(src)`。
pub fn cast_dtype_into(src: &Tensor, dst: &mut Tensor) -> Result<()> {
    if src.shape() != dst.shape() {
        return Err(Error::InvalidArgument(format!(
            "cast_dtype_into: shape mismatch src={:?} dst={:?}",
            src.shape(), dst.shape())).into());
    }
    if src.device() != dst.device() {
        return Err(Error::InvalidArgument(format!(
            "cast_dtype_into: device mismatch src={:?} dst={:?}",
            src.device(), dst.device())).into());
    }

    // identity fast path
    if src.dtype() == dst.dtype() {
        return dst.copy_from(src);
    }

    match src.device() {
        DeviceType::Cpu => {
            // CPU 路径直接用 Tensor::to_dtype 语义 —— 分配一次中间再拷贝。
            // CPU 不是 hot path，这样实现最简且保持与旧行为一致。
            let tmp = src.to_dtype(dst.dtype())?;
            dst.copy_from(&tmp)
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let n = src.num_elements() as i32;
            let stream = crate::cuda::get_current_cuda_stream();
            match (src.dtype(), dst.dtype()) {
                (DataType::F32, DataType::BF16) => unsafe {
                    cast_f32_to_bf16_forward(
                        dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                        src.as_f32()?.buffer().as_ptr() as *const f32, n, stream);
                }
                (DataType::BF16, DataType::F32) => unsafe {
                    cast_bf16_to_f32_forward(
                        dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                        src.as_bf16()?.buffer().as_ptr() as *const half::bf16, n, stream);
                }
                (DataType::F32, DataType::F16) => unsafe {
                    cast_f32_to_f16_forward(
                        dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16,
                        src.as_f32()?.buffer().as_ptr() as *const f32, n, stream);
                }
                (DataType::F16, DataType::F32) => unsafe {
                    cast_f16_to_f32_forward(
                        dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                        src.as_f16()?.buffer().as_ptr() as *const half::f16, n, stream);
                }
                (from, to) => return Err(Error::InvalidArgument(format!(
                    "cast_dtype_into CUDA: unsupported {:?} → {:?}", from, to)).into()),
            }
            Ok(())
        }
    }
}

/// Pad [n_src, D] by repeating the last row to [target_len, D]. Device-agnostic.
///
/// 分配路径的便利包装；零分配 hot path 请用 [`pad_last_row_into`]。
pub fn pad_last_row(src: &Tensor, target_len: usize) -> Result<Tensor> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row: expected [N, D], got {:?}", shape)).into());
    }
    let (_n, d) = (shape[0], shape[1]);
    let mut dst = Tensor::new(&[target_len, d], src.dtype(), src.device())?;
    pad_last_row_into(src, &mut dst)?;
    Ok(dst)
}

/// Dst-write pad-last-row: writes into pre-allocated `dst` (shape
/// `[target_len, D]`) without allocating.
///
/// `target_len` is inferred from `dst.shape()[0]`.
pub fn pad_last_row_into(src: &Tensor, dst: &mut Tensor) -> Result<()> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: expected [N, D], got {:?}", shape)).into());
    }
    let (n, d) = (shape[0], shape[1]);
    if dst.shape().len() != 2 || dst.shape()[1] != d {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: dst shape {:?} incompatible with [target_len, {}]",
            dst.shape(), d)).into());
    }
    let target_len = dst.shape()[0];
    if target_len < n {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: target_len ({}) < n ({})", target_len, n)).into());
    }
    if src.dtype() != dst.dtype() || src.device() != dst.device() {
        return Err(Error::InvalidArgument(format!(
            "pad_last_row_into: dtype/device mismatch").into()).into());
    }

    // Short-circuit: no padding needed.
    if target_len == n {
        return dst.copy_from(src);
    }

    let bytes_per_row = d * src.dtype().size_in_bytes();
    let src_bytes = n * bytes_per_row;

    match src.device() {
        DeviceType::Cpu => unsafe {
            std::ptr::copy_nonoverlapping(
                src.buffer().as_ptr(),
                dst.buffer_mut().as_mut_ptr(),
                src_bytes);
            // For padding rows, repeat row n-1 (well-defined since target_len > n ⇒ n ≥ 1).
            let last_ptr = src.buffer().as_ptr().add((n - 1) * bytes_per_row);
            for r in n..target_len {
                let dst_ptr = dst.buffer_mut().as_mut_ptr().add(r * bytes_per_row);
                std::ptr::copy_nonoverlapping(last_ptr, dst_ptr, bytes_per_row);
            }
            Ok(())
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            unsafe {
                d2d_memcpy_async(
                    dst.buffer_mut().as_mut_ptr() as *mut core::ffi::c_void,
                    src.buffer().as_ptr() as *const core::ffi::c_void,
                    src_bytes, stream)?;
            }
            match src.dtype() {
                DataType::BF16 => unsafe {
                    fill_repeat_last_row_bf16_forward(
                        dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16,
                        n as i32, target_len as i32, d as i32, stream);
                }
                DataType::F32 => unsafe {
                    fill_repeat_last_row_f32_forward(
                        dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32,
                        n as i32, target_len as i32, d as i32, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "pad_last_row_into CUDA: unsupported dtype {:?}", other)).into()),
            }
            Ok(())
        }
    }
}

/// Pad [n_src, D] to [target_len, D], writing a broadcast single-row `pad_token` to positions [n_src..].
///
/// 分配路径的便利包装；零分配 hot path 请用 [`pad_with_token_into`]。
pub fn pad_with_token(src: &Tensor, pad_token: &Tensor, target_len: usize) -> Result<Tensor> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token: expected [N, D], got {:?}", shape)).into());
    }
    let d = shape[1];
    let mut dst = Tensor::new(&[target_len, d], src.dtype(), src.device())?;
    pad_with_token_into(src, pad_token, &mut dst)?;
    Ok(dst)
}

/// Dst-write pad-with-token: `dst[..n_src] = src; dst[n_src..] = pad_token broadcasted`.
///
/// `target_len` is inferred from `dst.shape()[0]`.
pub fn pad_with_token_into(
    src: &Tensor, pad_token: &Tensor, dst: &mut Tensor,
) -> Result<()> {
    let shape = src.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: expected [N, D], got {:?}", shape)).into());
    }
    let (n, d) = (shape[0], shape[1]);
    if dst.shape().len() != 2 || dst.shape()[1] != d {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: dst shape {:?} incompatible with [target_len, {}]",
            dst.shape(), d)).into());
    }
    let target_len = dst.shape()[0];
    if target_len < n {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: target_len ({}) < n ({})", target_len, n)).into());
    }
    if pad_token.num_elements() != d {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: pad_token has {} elems, expected {}",
            pad_token.num_elements(), d)).into());
    }
    if src.dtype() != dst.dtype() || src.device() != dst.device() {
        return Err(Error::InvalidArgument(format!(
            "pad_with_token_into: dtype/device mismatch").into()).into());
    }

    if target_len == n {
        return dst.copy_from(src);
    }

    let bytes_per_row = d * src.dtype().size_in_bytes();
    let src_bytes = n * bytes_per_row;

    match src.device() {
        DeviceType::Cpu => unsafe {
            std::ptr::copy_nonoverlapping(
                src.buffer().as_ptr(),
                dst.buffer_mut().as_mut_ptr(),
                src_bytes);
            let pad_ptr = pad_token.buffer().as_ptr();
            for r in n..target_len {
                let dst_ptr = dst.buffer_mut().as_mut_ptr().add(r * bytes_per_row);
                std::ptr::copy_nonoverlapping(pad_ptr, dst_ptr, bytes_per_row);
            }
            Ok(())
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            if src_bytes > 0 {
                unsafe {
                    d2d_memcpy_async(
                        dst.buffer_mut().as_mut_ptr() as *mut core::ffi::c_void,
                        src.buffer().as_ptr() as *const core::ffi::c_void,
                        src_bytes, stream)?;
                }
            }
            let pad_rows = target_len - n;
            if pad_rows > 0 {
                let offset_bytes = n * bytes_per_row;
                match src.dtype() {
                    DataType::BF16 => unsafe {
                        let dst_ptr = (dst.buffer_mut().as_mut_ptr().add(offset_bytes)) as *mut half::bf16;
                        let pad_ptr = pad_token.buffer().as_ptr() as *const half::bf16;
                        broadcast_row_bf16_forward(dst_ptr, pad_ptr, pad_rows as i32, d as i32, stream);
                    }
                    DataType::F32 => unsafe {
                        let dst_ptr = (dst.buffer_mut().as_mut_ptr().add(offset_bytes)) as *mut f32;
                        let pad_ptr = pad_token.buffer().as_ptr() as *const f32;
                        broadcast_row_f32_forward(dst_ptr, pad_ptr, pad_rows as i32, d as i32, stream);
                    }
                    other => return Err(Error::InvalidArgument(format!(
                        "pad_with_token_into CUDA: unsupported dtype {:?}", other)).into()),
                }
            }
            Ok(())
        }
    }
}

/// Overwrite rows [keep_prefix..N] of `x` with a broadcast of `pad_token`
/// and return a new tensor (legacy convenience). 原地版本请用
/// [`overwrite_pad_tokens_inplace`]。
pub fn overwrite_pad_tokens(x: &Tensor, pad_token: &Tensor, keep_prefix: usize) -> Result<Tensor> {
    let mut dst = clone_tensor(x)?;
    overwrite_pad_tokens_inplace(&mut dst, pad_token, keep_prefix)?;
    Ok(dst)
}

/// In-place overwrite: `x[keep_prefix.., :] = pad_token broadcasted`.
pub fn overwrite_pad_tokens_inplace(
    x: &mut Tensor, pad_token: &Tensor, keep_prefix: usize,
) -> Result<()> {
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "overwrite_pad_tokens_inplace: expected [N, D], got {:?}", shape)).into());
    }
    let (total, d) = (shape[0], shape[1]);
    if keep_prefix > total {
        return Err(Error::InvalidArgument(format!(
            "overwrite_pad_tokens_inplace: keep_prefix ({}) > total ({})",
            keep_prefix, total)).into());
    }
    if pad_token.num_elements() != d {
        return Err(Error::InvalidArgument(format!(
            "overwrite_pad_tokens_inplace: pad_token has {} elems, expected {}",
            pad_token.num_elements(), d)).into());
    }
    let pad_rows = total - keep_prefix;
    if pad_rows == 0 { return Ok(()); }

    let bytes_per_row = d * x.dtype().size_in_bytes();
    let offset_bytes = keep_prefix * bytes_per_row;

    match x.device() {
        DeviceType::Cpu => unsafe {
            let pad_ptr = pad_token.buffer().as_ptr();
            for r in keep_prefix..total {
                let dst_ptr = x.buffer_mut().as_mut_ptr().add(r * bytes_per_row);
                std::ptr::copy_nonoverlapping(pad_ptr, dst_ptr, bytes_per_row);
            }
        }
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            match x.dtype() {
                DataType::BF16 => unsafe {
                    let dst_ptr = (x.buffer_mut().as_mut_ptr().add(offset_bytes)) as *mut half::bf16;
                    let pad_ptr = pad_token.buffer().as_ptr() as *const half::bf16;
                    broadcast_row_bf16_forward(dst_ptr, pad_ptr, pad_rows as i32, d as i32, stream);
                }
                DataType::F32 => unsafe {
                    let dst_ptr = (x.buffer_mut().as_mut_ptr().add(offset_bytes)) as *mut f32;
                    let pad_ptr = pad_token.buffer().as_ptr() as *const f32;
                    broadcast_row_f32_forward(dst_ptr, pad_ptr, pad_rows as i32, d as i32, stream);
                }
                other => return Err(Error::InvalidArgument(format!(
                    "overwrite_pad_tokens_inplace CUDA: unsupported dtype {:?}", other)).into()),
            }
        }
    }
    Ok(())
}

/// Concat two 2D tensors along dim 0: [S_a, D] + [S_b, D] → [S_a + S_b, D].
/// Device-agnostic.
///
/// 分配路径的便利包装；零分配 hot path 请用 [`concat_seq_into`]。
pub fn concat_seq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(Error::InvalidArgument("concat_seq: expected 2D tensors".into()).into());
    }
    let (s_a, d_a) = (a.shape()[0], a.shape()[1]);
    let (s_b, d_b) = (b.shape()[0], b.shape()[1]);
    if d_a != d_b {
        return Err(Error::InvalidArgument(format!(
            "concat_seq: dim mismatch {} vs {}", d_a, d_b)).into());
    }
    let mut dst = Tensor::new(&[s_a + s_b, d_a], a.dtype(), a.device())?;
    concat_seq_into(a, b, &mut dst)?;
    Ok(dst)
}

/// Dst-write concat: `dst[..s_a] = a; dst[s_a..] = b` along dim 0.
pub fn concat_seq_into(a: &Tensor, b: &Tensor, dst: &mut Tensor) -> Result<()> {
    if a.shape().len() != 2 || b.shape().len() != 2 {
        return Err(Error::InvalidArgument("concat_seq_into: expected 2D tensors".into()).into());
    }
    let (s_a, d_a) = (a.shape()[0], a.shape()[1]);
    let (s_b, d_b) = (b.shape()[0], b.shape()[1]);
    if d_a != d_b {
        return Err(Error::InvalidArgument(format!(
            "concat_seq_into: dim mismatch {} vs {}", d_a, d_b)).into());
    }
    if a.dtype() != b.dtype() || a.device() != b.device() {
        return Err(Error::InvalidArgument("concat_seq_into: dtype/device mismatch".into()).into());
    }
    if dst.shape() != [s_a + s_b, d_a].as_slice() {
        return Err(Error::InvalidArgument(format!(
            "concat_seq_into: dst shape {:?} incompatible with [{}, {}]",
            dst.shape(), s_a + s_b, d_a)).into());
    }
    if dst.dtype() != a.dtype() || dst.device() != a.device() {
        return Err(Error::InvalidArgument("concat_seq_into: dst dtype/device mismatch".into()).into());
    }

    let bytes_per_row = d_a * a.dtype().size_in_bytes();
    let a_bytes = s_a * bytes_per_row;
    let b_bytes = s_b * bytes_per_row;

    match a.device() {
        DeviceType::Cpu => unsafe {
            std::ptr::copy_nonoverlapping(
                a.buffer().as_ptr(), dst.buffer_mut().as_mut_ptr(), a_bytes);
            std::ptr::copy_nonoverlapping(
                b.buffer().as_ptr(),
                dst.buffer_mut().as_mut_ptr().add(a_bytes), b_bytes);
            Ok(())
        },
        #[cfg(feature = "cuda")]
        DeviceType::Cuda(_) => {
            let stream = crate::cuda::get_current_cuda_stream();
            unsafe {
                d2d_memcpy_async(
                    dst.buffer_mut().as_mut_ptr() as *mut core::ffi::c_void,
                    a.buffer().as_ptr() as *const core::ffi::c_void,
                    a_bytes, stream)?;
                d2d_memcpy_async(
                    dst.buffer_mut().as_mut_ptr().add(a_bytes) as *mut core::ffi::c_void,
                    b.buffer().as_ptr() as *const core::ffi::c_void,
                    b_bytes, stream)?;
            }
            Ok(())
        }
    }
}

// ─────────────────── Tests ───────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::tensor::Tensor;

    // ── helpers ──

    /// 将任意 dtype Tensor 转为 Vec<f32>（用于结果比较）
    fn to_f32_vec(t: &Tensor) -> Vec<f32> {
        match t.dtype() {
            DataType::F32 => t.as_f32().unwrap().as_slice().unwrap().to_vec(),
            DataType::BF16 => t.as_bf16().unwrap().as_slice().unwrap()
                .iter().map(|v| v.to_f32()).collect(),
            DataType::F16 => t.as_f16().unwrap().as_slice().unwrap()
                .iter().map(|v| v.to_f32()).collect(),
            other => panic!("to_f32_vec: unsupported dtype {:?}", other),
        }
    }

    /// 在 CPU 上构造 Tensor，再 optionally 拷到 CUDA
    fn make_tensor(shape: &[usize], dtype: DataType, device: DeviceType, values: &[f32]) -> Tensor {
        let mut cpu = Tensor::new(shape, dtype, DeviceType::Cpu).unwrap();
        match dtype {
            DataType::F32 => cpu.as_f32_mut().unwrap().as_slice_mut().unwrap()
                .copy_from_slice(values),
            DataType::BF16 => {
                let sl = cpu.as_bf16_mut().unwrap().as_slice_mut().unwrap();
                for (i, v) in values.iter().enumerate() { sl[i] = half::bf16::from_f32(*v); }
            }
            DataType::F16 => {
                let sl = cpu.as_f16_mut().unwrap().as_slice_mut().unwrap();
                for (i, v) in values.iter().enumerate() { sl[i] = half::f16::from_f32(*v); }
            }
            other => panic!("make_tensor: unsupported dtype {:?}", other),
        }
        if device != DeviceType::Cpu {
            #[cfg(feature = "cuda")]
            { cpu.to_device(device).unwrap() }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA not enabled") }
        } else {
            cpu
        }
    }

    /// 检查两个 f32 slice 是否在容差内相等
    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{}: length mismatch {} vs {}", msg, a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol,
                "{}: mismatch at {}: {} vs {} (diff={})", msg, i, x, y, (x - y).abs());
        }
    }

    // ── permute_nd tests ──

    #[test]
    fn test_permute_nd_f32_cpu() -> Result<()> {
        // [2,3,4] → permute(2,0,1) → [4,2,3]
        let mut src = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        let data = src.as_f32_mut()?.as_slice_mut()?;
        for i in 0..24 { data[i] = i as f32; }

        let dst = permute_nd(&src, &[2, 0, 1])?;
        assert_eq!(dst.shape(), &[4, 2, 3]);
        let r = dst.as_f32()?.as_slice()?;
        // new[w,i,j] = old[i,j,w]
        assert_eq!(r[0], 0.0);   // new[0,0,0] = old[0,0,0]
        assert_eq!(r[6], 1.0);   // new[1,0,0] = old[0,0,1]
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_nd_f32_cuda() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let mut src = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        let data = src.as_f32_mut()?.as_slice_mut()?;
        for i in 0..24 { data[i] = i as f32; }
        let src = src.to_device(device)?;

        let dst = permute_nd(&src, &[2, 0, 1])?;
        assert_eq!(dst.shape(), &[4, 2, 3]);
        let dst_cpu = dst.to_device(DeviceType::Cpu)?;
        let r = dst_cpu.as_f32()?.as_slice()?;
        assert_eq!(r[0], 0.0);
        assert_eq!(r[6], 1.0);
        Ok(())
    }

    /// CPU 与 CUDA permute_nd 结果一致
    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_nd_cpu_cuda_consistency() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let mut src_cpu = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        for (i, v) in src_cpu.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = (i as f32 * 0.1) - 1.0;
        }
        let src_cuda = src_cpu.to_device(device)?;

        let perm = &[2, 0, 1];
        let dst_cpu = permute_nd(&src_cpu, perm)?;
        let dst_cuda = permute_nd(&src_cuda, perm)?;
        let dst_cuda_cpu = dst_cuda.to_device(DeviceType::Cpu)?;

        let cpu_v = dst_cpu.as_f32()?.as_slice()?;
        let cuda_v = dst_cuda_cpu.as_f32()?.as_slice()?;
        assert_close(cpu_v, cuda_v, 1e-5, "permute_nd f32");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_nd_bf16_cuda() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let values: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let src = make_tensor(&[2, 3, 4], DataType::BF16, device, &values);

        let dst = permute_nd(&src, &[2, 0, 1])?;
        assert_eq!(dst.shape(), &[4, 2, 3]);
        let dst_cpu = dst.to_device(DeviceType::Cpu)?;
        let r: Vec<f32> = dst_cpu.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        assert!((r[0] - 0.0).abs() < 0.1, "bf16 permute[0] mismatch: {}", r[0]);
        assert!((r[6] - 1.0).abs() < 0.1, "bf16 permute[6] mismatch: {}", r[6]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_permute_nd_i32_cuda() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let mut src = Tensor::new(&[2, 3], DataType::I32, DeviceType::Cpu)?;
        let data = src.as_i32_mut()?.as_slice_mut()?;
        for i in 0..6 { data[i] = i as i32; }
        let src = src.to_device(device)?;

        let dst = permute_nd(&src, &[1, 0])?;  // transpose
        assert_eq!(dst.shape(), &[3, 2]);
        let dst_cpu = dst.to_device(DeviceType::Cpu)?;
        let r = dst_cpu.as_i32()?.as_slice()?;
        assert_eq!(r[0], 0);
        assert_eq!(r[1], 3);
        assert_eq!(r[2], 1);
        Ok(())
    }

    // ── apply_rope_interleaved_dev tests ──

    fn make_rope_tensors(
        seq: usize, n_heads: usize, head_dim: usize, device: DeviceType
    ) -> (Tensor, Tensor, Tensor) {
        let total = seq * n_heads * head_dim;
        let values: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1) + 0.5).collect();
        let x = make_tensor(&[seq, n_heads, head_dim], DataType::F32, device, &values);

        let half = head_dim / 2;
        let mut cos = Tensor::new(&[seq, half], DataType::F32, DeviceType::Cpu).unwrap();
        let mut sin = Tensor::new(&[seq, half], DataType::F32, DeviceType::Cpu).unwrap();
        cos.as_f32_mut().unwrap().as_slice_mut().unwrap().fill(1.0);
        sin.as_f32_mut().unwrap().as_slice_mut().unwrap().fill(0.0);

        if device != DeviceType::Cpu {
            #[cfg(feature = "cuda")]
            {
                let cos = cos.to_device(device).unwrap();
                let sin = sin.to_device(device).unwrap();
                return (x, cos, sin);
            }
        }
        (x, cos, sin)
    }

    #[test]
    fn test_rope_interleaved_dev_f32_cpu_identity() -> Result<()> {
        // cos=1, sin=0 → identity
        let (mut x, cos, sin) = make_rope_tensors(2, 3, 8, DeviceType::Cpu);
        let original = to_f32_vec(&x);
        apply_rope_interleaved_dev(&mut x, &cos, &sin, 8)?;
        let result = to_f32_vec(&x);
        assert_close(&original, &result, 1e-5, "rope identity cpu");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_rope_interleaved_dev_f32_cuda_identity() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let (mut x, cos, sin) = make_rope_tensors(2, 3, 8, device);
        let original = {
            let tmp = x.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        apply_rope_interleaved_dev(&mut x, &cos, &sin, 8)?;
        let result = {
            let tmp = x.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        assert_close(&original, &result, 1e-4, "rope identity cuda");
        Ok(())
    }

    /// CPU 与 CUDA apply_rope_interleaved_dev 结果一致（非 identity，用真实 cos/sin）
    #[test]
    #[cfg(feature = "cuda")]
    fn test_rope_interleaved_dev_cpu_cuda_consistency() -> Result<()> {
        let seq = 2;
        let n_heads = 2;
        let head_dim = 8;
        let half = head_dim / 2;

        let total = seq * n_heads * head_dim;
        let values: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2) - 1.0).collect();

        let mut x_cpu = make_tensor(&[seq, n_heads, head_dim], DataType::F32, DeviceType::Cpu, &values);
        let mut x_cuda = make_tensor(&[seq, n_heads, head_dim], DataType::F32, DeviceType::Cuda(0), &values);

        // 非 identity 的 cos/sin
        let mut cos = Tensor::new(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        let mut sin = Tensor::new(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        let csl = cos.as_f32_mut()?.as_slice_mut()?;
        let ssl = sin.as_f32_mut()?.as_slice_mut()?;
        for i in 0..seq * half {
            csl[i] = (i as f32 * 0.3).cos();
            ssl[i] = (i as f32 * 0.3).sin();
        }
        let cos_cuda = cos.to_device(DeviceType::Cuda(0))?;
        let sin_cuda = sin.to_device(DeviceType::Cuda(0))?;

        apply_rope_interleaved_dev(&mut x_cpu, &cos, &sin, head_dim)?;
        apply_rope_interleaved_dev(&mut x_cuda, &cos_cuda, &sin_cuda, head_dim)?;

        let r_cpu = to_f32_vec(&x_cpu);
        let r_cuda = {
            let tmp = x_cuda.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        // f32 kernel 精度高，1e-4 容差
        assert_close(&r_cpu, &r_cuda, 1e-4, "rope_interleaved f32 cpu vs cuda");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_rope_interleaved_dev_bf16_cuda() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let seq = 1;
        let n_heads = 1;
        let head_dim = 4;
        let half = 2;

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mut x = make_tensor(&[seq, n_heads, head_dim], DataType::BF16, device, &values);

        let mut cos = Tensor::new(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        let mut sin = Tensor::new(&[seq, half], DataType::F32, DeviceType::Cpu)?;
        cos.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.8, 0.6]);
        sin.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.3, 0.7]);
        let cos = cos.to_device(device)?;
        let sin = sin.to_device(device)?;

        apply_rope_interleaved_dev(&mut x, &cos, &sin, head_dim)?;

        let r: Vec<f32> = {
            let tmp = x.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        // expected: [1*0.8-2*0.3, 2*0.8+1*0.3, 3*0.6-4*0.7, 4*0.6+3*0.7]
        let expected = [0.2, 1.9, -1.0, 4.5];
        for i in 0..4 {
            assert!((r[i] - expected[i]).abs() < 0.1,
                "bf16 rope mismatch at {}: {} vs {}", i, r[i], expected[i]);
        }
        Ok(())
    }

    // ── cast_dtype tests ──

    #[test]
    fn test_cast_dtype_f32_to_bf16_cpu() -> Result<()> {
        let mut src = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        src.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let dst = cast_dtype(&src, DataType::BF16)?;
        assert_eq!(dst.dtype(), DataType::BF16);
        let r: Vec<f32> = dst.as_bf16()?.as_slice()?.iter().map(|v| v.to_f32()).collect();
        assert_close(&[1.0, 2.0, 3.0, 4.0], &r, 0.01, "cast f32->bf16 cpu");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cast_dtype_f32_to_bf16_cuda() -> Result<()> {
        let device = DeviceType::Cuda(0);
        let src = make_tensor(&[4], DataType::F32, device, &[1.0, 2.0, 3.0, 4.0]);
        let dst = cast_dtype(&src, DataType::BF16)?;
        assert_eq!(dst.dtype(), DataType::BF16);
        let r: Vec<f32> = {
            let tmp = dst.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        assert_close(&[1.0, 2.0, 3.0, 4.0], &r, 0.01, "cast f32->bf16 cuda");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cast_dtype_cpu_cuda_consistency() -> Result<()> {
        let values: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1) - 5.0).collect();
        let src_cpu = make_tensor(&[128], DataType::F32, DeviceType::Cpu, &values);
        let src_cuda = make_tensor(&[128], DataType::F32, DeviceType::Cuda(0), &values);

        let dst_cpu = cast_dtype(&src_cpu, DataType::BF16)?;
        let dst_cuda = cast_dtype(&src_cuda, DataType::BF16)?;
        let r_cpu = {
            let tmp = dst_cpu.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        let r_cuda = {
            let tmp = dst_cuda.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        assert_close(&r_cpu, &r_cuda, 0.01, "cast f32->bf16 cpu vs cuda");
        Ok(())
    }

    #[test]
    fn test_cast_dtype_bf16_to_f32_cpu() -> Result<()> {
        let values = [1.0f32, 2.0, 3.0];
        let src = make_tensor(&[3], DataType::BF16, DeviceType::Cpu, &values);
        let dst = cast_dtype(&src, DataType::F32)?;
        assert_eq!(dst.dtype(), DataType::F32);
        let r = dst.as_f32()?.as_slice()?;
        assert_close(&values, r, 1e-5, "cast bf16->f32 cpu");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cast_dtype_identity() -> Result<()> {
        // same dtype → clone, values unchanged
        let src = make_tensor(&[8], DataType::F32, DeviceType::Cuda(0), &[1.0; 8]);
        let dst = cast_dtype(&src, DataType::F32)?;
        assert_eq!(dst.dtype(), DataType::F32);
        let r: Vec<f32> = {
            let tmp = dst.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        assert_close(&[1.0; 8], &r, 1e-5, "cast identity");
        Ok(())
    }

    // ── ewise_mul tests ──

    #[test]
    fn test_ewise_mul_f32_cpu() -> Result<()> {
        let mut a = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[2.0, 3.0, 4.0, 5.0]);
        let mut dst = Tensor::new(&[4], DataType::F32, DeviceType::Cpu)?;
        ewise_mul(&a, &b, &mut dst)?;
        let r = dst.as_f32()?.as_slice()?;
        assert_close(&[2.0, 6.0, 12.0, 20.0], r, 1e-5, "ewise_mul f32 cpu");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_ewise_mul_f32_cuda() -> Result<()> {
        let a = make_tensor(&[4], DataType::F32, DeviceType::Cuda(0), &[1.0, 2.0, 3.0, 4.0]);
        let b = make_tensor(&[4], DataType::F32, DeviceType::Cuda(0), &[2.0, 3.0, 4.0, 5.0]);
        let mut dst = Tensor::new(&[4], DataType::F32, DeviceType::Cuda(0))?;
        ewise_mul(&a, &b, &mut dst)?;
        let r: Vec<f32> = {
            let tmp = dst.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        assert_close(&[2.0, 6.0, 12.0, 20.0], &r, 1e-5, "ewise_mul f32 cuda");
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_ewise_mul_cpu_cuda_consistency() -> Result<()> {
        let vals: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 + 0.5).collect();
        let a_cpu = make_tensor(&[128], DataType::F32, DeviceType::Cpu, &vals);
        let b_cpu = make_tensor(&[128], DataType::F32, DeviceType::Cpu, &vals.iter().map(|v| v * 2.0).collect::<Vec<_>>());
        let a_cuda = a_cpu.to_device(DeviceType::Cuda(0))?;
        let b_cuda = b_cpu.to_device(DeviceType::Cuda(0))?;

        let mut dst_cpu = Tensor::new(&[128], DataType::F32, DeviceType::Cpu)?;
        let mut dst_cuda = Tensor::new(&[128], DataType::F32, DeviceType::Cuda(0))?;
        ewise_mul(&a_cpu, &b_cpu, &mut dst_cpu)?;
        ewise_mul(&a_cuda, &b_cuda, &mut dst_cuda)?;

        let r_cpu = to_f32_vec(&dst_cpu);
        let r_cuda: Vec<f32> = {
            let tmp = dst_cuda.to_device(DeviceType::Cpu)?;
            to_f32_vec(&tmp)
        };
        assert_close(&r_cpu, &r_cuda, 1e-4, "ewise_mul f32 cpu vs cuda");
        Ok(())
    }

    // ─── S2.3 dst-write equivalence tests ───
    //
    // For each `*_into` added in S2.3, assert byte-level equality against
    // the legacy allocating API on both CPU and CUDA. This locks in the
    // invariant that the dst-write path produces identical output to the
    // old path.

    #[test]
    fn permute_into_matches_permute_nd_cpu_f32() -> Result<()> {
        let mut src = Tensor::new(&[2, 3, 4], DataType::F32, DeviceType::Cpu)?;
        for (i, v) in src.as_f32_mut()?.as_slice_mut()?.iter_mut().enumerate() {
            *v = i as f32;
        }
        let legacy = permute_nd(&src, &[2, 0, 1])?;
        let mut dst = Tensor::new(legacy.shape(), DataType::F32, DeviceType::Cpu)?;
        permute(&src, &[2, 0, 1], &mut dst)?;
        assert_eq!(legacy.as_f32()?.as_slice()?, dst.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn permute_into_matches_permute_nd_cuda_bf16() -> Result<()> {
        let values: Vec<f32> = (0..24).map(|i| i as f32 * 0.25).collect();
        let src = make_tensor(&[2, 3, 4], DataType::BF16, DeviceType::Cuda(0), &values);
        let legacy = permute_nd(&src, &[2, 0, 1])?;
        let mut dst = Tensor::new(legacy.shape(), DataType::BF16, DeviceType::Cuda(0))?;
        permute(&src, &[2, 0, 1], &mut dst)?;

        let a_cpu = legacy.to_cpu()?;
        let b_cpu = dst.to_cpu()?;
        assert_eq!(
            a_cpu.as_bf16()?.as_slice()?,
            b_cpu.as_bf16()?.as_slice()?,
        );
        Ok(())
    }

    #[test]
    fn cast_dtype_into_matches_cast_dtype_cpu_f32_to_bf16() -> Result<()> {
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 2.5).collect();
        let src = make_tensor(&[64], DataType::F32, DeviceType::Cpu, &data);
        let legacy = cast_dtype(&src, DataType::BF16)?;
        let mut dst = Tensor::new(&[64], DataType::BF16, DeviceType::Cpu)?;
        cast_dtype_into(&src, &mut dst)?;
        assert_eq!(
            legacy.as_bf16()?.as_slice()?,
            dst.as_bf16()?.as_slice()?,
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cast_dtype_into_matches_cast_dtype_cuda_f32_to_bf16() -> Result<()> {
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.05 - 4.0).collect();
        let src = make_tensor(&[256], DataType::F32, DeviceType::Cuda(0), &data);
        let legacy = cast_dtype(&src, DataType::BF16)?;
        let mut dst = Tensor::new(&[256], DataType::BF16, DeviceType::Cuda(0))?;
        cast_dtype_into(&src, &mut dst)?;

        let a = legacy.to_cpu()?;
        let b = dst.to_cpu()?;
        assert_eq!(a.as_bf16()?.as_slice()?, b.as_bf16()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn pad_last_row_into_matches_pad_last_row_cpu_f32() -> Result<()> {
        let src = make_tensor(
            &[3, 4], DataType::F32, DeviceType::Cpu,
            &[1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,  9.0, 10.0, 11.0, 12.0]);
        let legacy = pad_last_row(&src, 6)?;
        let mut dst = Tensor::new(&[6, 4], DataType::F32, DeviceType::Cpu)?;
        pad_last_row_into(&src, &mut dst)?;
        assert_eq!(legacy.as_f32()?.as_slice()?, dst.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn pad_last_row_into_matches_pad_last_row_cuda_bf16() -> Result<()> {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let src = make_tensor(&[3, 4], DataType::BF16, DeviceType::Cuda(0), &data);
        let legacy = pad_last_row(&src, 6)?;
        let mut dst = Tensor::new(&[6, 4], DataType::BF16, DeviceType::Cuda(0))?;
        pad_last_row_into(&src, &mut dst)?;

        let a = legacy.to_cpu()?;
        let b = dst.to_cpu()?;
        assert_eq!(a.as_bf16()?.as_slice()?, b.as_bf16()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn pad_with_token_into_matches_pad_with_token_cpu_f32() -> Result<()> {
        let src = make_tensor(&[2, 3], DataType::F32, DeviceType::Cpu,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let pad = make_tensor(&[3], DataType::F32, DeviceType::Cpu, &[-1.0, -2.0, -3.0]);
        let legacy = pad_with_token(&src, &pad, 5)?;
        let mut dst = Tensor::new(&[5, 3], DataType::F32, DeviceType::Cpu)?;
        pad_with_token_into(&src, &pad, &mut dst)?;
        assert_eq!(legacy.as_f32()?.as_slice()?, dst.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn pad_with_token_into_matches_pad_with_token_cuda_bf16() -> Result<()> {
        let src_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let pad_data: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let src = make_tensor(&[2, 3], DataType::BF16, DeviceType::Cuda(0), &src_data);
        let pad = make_tensor(&[3], DataType::BF16, DeviceType::Cuda(0), &pad_data);
        let legacy = pad_with_token(&src, &pad, 5)?;
        let mut dst = Tensor::new(&[5, 3], DataType::BF16, DeviceType::Cuda(0))?;
        pad_with_token_into(&src, &pad, &mut dst)?;
        let a = legacy.to_cpu()?;
        let b = dst.to_cpu()?;
        assert_eq!(a.as_bf16()?.as_slice()?, b.as_bf16()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn overwrite_pad_tokens_inplace_matches_legacy_cpu() -> Result<()> {
        let x = make_tensor(&[4, 3], DataType::F32, DeviceType::Cpu,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let pad = make_tensor(&[3], DataType::F32, DeviceType::Cpu, &[-1.0, -1.0, -1.0]);
        let legacy = overwrite_pad_tokens(&x, &pad, 2)?;

        let mut clone = Tensor::new(&[4, 3], DataType::F32, DeviceType::Cpu)?;
        clone.copy_from(&x)?;
        overwrite_pad_tokens_inplace(&mut clone, &pad, 2)?;

        assert_eq!(legacy.as_f32()?.as_slice()?, clone.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn overwrite_pad_tokens_inplace_matches_legacy_cuda_bf16() -> Result<()> {
        let data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let pad_data = vec![-1.0_f32; 3];
        let x = make_tensor(&[4, 3], DataType::BF16, DeviceType::Cuda(0), &data);
        let pad = make_tensor(&[3], DataType::BF16, DeviceType::Cuda(0), &pad_data);
        let legacy = overwrite_pad_tokens(&x, &pad, 2)?;

        let mut clone = Tensor::new(&[4, 3], DataType::BF16, DeviceType::Cuda(0))?;
        clone.copy_from(&x)?;
        overwrite_pad_tokens_inplace(&mut clone, &pad, 2)?;

        let a = legacy.to_cpu()?;
        let b = clone.to_cpu()?;
        assert_eq!(a.as_bf16()?.as_slice()?, b.as_bf16()?.as_slice()?);
        Ok(())
    }

    #[test]
    fn concat_seq_into_matches_concat_seq_cpu_f32() -> Result<()> {
        let a = make_tensor(&[2, 3], DataType::F32, DeviceType::Cpu,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_tensor(&[1, 3], DataType::F32, DeviceType::Cpu,
            &[7.0, 8.0, 9.0]);
        let legacy = concat_seq(&a, &b)?;
        let mut dst = Tensor::new(&[3, 3], DataType::F32, DeviceType::Cpu)?;
        concat_seq_into(&a, &b, &mut dst)?;
        assert_eq!(legacy.as_f32()?.as_slice()?, dst.as_f32()?.as_slice()?);
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn concat_seq_into_matches_concat_seq_cuda_bf16() -> Result<()> {
        let a_data: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let b_data: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0];
        let a = make_tensor(&[3, 4], DataType::BF16, DeviceType::Cuda(0), &a_data);
        let b = make_tensor(&[1, 4], DataType::BF16, DeviceType::Cuda(0), &b_data);
        let legacy = concat_seq(&a, &b)?;
        let mut dst = Tensor::new(&[4, 4], DataType::BF16, DeviceType::Cuda(0))?;
        concat_seq_into(&a, &b, &mut dst)?;

        let lc = legacy.to_cpu()?;
        let dc = dst.to_cpu()?;
        assert_eq!(lc.as_bf16()?.as_slice()?, dc.as_bf16()?.as_slice()?);
        Ok(())
    }
}
