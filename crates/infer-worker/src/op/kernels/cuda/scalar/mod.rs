use crate::base::error::{Error, Result};
use crate::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::cuda::ffi::cudaStream_t;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn scalar_mul_f32_forward(dst: *mut f32, src: *const f32, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_bf16_forward(dst: *mut half::bf16, src: *const half::bf16, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_f16_forward(dst: *mut half::f16, src: *const half::f16, val: f32, n: i32, stream: cudaStream_t);

    fn scalar_add_f32_forward(dst: *mut f32, src: *const f32, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_add_bf16_forward(dst: *mut half::bf16, src: *const half::bf16, val: f32, n: i32, stream: cudaStream_t);
    fn scalar_add_f16_forward(dst: *mut half::f16, src: *const half::f16, val: f32, n: i32, stream: cudaStream_t);

    fn silu_inplace_f32_forward(data: *mut f32, n: i32, stream: cudaStream_t);
    fn silu_inplace_bf16_forward(data: *mut half::bf16, n: i32, stream: cudaStream_t);
    fn silu_inplace_f16_forward(data: *mut half::f16, n: i32, stream: cudaStream_t);

    fn tanh_inplace_f32_forward(data: *mut f32, n: i32, stream: cudaStream_t);
    fn tanh_inplace_bf16_forward(data: *mut half::bf16, n: i32, stream: cudaStream_t);
    fn tanh_inplace_f16_forward(data: *mut half::f16, n: i32, stream: cudaStream_t);

    // Device-scalar variants used by CUDA Graph denoise capture.
    fn scalar_mul_inplace_from_dev_f32_forward (x: *mut f32,         d_val: *const f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_inplace_from_dev_bf16_forward(x: *mut half::bf16,  d_val: *const f32, n: i32, stream: cudaStream_t);
    fn scalar_mul_inplace_from_dev_f16_forward (x: *mut half::f16,   d_val: *const f32, n: i32, stream: cudaStream_t);

    fn sinusoid_embedding_from_dev_f32_forward (out: *mut f32,         d_t: *const f32, dim: i32, stream: cudaStream_t);
    fn sinusoid_embedding_from_dev_bf16_forward(out: *mut half::bf16,  d_t: *const f32, dim: i32, stream: cudaStream_t);
    fn sinusoid_embedding_from_dev_f16_forward (out: *mut half::f16,   d_t: *const f32, dim: i32, stream: cudaStream_t);
}

/// dst[i] = src[i] * val  (CUDA)
#[cfg(feature = "cuda")]
pub fn scalar_mul(src: &Tensor, dst: &mut Tensor, val: f32, stream: cudaStream_t) -> Result<()> {
    let n = src.num_elements() as i32;
    match src.dtype() {
        crate::base::DataType::F32 => {
            let s = src.as_f32()?.buffer().as_ptr() as *const f32;
            let d = dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_mul_f32_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let s = src.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let d = dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { scalar_mul_bf16_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::F16 => {
            let s = src.as_f16()?.buffer().as_ptr() as *const half::f16;
            let d = dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { scalar_mul_f16_forward(d, s, val, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA scalar_mul: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}

/// dst[i] = src[i] + val  (CUDA)
#[cfg(feature = "cuda")]
pub fn scalar_add(src: &Tensor, dst: &mut Tensor, val: f32, stream: cudaStream_t) -> Result<()> {
    let n = src.num_elements() as i32;
    match src.dtype() {
        crate::base::DataType::F32 => {
            let s = src.as_f32()?.buffer().as_ptr() as *const f32;
            let d = dst.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_add_f32_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let s = src.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let d = dst.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { scalar_add_bf16_forward(d, s, val, n, stream); }
        }
        crate::base::DataType::F16 => {
            let s = src.as_f16()?.buffer().as_ptr() as *const half::f16;
            let d = dst.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { scalar_add_f16_forward(d, s, val, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA scalar_add: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}

/// 原地 SiLU: x[i] = x[i] * sigmoid(x[i])  (CUDA)
#[cfg(feature = "cuda")]
pub fn silu_inplace(x: &mut Tensor, stream: cudaStream_t) -> Result<()> {
    let n = x.num_elements() as i32;
    match x.dtype() {
        crate::base::DataType::F32 => {
            let p = x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { silu_inplace_f32_forward(p, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let p = x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { silu_inplace_bf16_forward(p, n, stream); }
        }
        crate::base::DataType::F16 => {
            let p = x.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { silu_inplace_f16_forward(p, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA silu_inplace: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}

/// 原地 tanh: x[i] = tanh(x[i])  (CUDA)
#[cfg(feature = "cuda")]
pub fn tanh_inplace(x: &mut Tensor, stream: cudaStream_t) -> Result<()> {
    let n = x.num_elements() as i32;
    match x.dtype() {
        crate::base::DataType::F32 => {
            let p = x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { tanh_inplace_f32_forward(p, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let p = x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { tanh_inplace_bf16_forward(p, n, stream); }
        }
        crate::base::DataType::F16 => {
            let p = x.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { tanh_inplace_f16_forward(p, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA tanh_inplace: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}

/// 原地标量乘，系数从 device `[1] f32` 读取。
///
/// `x` 与 `d_scalar` 在 capture/replay 之间指针保持稳定；host 只需在每次
/// replay 前用一次 `cudaMemcpyAsync` 改写 `d_scalar` 指向的字节，kernel
/// 参数（包括 `d_scalar` 指针本身）则保持不变。
#[cfg(feature = "cuda")]
pub fn scalar_mul_inplace_from_dev(
    x: &mut Tensor,
    d_scalar: &Tensor,
    stream: cudaStream_t,
) -> Result<()> {
    if d_scalar.dtype() != crate::base::DataType::F32
        || d_scalar.num_elements() != 1
    {
        return Err(Error::InvalidArgument(format!(
            "scalar_mul_inplace_from_dev: d_scalar must be [1] F32, got {:?} {:?}",
            d_scalar.shape(), d_scalar.dtype(),
        )).into());
    }
    let n = x.num_elements() as i32;
    let d = d_scalar.as_f32()?.buffer().as_ptr() as *const f32;
    match x.dtype() {
        crate::base::DataType::F32 => {
            let p = x.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { scalar_mul_inplace_from_dev_f32_forward(p, d, n, stream); }
        }
        crate::base::DataType::BF16 => {
            let p = x.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { scalar_mul_inplace_from_dev_bf16_forward(p, d, n, stream); }
        }
        crate::base::DataType::F16 => {
            let p = x.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { scalar_mul_inplace_from_dev_f16_forward(p, d, n, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA scalar_mul_inplace_from_dev: unsupported dtype {:?}", other
        )).into()),
    }
    Ok(())
}

/// Sinusoidal timestep embedding, reading the (already-scaled) scalar
/// timestep from device memory.
///
/// - `d_t`  : `[1]` F32 device — contains `t_value * t_scale`.
/// - `out`  : `[1, dim]` any of F32/BF16/F16 device — receives the
///            `[cos(arg) | sin(arg)]` embedding. `dim` must be even.
///
/// Designed to be the **only** timestep-dependent kernel in the denoise
/// graph: host stores the entire per-step schedule up-front in a
/// `TValueDevVec: [N] f32` slot, then passes `d_t = TValueDevVec + i`
/// each step; kernel args (the pointer) are constant across replays.
#[cfg(feature = "cuda")]
pub fn sinusoid_embedding_from_dev(
    out: &mut Tensor,
    d_t: &Tensor,
    stream: cudaStream_t,
) -> Result<()> {
    if d_t.dtype() != crate::base::DataType::F32 || d_t.num_elements() != 1 {
        return Err(Error::InvalidArgument(format!(
            "sinusoid_embedding_from_dev: d_t must be [1] F32, got {:?} {:?}",
            d_t.shape(), d_t.dtype(),
        )).into());
    }
    let dim = out.num_elements() as i32;
    if dim % 2 != 0 {
        return Err(Error::InvalidArgument(format!(
            "sinusoid_embedding_from_dev: dim={} must be even", dim,
        )).into());
    }
    let d = d_t.as_f32()?.buffer().as_ptr() as *const f32;
    match out.dtype() {
        crate::base::DataType::F32 => {
            let p = out.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;
            unsafe { sinusoid_embedding_from_dev_f32_forward(p, d, dim, stream); }
        }
        crate::base::DataType::BF16 => {
            let p = out.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            unsafe { sinusoid_embedding_from_dev_bf16_forward(p, d, dim, stream); }
        }
        crate::base::DataType::F16 => {
            let p = out.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            unsafe { sinusoid_embedding_from_dev_f16_forward(p, d, dim, stream); }
        }
        other => return Err(Error::InvalidArgument(format!(
            "CUDA sinusoid_embedding_from_dev: unsupported out dtype {:?}", other
        )).into()),
    }
    Ok(())
}
