use crate::base::error::{Result, Error};
use crate::tensor::{Tensor, TypedTensor};
use crate::cuda::{self, CudaConfig};

unsafe extern "C" {
    fn scatter_kernel_bf16(
        dst: *mut half::bf16,
        src: *const half::bf16,
        pos: *const i32,
        kvdim: i32,
        max_seq_len: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kernel_fp16(
        dst: *mut half::f16,
        src: *const half::f16,
        pos: *const i32,
        kvdim: i32,
        max_seq_len: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kernel_f32(
        dst: *mut f32,
        src: *const f32,
        pos: *const i32,
        kvdim: i32,
        max_seq_len: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_kernel_bf16(
        dst_k: *mut half::bf16,
        src_k: *const half::bf16,
        dst_v: *mut half::bf16,
        src_v: *const half::bf16,
        pos: *const i32,
        kvdim: i32,
        max_seq_len: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_kernel_fp16(
        dst_k: *mut half::f16,
        src_k: *const half::f16,
        dst_v: *mut half::f16,
        src_v: *const half::f16,
        pos: *const i32,
        kvdim: i32,
        max_seq_len: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_batch_kernel_bf16(
        dst_k_ptrs: *mut *mut half::bf16,
        dst_v_ptrs: *mut *mut half::bf16,
        src_k: *const half::bf16,
        src_v: *const half::bf16,
        positions: *const i32,
        batch_size: i32,
        kvdim: i32,
        src_k_row_stride: i32,
        src_v_row_stride: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn scatter_kv_batch_kernel_fp16(
        dst_k_ptrs: *mut *mut half::f16,
        dst_v_ptrs: *mut *mut half::f16,
        src_k: *const half::f16,
        src_v: *const half::f16,
        positions: *const i32,
        batch_size: i32,
        kvdim: i32,
        src_k_row_stride: i32,
        src_v_row_stride: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// Scatter operation: copies src[0, :] to dst[pos, :]
///
/// # Arguments
/// * `dst` - Destination tensor with shape [max_seq_len, kvdim]
/// * `src` - Source tensor with shape [1, kvdim]
/// * `pos` - Position offset in the destination tensor
/// * `cuda_config` - Optional CUDA configuration for stream
pub fn scatter(
    dst: &mut Tensor,
    src: &Tensor,
    pos: &Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // Validate data types match
    let dtype = dst.dtype();
    if src.dtype() != dtype {
        return Err(Error::InvalidArgument(format!(
            "Data type mismatch: dst={:?}, src={:?}",
            dtype, src.dtype()
        )).into());
    }

    // Validate shapes
    let dst_shape = dst.shape();
    let src_shape = src.shape();

    if dst_shape.len() != 2 || src_shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "Both tensors must be 2D. dst_shape={:?}, src_shape={:?}",
            dst_shape, src_shape
        )).into());
    }

    if src_shape[0] != 1 {
        return Err(Error::InvalidArgument(format!(
            "Source tensor must have shape [1, kvdim], got {:?}",
            src_shape
        )).into());
    }

    let max_seq_len = dst_shape[0];
    let kvdim = dst_shape[1];

    if src_shape[1] != kvdim {
        return Err(Error::InvalidArgument(format!(
            "KV dimension mismatch: dst kvdim={}, src kvdim={}",
            kvdim, src_shape[1]
        )).into());
    }
    // Get CUDA stream
    let stream = CudaConfig::resolve_stream(cuda_config);

    // Dispatch based on data type
    match dtype {
        crate::base::DataType::BF16 => {
            let dst_typed: &mut TypedTensor<half::bf16> = dst.as_bf16_mut()?;
            let src_typed = src.as_bf16()?;

            let dst_ptr = dst_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let src_ptr = src_typed.buffer().as_ptr() as *const half::bf16;

            unsafe {
                scatter_kernel_bf16(
                    dst_ptr,
                    src_ptr,
                    pos.as_i32()?.buffer().as_ptr() as *const i32,
                    kvdim as i32,
                    max_seq_len as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            let dst_typed: &mut TypedTensor<half::f16> = dst.as_f16_mut()?;
            let src_typed = src.as_f16()?;

            let dst_ptr = dst_typed.buffer_mut().as_mut_ptr() as *mut half::f16;
            let src_ptr = src_typed.buffer().as_ptr() as *const half::f16;

            unsafe {
                scatter_kernel_fp16(
                    dst_ptr,
                    src_ptr,
                    pos.as_i32()?.buffer().as_ptr() as *const i32,
                    kvdim as i32,
                    max_seq_len as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::F32 => {
            let dst_typed: &mut TypedTensor<f32> = dst.as_f32_mut()?;
            let src_typed = src.as_f32()?;

            let dst_ptr = dst_typed.buffer_mut().as_mut_ptr() as *mut f32;
            let src_ptr = src_typed.buffer().as_ptr() as *const f32;

            unsafe {
                scatter_kernel_f32(
                    dst_ptr,
                    src_ptr,
                    pos.as_i32()?.buffer().as_ptr() as *const i32,
                    kvdim as i32,
                    max_seq_len as i32,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for scatter: {:?}", dtype
            )).into());
        }
    }

    Ok(())
}

/// Fused scatter for K and V caches: writes both K and V in a single kernel launch.
/// Saves one kernel launch + gap per layer in decode phase.
///
/// dst_k, dst_v: [max_seq_len, kvdim] cache tensors
/// src_k, src_v: [1, kvdim] current step data
/// pos: scalar position tensor
pub fn scatter_kv(
    dst_k: &mut Tensor,
    src_k: &Tensor,
    dst_v: &mut Tensor,
    src_v: &Tensor,
    pos: &Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let kvdim = dst_k.shape()[1];
    let max_seq_len = dst_k.shape()[0];
    let stream = CudaConfig::resolve_stream(cuda_config);
    let dtype = dst_k.dtype();
    let pos_ptr = pos.as_i32()?.buffer().as_ptr() as *const i32;

    match dtype {
        crate::base::DataType::BF16 => {
            let dst_k_ptr = dst_k.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let src_k_ptr = src_k.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let dst_v_ptr = dst_v.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let src_v_ptr = src_v.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            unsafe {
                scatter_kv_kernel_bf16(
                    dst_k_ptr, src_k_ptr, dst_v_ptr, src_v_ptr,
                    pos_ptr, kvdim as i32, max_seq_len as i32, stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            let dst_k_ptr = dst_k.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            let src_k_ptr = src_k.as_f16()?.buffer().as_ptr() as *const half::f16;
            let dst_v_ptr = dst_v.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
            let src_v_ptr = src_v.as_f16()?.buffer().as_ptr() as *const half::f16;
            unsafe {
                scatter_kv_kernel_fp16(
                    dst_k_ptr, src_k_ptr, dst_v_ptr, src_v_ptr,
                    pos_ptr, kvdim as i32, max_seq_len as i32, stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for scatter_kv: {:?}", dtype
            )).into());
        }
    }

    Ok(())
}

/// Batched scatter_kv: 一次 kernel launch 把 B 行 K/V 写入 B 个不同 cache 的各自位置。
///
/// 调用方需提供 `k_ptrs_dev` / `v_ptrs_dev`：device 上的指针数组 buffer
/// （容量 ≥ batch_size * sizeof(u64)），本函数负责把 host 指针 async 拷入。
#[allow(clippy::too_many_arguments)]
pub fn scatter_kv_batch(
    k_caches: &mut [&mut Tensor],
    v_caches: &mut [&mut Tensor],
    src_k: &Tensor,
    src_v: &Tensor,
    positions_dev: &Tensor, // [B] i32, device
    k_ptrs_dev: *mut u64,   // device buffer，容量 ≥ B*8 bytes
    v_ptrs_dev: *mut u64,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let batch_size = k_caches.len();
    if batch_size == 0 { return Ok(()); }
    let kvdim = k_caches[0].shape()[1];
    let dtype = k_caches[0].dtype();
    let stream = CudaConfig::resolve_stream(cuda_config);

    // 1. 收集 host 指针数组
    let mut k_host: Vec<u64> = Vec::with_capacity(batch_size);
    let mut v_host: Vec<u64> = Vec::with_capacity(batch_size);
    match dtype {
        crate::base::DataType::BF16 => {
            for i in 0..batch_size {
                k_host.push(k_caches[i].as_bf16_mut()?.buffer_mut().as_mut_ptr() as u64);
                v_host.push(v_caches[i].as_bf16_mut()?.buffer_mut().as_mut_ptr() as u64);
            }
        }
        crate::base::DataType::F16 => {
            for i in 0..batch_size {
                k_host.push(k_caches[i].as_f16_mut()?.buffer_mut().as_mut_ptr() as u64);
                v_host.push(v_caches[i].as_f16_mut()?.buffer_mut().as_mut_ptr() as u64);
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for scatter_kv_batch: {:?}", dtype
            )).into());
        }
    }

    // 2. host → device copy（async，在 stream 上）
    let bytes = batch_size * std::mem::size_of::<u64>();
    unsafe {
        crate::cuda_check!(cuda::ffi::cudaMemcpyAsync(
            k_ptrs_dev as *mut _,
            k_host.as_ptr() as *const _,
            bytes,
            cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream,
        ))?;
        crate::cuda_check!(cuda::ffi::cudaMemcpyAsync(
            v_ptrs_dev as *mut _,
            v_host.as_ptr() as *const _,
            bytes,
            cuda::ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream,
        ))?;
    }
    // host buffer 必须在 async memcpy 完成之前保活。这里做一次 stream sync 保证安全性。
    // (不是性能关键：仅 16 bytes 级别的 copy；之后的 kernel 仍在同一 stream 排队)
    unsafe { crate::cuda_check!(cuda::ffi::cudaStreamSynchronize(stream))?; }
    drop(k_host);
    drop(v_host);

    // 3. Launch batched kernel (src_{k,v}_row_stride = kvdim, 连续情形)
    let pos_ptr = positions_dev.as_i32()?.buffer().as_ptr() as *const i32;
    match dtype {
        crate::base::DataType::BF16 => {
            let src_k_ptr = src_k.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let src_v_ptr = src_v.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            unsafe {
                scatter_kv_batch_kernel_bf16(
                    k_ptrs_dev as *mut *mut half::bf16,
                    v_ptrs_dev as *mut *mut half::bf16,
                    src_k_ptr, src_v_ptr,
                    pos_ptr, batch_size as i32, kvdim as i32,
                    kvdim as i32, kvdim as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            let src_k_ptr = src_k.as_f16()?.buffer().as_ptr() as *const half::f16;
            let src_v_ptr = src_v.as_f16()?.buffer().as_ptr() as *const half::f16;
            unsafe {
                scatter_kv_batch_kernel_fp16(
                    k_ptrs_dev as *mut *mut half::f16,
                    v_ptrs_dev as *mut *mut half::f16,
                    src_k_ptr, src_v_ptr,
                    pos_ptr, batch_size as i32, kvdim as i32,
                    kvdim as i32, kvdim as i32,
                    stream,
                );
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Batched scatter_kv "ready" 版：假设 `k_ptrs_dev` / `v_ptrs_dev` 已包含 B 个
/// 正确的 device cache 指针（调用方负责填充），**不做任何 cudaMemcpyAsync 或
/// stream sync**——纯 kernel launch，CUDA Graph 可捕获。
///
/// 支持 src 非连续：通过 `src_{k,v}_row_stride`（元素单位）和 `src_{k,v}_col_offset`
/// 指定每行起点，可直接从 fused qkv tensor 读 k/v 段。
#[allow(clippy::too_many_arguments)]
pub fn scatter_kv_batch_launch_ready(
    dtype: crate::base::DataType,
    kvdim: usize,
    batch_size: usize,
    src_k: &Tensor,
    src_v: &Tensor,
    src_k_row_stride: usize,
    src_v_row_stride: usize,
    src_k_col_offset: usize,
    src_v_col_offset: usize,
    positions_dev: &Tensor,
    k_ptrs_dev: *mut u64,
    v_ptrs_dev: *mut u64,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    if batch_size == 0 { return Ok(()); }
    let stream = CudaConfig::resolve_stream(cuda_config);
    let pos_ptr = positions_dev.as_i32()?.buffer().as_ptr() as *const i32;
    match dtype {
        crate::base::DataType::BF16 => {
            let k_base = src_k.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let v_base = src_v.as_bf16()?.buffer().as_ptr() as *const half::bf16;
            let src_k_ptr = unsafe { k_base.add(src_k_col_offset) };
            let src_v_ptr = unsafe { v_base.add(src_v_col_offset) };
            unsafe {
                scatter_kv_batch_kernel_bf16(
                    k_ptrs_dev as *mut *mut half::bf16,
                    v_ptrs_dev as *mut *mut half::bf16,
                    src_k_ptr, src_v_ptr,
                    pos_ptr, batch_size as i32, kvdim as i32,
                    src_k_row_stride as i32, src_v_row_stride as i32,
                    stream,
                );
            }
        }
        crate::base::DataType::F16 => {
            let k_base = src_k.as_f16()?.buffer().as_ptr() as *const half::f16;
            let v_base = src_v.as_f16()?.buffer().as_ptr() as *const half::f16;
            let src_k_ptr = unsafe { k_base.add(src_k_col_offset) };
            let src_v_ptr = unsafe { v_base.add(src_v_col_offset) };
            unsafe {
                scatter_kv_batch_kernel_fp16(
                    k_ptrs_dev as *mut *mut half::f16,
                    v_ptrs_dev as *mut *mut half::f16,
                    src_k_ptr, src_v_ptr,
                    pos_ptr, batch_size as i32, kvdim as i32,
                    src_k_row_stride as i32, src_v_row_stride as i32,
                    stream,
                );
            }
        }
        _ => return Err(Error::InvalidArgument(format!(
            "Unsupported data type for scatter_kv_batch_launch_ready: {:?}", dtype
        )).into()),
    }
    Ok(())
}
