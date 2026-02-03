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

    /// Scatter KV到paged cache的CUDA kernel (仅BF16)
    /// slot_mapping: [batch_size] - 每个请求应该写入的slot位置
    /// key/value: [batch_size, kv_dim] - 当前批次的K/V
    /// k_cache/v_cache: [num_blocks, block_size, kv_dim] - 全局paged cache
    fn scatter_kv_kernel_bf16(
        k_cache: *mut half::bf16,
        v_cache: *mut half::bf16,
        key: *const half::bf16,
        value: *const half::bf16,
        slot_mapping: *const i32,
        kv_dim: i32,
        block_size: i32,
        batch_size: i32,
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
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

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
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for scatter: {:?} (only BF16 supported)", dtype
            )).into());
        }
    }

    Ok(())
}

/// Scatter KV到paged KV cache
///
/// # Arguments
/// * `key` - Key张量 [batch_size, kv_dim]
/// * `value` - Value张量 [batch_size, kv_dim]
/// * `slot_mapping` - 槽位映射 [batch_size] - 每个K/V对应该写入的slot位置
/// * `k_cache` - Key cache张量 [num_blocks, block_size, kv_dim]
/// * `v_cache` - Value cache张量 [num_blocks, block_size, kv_dim]
/// * `cuda_config` - Optional CUDA configuration for stream
pub fn scatter_kv(
    key: &Tensor,
    value: &Tensor,
    slot_mapping: &Tensor,
    k_cache: &mut Tensor,
    v_cache: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    // Validate data types match
    let dtype = key.dtype();
    if value.dtype() != dtype || k_cache.dtype() != dtype || v_cache.dtype() != dtype {
        return Err(Error::InvalidArgument(format!(
            "Data type mismatch: key={:?}, value={:?}, k_cache={:?}, v_cache={:?}",
            dtype, value.dtype(), k_cache.dtype(), v_cache.dtype()
        )).into());
    }

    // Validate shapes
    let key_shape = key.shape();
    let value_shape = value.shape();
    let k_cache_shape = k_cache.shape();
    let v_cache_shape = v_cache.shape();
    let slot_shape = slot_mapping.shape();

    if key_shape.len() != 2 || value_shape.len() != 2 {
        return Err(Error::InvalidArgument(format!(
            "Key and Value must be 2D. key_shape={:?}, value_shape={:?}",
            key_shape, value_shape
        )).into());
    }

    let batch_size = key_shape[0];
    let kv_dim = key_shape[1];

    if value_shape[0] != batch_size || value_shape[1] != kv_dim {
        return Err(Error::InvalidArgument(format!(
            "Value shape mismatch. Expected [{}, {}], got {:?}",
            batch_size, kv_dim, value_shape
        )).into());
    }

    if slot_shape[0] != batch_size {
        return Err(Error::InvalidArgument(format!(
            "Slot mapping size mismatch. Expected {}, got {}",
            batch_size, slot_shape[0]
        )).into());
    }

    // Paged cache期望形状：[num_blocks, block_size, kv_dim]
    if k_cache_shape.len() != 3 || v_cache_shape.len() != 3 {
        return Err(Error::InvalidArgument(format!(
            "K/V cache must be 3D [num_blocks, block_size, kv_dim]. k_cache={:?}, v_cache={:?}",
            k_cache_shape, v_cache_shape
        )).into());
    }

    if k_cache_shape != v_cache_shape {
        return Err(Error::InvalidArgument(format!(
            "K and V cache shapes must match. k_cache={:?}, v_cache={:?}",
            k_cache_shape, v_cache_shape
        )).into());
    }

    let block_size = k_cache_shape[1];

    if k_cache_shape[2] != kv_dim {
        return Err(Error::InvalidArgument(format!(
            "KV dimension mismatch: expected {}, got {}",
            kv_dim, k_cache_shape[2]
        )).into());
    }

    // Get CUDA stream
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // Dispatch based on data type
    match dtype {
        crate::base::DataType::BF16 => {
            let k_cache_typed: &mut TypedTensor<half::bf16> = k_cache.as_bf16_mut()?;
            let v_cache_typed: &mut TypedTensor<half::bf16> = v_cache.as_bf16_mut()?;
            let key_typed = key.as_bf16()?;
            let value_typed = value.as_bf16()?;

            let k_cache_ptr = k_cache_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let v_cache_ptr = v_cache_typed.buffer_mut().as_mut_ptr() as *mut half::bf16;
            let key_ptr = key_typed.buffer().as_ptr() as *const half::bf16;
            let value_ptr = value_typed.buffer().as_ptr() as *const half::bf16;
            let slot_ptr = slot_mapping.as_i32()?.buffer().as_ptr() as *const i32;

            unsafe {
                scatter_kv_kernel_bf16(
                    k_cache_ptr,
                    v_cache_ptr,
                    key_ptr,
                    value_ptr,
                    slot_ptr,
                    kv_dim as i32,
                    block_size as i32,
                    batch_size as i32,
                    stream,
                );
            }
        }
        _ => {
            return Err(Error::InvalidArgument(format!(
                "Unsupported data type for scatter_kv: {:?} (only BF16 supported)", dtype
            )).into());
        }
    }

    Ok(())
}
