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

    fn scatter_kernel_f32(
        dst: *mut f32,
        src: *const f32,
        pos: *const i32,
        kvdim: i32,
        max_seq_len: i32,
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
