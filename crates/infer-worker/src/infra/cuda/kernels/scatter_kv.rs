//! Scatter K/V into cache CUDA kernel.

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::Dtype;
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi;

/// Scatter K/V rows to cache at given positions.
/// k: [num_tokens, kv_dim], v: [num_tokens, kv_dim]
/// k_cache/v_cache: [capacity, kv_dim]
/// positions: [num_tokens] — device-resident position indices.
pub fn scatter_kv<T: Dtype>(
    k: &Tensor<T, Cuda>,
    v: &Tensor<T, Cuda>,
    k_cache: &mut Tensor<T, Cuda>,
    v_cache: &mut Tensor<T, Cuda>,
    positions: &Tensor<i32, Cuda>,
    kv_dim: usize,
) -> OpResult<()> {
    let num_tokens = k.numel() / kv_dim;
    let stream = k.device().config.stream;
    let capacity = k_cache.numel() / kv_dim;

    let elem = T::SIZE_BYTES;
    let row_bytes = kv_dim * elem;

    for t in 0..num_tokens {
        // Read position from device
        let mut pos: i32 = 0;
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                &mut pos as *mut i32 as *mut std::ffi::c_void,
                positions.data_ptr().add(t) as *const std::ffi::c_void,
                4,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("scatter_kv memcpy pos failed: {:?}", code)));
            }
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("scatter_kv sync failed: {:?}", code)));
            }
        }

        let pos = pos as usize;
        if pos >= capacity {
            return Err(OpError::Shape(format!("scatter_kv: pos {} >= capacity {}", pos, capacity)));
        }

        unsafe {
            // K: copy row t → cache[pos]
            let k_src = (k.data_ptr() as *const u8).add(t * row_bytes);
            let k_dst = (k_cache.data_ptr_mut() as *mut u8).add(pos * row_bytes);
            let code = ffi::cudaMemcpyAsync(
                k_dst as *mut std::ffi::c_void,
                k_src as *const std::ffi::c_void,
                row_bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("scatter_kv k copy failed: {:?}", code)));
            }

            // V: copy row t → cache[pos]
            let v_src = (v.data_ptr() as *const u8).add(t * row_bytes);
            let v_dst = (v_cache.data_ptr_mut() as *mut u8).add(pos * row_bytes);
            let code = ffi::cudaMemcpyAsync(
                v_dst as *mut std::ffi::c_void,
                v_src as *const std::ffi::c_void,
                row_bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!("scatter_kv v copy failed: {:?}", code)));
            }
        }
    }
    Ok(())
}

