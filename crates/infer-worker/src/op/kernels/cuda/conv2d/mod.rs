use std::os::raw::c_void;

use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::{ffi, CudaConfig};

mod cache;
pub use cache::Conv2dCache;
use cache::ConvKey;

/// cuDNN 状态检查宏
macro_rules! cudnn_check {
    ($expr:expr) => {{
        let status = $expr;
        if status != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(Error::InvalidArgument(format!(
                "cuDNN error: {:?}", status
            )).into());
        }
    }};
}

/// Conv2d via cuDNN: `input[B, Cin, H, W] * weight[Cout, Cin, kH, kW] + bias[Cout]`.
///
/// Uses `cuda_config`'s per-handle [`Conv2dCache`] to amortize cuDNN
/// descriptor creation, algorithm selection, and workspace allocation
/// across all calls with the same `(input shape, weight shape, stride,
/// padding, dtype, has_bias)` tuple. This collapses per-call host
/// overhead from several milliseconds down to a single `cudnnConvolutionForward`
/// (+ optional `cudnnAddTensor`) launch once the cache is warmed up.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_cudnn(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    output: &mut Tensor,
    stride: usize,
    padding: usize,
    cuda_config: &CudaConfig,
) -> Result<()> {
    let handle = cuda_config.cudnn_handle;
    if handle.is_null() {
        return Err(Error::InvalidArgument("cuDNN handle is null".into()).into());
    }

    let in_shape = input.shape();
    let w_shape = weight.shape();
    let out_shape = output.shape();

    let key = ConvKey {
        in_shape: [in_shape[0] as i32, in_shape[1] as i32, in_shape[2] as i32, in_shape[3] as i32],
        w_shape: [w_shape[0] as i32, w_shape[1] as i32, w_shape[2] as i32, w_shape[3] as i32],
        stride: stride as i32,
        padding: padding as i32,
        dtype: input.dtype(),
        has_bias: bias.is_some(),
    };
    let out_shape_arr = [
        out_shape[0] as i32, out_shape[1] as i32,
        out_shape[2] as i32, out_shape[3] as i32,
    ];

    // Populate (if needed) and read back the cached descriptors / algo /
    // workspace requirement, then ensure the shared workspace is at least
    // as large as the largest entry we've seen so far.
    let mut cache = cuda_config.conv2d_cache.lock()
        .map_err(|_| Error::InvalidArgument("conv2d_cache mutex poisoned".into()))?;
    let (input_desc, output_desc, filter_desc, conv_desc, bias_desc, algo, cudnn_dtype, ws_size);
    {
        let entry = cache.get_or_insert(key, handle, out_shape_arr)?;
        input_desc = entry.input_desc;
        output_desc = entry.output_desc;
        filter_desc = entry.filter_desc;
        conv_desc = entry.conv_desc;
        bias_desc = entry.bias_desc;
        algo = entry.algo;
        cudnn_dtype = entry.cudnn_dtype;
        ws_size = entry.ws_size;
    }
    cache.ensure_workspace(ws_size)?;
    let (ws_ptr, _) = cache.workspace();
    drop(cache); // release the mutex before the launch

    // cuDNN launch: alpha·conv(input, weight) + beta·output, then +bias.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let in_ptr = input.buffer().as_ptr() as *const c_void;
    let w_ptr = weight.buffer().as_ptr() as *const c_void;
    let out_ptr = output.buffer_mut().as_mut_ptr() as *mut c_void;

    unsafe {
        cudnn_check!(ffi::cudnnConvolutionForward(
            handle,
            &alpha as *const f32 as *const c_void,
            input_desc, in_ptr,
            filter_desc, w_ptr,
            conv_desc, algo,
            ws_ptr, ws_size,
            &beta as *const f32 as *const c_void,
            output_desc, out_ptr,
        ));

        if let (Some(bias_t), Some(bd)) = (bias, bias_desc) {
            let alpha_bias: f32 = 1.0;
            let bias_ptr = bias_t.buffer().as_ptr() as *const c_void;
            cudnn_check!(ffi::cudnnAddTensor(
                handle,
                &alpha_bias as *const f32 as *const c_void,
                bd, bias_ptr,
                &alpha_bias as *const f32 as *const c_void,
                output_desc, out_ptr,
            ));
        }
    }

    // `cudnn_dtype` is captured for diagnostic symmetry with the key;
    // no runtime behaviour depends on its value here.
    let _ = cudnn_dtype;
    Ok(())
}
