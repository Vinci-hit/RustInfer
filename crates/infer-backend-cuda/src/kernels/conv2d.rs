//! Conv2D via cuDNN.
//!
//! Uses cudnnConvolutionForward with pre-allocated workspace from CudaConfig.
//! Descriptor creation is done per-call (caching can be added later for perf).

use crate::Cuda;
use crate::ffi::{self, cudaStream_t};
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

/// Conv2D forward using cuDNN.
/// input: [N, Cin, H, W], weight: [Cout, Cin, Kh, Kw], output: [N, Cout, Hout, Wout]
pub fn conv2d<T: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    bias: Option<&Tensor<T, Cuda>>,
    output: &mut Tensor<T, Cuda>,
    stride: usize,
    padding: usize,
) -> OpResult<()> {
    let config = &input.device().config;
    let cudnn = config.cudnn_handle;
    check_cudnn(unsafe { ffi::cudnnSetStream(cudnn, stream) })?;

    let i_shape = input.shape().as_slice();
    let w_shape = weight.shape().as_slice();
    let o_shape = output.shape().as_slice();

    let (n, c_in, h_in, w_in) = (
        i_shape[0] as i32,
        i_shape[1] as i32,
        i_shape[2] as i32,
        i_shape[3] as i32,
    );
    let (c_out, _c_in2, kh, kw) = (
        w_shape[0] as i32,
        w_shape[1] as i32,
        w_shape[2] as i32,
        w_shape[3] as i32,
    );
    let (h_out, w_out) = (o_shape[2] as i32, o_shape[3] as i32);

    let cudnn_dtype = match T::DATA_TYPE {
        DataType::F32 => ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
        DataType::F16 => ffi::cudnnDataType_t::CUDNN_DATA_HALF,
        DataType::BF16 => ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        _ => {
            return Err(OpError::Kernel(format!(
                "conv2d cuDNN: unsupported dtype {:?}",
                T::DATA_TYPE
            )));
        }
    };

    unsafe {
        // Create descriptors
        let mut input_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
        let mut output_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
        let mut filter_desc: ffi::cudnnFilterDescriptor_t = std::ptr::null_mut();
        let mut conv_desc: ffi::cudnnConvolutionDescriptor_t = std::ptr::null_mut();

        check_cudnn(ffi::cudnnCreateTensorDescriptor(&mut input_desc))?;
        check_cudnn(ffi::cudnnCreateTensorDescriptor(&mut output_desc))?;
        check_cudnn(ffi::cudnnCreateFilterDescriptor(&mut filter_desc))?;
        check_cudnn(ffi::cudnnCreateConvolutionDescriptor(&mut conv_desc))?;

        check_cudnn(ffi::cudnnSetTensor4dDescriptor(
            input_desc,
            ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_dtype,
            n,
            c_in,
            h_in,
            w_in,
        ))?;
        check_cudnn(ffi::cudnnSetTensor4dDescriptor(
            output_desc,
            ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            cudnn_dtype,
            n,
            c_out,
            h_out,
            w_out,
        ))?;
        check_cudnn(ffi::cudnnSetFilter4dDescriptor(
            filter_desc,
            cudnn_dtype,
            ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            c_out,
            c_in,
            kh,
            kw,
        ))?;

        // cuDNN accumulates every supported input dtype in f32.
        let compute_type = ffi::cudnnDataType_t::CUDNN_DATA_FLOAT;
        check_cudnn(ffi::cudnnSetConvolution2dDescriptor(
            conv_desc,
            padding as i32,
            padding as i32,
            stride as i32,
            stride as i32,
            1,
            1,
            ffi::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            compute_type,
        ))?;
        check_cudnn(ffi::cudnnSetConvolutionMathType(
            conv_desc,
            ffi::cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        ))?;

        // Use IMPLICIT_GEMM as default algo
        let algo = ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

        // Get workspace size
        let mut ws_size: usize = 0;
        check_cudnn(ffi::cudnnGetConvolutionForwardWorkspaceSize(
            cudnn,
            input_desc,
            filter_desc,
            conv_desc,
            output_desc,
            algo,
            &mut ws_size,
        ))?;

        // Use the pre-allocated workspace from CudaConfig
        let ws_ptr = if ws_size <= config.workspace_size {
            config.workspace
        } else {
            // Fallback: use algo that needs no workspace
            return Err(OpError::Kernel(format!(
                "conv2d: workspace {} > available {}",
                ws_size, config.workspace_size
            )));
        };

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        check_cudnn(ffi::cudnnConvolutionForward(
            cudnn,
            &alpha as *const f32 as *const std::ffi::c_void,
            input_desc,
            input.data_ptr() as *const std::ffi::c_void,
            filter_desc,
            weight.data_ptr() as *const std::ffi::c_void,
            conv_desc,
            algo,
            ws_ptr,
            ws_size,
            &beta as *const f32 as *const std::ffi::c_void,
            output_desc,
            output.data_ptr_mut() as *mut std::ffi::c_void,
        ))?;

        // Add bias if present
        if let Some(b) = bias {
            let mut bias_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
            check_cudnn(ffi::cudnnCreateTensorDescriptor(&mut bias_desc))?;
            check_cudnn(ffi::cudnnSetTensor4dDescriptor(
                bias_desc,
                ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                cudnn_dtype,
                1,
                c_out,
                1,
                1,
            ))?;
            let one: f32 = 1.0;
            check_cudnn(ffi::cudnnAddTensor(
                cudnn,
                &one as *const f32 as *const std::ffi::c_void,
                bias_desc,
                b.data_ptr() as *const std::ffi::c_void,
                &one as *const f32 as *const std::ffi::c_void,
                output_desc,
                output.data_ptr_mut() as *mut std::ffi::c_void,
            ))?;
            ffi::cudnnDestroyTensorDescriptor(bias_desc);
        }

        // Cleanup descriptors
        ffi::cudnnDestroyTensorDescriptor(input_desc);
        ffi::cudnnDestroyTensorDescriptor(output_desc);
        ffi::cudnnDestroyFilterDescriptor(filter_desc);
        ffi::cudnnDestroyConvolutionDescriptor(conv_desc);
    }
    Ok(())
}

fn check_cudnn(status: ffi::cudnnStatus_t) -> OpResult<()> {
    if status != ffi::cudnnStatus_t::CUDNN_STATUS_SUCCESS {
        Err(OpError::Kernel(format!("cuDNN error: {:?}", status)))
    } else {
        Ok(())
    }
}
