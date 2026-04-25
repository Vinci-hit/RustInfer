use std::os::raw::c_void;

use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::{ffi, CudaConfig};

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

/// Conv2d via cuDNN: input[B,Cin,H,W] * weight[Cout,Cin,kH,kW] + bias[Cout]
///
/// 使用 cuDNN legacy API (cudnnConvolutionForward)。
/// cuDNN 会自动选择最优算法（IMPLICIT_GEMM / WINOGRAD / FFT 等）。
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

    let in_shape = input.shape();   // [B, Cin, H, W]
    let w_shape = weight.shape();   // [Cout, Cin, kH, kW]
    let out_shape = output.shape(); // [B, Cout, Hout, Wout]

    let batch = in_shape[0] as i32;
    let c_in = in_shape[1] as i32;
    let h_in = in_shape[2] as i32;
    let w_in = in_shape[3] as i32;
    let c_out = w_shape[0] as i32;
    let kh = w_shape[2] as i32;
    let kw = w_shape[3] as i32;
    let h_out = out_shape[2] as i32;
    let w_out = out_shape[3] as i32;
    let stride_i = stride as i32;
    let padding_i = padding as i32;

    // 确定 cuDNN 数据类型
    let (cudnn_dtype, compute_type) = match input.dtype() {
        crate::base::DataType::F32 => (
            ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
            ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
        ),
        crate::base::DataType::F16 => (
            ffi::cudnnDataType_t::CUDNN_DATA_HALF,
            ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
        ),
        crate::base::DataType::BF16 => (
            ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
            ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
        ),
        other => return Err(Error::InvalidArgument(format!(
            "cuDNN conv2d: unsupported dtype {:?}", other
        )).into()),
    };

    let format = ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;

    unsafe {
        // --- 创建 descriptors ---
        let mut input_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
        let mut output_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
        let mut filter_desc: ffi::cudnnFilterDescriptor_t = std::ptr::null_mut();
        let mut conv_desc: ffi::cudnnConvolutionDescriptor_t = std::ptr::null_mut();

        cudnn_check!(ffi::cudnnCreateTensorDescriptor(&mut input_desc));
        cudnn_check!(ffi::cudnnCreateTensorDescriptor(&mut output_desc));
        cudnn_check!(ffi::cudnnCreateFilterDescriptor(&mut filter_desc));
        cudnn_check!(ffi::cudnnCreateConvolutionDescriptor(&mut conv_desc));

        // 设置 descriptors
        cudnn_check!(ffi::cudnnSetTensor4dDescriptor(
            input_desc, format, cudnn_dtype, batch, c_in, h_in, w_in
        ));
        cudnn_check!(ffi::cudnnSetTensor4dDescriptor(
            output_desc, format, cudnn_dtype, batch, c_out, h_out, w_out
        ));
        cudnn_check!(ffi::cudnnSetFilter4dDescriptor(
            filter_desc, cudnn_dtype, format, c_out, c_in, kh, kw
        ));
        cudnn_check!(ffi::cudnnSetConvolution2dDescriptor(
            conv_desc, padding_i, padding_i, stride_i, stride_i,
            1, 1, // dilation
            ffi::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            compute_type,
        ));
        // 允许 Tensor Core
        cudnn_check!(ffi::cudnnSetConvolutionMathType(
            conv_desc, ffi::cudnnMathType_t::CUDNN_DEFAULT_MATH
        ));

        // --- 选择算法 ---
        let mut perf_results: [ffi::cudnnConvolutionFwdAlgoPerf_t; 1] = std::mem::zeroed();
        let mut returned_algo_count: i32 = 0;
        cudnn_check!(ffi::cudnnGetConvolutionForwardAlgorithm_v7(
            handle,
            input_desc,
            filter_desc,
            conv_desc,
            output_desc,
            1, // requested algo count
            &mut returned_algo_count,
            perf_results.as_mut_ptr(),
        ));
        let algo = perf_results[0].algo;

        // --- 获取 workspace 大小 ---
        let mut ws_size: usize = 0;
        cudnn_check!(ffi::cudnnGetConvolutionForwardWorkspaceSize(
            handle, input_desc, filter_desc, conv_desc, output_desc, algo, &mut ws_size,
        ));

        // 分配 workspace（如果需要）
        let mut ws_ptr: *mut c_void = std::ptr::null_mut();
        if ws_size > 0 {
            crate::cuda_check!(ffi::cudaMalloc(&mut ws_ptr, ws_size))?;
        }

        // --- 执行卷积 ---
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let in_ptr = input.buffer().as_ptr() as *const c_void;
        let w_ptr = weight.buffer().as_ptr() as *const c_void;
        let out_ptr = output.buffer_mut().as_mut_ptr() as *mut c_void;

        cudnn_check!(ffi::cudnnConvolutionForward(
            handle,
            &alpha as *const f32 as *const c_void,
            input_desc,
            in_ptr,
            filter_desc,
            w_ptr,
            conv_desc,
            algo,
            ws_ptr,
            ws_size,
            &beta as *const f32 as *const c_void,
            output_desc,
            out_ptr,
        ));

        // --- 加 bias ---
        if let Some(bias_t) = bias {
            // bias descriptor: [1, Cout, 1, 1]
            let mut bias_desc: ffi::cudnnTensorDescriptor_t = std::ptr::null_mut();
            cudnn_check!(ffi::cudnnCreateTensorDescriptor(&mut bias_desc));
            cudnn_check!(ffi::cudnnSetTensor4dDescriptor(
                bias_desc, format, cudnn_dtype, 1, c_out, 1, 1
            ));

            let alpha_bias: f32 = 1.0;
            let bias_ptr = bias_t.buffer().as_ptr() as *const c_void;
            cudnn_check!(ffi::cudnnAddTensor(
                handle,
                &alpha_bias as *const f32 as *const c_void,
                bias_desc,
                bias_ptr,
                &alpha_bias as *const f32 as *const c_void,
                output_desc,
                out_ptr,
            ));
            ffi::cudnnDestroyTensorDescriptor(bias_desc);
        }

        // --- 清理 ---
        if !ws_ptr.is_null() {
            let _ = ffi::cudaFree(ws_ptr);
        }
        ffi::cudnnDestroyTensorDescriptor(input_desc);
        ffi::cudnnDestroyTensorDescriptor(output_desc);
        ffi::cudnnDestroyFilterDescriptor(filter_desc);
        ffi::cudnnDestroyConvolutionDescriptor(conv_desc);
    }

    Ok(())
}
