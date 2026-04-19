use std::os::raw::c_void;

use crate::base::DataType;
use crate::base::error::{Error, Result};
use crate::tensor::Tensor;
use crate::cuda::{self, CudaConfig};

// --- FFI 声明 ---
unsafe extern "C" {
    pub fn sgemv_cu_fp32x4(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        M: i32,
        K: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    // SGEMM
    fn sgemm_naive_f32_cu(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        M: i32,
        N: i32,
        K: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
    // SGEMM BF16
    fn gemm_cublaslt_bf16(
        a: *const half::bf16,
        b: *const half::bf16,
        c: *mut half::bf16,
        M: i32,
        N: i32,
        K: i32,
        stream: crate::cuda::ffi::cudaStream_t,
        handle: cuda::ffi::cublasLtHandle_t,
        workspace: *mut c_void,
        workspaceSize: usize,
    );
    // BF16 GEMV for decode (M=1)
    fn hgemv_bf16_cu(
        input: *const half::bf16,
        weight: *const half::bf16,
        output: *mut half::bf16,
        N: i32,
        K: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
    // FP16 GEMV
    fn hgemv_fp16_cu(
        input: *const half::f16,
        weight: *const half::f16,
        output: *mut half::f16,
        N: i32,
        K: i32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
    // FP16 GEMM via cublasLt
    fn gemm_cublaslt_fp16(
        a: *const half::f16,
        b: *const half::f16,
        c: *mut half::f16,
        M: i32,
        N: i32,
        K: i32,
        stream: crate::cuda::ffi::cudaStream_t,
        handle: cuda::ffi::cublasLtHandle_t,
        workspace: *mut c_void,
        workspaceSize: usize,
    );

    // INT4 quantized GEMV/GEMM (K-packed, BF16)
    fn kpack_gemv_cu(
        input: *const std::ffi::c_void,
        weight_packed: *const std::ffi::c_void,
        weight_zero_point: *const std::ffi::c_void,
        weight_scale: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        N: i32,
        K: i32,
        group_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn kpack_gemm_cu(
        input: *const std::ffi::c_void,
        weight_packed: *const std::ffi::c_void,
        weight_zero_point: *const std::ffi::c_void,
        weight_scale: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        M: i32,
        N: i32,
        K: i32,
        group_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}

/// BF16 GEMV for decode phase (M=1): y = W * x
/// input: [1, K], weight: [N, K], output: [1, N]
/// Uses custom CUDA kernel with bf16x8 vectorized loads, FP32 accumulation,
/// and warp shuffle reduction. ~1.5x faster than cublasLt for M=1.
pub fn hgemv_bf16(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let a_shape = input.shape();
    let b_shape = weight.shape();

    let k = a_shape[1];     // inner dimension
    let n = b_shape[0];     // output dimension (rows of weight)

    let input_ptr = input.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let weight_ptr = weight.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let output_ptr = output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    unsafe {
        hgemv_bf16_cu(
            input_ptr,
            weight_ptr,
            output_ptr,
            n as i32,
            k as i32,
            stream,
        );
    }

    Ok(())
}

pub fn hgemm_bf16(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config:Option<&CudaConfig>) -> Result<()> {
    let a_shape = input.shape();
    let b_shape = weight.shape();

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    
    // (这里可以添加形状检查)

    let a_ptr = input.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    // **注意**: 我们的 sgemm 内核不支持转置，所以 B 必须已经是 W^T
    // 在实践中，我们会使用 cuBLAS，它支持转置。
    // 为了使用您的 naive_kernel，我们需要一个已经转置好的 weight。
    // 这里我们假设 weight 就是 B，而不是 W。
    let b_ptr = weight.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let c_ptr = output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);
    let cublaslt_handle = cuda_config.map_or(std::ptr::null_mut(), |config| config.cublaslt_handle);
    let workspace = cuda_config.map_or(std::ptr::null_mut(), |config| config.workspace);
    let workspace_size = cuda_config.map_or(0, |config| config.workspace_size);
    
    unsafe {
        gemm_cublaslt_bf16(
            a_ptr,
            b_ptr,
            c_ptr,
            m as i32,
            n as i32,
            k as i32,
            stream,
            cublaslt_handle,
            workspace,
            workspace_size,
        );
    }
    
    Ok(())
}

/// FP16 GEMV for decode phase (M=1): y = W * x
pub fn hgemv_fp16(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let qweight_shape = weight.shape();
    let n = qweight_shape[0] as i32;
    let k = qweight_shape[1] as i32;
    let input_ptr = input.as_f16()?.buffer().as_ptr() as *const half::f16;
    let weight_ptr = weight.as_f16()?.buffer().as_ptr() as *const half::f16;
    let output_ptr = output.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);
    unsafe { hgemv_fp16_cu(input_ptr, weight_ptr, output_ptr, n, k, stream); }
    Ok(())
}

/// FP16 GEMM via cublasLt
pub fn hgemm_fp16(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    let a_shape = input.shape();
    let b_shape = weight.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    let a_ptr = input.as_f16()?.buffer().as_ptr() as *const half::f16;
    let b_ptr = weight.as_f16()?.buffer().as_ptr() as *const half::f16;
    let c_ptr = output.as_f16_mut()?.buffer_mut().as_mut_ptr() as *mut half::f16;
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);
    let cublaslt_handle = cuda_config.map_or(std::ptr::null_mut(), |config| config.cublaslt_handle);
    let workspace = cuda_config.map_or(std::ptr::null_mut(), |config| config.workspace);
    let workspace_size = cuda_config.map_or(0, |config| config.workspace_size);
    unsafe {
        gemm_cublaslt_fp16(a_ptr, b_ptr, c_ptr, m as i32, n as i32, k as i32,
            stream, cublaslt_handle, workspace, workspace_size);
    }
    Ok(())
}

/// SGEMV: y = A * x 的 CUDA 内核包装函数
pub fn sgemv(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config:Option<&CudaConfig>) -> Result<()> {
    if input.dtype() == DataType::BF16 {
        return hgemm_bf16(input, weight, output, cuda_config);
    }
    // --- 1. 获取具体类型和指针 ---
    let a_typed = weight.as_f32()?;
    let x_typed = input.as_f32()?;
    let y_typed = output.as_f32_mut()?;
    
    let a_ptr = a_typed.buffer().as_ptr() as *const f32;
    let x_ptr = x_typed.buffer().as_ptr() as *const f32;
    let y_ptr = y_typed.buffer_mut().as_mut_ptr() as *mut f32;

    // --- 2. 形状检查和维度计算 ---
    let a_shape = weight.shape();
    
    let k = a_shape[0];
    let m = a_shape[1];

    if !m.is_multiple_of(4) {
        return Err(Error::InvalidArgument("SGEMV float4 kernel requires the inner dimension (N) to be a multiple of 4.".into()).into());
    }

    // --- 3. 获取 CUDA stream ---
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    // --- 4. 调用 FFI 函数 ---
    unsafe {
        sgemv_cu_fp32x4(
            x_ptr,
            a_ptr,
            y_ptr,
            m as i32,
            k as i32,
            stream,
        );
    }
    
    Ok(())
}

/// CUDA SGEMM (矩阵-矩阵乘法): Y = X * W^T
/// 在我们的场景中: C=A*B, A=X, B=W^T, C=Y
/// A(X): [seq_len, dim], B(W^T): [dim, vocab_size], C(Y): [seq_len, vocab_size]
pub fn sgemm(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config: Option<&CudaConfig>) -> Result<()> {
    if input.dtype() == DataType::BF16 {
        return hgemm_bf16(input, weight, output, cuda_config);
    }
    let stream = cuda_config.map_or(std::ptr::null_mut(), |c| c.stream);
    let a_shape = input.shape();
    let b_shape = weight.shape();

    let m = b_shape[0];
    let k = b_shape[1];
    let n = a_shape[0];
    
    // (这里可以添加形状检查)

    let a_ptr = input.as_f32()?.buffer().as_ptr() as *const f32;
    let b_ptr = weight.as_f32()?.buffer().as_ptr() as *const f32;
    let c_ptr = output.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;

    unsafe {
        sgemm_naive_f32_cu(a_ptr, b_ptr, c_ptr, n as i32, m as i32, k as i32, stream);
    }
    Ok(())
}

// ============================================================================
//  INT4 Quantized GEMV / GEMM (K-packed, BF16)
// ============================================================================

/// INT4 GEMV (decode, M=1) — K-packed format, BF16
/// input: [1, K] (BF16)
/// weight_packed: [N, K/8] (I32)
/// output: [1, N] (BF16)
pub fn kpack_gemv(
    input: &Tensor,
    weight_packed: &Tensor,
    weight_zero_point: &Tensor,
    weight_scale: &Tensor,
    group_size: usize,
    output: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let wp_shape = weight_packed.shape();
    let n = wp_shape[0] as i32;
    let k = (wp_shape[1] * 8) as i32;

    let stream = cuda_config.map_or(std::ptr::null_mut(), |c| c.stream);

    unsafe {
        kpack_gemv_cu(
            input.buffer().as_ptr() as *const std::ffi::c_void,
            weight_packed.buffer().as_ptr() as *const std::ffi::c_void,
            weight_zero_point.buffer().as_ptr() as *const std::ffi::c_void,
            weight_scale.buffer().as_ptr() as *const std::ffi::c_void,
            output.buffer_mut().as_mut_ptr() as *mut std::ffi::c_void,
            n,
            k,
            group_size as i32,
            stream,
        );
    }

    Ok(())
}

/// INT4 GEMM (prefill, M>1) — K-packed format, BF16
/// input: [M, K] (BF16)
/// weight_packed: [N, K/8] (I32)
/// output: [M, N] (BF16)
pub fn kpack_gemm(
    input: &Tensor,
    weight_packed: &Tensor,
    weight_zero_point: &Tensor,
    weight_scale: &Tensor,
    group_size: usize,
    output: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let input_shape = input.shape();
    let wp_shape = weight_packed.shape();

    let m = input_shape[0] as i32;
    let n = wp_shape[0] as i32;
    let k = (wp_shape[1] * 8) as i32;

    let stream = cuda_config.map_or(std::ptr::null_mut(), |c| c.stream);

    unsafe {
        kpack_gemm_cu(
            input.buffer().as_ptr() as *const std::ffi::c_void,
            weight_packed.buffer().as_ptr() as *const std::ffi::c_void,
            weight_zero_point.buffer().as_ptr() as *const std::ffi::c_void,
            weight_scale.buffer().as_ptr() as *const std::ffi::c_void,
            output.buffer_mut().as_mut_ptr() as *mut std::ffi::c_void,
            m,
            n,
            k,
            group_size as i32,
            stream,
        );
    }

    Ok(())
}