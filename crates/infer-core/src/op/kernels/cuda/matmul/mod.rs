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
}

pub fn hgemm_bf16(input: &Tensor, weight: &Tensor, output: &mut Tensor, cuda_config:Option<&CudaConfig>) -> Result<()> {
    let a_shape = input.shape();
    let b_shape = weight.shape();

    let m = b_shape[0];
    let k = b_shape[1];
    let n = a_shape[0];
    
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
    // **注意**: 我们的 sgemm 内核不支持转置，所以 B 必须已经是 W^T
    // 在实践中，我们会使用 cuBLAS，它支持转置。
    // 为了使用您的 naive_kernel，我们需要一个已经转置好的 weight。
    // 这里我们假设 weight 就是 B，而不是 W。
    let b_ptr = weight.as_f32()?.buffer().as_ptr() as *const f32;
    let c_ptr = output.as_f32_mut()?.buffer_mut().as_mut_ptr() as *mut f32;

    unsafe {
        sgemm_naive_f32_cu(a_ptr, b_ptr, c_ptr, n as i32, m as i32, k as i32, stream);
    }
    Ok(())
}