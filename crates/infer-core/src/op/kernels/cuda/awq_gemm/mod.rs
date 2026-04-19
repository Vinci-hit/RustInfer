use crate::base::error::Result;
use crate::cuda::{self, CudaConfig};
use crate::tensor::Tensor;

// --- FFI 声明 (AWQ N-packed, 转置后的布局) ---
unsafe extern "C" {
    fn awq_gemv_cu(
        input: *const std::ffi::c_void,
        qweight_t: *const std::ffi::c_void,   // [N/8, K]
        qzeros_t: *const std::ffi::c_void,    // [N/8, num_groups]
        scales_t: *const std::ffi::c_void,     // [N, num_groups]
        output: *mut std::ffi::c_void,
        N: i32,
        K: i32,
        group_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    fn awq_gemm_cu(
        input: *const std::ffi::c_void,
        qweight_t: *const std::ffi::c_void,
        qzeros_t: *const std::ffi::c_void,
        scales_t: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        M: i32,
        N: i32,
        K: i32,
        group_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );

    // --- FFI 声明 (K-packed compressed-tensors 格式, BF16) ---
    fn kpack_gemv_cu(
        input: *const std::ffi::c_void,
        weight_packed: *const std::ffi::c_void,       // [N, K/8]
        weight_zero_point: *const std::ffi::c_void,   // [N/8, num_groups]
        weight_scale: *const std::ffi::c_void,         // [N, num_groups]
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

/// AWQ INT4 GEMV (decode, M=1) — N-packed format, FP16
/// input: [1, K] (FP16)
/// qweight_t: [N/8, K] (I32, transposed)
/// output: [1, N] (FP16)
pub fn awq_gemv(
    input: &Tensor,
    qweight_t: &Tensor,
    qzeros_t: &Tensor,
    scales_t: &Tensor,
    group_size: usize,
    output: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let qweight_shape = qweight_t.shape();
    // qweight_t shape: [N/8, K]
    let n = (qweight_shape[0] * 8) as i32;    // out_features
    let k = qweight_shape[1] as i32;          // in_features

    let stream = cuda_config.map_or(std::ptr::null_mut(), |c| c.stream);

    unsafe {
        awq_gemv_cu(
            input.buffer().as_ptr() as *const std::ffi::c_void,
            qweight_t.buffer().as_ptr() as *const std::ffi::c_void,
            qzeros_t.buffer().as_ptr() as *const std::ffi::c_void,
            scales_t.buffer().as_ptr() as *const std::ffi::c_void,
            output.buffer_mut().as_mut_ptr() as *mut std::ffi::c_void,
            n,
            k,
            group_size as i32,
            stream,
        );
    }

    Ok(())
}

/// AWQ INT4 GEMM (prefill, M>1) — N-packed format, FP16
/// input: [M, K] (FP16)
/// qweight_t: [N/8, K] (I32, transposed)
/// output: [M, N] (FP16)
pub fn awq_gemm(
    input: &Tensor,
    qweight_t: &Tensor,
    qzeros_t: &Tensor,
    scales_t: &Tensor,
    group_size: usize,
    output: &mut Tensor,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let input_shape = input.shape();
    let qweight_shape = qweight_t.shape();

    let m = input_shape[0] as i32;
    let n = (qweight_shape[0] * 8) as i32;
    let k = qweight_shape[1] as i32;

    let stream = cuda_config.map_or(std::ptr::null_mut(), |c| c.stream);

    unsafe {
        awq_gemm_cu(
            input.buffer().as_ptr() as *const std::ffi::c_void,
            qweight_t.buffer().as_ptr() as *const std::ffi::c_void,
            qzeros_t.buffer().as_ptr() as *const std::ffi::c_void,
            scales_t.buffer().as_ptr() as *const std::ffi::c_void,
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

/// K-packed INT4 GEMV (decode, M=1) — compressed-tensors format, BF16
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
    // weight_packed shape: [N, K/8]
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

/// K-packed INT4 GEMM (prefill, M>1) — compressed-tensors format, BF16
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
