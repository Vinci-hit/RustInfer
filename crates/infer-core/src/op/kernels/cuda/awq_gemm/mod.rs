use crate::base::error::Result;
use crate::cuda::{self, CudaConfig};
use crate::tensor::Tensor;

// --- FFI 声明 (K-packed INT4, BF16) ---
unsafe extern "C" {
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
