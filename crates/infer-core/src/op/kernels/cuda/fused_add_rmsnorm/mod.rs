use crate::base::error::Result;
use crate::cuda::config::CudaConfig;
use crate::tensor::Tensor;

unsafe extern "C" {
    fn fused_add_rmsnorm_kernel_cu_bf16(
        norm_output: *mut half::bf16,
        residual: *mut half::bf16,
        input: *const half::bf16,
        weight: *const half::bf16,
        rows: i32,
        dim: i32,
        eps: f32,
        stream: crate::cuda::ffi::cudaStream_t,
    );
}

/// Fused: residual += input; norm_output = rmsnorm(residual, weight)
/// BF16 only. Saves one kernel launch vs separate add + rmsnorm.
pub fn fused_add_rmsnorm(
    norm_output: &mut Tensor,
    residual: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    cuda_config: Option<&CudaConfig>,
) -> Result<()> {
    let dim = weight.shape()[0];
    let rows = input.num_elements() / dim;
    let stream = cuda_config.map_or(std::ptr::null_mut(), |config| config.stream);

    let norm_out_ptr = norm_output.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
    let res_ptr = residual.as_bf16_mut()?.buffer_mut().as_mut_ptr() as *mut half::bf16;
    let input_ptr = input.as_bf16()?.buffer().as_ptr() as *const half::bf16;
    let weight_ptr = weight.as_bf16()?.buffer().as_ptr() as *const half::bf16;

    unsafe {
        fused_add_rmsnorm_kernel_cu_bf16(
            norm_out_ptr,
            res_ptr,
            input_ptr,
            weight_ptr,
            rows as i32,
            dim as i32,
            eps,
            stream,
        );
    }
    Ok(())
}
