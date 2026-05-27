//! Matmul CUDA kernel wrapper — dispatches to GEMV/GEMM by dtype and shape.
//! Also provides AWQ int4 quantized matmul (kpack_gemv/kpack_gemm).

use crate::domain::ports::{OpResult, OpError};
use crate::domain::types::{DataType, Dtype};
use crate::domain::tensor::Tensor;
use crate::infra::cuda::Cuda;
use crate::infra::cuda::ffi::cudaStream_t;

unsafe extern "C" {
    fn sgemv_cu_fp32x4(
        input: *const f32, weight: *const f32, output: *mut f32,
        m: i32, k: i32, stream: cudaStream_t,
    );
    fn sgemm_naive_f32_cu(
        a: *const f32, b: *const f32, c: *mut f32,
        m: i32, n: i32, k: i32,
        stream: cudaStream_t,
    );
    fn hgemv_bf16_cu(
        input: *const half::bf16, weight: *const half::bf16, output: *mut half::bf16,
        n: i32, k: i32, stream: cudaStream_t,
    );
    fn gemm_cublaslt_bf16(
        a: *const half::bf16, b: *const half::bf16, c: *mut half::bf16,
        m: i32, n: i32, k: i32,
        stream: cudaStream_t,
        handle: crate::infra::cuda::ffi::cublasLtHandle_t,
        workspace: *mut std::ffi::c_void, workspace_size: usize,
    );
    // AWQ int4 quantized kernels
    fn kpack_gemv_cu(
        input: *const std::ffi::c_void, weight_packed: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void, scales: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        n: i32, k: i32, group_size: i32, stream: cudaStream_t,
    );
    fn kpack_gemm_cu(
        input: *const std::ffi::c_void, weight_packed: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void, scales: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32, n: i32, k: i32, group_size: i32, stream: cudaStream_t,
    );
}

/// Standard matmul: output = input @ weight^T (same dtype)
pub fn matmul<T: Dtype>(
    input: &Tensor<T, Cuda>,
    weight: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
) -> OpResult<()> {
    let in_shape = input.shape().as_slice();
    let w_shape = weight.shape().as_slice();
    if in_shape.len() < 2 || w_shape.len() < 2 {
        return Err(OpError::Shape("matmul: need 2D".into()));
    }
    let m = in_shape[0];
    let k = in_shape[1];
    let n = w_shape[0];

    let cfg = &input.device().config;
    let stream = cfg.stream;

    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => {
                if m == 1 {
                    sgemv_cu_fp32x4(
                        input.data_ptr() as _, weight.data_ptr() as _, output.data_ptr_mut() as _,
                        n as i32, k as i32, stream,
                    );
                } else {
                    sgemm_naive_f32_cu(
                        input.data_ptr() as _, weight.data_ptr() as _, output.data_ptr_mut() as _,
                        m as i32, n as i32, k as i32, stream,
                    );
                }
            }
            DataType::BF16 => {
                // GEMV vs GEMM threshold: hand-rolled bf16 GEMV (1 warp = 1
                // output row) wins on small/medium N (qkv_proj / o_proj /
                // gate_up / down_proj — N≈2k-12k). For very large N (lm_head
                // with vocab≈128k) cuBLASLt's parallel scheduling wins.
                // Threshold validated empirically (commit 6b64bfd).
                const GEMV_VS_GEMM_N_THRESHOLD: usize = 16_384;
                let force_gemm = std::env::var("RUSTINFER_FORCE_GEMM").is_ok();
                if m == 1 && n <= GEMV_VS_GEMM_N_THRESHOLD && !force_gemm {
                    hgemv_bf16_cu(
                        input.data_ptr() as _, weight.data_ptr() as _, output.data_ptr_mut() as _,
                        n as i32, k as i32, stream,
                    );
                } else {
                    gemm_cublaslt_bf16(
                        input.data_ptr() as _, weight.data_ptr() as _, output.data_ptr_mut() as _,
                        m as i32, n as i32, k as i32,
                        stream, cfg.cublaslt_handle, cfg.workspace, cfg.workspace_size,
                    );
                }
            }
            _ => return Err(OpError::Kernel(format!("matmul: unsupported dtype {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

/// AWQ int4 quantized matmul.
/// - input: activation [M, K] (A dtype, typically bf16)
/// - weight_packed: [N, K/8] (W dtype, i32 — 8 int4 values packed)
/// - scales: [N, num_groups] (A dtype)
/// - zeros: [N/8, num_groups] (W dtype, i32 packed)
/// - output: [M, N] (O dtype, typically bf16)
/// - group_size: quantization group size (e.g. 128)
pub fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
    input: &Tensor<A, Cuda>,
    weight_packed: &Tensor<W, Cuda>,
    output: &mut Tensor<O, Cuda>,
    scales: &Tensor<A, Cuda>,
    zeros: Option<&Tensor<W, Cuda>>,
    group_size: usize,
) -> OpResult<()> {
    let wp_shape = weight_packed.shape().as_slice();
    let n = wp_shape[0];
    let k = wp_shape[1] * 8; // 8 int4 per int32
    let m = input.shape().as_slice()[0];

    let stream = input.device().config.stream;
    let zeros_ptr = zeros.map_or(std::ptr::null(), |z| z.data_ptr() as *const _);

    unsafe {
        if m == 1 {
            kpack_gemv_cu(
                input.data_ptr() as _, weight_packed.data_ptr() as _,
                zeros_ptr, scales.data_ptr() as _,
                output.data_ptr_mut() as _,
                n as i32, k as i32, group_size as i32, stream,
            );
        } else {
            kpack_gemm_cu(
                input.data_ptr() as _, weight_packed.data_ptr() as _,
                zeros_ptr, scales.data_ptr() as _,
                output.data_ptr_mut() as _,
                m as i32, n as i32, k as i32, group_size as i32, stream,
            );
        }
    }
    Ok(())
}
