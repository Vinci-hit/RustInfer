//! Matmul CUDA kernel wrapper — dispatches to GEMV/GEMM by dtype and shape.
//! Also provides AWQ int4 quantized matmul (kpack_gemv/kpack_gemm).

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
    fn sgemv_cu_fp32x4(
        input: *const f32,
        weight: *const f32,
        output: *mut f32,
        m: i32,
        k: i32,
        stream: cudaStream_t,
    );
    fn gemm_cublas_f32_axbt(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: cudaStream_t,
    );
    fn hgemv_bf16_cu(
        input: *const half::bf16,
        weight: *const half::bf16,
        output: *mut half::bf16,
        n: i32,
        k: i32,
        stream: cudaStream_t,
    );
    fn gemm_cublaslt_bf16(
        a: *const half::bf16,
        b: *const half::bf16,
        c: *mut half::bf16,
        m: i32,
        n: i32,
        k: i32,
        stream: cudaStream_t,
        handle: crate::ffi::cublasLtHandle_t,
        workspace: *mut std::ffi::c_void,
        workspace_size: usize,
    );
    // AWQ int4 quantized kernels
    fn kpack_gemv_cu(
        input: *const std::ffi::c_void,
        weight_packed: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        scales: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        n: i32,
        k: i32,
        group_size: i32,
        stream: cudaStream_t,
    );
    fn kpack_gemm_cu(
        input: *const std::ffi::c_void,
        weight_packed: *const std::ffi::c_void,
        zeros: *const std::ffi::c_void,
        scales: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32,
        n: i32,
        k: i32,
        group_size: i32,
        stream: cudaStream_t,
    );
    fn zimage_set_eager_prefill_gemm(on: i32);
}

/// Toggle the eager-prefill bf16 GEMM mode. When `on`, eager (non-capturing)
/// bf16 GEMMs use the build-free chunked `cublasGemmEx` path instead of the
/// per-shape cuBLASLt heuristic+probe cache build — eliminating ~9-18ms of
/// cold-shape build from every distinct-length prefill's TTFT. Decode (graph
/// capture + its warmup) must leave this off so the cuBLASLt cache is built.
pub fn set_eager_prefill_gemm(on: bool) {
    // SAFETY: stores an int into a process-global atomic; no aliasing.
    unsafe { zimage_set_eager_prefill_gemm(on as i32) };
}

/// Standard matmul: output = input @ weight^T (same dtype)
pub fn matmul<T: Dtype>(
    stream: cudaStream_t,
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

    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => {
                if m == 1 {
                    sgemv_cu_fp32x4(
                        input.data_ptr() as _,
                        weight.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        n as i32,
                        k as i32,
                        stream,
                    );
                } else {
                    // Tensor-Cores TF32 path via cuBLAS sgemm. Beats the
                    // naïve kernel by ~10× on H100/H20 for the shapes we use
                    // here (DiT 30 × 3840×3840 GEMMs per step).
                    gemm_cublas_f32_axbt(
                        input.data_ptr() as _,
                        weight.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        m as i32,
                        n as i32,
                        k as i32,
                        stream,
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
                let force_gemm = infer_core::env_flags::force_gemm();
                if m == 1 && n <= GEMV_VS_GEMM_N_THRESHOLD && !force_gemm {
                    hgemv_bf16_cu(
                        input.data_ptr() as _,
                        weight.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        n as i32,
                        k as i32,
                        stream,
                    );
                } else {
                    gemm_cublaslt_bf16(
                        input.data_ptr() as _,
                        weight.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        m as i32,
                        n as i32,
                        k as i32,
                        stream,
                        cfg.cublaslt_handle,
                        cfg.workspace,
                        cfg.workspace_size,
                    );
                }
            }
            _ => {
                return Err(OpError::Kernel(format!(
                    "matmul: unsupported dtype {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}

/// AWQ int4 quantized matmul.
/// - input: activation [M, K] (A dtype, typically bf16)
/// - weight_packed: [N, K/per_word] (W dtype, i32 — `per_word` int4 values packed)
/// - scales: [N, num_groups] (A dtype)
/// - zeros: [N/8, num_groups] (W dtype, i32 packed)
/// - output: [M, N] (O dtype, typically bf16)
/// - scheme: quantization scheme — `group` (e.g. 128) and `packing` (must be
///   `AwqInt4` for this build; `logical_per_word()` gives the pack factor)
pub fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<A, Cuda>,
    weight_packed: &Tensor<W, Cuda>,
    output: &mut Tensor<O, Cuda>,
    scales: &Tensor<A, Cuda>,
    zeros: Option<&Tensor<W, Cuda>>,
    scheme: &infer_core::dtype::quant::QuantScheme,
) -> OpResult<()> {
    use infer_core::dtype::quant::Packing;
    // This build ships the AWQ int4 kernels (kpack_gemv/kpack_gemm). Reject any
    // other packing loudly instead of silently mis-decoding K — the packing is
    // an attribute of the scheme, so the per-word factor comes from it.
    if scheme.packing != Packing::AwqInt4 {
        return Err(OpError::Kernel(format!(
            "matmul_quant: unsupported packing {:?} (this build supports AwqInt4)",
            scheme.packing
        )));
    }
    let per_word = scheme.logical_per_word(); // 8 int4 per int32 word
    let group_size = scheme.group;
    let wp_shape = weight_packed.shape().as_slice();
    let n = wp_shape[0];
    let k = wp_shape[1] * per_word;
    let m = input.shape().as_slice()[0];

    let zeros_ptr = zeros.map_or(std::ptr::null(), |z| z.data_ptr() as *const _);

    unsafe {
        if m == 1 {
            kpack_gemv_cu(
                input.data_ptr() as _,
                weight_packed.data_ptr() as _,
                zeros_ptr,
                scales.data_ptr() as _,
                output.data_ptr_mut() as _,
                n as i32,
                k as i32,
                group_size as i32,
                stream,
            );
        } else {
            kpack_gemm_cu(
                input.data_ptr() as _,
                weight_packed.data_ptr() as _,
                zeros_ptr,
                scales.data_ptr() as _,
                output.data_ptr_mut() as _,
                m as i32,
                n as i32,
                k as i32,
                group_size as i32,
                stream,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod xemb_tests {
    use super::*;
    #[test]
    fn matmul_bf16_x_embedder_shape_no_explosion() {
        let cuda = Cuda::new(0).unwrap();
        let (m, n, k) = (4096usize, 3840usize, 64usize);
        fn fill_normal(v: &mut [f32], seed: u64, scale: f32) {
            let mut s = seed;
            for x in v.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u1 = (s >> 32) as f32 / u32::MAX as f32 + 1e-7;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u2 = (s >> 32) as f32 / u32::MAX as f32;
                *x = (-2.0_f32 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * scale;
            }
        }
        let mut inp_host = vec![0.0_f32; m * k];
        fill_normal(&mut inp_host, 0xCAFE_BABE, 1.0);
        let mut wgt_host = vec![0.0_f32; n * k];
        fill_normal(&mut wgt_host, 0xDEAD_BEEF, 0.05);
        let inp_bf16: Vec<half::bf16> = inp_host.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let wgt_bf16: Vec<half::bf16> = wgt_host.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let inp: Tensor<half::bf16, Cuda> =
            Tensor::from_host_slice(&inp_bf16, [m, k], &cuda).unwrap();
        let wgt: Tensor<half::bf16, Cuda> =
            Tensor::from_host_slice(&wgt_bf16, [n, k], &cuda).unwrap();
        let mut out: Tensor<half::bf16, Cuda> = Tensor::zeros([m, n], &cuda).unwrap();
        matmul(cuda.config.stream, &inp, &wgt, &mut out).unwrap();
        let got: Vec<f32> = out
            .to_host_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect();
        let nan = got.iter().filter(|v| !v.is_finite()).count();
        let big: usize = got
            .iter()
            .filter(|v| v.is_finite() && v.abs() > 100.0)
            .count();
        let max_abs = got
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "matmul (M={} N={} K={}) BF16: nan={} big(>100)={} max_abs={}",
            m, n, k, nan, big, max_abs
        );
        if nan > 0 || big > 0 {
            for (i, &v) in got.iter().enumerate() {
                if !v.is_finite() || v.abs() > 100.0 {
                    eprintln!("  bad [{}, {}]: {}", i / n, i % n, v);
                    if i > 200 {
                        break;
                    }
                }
            }
        }
        assert_eq!(nan, 0, "matmul produced {} NaN outputs", nan);
        assert!(
            big < 10,
            "matmul produced {} big outputs (max_abs={})",
            big,
            max_abs
        );
    }
}
