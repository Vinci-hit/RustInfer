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
    fn fp8_block_matmul_init_cu(device_id: i32) -> i32;
    fn fp8_block_matmul_bf16_cu(
        input: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        weight_scale_inv: *const std::ffi::c_void,
        output: *mut std::ffi::c_void,
        m: i32,
        n: i32,
        k: i32,
        scale_cols: i32,
        block_n: i32,
        block_k: i32,
        device_id: i32,
        workspace: *mut std::ffi::c_void,
        workspace_size: usize,
        stream: cudaStream_t,
    ) -> i32;
    fn zimage_set_eager_prefill_gemm(on: i32);
}

/// Prepare the accelerated FP8 kernel before any CUDA stream capture.
/// Builds without an accelerated implementation expose the same symbol as a no-op.
pub(crate) fn init_fp8_block_matmul(device_id: i32) -> OpResult<()> {
    let status = unsafe { fp8_block_matmul_init_cu(device_id) };
    if status == 0 {
        Ok(())
    } else {
        Err(OpError::Kernel(format!(
            "accelerated fp8 block matmul initialization failed with CUDA status {status}"
        )))
    }
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
                    let workspace = cfg.kernel_workspace();
                    gemm_cublaslt_bf16(
                        input.data_ptr() as _,
                        weight.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        m as i32,
                        n as i32,
                        k as i32,
                        stream,
                        cfg.cublaslt_handle,
                        workspace.ptr(),
                        workspace.size(),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8BlockMatmulPath {
    DynamicW8A8Gemv,
    AcceleratedCutlass,
    CudaW8A8Fallback,
}

/// Block-scaled E4M3FN matmul with in-op dynamic 1x128 activation quantization.
///
/// Weight bytes remain FP8 on device. `workspace` must be address-stable across
/// graph capture and is partitioned into the transient FP8 activation, f32
/// activation scales, and CUTLASS scheduler workspace.
pub(crate) fn matmul_fp8_block_with_path<T: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<infer_core::dtype::Fp8E4m3, Cuda>,
    output: &mut Tensor<T, Cuda>,
    weight_scale_inv: &Tensor<f32, Cuda>,
    [block_n, block_k]: [usize; 2],
    workspace: *mut std::ffi::c_void,
    workspace_size: usize,
) -> OpResult<Fp8BlockMatmulPath> {
    if T::DATA_TYPE != DataType::BF16 {
        return Err(OpError::Kernel(format!(
            "matmul_fp8_block: activation/output must be BF16, got {:?}",
            T::DATA_TYPE
        )));
    }
    if [block_n, block_k] != [128, 128] {
        return Err(OpError::Shape(format!(
            "matmul_fp8_block: Hopper blockwise path requires block [128, 128], got [{}, {}]",
            block_n, block_k
        )));
    }
    if !input.is_contiguous()
        || !weight.is_contiguous()
        || !output.is_contiguous()
        || !weight_scale_inv.is_contiguous()
    {
        return Err(OpError::NotContiguous(*input.shape()));
    }

    let input_shape = input.shape().as_slice();
    let weight_shape = weight.shape().as_slice();
    let output_shape = output.shape().as_slice();
    let scale_shape = weight_scale_inv.shape().as_slice();
    if input_shape.len() != 2
        || weight_shape.len() != 2
        || output_shape.len() != 2
        || scale_shape.len() != 2
    {
        return Err(OpError::Shape(format!(
            "matmul_fp8_block: expected rank-2 input/weight/output/scale, got {:?}/{:?}/{:?}/{:?}",
            input_shape, weight_shape, output_shape, scale_shape
        )));
    }

    let (m, k) = (input_shape[0], input_shape[1]);
    let (n, weight_k) = (weight_shape[0], weight_shape[1]);
    if m == 0 || n == 0 || k == 0 {
        return Err(OpError::Shape(format!(
            "matmul_fp8_block: dimensions must be nonzero, got M={} N={} K={}",
            m, n, k
        )));
    }
    if weight_k != k || output_shape != [m, n] {
        return Err(OpError::Shape(format!(
            "matmul_fp8_block: input {:?}, weight {:?}, output {:?} are incompatible",
            input_shape, weight_shape, output_shape
        )));
    }
    if n % block_n != 0 || k % block_k != 0 {
        return Err(OpError::Shape(format!(
            "matmul_fp8_block: N and K must be multiples of 128, got N={} K={}",
            n, k
        )));
    }
    let expected_scale = [n / block_n, k / block_k];
    if scale_shape != expected_scale {
        return Err(OpError::Shape(format!(
            "matmul_fp8_block: expected scale shape {:?}, got {:?}",
            expected_scale, scale_shape
        )));
    }

    let to_i32 = |name: &str, value: usize| {
        i32::try_from(value).map_err(|_| {
            OpError::Shape(format!(
                "matmul_fp8_block: {}={} exceeds CUDA kernel i32 range",
                name, value
            ))
        })
    };
    let (m_i32, n_i32, k_i32) = (to_i32("M", m)?, to_i32("N", n)?, to_i32("K", k)?);
    let scale_cols = to_i32("scale_cols", expected_scale[1])?;
    let block_n = to_i32("block_n", block_n)?;
    let block_k = to_i32("block_k", block_k)?;

    let status = unsafe {
        fp8_block_matmul_bf16_cu(
            input.data_ptr() as _,
            weight.data_ptr() as _,
            weight_scale_inv.data_ptr() as _,
            output.data_ptr_mut() as _,
            m_i32,
            n_i32,
            k_i32,
            scale_cols,
            block_n,
            block_k,
            input.device().device_id,
            workspace,
            workspace_size,
            stream,
        )
    };
    match status {
        1 => Ok(Fp8BlockMatmulPath::DynamicW8A8Gemv),
        2 => Ok(Fp8BlockMatmulPath::AcceleratedCutlass),
        3 => Ok(Fp8BlockMatmulPath::CudaW8A8Fallback),
        other => Err(OpError::Kernel(format!(
            "matmul_fp8_block CUDA dispatch failed with status {other}"
        ))),
    }
}

pub(crate) fn matmul_fp8_block<T: Dtype>(
    stream: cudaStream_t,
    input: &Tensor<T, Cuda>,
    weight: &Tensor<infer_core::dtype::Fp8E4m3, Cuda>,
    output: &mut Tensor<T, Cuda>,
    weight_scale_inv: &Tensor<f32, Cuda>,
    block: [usize; 2],
    workspace: *mut std::ffi::c_void,
    workspace_size: usize,
) -> OpResult<()> {
    matmul_fp8_block_with_path(
        stream,
        input,
        weight,
        output,
        weight_scale_inv,
        block,
        workspace,
        workspace_size,
    )
    .map(|_| ())
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

#[cfg(test)]
mod fp8_block_tests {
    use super::*;
    use infer_core::dtype::{Dtype as NumericDtype, Fp8E4m3};
    use std::sync::Mutex;

    // CUDA stream capture is process-global with respect to unsafe API calls.
    // Keep the decode test from creating another CudaConfig while the prefill
    // test is capturing a graph on the same visible device.
    static FP8_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn run_case(m: usize, expected_path: Fp8BlockMatmulPath, check_capture: bool) {
        let _test_guard = FP8_TEST_LOCK.lock().expect("FP8 CUDA test lock");
        // Two blocks in both N and K make the asymmetric scale grid catch a
        // transposed [Kblocks,Nblocks] interpretation while satisfying the
        // SM90 blockwise kernel's 128-element alignment contract.
        let (n, k) = (256usize, 256usize);
        let block = [128usize, 128usize];
        let scale_values = vec![0.25f32, 0.5, 1.0, 2.0]; // row-major [2,2]
        let input_values: Vec<half::bf16> = (0..m * k)
            .map(|i| half::bf16::from_f32(((i * 7 % 13) as f32 - 6.0) * 0.125))
            .collect();
        let weight_codes = [0xb8u8, 0xb0, 0x00, 0x30, 0x38, 0x3c, 0x40, 0xbc];
        let weight_values: Vec<Fp8E4m3> = (0..n * k)
            .map(|i| Fp8E4m3(weight_codes[(i * 5 + i / k) % weight_codes.len()]))
            .collect();

        // Reference the same in-op dynamic 1x128 activation quantization.
        let mut quantized_input = vec![0.0f32; m * k];
        for row in 0..m {
            for scale_col in 0..k / 128 {
                let start = row * k + scale_col * 128;
                let max_abs = input_values[start..start + 128]
                    .iter()
                    .map(|value| value.to_f32().abs())
                    .fold(0.0f32, f32::max);
                let scale = if max_abs > 0.0 { max_abs / 448.0 } else { 1.0 };
                for inner in 0..128 {
                    let value = input_values[start + inner].to_f32() / scale;
                    let fp8 = <Fp8E4m3 as NumericDtype>::write_f64(value as f64);
                    quantized_input[start + inner] =
                        <Fp8E4m3 as NumericDtype>::read_f64(&fp8) as f32 * scale;
                }
            }
        }

        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for inner in 0..k {
                    let a = quantized_input[row * k + inner];
                    let w =
                        <Fp8E4m3 as NumericDtype>::read_f64(&weight_values[col * k + inner]) as f32;
                    let scale = scale_values[(col / block[0]) * 2 + inner / block[1]];
                    sum = a.mul_add(w * scale, sum);
                }
                expected[row * n + col] = sum;
            }
        }

        let cuda = Cuda::new(0).expect("Cuda::new");
        let input = Tensor::from_host_slice(&input_values, [m, k], &cuda).expect("input upload");
        let weight = Tensor::from_host_slice(&weight_values, [n, k], &cuda).expect("weight upload");
        let scales = Tensor::from_host_slice(&scale_values, [2, 2], &cuda).expect("scale upload");
        let mut output: Tensor<half::bf16, Cuda> =
            Tensor::zeros([m, n], &cuda).expect("output alloc");

        let path = matmul_fp8_block_with_path(
            cuda.config.stream,
            &input,
            &weight,
            &mut output,
            &scales,
            block,
            cuda.config.kernel_workspace().ptr(),
            cuda.config.kernel_workspace().size(),
        )
        .expect("fp8 block matmul");
        assert_eq!(path, expected_path, "unexpected FP8 dispatch path");

        if check_capture {
            let slot = crate::GraphSlot::LlmDecode {
                batch: m,
                buffer_id: 0,
                slot_signature: 0xf8,
            };
            cuda.config
                .capture_begin_relaxed()
                .expect("begin FP8 graph capture");
            let captured_path = matmul_fp8_block_with_path(
                cuda.config.stream,
                &input,
                &weight,
                &mut output,
                &scales,
                block,
                cuda.config.kernel_workspace().ptr(),
                cuda.config.kernel_workspace().size(),
            )
            .expect("capture FP8 block matmul");
            assert_eq!(captured_path, expected_path);
            cuda.config
                .capture_end(slot)
                .expect("instantiate FP8 graph");
            cuda.config.launch(slot).expect("replay FP8 graph");
        }

        let got = output.to_host_vec().expect("output download");
        for (index, (got, expected)) in got.iter().zip(&expected).enumerate() {
            let got = got.to_f32();
            let tolerance = 0.03 + expected.abs() * 0.01;
            assert!(
                (got - expected).abs() <= tolerance,
                "M={m} output[{index}] got {got}, expected {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn fp8_block_matmul_decode_m1_matches_reference() {
        run_case(1, Fp8BlockMatmulPath::DynamicW8A8Gemv, false);
    }

    #[test]
    fn fp8_block_matmul_prefill_uses_cutlass_and_matches_reference() {
        let expected_path = if env!("RUSTINFER_CUDA_ARCH").starts_with("sm_90") {
            Fp8BlockMatmulPath::AcceleratedCutlass
        } else {
            Fp8BlockMatmulPath::CudaW8A8Fallback
        };
        run_case(3, expected_path, true);
    }
}
