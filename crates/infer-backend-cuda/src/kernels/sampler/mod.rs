//! Argmax sampler CUDA kernel.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

unsafe extern "C" {
    // BF16 C signature: (logits, selected_rows_device, batch_size, vocab_size,
    // result_gpu, workspace, stream). `selected_rows_device` is nullable — null
    // means "argmax every row 0..batch". The previous binding OMITTED this
    // pointer, shifting every subsequent argument by one (batch→selected_rows,
    // vocab→batch, output→vocab, ...) so the kernel never wrote `output` and
    // decode emitted token 0 ("!") for every position.
    fn argmax_cu_bf16_ffi(
        input: *const half::bf16,
        selected_rows: *const i32,
        batch_size: i32,
        vocab_size: i32,
        output: *mut i32,
        workspace: *mut f32,
        stream: cudaStream_t,
    );
    fn argmax_cu_fp16_ffi(
        input: *const half::f16,
        vocab_size: i32,
        output: *mut i32,
        workspace: *mut f32,
        stream: cudaStream_t,
    );
    fn argmax_cu_f32_ffi(
        input: *const f32,
        vocab_size: i32,
        output: *mut i32,
        workspace: *mut f32,
        stream: cudaStream_t,
    );
}

pub fn argmax<T: Dtype>(
    stream: cudaStream_t,
    logits: &Tensor<T, Cuda>,
    output: &mut Tensor<i32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
    // BF16 only: device i32 list of rows to argmax. `None` → every row.
    // Length determines the kernel's `batch_size` (one output id per selected
    // row). f16/f32 bindings have no selector pin yet — `Some` returns Err.
    selected_rows: Option<&Tensor<i32, Cuda>>,
) -> OpResult<()> {
    let vocab_size = *logits.shape().as_slice().last().unwrap() as i32;
    let logits_rows = *logits.shape().as_slice().first().unwrap() as i32;
    // When `selected_rows` is provided, the kernel argmaxes ONLY those rows;
    // the C entry's `batch_size` is the selector length, not logits rows.
    let (sel_ptr, eff_batch) = match selected_rows {
        None => (std::ptr::null::<i32>(), logits_rows),
        Some(sel) => (sel.data_ptr(), sel.numel() as i32),
    };
    unsafe {
        match T::DATA_TYPE {
            DataType::F32 => {
                if selected_rows.is_some() {
                    return Err(OpError::Kernel(
                        "argmax: selected_rows is bf16-only (f32 binding has no selector)".into(),
                    ));
                }
                argmax_cu_f32_ffi(
                    logits.data_ptr() as _,
                    vocab_size,
                    output.data_ptr_mut(),
                    workspace.data_ptr_mut(),
                    stream,
                )
            }
            DataType::BF16 => argmax_cu_bf16_ffi(
                logits.data_ptr() as _,
                sel_ptr,
                eff_batch,
                vocab_size,
                output.data_ptr_mut(),
                workspace.data_ptr_mut(),
                stream,
            ),
            DataType::F16 => {
                if selected_rows.is_some() {
                    return Err(OpError::Kernel(
                        "argmax: selected_rows is bf16-only (f16 binding has no selector)".into(),
                    ));
                }
                argmax_cu_fp16_ffi(
                    logits.data_ptr() as _,
                    vocab_size,
                    output.data_ptr_mut(),
                    workspace.data_ptr_mut(),
                    stream,
                )
            }
            _ => return Err(OpError::Kernel(format!("argmax: {:?}", T::DATA_TYPE))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;
    use infer_core::exec::ExecScope;

    #[test]
    fn bf16_argmax_ties_choose_first_across_lanes_blocks_and_selected_rows() {
        let cuda = Cuda::new(0).unwrap();
        let scope = crate::CudaScope::new(cuda.clone());
        let vocab = 248320;
        let mut values = vec![bf16::from_f32(-1.0); 3 * vocab];
        for (row, indices) in [
            (0, vec![198, 271]),
            (1, vec![2, 3, 2050, 200000]),
            (2, vec![3501, 3502, 247999]),
        ] {
            for i in indices {
                values[row * vocab + i] = bf16::from_f32(23.625);
            }
        }
        let logits = Tensor::from_host_slice(&values, [3, vocab], &cuda).unwrap();
        let workspace = Tensor::zeros([3, 256], &cuda).unwrap();
        let mut output = Tensor::zeros([3], &cuda).unwrap();
        argmax(
            crate::scope_stream(&scope),
            &logits,
            &mut output,
            &workspace,
            None,
        )
        .unwrap();
        scope.synchronize().unwrap();
        assert_eq!(output.to_host_vec().unwrap(), vec![198, 2, 3501]);
        let selected = Tensor::from_host_slice(&[2, 0], [2], &cuda).unwrap();
        let mut picked = Tensor::zeros([2], &cuda).unwrap();
        argmax(
            crate::scope_stream(&scope),
            &logits,
            &mut picked,
            &workspace,
            Some(&selected),
        )
        .unwrap();
        scope.synchronize().unwrap();
        assert_eq!(picked.to_host_vec().unwrap(), vec![3501, 198]);
    }
}

unsafe extern "C" {
    fn filtered_workspace_bytes(n: i32, bytes: *mut usize) -> i32;
    fn filtered_sample(
        logits: *const std::ffi::c_void,
        dtype: i32,
        n: i32,
        temperature: f32,
        k: i32,
        top_p: f32,
        min_p: f32,
        draw: f64,
        out: *mut i32,
        logprob: *mut f32,
        workspace: *mut std::ffi::c_void,
        workspace_bytes: usize,
        stream: cudaStream_t,
    ) -> i32;
}

pub fn sampling_workspace_words(vocab: usize) -> OpResult<usize> {
    let n =
        i32::try_from(vocab).map_err(|_| OpError::Shape("sampling vocab exceeds i32".into()))?;
    let mut bytes = 0;
    let status = unsafe { filtered_workspace_bytes(n, &mut bytes) };
    if status != 0 {
        return Err(OpError::Kernel(format!(
            "sampling workspace query: CUDA error {status}"
        )));
    }
    Ok(bytes.div_ceil(4))
}

#[allow(clippy::too_many_arguments)]
pub fn sample_filtered_into<T: Dtype>(
    stream: cudaStream_t,
    logits: &Tensor<T, Cuda>,
    params: infer_core::ports::sampler::SamplingParams,
    draw: f64,
    out: &mut Tensor<i32, Cuda>,
    logprob: &mut Tensor<f32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
) -> OpResult<bool> {
    let dtype = match T::DATA_TYPE {
        DataType::BF16 => 0,
        DataType::F32 => 1,
        _ => return Ok(false),
    };
    if params.want_logprobs || params.repetition_penalty != 1.0 {
        return Ok(false);
    }
    let n = i32::try_from(logits.numel())
        .map_err(|_| OpError::Shape("sampling row exceeds i32".into()))?;
    if n == 0
        || logits.ndim() != 1
        || !logits.is_contiguous()
        || out.numel() != 1
        || !out.is_contiguous()
        || logprob.numel() != 1
        || !logprob.is_contiguous()
        || !workspace.is_contiguous()
        || !params.temperature.is_finite()
        || params.temperature < 0.0
        || !params.top_p.is_finite()
        || !(0.0..=1.0).contains(&params.top_p)
        || !params.min_p.is_finite()
        || !(0.0..=1.0).contains(&params.min_p)
        || !draw.is_finite()
        || !(0.0..1.0).contains(&draw)
    {
        return Err(OpError::Shape(
            "invalid filtered sampling row, parameters or output".into(),
        ));
    }
    let status = unsafe {
        filtered_sample(
            logits.data_ptr().cast(),
            dtype,
            n,
            params.temperature,
            params.top_k.min(n as u32) as i32,
            params.top_p,
            params.min_p,
            draw,
            out.data_ptr_mut(),
            logprob.data_ptr_mut(),
            workspace.data_ptr_mut().cast(),
            workspace.numel() * 4,
            stream,
        )
    };
    if status != 0 {
        return Err(OpError::Kernel(format!(
            "filtered sampling: CUDA error {status}"
        )));
    }
    Ok(true)
}

unsafe extern "C" {
    fn beam_candidates(
        logits: *const std::ffi::c_void,
        dtype: i32,
        n: i32,
        k: i32,
        out: *mut i32,
        logprobs: *mut f32,
        workspace: *mut std::ffi::c_void,
        bytes: usize,
        stream: cudaStream_t,
    ) -> i32;
}

pub fn beam_candidates_into<T: Dtype>(
    stream: cudaStream_t,
    logits: &Tensor<T, Cuda>,
    ids: &mut Tensor<i32, Cuda>,
    logprobs: &mut Tensor<f32, Cuda>,
    workspace: &Tensor<f32, Cuda>,
) -> OpResult<bool> {
    let dtype = match T::DATA_TYPE {
        DataType::BF16 => 0,
        DataType::F32 => 1,
        _ => return Ok(false),
    };
    if logits.ndim() != 1
        || logits.numel() > i32::MAX as usize
        || ids.numel() == 0
        || ids.numel() > logits.numel()
        || ids.numel() != logprobs.numel()
        || !logits.is_contiguous()
        || !ids.is_contiguous()
        || !logprobs.is_contiguous()
        || !workspace.is_contiguous()
    {
        return Err(OpError::Shape("invalid beam candidate buffers".into()));
    }
    let status = unsafe {
        beam_candidates(
            logits.data_ptr().cast(),
            dtype,
            logits.numel() as i32,
            ids.numel() as i32,
            ids.data_ptr_mut(),
            logprobs.data_ptr_mut(),
            workspace.data_ptr_mut().cast(),
            workspace.numel() * 4,
            stream,
        )
    };
    if status != 0 {
        return Err(OpError::Kernel(format!(
            "beam candidates: CUDA error {status}"
        )));
    }
    Ok(true)
}

#[cfg(test)]
mod filtered_tests {
    use super::*;
    use infer_core::ports::sampler::SamplingParams;
    #[test]
    fn filtered_gpu_sampling_and_beam_scores() {
        let cuda = Cuda::new(0).unwrap();
        let scope = crate::CudaScope::new(cuda.clone());
        let stream = crate::scope_stream(&scope);
        // Production Qwen vocabulary size, including token IDs beyond 200k.
        let vocab = 248320;
        let mut values = vec![half::bf16::from_f32(-1000.0); vocab];
        values[200001] = half::bf16::from_f32(3.0);
        values[2] = half::bf16::from_f32(2.0);
        values[7] = half::bf16::from_f32(1.0);
        let logits = Tensor::from_host_slice(&values, [vocab], &cuda).unwrap();
        let workspace = Tensor::zeros([sampling_workspace_words(vocab).unwrap()], &cuda).unwrap();
        let address = workspace.data_ptr();
        let mut ids = Tensor::zeros([1], &cuda).unwrap();
        let mut logprobs = Tensor::zeros([1], &cuda).unwrap();
        for (k, p, draw, expected) in [
            (2, 1.0, 0.99, 2),
            (0, 0.5, 0.99, 200001),
            (0, 1.0, 0.99, 7),
            (2, 0.8, 0.99, 2),
            (1, 1.0, 0.99, 200001),
        ] {
            let params = SamplingParams {
                temperature: 1.0,
                top_k: k,
                top_p: p,
                ..Default::default()
            };
            assert!(
                sample_filtered_into(
                    stream,
                    &logits,
                    params,
                    draw,
                    &mut ids,
                    &mut logprobs,
                    &workspace
                )
                .unwrap()
            );
            assert_eq!(ids.to_host_vec().unwrap(), [expected]);
            assert!(logprobs.to_host_vec().unwrap()[0].is_finite());
            assert_eq!(workspace.data_ptr(), address);
        }
        let mut top_ids = Tensor::zeros([3], &cuda).unwrap();
        let mut top_probs = Tensor::zeros([3], &cuda).unwrap();
        beam_candidates_into(stream, &logits, &mut top_ids, &mut top_probs, &workspace).unwrap();
        assert_eq!(top_ids.to_host_vec().unwrap(), [200001, 2, 7]);
        let expected = -(1f32 + (-1f32).exp() + (-2f32).exp()).ln();
        for (i, logprob) in top_probs.to_host_vec().unwrap().iter().enumerate() {
            assert!((logprob - (expected - i as f32)).abs() < 2e-5);
        }
        // Equal positive infinities have equal mass; ties retain token order.
        let tied =
            Tensor::from_host_slice(&[f32::INFINITY, 0., f32::INFINITY], [3], &cuda).unwrap();
        let scratch = Tensor::zeros([sampling_workspace_words(3).unwrap()], &cuda).unwrap();
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        for (draw, expected) in [(0.0, 0), (0.49, 0), (0.5, 2), (0.99, 2)] {
            sample_filtered_into(
                stream,
                &tied,
                params,
                draw,
                &mut ids,
                &mut logprobs,
                &scratch,
            )
            .unwrap();
            assert_eq!(ids.to_host_vec().unwrap(), [expected]);
        }
        let tiny = Tensor::zeros([1], &cuda).unwrap();
        assert!(
            sample_filtered_into(stream, &logits, params, 0.5, &mut ids, &mut logprobs, &tiny)
                .is_err()
        );
    }
}
