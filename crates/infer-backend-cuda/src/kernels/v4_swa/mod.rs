//! Allocation-free, single-request V4 SWA. See FusedOps for the contracts.
use crate::{Cuda, ffi};
use half::bf16;
use infer_core::dtype::Dtype;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn rustinfer_v4_swa_prefill_bf16(
        query: *const bf16,
        new_kv: *const bf16,
        sink: *const f32,
        start_position: *const i32,
        cache: *mut bf16,
        output: *mut bf16,
        tokens: i32,
        heads: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
    fn rustinfer_v4_swa_decode_bf16(
        query: *const bf16,
        new_kv: *const bf16,
        sink: *const f32,
        position: *const i32,
        cache: *mut bf16,
        output: *mut bf16,
        heads: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
}

fn range<T: Dtype>(t: &Tensor<T, Cuda>) -> (usize, usize) {
    let start = t.data_ptr() as usize;
    (start, start + t.numel() * std::mem::size_of::<T>())
}

fn overlaps(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

fn check<T: Dtype>(t: &Tensor<T, Cuda>, device: i32) -> OpResult<()> {
    if !t.is_contiguous() || t.device().device_id != device {
        return Err(OpError::Shape(
            "v4_swa: all tensors must be contiguous and on the scope device".into(),
        ));
    }
    // BF16 pair loads require four-byte alignment, including offset views.
    if !(t.data_ptr() as usize).is_multiple_of(4) {
        return Err(OpError::Shape("v4_swa: unaligned tensor view".into()));
    }
    Ok(())
}

pub fn decode(
    stream: ffi::cudaStream_t,
    device: i32,
    query: &Tensor<bf16, Cuda>,
    new_kv: &Tensor<bf16, Cuda>,
    sink: &Tensor<f32, Cuda>,
    position: &Tensor<i32, Cuda>,
    cache: &mut Tensor<bf16, Cuda>,
    output: &mut Tensor<bf16, Cuda>,
) -> OpResult<()> {
    let heads = query.shape().as_slice().first().copied().unwrap_or(0);
    if heads == 0
        || heads > 128
        || query.shape().as_slice() != [heads, 512]
        || output.shape() != query.shape()
        || new_kv.shape().as_slice() != [512]
        || cache.shape().as_slice() != [128, 512]
        || sink.shape().as_slice() != [heads]
        || position.shape().as_slice() != [1]
    {
        return Err(OpError::Shape(
            "v4_swa_decode: expected Q/O [1..=128,512], KV [512], cache [128,512], sink [heads], position [1]".into(),
        ));
    }
    check(query, device)?;
    check(new_kv, device)?;
    check(sink, device)?;
    check(position, device)?;
    check(cache, device)?;
    check(output, device)?;
    let reads = [range(query), range(new_kv), range(sink), range(position)];
    let c = range(cache);
    let o = range(output);
    if overlaps(c, o) || reads.iter().any(|&r| overlaps(r, c) || overlaps(r, o)) {
        return Err(OpError::Shape(
            "v4_swa_decode: cache/output must not overlap another argument".into(),
        ));
    }
    // No allocations, synchronization, host reads or device-value inspection.
    let status = unsafe {
        rustinfer_v4_swa_decode_bf16(
            query.data_ptr(),
            new_kv.data_ptr(),
            sink.data_ptr(),
            position.data_ptr(),
            cache.data_ptr_mut(),
            output.data_ptr_mut(),
            heads as i32,
            stream,
        )
    };
    if status != ffi::cudaError_cudaSuccess as i32 {
        return Err(OpError::Kernel(format!(
            "v4_swa_decode: CUDA launch error {status}"
        )));
    }
    Ok(())
}

pub fn prefill(
    stream: ffi::cudaStream_t,
    device: i32,
    query: &Tensor<bf16, Cuda>,
    new_kv: &Tensor<bf16, Cuda>,
    sink: &Tensor<f32, Cuda>,
    start_position: &Tensor<i32, Cuda>,
    cache: &mut Tensor<bf16, Cuda>,
    output: &mut Tensor<bf16, Cuda>,
) -> OpResult<()> {
    let shape = query.shape().as_slice();
    let tokens = shape.first().copied().unwrap_or(0);
    let heads = shape.get(1).copied().unwrap_or(0);
    if tokens == 0
        || tokens > i32::MAX as usize
        || !(1..=128).contains(&heads)
        || shape != [tokens, heads, 512]
        || output.shape() != query.shape()
        || new_kv.shape().as_slice() != [tokens, 512]
        || cache.shape().as_slice() != [128, 512]
        || sink.shape().as_slice() != [heads]
        || start_position.shape().as_slice() != [1]
    {
        return Err(OpError::Shape(
            "v4_swa_prefill: expected Q/O [tokens,1..=128,512], KV [tokens,512], cache [128,512], sink [heads], start [1]".into(),
        ));
    }
    check(query, device)?;
    check(new_kv, device)?;
    check(sink, device)?;
    check(start_position, device)?;
    check(cache, device)?;
    check(output, device)?;
    let reads = [
        range(query),
        range(new_kv),
        range(sink),
        range(start_position),
    ];
    let c = range(cache);
    let o = range(output);
    if overlaps(c, o) || reads.iter().any(|&r| overlaps(r, c) || overlaps(r, o)) {
        return Err(OpError::Shape(
            "v4_swa_prefill: cache/output must not overlap another argument".into(),
        ));
    }
    let status = unsafe {
        rustinfer_v4_swa_prefill_bf16(
            query.data_ptr(),
            new_kv.data_ptr(),
            sink.data_ptr(),
            start_position.data_ptr(),
            cache.data_ptr_mut(),
            output.data_ptr_mut(),
            tokens as i32,
            heads as i32,
            stream,
        )
    };
    if status != ffi::cudaError_cudaSuccess as i32 {
        return Err(OpError::Kernel(format!(
            "v4_swa_prefill: CUDA launch error {status}"
        )));
    }
    Ok(())
}
