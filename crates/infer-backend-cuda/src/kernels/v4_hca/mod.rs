//! Single-request HCA: streaming compression and joint local/compressed attention.
use crate::{Cuda, ffi};
use half::bf16;
use infer_core::dtype::Dtype;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn rustinfer_v4_hca_compress(
        values: *const f32,
        gates: *const f32,
        ape: *const f32,
        norm: *const f32,
        rope: *const f32,
        start: *const i32,
        state: *mut f32,
        cache: *mut bf16,
        tokens: i32,
        capacity: i32,
        eps: f32,
        stream: ffi::cudaStream_t,
    ) -> i32;
    fn rustinfer_v4_hca_decode_bf16(
        query: *const bf16,
        kv: *const bf16,
        sink: *const f32,
        start: *const i32,
        compressed: *const bf16,
        cache: *mut bf16,
        output: *mut bf16,
        heads: i32,
        capacity: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
    fn rustinfer_v4_hca_prefill_bf16(
        query: *const bf16,
        kv: *const bf16,
        sink: *const f32,
        start: *const i32,
        compressed: *const bf16,
        cache: *mut bf16,
        output: *mut bf16,
        tokens: i32,
        heads: i32,
        capacity: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
}

fn check<T: Dtype>(t: &Tensor<T, Cuda>, device: i32) -> OpResult<(usize, usize)> {
    if !t.is_contiguous()
        || t.device().device_id != device
        || !(t.data_ptr() as usize).is_multiple_of(4)
    {
        return Err(OpError::Shape(
            "v4_hca: tensors must be contiguous, four-byte aligned, and on the scope device".into(),
        ));
    }
    let start = t.data_ptr() as usize;
    Ok((start, start + t.numel() * std::mem::size_of::<T>()))
}

fn disjoint(reads: &[(usize, usize)], writes: &[(usize, usize)]) -> OpResult<()> {
    let overlaps = |a: (usize, usize), b: (usize, usize)| a.0 < b.1 && b.0 < a.1;
    for (i, &w) in writes.iter().enumerate() {
        if reads.iter().chain(&writes[..i]).any(|&r| overlaps(r, w)) {
            return Err(OpError::Shape(
                "v4_hca: writable tensors must not overlap another argument".into(),
            ));
        }
    }
    Ok(())
}

fn launched(status: i32) -> OpResult<()> {
    if status != ffi::cudaError_cudaSuccess as i32 {
        Err(OpError::Kernel(format!(
            "v4_hca: CUDA launch error {status}"
        )))
    } else {
        Ok(())
    }
}

pub fn compress(
    stream: ffi::cudaStream_t,
    device: i32,
    values: &Tensor<f32, Cuda>,
    gates: &Tensor<f32, Cuda>,
    ape: &Tensor<f32, Cuda>,
    norm: &Tensor<f32, Cuda>,
    rope: &Tensor<f32, Cuda>,
    start: &Tensor<i32, Cuda>,
    state: &mut Tensor<f32, Cuda>,
    compressed: &mut Tensor<bf16, Cuda>,
    eps: f32,
) -> OpResult<()> {
    let tokens = values.shape().as_slice().first().copied().unwrap_or(0);
    let capacity = compressed.shape().as_slice().first().copied().unwrap_or(0);
    if !(1..=i32::MAX as usize).contains(&tokens)
        || !(1..=i32::MAX as usize / 128 + 1).contains(&capacity)
        || values.shape().as_slice() != [tokens, 512]
        || gates.shape() != values.shape()
        || ape.shape().as_slice() != [128, 512]
        || norm.shape().as_slice() != [512]
        || rope.shape().as_slice() != [capacity, 32, 2]
        || start.shape().as_slice() != [1]
        || state.shape().as_slice() != [3, 512]
        || compressed.shape().as_slice() != [capacity, 512]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(OpError::Shape("v4_hca_compress: expected values/gates [N,512], ape [128,512], norm [512], rope [C,32,2], start [1], state [3,512], compressed [C,512], positive finite eps".into()));
    }
    let reads = [
        check(values, device)?,
        check(gates, device)?,
        check(ape, device)?,
        check(norm, device)?,
        check(rope, device)?,
        check(start, device)?,
    ];
    disjoint(&reads, &[check(state, device)?, check(compressed, device)?])?;
    launched(unsafe {
        rustinfer_v4_hca_compress(
            values.data_ptr(),
            gates.data_ptr(),
            ape.data_ptr(),
            norm.data_ptr(),
            rope.data_ptr(),
            start.data_ptr(),
            state.data_ptr_mut(),
            compressed.data_ptr_mut(),
            tokens as i32,
            capacity as i32,
            eps,
            stream,
        )
    })
}

pub fn attention(
    stream: ffi::cudaStream_t,
    device: i32,
    decode: bool,
    query: &Tensor<bf16, Cuda>,
    kv: &Tensor<bf16, Cuda>,
    sink: &Tensor<f32, Cuda>,
    start: &Tensor<i32, Cuda>,
    compressed: &Tensor<bf16, Cuda>,
    cache: &mut Tensor<bf16, Cuda>,
    output: &mut Tensor<bf16, Cuda>,
) -> OpResult<()> {
    let shape = query.shape().as_slice();
    let tokens = if decode {
        1
    } else {
        shape.first().copied().unwrap_or(0)
    };
    let heads = shape.get(usize::from(!decode)).copied().unwrap_or(0);
    let capacity = compressed.shape().as_slice().first().copied().unwrap_or(0);
    let qshape: &[usize] = if decode {
        &[heads, 512]
    } else {
        &[tokens, heads, 512]
    };
    let kvshape: &[usize] = if decode { &[512] } else { &[tokens, 512] };
    if !(1..=i32::MAX as usize).contains(&tokens)
        || !(1..=128).contains(&heads)
        || !(1..=i32::MAX as usize / 128 + 1).contains(&capacity)
        || shape != qshape
        || output.shape() != query.shape()
        || kv.shape().as_slice() != kvshape
        || sink.shape().as_slice() != [heads]
        || start.shape().as_slice() != [1]
        || cache.shape().as_slice() != [128, 512]
        || compressed.shape().as_slice() != [capacity, 512]
    {
        return Err(OpError::Shape("v4_hca_attention: invalid Q/KV/output, sink, start or cache shapes (D=512,W=128,1<=H<=128,N>0,C>0)".into()));
    }
    let reads = [
        check(query, device)?,
        check(kv, device)?,
        check(sink, device)?,
        check(start, device)?,
        check(compressed, device)?,
    ];
    disjoint(&reads, &[check(cache, device)?, check(output, device)?])?;
    launched(unsafe {
        if decode {
            rustinfer_v4_hca_decode_bf16(
                query.data_ptr(),
                kv.data_ptr(),
                sink.data_ptr(),
                start.data_ptr(),
                compressed.data_ptr(),
                cache.data_ptr_mut(),
                output.data_ptr_mut(),
                heads as i32,
                capacity as i32,
                stream,
            )
        } else {
            rustinfer_v4_hca_prefill_bf16(
                query.data_ptr(),
                kv.data_ptr(),
                sink.data_ptr(),
                start.data_ptr(),
                compressed.data_ptr(),
                cache.data_ptr_mut(),
                output.data_ptr_mut(),
                tokens as i32,
                heads as i32,
                capacity as i32,
                stream,
            )
        }
    })
}
