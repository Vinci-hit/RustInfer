//! Allocation-free, single-request V4 SWA. See FusedOps for the contracts.
use super::v4_common::Validation;
use crate::{Cuda, ffi};
use half::bf16;
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
    let check = Validation::new("v4_swa_decode", device);
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
    let reads = [
        check.tensor(query)?,
        check.tensor(new_kv)?,
        check.tensor(sink)?,
        check.tensor(position)?,
    ];
    check.disjoint(&reads, &[check.tensor(cache)?, check.tensor(output)?])?;
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
    check.launched(status)
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
    let check = Validation::new("v4_swa_prefill", device);
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
    let reads = [
        check.tensor(query)?,
        check.tensor(new_kv)?,
        check.tensor(sink)?,
        check.tensor(start_position)?,
    ];
    check.disjoint(&reads, &[check.tensor(cache)?, check.tensor(output)?])?;
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
    check.launched(status)
}
