//! Lightning Indexer scoring and deterministic top-k selection.
use crate::{Cuda, ffi};
use half::bf16;
use infer_core::dtype::Dtype;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;

unsafe extern "C" {
    fn rustinfer_v4_indexer_topk(
        scores: *const f32,
        start: *const i32,
        workspace: *mut i32,
        indices: *mut i32,
        tokens: i32,
        capacity: i32,
        k: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
    fn rustinfer_v4_indexer_scores_bf16(
        query: *const bf16,
        keys: *const bf16,
        weights: *const f32,
        start: *const i32,
        output: *mut f32,
        tokens: i32,
        heads: i32,
        capacity: i32,
        stream: ffi::cudaStream_t,
    ) -> i32;
}

pub fn topk_workspace_words(tokens: usize, capacity: usize, k: usize) -> OpResult<usize> {
    let parts = capacity.div_ceil(2048);
    if !(1..=i32::MAX as usize).contains(&tokens)
        || !(1..=i32::MAX as usize / 4 + 1).contains(&capacity)
        || !(1..=512).contains(&k)
        || tokens.saturating_mul(parts) > i32::MAX as usize
    {
        return Err(OpError::Shape(
            "v4_indexer_topk: N,C>0, 1<=K<=512, N*ceil(C/2048)<=i32::MAX required".into(),
        ));
    }
    if parts == 1 {
        Ok(1)
    } else {
        tokens
            .checked_mul(parts)
            .and_then(|v| v.checked_mul(k))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| OpError::Shape("v4_indexer_topk: workspace size overflow".into()))
    }
}

pub fn topk(
    stream: ffi::cudaStream_t,
    device: i32,
    scores: &Tensor<f32, Cuda>,
    start: &Tensor<i32, Cuda>,
    workspace: &mut Tensor<i32, Cuda>,
    indices: &mut Tensor<i32, Cuda>,
) -> OpResult<()> {
    let shape = scores.shape().as_slice();
    let tokens = shape.first().copied().unwrap_or(0);
    let capacity = shape.get(1).copied().unwrap_or(0);
    let k = indices.shape().as_slice().get(1).copied().unwrap_or(0);
    let words = topk_workspace_words(tokens, capacity, k)?;
    if shape != [tokens, capacity]
        || start.shape().as_slice() != [1]
        || indices.shape().as_slice() != [tokens, k]
        || workspace.shape().as_slice().len() != 1
        || workspace.numel() < words
    {
        return Err(OpError::Shape("v4_indexer_topk: expected scores [N,C], start [1], indices [N,K], and sufficient one-dimensional I32 workspace".into()));
    }
    let reads = [check(scores, device)?, check(start, device)?];
    let writes = [check(workspace, device)?, check(indices, device)?];
    for (i, &w) in writes.iter().enumerate() {
        if reads
            .iter()
            .chain(&writes[..i])
            .any(|r| r.0 < w.1 && w.0 < r.1)
        {
            return Err(OpError::Shape(
                "v4_indexer_topk: writable tensors must not overlap another argument".into(),
            ));
        }
    }
    let status = unsafe {
        rustinfer_v4_indexer_topk(
            scores.data_ptr(),
            start.data_ptr(),
            workspace.data_ptr_mut(),
            indices.data_ptr_mut(),
            tokens as i32,
            capacity as i32,
            k as i32,
            stream,
        )
    };
    if status == ffi::cudaError_cudaSuccess as i32 {
        Ok(())
    } else {
        Err(OpError::Kernel(format!(
            "v4_indexer_topk: CUDA launch error {status}"
        )))
    }
}

fn check<T: Dtype>(t: &Tensor<T, Cuda>, device: i32) -> OpResult<(usize, usize)> {
    if !t.is_contiguous()
        || t.device().device_id != device
        || !(t.data_ptr() as usize).is_multiple_of(4)
    {
        return Err(OpError::Shape(
            "v4_indexer: tensors must be contiguous, four-byte aligned, and on the scope device"
                .into(),
        ));
    }
    let ptr = t.data_ptr() as usize;
    Ok((ptr, ptr + t.numel() * std::mem::size_of::<T>()))
}

pub fn scores(
    stream: ffi::cudaStream_t,
    device: i32,
    query: &Tensor<bf16, Cuda>,
    keys: &Tensor<bf16, Cuda>,
    weights: &Tensor<f32, Cuda>,
    start: &Tensor<i32, Cuda>,
    output: &mut Tensor<f32, Cuda>,
) -> OpResult<()> {
    let shape = query.shape().as_slice();
    let tokens = shape.first().copied().unwrap_or(0);
    let heads = shape.get(1).copied().unwrap_or(0);
    let capacity = keys.shape().as_slice().first().copied().unwrap_or(0);
    if !(1..=i32::MAX as usize).contains(&tokens)
        || !(1..=128).contains(&heads)
        || !(1..=i32::MAX as usize / 4 + 1).contains(&capacity)
        || shape != [tokens, heads, 128]
        || keys.shape().as_slice() != [capacity, 128]
        || weights.shape().as_slice() != [tokens, heads]
        || start.shape().as_slice() != [1]
        || output.shape().as_slice() != [tokens, capacity]
        || tokens.saturating_mul(capacity.div_ceil(64)) > i32::MAX as usize
    {
        return Err(OpError::Shape("v4_indexer_scores: expected query [N,H,128], keys [C,128], weights [N,H], start [1], output [N,C]; N,C>0, 1<=H<=128, CUDA tile grid <=i32::MAX".into()));
    }
    let reads = [
        check(query, device)?,
        check(keys, device)?,
        check(weights, device)?,
        check(start, device)?,
    ];
    let w = check(output, device)?;
    if reads.iter().any(|r| r.0 < w.1 && w.0 < r.1) {
        return Err(OpError::Shape(
            "v4_indexer_scores: output must not overlap any input".into(),
        ));
    }
    let status = unsafe {
        rustinfer_v4_indexer_scores_bf16(
            query.data_ptr(),
            keys.data_ptr(),
            weights.data_ptr(),
            start.data_ptr(),
            output.data_ptr_mut(),
            tokens as i32,
            heads as i32,
            capacity as i32,
            stream,
        )
    };
    if status == ffi::cudaError_cudaSuccess as i32 {
        Ok(())
    } else {
        Err(OpError::Kernel(format!(
            "v4_indexer_scores: CUDA launch error {status}"
        )))
    }
}
