//! Reinterpret a small causal query batch as independent paged decode rows.
//! Only page indices are expanded, once per forward; Q and KV stay in place.
use super::{PagedAttentionKind, PagedAttentionPlan, try_cudnn_paged_decode};
use crate::{Cuda, ffi::cudaStream_t};
use infer_core::device::Device;
use infer_core::kv::{KvIndexTensors, PagedDecodeRows};
use infer_core::plan::{BatchKind, BatchPlan, MaskMode};
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};

const MAX_ROWS: usize = 8;
const DISABLE_ENV: &str = "RUSTINFER_DISABLE_SHORT_QUERY_ATTENTION";

unsafe extern "C" {
    fn rustinfer_prepare_short_query_rows(
        tables: *const i32,
        cu_q: *const i32,
        kv_lens: *const i32,
        row_tables: *mut i32,
        row_q_lens: *mut i32,
        row_kv_lens: *mut i32,
        batch: i32,
        rows: i32,
        max_blocks: i32,
        block_size: i32,
        stream: cudaStream_t,
    ) -> i32;
}

fn enabled() -> bool {
    std::env::var_os(DISABLE_ENV).is_none()
        && std::env::var_os(super::DISABLE_CUDNN_ATTENTION_ENV).is_none()
}

pub(crate) fn allocate(
    device: &Cuda,
    cap_num_tokens: usize,
    max_blocks_per_seq: usize,
) -> OpResult<Option<PagedDecodeRows<Cuda>>> {
    let capacity = cap_num_tokens.min(MAX_ROWS);
    if capacity < 2 || max_blocks_per_seq == 0 || !enabled() {
        return Ok(None);
    }
    Ok(Some(PagedDecodeRows {
        block_tables: Tensor::zeros([capacity, max_blocks_per_seq], device)?,
        q_lens: Tensor::zeros([capacity], device)?,
        kv_lens: Tensor::zeros([capacity], device)?,
        num_rows: 0,
    }))
}

fn eligible(plan: &BatchPlan) -> bool {
    matches!(
        plan.kind,
        BatchKind::Ragged
            | BatchKind::Spec {
                mask: MaskMode::Causal,
                mask_handle: None
            }
    ) && (2..=MAX_ROWS).contains(&plan.num_tokens)
        && (1..=MAX_ROWS).contains(&plan.batch)
        && plan.q_lens.len() == plan.batch
        && plan.kv_lens.len() == plan.batch
        && plan
            .q_lens
            .iter()
            .all(|&n| (1..=MAX_ROWS as i32).contains(&n))
        && plan.q_lens.iter().any(|&n| n > 1)
        && plan.q_lens.iter().map(|&n| n as usize).sum::<usize>() == plan.num_tokens
        && plan.block_size > 0
        && plan.max_blocks_per_seq > 0
        && plan
            .max_blocks_per_seq
            .checked_mul(plan.block_size)
            .is_some_and(|max_kv| {
                max_kv <= i32::MAX as usize
                    && plan
                        .q_lens
                        .iter()
                        .zip(&plan.kv_lens)
                        .all(|(&q, &kv)| kv >= q && kv as usize <= max_kv)
            })
}

pub(crate) fn prepare(
    stream: cudaStream_t,
    plan: &BatchPlan,
    index: &mut KvIndexTensors<Cuda>,
) -> OpResult<()> {
    let Some(rows) = index.decode_rows.as_mut() else {
        return Ok(());
    };
    rows.num_rows = 0;
    if !eligible(plan) || !enabled() {
        return Ok(());
    }
    let max_blocks = plan.max_blocks_per_seq;
    let table_shape = rows.block_tables.shape().as_slice();
    let source_shape = index.block_tables.shape().as_slice();
    if table_shape.len() != 2
        || table_shape[0] < plan.num_tokens
        || table_shape[1] != max_blocks
        || source_shape.len() != 2
        || source_shape[0] < plan.batch
        || source_shape[1] != max_blocks
        || index.cu_q_lens.numel() < plan.batch + 1
        || index.kv_lens.numel() < plan.batch
        || rows.q_lens.numel() < plan.num_tokens
        || rows.kv_lens.numel() < plan.num_tokens
    {
        return Err(OpError::Shape(
            "short query attention index capacity/layout mismatch".into(),
        ));
    }
    let tensors = [
        &index.block_tables,
        &index.cu_q_lens,
        &index.kv_lens,
        &rows.block_tables,
        &rows.q_lens,
        &rows.kv_lens,
    ];
    if tensors.iter().any(|t| {
        !t.is_contiguous() || t.device().device_id() != index.block_tables.device().device_id()
    }) {
        return Err(OpError::Shape(
            "short query attention indices must be contiguous on one device".into(),
        ));
    }
    let status = unsafe {
        rustinfer_prepare_short_query_rows(
            index.block_tables.data_ptr(),
            index.cu_q_lens.data_ptr(),
            index.kv_lens.data_ptr(),
            rows.block_tables.data_ptr_mut(),
            rows.q_lens.data_ptr_mut(),
            rows.kv_lens.data_ptr_mut(),
            plan.batch as i32,
            plan.num_tokens as i32,
            max_blocks as i32,
            plan.block_size as i32,
            stream,
        )
    };
    if status != 0 {
        return Err(OpError::Kernel(format!(
            "short query index expansion failed: CUDA status {status}"
        )));
    }
    rows.num_rows = plan.num_tokens;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_attention<T: Dtype>(
    stream: cudaStream_t,
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: PagedAttentionPlan<'_>,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<bool> {
    let Some(rows) = plan.decode_rows else {
        return Ok(false);
    };
    if rows.num_rows != plan.num_tokens
        || !(2..=MAX_ROWS).contains(&rows.num_rows)
        || !matches!(T::DATA_TYPE, DataType::BF16 | DataType::F16)
        || !matches!(head_dim, 64 | 128)
    {
        return Ok(false);
    }
    let decode = PagedAttentionPlan {
        decode_rows: None,
        kind: PagedAttentionKind::DecodeOnly,
        batch: rows.num_rows,
        q_lens: &[1; MAX_ROWS][..rows.num_rows],
        block_tables: &rows.block_tables,
        kv_lens: &rows.kv_lens,
        seq_lens_step: &rows.q_lens,
        ..plan
    };
    let status = unsafe {
        try_cudnn_paged_decode(
            q.device(),
            q,
            k_pool,
            v_pool,
            output,
            decode,
            q.strides().as_slice()[0] as i64,
            head_dim as i64,
            (head_num * head_dim) as i64,
            head_dim as i64,
            head_num,
            kv_head_num,
            head_dim,
            scale,
            rows.num_rows as i32,
            stream,
        )
    };
    match status {
        Some(0) => Ok(true),
        Some(status) if std::env::var_os(super::STRICT_CUDNN_ATTENTION_ENV).is_some() => {
            Err(OpError::Kernel(format!(
                "cuDNN short query attention failed: status {status}"
            )))
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests;
