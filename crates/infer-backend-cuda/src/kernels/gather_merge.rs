//! ABC decode merge helpers.
//!
//! Serving decode keeps:
//! - **A**: stable input/output token buffer, read by forward and by host.
//! - **B**: newly-admitted first decode tokens.
//! - **C**: graph argmax output.
//!
//! The compact merge consumes C, writes surviving rows back to A, and emits
//! active/finished counts plus source-row mappings. Host never reads C.

use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;
use crate::Cuda;
use crate::ffi::cudaStream_t;

unsafe extern "C" {
    fn append_decode_admissions(
        a: *mut i32,
        b: *const i32,
        start: i32,
        count: i32,
        stream: cudaStream_t,
    );
    fn merge_compact_decode(
        a: *mut i32,
        c: *const i32,
        generated_counts: *const i32,
        max_tokens: *const i32,
        ignore_eos: *const i32,
        eos_ids: *const i32,
        eos_len: i32,
        old_batch: i32,
        active_src_rows: *mut i32,
        finished_src_rows: *mut i32,
        finished_tokens: *mut i32,
        counts: *mut i32,
        stream: cudaStream_t,
    );
}

/// Copy new-admission seed tokens from B into A at `start..start+count`.
pub fn append_decode_admissions_into(
    a_out: &mut Tensor<i32, Cuda>,
    b_new: &Tensor<i32, Cuda>,
    start: usize,
    count: usize,
    stream: cudaStream_t,
) -> OpResult<()> {
    if count == 0 {
        return Ok(());
    }
    unsafe {
        append_decode_admissions(
            a_out.data_ptr_mut(),
            b_new.data_ptr(),
            start as i32,
            count as i32,
            stream,
        );
    }
    Ok(())
}

pub struct MergeCompactDecodeArgs<'a> {
    pub a_out: &'a mut Tensor<i32, Cuda>,
    pub c_prev: &'a Tensor<i32, Cuda>,
    pub generated_counts: &'a Tensor<i32, Cuda>,
    pub max_tokens: &'a Tensor<i32, Cuda>,
    pub ignore_eos: &'a Tensor<i32, Cuda>,
    pub eos_ids: &'a Tensor<i32, Cuda>,
    pub eos_len: usize,
    pub old_batch: usize,
    pub active_src_rows: &'a Tensor<i32, Cuda>,
    pub finished_src_rows: &'a Tensor<i32, Cuda>,
    pub finished_tokens: &'a Tensor<i32, Cuda>,
    pub counts: &'a Tensor<i32, Cuda>,
    pub stream: cudaStream_t,
}

/// Commit decode output C into stable A, compacting non-finished rows.
pub fn merge_compact_decode_into(args: MergeCompactDecodeArgs<'_>) -> OpResult<()> {
    if args.old_batch == 0 {
        return Ok(());
    }
    unsafe {
        merge_compact_decode(
            args.a_out.data_ptr_mut(),
            args.c_prev.data_ptr(),
            args.generated_counts.data_ptr(),
            args.max_tokens.data_ptr(),
            args.ignore_eos.data_ptr(),
            args.eos_ids.data_ptr(),
            args.eos_len as i32,
            args.old_batch as i32,
            args.active_src_rows.data_ptr_mut(),
            args.finished_src_rows.data_ptr_mut(),
            args.finished_tokens.data_ptr_mut(),
            args.counts.data_ptr_mut(),
            args.stream,
        );
    }
    Ok(())
}
