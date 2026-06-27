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
    #[allow(clippy::too_many_arguments)]
    fn compact_extend_control(
        block_tables: *mut i32,
        block_tables_scratch: *mut i32,
        kv_lens: *mut i32,
        kv_lens_scratch: *mut i32,
        seq_positions_out: *mut i32,
        seq_lens_step_out: *mut i32,
        rope_positions_out: *mut i32,
        cu_q_lens_out: *mut i32,
        block2req_out: *mut i32,
        block2tile_out: *mut i32,
        active_src_rows: *const i32,
        counts: *const i32,
        new_slots: *const i32,
        mbps: i32,
        cap_batch: i32,
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

/// Arguments for the device-resident decode control-plane builder.
pub struct CompactExtendControlArgs<'a> {
    /// Live block tables (this step's order). The gather reads it, then the
    /// launcher copies the scratch result back over it.
    pub block_tables: &'a mut Tensor<i32, Cuda>,
    /// Scratch gather target for block tables ([cap_batch * mbps]).
    pub block_tables_scratch: &'a mut Tensor<i32, Cuda>,
    /// Live kv_lens (read by the gather, then overwritten by the copy-back).
    pub kv_lens: &'a mut Tensor<i32, Cuda>,
    /// Scratch gather target for kv_lens ([cap_batch]).
    pub kv_lens_scratch: &'a mut Tensor<i32, Cuda>,
    pub seq_positions_out: &'a mut Tensor<i32, Cuda>,
    pub seq_lens_step_out: &'a mut Tensor<i32, Cuda>,
    pub rope_positions_out: &'a mut Tensor<i32, Cuda>,
    pub cu_q_lens_out: &'a mut Tensor<i32, Cuda>,
    pub block2req_out: &'a mut Tensor<i32, Cuda>,
    pub block2tile_out: &'a mut Tensor<i32, Cuda>,
    pub active_src_rows: &'a Tensor<i32, Cuda>,
    pub counts: &'a Tensor<i32, Cuda>,
    pub new_slots: &'a Tensor<i32, Cuda>,
    pub mbps: usize,
    pub cap_batch: usize,
    pub stream: cudaStream_t,
}

/// Build the next decode step's control plane on-device from this step's
/// survivors. See the `compact_extend_control` C declaration for semantics.
pub fn compact_extend_control_into(args: CompactExtendControlArgs<'_>) -> OpResult<()> {
    if args.cap_batch == 0 {
        return Ok(());
    }
    unsafe {
        compact_extend_control(
            args.block_tables.data_ptr_mut(),
            args.block_tables_scratch.data_ptr_mut(),
            args.kv_lens.data_ptr_mut(),
            args.kv_lens_scratch.data_ptr_mut(),
            args.seq_positions_out.data_ptr_mut(),
            args.seq_lens_step_out.data_ptr_mut(),
            args.rope_positions_out.data_ptr_mut(),
            args.cu_q_lens_out.data_ptr_mut(),
            args.block2req_out.data_ptr_mut(),
            args.block2tile_out.data_ptr_mut(),
            args.active_src_rows.data_ptr(),
            args.counts.data_ptr(),
            args.new_slots.data_ptr(),
            args.mbps as i32,
            args.cap_batch as i32,
            args.stream,
        );
    }
    Ok(())
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
