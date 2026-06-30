//! ABC decode merge helpers.
//!
//! Serving decode keeps:
//! - **A**: stable input/output token buffer, read by forward and by host.
//! - **B**: newly-admitted first decode tokens.
//! - **C**: graph argmax output.
//!
//! The compact merge consumes C, writes surviving rows back to A, and emits
//! active/finished counts plus source-row mappings. Host never reads C.

use crate::Cuda;
use crate::ffi::cudaStream_t;
use infer_core::ports::OpResult;
use infer_core::tensor::Tensor;

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
    fn merge_compact_mixed(
        a: *mut i32,
        c: *const i32,
        row_kind: *const i32,
        generated_counts: *const i32,
        max_tokens: *const i32,
        ignore_eos: *const i32,
        eos_ids: *const i32,
        eos_len: i32,
        old_rows: i32,
        active_src_rows: *mut i32,
        active_tokens: *mut i32,
        finished_src_rows: *mut i32,
        finished_tokens: *mut i32,
        prefill_final_src_rows: *mut i32,
        prefill_final_tokens: *mut i32,
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

pub struct MergeCompactMixedArgs<'a> {
    pub a_out: &'a mut Tensor<i32, Cuda>,
    pub c_prev: &'a Tensor<i32, Cuda>,
    pub row_kind: &'a Tensor<i32, Cuda>,
    pub generated_counts: &'a Tensor<i32, Cuda>,
    pub max_tokens: &'a Tensor<i32, Cuda>,
    pub ignore_eos: &'a Tensor<i32, Cuda>,
    pub eos_ids: &'a Tensor<i32, Cuda>,
    pub eos_len: usize,
    pub old_rows: usize,
    pub active_src_rows: &'a Tensor<i32, Cuda>,
    pub active_tokens: &'a Tensor<i32, Cuda>,
    pub finished_src_rows: &'a Tensor<i32, Cuda>,
    pub finished_tokens: &'a Tensor<i32, Cuda>,
    pub prefill_final_src_rows: &'a Tensor<i32, Cuda>,
    pub prefill_final_tokens: &'a Tensor<i32, Cuda>,
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

/// Commit mixed ragged output C into flat token tape A, compacting rows that
/// become next-step decode inputs and splitting row-kind sidebands.
pub fn merge_compact_mixed_into(args: MergeCompactMixedArgs<'_>) -> OpResult<()> {
    if args.old_rows == 0 {
        return Ok(());
    }
    unsafe {
        merge_compact_mixed(
            args.a_out.data_ptr_mut(),
            args.c_prev.data_ptr(),
            args.row_kind.data_ptr(),
            args.generated_counts.data_ptr(),
            args.max_tokens.data_ptr(),
            args.ignore_eos.data_ptr(),
            args.eos_ids.data_ptr(),
            args.eos_len as i32,
            args.old_rows as i32,
            args.active_src_rows.data_ptr_mut(),
            args.active_tokens.data_ptr_mut(),
            args.finished_src_rows.data_ptr_mut(),
            args.finished_tokens.data_ptr_mut(),
            args.prefill_final_src_rows.data_ptr_mut(),
            args.prefill_final_tokens.data_ptr_mut(),
            args.counts.data_ptr_mut(),
            args.stream,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use infer_core::types::Shape;

    #[test]
    fn merge_compact_decode_preserves_order() {
        let cuda = Cuda::new(0).expect("cuda init");
        let mut a = Tensor::<i32, Cuda>::from_host_slice(&[99; 5], Shape::from_slice(&[5]), &cuda)
            .expect("A");
        let c = Tensor::<i32, Cuda>::from_host_slice(&[10, 20, 30, 40, 50], [5], &cuda).expect("C");
        let generated_counts =
            Tensor::<i32, Cuda>::from_host_slice(&[0, 1, 4, 0, 2], [5], &cuda).expect("gen");
        let max_tokens =
            Tensor::<i32, Cuda>::from_host_slice(&[10, 10, 5, 99, 3], [5], &cuda).expect("max");
        let ignore_eos =
            Tensor::<i32, Cuda>::from_host_slice(&[1, 1, 1, 0, 1], [5], &cuda).expect("ignore");
        let eos_ids = Tensor::<i32, Cuda>::from_host_slice(&[40], [1], &cuda).expect("eos");
        let active_src_rows = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("active_src");
        let finished_src_rows = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("finished_src");
        let finished_tokens = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("finished_tokens");
        let counts = Tensor::<i32, Cuda>::zeros([3], &cuda).expect("counts");

        merge_compact_decode_into(MergeCompactDecodeArgs {
            a_out: &mut a,
            c_prev: &c,
            generated_counts: &generated_counts,
            max_tokens: &max_tokens,
            ignore_eos: &ignore_eos,
            eos_ids: &eos_ids,
            eos_len: 1,
            old_batch: 5,
            active_src_rows: &active_src_rows,
            finished_src_rows: &finished_src_rows,
            finished_tokens: &finished_tokens,
            counts: &counts,
            stream: cuda.config.stream,
        })
        .expect("merge");

        assert_eq!(&counts.to_host_vec().expect("counts")[..3], &[2, 3, 5]);
        assert_eq!(&a.to_host_vec().expect("A")[..2], &[10, 20]);
        assert_eq!(
            &active_src_rows.to_host_vec().expect("active src")[..2],
            &[0, 1]
        );
        assert_eq!(
            &finished_src_rows.to_host_vec().expect("finished src")[..3],
            &[2, 3, 4]
        );
        assert_eq!(
            &finished_tokens.to_host_vec().expect("finished tokens")[..3],
            &[30, 40, 50]
        );
    }

    #[test]
    fn merge_compact_mixed_respects_row_kind() {
        let cuda = Cuda::new(0).expect("cuda init");
        let mut a = Tensor::<i32, Cuda>::from_host_slice(&[99; 5], Shape::from_slice(&[5]), &cuda)
            .expect("A");
        let c = Tensor::<i32, Cuda>::from_host_slice(&[10, 20, 30, 40, 50], [5], &cuda).expect("C");
        let row_kind =
            Tensor::<i32, Cuda>::from_host_slice(&[0, 2, 1, 3, 0], [5], &cuda).expect("row_kind");
        let generated_counts =
            Tensor::<i32, Cuda>::from_host_slice(&[0, 0, 0, 0, 4], [5], &cuda).expect("gen");
        let max_tokens =
            Tensor::<i32, Cuda>::from_host_slice(&[10, 10, 10, 10, 5], [5], &cuda).expect("max");
        let ignore_eos =
            Tensor::<i32, Cuda>::from_host_slice(&[1, 1, 1, 1, 1], [5], &cuda).expect("ignore");
        let eos_ids = Tensor::<i32, Cuda>::zeros([1], &cuda).expect("eos");
        let active_src_rows = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("active_src");
        let active_tokens = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("active_tokens");
        let finished_src_rows = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("finished_src");
        let finished_tokens = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("finished_tokens");
        let prefill_final_src_rows =
            Tensor::<i32, Cuda>::zeros([5], &cuda).expect("prefill_final_src");
        let prefill_final_tokens =
            Tensor::<i32, Cuda>::zeros([5], &cuda).expect("prefill_final_tokens");
        let counts = Tensor::<i32, Cuda>::zeros([5], &cuda).expect("counts");

        merge_compact_mixed_into(MergeCompactMixedArgs {
            a_out: &mut a,
            c_prev: &c,
            row_kind: &row_kind,
            generated_counts: &generated_counts,
            max_tokens: &max_tokens,
            ignore_eos: &ignore_eos,
            eos_ids: &eos_ids,
            eos_len: 0,
            old_rows: 5,
            active_src_rows: &active_src_rows,
            active_tokens: &active_tokens,
            finished_src_rows: &finished_src_rows,
            finished_tokens: &finished_tokens,
            prefill_final_src_rows: &prefill_final_src_rows,
            prefill_final_tokens: &prefill_final_tokens,
            counts: &counts,
            stream: cuda.config.stream,
        })
        .expect("merge");

        assert_eq!(&counts.to_host_vec().expect("counts")[..4], &[2, 1, 1, 5]);
        assert_eq!(&a.to_host_vec().expect("A")[..2], &[10, 30]);
        assert_eq!(
            &active_src_rows.to_host_vec().expect("active src")[..2],
            &[0, 2]
        );
        assert_eq!(
            &active_tokens.to_host_vec().expect("active tokens")[..2],
            &[10, 30]
        );
        assert_eq!(
            &finished_src_rows.to_host_vec().expect("finished src")[..1],
            &[4]
        );
        assert_eq!(
            &finished_tokens.to_host_vec().expect("finished tokens")[..1],
            &[50]
        );
        assert_eq!(
            &prefill_final_src_rows
                .to_host_vec()
                .expect("prefill final src")[..1],
            &[2]
        );
        assert_eq!(
            &prefill_final_tokens
                .to_host_vec()
                .expect("prefill final tokens")[..1],
            &[30]
        );
    }
}
