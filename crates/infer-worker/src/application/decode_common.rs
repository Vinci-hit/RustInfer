//! Shared decode-step building blocks used by [`DecodeEngine`] and the
//! prefill-side failure paths.
//!
//! These types and helpers used to be copy-pasted between `decode_engine.rs`
//! and `worker_scheduler.rs`. The scheduler-side `run_decode_step` was dead
//! (fully replaced by [`DecodeEngine`]); its still-live sibling — the prefill
//! path — only needs [`send_step_error`], so the decode-specific pieces now
//! live here once and `decode_engine.rs` consumes them.
//!
//! [`DecodeEngine`]: crate::application::decode_engine::DecodeEngine

use std::collections::HashSet;

use infer_protocol::worker_to_scheduler_control::{WorkerControlMessage, WorkerStepError};
use infer_protocol::worker_to_scheduler_data::AssignedIndices;

use crate::application::model_runner::SeqStep;
use crate::application::worker_state::ActiveSeqMap;
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::infrastructure::transport::control_pump::ControlPump;

/// Outcome of decode-step preparation: either a ready-to-forward batch or a
/// signal that the caller should return `Ok(())` (nothing to run, or the step
/// already failed and reported its sequences).
pub(crate) enum DecodePrep {
    Ready {
        order: Vec<u64>,
        new_indices: Vec<u32>,
        append_start_row: usize,
        append_tokens: Vec<i32>,
    },
    Done,
}

/// Per-row inputs for one decode forward. The three trailing vectors mirror
/// `steps` row-for-row and are passed as separate slices because
/// `step_decode_abc_compact` consumes them that way.
pub(crate) struct DecodeInputs {
    pub steps: Vec<SeqStep>,
    pub assigned: Vec<AssignedIndices>,
    pub generated_counts: Vec<usize>,
    pub max_tokens: Vec<usize>,
    pub ignore_eos: Vec<bool>,
}

/// Build the per-row decode forward inputs: one appended KV slot per row, the
/// single-token step, and the metadata slices the sampler consumes.
pub(crate) fn build_decode_inputs(
    order: &[u64],
    new_indices: &[u32],
    active: &ActiveSeqMap,
    enable_prefix_caching: bool,
) -> DecodeInputs {
    let mut inputs = DecodeInputs {
        steps: Vec::with_capacity(order.len()),
        assigned: Vec::with_capacity(order.len()),
        generated_counts: Vec::with_capacity(order.len()),
        max_tokens: Vec::with_capacity(order.len()),
        ignore_eos: Vec::with_capacity(order.len()),
    };
    for (i, &sid) in order.iter().enumerate() {
        let new_idx = new_indices[i];
        let seq = active.get(&sid).unwrap();
        // C1: build the step's block table in one allocation. The old
        // `clone()` then `push()` allocated twice (clone sized to len, push
        // reallocated to len+1); reserving len+1 up front does it once.
        let mut bt = Vec::with_capacity(seq.block_table.len() + 1);
        bt.extend_from_slice(&seq.block_table);
        bt.push(new_idx);
        inputs.steps.push(SeqStep {
            input_ids: vec![seq.last_token],
            positions: vec![seq.kv_len as i32],
            kv_write_start: seq.kv_len as i32,
            kv_len_after: (seq.kv_len + 1) as i32,
            block_table: bt,
        });
        inputs.assigned.push(AssignedIndices {
            sequence_id: sid,
            base: new_idx,
            len: 1,
            token_ids: if enable_prefix_caching {
                vec![seq.last_token]
            } else {
                Vec::new()
            },
        });
        inputs.generated_counts.push(seq.generated_count);
        inputs.max_tokens.push(seq.max_tokens);
        inputs.ignore_eos.push(seq.ignore_eos);
    }
    inputs
}

/// Compute the A-buffer append window: the first row index whose sequence was
/// newly admitted this step and the last tokens to seed for `[start..]`.
pub(crate) fn build_a_append(
    order: &[u64],
    pending_admissions: &[u64],
    active: &ActiveSeqMap,
) -> (usize, Vec<i32>) {
    if pending_admissions.is_empty() {
        return (order.len(), Vec::new());
    }

    let pending: HashSet<u64> = pending_admissions.iter().copied().collect();
    let start = order
        .iter()
        .position(|sid| pending.contains(sid))
        .unwrap_or(order.len());
    let tokens = order[start..]
        .iter()
        .filter_map(|sid| active.get(sid).map(|seq| seq.last_token))
        .collect();
    (start, tokens)
}

/// Report a non-fatal decode failure for `sids`, evict them from `active`, and
/// release their KV. The shared rollback for every decode alloc/forward
/// failure path (previously copy-pasted four times inline).
pub(crate) fn fail_decode_seqs(
    control: &ControlPump,
    active: &mut ActiveSeqMap,
    kv_allocator: &mut GlobalKvAllocator,
    sids: &[u64],
    message: String,
    enable_prefix_caching: bool,
) {
    send_step_error(control, sids.to_vec(), message);
    for sid in sids {
        if let Some(removed) = active.remove(sid) {
            kv_allocator.release_owned(&removed.block_table, enable_prefix_caching);
        }
    }
}

/// Send a non-fatal StepError to the scheduler, logging (not silently
/// dropping) if the control channel is broken. Centralizes the boilerplate
/// previously copy-pasted across every prefill/decode failure path, and
/// makes a torn control plane observable instead of a silent hang (H8/M6).
pub(crate) fn send_step_error(control: &ControlPump, sequence_ids: Vec<u64>, message: String) {
    if let Err(e) = control.send(
        WorkerControlMessage::StepError(WorkerStepError {
            sequence_ids,
            message,
            fatal: false,
        }),
        infer_protocol::control_envelope::RequestId::NONE,
    ) {
        tracing::error!(error = %e, "failed to send StepError to scheduler (control plane may be down)");
    }
}
