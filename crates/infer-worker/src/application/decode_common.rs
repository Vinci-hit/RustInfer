//! Shared failure-path helpers used by the prefill scheduler and the decode
//! engine. Both report non-fatal step errors to the scheduler and roll back
//! worker-owned KV the same way; that logic lives here once.

use infer_protocol::worker_to_scheduler_control::{WorkerControlMessage, WorkerStepError};

use crate::application::worker_state::ActiveSeqMap;
use crate::domain::global_kv_alloc::GlobalKvAllocator;
use crate::infrastructure::transport::control_pump::ControlPump;

/// Report a non-fatal decode failure for `sids`, evict them from `active`, and
/// release their KV. The shared rollback for every decode alloc/forward
/// failure path.
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
/// makes a torn control plane observable instead of a silent hang.
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

/// Send a **fatal** StepError for `sequence_ids` (poisoned device/context).
/// Unlike [`send_step_error`], this tells the scheduler to terminate rather
/// than retry the sequences; the worker is expected to exit immediately after.
pub(crate) fn send_fatal_step_error(
    control: &ControlPump,
    sequence_ids: Vec<u64>,
    message: String,
) {
    if let Err(e) = control.send(
        WorkerControlMessage::StepError(WorkerStepError {
            sequence_ids,
            message,
            fatal: true,
        }),
        infer_protocol::control_envelope::RequestId::NONE,
    ) {
        tracing::error!(error = %e, "failed to send FATAL StepError to scheduler (control plane may be down)");
    }
}
