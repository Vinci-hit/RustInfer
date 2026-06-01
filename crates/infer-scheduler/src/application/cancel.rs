//! Cancel pathway helpers.
//!
//! Cancel is small enough to stay alongside `OutputProcessingSystem`
//! conceptually (it ultimately drives KV release + worker
//! notification), but it neither generates a client response nor
//! a downstream session transition the way `complete_session` does.
//! We keep it as a free helper for now; if the surface grows
//! (priorities, partial cancels, batched cancels) we'll promote it
//! to its own System.

use infer_protocol::scheduler_to_worker_control::{CancelSequence, SchedulerControlMessage};

use crate::application::OutputProcessingSystem;
use crate::domain::inference_session::lifecycle::{RequestId, SequenceId};
use crate::domain::inference_session::table::{CancelOutcome, RequestTable};
use crate::error::{Result, SchedulerError};
use crate::infrastructure::transport::control_plane::{ControlPlaneCmdTx, WorkerId};

/// Attempt to cancel a request by its internal id. Drives the
/// session repository transition and a `Cancel` control unicast to
/// the worker. Scheduler-side KV release rides on
/// `RadixTree::mark_finished_chain` from the engine; there is no
/// per-cancel KV hook to call here.
pub async fn cancel_request(
    sessions: &mut RequestTable,
    output: &OutputProcessingSystem,
    control_cmd: &ControlPlaneCmdTx,
    default_worker: &WorkerId,
    request_id: RequestId,
) -> Result<()> {
    let _ = output; // retained as a parameter for future hooks
    match sessions.cancel_request(&request_id)? {
        CancelOutcome::RemovedWaiting { .. } | CancelOutcome::NotFound => Ok(()),
        CancelOutcome::RemovedPrefilling { sequence_id, .. }
        | CancelOutcome::RemovedDecoding { sequence_id, .. } => {
            send_cancel_to_worker(control_cmd, default_worker, sequence_id)
        }
    }
}

/// Cancel by client-supplied external id (resolves through the
/// `by_external_id` index, then delegates to [`cancel_request`]).
/// Unknown external ids are silently ignored.
pub async fn cancel_request_by_external_id(
    sessions: &mut RequestTable,
    output: &OutputProcessingSystem,
    control_cmd: &ControlPlaneCmdTx,
    default_worker: &WorkerId,
    external_id: &str,
) -> Result<()> {
    let Some(seq_id) = sessions.sequence_id_for_external(external_id) else {
        tracing::debug!("Cancel for unknown external_id={}", external_id);
        return Ok(());
    };
    let Some(request_id) = sessions.request_id_for_sequence(seq_id) else {
        tracing::debug!("Cancel: sequence_id={} no longer active", seq_id);
        return Ok(());
    };
    cancel_request(sessions, output, control_cmd, default_worker, request_id).await
}

/// Unicast a `Cancel` control message to the worker that owns this
/// sequence. Single-rank deployment today; a future TP/PP variant
/// will thread per-sequence affinity through here.
fn send_cancel_to_worker(
    control_cmd: &ControlPlaneCmdTx,
    worker: &WorkerId,
    sequence_id: SequenceId,
) -> Result<()> {
    control_cmd
        .send_to(
            worker,
            SchedulerControlMessage::Cancel(CancelSequence {
                sequence_id: sequence_id.0,
            }),
        )
        .map_err(|e| SchedulerError::WorkerError(format!("cancel send: {}", e)))
}
