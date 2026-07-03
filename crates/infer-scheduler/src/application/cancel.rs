//! Cancel pathway helpers.
//!
//! Cancel is small enough to stay alongside the output processing
//! conceptually (it ultimately drives KV release + worker
//! notification), but it neither generates a client response nor
//! a downstream session transition the way `complete_session` does.
//! We keep it as a free helper.

use infer_protocol::scheduler_to_worker_control::{CancelSequence, SchedulerControlMessage};

use crate::application::kv_reclaim::{KvReclaimer, SeqKv};
use crate::domain::inference_session::lifecycle::{RequestId, SequenceId};
use crate::domain::inference_session::table::{CancelOutcome, RequestTable};
use crate::domain::kv_budget::KvBudget;
use crate::error::{Result, SchedulerError};
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::transport::control_plane::{ControlPlaneCmdTx, WorkerId};

/// Attempt to cancel a request by its internal id.
pub async fn cancel_request(
    sessions: &mut RequestTable,
    control_cmd: &ControlPlaneCmdTx,
    default_worker: &WorkerId,
    request_id: RequestId,
) -> Result<()> {
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
pub async fn cancel_request_by_external_id(
    sessions: &mut RequestTable,
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
    cancel_request(sessions, control_cmd, default_worker, request_id).await
}

/// Cancel by external id and keep scheduler-side KV state aligned with the
/// worker-side active removal.
#[allow(clippy::too_many_arguments)]
pub async fn cancel_request_by_external_id_with_kv(
    sessions: &mut RequestTable,
    radix: &mut RadixTree,
    kv_budget: &mut KvBudget,
    control_cmd: &ControlPlaneCmdTx,
    default_worker: &WorkerId,
    model_instance_id: &str,
    external_id: &str,
    enable_prefix_caching: bool,
) -> Result<()> {
    let Some(seq_id) = sessions.sequence_id_for_external(external_id) else {
        tracing::debug!("Cancel for unknown external_id={}", external_id);
        return Ok(());
    };
    let Some(request_id) = sessions.request_id_for_sequence(seq_id) else {
        tracing::debug!("Cancel: sequence_id={} no longer active", seq_id);
        return Ok(());
    };
    let kv_slots = sessions.kv_slots_for_sequence(seq_id);

    match sessions.cancel_request(&request_id)? {
        CancelOutcome::RemovedWaiting { .. } | CancelOutcome::NotFound => Ok(()),
        CancelOutcome::RemovedPrefilling { sequence_id, .. }
        | CancelOutcome::RemovedDecoding { sequence_id, .. } => {
            let mut reclaimer = KvReclaimer {
                radix,
                kv_budget,
                control_cmd,
                model_instance_id,
                enable_prefix_caching,
            };
            reclaimer.reclaim_terminated_collect(
                &[SeqKv {
                    sequence_id: sequence_id.0,
                    kv_slots: kv_slots.unwrap_or(0),
                }],
                0,
                "cancel",
            );
            send_cancel_to_worker(control_cmd, default_worker, sequence_id)
        }
    }
}

/// Unicast a `Cancel` control message to the worker that owns this
/// sequence.
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
