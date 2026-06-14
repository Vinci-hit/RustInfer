//! Free-function counterpart for [`super::ControlEventSystem`].
//!
//! `ControlEventSystem` is a stateless singleton (`PhantomData<()>`).
//! Converting to a free function simplifies the borrow shape and
//! prepares for the `EngineWorkflow` trait.

use infer_protocol::worker_to_scheduler_control::{AllocFailed, WorkerStepError};

use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::{PreemptCandidate, RequestTable};
use crate::domain::kv_budget::KvBudget;
use crate::error::SchedulerError;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::transport::control_plane::{
    ControlEvent, ControlPlaneCmdTx, WorkerGroup, WorkerId,
};

use super::outcomes::ControlOutcome;

/// Translate a single control-plane event.
///
/// **Does not** call any output functions. The orchestrator is
/// responsible for driving any returned `failed_request_ids`
/// through `output_fns::fail_sessions(...)`.
pub fn handle_control_event(
    event: ControlEvent,
    sessions: &mut RequestTable,
    radix: &mut RadixTree,
    kv_budget: &mut KvBudget,
    control_cmd: &ControlPlaneCmdTx,
    worker_group: &WorkerGroup,
    default_worker: &WorkerId,
    enable_prefix_caching: bool,
) -> ControlOutcome {
    match event {
        ControlEvent::StepError { worker, err } => handle_worker_step_error(
            err,
            sessions,
            radix,
            kv_budget,
            enable_prefix_caching,
            control_cmd,
            worker_group,
            &worker,
        ),
        ControlEvent::WorkerLost {
            worker,
            last_seen_ms,
        } => handle_worker_lost(worker, last_seen_ms, sessions),
        ControlEvent::WorkerError {
            worker,
            message,
            fatal,
        } => {
            tracing::error!(
                "WorkerError on control plane: worker={} fatal={} message={}",
                worker,
                fatal,
                message
            );
            if fatal {
                ControlOutcome::Terminate {
                    lost: None,
                    error: SchedulerError::WorkerError(message),
                }
            } else {
                ControlOutcome::noop()
            }
        }
        ControlEvent::Heartbeat { worker, hb } => {
            tracing::trace!(
                "Heartbeat: worker={} state={:?} active={}",
                worker,
                hb.state,
                hb.active_requests,
            );
            ControlOutcome::noop()
        }
        ControlEvent::AllocFailed { worker, req } => handle_alloc_failed(
            req,
            sessions,
            radix,
            kv_budget,
            control_cmd,
            worker_group,
            &worker,
            default_worker,
            enable_prefix_caching,
        ),
    }
}

/// Worker-driven KV pressure relief.
fn handle_alloc_failed(
    req: AllocFailed,
    sessions: &mut RequestTable,
    radix: &mut RadixTree,
    kv_budget: &mut KvBudget,
    control_cmd: &ControlPlaneCmdTx,
    worker_group: &WorkerGroup,
    worker_from_event: &WorkerId,
    default_worker: &WorkerId,
    enable_prefix_caching: bool,
) -> ControlOutcome {
    let _ = default_worker;
    let target_worker = worker_from_event;

    let total = worker_group
        .effective_capacity
        .max_total_kv_tokens
        .map(|t| u32::try_from(t).unwrap_or(u32::MAX))
        .unwrap_or(0);
    if total == 0 {
        tracing::debug!(
            worker_id = %req.worker_id,
            round = req.round,
            shortfall = req.shortfall,
            "AllocFailed: total capacity unknown — skipping relief"
        );
        return ControlOutcome::noop();
    }

    let five_pct = (total / 20).max(1);
    let target_slots = req.shortfall.max(five_pct).max(1);

    if req.round == 0 {
        let lru_total = radix.lru_total_indices() as u32;
        let target = target_slots.min(lru_total);
        if target > 0 {
            let indices = radix.evict_collect_at_least(target as usize);
            if !indices.is_empty() {
                release_budget_up_to(kv_budget, indices.len() as u32, "alloc_failed_round0");
                tracing::info!(
                    worker_id = %req.worker_id,
                    shortfall = req.shortfall,
                    target,
                    freed = indices.len(),
                    "AllocFailed round=0: evicting RadixTree LRU leaves"
                );
                let msg = infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::FreeKvIndices(
                    infer_protocol::scheduler_to_worker_control::FreeKvIndices {
                        model_instance_id: worker_group.model_instance_id.clone(),
                        indices,
                    },
                );
                let _ = control_cmd.send_to(target_worker, msg);
                return ControlOutcome::noop();
            }
        }
        tracing::debug!(
            worker_id = %req.worker_id,
            shortfall = req.shortfall,
            five_pct,
            lru_total,
            "AllocFailed round=0: nothing in LRU — escalating to preemption"
        );
    }

    // Level 2: victim preemption.
    let target = target_slots;

    let mut candidates: Vec<PreemptCandidate> = sessions.preemption_candidates();
    candidates.sort_by(|a, b| {
        b.output_len
            .cmp(&a.output_len)
            .then(a.input_len.cmp(&b.input_len))
    });

    let mut victims: Vec<u64> = Vec::new();
    let mut freed: u32 = 0;
    for cand in &candidates {
        victims.push(cand.sequence_id);
        freed = freed.saturating_add(cand.kv_used);
        if freed >= target {
            break;
        }
    }

    if freed < target && !victims.is_empty() {
        tracing::warn!(
            worker_id = %req.worker_id,
            target,
            freed,
            victims = victims.len(),
            "AllocFailed round=1: preempted everything but target unmet"
        );
    }

    if victims.is_empty() {
        tracing::warn!(
            worker_id = %req.worker_id,
            "AllocFailed round=1: no preemption candidates — relief timeout will fail batch"
        );
        return ControlOutcome::noop();
    }

    let mut scheduler_released_slots = 0u32;
    for sid in &victims {
        if enable_prefix_caching {
            radix.mark_finished_chain(*sid);
        } else if let Some(n) = sessions.kv_slots_for_sequence(SequenceId(*sid)) {
            scheduler_released_slots = scheduler_released_slots.saturating_add(n);
        }
        if let Err(e) = sessions.preempt_to_queued(SequenceId(*sid)) {
            tracing::error!(sequence_id = sid, "preempt_to_queued failed: {}", e);
        }
    }
    let free_indices = if enable_prefix_caching {
        let indices = radix.evict_collect_at_least(target as usize);
        if !indices.is_empty() {
            release_budget_up_to(
                kv_budget,
                indices.len() as u32,
                "alloc_failed_round1_prefix",
            );
        }
        indices
    } else {
        if scheduler_released_slots > 0 {
            release_budget_up_to(
                kv_budget,
                scheduler_released_slots,
                "alloc_failed_round1_preempt",
            );
        }
        Vec::new()
    };

    tracing::info!(
        worker_id = %req.worker_id,
        shortfall = req.shortfall,
        target,
        freed,
        free_indices = free_indices.len(),
        victims = victims.len(),
        "AllocFailed round=1: victim preemption"
    );

    let msg = infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::Preempt(
        infer_protocol::scheduler_to_worker_control::Preempt {
            model_instance_id: worker_group.model_instance_id.clone(),
            sequence_ids: victims,
            free_indices,
        },
    );
    let _ = control_cmd.send_to(target_worker, msg);

    ControlOutcome::noop()
}

/// Worker-reported step error.
fn handle_worker_step_error(
    err: WorkerStepError,
    sessions: &mut RequestTable,
    radix: &mut RadixTree,
    kv_budget: &mut KvBudget,
    enable_prefix_caching: bool,
    control_cmd: &ControlPlaneCmdTx,
    worker_group: &WorkerGroup,
    target_worker: &WorkerId,
) -> ControlOutcome {
    let failed_sequence_ids = collect_failed_sequence_ids(&err, sessions);
    let failed_kv_slots: u32 = failed_sequence_ids
        .iter()
        .filter_map(|raw| sessions.kv_slots_for_sequence(SequenceId(*raw)))
        .sum();
    for raw in &failed_sequence_ids {
        let sid = SequenceId(*raw);
        if enable_prefix_caching {
            radix.mark_finished_chain(*raw);
        } else if let Some(n) = sessions.kv_slots_for_sequence(sid) {
            release_budget_up_to(kv_budget, n, "worker_step_error");
        }
    }
    if enable_prefix_caching && failed_kv_slots > 0 {
        let indices = radix.evict_collect_at_least(failed_kv_slots as usize);
        if !indices.is_empty() {
            release_budget_up_to(kv_budget, indices.len() as u32, "worker_step_error_prefix");
            let msg =
                infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::FreeKvIndices(
                    infer_protocol::scheduler_to_worker_control::FreeKvIndices {
                        model_instance_id: worker_group.model_instance_id.clone(),
                        indices,
                    },
                );
            let _ = control_cmd.send_to(target_worker, msg);
        }
    }
    let failed_ids = failed_sequence_ids
        .iter()
        .filter_map(|raw| sessions.request_id_for_sequence(SequenceId(*raw)))
        .collect();
    let fatal = err.fatal;
    let message = err.message;
    let outcome_continue = ControlOutcome::Continue {
        failed_request_ids: failed_ids,
        fail_message: Some(message.clone()),
    };
    if fatal {
        return ControlOutcome::Terminate {
            lost: None,
            error: SchedulerError::WorkerError(message),
        };
    }
    outcome_continue
}

fn release_budget_up_to(kv_budget: &mut KvBudget, requested: u32, reason: &'static str) -> u32 {
    let releasable = requested.min(kv_budget.outstanding());
    if releasable < requested {
        tracing::warn!(
            requested,
            outstanding = kv_budget.outstanding(),
            released = releasable,
            reason,
            "KV budget release exceeds outstanding; clamping"
        );
    }
    if releasable > 0 {
        kv_budget.release(releasable);
    }
    releasable
}

/// Worker liveness watchdog timed out.
fn handle_worker_lost(
    worker: WorkerId,
    last_seen_ms: u64,
    _sessions: &mut RequestTable,
) -> ControlOutcome {
    tracing::error!(
        "Worker lost: worker={} last_seen_ms={}",
        worker,
        last_seen_ms
    );
    ControlOutcome::Terminate {
        lost: None,
        error: SchedulerError::WorkerError(format!("worker {} lost", worker)),
    }
}

/// Resolve raw sequence ids affected by a `WorkerStepError`.
fn collect_failed_sequence_ids(err: &WorkerStepError, sessions: &RequestTable) -> Vec<u64> {
    let mut sequence_ids = err.sequence_ids.clone();
    if err.fatal || sequence_ids.is_empty() {
        sequence_ids.extend(sessions.running_sequence_ids().into_iter().map(|id| id.0));
    }
    sequence_ids.sort_unstable();
    sequence_ids.dedup();
    sequence_ids
}
