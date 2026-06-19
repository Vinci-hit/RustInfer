//! Control-plane event handling (free functions).
//!
//! Translates a single `ControlEvent` into a [`ControlOutcome`] the engine
//! then dispatches. Stateless: the engine drives any returned
//! `failed_request_ids` through `output_fns::fail_sessions(...)` with a fresh
//! borrow, keeping the `&mut requests` borrow here disjoint from the follow-up
//! `&mut output`.

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

/// P2: Context struct grouping the repeated parameters that every internal
/// handler in this module passes around. The public `handle_control_event`
/// signature is kept stable for the `EngineWorkflow` trait, but internally
/// the helpers use this struct.
struct ControlCtx<'a> {
    sessions: &'a mut RequestTable,
    radix: &'a mut RadixTree,
    kv_budget: &'a mut KvBudget,
    control_cmd: &'a ControlPlaneCmdTx,
    worker_group: &'a WorkerGroup,
    enable_prefix_caching: bool,
}

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
    let mut ctx = ControlCtx {
        sessions,
        radix,
        kv_budget,
        control_cmd,
        worker_group,
        enable_prefix_caching,
    };
    match event {
        ControlEvent::StepError { worker, err } => handle_worker_step_error(
            err,
            &mut ctx,
            &worker,
        ),
        ControlEvent::WorkerLost {
            worker,
            last_seen_ms,
        } => handle_worker_lost(worker, last_seen_ms, ctx.sessions),
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
            // ── KV drift detection ──
            // If the worker reports its outstanding KV count, compare it
            // with the scheduler's own `KvBudget::outstanding()`. A drift
            // beyond the threshold indicates lost messages / decode errors
            // that caused the two counters to permanently diverge.
            if let Some(worker_outstanding) = hb.kv_outstanding {
                let sched_outstanding = ctx.kv_budget.outstanding();
                let drift = (sched_outstanding as i64) - (worker_outstanding as i64);
                // Threshold: max(8, capacity / 1000) — covers normal in-flight
                // jitter while catching systematic drift.
                let cap = ctx.kv_budget.capacity();
                let threshold = 8i64.max(cap as i64 / 1000);
                if drift.unsigned_abs() > threshold as u64 {
                    tracing::warn!(
                        scheduler = sched_outstanding,
                        worker = worker_outstanding,
                        drift,
                        threshold,
                        "KV budget drift detected; recalibrating to worker value"
                    );
                    ctx.kv_budget.force_set_outstanding(worker_outstanding);
                }
            }
            ControlOutcome::noop()
        }
        ControlEvent::AllocFailed { worker, req } => handle_alloc_failed(
            req,
            &mut ctx,
            &worker,
            default_worker,
        ),
    }
}

/// Worker-driven KV pressure relief.
fn handle_alloc_failed(
    req: AllocFailed,
    ctx: &mut ControlCtx<'_>,
    worker_from_event: &WorkerId,
    default_worker: &WorkerId,
) -> ControlOutcome {
    let _ = default_worker;
    let target_worker = worker_from_event;

    let total = ctx.worker_group
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
        let target = target_slots;
        let indices = ctx.radix.evict_collect_at_least(target as usize);
        if !indices.is_empty() {
            release_budget_up_to(ctx.kv_budget, indices.len() as u32, "alloc_failed_round0");
            tracing::info!(
                worker_id = %req.worker_id,
                shortfall = req.shortfall,
                target,
                freed = indices.len(),
                "AllocFailed round=0: evicting RadixTree LRU leaves"
            );
            let msg =
                infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::FreeKvIndices(
                    infer_protocol::scheduler_to_worker_control::FreeKvIndices {
                        model_instance_id: ctx.worker_group.model_instance_id.clone(),
                        indices,
                    },
                );
            if let Err(e) = ctx.control_cmd.send_to(target_worker, msg) {
                tracing::error!(
                    worker = %target_worker,
                    error = %e,
                    "failed to send AllocFailed round=0 FreeKvIndices"
                );
            }
            return ControlOutcome::noop();
        }
        tracing::debug!(
            worker_id = %req.worker_id,
            shortfall = req.shortfall,
            target,
            "AllocFailed round=0: nothing in LRU — escalating to preemption"
        );
    }

    // Level 2: victim preemption.
    let target = target_slots;

    let mut candidates: Vec<PreemptCandidate> = ctx.sessions.preemption_candidates();
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
        if ctx.enable_prefix_caching {
            ctx.radix.mark_finished_chain(*sid);
        } else if let Some(n) = ctx.sessions.kv_slots_for_sequence(SequenceId(*sid)) {
            scheduler_released_slots = scheduler_released_slots.saturating_add(n);
        }
        if let Err(e) = ctx.sessions.preempt_to_queued(SequenceId(*sid)) {
            tracing::error!(sequence_id = sid, "preempt_to_queued failed: {}", e);
        }
    }
    let free_indices = if ctx.enable_prefix_caching {
        let indices = ctx.radix.evict_collect_at_least(target as usize);
        if !indices.is_empty() {
            release_budget_up_to(
                ctx.kv_budget,
                indices.len() as u32,
                "alloc_failed_round1_prefix",
            );
        }
        indices
    } else {
        if scheduler_released_slots > 0 {
            release_budget_up_to(
                ctx.kv_budget,
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
            model_instance_id: ctx.worker_group.model_instance_id.clone(),
            sequence_ids: victims,
            free_indices,
        },
    );
    if let Err(e) = ctx.control_cmd.send_to(target_worker, msg) {
        tracing::error!(
            worker = %target_worker,
            error = %e,
            "failed to send AllocFailed round=1 Preempt"
        );
    }

    ControlOutcome::noop()
}

/// Worker-reported step error.
fn handle_worker_step_error(
    err: WorkerStepError,
    ctx: &mut ControlCtx<'_>,
    target_worker: &WorkerId,
) -> ControlOutcome {
    let failed_sequence_ids = collect_failed_sequence_ids(&err, ctx.sessions);
    let failed_kv_slots: u32 = failed_sequence_ids
        .iter()
        .filter_map(|raw| ctx.sessions.kv_slots_for_sequence(SequenceId(*raw)))
        .sum();
    for raw in &failed_sequence_ids {
        let sid = SequenceId(*raw);
        if ctx.enable_prefix_caching {
            ctx.radix.mark_finished_chain(*raw);
        } else if let Some(n) = ctx.sessions.kv_slots_for_sequence(sid) {
            release_budget_up_to(ctx.kv_budget, n, "worker_step_error");
        }
    }
    if ctx.enable_prefix_caching && failed_kv_slots > 0 {
        let indices = ctx.radix.evict_collect_at_least(failed_kv_slots as usize);
        if !indices.is_empty() {
            release_budget_up_to(ctx.kv_budget, indices.len() as u32, "worker_step_error_prefix");
            let msg =
                infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::FreeKvIndices(
                    infer_protocol::scheduler_to_worker_control::FreeKvIndices {
                        model_instance_id: ctx.worker_group.model_instance_id.clone(),
                        indices,
                    },
                );
            if let Err(e) = ctx.control_cmd.send_to(target_worker, msg) {
                tracing::error!(
                    worker = %target_worker,
                    error = %e,
                    "failed to send worker StepError FreeKvIndices"
                );
            }
        }
    }
    let failed_ids = failed_sequence_ids
        .iter()
        .filter_map(|raw| ctx.sessions.request_id_for_sequence(SequenceId(*raw)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::{
        Priority, RequestId, RequestMeta, SamplingParams,
    };
    use infer_protocol::worker_to_scheduler_control::{WorkerHeartbeat, WorkerState};
    use std::sync::Arc;
    use std::time::Instant;

    fn empty_table() -> RequestTable {
        RequestTable::new()
    }

    fn fresh_radix() -> RadixTree {
        RadixTree::new()
    }

    fn fresh_budget(cap: u32) -> KvBudget {
        KvBudget::new(cap)
    }

    fn worker_group_for_test() -> WorkerGroup {
        use infer_protocol::worker_to_scheduler_control::{WorkerCapacity, WorkerReady};
        WorkerGroup::from_single_ready(WorkerReady {
            worker_id: "worker-test".into(),
            model_instance_id: "default".into(),
            model_path: "model".into(),
            model_type: "llama".into(),
            device: "cuda:0".into(),
            capacity: WorkerCapacity {
                max_batch_tokens: 0,
                max_batch_seqs: 0,
                max_running_requests: 0,
                max_total_kv_tokens: None,
                free_mem_before_load_gb: None,
                free_mem_after_load_gb: None,
                weight_mem_usage_gb: None,
                workspace_mem_usage_gb: None,
                graph_mem_usage_gb: None,
            },
        })
    }

    fn dummy_running_session(table: &mut RequestTable, external_id: &str, sid: u64) -> RequestId {
        let request_id = RequestId::new_v4();
        let meta = Arc::new(RequestMeta {
            id: request_id.clone(),
            external_id: external_id.into(),
            sequence_id: SequenceId(sid),
            input_ids: vec![1, 2, 3, 4],
            max_tokens: 8,
            sampling: SamplingParams::default(),
            priority: Priority::default(),
            stream: false,
            stop_sequences: vec![],
            ignore_eos: false,
            diffusion: None,
            arrival_time: Instant::now(),
        });
        table.insert_new(meta, RequestHandle::noop()).unwrap();
        // Promote to decoding so running_sequence_ids() picks it up.
        let queued = table.take_waiting(&request_id).unwrap();
        table
            .commit_prefill_start(
                queued,
                crate::infrastructure::kv_cache::traits::PrefixMatch::none(),
                4,
            )
            .unwrap();
        let _ = table.ack_prefill(SequenceId(sid)).unwrap();
        request_id
    }

    /// Build a `ControlPlaneCmdTx` whose router channel has a live
    /// receiver, so `send_to(...)` never fails (it would otherwise
    /// hit `ControlError::Shutdown` and our outcomes would all be
    /// `Terminate`). The receiver handle is returned so the caller
    /// keeps it alive for the duration of the test.
    fn dummy_cmd_tx_with_rx() -> (
        ControlPlaneCmdTx,
        tokio::sync::mpsc::UnboundedReceiver<
            crate::infrastructure::transport::control_plane::handle::RouterCommand,
        >,
    ) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let pending =
            crate::infrastructure::transport::control_plane::pending_calls::PendingCalls::new();
        (
            ControlPlaneCmdTx {
                tx,
                pending,
                default_rpc_deadline: std::time::Duration::from_secs(5),
            },
            rx,
        )
    }

    /// Invoke the free handler with `enable_prefix_caching = false` and the
    /// test worker identity. The migrated tests do not exercise the
    /// prefix-caching branches, so the flag value is immaterial to their
    /// assertions.
    fn invoke(
        event: ControlEvent,
        sessions: &mut RequestTable,
        radix: &mut RadixTree,
        budget: &mut KvBudget,
        cmd: &ControlPlaneCmdTx,
        wg: &WorkerGroup,
    ) -> ControlOutcome {
        handle_control_event(
            event,
            sessions,
            radix,
            budget,
            cmd,
            wg,
            &WorkerId::from_identity(b"worker-test"),
            false,
        )
    }

    #[test]
    fn worker_error_fatal_emits_terminate() {
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            ControlEvent::WorkerError {
                worker: WorkerId::from_identity(b"w"),
                message: "boom".into(),
                fatal: true,
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );
        match outcome {
            ControlOutcome::Terminate {
                lost: None,
                error: SchedulerError::WorkerError(msg),
            } => assert_eq!(msg, "boom"),
            other => panic!("expected Terminate(WorkerError), got {:?}", other),
        }
    }

    #[test]
    fn worker_error_nonfatal_is_noop() {
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            ControlEvent::WorkerError {
                worker: WorkerId::from_identity(b"w"),
                message: "transient".into(),
                fatal: false,
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );
        assert!(matches!(
            outcome,
            ControlOutcome::Continue {
                ref failed_request_ids,
                fail_message: None,
            } if failed_request_ids.is_empty()
        ));
    }

    #[test]
    fn worker_lost_terminates() {
        let mut sessions = empty_table();
        // One running session: the synthetic StepError gathers it.
        let _ = dummy_running_session(&mut sessions, "ext-1", 1);
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            ControlEvent::WorkerLost {
                worker: WorkerId::from_identity(b"w"),
                last_seen_ms: 5000,
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );
        assert!(matches!(
            outcome,
            ControlOutcome::Terminate {
                lost: None,
                error: SchedulerError::WorkerError(_),
            }
        ));
    }

    #[test]
    fn step_error_nonfatal_returns_failed_request_ids() {
        let mut sessions = empty_table();
        let rid = dummy_running_session(&mut sessions, "ext-2", 7);
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            ControlEvent::StepError {
                worker: WorkerId::from_identity(b"w"),
                err: WorkerStepError {
                    sequence_ids: vec![7],
                    message: "step glitch".into(),
                    fatal: false,
                },
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );
        match outcome {
            ControlOutcome::Continue {
                failed_request_ids,
                fail_message,
            } => {
                assert_eq!(failed_request_ids, vec![rid]);
                assert_eq!(fail_message.as_deref(), Some("step glitch"));
            }
            other => panic!("expected Continue, got {:?}", other),
        }
    }

    /// Heartbeat is liveness-only now — no KV pressure inspection.
    #[test]
    fn heartbeat_is_noop() {
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(100);
        let (cmd, mut cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            ControlEvent::Heartbeat {
                worker: WorkerId::from_identity(b"w"),
                hb: WorkerHeartbeat {
                    worker_id: "w".into(),
                    state: WorkerState::Running,
                    active_requests: 0,
                    kv_outstanding: None,
                    kv_total_free: None,
                    kv_released_pending: None,
                },
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );
        assert!(matches!(outcome, ControlOutcome::Continue { .. }));
        assert!(cmd_rx.try_recv().is_err(), "no FreeKvIndices expected");
    }
}
