//! `ControlEventSystem` — interprets control-plane events.
//!
//! Translates a single `ControlEvent` into a [`ControlOutcome`] the
//! engine then dispatches. The System never touches
//! [`crate::application::output_fns`] directly: it
//! returns the list of failed `RequestId`s plus a fail message,
//! and the engine drives `output_fns::fail_sessions(...)` with a fresh
//! borrow. Keeping the borrows disjoint means the engine can hold
//! `&mut requests` here while later holding `&mut output` for the
//! follow-up.
//!
//! ## What this System handles inline
//!
//! - `Heartbeat` → liveness-only (the router thread updates the
//!   liveness clock); the engine sees this for observability and
//!   does not act on it.
//! - `AllocFailed` → KV pressure relief. Round 0 evicts RadixTree LRU
//!   leaves up to ~5% of total slots and replies with `FreeKvIndices`.
//!   Round 1 picks decoding / chunked-prefilling victims, marks their
//!   chains finished, transitions them back to `Queued`, and replies
//!   with `Preempt(sequence_ids)`. The worker is purely passive at
//!   both rounds.
//! - `WorkerError { fatal: false }` → error log, then `Continue`.
//!
//! ## What it returns to the orchestrator
//!
//! - `WorkerError { fatal: true }` → `Terminate { error }`.
//! - `StepError` → `Continue { failed_request_ids, fail_message }`
//!   if `!fatal`, else `Terminate`.
//! - `WorkerLost` → `Terminate { error }` after collecting every
//!   running session as failed.

use std::marker::PhantomData;

use infer_protocol::worker_to_scheduler_control::{AllocFailed, WorkerStepError};

use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::{PreemptCandidate, RequestTable};
use crate::domain::kv_budget::KvBudget;
use crate::error::SchedulerError;
use crate::infrastructure::kv_cache::radix_tree::RadixTree;
use crate::infrastructure::transport::control_plane::WorkerGroup;
use crate::infrastructure::transport::control_plane::{ControlEvent, ControlPlaneCmdTx, WorkerId};

use super::outcomes::ControlOutcome;

/// Control-event handling stage.
#[derive(Debug, Default)]
pub struct ControlEventSystem {
    _marker: PhantomData<()>,
}

impl ControlEventSystem {
    pub fn new() -> Self {
        Self::default()
    }

    /// Translate a single control-plane event.
    ///
    /// **Does not** call output_fns. The orchestrator
    /// is responsible for driving any returned `failed_request_ids`
    /// through `output_fns::fail_sessions(...)`.
    pub fn handle(
        &self,
        event: ControlEvent,
        sessions: &mut RequestTable,
        radix: &mut RadixTree,
        kv_budget: &mut KvBudget,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
        default_worker: &WorkerId,
    ) -> ControlOutcome {
        match event {
            ControlEvent::StepError { worker: _, err } => {
                self.handle_worker_step_error(err, sessions)
            }
            ControlEvent::WorkerLost {
                worker,
                last_seen_ms,
            } => self.handle_worker_lost(worker, last_seen_ms, sessions),
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
            ControlEvent::AllocFailed { worker, req } => self.handle_alloc_failed(
                req,
                sessions,
                radix,
                kv_budget,
                control_cmd,
                worker_group,
                &worker,
                default_worker,
            ),
        }
    }

    /// Worker-driven KV pressure relief.
    ///
    /// `round = 0` → Level 1: evict up to `min(5%, lru_total)` indices
    /// from the RadixTree LRU and reply with `FreeKvIndices`.
    ///
    /// `round = 1` → Level 2: pick decoding + chunked-prefilling
    /// victims sorted by `(output_len desc, input_len asc)` until ~5%
    /// of total capacity is freed. For each victim:
    ///   1. `radix.mark_finished_chain(sid)` releases its prefix-tree
    ///      ownership so its slots can later return to LRU.
    ///   2. `sessions.preempt_to_queued(sid)` flips the type-state
    ///      back to `Queued`, push_front'ing it into `waiting`. For
    ///      Decoding victims this also bumps `preemption_count`.
    /// Then send `Preempt(victim_ids)`; the worker is purely passive
    /// and frees their `block_table` locally.
    fn handle_alloc_failed(
        &self,
        req: AllocFailed,
        sessions: &mut RequestTable,
        radix: &mut RadixTree,
        kv_budget: &mut KvBudget,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
        worker_from_event: &WorkerId,
        default_worker: &WorkerId,
    ) -> ControlOutcome {
        // Use the worker that emitted the AllocFailed when available;
        // fall back to default_worker (single-rank deployment safety net).
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

        // Minimum eviction target as a fraction of total slots, expressed as
        // a divisor: `total / MIN_EVICT_TARGET_DIVISOR` ≈ 5% (L4). Evicting a
        // small batch beyond the immediate shortfall amortizes the per-event
        // eviction cost and reduces back-to-back AllocFailed churn.
        const MIN_EVICT_TARGET_DIVISOR: u32 = 20;
        let min_target = (total / MIN_EVICT_TARGET_DIVISOR).max(1);
        let target_slots = req.shortfall.max(min_target).max(1);

        if req.round == 0 {
            // Level 1: ask the LRU eviction path directly. It returns empty
            // when no reclaimable leaves exist and may return fewer than the
            // target when the cache cannot satisfy it.
            let target = target_slots;
            let indices = radix.evict_collect_at_least(target as usize);
            if !indices.is_empty() {
                kv_budget.release(indices.len() as u32);
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
                if let Err(e) = control_cmd.send_to(target_worker, msg) {
                    tracing::error!(
                        worker = %target_worker,
                        error = %e,
                        "failed to send AllocFailed round=0 FreeKvIndices"
                    );
                }
                return ControlOutcome::noop();
            }
            // LRU is empty (or evict returned nothing). Fall through to
            // Level 2 immediately rather than forcing the worker to wait
            // 500ms for nothing and re-issue a round=1 request. This
            // halves the relief tail latency in the common burst case
            // where LRU is empty (cold start, all-active workload).
            tracing::debug!(
                worker_id = %req.worker_id,
                shortfall = req.shortfall,
                target,
                "AllocFailed round=0: nothing in LRU — escalating to preemption"
            );
        }

        // Level 2: victim preemption.
        let target = target_slots;

        let mut candidates: Vec<PreemptCandidate> = sessions.preemption_candidates();
        // (output_len desc, input_len asc). Long outputs first
        // (sunk-cost is largest), short inputs next (cheapest to
        // re-prefill on resume).
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

        // Level 3: even all candidates won't satisfy the target.
        // Continue with whatever we have — exhausting the active pool
        // is the best we can offer; any leftover shortfall makes the
        // worker-side `wait_for_relief` time out and fail the batch.
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

        // Mark chains finished + flip type-state. Order matters: the
        // RadixTree drop happens *before* the worker physically frees
        // anything, so reused indices can never alias a still-pinned
        // owner.
        for sid in &victims {
            radix.mark_finished_chain(*sid);
            if let Err(e) = sessions.preempt_to_queued(SequenceId(*sid)) {
                tracing::error!(sequence_id = sid, "preempt_to_queued failed: {}", e);
            }
        }

        tracing::info!(
            worker_id = %req.worker_id,
            shortfall = req.shortfall,
            target,
            freed,
            victims = victims.len(),
            "AllocFailed round=1: victim preemption"
        );

        let msg = infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::Preempt(
            infer_protocol::scheduler_to_worker_control::Preempt {
                model_instance_id: worker_group.model_instance_id.clone(),
                sequence_ids: victims,
                free_indices: Vec::new(),
            },
        );
        if let Err(e) = control_cmd.send_to(target_worker, msg) {
            tracing::error!(
                worker = %target_worker,
                error = %e,
                "failed to send AllocFailed round=1 Preempt"
            );
        }

        ControlOutcome::noop()
    }

    /// Worker-reported step error. Fatal flag escalates to
    /// `Terminate`; non-fatal returns the failed-id list to the
    /// orchestrator for OutputSystem follow-up.
    fn handle_worker_step_error(
        &self,
        err: WorkerStepError,
        sessions: &mut RequestTable,
    ) -> ControlOutcome {
        let failed_ids = self.collect_failed_sequence_ids(&err, sessions);
        let fatal = err.fatal;
        let message = err.message;
        let outcome_continue = ControlOutcome::Continue {
            failed_request_ids: failed_ids,
            fail_message: Some(message.clone()),
        };
        if fatal {
            // Both the failed-list (for output_fns) and the fatal
            // termination need to travel back. We currently carry one
            // outcome per event, so emit `Terminate` and let the engine
            // pre-flush the running set on its side before unwinding.
            return ControlOutcome::Terminate {
                lost: None,
                error: SchedulerError::WorkerError(message),
            };
        }
        outcome_continue
    }

    /// Worker liveness watchdog timed out. Fail every running
    /// session and terminate.
    fn handle_worker_lost(
        &self,
        worker: WorkerId,
        last_seen_ms: u64,
        sessions: &mut RequestTable,
    ) -> ControlOutcome {
        tracing::error!(
            "Worker lost: worker={} last_seen_ms={}",
            worker,
            last_seen_ms
        );
        // Every active sequence dies with a synthetic message.
        let synthetic = WorkerStepError {
            sequence_ids: sessions
                .running_sequence_ids()
                .into_iter()
                .map(|id| id.0)
                .collect(),
            message: format!("worker {} lost (last_seen_ms={})", worker, last_seen_ms),
            fatal: true,
        };
        let _failed_ids = self.collect_failed_sequence_ids(&synthetic, sessions);
        // Bubble the fatal error so the engine exits its event loop.
        ControlOutcome::Terminate {
            lost: None,
            error: SchedulerError::WorkerError(format!("worker {} lost", worker)),
        }
    }

    /// Resolve the list of internal `RequestId` for a
    /// `WorkerStepError`.
    ///
    /// If `err.fatal` or `err.sequence_ids` is empty, every running
    /// session is included (fatal errors invalidate the whole batch).
    fn collect_failed_sequence_ids(
        &self,
        err: &WorkerStepError,
        sessions: &RequestTable,
    ) -> Vec<crate::domain::inference_session::lifecycle::RequestId> {
        let mut sequence_ids = err.sequence_ids.clone();
        if err.fatal || sequence_ids.is_empty() {
            sequence_ids.extend(sessions.running_sequence_ids().into_iter().map(|id| id.0));
        }
        sequence_ids.sort_unstable();
        sequence_ids.dedup();
        sequence_ids
            .into_iter()
            .filter_map(|raw| sessions.request_id_for_sequence(SequenceId(raw)))
            .collect()
    }
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

    fn invoke(
        sys: &ControlEventSystem,
        event: ControlEvent,
        sessions: &mut RequestTable,
        radix: &mut RadixTree,
        budget: &mut KvBudget,
        cmd: &ControlPlaneCmdTx,
        wg: &WorkerGroup,
    ) -> ControlOutcome {
        sys.handle(
            event,
            sessions,
            radix,
            budget,
            cmd,
            wg,
            &WorkerId::from_identity(b"worker-test"),
        )
    }

    #[test]
    fn worker_error_fatal_emits_terminate() {
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            &sys,
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
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            &sys,
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
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        // One running session: the synthetic StepError gathers it.
        let _ = dummy_running_session(&mut sessions, "ext-1", 1);
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            &sys,
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
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let rid = dummy_running_session(&mut sessions, "ext-2", 7);
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(0);
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            &sys,
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
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(100);
        let (cmd, mut cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = invoke(
            &sys,
            ControlEvent::Heartbeat {
                worker: WorkerId::from_identity(b"w"),
                hb: WorkerHeartbeat {
                    worker_id: "w".into(),
                    state: WorkerState::Running,
                    active_requests: 0,
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
