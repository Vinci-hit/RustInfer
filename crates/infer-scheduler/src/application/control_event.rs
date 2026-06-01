//! `ControlEventSystem` — interprets control-plane events.
//!
//! Translates a single `ControlEvent` into a [`ControlOutcome`] the
//! engine then dispatches. The System never touches
//! [`crate::application::OutputProcessingSystem`] directly: it
//! returns the list of failed `RequestId`s plus a fail message,
//! and the engine drives `output.fail_sessions(...)` with a fresh
//! borrow. Keeping the borrows disjoint means the engine can hold
//! `&mut requests` here while later holding `&mut output` for the
//! follow-up.
//!
//! ## What this System handles inline
//!
//! - `Heartbeat` → check the worker-reported KV pool free ratio;
//!   when `kv_free_slots / kv_total_slots` falls below
//!   `KV_LOW_WATER_RATIO`, evict RadixTree LRU leaves and ship a
//!   `FreeKvIndices` control message back to the worker.
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

use infer_protocol::worker_to_scheduler_control::WorkerStepError;

use crate::error::SchedulerError;
use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::RequestTable;
use crate::domain::kv_budget::KvBudget;
use crate::infrastructure::kv_cache::radix_tree_v2::RadixTree;
use crate::infrastructure::transport::control_plane::{ControlEvent, ControlPlaneCmdTx, WorkerId};
use crate::infrastructure::transport::control_plane::WorkerGroup;

use super::outcomes::ControlOutcome;

/// Free-ratio threshold below which the scheduler kicks RadixTree LRU
/// eviction in response to a worker Heartbeat. 15 % free → react.
const KV_LOW_WATER_RATIO: f32 = 0.15;
/// Free-ratio target after eviction completes. We free as many LRU
/// leaves as needed to bring the worker's free ratio above this number.
const KV_HIGH_WATER_RATIO: f32 = 0.30;

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
    /// **Does not** call the OutputProcessingSystem. The orchestrator
    /// is responsible for driving any returned `failed_request_ids`
    /// through `output.fail_sessions(...)`.
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
            ControlEvent::WorkerLost { worker, last_seen_ms } => {
                self.handle_worker_lost(worker, last_seen_ms, sessions)
            }
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
                    "Heartbeat: worker={} state={:?} active={} kv_free={}/{}",
                    worker,
                    hb.state,
                    hb.active_requests,
                    hb.kv_free_slots,
                    hb.kv_total_slots,
                );

                // Worker-driven KV pressure response.
                //
                // Total == 0 means the worker did not report KV info
                // (legacy heartbeats, diffusion mode, bootstrap-phase
                // heartbeat). Skip silently in that case.
                if hb.kv_total_slots > 0 {
                    let free_ratio =
                        hb.kv_free_slots as f32 / hb.kv_total_slots as f32;
                    if free_ratio < KV_LOW_WATER_RATIO {
                        let target_free = ((hb.kv_total_slots as f32)
                            * KV_HIGH_WATER_RATIO)
                            as u32;
                        let need_to_free =
                            target_free.saturating_sub(hb.kv_free_slots);
                        if need_to_free > 0 {
                            self.run_kv_pressure_relief(
                                radix,
                                kv_budget,
                                control_cmd,
                                worker_group,
                                default_worker,
                                need_to_free,
                            );
                        }
                    }
                }
                ControlOutcome::noop()
            }
        }
    }

    /// Evict RadixTree LRU leaves to satisfy the worker's pressure
    /// signal, account the freed slots in `KvBudget`, and ship a
    /// `FreeKvIndices` control message back to the worker. All errors
    /// are swallowed — pressure relief is best-effort and the next
    /// Heartbeat will retry if we couldn't free enough.
    fn run_kv_pressure_relief(
        &self,
        radix: &mut RadixTree,
        kv_budget: &mut KvBudget,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
        default_worker: &WorkerId,
        need_to_free: u32,
    ) {
        let freed = radix.evict(need_to_free as usize);
        if freed.is_empty() {
            tracing::debug!(
                need = need_to_free,
                "KV pressure: RadixTree LRU empty — nothing to evict"
            );
            return;
        }
        kv_budget.release(freed.len() as u32);
        tracing::info!(
            need = need_to_free,
            freed = freed.len(),
            "KV pressure: evicted RadixTree LRU leaves"
        );
        let msg = infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::FreeKvIndices(
            infer_protocol::scheduler_to_worker_control::FreeKvIndices {
                model_instance_id: worker_group.model_instance_id.clone(),
                indices: freed,
            },
        );
        let _ = control_cmd.send_to(default_worker, msg);
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
            // Both the failed-list (for OutputSystem) and the fatal
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
            sequence_ids
                .extend(sessions.running_sequence_ids().into_iter().map(|id| id.0));
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
    use crate::domain::inference_session::lifecycle::{Priority, RequestId, RequestMeta, SamplingParams};
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
        let pending = crate::infrastructure::transport::control_plane::pending_calls::PendingCalls::new();
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

    /// Heartbeat with no KV info → noop, no eviction, no FreeKvIndices.
    #[test]
    fn heartbeat_without_kv_info_is_noop() {
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
                    kv_total_slots: 0,
                    kv_free_slots: 0,
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

    /// Heartbeat above low-water → noop.
    #[test]
    fn heartbeat_above_low_water_is_noop() {
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let mut radix = fresh_radix();
        let mut budget = fresh_budget(100);
        let (cmd, mut cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let _ = invoke(
            &sys,
            ControlEvent::Heartbeat {
                worker: WorkerId::from_identity(b"w"),
                hb: WorkerHeartbeat {
                    worker_id: "w".into(),
                    state: WorkerState::Running,
                    active_requests: 0,
                    kv_total_slots: 100,
                    kv_free_slots: 50, // 50% free
                },
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );
        assert!(cmd_rx.try_recv().is_err(), "no FreeKvIndices expected");
    }

    /// Heartbeat below low-water with LRU-eligible nodes → evict + send
    /// `FreeKvIndices` and adjust `KvBudget`.
    #[tokio::test]
    async fn heartbeat_low_water_evicts_and_sends_free_kv_indices() {
        use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;

        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        // Build a RadixTree with 3 chains × 4 slots each, all finished
        // → 12 slots in LRU.
        let mut radix = fresh_radix();
        for s in 1..=3u64 {
            for k in 0..4 {
                let token = (10 * s as i32) + k;
                let idx = ((s as u32 - 1) * 4) + k as u32;
                radix.append_token(s, token, idx);
            }
            radix.mark_finished_chain(s);
        }
        let mut budget = fresh_budget(100);
        // Pretend the worker reported these 12 slots.
        budget.try_reserve(12).unwrap();

        let (cmd, mut cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let _ = invoke(
            &sys,
            ControlEvent::Heartbeat {
                worker: WorkerId::from_identity(b"w"),
                hb: WorkerHeartbeat {
                    worker_id: "w".into(),
                    state: WorkerState::Running,
                    active_requests: 0,
                    kv_total_slots: 100,
                    kv_free_slots: 10, // 10% free → trigger
                },
            },
            &mut sessions,
            &mut radix,
            &mut budget,
            &cmd,
            &wg,
        );

        // Budget should have shrunk by the number of evicted indices.
        // Target free = 30, current free = 10, need_to_free = 20.
        // LRU has 12 slots → evict yields all 12.
        assert_eq!(budget.outstanding(), 0, "all 12 LRU slots released");

        let cmd_msg = cmd_rx.try_recv().expect("FreeKvIndices expected");
        match cmd_msg {
            crate::infrastructure::transport::control_plane::handle::RouterCommand::SendTo {
                env, ..
            } => match env.payload {
                SchedulerControlMessage::FreeKvIndices(free) => {
                    assert_eq!(free.indices.len(), 12, "all LRU slots returned");
                }
                other => panic!("expected FreeKvIndices, got {:?}", other),
            },
            _ => panic!("expected SendTo router command"),
        }
    }
}
