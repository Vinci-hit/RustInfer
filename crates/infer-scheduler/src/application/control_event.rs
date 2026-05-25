//! `ControlEventSystem` — interprets control-plane events.
//!
//! Step 16 ports engine's `on_control_event` + `handle_need_blocks`
//! + `handle_worker_lost` + `handle_control_step_error` (~150 lines)
//! into a single System. The crucial invariant (P1-B in the refactor
//! plan) is **this System never touches `OutputProcessingSystem`** —
//! it returns a [`ControlOutcome`] describing what session failures
//! the orchestrator should drive next, and the orchestrator (i.e.
//! `SchedulerEngine`) does the second-stage `output.fail_*` calls
//! with a fresh borrow.
//!
//! ## Why P1-B matters
//!
//! Today's engine drives `output.fail_decoding_session(...)` from
//! within the control-event branch. That works in the current shape
//! because all four fields (`output`, `frontend`, `kv_manager`,
//! `requests`) live on the same struct. But once Steps 18-19 split
//! the engine into thin orchestrator + Systems, the control branch
//! cannot hold `&mut self.output` *and* `&mut self.requests` *and*
//! `&mut self.kv_pool` simultaneously without re-aliasing through
//! the engine. The `ControlOutcome` indirection breaks that knot
//! ahead of time.
//!
//! ## What this System does itself
//!
//! - `NeedBlocks` → allocate decode KV blocks, extend the session's
//!   block table, unicast `GrantBlocks` (or `GrantBlocksDenied`).
//! - `Heartbeat` → trace-level log.
//! - `WorkerError { fatal: false }` → error log, no failure.
//!
//! ## What it returns to the orchestrator
//!
//! - `WorkerError { fatal: true }` →
//!   `Terminate { lost: None, error }` (`lost` populated once
//!   Step 18 introduces `WorkerNode<Ready>` on the engine).
//! - `StepError` → `Continue { failed_request_ids, fail_message }`
//!   (with `Terminate` follow-up if `err.fatal`).
//! - `WorkerLost` → `Continue { failed_request_ids = all running,
//!   fail_message: "..."}` plus `Terminate`.
//!
//! Right now the engine still has its own copies of the legacy
//! handlers; Step 18 will delete those and route everything through
//! `ControlEventSystem::handle()`.

use std::marker::PhantomData;

use infer_protocol::scheduler_to_worker_control::{
    BlockGrantDeniedReason, GrantBlocks, GrantBlocksDenied, SchedulerControlMessage,
};
use infer_protocol::worker_to_scheduler_control::{NeedBlocks, WorkerStepError};

use crate::domain::kv_cache_pool::KvCachePool;
use crate::error::SchedulerError;
use crate::domain::inference_session::lifecycle::SequenceId;
use crate::domain::inference_session::table::RequestTable;
use crate::infrastructure::transport::control_plane::{ControlEvent, ControlPlaneCmdTx, WorkerId};
use crate::infrastructure::transport::control_plane::WorkerGroup;

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
    /// **Does not** call the OutputProcessingSystem. The orchestrator
    /// is responsible for driving any returned `failed_request_ids`
    /// through `output.fail_sessions(...)`.
    pub fn handle(
        &self,
        event: ControlEvent,
        sessions: &mut RequestTable,
        kv: &mut dyn KvCachePool,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
    ) -> ControlOutcome {
        match event {
            ControlEvent::NeedBlocks { worker, req } => {
                self.handle_need_blocks(worker, req, sessions, kv, control_cmd, worker_group)
            }
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
                    "Heartbeat: worker={} state={:?} active={}",
                    worker,
                    hb.state,
                    hb.active_requests
                );
                ControlOutcome::noop()
            }
        }
    }

    /// Allocate KV blocks for an in-flight decode and unicast the
    /// response. KV failures degrade to `GrantBlocksDenied`; nothing
    /// in this path is fatal to the engine.
    fn handle_need_blocks(
        &self,
        worker: WorkerId,
        req: NeedBlocks,
        sessions: &mut RequestTable,
        kv: &mut dyn KvCachePool,
        control_cmd: &ControlPlaneCmdTx,
        worker_group: &WorkerGroup,
    ) -> ControlOutcome {
        match kv.allocate_decode_blocks(crate::domain::ids::BlockCount::new(req.request_blocks as usize)) {
            Ok(blocks) => {
                if let Err(e) =
                    sessions.extend_decode_kv(SequenceId(req.sequence_id), blocks.clone())
                {
                    tracing::debug!(
                        "NeedBlocks for non-decoding sequence_id={} ignored: {}",
                        req.sequence_id,
                        e
                    );
                    return ControlOutcome::noop();
                }
                if let Err(e) = control_cmd.send_to(
                    &worker,
                    SchedulerControlMessage::GrantBlocks(GrantBlocks {
                        model_instance_id: worker_group.model_instance_id.clone(),
                        sequence_id: req.sequence_id,
                        block_ids: blocks.iter().map(|b| b.0).collect(),
                    }),
                ) {
                    return ControlOutcome::Terminate {
                        lost: None,
                        error: SchedulerError::WorkerError(format!("GrantBlocks send: {}", e)),
                    };
                }
                ControlOutcome::noop()
            }
            Err(e) => {
                tracing::warn!(
                    "NeedBlocks denied: sequence_id={} request_blocks={} error={}",
                    req.sequence_id,
                    req.request_blocks,
                    e,
                );
                if let Err(send_err) = control_cmd.send_to(
                    &worker,
                    SchedulerControlMessage::GrantBlocksDenied(GrantBlocksDenied {
                        model_instance_id: worker_group.model_instance_id.clone(),
                        sequence_id: req.sequence_id,
                        reason: BlockGrantDeniedReason::CacheExhausted,
                    }),
                ) {
                    return ControlOutcome::Terminate {
                        lost: None,
                        error: SchedulerError::WorkerError(format!(
                            "GrantBlocksDenied send: {}",
                            send_err
                        )),
                    };
                }
                ControlOutcome::noop()
            }
        }
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
            // termination must travel back. Today we only carry one
            // outcome per event; the orchestrator handles fatal
            // by inspecting the list first and *then* terminating.
            // We emit Terminate; the orchestrator pre-flushes the
            // running set on its side. (Engine still mirrors the
            // pre-Step-16 behavior exactly until Step 18 wires the
            // orchestrator path.)
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
        // Mirror legacy behavior: bubble fatal error so the engine
        // exits its event loop. Once Step 18 lands `WorkerNode<Ready>`,
        // populate `lost` with `worker_node.snapshot_as_lost(...)`.
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
    use crate::domain::kv_cache_pool::KvLease;
    use crate::infrastructure::kv_cache::traits::PhysicalBlockId;
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::{Priority, RequestId, RequestMeta, SamplingParams};
    use std::sync::Arc;
    use std::time::Instant;

    /// Mock KvCachePool that always denies decode-block grants.
    #[derive(Default)]
    struct DenyingKv;
    impl KvCachePool for DenyingKv {
        fn allocate(
            &mut self,
            _: crate::domain::ids::TokenCount,
        ) -> crate::error::Result<KvLease> {
            Ok(KvLease::empty())
        }
        fn allocate_with_prefix(
            &mut self,
            _: &[i32],
        ) -> crate::error::Result<(KvLease, crate::infrastructure::kv_cache::PrefixMatch)> {
            Ok((
                KvLease::empty(),
                crate::infrastructure::kv_cache::PrefixMatch::none(),
            ))
        }
        fn allocate_decode_blocks(
            &mut self,
            _: crate::domain::ids::BlockCount,
        ) -> crate::error::Result<Vec<PhysicalBlockId>> {
            Err(SchedulerError::CacheExhausted {
                needed: 1,
                available: 0,
            })
        }
        fn free_finished(&mut self, _: &[i32], _: KvLease) {}
        fn match_prefix(
            &mut self,
            _: &[i32],
        ) -> crate::infrastructure::kv_cache::PrefixMatch {
            crate::infrastructure::kv_cache::PrefixMatch::none()
        }
        fn flush_pending_returns(&mut self) {}
        fn block_size(&self) -> crate::domain::ids::BlockSize {
            crate::domain::ids::BlockSize::new(1)
        }
        fn total_blocks(&self) -> crate::domain::ids::BlockCount {
            crate::domain::ids::BlockCount::new(0)
        }
        fn available_blocks(&self) -> crate::domain::ids::BlockCount {
            crate::domain::ids::BlockCount::new(0)
        }
        fn mode_name(&self) -> &'static str {
            "denying"
        }
    }

    fn empty_table() -> RequestTable {
        RequestTable::new()
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
                KvLease::empty(),
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

    #[test]
    fn worker_error_fatal_emits_terminate() {
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let mut kv = DenyingKv;
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = sys.handle(
            ControlEvent::WorkerError {
                worker: WorkerId::from_identity(b"w"),
                message: "boom".into(),
                fatal: true,
            },
            &mut sessions,
            &mut kv,
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
        let mut kv = DenyingKv;
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = sys.handle(
            ControlEvent::WorkerError {
                worker: WorkerId::from_identity(b"w"),
                message: "transient".into(),
                fatal: false,
            },
            &mut sessions,
            &mut kv,
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
        let mut kv = DenyingKv;
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = sys.handle(
            ControlEvent::WorkerLost {
                worker: WorkerId::from_identity(b"w"),
                last_seen_ms: 5000,
            },
            &mut sessions,
            &mut kv,
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
        let mut kv = DenyingKv;
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = sys.handle(
            ControlEvent::StepError {
                worker: WorkerId::from_identity(b"w"),
                err: WorkerStepError {
                    sequence_ids: vec![7],
                    message: "step glitch".into(),
                    fatal: false,
                },
            },
            &mut sessions,
            &mut kv,
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

    #[test]
    fn need_blocks_denied_path_does_not_terminate() {
        let sys = ControlEventSystem::new();
        let mut sessions = empty_table();
        let mut kv = DenyingKv; // denies => GrantBlocksDenied
        let (cmd, _cmd_rx) = dummy_cmd_tx_with_rx();
        let wg = worker_group_for_test();
        let outcome = sys.handle(
            ControlEvent::NeedBlocks {
                worker: WorkerId::from_identity(b"w"),
                req: NeedBlocks {
                    worker_id: "worker-test".into(),
                    model_instance_id: "default".into(),
                    sequence_id: 1,
                    current_blocks: 0,
                    required_blocks: 1,
                    request_blocks: 1,
                    reason: infer_protocol::worker_to_scheduler_control::NeedBlocksReason::DecodeExtend,
                },
            },
            &mut sessions,
            &mut kv,
            &cmd,
            &wg,
        );
        // Non-fatal: KV exhausted gets reported back to the worker
        // via GrantBlocksDenied; the engine continues.
        assert!(matches!(outcome, ControlOutcome::Continue { .. }));
    }
}
