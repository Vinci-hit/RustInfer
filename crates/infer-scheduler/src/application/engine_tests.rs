    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
    use infer_protocol::scheduler_to_worker_data::BatchCommand;
    use infer_protocol::worker_to_scheduler_control::WorkerStepError;
    use infer_protocol::worker_to_scheduler_data::{GeneratedToken, StepOutput};

    use crate::config::SchedulerConfig;
    use crate::domain::ids::BlockSize;
    use crate::domain::policy::ContinuousBatchingPolicy;
    use crate::domain::inference_session::handle::{ClientId, RequestHandle};
    use crate::error::SchedulerError;
    use crate::domain::inference_session::lifecycle::{
        InFlightPrefillSegment, InferenceSession, Prefilling, Priority, RequestId, RequestMeta,
        SamplingParams, SequenceId,
    };
    use crate::infrastructure::transport::codec::{Codec, MsgPackCodec};
    use crate::infrastructure::transport::control_plane::WorkerId;
    use crate::infrastructure::transport::traits::{FrontendEvent, FrontendTransport, WorkerTransport};
    use crate::infrastructure::transport::control_plane::WorkerGroup;
    use infer_protocol::scheduler_to_server::{InferenceResponse, StreamChunk};
    use infer_protocol::worker_to_scheduler_control::{WorkerCapacity, WorkerReady};

    /// Build a `(ControlPlaneCmdTx, ControlPlaneEventRx, sent: Arc<Mutex<Vec<RouterCommand>>>)`
    /// trio for engine tests. The cmd_tx records every queued message; the
    /// event_rx is fed by the test by writing to its sender.
    fn mock_control_plane() -> (
        crate::infrastructure::transport::control_plane::ControlPlaneCmdTx,
        crate::infrastructure::transport::control_plane::ControlPlaneEventRx,
        tokio::sync::mpsc::UnboundedSender<crate::infrastructure::transport::control_plane::ControlEvent>,
        tokio::sync::mpsc::UnboundedReceiver<
            crate::infrastructure::transport::control_plane::handle::RouterCommand,
        >,
    ) {
        use crate::infrastructure::transport::control_plane::handle::RouterCommand;
        use crate::infrastructure::transport::control_plane::pending_calls::PendingCalls;
        use crate::infrastructure::transport::control_plane::{ControlPlaneCmdTx, ControlPlaneEventRx};
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<RouterCommand>();
        let (event_tx, event_rx) =
            tokio::sync::mpsc::unbounded_channel::<crate::infrastructure::transport::control_plane::ControlEvent>();
        let pending = PendingCalls::new();
        let cmd = ControlPlaneCmdTx {
            tx: cmd_tx,
            pending,
            default_rpc_deadline: std::time::Duration::from_secs(5),
        };
        let events = ControlPlaneEventRx { rx: event_rx };
        (cmd, events, event_tx, cmd_rx)
    }

    #[derive(Default)]
    struct MockFrontend;

    #[async_trait]
    impl FrontendTransport for MockFrontend {
        async fn recv_event(&mut self) -> Result<FrontendEvent> {
            Err(crate::error::SchedulerError::Shutdown)
        }

        async fn send_response(
            &mut self,
            _client: &ClientId,
            _response: InferenceResponse,
        ) -> Result<()> {
            Ok(())
        }

        async fn send_stream_chunk(
            &mut self,
            _client: &ClientId,
            _chunk: StreamChunk,
        ) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct MockWorker {
        sent: Arc<Mutex<Vec<Vec<u8>>>>,
    }

    #[async_trait]
    impl WorkerTransport for MockWorker {
        async fn send_batch(&mut self, cmd: Vec<u8>) -> Result<()> {
            self.sent.lock().unwrap().push(cmd);
            Ok(())
        }

        async fn recv_step_output(&mut self) -> Result<Vec<u8>> {
            Err(crate::error::SchedulerError::Shutdown)
        }
    }

    fn worker_group() -> WorkerGroup {
        WorkerGroup::from_single_ready(WorkerReady {
            worker_id: "worker-test".to_string(),
            model_instance_id: "default".to_string(),
            model_path: "model".to_string(),
            model_type: "llama3".to_string(),
            device: "cuda:0".to_string(),
            capacity: WorkerCapacity {
                max_batch_tokens: 256,
                max_batch_seqs: 4,
                max_running_requests: 4,
                max_total_kv_tokens: Some(32),
                free_mem_before_load_gb: None,
                free_mem_after_load_gb: None,
                weight_mem_usage_gb: None,
                workspace_mem_usage_gb: None,
                graph_mem_usage_gb: None,
            },
        })
    }

    fn prefilling_sequence() -> InferenceSession<Prefilling> {
        let meta = Arc::new(RequestMeta {
            id: RequestId::new_v4(), external_id: "req-need-blocks".to_string(),
            sequence_id: SequenceId(7),
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
        InferenceSession {
            meta,
            handle: RequestHandle::noop(),
            state: Prefilling {
                num_computed_tokens: 0,
                inflight: Some(InFlightPrefillSegment {
                    segment_start: 0,
                    segment_end: 4,
                    is_final: true,
                }),
                prompt_len: 4,
                prefill_start: Instant::now(),
            },
        }
    }

    #[tokio::test]
    async fn step_output_final_prefill_decodes_with_existing_blocks() -> Result<()> {
        let config = SchedulerConfig {
            paged_block_size: BlockSize::new(4),
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let worker = MockWorker::default();
        let _sent = Arc::clone(&worker.sent);
        let (control_cmd, control_events, _event_tx, mut cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let mut engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            worker_group(),
            MockFrontend,
            worker,
            control_cmd,
            control_events,
            default_worker.clone(),
        );
        let seq = prefilling_sequence();
        let request_id = seq.meta.id.clone();
        engine
            .requests
            .insert_new(Arc::clone(&seq.meta), RequestHandle::noop())?;
        let queued = engine.requests.take_waiting(&request_id)?;
        engine.requests.commit_prefill_start(
            queued,
            crate::infrastructure::kv_cache::traits::PrefixMatch::none(),
            4,
        )?;

        let codec = MsgPackCodec;
        // Slim StepOutput: only prefill_done + tokens.
        let output = StepOutput {
            prefill_done: vec![7],
            tokens: vec![GeneratedToken {
                sequence_id: 7,
                token_id: 42,
                finished: false,
            }],
            assigned_indices: vec![],
        };
        engine
            .handle_step_output_llm(codec.encode(&output)?)
            .await?;

        // After step_output: sequence is decoding with the original block.
        assert_eq!(engine.requests.prefilling_len(), 0);
        assert_eq!(engine.requests.decoding_len(), 1);
        assert_eq!(engine.requests.decoding()[0].state.output_tokens, vec![42]);

        // Drain the cmd_rx — no scheduler control msg should be emitted on the
        // worker-owned-KV path during a normal decode transition.
        assert!(cmd_rx.try_recv().is_err(), "no scheduler control msg expected");

        // Suppress unused suppression: confirm the data-plane sent vec has the
        // expected single prefill batch (none in this test path).
        let _ = _sent;
        // Use `BatchCommand` to confirm the import remains valid even though no
        // batch was sent on this path.
        let _: Option<BatchCommand> = None;
        let _ = default_worker;
        let _ = SchedulerControlMessage::Ping;
        Ok(())
    }

    /// Build a SchedulerEngine ready for control-plane event injection. Tests
    /// hold the test side of the `cmd_rx` so they can assert what the engine
    /// emitted, and the test side of `event_tx` so they can drive events.
    fn make_engine() -> (
        SchedulerEngine,
        WorkerId,
        tokio::sync::mpsc::UnboundedSender<crate::infrastructure::transport::control_plane::ControlEvent>,
        tokio::sync::mpsc::UnboundedReceiver<
            crate::infrastructure::transport::control_plane::handle::RouterCommand,
        >,
    ) {
        let config = SchedulerConfig {
            paged_block_size: BlockSize::new(4),
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let (control_cmd, control_events, event_tx, cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            worker_group(),
            MockFrontend,
            MockWorker::default(),
            control_cmd,
            control_events,
            default_worker.clone(),
        );
        (engine, default_worker, event_tx, cmd_rx)
    }

    /// `WorkerStepError` arriving on the control plane should fail the listed
    /// in-flight sequences and (when fatal) bubble up as a SchedulerError.
    #[tokio::test]
    async fn step_error_via_control_plane_fails_inflight() -> Result<()> {
        let (mut engine, default_worker, _event_tx, _cmd_rx) = make_engine();
        // Inject a fatal StepError event.
        let err = WorkerStepError {
            sequence_ids: vec![],
            message: "synthetic fatal".to_string(),
            fatal: true,
        };
        let result = engine
            .on_control_event(crate::infrastructure::transport::control_plane::ControlEvent::StepError {
                worker: default_worker.clone(),
                err,
            })
            .await;
        assert!(matches!(result, Err(SchedulerError::WorkerError(_))));
        Ok(())
    }

    /// A `WorkerLost` event should bubble out a fatal `SchedulerError` so the
    /// event loop terminates rather than continuing to send to a dead worker.
    #[tokio::test]
    async fn worker_lost_fails_all_inflight() -> Result<()> {
        let (mut engine, default_worker, _event_tx, _cmd_rx) = make_engine();
        let result = engine
            .on_control_event(crate::infrastructure::transport::control_plane::ControlEvent::WorkerLost {
                worker: default_worker.clone(),
                last_seen_ms: 9_999,
            })
            .await;
        assert!(matches!(result, Err(SchedulerError::WorkerError(_))));
        Ok(())
    }

    /// `cancel_request` on a prefilling sequence should unicast a Cancel
    /// SchedulerControlMessage on the control plane (not on the data plane).
    #[tokio::test]
    async fn cancel_emits_control_unicast_not_data_plane() -> Result<()> {
        let (mut engine, default_worker, _event_tx, mut cmd_rx) = make_engine();
        let seq = prefilling_sequence();
        let request_id = seq.meta.id.clone();
        engine
            .requests
            .insert_new(Arc::clone(&seq.meta), RequestHandle::noop())?;
        let queued = engine.requests.take_waiting(&request_id)?;
        engine.requests.commit_prefill_start(
            queued,
            crate::infrastructure::kv_cache::traits::PrefixMatch::none(),
            4,
        )?;

        crate::application::cancel::cancel_request(
            &mut engine.requests,
            &engine.control_cmd,
            &engine.default_worker,
            request_id,
        )
        .await?;

        let cmd = cmd_rx.try_recv().expect("expected RouterCommand on cmd_rx");
        match cmd {
            crate::infrastructure::transport::control_plane::handle::RouterCommand::SendTo { worker, env } => {
                assert_eq!(worker, default_worker);
                match env.payload {
                    SchedulerControlMessage::Cancel(c) => {
                        assert_eq!(c.sequence_id, 7);
                    }
                    other => panic!("expected Cancel, got {:?}", other),
                }
            }
            _ => panic!("expected SendTo"),
        }
        Ok(())
    }

    // ─── Admission cascade activation ────────────────────────────────

    /// Verifies the engine's `KvBudget` was sized from
    /// `worker_group.effective_capacity.max_total_kv_tokens` (32 in the
    /// test fixture).
    #[tokio::test]
    async fn engine_kv_budget_capacity_taken_from_worker_group() -> Result<()> {
        let (control_cmd, control_events, _evt_tx, _cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let config = SchedulerConfig {
            paged_block_size: BlockSize::new(4),
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            worker_group(),
            MockFrontend,
            MockWorker::default(),
            control_cmd,
            control_events,
            default_worker,
        );
        // The worker_group fixture reports max_total_kv_tokens = 32; engine
        // should pick that up as the KvBudget capacity.
        assert_eq!(engine.kv_budget.capacity(), 32);
        assert_eq!(engine.kv_budget.outstanding(), 0);
        // RadixTree starts empty.
        assert_eq!(engine.radix.token_count(), 0);
        Ok(())
    }

    /// `StepOutput.assigned_indices` flows into RadixTree + KvBudget when
    /// LLM step output arrives.
    #[tokio::test]
    async fn step_output_assigned_indices_drive_radix_and_budget() -> Result<()> {
        use infer_protocol::worker_to_scheduler_data::AssignedIndices;
        let (control_cmd, control_events, _evt_tx, _cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let config = SchedulerConfig {
            paged_block_size: BlockSize::new(4),
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let mut engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            worker_group(),
            MockFrontend,
            MockWorker::default(),
            control_cmd,
            control_events,
            default_worker,
        );

        // Build a StepOutput with assigned_indices populated. We skip the
        // prefill_done / tokens machinery (those sequences aren't actually
        // registered) — `handle_step_output_llm` first peels the
        // assigned_indices fields, which exercise the radix/budget path,
        // and then `process_llm_step` warns on unknown sequence_ids but
        // does not fail.
        let codec = MsgPackCodec;
        let output = StepOutput {
            prefill_done: vec![],
            tokens: vec![GeneratedToken {
                sequence_id: 100,
                token_id: 42,
                finished: false,
            }],
            assigned_indices: vec![
                AssignedIndices {
                    sequence_id: 100,
                    base: 0,
                    len: 3,
                },
                AssignedIndices {
                    sequence_id: 200,
                    base: 3,
                    len: 5,
                },
            ],
        };
        engine
            .handle_step_output_llm(codec.encode(&output)?)
            .await?;

        // Budget reflects the 3 + 5 = 8 reserved slots.
        assert_eq!(engine.kv_budget.outstanding(), 8);
        // RadixTree saw 8 token append calls (one per slot).
        assert_eq!(engine.radix.token_count(), 8);
        Ok(())
    }

    /// `mark_finished_chain` is invoked when a token's `finished` flag is
    /// set in the StepOutput. After that, evicting from the tree returns
    /// the slots that belonged to the finished sequence.
    #[tokio::test]
    async fn finished_token_marks_chain_for_lru_eviction() -> Result<()> {
        use infer_protocol::worker_to_scheduler_data::AssignedIndices;
        let (control_cmd, control_events, _evt_tx, _cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let config = SchedulerConfig {
            paged_block_size: BlockSize::new(4),
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let mut engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            worker_group(),
            MockFrontend,
            MockWorker::default(),
            control_cmd,
            control_events,
            default_worker,
        );

        let codec = MsgPackCodec;
        // Write 4 slots for seq 7.
        let out1 = StepOutput {
            prefill_done: vec![],
            tokens: vec![GeneratedToken {
                sequence_id: 7,
                token_id: 11,
                finished: false,
            }],
            assigned_indices: vec![AssignedIndices {
                sequence_id: 7,
                base: 0,
                len: 4,
            }],
        };
        engine.handle_step_output_llm(codec.encode(&out1)?).await?;
        assert_eq!(engine.radix.lru_len_estimate(), 0);

        // Finish seq 7 (token.finished = true).
        let out2 = StepOutput {
            prefill_done: vec![],
            tokens: vec![GeneratedToken {
                sequence_id: 7,
                token_id: 22,
                finished: true,
            }],
            assigned_indices: vec![AssignedIndices {
                sequence_id: 7,
                base: 4,
                len: 1,
            }],
        };
        engine.handle_step_output_llm(codec.encode(&out2)?).await?;

        // After mark_finished_chain, the chain's leaf must be in LRU.
        assert!(engine.radix.lru_len_estimate() >= 1);
        // Eviction yields the 5 slots used by seq 7.
        let evicted = engine.radix.evict(10);
        assert_eq!(evicted.len(), 5);
        let mut sorted = evicted.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        Ok(())
    }

    // ─── Full admission cycle integration ───────────────────────────────

    /// Drives the complete worker-owned-KV cycle through the scheduler
    /// using synthetic StepOutputs (no real worker). Verifies that:
    ///
    /// 1. Multiple seqs writing assigned_indices correctly account in
    ///    `KvBudget` (RadixTree append).
    /// 2. Finishing a seq makes its slots LRU-evictable.
    /// 3. The next `evict()` call returns those slots ready for
    ///    `FreeKvIndices` dispatch.
    /// 4. Capacity is restored — outstanding can drop back to a level
    ///    where new work fits.
    ///
    /// This test does **not** spin up real ZMQ transports. It exercises
    /// the engine's hot path (handle_step_output_llm) and the admission
    /// substrate (RadixTree + KvBudget) end-to-end.
    #[tokio::test]
    async fn phase8_full_admission_cycle_through_engine() -> Result<()> {
        use infer_protocol::worker_to_scheduler_data::AssignedIndices;
        let (control_cmd, control_events, _evt_tx, _cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let config = SchedulerConfig {
            paged_block_size: BlockSize::new(4),
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let mut engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            worker_group(), // capacity=32 from fixture
            MockFrontend,
            MockWorker::default(),
            control_cmd,
            control_events,
            default_worker,
        );
        let codec = MsgPackCodec;

        // ── 1. Three seqs each consume 8 slots (total 24/32 = 75%). ──
        // Use distinct first tokens so the seqs hang off different
        // root-level branches and don't collide in the prefix tree.
        for sid in [101u64, 102, 103] {
            let out = StepOutput {
                prefill_done: vec![],
                tokens: vec![GeneratedToken {
                    sequence_id: sid,
                    // Token id chosen so each seq starts on its own branch.
                    token_id: sid as i32,
                    finished: false,
                }],
                assigned_indices: vec![AssignedIndices {
                    sequence_id: sid,
                    base: ((sid - 101) * 8) as u32,
                    len: 8,
                }],
            };
            engine.handle_step_output_llm(codec.encode(&out)?).await?;
        }
        assert_eq!(engine.kv_budget.outstanding(), 24, "3*8=24 reserved");
        assert_eq!(engine.radix.token_count(), 24);
        assert_eq!(engine.radix.lru_len_estimate(), 0, "no chains finished yet");

        // ── 2. Finish seq 102 — its 8 slots must enter LRU. ──
        let finish_out = StepOutput {
            prefill_done: vec![],
            tokens: vec![GeneratedToken {
                sequence_id: 102,
                token_id: 999,
                finished: true,
            }],
            assigned_indices: vec![AssignedIndices {
                sequence_id: 102,
                base: 16, // append one more slot for the final token
                len: 1,
            }],
        };
        engine.handle_step_output_llm(codec.encode(&finish_out)?).await?;
        assert_eq!(
            engine.kv_budget.outstanding(),
            25,
            "24 + 1 final-token slot"
        );
        // Seq 102 chain (9 slots) should now be in LRU.
        assert!(engine.radix.lru_len_estimate() >= 1);

        // ── 3. Eviction recovers exactly the 9 slots seq 102 owned. ──
        let freed = engine.radix.evict(100);
        assert_eq!(freed.len(), 9, "seq 102 had 9 slots total");
        let mut sorted = freed.clone();
        sorted.sort_unstable();
        // Seq 102 used [8..16) and [16..17).
        assert_eq!(sorted, vec![8, 9, 10, 11, 12, 13, 14, 15, 16]);

        // ── 4. After release, capacity returns to its pre-finish level. ──
        engine.kv_budget.release(freed.len() as u32);
        assert_eq!(
            engine.kv_budget.outstanding(),
            16,
            "25 reserved − 9 freed = 16"
        );

        // ── 5. New seq 104 with 16 slots fits exactly into the headroom. ──
        let out104 = StepOutput {
            prefill_done: vec![],
            tokens: vec![GeneratedToken {
                sequence_id: 104,
                token_id: 7,
                finished: false,
            }],
            assigned_indices: vec![AssignedIndices {
                sequence_id: 104,
                base: 100,
                len: 16,
            }],
        };
        engine.handle_step_output_llm(codec.encode(&out104)?).await?;
        assert_eq!(engine.kv_budget.outstanding(), 32, "exactly at capacity");
        // Engine-level cycle confirmed: account → mark finished → evict →
        // reuse capacity.
        Ok(())
    }

    // ─── AllocFailed pressure-relief paths ───────────────────────────

    /// Round 0 with finished leaves in LRU should evict (up to ~5% of
    /// total capacity) and reply with `FreeKvIndices` on the cmd_rx.
    #[tokio::test]
    async fn alloc_failed_round_0_evicts_lru() -> Result<()> {
        use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
        use infer_protocol::worker_to_scheduler_control::AllocFailed;
        let (mut engine, default_worker, _event_tx, mut cmd_rx) = make_engine();
        // worker_group fixture has max_total_kv_tokens=32 → 5% = 1 slot.
        // Plant a single finished chain of 1 token in the radix tree.
        engine.radix.append_token(42, 7, 1000);
        engine.radix.mark_finished_chain(42);
        assert_eq!(engine.radix.lru_total_indices(), 1);
        // Budget must reflect the 1 outstanding slot the radix-planted
        // node represents — relief releases that slot.
        engine.kv_budget.try_reserve(1).unwrap();

        engine
            .on_control_event(crate::infrastructure::transport::control_plane::ControlEvent::AllocFailed {
                worker: default_worker.clone(),
                req: AllocFailed {
                    worker_id: "worker-test".to_string(),
                    shortfall: 4,
                    round: 0,
                },
            })
            .await?;

        let cmd = cmd_rx.try_recv().expect("expected FreeKvIndices");
        match cmd {
            crate::infrastructure::transport::control_plane::handle::RouterCommand::SendTo {
                env,
                ..
            } => match env.payload {
                SchedulerControlMessage::FreeKvIndices(f) => {
                    assert_eq!(f.indices, vec![1000]);
                }
                other => panic!("expected FreeKvIndices, got {:?}", other),
            },
            _ => panic!("expected SendTo router command"),
        }
        Ok(())
    }

    /// Round 0 with empty LRU is a no-op — no scheduler control message
    /// should be emitted (worker's wait_for_relief will time out and
    /// retry at round 1).
    #[tokio::test]
    async fn alloc_failed_lru_total_zero_round_0_noop() -> Result<()> {
        use infer_protocol::worker_to_scheduler_control::AllocFailed;
        let (mut engine, default_worker, _event_tx, mut cmd_rx) = make_engine();
        // RadixTree is empty by default → lru_total_indices == 0.
        assert_eq!(engine.radix.lru_total_indices(), 0);

        engine
            .on_control_event(crate::infrastructure::transport::control_plane::ControlEvent::AllocFailed {
                worker: default_worker.clone(),
                req: AllocFailed {
                    worker_id: "worker-test".to_string(),
                    shortfall: 4,
                    round: 0,
                },
            })
            .await?;
        assert!(
            cmd_rx.try_recv().is_err(),
            "no scheduler control msg expected when LRU is empty"
        );
        Ok(())
    }

    /// Round 1 with two Decoding seqs → victim picks the longer-output
    /// sequence first, sends a `Preempt(sid)`, and the picked seq
    /// transitions back to Queued with `preemption_count` bumped.
    #[tokio::test]
    async fn alloc_failed_round_1_preempts_decoding() -> Result<()> {
        use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
        use infer_protocol::worker_to_scheduler_control::AllocFailed;
        let (mut engine, default_worker, _event_tx, mut cmd_rx) = make_engine();

        // ── Build two Decoding sessions:
        //   seq 11: input=10, output=20  → preferred victim
        //   seq 12: input=20, output=5   → second
        let mut build_decoding = |sid: u64, input_len: usize, output_n: usize| {
            let meta = Arc::new(RequestMeta {
                id: RequestId::new_v4(),
                external_id: format!("ext-{}", sid),
                sequence_id: SequenceId(sid),
                input_ids: (0..input_len as i32).collect(),
                max_tokens: 32,
                sampling: SamplingParams::default(),
                priority: Priority::default(),
                stream: false,
                stop_sequences: vec![],
                ignore_eos: false,
                diffusion: None,
                arrival_time: Instant::now(),
            });
            engine
                .requests
                .insert_new(Arc::clone(&meta), RequestHandle::noop())
                .unwrap();
            let queued = engine.requests.take_waiting(&meta.id).unwrap();
            engine
                .requests
                .commit_prefill_start(
                    queued,
                    crate::infrastructure::kv_cache::traits::PrefixMatch::none(),
                    input_len,
                )
                .unwrap();
            let _ = engine.requests.ack_prefill(SequenceId(sid)).unwrap();
            for k in 0..output_n {
                let _ = engine
                    .requests
                    .append_generated_token(SequenceId(sid), 100 + k as i32, false)
                    .unwrap();
            }
        };
        build_decoding(11, 10, 20);
        build_decoding(12, 20, 5);

        engine
            .on_control_event(crate::infrastructure::transport::control_plane::ControlEvent::AllocFailed {
                worker: default_worker.clone(),
                req: AllocFailed {
                    worker_id: "worker-test".to_string(),
                    shortfall: 4,
                    round: 1,
                },
            })
            .await?;

        // 5% of 32 == 1, so victim selection stops as soon as we pick
        // the first Decoding seq. (output_len=20 wins.)
        let cmd = cmd_rx.try_recv().expect("expected Preempt");
        match cmd {
            crate::infrastructure::transport::control_plane::handle::RouterCommand::SendTo {
                env,
                ..
            } => match env.payload {
                SchedulerControlMessage::Preempt(p) => {
                    assert_eq!(p.sequence_ids, vec![11], "longest-output seq picked");
                }
                other => panic!("expected Preempt, got {:?}", other),
            },
            _ => panic!("expected SendTo router command"),
        }

        // The picked seq is now Queued (front of waiting). Decoding now
        // has only seq 12.
        assert_eq!(engine.requests.decoding_len(), 1);
        let waiting_front = engine.requests.waiting().front().unwrap();
        assert_eq!(waiting_front.meta.sequence_id, SequenceId(11));
        Ok(())
    }

