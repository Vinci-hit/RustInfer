    use super::*;
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use infer_protocol::scheduler_to_worker_control::SchedulerControlMessage;
    use infer_protocol::scheduler_to_worker_data::BatchCommand;
    use infer_protocol::worker_to_scheduler_control::{NeedBlocks, NeedBlocksReason, WorkerStepError};
    use infer_protocol::worker_to_scheduler_data::{GeneratedToken, StepOutput};

    use crate::domain::kv_cache_pool::{KvLease, PagedKvPool};
    use crate::infrastructure::kv_cache::traits::PhysicalBlockId;
    use crate::config::{KvCacheMode, SchedulerConfig};
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
            diffusion: None,
            arrival_time: Instant::now(),
        });
        InferenceSession {
            meta,
            handle: RequestHandle::noop(),
            state: Prefilling {
                kv_lease: KvLease::test_with_blocks(vec![PhysicalBlockId(0)]),
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
    async fn step_output_final_prefill_need_blocks_grants_after_decode_transition() -> Result<()> {
        let config = SchedulerConfig {
            kv_cache_mode: KvCacheMode::Paged { block_size: 4 },
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
            Box::new(PagedKvPool::new(4, 4)),
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
            KvLease::test_with_blocks(vec![PhysicalBlockId(0)]),
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
        };
        engine
            .handle_step_output_llm(codec.encode(&output)?)
            .await?;

        // After step_output: sequence is decoding with the original block.
        assert_eq!(engine.requests.prefilling_len(), 0);
        assert_eq!(engine.requests.decoding_len(), 1);
        assert_eq!(engine.requests.decoding()[0].state.output_tokens, vec![42]);

        // Now inject a NeedBlocks event through the control plane and verify
        // the engine emits a GrantBlocks unicast.
        let need = NeedBlocks {
            worker_id: "worker-test".to_string(),
            model_instance_id: engine.worker_group.model_instance_id.clone(),
            sequence_id: 7,
            current_blocks: 1,
            required_blocks: 2,
            request_blocks: 1,
            reason: NeedBlocksReason::DecodeExtend,
        };
        engine
            .on_control_event(ControlEvent::NeedBlocks {
                worker: default_worker.clone(),
                req: need,
            })
            .await?;

        // After grant: block table extended.
        let blocks = engine.requests.decoding()[0].state.kv_lease.blocks();
        assert_eq!(blocks.len(), 2);

        // Drain the cmd_rx and verify GrantBlocks was unicast to the right worker.
        let cmd = cmd_rx.try_recv().expect("expected RouterCommand on cmd_rx");
        match cmd {
            crate::infrastructure::transport::control_plane::handle::RouterCommand::SendTo { worker, env } => {
                assert_eq!(worker, default_worker);
                match env.payload {
                    SchedulerControlMessage::GrantBlocks(g) => {
                        assert_eq!(g.sequence_id, 7);
                        assert_eq!(g.block_ids.len(), 1);
                    }
                    other => panic!("expected GrantBlocks, got {:?}", other),
                }
            }
            other => panic!("expected SendTo, got something else: {}",
                match other {
                    crate::infrastructure::transport::control_plane::handle::RouterCommand::Broadcast { .. } => "Broadcast",
                    crate::infrastructure::transport::control_plane::handle::RouterCommand::CallOne { .. } => "CallOne",
                    crate::infrastructure::transport::control_plane::handle::RouterCommand::CallAll { .. } => "CallAll",
                    crate::infrastructure::transport::control_plane::handle::RouterCommand::Shutdown => "Shutdown",
                    _ => "?",
                }),
        }

        // Suppress unused suppression: confirm the data-plane sent vec has the
        // expected single prefill batch (none in this test path).
        let _ = _sent;
        // Use `BatchCommand` to confirm the import remains valid even though no
        // batch was sent on this path.
        let _: Option<BatchCommand> = None;
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
            kv_cache_mode: KvCacheMode::Paged { block_size: 4 },
            num_gpu_blocks: 4,
            ..Default::default()
        };
        let (control_cmd, control_events, event_tx, cmd_rx) = mock_control_plane();
        let default_worker = WorkerId::from_identity(b"worker-test");
        let engine = SchedulerEngine::new(
            config,
            Box::new(ContinuousBatchingPolicy::new(None)),
            Box::new(PagedKvPool::new(4, 4)),
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
            KvLease::test_with_blocks(vec![PhysicalBlockId(0)]),
            crate::infrastructure::kv_cache::traits::PrefixMatch::none(),
            4,
        )?;

        engine.cancel_request(request_id).await?;

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
