//! Output helpers — shared types for [`super::output_fns`].
//!
//! The actual response emission, error fan-out, and `RadixTree`
//! extension logic lives in `output_fns.rs` as free functions.
//! This file keeps the [`CompleteOutcome`] struct (shared across
//! LLM and Diffusion paths) and the unit tests.

/// Diagnostic bundle returned by [`super::output_fns::complete_session`].
///
/// The engine's tracing line wants these fields together; instead
/// of forcing a destructure we hand back a small typed record so the
/// log line is unambiguous and future fields (e.g. `prefill_latency`)
/// extend without breaking callers.
#[derive(Debug)]
pub struct CompleteOutcome {
    pub request_id_display: String,
    pub num_tokens: u32,
    pub elapsed_ms: u64,
    pub ttft_ms: u64,
    pub decode_ms: u64,
    pub tokens_per_second: f64,
    pub decode_tokens_per_second: f64,
}

#[cfg(test)]
mod tests {
    use crate::domain::inference_session::handle::RequestHandle;
    use crate::domain::inference_session::lifecycle::{
        Decoding, InferenceSession, Prefilling, Priority, RequestId, RequestMeta, SamplingParams,
        SequenceId,
    };
    use crate::infrastructure::metrics::MetricsRecorder;
    use async_trait::async_trait;
    use infer_protocol::scheduler_to_server::{
        ChunkType, InferenceResponse, ResponseStatus, StreamChunk,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    /// Capturing FrontendTransport: records every send_* call so tests
    /// can assert on emitted responses/chunks.
    #[derive(Default, Clone)]
    struct CapturingFrontend {
        responses: Arc<Mutex<Vec<InferenceResponse>>>,
        chunks: Arc<Mutex<Vec<StreamChunk>>>,
    }

    #[async_trait]
    impl crate::infrastructure::transport::traits::FrontendTransport for CapturingFrontend {
        async fn recv_event(
            &mut self,
        ) -> crate::error::Result<crate::infrastructure::transport::traits::FrontendEvent> {
            Err(crate::error::SchedulerError::Shutdown)
        }
        async fn send_response(
            &mut self,
            _client: &crate::domain::inference_session::handle::ClientId,
            response: InferenceResponse,
        ) -> crate::error::Result<()> {
            self.responses.lock().unwrap().push(response);
            Ok(())
        }
        async fn send_stream_chunk(
            &mut self,
            _client: &crate::domain::inference_session::handle::ClientId,
            chunk: StreamChunk,
        ) -> crate::error::Result<()> {
            self.chunks.lock().unwrap().push(chunk);
            Ok(())
        }
    }

    fn meta_for_test(stream: bool, prompt: Vec<i32>) -> Arc<RequestMeta> {
        Arc::new(RequestMeta {
            id: RequestId::new_v4(),
            external_id: "test-ext".to_string(),
            sequence_id: SequenceId(1),
            input_ids: prompt,
            max_tokens: 8,
            sampling: SamplingParams::default(),
            priority: Priority::default(),
            stream,
            stop_sequences: vec![],
            ignore_eos: false,
            diffusion: None,
            arrival_time: Instant::now(),
        })
    }

    fn prefilling_session(stream: bool) -> InferenceSession<Prefilling> {
        InferenceSession {
            meta: meta_for_test(stream, vec![1, 2, 3, 4]),
            handle: RequestHandle::noop(),
            state: Prefilling {
                num_computed_tokens: 0,
                inflight: None,
                prompt_len: 4,
                prefill_start: Instant::now(),
                resume: None,
            },
        }
    }

    fn decoding_session(stream: bool, output: Vec<i32>) -> InferenceSession<Decoding> {
        InferenceSession {
            meta: meta_for_test(stream, vec![1, 2, 3, 4]),
            handle: RequestHandle::noop(),
            state: Decoding {
                output_tokens: output,
                num_streamed_tokens: 0,
                stop_sequence_matched: false,
                seq_position: 4,
                prompt_len: 4,
                original_prompt_len: 4,
                first_token_time: Instant::now(),
                preemption_count: 0,
            },
        }
    }

    #[tokio::test]
    async fn fail_prefilling_emits_error_response() {
        let mut frontend = CapturingFrontend::default();
        crate::application::output_fns::fail_prefilling_session(
            &mut frontend,
            prefilling_session(false),
            "bad",
        )
        .await
        .unwrap();
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].status, ResponseStatus::Error));
        assert_eq!(resps[0].metrics.num_tokens, 0);
    }

    #[tokio::test]
    async fn fail_decoding_carries_partial_token_count_to_metrics() {
        let mut frontend = CapturingFrontend::default();
        crate::application::output_fns::fail_decoding_session(
            &mut frontend,
            decoding_session(false, vec![1, 2, 3]),
            "bang",
        )
        .await
        .unwrap();
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert_eq!(resps[0].metrics.num_tokens, 3, "3 partial output tokens");
    }

    #[tokio::test]
    async fn fail_decoding_streams_error_chunk_when_stream_true() {
        let mut frontend = CapturingFrontend::default();
        crate::application::output_fns::fail_decoding_session(
            &mut frontend,
            decoding_session(true, vec![42, 43]),
            "boom",
        )
        .await
        .unwrap();
        // Streaming path emits a chunk, not a unary response.
        assert_eq!(frontend.responses.lock().unwrap().len(), 0);
        let chunks = frontend.chunks.lock().unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(matches!(chunks[0].chunk_type, ChunkType::Error));
        assert_eq!(chunks[0].finish_reason.as_deref(), Some("boom"));
    }

    #[tokio::test]
    async fn complete_session_emits_success_response() {
        let mut frontend = CapturingFrontend::default();
        let metrics = MetricsRecorder::new(false);
        let outcome = crate::application::output_fns::complete_session(
            &mut frontend,
            &metrics,
            decoding_session(false, vec![10, 11, 12]),
        )
        .await
        .unwrap();
        // Response is success with the output token list.
        let resps = frontend.responses.lock().unwrap();
        assert_eq!(resps.len(), 1);
        assert!(matches!(resps[0].status, ResponseStatus::Success));
        assert_eq!(resps[0].output_token_ids, vec![10, 11, 12]);
        assert!(outcome.num_tokens >= 1);
    }

    #[tokio::test]
    async fn complete_session_reports_scheduler_stop_reason() {
        let mut frontend = CapturingFrontend::default();
        let metrics = MetricsRecorder::new(false);
        let mut sequence = decoding_session(false, vec![10, 11]);
        sequence.state.stop_sequence_matched = true;

        crate::application::output_fns::complete_session(&mut frontend, &metrics, sequence)
            .await
            .unwrap();

        let responses = frontend.responses.lock().unwrap();
        assert_eq!(responses[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(responses[0].output_token_ids, vec![10, 11]);
    }
}
