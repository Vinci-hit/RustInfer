//! `IngestionSystem` — entrypoint for new client requests.
//!
//! Responsibilities:
//!
//! 1. Reject disabled diffusion payloads and validate the LLM protocol payload:
//!    non-empty `input_ids`, length within `config.max_model_len`.
//! 2. Mint a fresh `RequestId` (uuid-backed) and the next monotonic
//!    `SequenceId`.
//! 3. Build the `RequestMeta` aggregate root and hand it to
//!    `RequestTable::insert_new`.
//! 4. Report success/rejection through an `IngestOutcome` enum so
//!    the engine can drive metrics + tracing without us depending
//!    on either.
//!
//! Owning the `SequenceId` counter inside this System (instead of
//! the engine) keeps the engine's job purely "wire systems together".

use std::sync::Arc;
use std::time::Instant;

use infer_protocol::server_to_scheduler::{InferenceModality, InferenceRequest};

use crate::config::{SchedulerConfig, SchedulerMode};
use crate::domain::inference_session::handle::{ClientId, RequestHandle};
use crate::domain::inference_session::lifecycle::{
    Priority, RequestId, RequestMeta, SamplingParams, SequenceId,
};
use crate::domain::inference_session::table::RequestTable;
use crate::error::SchedulerError;

/// Outcome of [`IngestionSystem::ingest`].
#[derive(Debug)]
pub enum IngestOutcome {
    /// Request was admitted into the waiting queue. Carries the
    /// minted internal id for tracing / metrics.
    Admitted {
        request_id: RequestId,
        sequence_id: SequenceId,
        external_id: String,
    },
    /// Request was rejected before reaching the queue (validation
    /// failure or duplicate). Carries a human-readable reason.
    Rejected {
        external_id: String,
        reason: RejectReason,
    },
}

/// Why a request was turned away at ingestion.
#[derive(Debug, Clone)]
pub enum RejectReason {
    /// LLM request had empty `input_ids`.
    EmptyPrompt,
    /// LLM prompt longer than `config.max_model_len`.
    PromptTooLong { len: usize, limit: usize },
    /// LLM max_tokens must be positive.
    MaxTokensZero,
    /// Prompt + generated tokens would exceed the model context window.
    TotalTokensTooLong { requested: usize, limit: usize },
    /// Diffusion requests are disabled in this release.
    DiffusionDisabled,
    /// `RequestTable::insert_new` rejected the entry (duplicate id /
    /// duplicate sequence). Usually indicates a bug, not user input.
    Repository(String),
}

impl RejectReason {
    pub fn as_message(&self) -> String {
        match self {
            Self::EmptyPrompt => "empty input_ids".to_string(),
            Self::PromptTooLong { len, limit } => {
                format!("prompt length {} exceeds max_model_len {}", len, limit)
            }
            Self::MaxTokensZero => "max_tokens must be > 0".to_string(),
            Self::TotalTokensTooLong { requested, limit } => {
                format!(
                    "prompt length + max_tokens {} exceeds max_model_len {}",
                    requested, limit
                )
            }
            Self::DiffusionDisabled => "diffusion is disabled in this release".to_string(),
            Self::Repository(msg) => format!("repository rejected: {}", msg),
        }
    }
}

/// New-request ingestion stage.
///
/// Holds the monotonic `SequenceId` counter. No async, no IO.
#[derive(Debug)]
pub struct IngestionSystem {
    next_sequence_id: u64,
}

impl Default for IngestionSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl IngestionSystem {
    pub fn new() -> Self {
        // SequenceId(0) is reserved as a sentinel "unassigned" value
        // in some debug paths; start at 1 to keep parity with the
        // previous engine counter.
        Self {
            next_sequence_id: 1,
        }
    }

    /// Validate + admit a single inbound request.
    ///
    /// On `Admitted`, the session is already in
    /// `RequestTable`'s waiting queue; on `Rejected`, the
    /// repository is untouched.
    pub fn ingest(
        &mut self,
        client_id: ClientId,
        request: InferenceRequest,
        config: &SchedulerConfig,
        sessions: &mut RequestTable,
    ) -> IngestOutcome {
        let external_id = request.request_id.clone();
        if request.modality == InferenceModality::Diffusion
            || request.diffusion.is_some()
            || matches!(config.mode, SchedulerMode::Diffusion)
        {
            return IngestOutcome::Rejected {
                external_id,
                reason: RejectReason::DiffusionDisabled,
            };
        }

        if let Err(reason) = self.validate(&request, config) {
            return IngestOutcome::Rejected {
                external_id,
                reason,
            };
        }

        let sequence_id = SequenceId(self.next_sequence_id);
        self.next_sequence_id += 1;

        let meta = Arc::new(RequestMeta {
            id: RequestId::new_v4(),
            external_id: external_id.clone(),
            sequence_id,
            input_ids: request.input_ids,
            max_tokens: request.max_tokens,
            sampling: SamplingParams {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            },
            priority: Priority(request.priority),
            stream: request.stream,
            stop_sequences: request.stop_sequences,
            ignore_eos: request.ignore_eos,
            diffusion: None,
            arrival_time: Instant::now(),
        });
        let request_id = meta.id;

        let handle = RequestHandle::new(client_id, request.stream);
        if let Err(SchedulerError::Internal(msg)) = sessions.insert_new(meta, handle) {
            return IngestOutcome::Rejected {
                external_id,
                reason: RejectReason::Repository(msg),
            };
        }

        IngestOutcome::Admitted {
            request_id,
            sequence_id,
            external_id,
        }
    }

    /// Pure validation. Borrowed `&self` because the counter does not
    /// advance for rejections.
    fn validate(
        &self,
        request: &InferenceRequest,
        config: &SchedulerConfig,
    ) -> Result<(), RejectReason> {
        if request.input_ids.is_empty() {
            return Err(RejectReason::EmptyPrompt);
        }
        if request.input_ids.len() > config.max_model_len {
            return Err(RejectReason::PromptTooLong {
                len: request.input_ids.len(),
                limit: config.max_model_len,
            });
        }
        if request.max_tokens == 0 {
            return Err(RejectReason::MaxTokensZero);
        }
        // vLLM 语义：prompt + max_tokens 必须 ≤ max_model_len。worker 的
        // SeqStep validate 在 kv_len_after > max_seq_len 时会抛 Shape error，
        // 所以这里要兜底 cap（OpenAI server 已先 cap，但其他客户端也可能
        // 直连 scheduler，避免硬错）。
        let total = request.input_ids.len().saturating_add(request.max_tokens);
        if total > config.max_model_len {
            return Err(RejectReason::TotalTokensTooLong {
                requested: total,
                limit: config.max_model_len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SchedulerConfig;
    use infer_protocol::server_to_scheduler::InferenceModality;

    fn config_llm(max_len: usize) -> SchedulerConfig {
        SchedulerConfig {
            max_model_len: max_len,
            mode: SchedulerMode::Llm,
            ..Default::default()
        }
    }

    fn dummy_llm_request(id: &str, prompt_tokens: usize) -> InferenceRequest {
        InferenceRequest {
            request_id: id.to_string(),
            modality: InferenceModality::Llm,
            input_ids: (0..prompt_tokens as i32).collect(),
            max_tokens: 16,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            priority: 0,
            stream: false,
            stop_sequences: vec![],
            ignore_eos: false,
            diffusion: None,
        }
    }

    #[test]
    fn admits_valid_llm_request() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_llm(4096);
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_llm_request("req-1", 10),
            &cfg,
            &mut sessions,
        );
        match outcome {
            IngestOutcome::Admitted {
                external_id,
                sequence_id,
                ..
            } => {
                assert_eq!(external_id, "req-1");
                assert_eq!(sequence_id.0, 1);
            }
            other => panic!("expected Admitted, got {:?}", other),
        }
        assert_eq!(sessions.active_count(), 1);
    }

    #[test]
    fn preserves_server_tokenized_stop_sequences() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_llm(4096);
        let mut request = dummy_llm_request("req-stop", 10);
        request.stop_sequences = vec![vec![7, 8], vec![9]];

        let outcome = sys.ingest(ClientId::dummy(), request, &cfg, &mut sessions);
        assert!(matches!(outcome, IngestOutcome::Admitted { .. }));
        assert_eq!(
            sessions.waiting().front().unwrap().meta.stop_sequences,
            vec![vec![7, 8], vec![9]]
        );
    }

    #[test]
    fn rejects_empty_prompt_in_llm_mode() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_llm(4096);
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_llm_request("req-empty", 0),
            &cfg,
            &mut sessions,
        );
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::EmptyPrompt,
                ..
            }
        ));
        assert_eq!(sessions.active_count(), 0);
    }

    #[test]
    fn rejects_oversize_prompt() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_llm(8);
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_llm_request("req-big", 100),
            &cfg,
            &mut sessions,
        );
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::PromptTooLong { len: 100, limit: 8 },
                ..
            }
        ));
    }

    #[test]
    fn rejects_diffusion_modality_when_disabled() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_llm(4096);
        let mut request = dummy_llm_request("img-1", 1);
        request.modality = InferenceModality::Diffusion;
        let outcome = sys.ingest(ClientId::dummy(), request, &cfg, &mut sessions);
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::DiffusionDisabled,
                ..
            }
        ));
        assert_eq!(sessions.active_count(), 0);
    }

    #[test]
    fn rejects_hidden_diffusion_payload_and_diffusion_mode() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let mut payload_request = dummy_llm_request("img-payload", 1);
        payload_request.diffusion = Some(Default::default());
        let outcome = sys.ingest(
            ClientId::dummy(),
            payload_request,
            &config_llm(4096),
            &mut sessions,
        );
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::DiffusionDisabled,
                ..
            }
        ));

        let cfg = SchedulerConfig {
            mode: SchedulerMode::Diffusion,
            ..Default::default()
        };
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_llm_request("img-mode", 1),
            &cfg,
            &mut sessions,
        );
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::DiffusionDisabled,
                ..
            }
        ));
        assert_eq!(sessions.active_count(), 0);
    }

    #[test]
    fn sequence_ids_are_monotonic_and_skip_zero() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_llm(4096);
        let mut ids = Vec::new();
        for i in 0..3 {
            let outcome = sys.ingest(
                ClientId::dummy(),
                dummy_llm_request(&format!("req-{}", i), 4),
                &cfg,
                &mut sessions,
            );
            match outcome {
                IngestOutcome::Admitted { sequence_id, .. } => ids.push(sequence_id.0),
                other => panic!("expected Admitted, got {:?}", other),
            }
        }
        assert_eq!(ids, vec![1, 2, 3]);
    }
}
