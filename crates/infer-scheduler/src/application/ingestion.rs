//! `IngestionSystem` — entrypoint for new client requests.
//!
//! Responsibilities:
//!
//! 1. Validate the protocol payload:
//!    - LLM mode: non-empty `input_ids`, length within
//!      `config.max_model_len`.
//!    - Diffusion mode: non-empty prompt + non-empty
//!      `prompt_input_ids` (server pre-tokenized).
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
use crate::error::SchedulerError;
use crate::domain::inference_session::handle::{ClientId, RequestHandle};
use crate::domain::inference_session::lifecycle::{Priority, RequestId, RequestMeta, SamplingParams, SequenceId};
use crate::domain::inference_session::table::RequestTable;

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
    /// Diffusion mode but request carried no `diffusion` payload.
    DiffusionPayloadMissing,
    /// Diffusion mode but `diffusion.prompt` was empty.
    DiffusionPromptEmpty,
    /// Diffusion mode but `diffusion.prompt_input_ids` was empty.
    DiffusionPromptIdsEmpty,
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
            Self::DiffusionPayloadMissing => "missing diffusion payload".to_string(),
            Self::DiffusionPromptEmpty => "empty diffusion prompt".to_string(),
            Self::DiffusionPromptIdsEmpty => {
                "empty server-tokenized prompt_input_ids".to_string()
            }
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
        Self { next_sequence_id: 1 }
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
        let is_diffusion = request.modality == InferenceModality::Diffusion
            || matches!(config.mode, SchedulerMode::Diffusion);

        if let Err(reason) = self.validate(&request, config, is_diffusion) {
            return IngestOutcome::Rejected {
                external_id,
                reason,
            };
        }

        let sequence_id = SequenceId(self.next_sequence_id);
        self.next_sequence_id += 1;

        // Diffusion uses a placeholder input_ids for the LLM-typed
        // session machinery; max_tokens is fixed to 1 because the
        // diffusion worker emits one image per request.
        let input_ids = if is_diffusion && request.input_ids.is_empty() {
            vec![0]
        } else {
            request.input_ids
        };
        let max_tokens = if is_diffusion { 1 } else { request.max_tokens };

        let meta = Arc::new(RequestMeta {
            id: RequestId::new_v4(),
            external_id: external_id.clone(),
            sequence_id,
            input_ids,
            max_tokens,
            sampling: SamplingParams {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            },
            priority: Priority(request.priority),
            stream: request.stream,
            stop_sequences: vec![],
            diffusion: request.diffusion,
            arrival_time: Instant::now(),
        });
        let request_id = meta.id.clone();

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
        is_diffusion: bool,
    ) -> Result<(), RejectReason> {
        if is_diffusion {
            let Some(diffusion) = request.diffusion.as_ref() else {
                return Err(RejectReason::DiffusionPayloadMissing);
            };
            if diffusion.prompt.is_empty() {
                return Err(RejectReason::DiffusionPromptEmpty);
            }
            if diffusion.prompt_input_ids.is_empty() {
                return Err(RejectReason::DiffusionPromptIdsEmpty);
            }
            return Ok(());
        }

        if request.input_ids.is_empty() {
            return Err(RejectReason::EmptyPrompt);
        }
        if request.input_ids.len() > config.max_model_len {
            return Err(RejectReason::PromptTooLong {
                len: request.input_ids.len(),
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
    use infer_protocol::server_to_scheduler::{DiffusionRequest, InferenceModality};

    fn config_llm(max_len: usize) -> SchedulerConfig {
        SchedulerConfig {
            max_model_len: max_len,
            mode: SchedulerMode::Llm,
            ..Default::default()
        }
    }

    fn config_diffusion() -> SchedulerConfig {
        SchedulerConfig {
            mode: SchedulerMode::Diffusion,
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
            diffusion: None,
        }
    }

    fn dummy_diffusion_request(id: &str, with_payload: bool, prompt: &str) -> InferenceRequest {
        InferenceRequest {
            request_id: id.to_string(),
            modality: InferenceModality::Diffusion,
            input_ids: vec![],
            max_tokens: 1,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            priority: 0,
            stream: false,
            stop_sequences: vec![],
            diffusion: with_payload.then(|| DiffusionRequest {
                prompt: prompt.to_string(),
                prompt_input_ids: vec![1, 2, 3],
                negative_prompt: None,
                negative_prompt_input_ids: None,
                height: 512,
                width: 512,
                num_inference_steps: 20,
                sigmas: None,
                guidance_scale: 7.5,
                seed: None,
                output_format: "png".to_string(),
            }),
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
    fn rejects_diffusion_without_payload() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_diffusion();
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_diffusion_request("img-1", false, "ignored"),
            &cfg,
            &mut sessions,
        );
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::DiffusionPayloadMissing,
                ..
            }
        ));
    }

    #[test]
    fn rejects_diffusion_with_empty_prompt_string() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_diffusion();
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_diffusion_request("img-2", true, ""),
            &cfg,
            &mut sessions,
        );
        assert!(matches!(
            outcome,
            IngestOutcome::Rejected {
                reason: RejectReason::DiffusionPromptEmpty,
                ..
            }
        ));
    }

    #[test]
    fn admits_valid_diffusion_request() {
        let mut sys = IngestionSystem::new();
        let mut sessions = RequestTable::new();
        let cfg = config_diffusion();
        let outcome = sys.ingest(
            ClientId::dummy(),
            dummy_diffusion_request("img-3", true, "a sunset"),
            &cfg,
            &mut sessions,
        );
        assert!(matches!(outcome, IngestOutcome::Admitted { .. }));
        assert_eq!(sessions.active_count(), 1);
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
