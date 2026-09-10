//! Exclusive beam admission on the existing worker. Ordinary decodes drain
//! first; continuation prefills remain schedulable while fresh admissions pause.
use super::*;
use infer_protocol::beam::{BeamCommand, BeamOutput};

pub(super) struct PendingBeam {
    client: ClientId,
    command: BeamCommand,
    cancelled: bool,
}

impl SchedulerEngine {
    pub(super) async fn enqueue_beam(
        &mut self,
        client: ClientId,
        request: InferenceRequest,
    ) -> Result<()> {
        let options = request.beam.as_ref().unwrap();
        let invalid = options
            .validate(self.config.max_num_seqs.min(self.config.max_batch_tokens))
            .err()
            .or_else(|| {
                (request.stream
                    || request.multimodal.is_some()
                    || request.diffusion.is_some()
                    || request.modality
                        != infer_protocol::server_to_scheduler::InferenceModality::Llm
                    || self.worker_group.rank_count() != 1
                    || !matches!(self.config.mode, SchedulerMode::Llm)
                    || request.input_ids.is_empty()
                    || request.input_ids.iter().any(|&id| id < 0)
                    || request.max_tokens == 0
                    || request.input_ids.len().saturating_add(request.max_tokens)
                        > self.config.max_model_len)
                    .then(|| {
                        "invalid beam request: non-streaming TP1 text and valid context required"
                            .into()
                    })
            })
            .or_else(|| {
                (self.beam_queue.len() >= self.config.max_num_seqs)
                    .then(|| "beam request queue is full".into())
            });
        if let Some(error) = invalid {
            return crate::application::output_fns::send_request_error(
                self.dispatch.frontend_mut(),
                client,
                request.request_id,
                request.stream,
                error,
                0,
            )
            .await;
        }
        // High-bit IDs are reserved for whole-search control messages; internal
        // beam rows use a separate runtime-local namespace while it is idle.
        let sequence_id = self.next_beam_id;
        self.next_beam_id = self.next_beam_id.checked_add(1).expect("beam ID exhausted");
        self.metrics.record_enqueue();
        self.beam_queue.push_back(PendingBeam {
            client,
            command: BeamCommand {
                sequence_id,
                request,
                free_indices: vec![],
            },
            cancelled: false,
        });
        Ok(())
    }

    /// True means beam work owns admission, including waiting for active decodes.
    pub(super) async fn schedule_beam(&mut self) -> Result<bool> {
        if self.active_beam.is_some() {
            return Ok(true);
        }
        if self.beam_queue.is_empty() {
            return Ok(false);
        }
        if self.requests.prefilling_len() != 0 {
            return Ok(false);
        }
        if self.requests.decoding_len() != 0 {
            return Ok(true);
        }
        let mut pending = self.beam_queue.pop_front().unwrap();
        // Carry evictions with the execution command so separate control/data
        // sockets cannot reorder reclamation after beam allocation.
        let request = &pending.command.request;
        let needed = request.input_ids.len().saturating_add(
            request
                .beam
                .as_ref()
                .unwrap()
                .width
                .saturating_mul(request.max_tokens),
        );
        pending.command.free_indices = self
            .radix
            .evict(needed.saturating_sub(self.kv_budget.headroom() as usize));
        let bytes = self.codec.encode(
            &infer_protocol::scheduler_to_worker_data::BatchCommand::Beam(Box::new(
                pending.command.clone(),
            )),
        )?;
        self.dispatch.send_batch(bytes).await?;
        self.active_beam = Some(pending);
        Ok(true)
    }

    pub(super) async fn finish_beam(&mut self, output: BeamOutput) -> Result<()> {
        let BeamOutput::Finished {
            sequence_id,
            allocated_kv_tokens,
            response,
        } = output;
        if self
            .active_beam
            .as_ref()
            .is_none_or(|p| p.command.sequence_id != sequence_id)
        {
            return Ok(());
        }
        let pending = self.active_beam.take().unwrap();
        if let Some(tokens) = allocated_kv_tokens {
            self.kv_budget.force_set_outstanding(tokens);
        }
        self.kv_budget.set_pending_prefill(0);
        if matches!(response.status, infer_protocol::ResponseStatus::Success) {
            self.metrics
                .record_completion(response.metrics.total_ms, response.metrics.num_tokens);
        }
        if !pending.cancelled {
            self.dispatch
                .frontend_mut()
                .send_response(&pending.client, response)
                .await?;
        }
        // Give ordinary waiters one admission round between beam searches;
        // a stream of new beam requests must not starve ordinary traffic.
        if self.workflow.can_schedule(&self.requests) {
            let (workflow, dispatch, mut ctx) = self.split_for_workflow();
            workflow.try_schedule(&mut ctx, dispatch).await?;
        }
        Ok(())
    }

    pub(super) fn cancel_beam(&mut self, external_id: &str) -> Result<bool> {
        if let Some(pos) = self
            .beam_queue
            .iter()
            .position(|p| p.command.request.request_id == external_id)
        {
            self.beam_queue.remove(pos);
            return Ok(true);
        }
        if let Some(pending) = self.active_beam.as_mut()
            && pending.command.request.request_id == external_id
        {
            pending.cancelled = true;
            self.control_cmd
                .send_to(
                    &self.default_worker,
                    infer_protocol::scheduler_to_worker_control::SchedulerControlMessage::Cancel(
                        infer_protocol::scheduler_to_worker_control::CancelSequence {
                            sequence_id: pending.command.sequence_id,
                        },
                    ),
                )
                .map_err(|e| crate::error::SchedulerError::Internal(e.to_string()))?;
            // Keep the admission barrier until the worker confirms cleanup.
            return Ok(true);
        }
        Ok(false)
    }

    pub(super) async fn fail_beams(&mut self, message: &str) {
        let all = self
            .active_beam
            .take()
            .into_iter()
            .chain(self.beam_queue.drain(..))
            .collect::<Vec<_>>();
        for pending in all {
            if !pending.cancelled {
                let _ = crate::application::output_fns::send_request_error(
                    self.dispatch.frontend_mut(),
                    pending.client,
                    pending.command.request.request_id,
                    false,
                    message.into(),
                    0,
                )
                .await;
            }
        }
    }
}
