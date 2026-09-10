//! Incremental serving adapter. Uses the resident model, activation workspace,
//! recurrent slots and the same physical KV allocator as ordinary decoding.
use super::beam_search::{BeamSearch, BeamSearchConfig};
use super::runtime::Runtime;
use crate::domain::{
    dtype::Dtype,
    exec::ExecScope,
    global_kv_alloc::GlobalKvAllocator,
    model::DecoderModel,
    ports::{OpError, OpResult, backend::LlmBackend},
    tensor::Tensor,
};
use infer_protocol::{
    beam::{BeamCommand, BeamOutput},
    scheduler_to_server::{InferenceMetrics, InferenceResponse, ResponseStatus},
};

pub(crate) struct BeamServing<D: LlmBackend> {
    ids: Tensor<i32, D>,
    logprobs: Tensor<f32, D>,
    active: Option<Active>,
}
struct Active {
    command: BeamCommand,
    search: BeamSearch,
    slots: Vec<u32>,
    started: std::time::Instant,
}

impl<D: LlmBackend> BeamServing<D> {
    pub(crate) fn reserve<T: Dtype, M: DecoderModel<T, D>>(
        runtime: &mut Runtime<T, D, M>,
        eos_count: usize,
    ) -> OpResult<Self> {
        runtime.prepare_beam_forks()?;
        let k = runtime
            .cap_batch
            .saturating_mul(eos_count.saturating_add(1))
            .min(runtime.dims.vocab_size);
        let n = runtime
            .cap_batch
            .checked_mul(k)
            .ok_or_else(|| OpError::Shape("beam candidate capacity overflow".into()))?;
        Ok(Self {
            ids: Tensor::zeros([n], runtime.scope.device())?,
            logprobs: Tensor::zeros([n], runtime.scope.device())?,
            active: None,
        })
    }
    pub(crate) fn sequence_id(&self) -> Option<u64> {
        self.active.as_ref().map(|a| a.command.sequence_id)
    }
    pub(crate) fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub(crate) fn start<T: Dtype, M: DecoderModel<T, D>>(
        &mut self,
        command: &BeamCommand,
        runtime: &Runtime<T, D, M>,
        allocator: &mut GlobalKvAllocator,
        eos: &[i32],
    ) -> Result<(), String> {
        let result = (|| -> OpResult<Active> {
            let request = &command.request;
            let options = request
                .beam
                .as_ref()
                .ok_or_else(|| OpError::Shape("missing beam options".into()))?;
            options
                .validate(
                    runtime
                        .cap_batch
                        .min(runtime.cap_num_tokens)
                        .min(runtime.dims.vocab_size),
                )
                .map_err(OpError::Shape)?;
            if self.active.is_some()
                || runtime.scope.topology().tp.size != 1
                || request.stream
                || request.multimodal.is_some()
                || request.max_tokens == 0
                || request.input_ids.is_empty()
                || request
                    .input_ids
                    .iter()
                    .any(|&id| id < 0 || id as usize >= runtime.dims.vocab_size)
                || request.input_ids.len().saturating_add(request.max_tokens) > runtime.max_seq_len
            {
                return Err(OpError::Shape("invalid serving beam request".into()));
            }
            let capacity = options
                .width
                .checked_mul(request.max_tokens)
                .and_then(|n| n.checked_add(request.input_ids.len()))
                .filter(|&n| n <= u32::MAX as usize)
                .ok_or_else(|| OpError::Shape("beam KV capacity overflow".into()))?;
            let slots = allocator
                .alloc_indices(capacity as u32)
                .map_err(|e| OpError::Shape(e.to_string()))?;
            let config = BeamSearchConfig {
                width: options.width,
                max_context: runtime.max_seq_len,
                max_step_tokens: runtime.cap_num_tokens,
                max_new_tokens: request.max_tokens,
                length_penalty: options.length_penalty,
                eos_ids: if request.ignore_eos {
                    vec![]
                } else {
                    eos.to_vec()
                },
            };
            let search = match BeamSearch::new(
                request.input_ids.clone(),
                config,
                slots.clone(),
                request.stop_sequences.clone(),
            ) {
                Ok(search) => search,
                Err(error) => {
                    allocator.free(&slots);
                    return Err(error);
                }
            };
            Ok(Active {
                command: command.clone(),
                search,
                slots,
                started: std::time::Instant::now(),
            })
        })();
        match result {
            Ok(active) => {
                self.active = Some(active);
                Ok(())
            }
            Err(error) => Err(error.to_string()),
        }
    }

    pub(crate) fn step<T: Dtype, M: DecoderModel<T, D>>(
        &mut self,
        runtime: &mut Runtime<T, D, M>,
        allocator: &mut GlobalKvAllocator,
        cancelled: &mut Option<u64>,
    ) -> OpResult<Option<BeamOutput>> {
        let Some(active) = self.active.as_mut() else {
            return Ok(None);
        };
        let result = if *cancelled == Some(active.command.sequence_id) {
            *cancelled = None;
            Err(OpError::Shape("beam request cancelled".into()))
        } else {
            active
                .search
                .step(runtime, &mut self.ids, &mut self.logprobs)
        };
        if result.as_ref().is_err_and(|error| error.is_fatal()) {
            return result.map(|_| None);
        }
        if matches!(result, Ok(None)) {
            return Ok(None);
        }
        // No slot is returned until all outstanding writes have completed.
        runtime.scope.synchronize()?;
        let active = self.active.take().unwrap();
        for id in 0..active.command.request.beam.as_ref().unwrap().width {
            runtime.release_sequence(id as u64);
        }
        allocator.free(&active.slots);
        let elapsed = active.started.elapsed();
        let response = match result {
            Ok(Some(mut beams)) => {
                let mut best = beams.remove(0);
                let mut stopped = best.ended_with_eos;
                for stop in &active.command.request.stop_sequences {
                    if !stop.is_empty() && best.token_ids.ends_with(stop) {
                        best.token_ids.truncate(best.token_ids.len() - stop.len());
                        stopped = true;
                        break;
                    }
                }
                let count = best.token_ids.len() as u32;
                InferenceResponse {
                    request_id: active.command.request.request_id.clone(),
                    status: ResponseStatus::Success,
                    output_token_ids: best.token_ids,
                    images: vec![],
                    finish_reason: Some(if stopped { "stop" } else { "length" }.into()),
                    error: None,
                    metrics: InferenceMetrics {
                        total_ms: elapsed.as_millis() as u64,
                        num_tokens: count,
                        tokens_per_second: count as f64 / elapsed.as_secs_f64().max(1e-9),
                    },
                }
            }
            Err(error) => {
                if error.is_fatal() {
                    return Err(error);
                }
                return Ok(Some(
                    error_output(active.command, error.to_string())
                        .with_kv_usage(allocator.outstanding()),
                ));
            }
            Ok(None) => unreachable!(),
        };
        Ok(Some(BeamOutput::Finished {
            sequence_id: active.command.sequence_id,
            allocated_kv_tokens: Some(allocator.outstanding()),
            response,
        }))
    }
}

pub(crate) fn error_output(command: BeamCommand, message: String) -> BeamOutput {
    BeamOutput::Finished {
        sequence_id: command.sequence_id,
        allocated_kv_tokens: None,
        response: InferenceResponse {
            request_id: command.request.request_id,
            status: ResponseStatus::Error,
            output_token_ids: vec![],
            images: vec![],
            finish_reason: None,
            error: Some(message),
            metrics: InferenceMetrics {
                total_ms: 0,
                num_tokens: 0,
                tokens_per_second: 0.0,
            },
        },
    }
}
