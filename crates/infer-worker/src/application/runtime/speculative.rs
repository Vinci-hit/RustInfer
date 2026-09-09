//! Eager verification transactions and caller-owned target hidden readout.
use super::*;
use crate::domain::model::DecoderReadout;

/// Hidden rows correspond to the retained input prefix of each request, packed
/// in request order. Speculative suffix rows are never exposed as committed.
pub struct TargetStep<T: Dtype, D: LlmBackend> {
    pub output: StepOutput,
    pub normalized_hidden: Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> Runtime<T, D, M> {
    fn validate_eager_transaction(&self, req: &StepRequest) -> OpResult<()> {
        if self.scope.topology().tp.size != 1 {
            return Err(OpError::unsupported("target readout", "tensor parallelism"));
        }
        if self
            .recurrent
            .as_ref()
            .is_some_and(|r| r.has_decode_in_flight())
        {
            return Err(OpError::Shape(
                "collect in-flight decode before speculative execution".into(),
            ));
        }
        if self.requires_multimodal_prefill(req)
            || req.seqs.iter().any(|s| {
                self.multimodal_decode_position(s.sequence_id, s.kv_write_start.max(0) as usize)
                    .is_some()
            })
        {
            return Err(OpError::unsupported(
                "speculative execution",
                "multimodal input",
            ));
        }
        for seq in &req.seqs {
            if seq
                .positions
                .iter()
                .enumerate()
                .any(|(i, &p)| seq.kv_write_start.checked_add(i as i32) != Some(p))
                || seq
                    .input_ids
                    .iter()
                    .any(|&id| id < 0 || id as usize >= self.dims.vocab_size)
            {
                return Err(OpError::Shape(
                    "speculative execution requires valid tokens and consecutive text positions"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn step_speculative(
        &mut self,
        req: &StepRequest,
        plan: &BatchPlan,
    ) -> OpResult<(StepOutput, BatchPlan)> {
        self.validate_eager_transaction(req)?;
        self.prepare_recurrent_verification(req, plan)?;
        let result = (|| {
            self.scope.synchronize()?;
            let snapshot = self
                .recurrent
                .as_ref()
                .map(|r| r.snapshot(req))
                .transpose()?;
            infer_core::device::MemoryPort::synchronize(self.scope.device())?;
            self.upload_index(plan, req)?;
            self.prepare_multimodal(req)?;
            let output = self.step_eager(plan, req)?;
            let mut retained = req.clone();
            retained.draft_tokens.clear();
            for (seq, &n) in retained.seqs.iter_mut().zip(&output.materialized_tokens) {
                let n = n as usize;
                if n == 0 || n > seq.input_ids.len() {
                    return Err(OpError::Shape(
                        "invalid speculative retention length".into(),
                    ));
                }
                seq.input_ids.truncate(n);
                seq.positions.truncate(n);
                seq.kv_len_after = seq.kv_write_start + n as i32;
            }
            let retained_plan = self.build_plan(&retained)?;
            if retained_plan.num_tokens != plan.num_tokens {
                // GDN has already consumed rejected inputs. Restore the entire
                // participating batch, then replay just its retained prefixes.
                // Full-attention KV suffix bytes become inaccessible via lengths;
                // the lease owner may release the corresponding blocks.
                self.scope.synchronize()?;
                if let (Some(saved), Some(state)) = (snapshot.as_ref(), self.recurrent.as_mut()) {
                    saved.restore(&mut state.layers)?;
                    infer_core::device::MemoryPort::synchronize(self.scope.device())?;
                    // Logical lengths still refer to the pre-verification prefix.
                    self.prepare_recurrent(&retained, &retained_plan)?;
                }
                self.upload_index(&retained_plan, &retained)?;
                let input = self.input_ids_tensor(&retained, &retained_plan)?;
                self.run_layers(&retained_plan, &input)?;
            }
            if let Some(state) = &mut self.recurrent {
                state.retain_step(&retained);
            }
            // No request may observe committed state until replay completes.
            self.scope.synchronize()?;
            Ok((output, retained_plan))
        })();
        self.finish_recurrent_step(result)
    }
}

impl<T: Dtype, D: LlmBackend, M: DecoderReadout<T, D>> Runtime<T, D, M> {
    /// Explicit text-only, TP=1 eager entry point for a hidden-conditioned
    /// proposer. Output storage survives later forwards. On an execution error,
    /// callers must release their KV lease and rebuild the sequence from zero.
    pub fn step_with_hidden(&mut self, req: &StepRequest) -> OpResult<TargetStep<T, D>> {
        let plan = self.build_plan(req)?;
        self.validate_eager_transaction(req)?;
        let (output, retained_plan) = if req.draft_tokens.is_empty() {
            self.prepare_recurrent(req, &plan)?;
            let result = (|| {
                self.upload_index(&plan, req)?;
                self.prepare_multimodal(req)?;
                self.step_eager(&plan, req)
            })();
            (self.finish_recurrent_step(result)?, plan)
        } else {
            self.step_speculative(req, &plan)?
        };
        let result = (|| {
            let n = retained_plan.num_tokens;
            let mut normalized_hidden = Tensor::zeros([n, self.dims.dim], self.scope.device())?;
            let hidden = Hidden {
                stream: self.hidden.stream.narrow(0, 0, n)?,
                pending: None,
            };
            let ctx = crate::domain::exec::StepCtx::new(&self.scope, &retained_plan);
            self.model
                .normalize_hidden_into(&hidden, &mut normalized_hidden, &ctx)?;
            self.scope.synchronize()?;
            Ok(TargetStep {
                output,
                normalized_hidden,
            })
        })();
        if result.is_err() {
            for seq in &req.seqs {
                self.release_sequence(seq.sequence_id);
            }
        }
        result
    }
}
