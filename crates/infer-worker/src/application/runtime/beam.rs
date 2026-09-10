//! Dedicated beam sessions use eager readout and explicit recurrent forking.
use super::*;

impl<T: Dtype, D: LlmBackend, M: DecoderModel<T, D>> Runtime<T, D, M> {
    pub(crate) fn beam_step(
        &mut self,
        req: &StepRequest,
        k: usize,
        ids: &mut Tensor<i32, D>,
        logprobs: &mut Tensor<f32, D>,
    ) -> OpResult<Vec<Vec<(i32, f32)>>> {
        if self.scope.topology().tp.size != 1
            || !req.draft_tokens.is_empty()
            || self.requires_multimodal_prefill(req)
            || k == 0
            || k > self.dims.vocab_size
        {
            return Err(OpError::Shape(
                "beam readout requires TP=1 text input and valid k".into(),
            ));
        }
        let plan = self.build_plan(req)?;
        if ids.numel() < plan.batch * k || logprobs.numel() < plan.batch * k {
            return Err(OpError::Shape("beam readout capacity too small".into()));
        }
        self.prepare_recurrent(req, &plan)?;
        let result = (|| {
            self.upload_index(&plan, req)?;
            let input = self.input_ids_tensor(req, &plan)?;
            let prefill = plan.num_tokens > plan.batch;
            if prefill {
                D::set_prefill_gemm_mode(true);
            }
            let _gemm = PrefillGemmGuard::<D>(prefill, std::marker::PhantomData);
            self.run_layers(&plan, &input)?;
            let hidden = Hidden {
                stream: self.hidden.stream.narrow(0, 0, plan.num_tokens)?,
                pending: None,
            };
            let ctx = crate::domain::exec::StepCtx::new(&self.scope, &plan);
            let _guard = self.scope.enter();
            let logits = self
                .model
                .finalize(&hidden, SampleRows::LastPerSeq, &ctx)?
                .0;
            let mut device = true;
            for row in 0..plan.batch {
                let input = logits
                    .narrow(0, row, 1)?
                    .view_contiguous([self.dims.vocab_size].into())?;
                if !D::beam_candidates_into(
                    &ctx,
                    &input,
                    &mut ids.narrow(0, row * k, k)?,
                    &mut logprobs.narrow(0, row * k, k)?,
                    &self.sampling_workspace,
                )? {
                    device = false;
                    break;
                }
            }
            if device {
                let ids = ids.narrow(0, 0, plan.batch * k)?.to_host_vec()?;
                let probs = logprobs.narrow(0, 0, plan.batch * k)?.to_host_vec()?;
                Ok(ids
                    .chunks(k)
                    .zip(probs.chunks(k))
                    .map(|(ids, probs)| ids.iter().copied().zip(probs.iter().copied()).collect())
                    .collect())
            } else {
                let host = logits.to_host_vec()?;
                host.chunks(self.dims.vocab_size)
                    .map(|row| crate::application::beam_search::top_candidates(row, k))
                    .collect()
            }
        })();
        self.finish_recurrent_step(result)
    }
}
