use crate::ports::backend::LlmBackend;
use crate::ports::{OpError, OpResult};
use infer_core::dtype::Dtype;
use infer_core::exec::StepCtx;
use infer_core::tensor::Tensor;

/// A single sampled token plus its log-probabilities. The sampling *result*
/// type, defined here next to the `Sampler` interface; the runtime's
/// `StepOutput` re-references it via `infer-worker::domain::plan`.
#[derive(Debug, Clone)]
pub struct SampledToken {
    pub token_id: i32,
    pub logprob: f32,
    pub top_logprobs: Vec<(i32, f32)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub want_logprobs: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: None,
            want_logprobs: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SampleBatch {
    pub tokens: Vec<SampledToken>,
}

#[derive(Debug, Clone)]
pub struct AcceptReject {
    pub accepted_count: Vec<u32>,
    pub bonus_token: Vec<SampledToken>,
}

pub trait Sampler<T: Dtype, D: LlmBackend>: Send + Sync {
    fn sample(
        &self,
        logits: &Tensor<T, D>,
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<SampleBatch> {
        let _ = (params, ctx);
        Err(OpError::unsupported(
            logits.device().name(),
            "sampler.sample",
        ))
    }

    fn probs(
        &self,
        logits: &Tensor<T, D>,
        params: &[SamplingParams],
        out: &mut Tensor<f32, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let _ = (params, out, ctx);
        Err(OpError::unsupported(
            logits.device().name(),
            "sampler.probs",
        ))
    }

    fn verify(
        &self,
        target_logits: &Tensor<T, D>,
        draft_tokens: &[i32],
        draft_probs: &Tensor<f32, D>,
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<AcceptReject> {
        let _ = (draft_tokens, draft_probs, params, ctx);
        Err(OpError::unsupported(
            target_logits.device().name(),
            "sampler.verify",
        ))
    }
}
