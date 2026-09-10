//! Single-sequence eager reference engine. Owns both caches and commits a round
//! only after target verification, recurrent recovery, and MTP catch-up succeed.
use super::MtpProposer;
use crate::application::runtime::Runtime;
use crate::application::sampler_stack::GreedySampler;
use crate::components::mtp::MtpHead;
use crate::domain::dtype::Dtype;
use crate::domain::exec::ExecScope;
use crate::domain::model::DecoderReadout;
use crate::domain::plan::{SeqStep, StepRequest, StopCriteria};
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};

#[derive(Clone, Debug)]
pub struct MtpLimits {
    pub max_context: usize,
    pub max_step_tokens: usize,
    pub max_output_tokens: u32,
    pub draft_tokens: usize,
    pub eos_ids: Vec<i32>,
}

#[derive(Debug)]
pub struct MtpStep {
    pub tokens: Vec<i32>,
    pub proposed: usize,
    /// Consecutive matching drafts before EOS truncation, not a KV increment.
    pub accepted: usize,
    pub finished: bool,
}

/// Static dispatch for both the target and head. This engine is deliberately
/// explicit: constructing a normal Runtime never loads or enables speculation.
/// It owns a dedicated single-request KV allocation, not a serving scheduler lease.
pub struct MtpSession<T: Dtype, D: LlmBackend, M: DecoderReadout<T, D>, H: DecoderReadout<T, D>> {
    target: Runtime<T, D, M>,
    proposer: MtpProposer<T, D, H>,
    limits: MtpLimits,
    blocks: Vec<u32>,
    len: usize,
    pending: Option<i32>,
    generated: u32,
    finished: bool,
    poisoned: bool,
    target_hidden: crate::domain::tensor::Tensor<T, D>,
}

impl<T: Dtype, D: LlmBackend, M: DecoderReadout<T, D>, H: DecoderReadout<T, D>>
    MtpSession<T, D, M, H>
{
    pub fn new(
        model: M,
        head: MtpHead<T, D, H>,
        scope: D::Scope,
        limits: MtpLimits,
    ) -> OpResult<Self> {
        if limits.max_context == 0
            || limits.max_context > i32::MAX as usize
            || limits.max_step_tokens == 0
            || limits.max_step_tokens > limits.max_context
            || limits.draft_tokens >= limits.max_step_tokens
            || limits.max_output_tokens == 0
            || model.dims().dim != head.dims().dim
            || model.dims().vocab_size != head.dims().vocab_size
            || limits
                .eos_ids
                .iter()
                .any(|&id| id < 0 || id as usize >= model.dims().vocab_size)
            || scope.topology().tp.size != 1
        {
            return Err(OpError::Shape(
                "invalid MTP session limits or model dimensions".into(),
            ));
        }
        let proposer = MtpProposer::new(
            head,
            limits.max_context,
            limits.max_step_tokens,
            scope.device(),
        )?;
        let block_size = 16;
        let num_blocks = limits.max_context.div_ceil(block_size);
        let target_hidden = crate::domain::tensor::Tensor::zeros(
            [limits.max_step_tokens, model.dims().dim],
            scope.device(),
        )?;
        let mut target = Runtime::new(
            model,
            scope,
            Box::new(GreedySampler),
            num_blocks,
            block_size,
            num_blocks,
            limits.max_context,
            limits.max_step_tokens,
            1,
            vec![],
        )?;
        target.prepare_speculative()?;
        Ok(Self {
            target_hidden,
            target,
            proposer,
            blocks: (0..num_blocks as u32).collect(),
            limits,
            len: 0,
            pending: None,
            generated: 0,
            finished: false,
            poisoned: false,
        })
    }

    pub fn cached_tokens(&self) -> usize {
        self.len
    }
    pub fn draft_cached_tokens(&self) -> usize {
        self.proposer.committed_len()
    }

    /// Clear ownership before reusing this engine for another prompt. Old KV
    /// bytes are unreachable at length zero and are overwritten during prefill.
    pub fn reset(&mut self) -> OpResult<()> {
        self.target.scope.synchronize()?;
        self.target.release_sequence(0);
        self.proposer.reset();
        self.len = 0;
        self.pending = None;
        self.generated = 0;
        self.finished = false;
        self.poisoned = false;
        Ok(())
    }

    pub fn prefill(&mut self, prompt: &[i32]) -> OpResult<MtpStep> {
        if self.poisoned
            || self.pending.is_some()
            || self.len != 0
            || prompt.is_empty()
            || prompt.len() > self.limits.max_context
            || prompt
                .iter()
                .any(|&id| id < 0 || id as usize >= self.target.dims.vocab_size)
        {
            return Err(OpError::Shape(
                "MTP prefill requires a fresh session and valid prompt".into(),
            ));
        }
        let result = (|| {
            let mut next = 0;
            for ids in prompt.chunks(self.limits.max_step_tokens) {
                let request = self.request(ids, vec![]);
                let output = self
                    .target
                    .step_with_hidden_into(&request, &mut self.target_hidden)?;
                self.proposer.observe(
                    ids,
                    self.len,
                    &self.target_hidden.narrow(0, 0, ids.len())?,
                    &self.target.scope,
                )?;
                self.len += ids.len();
                next = output.tokens[0][0].token_id;
            }
            self.pending = Some(next);
            self.generated = 1;
            self.finished = self.limits.eos_ids.contains(&next)
                || self.generated >= self.limits.max_output_tokens
                || self.len >= self.limits.max_context;
            Ok(MtpStep {
                tokens: vec![next],
                proposed: 0,
                accepted: 0,
                finished: self.finished,
            })
        })();
        self.finish(result)
    }

    pub fn decode(&mut self) -> OpResult<MtpStep> {
        if self.poisoned || self.finished || self.pending.is_none() {
            return Err(OpError::Shape(
                "MTP decode requires an active healthy session".into(),
            ));
        }
        let result = (|| {
            let available = (self.limits.max_output_tokens - self.generated) as usize;
            let capacity = self.limits.max_context - self.len;
            let k = self
                .limits
                .draft_tokens
                .min(available - 1)
                .min(capacity - 1);
            let (drafts, device_input) =
                self.proposer
                    .draft_with_device(self.pending.unwrap(), k, &self.target.scope)?;
            let mut ids = Vec::with_capacity(k + 1);
            ids.push(self.pending.unwrap());
            ids.extend_from_slice(&drafts);
            let req = self.request(&ids, vec![drafts]);
            let output = self.target.step_with_hidden_input(
                &req,
                &mut self.target_hidden,
                Some(&device_input),
            )?;
            let retained = output.materialized_tokens[0] as usize;
            self.proposer.observe_with_input(
                &ids[..retained],
                self.len,
                &self.target_hidden.narrow(0, 0, retained)?,
                &self.target.scope,
                Some(&device_input.narrow(0, 0, retained)?),
            )?;
            let tokens = output.tokens[0]
                .iter()
                .map(|t| t.token_id)
                .collect::<Vec<_>>();
            self.len += retained;
            self.generated += tokens.len() as u32;
            self.pending = tokens.last().copied();
            self.finished = output.finished[0] || self.len >= self.limits.max_context;
            debug_assert_eq!(self.proposer.committed_len() + 1, self.len);
            Ok(MtpStep {
                tokens,
                proposed: k,
                accepted: output.accepted_drafts.as_ref().unwrap()[0] as usize,
                finished: self.finished,
            })
        })();
        self.finish(result)
    }

    fn finish(&mut self, result: OpResult<MtpStep>) -> OpResult<MtpStep> {
        if result.is_err() {
            self.poisoned = true;
            self.target.release_sequence(0);
        }
        result
    }

    fn request(&self, ids: &[i32], draft_tokens: Vec<Vec<i32>>) -> StepRequest {
        StepRequest {
            seqs: vec![SeqStep {
                sequence_id: 0,
                input_ids: ids.to_vec(),
                positions: (self.len as i32..(self.len + ids.len()) as i32).collect(),
                kv_write_start: self.len as i32,
                kv_len_after: (self.len + ids.len()) as i32,
                block_table: self.blocks.clone(),
            }],
            sampling: vec![],
            stop: StopCriteria {
                eos_ids: self.limits.eos_ids.clone(),
                generated_counts: vec![self.generated],
                max_tokens: vec![self.limits.max_output_tokens],
                ignore_eos: vec![false],
            },
            draft_tokens,
        }
    }
}
