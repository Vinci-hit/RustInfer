//! Explicit MTP adapter for the existing worker control and data planes.
use super::{MtpProposer, commit::commit_decode, validate_sampling};
use crate::application::decode_common::send_step_error;
use crate::application::execution::{ExecutionPlan, Phase, WorkspaceUse};
use crate::application::serve_execution::{ServingExecution, ServingStep};
use crate::application::worker_scheduler::handle_eager_prefill;
use crate::components::mtp::MtpHead;
use crate::domain::model::DecoderReadout;
use crate::domain::plan::StepRequest;
use crate::domain::ports::{OpError, OpResult};
use crate::infrastructure::cuda::Cuda;
use half::bf16;
use infer_protocol::scheduler_to_worker_data::PrefillBatchCmd;

pub struct MtpServing<H: DecoderReadout<bf16, Cuda>> {
    proposer: MtpProposer<bf16, Cuda, H>,
    owner: Option<u64>,
    draft_tokens: usize,
    request: StepRequest,
    target_hidden: crate::domain::tensor::Tensor<bf16, Cuda>,
}

impl<H: DecoderReadout<bf16, Cuda>> MtpServing<H> {
    pub fn new(
        head: MtpHead<bf16, Cuda, H>,
        max_context: usize,
        max_step_tokens: usize,
        draft_tokens: usize,
        device: &Cuda,
    ) -> OpResult<Self> {
        if draft_tokens == 0 || draft_tokens >= max_step_tokens {
            return Err(OpError::Shape(
                "MTP serving draft width must fit K+1 target rows".into(),
            ));
        }
        let target_hidden =
            crate::domain::tensor::Tensor::zeros([max_step_tokens, head.dims().dim], device)?;
        let mut request = StepRequest::workspace(1, draft_tokens + 1, max_context);
        request.draft_tokens.push(Vec::with_capacity(draft_tokens));
        Ok(Self {
            request,
            target_hidden,
            proposer: MtpProposer::new(
                head,
                max_context,
                max_step_tokens.min(max_context),
                device,
            )?,
            owner: None,
            draft_tokens,
        })
    }

    fn prefill<M: DecoderReadout<bf16, Cuda>>(
        &mut self,
        ctx: &mut ServingStep<'_, M>,
        cmd: &PrefillBatchCmd,
    ) -> OpResult<()> {
        cmd.validate(ctx.runner.cap_num_tokens, 1)
            .map_err(|e| OpError::Shape(e.to_string()))?;
        let seg = &cmd.segments[0];
        if seg.has_multimodal
            || seg.multimodal.is_some()
            || seg.prefix_hint.as_ref().is_some_and(|p| !p.is_empty())
        {
            return Err(OpError::unsupported(
                "MTP serving",
                "multimodal inputs or prefix-cache hits",
            ));
        }
        let params = crate::domain::ports::SamplingParams {
            temperature: seg.sampling_params.temperature,
            top_k: seg.sampling_params.top_k.max(0) as u32,
            top_p: seg.sampling_params.top_p,
            ..Default::default()
        };
        validate_sampling(&[params], 1)?;
        if seg.segment_start == 0 {
            if ctx.prefilling.contains_key(&seg.sequence_id) {
                return Err(OpError::Shape("duplicate MTP initial prefill".into()));
            }
            ctx.runner.release_sequence(seg.sequence_id);
            self.proposer.reset();
            self.owner = Some(seg.sequence_id);
        } else if self.owner != Some(seg.sequence_id) {
            return Err(OpError::Shape(
                "MTP prefill continuation has no draft history".into(),
            ));
        }
        let wire = handle_eager_prefill(
            cmd,
            ctx.runner,
            ctx.active,
            ctx.prefilling,
            ctx.allocator,
            ctx.eos_ids,
            |runner, req| {
                let mut output = runner.step_with_hidden_into(req, &mut self.target_hidden)?;
                let seq = &req.seqs[0];
                self.proposer.observe(
                    &seq.input_ids,
                    seq.kv_write_start as usize,
                    &self.target_hidden.narrow(0, 0, seq.input_ids.len())?,
                    &runner.scope,
                )?;
                if seq.kv_len_after as usize >= runner.max_seq_len {
                    output.finished[0] = true;
                }
                Ok(output)
            },
        )?;
        ctx.data
            .send_step_output(&wire)
            .map_err(|e| OpError::Kernel(e.to_string()))
    }

    fn decode<M: DecoderReadout<bf16, Cuda>>(
        &mut self,
        ctx: &mut ServingStep<'_, M>,
        id: u64,
    ) -> OpResult<()> {
        let seq = &ctx.active[&id];
        if self.owner != Some(id) || self.proposer.committed_len() + 1 != seq.kv_len {
            return Err(OpError::Shape("MTP target/draft history mismatch".into()));
        }
        validate_sampling(std::slice::from_ref(&seq.sampling), 1)?;
        let remaining = seq.max_tokens.saturating_sub(seq.generated_count);
        let capacity = ctx.runner.max_seq_len.saturating_sub(seq.kv_len);
        if remaining == 0 || capacity == 0 {
            return Err(OpError::Shape("MTP decode has exhausted its budget".into()));
        }
        let k = self
            .draft_tokens
            .min(remaining - 1)
            .min(capacity - 1)
            .min(ctx.allocator.total_free().saturating_sub(1) as usize);
        let mut lease = ctx
            .allocator
            .lease((k + 1) as u32)
            .map_err(|e| OpError::Shape(e.to_string()))?;
        let result = (|| {
            let started = std::time::Instant::now();
            let (drafts, device_input) =
                self.proposer
                    .draft_with_device(seq.last_token, k, &ctx.runner.scope)?;
            let drafted = started.elapsed();
            let req = &mut self.request;
            let staged = &mut req.seqs[0];
            staged.sequence_id = id;
            staged.input_ids.clear();
            staged.input_ids.push(seq.last_token);
            staged.input_ids.extend_from_slice(&drafts);
            staged.positions.clear();
            staged
                .positions
                .extend(seq.kv_len as i32..(seq.kv_len + k + 1) as i32);
            staged.kv_write_start = seq.kv_len as i32;
            staged.kv_len_after = (seq.kv_len + k + 1) as i32;
            staged.block_table.clone_from(&seq.block_table);
            staged.block_table.extend_from_slice(lease.as_slice());
            req.sampling.clear();
            req.sampling.push(seq.sampling);
            req.draft_tokens[0].clear();
            req.draft_tokens[0].extend_from_slice(&drafts);
            req.stop.eos_ids.clear();
            req.stop.eos_ids.extend_from_slice(ctx.eos_ids);
            req.stop.generated_counts.clear();
            req.stop.generated_counts.push(seq.generated_count as u32);
            req.stop.max_tokens.clear();
            req.stop.max_tokens.push(seq.max_tokens as u32);
            req.stop.ignore_eos.clear();
            req.stop.ignore_eos.push(seq.ignore_eos);
            let mut output = ctx.runner.step_with_hidden_input(
                req,
                &mut self.target_hidden,
                Some(&device_input),
            )?;
            let verified = started.elapsed();
            let kept = output.materialized_tokens[0] as usize;
            self.proposer.observe_with_input(
                &req.seqs[0].input_ids[..kept],
                seq.kv_len,
                &self.target_hidden.narrow(0, 0, kept)?,
                &ctx.runner.scope,
                Some(&device_input.narrow(0, 0, kept)?),
            )?;
            if seq.kv_len + kept >= ctx.runner.max_seq_len {
                output.finished[0] = true;
            }
            tracing::debug!(
                sequence_id = id,
                proposed = k,
                accepted = output.accepted_drafts.as_ref().unwrap()[0],
                emitted = output.tokens[0].len(),
                draft_ms = drafted.as_secs_f64() * 1e3,
                verify_ms = (verified - drafted).as_secs_f64() * 1e3,
                catchup_ms = (started.elapsed() - verified).as_secs_f64() * 1e3,
                "MTP round"
            );
            Ok(output)
        })();
        let output = match result {
            Ok(out) => out,
            Err(e) => {
                lease.release(ctx.allocator);
                return Err(e);
            }
        };
        let accepted = output.accepted_drafts.as_ref().unwrap()[0] as usize;
        let emitted = output.tokens[0].len();
        let materialized = output.materialized_tokens[0] as usize;
        let wire = ExecutionPlan::eager(Phase::Commit, 1, materialized, WorkspaceUse::Runtime)
            .execute(&ctx.runner.execution_metrics, |_| {
                commit_decode(ctx.active, ctx.allocator, id, lease.take(), output)
            })?;
        ctx.runner
            .execution_metrics
            .committed(k, accepted, emitted, materialized);
        ctx.data
            .send_step_output(&wire)
            .map_err(|e| OpError::Kernel(e.to_string()))
    }
}

impl<M: DecoderReadout<bf16, Cuda>, H: DecoderReadout<bf16, Cuda>> ServingExecution<M>
    for MtpServing<H>
{
    const SPECULATIVE: bool = true;
    fn prepare(
        &mut self,
        runner: &crate::application::runtime::Runtime<bf16, Cuda, M>,
    ) -> OpResult<()> {
        self.proposer.prepare_metrics(&runner.scope)
    }
    fn step(&mut self, mut ctx: ServingStep<'_, M>) -> OpResult<()> {
        if ctx.runner.cap_batch != 1 || ctx.active.len() + ctx.prefilling.len() > 1 {
            return Err(OpError::Shape(
                "MTP serving requires one active request".into(),
            ));
        }
        if self
            .owner
            .is_some_and(|id| !ctx.active.contains_key(&id) && !ctx.prefilling.contains_key(&id))
        {
            self.proposer.reset();
            self.owner = None;
        }
        let (ids, result) = if let Some(&id) = ctx.active.keys().next() {
            (vec![id], self.decode(&mut ctx, id))
        } else if let Some(index) = ctx.prefills.iter().position(|cmd| {
            ctx.prefilling.is_empty()
                || cmd
                    .segments
                    .iter()
                    .any(|s| ctx.prefilling.contains_key(&s.sequence_id))
        }) {
            let cmd = ctx.prefills.remove(index);
            let ids = cmd
                .segments
                .iter()
                .map(|s| s.sequence_id)
                .collect::<Vec<_>>();
            (ids, self.prefill(&mut ctx, &cmd))
        } else {
            return Ok(());
        };
        if let Err(error) = result {
            if error.is_fatal() {
                return Err(error);
            }
            for &id in &ids {
                if let Some(seq) = ctx.active.remove(&id) {
                    ctx.allocator.release_owned(&seq.block_table, false);
                }
                if let Some(seq) = ctx.prefilling.remove(&id) {
                    ctx.allocator.release_owned(&seq.block_table, false);
                }
                ctx.runner.release_sequence(id);
            }
            self.proposer.reset();
            self.owner = None;
            send_step_error(ctx.control, ids, error.to_string());
        }
        Ok(())
    }
}
