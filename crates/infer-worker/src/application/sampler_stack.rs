use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::sampler::{AcceptReject, SampleBatch, Sampler, SamplingParams};
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;

pub struct GreedySampler;

impl<T: Dtype, D: LlmBackend> Sampler<T, D> for GreedySampler {
    fn sample(
        &self,
        logits: &Tensor<T, D>,
        _params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<SampleBatch> {
        let shape = logits.shape().as_slice();
        if shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "GreedySampler::sample: expected 2D logits, got {:?}",
                shape
            )));
        }
        let vocab = shape[1];
        let host = logits.to_host_vec()?;
        let rows = sampled_rows(ctx.plan());
        let mut tokens = Vec::with_capacity(rows.len());
        for row in rows {
            let start = row * vocab;
            let slice = &host[start..start + vocab];
            let (token_id, max_logit) = slice
                .iter()
                .enumerate()
                .map(|(i, v)| (i as i32, T::read_f64(v)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| OpError::Shape("GreedySampler::sample: empty vocab".into()))?;
            let logprob = log_softmax_at(slice, token_id as usize) as f32;
            let _ = max_logit;
            tokens.push(crate::domain::plan::SampledToken {
                token_id,
                logprob,
                top_logprobs: Vec::new(),
            });
        }
        Ok(SampleBatch { tokens })
    }

    fn probs(
        &self,
        logits: &Tensor<T, D>,
        _params: &[SamplingParams],
        out: &mut Tensor<f32, D>,
        _ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let shape = logits.shape().as_slice();
        if shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "GreedySampler::probs: expected 2D logits, got {:?}",
                shape
            )));
        }
        if out.numel() != logits.numel() {
            return Err(OpError::Shape(format!(
                "GreedySampler::probs: out numel {} != logits {}",
                out.numel(),
                logits.numel()
            )));
        }
        let vocab = shape[1];
        let host = logits.to_host_vec()?;
        let mut probs = vec![0.0f32; host.len()];
        for (row_in, row_out) in host.chunks(vocab).zip(probs.chunks_mut(vocab)) {
            let max = row_in
                .iter()
                .map(T::read_f64)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0f64;
            for (src, dst) in row_in.iter().zip(row_out.iter_mut()) {
                let p = (T::read_f64(src) - max).exp();
                *dst = p as f32;
                sum += p;
            }
            if sum > 0.0 {
                for dst in row_out {
                    *dst /= sum as f32;
                }
            }
        }
        out.upload_from_host(&probs)
    }

    fn verify(
        &self,
        target_logits: &Tensor<T, D>,
        draft_tokens: &[i32],
        _draft_probs: &Tensor<f32, D>,
        _params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<AcceptReject> {
        let shape = target_logits.shape().as_slice();
        if shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "GreedySampler::verify: expected 2D logits, got {:?}",
                shape
            )));
        }
        let vocab = shape[1];
        let expected = ctx
            .plan()
            .q_lens
            .iter()
            .map(|&q| q.max(0) as usize)
            .sum::<usize>();
        if draft_tokens.len() != expected {
            return Err(OpError::Shape(format!(
                "GreedySampler::verify: draft_tokens {} != planned tokens {}",
                draft_tokens.len(),
                expected
            )));
        }
        let host = target_logits.to_host_vec()?;
        let mut accepted_count = Vec::with_capacity(ctx.plan().batch);
        let mut bonus_token = Vec::with_capacity(ctx.plan().batch);
        let mut offset = 0usize;
        for &q_len in &ctx.plan().q_lens {
            let q = q_len.max(0) as usize;
            let mut accepted = 0usize;
            for row_in_seq in 0..q {
                let row = offset + row_in_seq;
                let (token_id, _) = argmax_row(&host[row * vocab..(row + 1) * vocab])?;
                if token_id == draft_tokens[row] {
                    accepted += 1;
                } else {
                    break;
                }
            }
            let bonus_row = offset + accepted.min(q.saturating_sub(1));
            let row = &host[bonus_row * vocab..(bonus_row + 1) * vocab];
            let (token_id, _) = argmax_row(row)?;
            bonus_token.push(crate::domain::plan::SampledToken {
                token_id,
                logprob: log_softmax_at(row, token_id as usize) as f32,
                top_logprobs: Vec::new(),
            });
            accepted_count.push(accepted as u32);
            offset += q;
        }
        Ok(AcceptReject {
            accepted_count,
            bonus_token,
        })
    }
}

pub struct ChainSampler;
impl<T: Dtype, D: LlmBackend> Sampler<T, D> for ChainSampler {
    fn sample(
        &self,
        logits: &Tensor<T, D>,
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<SampleBatch> {
        GreedySampler.sample(logits, params, ctx)
    }

    fn probs(
        &self,
        logits: &Tensor<T, D>,
        params: &[SamplingParams],
        out: &mut Tensor<f32, D>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        GreedySampler.probs(logits, params, out, ctx)
    }

    fn verify(
        &self,
        target_logits: &Tensor<T, D>,
        draft_tokens: &[i32],
        draft_probs: &Tensor<f32, D>,
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<AcceptReject> {
        GreedySampler.verify(target_logits, draft_tokens, draft_probs, params, ctx)
    }
}

fn sampled_rows(plan: &crate::domain::plan::BatchPlan) -> Vec<usize> {
    let mut rows = Vec::with_capacity(plan.batch);
    let mut offset = 0usize;
    for &q_len in &plan.q_lens {
        let q = q_len.max(1) as usize;
        rows.push(offset + q - 1);
        offset += q;
    }
    rows
}

fn log_softmax_at<T: Dtype>(row: &[T], idx: usize) -> f64 {
    let max = row
        .iter()
        .map(T::read_f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum = row
        .iter()
        .map(|v| (T::read_f64(v) - max).exp())
        .sum::<f64>();
    T::read_f64(&row[idx]) - max - sum.ln()
}

fn argmax_row<T: Dtype>(row: &[T]) -> OpResult<(i32, f64)> {
    row.iter()
        .enumerate()
        .map(|(i, v)| (i as i32, T::read_f64(v)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| OpError::Shape("argmax_row: empty vocab".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::exec::{HostScope, StepCtx};
    use crate::domain::plan::{BatchKind, BatchPlan};
    use crate::infrastructure::cpu::Cpu;

    #[test]
    fn greedy_verify_returns_per_sequence_acceptance() {
        let cpu = Cpu;
        let scope = HostScope::new(cpu);
        let plan = BatchPlan {
            kind: BatchKind::Spec {
                mask: crate::domain::plan::MaskMode::Causal,
                mask_handle: None,
            },
            num_tokens: 3,
            batch: 2,
            q_lens: vec![2, 1],
            kv_lens: vec![2, 1],
            seq_positions: vec![0, 0],
            rope_positions: vec![0, 1, 0],
            max_blocks_per_seq: 2,
            block_size: 16,
            total_q_tiles: 1,
        };
        let ctx = StepCtx::new(&scope, &plan);
        let logits = Tensor::from_host_slice(
            &[
                0.0f32, 3.0, 1.0, 0.0, // row 0 predicts token 1: accepted
                0.0, 1.0, 4.0, 0.0, // row 1 predicts token 2: rejects draft token 3
                5.0, 0.0, 1.0, 0.0, // row 2 predicts token 0: accepted
            ],
            [3, 4],
            &cpu,
        )
        .unwrap();
        let draft_probs = Tensor::from_host_slice(&[1.0f32], [1], &cpu).unwrap();

        let verdict = GreedySampler
            .verify(&logits, &[1, 3, 0], &draft_probs, &[], &ctx)
            .unwrap();

        assert_eq!(verdict.accepted_count, vec![1, 1]);
        assert_eq!(verdict.bonus_token[0].token_id, 2);
        assert_eq!(verdict.bonus_token[1].token_id, 0);
    }
}
