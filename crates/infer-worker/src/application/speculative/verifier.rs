use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult, SampledToken, SamplingParams};
use crate::domain::speculative::{DraftBatch, Verification};
use crate::domain::tensor::Tensor;

/// Deterministic prefix verification. Backend argmax resolves ties, exactly as
/// in ordinary greedy sampling; CUDA transfers only the per-row token IDs.
pub struct GreedyVerifier;

impl GreedyVerifier {
    pub fn verify<T: Dtype, D: LlmBackend>(
        &self,
        target_logits: &Tensor<T, D>,
        drafts: &[Vec<i32>],
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<Vec<Verification>> {
        let plan = ctx.plan();
        validate_sampling(params, plan.batch)?;
        if drafts.len() != plan.batch {
            return Err(OpError::Shape(format!(
                "GreedyVerifier: {} draft sequences != batch {}",
                drafts.len(),
                plan.batch
            )));
        }
        let batch = DraftBatch::new(drafts, &plan.q_lens, plan.num_tokens)?;
        let shape = target_logits.shape().as_slice();
        if shape.len() != 2 || shape[0] != batch.target_rows() {
            return Err(OpError::Shape(format!(
                "GreedyVerifier: expected logits [{}, vocab], got {shape:?}",
                batch.target_rows()
            )));
        }
        if !target_logits.is_contiguous() {
            return Err(OpError::Shape(
                "GreedyVerifier: logits must be contiguous".into(),
            ));
        }
        let vocab = shape[1];
        if vocab == 0 || vocab > i32::MAX as usize {
            return Err(OpError::Shape(format!(
                "GreedyVerifier: vocabulary size {vocab} must fit a positive i32"
            )));
        }
        for seq in batch.sequences() {
            validate_token_ids(seq.drafts, vocab, "draft")?;
        }
        let target_ids = D::argmax(ctx, target_logits)?;
        verify_predictions(&target_ids, batch, vocab)
    }
}

/// Preflight validation, usable before running the target or writing caches.
/// Empty parameters select the ordinary greedy defaults for every sequence.
pub fn validate_sampling(params: &[SamplingParams], batch: usize) -> OpResult<()> {
    if !params.is_empty() && params.len() != batch {
        return Err(OpError::Shape(format!(
            "GreedyVerifier: {} sampling parameters != batch {batch}",
            params.len()
        )));
    }
    for param in params {
        if !param.temperature.is_finite() || param.temperature < 0.0 {
            return Err(OpError::Shape(
                "GreedyVerifier: temperature must be finite and non-negative".into(),
            ));
        }
        if !param.top_p.is_finite() || !(0.0..=1.0).contains(&param.top_p) {
            return Err(OpError::Shape(
                "GreedyVerifier: top_p must be in [0, 1]".into(),
            ));
        }
        if !param.min_p.is_finite() || !(0.0..=1.0).contains(&param.min_p) {
            return Err(OpError::Shape(
                "GreedyVerifier: min_p must be in [0, 1]".into(),
            ));
        }
        if param.repetition_penalty != 1.0 {
            return Err(OpError::unsupported(
                "worker",
                "greedy verification with repetition penalty requires token history",
            ));
        }
        if !param.is_greedy() {
            return Err(OpError::unsupported(
                "worker",
                "stochastic speculative verification requires draft distributions",
            ));
        }
    }
    Ok(())
}

fn validate_token_ids(ids: &[i32], vocab: usize, source: &str) -> OpResult<()> {
    if let Some(&id) = ids.iter().find(|&&id| id < 0 || id as usize >= vocab) {
        return Err(OpError::Shape(format!(
            "GreedyVerifier: {source} token {id} outside vocabulary [0, {vocab})"
        )));
    }
    Ok(())
}

fn verify_predictions(
    target_ids: &[i32],
    batch: DraftBatch<'_>,
    vocab: usize,
) -> OpResult<Vec<Verification>> {
    if target_ids.len() != batch.target_rows() {
        return Err(OpError::Shape(format!(
            "GreedyVerifier: backend returned {} predictions for {} target rows",
            target_ids.len(),
            batch.target_rows()
        )));
    }
    validate_token_ids(target_ids, vocab, "target")?;
    let mut decisions = Vec::with_capacity(batch.batch_size());
    for seq in batch.sequences() {
        let predicted = &target_ids[seq.target_rows];
        let accepted_drafts = seq
            .drafts
            .iter()
            .zip(predicted)
            .take_while(|(draft, target)| draft == target)
            .count();
        decisions.push(Verification {
            accepted_drafts,
            correction_or_bonus: SampledToken {
                token_id: predicted[accepted_drafts],
                // Match ordinary greedy sampling, which does not compute a
                // probability distribution or top-logprob list on this path.
                logprob: 0.0,
                top_logprobs: Vec::new(),
            },
        });
    }
    Ok(decisions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::exec::HostScope;
    use crate::domain::plan::{BatchKind, BatchPlan, MaskMode};
    use crate::infrastructure::cpu::Cpu;

    fn plan(q_lens: &[i32]) -> BatchPlan {
        BatchPlan {
            kind: BatchKind::Spec {
                mask: MaskMode::Causal,
                mask_handle: None,
            },
            num_tokens: q_lens.iter().map(|&q| q as usize).sum(),
            batch: q_lens.len(),
            q_lens: q_lens.to_vec(),
            kv_lens: q_lens.to_vec(),
            seq_positions: vec![0; q_lens.len()],
            rope_positions: q_lens.iter().flat_map(|&q| 0..q).collect(),
            max_blocks_per_seq: 2,
            block_size: 16,
            total_q_tiles: 1,
        }
    }

    fn logits(predictions: &[i32], vocab: usize) -> Tensor<f32, Cpu> {
        let mut values = vec![-1.0; predictions.len() * vocab];
        for (row, &token) in predictions.iter().enumerate() {
            values[row * vocab + token as usize] = 4.0;
        }
        Tensor::from_host_slice(&values, [predictions.len(), vocab], &Cpu).unwrap()
    }

    fn decisions(predictions: &[i32], drafts: &[Vec<i32>]) -> Vec<(usize, i32)> {
        let scope = HostScope::new(Cpu);
        let q_lens = drafts
            .iter()
            .map(|draft| draft.len() as i32 + 1)
            .collect::<Vec<_>>();
        let plan = plan(&q_lens);
        let ctx = StepCtx::new(&scope, &plan);
        GreedyVerifier
            .verify(&logits(predictions, 8), drafts, &[], &ctx)
            .unwrap()
            .into_iter()
            .map(|decision| {
                (
                    decision.accepted_drafts,
                    decision.correction_or_bonus.token_id,
                )
            })
            .collect()
    }

    #[test]
    fn first_rejection_uses_the_first_prediction_and_accepts_no_later_drafts() {
        assert_eq!(decisions(&[4, 2, 3], &[vec![1, 2]]), vec![(0, 4)]);
    }

    #[test]
    fn partial_acceptance_uses_the_rejection_row() {
        assert_eq!(decisions(&[1, 4, 3, 5], &[vec![1, 2, 3]]), vec![(1, 4)]);
    }

    #[test]
    fn full_acceptance_uses_an_independent_bonus_row() {
        assert_eq!(decisions(&[1, 2, 7], &[vec![1, 2]]), vec![(2, 7)]);
        assert_eq!(decisions(&[1, 7], &[vec![1]]), vec![(1, 7)]);
    }

    #[test]
    fn zero_drafts_still_requires_one_target_prediction() {
        assert_eq!(decisions(&[6], &[vec![]]), vec![(0, 6)]);
    }

    #[test]
    fn ragged_sequences_have_independent_prefixes_and_bonus_rows() {
        assert_eq!(
            decisions(&[1, 2, 3, 4, 6, 7], &[vec![1, 2], vec![], vec![5]]),
            vec![(2, 3), (0, 4), (0, 6)]
        );
    }

    #[test]
    fn tied_logits_follow_the_backend_lowest_id_rule() {
        let scope = HostScope::new(Cpu);
        let plan = plan(&[2]);
        let ctx = StepCtx::new(&scope, &plan);
        let logits =
            Tensor::from_host_slice(&[3.0f32, 3.0, 0.0, 0.0, 5.0, 5.0], [2, 3], &Cpu).unwrap();
        let result = GreedyVerifier
            .verify(&logits, &[vec![0]], &[], &ctx)
            .unwrap();
        assert_eq!(result[0].accepted_drafts, 1);
        assert_eq!(result[0].correction_or_bonus.token_id, 1);
    }

    #[test]
    fn rejects_non_greedy_and_invalid_sampling_parameters() {
        let default = SamplingParams::default();
        assert!(validate_sampling(&[default], 2).is_err());
        for param in [
            SamplingParams {
                temperature: 1.0,
                ..default
            },
            SamplingParams {
                temperature: -1.0,
                ..default
            },
            SamplingParams {
                temperature: f32::NAN,
                ..default
            },
            SamplingParams {
                top_p: f32::NAN,
                ..default
            },
            SamplingParams {
                top_p: -0.1,
                ..default
            },
            SamplingParams {
                top_p: 1.1,
                ..default
            },
            SamplingParams {
                min_p: f32::NAN,
                ..default
            },
            SamplingParams {
                min_p: -0.1,
                ..default
            },
            SamplingParams {
                min_p: 1.1,
                ..default
            },
            SamplingParams {
                repetition_penalty: 1.1,
                ..default
            },
        ] {
            assert!(validate_sampling(&[param], 1).is_err(), "{param:?}");
        }
        assert!(validate_sampling(&[], 2).is_ok());
        assert!(validate_sampling(&[default], 1).is_ok());
        assert!(
            validate_sampling(
                &[SamplingParams {
                    temperature: 1.0,
                    top_k: 1,
                    ..default
                }],
                1
            )
            .is_ok()
        );
        assert!(
            validate_sampling(
                &[SamplingParams {
                    temperature: 1.0,
                    top_p: 0.0,
                    ..default
                }],
                1
            )
            .is_ok()
        );
    }

    #[test]
    fn rejects_bad_logits_and_plan_shapes_before_backend_dispatch() {
        let scope = HostScope::new(Cpu);
        let valid_plan = plan(&[2]);
        let ctx = StepCtx::new(&scope, &valid_plan);
        for logits in [
            Tensor::<f32, Cpu>::zeros([2], &Cpu).unwrap(),
            Tensor::zeros([1, 4], &Cpu).unwrap(),
            Tensor::zeros([3, 4], &Cpu).unwrap(),
            Tensor::zeros([2, 0], &Cpu).unwrap(),
            Tensor::zeros([2, 4], &Cpu)
                .unwrap()
                .narrow(1, 0, 2)
                .unwrap(),
        ] {
            assert!(
                GreedyVerifier
                    .verify(&logits, &[vec![1]], &[], &ctx)
                    .is_err()
            );
        }
        assert!(
            GreedyVerifier
                .verify(&logits(&[1, 2], 4), &[], &[], &ctx)
                .is_err()
        );
        assert!(
            GreedyVerifier
                .verify(&logits(&[1, 2], 4), &[vec![]], &[], &ctx)
                .is_err()
        );
        let mut wrong_batch = valid_plan.clone();
        wrong_batch.batch = 2;
        assert!(
            GreedyVerifier
                .verify(
                    &logits(&[1, 2], 4),
                    &[vec![1]],
                    &[],
                    &StepCtx::new(&scope, &wrong_batch)
                )
                .is_err()
        );
        let mut wrong_rows = valid_plan;
        wrong_rows.num_tokens = 1;
        assert!(
            GreedyVerifier
                .verify(
                    &logits(&[1, 2], 4),
                    &[vec![1]],
                    &[],
                    &StepCtx::new(&scope, &wrong_rows)
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_out_of_vocabulary_draft_and_backend_ids() {
        let scope = HostScope::new(Cpu);
        let plan = plan(&[2]);
        let ctx = StepCtx::new(&scope, &plan);
        let logits = logits(&[1, 2], 4);
        for draft in [-1, 4] {
            assert!(
                GreedyVerifier
                    .verify(&logits, &[vec![draft]], &[], &ctx)
                    .is_err()
            );
        }
        let drafts = [vec![1]];
        let batch = DraftBatch::new(&drafts, &[2], 2).unwrap();
        assert!(verify_predictions(&[1], batch, 4).is_err());
        assert!(verify_predictions(&[1, 2, 3], batch, 4).is_err());
        assert!(verify_predictions(&[1, -1], batch, 4).is_err());
        assert!(verify_predictions(&[1, 4], batch, 4).is_err());
    }
}
