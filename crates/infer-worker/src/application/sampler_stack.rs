use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::sampler::{SampleBatch, Sampler, SamplingParams};
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct GreedySampler;

impl<T: Dtype, D: LlmBackend> Sampler<T, D> for GreedySampler {
    fn sample_with_workspace(
        &self,
        logits: &Tensor<T, D>,
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
        workspace: &Tensor<f32, D>,
        ids: &mut Tensor<i32, D>,
        logprobs: &mut Tensor<f32, D>,
    ) -> OpResult<SampleBatch> {
        let shape = logits.shape().as_slice();
        if shape.len() != 2
            || shape[0] != params.len()
            || params.is_empty()
            || params.iter().any(|p| p.want_logprobs)
            || workspace.numel() <= 1
        {
            return self.sample(logits, params, ctx);
        }
        for &p in params {
            validate_sampling_params(p)?;
        }
        if ids.numel() < params.len() || logprobs.numel() < params.len() {
            return Err(OpError::Shape("sampling output capacity too small".into()));
        }
        for (row, &p) in params.iter().enumerate() {
            let position = ctx
                .plan()
                .seq_positions
                .get(row)
                .copied()
                .unwrap_or_default() as u64;
            let draw = sampling_draw(p.seed, position, row);
            let input = logits
                .narrow(0, row, 1)?
                .view_contiguous([shape[1]].into())?;
            if !D::sample_filtered_into(
                ctx,
                &input,
                p,
                draw,
                &mut ids.narrow(0, row, 1)?,
                &mut logprobs.narrow(0, row, 1)?,
                workspace,
            )? {
                return self.sample(logits, params, ctx);
            }
        }
        let ids = ids.narrow(0, 0, params.len())?.to_host_vec()?;
        let logprobs = logprobs.narrow(0, 0, params.len())?.to_host_vec()?;
        Ok(SampleBatch {
            tokens: ids
                .into_iter()
                .zip(logprobs)
                .map(|(token_id, logprob)| crate::domain::plan::SampledToken {
                    token_id,
                    logprob,
                    top_logprobs: Vec::new(),
                })
                .collect(),
        })
    }

    fn sample(
        &self,
        logits: &Tensor<T, D>,
        params: &[SamplingParams],
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<SampleBatch> {
        let shape = logits.shape().as_slice();
        if shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "GreedySampler::sample: expected 2D logits, got {:?}",
                shape
            )));
        }
        let rows = if shape[0] == ctx.plan().batch {
            // `finalize(LastPerSeq)` has already projected one row per request.
            (0..shape[0]).collect()
        } else {
            sampled_rows(ctx.plan())
        };
        if !params.is_empty() && params.len() != rows.len() {
            return Err(OpError::Shape(format!(
                "GreedySampler::sample: params len {} != sampled rows {}",
                params.len(),
                rows.len()
            )));
        }

        for &params in params {
            validate_sampling_params(params)?;
        }

        // Preserve the device argmax path when every row is deterministic. A
        // stochastic row needs its distribution on the host until the backend
        // exposes a filtered multinomial kernel; mixed batches take that same
        // correctness path so row ordering remains exact.
        if params.is_empty() || params.iter().all(|p| p.is_greedy() && !p.want_logprobs) {
            let ids = D::argmax(ctx, logits)?;
            let mut tokens = Vec::with_capacity(rows.len());
            for row in rows {
                let token_id = *ids.get(row).ok_or_else(|| {
                    OpError::Shape(format!(
                        "GreedySampler::sample: sampled row {} out of argmax range {}",
                        row,
                        ids.len()
                    ))
                })?;
                tokens.push(crate::domain::plan::SampledToken {
                    token_id,
                    logprob: 0.0,
                    top_logprobs: Vec::new(),
                });
            }
            return Ok(SampleBatch { tokens });
        }

        let vocab = shape[1];
        let host = logits.to_host_vec()?;
        let default_params = SamplingParams::default();
        let mut tokens = Vec::with_capacity(rows.len());
        for (seq_index, row) in rows.into_iter().enumerate() {
            let params = params.get(seq_index).unwrap_or(&default_params);
            let position = ctx
                .plan()
                .seq_positions
                .get(seq_index)
                .copied()
                .unwrap_or_default() as u64;
            let draw = sampling_draw(params.seed, position, seq_index);
            let start = row.checked_mul(vocab).ok_or_else(|| {
                OpError::Shape("GreedySampler::sample: logits row offset overflow".into())
            })?;
            let end = start.checked_add(vocab).ok_or_else(|| {
                OpError::Shape("GreedySampler::sample: logits row end overflow".into())
            })?;
            let row_logits = host.get(start..end).ok_or_else(|| {
                OpError::Shape(format!(
                    "GreedySampler::sample: sampled row {} outside logits shape {:?}",
                    row, shape
                ))
            })?;
            tokens.push(sample_filtered_row(row_logits, *params, draw)?);
        }
        Ok(SampleBatch { tokens })
    }
}

fn sampling_draw(seed: Option<u64>, position: u64, row: usize) -> f64 {
    match seed {
        Some(seed) => {
            let mixed = seed
                ^ position.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (row as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            StdRng::seed_from_u64(mixed).random::<f64>()
        }
        None => rand::rng().random::<f64>(),
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

fn validate_sampling_params(params: SamplingParams) -> OpResult<()> {
    if !params.temperature.is_finite() || params.temperature < 0.0 {
        return Err(OpError::Shape(
            "sampling temperature must be finite and non-negative".into(),
        ));
    }
    if !params.top_p.is_finite() || !(0.0..=1.0).contains(&params.top_p) {
        return Err(OpError::Shape("sampling top_p must be in [0, 1]".into()));
    }
    if !params.min_p.is_finite() || !(0.0..=1.0).contains(&params.min_p) {
        return Err(OpError::Shape("sampling min_p must be in [0, 1]".into()));
    }
    if params.repetition_penalty != 1.0 {
        return Err(OpError::unsupported(
            "worker",
            "repetition-penalty sampling requires token history",
        ));
    }
    Ok(())
}

fn sample_filtered_row<T: Dtype>(
    row: &[T],
    params: SamplingParams,
    draw: f64,
) -> OpResult<crate::domain::plan::SampledToken> {
    if row.is_empty() {
        return Err(OpError::Shape("sample_filtered_row: empty vocab".into()));
    }
    if params.is_greedy() {
        let (token_id, _) = argmax_row(row)?;
        return Ok(crate::domain::plan::SampledToken {
            token_id,
            logprob: log_softmax_at(row, token_id as usize) as f32,
            top_logprobs: Vec::new(),
        });
    }

    let inv_temperature = 1.0 / f64::from(params.temperature);
    let mut candidates: Vec<(i32, f64)> = row
        .iter()
        .enumerate()
        .map(|(token_id, value)| {
            let logit = T::read_f64(value) * inv_temperature;
            (
                token_id as i32,
                if logit.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    logit
                },
            )
        })
        .collect();
    let order = |a: &(i32, f64), b: &(i32, f64)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    if params.top_k > 0 && (params.top_k as usize) < candidates.len() {
        let k = params.top_k as usize;
        candidates.select_nth_unstable_by(k, order);
        candidates.truncate(k);
    }
    candidates.sort_unstable_by(order);

    let max = candidates[0].1;
    // Convert scores in place: no second vocabulary-sized allocation.
    let mut weighted = candidates;
    for (_, score) in &mut weighted {
        *score = if max == f64::INFINITY {
            if *score == f64::INFINITY { 1.0 } else { 0.0 }
        } else if max == f64::NEG_INFINITY {
            1.0
        } else {
            (*score - max).exp()
        };
    }

    if params.min_p > 0.0 {
        let threshold = weighted[0].1 * f64::from(params.min_p);
        weighted.retain(|(_, weight)| *weight >= threshold);
    }
    if weighted.is_empty() {
        return Err(OpError::Shape(
            "sample_filtered_row: filtering removed every token".into(),
        ));
    }

    let total = weighted.iter().map(|(_, weight)| weight).sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(OpError::Shape(
            "sample_filtered_row: invalid probability mass".into(),
        ));
    }
    let nucleus = f64::from(params.top_p);
    if nucleus < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0usize;
        for (_, weight) in &weighted {
            cumulative += *weight / total;
            keep += 1;
            if cumulative >= nucleus {
                break;
            }
        }
        weighted.truncate(keep.max(1));
    }

    let filtered_total = weighted.iter().map(|(_, weight)| weight).sum::<f64>();
    let mut target = draw.clamp(0.0, 1.0 - f64::EPSILON) * filtered_total;
    let mut selected = weighted[0].0;
    let mut selected_weight = weighted[0].1;
    for &(token_id, weight) in &weighted {
        selected = token_id;
        selected_weight = weight;
        if target < weight {
            break;
        }
        target -= weight;
    }
    let top_logprobs = if params.want_logprobs {
        weighted
            .iter()
            .take(5)
            .map(|&(token_id, weight)| (token_id, (weight / filtered_total).ln() as f32))
            .collect()
    } else {
        Vec::new()
    };
    Ok(crate::domain::plan::SampledToken {
        token_id: selected,
        logprob: (selected_weight / filtered_total).ln() as f32,
        top_logprobs,
    })
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
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .ok_or_else(|| OpError::Shape("argmax_row: empty vocab".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_then_nucleus_renormalizes_and_breaks_ties_by_id() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 0.8,
            ..SamplingParams::default()
        };
        let selected = sample_filtered_row(&[3f32, 2., 1.], params, 0.99).unwrap();
        assert_eq!(selected.token_id, 1);
        assert!((selected.logprob - (1f32 / (1f32.exp() + 1.)).ln()).abs() < 1e-6);
        let tied = sample_filtered_row(&[0f32; 32], params, 0.99).unwrap();
        assert_eq!(tied.token_id, 1);
        assert!(
            validate_sampling_params(SamplingParams {
                temperature: f32::NAN,
                top_k: 1,
                ..params
            })
            .is_err()
        );
    }

    #[test]
    fn greedy_ties_choose_the_lowest_token_id() {
        assert_eq!(argmax_row(&[23.625f32, 23.625, 0.0]).unwrap().0, 0);
        assert_eq!(argmax_row(&[0.0f32, 23.625, 23.625]).unwrap().0, 1);
    }

    #[test]
    fn nucleus_sampling_keeps_at_least_the_highest_probability_token() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.5,
            ..SamplingParams::default()
        };

        let sampled = sample_filtered_row(&[3.0f32, 2.0, 1.0], params, 0.99).unwrap();

        assert_eq!(sampled.token_id, 0);
    }

    #[test]
    fn top_k_sampling_never_selects_outside_the_candidate_set() {
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 1.0,
            ..SamplingParams::default()
        };

        let sampled = sample_filtered_row(&[3.0f32, 2.0, 100.0, 0.0], params, 0.99).unwrap();

        assert!(matches!(sampled.token_id, 0 | 2));
    }
}
