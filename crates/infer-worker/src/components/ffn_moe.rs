use crate::components::ffn_dense::DenseFfn;
use crate::components::linear::Linear;
use crate::components::norm::RmsNorm;
use crate::domain::component::{Component, Hidden, StageKind};
use crate::domain::dtype::Dtype;
use crate::domain::exec::StepCtx;
use crate::domain::kv::KvView;
use crate::domain::ports::backend::LlmBackend;
use crate::domain::ports::{OpError, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::Shape;

/// Pre-norm sparse MoE FFN sublayer. Same `Component` contract as `DenseFfn`,
/// so swapping it into a `DecoderBlock` is the entire MoE model change. Owns its
/// post-attention norm; routed experts and the optional shared expert both read
/// the normalized input and accumulate into the residual.
pub struct MoeFfn<T: Dtype, D: LlmBackend> {
    pub post_attention_layernorm: RmsNorm<T, D>,
    pub router: Linear<T, D>,
    pub expert_gate_up: Tensor<T, D>,
    pub expert_down: Tensor<T, D>,
    pub shared: Option<DenseFfn<T, D>>,
    pub experts_per_tok: usize,
}

impl<T: Dtype, D: LlmBackend> Component<T, D> for MoeFfn<T, D> {
    fn kind(&self) -> StageKind {
        StageKind::Ffn
    }

    fn run(
        &self,
        hidden: &mut Hidden<T, D>,
        kv: Option<&mut KvView<'_, T, D>>,
        ctx: &StepCtx<'_, D>,
    ) -> OpResult<()> {
        let _ = kv;
        let hidden_shape = hidden.stream.shape().as_slice();
        let gate_shape = self.expert_gate_up.shape().as_slice();
        let down_shape = self.expert_down.shape().as_slice();
        let router_shape = self.router.weight.shape().as_slice();
        if hidden_shape.len() != 2 || gate_shape.len() != 3 || down_shape.len() != 3 {
            return Err(OpError::Shape(format!(
                "MoeFfn::run: expected hidden [tokens,dim], gate [experts,2*inter,dim], down [experts,dim,inter], got {:?} {:?} {:?}",
                hidden_shape, gate_shape, down_shape
            )));
        }
        let num_tokens = hidden_shape[0];
        let dim = hidden_shape[1];
        let num_experts = gate_shape[0];
        let gate_cols = gate_shape[1];
        let inter = gate_cols / 2;
        if gate_cols == 0 || gate_cols % 2 != 0 || gate_shape[2] != dim {
            return Err(OpError::Shape(format!(
                "MoeFfn::run: invalid gate shape {:?} for dim {}",
                gate_shape, dim
            )));
        }
        if down_shape != [num_experts, dim, inter] {
            return Err(OpError::Shape(format!(
                "MoeFfn::run: down shape {:?} != [{}, {}, {}]",
                down_shape, num_experts, dim, inter
            )));
        }
        if router_shape.len() != 2 || router_shape[0] != num_experts || router_shape[1] != dim {
            return Err(OpError::Shape(format!(
                "MoeFfn::run: router weight shape {:?} != [{}, {}]",
                router_shape, num_experts, dim
            )));
        }
        if self.experts_per_tok == 0 || self.experts_per_tok > num_experts {
            return Err(OpError::Shape(format!(
                "MoeFfn::run: experts_per_tok {} invalid for {} experts",
                self.experts_per_tok, num_experts
            )));
        }

        let dev = hidden.stream.device().clone();
        // Pre-FFN norm — residual stays intact in `hidden.stream`.
        let mut normed = D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?;
        self.post_attention_layernorm
            .forward(&hidden.stream, &mut normed, ctx)?;
        let normed_host = normed.to_host_vec()?;

        let mut router_logits =
            D::alloc_tensor(Shape::from_slice(&[num_tokens, num_experts]), &dev)?;
        self.router.forward(&normed, &mut router_logits, ctx)?;
        let router_host = router_logits.to_host_vec()?;

        let mut by_expert: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_experts];
        for token_idx in 0..num_tokens {
            let row = &router_host[token_idx * num_experts..(token_idx + 1) * num_experts];
            let mut ranked: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .map(|(expert, value)| (expert, T::read_f64(value)))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top = &ranked[..self.experts_per_tok];
            let max = top
                .iter()
                .map(|(_, score)| *score)
                .fold(f64::NEG_INFINITY, f64::max);
            let denom = top
                .iter()
                .map(|(_, score)| (*score - max).exp())
                .sum::<f64>();
            for &(expert, score) in top {
                let weight = if denom > 0.0 {
                    (score - max).exp() / denom
                } else {
                    1.0 / self.experts_per_tok as f64
                };
                by_expert[expert].push((token_idx, weight));
            }
        }

        let grouped_rows = num_tokens * self.experts_per_tok;
        let mut offsets = Vec::with_capacity(num_experts + 1);
        let mut routes = Vec::with_capacity(grouped_rows);
        let mut grouped_input = Vec::with_capacity(grouped_rows * dim);
        offsets.push(0i32);
        for assignments in &by_expert {
            for &(token_idx, weight) in assignments {
                routes.push((token_idx, weight));
                let start = token_idx * dim;
                grouped_input.extend_from_slice(&normed_host[start..start + dim]);
            }
            offsets.push(routes.len() as i32);
        }

        let grouped_input =
            Tensor::from_host_slice(&grouped_input, Shape::from_slice(&[grouped_rows, dim]), &dev)?;
        let expert_offsets =
            Tensor::from_host_slice(&offsets, Shape::from_slice(&[num_experts + 1]), &dev)?;
        let mut gate_up: Tensor<T, D> =
            D::alloc_tensor(Shape::from_slice(&[grouped_rows, gate_cols]), &dev)?;
        let mut swiglu: Tensor<T, D> =
            D::alloc_tensor(Shape::from_slice(&[grouped_rows, inter]), &dev)?;
        let mut expert_out: Tensor<T, D> =
            D::alloc_tensor(Shape::from_slice(&[grouped_rows, dim]), &dev)?;

        D::grouped_expert_gemm(
            ctx,
            &grouped_input,
            &self.expert_gate_up,
            &mut gate_up,
            &expert_offsets,
            None,
            None,
            None,
        )?;
        D::swiglu_packed(ctx, &gate_up, &mut swiglu, grouped_rows, inter)?;
        D::grouped_expert_gemm(
            ctx,
            &swiglu,
            &self.expert_down,
            &mut expert_out,
            &expert_offsets,
            None,
            None,
            None,
        )?;

        let expert_host = expert_out.to_host_vec()?;
        let mut combined = vec![T::write_f64(0.0); num_tokens * dim];
        for (row, &(token_idx, route_weight)) in routes.iter().enumerate() {
            for col in 0..dim {
                let dst = token_idx * dim + col;
                let value = T::read_f64(&combined[dst])
                    + route_weight * T::read_f64(&expert_host[row * dim + col]);
                combined[dst] = T::write_f64(value);
            }
        }
        let moe_out = Tensor::from_host_slice(&combined, Shape::from_slice(&[num_tokens, dim]), &dev)?;
        D::add_inplace(ctx.scope(), &mut hidden.stream, &moe_out)?;

        // Shared expert reuses the already-normalized input (no second norm).
        if let Some(shared) = &self.shared {
            let mut shared_out = D::alloc_tensor(Shape::from_slice(&[num_tokens, dim]), &dev)?;
            shared.project(&normed, &mut shared_out, ctx)?;
            D::add_inplace(ctx.scope(), &mut hidden.stream, &shared_out)?;
        }
        Ok(())
    }
}
