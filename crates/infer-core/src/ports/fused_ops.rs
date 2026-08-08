// Fused kernel ports mirror launch signatures, and tensor triples are the
// natural Q/K/V result. These shapes are clearer than one-off argument structs.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use crate::kv::{KvView, LayerKv};
use crate::ports::math_ops::MathOps;
use crate::ports::{OpError, OpResult};
use infer_core::dtype::Dtype;
use infer_core::dtype::quant::QuantScheme;
use infer_core::exec::StepCtx;
use infer_core::tensor::Tensor;
use infer_core::types::Shape;

pub trait FusedOps: MathOps {
    /// Toggle build-free eager-prefill GEMM mode. When `on`, eager (non-graph)
    /// bf16 GEMMs skip the per-shape cuBLASLt heuristic+probe cache build and use
    /// the build-free chunked path, removing ~9-18ms of cold-shape build from
    /// every distinct-length prefill's TTFT. Must be left off for decode (graph
    /// capture + its warmup) so the amortized cuBLASLt cache is populated.
    /// Default: no-op (only the CUDA backend implements it).
    fn set_prefill_gemm_mode(_on: bool) {}

    /// True when the backend serves an EAGER ragged/mixed forward with a
    /// unified single-kernel varlen attention (FA3 on Hopper) for this
    /// dtype/head_dim, so the runtime should default mixed steps to eager
    /// instead of bucketed mixed-graph replay (whose captured region carries
    /// the slower legacy split attention plus row/token padding).
    /// Default: false (only the CUDA backend implements it).
    fn unified_mixed_attention_available<T: Dtype>(_head_dim: usize) -> bool {
        false
    }

    /// Permit the unified varlen attention (FA3) to run *under CUDA graph
    /// capture*. Normally FA3 declines to launch while a stream is capturing
    /// (its grid/`max_q` bounds bake host-side, wrong at replay); the runtime
    /// sets this only around the mixed FA3-graph capture region, where the
    /// bucket plan bakes `max_q` to a proven upper bound over every replay
    /// composition, so the captured FA3 node stays correct. Cleared right
    /// after the region. Default: no-op (only the CUDA backend implements it).
    fn set_unified_mixed_capture(_on: bool) {}

    /// True when the backend can execute the complete single-device routed MoE
    /// path for `T`. The default is false so host and newly added backends never
    /// enter a partially supported MoE sublayer before returning `Unsupported`.
    fn local_moe_available<T: Dtype>() -> bool {
        false
    }

    fn fused_add_rmsnorm<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        output: &mut Tensor<T, Self>,
        residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        Self::add_inplace(ctx.scope(), residual, input)?;
        Self::rmsnorm(ctx.scope(), residual, weight, output, eps)
    }

    fn swiglu_packed<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        let mut gate = Self::alloc_tensor(Shape::from_slice(&[rows, inter]), gate_up.device())?;
        let mut up = Self::alloc_tensor(Shape::from_slice(&[rows, inter]), gate_up.device())?;
        Self::split_cols(ctx.scope(), gate_up, &mut gate, rows, 2 * inter, 0, inter)?;
        Self::split_cols(ctx.scope(), gate_up, &mut up, rows, 2 * inter, inter, inter)?;
        Self::silu_inplace(ctx.scope(), &mut gate)?;
        Self::ewise_mul(ctx.scope(), &gate, &up, out)
    }

    fn split_qkv<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        qkv: &Tensor<T, Self>,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        v: &mut Tensor<T, Self>,
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        let total = q_dim + 2 * kv_dim;
        Self::split_cols(ctx.scope(), qkv, q, num_tokens, total, 0, q_dim)?;
        Self::split_cols(ctx.scope(), qkv, k, num_tokens, total, q_dim, kv_dim)?;
        Self::split_cols(
            ctx.scope(),
            qkv,
            v,
            num_tokens,
            total,
            q_dim + kv_dim,
            kv_dim,
        )
    }

    /// Produce Q/K/V from a fused `[num_tokens, q_dim + 2*kv_dim]` buffer.
    ///
    /// Default: materialize CONTIGUOUS q/k/v (one copy each) — required by
    /// backends whose attention kernels assume a contiguous row stride (the CPU
    /// reference indexes `t*q_dim + ...`). CUDA overrides this to return
    /// zero-copy column narrows of `qkv` (its kernels read row/col strides
    /// directly), so the GPU path allocates and copies nothing here.
    fn qkv_split<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        qkv: &Tensor<T, Self>,
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<(Tensor<T, Self>, Tensor<T, Self>, Tensor<T, Self>)> {
        let dev = qkv.device();
        let mut q = Self::alloc_tensor(Shape::from_slice(&[num_tokens, q_dim]), dev)?;
        let mut k = Self::alloc_tensor(Shape::from_slice(&[num_tokens, kv_dim]), dev)?;
        let mut v = Self::alloc_tensor(Shape::from_slice(&[num_tokens, kv_dim]), dev)?;
        Self::split_qkv(ctx, qkv, &mut q, &mut k, &mut v, num_tokens, q_dim, kv_dim)?;
        Ok((q, k, v))
    }

    fn attention_paged<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        q: &Tensor<T, Self>,
        kv: &KvView<'_, T, Self>,
        output: &mut Tensor<T, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        scale: f32,
        // `Some(ws)` → backend reuses the caller-provided `[f32; >=
        // flash_attention_workspace_capacity_f32(batch, num_tokens,
        // head_num, head_dim)]`
        // scratch (zero alloc, capturable). `None` → backend self-allocates
        // (legacy path). The CPU reference ignores this parameter.
        workspace: Option<&mut Tensor<f32, Self>>,
    ) -> OpResult<()> {
        let _ = workspace;
        attention_paged_reference(ctx, q, kv, output, head_num, kv_head_num, head_dim, scale)
    }

    /// f32 element count required by `attention_paged` for the largest planned
    /// decode or ragged batch. Backends that do not need scratch return 0.
    fn flash_attention_workspace_capacity_f32(
        _batch: usize,
        _num_tokens: usize,
        _num_q_heads: usize,
        _head_dim: usize,
    ) -> usize {
        0
    }

    fn scatter_kv_paged<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        k_src: &Tensor<T, Self>,
        v_src: &Tensor<T, Self>,
        layer: &mut LayerKv<'_, T, Self>,
        kv_dim: usize,
    ) -> OpResult<()> {
        scatter_kv_paged_reference(ctx, k_src, v_src, layer, kv_dim)
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_norm_rope_scatter<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        v: &Tensor<T, Self>,
        q_weight: Option<&Tensor<T, Self>>,
        k_weight: Option<&Tensor<T, Self>>,
        q_eps: f32,
        k_eps: f32,
        sin: &Tensor<T, Self>,
        cos: &Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        layer: &mut LayerKv<'_, T, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        if let Some(weight) = q_weight {
            rmsnorm_heads(q, weight, head_num, head_dim, q_eps)?;
        }
        if let Some(weight) = k_weight {
            rmsnorm_heads(k, weight, kv_head_num, head_dim, k_eps)?;
        }
        Self::rope_inplace(
            ctx.scope(),
            q,
            k,
            sin,
            cos,
            positions,
            head_num,
            kv_head_num,
            head_dim,
        )?;
        Self::scatter_kv_paged(ctx, k, v, layer, kv_dim)
    }

    /// Select routed experts from dense router logits.
    ///
    /// `logits` is `[tokens, experts]`; `expert_ids` and `expert_weights` are
    /// `[tokens, top_k]`. Weights are computed in FP32. When `renormalize` is
    /// true, the selected weights sum to one; otherwise they retain their
    /// probability mass from the full-expert softmax.
    ///
    /// There is deliberately no CPU fallback. Each accelerator backend must
    /// opt in with an implementation before MoE routing is available.
    fn moe_route_topk<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        logits: &Tensor<T, Self>,
        expert_ids: &mut Tensor<i32, Self>,
        expert_weights: &mut Tensor<f32, Self>,
        top_k: usize,
        renormalize: bool,
    ) -> OpResult<()> {
        let _ = (ctx, expert_ids, expert_weights, top_k, renormalize);
        Err(OpError::unsupported(
            logits.device().name(),
            "moe_route_topk",
        ))
    }

    /// Group routed token rows into stable expert-major order.
    ///
    /// `input` is `[tokens, hidden]`; IDs and weights are `[tokens, top_k]`.
    /// The outputs contain exactly `tokens * top_k` routes:
    ///
    /// - `permuted_input`: `[routes, hidden]`, grouped by expert;
    /// - `source_tokens`: `[routes]`, mapping each grouped row back to a token;
    /// - `route_weights`: `[routes]`, reordered with the grouped rows;
    /// - `expert_offsets`: `[experts + 1]`, delimiting each expert's rows.
    ///
    /// Ordering within one expert is stable with respect to flattened
    /// token-major/route-major input order. Expert IDs must be in
    /// `[0, experts)`, as guaranteed by `moe_route_topk`. There is deliberately
    /// no CPU fallback.
    #[allow(clippy::too_many_arguments)]
    fn moe_permute_tokens<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        expert_ids: &Tensor<i32, Self>,
        expert_weights: &Tensor<f32, Self>,
        permuted_input: &mut Tensor<T, Self>,
        source_tokens: &mut Tensor<i32, Self>,
        route_weights: &mut Tensor<f32, Self>,
        expert_offsets: &mut Tensor<i32, Self>,
    ) -> OpResult<()> {
        let _ = (
            ctx,
            expert_ids,
            expert_weights,
            permuted_input,
            source_tokens,
            route_weights,
            expert_offsets,
        );
        Err(OpError::unsupported(
            input.device().name(),
            "moe_permute_tokens",
        ))
    }

    /// Apply one expert matrix to each contiguous group of input rows.
    ///
    /// `input` is `[rows, in]`, `weights` is `[experts, out, in]`, `output` is
    /// `[rows, out]`, and `expert_offsets` is a monotonic `[experts + 1]`
    /// partition spanning `0..rows`. Quantization tensors are backend-specific.
    /// There is deliberately no CPU fallback.
    fn grouped_expert_gemm<A: Dtype, W: Dtype, O: Dtype>(
        ctx: &StepCtx<'_, Self>,
        input: &Tensor<A, Self>,
        weights: &Tensor<W, Self>,
        output: &mut Tensor<O, Self>,
        expert_offsets: &Tensor<i32, Self>,
        scales: Option<&Tensor<A, Self>>,
        zeros: Option<&Tensor<W, Self>>,
        scheme: Option<&QuantScheme>,
    ) -> OpResult<()> {
        let _ = (ctx, weights, output, expert_offsets, scales, zeros, scheme);
        // MoE deliberately has no host reference fallback. A backend must opt
        // in with a real implementation; otherwise CUDA could silently execute
        // an entire expert layer on the CPU through this trait default.
        Err(OpError::unsupported(
            input.device().name(),
            "grouped_expert_gemm",
        ))
    }

    /// Weight expert-major route rows and accumulate them back to token order.
    ///
    /// `expert_output` is `[routes, hidden]`; `source_tokens` and
    /// `route_weights` are `[routes]`; `output` and the FP32 `accumulator` are
    /// `[tokens, hidden]`. Source token indices must be in `0..tokens`, as
    /// guaranteed by `moe_permute_tokens`. There is deliberately no CPU
    /// fallback.
    fn moe_combine<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        expert_output: &Tensor<T, Self>,
        source_tokens: &Tensor<i32, Self>,
        route_weights: &Tensor<f32, Self>,
        output: &mut Tensor<T, Self>,
        accumulator: &mut Tensor<f32, Self>,
    ) -> OpResult<()> {
        let _ = (ctx, source_tokens, route_weights, output, accumulator);
        Err(OpError::unsupported(
            expert_output.device().name(),
            "moe_combine",
        ))
    }

    /// Greedy argmax over the last (vocab) dimension. `logits` is `[rows, vocab]`;
    /// returns the winning column index for every row as a host `Vec<i32>` of
    /// length `rows`.
    ///
    /// Default is a host reference implementation (copies the full logits to
    /// host). The CUDA backend overrides this with an on-device two-phase argmax
    /// that copies back ONLY the per-row ids — avoiding the multi-MB
    /// logits download that otherwise stalls the decode loop at batch > 1.
    fn argmax<T: Dtype>(ctx: &StepCtx<'_, Self>, logits: &Tensor<T, Self>) -> OpResult<Vec<i32>> {
        let _ = ctx;
        let shape = logits.shape().as_slice();
        if shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "argmax: expected 2D logits [rows, vocab], got {:?}",
                shape
            )));
        }
        let rows = shape[0];
        let vocab = shape[1];
        let host = logits.to_host_vec()?;
        let mut ids = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * vocab;
            let slice = &host[start..start + vocab];
            let (idx, _) = slice
                .iter()
                .enumerate()
                .map(|(i, v)| (i as i32, T::read_f64(v)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| OpError::Shape("argmax: empty vocab".into()))?;
            ids.push(idx);
        }
        Ok(ids)
    }

    /// Capturable greedy argmax: writes the per-row winning index into the
    /// caller-provided device buffer `out` using the caller-provided
    /// `workspace` scratch. Allocates nothing, so it is safe to invoke INSIDE
    /// CUDA-graph capture (unlike [`Self::argmax`], which the CUDA backend
    /// implements with a fresh per-call output/workspace).
    ///
    /// `selected_rows`:
    /// - `None` → argmax every row of `logits`; `out` must have `numel == rows`.
    /// - `Some(idx)` → argmax ONLY the rows listed in `idx` (device i32 tensor
    ///   of length `K`); `out` must have `numel == K` and receives one id per
    ///   selected row, in order. Used by prefill to skip the `num_tokens - K`
    ///   rows that the sampler discards anyway (the per-sequence last row is
    ///   the only one whose token id matters).
    ///
    /// Default is a host reference: CPU argmax + host-side row pick, then
    /// upload only the selected ids into `out`.
    fn argmax_into<T: Dtype>(
        ctx: &StepCtx<'_, Self>,
        logits: &Tensor<T, Self>,
        out: &mut Tensor<i32, Self>,
        workspace: &Tensor<f32, Self>,
        selected_rows: Option<&Tensor<i32, Self>>,
    ) -> OpResult<()> {
        let _ = workspace;
        let all_ids = Self::argmax(ctx, logits)?;
        match selected_rows {
            None => out.upload_from_host(&all_ids),
            Some(idx) => {
                let idx_host = idx.to_host_vec()?;
                let mut picked = Vec::with_capacity(idx_host.len());
                for &row in &idx_host {
                    let row = row as usize;
                    let id = all_ids.get(row).copied().ok_or_else(|| {
                        OpError::Shape(format!(
                            "argmax_into: selected row {} >= rows {}",
                            row,
                            all_ids.len()
                        ))
                    })?;
                    picked.push(id);
                }
                out.upload_from_host(&picked)
            }
        }
    }
}

fn scatter_kv_paged_reference<T, D>(
    ctx: &StepCtx<'_, D>,
    k_src: &Tensor<T, D>,
    v_src: &Tensor<T, D>,
    layer: &mut LayerKv<'_, T, D>,
    kv_dim: usize,
) -> OpResult<()>
where
    T: Dtype,
    D: MathOps,
{
    let plan = ctx.plan();
    let k_shape = k_src.shape().as_slice();
    let v_shape = v_src.shape().as_slice();
    if k_shape != [plan.num_tokens, kv_dim] || v_shape != [plan.num_tokens, kv_dim] {
        return Err(OpError::Shape(format!(
            "scatter_kv_paged: expected k/v [{}, {}], got {:?} {:?}",
            plan.num_tokens, kv_dim, k_shape, v_shape
        )));
    }

    validate_plan_index_lengths(plan)?;
    let block_tables = layer.index.block_tables.to_host_vec()?;
    let cu_q_lens = layer.index.cu_q_lens.to_host_vec()?;
    let seq_positions = layer.index.seq_positions.to_host_vec()?;
    let seq_lens_step = layer.index.seq_lens_step.to_host_vec()?;
    let k_host = k_src.to_host_vec()?;
    let v_host = v_src.to_host_vec()?;
    let mut k_pool = layer.k.to_host_vec()?;
    let mut v_pool = layer.v.to_host_vec()?;

    for b in 0..plan.batch {
        let start = cu_q_lens[b] as usize;
        let q_len = seq_lens_step[b].max(0) as usize;
        let write_start = seq_positions[b].max(0) as usize;
        for t in 0..q_len {
            let src_row = start + t;
            if src_row >= plan.num_tokens {
                return Err(OpError::Shape(format!(
                    "scatter_kv_paged: src row {} >= num_tokens {}",
                    src_row, plan.num_tokens
                )));
            }
            let pos = write_start + t;
            let block = block_for_position(&block_tables, plan, b, pos)?;
            let offset = pos % plan.block_size;
            let dst_base = ((block * plan.block_size + offset) * kv_dim) as usize;
            let src_base = src_row * kv_dim;
            for col in 0..kv_dim {
                k_pool[dst_base + col] = T::write_f64(T::read_f64(&k_host[src_base + col]));
                v_pool[dst_base + col] = T::write_f64(T::read_f64(&v_host[src_base + col]));
            }
        }
    }

    layer.k.upload_from_host(&k_pool)?;
    layer.v.upload_from_host(&v_pool)
}

fn attention_paged_reference<T, D>(
    ctx: &StepCtx<'_, D>,
    q: &Tensor<T, D>,
    kv: &KvView<'_, T, D>,
    output: &mut Tensor<T, D>,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<()>
where
    T: Dtype,
    D: MathOps,
{
    let plan = ctx.plan();
    let q_dim = head_num * head_dim;
    let kv_dim = kv_head_num * head_dim;
    if head_num == 0 || kv_head_num == 0 || head_dim == 0 {
        return Err(OpError::Shape(format!(
            "attention_paged: invalid heads head_num={} kv_head_num={} head_dim={}",
            head_num, kv_head_num, head_dim
        )));
    }
    if q.shape().as_slice() != [plan.num_tokens, q_dim]
        || output.shape().as_slice() != [plan.num_tokens, q_dim]
    {
        return Err(OpError::Shape(format!(
            "attention_paged: expected q/output [{}, {}], got {:?} {:?}",
            plan.num_tokens,
            q_dim,
            q.shape().as_slice(),
            output.shape().as_slice()
        )));
    }
    validate_plan_index_lengths(plan)?;

    let block_tables = kv.index.block_tables.to_host_vec()?;
    let cu_q_lens = kv.index.cu_q_lens.to_host_vec()?;
    let seq_positions = kv.index.seq_positions.to_host_vec()?;
    let q_lens = kv.index.seq_lens_step.to_host_vec()?;
    let kv_lens = kv.index.kv_lens.to_host_vec()?;
    let q_host = q.to_host_vec()?;
    let (k_pool, v_pool) = kv.layer(0);
    let k_host = k_pool.to_host_vec()?;
    let v_host = v_pool.to_host_vec()?;
    let mut out_host = (0..plan.num_tokens * q_dim)
        .map(|_| T::write_f64(0.0))
        .collect::<Vec<_>>();

    for b in 0..plan.batch {
        let q_start = cu_q_lens[b] as usize;
        let q_len = q_lens[b].max(0) as usize;
        let seq_start = seq_positions[b].max(0) as usize;
        let seq_kv_len = kv_lens[b].max(0) as usize;
        for tq in 0..q_len {
            let q_row = q_start + tq;
            if q_row >= plan.num_tokens {
                return Err(OpError::Shape(format!(
                    "attention_paged: q row {} >= num_tokens {}",
                    q_row, plan.num_tokens
                )));
            }
            let visible = seq_kv_len.min(seq_start + tq + 1);
            if visible == 0 {
                continue;
            }
            for h in 0..head_num {
                let kv_h = h * kv_head_num / head_num;
                let q_base = q_row * q_dim + h * head_dim;
                let mut scores = Vec::with_capacity(visible);
                for pos in 0..visible {
                    let block = block_for_position(&block_tables, plan, b, pos)?;
                    let offset = pos % plan.block_size;
                    let k_base = (block * plan.block_size + offset) * kv_dim + kv_h * head_dim;
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        dot += T::read_f64(&q_host[q_base + d]) * T::read_f64(&k_host[k_base + d]);
                    }
                    scores.push(dot * scale as f64);
                }

                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let denom = scores.iter().map(|score| (*score - max).exp()).sum::<f64>();
                if denom <= 0.0 {
                    continue;
                }
                let out_base = q_row * q_dim + h * head_dim;
                let mut acc = vec![0.0f64; head_dim];
                for (pos, score) in scores.iter().enumerate() {
                    let weight = (*score - max).exp() / denom;
                    let block = block_for_position(&block_tables, plan, b, pos)?;
                    let offset = pos % plan.block_size;
                    let v_base = (block * plan.block_size + offset) * kv_dim + kv_h * head_dim;
                    for d in 0..head_dim {
                        acc[d] += weight * T::read_f64(&v_host[v_base + d]);
                    }
                }
                for d in 0..head_dim {
                    out_host[out_base + d] = T::write_f64(acc[d]);
                }
            }
        }
    }

    output.upload_from_host(&out_host)
}

fn rmsnorm_heads<T, D>(
    x: &mut Tensor<T, D>,
    weight: &Tensor<T, D>,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> OpResult<()>
where
    T: Dtype,
    D: MathOps,
{
    let shape = x.shape().as_slice();
    if shape.len() != 2 || shape[1] != heads * head_dim {
        return Err(OpError::Shape(format!(
            "rmsnorm_heads: x shape {:?} != [rows, {}]",
            shape,
            heads * head_dim
        )));
    }
    let weight_len = weight.numel();
    if weight_len != head_dim && weight_len != heads * head_dim {
        return Err(OpError::Shape(format!(
            "rmsnorm_heads: weight len {} != {} or {}",
            weight_len,
            head_dim,
            heads * head_dim
        )));
    }

    let rows = shape[0];
    let mut x_host = x.to_host_vec()?;
    let weight_host = weight.to_host_vec()?;
    for row in 0..rows {
        for head in 0..heads {
            let base = row * heads * head_dim + head * head_dim;
            let mean_square = (0..head_dim)
                .map(|d| {
                    let v = T::read_f64(&x_host[base + d]);
                    v * v
                })
                .sum::<f64>()
                / head_dim as f64;
            let inv = 1.0 / (mean_square + eps as f64).sqrt();
            for d in 0..head_dim {
                let weight_idx = if weight_len == head_dim {
                    d
                } else {
                    head * head_dim + d
                };
                let value =
                    T::read_f64(&x_host[base + d]) * inv * T::read_f64(&weight_host[weight_idx]);
                x_host[base + d] = T::write_f64(value);
            }
        }
    }
    x.upload_from_host(&x_host)
}

fn validate_plan_index_lengths(plan: &infer_core::plan::BatchPlan) -> OpResult<()> {
    if plan.block_size == 0 || plan.max_blocks_per_seq == 0 {
        return Err(OpError::Shape(format!(
            "invalid plan block_size={} max_blocks_per_seq={}",
            plan.block_size, plan.max_blocks_per_seq
        )));
    }
    if plan.q_lens.len() != plan.batch
        || plan.kv_lens.len() != plan.batch
        || plan.seq_positions.len() != plan.batch
    {
        return Err(OpError::Shape(format!(
            "plan vector length mismatch batch={} q={} kv={} pos={}",
            plan.batch,
            plan.q_lens.len(),
            plan.kv_lens.len(),
            plan.seq_positions.len()
        )));
    }
    Ok(())
}

fn block_for_position(
    block_tables: &[i32],
    plan: &infer_core::plan::BatchPlan,
    batch_idx: usize,
    position: usize,
) -> OpResult<usize> {
    let block_slot = position / plan.block_size;
    if block_slot >= plan.max_blocks_per_seq {
        return Err(OpError::Shape(format!(
            "position {} requires block slot {} >= max {}",
            position, block_slot, plan.max_blocks_per_seq
        )));
    }
    let table_idx = batch_idx * plan.max_blocks_per_seq + block_slot;
    let Some(&block) = block_tables.get(table_idx) else {
        return Err(OpError::Shape(format!(
            "block table index {} out of range {}",
            table_idx,
            block_tables.len()
        )));
    };
    if block < 0 {
        return Err(OpError::Shape(format!(
            "negative block id {} at table index {}",
            block, table_idx
        )));
    }
    Ok(block as usize)
}
