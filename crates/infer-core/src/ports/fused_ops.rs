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
    fn layer_norm<T: Dtype>(
        _scope: &Self::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let dim = weight.numel();
        if dim == 0
            || input.shape().len() != 2
            || input.shape()[1] != dim
            || bias.numel() != dim
            || output.shape() != input.shape()
            || !eps.is_finite()
            || !input.is_contiguous()
            || !output.is_contiguous()
            || !weight.is_contiguous()
            || !bias.is_contiguous()
            || eps <= 0.0
        {
            return Err(OpError::Shape(
                "layer_norm: invalid dimensions or epsilon".into(),
            ));
        }
        let x = input.to_host_vec()?;
        let w = weight.to_host_vec()?;
        let b = bias.to_host_vec()?;
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks_exact(dim) {
            let mean = row.iter().map(|v| T::read_f64(v) as f32).sum::<f32>() / dim as f32;
            let var = row
                .iter()
                .map(|v| (T::read_f64(v) as f32 - mean).powi(2))
                .sum::<f32>()
                / dim as f32;
            let inv = (var + eps).sqrt().recip();
            for i in 0..dim {
                out.push(T::write_f64(
                    ((T::read_f64(&row[i]) as f32 - mean) * inv * T::read_f64(&w[i]) as f32
                        + T::read_f64(&b[i]) as f32) as f64,
                ));
            }
        }
        output.upload_from_host(&out)
    }

    /// Exact (erf) and tanh GELU are different checkpoint operations.
    fn gelu_inplace<T: Dtype>(
        _scope: &Self::Scope,
        x: &mut Tensor<T, Self>,
        tanh: bool,
    ) -> OpResult<()> {
        if !x.is_contiguous() {
            return Err(OpError::NotContiguous(*x.shape()));
        }
        let mut values = x.to_host_vec()?;
        for v in &mut values {
            let a = T::read_f64(v) as f32;
            let c = if tanh {
                (0.7978846 * (a + 0.044715 * a * a * a)).tanh()
            } else {
                libm::erff(a * std::f32::consts::FRAC_1_SQRT_2)
            };
            *v = T::write_f64((0.5 * a * (1.0 + c)) as f64);
        }
        x.upload_from_host(&values)
    }

    /// Half-split rotation with caller-prepared per-token FP32 angles.
    /// Supports strided [tokens, heads * head_dim] Q/K views and partial RoPE.
    fn rope_with_angles<T: Dtype>(
        _scope: &Self::Scope,
        x: &mut Tensor<T, Self>,
        sin: &Tensor<f32, Self>,
        cos: &Tensor<f32, Self>,
        head_dim: usize,
    ) -> OpResult<()> {
        let half = sin.shape().as_slice().get(1).copied().unwrap_or(0);
        if x.shape().len() != 2
            || sin.shape().len() != 2
            || sin.shape() != cos.shape()
            || head_dim == 0
            || half == 0
            || half * 2 > head_dim
            || !x.shape()[1].is_multiple_of(head_dim)
            || x.shape()[0] != sin.shape()[0]
            || x.strides()[1] != 1
        {
            return Err(OpError::Shape(
                "rope_with_angles: invalid dimensions".into(),
            ));
        }
        let mut values = Vec::with_capacity(x.numel());
        for row in 0..x.shape()[0] {
            let shape = Shape::from_slice(&[1, x.shape()[1]]);
            let row_view = x.view_raw(
                shape,
                shape.contiguous_strides(),
                x.offset_elems() + row * x.strides()[0],
                true,
            );
            values.extend(row_view.to_host_vec()?);
        }
        let s = sin.to_host_vec()?;
        let c = cos.to_host_vec()?;
        for (row, values) in values.chunks_exact_mut(x.shape()[1]).enumerate() {
            for head in values.chunks_exact_mut(head_dim) {
                for j in 0..half {
                    let a = T::read_f64(&head[j]) as f32;
                    let b = T::read_f64(&head[j + half]) as f32;
                    head[j] = T::write_f64((a * c[row * half + j] - b * s[row * half + j]) as f64);
                    head[j + half] =
                        T::write_f64((a * s[row * half + j] + b * c[row * half + j]) as f64);
                }
            }
        }
        for (row, values) in values.chunks_exact(x.shape()[1]).enumerate() {
            x.narrow(0, row, 1)?.upload_from_host(values)?;
        }
        Ok(())
    }

    /// Qwen3.5 RMSNorm: FP32 normalization and (1 + weight), one final cast.
    /// Each contiguous group of weight.len() columns is an independent head.
    fn rmsnorm_zero_centered<T: Dtype>(
        _scope: &Self::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let dim = weight.numel();
        if dim == 0
            || input.shape() != output.shape()
            || input.shape().len() != 2
            || !input.shape()[1].is_multiple_of(dim)
            || !eps.is_finite()
            || eps < 0.0
        {
            return Err(OpError::Shape(
                "rmsnorm_zero_centered: invalid shapes or epsilon".into(),
            ));
        }
        let mut values = input.to_host_vec()?;
        let weights = weight.to_host_vec()?;
        for row in values.chunks_mut(dim) {
            let sum: f32 = row.iter().map(|v| (T::read_f64(v) as f32).powi(2)).sum();
            let inv = (sum / dim as f32 + eps).sqrt().recip();
            for (v, w) in row.iter_mut().zip(&weights) {
                *v = T::write_f64(
                    ((T::read_f64(v) as f32 * inv) * (1.0 + T::read_f64(w) as f32)) as f64,
                );
            }
        }
        output.upload_from_host(&values)
    }

    /// `output *= sigmoid(gate)`, with sigmoid rounded to the activation dtype.
    /// Both tensors must have identical shapes and contiguous storage.
    fn sigmoid_mul<T: Dtype>(
        _scope: &Self::Scope,
        output: &mut Tensor<T, Self>,
        gate: &Tensor<T, Self>,
    ) -> OpResult<()> {
        if output.shape() != gate.shape() || !output.is_contiguous() || !gate.is_contiguous() {
            return Err(OpError::Shape(
                "sigmoid_mul: expected matching contiguous tensors".into(),
            ));
        }
        let mut values = output.to_host_vec()?;
        let gates = gate.to_host_vec()?;
        for (value, gate) in values.iter_mut().zip(gates.iter()) {
            let g = T::read_f64(gate) as f32;
            let sigmoid = T::write_f64((1.0 / (1.0 + (-g).exp())) as f64);
            *value =
                T::write_f64(((T::read_f64(value) as f32) * (T::read_f64(&sigmoid) as f32)) as f64);
        }
        output.upload_from_host(&values)
    }

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

    /// Causal depthwise Conv1d followed by SiLU. The operator owns no state:
    /// all persistent sequence state is supplied by the caller through
    /// `conv_state` and updated in place.
    ///
    /// Layout:
    /// - `input` / `output`: flattened ragged tape `[num_tokens, channels]`
    /// - `weight`: depthwise weights `[channels, 1, kernel_size]`
    /// - `conv_state`: caller-owned `[num_slots, channels, kernel_size]`
    /// - `state_slots`: one distinct slot id per sequence, `[batch]`
    /// - `cu_seqlens`: ragged row offsets, `[batch + 1]`
    ///
    /// A state row stores the latest `kernel_size` *pre-convolution* input
    /// values. For each output only the latest `kernel_size - 1` cached values
    /// plus the current input participate; retaining one full kernel matches
    /// Qwen3.5 / causal-conv1d cache layout and makes single-token updates a
    /// shift-and-append operation. Distinct sequences must not name the same
    /// mutable slot in one call.
    fn causal_conv1d_silu<T: Dtype>(
        _scope: &Self::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        conv_state: &mut Tensor<T, Self>,
        state_slots: &Tensor<i32, Self>,
        cu_seqlens: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        causal_conv1d_silu_reference(input, weight, conv_state, state_slots, cu_seqlens, output)
    }

    /// Sequential Gated DeltaNet recurrence for a flattened ragged batch.
    /// The operator owns no recurrent state: `recurrent_state` is supplied by
    /// the caller, read as the initial state, and updated in place.
    ///
    /// Layout:
    /// - `query` / `key`: `[num_tokens, num_key_heads * key_head_dim]`
    /// - `value` / `output`: `[num_tokens, num_value_heads * value_head_dim]`
    /// - `a` / `b`: raw projections `[num_tokens, num_value_heads]`
    /// - `a_log`: fp32 `[num_value_heads]`; `dt_bias`: `[num_value_heads]`
    /// - `recurrent_state`: caller-owned fp32
    ///   `[num_slots, num_value_heads, key_head_dim, value_head_dim]`
    /// - `state_slots`: one distinct slot id per sequence, `[batch]`
    /// - `cu_seqlens`: ragged row offsets, `[batch + 1]`
    ///
    /// The state shape determines the head dimensions. Value heads are mapped
    /// to key/query heads in repeat-interleave order, so
    /// `num_value_heads` must be divisible by `num_key_heads`.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_rule<T: Dtype>(
        _scope: &Self::Scope,
        query: &Tensor<T, Self>,
        key: &Tensor<T, Self>,
        value: &Tensor<T, Self>,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        a_log: &Tensor<f32, Self>,
        dt_bias: &Tensor<T, Self>,
        recurrent_state: &mut Tensor<f32, Self>,
        state_slots: &Tensor<i32, Self>,
        cu_seqlens: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        gated_delta_rule_reference(
            query,
            key,
            value,
            a,
            b,
            a_log,
            dt_bias,
            recurrent_state,
            state_slots,
            cu_seqlens,
            output,
        )
    }

    /// Per-head RMSNorm followed by a SiLU output gate.
    ///
    /// Layout:
    /// - `input` / `gate` / `output`: the same `[..., head_dim]` shape
    /// - `weight`: fp32 `[head_dim]`
    ///
    /// The operator owns no state and allocates no persistent storage. For
    /// low-precision activations it follows Qwen3.5's numerical order: RMS
    /// statistics are computed in fp32, the normalized activation is rounded
    /// back to `T`, then the fp32 weight and fp32 SiLU gate are applied before
    /// the final result is rounded to `T`.
    fn gated_rmsnorm<T: Dtype>(
        _scope: &Self::Scope,
        input: &Tensor<T, Self>,
        gate: &Tensor<T, Self>,
        weight: &Tensor<f32, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        gated_rmsnorm_reference(input, gate, weight, output, eps)
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
        rotary_dim: usize,
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
            rotary_dim,
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
    /// length `rows`. Equal maxima choose the lowest column index.
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
                .max_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b.0.cmp(&a.0))
                })
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

fn causal_conv1d_silu_reference<T, D>(
    input: &Tensor<T, D>,
    weight: &Tensor<T, D>,
    conv_state: &mut Tensor<T, D>,
    state_slots: &Tensor<i32, D>,
    cu_seqlens: &Tensor<i32, D>,
    output: &mut Tensor<T, D>,
) -> OpResult<()>
where
    T: Dtype,
    D: MathOps,
{
    let input_shape = input.shape().as_slice();
    if input_shape.len() != 2 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: input must be [num_tokens, channels], got {:?}",
            input_shape
        )));
    }
    let num_tokens = input_shape[0];
    let channels = input_shape[1];
    if channels == 0 {
        return Err(OpError::Shape(
            "causal_conv1d_silu: channels must be non-zero".into(),
        ));
    }
    if output.shape().as_slice() != input_shape {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: output shape {:?} != input shape {:?}",
            output.shape().as_slice(),
            input_shape
        )));
    }

    let weight_shape = weight.shape().as_slice();
    if weight_shape.len() != 3 || weight_shape[0] != channels || weight_shape[1] != 1 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: weight must be [{channels}, 1, kernel_size], got {:?}",
            weight_shape
        )));
    }
    let kernel_size = weight_shape[2];
    if kernel_size == 0 {
        return Err(OpError::Shape(
            "causal_conv1d_silu: kernel_size must be non-zero".into(),
        ));
    }

    let state_shape = conv_state.shape().as_slice();
    if state_shape.len() != 3
        || state_shape[0] == 0
        || state_shape[1] != channels
        || state_shape[2] != kernel_size
    {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: conv_state must be [num_slots, {channels}, {kernel_size}], got {:?}",
            state_shape
        )));
    }
    let num_slots = state_shape[0];

    let slots_shape = state_slots.shape().as_slice();
    let cu_shape = cu_seqlens.shape().as_slice();
    if slots_shape.len() != 1 {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: state_slots must be rank 1, got {:?}",
            slots_shape
        )));
    }
    let batch = slots_shape[0];
    if batch == 0 {
        return Err(OpError::Shape(
            "causal_conv1d_silu: batch must be non-zero".into(),
        ));
    }
    let cu_len = batch
        .checked_add(1)
        .ok_or_else(|| OpError::Shape("causal_conv1d_silu: batch size overflows".into()))?;
    if cu_shape != [cu_len] {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: cu_seqlens shape {:?} != [{}]",
            cu_shape, cu_len
        )));
    }

    for (shape, contiguous) in [
        (*input.shape(), input.is_contiguous()),
        (*weight.shape(), weight.is_contiguous()),
        (*conv_state.shape(), conv_state.is_contiguous()),
        (*state_slots.shape(), state_slots.is_contiguous()),
        (*cu_seqlens.shape(), cu_seqlens.is_contiguous()),
        (*output.shape(), output.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::NotContiguous(shape));
        }
    }

    let input_host = input.to_host_vec()?;
    let weight_host = weight.to_host_vec()?;
    let mut state_host = conv_state.to_host_vec()?;
    let slots_host = state_slots.to_host_vec()?;
    let cu_host = cu_seqlens.to_host_vec()?;

    if cu_host.first().copied() != Some(0) {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: cu_seqlens must start at 0, got {:?}",
            cu_host.first()
        )));
    }
    let num_tokens_i32 = i32::try_from(num_tokens)
        .map_err(|_| OpError::Shape("causal_conv1d_silu: num_tokens exceeds i32".into()))?;
    if cu_host.last().copied() != Some(num_tokens_i32) {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: cu_seqlens must end at num_tokens {}, got {:?}",
            num_tokens,
            cu_host.last()
        )));
    }
    if cu_host
        .windows(2)
        .any(|pair| pair[0] < 0 || pair[0] > pair[1])
    {
        return Err(OpError::Shape(format!(
            "causal_conv1d_silu: cu_seqlens must be monotonic and non-negative, got {:?}",
            cu_host
        )));
    }

    let mut seen_slots = std::collections::HashSet::with_capacity(batch);
    for (seq, &slot) in slots_host.iter().enumerate() {
        if slot < 0 || slot as usize >= num_slots {
            return Err(OpError::Shape(format!(
                "causal_conv1d_silu: state slot {} for sequence {} outside [0, {})",
                slot, seq, num_slots
            )));
        }
        if !seen_slots.insert(slot) {
            return Err(OpError::Shape(format!(
                "causal_conv1d_silu: duplicate mutable state slot {}",
                slot
            )));
        }
    }

    let mut output_host = vec![T::write_f64(0.0); num_tokens * channels];
    for seq in 0..batch {
        let start = cu_host[seq] as usize;
        let end = cu_host[seq + 1] as usize;
        let seq_len = end - start;
        let slot = slots_host[seq] as usize;

        for channel in 0..channels {
            let state_base = (slot * channels + channel) * kernel_size;
            let weight_base = channel * kernel_size;

            for token in 0..seq_len {
                let mut acc = 0.0f64;
                for tap in 0..kernel_size {
                    let relative = token as isize + tap as isize + 1 - kernel_size as isize;
                    let value = if relative >= 0 {
                        let row = start + relative as usize;
                        T::read_f64(&input_host[row * channels + channel])
                    } else {
                        let state_col = (kernel_size as isize + relative) as usize;
                        T::read_f64(&state_host[state_base + state_col])
                    };
                    acc += value * T::read_f64(&weight_host[weight_base + tap]);
                }
                let activated = acc / (1.0 + (-acc).exp());
                output_host[(start + token) * channels + channel] = T::write_f64(activated);
            }

            // Keep the last `kernel_size` raw inputs from
            // `[old_state, current_sequence]`. Iterate low-to-high: when the
            // chunk is shorter than the kernel, every old-state read is from a
            // strictly higher index than its write, so no temporary is needed.
            for state_col in 0..kernel_size {
                let concat_col = seq_len + state_col;
                state_host[state_base + state_col] = if concat_col < kernel_size {
                    state_host[state_base + concat_col]
                } else {
                    let input_row = start + concat_col - kernel_size;
                    input_host[input_row * channels + channel]
                };
            }
        }
    }

    output.upload_from_host(&output_host)?;
    conv_state.upload_from_host(&state_host)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_reference<T, D>(
    query: &Tensor<T, D>,
    key: &Tensor<T, D>,
    value: &Tensor<T, D>,
    a: &Tensor<T, D>,
    b: &Tensor<T, D>,
    a_log: &Tensor<f32, D>,
    dt_bias: &Tensor<T, D>,
    recurrent_state: &mut Tensor<f32, D>,
    state_slots: &Tensor<i32, D>,
    cu_seqlens: &Tensor<i32, D>,
    output: &mut Tensor<T, D>,
) -> OpResult<()>
where
    T: Dtype,
    D: MathOps,
{
    let query_shape = query.shape().as_slice();
    if query_shape.len() != 2 || query_shape[0] == 0 || query_shape[1] == 0 {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: query must be non-empty [num_tokens, key_width], got {:?}",
            query_shape
        )));
    }
    let (num_tokens, key_width) = (query_shape[0], query_shape[1]);
    if key.shape().as_slice() != query_shape {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: key shape {:?} != query shape {:?}",
            key.shape().as_slice(),
            query_shape
        )));
    }

    let value_shape = value.shape().as_slice();
    if value_shape.len() != 2 || value_shape[0] != num_tokens || value_shape[1] == 0 {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: value must be [{num_tokens}, value_width], got {:?}",
            value_shape
        )));
    }
    if output.shape().as_slice() != value_shape {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: output shape {:?} != value shape {:?}",
            output.shape().as_slice(),
            value_shape
        )));
    }

    let state_shape = recurrent_state.shape().as_slice();
    if state_shape.len() != 4
        || state_shape[0] == 0
        || state_shape[1] == 0
        || state_shape[2] == 0
        || state_shape[3] == 0
    {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: recurrent_state must be non-empty [num_slots, num_value_heads, key_head_dim, value_head_dim], got {:?}",
            state_shape
        )));
    }
    let (num_slots, num_value_heads, key_head_dim, value_head_dim) = (
        state_shape[0],
        state_shape[1],
        state_shape[2],
        state_shape[3],
    );
    let value_width = num_value_heads
        .checked_mul(value_head_dim)
        .ok_or_else(|| OpError::Shape("gated_delta_rule: value width overflows".into()))?;
    if value_shape[1] != value_width {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: value width {} != num_value_heads {} * value_head_dim {}",
            value_shape[1], num_value_heads, value_head_dim
        )));
    }
    if !key_width.is_multiple_of(key_head_dim) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: key width {key_width} is not divisible by key_head_dim {key_head_dim}"
        )));
    }
    let num_key_heads = key_width / key_head_dim;
    if num_key_heads == 0 || !num_value_heads.is_multiple_of(num_key_heads) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: num_value_heads {num_value_heads} must be divisible by num_key_heads {num_key_heads}"
        )));
    }
    let value_heads_per_key = num_value_heads / num_key_heads;

    let head_projection_shape = [num_tokens, num_value_heads];
    for (name, shape) in [("a", a.shape()), ("b", b.shape())] {
        if shape.as_slice() != head_projection_shape {
            return Err(OpError::Shape(format!(
                "gated_delta_rule: {name} shape {:?} != {:?}",
                shape.as_slice(),
                head_projection_shape
            )));
        }
    }
    if a_log.shape().as_slice() != [num_value_heads]
        || dt_bias.shape().as_slice() != [num_value_heads]
    {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: a_log/dt_bias must both be [{num_value_heads}], got {:?}/{:?}",
            a_log.shape().as_slice(),
            dt_bias.shape().as_slice()
        )));
    }

    let slots_shape = state_slots.shape().as_slice();
    if slots_shape.len() != 1 || slots_shape[0] == 0 {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: state_slots must be non-empty rank 1, got {:?}",
            slots_shape
        )));
    }
    let batch = slots_shape[0];
    let cu_len = batch
        .checked_add(1)
        .ok_or_else(|| OpError::Shape("gated_delta_rule: batch size overflows".into()))?;
    if cu_seqlens.shape().as_slice() != [cu_len] {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: cu_seqlens shape {:?} != [{cu_len}]",
            cu_seqlens.shape().as_slice()
        )));
    }

    for (shape, contiguous) in [
        (*query.shape(), query.is_contiguous()),
        (*key.shape(), key.is_contiguous()),
        (*value.shape(), value.is_contiguous()),
        (*a.shape(), a.is_contiguous()),
        (*b.shape(), b.is_contiguous()),
        (*a_log.shape(), a_log.is_contiguous()),
        (*dt_bias.shape(), dt_bias.is_contiguous()),
        (*recurrent_state.shape(), recurrent_state.is_contiguous()),
        (*state_slots.shape(), state_slots.is_contiguous()),
        (*cu_seqlens.shape(), cu_seqlens.is_contiguous()),
        (*output.shape(), output.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::NotContiguous(shape));
        }
    }

    let query_host = query.to_host_vec()?;
    let key_host = key.to_host_vec()?;
    let value_host = value.to_host_vec()?;
    let a_host = a.to_host_vec()?;
    let b_host = b.to_host_vec()?;
    let a_log_host = a_log.to_host_vec()?;
    let dt_bias_host = dt_bias.to_host_vec()?;
    let mut state_host = recurrent_state.to_host_vec()?;
    let slots_host = state_slots.to_host_vec()?;
    let cu_host = cu_seqlens.to_host_vec()?;

    if cu_host.first().copied() != Some(0) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: cu_seqlens must start at 0, got {:?}",
            cu_host.first()
        )));
    }
    let num_tokens_i32 = i32::try_from(num_tokens)
        .map_err(|_| OpError::Shape("gated_delta_rule: num_tokens exceeds i32".into()))?;
    if cu_host.last().copied() != Some(num_tokens_i32) {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: cu_seqlens must end at num_tokens {num_tokens}, got {:?}",
            cu_host.last()
        )));
    }
    if cu_host
        .windows(2)
        .any(|pair| pair[0] < 0 || pair[0] > pair[1])
    {
        return Err(OpError::Shape(format!(
            "gated_delta_rule: cu_seqlens must be monotonic and non-negative, got {:?}",
            cu_host
        )));
    }
    let mut seen_slots = std::collections::HashSet::with_capacity(batch);
    for (seq, &slot) in slots_host.iter().enumerate() {
        if slot < 0 || slot as usize >= num_slots {
            return Err(OpError::Shape(format!(
                "gated_delta_rule: state slot {slot} for sequence {seq} outside [0, {num_slots})"
            )));
        }
        if !seen_slots.insert(slot) {
            return Err(OpError::Shape(format!(
                "gated_delta_rule: duplicate mutable state slot {slot}"
            )));
        }
    }

    #[inline]
    fn sigmoid(x: f32) -> f32 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }
    #[inline]
    fn softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { x.exp().ln_1p() }
    }

    const QK_L2_EPS: f32 = 1e-6;
    let mut output_host = vec![T::write_f64(0.0); num_tokens * value_width];
    for seq in 0..batch {
        let start = cu_host[seq] as usize;
        let end = cu_host[seq + 1] as usize;
        let slot = slots_host[seq] as usize;
        for value_head in 0..num_value_heads {
            let key_head = value_head / value_heads_per_key;
            let state_base =
                ((slot * num_value_heads + value_head) * key_head_dim) * value_head_dim;
            let neg_a = -a_log_host[value_head].exp();
            let dt = T::read_f64(&dt_bias_host[value_head]) as f32;

            for token in start..end {
                let query_base = token * key_width + key_head * key_head_dim;
                let key_base = query_base;
                let mut query_norm_sq = 0.0f32;
                let mut key_norm_sq = 0.0f32;
                for dim in 0..key_head_dim {
                    let q = T::read_f64(&query_host[query_base + dim]) as f32;
                    let k = T::read_f64(&key_host[key_base + dim]) as f32;
                    query_norm_sq = q.mul_add(q, query_norm_sq);
                    key_norm_sq = k.mul_add(k, key_norm_sq);
                }
                let query_scale =
                    1.0 / ((query_norm_sq + QK_L2_EPS).sqrt() * (key_head_dim as f32).sqrt());
                let key_scale = 1.0 / (key_norm_sq + QK_L2_EPS).sqrt();

                let head_offset = token * num_value_heads + value_head;
                let raw_b = T::read_f64(&b_host[head_offset]) as f32;
                // HF computes sigmoid in the activation dtype before the
                // recurrence promotes beta to fp32. Preserve that rounding.
                let beta_t = T::write_f64(sigmoid(raw_b) as f64);
                let beta = T::read_f64(&beta_t) as f32;
                let raw_a = T::read_f64(&a_host[head_offset]) as f32;
                let log_decay = neg_a * softplus(raw_a + dt);
                let decay = log_decay.exp();
                let value_base = token * value_width + value_head * value_head_dim;

                for value_dim in 0..value_head_dim {
                    let mut memory = 0.0f32;
                    for key_dim in 0..key_head_dim {
                        let state_index = state_base + key_dim * value_head_dim + value_dim;
                        let decayed = state_host[state_index] * decay;
                        state_host[state_index] = decayed;
                        let k = T::read_f64(&key_host[key_base + key_dim]) as f32 * key_scale;
                        memory = decayed.mul_add(k, memory);
                    }
                    let v = T::read_f64(&value_host[value_base + value_dim]) as f32;
                    let delta = (v - memory) * beta;
                    let mut out = 0.0f32;
                    for key_dim in 0..key_head_dim {
                        let state_index = state_base + key_dim * value_head_dim + value_dim;
                        let k = T::read_f64(&key_host[key_base + key_dim]) as f32 * key_scale;
                        let updated = k.mul_add(delta, state_host[state_index]);
                        state_host[state_index] = updated;
                        let q = T::read_f64(&query_host[query_base + key_dim]) as f32 * query_scale;
                        out = updated.mul_add(q, out);
                    }
                    output_host[value_base + value_dim] = T::write_f64(out as f64);
                }
            }
        }
    }

    output.upload_from_host(&output_host)?;
    recurrent_state.upload_from_host(&state_host)
}

fn gated_rmsnorm_reference<T, D>(
    input: &Tensor<T, D>,
    gate: &Tensor<T, D>,
    weight: &Tensor<f32, D>,
    output: &mut Tensor<T, D>,
    eps: f32,
) -> OpResult<()>
where
    T: Dtype,
    D: MathOps,
{
    let shape = input.shape().as_slice();
    if shape.is_empty() || input.numel() == 0 {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: input must be non-empty [..., head_dim], got {:?}",
            shape
        )));
    }
    if gate.shape().as_slice() != shape || output.shape().as_slice() != shape {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: input/gate/output shapes must match, got {:?}/{:?}/{:?}",
            shape,
            gate.shape().as_slice(),
            output.shape().as_slice()
        )));
    }
    let head_dim = *shape.last().expect("non-empty shape checked above");
    if head_dim == 0 || weight.shape().as_slice() != [head_dim] {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: weight must be fp32 [{head_dim}], got {:?}",
            weight.shape().as_slice()
        )));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(OpError::Shape(format!(
            "gated_rmsnorm: eps must be finite and non-negative, got {eps}"
        )));
    }
    for (tensor_shape, contiguous) in [
        (*input.shape(), input.is_contiguous()),
        (*gate.shape(), gate.is_contiguous()),
        (*weight.shape(), weight.is_contiguous()),
        (*output.shape(), output.is_contiguous()),
    ] {
        if !contiguous {
            return Err(OpError::NotContiguous(tensor_shape));
        }
    }

    let input_host = input.to_host_vec()?;
    let gate_host = gate.to_host_vec()?;
    let weight_host = weight.to_host_vec()?;
    let rows = input.numel() / head_dim;
    let mut output_host = vec![T::write_f64(0.0); input.numel()];
    for row in 0..rows {
        let base = row * head_dim;
        let mut square_sum = 0.0f32;
        for col in 0..head_dim {
            let value = T::read_f64(&input_host[base + col]) as f32;
            square_sum = value.mul_add(value, square_sum);
        }
        let inv_rms = (square_sum / head_dim as f32 + eps).sqrt().recip();
        for col in 0..head_dim {
            let value = T::read_f64(&input_host[base + col]) as f32;
            // Qwen3.5 casts normalized values back to the activation dtype
            // before multiplying by its fp32 norm weight.
            let normalized_t = T::write_f64((value * inv_rms) as f64);
            let normalized = T::read_f64(&normalized_t) as f32;
            let gate_value = T::read_f64(&gate_host[base + col]) as f32;
            let silu_gate = gate_value / (1.0 + (-gate_value).exp());
            let result = normalized * weight_host[col] * silu_gate;
            output_host[base + col] = T::write_f64(result as f64);
        }
    }
    output.upload_from_host(&output_host)
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
