use super::device::MemoryPort;
use super::error::OpResult;
use crate::domain::batch::BatchPlan;
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape, Strides};

/// Core operator port — primitives shared by **all** model families
/// (LLM decoders and diffusion pipelines alike).
///
/// Infrastructure provides `impl CoreOps for Cpu` / `impl CoreOps for Cuda`.
/// A backend may explicitly reject an op with [`OpError::Unsupported`] when
/// that device intentionally does not implement the kernel. That is still a
/// valid implementation as long as the failure is clear and deterministic.
///
/// Generic bounds use `Dtype` (not `Float`) so quantized types (i8, i4)
/// can participate in mixed-precision operations (e.g. int8 weight × bf16 activation).
///
/// `CoreOps: MemoryPort` so any backend can allocate / upload / download
/// (required for `Tensor::zeros`, `Tensor::from_host_slice`, `Tensor::to_host_vec`).
///
/// Family-specific ops live in [`LlmOps`] (decoder / paged-KV) and
/// [`DiffusionOps`] (conv / VAE / DiT). These capability traits describe the
/// intended surface area, not a guarantee that every backend accelerates every
/// method. Production paths should choose a backend whose required ops are
/// implemented; reference or partial backends may return `Unsupported`.
pub trait CoreOps: MemoryPort {
    // ─── Allocation ──────────────────────────────────────────────────
    /// Allocate a contiguous, zeroed tensor on this device. Default
    /// implementation routes through `MemoryPort` — overridable for
    /// backends that have a smarter pool.
    fn alloc_tensor<T: Dtype>(shape: Shape, device: &Self) -> OpResult<Tensor<T, Self>> {
        Tensor::<T, Self>::zeros(shape, device)
    }

    // ─── Element-wise ────────────────────────────────────────────────
    fn add<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;
    fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()>;

    // ─── Element-wise multiply ───────────────────────────────────────
    fn ewise_mul<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    // ─── Linear algebra ──────────────────────────────────────────────
    /// Standard matmul: input [M,K] × weight [N,K]^T → output [M,N]
    /// Same dtype for input, weight, output.
    fn matmul<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    /// Quantized matmul: activation A × weight W → output O
    /// A: activation dtype, W: weight dtype (e.g. i8, i4-packed-as-i32), O: output dtype
    /// Supports mixed-precision: bf16 activation × int8 weight → bf16 output
    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        input: &Tensor<A, Self>,
        weight: &Tensor<W, Self>,
        output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>,
        zeros: Option<&Tensor<W, Self>>,
        group_size: usize,
    ) -> OpResult<()>;

    // ─── Activations ─────────────────────────────────────────────────
    fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()>;

    // ─── Softmax ─────────────────────────────────────────────────────
    fn softmax<T: Dtype>(input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;

    // ─── Embedding ───────────────────────────────────────────────────
    fn embedding<T: Dtype>(
        table: &Tensor<T, Self>,
        indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    // ─── Scalar ──────────────────────────────────────────────────────
    fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()>;

    /// `x += scalar` (in place).
    fn scalar_add_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()>;

    /// CUDA-Graph-friendly scalar multiply: `x *= *d_scalar` where
    /// `d_scalar` is a single-element f32 device tensor.
    fn scalar_mul_inplace_from_dev<T: Dtype>(
        x: &mut Tensor<T, Self>,
        d_scalar: &Tensor<f32, Self>,
    ) -> OpResult<()>;

    // ─── Broadcast ───────────────────────────────────────────────────
    /// In-place broadcast multiply: x[i,j] *= scale[j].
    /// x: [rows, dim], scale: [dim] or [1, dim].
    fn broadcast_mul_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()>;

    /// In-place broadcast add: `x[i, j] += bias[j]` over a `[*, D]` tensor.
    fn broadcast_add_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        bias: &Tensor<T, Self>,
    ) -> OpResult<()>;

    // ─── Shape / layout ──────────────────────────────────────────────
    /// Generic column slice: `dst[r, j] = src[r, col_offset + j]` for a
    /// 2D src of shape `[rows, total_cols]` and dst `[rows, dst_cols]`.
    fn split_cols<T: Dtype>(
        src: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        dst_cols: usize,
    ) -> OpResult<()>;

    /// Concat two `[*, D]` tensors along dim 0 into a pre-allocated dst.
    fn concat_seq<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    /// Cast between dtypes within the same device.
    fn cast_dtype<S: Dtype, D2: Dtype>(
        src: &Tensor<S, Self>,
        dst: &mut Tensor<D2, Self>,
    ) -> OpResult<()>;
}

/// LLM decoder operator port — ops used by autoregressive decoder models
/// (Llama3, Qwen3, Qwen3_5, …). These models share **all** of these ops;
/// per-model differences (layer count, dims, weight layout) live in the
/// model code, not here.
///
/// A backend that only serves LLMs implements `CoreOps + LlmOps` and can
/// skip [`DiffusionOps`] entirely.
pub trait LlmOps: CoreOps {
    // ─── Normalization ───────────────────────────────────────────────
    fn rmsnorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;
    fn rmsnorm_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    /// Fused: residual += input; output = rmsnorm(residual, weight, eps)
    /// Saves one global memory pass vs separate add + rmsnorm.
    fn fused_add_rmsnorm<T: Dtype>(
        output: &mut Tensor<T, Self>,
        residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    // ─── Activations ─────────────────────────────────────────────────
    fn swiglu_inplace<T: Dtype>(x: &mut Tensor<T, Self>, gate: &Tensor<T, Self>) -> OpResult<()>;

    /// Packed SwiGLU: `gate_up [rows, 2*inter]` → `out [rows, inter]`,
    /// `out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]`.
    ///
    /// Replaces 2 × split + swiglu_inplace with one fused kernel launch.
    /// Only required to be implemented for backends that need it; CPU
    /// backend can fall back to the split + swiglu path.
    fn swiglu_packed<T: Dtype>(
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()>;

    // ─── RoPE ────────────────────────────────────────────────────────
    /// Apply Rotary Position Embedding in-place.
    fn rope_inplace<T: Dtype>(
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>,
        cos: &Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
    ) -> OpResult<()>;

    // ─── Attention (paged) ───────────────────────────────────────────
    /// Batched / ragged attention over a paged KV pool.
    ///
    /// - `q`               : `[num_tokens, q_dim]` (q_dim = head_num * head_dim)
    /// - `k_pool / v_pool` : layer-local paged tensors `[num_blocks, block_size, kv_dim]`
    /// - `output`          : `[num_tokens, q_dim]`
    /// - `plan`            : carries `block_tables`, `kv_lens`, `cu_q_lens`,
    ///                       `block2req`, `block2tile`, `total_q_tiles`, `kind`,
    ///                       `block_size`, `max_blocks_per_seq`
    ///
    /// `BatchKind::DecodeOnly` dispatches to a Flash-Decoding kernel
    /// (q_len=1 per seq, gather K/V from paged blocks).
    /// `BatchKind::Ragged` dispatches to a tile-scheduled paged prefill
    /// kernel (variable q_len + kv_len per seq, GQA-aware).
    fn attention_paged<T: Dtype>(
        q: &Tensor<T, Self>,
        k_pool: &Tensor<T, Self>,
        v_pool: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        plan: &BatchPlan<Self>,
        workspace: &mut Tensor<f32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()>;

    // ─── KV Cache ────────────────────────────────────────────────────
    /// Split a fused [num_tokens, qkv_dim] tensor into Q, K, V.
    fn split_qkv<T: Dtype>(
        qkv: &Tensor<T, Self>,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        v: &mut Tensor<T, Self>,
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()>;

    /// Scatter K/V rows into a layer's paged KV pool.
    ///
    /// - `k_src/v_src`    : `[num_tokens, kv_dim]`
    /// - `k_pool/v_pool`  : `[num_blocks, block_size, kv_dim]`
    /// - `block_tables`   : `[batch, max_blocks_per_seq]` device i32, physical block ids
    /// - `seq_positions`  : `[batch]` — first cache row per seq this step writes
    /// - `cu_q_lens`      : `[batch + 1]` — prefix sum of per-seq q_len
    /// - `seq_lens_step`  : `[batch]` — tokens this step writes per seq
    fn scatter_kv_paged<T: Dtype>(
        k_src: &Tensor<T, Self>,
        v_src: &Tensor<T, Self>,
        k_pool: &mut Tensor<T, Self>,
        v_pool: &mut Tensor<T, Self>,
        block_tables: &Tensor<i32, Self>,
        seq_positions: &Tensor<i32, Self>,
        cu_q_lens: &Tensor<i32, Self>,
        seq_lens_step: &Tensor<i32, Self>,
        max_blocks_per_seq: usize,
        block_size: usize,
        kv_dim: usize,
    ) -> OpResult<()>;

    /// Fused Qwen3 Q/K RMSNorm + RoPE + paged K/V scatter.
    fn qkv_norm_rope_scatter<T: Dtype>(
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
        k_pool: &mut Tensor<T, Self>,
        v_pool: &mut Tensor<T, Self>,
        block_tables: &Tensor<i32, Self>,
        seq_positions: &Tensor<i32, Self>,
        cu_q_lens: &Tensor<i32, Self>,
        seq_lens_step: &Tensor<i32, Self>,
        max_blocks_per_seq: usize,
        block_size: usize,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        if let Some(q_weight) = q_weight {
            let q_cols = q.strides().as_slice()[0];
            let q_offset = q.offset_elems();
            let mut q3 = q.view_raw(
                Shape::from_slice(&[positions.numel(), head_num, head_dim]),
                Strides::from_slice(&[q_cols, head_dim, 1]),
                q_offset,
                false,
            );
            Self::rmsnorm_inplace(&mut q3, q_weight, q_eps)?;
        }
        if let Some(k_weight) = k_weight {
            let k_cols = k.strides().as_slice()[0];
            let k_offset = k.offset_elems();
            let mut k3 = k.view_raw(
                Shape::from_slice(&[positions.numel(), kv_head_num, head_dim]),
                Strides::from_slice(&[k_cols, head_dim, 1]),
                k_offset,
                false,
            );
            Self::rmsnorm_inplace(&mut k3, k_weight, k_eps)?;
        }
        Self::rope_inplace(q, k, sin, cos, positions, head_num, kv_head_num, head_dim)?;
        Self::scatter_kv_paged(
            k,
            v,
            k_pool,
            v_pool,
            block_tables,
            seq_positions,
            cu_q_lens,
            seq_lens_step,
            max_blocks_per_seq,
            block_size,
            kv_dim,
        )
    }

    // ─── Sampling ────────────────────────────────────────────────────
    /// Argmax over each sequence's last logits row.
    /// `logits` : `[num_tokens, vocab_size]`
    /// `cu_q_lens` : `[batch+1]` — used to find each seq's last row
    /// `out_dev` : `[batch]` pre-allocated device buffer for kernel output
    /// `workspace` : `[batch, 256]` per-seq scratch for the argmax kernel
    /// `rows` : `[batch]` pre-allocated device buffer for selected row indices
    /// Returns `Vec<i32>` of length `batch`.
    fn argmax_batched<T: Dtype>(
        logits: &Tensor<T, Self>,
        cu_q_lens: &Tensor<i32, Self>,
        batch: usize,
        out_dev: &mut Tensor<i32, Self>,
        workspace: &Tensor<f32, Self>,
        rows: &mut Tensor<i32, Self>,
    ) -> OpResult<Vec<i32>>;
}

/// Diffusion operator port — Conv / Norm / Spatial / DiT ops used by
/// image-diffusion pipelines (Z_Image VAE + DiT + text encoder). No LLM
/// decoder uses these.
///
/// A backend that only serves LLMs may skip implementing this trait; the
/// type system then prevents constructing a diffusion model on that backend.
pub trait DiffusionOps: CoreOps {
    // ─── Conv2D (VAE) ────────────────────────────────────────────────
    /// 2D convolution: input [N, Cin, H, W] × weight [Cout, Cin, Kh, Kw] → output [N, Cout, Hout, Wout]
    fn conv2d<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: Option<&Tensor<T, Self>>,
        output: &mut Tensor<T, Self>,
        stride: usize,
        padding: usize,
    ) -> OpResult<()>;

    // ─── GroupNorm (VAE) ─────────────────────────────────────────────
    /// Group normalization: input [N, C, H, W], num_groups divides C.
    fn groupnorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()>;

    /// Fused GroupNorm + SiLU activation.
    fn groupnorm_silu<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()>;

    // ─── LayerNorm (DiT) ─────────────────────────────────────────────
    /// Layer normalization (mean + variance, unlike RMSNorm which is RMS only).
    fn layernorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    // ─── Upsample (VAE) ──────────────────────────────────────────────
    /// Nearest-neighbor 2× upsample: [N, C, H, W] → [N, C, 2H, 2W].
    fn upsample_nearest_2x<T: Dtype>(
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    // ─── SDPA for DiT (no KV cache, fixed seq len) ───────────────────
    /// Scaled dot-product attention for DiT blocks (self-attention, no cache).
    /// q/k/v: [seq_len, heads * head_dim], output: [seq_len, heads * head_dim]
    fn sdpa<T: Dtype>(
        q: &Tensor<T, Self>,
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()>;

    /// SDPA with an additive attention mask broadcast across all heads.
    /// `mask` is `[seq, seq]` (T-dtype). Entries should be `0.0` for
    /// "attend" and `-inf` (or large negative) for "do not attend". The mask
    /// is added to scaled scores **before** softmax.
    ///
    /// Used by Qwen3 text encoder (causal + padding) and any other attention
    /// that needs masking; DiT self-attention typically passes `None` and
    /// should call `sdpa` instead.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_masked<T: Dtype>(
        q: &Tensor<T, Self>,
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        mask: &Tensor<T, Self>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()>;

    /// Interleaved RoPE for DiT — applies per-head rotation on
    /// `[seq, n_heads, head_dim]` from F32 cos/sin caches `[seq, head_dim/2]`.
    fn apply_rope_interleaved<T: Dtype>(
        x: &mut Tensor<T, Self>,
        cos: &Tensor<f32, Self>,
        sin: &Tensor<f32, Self>,
        head_dim: usize,
    ) -> OpResult<()>;

    /// `dst[..n] = src; dst[n..target] = pad_token`.
    fn pad_with_token<T: Dtype>(
        src: &Tensor<T, Self>,
        pad_token: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()>;

    /// `dst[..n] = src; dst[n..target] = src[n-1]`.
    fn pad_last_row<T: Dtype>(src: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;

    /// `dst[keep_prefix..] = pad_token`.
    fn overwrite_pad_tokens_inplace<T: Dtype>(
        dst: &mut Tensor<T, Self>,
        pad_token: &Tensor<T, Self>,
        keep_prefix: usize,
    ) -> OpResult<()>;

    /// In-place SiLU activation (`x = x * sigmoid(x)`).
    fn silu_inplace_diff<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()>;

    /// In-place tanh activation.
    fn tanh_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()>;
}

/// Backward-compatible "all ops" alias. Code that is generic over an
/// arbitrary backend supporting *everything* can keep using `D: OpBackend`.
/// Implemented automatically for any backend that implements all three
/// capability traits — no explicit `impl` required.
pub trait OpBackend: CoreOps + LlmOps + DiffusionOps {}
impl<D: CoreOps + LlmOps + DiffusionOps> OpBackend for D {}
