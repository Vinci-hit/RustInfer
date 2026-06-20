use super::device::MemoryPort;
use super::error::OpResult;
use infer_core::tensor::Tensor;
use infer_core::types::{Dtype, Shape};

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
/// Diffusion-family ops live in [`DiffusionOps`] (conv / VAE / DiT). These capability traits describe the
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

    // ─── Normalization / RoPE / activation (shared by the LLM decode path,
    //     the diffusion text encoder, and the CPU `MathOps` bridge) ─────
    /// RMSNorm: `output = x / rms(x) * weight`.
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

    /// Apply Rotary Position Embedding in-place to Q and K.
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

    /// Packed SwiGLU: `gate_up [rows, 2*inter]` → `out [rows, inter]`,
    /// `out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]`.
    fn swiglu_packed<T: Dtype>(
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()>;
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

/// "All ops" alias for code that is generic over a backend supporting every
/// model-family capability.
/// Implemented automatically for any backend that implements all three
/// capability traits — no explicit `impl` required.
pub trait OpBackend: CoreOps + DiffusionOps {}
impl<D: CoreOps + DiffusionOps> OpBackend for D {}
