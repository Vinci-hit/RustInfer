//! Domain ports — trait definitions that Infrastructure must implement.
//!
//! This is the hexagonal "port" layer. Domain code programs against these
//! traits; infra/ provides the concrete adapters (CPU, CUDA).

use std::alloc::Layout;
use std::ptr::NonNull;
use std::fmt::Debug;

use super::types::{Dtype, Shape};
use super::tensor::Tensor;

/// A compute device (CPU or CUDA). The TypeState axis for Tensor<T, D>.
pub trait Device: Clone + Send + Sync + Debug + 'static {
    /// Execution context (CUDA: handles+stream; CPU: ()).
    type ExecCtx: Send + Sync;
    /// Borrow the exec context.
    fn exec_ctx(&self) -> &Self::ExecCtx;
    /// Human-readable name for errors.
    fn name(&self) -> &'static str;
}

/// Marker: "this device has host-accessible memory" (enables as_slice).
pub trait HostDevice: Device {}

/// Memory allocator port.
pub trait Allocator: Debug + Send + Sync {
    /// Allocate raw bytes.
    /// # Safety: caller must dealloc with same allocator + layout.
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;
    /// Deallocate.
    /// # Safety: ptr must come from this allocator with matching layout.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

#[derive(Debug, thiserror::Error)]
#[error("allocation failed: {0}")]
pub struct AllocError(pub String);

/// Operator backend port — defines WHAT each op does, not how.
///
/// Infrastructure provides `impl OpBackend for Cpu` and `impl OpBackend for Cuda`.
/// Every method MUST have a real implementation — no `Unsupported` allowed.
///
/// Generic bounds use `Dtype` (not `Float`) so quantized types (i8, i4)
/// can participate in mixed-precision operations (e.g. int8 weight × bf16 activation).
pub trait OpBackend: Device {
    // ─── Allocation ──────────────────────────────────────────────────
    /// Allocate a contiguous, zeroed tensor on this device.
    fn alloc_tensor<T: Dtype>(shape: Shape, device: &Self) -> OpResult<Tensor<T, Self>>;

    // ─── Element-wise ────────────────────────────────────────────────
    fn add<T: Dtype>(a: &Tensor<T, Self>, b: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()>;
    fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()>;

    // ─── Normalization ───────────────────────────────────────────────
    fn rmsnorm<T: Dtype>(input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>, eps: f32) -> OpResult<()>;
    fn rmsnorm_inplace<T: Dtype>(x: &mut Tensor<T, Self>, weight: &Tensor<T, Self>, eps: f32) -> OpResult<()>;

    /// Fused: residual += input; output = rmsnorm(residual, weight, eps)
    /// Saves one global memory pass vs separate add + rmsnorm.
    fn fused_add_rmsnorm<T: Dtype>(
        output: &mut Tensor<T, Self>,
        residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()>;

    // ─── Linear algebra ──────────────────────────────────────────────
    /// Standard matmul: input [M,K] × weight [N,K]^T → output [M,N]
    /// Same dtype for input, weight, output.
    fn matmul<T: Dtype>(input: &Tensor<T, Self>, weight: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;

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
    fn swiglu_inplace<T: Dtype>(x: &mut Tensor<T, Self>, gate: &Tensor<T, Self>) -> OpResult<()>;

    // ─── Softmax ─────────────────────────────────────────────────────
    fn softmax<T: Dtype>(input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;

    // ─── Scalar ──────────────────────────────────────────────────────
    fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()>;

    // ─── Embedding ───────────────────────────────────────────────────
    fn embedding<T: Dtype>(table: &Tensor<T, Self>, indices: &Tensor<i32, Self>, output: &mut Tensor<T, Self>) -> OpResult<()>;

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

    // ─── Attention ───────────────────────────────────────────────────
    /// Scaled dot-product attention.
    /// - q: [num_tokens, q_dim]
    /// - k: [total_kv_len, kv_dim] (from KV cache)
    /// - v: [total_kv_len, kv_dim]
    /// - output: [num_tokens, q_dim]
    /// - seq_starts: [batch+1] prefix-sum of sequence lengths (for ragged batch)
    /// - head_num / kv_head_num / head_dim: model geometry
    /// - scale: 1/sqrt(head_dim)
    fn attention<T: Dtype>(
        q: &Tensor<T, Self>,
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        seq_starts: &Tensor<i32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()>;

    // ─── KV Cache ────────────────────────────────────────────────────
    /// Split a fused [num_tokens, qkv_dim] tensor into Q, K, V.
    /// Copies columns [0..q_dim), [q_dim..q_dim+kv_dim), [q_dim+kv_dim..qkv_dim).
    fn split_qkv<T: Dtype>(
        qkv: &Tensor<T, Self>,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        v: &mut Tensor<T, Self>,
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()>;

    /// Scatter K/V rows into the KV cache at given positions.
    /// k: [num_tokens, kv_dim], v: [num_tokens, kv_dim]
    /// positions: position[t] = which cache row to write token t into.
    fn scatter_kv<T: Dtype>(
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        k_cache: &mut Tensor<T, Self>,
        v_cache: &mut Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        kv_dim: usize,
    ) -> OpResult<()>;

    // ─── Sampling ────────────────────────────────────────────────────
    /// Argmax over the last row of logits → output token ID.
    /// logits: [num_rows, vocab_size], returns argmax of last row.
    fn argmax<T: Dtype>(logits: &Tensor<T, Self>, num_rows: usize) -> OpResult<i32>;

    // ═══════════════════════════════════════════════════════════════════
    // ─── Diffusion ops (Conv / Norm / Spatial) ─────────────────────────
    // ═══════════════════════════════════════════════════════════════════

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

    // ─── Broadcast multiply (adaLN) ──────────────────────────────────
    /// In-place broadcast multiply: x[i,j] *= scale[j].
    /// x: [rows, dim], scale: [dim] or [1, dim].
    fn broadcast_mul_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()>;

    // ─── Element-wise multiply ───────────────────────────────────────
    fn ewise_mul<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
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
}

/// Operator-level error — no `Unsupported` variant.
/// All ops must be implemented; if it can't run, it's a Shape or Kernel error.
#[derive(Debug, thiserror::Error)]
pub enum OpError {
    #[error("shape error: {0}")]
    Shape(String),
    #[error("not contiguous: shape={0:?}")]
    NotContiguous(Shape),
    #[error("kernel failed: {0}")]
    Kernel(String),
}

pub type OpResult<T> = std::result::Result<T, OpError>;
