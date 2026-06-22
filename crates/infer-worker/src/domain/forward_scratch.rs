//! `ForwardScratch` — address-stable, preallocated per-layer forward scratch.
//!
//! Restores the pre-refactor zero-allocation forward. The decoder's attention
//! and dense-FFN sublayers reuse fixed workspace buffers across every layer
//! instead of `cudaMalloc`-ing fresh tensors each call. The component refactor
//! lost this and reintroduced ~11 device allocations PER LAYER (≈396 per
//! 36-layer forward) — a `cudaMalloc`/`cudaFree`/memset storm that dominated
//! eager-prefill TTFT and also baked ~396 per-step memsets into the captured
//! decode graph (replayed every token → TTOT regression).
//!
//! One set of worst-case-sized buffers is allocated once at `Runtime::new` and
//! shared across all layers via `Rc`; each sublayer takes a smaller-row
//! `view_raw` per call (cols fixed = buffer cols, only the row count shrinks).
//! Buffer addresses never change for the runner's lifetime, so they bake
//! cleanly into captured CUDA graphs — strictly safer than the capture arena,
//! whose stability depended on deterministic per-step allocation order.
//!
//! Q/K/V are NOT held here — they come from `D::qkv_split`, which on CUDA
//! returns zero-copy column narrows of `qkv` (kernels honor strides) and on the
//! CPU reference materializes contiguous copies (its kernels index a contiguous
//! row stride). `finalize`'s `logits` and norm reuse buffers here too, so a
//! whole forward (prefill or decode) ALLOCATES nothing per step on CUDA.
//! `logits [cap_num_tokens, vocab]` is the one large buffer; it is preallocated
//! rather than pooled so ragged prompt lengths cannot grow the allocator's free
//! list without bound.

use std::rc::Rc;

use super::model::ModelDims;
use super::ports::{MemoryPort, OpResult};
use super::tensor::Tensor;
use super::types::{Dtype, Shape};

/// Fixed-capacity scratch for the dense decoder forward (attention + dense FFN).
/// Sized for the worst-case `cap_num_tokens` rows; sublayers view a row-prefix.
pub struct ForwardScratch<T: Dtype, D: MemoryPort> {
    cap_num_tokens: usize,
    dims: ModelDims,

    normed: Tensor<T, D>,   // [cap, dim]   (attn pre-norm + ffn pre-norm + finalize norm)
    qkv: Tensor<T, D>,      // [cap, qkv_dim]  (Q/K/V come from D::qkv_split: CUDA narrows this)
    attn_out: Tensor<T, D>, // [cap, q_dim]
    o_out: Tensor<T, D>,    // [cap, dim]
    gate_up: Tensor<T, D>,  // [cap, 2*intermediate_size]
    swiglu: Tensor<T, D>,   // [cap, intermediate_size]
    ffn_out: Tensor<T, D>,  // [cap, dim]
    logits: Tensor<T, D>,   // [cap, vocab_size]  (finalize lm_head output)
}

impl<T: Dtype, D: MemoryPort> ForwardScratch<T, D> {
    /// Allocate every buffer once at worst-case capacity. Returned behind an
    /// `Rc` so it can be cloned into each decoder block's sublayers.
    pub fn new(device: &D, dims: ModelDims, cap_num_tokens: usize) -> OpResult<Rc<Self>> {
        let cap = cap_num_tokens.max(1);
        let alloc = |cols: usize| -> OpResult<Tensor<T, D>> {
            Tensor::<T, D>::zeros(Shape::from_slice(&[cap, cols.max(1)]), device)
        };
        Ok(Rc::new(Self {
            cap_num_tokens: cap,
            dims,
            normed: alloc(dims.dim)?,
            qkv: alloc(dims.qkv_dim)?,
            attn_out: alloc(dims.q_dim)?,
            o_out: alloc(dims.dim)?,
            gate_up: alloc(2 * dims.intermediate_size)?,
            swiglu: alloc(dims.intermediate_size)?,
            ffn_out: alloc(dims.dim)?,
            logits: alloc(dims.vocab_size)?,
        }))
    }

    /// Row-prefix view: shape `[n, cols]` over a `[cap, cols]` buffer. `cols`
    /// MUST equal the buffer's column count so the contiguous prefix is valid.
    #[inline]
    fn view(t: &Tensor<T, D>, n: usize, cols: usize) -> Tensor<T, D> {
        let shape = Shape::from_slice(&[n, cols]);
        let strides = shape.contiguous_strides();
        t.view_raw(shape, strides, 0, true)
    }

    /// True when the attention buffers can serve `num_tokens` rows.
    pub fn fits(&self, num_tokens: usize) -> bool {
        num_tokens <= self.cap_num_tokens
    }

    /// True when the FFN buffers fit AND the model's fused gate/up width matches
    /// this scratch's column geometry (guards an FFN with a different
    /// intermediate size, e.g. an MoE shared expert, from reusing it).
    pub fn fits_ffn(&self, num_tokens: usize, gate_cols: usize) -> bool {
        num_tokens <= self.cap_num_tokens && gate_cols == 2 * self.dims.intermediate_size
    }

    pub fn normed(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.normed, n, self.dims.dim)
    }
    pub fn qkv(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.qkv, n, self.dims.qkv_dim)
    }
    pub fn attn_out(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.attn_out, n, self.dims.q_dim)
    }
    pub fn o_out(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.o_out, n, self.dims.dim)
    }
    pub fn gate_up(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.gate_up, n, 2 * self.dims.intermediate_size)
    }
    pub fn swiglu(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.swiglu, n, self.dims.intermediate_size)
    }
    pub fn ffn_out(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.ffn_out, n, self.dims.dim)
    }
    /// `finalize` lm_head output, `[n, vocab_size]`.
    pub fn logits(&self, n: usize) -> Tensor<T, D> {
        Self::view(&self.logits, n, self.dims.vocab_size)
    }
}
