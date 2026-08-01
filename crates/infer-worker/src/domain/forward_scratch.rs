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
//! A small set of worst-case-sized physical slots is allocated once at
//! `Runtime::new` and shared across all layers via `Rc`. Logical intermediates
//! declare a slot and column rule in `define_forward_buffers!`; values with
//! disjoint lifetimes alias the same slot. Each sublayer takes a contiguous
//! prefix view for its current row/column shape. Slot addresses never change
//! for the runner's lifetime, so they bake cleanly into captured CUDA graphs.
//!
//! Q/K/V are NOT held here — they come from `D::qkv_split`, which on CUDA
//! returns zero-copy column narrows of `qkv` (kernels honor strides) and on the
//! CPU reference materializes contiguous copies (its kernels index a contiguous
//! row stride). `finalize`'s `logits` and norm reuse buffers here too, so a
//! whole forward (prefill or decode) ALLOCATES nothing per step on CUDA.
//! The slot containing `logits [cap_num_tokens, vocab]` is the large buffer; it
//! is also reused by earlier, non-overlapping stages rather than allocating
//! separate buffers. `flash_ws [flash_ws_elems]` is the attention kernel's
//! batched-decode scratch, sized once per `Runtime` to the worst-case
//! `(cap_batch, head_num, head_dim)`; backends that don't need it (CPU
//! reference) get a 0-elem placeholder via `D::flash_decode_workspace_capacity_f32`.

use std::cell::UnsafeCell;
use std::rc::Rc;

use super::model::ModelDims;
use super::ports::{FusedOps, MemoryPort, OpResult};
use super::tensor::Tensor;
use super::types::{Dtype, Shape};

/// Fixed-capacity scratch for the dense decoder forward (attention + dense FFN).
/// Sized for the worst-case `cap_num_tokens` rows; sublayers view a row-prefix.
pub struct ForwardScratch<T: Dtype, D: FusedOps> {
    cap_num_tokens: usize,
    dims: ModelDims,
    /// Physical buffers backing the logical fields declared by
    /// `define_forward_buffers!`. Fields in the same slot have disjoint
    /// lifetimes and reuse the slot's contiguous prefix.
    buffers: Vec<Tensor<T, D>>,
    /// Flash-attention decode workspace, `[flash_ws_elems] f32`. Held in
    /// `UnsafeCell` so per-call `Attention::run` can hand the kernel a `&mut`
    /// view via `flash_workspace_mut`. Concurrent layers are NOT possible (the
    /// decoder runs layers serially on a single stream), and inside one layer
    /// the kernel reads/writes are stream-ordered — so the only thing the
    /// shared-mutability guard needs to protect is the (single-threaded)
    /// borrow discipline, which `&self` enforces by giving out fresh views
    /// each call. Tensor handles are reference-counted views into immutable
    /// storage; this `UnsafeCell` exists purely so we can hand out a `&mut`
    /// handle (NOT mutate the cell itself).
    flash_ws: UnsafeCell<Tensor<f32, D>>,
    flash_ws_elems: usize,
}

#[derive(Debug, Clone, Copy)]
struct BufferSpec {
    slot: usize,
    columns: fn(ModelDims) -> usize,
}

// A logical field is defined exactly once: accessor name, reuse slot and
// column rule live together. Add/remove one row to evolve the forward scratch.
// Adjacent stages must use different slots because GEMM reads one while writing
// the next. Stages two or more steps apart may safely reuse a slot: execution is
// serial on one stream and the earlier value is dead before reuse.
macro_rules! define_forward_buffers {
    ($($field:ident => $accessor:ident { slot: $slot:expr, columns: $columns:expr }),+ $(,)?) => {
        #[derive(Debug, Clone, Copy)]
        enum ForwardBuffer {
            $($field),+
        }

        impl ForwardBuffer {
            const ALL: &'static [Self] = &[$(Self::$field),+];

            fn spec(self) -> BufferSpec {
                match self {
                    $(Self::$field => BufferSpec { slot: $slot, columns: $columns }),+
                }
            }
        }

        impl<T: Dtype, D: FusedOps + MemoryPort> ForwardScratch<T, D> {
            $(
                pub fn $accessor(&self, rows: usize) -> Tensor<T, D> {
                    self.view(ForwardBuffer::$field, rows)
                }
            )+
        }
    };
}

define_forward_buffers! {
    Normed => normed { slot: 0, columns: |d: ModelDims| d.dim },
    Qkv => qkv { slot: 1, columns: |d: ModelDims| d.qkv_dim },
    AttnOut => attn_out { slot: 0, columns: |d: ModelDims| d.q_dim },
    OOut => o_out { slot: 1, columns: |d: ModelDims| d.dim },
    GateUp => gate_up { slot: 1, columns: |d: ModelDims| 2 * d.intermediate_size },
    SwiGlu => swiglu { slot: 0, columns: |d: ModelDims| d.intermediate_size },
    FfnOut => ffn_out { slot: 1, columns: |d: ModelDims| d.dim },
    Logits => logits { slot: 1, columns: |d: ModelDims| d.vocab_size },
}

fn forward_slot_columns(dims: ModelDims) -> Vec<usize> {
    let slot_count = ForwardBuffer::ALL
        .iter()
        .map(|field| field.spec().slot)
        .max()
        .map_or(0, |slot| slot + 1);
    let mut slot_columns = vec![0usize; slot_count];
    for field in ForwardBuffer::ALL {
        let spec = field.spec();
        slot_columns[spec.slot] = slot_columns[spec.slot].max((spec.columns)(dims));
    }
    slot_columns
}

impl<T: Dtype, D: FusedOps + MemoryPort> ForwardScratch<T, D> {
    /// Plan and allocate every physical slot once at worst-case capacity.
    /// Returned behind an `Rc` so it can be cloned into each decoder block.
    ///
    /// `cap_batch` sizes the flash-attention decode workspace (the kernel's
    /// scratch grows with batch, not token count); CPU returns 0 and the
    /// placeholder is unused.
    pub fn new(
        device: &D,
        dims: ModelDims,
        cap_num_tokens: usize,
        cap_batch: usize,
    ) -> OpResult<Rc<Self>> {
        let cap = cap_num_tokens.max(1);
        let buffers = forward_slot_columns(dims)
            .into_iter()
            .map(|cols| Tensor::<T, D>::zeros(Shape::from_slice(&[cap, cols.max(1)]), device))
            .collect::<OpResult<Vec<_>>>()?;
        let flash_ws_elems =
            D::flash_decode_workspace_capacity_f32(cap_batch, dims.head_num, dims.head_dim).max(1);
        let flash_ws = Tensor::<f32, D>::zeros(Shape::from_slice(&[flash_ws_elems]), device)?;
        Ok(Rc::new(Self {
            cap_num_tokens: cap,
            dims,
            buffers,
            flash_ws: UnsafeCell::new(flash_ws),
            flash_ws_elems,
        }))
    }

    /// Row-prefix view: shape `[n, cols]` over a `[cap, cols]` buffer. `cols`
    /// MUST equal the buffer's column count so the contiguous prefix is valid.
    #[inline]
    fn view(&self, field: ForwardBuffer, n: usize) -> Tensor<T, D> {
        debug_assert!(n <= self.cap_num_tokens);
        let spec = field.spec();
        let cols = (spec.columns)(self.dims);
        let shape = Shape::from_slice(&[n, cols]);
        let strides = shape.contiguous_strides();
        self.buffers[spec.slot].view_raw(shape, strides, 0, true)
    }

    /// True when the attention buffers can serve `num_tokens` rows.
    pub fn fits(&self, num_tokens: usize) -> bool {
        num_tokens <= self.cap_num_tokens
    }

    /// True when the FFN buffers fit AND the model's fused gate/up width matches
    /// this scratch's column geometry (guards an FFN with a different
    /// intermediate size, e.g. an MoE shared expert, from reusing it).
    pub fn fits_ffn(&self, num_tokens: usize, gate_cols: usize) -> bool {
        let gate_up = ForwardBuffer::GateUp.spec();
        num_tokens <= self.cap_num_tokens && gate_cols == (gate_up.columns)(self.dims)
    }

    /// Borrow the flash-attention decode workspace mutably for one
    /// kernel call. Single-threaded, layers run serially on the same stream,
    /// so handing the same buffer to each layer in turn is safe (the kernel's
    /// reads-and-writes are stream-ordered).
    ///
    /// Returns a fresh full-buffer view rather than `&mut Tensor` so the
    /// `Rc<Self>` borrow rules stay clean (the returned view owns its
    /// storage handle through the underlying tensor's `Arc`).
    pub fn flash_workspace_mut(&self) -> Tensor<f32, D> {
        // SAFETY: `&self` precludes aliasing across threads (`ForwardScratch`
        // is `!Sync` via `UnsafeCell`). Within one thread, layers run
        // serially: each call obtains a fresh view, hands it to one
        // `attention_paged` invocation, and drops it before the next layer
        // runs. We never hold two live views simultaneously.
        let cell = unsafe { &*self.flash_ws.get() };
        let shape = Shape::from_slice(&[self.flash_ws_elems]);
        let strides = shape.contiguous_strides();
        cell.view_raw(shape, strides, 0, true)
    }

    /// f32 element count of the flash workspace. Mostly useful for tests /
    /// debug assertions.
    pub fn flash_workspace_elems(&self) -> usize {
        self.flash_ws_elems
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_4b_layout_reuses_two_physical_slots() {
        let dims = ModelDims {
            dim: 2560,
            q_dim: 4096,
            kv_dim: 1024,
            qkv_dim: 6144,
            intermediate_size: 9728,
            vocab_size: 151_936,
            ..ModelDims::default()
        };

        assert_eq!(forward_slot_columns(dims), vec![9728, 151_936]);
        let logical_columns: usize = ForwardBuffer::ALL
            .iter()
            .map(|field| (field.spec().columns)(dims))
            .sum();
        assert_eq!(logical_columns, 199_040);
        assert!(forward_slot_columns(dims).into_iter().sum::<usize>() < logical_columns);
    }
}
