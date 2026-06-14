//! `ForwardWorkspace` — fixed-capacity, address-stable scratch for one
//! `LlmModel::forward` call.
//!
//! All per-forward intermediates (`x`, `h`, `qkv_buf`, `attn_out`, `q_buf`,
//! `k_buf`, `v_buf`, `gate_buf`, `up_buf`, `ffn_out`, `o_out`, `logits`)
//! are pre-allocated at runner construction with the worst-case
//! `cap_num_tokens` rows. The model layers use `*_view(num_tokens)` to get
//! a smaller-shape view that shares the same `Arc<Storage>`.
//!
//! The flash-decode attention scratch (`flash_decode_workspace_f32`) and
//! the per-step argmax output (`argmax_out_dev` / `argmax_out_host`) live
//! here too, so attention and sampling do not allocate per call either.
//!
//! This is application/runtime state: it encodes runner capacity, address
//! stability, and CUDA-graph-friendly buffer reuse.
//!
//! # Address stability invariant
//!
//! Once constructed, no method may resize, replace, or grow the tensors —
//! their `data_ptr()`s are baked into captured CUDA graphs. Future code
//! must only update **contents** via `upload_async`, never reallocate.

use crate::domain::model::LlmForwardWorkspace;
use crate::domain::ports::{MemoryPort, OpResult};
use crate::domain::tensor::Tensor;
use crate::domain::types::{Dtype, Shape};

/// Geometric description of a model, used to size the workspace at
/// construction. Pulled out of the `LlmModel` accessors so the workspace
/// doesn't need a generic `M: LlmModel` bound.
#[derive(Debug, Clone, Copy)]
pub struct ModelDims {
    pub dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub qkv_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_num: usize,
    pub head_dim: usize,
}

impl ModelDims {
    /// Geometry sanity checks.
    pub fn validate(&self) -> OpResult<()> {
        use crate::domain::ports::OpError;
        if self.q_dim != self.head_num * self.head_dim {
            return Err(OpError::Shape(format!(
                "ModelDims: q_dim ({}) != head_num ({}) * head_dim ({})",
                self.q_dim, self.head_num, self.head_dim,
            )));
        }
        if self.qkv_dim < self.q_dim + 2 * self.kv_dim {
            return Err(OpError::Shape(format!(
                "ModelDims: qkv_dim ({}) < q_dim ({}) + 2 * kv_dim ({})",
                self.qkv_dim, self.q_dim, self.kv_dim,
            )));
        }
        Ok(())
    }
}

pub struct ForwardWorkspace<T: Dtype, D: MemoryPort> {
    pub cap_num_tokens: usize,
    pub cap_batch: usize,
    pub dims: ModelDims,

    // Per-forward intermediates (alloc once, view via `view_raw`).
    x: Tensor<T, D>,        // [cap_num_tokens, dim]
    h: Tensor<T, D>,        // [cap_num_tokens, dim]
    qkv_buf: Tensor<T, D>,  // [cap_num_tokens, qkv_dim]
    attn_out: Tensor<T, D>, // [cap_num_tokens, q_dim]
    gate_buf: Tensor<T, D>, // [cap_num_tokens, intermediate_size]
    up_buf: Tensor<T, D>,   // [cap_num_tokens, intermediate_size]
    /// Fused gate_up output: [cap_num_tokens, 2*intermediate_size].
    /// The fused gate_up GEMV writes here; swiglu_packed reads it.
    gate_up_buf: Tensor<T, D>,
    ffn_out: Tensor<T, D>, // [cap_num_tokens, dim]
    q_buf: Tensor<T, D>,   // [cap_num_tokens, q_dim]
    k_buf: Tensor<T, D>,   // [cap_num_tokens, kv_dim]
    v_buf: Tensor<T, D>,   // [cap_num_tokens, kv_dim]
    o_out: Tensor<T, D>,   // [cap_num_tokens, dim]
    logits: Tensor<T, D>,  // [cap_num_tokens, vocab_size]

    /// Long-lived attention scratch. Used by paged decode-attention to avoid
    /// per-call `cudaMalloc`. Sized for the worst-case batch.
    flash_decode_workspace_f32: Tensor<f32, D>,

    /// Long-lived argmax output (one i32 per sequence). Written inside the
    /// captured graph. This is buffer **C**; the host reads compacted A and
    /// merge sidebands, never C directly.
    argmax_out_dev: Tensor<i32, D>,

    /// Reusable host slot for the single-D2H read after each step.
    argmax_out_host: Vec<i32>,

    /// Per-seq argmax scratch workspace: [cap_batch, 256].
    /// Passed to the CUDA argmax kernel as a temporary buffer.
    /// `256` is the per-seq scratch size expected by the kernel.
    argmax_workspace: Tensor<f32, D>,

    /// Pre-allocated device buffer for `argmax_batched` row indices.
    /// Shape `[cap_batch]`.  Caller uploads `selected_rows` each step;
    /// passed to the kernel as `selected_rows_device`.
    argmax_rows_dev: Tensor<i32, D>,

    // ─── ABC decode pipeline buffers ────────────────────────────────
    //
    // Buffer A lives in BatchWorkspace::input_ids. Buffer B is used for
    // newly-admitted first decode tokens before they are appended into A.
    // Buffer C is `argmax_out_dev`, produced by graph forward.
    /// Buffer **B**: first tokens of newly-admitted sequences. Uploaded
    /// async on the copy-in stream each step. Len `cap_batch`.
    new_token_dev: Tensor<i32, D>,

    /// Host staging for `new_token_dev` (owned for the runner's lifetime so
    /// `upload_async` is safe without an immediate sync). Len `cap_batch`.
    new_token_host: Vec<i32>,

    // ─── Decode compact metadata/output buffers ─────────────────────
    //
    // The ABC serving path runs graph forward A -> C, then a compact merge
    // kernel decides finish/non-finish, writes surviving C tokens back to A,
    // and reports row counts. These buffers are address-stable and reused.
    decode_generated_counts_dev: Tensor<i32, D>,
    decode_max_tokens_dev: Tensor<i32, D>,
    decode_ignore_eos_dev: Tensor<i32, D>,
    decode_eos_ids_dev: Tensor<i32, D>,
    decode_active_src_rows_dev: Tensor<i32, D>,
    decode_finished_src_rows_dev: Tensor<i32, D>,
    decode_finished_tokens_dev: Tensor<i32, D>,
    decode_counts_dev: Tensor<i32, D>,

    decode_generated_counts_host: Vec<i32>,
    decode_max_tokens_host: Vec<i32>,
    decode_ignore_eos_host: Vec<i32>,
    decode_eos_ids_host: Vec<i32>,
}

impl<T: Dtype, D: MemoryPort> ForwardWorkspace<T, D> {
    /// Allocate at the worst-case capacity. Sizes are baked in for the
    /// runner's lifetime — no resize after construction.
    ///
    /// `flash_decode_capacity` controls the f32 attention scratch (the
    /// existing kernel reports its requirement as `≈ batch * head_num *
    /// head_dim` floats; we let the caller pass it explicitly).
    pub fn new(
        device: &D,
        dims: ModelDims,
        cap_num_tokens: usize,
        cap_batch: usize,
        flash_decode_capacity: usize,
    ) -> OpResult<Self> {
        dims.validate()?;
        let alloc = |rows: usize, cols: usize| -> OpResult<Tensor<T, D>> {
            Tensor::<T, D>::zeros(Shape::from_slice(&[rows, cols]), device)
        };
        let x = alloc(cap_num_tokens, dims.dim)?;
        let h = alloc(cap_num_tokens, dims.dim)?;
        let qkv_buf = alloc(cap_num_tokens, dims.qkv_dim)?;
        let attn_out = alloc(cap_num_tokens, dims.q_dim)?;
        let gate_buf = alloc(cap_num_tokens, dims.intermediate_size)?;
        let up_buf = alloc(cap_num_tokens, dims.intermediate_size)?;
        let gate_up_buf = alloc(cap_num_tokens, 2 * dims.intermediate_size)?;
        let ffn_out = alloc(cap_num_tokens, dims.dim)?;
        let q_buf = alloc(cap_num_tokens, dims.q_dim)?;
        let k_buf = alloc(cap_num_tokens, dims.kv_dim)?;
        let v_buf = alloc(cap_num_tokens, dims.kv_dim)?;
        let o_out = alloc(cap_num_tokens, dims.dim)?;
        let logits = alloc(cap_num_tokens, dims.vocab_size)?;

        let flash_decode_workspace_f32 =
            Tensor::<f32, D>::zeros(Shape::from_slice(&[flash_decode_capacity.max(1)]), device)?;
        let argmax_out_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let argmax_out_host = vec![0i32; cap_batch.max(1)];
        let argmax_workspace =
            Tensor::<f32, D>::zeros(Shape::from_slice(&[cap_batch.max(1), 256]), device)?;
        let argmax_rows_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;

        // ABC decode pipeline buffer B, address-stable at cap_batch.
        let new_token_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let new_token_host = vec![0i32; cap_batch.max(1)];

        let decode_generated_counts_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let decode_max_tokens_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let decode_ignore_eos_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let decode_eos_ids_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_num_tokens.max(1)]), device)?;
        let decode_active_src_rows_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let decode_finished_src_rows_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let decode_finished_tokens_dev =
            Tensor::<i32, D>::zeros(Shape::from_slice(&[cap_batch.max(1)]), device)?;
        let decode_counts_dev = Tensor::<i32, D>::zeros(Shape::from_slice(&[3]), device)?;

        let decode_generated_counts_host = vec![0i32; cap_batch.max(1)];
        let decode_max_tokens_host = vec![0i32; cap_batch.max(1)];
        let decode_ignore_eos_host = vec![0i32; cap_batch.max(1)];
        let decode_eos_ids_host = vec![0i32; cap_num_tokens.max(1)];

        Ok(Self {
            cap_num_tokens,
            cap_batch,
            dims,
            x,
            h,
            qkv_buf,
            attn_out,
            gate_buf,
            up_buf,
            gate_up_buf,
            ffn_out,
            q_buf,
            k_buf,
            v_buf,
            o_out,
            logits,
            flash_decode_workspace_f32,
            argmax_out_dev,
            argmax_out_host,
            argmax_workspace,
            argmax_rows_dev,
            new_token_dev,
            new_token_host,
            decode_generated_counts_dev,
            decode_max_tokens_dev,
            decode_ignore_eos_dev,
            decode_eos_ids_dev,
            decode_active_src_rows_dev,
            decode_finished_src_rows_dev,
            decode_finished_tokens_dev,
            decode_counts_dev,
            decode_generated_counts_host,
            decode_max_tokens_host,
            decode_ignore_eos_host,
            decode_eos_ids_host,
        })
    }

    // ─── Per-call shape views ────────────────────────────────────────

    fn narrow_rows(t: &Tensor<T, D>, n: usize, cols: usize) -> Tensor<T, D> {
        let strides = Shape::from_slice(&[n.max(1), cols]).contiguous_strides();
        t.view_raw(Shape::from_slice(&[n, cols]), strides, 0, true)
    }

    pub fn x_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.x, n, self.dims.dim)
    }
    pub fn h_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.h, n, self.dims.dim)
    }
    pub fn qkv_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.qkv_buf, n, self.dims.qkv_dim)
    }
    pub fn attn_out_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.attn_out, n, self.dims.q_dim)
    }
    pub fn q_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.q_buf, n, self.dims.q_dim)
    }
    pub fn k_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.k_buf, n, self.dims.kv_dim)
    }
    pub fn v_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.v_buf, n, self.dims.kv_dim)
    }
    pub fn o_out_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.o_out, n, self.dims.dim)
    }
    pub fn gate_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.gate_buf, n, self.dims.intermediate_size)
    }
    pub fn up_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.up_buf, n, self.dims.intermediate_size)
    }
    pub fn gate_up_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.gate_up_buf, n, 2 * self.dims.intermediate_size)
    }
    pub fn ffn_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.ffn_out, n, self.dims.dim)
    }
    pub fn logits_view(&self, n: usize) -> Tensor<T, D> {
        Self::narrow_rows(&self.logits, n, self.dims.vocab_size)
    }

    pub fn flash_decode_workspace(&mut self) -> &mut Tensor<f32, D> {
        &mut self.flash_decode_workspace_f32
    }
    pub fn argmax_out_dev_mut(&mut self) -> &mut Tensor<i32, D> {
        &mut self.argmax_out_dev
    }
    pub fn argmax_out_dev(&self) -> &Tensor<i32, D> {
        &self.argmax_out_dev
    }
    pub fn argmax_out_host_mut(&mut self) -> &mut [i32] {
        &mut self.argmax_out_host
    }
    pub fn argmax_workspace_mut(&mut self) -> &mut Tensor<f32, D> {
        &mut self.argmax_workspace
    }
    pub fn argmax_workspace(&self) -> &Tensor<f32, D> {
        &self.argmax_workspace
    }
    pub fn argmax_rows_dev_mut(&mut self) -> &mut Tensor<i32, D> {
        &mut self.argmax_rows_dev
    }
    pub fn argmax_rows_dev(&self) -> &Tensor<i32, D> {
        &self.argmax_rows_dev
    }
    /// 同时借出 `argmax_out_dev` (mut)、`argmax_workspace` (imm) 和
    /// `argmax_rows_dev` (mut)，绕过 Rust 借用检查器无法在同一调用中
    /// 同时取 `&mut self` + `&self` 的限制。
    pub fn argmax_args(&mut self) -> (&mut Tensor<i32, D>, &Tensor<f32, D>, &mut Tensor<i32, D>) {
        (
            &mut self.argmax_out_dev,
            &self.argmax_workspace,
            &mut self.argmax_rows_dev,
        )
    }

    // ─── Bubble-free decode pipeline accessors ───────────────────────

    /// Buffer **B** (`new_token_dev`): newly-admitted seqs' first tokens.
    pub fn new_token_dev(&self) -> &Tensor<i32, D> {
        &self.new_token_dev
    }
    pub fn new_token_dev_mut(&mut self) -> &mut Tensor<i32, D> {
        &mut self.new_token_dev
    }
    /// Host staging mirrors (owned for the runner's lifetime; safe for
    /// `upload_async`).
    pub fn new_token_host_mut(&mut self) -> &mut [i32] {
        &mut self.new_token_host
    }
    /// View of B sized to the active batch (for the kernel launch).
    pub fn new_token_view(&self, batch: usize) -> Tensor<i32, D> {
        let strides = Shape::from_slice(&[batch.max(1)]).contiguous_strides();
        self.new_token_dev
            .view_raw(Shape::from_slice(&[batch]), strides, 0, true)
    }

    pub fn decode_generated_counts_dev(&self) -> &Tensor<i32, D> {
        &self.decode_generated_counts_dev
    }
    pub fn decode_max_tokens_dev(&self) -> &Tensor<i32, D> {
        &self.decode_max_tokens_dev
    }
    pub fn decode_ignore_eos_dev(&self) -> &Tensor<i32, D> {
        &self.decode_ignore_eos_dev
    }
    pub fn decode_eos_ids_dev(&self) -> &Tensor<i32, D> {
        &self.decode_eos_ids_dev
    }
    pub fn decode_active_src_rows_dev(&self) -> &Tensor<i32, D> {
        &self.decode_active_src_rows_dev
    }
    pub fn decode_finished_src_rows_dev(&self) -> &Tensor<i32, D> {
        &self.decode_finished_src_rows_dev
    }
    pub fn decode_finished_tokens_dev(&self) -> &Tensor<i32, D> {
        &self.decode_finished_tokens_dev
    }
    pub fn decode_counts_dev(&self) -> &Tensor<i32, D> {
        &self.decode_counts_dev
    }

    pub fn decode_generated_counts_host_mut(&mut self) -> &mut [i32] {
        &mut self.decode_generated_counts_host
    }
    pub fn decode_max_tokens_host_mut(&mut self) -> &mut [i32] {
        &mut self.decode_max_tokens_host
    }
    pub fn decode_ignore_eos_host_mut(&mut self) -> &mut [i32] {
        &mut self.decode_ignore_eos_host
    }
    pub fn decode_eos_ids_host_mut(&mut self) -> &mut [i32] {
        &mut self.decode_eos_ids_host
    }
}

impl<T: Dtype, D: MemoryPort> LlmForwardWorkspace<T, D> for ForwardWorkspace<T, D> {
    fn x_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::x_view(self, n)
    }

    fn h_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::h_view(self, n)
    }

    fn qkv_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::qkv_view(self, n)
    }

    fn attn_out_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::attn_out_view(self, n)
    }

    fn gate_up_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::gate_up_view(self, n)
    }

    fn gate_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::gate_view(self, n)
    }

    fn ffn_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::ffn_view(self, n)
    }

    fn o_out_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::o_out_view(self, n)
    }

    fn logits_view(&self, n: usize) -> Tensor<T, D> {
        ForwardWorkspace::logits_view(self, n)
    }

    fn flash_decode_workspace(&mut self) -> &mut Tensor<f32, D> {
        ForwardWorkspace::flash_decode_workspace(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::cpu::Cpu;

    fn dims() -> ModelDims {
        ModelDims {
            dim: 16,
            q_dim: 16,
            kv_dim: 16,
            qkv_dim: 48,
            intermediate_size: 32,
            vocab_size: 64,
            head_num: 2,
            head_dim: 8,
        }
    }

    #[test]
    fn alloc_and_view_aliases_storage() {
        let ws: ForwardWorkspace<f32, Cpu> = ForwardWorkspace::new(&Cpu, dims(), 8, 4, 32).unwrap();
        let v1 = ws.x_view(3);
        let v2 = ws.x_view(7);
        // Both views share the same underlying storage pointer (Arc).
        assert_eq!(v1.data_ptr(), v2.data_ptr());
        assert_eq!(v1.shape().as_slice(), &[3, 16]);
        assert_eq!(v2.shape().as_slice(), &[7, 16]);
    }

    #[test]
    fn dims_validation_catches_mismatch() {
        let mut bad = dims();
        bad.q_dim = 17; // not head_num*head_dim
        assert!(ForwardWorkspace::<f32, Cpu>::new(&Cpu, bad, 4, 4, 32).is_err());
    }
}
