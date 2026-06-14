//! CPU infrastructure adapter.
//!
//! Implements `Device`, `HostDevice`, `MemoryPort`, and reference CPU ops.
//! Some GPU-oriented ops intentionally return `OpError::Unsupported`.

use std::alloc::Layout;
use std::ptr::NonNull;

use half::{bf16, f16};

use crate::domain::ports::{
    AllocError, Allocator, CoreOps, Device, DiffusionOps, HostDevice, LlmOps, MemoryPort, OpError,
    OpResult,
};
use crate::domain::tensor::Tensor;
use crate::domain::types::{DataType, Dtype, Shape};

// ─── Cpu Device ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu;

impl Device for Cpu {
    type ExecCtx = ();
    fn exec_ctx(&self) -> &() {
        &()
    }
    fn name(&self) -> &'static str {
        "cpu"
    }
}
impl HostDevice for Cpu {}

// ─── Cpu MemoryPort ──────────────────────────────────────────────────────────

/// CPU layout: 16-byte aligned, size rounded up to 16 (matches existing kernel
/// expectations and SIMD alignment).
#[inline]
fn cpu_layout(size: usize) -> Layout {
    Layout::from_size_align(size.max(1), 16).expect("invalid layout")
}

impl MemoryPort for Cpu {
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>> {
        let layout = cpu_layout(size);
        // SAFETY: layout is valid (size>=1, align=16).
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        NonNull::new(ptr).ok_or_else(|| OpError::Kernel("CPU alloc returned null".into()))
    }

    unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize) {
        let layout = cpu_layout(size);
        // SAFETY: layout matches the one used in alloc_bytes.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }

    unsafe fn upload(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        // SAFETY: caller provides valid src/dst with `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_ptr(), size) };
        Ok(())
    }

    unsafe fn upload_async(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        // CPU: memcpy is already synchronous; "async" semantics are a no-op.
        unsafe { std::ptr::copy_nonoverlapping(src, dst.as_ptr(), size) };
        Ok(())
    }

    unsafe fn download(&self, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()> {
        // SAFETY: caller provides valid src/dst with `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, size) };
        Ok(())
    }

    fn synchronize(&self) -> OpResult<()> {
        Ok(())
    }

    unsafe fn copy_device_to_device(
        &self,
        dst: NonNull<u8>,
        src: NonNull<u8>,
        size: usize,
    ) -> OpResult<()> {
        // SAFETY: caller guarantees dst/src are valid device (host) pointers,
        // size is correct, and regions do not overlap.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_ptr(), size) };
        Ok(())
    }
}

// ─── Cpu Allocator (legacy, used by VAE pool) ────────────────────────────────

#[derive(Debug)]
pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        NonNull::new(ptr).ok_or_else(|| AllocError("CPU alloc returned null".into()))
    }
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

// ─── Tensor construction helpers (CPU) ───────────────────────────────────────
//
// `Tensor::zeros(shape, &Cpu)` is the canonical entry point; the convenience
// wrappers below preserve the historical CPU-only API surface.

impl<T: Dtype> Tensor<T, Cpu> {
    /// CPU-only convenience: allocate a contiguous, zero-initialized tensor.
    /// Equivalent to `Tensor::zeros(shape, &Cpu).unwrap()`.
    pub fn zeros_cpu(shape: impl Into<Shape>) -> Tensor<T, Cpu> {
        Tensor::<T, Cpu>::zeros(shape, &Cpu).expect("CPU alloc cannot fail")
    }

    /// Create from existing host data (copies bytes into a fresh allocation).
    pub fn from_slice(data: &[T], shape: impl Into<Shape>) -> Tensor<T, Cpu> {
        Tensor::<T, Cpu>::from_host_slice(data, shape, &Cpu)
            .expect("CPU from_host_slice cannot fail")
    }

    /// Borrow the tensor as a typed slice (CPU + contiguous only).
    pub fn as_slice(&self) -> &[T] {
        assert!(self.is_contiguous(), "as_slice requires contiguous");
        // SAFETY: CPU storage is host-accessible; pointer is valid for `numel` elements.
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.numel()) }
    }

    /// Mutable typed slice. Note: takes `&mut self` to encode exclusive
    /// access at the call site, even though the underlying Arc is shared
    /// — callers that hold multiple Arcs to the same storage must
    /// coordinate access themselves.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        assert!(self.is_contiguous(), "as_slice_mut requires contiguous");
        // SAFETY: CPU storage is host-accessible; pointer is valid for `numel` elements.
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), self.numel()) }
    }
}

// ─── CoreOps for Cpu ─────────────────────────────────────────────────────────
// Primitives shared by every model family.

impl CoreOps for Cpu {
    fn add<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        check_contiguous3(a, b, dst)?;
        check_numel3(a, b, dst)?;
        for i in 0..a.numel() {
            unsafe {
                write_f64(
                    dst.data_ptr_mut().add(i),
                    read_f64(a.data_ptr().add(i)) + read_f64(b.data_ptr().add(i)),
                )
            };
        }
        Ok(())
    }

    fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()> {
        assert_eq!(dst.numel(), src.numel());
        for i in 0..dst.numel() {
            unsafe {
                let v = read_f64(dst.data_ptr().add(i)) + read_f64(src.data_ptr().add(i));
                write_f64(dst.data_ptr_mut().add(i), v);
            }
        }
        Ok(())
    }

    fn ewise_mul<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        for i in 0..a.numel() {
            unsafe {
                let va = read_f64(a.data_ptr().add(i));
                let vb = read_f64(b.data_ptr().add(i));
                write_f64(dst.data_ptr_mut().add(i), va * vb);
            }
        }
        Ok(())
    }

    fn matmul<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let m = input.shape().as_slice()[0];
        let k = input.shape().as_slice()[1];
        let n = weight.shape().as_slice()[0];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    unsafe {
                        sum += read_f64(input.data_ptr().add(i * k + p))
                            * read_f64(weight.data_ptr().add(j * k + p));
                    }
                }
                unsafe { write_f64(output.data_ptr_mut().add(i * n + j), sum) };
            }
        }
        Ok(())
    }

    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        input: &Tensor<A, Self>,
        weight: &Tensor<W, Self>,
        output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>,
        _zeros: Option<&Tensor<W, Self>>,
        group_size: usize,
    ) -> OpResult<()> {
        // CPU reference: dequantize weight per group, then matmul
        let m = input.shape().as_slice()[0];
        let k = input.shape().as_slice()[1];
        let n = weight.shape().as_slice()[0];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let group = p / group_size;
                    let scale =
                        unsafe { read_f64(scales.data_ptr().add(j * (k / group_size) + group)) };
                    let w = unsafe { read_f64(weight.data_ptr().add(j * k + p)) };
                    let a = unsafe { read_f64(input.data_ptr().add(i * k + p)) };
                    sum += a * w * scale;
                }
                unsafe { write_f64(output.data_ptr_mut().add(i * n + j), sum) };
            }
        }
        Ok(())
    }

    fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        for i in 0..x.numel() {
            unsafe {
                let v = read_f64((x.data_ptr() as *const T).add(i));
                write_f64(x.data_ptr_mut().add(i), v / (1.0 + (-v).exp()));
            }
        }
        Ok(())
    }

    fn softmax<T: Dtype>(input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()> {
        let dim = *input.shape().as_slice().last().unwrap();
        let rows = input.numel() / dim;
        for row in 0..rows {
            let off = row * dim;
            let mut max_v = f64::NEG_INFINITY;
            for i in 0..dim {
                unsafe {
                    let v = read_f64(input.data_ptr().add(off + i));
                    if v > max_v {
                        max_v = v;
                    }
                }
            }
            let mut sum = 0.0f64;
            for i in 0..dim {
                unsafe {
                    let e = (read_f64(input.data_ptr().add(off + i)) - max_v).exp();
                    write_f64(output.data_ptr_mut().add(off + i), e);
                    sum += e;
                }
            }
            for i in 0..dim {
                unsafe {
                    let v = read_f64((output.data_ptr() as *const T).add(off + i));
                    write_f64(output.data_ptr_mut().add(off + i), v / sum);
                }
            }
        }
        Ok(())
    }

    fn embedding<T: Dtype>(
        table: &Tensor<T, Self>,
        indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let dim = table.shape().as_slice()[1];
        let seq_len = indices.numel();
        let idx_slice = unsafe { std::slice::from_raw_parts(indices.data_ptr(), seq_len) };
        for i in 0..seq_len {
            let idx = idx_slice[i] as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (table.data_ptr() as *const u8).add(idx * dim * T::SIZE_BYTES),
                    (output.data_ptr_mut() as *mut u8).add(i * dim * T::SIZE_BYTES),
                    dim * T::SIZE_BYTES,
                );
            }
        }
        Ok(())
    }

    fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()> {
        for i in 0..x.numel() {
            unsafe {
                let v = read_f64((x.data_ptr() as *const T).add(i));
                write_f64(x.data_ptr_mut().add(i), v * scalar);
            }
        }
        Ok(())
    }

    fn broadcast_mul_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let dim = scale.numel();
        let rows = x.numel() / dim;
        for row in 0..rows {
            for i in 0..dim {
                unsafe {
                    let v = read_f64((x.data_ptr() as *const T).add(row * dim + i));
                    let s = read_f64(scale.data_ptr().add(i));
                    write_f64(x.data_ptr_mut().add(row * dim + i), v * s);
                }
            }
        }
        Ok(())
    }

    fn scalar_add_inplace<T: Dtype>(_x: &mut Tensor<T, Self>, _scalar: f64) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "scalar_add_inplace"))
    }

    fn scalar_mul_inplace_from_dev<T: Dtype>(
        _x: &mut Tensor<T, Self>,
        _d_scalar: &Tensor<f32, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "scalar_mul_inplace_from_dev"))
    }

    fn broadcast_add_inplace<T: Dtype>(
        _x: &mut Tensor<T, Self>,
        _bias: &Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "broadcast_add_inplace"))
    }

    fn split_cols<T: Dtype>(
        _src: &Tensor<T, Self>,
        _dst: &mut Tensor<T, Self>,
        _rows: usize,
        _total_cols: usize,
        _col_offset: usize,
        _dst_cols: usize,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "split_cols"))
    }

    fn concat_seq<T: Dtype>(
        _a: &Tensor<T, Self>,
        _b: &Tensor<T, Self>,
        _dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "concat_seq"))
    }

    fn cast_dtype<S: Dtype, D2: Dtype>(
        _src: &Tensor<S, Self>,
        _dst: &mut Tensor<D2, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "cast_dtype"))
    }
}

// ─── LlmOps for Cpu ──────────────────────────────────────────────────────────
// Decoder + paged-KV ops shared by Llama3 / Qwen3 / Qwen3_5.

impl LlmOps for Cpu {
    fn rmsnorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let dim = *input.shape().as_slice().last().unwrap();
        let rows = input.numel() / dim;
        for row in 0..rows {
            let off = row * dim;
            let mut ss = 0.0f64;
            for i in 0..dim {
                unsafe {
                    let v = read_f64(input.data_ptr().add(off + i));
                    ss += v * v;
                }
            }
            let inv_rms = 1.0 / ((ss / dim as f64) + eps as f64).sqrt();
            for i in 0..dim {
                unsafe {
                    let v = read_f64(input.data_ptr().add(off + i));
                    let w = read_f64(weight.data_ptr().add(i));
                    write_f64(output.data_ptr_mut().add(off + i), v * w * inv_rms);
                }
            }
        }
        Ok(())
    }

    fn rmsnorm_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let dim = *x.shape().as_slice().last().unwrap();
        let rows = x.numel() / dim;
        for row in 0..rows {
            let off = row * dim;
            let mut ss = 0.0f64;
            for i in 0..dim {
                unsafe {
                    let v = read_f64((x.data_ptr() as *const T).add(off + i));
                    ss += v * v;
                }
            }
            let inv_rms = 1.0 / ((ss / dim as f64) + eps as f64).sqrt();
            for i in 0..dim {
                unsafe {
                    let v = read_f64((x.data_ptr() as *const T).add(off + i));
                    let w = read_f64(weight.data_ptr().add(i));
                    write_f64(x.data_ptr_mut().add(off + i), v * w * inv_rms);
                }
            }
        }
        Ok(())
    }

    fn fused_add_rmsnorm<T: Dtype>(
        output: &mut Tensor<T, Self>,
        residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        // residual += input; output = rmsnorm(residual, weight, eps)
        let dim = weight.numel();
        let rows = input.numel() / dim;
        for row in 0..rows {
            let off = row * dim;
            // Step 1: residual += input
            for i in 0..dim {
                unsafe {
                    let r = read_f64(residual.data_ptr().add(off + i));
                    let inp = read_f64(input.data_ptr().add(off + i));
                    write_f64(residual.data_ptr_mut().add(off + i), r + inp);
                }
            }
            // Step 2: output = rmsnorm(residual)
            let mut ss = 0.0f64;
            for i in 0..dim {
                unsafe {
                    let v = read_f64(residual.data_ptr().add(off + i));
                    ss += v * v;
                }
            }
            let inv_rms = 1.0 / ((ss / dim as f64) + eps as f64).sqrt();
            for i in 0..dim {
                unsafe {
                    let v = read_f64(residual.data_ptr().add(off + i));
                    let w = read_f64(weight.data_ptr().add(i));
                    write_f64(output.data_ptr_mut().add(off + i), v * w * inv_rms);
                }
            }
        }
        Ok(())
    }

    fn swiglu_inplace<T: Dtype>(x: &mut Tensor<T, Self>, gate: &Tensor<T, Self>) -> OpResult<()> {
        for i in 0..x.numel() {
            unsafe {
                let v = read_f64((x.data_ptr() as *const T).add(i));
                let g = read_f64(gate.data_ptr().add(i));
                let silu = v / (1.0 + (-v).exp());
                write_f64(x.data_ptr_mut().add(i), silu * g);
            }
        }
        Ok(())
    }

    fn swiglu_packed<T: Dtype>(
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        // Naive CPU reference: out[r,d] = silu(gate_up[r,d]) * gate_up[r, inter+d]
        for r in 0..rows {
            for d in 0..inter {
                unsafe {
                    let g = read_f64((gate_up.data_ptr() as *const T).add(r * 2 * inter + d));
                    let u =
                        read_f64((gate_up.data_ptr() as *const T).add(r * 2 * inter + inter + d));
                    let silu = g / (1.0 + (-g).exp());
                    write_f64(out.data_ptr_mut().add(r * inter + d), silu * u);
                }
            }
        }
        Ok(())
    }

    fn rope_inplace<T: Dtype>(
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>,
        cos: &Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
    ) -> OpResult<()> {
        // CPU RoPE implementation
        let num_tokens = positions.numel();
        let pos_slice = unsafe { std::slice::from_raw_parts(positions.data_ptr(), num_tokens) };
        let q_dim = head_num * head_dim;
        let kv_dim = kv_head_num * head_dim;

        for t in 0..num_tokens {
            let pos = pos_slice[t] as usize;
            for h in 0..head_num {
                for i in 0..(head_dim / 2) {
                    let sin_val = unsafe { read_f64(sin.data_ptr().add(pos * head_dim + i)) };
                    let cos_val = unsafe { read_f64(cos.data_ptr().add(pos * head_dim + i)) };
                    let idx0 = t * q_dim + h * head_dim + i * 2;
                    let idx1 = idx0 + 1;
                    unsafe {
                        let q0 = read_f64(q.data_ptr().add(idx0));
                        let q1 = read_f64(q.data_ptr().add(idx1));
                        write_f64(q.data_ptr_mut().add(idx0), q0 * cos_val - q1 * sin_val);
                        write_f64(q.data_ptr_mut().add(idx1), q0 * sin_val + q1 * cos_val);
                    }
                }
            }
            for h in 0..kv_head_num {
                for i in 0..(head_dim / 2) {
                    let sin_val = unsafe { read_f64(sin.data_ptr().add(pos * head_dim + i)) };
                    let cos_val = unsafe { read_f64(cos.data_ptr().add(pos * head_dim + i)) };
                    let idx0 = t * kv_dim + h * head_dim + i * 2;
                    let idx1 = idx0 + 1;
                    unsafe {
                        let k0 = read_f64(k.data_ptr().add(idx0));
                        let k1 = read_f64(k.data_ptr().add(idx1));
                        write_f64(k.data_ptr_mut().add(idx0), k0 * cos_val - k1 * sin_val);
                        write_f64(k.data_ptr_mut().add(idx1), k0 * sin_val + k1 * cos_val);
                    }
                }
            }
        }
        Ok(())
    }

    fn attention_paged<T: Dtype>(
        q: &Tensor<T, Self>,
        k_pool: &Tensor<T, Self>,
        v_pool: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        plan: &crate::domain::batch::BatchPlan<Self>,
        _workspace: &mut Tensor<f32, Self>, // CPU ignores; no flash-decode scratch needed
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()> {
        // CPU reference: naive ragged-batch causal attention over a paged
        // KV pool. Layout:
        //   k_pool / v_pool : [num_blocks, block_size, kv_dim] contiguous
        //   block_tables    : [batch, max_blocks_per_seq] (in plan)
        // For each seq i, q has rows [cu_q_lens[i] .. cu_q_lens[i+1]); attend
        // to the first `kv_lens[i]` tokens routed through `block_tables[i]`.
        let kv_mul = head_num / kv_head_num;
        let q_dim = head_num * head_dim;
        let kv_dim = kv_head_num * head_dim;
        let block_size = plan.block_size;
        let max_blocks = plan.max_blocks_per_seq;

        let cu_q = plan.cu_q_lens.as_slice();
        let kv_lens_h = plan.kv_lens.as_slice();
        let block_tables_h = plan.block_tables.as_slice();
        let batch = plan.batch;

        // Helper: fetch K/V pool row for `(seq, token_pos)`. Returns f64.
        let fetch =
            |pool: &Tensor<T, Self>, seq: usize, pos: usize, kv_h: usize, d: usize| -> f64 {
                let logical_block = pos / block_size;
                let block_off = pos % block_size;
                let physical = block_tables_h[seq * max_blocks + logical_block] as usize;
                let row_idx = physical * block_size + block_off;
                unsafe { read_f64(pool.data_ptr().add(row_idx * kv_dim + kv_h * head_dim + d)) }
            };

        for seq in 0..batch {
            let q_start = cu_q[seq] as usize;
            let q_end = cu_q[seq + 1] as usize;
            let q_len = q_end - q_start;
            let kv_len = kv_lens_h[seq] as usize;

            // Causal: q-row r in this seq attends to KV [0..kv_len-q_len+r+1].
            let causal_shift = (kv_len as i64) - (q_len as i64);

            for h in 0..head_num {
                let kv_h = h / kv_mul;
                for r in 0..q_len {
                    let q_row_global = q_start + r;
                    let kv_upper = (causal_shift + r as i64 + 1).max(0) as usize;
                    let mut scores = vec![0.0f64; kv_upper];
                    for s in 0..kv_upper {
                        let mut dot = 0.0f64;
                        for d in 0..head_dim {
                            unsafe {
                                let qi = read_f64(
                                    q.data_ptr().add(q_row_global * q_dim + h * head_dim + d),
                                );
                                let ki = fetch(k_pool, seq, s, kv_h, d);
                                dot += qi * ki;
                            }
                        }
                        scores[s] = dot * scale as f64;
                    }
                    let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum = 0.0f64;
                    for s in scores.iter_mut() {
                        *s = (*s - max_s).exp();
                        sum += *s;
                    }
                    if sum == 0.0 {
                        sum = 1.0;
                    }
                    for s in scores.iter_mut() {
                        *s /= sum;
                    }
                    for d in 0..head_dim {
                        let mut val = 0.0f64;
                        for s in 0..kv_upper {
                            let vi = fetch(v_pool, seq, s, kv_h, d);
                            val += scores[s] * vi;
                        }
                        unsafe {
                            write_f64(
                                output
                                    .data_ptr_mut()
                                    .add(q_row_global * q_dim + h * head_dim + d),
                                val,
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn split_qkv<T: Dtype>(
        qkv: &Tensor<T, Self>,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        v: &mut Tensor<T, Self>,
        num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        let qkv_dim = q_dim + 2 * kv_dim;
        let elem = T::SIZE_BYTES;
        let src = qkv.data_ptr() as *const u8;
        let q_dst = q.data_ptr_mut() as *mut u8;
        let k_dst = k.data_ptr_mut() as *mut u8;
        let v_dst = v.data_ptr_mut() as *mut u8;
        for t in 0..num_tokens {
            unsafe {
                let row = src.add(t * qkv_dim * elem);
                std::ptr::copy_nonoverlapping(row, q_dst.add(t * q_dim * elem), q_dim * elem);
                std::ptr::copy_nonoverlapping(
                    row.add(q_dim * elem),
                    k_dst.add(t * kv_dim * elem),
                    kv_dim * elem,
                );
                std::ptr::copy_nonoverlapping(
                    row.add((q_dim + kv_dim) * elem),
                    v_dst.add(t * kv_dim * elem),
                    kv_dim * elem,
                );
            }
        }
        Ok(())
    }

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
    ) -> OpResult<()> {
        let elem = T::SIZE_BYTES;
        let cu_q = cu_q_lens.as_slice();
        let seq_pos = seq_positions.as_slice();
        let seq_lens = seq_lens_step.as_slice();
        let block_tables_h = block_tables.as_slice();
        let batch = seq_pos.len();
        for seq in 0..batch {
            let q_start = cu_q[seq] as usize;
            let q_len = seq_lens[seq] as usize;
            let dst_pos_start = seq_pos[seq] as usize;
            for t in 0..q_len {
                let dst_pos = dst_pos_start + t;
                let logical_block = dst_pos / block_size;
                let block_off = dst_pos % block_size;
                let physical = block_tables_h[seq * max_blocks_per_seq + logical_block] as usize;
                let row_idx = physical * block_size + block_off;
                unsafe {
                    let k_src_row =
                        (k_src.data_ptr() as *const u8).add((q_start + t) * kv_dim * elem);
                    let k_dst_row = (k_pool.data_ptr_mut() as *mut u8).add(row_idx * kv_dim * elem);
                    std::ptr::copy_nonoverlapping(k_src_row, k_dst_row, kv_dim * elem);
                    let v_src_row =
                        (v_src.data_ptr() as *const u8).add((q_start + t) * kv_dim * elem);
                    let v_dst_row = (v_pool.data_ptr_mut() as *mut u8).add(row_idx * kv_dim * elem);
                    std::ptr::copy_nonoverlapping(v_src_row, v_dst_row, kv_dim * elem);
                }
            }
        }
        Ok(())
    }

    fn argmax_batched<T: Dtype>(
        logits: &Tensor<T, Self>,
        cu_q_lens: &Tensor<i32, Self>,
        batch: usize,
        out_dev: &mut Tensor<i32, Self>,
        _workspace: &Tensor<f32, Self>,
        _rows: &mut Tensor<i32, Self>,
    ) -> OpResult<Vec<i32>> {
        let total_rows = if logits.shape().as_slice().len() >= 1 {
            logits.shape().as_slice()[0]
        } else {
            return Err(crate::domain::ports::OpError::Shape(
                "logits must be 2D".into(),
            ));
        };
        let vocab = logits.numel() / total_rows;
        let cu_q = cu_q_lens.as_slice();
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_dev.data_ptr_mut(), batch) };
        for seq in 0..batch {
            let last_row = (cu_q[seq + 1] - 1) as usize;
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx = 0i32;
            for i in 0..vocab {
                let val = unsafe { read_f64(logits.data_ptr().add(last_row * vocab + i)) };
                if val > max_val {
                    max_val = val;
                    max_idx = i as i32;
                }
            }
            out_slice[seq] = max_idx;
        }
        Ok(out_slice.to_vec())
    }
}

// ─── DiffusionOps for Cpu ────────────────────────────────────────────────────
// Conv / Norm / Spatial / DiT ops. Most are CPU reference or unimplemented —
// the diffusion pipeline runs on CUDA only (per project decision).

impl DiffusionOps for Cpu {
    fn conv2d<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: Option<&Tensor<T, Self>>,
        output: &mut Tensor<T, Self>,
        stride: usize,
        padding: usize,
    ) -> OpResult<()> {
        // input: [N, Cin, H, W], weight: [Cout, Cin, Kh, Kw], output: [N, Cout, Hout, Wout]
        let shape_i = input.shape().as_slice();
        let shape_w = weight.shape().as_slice();
        let shape_o = output.shape().as_slice();
        let (n, cin, h, w) = (shape_i[0], shape_i[1], shape_i[2], shape_i[3]);
        let (cout, _cin2, kh, kw) = (shape_w[0], shape_w[1], shape_w[2], shape_w[3]);
        let (ho, wo) = (shape_o[2], shape_o[3]);

        for batch in 0..n {
            for oc in 0..cout {
                for oh in 0..ho {
                    for ow in 0..wo {
                        let mut sum = 0.0f64;
                        for ic in 0..cin {
                            for fh in 0..kh {
                                for fw in 0..kw {
                                    let ih = oh * stride + fh;
                                    let iw = ow * stride + fw;
                                    let ih = ih as isize - padding as isize;
                                    let iw = iw as isize - padding as isize;
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        let i_idx = batch * cin * h * w
                                            + ic * h * w
                                            + ih as usize * w
                                            + iw as usize;
                                        let w_idx =
                                            oc * cin * kh * kw + ic * kh * kw + fh * kw + fw;
                                        unsafe {
                                            sum += read_f64(input.data_ptr().add(i_idx))
                                                * read_f64(weight.data_ptr().add(w_idx));
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(b) = bias {
                            sum += unsafe { read_f64(b.data_ptr().add(oc)) };
                        }
                        let o_idx = batch * cout * ho * wo + oc * ho * wo + oh * wo + ow;
                        unsafe {
                            write_f64(output.data_ptr_mut().add(o_idx), sum);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn groupnorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()> {
        // input: [N, C, ...spatial], weight/bias: [C]
        let shape = input.shape().as_slice();
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape[2..].iter().product();
        let group_size = c / num_groups;

        for batch in 0..n {
            for g in 0..num_groups {
                let ch_start = g * group_size;
                let count = group_size * spatial;
                // Compute mean + variance
                let mut sum = 0.0f64;
                for ch in 0..group_size {
                    for s in 0..spatial {
                        let idx = batch * c * spatial + (ch_start + ch) * spatial + s;
                        sum += unsafe { read_f64(input.data_ptr().add(idx)) };
                    }
                }
                let mean = sum / count as f64;
                let mut var_sum = 0.0f64;
                for ch in 0..group_size {
                    for s in 0..spatial {
                        let idx = batch * c * spatial + (ch_start + ch) * spatial + s;
                        let v = unsafe { read_f64(input.data_ptr().add(idx)) } - mean;
                        var_sum += v * v;
                    }
                }
                let inv_std = 1.0 / ((var_sum / count as f64) + eps as f64).sqrt();
                // Normalize + affine
                for ch in 0..group_size {
                    let c_idx = ch_start + ch;
                    let w = unsafe { read_f64(weight.data_ptr().add(c_idx)) };
                    let b = unsafe { read_f64(bias.data_ptr().add(c_idx)) };
                    for s in 0..spatial {
                        let idx = batch * c * spatial + c_idx * spatial + s;
                        let v = unsafe { read_f64(input.data_ptr().add(idx)) };
                        let normed = (v - mean) * inv_std * w + b;
                        unsafe {
                            write_f64(output.data_ptr_mut().add(idx), normed);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn groupnorm_silu<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()> {
        // Same as groupnorm but apply SiLU after
        Self::groupnorm(input, weight, bias, output, num_groups, eps)?;
        let n = output.numel();
        for i in 0..n {
            unsafe {
                let v = read_f64((output.data_ptr() as *const T).add(i));
                write_f64(output.data_ptr_mut().add(i), v / (1.0 + (-v).exp()));
            }
        }
        Ok(())
    }

    fn layernorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let dim = *input.shape().as_slice().last().unwrap();
        let rows = input.numel() / dim;
        for row in 0..rows {
            let off = row * dim;
            // Compute mean
            let mut sum = 0.0f64;
            for i in 0..dim {
                sum += unsafe { read_f64(input.data_ptr().add(off + i)) };
            }
            let mean = sum / dim as f64;
            // Compute variance
            let mut var = 0.0f64;
            for i in 0..dim {
                let v = unsafe { read_f64(input.data_ptr().add(off + i)) } - mean;
                var += v * v;
            }
            let inv_std = 1.0 / ((var / dim as f64) + eps as f64).sqrt();
            // Normalize + affine
            for i in 0..dim {
                unsafe {
                    let v = read_f64(input.data_ptr().add(off + i));
                    let w = read_f64(weight.data_ptr().add(i));
                    let b = read_f64(bias.data_ptr().add(i));
                    write_f64(
                        output.data_ptr_mut().add(off + i),
                        (v - mean) * inv_std * w + b,
                    );
                }
            }
        }
        Ok(())
    }

    fn upsample_nearest_2x<T: Dtype>(
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        // input: [N, C, H, W] → output: [N, C, 2H, 2W]
        let shape = input.shape().as_slice();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        for batch in 0..n {
            for ch in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let val = unsafe {
                            read_f64(
                                input
                                    .data_ptr()
                                    .add(batch * c * h * w + ch * h * w + y * w + x),
                            )
                        };
                        // Write to 4 output pixels
                        for dy in 0..2usize {
                            for dx in 0..2usize {
                                let oy = y * 2 + dy;
                                let ox = x * 2 + dx;
                                let o_idx = batch * c * (h * 2) * (w * 2)
                                    + ch * (h * 2) * (w * 2)
                                    + oy * (w * 2)
                                    + ox;
                                unsafe {
                                    write_f64(output.data_ptr_mut().add(o_idx), val);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn sdpa<T: Dtype>(
        q: &Tensor<T, Self>,
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()> {
        // Self-attention (no KV cache): same as LLM attention but no seq_starts
        let seq_len = q.numel() / (num_heads * head_dim);
        let kv_mul = num_heads / num_kv_heads;

        for h in 0..num_heads {
            let kv_h = h / kv_mul;
            for t in 0..seq_len {
                let mut scores = vec![0.0f64; seq_len];
                for s in 0..seq_len {
                    let mut dot = 0.0f64;
                    for d in 0..head_dim {
                        unsafe {
                            let qi = read_f64(
                                q.data_ptr()
                                    .add(t * num_heads * head_dim + h * head_dim + d),
                            );
                            let ki = read_f64(
                                k.data_ptr()
                                    .add(s * num_kv_heads * head_dim + kv_h * head_dim + d),
                            );
                            dot += qi * ki;
                        }
                    }
                    scores[s] = dot * scale as f64;
                }
                // Softmax
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                for s in scores.iter_mut() {
                    *s /= sum;
                }
                // Weighted sum
                for d in 0..head_dim {
                    let mut val = 0.0f64;
                    for s in 0..seq_len {
                        unsafe {
                            let vi = read_f64(
                                v.data_ptr()
                                    .add(s * num_kv_heads * head_dim + kv_h * head_dim + d),
                            );
                            val += scores[s] * vi;
                        }
                    }
                    unsafe {
                        write_f64(
                            output
                                .data_ptr_mut()
                                .add(t * num_heads * head_dim + h * head_dim + d),
                            val,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn sdpa_masked<T: Dtype>(
        _q: &Tensor<T, Self>,
        _k: &Tensor<T, Self>,
        _v: &Tensor<T, Self>,
        _output: &mut Tensor<T, Self>,
        _mask: &Tensor<T, Self>,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _scale: f32,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "sdpa_masked"))
    }

    // ─── Diffusion-only ops — not maintained for CPU ───
    // The diffusion pipeline runs on CUDA only (per project decision).

    fn apply_rope_interleaved<T: Dtype>(
        _x: &mut Tensor<T, Self>,
        _cos: &Tensor<f32, Self>,
        _sin: &Tensor<f32, Self>,
        _head_dim: usize,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "apply_rope_interleaved"))
    }

    fn pad_with_token<T: Dtype>(
        _src: &Tensor<T, Self>,
        _pad_token: &Tensor<T, Self>,
        _dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "pad_with_token"))
    }

    fn pad_last_row<T: Dtype>(_src: &Tensor<T, Self>, _dst: &mut Tensor<T, Self>) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "pad_last_row"))
    }

    fn overwrite_pad_tokens_inplace<T: Dtype>(
        _dst: &mut Tensor<T, Self>,
        _pad_token: &Tensor<T, Self>,
        _keep_prefix: usize,
    ) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "overwrite_pad_tokens_inplace"))
    }

    fn silu_inplace_diff<T: Dtype>(_x: &mut Tensor<T, Self>) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "silu_inplace_diff"))
    }

    fn tanh_inplace<T: Dtype>(_x: &mut Tensor<T, Self>) -> OpResult<()> {
        Err(OpError::unsupported("cpu", "tanh_inplace"))
    }
}

// ─── Generic numeric helpers ─────────────────────────────────────────────────

#[inline]
unsafe fn read_f64<T: Dtype>(ptr: *const T) -> f64 {
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, T::SIZE_BYTES) };
    match T::DATA_TYPE {
        DataType::F32 => f64::from(f32::from_le_bytes(bytes[..4].try_into().unwrap())),
        DataType::BF16 => f64::from(bf16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32()),
        DataType::F16 => f64::from(f16::from_le_bytes(bytes[..2].try_into().unwrap()).to_f32()),
        _ => 0.0,
    }
}

#[inline]
unsafe fn write_f64<T: Dtype>(ptr: *mut T, val: f64) {
    let dst = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, T::SIZE_BYTES) };
    match T::DATA_TYPE {
        DataType::F32 => dst.copy_from_slice(&(val as f32).to_le_bytes()),
        DataType::BF16 => dst.copy_from_slice(&bf16::from_f64(val).to_le_bytes()),
        DataType::F16 => dst.copy_from_slice(&f16::from_f64(val).to_le_bytes()),
        _ => {}
    }
}

// ─── Validation helpers ──────────────────────────────────────────────────────

fn check_contiguous3<T: Dtype, D: MemoryPort>(
    a: &Tensor<T, D>,
    b: &Tensor<T, D>,
    c: &Tensor<T, D>,
) -> OpResult<()> {
    if !a.is_contiguous() || !b.is_contiguous() || !c.is_contiguous() {
        return Err(OpError::NotContiguous(*a.shape()));
    }
    Ok(())
}

fn check_numel3<T: Dtype, D: MemoryPort>(
    a: &Tensor<T, D>,
    b: &Tensor<T, D>,
    c: &Tensor<T, D>,
) -> OpResult<()> {
    if a.numel() != b.numel() || a.numel() != c.numel() {
        return Err(OpError::Shape(format!(
            "numel mismatch: {} {} {}",
            a.numel(),
            b.numel(),
            c.numel()
        )));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_zeros_and_slice() {
        let t = Tensor::<f32, Cpu>::zeros_cpu([2, 3]);
        assert_eq!(t.shape().as_slice(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tensor_from_slice_roundtrip() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::<f32, Cpu>::from_slice(&data, [4]);
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn op_add_f32() {
        let a = Tensor::<f32, Cpu>::from_slice(&[1.0, 2.0, 3.0], [3]);
        let b = Tensor::<f32, Cpu>::from_slice(&[10.0, 20.0, 30.0], [3]);
        let mut dst = Tensor::<f32, Cpu>::zeros_cpu([3]);
        Cpu::add(&a, &b, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn op_rmsnorm_unit_weight() {
        let dim = 64;
        let input_data: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1 + 1.0).collect();
        let weight_data = vec![1.0f32; dim];
        let input = Tensor::<f32, Cpu>::from_slice(&input_data, [1, dim]);
        let weight = Tensor::<f32, Cpu>::from_slice(&weight_data, [dim]);
        let mut output = Tensor::<f32, Cpu>::zeros_cpu([1, dim]);
        Cpu::rmsnorm(&input, &weight, &mut output, 1e-6).unwrap();
        let mean_sq: f32 = output.as_slice().iter().map(|x| x * x).sum::<f32>() / dim as f32;
        assert!((mean_sq - 1.0).abs() < 0.01);
    }

    #[test]
    fn op_matmul_identity() {
        let input = Tensor::<f32, Cpu>::from_slice(&[1.0, 0.0, 0.0, 1.0], [2, 2]);
        let weight = Tensor::<f32, Cpu>::from_slice(&[1.0, 0.0, 0.0, 1.0], [2, 2]);
        let mut output = Tensor::<f32, Cpu>::zeros_cpu([2, 2]);
        Cpu::matmul(&input, &weight, &mut output).unwrap();
        assert_eq!(output.as_slice(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn op_softmax_sums_to_one() {
        let input = Tensor::<f32, Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], [1, 4]);
        let mut output = Tensor::<f32, Cpu>::zeros_cpu([1, 4]);
        Cpu::softmax(&input, &mut output).unwrap();
        let sum: f32 = output.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn op_silu() {
        let mut x = Tensor::<f32, Cpu>::from_slice(&[0.0, 1.0, -1.0], [3]);
        Cpu::silu_inplace(&mut x).unwrap();
        let r = x.as_slice();
        assert!((r[0] - 0.0).abs() < 1e-6);
        assert!((r[1] - 0.7311).abs() < 0.001);
    }

    #[test]
    fn op_embedding() {
        let table =
            Tensor::<f32, Cpu>::from_slice(&[0.1, 0.2, 0.3, 1.1, 1.2, 1.3, 2.1, 2.2, 2.3], [3, 3]);
        let indices = Tensor::<i32, Cpu>::from_slice(&[2, 0, 1], [3]);
        let mut output = Tensor::<f32, Cpu>::zeros_cpu([3, 3]);
        Cpu::embedding(&table, &indices, &mut output).unwrap();
        assert_eq!(&output.as_slice()[..3], &[2.1, 2.2, 2.3]);
        assert_eq!(&output.as_slice()[3..6], &[0.1, 0.2, 0.3]);
    }

    // ─── Diffusion op tests ─────────────────────────────────────────

    #[test]
    fn op_conv2d_identity_kernel() {
        // 1×1 conv with identity-like weight acts as a linear per-pixel
        // input: [1, 1, 3, 3], weight: [1, 1, 1, 1] = 2.0 → output = input * 2
        let input_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let input = Tensor::<f32, Cpu>::from_slice(&input_data, Shape::from_slice(&[1, 1, 3, 3]));
        let weight = Tensor::<f32, Cpu>::from_slice(&[2.0f32], Shape::from_slice(&[1, 1, 1, 1]));
        let mut output = Tensor::<f32, Cpu>::zeros_cpu(Shape::from_slice(&[1, 1, 3, 3]));
        Cpu::conv2d(&input, &weight, None, &mut output, 1, 0).unwrap();
        let out = output.as_slice();
        for i in 0..9 {
            assert!((out[i] - (i + 1) as f32 * 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn op_groupnorm_single_group() {
        // GroupNorm with 1 group = LayerNorm over channels
        let input =
            Tensor::<f32, Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[1, 4, 1, 1]));
        let weight = Tensor::<f32, Cpu>::from_slice(&[1.0; 4], Shape::from_slice(&[4]));
        let bias = Tensor::<f32, Cpu>::from_slice(&[0.0; 4], Shape::from_slice(&[4]));
        let mut output = Tensor::<f32, Cpu>::zeros_cpu(Shape::from_slice(&[1, 4, 1, 1]));
        Cpu::groupnorm(&input, &weight, &bias, &mut output, 1, 1e-5).unwrap();
        // Output should be normalized: mean≈0, var≈1
        let out = output.as_slice();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean = {}", mean);
    }

    #[test]
    fn op_layernorm_normalized() {
        let input =
            Tensor::<f32, Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[1, 4]));
        let weight = Tensor::<f32, Cpu>::from_slice(&[1.0; 4], Shape::from_slice(&[4]));
        let bias = Tensor::<f32, Cpu>::from_slice(&[0.0; 4], Shape::from_slice(&[4]));
        let mut output = Tensor::<f32, Cpu>::zeros_cpu(Shape::from_slice(&[1, 4]));
        Cpu::layernorm(&input, &weight, &bias, &mut output, 1e-5).unwrap();
        let out = output.as_slice();
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        let var: f32 = out.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean = {}", mean);
        assert!((var - 1.0).abs() < 0.01, "var = {}", var);
    }

    #[test]
    fn op_upsample_nearest_2x() {
        // [1, 1, 2, 2] → [1, 1, 4, 4]
        let input =
            Tensor::<f32, Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], Shape::from_slice(&[1, 1, 2, 2]));
        let mut output = Tensor::<f32, Cpu>::zeros_cpu(Shape::from_slice(&[1, 1, 4, 4]));
        Cpu::upsample_nearest_2x(&input, &mut output).unwrap();
        let out = output.as_slice();
        // Top-left 2×2 should all be 1.0
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[4], 1.0);
        assert_eq!(out[5], 1.0);
        // Top-right 2×2 should all be 2.0
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 2.0);
    }

    #[test]
    fn op_ewise_mul() {
        let a = Tensor::<f32, Cpu>::from_slice(&[1.0, 2.0, 3.0, 4.0], [4]);
        let b = Tensor::<f32, Cpu>::from_slice(&[2.0, 3.0, 4.0, 5.0], [4]);
        let mut dst = Tensor::<f32, Cpu>::zeros_cpu([4]);
        Cpu::ewise_mul(&a, &b, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), &[2.0, 6.0, 12.0, 20.0]);
    }
}
