//! CPU infrastructure adapter.
//!
//! Implements `Device`, `HostDevice`, `MemoryPort`, and reference CPU ops.
//! Some GPU-oriented ops intentionally return `OpError::Unsupported`.

use std::alloc::Layout;
use std::ptr::NonNull;

use infer_core::ports::{
    AllocError, Allocator, CollectiveOps, CommAxis, CoreOps, Device, DiffusionOps, HostDevice,
    MemoryPort, OpError, OpResult, ReduceOp, VocabOps,
};
use infer_core::tensor::Tensor;
use infer_core::types::{Dtype, Shape};

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

impl infer_core::exec::ExecDevice for Cpu {
    type Scope = infer_core::exec::HostScope<Self>;
}

impl infer_core::exec::ExecHostDevice for Cpu {}

infer_core::impl_math_ops_via_core_ops!(Cpu);

impl infer_core::ports::FusedOps for Cpu {}

// Decode-pipeline port: the host reference defaults ARE the CPU implementation.
impl infer_core::ports::DecodePipelineOps for Cpu {}

fn require_single_rank(
    scope: &infer_core::exec::HostScope<Cpu>,
    axis: CommAxis,
    op: &str,
) -> OpResult<()> {
    let size = infer_core::exec::ExecScope::topology(scope).group_size(axis);
    if size != 1 {
        return Err(OpError::Kernel(format!(
            "CPU {op} requires a single-rank {axis:?} group, got size {size}"
        )));
    }
    Ok(())
}

impl CollectiveOps for Cpu {
    type Comm = infer_core::ports::collective::SingleRankComm;

    fn comm(_scope: &Self::Scope, _axis: CommAxis) -> Option<&Self::Comm> {
        None
    }

    fn all_reduce<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        _op: ReduceOp,
        _buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_single_rank(scope, axis, "all_reduce")
    }

    fn all_gather<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        _dim: usize,
        shard: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_single_rank(scope, axis, "all_gather")?;
        out.copy_from(shard)
    }

    fn reduce_scatter<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        _op: ReduceOp,
        _dim: usize,
        buf: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_single_rank(scope, axis, "reduce_scatter")?;
        out.copy_from(buf)
    }

    fn broadcast<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        root: usize,
        _buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        require_single_rank(scope, axis, "broadcast")?;
        if root != 0 {
            return Err(OpError::Shape(format!(
                "single-rank broadcast root must be 0, got {root}"
            )));
        }
        Ok(())
    }

    fn send<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _peer: usize,
        buf: &Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported(buf.device().name(), "send"))
    }

    fn recv<T: Dtype>(
        _scope: &Self::Scope,
        _axis: CommAxis,
        _peer: usize,
        buf: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        Err(OpError::unsupported(buf.device().name(), "recv"))
    }

    fn all_to_all<T: Dtype>(
        scope: &Self::Scope,
        axis: CommAxis,
        send_chunks: &[Tensor<T, Self>],
        recv_chunks: &mut [Tensor<T, Self>],
    ) -> OpResult<()> {
        require_single_rank(scope, axis, "all_to_all")?;
        if send_chunks.len() != recv_chunks.len() {
            return Err(OpError::Shape(format!(
                "all_to_all: send_chunks={} recv_chunks={}",
                send_chunks.len(),
                recv_chunks.len()
            )));
        }
        for (src, dst) in send_chunks.iter().zip(recv_chunks.iter_mut()) {
            dst.copy_from(src)?;
        }
        Ok(())
    }

    fn barrier(scope: &Self::Scope, axis: CommAxis) -> OpResult<()> {
        require_single_rank(scope, axis, "barrier")
    }
}

impl VocabOps for Cpu {
    fn vocab_embedding<T: Dtype>(
        _scope: &Self::Scope,
        table: &Tensor<T, Self>,
        global_indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
        vocab_start: usize,
        global_vocab_size: usize,
    ) -> OpResult<()> {
        let table_shape = table.shape().as_slice();
        if table_shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "vocab_embedding table must be rank 2, got {:?}",
                table_shape
            )));
        }
        let (local_vocab, dim) = (table_shape[0], table_shape[1]);
        if local_vocab == 0 || dim == 0 {
            return Err(OpError::Shape(format!(
                "vocab_embedding table dimensions must be non-zero, got {:?}",
                table_shape
            )));
        }
        let vocab_end = vocab_start
            .checked_add(local_vocab)
            .ok_or_else(|| OpError::Shape("vocab_embedding shard range overflows".into()))?;
        if vocab_end > global_vocab_size {
            return Err(OpError::Shape(format!(
                "vocab_embedding shard [{vocab_start}, {vocab_end}) exceeds global vocabulary {global_vocab_size}"
            )));
        }
        if output.shape().as_slice() != [global_indices.numel(), dim] {
            return Err(OpError::Shape(format!(
                "vocab_embedding output shape {:?}, expected [{}, {}]",
                output.shape().as_slice(),
                global_indices.numel(),
                dim
            )));
        }

        let indices = unsafe {
            std::slice::from_raw_parts(global_indices.data_ptr(), global_indices.numel())
        };
        for (row, &raw) in indices.iter().enumerate() {
            if raw < 0 || raw as usize >= global_vocab_size {
                return Err(OpError::Shape(format!(
                    "vocab_embedding token id {raw} at position {row} outside [0, {global_vocab_size})"
                )));
            }
            let dst = unsafe { output.data_ptr_mut().add(row * dim) };
            let token = raw as usize;
            if (vocab_start..vocab_end).contains(&token) {
                let local = token - vocab_start;
                unsafe {
                    std::ptr::copy_nonoverlapping(table.data_ptr().add(local * dim), dst, dim);
                }
            } else {
                unsafe {
                    std::ptr::write_bytes(dst, 0, dim);
                }
            }
        }
        Ok(())
    }
}

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

// ─── Cpu Allocator (used by VAE pool) ────────────────────────────────────────

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

/// CPU tensor construction helpers. An *extension trait* rather than inherent
/// methods, because `Tensor` lives in `infer-core`: a crate may add inherent
/// methods only to types it defines, but it may implement its own trait for a
/// foreign type. Bring `CpuTensorExt` into scope to call
/// `Tensor::<T, Cpu>::zeros_cpu(..)` / `::from_slice(..)`.
pub trait CpuTensorExt<T: Dtype>: Sized {
    fn zeros_cpu(shape: impl Into<Shape>) -> Tensor<T, Cpu>;
    fn from_slice(data: &[T], shape: impl Into<Shape>) -> Tensor<T, Cpu>;
}

impl<T: Dtype> CpuTensorExt<T> for Tensor<T, Cpu> {
    /// CPU-only convenience: allocate a contiguous, zero-initialized tensor.
    fn zeros_cpu(shape: impl Into<Shape>) -> Tensor<T, Cpu> {
        Tensor::<T, Cpu>::zeros(shape, &Cpu).expect("CPU alloc cannot fail")
    }

    /// Create from existing host data (copies bytes into a fresh allocation).
    fn from_slice(data: &[T], shape: impl Into<Shape>) -> Tensor<T, Cpu> {
        Tensor::<T, Cpu>::from_host_slice(data, shape, &Cpu)
            .expect("CPU from_host_slice cannot fail")
    }
}
// `as_slice` / `as_slice_mut` are now host-generic inherent methods on
// `Tensor<T, D: HostDevice>` in `domain/tensor.rs` (they apply to any
// host-addressable backend, not just Cpu), so they move with Tensor when the
// foundation collapses into infer-core.

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
        check_contiguous3(a, b, dst)?;
        check_numel3(a, b, dst)?;
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
        scheme: &infer_core::dtype::quant::QuantScheme,
    ) -> OpResult<()> {
        // CPU reference: dequantize weight per group, then matmul. The weight is
        // stored already-expanded `[N, K]` here (not bit-packed), so the packing
        // factor is 1 for this reference path; only the group size matters.
        let group_size = scheme.group;
        let m = input.shape().as_slice()[0];
        let k = input.shape().as_slice()[1];
        let n = weight.shape().as_slice()[0];
        if !input.is_contiguous() || !weight.is_contiguous() || !output.is_contiguous() {
            return Err(OpError::NotContiguous(*input.shape()));
        }
        if group_size == 0 || !k.is_multiple_of(group_size) {
            return Err(OpError::Shape(format!(
                "matmul_quant: k {} not divisible by group_size {}",
                k, group_size
            )));
        }
        let groups = k / group_size;
        // Bound checks against the actual backing storage before any raw reads.
        if weight.numel() < n * k
            || input.numel() < m * k
            || output.numel() < m * n
            || scales.numel() < n * groups
        {
            return Err(OpError::Shape(format!(
                "matmul_quant: operand smaller than declared m={} n={} k={} groups={} (input={}, weight={}, scales={}, output={})",
                m,
                n,
                k,
                groups,
                input.numel(),
                weight.numel(),
                scales.numel(),
                output.numel()
            )));
        }
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let group = p / group_size;
                    let scale = unsafe { read_f64(scales.data_ptr().add(j * groups + group)) };
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
                let v = read_f64(x.data_ptr().add(i));
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
                    let v = read_f64(output.data_ptr().add(off + i));
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
        let vocab = table.shape().as_slice()[0];
        let seq_len = indices.numel();
        let idx_slice = unsafe { std::slice::from_raw_parts(indices.data_ptr(), seq_len) };
        for (i, &raw) in idx_slice.iter().enumerate() {
            if raw < 0 || (raw as usize) >= vocab {
                return Err(OpError::Shape(format!(
                    "embedding: index {} at position {} out of range [0, {})",
                    raw, i, vocab
                )));
            }
            let idx = raw as usize;
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
                let v = read_f64(x.data_ptr().add(i));
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
                    let v = read_f64(x.data_ptr().add(row * dim + i));
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
        src: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        dst_cols: usize,
    ) -> OpResult<()> {
        if col_offset + dst_cols > total_cols {
            return Err(OpError::Shape(format!(
                "split_cols: col_offset {} + dst_cols {} > total_cols {}",
                col_offset, dst_cols, total_cols
            )));
        }
        if !src.is_contiguous() || !dst.is_contiguous() {
            return Err(OpError::NotContiguous(*src.shape()));
        }
        // The declared [rows, total_cols] / [rows, dst_cols] logical shapes must
        // fit inside the backing tensors; otherwise the pointer arithmetic below
        // (`r * total_cols + col_offset`, `r * dst_cols`) walks out of bounds.
        if rows * total_cols > src.numel() || rows * dst_cols > dst.numel() {
            return Err(OpError::Shape(format!(
                "split_cols: declared shape exceeds storage (rows={}, total_cols={}, dst_cols={}, src_numel={}, dst_numel={})",
                rows,
                total_cols,
                dst_cols,
                src.numel(),
                dst.numel()
            )));
        }
        let src_ptr = src.data_ptr();
        let dst_ptr = dst.data_ptr_mut();
        for r in 0..rows {
            // SAFETY: contiguous [rows, total_cols] src and [rows, dst_cols] dst;
            // the bound check above keeps the read window inside each src row.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_ptr.add(r * total_cols + col_offset),
                    dst_ptr.add(r * dst_cols),
                    dst_cols,
                );
            }
        }
        Ok(())
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
                    let v = read_f64(x.data_ptr().add(off + i));
                    ss += v * v;
                }
            }
            let inv_rms = 1.0 / ((ss / dim as f64) + eps as f64).sqrt();
            for i in 0..dim {
                unsafe {
                    let v = read_f64(x.data_ptr().add(off + i));
                    let w = read_f64(weight.data_ptr().add(i));
                    write_f64(x.data_ptr_mut().add(off + i), v * w * inv_rms);
                }
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
                    let g = read_f64(gate_up.data_ptr().add(r * 2 * inter + d));
                    let u = read_f64(gate_up.data_ptr().add(r * 2 * inter + inter + d));
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
        if head_num == 0 || kv_head_num == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(OpError::Shape(format!(
                "rope_inplace: invalid heads head_num={head_num} kv_head_num={kv_head_num} head_dim={head_dim}"
            )));
        }
        let q_dim = head_num * head_dim;
        let kv_dim = kv_head_num * head_dim;
        let q_shape = q.shape().as_slice();
        let k_shape = k.shape().as_slice();
        if q_shape.len() != 2 || q_shape[1] != q_dim {
            return Err(OpError::Shape(format!(
                "rope_inplace: q shape {q_shape:?} is not [tokens, {q_dim}]"
            )));
        }
        let num_tokens = q_shape[0];
        if k_shape != [num_tokens, kv_dim] {
            return Err(OpError::Shape(format!(
                "rope_inplace: k shape {k_shape:?} is not [{num_tokens}, {kv_dim}]"
            )));
        }
        if !q.is_contiguous()
            || !k.is_contiguous()
            || !positions.is_contiguous()
            || !sin.is_contiguous()
            || !cos.is_contiguous()
        {
            return Err(OpError::NotContiguous(*q.shape()));
        }
        if positions.numel() < num_tokens {
            return Err(OpError::Shape(format!(
                "rope_inplace: positions has {} entries for {num_tokens} active tokens",
                positions.numel()
            )));
        }
        let sin_shape = sin.shape().as_slice();
        let cos_shape = cos.shape().as_slice();
        if sin_shape.len() != 2
            || cos_shape.len() != 2
            || sin_shape[1] != head_dim
            || cos_shape[1] != head_dim
        {
            return Err(OpError::Shape(format!(
                "rope_inplace: sin/cos shapes {sin_shape:?}/{cos_shape:?} must have width {head_dim}"
            )));
        }
        let cache_rows = sin_shape[0].min(cos_shape[0]);
        // The index tensors are capacity-sized; only Q's active row count is
        // valid for this step. Iterating positions.numel() would write past Q/K.
        let pos_slice = unsafe { std::slice::from_raw_parts(positions.data_ptr(), num_tokens) };

        for (t, &raw_pos) in pos_slice.iter().enumerate() {
            if raw_pos < 0 || raw_pos as usize >= cache_rows {
                return Err(OpError::Shape(format!(
                    "rope_inplace: position {raw_pos} at token {t} is outside [0, {cache_rows})"
                )));
            }
            let pos = raw_pos as usize;
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
                let v = read_f64(output.data_ptr().add(i));
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
                for (s, score) in scores.iter_mut().enumerate() {
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
                    *score = dot * scale as f64;
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
                    for (s, &score) in scores.iter().enumerate() {
                        unsafe {
                            let vi = read_f64(
                                v.data_ptr()
                                    .add(s * num_kv_heads * head_dim + kv_h * head_dim + d),
                            );
                            val += score * vi;
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

// Scalar conversion is a capability of `Dtype`, so the CPU reference backend
// stays generic and never reinterprets a `T` pointer through a runtime dtype tag.
#[inline]
unsafe fn read_f64<T: Dtype>(ptr: *const T) -> f64 {
    T::read_f64(unsafe { &*ptr })
}

#[inline]
unsafe fn write_f64<T: Dtype>(ptr: *mut T, val: f64) {
    unsafe { *ptr = T::write_f64(val) }
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
    use infer_core::storage::Storage;

    // Relocated from infer-core's storage.rs: these exercise Storage against a
    // concrete backend (Cpu), which lives here in the worker, not in infer-core.
    #[test]
    fn storage_alloc_and_drop_is_safe() {
        let s = Storage::alloc(&Cpu, 1024).unwrap();
        assert_eq!(s.size(), 1024);
        drop(s);
    }

    #[test]
    fn storage_arc_clone_shares_storage() {
        let s = Storage::alloc(&Cpu, 64).unwrap();
        let s2 = std::sync::Arc::clone(&s);
        assert_eq!(s.ptr(), s2.ptr());
        assert_eq!(std::sync::Arc::strong_count(&s), 2);
        drop(s);
        assert_eq!(std::sync::Arc::strong_count(&s2), 1);
    }

    #[test]
    fn storage_zero_size_alloc_is_safe() {
        let s = Storage::alloc(&Cpu, 0).unwrap();
        assert_eq!(s.size(), 0);
    }

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
    fn read_write_f64_handles_integer_dtypes() {
        // Scalar conversion must preserve integer types without a backend-local
        // dtype dispatch table.
        for v in [0i32, 7, -13, 12345] {
            let mut cell = v;
            let got = unsafe { read_f64(&cell as *const i32) };
            assert_eq!(got, v as f64, "i32 read_f64 must not zero {}", v);
            unsafe { write_f64(&mut cell as *mut i32, 99.0) };
            assert_eq!(cell, 99, "i32 write_f64 must round-trip");
        }
        let mut b = -5i8;
        assert_eq!(unsafe { read_f64(&b as *const i8) }, -5.0);
        unsafe { write_f64(&mut b as *mut i8, 42.0) };
        assert_eq!(b, 42);
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

    #[test]
    fn vocab_embedding_masks_tokens_owned_by_other_ranks() {
        let scope = infer_core::exec::HostScope::new(Cpu);
        let table =
            Tensor::<f32, Cpu>::from_slice(&[20.0, 21.0, 30.0, 31.0], Shape::from_slice(&[2, 2]));
        let indices = Tensor::<i32, Cpu>::from_slice(&[0, 2, 3, 1], [4]);
        let mut output = Tensor::<f32, Cpu>::zeros_cpu([4, 2]);

        <Cpu as VocabOps>::vocab_embedding(&scope, &table, &indices, &mut output, 2, 4).unwrap();

        assert_eq!(
            output.as_slice(),
            &[0.0, 0.0, 20.0, 21.0, 30.0, 31.0, 0.0, 0.0]
        );
    }

    #[test]
    fn rope_uses_active_q_rows_not_position_capacity() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut q = Tensor::<f32, Cpu>::from_slice(&values, [2, 4]);
        let mut k = Tensor::<f32, Cpu>::from_slice(&values, [2, 4]);
        let sin = Tensor::<f32, Cpu>::from_slice(&[0.0; 16], [4, 4]);
        let cos = Tensor::<f32, Cpu>::from_slice(&[1.0; 16], [4, 4]);
        // Runtime index buffers are capacity-sized. Only the first q.shape()[0]
        // positions describe active rows in this step.
        let positions = Tensor::<i32, Cpu>::from_slice(&[0, 1, 3, 3, 3, 3], [6]);

        Cpu::rope_inplace(&mut q, &mut k, &sin, &cos, &positions, 1, 1, 4).unwrap();

        assert_eq!(q.as_slice(), &values);
        assert_eq!(k.as_slice(), &values);
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
        for (i, &actual) in out.iter().enumerate().take(9) {
            assert!((actual - (i + 1) as f32 * 2.0).abs() < 1e-5);
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
