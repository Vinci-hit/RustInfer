//! CUDA infrastructure adapter.
//!
//! Implements `Device`, `MemoryPort`, and `OpBackend` for `Cuda`.
//! Contains: FFI bindings, CudaConfig (handles), and kernel dispatch wrappers.

pub mod config;
pub mod device_utils;
pub mod error;
pub mod ffi;
mod nccl;
// Raw kernel launch wrappers are an implementation detail. Keeping this module
// private prevents external callers from manufacturing invalid CUDA streams or
// device pointers; the safe backend traits below are the supported API.
mod kernels;

pub use config::{CudaConfig, CudaMemoryPlan, CudaWorkspace, GraphSlot};
pub use error::CudaError;
pub use nccl::{NCCL_UNIQUE_ID_BYTES, NcclCommunicator, NcclUniqueId};

use std::ptr::NonNull;
use std::sync::Arc;

use infer_core::ports::{CoreOps, Device, DiffusionOps, MemoryPort, OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::Dtype;

/// CUDA device — carries device_id + shared CudaConfig (handles, stream).
#[derive(Debug, Clone)]
pub struct Cuda {
    pub device_id: i32,
    pub config: Arc<CudaConfig>,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaStream(pub ffi::cudaStream_t);

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl infer_core::exec::Stream for CudaStream {}

#[derive(Debug)]
pub struct CudaScope {
    device: Cuda,
    stream: CudaStream,
    rank: infer_core::exec::Rank,
    topology: infer_core::exec::TopologyShape,
    tp_comm: Option<Arc<NcclCommunicator>>,
    quant_tier: infer_core::exec::QuantTier,
    workspace: infer_core::exec::Workspace<Cuda>,
}

impl CudaScope {
    pub fn new(device: Cuda) -> Self {
        let stream = CudaStream(device.config.stream);
        let kernel_workspace = device.config.kernel_workspace();
        let workspace = infer_core::exec::Workspace::from_raw(
            NonNull::new(kernel_workspace.ptr() as *mut u8),
            kernel_workspace.size(),
        );
        Self {
            device,
            stream,
            rank: infer_core::exec::Rank::SINGLE,
            topology: infer_core::exec::TopologyShape::SINGLE,
            tp_comm: None,
            quant_tier: infer_core::exec::QuantTier::None,
            workspace,
        }
    }

    pub fn with_topology(mut self, topology: infer_core::exec::TopologyShape) -> OpResult<Self> {
        let world_rank = topology.world_rank()?;
        if let Some(comm) = &self.tp_comm
            && comm.rank_pair() != topology.tp
        {
            return Err(OpError::Shape(format!(
                "TP communicator rank {}/{} does not match requested topology {}/{}",
                comm.rank_pair().rank,
                comm.rank_pair().size,
                topology.tp.rank,
                topology.tp.size
            )));
        }
        self.rank = infer_core::exec::Rank {
            tp_rank: topology.tp.rank,
            pp_rank: topology.pp.rank,
            dp_rank: topology.dp.rank,
            node_rank: topology.node.rank,
            world_rank,
        };
        self.topology = topology;
        Ok(self)
    }
}

impl infer_core::exec::ExecScope for CudaScope {
    type Device = Cuda;
    type Stream = CudaStream;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn enter(&self) -> infer_core::exec::ActiveGuard<'_, Self::Device> {
        let previous = <Cuda as infer_core::exec::ExecDevice>::enter_device(&self.device);
        infer_core::exec::ActiveGuard::new(self, previous)
    }

    fn stream(&self) -> &Self::Stream {
        &self.stream
    }

    fn rank(&self) -> infer_core::exec::Rank {
        self.rank
    }

    fn topology(&self) -> infer_core::exec::TopologyShape {
        self.topology
    }

    fn quant_tier(&self) -> infer_core::exec::QuantTier {
        self.quant_tier
    }

    fn workspace(&self) -> &infer_core::exec::Workspace<Self::Device> {
        &self.workspace
    }

    fn supports_graphs(&self) -> bool {
        // TP collectives are intentionally eager in the first NCCL release.
        // Capturing them is only safe once every rank captures and replays the
        // same collective sequence in lockstep, which the multi-worker control
        // plane does not guarantee yet.
        self.topology.tp.size == 1 && self.device.config.arena_available()
    }

    fn graph_capture_begin(&self) -> OpResult<()> {
        // Enable the capture arena BEFORE entering stream capture so the
        // forward's scratch allocations are alloc-free.
        self.device.config.arena_begin()?;
        if let Err(e) = self.device.config.capture_begin_relaxed() {
            self.device.config.arena_end();
            return Err(e);
        }
        Ok(())
    }

    fn graph_capture_end(&self, key: u64) -> OpResult<()> {
        let slot = GraphSlot::LlmDecode {
            batch: key as usize,
            buffer_id: 0,
            slot_signature: 0,
        };
        let r = self.device.config.capture_end(slot);
        self.device.config.arena_end();
        r
    }

    fn graph_launch(&self, key: u64) -> OpResult<()> {
        let slot = GraphSlot::LlmDecode {
            batch: key as usize,
            buffer_id: 0,
            slot_signature: 0,
        };
        self.device.config.launch(slot)
    }

    fn graph_ready(&self, key: u64) -> bool {
        let slot = GraphSlot::LlmDecode {
            batch: key as usize,
            buffer_id: 0,
            slot_signature: 0,
        };
        self.device.config.graph_ready(slot)
    }

    fn graph_debug_state(&self) -> &'static str {
        self.device.config.capture_state()
    }

    fn synchronize(&self) -> OpResult<()> {
        let _guard = self.enter();
        // A blocking stream sync cannot unwind itself on a distributed stall.
        // Production TP therefore wraps each mirrored operation in a separate
        // fail-stop watchdog; expiry terminates the worker without touching the
        // communicator from another thread. This post-sync check still turns
        // completed NCCL asynchronous failures into ordinary fatal errors.
        self.device.config.synchronize()?;
        if let Some(comm) = &self.tp_comm {
            comm.check_async_error()?;
        }
        Ok(())
    }
}

impl Device for Cuda {
    type ExecCtx = CudaConfig;
    fn exec_ctx(&self) -> &CudaConfig {
        &self.config
    }
    fn device_id(&self) -> i32 {
        self.device_id
    }
    fn name(&self) -> &'static str {
        "cuda"
    }
}

impl infer_core::exec::ExecDevice for Cuda {
    type Scope = CudaScope;

    fn enter_device(&self) -> infer_core::exec::DeviceId {
        let previous = device_utils::current_device().unwrap_or(self.device_id);
        device_utils::set_current_device(self.device_id).unwrap_or_else(|e| {
            panic!(
                "cudaSetDevice({}) in ExecScope::enter failed: {e:?}",
                self.device_id
            )
        });
        infer_core::exec::DeviceId(previous)
    }

    fn restore_device(&self, previous: infer_core::exec::DeviceId) {
        if previous.0 >= 0 && previous.0 != self.device_id {
            let _ = device_utils::set_current_device(previous.0);
        }
    }
}

#[inline]
fn scope_stream(scope: &CudaScope) -> ffi::cudaStream_t {
    infer_core::exec::ExecScope::stream(scope).0
}

pub(crate) fn require_scope_tensor<T: Dtype>(
    scope: &CudaScope,
    tensor: &Tensor<T, Cuda>,
    what: &str,
) -> OpResult<()> {
    let tensor_device = tensor.device();
    if tensor_device.device_id != scope.device.device_id
        || !Arc::ptr_eq(&tensor_device.config, &scope.device.config)
    {
        return Err(OpError::Kernel(format!(
            "{what} tensor belongs to CUDA device/config {}, but scope uses device {}",
            tensor_device.device_id, scope.device.device_id
        )));
    }
    Ok(())
}

impl infer_core::ports::MathOps for Cuda {
    fn add<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "add", |F| {
            kernels::add::add::<F>(
                stream,
                &a.reinterpret::<F>(),
                &b.reinterpret::<F>(),
                &mut dst.reinterpret::<F>(),
            )
        })
    }

    fn add_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        dst: &mut Tensor<T, Self>,
        src: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "add_inplace", |F| {
            kernels::add::add_inplace::<F>(
                stream,
                &mut dst.reinterpret::<F>(),
                &src.reinterpret::<F>(),
            )
        })
    }

    fn ewise_mul<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "ewise_mul", |F| {
            kernels::ewise_mul::ewise_mul::<F>(
                stream,
                &a.reinterpret::<F>(),
                &b.reinterpret::<F>(),
                &mut dst.reinterpret::<F>(),
            )
        })
    }

    fn scalar_mul_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        scalar: f64,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "scalar_mul_inplace", |F| {
            kernels::scalar::scalar_mul_inplace::<F>(stream, &mut x.reinterpret::<F>(), scalar)
        })
    }

    fn broadcast_mul_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "broadcast_mul_inplace", |F| {
            kernels::broadcast_mul::broadcast_mul_inplace::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                &scale.reinterpret::<F>(),
            )
        })
    }

    fn broadcast_add_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        bias: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "broadcast_add_inplace", |F| {
            kernels::broadcast_mul::broadcast_add_inplace::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                &bias.reinterpret::<F>(),
            )
        })
    }

    fn matmul<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::matmul::matmul(scope_stream(scope), input, weight, output)
    }

    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        input: &Tensor<A, Self>,
        weight: &Tensor<W, Self>,
        output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>,
        zeros: Option<&Tensor<W, Self>>,
        scheme: &infer_core::dtype::quant::QuantScheme,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::matmul::matmul_quant(
            scope_stream(scope),
            input,
            weight,
            output,
            scales,
            zeros,
            scheme,
        )
    }

    fn matmul_fp8_block<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<infer_core::dtype::Fp8E4m3, Self>,
        output: &mut Tensor<T, Self>,
        weight_scale_inv: &Tensor<f32, Self>,
        block: [usize; 2],
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let workspace = infer_core::exec::ExecScope::workspace(scope);
        kernels::matmul::matmul_fp8_block(
            scope_stream(scope),
            input,
            weight,
            output,
            weight_scale_inv,
            block,
            workspace
                .ptr()
                .map_or(std::ptr::null_mut(), |ptr| ptr.as_ptr().cast()),
            workspace.size(),
        )
    }

    fn rmsnorm<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "rmsnorm", |F| {
            kernels::rmsnorm::rmsnorm::<F>(
                stream,
                &input.reinterpret::<F>(),
                &weight.reinterpret::<F>(),
                &mut output.reinterpret::<F>(),
                eps,
            )
        })
    }

    fn rmsnorm_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "rmsnorm_inplace", |F| {
            kernels::rmsnorm::rmsnorm_inplace::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                &weight.reinterpret::<F>(),
                eps,
            )
        })
    }

    fn silu_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "silu_inplace", |F| {
            kernels::swiglu::silu_inplace::<F>(stream, &mut x.reinterpret::<F>())
        })
    }

    fn softmax<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "softmax", |F| {
            kernels::softmax::softmax::<F>(
                stream,
                &input.reinterpret::<F>(),
                &mut output.reinterpret::<F>(),
            )
        })
    }

    fn rope_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        q: &mut Tensor<T, Self>,
        k: &mut Tensor<T, Self>,
        sin: &Tensor<T, Self>,
        cos: &Tensor<T, Self>,
        positions: &Tensor<i32, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let num_tokens = q.shape().as_slice()[0] as i32;
        let stream = scope_stream(scope);
        narrow_float!(T, "rope_inplace", |F| {
            kernels::rope::rope_inplace::<F>(
                stream,
                &mut q.reinterpret::<F>(),
                &mut k.reinterpret::<F>(),
                &sin.reinterpret::<F>(),
                &cos.reinterpret::<F>(),
                positions.data_ptr(),
                num_tokens,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
            )
        })
    }

    fn sdpa<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        q: &Tensor<T, Self>,
        k: &Tensor<T, Self>,
        v: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        mask: Option<&Tensor<T, Self>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        match mask {
            Some(mask) => kernels::sdpa::sdpa_masked(
                stream,
                q,
                k,
                v,
                output,
                mask,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            ),
            None => kernels::sdpa::sdpa(
                stream,
                q,
                k,
                v,
                output,
                num_heads,
                num_kv_heads,
                head_dim,
                scale,
            ),
        }
    }

    fn embedding<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        table: &Tensor<T, Self>,
        indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "embedding", |F| {
            kernels::embedding::embedding::<F>(
                stream,
                &table.reinterpret::<F>(),
                indices,
                &mut output.reinterpret::<F>(),
            )
        })
    }

    fn split_cols<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        src: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        dst_cols: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "split_cols", |F| {
            kernels::split_cols::split_cols::<F>(
                stream,
                &src.reinterpret::<F>(),
                &mut dst.reinterpret::<F>(),
                rows as i32,
                total_cols as i32,
                col_offset as i32,
                dst_cols as i32,
            )
        })
    }

    fn concat_seq<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::concat_seq::concat_seq_into(scope_stream(scope), a, b, dst)
    }

    fn cast<S: Dtype, T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        src: &Tensor<S, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::cast_dtype::cast_dtype(scope_stream(scope), src, dst)
    }
}

impl infer_core::ports::VocabOps for Cuda {
    fn vocab_embedding<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        table: &Tensor<T, Self>,
        global_indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
        vocab_start: usize,
        global_vocab_size: usize,
    ) -> OpResult<()> {
        i32::try_from(global_vocab_size)
            .map_err(|_| OpError::Shape("global vocabulary exceeds i32 token ids".into()))?;
        require_scope_tensor(scope, table, "vocab_embedding table")?;
        require_scope_tensor(scope, global_indices, "vocab_embedding indices")?;
        require_scope_tensor(scope, output, "vocab_embedding output")?;
        for (tensor_is_contiguous, what) in [
            (table.is_contiguous(), "table"),
            (global_indices.is_contiguous(), "indices"),
            (output.is_contiguous(), "output"),
        ] {
            if !tensor_is_contiguous {
                return Err(OpError::Shape(format!(
                    "vocab_embedding {what} must be contiguous"
                )));
            }
        }
        let table_shape = table.shape().as_slice();
        if table_shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "vocab_embedding table must be rank 2, got {:?}",
                table_shape
            )));
        }
        if table_shape[0] == 0 || table_shape[1] == 0 {
            return Err(OpError::Shape(format!(
                "vocab_embedding table dimensions must be non-zero, got {:?}",
                table_shape
            )));
        }
        let vocab_end = vocab_start
            .checked_add(table_shape[0])
            .ok_or_else(|| OpError::Shape("vocab_embedding shard range overflows".into()))?;
        if vocab_end > global_vocab_size {
            return Err(OpError::Shape(format!(
                "vocab_embedding shard [{vocab_start}, {vocab_end}) exceeds global vocabulary {global_vocab_size}"
            )));
        }
        if output.shape().as_slice() != [global_indices.numel(), table_shape[1]] {
            return Err(OpError::Shape(format!(
                "vocab_embedding output shape {:?} does not match {} tokens x dim {}",
                output.shape().as_slice(),
                global_indices.numel(),
                table_shape[1]
            )));
        }

        let _guard = infer_core::exec::ExecScope::enter(scope);
        let stream = scope_stream(scope);
        narrow_float!(T, "vocab_embedding", |F| {
            kernels::embedding::vocab_embedding::<F>(
                stream,
                &table.reinterpret::<F>(),
                global_indices,
                &mut output.reinterpret::<F>(),
                vocab_start,
            )
        })
    }
}

impl infer_core::ports::FusedOps for Cuda {
    fn set_prefill_gemm_mode(on: bool) {
        kernels::matmul::set_eager_prefill_gemm(on);
    }

    fn set_unified_mixed_capture(on: bool) {
        kernels::flash_attn_gqa::set_fa3_capture_allowed(on);
    }

    fn unified_mixed_attention_available<T: infer_core::dtype::Dtype>(head_dim: usize) -> bool {
        kernels::flash_attn_gqa::fa3_unified_available::<T>(head_dim)
    }

    fn fused_add_rmsnorm<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        output: &mut Tensor<T, Self>,
        residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        let stream = scope_stream(ctx.scope());
        narrow_float!(T, "fused_add_rmsnorm", |F| {
            kernels::fused_add_rmsnorm::fused_add_rmsnorm::<F>(
                stream,
                &mut output.reinterpret::<F>(),
                &mut residual.reinterpret::<F>(),
                &input.reinterpret::<F>(),
                &weight.reinterpret::<F>(),
                eps,
            )
        })
    }

    fn swiglu_packed<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        let stream = scope_stream(ctx.scope());
        narrow_float_no_f16!(T, "swiglu_packed", |F| {
            kernels::swiglu::swiglu_packed::<F>(
                stream,
                &gate_up.reinterpret::<F>(),
                &mut out.reinterpret::<F>(),
                rows,
                inter,
            )
        })
    }

    fn argmax<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        logits: &Tensor<T, Self>,
    ) -> OpResult<Vec<i32>> {
        let shape = logits.shape().as_slice();
        if shape.len() != 2 {
            return Err(OpError::Shape(format!(
                "argmax: expected 2D logits [rows, vocab], got {:?}",
                shape
            )));
        }
        let rows = shape[0].max(1);
        let dev = logits.device();
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        // On-device two-phase argmax. Only the per-row token ids are copied back
        // to host (`rows` i32), instead of the full [rows, vocab] logits.
        // Workspace: the bf16 kernel uses batch*512 bf16 (= batch*256 f32) of
        // scratch; allocate 512 f32/row for headroom.
        let mut out = Tensor::<i32, Self>::zeros([rows], dev)?;
        let ws = Tensor::<f32, Self>::zeros([rows * 512], dev)?;
        kernels::sampler::argmax(scope_stream(ctx.scope()), logits, &mut out, &ws, None)?;
        out.to_host_vec()
    }

    fn argmax_into<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        logits: &Tensor<T, Self>,
        out: &mut Tensor<i32, Self>,
        workspace: &Tensor<f32, Self>,
        selected_rows: Option<&Tensor<i32, Self>>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        // Writes per-row argmax into the caller's persistent `out`/`workspace`
        // (no allocation) so this is safe inside CUDA-graph capture. When
        // `selected_rows` is `Some`, the kernel argmaxes ONLY those rows and
        // writes K = selected_rows.numel() ids into `out` (in order) — used by
        // prefill to skip num_tokens-K rows the sampler would discard.
        kernels::sampler::argmax(
            scope_stream(ctx.scope()),
            logits,
            out,
            workspace,
            selected_rows,
        )
    }

    fn scatter_kv_paged<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        k_src: &Tensor<T, Self>,
        v_src: &Tensor<T, Self>,
        layer: &mut infer_core::kv::LayerKv<'_, T, Self>,
        kv_dim: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::kv_cache::scatter_kv_paged(
            scope_stream(ctx.scope()),
            k_src,
            v_src,
            layer.k,
            layer.v,
            &layer.index.block_tables,
            &layer.index.seq_positions,
            &layer.index.cu_q_lens,
            &layer.index.seq_lens_step,
            ctx.plan().max_blocks_per_seq,
            ctx.plan().block_size,
            kv_dim,
        )
    }

    fn qkv_split<T: infer_core::dtype::Dtype>(
        _ctx: &infer_core::exec::StepCtx<'_, Self>,
        qkv: &Tensor<T, Self>,
        _num_tokens: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> OpResult<(Tensor<T, Self>, Tensor<T, Self>, Tensor<T, Self>)> {
        // Zero-copy: Q/K/V are column narrows of the fused `qkv` buffer (row
        // stride = qkv_dim). Every CUDA attention kernel (qkv_norm_rope_scatter,
        // rope_inplace, scatter_kv_paged, attention_paged) reads row/col strides
        // directly, so no split copy and no per-layer q/k/v allocation.
        let q = qkv.narrow(1, 0, q_dim)?;
        let k = qkv.narrow(1, q_dim, kv_dim)?;
        let v = qkv.narrow(1, q_dim + kv_dim, kv_dim)?;
        Ok((q, k, v))
    }

    fn qkv_norm_rope_scatter<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
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
        layer: &mut infer_core::kv::LayerKv<'_, T, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        // Grid-size the scatter by the ACTUAL active batch, not the
        // capacity-allocated control-buffer shape. The KV-index buffers
        // (`seq_positions` etc.) are allocated at `cap_batch` and zero-padded, and
        // the kernel derives its sequence-grid from `seq_positions.shape()[0]`. So
        // without this narrow a single-request prefill (batch=1) launches
        // `cap_batch` (e.g. 256) sequence-blocks per layer — 255 of them empty —
        // which made `qkv_norm_rope_scatter` ~3x slower than the pre-refactor path
        // and was the dominant prefill-forward (TTFT) regression. The data at
        // [0, batch) is identical; only the launched grid shrinks. Decode-only
        // steps are CUDA-graph-captured at fixed shapes, so leave their grid
        // exactly as-is and shrink only the eager prefill/ragged path.
        let active_batch = if ctx.plan().is_decode_only() {
            layer.index.seq_positions.shape().as_slice()[0]
        } else {
            ctx.plan().batch.max(1)
        };
        let seq_positions_active = layer.index.seq_positions.narrow(0, 0, active_batch)?;
        match (q_weight, k_weight) {
            (Some(q_weight), Some(k_weight)) => {
                kernels::qkv_norm_rope_scatter::qkv_norm_rope_scatter(
                    scope_stream(ctx.scope()),
                    q,
                    k,
                    v,
                    Some(q_weight),
                    Some(k_weight),
                    q_eps,
                    k_eps,
                    sin,
                    cos,
                    positions,
                    layer.k,
                    layer.v,
                    &layer.index.block_tables,
                    &seq_positions_active,
                    &layer.index.cu_q_lens,
                    &layer.index.seq_lens_step,
                    ctx.plan().max_blocks_per_seq,
                    ctx.plan().block_size,
                    head_num,
                    kv_head_num,
                    head_dim,
                    kv_dim,
                )
            }
            (None, None) => {
                let num_tokens = q.shape().as_slice()[0] as i32;
                let stream = scope_stream(ctx.scope());
                narrow_float!(T, "rope_inplace", |F| {
                    kernels::rope::rope_inplace::<F>(
                        stream,
                        &mut q.reinterpret::<F>(),
                        &mut k.reinterpret::<F>(),
                        &sin.reinterpret::<F>(),
                        &cos.reinterpret::<F>(),
                        positions.data_ptr(),
                        num_tokens,
                        head_num as i32,
                        kv_head_num as i32,
                        head_dim as i32,
                    )
                })?;
                kernels::kv_cache::scatter_kv_paged(
                    scope_stream(ctx.scope()),
                    k,
                    v,
                    layer.k,
                    layer.v,
                    &layer.index.block_tables,
                    &layer.index.seq_positions,
                    &layer.index.cu_q_lens,
                    &layer.index.seq_lens_step,
                    ctx.plan().max_blocks_per_seq,
                    ctx.plan().block_size,
                    kv_dim,
                )
            }
            _ => Err(OpError::Kernel(
                "qkv_norm_rope_scatter: q/k norm weights must be both present or both absent"
                    .into(),
            )),
        }
    }

    fn attention_paged<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        q: &Tensor<T, Self>,
        kv: &infer_core::kv::KvView<'_, T, Self>,
        output: &mut Tensor<T, Self>,
        head_num: usize,
        kv_head_num: usize,
        head_dim: usize,
        scale: f32,
        workspace: Option<&mut Tensor<f32, Self>>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        let (k_pool, v_pool) = kv.layer(0);
        // Decode uses this region for split-K partials; FA3 ragged attention
        // uses it for LSE plus its scheduler semaphore. Caller-owned scratch is
        // the hot path here — every layer of every step used to allocate and
        // memset fresh storage. Its stable address also bakes cleanly into
        // captured graphs.
        match workspace {
            Some(ws) => kernels::flash_attn_gqa::attention_paged(
                scope_stream(ctx.scope()),
                q,
                k_pool,
                v_pool,
                output,
                kernels::flash_attn_gqa::PagedAttentionPlan::from_v2(ctx.plan(), kv.index),
                ws,
                head_num,
                kv_head_num,
                head_dim,
                scale,
            ),
            None => {
                // Legacy fallback: still works (and graph-captures correctly
                // through the capture arena), but pays the per-call alloc.
                let workspace_elems =
                    kernels::flash_attn_gqa::flash_attention_workspace_capacity_f32(
                        ctx.plan().batch,
                        q.shape().as_slice()[0],
                        head_num,
                        head_dim,
                    )
                    .max(1);
                let mut owned = Tensor::<f32, Cuda>::zeros([workspace_elems], q.device())?;
                kernels::flash_attn_gqa::attention_paged(
                    scope_stream(ctx.scope()),
                    q,
                    k_pool,
                    v_pool,
                    output,
                    kernels::flash_attn_gqa::PagedAttentionPlan::from_v2(ctx.plan(), kv.index),
                    &mut owned,
                    head_num,
                    kv_head_num,
                    head_dim,
                    scale,
                )
            }
        }
    }

    fn flash_attention_workspace_capacity_f32(
        batch: usize,
        num_tokens: usize,
        num_q_heads: usize,
        head_dim: usize,
    ) -> usize {
        kernels::flash_attn_gqa::flash_attention_workspace_capacity_f32(
            batch,
            num_tokens,
            num_q_heads,
            head_dim,
        )
    }
}

impl Cuda {
    /// Create a new Cuda device (allocates stream + handles).
    pub fn new(device_id: i32) -> Result<Self, OpError> {
        Self::with_memory_plan(device_id, config::CudaMemoryPlan::default())
    }

    pub fn with_memory_plan(
        device_id: i32,
        memory_plan: config::CudaMemoryPlan,
    ) -> Result<Self, OpError> {
        device_utils::set_current_device(device_id)
            .map_err(|e| OpError::Kernel(format!("set device failed: {}", e)))?;
        let config = Arc::new(
            CudaConfig::with_memory_plan(memory_plan)
                .map_err(|e| OpError::Kernel(format!("CudaConfig creation failed: {}", e)))?,
        );
        kernels::initialize_device(config.device_info())?;
        Ok(Self { device_id, config })
    }

    pub fn scope(&self) -> CudaScope {
        CudaScope::new(self.clone())
    }
}

// ─── Cuda MemoryPort ─────────────────────────────────────────────────────────

impl MemoryPort for Cuda {
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>> {
        // During graph capture/replay, serve scratch from the capture arena so
        // no `cudaMalloc` is issued (illegal while a stream is capturing).
        if let Some(arena_ptr) = self.config.arena_alloc(size) {
            return NonNull::new(arena_ptr as *mut u8)
                .ok_or_else(|| OpError::Kernel("graph arena returned null".into()));
        }
        // Recycle a previously-freed block of the same size class if one is
        // available — avoids cudaMalloc entirely once the pool warms up.
        let n = self.config.round_up_256(size);
        if let Some(ptr) = self.config.pool_pop(n) {
            // The reused block holds a prior tenant's bytes; zero it stream-
            // ordered (async, no host stall) to preserve `Tensor::zeros`
            // semantics for every caller.
            unsafe {
                ffi::cudaMemsetAsync(ptr, 0, n, self.config.stream);
            }
            return NonNull::new(ptr as *mut u8)
                .ok_or_else(|| OpError::Kernel("cuda pool returned null".into()));
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cudaMalloc/cudaMemsetAsync are safe to call with valid args.
        unsafe {
            let code = ffi::cudaMalloc(&mut ptr, n);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaMalloc({}) failed: {:?}",
                    n, code
                )));
            }
            // Async, stream-ordered zero — replaces the host-blocking
            // synchronous cudaMemset that dominated eager-forward TTFT.
            ffi::cudaMemsetAsync(ptr, 0, n, self.config.stream);
        }
        self.config.pool_note_cold_alloc(n);
        NonNull::new(ptr as *mut u8)
            .ok_or_else(|| OpError::Kernel("cudaMalloc returned null".into()))
    }

    unsafe fn free_bytes(&self, ptr: NonNull<u8>, size: usize) {
        // Arena-owned scratch is not individually freed — the bump arena is
        // reset wholesale at the next capture. A real `cudaFree` here would
        // both corrupt the arena and (if mid-capture) be illegal.
        if self
            .config
            .arena_contains(ptr.as_ptr() as *mut std::ffi::c_void)
        {
            return;
        }
        // Recycle into the size-keyed pool instead of `cudaFree` (which would
        // device-synchronize). The block is reused by the next same-size alloc;
        // all retained blocks are released in `CudaConfig::Drop` via `pool_drain`.
        let n = self.config.round_up_256(size);
        self.config
            .pool_push(n, ptr.as_ptr() as *mut std::ffi::c_void);
    }

    unsafe fn upload(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        if size == 0 {
            return Ok(());
        }
        let stream = self.config.stream;
        // SAFETY: caller asserts dst is a device ptr with `size` bytes,
        // src is a host ptr with `size` bytes.
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst.as_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaMemcpyAsync H2D failed: {:?}",
                    code
                )));
            }
            // Sync so the host buffer can be freed/reused safely after this returns.
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaStreamSynchronize failed: {:?}",
                    code
                )));
            }
            error::check_last_error("cuda upload sync observed prior kernel error")?;
        }
        Ok(())
    }

    unsafe fn upload_async(&self, dst: NonNull<u8>, src: *const u8, size: usize) -> OpResult<()> {
        if size == 0 {
            return Ok(());
        }
        let stream = self.config.stream;
        // SAFETY: caller asserts the host pointer remains valid until the
        // device stream consumes the copy (workspaces own host staging
        // buffers for their entire lifetime, so this is upheld).
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst.as_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaMemcpyAsync H2D async failed: {:?}",
                    code
                )));
            }
            // NO cudaStreamSynchronize — graph capture friendly.
        }
        Ok(())
    }

    unsafe fn download(&self, dst: *mut u8, src: NonNull<u8>, size: usize) -> OpResult<()> {
        if size == 0 {
            return Ok(());
        }
        let stream = self.config.stream;
        // SAFETY: caller asserts ptrs and size.
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaMemcpyAsync D2H failed: {:?}",
                    code
                )));
            }
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaStreamSynchronize failed: {:?}",
                    code
                )));
            }
            error::check_last_error("cuda download sync observed prior kernel error")?;
        }
        Ok(())
    }

    fn synchronize(&self) -> OpResult<()> {
        let stream = self.config.stream;
        // SAFETY: stream is owned by self.config.
        let code = unsafe { ffi::cudaStreamSynchronize(stream) };
        if code != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!(
                "cudaStreamSynchronize failed: {:?}",
                code
            )));
        }
        error::check_last_error("cuda synchronize observed prior kernel error")?;
        Ok(())
    }

    unsafe fn copy_device_to_device(
        &self,
        dst: NonNull<u8>,
        src: NonNull<u8>,
        size: usize,
    ) -> OpResult<()> {
        if size == 0 {
            return Ok(());
        }
        let stream = self.config.stream;
        // SAFETY: caller asserts dst/src are device ptrs with `size` bytes,
        // and regions do not overlap.
        unsafe {
            let code = ffi::cudaMemcpyAsync(
                dst.as_ptr() as *mut std::ffi::c_void,
                src.as_ptr() as *const std::ffi::c_void,
                size,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                stream,
            );
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaMemcpyAsync D2D failed: {:?}",
                    code
                )));
            }
            let code = ffi::cudaStreamSynchronize(stream);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaStreamSynchronize failed: {:?}",
                    code
                )));
            }
            error::check_last_error("cuda D2D sync observed prior kernel error")?;
        }
        Ok(())
    }
}

/// CoreOps for Cuda.
/// The stream choice is explicit at the kernel boundary even though this trait
/// cannot carry `ExecScope`.
impl CoreOps for Cuda {
    // alloc_tensor uses the default impl (Tensor::zeros via MemoryPort).

    fn add<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let stream = a.device().config.stream;
        narrow_float!(T, "add", |F| {
            kernels::add::add::<F>(
                stream,
                &a.reinterpret::<F>(),
                &b.reinterpret::<F>(),
                &mut dst.reinterpret::<F>(),
            )
        })
    }
    fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()> {
        let stream = dst.device().config.stream;
        narrow_float!(T, "add_inplace", |F| {
            kernels::add::add_inplace::<F>(
                stream,
                &mut dst.reinterpret::<F>(),
                &src.reinterpret::<F>(),
            )
        })
    }
    fn ewise_mul<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let stream = a.device().config.stream;
        narrow_float!(T, "ewise_mul", |F| {
            kernels::ewise_mul::ewise_mul::<F>(
                stream,
                &a.reinterpret::<F>(),
                &b.reinterpret::<F>(),
                &mut dst.reinterpret::<F>(),
            )
        })
    }
    fn matmul<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::matmul::matmul(input.device().config.stream, input, weight, output)
    }
    fn matmul_quant<A: Dtype, W: Dtype, O: Dtype>(
        input: &Tensor<A, Self>,
        weight: &Tensor<W, Self>,
        output: &mut Tensor<O, Self>,
        scales: &Tensor<A, Self>,
        zeros: Option<&Tensor<W, Self>>,
        scheme: &infer_core::dtype::quant::QuantScheme,
    ) -> OpResult<()> {
        kernels::matmul::matmul_quant(
            input.device().config.stream,
            input,
            weight,
            output,
            scales,
            zeros,
            scheme,
        )
    }
    fn matmul_fp8_block<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<infer_core::dtype::Fp8E4m3, Self>,
        output: &mut Tensor<T, Self>,
        weight_scale_inv: &Tensor<f32, Self>,
        block: [usize; 2],
    ) -> OpResult<()> {
        let cfg = &input.device().config;
        let workspace = cfg.kernel_workspace();
        kernels::matmul::matmul_fp8_block(
            cfg.stream,
            input,
            weight,
            output,
            weight_scale_inv,
            block,
            workspace.ptr(),
            workspace.size(),
        )
    }
    fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "silu_inplace", |F| {
            kernels::swiglu::silu_inplace::<F>(stream, &mut x.reinterpret::<F>())
        })
    }
    fn softmax<T: Dtype>(input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()> {
        let stream = input.device().config.stream;
        narrow_float!(T, "softmax", |F| {
            kernels::softmax::softmax::<F>(
                stream,
                &input.reinterpret::<F>(),
                &mut output.reinterpret::<F>(),
            )
        })
    }
    fn embedding<T: Dtype>(
        table: &Tensor<T, Self>,
        indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let stream = table.device().config.stream;
        narrow_float!(T, "embedding", |F| {
            kernels::embedding::embedding::<F>(
                stream,
                &table.reinterpret::<F>(),
                indices,
                &mut output.reinterpret::<F>(),
            )
        })
    }
    fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "scalar_mul_inplace", |F| {
            kernels::scalar::scalar_mul_inplace::<F>(stream, &mut x.reinterpret::<F>(), scalar)
        })
    }
    fn scalar_add_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "scalar_add_inplace", |F| {
            kernels::scalar::scalar_add_inplace::<F>(stream, &mut x.reinterpret::<F>(), scalar)
        })
    }
    fn scalar_mul_inplace_from_dev<T: Dtype>(
        x: &mut Tensor<T, Self>,
        d_scalar: &Tensor<f32, Self>,
    ) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "scalar_mul_inplace_from_dev", |F| {
            kernels::scalar::scalar_mul_inplace_from_dev::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                d_scalar,
            )
        })
    }
    fn broadcast_mul_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "broadcast_mul_inplace", |F| {
            kernels::broadcast_mul::broadcast_mul_inplace::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                &scale.reinterpret::<F>(),
            )
        })
    }
    fn broadcast_add_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        bias: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "broadcast_add_inplace", |F| {
            kernels::broadcast_mul::broadcast_add_inplace::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                &bias.reinterpret::<F>(),
            )
        })
    }
    fn split_cols<T: Dtype>(
        src: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        dst_cols: usize,
    ) -> OpResult<()> {
        let stream = src.device().config.stream;
        narrow_float!(T, "split_cols", |F| {
            kernels::split_cols::split_cols::<F>(
                stream,
                &src.reinterpret::<F>(),
                &mut dst.reinterpret::<F>(),
                rows as i32,
                total_cols as i32,
                col_offset as i32,
                dst_cols as i32,
            )
        })
    }
    fn concat_seq<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::concat_seq::concat_seq_into(a.device().config.stream, a, b, dst)
    }
    fn cast_dtype<S: Dtype, D2: Dtype>(
        src: &Tensor<S, Self>,
        dst: &mut Tensor<D2, Self>,
    ) -> OpResult<()> {
        kernels::cast_dtype::cast_dtype(src.device().config.stream, src, dst)
    }

    fn rmsnorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let stream = input.device().config.stream;
        narrow_float!(T, "rmsnorm", |F| {
            kernels::rmsnorm::rmsnorm::<F>(
                stream,
                &input.reinterpret::<F>(),
                &weight.reinterpret::<F>(),
                &mut output.reinterpret::<F>(),
                eps,
            )
        })
    }
    fn rmsnorm_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "rmsnorm_inplace", |F| {
            kernels::rmsnorm::rmsnorm_inplace::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                &weight.reinterpret::<F>(),
                eps,
            )
        })
    }
    fn swiglu_packed<T: Dtype>(
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        let stream = gate_up.device().config.stream;
        narrow_float_no_f16!(T, "swiglu_packed", |F| {
            kernels::swiglu::swiglu_packed::<F>(
                stream,
                &gate_up.reinterpret::<F>(),
                &mut out.reinterpret::<F>(),
                rows,
                inter,
            )
        })
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
        let num_tokens = q.shape().as_slice()[0] as i32;
        let stream = q.device().config.stream;
        narrow_float!(T, "rope_inplace", |F| {
            kernels::rope::rope_inplace::<F>(
                stream,
                &mut q.reinterpret::<F>(),
                &mut k.reinterpret::<F>(),
                &sin.reinterpret::<F>(),
                &cos.reinterpret::<F>(),
                positions.data_ptr(),
                num_tokens,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
            )
        })
    }
}

/// DiffusionOps for Cuda — Conv / Norm / Spatial / DiT kernels (Z_Image).
impl DiffusionOps for Cuda {
    fn conv2d<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: Option<&Tensor<T, Self>>,
        output: &mut Tensor<T, Self>,
        stride: usize,
        padding: usize,
    ) -> OpResult<()> {
        kernels::conv2d::conv2d(
            input.device().config.stream,
            input,
            weight,
            bias,
            output,
            stride,
            padding,
        )
    }

    fn groupnorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()> {
        kernels::groupnorm::groupnorm(
            input.device().config.stream,
            input,
            weight,
            bias,
            output,
            num_groups,
            eps,
        )
    }

    fn groupnorm_silu<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()> {
        kernels::groupnorm::groupnorm_silu(
            input.device().config.stream,
            input,
            weight,
            bias,
            output,
            num_groups,
            eps,
        )
    }

    fn layernorm<T: Dtype>(
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let stream = input.device().config.stream;
        narrow_float!(T, "layernorm", |F| {
            kernels::layernorm::layernorm::<F>(
                stream,
                &input.reinterpret::<F>(),
                &weight.reinterpret::<F>(),
                &bias.reinterpret::<F>(),
                &mut output.reinterpret::<F>(),
                eps,
            )
        })
    }

    fn upsample_nearest_2x<T: Dtype>(
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::upsample::upsample_nearest_2x(input.device().config.stream, input, output)
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
        kernels::sdpa::sdpa(
            q.device().config.stream,
            q,
            k,
            v,
            output,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }

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
    ) -> OpResult<()> {
        kernels::sdpa::sdpa_masked(
            q.device().config.stream,
            q,
            k,
            v,
            output,
            mask,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        )
    }

    fn apply_rope_interleaved<T: Dtype>(
        x: &mut Tensor<T, Self>,
        cos: &Tensor<f32, Self>,
        sin: &Tensor<f32, Self>,
        head_dim: usize,
    ) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float_no_f16!(T, "apply_rope_interleaved", |F| {
            kernels::rope_interleaved::apply_rope_interleaved::<F>(
                stream,
                &mut x.reinterpret::<F>(),
                cos,
                sin,
                head_dim,
            )
        })
    }

    fn pad_with_token<T: Dtype>(
        src: &Tensor<T, Self>,
        pad_token: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::pad::pad_with_token_into(src.device().config.stream, src, pad_token, dst)
    }

    fn pad_last_row<T: Dtype>(src: &Tensor<T, Self>, dst: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::pad::pad_last_row_into(src.device().config.stream, src, dst)
    }

    fn overwrite_pad_tokens_inplace<T: Dtype>(
        dst: &mut Tensor<T, Self>,
        pad_token: &Tensor<T, Self>,
        keep_prefix: usize,
    ) -> OpResult<()> {
        kernels::pad::overwrite_pad_tokens_inplace(
            dst.device().config.stream,
            dst,
            pad_token,
            keep_prefix,
        )
    }

    fn silu_inplace_diff<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "silu_inplace", |F| {
            kernels::scalar::silu_inplace::<F>(stream, &mut x.reinterpret::<F>())
        })
    }

    fn tanh_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        let stream = x.device().config.stream;
        narrow_float!(T, "tanh_inplace", |F| {
            kernels::scalar::tanh_inplace::<F>(stream, &mut x.reinterpret::<F>())
        })
    }
}

// ─── DecodePipelineOps: the real CUDA implementation ─────────────────────────
// Delegates the merge/control kernels to `kernels::gather_merge` and the
// event/copy choreography to the device's dual-stream `CudaConfig` machinery.
impl infer_core::ports::DecodePipelineOps for Cuda {
    fn append_decode_admissions(
        scope: &CudaScope,
        a_out: &mut Tensor<i32, Cuda>,
        b_new: &Tensor<i32, Cuda>,
        start: usize,
        count: usize,
    ) -> OpResult<()> {
        let stream = infer_core::exec::ExecScope::stream(scope).0;
        kernels::gather_merge::append_decode_admissions_into(a_out, b_new, start, count, stream)
    }

    fn merge_compact_decode(
        scope: &CudaScope,
        args: infer_core::ports::MergeCompactDecodeArgs<'_, Cuda>,
    ) -> OpResult<()> {
        let stream = infer_core::exec::ExecScope::stream(scope).0;
        kernels::gather_merge::merge_compact_decode_into(
            kernels::gather_merge::MergeCompactDecodeArgs {
                a_out: args.a_out,
                c_prev: args.c_prev,
                generated_counts: args.generated_counts,
                max_tokens: args.max_tokens,
                ignore_eos: args.ignore_eos,
                eos_ids: args.eos_ids,
                eos_len: args.eos_len,
                old_batch: args.old_batch,
                active_src_rows: &*args.active_src_rows,
                finished_src_rows: &*args.finished_src_rows,
                finished_tokens: &*args.finished_tokens,
                counts: &*args.counts,
                stream,
            },
        )
    }

    fn merge_compact_mixed(
        scope: &CudaScope,
        args: infer_core::ports::MergeCompactMixedArgs<'_, Cuda>,
    ) -> OpResult<()> {
        let stream = infer_core::exec::ExecScope::stream(scope).0;
        kernels::gather_merge::merge_compact_mixed_into(
            kernels::gather_merge::MergeCompactMixedArgs {
                a_out: args.a_out,
                c_prev: args.c_prev,
                row_kind: args.row_kind,
                generated_counts: args.generated_counts,
                max_tokens: args.max_tokens,
                ignore_eos: args.ignore_eos,
                eos_ids: args.eos_ids,
                eos_len: args.eos_len,
                old_rows: args.old_rows,
                active_src_rows: &*args.active_src_rows,
                active_tokens: &*args.active_tokens,
                finished_src_rows: &*args.finished_src_rows,
                finished_tokens: &*args.finished_tokens,
                prefill_final_src_rows: &*args.prefill_final_src_rows,
                prefill_final_tokens: &*args.prefill_final_tokens,
                counts: &*args.counts,
                stream,
            },
        )
    }

    fn compact_extend_control(
        scope: &CudaScope,
        args: infer_core::ports::CompactExtendControlArgs<'_, Cuda>,
    ) -> OpResult<()> {
        let stream = infer_core::exec::ExecScope::stream(scope).0;
        kernels::gather_merge::compact_extend_control_into(
            kernels::gather_merge::CompactExtendControlArgs {
                block_tables: args.block_tables,
                block_tables_scratch: args.block_tables_scratch,
                kv_lens: args.kv_lens,
                kv_lens_scratch: args.kv_lens_scratch,
                seq_positions_out: args.seq_positions_out,
                seq_lens_step_out: args.seq_lens_step_out,
                rope_positions_out: args.rope_positions_out,
                cu_q_lens_out: args.cu_q_lens_out,
                block2req_out: args.block2req_out,
                block2tile_out: args.block2tile_out,
                active_src_rows: args.active_src_rows,
                counts: args.counts,
                new_slots: args.new_slots,
                mbps: args.mbps,
                cap_batch: args.cap_batch,
                stream,
            },
        )
    }

    fn pipeline_pin_host_i32(scope: &CudaScope, buf: &[i32]) -> OpResult<()> {
        scope.device.config.pin_host_i32(buf)
    }

    fn pipeline_record_copy_in(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.record_copy_in()
    }
    fn pipeline_compute_wait_copy_in(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.compute_wait_copy_in()
    }
    fn pipeline_record_compute_a(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.record_compute_a()
    }
    fn pipeline_copy_out_wait_compute_a(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.copy_out_wait_compute_a()
    }
    fn pipeline_record_copy_out(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.record_copy_out()
    }
    fn pipeline_compute_wait_copy_out(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.compute_wait_copy_out()
    }
    fn pipeline_synchronize_copy_in(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.synchronize_copy_in()
    }
    fn pipeline_synchronize_copy_out(scope: &CudaScope) -> OpResult<()> {
        scope.device.config.synchronize_copy_out()
    }

    unsafe fn pipeline_upload_h2d_copy_in(
        scope: &CudaScope,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> OpResult<()> {
        unsafe { scope.device.config.upload_h2d_copy_in(dst, src, bytes) }
    }

    unsafe fn pipeline_download_d2h_copy_out(
        scope: &CudaScope,
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        bytes: usize,
    ) -> OpResult<()> {
        unsafe { scope.device.config.download_d2h_copy_out(dst, src, bytes) }
    }

    fn pipeline_arena_begin(scope: &CudaScope) -> OpResult<()> {
        if scope.device.config.arena_available() {
            scope.device.config.arena_begin()
        } else {
            // Eager execution can always fall back to the recycling allocator.
            // A zero-sized/failed graph arena only disables capture support; it
            // must not make the eager mixed path unusable.
            Ok(())
        }
    }
    fn pipeline_arena_end(scope: &CudaScope) {
        scope.device.config.arena_end();
    }
}
