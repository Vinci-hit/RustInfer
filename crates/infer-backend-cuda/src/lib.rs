//! CUDA infrastructure adapter.
//!
//! Implements `Device`, `MemoryPort`, and `OpBackend` for `Cuda`.
//! Contains: FFI bindings, CudaConfig (handles), and kernel dispatch wrappers.

pub mod config;
pub mod device_utils;
pub mod error;
pub mod ffi;
pub mod kernels;

pub use config::{CudaConfig, GraphSlot};
pub use error::CudaError;

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
    quant_tier: infer_core::exec::QuantTier,
    workspace: infer_core::exec::Workspace<Cuda>,
}

impl CudaScope {
    pub fn new(device: Cuda) -> Self {
        let stream = CudaStream(device.config.stream);
        let workspace = infer_core::exec::Workspace::from_raw(
            NonNull::new(device.config.workspace as *mut u8),
            device.config.workspace_size,
        );
        Self {
            device,
            stream,
            rank: infer_core::exec::Rank::SINGLE,
            topology: infer_core::exec::TopologyShape::SINGLE,
            quant_tier: infer_core::exec::QuantTier::None,
            workspace,
        }
    }

    pub fn with_topology(mut self, topology: infer_core::exec::TopologyShape) -> Self {
        self.rank = infer_core::exec::Rank {
            tp_rank: topology.tp.rank,
            pp_rank: topology.pp.rank,
            dp_rank: topology.dp.rank,
            node_rank: topology.node.rank,
            world_rank: 0,
        };
        self.topology = topology;
        self
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

    fn synchronize(&self) -> OpResult<()> {
        self.device.config.synchronize()
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

impl infer_core::ports::MathOps for Cuda {
    fn add<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::add::add(scope_stream(scope), a, b, dst)
    }

    fn add_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        dst: &mut Tensor<T, Self>,
        src: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::add::add_inplace(scope_stream(scope), dst, src)
    }

    fn ewise_mul<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::ewise_mul::ewise_mul(scope_stream(scope), a, b, dst)
    }

    fn scalar_mul_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        scalar: f64,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::scalar::scalar_mul_inplace(scope_stream(scope), x, scalar)
    }

    fn broadcast_mul_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::broadcast_mul::broadcast_mul_inplace(scope_stream(scope), x, scale)
    }

    fn broadcast_add_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        bias: &Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::broadcast_mul::broadcast_add_inplace(scope_stream(scope), x, bias)
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
            scheme.group,
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
        kernels::rmsnorm::rmsnorm(scope_stream(scope), input, weight, output, eps)
    }

    fn rmsnorm_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::rmsnorm::rmsnorm_inplace(scope_stream(scope), x, weight, eps)
    }

    fn silu_inplace<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        x: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::activation::silu_inplace(scope_stream(scope), x)
    }

    fn softmax<T: Dtype>(
        scope: &<Self as infer_core::exec::ExecDevice>::Scope,
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(scope);
        kernels::softmax::softmax(scope_stream(scope), input, output)
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
        kernels::rope::rope_inplace(
            scope_stream(scope),
            q,
            k,
            sin,
            cos,
            positions.data_ptr(),
            num_tokens,
            head_num as i32,
            kv_head_num as i32,
            head_dim as i32,
        )
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
        kernels::embedding::embedding(scope_stream(scope), table, indices, output)
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
        kernels::split_cols::split_cols(
            scope_stream(scope),
            src,
            dst,
            rows as i32,
            total_cols as i32,
            col_offset as i32,
            dst_cols as i32,
        )
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

impl infer_core::ports::FusedOps for Cuda {
    fn fused_add_rmsnorm<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        output: &mut Tensor<T, Self>,
        residual: &mut Tensor<T, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::fused_add_rmsnorm::fused_add_rmsnorm(
            scope_stream(ctx.scope()),
            output,
            residual,
            input,
            weight,
            eps,
        )
    }

    fn swiglu_packed<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::activation::swiglu_packed(scope_stream(ctx.scope()), gate_up, out, rows, inter)
    }

    fn scatter_kv_paged<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        k_src: &Tensor<T, Self>,
        v_src: &Tensor<T, Self>,
        layer: &mut infer_core::kv::LayerKv<'_, T, Self>,
        kv_dim: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::scatter_kv_paged::scatter_kv_paged(
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
                    &layer.index.seq_positions,
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
                kernels::rope::rope_inplace(
                    scope_stream(ctx.scope()),
                    q,
                    k,
                    sin,
                    cos,
                    positions.data_ptr(),
                    q.shape().as_slice()[0] as i32,
                    head_num as i32,
                    kv_head_num as i32,
                    head_dim as i32,
                )?;
                kernels::scatter_kv_paged::scatter_kv_paged(
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
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        let (k_pool, v_pool) = kv.layer(0);
        let workspace_elems = if ctx.plan().is_decode_only() {
            kernels::attention_paged::flash_decode_workspace_capacity_f32(
                ctx.plan().batch,
                head_num,
                head_dim,
            )
            .max(1)
        } else {
            1
        };
        let mut workspace = Tensor::<f32, Cuda>::zeros([workspace_elems], q.device())?;
        kernels::attention_paged::attention_paged(
            scope_stream(ctx.scope()),
            q,
            k_pool,
            v_pool,
            output,
            kernels::attention_paged::PagedAttentionPlan::from_v2(ctx.plan(), kv.index),
            &mut workspace,
            head_num,
            kv_head_num,
            head_dim,
            scale,
        )
    }
}

impl Cuda {
    /// Create a new Cuda device (allocates stream + handles).
    pub fn new(device_id: i32) -> Result<Self, OpError> {
        device_utils::set_current_device(device_id)
            .map_err(|e| OpError::Kernel(format!("set device failed: {}", e)))?;
        let config = Arc::new(
            CudaConfig::new()
                .map_err(|e| OpError::Kernel(format!("CudaConfig::new failed: {}", e)))?,
        );
        Ok(Self { device_id, config })
    }

    pub fn scope(&self) -> CudaScope {
        CudaScope::new(self.clone())
    }
}

// ─── Cuda MemoryPort ─────────────────────────────────────────────────────────

impl MemoryPort for Cuda {
    fn alloc_bytes(&self, size: usize) -> OpResult<NonNull<u8>> {
        let n = size.max(1);
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cudaMalloc/cudaMemset are safe to call with valid args.
        unsafe {
            let code = ffi::cudaMalloc(&mut ptr, n);
            if code != ffi::cudaError_cudaSuccess {
                return Err(OpError::Kernel(format!(
                    "cudaMalloc({}) failed: {:?}",
                    n, code
                )));
            }
            let code = ffi::cudaMemset(ptr, 0, n);
            if code != ffi::cudaError_cudaSuccess {
                ffi::cudaFree(ptr);
                return Err(OpError::Kernel(format!("cudaMemset failed: {:?}", code)));
            }
        }
        NonNull::new(ptr as *mut u8)
            .ok_or_else(|| OpError::Kernel("cudaMalloc returned null".into()))
    }

    unsafe fn free_bytes(&self, ptr: NonNull<u8>, _size: usize) {
        // SAFETY: ptr came from cudaMalloc.
        unsafe {
            ffi::cudaFree(ptr.as_ptr() as *mut std::ffi::c_void);
        }
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
        kernels::add::add(a.device().config.stream, a, b, dst)
    }
    fn add_inplace<T: Dtype>(dst: &mut Tensor<T, Self>, src: &Tensor<T, Self>) -> OpResult<()> {
        kernels::add::add_inplace(dst.device().config.stream, dst, src)
    }
    fn ewise_mul<T: Dtype>(
        a: &Tensor<T, Self>,
        b: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::ewise_mul::ewise_mul(a.device().config.stream, a, b, dst)
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
        group_size: usize,
    ) -> OpResult<()> {
        kernels::matmul::matmul_quant(
            input.device().config.stream,
            input,
            weight,
            output,
            scales,
            zeros,
            group_size,
        )
    }
    fn silu_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::activation::silu_inplace(x.device().config.stream, x)
    }
    fn softmax<T: Dtype>(input: &Tensor<T, Self>, output: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::softmax::softmax(input.device().config.stream, input, output)
    }
    fn embedding<T: Dtype>(
        table: &Tensor<T, Self>,
        indices: &Tensor<i32, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::embedding::embedding(table.device().config.stream, table, indices, output)
    }
    fn scalar_mul_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()> {
        kernels::scalar::scalar_mul_inplace(x.device().config.stream, x, scalar)
    }
    fn scalar_add_inplace<T: Dtype>(x: &mut Tensor<T, Self>, scalar: f64) -> OpResult<()> {
        kernels::scalar::scalar_add_inplace(x.device().config.stream, x, scalar)
    }
    fn scalar_mul_inplace_from_dev<T: Dtype>(
        x: &mut Tensor<T, Self>,
        d_scalar: &Tensor<f32, Self>,
    ) -> OpResult<()> {
        kernels::scalar::scalar_mul_inplace_from_dev(x.device().config.stream, x, d_scalar)
    }
    fn broadcast_mul_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        scale: &Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::broadcast_mul::broadcast_mul_inplace(x.device().config.stream, x, scale)
    }
    fn broadcast_add_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        bias: &Tensor<T, Self>,
    ) -> OpResult<()> {
        kernels::broadcast_mul::broadcast_add_inplace(x.device().config.stream, x, bias)
    }
    fn split_cols<T: Dtype>(
        src: &Tensor<T, Self>,
        dst: &mut Tensor<T, Self>,
        rows: usize,
        total_cols: usize,
        col_offset: usize,
        dst_cols: usize,
    ) -> OpResult<()> {
        kernels::split_cols::split_cols(
            src.device().config.stream,
            src,
            dst,
            rows as i32,
            total_cols as i32,
            col_offset as i32,
            dst_cols as i32,
        )
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
        kernels::rmsnorm::rmsnorm(input.device().config.stream, input, weight, output, eps)
    }
    fn rmsnorm_inplace<T: Dtype>(
        x: &mut Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        kernels::rmsnorm::rmsnorm_inplace(x.device().config.stream, x, weight, eps)
    }
    fn swiglu_packed<T: Dtype>(
        gate_up: &Tensor<T, Self>,
        out: &mut Tensor<T, Self>,
        rows: usize,
        inter: usize,
    ) -> OpResult<()> {
        kernels::activation::swiglu_packed(
            gate_up.device().config.stream,
            gate_up,
            out,
            rows,
            inter,
        )
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
        kernels::rope::rope_inplace(
            q.device().config.stream,
            q,
            k,
            sin,
            cos,
            positions.data_ptr(),
            num_tokens,
            head_num as i32,
            kv_head_num as i32,
            head_dim as i32,
        )
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
        kernels::layernorm::layernorm(
            input.device().config.stream,
            input,
            weight,
            bias,
            output,
            eps,
        )
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
        kernels::rope_interleaved::apply_rope_interleaved(
            x.device().config.stream,
            x,
            cos,
            sin,
            head_dim,
        )
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
        kernels::scalar::silu_inplace(x.device().config.stream, x)
    }

    fn tanh_inplace<T: Dtype>(x: &mut Tensor<T, Self>) -> OpResult<()> {
        kernels::scalar::tanh_inplace(x.device().config.stream, x)
    }
}

impl infer_core::ports::diffusion_ops::DiffusionOps for Cuda {
    fn conv2d<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: Option<&Tensor<T, Self>>,
        output: &mut Tensor<T, Self>,
        stride: usize,
        padding: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::conv2d::conv2d(
            scope_stream(ctx.scope()),
            input,
            weight,
            bias,
            output,
            stride,
            padding,
        )
    }

    fn groupnorm<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::groupnorm::groupnorm(
            scope_stream(ctx.scope()),
            input,
            weight,
            bias,
            output,
            num_groups,
            eps,
        )
    }

    fn groupnorm_silu<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        num_groups: usize,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::groupnorm::groupnorm_silu(
            scope_stream(ctx.scope()),
            input,
            weight,
            bias,
            output,
            num_groups,
            eps,
        )
    }

    fn layernorm<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        weight: &Tensor<T, Self>,
        bias: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
        eps: f32,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::layernorm::layernorm(scope_stream(ctx.scope()), input, weight, bias, output, eps)
    }

    fn upsample_nearest_2x<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        input: &Tensor<T, Self>,
        output: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::upsample::upsample_nearest_2x(scope_stream(ctx.scope()), input, output)
    }

    fn apply_rope_interleaved<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        x: &mut Tensor<T, Self>,
        cos: &Tensor<f32, Self>,
        sin: &Tensor<f32, Self>,
        head_dim: usize,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::rope_interleaved::apply_rope_interleaved(
            scope_stream(ctx.scope()),
            x,
            cos,
            sin,
            head_dim,
        )
    }

    fn tanh_inplace<T: infer_core::dtype::Dtype>(
        ctx: &infer_core::exec::StepCtx<'_, Self>,
        x: &mut Tensor<T, Self>,
    ) -> OpResult<()> {
        let _guard = infer_core::exec::ExecScope::enter(ctx.scope());
        kernels::scalar::tanh_inplace(scope_stream(ctx.scope()), x)
    }
}
