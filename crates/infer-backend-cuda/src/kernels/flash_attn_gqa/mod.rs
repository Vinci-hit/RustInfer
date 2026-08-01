//! Paged ragged / paged decode attention wrapper.
//!
//! - `BatchKind::DecodeOnly` → cuDNN frontend SDPA, then Flash fallback
//! - `BatchKind::Ragged`     → ONE FA3 varlen+paged launch for the whole
//!   batch (Hopper, eager, bf16 hd128); otherwise the legacy split — decode
//!   prefix on cuDNN SDPA, prefill suffix on the CuTe ragged kernel (also
//!   used inside CUDA graph capture)
//!
//! Both kernels read K/V from a single `[num_blocks, block_size, kv_dim]`
//! pool per layer; per-seq routing comes from the device `block_tables`
//! tensor (`[batch, max_blocks_per_seq]` u32).

use crate::Cuda;
use crate::ffi::{cudaStream_t, cudnnHandle_t};
use infer_core::kv::KvIndexTensors;
use infer_core::plan;
use infer_core::ports::{OpError, OpResult};
use infer_core::tensor::Tensor;
use infer_core::types::{DataType, Dtype};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

/// When set, FA3 is allowed to launch under CUDA graph capture. The runtime
/// raises this only around the mixed FA3-graph capture region (where the bucket
/// plan bakes `max_q`/`b` to proven upper bounds over every replay composition)
/// and lowers it immediately after. Replay does not re-enter this dispatch — the
/// captured FA3 node runs directly — so the flag matters only during capture.
static FA3_CAPTURE_ALLOWED: AtomicBool = AtomicBool::new(false);

/// Toggle FA3-under-capture (see `FA3_CAPTURE_ALLOWED`). Called from the Cuda
/// `FusedOps::set_unified_mixed_capture` impl.
pub fn set_fa3_capture_allowed(on: bool) {
    FA3_CAPTURE_ALLOWED.store(on, Ordering::Relaxed);
}

unsafe extern "C" {
    fn launch_flash_attn_paged_decode_bf16(
        q: *const half::bf16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        workspace: *mut f32,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    );
    fn launch_flash_attn_paged_decode_fp16(
        q: *const half::f16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        workspace: *mut f32,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    );
    fn flash_attn_batched_decode_workspace_bytes(
        batch: i32,
        num_q_heads: i32,
        head_dim: i32,
    ) -> i64;
    fn launch_cudnn_paged_decode_bf16(
        handle: cudnnHandle_t,
        q: *const half::bf16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        q_lens: *const i32,
        kv_lens: *const i32,
        num_blocks: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    fn launch_cudnn_paged_decode_fp16(
        handle: cudnnHandle_t,
        q: *const half::f16,
        qsb: i64,
        qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16,
        osb: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        q_lens: *const i32,
        kv_lens: *const i32,
        num_blocks: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        batch: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> i32;

    fn launch_flash_attn_paged_ragged_cute_bf16(
        q: *const half::bf16,
        qss: i64,
        qsh: i64,
        k_pool: *const half::bf16,
        v_pool: *const half::bf16,
        o: *mut half::bf16,
        oss: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        block2req: *const i32,
        block2tile: *const i32,
        valid_q_tiles: *const i32,
        total_q_tiles: i32,
        batch: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: cudaStream_t,
    );
    fn launch_flash_attn_paged_ragged_cute_fp16(
        q: *const half::f16,
        qss: i64,
        qsh: i64,
        k_pool: *const half::f16,
        v_pool: *const half::f16,
        o: *mut half::f16,
        oss: i64,
        osh: i64,
        block_tables: *const u32,
        max_blocks_per_seq: i32,
        block_size: i32,
        kv_lens: *const i32,
        cu_q_lens: *const i32,
        block2req: *const i32,
        block2tile: *const i32,
        valid_q_tiles: *const i32,
        total_q_tiles: i32,
        batch: i32,
        total_q_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: cudaStream_t,
    );
}

const DISABLE_CUDNN_ATTENTION_ENV: &str = "RUSTINFER_DISABLE_CUDNN_ATTENTION";
const STRICT_CUDNN_ATTENTION_ENV: &str = "RUSTINFER_STRICT_CUDNN_ATTENTION";
const PREFILL_FA3_ENV: &str = "RUSTINFER_PREFILL_FA3";

#[cfg(rustinfer_fa3)]
unsafe extern "C" {
    fn rustinfer_fa3_varlen_paged_bf16_hd128(
        q: *const c_void,
        k_pool: *const c_void,
        v_pool: *const c_void,
        o: *mut c_void,
        softmax_lse: *mut f32,
        cu_seqlens_q: *const i32,
        seqused_k: *const i32,
        page_table: *const i32,
        page_table_batch_stride: i64,
        tile_count_semaphore: *mut i32,
        b: i32,
        max_seqlen_q: i32,
        max_pages_per_seq: i32,
        page_size: i32,
        num_pages: i32,
        q_extent: i32,
        h: i32,
        h_k: i32,
        q_row_stride: i64,
        q_head_stride: i64,
        k_row_stride: i64,
        k_head_stride: i64,
        k_page_stride: i64,
        v_row_stride: i64,
        v_head_stride: i64,
        v_page_stride: i64,
        o_row_stride: i64,
        o_head_stride: i64,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> i32;
}

/// Process-lifetime, per-device FA3 scratch: softmax LSE
/// (`head_num * q_extent` f32) and the tile-count semaphore. Grow-only, so
/// steady state issues zero allocations (a per-step `cudaMalloc` is the known
/// TTFT-regression shape).
///
/// A captured CUDA graph stores the LSE address passed to its FA3 node. Old
/// allocations therefore remain alive when a larger eager request grows the
/// current scratch; freeing them would leave every graph captured with the old
/// address holding a dangling pointer. Growth is rare and geometric in normal
/// workloads, so retaining those allocations until process exit is a small,
/// bounded trade for address stability.
#[cfg(rustinfer_fa3)]
struct Fa3Scratch {
    lse: *mut f32,
    lse_cap: usize,
    semaphore: *mut i32,
    retired_lse: Vec<*mut f32>,
}

#[cfg(rustinfer_fa3)]
unsafe impl Send for Fa3Scratch {}

#[cfg(rustinfer_fa3)]
static FA3_SCRATCH: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<i32, Fa3Scratch>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

#[cfg(rustinfer_fa3)]
fn fa3_scratch(head_num: usize, q_extent: usize) -> OpResult<(*mut f32, *mut i32)> {
    use crate::ffi;
    let device = crate::device_utils::current_device()?;
    let mut all_scratch = FA3_SCRATCH
        .lock()
        .map_err(|_| OpError::Kernel("FA3 scratch lock poisoned".into()))?;
    let scratch = all_scratch.entry(device).or_insert_with(|| Fa3Scratch {
        lse: std::ptr::null_mut(),
        lse_cap: 0,
        semaphore: std::ptr::null_mut(),
        retired_lse: Vec::new(),
    });
    if scratch.semaphore.is_null() {
        let mut p: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut p, std::mem::size_of::<i32>()) };
        if err != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!(
                "FA3 semaphore cudaMalloc failed: {:?}",
                err
            )));
        }
        scratch.semaphore = p as *mut i32;
    }
    // Floor at 8192 rows so the common config allocates exactly once.
    let need = head_num * q_extent;
    if scratch.lse_cap < need {
        let want = need.max(head_num * 8192);
        let mut p: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut p, want * std::mem::size_of::<f32>()) };
        if err != ffi::cudaError_cudaSuccess {
            return Err(OpError::Kernel(format!(
                "FA3 LSE scratch cudaMalloc({} f32) failed: {:?}",
                want, err
            )));
        }
        if !scratch.lse.is_null() {
            scratch.retired_lse.push(scratch.lse);
        }
        scratch.lse = p as *mut f32;
        scratch.lse_cap = want;
    }
    Ok((scratch.lse, scratch.semaphore))
}

/// FA3 serves ragged prefill launches outside CUDA graph capture unconditionally;
/// under capture it runs only when the runtime has raised `FA3_CAPTURE_ALLOWED`
/// (the mixed FA3-graph path, whose bucket plan bakes `max_q`/`b` to proven upper
/// bounds). Otherwise capture stays on the CuTe kernel, which holds correct at
/// padded bucket shapes through its device-side `valid_q_tiles` cutoff while FA3
/// trusts host-side `b`/`cu_seqlens`.
#[cfg(rustinfer_fa3)]
fn fa3_ragged_eligible<T: Dtype>(head_dim: usize, stream: cudaStream_t) -> bool {
    if T::DATA_TYPE != DataType::BF16 || head_dim != 128 {
        return false;
    }
    if std::env::var_os(PREFILL_FA3_ENV).is_some_and(|v| v == "0") {
        return false;
    }
    let mut status: crate::ffi::cudaStreamCaptureStatus = Default::default();
    let err = unsafe { crate::ffi::cudaStreamIsCapturing(stream, &mut status) };
    if err != crate::ffi::cudaError_cudaSuccess {
        return false;
    }
    // 0 == cudaStreamCaptureStatusNone. When capturing, only the opted-in mixed
    // FA3-graph region (bounded max_q/b) may launch FA3.
    status as u32 == 0 || FA3_CAPTURE_ALLOWED.load(Ordering::Relaxed)
}

#[cfg(not(rustinfer_fa3))]
fn fa3_ragged_eligible<T: Dtype>(_head_dim: usize, _stream: cudaStream_t) -> bool {
    false
}

/// Build/dtype/env eligibility for the unified FA3 ragged path, WITHOUT the
/// stream-capture check. This is the POLICY predicate the runtime consults to
/// pick the mixed-step mode (eager-FA3 vs bucketed mixed-graph replay) before
/// any forward is issued; the per-launch dispatch still goes through
/// `fa3_ragged_eligible`, whose capture check keeps captured graphs on CuTe.
#[cfg(rustinfer_fa3)]
pub fn fa3_unified_available<T: Dtype>(head_dim: usize) -> bool {
    T::DATA_TYPE == DataType::BF16
        && head_dim == 128
        && std::env::var_os(PREFILL_FA3_ENV).is_none_or(|v| v != "0")
}

#[cfg(not(rustinfer_fa3))]
pub fn fa3_unified_available<T: Dtype>(_head_dim: usize) -> bool {
    false
}

/// FA3 varlen + paged-KV forward over the whole ragged batch — one launch,
/// any row composition (q=1 decode rows included).
#[cfg(rustinfer_fa3)]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fa3_ragged<T: Dtype>(
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: &PagedAttentionPlan<'_>,
    q_stride_seq: i64,
    q_stride_head: i64,
    o_stride_seq: i64,
    o_stride_head: i64,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
    stream: cudaStream_t,
) -> OpResult<()> {
    let max_q = plan.q_lens[..plan.batch].iter().copied().max().unwrap_or(0);
    if plan.batch == 0 || max_q <= 0 {
        return Ok(());
    }
    let q_extent = q.shape().as_slice()[0];
    let num_pages = k_pool.shape().as_slice()[0];
    let (lse, semaphore) = fa3_scratch(head_num, q_extent)?;
    let kv_row = (kv_head_num * head_dim) as i64;
    let page_stride = plan.block_size as i64 * kv_row;
    let rc = unsafe {
        rustinfer_fa3_varlen_paged_bf16_hd128(
            q.data_ptr() as *const c_void,
            k_pool.data_ptr() as *const c_void,
            v_pool.data_ptr() as *const c_void,
            output.data_ptr_mut() as *mut c_void,
            lse,
            plan.cu_q_lens.data_ptr(),
            plan.kv_lens.data_ptr(),
            plan.block_tables.data_ptr(),
            plan.max_blocks_per_seq as i64,
            semaphore,
            plan.batch as i32,
            max_q,
            plan.max_blocks_per_seq as i32,
            plan.block_size as i32,
            num_pages as i32,
            q_extent as i32,
            head_num as i32,
            kv_head_num as i32,
            q_stride_seq,
            q_stride_head,
            kv_row,
            head_dim as i64,
            page_stride,
            kv_row,
            head_dim as i64,
            page_stride,
            o_stride_seq,
            o_stride_head,
            scale,
            stream,
        )
    };
    if rc != 0 {
        return Err(OpError::Kernel(format!(
            "FA3 varlen paged prefill failed: cudaError {rc}"
        )));
    }
    Ok(())
}

#[cfg(not(rustinfer_fa3))]
#[allow(clippy::too_many_arguments)]
unsafe fn launch_fa3_ragged<T: Dtype>(
    _q: &Tensor<T, Cuda>,
    _k_pool: &Tensor<T, Cuda>,
    _v_pool: &Tensor<T, Cuda>,
    _output: &mut Tensor<T, Cuda>,
    _plan: &PagedAttentionPlan<'_>,
    _q_stride_seq: i64,
    _q_stride_head: i64,
    _o_stride_seq: i64,
    _o_stride_head: i64,
    _head_num: usize,
    _kv_head_num: usize,
    _head_dim: usize,
    _scale: f32,
    _stream: cudaStream_t,
) -> OpResult<()> {
    Err(OpError::Kernel(
        "FA3 kernels not built for this arch".into(),
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagedAttentionKind {
    DecodeOnly,
    Ragged,
}

#[derive(Clone, Copy)]
pub struct PagedAttentionPlan<'a> {
    pub kind: PagedAttentionKind,
    pub num_tokens: usize,
    pub batch: usize,
    /// Per-request query lengths (host), in row order. The fused mixed batch is
    /// laid out decode-first, so the leading run of `1`s is the decode prefix and
    /// the remainder are prefill chunks. Used to split a `Ragged` batch's
    /// attention by row type (decode → cuDNN decode SDPA; prefill → cuDNN
    /// bottom-right-causal SDPA), keeping q=1 rows off the CuTe ragged kernel.
    pub q_lens: &'a [i32],
    pub block_tables: &'a Tensor<i32, Cuda>,
    pub cu_q_lens: &'a Tensor<i32, Cuda>,
    pub kv_lens: &'a Tensor<i32, Cuda>,
    pub seq_lens_step: &'a Tensor<i32, Cuda>,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
    pub block2req: &'a Tensor<i32, Cuda>,
    pub block2tile: &'a Tensor<i32, Cuda>,
    pub valid_q_tiles: &'a Tensor<i32, Cuda>,
    pub valid_suffix_q_tiles: &'a Tensor<i32, Cuda>,
    pub total_q_tiles: i32,
}

impl<'a> PagedAttentionPlan<'a> {
    pub fn from_v2(plan: &'a plan::BatchPlan, index: &'a KvIndexTensors<Cuda>) -> Self {
        let kind = match plan.kind {
            plan::BatchKind::DecodeOnly => PagedAttentionKind::DecodeOnly,
            plan::BatchKind::Ragged | plan::BatchKind::Spec { .. } => PagedAttentionKind::Ragged,
        };
        Self {
            kind,
            num_tokens: plan.num_tokens,
            batch: plan.batch,
            q_lens: &plan.q_lens,
            block_tables: &index.block_tables,
            cu_q_lens: &index.cu_q_lens,
            kv_lens: &index.kv_lens,
            seq_lens_step: &index.seq_lens_step,
            max_blocks_per_seq: plan.max_blocks_per_seq,
            block_size: plan.block_size,
            block2req: &index.block2req,
            block2tile: &index.block2tile,
            valid_q_tiles: &index.valid_q_tiles,
            valid_suffix_q_tiles: &index.valid_suffix_q_tiles,
            total_q_tiles: plan.total_q_tiles,
        }
    }
}

/// f32 element count needed by the paged-decode flash attention kernel for
/// a worst-case `(batch, num_q_heads, head_dim)`. Callers (`ForwardWorkspace`)
/// use this to size the long-lived attention scratch.
pub fn flash_decode_workspace_capacity_f32(
    batch: usize,
    num_q_heads: usize,
    head_dim: usize,
) -> usize {
    let bytes = unsafe {
        flash_attn_batched_decode_workspace_bytes(batch as i32, num_q_heads as i32, head_dim as i32)
    } as usize;
    bytes.div_ceil(4)
}

/// Launch the CuTe paged ragged Flash kernel over a (sub)set of q tiles. The
/// kernel indexes q / kv_lens / block_tables by the *absolute* request id held
/// in `block2req[tile]` (and `cu_q_lens[req]`), so a fused batch's prefill
/// suffix runs by passing the full base pointers but with `block2req` /
/// `block2tile` shifted past the leading decode tiles and `total_q_tiles`
/// reduced to match. Always causal (chunked-prefill bottom-right via the
/// kernel's `kv_len - q_len` mask shift).
#[allow(clippy::too_many_arguments)]
unsafe fn launch_cute_ragged<T: Dtype>(
    q_ptr: *const T,
    q_stride_seq: i64,
    q_stride_head: i64,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    o_ptr: *mut T,
    o_stride_seq: i64,
    o_stride_head: i64,
    block_tables_ptr: *const u32,
    max_blocks_per_seq: i32,
    block_size: i32,
    kv_lens_ptr: *const i32,
    cu_q_lens_ptr: *const i32,
    block2req_ptr: *const i32,
    block2tile_ptr: *const i32,
    valid_q_tiles_ptr: *const i32,
    total_q_tiles: i32,
    batch: i32,
    num_tokens: i32,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
    stream: cudaStream_t,
) -> OpResult<()> {
    unsafe {
        match T::DATA_TYPE {
            DataType::BF16 => launch_flash_attn_paged_ragged_cute_bf16(
                q_ptr as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                o_ptr as _,
                o_stride_seq,
                o_stride_head,
                block_tables_ptr,
                max_blocks_per_seq,
                block_size,
                kv_lens_ptr,
                cu_q_lens_ptr,
                block2req_ptr,
                block2tile_ptr,
                valid_q_tiles_ptr,
                total_q_tiles,
                batch,
                num_tokens,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                1,
                stream,
            ),
            DataType::F16 => launch_flash_attn_paged_ragged_cute_fp16(
                q_ptr as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                o_ptr as _,
                o_stride_seq,
                o_stride_head,
                block_tables_ptr,
                max_blocks_per_seq,
                block_size,
                kv_lens_ptr,
                cu_q_lens_ptr,
                block2req_ptr,
                block2tile_ptr,
                valid_q_tiles_ptr,
                total_q_tiles,
                batch,
                num_tokens,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                1,
                stream,
            ),
            _ => {
                return Err(OpError::Kernel(format!(
                    "attention_paged Ragged: dtype {:?}",
                    T::DATA_TYPE
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn try_cudnn_paged_decode<T: Dtype>(
    device: &Cuda,
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: PagedAttentionPlan<'_>,
    q_stride_seq: i64,
    q_stride_head: i64,
    o_stride_seq: i64,
    o_stride_head: i64,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
    batch: i32,
    stream: cudaStream_t,
) -> Option<i32> {
    if std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_some() {
        return None;
    }

    let num_blocks = k_pool
        .shape()
        .as_slice()
        .first()
        .copied()
        .unwrap_or_default() as i32;
    let kernel_workspace = device.config.kernel_workspace();
    let status = match T::DATA_TYPE {
        DataType::BF16 => unsafe {
            launch_cudnn_paged_decode_bf16(
                device.config.cudnn_handle,
                q.data_ptr() as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                output.data_ptr_mut() as _,
                o_stride_seq,
                o_stride_head,
                plan.block_tables.data_ptr() as *const u32,
                plan.max_blocks_per_seq as i32,
                plan.block_size as i32,
                plan.seq_lens_step.data_ptr(),
                plan.kv_lens.data_ptr(),
                num_blocks,
                kernel_workspace.ptr(),
                kernel_workspace.size(),
                batch,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                stream,
            )
        },
        DataType::F16 => unsafe {
            launch_cudnn_paged_decode_fp16(
                device.config.cudnn_handle,
                q.data_ptr() as _,
                q_stride_seq,
                q_stride_head,
                k_pool.data_ptr() as _,
                v_pool.data_ptr() as _,
                output.data_ptr_mut() as _,
                o_stride_seq,
                o_stride_head,
                plan.block_tables.data_ptr() as *const u32,
                plan.max_blocks_per_seq as i32,
                plan.block_size as i32,
                plan.seq_lens_step.data_ptr(),
                plan.kv_lens.data_ptr(),
                num_blocks,
                kernel_workspace.ptr(),
                kernel_workspace.size(),
                batch,
                head_num as i32,
                kv_head_num as i32,
                head_dim as i32,
                scale,
                stream,
            )
        },
        _ => return None,
    };
    Some(status)
}

pub fn attention_paged<T: Dtype>(
    stream: cudaStream_t,
    q: &Tensor<T, Cuda>,
    k_pool: &Tensor<T, Cuda>,
    v_pool: &Tensor<T, Cuda>,
    output: &mut Tensor<T, Cuda>,
    plan: PagedAttentionPlan<'_>,
    workspace: &mut Tensor<f32, Cuda>,
    head_num: usize,
    kv_head_num: usize,
    head_dim: usize,
    scale: f32,
) -> OpResult<()> {
    let device = q.device();
    let batch = plan.batch as i32;

    // Q / O layout: [num_tokens, num_q_heads * head_dim]; q may be a
    // strided view (zero-copy slice of a fused QKV buffer), so read its
    // row stride directly. O is always contiguous (workspace-owned).
    //   stride seq  = q's row stride (== num_q_heads * head_dim if contig)
    //   stride head = head_dim
    let q_stride_seq = q.strides().as_slice()[0] as i64;
    let q_stride_head = head_dim as i64;
    let o_stride_seq = (head_num * head_dim) as i64;
    let o_stride_head = head_dim as i64;

    // Pool layout: [num_blocks, block_size, kv_dim]; the kernels handle
    // their own indexing through block_tables + block_size, so we only pass
    // the base pointers.

    match plan.kind {
        PagedAttentionKind::DecodeOnly => {
            // Decode attention uses cuDNN SDPA by default (matches commit
            // 96f7b4e: correct + ~3.3x faster than the custom flash-decode
            // kernel). cuDNN IS CUDA-graph-capturable here — warmup builds &
            // caches the SDPA plan and capture reuses the cached plan (see
            // cudnn_paged_attention.cu), so warmup and the captured graph run the
            // SAME cuDNN kernel. Opt out (custom kernel) via env.
            let use_cudnn = std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_none();
            if use_cudnn
                && let Some(cudnn_status) = unsafe {
                    try_cudnn_paged_decode(
                        device,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        plan,
                        q_stride_seq,
                        q_stride_head,
                        o_stride_seq,
                        o_stride_head,
                        head_num,
                        kv_head_num,
                        head_dim,
                        scale,
                        batch,
                        stream,
                    )
                }
            {
                if cudnn_status == 0 {
                    return Ok(());
                }
                if std::env::var_os(STRICT_CUDNN_ATTENTION_ENV).is_some() {
                    return Err(OpError::Kernel(format!(
                        "cuDNN paged decode attention failed with status {}",
                        cudnn_status
                    )));
                }
            }

            // The caller pre-allocated `workspace` with at least
            // `flash_attn_batched_decode_workspace_bytes(cap_batch, head_num, head_dim)`
            // bytes. A too-small workspace would let the flash kernel write out
            // of bounds (silent device memory corruption), so validate ALWAYS —
            // computing `need` is a couple integer ops plus one cheap FFI call,
            // negligible next to the kernel launch, and worth it to fail loudly
            // instead of corrupting memory in release builds.
            {
                let need = unsafe {
                    flash_attn_batched_decode_workspace_bytes(
                        batch,
                        head_num as i32,
                        head_dim as i32,
                    )
                } as usize;
                let have = workspace.numel() * std::mem::size_of::<f32>();
                if have < need {
                    return Err(OpError::Kernel(format!(
                        "attention_paged decode workspace too small: have {} bytes, need {} \
                         (batch={}, head_num={}, head_dim={})",
                        have, need, batch, head_num, head_dim
                    )));
                }
            }

            unsafe {
                match T::DATA_TYPE {
                    DataType::BF16 => launch_flash_attn_paged_decode_bf16(
                        q.data_ptr() as _,
                        q_stride_seq,
                        q_stride_head,
                        k_pool.data_ptr() as _,
                        v_pool.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        o_stride_seq,
                        o_stride_head,
                        plan.block_tables.data_ptr() as *const u32,
                        plan.max_blocks_per_seq as i32,
                        plan.block_size as i32,
                        plan.kv_lens.data_ptr(),
                        workspace.data_ptr_mut(),
                        batch,
                        head_num as i32,
                        kv_head_num as i32,
                        head_dim as i32,
                        scale,
                        stream,
                    ),
                    DataType::F16 => launch_flash_attn_paged_decode_fp16(
                        q.data_ptr() as _,
                        q_stride_seq,
                        q_stride_head,
                        k_pool.data_ptr() as _,
                        v_pool.data_ptr() as _,
                        output.data_ptr_mut() as _,
                        o_stride_seq,
                        o_stride_head,
                        plan.block_tables.data_ptr() as *const u32,
                        plan.max_blocks_per_seq as i32,
                        plan.block_size as i32,
                        plan.kv_lens.data_ptr(),
                        workspace.data_ptr_mut(),
                        batch,
                        head_num as i32,
                        kv_head_num as i32,
                        head_dim as i32,
                        scale,
                        stream,
                    ),
                    _ => {
                        return Err(OpError::Kernel(format!(
                            "attention_paged DecodeOnly: dtype {:?}",
                            T::DATA_TYPE
                        )));
                    }
                }
            }
        }
        PagedAttentionKind::Ragged => {
            // FA3 (Hopper, eager) takes the WHOLE ragged batch — decode q=1
            // rows included — in one varlen+paged launch. The historical
            // decode/prefill split below exists only because the CuTe ragged
            // kernel runs q=1 rows at 1/128 tile utilization; FA3's persistent
            // varlen scheduler has no such penalty, so no rescue is needed.
            if fa3_ragged_eligible::<T>(head_dim, stream) {
                unsafe {
                    launch_fa3_ragged(
                        q,
                        k_pool,
                        v_pool,
                        output,
                        &plan,
                        q_stride_seq,
                        q_stride_head,
                        o_stride_seq,
                        o_stride_head,
                        head_num,
                        kv_head_num,
                        head_dim,
                        scale,
                        stream,
                    )?;
                }
                return Ok(());
            }

            // Legacy split path — non-Hopper builds, non-bf16/hd128 models,
            // RUSTINFER_PREFILL_FA3=0, and always inside CUDA graph capture
            // (FA3 trusts host-side b/cu_seqlens; only CuTe's device-side
            // valid_q_tiles cutoff stays correct at padded bucket shapes).
            //
            // The fused mixed batch is laid out decode-first, so the leading
            // run of q==1 rows is the decode prefix. Rescue it onto cuDNN's
            // fast decode SDPA and run the q>1 prefill suffix on the CuTe
            // ragged kernel: each decode row is one tile, so shifting
            // block2req/block2tile past the first `decode_count` tiles and
            // reducing total_q_tiles selects exactly the prefill tiles, and
            // the kernel's absolute block2req[tile] / cu_q_lens[req] indexing
            // keeps every lookup correct against the full base pointers.
            let cudnn_enabled = std::env::var_os(DISABLE_CUDNN_ATTENTION_ENV).is_none();
            let decode_count = plan.q_lens.iter().take_while(|&&q| q == 1).count();

            if cudnn_enabled && decode_count > 0 && decode_count < plan.batch {
                let dec = unsafe {
                    try_cudnn_paged_decode(
                        device,
                        q,
                        k_pool,
                        v_pool,
                        output,
                        plan,
                        q_stride_seq,
                        q_stride_head,
                        o_stride_seq,
                        o_stride_head,
                        head_num,
                        kv_head_num,
                        head_dim,
                        scale,
                        decode_count as i32,
                        stream,
                    )
                };
                if dec == Some(0) {
                    // decode prefix done on cuDNN → prefill suffix on CuTe.
                    let suffix_tiles = plan.total_q_tiles - decode_count as i32;
                    if suffix_tiles > 0 {
                        unsafe {
                            launch_cute_ragged(
                                q.data_ptr(),
                                q_stride_seq,
                                q_stride_head,
                                k_pool,
                                v_pool,
                                output.data_ptr_mut(),
                                o_stride_seq,
                                o_stride_head,
                                plan.block_tables.data_ptr() as *const u32,
                                plan.max_blocks_per_seq as i32,
                                plan.block_size as i32,
                                plan.kv_lens.data_ptr(),
                                plan.cu_q_lens.data_ptr(),
                                plan.block2req.data_ptr().add(decode_count),
                                plan.block2tile.data_ptr().add(decode_count),
                                plan.valid_suffix_q_tiles.data_ptr(),
                                suffix_tiles,
                                batch,
                                plan.num_tokens as i32,
                                head_num,
                                kv_head_num,
                                head_dim,
                                scale,
                                stream,
                            )?;
                        }
                    }
                    return Ok(());
                }
                if dec.is_some() && std::env::var_os(STRICT_CUDNN_ATTENTION_ENV).is_some() {
                    return Err(OpError::Kernel(
                        "cuDNN fused decode-prefix attention failed".into(),
                    ));
                }
                // else: decode rescue unavailable → full-batch CuTe ragged below.
            }

            // Full-batch CuTe ragged: no decode prefix, cuDNN disabled, or the
            // decode rescue failed (non-strict). Correct for any composition.
            unsafe {
                launch_cute_ragged(
                    q.data_ptr(),
                    q_stride_seq,
                    q_stride_head,
                    k_pool,
                    v_pool,
                    output.data_ptr_mut(),
                    o_stride_seq,
                    o_stride_head,
                    plan.block_tables.data_ptr() as *const u32,
                    plan.max_blocks_per_seq as i32,
                    plan.block_size as i32,
                    plan.kv_lens.data_ptr(),
                    plan.cu_q_lens.data_ptr(),
                    plan.block2req.data_ptr(),
                    plan.block2tile.data_ptr(),
                    plan.valid_q_tiles.data_ptr(),
                    plan.total_q_tiles,
                    batch,
                    plan.num_tokens as i32,
                    head_num,
                    kv_head_num,
                    head_dim,
                    scale,
                    stream,
                )?;
            }
        }
    }
    Ok(())
}
