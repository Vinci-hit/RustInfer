//! Attention operators.
//!
//! Split into two orthogonal operators:
//!
//! - [`FlashAttnDecodeBatch`] — batched Flash-Decoding (`q_len = 1` per
//!   request, many requests per launch).  Split-KV internally, graph-ready.
//! - [`FlashAttnRagged`]      — ragged-batch prefill / chunked prefill
//!   (arbitrary `q_len_i`, `kv_len_i`).
//!
//! Models should not touch these two directly. They should use the thin
//! [`Attention`] facade, which dispatches based on an [`AttentionPlan`]
//! prepared by the runner / scheduler at step entry.

pub mod decode_batch;
pub mod ragged;

pub use decode_batch::FlashAttnDecodeBatch;
pub use ragged::FlashAttnRagged;

use std::ffi::c_void;

use crate::OpConfig;
use crate::base::error::Result;
use crate::tensor::Tensor;

// ============================================================================
//  Attention facade —— 模型层唯一看到的 attention 算子
// ============================================================================

/// 本 step 的 attention 类型。由 runner / scheduler 在 step 入口决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// 所有 seq `q_len == 1` —— 走 [`FlashAttnDecodeBatch`]（split-KV 优化）。
    DecodeOnly,
    /// 变长 `q_len_i` —— 走 [`FlashAttnRagged`]。
    /// 混合 batch（有些 seq q_len>1、有些 q_len==1）也归这里。
    Ragged,
    /// Paged KV decode path.
    PagedDecode,
    /// Paged KV ragged prefill / mixed path.
    PagedRagged,
}

/// 一次 forward step 的 attention 调度计划。
///
/// 所有字段都是**设备常驻、地址稳定**的裸指针（CUDA-Graph-capturable）。由
/// runner 在每个 step 入口一次性准备，随后**所有层共用**同一份 plan。
///
/// 字段语义与 [`FlashAttnDecodeBatch::forward`] / [`FlashAttnRagged::forward`]
/// 完全一致，此处只做聚合。
///
/// 注意：`k_cache_ptrs_dev` / `v_cache_ptrs_dev` 是整张
/// `[layer_num × max_batch_seqs]` u64 指针表的**起始地址**；模型每层调用
/// [`Attention::forward`] 时需要传入 `layer_idx` 与 `max_batch_seqs`，本 facade
/// 内部将这两张表按 `layer_idx * max_batch_seqs` 偏移到本层对应行起始处再下传给
/// kernel —— 因为 attention kernel（ragged / decode-batch）只接受按 slot 索引
/// 的"单层指针表"，与 scatter 的 `[layer × slot]` 大表索引语义不同。
pub struct AttentionPlan {
    pub kind: AttentionKind,

    // 两种 kind 都要的
    pub k_cache_ptrs_dev: *const *const c_void,
    pub v_cache_ptrs_dev: *const *const c_void,
    pub kv_stride_s: i64,
    pub kv_stride_h: i64,
    pub req_to_slot_dev: *const i32,
    pub kv_lens_dev: *const i32,
    /// 指针表中每层占用的列数（= `BatchWorkspace::max_batch_seqs`），用于按
    /// `layer_idx` 偏移定位本层对应行的起始指针。
    pub max_batch_seqs: usize,
    /// Current number of sequences in the batch.
    pub batch: usize,

    // DecodeOnly 专用
    pub workspace: *mut f32,

    // Ragged 专用
    pub cu_q_lens_dev: *const i32,
    pub block2req_dev: *const i32,
    pub block2tile_dev: *const i32,
    pub total_q_tiles: i32,

    // Paged KV 专用。地址由 BatchWorkspace 固定持有，Graph-capturable。
    pub paged_block_tables_dev: *const u32,
    pub paged_block_counts_dev: *const i32,
    pub paged_max_blocks_per_seq: usize,
    /// Host-side device addresses for each layer's global paged K/V pool.
    /// Values are stable after PagedKvPool initialization.
    pub paged_k_pool_ptrs: Vec<usize>,
    pub paged_v_pool_ptrs: Vec<usize>,
    pub paged_block_size: usize,
}

// 裸指针不自动 Send/Sync；此结构由单 runner 线程构造、传递给各层共用，
// 手动 mark 以便能嵌入 `ForwardCtx`。
unsafe impl Send for AttentionPlan {}
unsafe impl Sync for AttentionPlan {}

impl AttentionPlan {
    /// 占位 plan：所有指针为 null、`kind = DecodeOnly`、`total_q_tiles = 0`。
    ///
    /// 仅用于"ForwardCtx 已经构造、但 runner 尚未填充真正 plan"的过渡期。
    /// 一旦调用 [`Attention::forward`] 会解引用这些 null 指针 —— 属于预期的
    /// 崩溃（故意不兜底），用以提醒调用路径必须提供真实 plan。
    pub fn empty() -> Self {
        Self {
            kind: AttentionKind::DecodeOnly,
            k_cache_ptrs_dev: std::ptr::null(),
            v_cache_ptrs_dev: std::ptr::null(),
            kv_stride_s: 0,
            kv_stride_h: 0,
            req_to_slot_dev: std::ptr::null(),
            kv_lens_dev: std::ptr::null(),
            max_batch_seqs: 0,
            batch: 0,
            workspace: std::ptr::null_mut(),
            cu_q_lens_dev: std::ptr::null(),
            block2req_dev: std::ptr::null(),
            block2tile_dev: std::ptr::null(),
            total_q_tiles: 0,
            paged_block_tables_dev: std::ptr::null(),
            paged_block_counts_dev: std::ptr::null(),
            paged_max_blocks_per_seq: 0,
            paged_k_pool_ptrs: Vec::new(),
            paged_v_pool_ptrs: Vec::new(),
            paged_block_size: 0,
        }
    }
}

/// 模型层唯一的 attention 算子：持两条实现路径，按 [`AttentionPlan::kind`]
/// 分派。无论 decode 还是 prefill，模型里只调用一次 [`Attention::forward`]。
pub struct Attention {
    decode: FlashAttnDecodeBatch,
    ragged: FlashAttnRagged,
}

impl Attention {
    pub fn new(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<Self> {
        Ok(Self {
            decode: FlashAttnDecodeBatch::new(num_q_heads, num_kv_heads, head_dim)?,
            ragged: FlashAttnRagged::new(num_q_heads, num_kv_heads, head_dim, causal)?,
        })
    }

    /// 输入 / 输出：
    /// - `q` / `o` 均为 `[total_q_tokens, num_q_heads, head_dim]` 的 3D view。
    ///   - `DecodeOnly` 时 `total_q_tokens == batch`（每 seq 一个 token）。
    ///   - `Ragged` 时 `total_q_tokens == Σ q_len_i`。
    /// - `layer_idx`：本次 attention 对应的 transformer 层；本 facade 据此把
    ///   `plan.k_cache_ptrs_dev` / `plan.v_cache_ptrs_dev` 偏移到该层在
    ///   `[layer_num × max_batch_seqs]` 大表里的行起始处，下传给 kernel。
    ///
    /// # Safety
    /// `plan` 里所有裸指针在本调用期间必须有效；且 `plan.kind` 与 `q` 的布局
    /// 必须自洽（由 runner 保证）。`layer_idx` 与 `plan.max_batch_seqs` 必须
    /// 与 [`crate::worker::BatchWorkspace`] 中实际填表时使用的同一组数值。
    pub unsafe fn forward(
        &self,
        q: &Tensor,
        o: &mut Tensor,
        layer_idx: usize,
        plan: &AttentionPlan,
        cuda_cfg: Option<&OpConfig>,
    ) -> Result<()> {
        // 同一张 [layer_num × max_batch_seqs] u64 指针大表，按 layer_idx 偏移到
        // 本层对应行起始处。Kernel 端只接受按 slot 索引的"单层指针表"。
        let row = layer_idx
            .checked_mul(plan.max_batch_seqs)
            .ok_or_else(|| crate::base::error::Error::InvalidArgument(format!(
                "Attention::forward: layer_idx {} * max_batch_seqs {} overflows",
                layer_idx, plan.max_batch_seqs,
            )))?;
        let k_layer = unsafe { plan.k_cache_ptrs_dev.add(row) };
        let v_layer = unsafe { plan.v_cache_ptrs_dev.add(row) };
        match plan.kind {
            AttentionKind::DecodeOnly => unsafe {
                self.decode.forward(
                    q,
                    k_layer,
                    v_layer,
                    plan.kv_stride_s,
                    plan.kv_stride_h,
                    plan.req_to_slot_dev,
                    plan.kv_lens_dev,
                    plan.workspace,
                    o,
                    cuda_cfg,
                )
            },
            AttentionKind::Ragged => unsafe {
                self.ragged.forward(
                    q,
                    k_layer,
                    v_layer,
                    plan.kv_stride_s,
                    plan.kv_stride_h,
                    plan.req_to_slot_dev,
                    plan.kv_lens_dev,
                    plan.cu_q_lens_dev,
                    plan.block2req_dev,
                    plan.block2tile_dev,
                    plan.total_q_tiles,
                    o,
                    cuda_cfg,
                )
            },
            AttentionKind::PagedDecode => unsafe {
                let k_pool = *plan.paged_k_pool_ptrs.get(layer_idx).ok_or_else(|| {
                    crate::base::error::Error::InvalidArgument(format!(
                        "PagedDecode missing K pool pointer for layer {}", layer_idx
                    ))
                })? as *const c_void;
                let v_pool = *plan.paged_v_pool_ptrs.get(layer_idx).ok_or_else(|| {
                    crate::base::error::Error::InvalidArgument(format!(
                        "PagedDecode missing V pool pointer for layer {}", layer_idx
                    ))
                })? as *const c_void;
                self.decode.forward_paged(
                    q,
                    k_pool,
                    v_pool,
                    plan.paged_block_tables_dev,
                    plan.paged_max_blocks_per_seq,
                    plan.paged_block_size,
                    plan.kv_lens_dev,
                    o,
                    cuda_cfg,
                )
            },
            AttentionKind::PagedRagged => unsafe {
                let k_pool = *plan.paged_k_pool_ptrs.get(layer_idx).ok_or_else(|| {
                    crate::base::error::Error::InvalidArgument(format!(
                        "PagedRagged missing K pool pointer for layer {}", layer_idx
                    ))
                })? as *const c_void;
                let v_pool = *plan.paged_v_pool_ptrs.get(layer_idx).ok_or_else(|| {
                    crate::base::error::Error::InvalidArgument(format!(
                        "PagedRagged missing V pool pointer for layer {}", layer_idx
                    ))
                })? as *const c_void;
                self.ragged.forward_paged(
                    q,
                    k_pool,
                    v_pool,
                    plan.paged_block_tables_dev,
                    plan.paged_max_blocks_per_seq,
                    plan.paged_block_size,
                    plan.kv_lens_dev,
                    plan.cu_q_lens_dev,
                    plan.block2req_dev,
                    plan.block2tile_dev,
                    plan.total_q_tiles,
                    plan.batch,
                    o,
                    cuda_cfg,
                )
            },
        }
    }
}
