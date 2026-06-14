//! Batch plan + paged KV pool.
//!
//! All KV addressing is paged: a single shared pool per layer holds
//! `[num_blocks, block_size, kv_dim]` K/V tensors; each sequence carries a
//! `block_table` (a list of physical block ids) and a current `kv_len`.
//! Token `t` of seq i lives at `pool[block_table[i][t / block_size]][t % block_size]`.
//!
//! This matches the scheduler's paged abstraction (`PrefillSegmentMeta`'s
//! `block_table` + `block_size`) and the underlying CUDA kernels
//! (`scatter_kv_paged_*`, `launch_flash_attn_paged_ragged_cute_*`,
//! `launch_flash_attn_paged_decode_*`).

use super::ports::MemoryPort;
use super::tensor::Tensor;
use super::types::Dtype;

/// Whether every sequence in the batch has `q_len == 1`. Drives the
/// attention dispatch: decode-only paths take Flash-Decoding (split-KV) on
/// the paged pool; mixed / pure-prefill paths take the ragged paged kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    DecodeOnly,
    Ragged,
}

/// Q-tile size used by the ragged paged attention kernel. Must match
/// `kBlockM` in `flash_attn_paged_prefill.cu`.
pub const RAGGED_Q_TILE: i32 = 128;

/// One forward step's batch plan — all device-resident metadata.
///
/// All tensors are i32/u32 device tensors freshly built by the runner per
/// step; Phase 4 (serve_loop) will lift them into long-lived workspaces.
pub struct BatchPlan<D: MemoryPort> {
    pub kind: BatchKind,
    /// Total number of Q tokens this step (= sum of per-seq `q_len`).
    pub num_tokens: usize,
    /// Number of sequences participating this step.
    pub batch: usize,

    /// `[num_tokens]` — absolute RoPE position for each token.
    pub rope_positions: Tensor<i32, D>,
    /// `[batch + 1]` — prefix sum of per-seq `q_len`.
    pub cu_q_lens: Tensor<i32, D>,
    /// `[batch]` — KV length AFTER this step writes.
    pub kv_lens: Tensor<i32, D>,
    /// `[batch]` — `seq_positions[i]` is the FIRST cache row this step writes
    /// for seq i (== current kv_len BEFORE the step).
    pub seq_positions: Tensor<i32, D>,
    /// `[batch]` — number of tokens this step writes per seq (== q_len[i]).
    pub seq_lens_step: Tensor<i32, D>,

    /// `[batch, max_blocks_per_seq]` — physical block ids per seq, padded
    /// with zeros. Each block holds `block_size` tokens of K/V.
    pub block_tables: Tensor<i32, D>,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,

    // Ragged-only schedule (DecodeOnly leaves these as length-1 placeholders).
    pub block2req: Tensor<i32, D>,
    pub block2tile: Tensor<i32, D>,
    pub total_q_tiles: i32,
}

impl<D: MemoryPort> BatchPlan<D> {
    /// Host-side helper: compute `cu_q_lens` (prefix sum) and the q-tile
    /// schedule (`block2req`, `block2tile`) from per-seq q_lens.
    pub fn plan_ragged_tiles(q_lens: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let mut cu = Vec::with_capacity(q_lens.len() + 1);
        cu.push(0i32);
        let mut acc = 0i32;
        for &q in q_lens {
            acc += q;
            cu.push(acc);
        }
        let mut b2r = Vec::new();
        let mut b2t = Vec::new();
        for (req, &q) in q_lens.iter().enumerate() {
            let n_tiles = (q + RAGGED_Q_TILE - 1) / RAGGED_Q_TILE;
            for t in 0..n_tiles {
                b2r.push(req as i32);
                b2t.push(t);
            }
        }
        (cu, b2r, b2t)
    }
}

/// One transformer layer's slice of the global paged KV pool.
///
/// `k` / `v` shape = `[num_blocks, block_size, kv_dim]`. The same `num_blocks`
/// and `block_size` are shared across all layers and sequences; each seq
/// owns a list of physical block ids (its `block_table`).
pub struct PagedKvLayer<T: Dtype, D: MemoryPort> {
    pub k: Tensor<T, D>,
    pub v: Tensor<T, D>,
}

/// Worker-owned paged KV pool. Owned by `ModelRunner`; borrowed by `forward`
/// through `ForwardContext`.
pub struct PagedKvPool<T: Dtype, D: MemoryPort> {
    pub layers: Vec<PagedKvLayer<T, D>>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub kv_dim: usize,
}

impl<T: Dtype, D: MemoryPort> PagedKvPool<T, D> {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
