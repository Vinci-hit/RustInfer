/*
 * Batch Prefill Parameters for Flash Attention (Forward-only, RadixTree-compatible)
 *
 * Simplified from FlashInfer design, removing:
 * - Backward pass support (lse, partition_kv, merge_indptr, etc.)
 * - Fused RoPE (rope_offset, rope_scale/theta) - use separate RoPE kernel
 * - Custom mask (maybe_custom_mask, mask_indptr) - use causal mask in kernel
 * - ALiBi (maybe_alibi_slopes) - Llama uses RoPE
 * - Logits soft cap (Gemma2 feature)
 * - Sliding window (window_left)
 * - Document masking (prefix_len, token_pos_in_items)
 *
 * RadixTree Support:
 * - Uses CSR-style kv_indices for token-level indexing (not block-based)
 * - Enables prefix sharing across requests in the same batch
 */
#ifndef BATCH_PREFILL_PARAMS_CUH_
#define BATCH_PREFILL_PARAMS_CUH_

#include <cuda_runtime.h>
#include <cstdint>

namespace flashinfer {

// =============================================================================
// paged_kv_t: Paged KV cache descriptor (CSR-style, RadixTree-compatible)
//
// This uses token-level CSR indexing instead of block-based page tables,
// which enables fine-grained prefix sharing via RadixTree.
// =============================================================================
template <typename DTypeKV_, typename IdType_>
struct paged_kv_prefill_t {
  using DTypeKV = DTypeKV_;
  using IdType = IdType_;

  // KV cache pool (all slots)
  DTypeKV* k_data;              // [total_slots, num_kv_heads, head_dim]
  DTypeKV* v_data;              // [total_slots, num_kv_heads, head_dim]

  // CSR-style indexing for RadixTree compatibility
  // kv_indices contains physical slot indices for all KV tokens
  // kv_indptr[i] to kv_indptr[i+1] gives the range for request i
  IdType* kv_indices;           // [nnz_kv_tokens] physical slot indices
  IdType* kv_indptr;            // [batch_size + 1] CSR row pointers

  // Dimensions
  uint32_t num_kv_heads;
  uint32_t head_dim;
  uint32_t batch_size;

  // Strides for accessing k_data/v_data
  uint32_t stride_slot;         // = num_kv_heads * head_dim
  uint32_t stride_head;         // = head_dim

  __host__ __device__ paged_kv_prefill_t()
      : k_data(nullptr), v_data(nullptr),
        kv_indices(nullptr), kv_indptr(nullptr),
        num_kv_heads(0), head_dim(0), batch_size(0),
        stride_slot(0), stride_head(0) {}

  __host__ __device__ paged_kv_prefill_t(
      DTypeKV* k_data, DTypeKV* v_data,
      IdType* kv_indices, IdType* kv_indptr,
      uint32_t num_kv_heads, uint32_t head_dim, uint32_t batch_size)
      : k_data(k_data), v_data(v_data),
        kv_indices(kv_indices), kv_indptr(kv_indptr),
        num_kv_heads(num_kv_heads), head_dim(head_dim), batch_size(batch_size),
        stride_slot(num_kv_heads * head_dim), stride_head(head_dim) {}

  // Get KV length for a specific batch item
  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
  }

  // Get element offset in k_data/v_data
  __host__ __device__ __forceinline__ uint32_t get_elem_offset(
      IdType slot, uint32_t head_idx, uint32_t dim_offset) const {
    return slot * stride_slot + head_idx * stride_head + dim_offset;
  }

  // Get physical slot for batch_idx's token at position token_idx
  __host__ __device__ __forceinline__ IdType get_slot(
      uint32_t batch_idx, uint32_t token_idx) const {
    return kv_indices[kv_indptr[batch_idx] + token_idx];
  }

  // Get K pointer for a specific slot and head
  __host__ __device__ __forceinline__ const DTypeKV* get_k_ptr(
      IdType slot, uint32_t head_idx) const {
    return k_data + slot * stride_slot + head_idx * stride_head;
  }

  // Get V pointer for a specific slot and head
  __host__ __device__ __forceinline__ const DTypeKV* get_v_ptr(
      IdType slot, uint32_t head_idx) const {
    return v_data + slot * stride_slot + head_idx * stride_head;
  }
};

// =============================================================================
// BatchPrefillPagedParams: Batch prefill with paged KV cache (RadixTree-compatible)
// =============================================================================
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchPrefillPagedParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  // Query tensor (ragged format)
  DTypeQ* q;                    // [total_q_tokens, num_qo_heads, head_dim]
  IdType* q_indptr;             // [batch_size + 1] Q sequence boundaries

  // Paged KV cache with CSR indexing
  paged_kv_prefill_t<DTypeKV, IdType> paged_kv;

  // Output tensor
  DTypeO* o;                    // [total_q_tokens, num_qo_heads, head_dim]

  // Dimensions
  uint32_t num_qo_heads;
  uint32_t head_dim;

  // Q strides
  uint32_t q_stride_n;
  uint32_t q_stride_h;

  // Attention scale
  float sm_scale;

  // GQA group size (num_qo_heads / num_kv_heads)
  uint32_t group_size;

  // Tiling info (for persistent kernel)
  IdType* request_indices;      // [num_tiles] which request each tile belongs to
  IdType* qo_tile_indices;      // [num_tiles] Q tile index within request
  uint32_t padded_batch_size;

  __host__ BatchPrefillPagedParams()
      : q(nullptr), q_indptr(nullptr), paged_kv(), o(nullptr),
        num_qo_heads(0), head_dim(0), q_stride_n(0), q_stride_h(0),
        sm_scale(0.0f), group_size(1),
        request_indices(nullptr), qo_tile_indices(nullptr), padded_batch_size(0) {}

  __host__ BatchPrefillPagedParams(
      DTypeQ* q, IdType* q_indptr,
      paged_kv_prefill_t<DTypeKV, IdType> paged_kv,
      DTypeO* o,
      uint32_t num_qo_heads, uint32_t head_dim,
      float sm_scale)
      : q(q), q_indptr(q_indptr), paged_kv(paged_kv), o(o),
        num_qo_heads(num_qo_heads), head_dim(head_dim),
        q_stride_n(num_qo_heads * head_dim), q_stride_h(head_dim),
        sm_scale(sm_scale),
        group_size(num_qo_heads / paged_kv.num_kv_heads),
        request_indices(nullptr), qo_tile_indices(nullptr), padded_batch_size(0) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {
    return q_indptr[batch_idx + 1] - q_indptr[batch_idx];
  }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return paged_kv.get_length(batch_idx);
  }
};

}  // namespace flashinfer

#endif  // BATCH_PREFILL_PARAMS_CUH_
