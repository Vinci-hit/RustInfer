#pragma once

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// paged_kv_t - Paged KV cache following FlashInfer's CSR-style design
//
// CSR (Compressed Sparse Row) approach:
// - kv_indices: [nnz_tokens] physical slot indices for all tokens
// - kv_indptr:  [batch_size + 1] CSR row pointers
//   - kv_indptr[i] = start position in kv_indices for request i
//   - kv_indptr[batch_size] = nnz_tokens (total number of tokens)
//   - kv_len[i] = kv_indptr[i+1] - kv_indptr[i]
//
// KV layout: [total_slots, num_kv_heads, head_dim] (NHD style)
// - Eliminates integer division/modulo in kernel
// - Host pre-computes physical slot index for each token
// ============================================================================
template <typename DTypeKV_, typename IdType_>
struct paged_kv_t {
    using DTypeKV = DTypeKV_;
    using IdType = IdType_;

    // KV cache data pointers
    DTypeKV* k_data;                   // [total_slots, num_kv_heads, head_dim]
    DTypeKV* v_data;                   // [total_slots, num_kv_heads, head_dim]

    // CSR-style indexing (following FlashInfer convention)
    const IdType* kv_indices;          // [nnz_tokens] physical slot indices
    const IdType* kv_indptr;           // [batch_size + 1] CSR row pointers

    // Dimensions
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t batch_size;

    // Strides for flexible memory layout
    uint32_t stride_slot;              // stride between slots (typically num_kv_heads * head_dim)
    uint32_t stride_head;              // stride between heads (typically head_dim)

    // RoPE position offset per request (optional, for fused RoPE)
    const IdType* rope_pos_offset;     // [batch_size] start position of each request

    __host__ __device__ paged_kv_t()
        : k_data(nullptr),
          v_data(nullptr),
          kv_indices(nullptr),
          kv_indptr(nullptr),
          num_kv_heads(0),
          head_dim(0),
          batch_size(0),
          stride_slot(0),
          stride_head(0),
          rope_pos_offset(nullptr) {}

    __host__ __device__ paged_kv_t(
        DTypeKV* k, DTypeKV* v,
        const IdType* indices, const IdType* indptr,
        uint32_t kv_heads, uint32_t dim, uint32_t batch,
        const IdType* rope_offset = nullptr)
        : k_data(k),
          v_data(v),
          kv_indices(indices),
          kv_indptr(indptr),
          num_kv_heads(kv_heads),
          head_dim(dim),
          batch_size(batch),
          stride_slot(kv_heads * dim),
          stride_head(dim),
          rope_pos_offset(rope_offset) {}

    // Constructor with custom strides
    __host__ __device__ paged_kv_t(
        DTypeKV* k, DTypeKV* v,
        const IdType* indices, const IdType* indptr,
        uint32_t kv_heads, uint32_t dim, uint32_t batch,
        uint32_t s_slot, uint32_t s_head,
        const IdType* rope_offset = nullptr)
        : k_data(k),
          v_data(v),
          kv_indices(indices),
          kv_indptr(indptr),
          num_kv_heads(kv_heads),
          head_dim(dim),
          batch_size(batch),
          stride_slot(s_slot),
          stride_head(s_head),
          rope_pos_offset(rope_offset) {}

    // Get KV sequence length for a specific batch (CSR style)
    __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
        return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
    }

    // Get physical slot index for token at (batch_idx, seq_idx)
    __device__ __forceinline__ IdType get_slot(uint32_t batch_idx, uint32_t seq_idx) const {
        return __ldg(kv_indices + kv_indptr[batch_idx] + seq_idx);
    }

    // Get slot by absolute position in kv_indices array
    __device__ __forceinline__ IdType get_slot_by_iter(uint32_t iter_idx) const {
        return __ldg(kv_indices + iter_idx);
    }

    // Compute offset in KV buffer
    __device__ __forceinline__ size_t get_elem_offset(IdType slot, uint32_t head_idx, uint32_t feat_idx) const {
        return static_cast<size_t>(slot) * stride_slot + head_idx * stride_head + feat_idx;
    }

    // Get K pointer for a specific (batch, seq, head)
    __device__ __forceinline__ DTypeKV* get_k_ptr(uint32_t batch_idx, uint32_t seq_idx, uint32_t head_idx) const {
        IdType slot = get_slot(batch_idx, seq_idx);
        return k_data + get_elem_offset(slot, head_idx, 0);
    }

    // Get V pointer for a specific (batch, seq, head)
    __device__ __forceinline__ DTypeKV* get_v_ptr(uint32_t batch_idx, uint32_t seq_idx, uint32_t head_idx) const {
        IdType slot = get_slot(batch_idx, seq_idx);
        return v_data + get_elem_offset(slot, head_idx, 0);
    }

    // Get K/V pointer by iterator position (for tile-based iteration)
    __device__ __forceinline__ DTypeKV* get_k_ptr_by_iter(uint32_t iter_idx, uint32_t head_idx) const {
        IdType slot = get_slot_by_iter(iter_idx);
        return k_data + get_elem_offset(slot, head_idx, 0);
    }

    __device__ __forceinline__ DTypeKV* get_v_ptr_by_iter(uint32_t iter_idx, uint32_t head_idx) const {
        IdType slot = get_slot_by_iter(iter_idx);
        return v_data + get_elem_offset(slot, head_idx, 0);
    }

    // Protective access (bounds checking for partial tiles)
    __device__ __forceinline__ DTypeKV* protective_get_k_ptr(
        uint32_t iter_idx, uint32_t head_idx, uint32_t last_indptr) const {
        if (iter_idx < last_indptr) {
            return get_k_ptr_by_iter(iter_idx, head_idx);
        }
        return k_data;  // Return base pointer for out-of-bounds (will be masked)
    }

    __device__ __forceinline__ DTypeKV* protective_get_v_ptr(
        uint32_t iter_idx, uint32_t head_idx, uint32_t last_indptr) const {
        if (iter_idx < last_indptr) {
            return get_v_ptr_by_iter(iter_idx, head_idx);
        }
        return v_data;
    }

    // Get RoPE position offset for a request
    __device__ __forceinline__ IdType get_rope_offset(uint32_t batch_idx) const {
        return rope_pos_offset ? __ldg(rope_pos_offset + batch_idx) : 0;
    }
};

// ============================================================================
// BatchDecodeParams - Parameter struct following FlashInfer's design
//
// Designed for decode phase with paged KV cache:
// - Q: [batch_size, num_qo_heads, head_dim] or with custom strides
// - KV: paged_kv_t with CSR-style indexing
// - O: [batch_size, num_qo_heads, head_dim] or with custom strides
// - LSE: [batch_size, num_qo_heads] log-sum-exp for online softmax
//
// Supports:
// - GQA (Grouped Query Attention)
// - RoPE (fused or pre-applied)
// - Sliding window attention
// - Logits soft capping
// - KV partitioning for long sequences
// ============================================================================
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchDecodeParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    // Query tensor
    DTypeQ* q;                         // [batch_size, num_qo_heads, head_dim]
    const IdType* q_rope_offset;       // [batch_size] RoPE position offset for Q (optional)

    // Paged KV cache (CSR-style)
    paged_kv_t<DTypeKV, IdType> paged_kv;

    // Output tensor
    DTypeO* o;                         // [batch_size, num_qo_heads, head_dim]
    float* lse;                        // [batch_size, num_qo_heads] log-sum-exp (optional)

    // Alibi slopes (optional, for ALiBi position encoding)
    const float* alibi_slopes;         // [num_qo_heads] or nullptr

    // Dimensions
    uint32_t batch_size;
    uint32_t padded_batch_size;        // For aligned memory access
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;

    // Q/O strides (following FlashInfer's naming: stride_n = between tokens, stride_h = between heads)
    IdType q_stride_n;                 // Stride between batch elements in Q
    IdType q_stride_h;                 // Stride between heads in Q
    IdType o_stride_n;                 // Stride between batch elements in O
    IdType o_stride_h;                 // Stride between heads in O

    // Attention parameters
    float sm_scale;                    // Softmax scale: 1/sqrt(head_dim)
    float sm_scale_log2;               // sm_scale * log2(e) for fast exp2
    float logits_soft_cap;             // Soft cap for logits (0 = disabled)
    int32_t window_left;               // Sliding window size (-1 = disabled)

    // RoPE parameters (for fused RoPE)
    float rope_rcp_scale;              // 1.0 / rope_scale
    float rope_rcp_theta;              // 1.0 / rope_theta

    // KV partitioning for long sequences (following FlashInfer)
    IdType* request_indices;           // [num_tiles] which request each tile belongs to
    IdType* kv_tile_indices;           // [num_tiles] which KV tile within the request
    IdType* o_indptr;                  // [batch_size + 1] CSR for output accumulation
    IdType* kv_chunk_size_ptr;         // [1] chunk size for KV partitioning
    bool* block_valid_mask;            // [num_tiles] validity mask for tiles
    bool partition_kv;                 // Whether KV is partitioned

    __host__ __device__ BatchDecodeParams()
        : q(nullptr),
          q_rope_offset(nullptr),
          paged_kv(),
          o(nullptr),
          lse(nullptr),
          alibi_slopes(nullptr),
          batch_size(0),
          padded_batch_size(0),
          num_qo_heads(0),
          num_kv_heads(0),
          head_dim(0),
          q_stride_n(0),
          q_stride_h(0),
          o_stride_n(0),
          o_stride_h(0),
          sm_scale(0.0f),
          sm_scale_log2(0.0f),
          logits_soft_cap(0.0f),
          window_left(-1),
          rope_rcp_scale(1.0f),
          rope_rcp_theta(0.0f),
          request_indices(nullptr),
          kv_tile_indices(nullptr),
          o_indptr(nullptr),
          kv_chunk_size_ptr(nullptr),
          block_valid_mask(nullptr),
          partition_kv(false) {}

    // Simple constructor for common use case
    __host__ __device__ BatchDecodeParams(
        DTypeQ* q_ptr,
        paged_kv_t<DTypeKV, IdType> kv,
        DTypeO* o_ptr,
        uint32_t batch, uint32_t qo_heads, uint32_t kv_heads, uint32_t dim,
        float scale)
        : q(q_ptr),
          q_rope_offset(nullptr),
          paged_kv(kv),
          o(o_ptr),
          lse(nullptr),
          alibi_slopes(nullptr),
          batch_size(batch),
          padded_batch_size(batch),
          num_qo_heads(qo_heads),
          num_kv_heads(kv_heads),
          head_dim(dim),
          q_stride_n(qo_heads * dim),
          q_stride_h(dim),
          o_stride_n(qo_heads * dim),
          o_stride_h(dim),
          sm_scale(scale),
          sm_scale_log2(scale * float(M_LOG2E)),
          logits_soft_cap(0.0f),
          window_left(-1),
          rope_rcp_scale(1.0f),
          rope_rcp_theta(0.0f),
          request_indices(nullptr),
          kv_tile_indices(nullptr),
          o_indptr(nullptr),
          kv_chunk_size_ptr(nullptr),
          block_valid_mask(nullptr),
          partition_kv(false) {}

    // Full constructor with all options
    __host__ __device__ BatchDecodeParams(
        DTypeQ* q_ptr, const IdType* q_rope_off,
        paged_kv_t<DTypeKV, IdType> kv,
        DTypeO* o_ptr, float* lse_ptr,
        const float* alibi,
        uint32_t batch, uint32_t qo_heads, uint32_t kv_heads, uint32_t dim,
        IdType stride_q_n, IdType stride_q_h,
        IdType stride_o_n, IdType stride_o_h,
        float scale, float soft_cap, int32_t win_left,
        float rope_scale, float rope_theta)
        : q(q_ptr),
          q_rope_offset(q_rope_off),
          paged_kv(kv),
          o(o_ptr),
          lse(lse_ptr),
          alibi_slopes(alibi),
          batch_size(batch),
          padded_batch_size(batch),
          num_qo_heads(qo_heads),
          num_kv_heads(kv_heads),
          head_dim(dim),
          q_stride_n(stride_q_n),
          q_stride_h(stride_q_h),
          o_stride_n(stride_o_n),
          o_stride_h(stride_o_h),
          sm_scale(scale),
          sm_scale_log2(scale * float(M_LOG2E)),
          logits_soft_cap(soft_cap),
          window_left(win_left),
          rope_rcp_scale(1.0f / rope_scale),
          rope_rcp_theta(1.0f / rope_theta),
          request_indices(nullptr),
          kv_tile_indices(nullptr),
          o_indptr(nullptr),
          kv_chunk_size_ptr(nullptr),
          block_valid_mask(nullptr),
          partition_kv(false) {}

    // ==================== Accessors ====================

    // QO length is always 1 for decode
    __host__ __device__ __forceinline__ int32_t get_qo_len(int32_t batch_idx) const {
        return 1;
    }

    // KV length from paged_kv CSR
    __host__ __device__ __forceinline__ int32_t get_kv_len(int32_t batch_idx) const {
        return paged_kv.get_length(batch_idx);
    }

    // GQA group size
    __host__ __device__ __forceinline__ uint32_t get_gqa_group_size() const {
        return num_qo_heads / num_kv_heads;
    }

    // Get Q pointer for (batch_idx, head_idx)
    __device__ __forceinline__ DTypeQ* get_q_ptr(int32_t batch_idx, int32_t head_idx) const {
        return q + batch_idx * q_stride_n + head_idx * q_stride_h;
    }

    // Get O pointer for (batch_idx, head_idx)
    __device__ __forceinline__ DTypeO* get_o_ptr(int32_t batch_idx, int32_t head_idx) const {
        return o + batch_idx * o_stride_n + head_idx * o_stride_h;
    }

    // Get LSE pointer for (batch_idx, head_idx)
    __device__ __forceinline__ float* get_lse_ptr(int32_t batch_idx, int32_t head_idx) const {
        return lse ? lse + batch_idx * num_qo_heads + head_idx : nullptr;
    }

    // Get RoPE position for Q
    __device__ __forceinline__ IdType get_q_rope_pos(int32_t batch_idx) const {
        if (q_rope_offset) {
            return __ldg(q_rope_offset + batch_idx);
        }
        // Default: Q position = KV length (for autoregressive decode)
        return get_kv_len(batch_idx);
    }
};
