#pragma once

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// PagedKV - Paged KV cache with slot_mapping (host pre-computed addresses)
//
// slot_mapping approach: host pre-computes physical slot index for each token
// - Eliminates integer division/modulo in kernel
// - More flexible: supports non-contiguous token layouts
// - KV layout: [total_slots, num_kv_heads, head_dim]
// ============================================================================
template <typename DTypeKV_, typename IdType_>
struct PagedKV {
    using DTypeKV = DTypeKV_;
    using IdType = IdType_;

    DTypeKV* k_ptr;                    // [total_slots, num_kv_heads, head_dim]
    DTypeKV* v_ptr;                    // [total_slots, num_kv_heads, head_dim]
    const IdType* slot_mapping;        // [total_tokens] -> physical slot index
    const IdType* kv_indptr;           // [batch_size + 1] CSR-style, kv_indptr[i] = start of batch i in slot_mapping
    IdType num_kv_heads;
    IdType head_dim;

    __host__ __device__ PagedKV()
        : k_ptr(nullptr),
          v_ptr(nullptr),
          slot_mapping(nullptr),
          kv_indptr(nullptr),
          num_kv_heads(0),
          head_dim(0) {}

    __host__ __device__ PagedKV(DTypeKV* k, DTypeKV* v,
                                const IdType* slots, const IdType* indptr,
                                IdType kv_heads, IdType dim)
        : k_ptr(k),
          v_ptr(v),
          slot_mapping(slots),
          kv_indptr(indptr),
          num_kv_heads(kv_heads),
          head_dim(dim) {}

    // Get KV length for a specific batch
    __device__ __forceinline__ IdType get_kv_len(IdType batch_idx) const {
        return kv_indptr[batch_idx + 1] - kv_indptr[batch_idx];
    }

    // Get physical slot index for token at (batch_idx, seq_idx)
    __device__ __forceinline__ IdType get_slot(IdType batch_idx, IdType seq_idx) const {
        return slot_mapping[kv_indptr[batch_idx] + seq_idx];
    }

    // Get K pointer for a specific token and head
    // No division/modulo needed - host already computed the slot!
    __device__ __forceinline__ DTypeKV* get_k_ptr(IdType batch_idx, IdType seq_idx, IdType head_idx) const {
        IdType slot = get_slot(batch_idx, seq_idx);
        return k_ptr + slot * num_kv_heads * head_dim + head_idx * head_dim;
    }

    // Get V pointer for a specific token and head
    __device__ __forceinline__ DTypeKV* get_v_ptr(IdType batch_idx, IdType seq_idx, IdType head_idx) const {
        IdType slot = get_slot(batch_idx, seq_idx);
        return v_ptr + slot * num_kv_heads * head_dim + head_idx * head_dim;
    }

    // Get K pointer directly from slot index (useful when iterating over tiles)
    __device__ __forceinline__ DTypeKV* get_k_ptr_by_slot(IdType slot, IdType head_idx) const {
        return k_ptr + slot * num_kv_heads * head_dim + head_idx * head_dim;
    }

    __device__ __forceinline__ DTypeKV* get_v_ptr_by_slot(IdType slot, IdType head_idx) const {
        return v_ptr + slot * num_kv_heads * head_dim + head_idx * head_dim;
    }
};

// ============================================================================
// BatchDecodeParams - Simplified parameter struct for forward-only paged attention
//
// Designed for decode phase:
// - Q: [batch_size, q_seq_len, num_qo_heads, head_dim]  (q_seq_len typically 1)
// - KV: paged with slot_mapping
// - O: [batch_size, q_seq_len, num_qo_heads, head_dim]
// ============================================================================
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchDecodeParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    // Query tensor - layout: [batch_size, q_seq_len, num_qo_heads, head_dim]
    DTypeQ* q;
    IdType q_stride_batch;
    IdType q_stride_seq;
    IdType q_stride_head;

    // Paged KV cache with slot_mapping
    PagedKV<DTypeKV, IdType> paged_kv;

    // Output tensor - layout: [batch_size, q_seq_len, num_qo_heads, head_dim]
    DTypeO* o;
    IdType o_stride_batch;
    IdType o_stride_seq;
    IdType o_stride_head;

    // Dimensions
    IdType batch_size;
    IdType q_seq_len;          // Typically 1 for decode
    IdType num_qo_heads;
    IdType num_kv_heads;
    IdType head_dim;

    // Softmax scale: typically 1/sqrt(head_dim)
    float sm_scale;
    float sm_scale_log2;

    __host__ __device__ BatchDecodeParams()
        : q(nullptr),
          q_stride_batch(0),
          q_stride_seq(0),
          q_stride_head(0),
          paged_kv(),
          o(nullptr),
          o_stride_batch(0),
          o_stride_seq(0),
          o_stride_head(0),
          batch_size(0),
          q_seq_len(0),
          num_qo_heads(0),
          num_kv_heads(0),
          head_dim(0),
          sm_scale(0.0f),
          sm_scale_log2(0.0f) {}

    __host__ __device__ BatchDecodeParams(
        DTypeQ* q_ptr, IdType q_stride_b, IdType q_stride_s, IdType q_stride_h,
        PagedKV<DTypeKV, IdType> kv,
        DTypeO* o_ptr, IdType o_stride_b, IdType o_stride_s, IdType o_stride_h,
        IdType batch, IdType q_len,
        IdType qo_heads, IdType kv_heads, IdType dim,
        float scale)
        : q(q_ptr),
          q_stride_batch(q_stride_b),
          q_stride_seq(q_stride_s),
          q_stride_head(q_stride_h),
          paged_kv(kv),
          o(o_ptr),
          o_stride_batch(o_stride_b),
          o_stride_seq(o_stride_s),
          o_stride_head(o_stride_h),
          batch_size(batch),
          q_seq_len(q_len),
          num_qo_heads(qo_heads),
          num_kv_heads(kv_heads),
          head_dim(dim),
          sm_scale(scale),
          sm_scale_log2(scale * float(M_LOG2E)) {}

    __host__ __device__ __forceinline__ IdType get_qo_len(IdType batch_idx) const {
        return q_seq_len;
    }

    __host__ __device__ __forceinline__ IdType get_kv_len(IdType batch_idx) const {
        return paged_kv.get_kv_len(batch_idx);
    }

    __host__ __device__ __forceinline__ IdType get_gqa_group_size() const {
        return num_qo_heads / num_kv_heads;
    }

    __device__ __forceinline__ DTypeQ* get_q_ptr(IdType batch_idx, IdType seq_idx, IdType head_idx) const {
        return q + batch_idx * q_stride_batch + seq_idx * q_stride_seq + head_idx * q_stride_head;
    }

    __device__ __forceinline__ DTypeO* get_o_ptr(IdType batch_idx, IdType seq_idx, IdType head_idx) const {
        return o + batch_idx * o_stride_batch + seq_idx * o_stride_seq + head_idx * o_stride_head;
    }
};
