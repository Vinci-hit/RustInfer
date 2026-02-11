/*
 * FlashInfer Rust Wrapper - Stub Implementation
 *
 * This file provides a C-compatible interface for FlashInfer's BatchDecode kernels.
 * Currently implements stub functions for testing the FFI bindings.
 * TODO: Integrate with actual FlashInfer library.
 *
 * Architecture: Init -> Plan -> Run lifecycle
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

// ============================================================================
// Plan Info Structure
// ============================================================================

struct FlashInferPlanInfo {
    int64_t padded_batch_size;
    int64_t v_offset;
    int64_t s_offset;
    int64_t request_indices_offset;
    int64_t kv_tile_indices_offset;
    int64_t o_indptr_offset;
    int64_t block_valid_mask_offset;
    int64_t kv_chunk_size_ptr_offset;
    bool enable_cuda_graph;
    bool split_kv;
};

// ============================================================================
// Context Structure - Holds all state for a batch decode handler
// ============================================================================

struct FlashInferContext {
    // Plan information
    FlashInferPlanInfo plan_info;

    // Configuration
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t page_size;
    uint32_t head_dim;
    float sm_scale;

    // Workspace pointers
    void* float_workspace;
    size_t float_workspace_size;
    void* int_workspace;
    size_t int_workspace_size;
    void* pinned_workspace;
    size_t pinned_workspace_size;

    // Is plan valid?
    bool plan_valid;
};

// ============================================================================
// C API - Extern functions for Rust FFI
// ============================================================================

extern "C" {

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

FlashInferContext* flashinfer_batch_decode_create() {
    auto* ctx = new FlashInferContext();
    ctx->plan_valid = false;
    ctx->num_qo_heads = 0;
    ctx->num_kv_heads = 0;
    ctx->page_size = 0;
    ctx->head_dim = 128;
    ctx->sm_scale = 0.0f;
    ctx->float_workspace = nullptr;
    ctx->int_workspace = nullptr;
    ctx->pinned_workspace = nullptr;
    return ctx;
}

void flashinfer_batch_decode_destroy(FlashInferContext* ctx) {
    if (ctx) {
        delete ctx;
    }
}

// ----------------------------------------------------------------------------
// Plan Phase
// ----------------------------------------------------------------------------

int flashinfer_batch_decode_plan(
    FlashInferContext* ctx,
    void* float_workspace,
    size_t float_ws_size,
    void* int_workspace,
    size_t int_ws_size,
    void* pinned_workspace,
    size_t pinned_ws_size,
    const int32_t* indptr_h,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t page_size,
    uint32_t head_dim,
    bool enable_cuda_graph,
    cudaStream_t stream
) {
    if (!ctx) return -1;
    if (head_dim != 128) {
        // Currently only support HeadDim=128
        return -2;
    }

    // Store configuration
    ctx->num_qo_heads = num_qo_heads;
    ctx->num_kv_heads = num_kv_heads;
    ctx->page_size = page_size;
    ctx->head_dim = head_dim;
    ctx->sm_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    ctx->float_workspace = float_workspace;
    ctx->float_workspace_size = float_ws_size;
    ctx->int_workspace = int_workspace;
    ctx->int_workspace_size = int_ws_size;
    ctx->pinned_workspace = pinned_workspace;
    ctx->pinned_workspace_size = pinned_ws_size;

    // Initialize plan info
    ctx->plan_info.padded_batch_size = batch_size;
    ctx->plan_info.enable_cuda_graph = enable_cuda_graph;
    ctx->plan_info.split_kv = false;
    ctx->plan_info.v_offset = 0;
    ctx->plan_info.s_offset = 0;
    ctx->plan_info.request_indices_offset = 0;
    ctx->plan_info.kv_tile_indices_offset = 0;
    ctx->plan_info.o_indptr_offset = 0;
    ctx->plan_info.block_valid_mask_offset = 0;
    ctx->plan_info.kv_chunk_size_ptr_offset = 0;

    ctx->plan_valid = true;
    return 0;
}

// ----------------------------------------------------------------------------
// Run Phase - Stub implementations
// ----------------------------------------------------------------------------

// Simple attention kernel for testing (not optimized)
__global__ void batch_decode_attention_kernel_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ o,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ last_page_len,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size,
    float sm_scale
) {
    // Stub kernel - just copy q to o for testing
    uint32_t batch_idx = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t dim_idx = threadIdx.x;

    if (batch_idx < batch_size && head_idx < num_qo_heads && dim_idx < head_dim) {
        uint32_t idx = (batch_idx * num_qo_heads + head_idx) * head_dim + dim_idx;
        o[idx] = q[idx];
    }
}

__global__ void batch_decode_attention_kernel_fp16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ o,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ last_page_len,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size,
    float sm_scale
) {
    // Stub kernel - just copy q to o for testing
    uint32_t batch_idx = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t dim_idx = threadIdx.x;

    if (batch_idx < batch_size && head_idx < num_qo_heads && dim_idx < head_dim) {
        uint32_t idx = (batch_idx * num_qo_heads + head_idx) * head_dim + dim_idx;
        o[idx] = q[idx];
    }
}

int flashinfer_batch_decode_run_bf16(
    FlashInferContext* ctx,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* o,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* last_page_len,
    uint32_t batch_size,
    cudaStream_t stream
) {
    if (!ctx || !ctx->plan_valid) return -1;

    dim3 grid(batch_size, ctx->num_qo_heads);
    dim3 block(ctx->head_dim);

    batch_decode_attention_kernel_bf16<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q),
        static_cast<const __nv_bfloat16*>(k_cache),
        static_cast<const __nv_bfloat16*>(v_cache),
        static_cast<__nv_bfloat16*>(o),
        kv_indices,
        kv_indptr,
        last_page_len,
        batch_size,
        ctx->num_qo_heads,
        ctx->num_kv_heads,
        ctx->head_dim,
        ctx->page_size,
        ctx->sm_scale
    );

    return cudaGetLastError();
}

int flashinfer_batch_decode_run_fp16(
    FlashInferContext* ctx,
    const void* q,
    const void* k_cache,
    const void* v_cache,
    void* o,
    const int32_t* kv_indices,
    const int32_t* kv_indptr,
    const int32_t* last_page_len,
    uint32_t batch_size,
    cudaStream_t stream
) {
    if (!ctx || !ctx->plan_valid) return -1;

    dim3 grid(batch_size, ctx->num_qo_heads);
    dim3 block(ctx->head_dim);

    batch_decode_attention_kernel_fp16<<<grid, block, 0, stream>>>(
        static_cast<const __half*>(q),
        static_cast<const __half*>(k_cache),
        static_cast<const __half*>(v_cache),
        static_cast<__half*>(o),
        kv_indices,
        kv_indptr,
        last_page_len,
        batch_size,
        ctx->num_qo_heads,
        ctx->num_kv_heads,
        ctx->head_dim,
        ctx->page_size,
        ctx->sm_scale
    );

    return cudaGetLastError();
}

// ----------------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------------

void flashinfer_get_workspace_sizes(
    uint32_t max_batch_size,
    uint32_t max_seq_len,
    uint32_t num_qo_heads,
    uint32_t head_dim,
    size_t* float_workspace_size,
    size_t* int_workspace_size,
    size_t* pinned_workspace_size
) {
    // Conservative workspace estimates
    size_t max_padded = max_batch_size * 4;

    // Float workspace: for tmp_v and tmp_s
    *float_workspace_size = max_padded * num_qo_heads * head_dim * sizeof(float) +
                            max_padded * num_qo_heads * sizeof(float);
    *float_workspace_size = ((*float_workspace_size + 255) / 256) * 256;

    // Int workspace: indices and masks
    *int_workspace_size = max_padded * sizeof(int32_t) * 4 +
                          max_padded * sizeof(bool) + 256;
    *int_workspace_size = ((*int_workspace_size + 255) / 256) * 256;

    // Pinned workspace: mirrors int workspace
    *pinned_workspace_size = *int_workspace_size;
}

bool flashinfer_is_plan_valid(FlashInferContext* ctx) {
    return ctx && ctx->plan_valid;
}

void flashinfer_get_plan_info(
    FlashInferContext* ctx,
    int64_t* padded_batch_size,
    bool* split_kv,
    bool* enable_cuda_graph
) {
    if (ctx && ctx->plan_valid) {
        *padded_batch_size = ctx->plan_info.padded_batch_size;
        *split_kv = ctx->plan_info.split_kv;
        *enable_cuda_graph = ctx->plan_info.enable_cuda_graph;
    } else {
        *padded_batch_size = 0;
        *split_kv = false;
        *enable_cuda_graph = false;
    }
}

} // extern "C"
