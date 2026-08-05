// Torch-free C entry point for FA3 (flash-attention v2.8.3 hopper) varlen-q +
// paged-KV forward, bf16 hdim128, causal, GQA. Used for the prefill suffix of
// mixed batches; decode rows stay on the cuDNN SDPA path.
//
// cu_seqlens_q may start at a non-zero value: entries are absolute row offsets
// into the full packed q/o tensors (q_extent rows), which lets the caller pass
// a pointer into the middle of the batch's cu_q_lens without rebasing.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include "cutlass/numeric_types.h"
#include "flash.h"
#include "rustinfer_fa3_api.h"

extern "C" int rustinfer_fa3_varlen_paged_bf16_hd128(
    const void* q,                    // [q_extent, h, 128] bf16, full-batch base
    const void* k_pool,               // [num_pages, page_size, h_k, 128] bf16
    const void* v_pool,               // same layout as k_pool
    void* o,                          // [q_extent, h, 128] bf16, full-batch base
    float* softmax_lse,               // [h * q_extent] f32 scratch
    const int32_t* cu_seqlens_q,      // device [b+1], absolute row offsets into q
    const int32_t* seqused_k,         // device [b], KV len per sequence
    const int32_t* page_table,        // device [b, max_pages_per_seq] i32/u32
    int64_t page_table_batch_stride,  // elements
    int32_t* tile_count_semaphore,    // device int, re-zeroed here each call
    int b,
    int max_seqlen_q,
    int max_pages_per_seq,
    int page_size,
    int num_pages,
    int q_extent,
    int h,
    int h_k,
    int device_id,
    int num_sm,
    int64_t q_row_stride, int64_t q_head_stride,
    int64_t k_row_stride, int64_t k_head_stride, int64_t k_page_stride,
    int64_t v_row_stride, int64_t v_head_stride, int64_t v_page_stride,
    int64_t o_row_stride, int64_t o_head_stride,
    float softmax_scale,
    cudaStream_t stream) {
    constexpr int kHeadDim = 128;

    Flash_fwd_params params;
    std::memset(&params, 0, sizeof(params));

    params.q_ptr = const_cast<void*>(q);
    params.k_ptr = const_cast<void*>(k_pool);
    params.v_ptr = const_cast<void*>(v_pool);
    params.o_ptr = o;
    params.softmax_lse_ptr = softmax_lse;

    params.q_row_stride = q_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_row_stride = k_row_stride;
    params.k_head_stride = k_head_stride;
    params.k_batch_stride = k_page_stride;
    params.v_row_stride = v_row_stride;
    params.v_head_stride = v_head_stride;
    params.v_batch_stride = v_page_stride;
    params.v_dim_stride = 1;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;

    params.h = h;
    params.h_k = h_k;
    params.b = b;
    params.b_k = b;
    params.d = kHeadDim;
    params.dv = kHeadDim;
    params.d_rounded = kHeadDim;
    params.dv_rounded = kHeadDim;

    params.seqlen_q = max_seqlen_q;
    params.total_q = q_extent;
    params.seqlen_k = max_pages_per_seq * page_size;
    params.total_k = num_pages * page_size;
    params.seqlen_q_rounded = (max_seqlen_q + 127) / 128 * 128;
    params.seqlen_k_rounded = (params.seqlen_k + 127) / 128 * 128;

    params.cu_seqlens_q = const_cast<int32_t*>(cu_seqlens_q);
    params.seqused_k = const_cast<int32_t*>(seqused_k);

    params.page_table = const_cast<int32_t*>(page_table);
    params.page_table_batch_stride = page_table_batch_stride;
    params.page_size = page_size;
    params.num_pages = num_pages;
    params.pagedkv_tma = false;

    params.scale_softmax = softmax_scale;
    params.p_dropout = 1.f;
    params.rp_dropout = 1.f;

    params.is_causal = true;
    params.is_local = false;
    params.window_size_left = -1;
    params.window_size_right = 0;
    params.attention_chunk = 0;

    params.is_bf16 = true;
    params.num_splits = 1;
    params.pack_gqa = true;

    // VarlenDynamicPersistentTileScheduler consumes the semaphore via atomicAdd
    // and leaves it dirty; the prepare kernel that normally resets it is
    // skipped (num_splits_dynamic_ptr == nullptr), so re-zero it here. The
    // memset is stream-ordered and gets captured into CUDA graphs.
    params.tile_count_semaphore = tile_count_semaphore;
    params.num_splits_dynamic_ptr = nullptr;
    params.skip_scheduler_metadata_computation = true;

    params.arch = 90;
    params.device_id = device_id;
    params.num_sm = num_sm;

    cudaError_t err = cudaMemsetAsync(tile_count_semaphore, 0, sizeof(int32_t), stream);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    run_mha_fwd_<90, cutlass::bfloat16_t, kHeadDim, kHeadDim, false /*Split*/,
                 true /*PagedKVNonTMA*/, false /*Has_softcap*/, true /*PackGQA*/>(params, stream);

    return static_cast<int>(cudaGetLastError());
}
