#include <stdint.h>
#include <cuda_runtime.h> // 包含 cudaStream_t 定义
#include <cuda_bf16.h>
#include <cublasLt.h>
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        }                                                                         \
    } while (0)
#ifdef __cplusplus
extern "C" {
#endif

// ======================================================================
// Flash-Attention / Flash-Decoding on CUDA: BF16 / FP16 only.
// Two kernels cover all attention paths:
//
//   1) launch_flash_attn_prefill_{bf16,fp16}
//      Stride-aware prefill (q_len > 1).  Arbitrary batch/stride layout,
//      head_dim ∈ {64, 128, 192, 256}.  See docs/FLASH_ATTN_PREFILL.md.
//
//   2) launch_flash_attn_batched_decode_{bf16,fp16}
//      Batched Flash-Decoding (q_len = 1) with split-KV.  Each request
//      carries its own KV cache pointed to by a device pointer array,
//      looked up via req_to_slot → graph-friendly.
//
// F32 is *not* supported on CUDA; run F32 attention on CPU.
// ======================================================================

// --- Prefill ---------------------------------------------------------------
void launch_flash_attn_prefill_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k, int64_t ksb, int64_t kss, int64_t ksh,
    const __nv_bfloat16* v, int64_t vsb, int64_t vss, int64_t vsh,
          __nv_bfloat16* o, int64_t osb, int64_t oss, int64_t osh,
    int batch, int q_len, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

void launch_flash_attn_prefill_fp16(
    const __half* q, int64_t qsb, int64_t qss, int64_t qsh,
    const __half* k, int64_t ksb, int64_t kss, int64_t ksh,
    const __half* v, int64_t vsb, int64_t vss, int64_t vsh,
          __half* o, int64_t osb, int64_t oss, int64_t osh,
    int batch, int q_len, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

// --- Ragged attention (variable q_len / kv_len per request) ---------------
// Packed Q / O layout:
//   q, o : [total_q_tokens, num_q_heads, head_dim]
//
// KV is per-request, accessed via a device pointer array like the decode op.
//
// All control arrays live on the device with stable addresses so the launch
// is CUDA-Graph-capturable:
//
//   req_to_slot[B]          — which KV-cache slot each request uses
//   kv_lens[B]              — current total KV length per request
//   cu_q_lens[B+1]          — prefix sum of q_len_i (Q tokens packed in order)
//   block2req[total_tiles]  — for each flattened (req, q_tile) slot, the req id
//   block2tile[total_tiles] — and the q-tile index within that request
//
//   total_q_tiles = Σ ceil(q_len_i / 128)  — Q-tile size is fixed at 128.
void launch_flash_attn_ragged_bf16(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* const* k_cache_ptrs,
    const __nv_bfloat16* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __nv_bfloat16* o, int64_t oss, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

void launch_flash_attn_ragged_fp16(
    const __half* q, int64_t qss, int64_t qsh,
    const __half* const* k_cache_ptrs,
    const __half* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __half* o, int64_t oss, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

// --- Batched Decode --------------------------------------------------------
// Shapes (logical):
//   q     : [batch, num_q_heads, head_dim]     (contiguous over last two)
//   o     : [batch, num_q_heads, head_dim]
//   K/V cache (per slot) : [max_seq_len, num_kv_heads, head_dim]
//
// Strides are in *elements*. kv_stride_s is the per-token stride of a single
// KV cache buffer; kv_stride_h is the per-kv-head stride (typically head_dim).
//
// `k_cache_ptrs` / `v_cache_ptrs` are device arrays (size = max_slots ≥ B)
// holding each slot's base pointer.  `req_to_slot[i]` tells the kernel
// which slot request i is currently occupying.
//
// `kv_lens[i]` is each request's current KV length (past + new).
//
// `workspace` must be at least `flash_attn_batched_decode_workspace_bytes`.
// Returns required workspace size in bytes for planning purposes.
int64_t flash_attn_batched_decode_workspace_bytes(
    int batch, int num_q_heads, int head_dim);

void launch_flash_attn_batched_decode_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qsh,
    const __nv_bfloat16* const* k_cache_ptrs,
    const __nv_bfloat16* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __nv_bfloat16* o, int64_t osb, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream);

void launch_flash_attn_batched_decode_fp16(
    const __half* q, int64_t qsb, int64_t qsh,
    const __half* const* k_cache_ptrs,
    const __half* const* v_cache_ptrs,
    int64_t kv_stride_s, int64_t kv_stride_h,
          __half* o, int64_t osb, int64_t osh,
    const int32_t* req_to_slot,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream);

void launch_flash_attn_paged_decode_bf16(
    const __nv_bfloat16* q, int64_t qsb, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream);

void launch_flash_attn_paged_decode_fp16(
    const __half* q, int64_t qsb, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    float* workspace,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream);

void launch_flash_attn_paged_ragged_bf16(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

void launch_flash_attn_paged_ragged_fp16(
    const __half* q, int64_t qss, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

void launch_flash_attn_paged_ragged_cute_bf16(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

void launch_flash_attn_paged_ragged_cute_fp16(
    const __half* q, int64_t qss, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t oss, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* kv_lens,
    const int32_t* cu_q_lens,
    const int32_t* block2req,
    const int32_t* block2tile,
    int total_q_tiles,
    int batch, int total_q_tokens,
    int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale, int is_causal,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
