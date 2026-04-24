#include <cuda_fp16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include "flash_attn_gqa.h"

// Async copy macros
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),     \
        "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// head_dim = 128
// 256 threads = 16 groups × 16 lanes
// Each lane handles 8 bf16 elements (1 float4 = 16 bytes)
// 16 lanes × 8 = 128 dimensions
#define HD128 128
#define BN128 16
#define THREADS_PER_KEY128 16
#define NUM_GROUPS128 16  // 256 / 16

__global__ void flash_decode_fp16_hdim128_kernel(
    const __half* __restrict__ Q,  // [num_q_heads, 128]
    const __half* __restrict__ K,  // [Seq_Len, num_kv_heads, 128]
    const __half* __restrict__ V,  // [Seq_Len, num_kv_heads, 128]
    __half* __restrict__ O,        // [num_q_heads, 128]
    int32_t* seq_len_ptr,
    int num_kv_heads,
    int group_size,
    float sm_scale
) {
    int32_t seq_len = *seq_len_ptr + 1;
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / group_size;

    // Shared memory layout:
    // s_q:   128 bf16 = 256 bytes
    // s_k:   2 stages × 16 tokens × 128 dims × 2 bytes = 8192 bytes
    // s_v:   2 stages × 16 tokens × 128 dims × 2 bytes = 8192 bytes
    // s_m:   16 floats = 64 bytes
    // s_s:   16 floats = 64 bytes
    // s_acc: 16 groups × 128 dims × 4 bytes = 8192 bytes
    // Total: ~25 KB
    extern __shared__ __half smem[];
    __half* s_q = smem;
    __half* s_k = s_q + HD128;
    __half* s_v = s_k + (2 * BN128 * HD128);

    float* s_m = (float*)(s_v + (2 * BN128 * HD128));
    float* s_s = s_m + NUM_GROUPS128;
    float* s_acc = s_s + NUM_GROUPS128;

    int tid = threadIdx.x;  // 0-255
    int gid = tid / THREADS_PER_KEY128;  // Group ID (0-15)
    int lane = tid % THREADS_PER_KEY128; // Lane ID (0-15), handles dims lane*8 to lane*8+7

    // 1. Load Q into shared memory
    // 256 threads, need to load 128 bf16 = 16 float4. Only first 16 threads do loads.
    if (tid < 16) {
        reinterpret_cast<float4*>(s_q)[tid] =
            reinterpret_cast<const float4*>(Q + q_head_idx * HD128)[tid];
    }

    // Register state for online softmax
    float row_max = -1e20f;
    float row_sum = 0.0f;
    float acc[8] = {0.0f};

    auto get_smem_ptr = [](__half* p) -> uint64_t {
        return (uint64_t)__cvta_generic_to_shared(p);
    };

    // 2. Software-pipelined K, V loading
    auto fetch_kv = [&](int iter, int stage) {
        int token_idx = iter * BN128 + gid;
        if (token_idx < seq_len) {
            const __half* k_ptr = K + (token_idx * num_kv_heads + kv_head_idx) * HD128 + lane * 8;
            const __half* v_ptr = V + (token_idx * num_kv_heads + kv_head_idx) * HD128 + lane * 8;
            __half* sk_dst = &s_k[(stage * BN128 + gid) * HD128 + lane * 8];
            __half* sv_dst = &s_v[(stage * BN128 + gid) * HD128 + lane * 8];

            CP_ASYNC_CG(get_smem_ptr(sk_dst), k_ptr, 16);
            CP_ASYNC_CG(get_smem_ptr(sv_dst), v_ptr, 16);
        }
    };

    // Pipeline warm-up
    fetch_kv(0, 0);
    CP_ASYNC_COMMIT_GROUP();

    for (int i = 0; i < seq_len; i += BN128) {
        int cur_stage = (i / BN128) % 2;
        int next_stage = 1 - cur_stage;

        if (i + BN128 < seq_len) {
            fetch_kv(i / BN128 + 1, next_stage);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        int current_token_idx = i + gid;
        if (current_token_idx < seq_len) {
            __half* sk_local = &s_k[(cur_stage * BN128 + gid) * HD128];
            __half* sv_local = &s_v[(cur_stage * BN128 + gid) * HD128];

            // A. Compute Q · K dot product
            float score = 0.0f;
            float4 q_vec = reinterpret_cast<float4*>(s_q)[lane];
            float4 k_vec = reinterpret_cast<float4*>(sk_local)[lane];

            half2* q_p2 = reinterpret_cast<half2*>(&q_vec);
            half2* k_p2 = reinterpret_cast<half2*>(&k_vec);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                half2 res = __hmul2(q_p2[j], k_p2[j]);
                score += __low2float(res) + __high2float(res);
            }

            // B. Reduction within group (16 threads -> 1 score)
            #pragma unroll
            for (int offset = 8; offset > 0; offset /= 2) {
                score += __shfl_xor_sync(0xffff, score, offset);
            }
            score *= sm_scale;

            // C. Online softmax update
            float old_max = row_max;
            row_max = max(row_max, score);
            float exp_scale = expf(old_max - row_max);
            float p = expf(score - row_max);
            row_sum = row_sum * exp_scale + p;

            // D. Accumulate V
            float4 v_vec = reinterpret_cast<float4*>(sv_local)[lane];
            __half* v_p = reinterpret_cast<__half*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                acc[d] = acc[d] * exp_scale + p * __half2float(v_p[d]);
            }
        }
    }

    // 3. Block-level final reduction (merge 16 groups)
    if (lane == 0) {
        s_m[gid] = row_max;
        s_s[gid] = row_sum;
    }
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        s_acc[gid * HD128 + lane * 8 + d] = acc[d];
    }
    __syncthreads();

    // First 128 threads handle the 128 output dimensions
    if (tid < HD128) {
        // A. Find global max across all 16 groups
        float global_max = -1e20f;
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS128; ++g) {
            global_max = max(global_max, s_m[g]);
        }

        // B. Compute weighted sum and accumulator for this dimension
        float final_sum = 0.0f;
        float final_acc_dim = 0.0f;

        #pragma unroll
        for (int g = 0; g < NUM_GROUPS128; ++g) {
            float exp_rescale = expf(s_m[g] - global_max);
            final_sum += s_s[g] * exp_rescale;
            final_acc_dim += s_acc[g * HD128 + tid] * exp_rescale;
        }

        // C. Write final result O[q_head, tid]
        if (final_sum > 0) {
            O[q_head_idx * HD128 + tid] = __float2half(final_acc_dim / final_sum);
        } else {
            O[q_head_idx * HD128 + tid] = __float2half(0.0f);
        }
    }
}

extern "C"
void flash_decoding_cu_fp16_hdim128(
    const __half* q_ptr,
    const __half* k_ptr,
    const __half* v_ptr,
    __half* o_ptr,
    int32_t* kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    // Shared memory: s_q(256) + s_k(8192) + s_v(8192) + s_m(64) + s_s(64) + s_acc(8192) = ~25KB
    size_t smem_size = (HD128 * sizeof(__half)) +
                       (2 * BN128 * HD128 * 2 * sizeof(__half)) +
                       (NUM_GROUPS128 * 2 * sizeof(float)) +
                       (NUM_GROUPS128 * HD128 * sizeof(float));

    float sm_scale = 1.0f / sqrtf((float)head_dim);
    int group_size = num_q_heads / num_kv_heads;

    // One block per query head, 256 threads per block
    flash_decode_fp16_hdim128_kernel<<<num_q_heads, 256, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr,
        kv_seq_len,
        num_kv_heads, group_size, sm_scale
    );
}
