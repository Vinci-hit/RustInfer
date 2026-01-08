#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include "flash_attn_gqa.h"

// 宏定义 (严格按照要求)
#define CP_ASYNC_CG(dst, src, bytes)                                           \
    asm volatile(                                                              \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),     \
        "l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

#define HEAD_DIM_1B 64
#define BN 16
#define THREADS_PER_KEY 8
#define NUM_GROUPS 16 // 128 / 8

__global__ void llama3_1B_decode_kernel_optimized(
    const __nv_bfloat16* __restrict__ Q,  // [num_q_heads, D]
    const __nv_bfloat16* __restrict__ K,  // [Seq_Len, num_kv_heads, D]
    const __nv_bfloat16* __restrict__ V,  // [Seq_Len, num_kv_heads, D]
    __nv_bfloat16* __restrict__ O,        // [num_q_heads, D]
    int seq_len,
    int num_kv_heads,
    int group_size,
    float sm_scale
) {
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / group_size;

    // 共享内存布局 (需要容纳 Q, 2阶段的K/V, 以及归约用的临时空间)
    // s_q: 64 * 2 = 128 bytes
    // s_kv: 2 (stage) * 16 (BN) * 64 (D) * 2 (K&V) * 2 (bf16) = 8192 bytes
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* s_q = smem;
    __nv_bfloat16* s_k = s_q + HEAD_DIM_1B;
    __nv_bfloat16* s_v = s_k + (2 * BN * HEAD_DIM_1B);
    
    // 归约专用空间 (接在 KV 后面，使用 float 保证精度)
    float* s_m = (float*)(s_v + (2 * BN * HEAD_DIM_1B)); // [16]
    float* s_s = s_m + NUM_GROUPS;                       // [16]
    float* s_acc = s_s + NUM_GROUPS;                     // [16 * 64]

    int tid = threadIdx.x;
    int gid = tid / THREADS_PER_KEY;  // Group ID (0-15)
    int lane = tid % THREADS_PER_KEY; // Lane ID (0-7), 负责维度 lane*8 到 lane*8+7

    // 1. 加载 Q 到共享内存 (使用 float4 提高效率)
    if (tid < 8) { 
        reinterpret_cast<float4*>(s_q)[tid] = 
            reinterpret_cast<const float4*>(Q + q_head_idx * HEAD_DIM_1B)[tid];
    }

    // 寄存器状态
    float row_max = -1e20f;
    float row_sum = 0.0f;
    float acc[8] = {0.0f};

    auto get_smem_ptr = [](__nv_bfloat16* p) -> uint64_t {
        return (uint64_t)__cvta_generic_to_shared(p);
    };

    // 2. 软件流水线加载 K, V
    auto fetch_kv = [&](int iter, int stage) {
        int token_idx = iter * BN + gid;
        if (token_idx < seq_len) {
            const __nv_bfloat16* k_ptr = K + (token_idx * num_kv_heads + kv_head_idx) * HEAD_DIM_1B + lane * 8;
            const __nv_bfloat16* v_ptr = V + (token_idx * num_kv_heads + kv_head_idx) * HEAD_DIM_1B + lane * 8;
            __nv_bfloat16* sk_dst = &s_k[(stage * BN + gid) * HEAD_DIM_1B + lane * 8];
            __nv_bfloat16* sv_dst = &s_v[(stage * BN + gid) * HEAD_DIM_1B + lane * 8];
            
            CP_ASYNC_CG(get_smem_ptr(sk_dst), k_ptr, 16);
            CP_ASYNC_CG(get_smem_ptr(sv_dst), v_ptr, 16);
        }
    };

    // 预热流水线
    fetch_kv(0, 0);
    CP_ASYNC_COMMIT_GROUP();

    for (int i = 0; i < seq_len; i += BN) {
        int cur_stage = (i / BN) % 2;
        int next_stage = 1 - cur_stage;

        if (i + BN < seq_len) {
            fetch_kv(i / BN + 1, next_stage);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1); // 等待当前 stage
        __syncthreads();

        int current_token_idx = i + gid;
        if (current_token_idx < seq_len) {
            __nv_bfloat16* sk_local = &s_k[(cur_stage * BN + gid) * HEAD_DIM_1B];
            __nv_bfloat16* sv_local = &s_v[(cur_stage * BN + gid) * HEAD_DIM_1B];

            // A. 计算点积 Q * K
            float score = 0.0f;
            float4 q_vec = reinterpret_cast<float4*>(s_q)[lane];
            float4 k_vec = reinterpret_cast<float4*>(sk_local)[lane];
            
            __nv_bfloat162* q_p2 = reinterpret_cast<__nv_bfloat162*>(&q_vec);
            __nv_bfloat162* k_p2 = reinterpret_cast<__nv_bfloat162*>(&k_vec);
            #pragma unroll
            for(int j=0; j<4; ++j) {
                __nv_bfloat162 res = __hmul2(q_p2[j], k_p2[j]);
                score += __low2float(res) + __high2float(res);
            }

            // B. Group 内归约 (8 threads -> 1 score)
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                score += __shfl_xor_sync(0xff, score, offset);
            }
            score *= sm_scale;

            // C. Online Softmax 更新
            float old_max = row_max;
            row_max = max(row_max, score);
            float exp_scale = expf(old_max - row_max);
            float p = expf(score - row_max);
            row_sum = row_sum * exp_scale + p;

            // D. 累加 V
            float4 v_vec = reinterpret_cast<float4*>(sv_local)[lane];
            __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(&v_vec);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                acc[d] = acc[d] * exp_scale + p * __bfloat162float(v_p[d]);
            }
        }
    }

    // 3. Block-level 最终归约 (16 个 Group 结果融合)
    // 存储局部结果到 Smem
    if (lane == 0) {
        s_m[gid] = row_max;
        s_s[gid] = row_sum;
    }
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        s_acc[gid * 64 + lane * 8 + d] = acc[d];
    }
    __syncthreads();

    // 只有前 64 个线程参与最后的合并 (处理 64 个维度)
    if (tid < 64) {
        // A. 找出 16 个 group 中的全局最大值
        float global_max = -1e20f;
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS; ++g) {
            global_max = max(global_max, s_m[g]);
        }

        // B. 计算加权后的全局 sum 和全局 acc[tid]
        float final_sum = 0.0f;
        float final_acc_dim = 0.0f;
        
        #pragma unroll
        for (int g = 0; g < NUM_GROUPS; ++g) {
            float exp_rescale = expf(s_m[g] - global_max);
            final_sum += s_s[g] * exp_rescale;
            final_acc_dim += s_acc[g * 64 + tid] * exp_rescale;
        }

        // C. 写回最终结果 O [q_head, tid]
        if (final_sum > 0) {
            O[q_head_idx * HEAD_DIM_1B + tid] = __float2bfloat16(final_acc_dim / final_sum);
        } else {
            O[q_head_idx * HEAD_DIM_1B + tid] = __float2bfloat16(0.0f);
        }
    }
}

void flash_decoding_cu_bf16(
    const __nv_bfloat16* q_ptr,
    const __nv_bfloat16* k_ptr,
    const __nv_bfloat16* v_ptr,
    __nv_bfloat16* o_ptr,
    int32_t kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    // 计算共享内存大小: Q(128) + KV_Stages(8192) + s_m(64) + s_s(64) + s_acc(4096)
    // 约 13KB，远小于硬件限制
    size_t smem_size = (HEAD_DIM_1B * sizeof(__nv_bfloat16)) + 
                       (2 * BN * HEAD_DIM_1B * 2 * sizeof(__nv_bfloat16)) +
                       (NUM_GROUPS * 2 * sizeof(float)) + 
                       (NUM_GROUPS * HEAD_DIM_1B * sizeof(float));

    float sm_scale = 1.0f / sqrtf((float)head_dim);
    int group_size = num_q_heads / num_kv_heads;

    llama3_1B_decode_kernel_optimized<<<num_q_heads, 128, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr,
        kv_seq_len, // 注意：如果 seq_len 包含当前 token，传入实际长度即可
        num_kv_heads, group_size, sm_scale
    );
}