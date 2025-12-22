#include <cuda_bf16.h>
#include <flash_attn_gqa.h>
#define CP_ASYNC_CG(dst, src, bytes)                                           \
asm volatile(                                                                \
"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),       \
"l"(src), "n"(bytes))
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)
#define HEAD_DIM_1B 64
#define WARP_SIZE 32
#define BN 16  // 一个循环处理 16 个 Token
#define THREAD_PER_BLOCK 128
#define VEC_SIZE 8     // float4 处理 8 个 bf16
#define THREADS_PER_KEY (HEAD_DIM_1B / VEC_SIZE) // float4 处理 8 个 bf16
__global__ void simple_llama3_1B_decode_kernel(
    const __nv_bfloat16* __restrict__ Q,  // [num_q_heads, D]
    const __nv_bfloat16* __restrict__ K,  // [Seq_Len, num_kv_heads, D]
    const __nv_bfloat16* __restrict__ V,  // [Seq_Len, num_kv_heads, D]
    __nv_bfloat16* __restrict__ O,        // [num_q_heads, D]
    int seq_len,
    int num_kv_heads,
    int group_size, // q_heads / kv_heads (Llama 3 8B通常是4或8, 1B如果是MQA则较大)
    float sm_scale
) {
    // 1. 静态计算布局
    // 每个Block处理一个 Q Head
    int q_head_idx = blockIdx.x;
    int kv_head_idx = q_head_idx / group_size; // GQA/MQA 核心逻辑

    // 共享内存：1个Q + 2个缓存块(K) + 2个缓存块(V)
    // 每个缓存块大小：BN * HEAD_DIM
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* s_q = smem;
    __nv_bfloat16* s_k = s_q + HEAD_DIM_1B;
    __nv_bfloat16* s_v = s_k + (2 * BN * HEAD_DIM_1B);

    int tid = threadIdx.x;
    const int row_idx_in_batch = tid / THREADS_PER_KEY; // 0-15 (负责哪一个Token)
    const int vec_offset = tid % THREADS_PER_KEY;      // 0-7  (负责向量的哪一段)

    // 2. 载入 Q 到 Smem (只有128个线程，每线程搬0.5个BF16即可，这里用前8线程搬完)
    if (tid < 8) {
        reinterpret_cast<float4*>(s_q)[tid] =
            reinterpret_cast<const float4*>(Q + q_head_idx * HEAD_DIM_1B)[tid];
    }
    __syncthreads();

    // 3. 寄存器初始化
    float row_max = -1e20f;
    float row_sum = 0.0f;
    float acc[8] = {0.0f}; // 每个线程维护自己负责的那 8 个维度的累加和

    // 4. 异步读取函数 (支持 GQA 索引)
    auto fetch_kv = [&](int iter, int stage) {
        int token_idx = iter * BN + row_idx_in_batch;
        if (token_idx < seq_len) {
            // 计算全局 K, V 偏移
            // 假设布局是 [Seq, KV_Heads, D]
            int base_offset = (token_idx * num_kv_heads + kv_head_idx) * HEAD_DIM_1B;

            // 目标 Smem 偏移
            int smem_offset = (stage * BN + row_idx_in_batch) * HEAD_DIM_1B + vec_offset * 8;

            // 128位异步拷贝
            CP_ASYNC_CG(
                &s_k[smem_offset],
                &K[base_offset + vec_offset * 8],
                sizeof(float4)
            );
            CP_ASYNC_CG(
                &s_v[smem_offset],
                &V[base_offset + vec_offset * 8],
                sizeof(float4)
            );
        }
    };

    // 5. 软件流水线循环
    fetch_kv(0, 0);
    CP_ASYNC_COMMIT_GROUP();

    for (int i = 0; i < seq_len; i += BN) {
        int next_stage = (i / BN + 1) % 2;
        if ((i + BN) < seq_len) {
            fetch_kv(i / BN + 1, next_stage);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(1);
        __syncthreads();

        int cur_stage = (i / BN) % 2;
        __nv_bfloat16* base_s_k = &s_k[cur_stage * BN * HEAD_DIM_1B];
        __nv_bfloat16* base_s_v = &s_v[cur_stage * BN * HEAD_DIM_1B];

        // --- 核心逻辑：遍历本批次的 BN 个 Token ---
        // 注意：这里每一个线程都在算自己那份 row_idx_in_batch 对应的 score
        for (int b = 0; b < BN; ++b) {
            int current_token_idx = i + b;
            if (current_token_idx >= seq_len) break;

            // A. 计算点积 (Q * K)
            float score = 0.0f;
            float4 q_v = reinterpret_cast<float4*>(s_q)[vec_offset];
            float4 k_v = reinterpret_cast<float4*>(&base_s_k[b * HEAD_DIM_1B])[vec_offset];

            __nv_bfloat162* q_p = reinterpret_cast<__nv_bfloat162*>(&q_v);
            __nv_bfloat162* k_p = reinterpret_cast<__nv_bfloat162*>(&k_v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_bfloat162 res = __hmul2(q_p[j], k_p[j]);
                score += __low2float(res) + __high2float(res);
            }

            // B. Warp内归约 (8个线程合并成一个 Score)
            // 使用 shfl_sync 归约同一个 Token 的 8 个片段
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                score += __shfl_xor_sync(0xffffffff, score, offset);
            }
            score *= sm_scale;

            // C. 在线 Softmax 更新
            float old_max = row_max;
            row_max = max(row_max, score);
            float exp_scale = exp2f((old_max - row_max) * 1.44269504f);
            float p = exp2f((score - row_max) * 1.44269504f);
            row_sum = row_sum * exp_scale + p;

            // D. 累加 V
            float4 v_v = reinterpret_cast<float4*>(&base_s_v[b * HEAD_DIM_1B])[vec_offset];
            __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(&v_v);
            #pragma unroll
            for (int d = 0; d < 8; ++d) {
                acc[d] = acc[d] * exp_scale + p * (float)v_p[d];
            }
        }
    }

    // 6. 最终写回
    // 注意：目前的 acc 在每 8 个线程里都是重复的（因为每个线程都跑了相同的 b 循环）
    // 但没关系，它们算的是相同的 Dim 映射。直接让每个线程写回自己负责的 8 个维度。
    // 索引：O[q_head_idx, vec_offset * 8 ... +7]
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        O[q_head_idx * HEAD_DIM_1B + vec_offset * 8 + d] = __float2bfloat16(acc[d] / row_sum);
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
    int threads = 128;
    dim3 block(threads);
    dim3 grid(num_q_heads);
    // s_q (64) + s_k (2 * 16 * 64) + s_v (2 * 16 * 64)
    size_t smem_size = (head_dim + 2 * 16 * head_dim + 2 * 16 * head_dim) * sizeof(__nv_bfloat16);
    float sm_scale = 1.0f / sqrtf((float)head_dim);
    int group_size = num_q_heads / num_kv_heads;

    simple_llama3_1B_decode_kernel<<<grid, block, smem_size, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        kv_seq_len + 1,
        num_kv_heads,
        group_size,
        sm_scale
    );

    CUDA_CHECK(cudaGetLastError());
}