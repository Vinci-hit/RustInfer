#include "flash_attn_gqa.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int mask = 32 >> 1; mask >= 1; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}
__device__ __forceinline__ float warp_reduce_sum(float val, unsigned mask = 0xffffffff) {
#pragma unroll
    for (int mask = 32 >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, 32);
    }
    return val;
}

#define CP_ASYNC_CG(dst, src, bytes)                                           \
asm volatile(                                                                \
"cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"l"(dst),       \
"l"(src), "n"(bytes))

#define CP_ASYNC_CG_ca(dst, src, bytes)                                           \
asm volatile(                                                                \
"cp.async.ca.shared.global [%0], [%1], %2;\n" ::"l"(dst),       \
"l"(src), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
template<const int THREADS_PER_BLOCK = 128,const int Br = 16>
__global__ void flash_attn_gqa_kernel(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    float* __restrict__ o_ptr,
    int Tc,
    const int tile_offset,//每个tile的大小，q_ptr[q_tile_id*Br:q_tile_id*(Br+1)][head_idx][head_dim]
    const int tile_size_inhead,
    const int kv_total_len, // 历史大小加上当前q输入新的qlen
    const float scale,
    int q_seq_len,
    int kv_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim)
{
    const unsigned Q_tile_id = blockIdx.x; // 处理某个Q的块。
    const unsigned q_head_idx = blockIdx.y; //第几个头，索引形式为q_ptr[seq_len][head_idx][head_dim]
    const unsigned kv_head_idx = blockIdx.y / (num_q_heads / num_kv_heads);
    const int kLoadNumElementsPerThread = Br * head_dim / THREADS_PER_BLOCK; //一个block128个线程，将tile内的元素平均分给每个线程。只管head内
    extern __shared__ float Q_tile_smem[];
    float *K_tile_smem = Q_tile_smem + tile_size_inhead;
    float *V_tile_smem = Q_tile_smem + 2 * tile_size_inhead;
    const int headdim_warp = head_dim / 32;
    uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);// 将通用指针转为共享内存专用指针。也许不转也行？
    uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
    uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);
    const unsigned tid = threadIdx.x;
    {
        const unsigned q_row_id = (tid / 4) + Q_tile_id * Br; //128个线程处理32行，也就是4个线程处理一行，也就是说tid/4 == tile内行id, 加上Q偏移，等于全局q行id
        //q_ptr[q_row_id][head_id][(tid % 4) * kLoadNumElementsPerThread]
        int gmem_start_idx = q_row_id * num_q_heads * head_dim + q_head_idx * head_dim + (tid % 4) * kLoadNumElementsPerThread; // 此时不同的线程的这个值处于各自的起始位置。
        //Q_smem[(tid / 4)][(tid % 4) * kLoadNumElementsPerThread]
        uint32_t q_load_smem_ptr = smem_Q_base_ptr + ((tid / 4) * head_dim + (tid % 4) * kLoadNumElementsPerThread)  * sizeof(float);//每个指针应该在的共享内存内的地址。
        if (q_row_id < q_seq_len)
        {
#pragma unroll
            for (int i =0;i<kLoadNumElementsPerThread;i+=4) //载入Q，边界判断
            {
                CP_ASYNC_CG(q_load_smem_ptr + i * sizeof(float), &q_ptr[gmem_start_idx + i], 4 * sizeof(float));
            }
        }
        CP_ASYNC_COMMIT_GROUP();//提交异步流
    }
    float m_old[8]; // 8 个 Br 行的局部最大值
    float l_old[8]; // 8 个 Br 行的局部指数和
    for (int i = 0; i < 8; ++i) {
        m_old[i] = -INFINITY;
        l_old[i] = 0.0f;
    }
    const int output_elements_per_thread = Br * head_dim / THREADS_PER_BLOCK;
    // 动态分配线程本地内存（堆上，64位CUDA支持）
    float* Output = (float*)malloc(output_elements_per_thread * sizeof(float));
    if (Output == nullptr) {  // 检查分配失败（避免内存不足）
        return;
    }
    for (int i =0 ; i< output_elements_per_thread;++i)
    {
        Output[i] = 0.0f;
    }
    
    //在循环外首次加载
    unsigned k_load_row_id = (tid / 4) + 0 * Br;
    //k_ptr[k_load_row_id][kv_head_id][tid%4 * kperthread]
    int kv_gmem_start_idx = k_load_row_id * num_kv_heads * head_dim + kv_head_idx * head_dim + (tid % 4) * kLoadNumElementsPerThread; // 此时不同的线程的这个值处于各自的起始位置。
    //k_smem[tid/4][tid%4 * kperthead] // 即每行由四个线程填充，转置存储只需改为k_smem[tid%4 * kperthead][tid/4]
    uint32_t k_load_smem_ptr = smem_K_base_ptr + ((tid % 4) * kLoadNumElementsPerThread * 32 + (tid / 4))  * sizeof(float);//每个指针应该在的共享内存内的地址。
    uint32_t v_load_smem_ptr = smem_V_base_ptr + ((tid / 4) * head_dim + (tid % 4) * kLoadNumElementsPerThread)  * sizeof(float);

#pragma unroll
    for (int i =0;i<kLoadNumElementsPerThread;i++)
    {
        CP_ASYNC_CG_ca(k_load_smem_ptr + (i * 32) * sizeof(float), &k_ptr[kv_gmem_start_idx + i], sizeof(float));
    }
    CP_ASYNC_COMMIT_GROUP();//提交异步流
    //开始主循环
    for (int K_tile_id = 0; K_tile_id < Tc; K_tile_id++)
    {
        
        k_load_row_id = (tid / 4) + K_tile_id * Br;
        kv_gmem_start_idx = k_load_row_id * num_kv_heads * head_dim + kv_head_idx * head_dim + (tid % 4) * kLoadNumElementsPerThread;

        if (k_load_row_id < kv_total_len)
        {
            //此时在载入qk中，还算不了矩阵，先发布一下载入v的命令
#pragma unroll
            for (int i =0;i<kLoadNumElementsPerThread;i+=4) //载入Q，边界判断
            {
                CP_ASYNC_CG(v_load_smem_ptr + i * sizeof(float), &v_ptr[kv_gmem_start_idx + i], 4 * sizeof(float));
            }
            CP_ASYNC_COMMIT_GROUP();//提交异步流
            CP_ASYNC_WAIT_GROUP(1);//等待qk加载完毕，允许最新的1组未完成
        }
        __syncthreads(); //等待块内所有线程加载数据完毕
        const unsigned k_row_id = (tid % Br) + K_tile_id * Br;
        //开始计算矩阵乘法，此时已保证k_row不会超过所需要的行
        float R_S[8]; //每个线程持有的block内的结果数量，为32*32 / 128 = 8
        for (int i = 0; i < 8; ++i) {
            R_S[i] = 0.0f;
        }
        const int q_row = Q_tile_id * Br + tid / 32 * 8;
        const int valid_row_id = q_seq_len - q_row > 8 ? 8 : q_seq_len - q_row;
        #ifndef HEAD_DIM_DIV_4
        #define HEAD_DIM_DIV_4 (head_dim / 4)
        #endif
        if (k_row_id < kv_total_len)
        {
        #pragma unroll
            for (int i = 0; i < 8; i++) 
            {
                const unsigned s_row_id = tid / Br * 8 + i; 
                
                // 1. 获取 Q 的 float4 指针 (假设 Q 的行是 16 字节对齐的)
                // Q_smem[s_row_id * head_dim] 是 Q 行的起始 float 地址
                const float4* Q_row_ptr_f4 = (const float4*)&Q_tile_smem[s_row_id * head_dim];

                // 2. K 的列索引 (转置 K 的行索引)
                const unsigned k_col_idx = k_row_id % 32; 

                // 3. 循环步长改为 4
        #pragma unroll
                for (int j_f4 = 0; j_f4 < HEAD_DIM_DIV_4; j_f4++)
                {
                    // --- 读取 Q 的 float4 ---
                    // 每次读取 4 个 float，减少 4 倍 SMem Load 指令数量
                    const float4 q_f4 = Q_row_ptr_f4[j_f4];
                    
                    // --- 读取 K 的 4 个 float ---
                    // K 的访问仍然是 4 个单独的 float 访问，以保持转置索引的正确性。
                    // 尽管是 4 次 SMem 访问，但它们依赖于前一个指令（Q Load）完成。

                    // K 的行索引 j 是 j_f4 * 4, j_f4 * 4 + 1, ...
                    const int j_base = j_f4 * 4;

                    float k_vals[4];
                    k_vals[0] = K_tile_smem[j_base * 32 + k_col_idx]; 
                    k_vals[1] = K_tile_smem[(j_base + 1) * 32 + k_col_idx];
                    k_vals[2] = K_tile_smem[(j_base + 2) * 32 + k_col_idx];
                    k_vals[3] = K_tile_smem[(j_base + 3) * 32 + k_col_idx];
                    
                    // 4. 点积累加 (手动展开 4 次浮点乘法)
                    R_S[i] += q_f4.x * k_vals[0];
                    R_S[i] += q_f4.y * k_vals[1];
                    R_S[i] += q_f4.z * k_vals[2];
                    R_S[i] += q_f4.w * k_vals[3];
                }
            }
        }
        

        k_load_row_id = (tid / 4) + (K_tile_id + 1) * Br;
        kv_gmem_start_idx = k_load_row_id * num_kv_heads * head_dim + kv_head_idx * head_dim + (tid % 4) * kLoadNumElementsPerThread;
        if (k_load_row_id < kv_total_len)
        {
#pragma unroll
            for (int i =0;i<kLoadNumElementsPerThread;i++)
            {
                CP_ASYNC_CG_ca(k_load_smem_ptr + (i * 32) * sizeof(float), &k_ptr[kv_gmem_start_idx + i], sizeof(float));
            }
            CP_ASYNC_COMMIT_GROUP();//提交异步流
        }

        float m_new;
        m_new = -INFINITY;
#pragma unroll
        for (int i = 0; i < valid_row_id; ++i) {
            const unsigned q_abs_pos = kv_seq_len + Q_tile_id * Br + i + tid / 32 * 8;
            // Causal Mask
            if (k_row_id > q_abs_pos) {
                R_S[i] = -INFINITY; // 掩盖 (设为极小负数)
            } else {
                R_S[i] *= scale;
            }
            m_new = warp_reduce_max(R_S[i]);
            m_new = m_new > m_old[i]?m_new:m_old[i];
            R_S[i] = __expf(R_S[i] - m_new);
            float e_factor = __expf(m_old[i] - m_new);
            l_old[i] = __expf(m_old[i] - m_new) * l_old[i] + warp_reduce_sum(R_S[i]);
            CP_ASYNC_WAIT_GROUP(1);//等待v加载完毕，允许最新的1组未完成
            __syncthreads(); //等待块内所有线程加载数据完毕
            int valid_col = kv_total_len - K_tile_id * Br;//计算现在p矩阵的有效列
            valid_col = valid_col < 32?valid_col :32;
            // P @ V 乘法，（32,32) @ (32,head_dim)
            for (int col = 0;col < head_dim;col+=32)//32个线程沿着headdim维度处理，无bank conflict,一个线程是一个col
            {
                float temp = 0;
                for (int row = 0;row < valid_col;row++)
                {
                    float val = __shfl_sync(0xffffffff,R_S[i],row,32);
                    
                    temp += V_tile_smem[row * head_dim + col + (tid % 32)] * val;
                    
                }
                
                const int local_col_block = col / 32; // 0, 1, 2, 3
                const int local_idx = i * headdim_warp + local_col_block; // [0, 31]
                // 现在只与 Q 有关，应该与 Q 的行数 q_seq_len 比较。
                if (Q_tile_id * Br + (tid / 32 * 8) + i < q_seq_len)
                    Output[local_idx] = temp + Output[local_idx] * e_factor;
                
            }
            
            m_old[i] = m_new;
        }
        
    }// 结束K循环
    
    int valid_rows_in_tile = q_seq_len - Q_tile_id * Br;
    valid_rows_in_tile = valid_rows_in_tile < Br ? valid_rows_in_tile : Br;
    const int q_row_idx = tid / 32 *8; 
    valid_rows_in_tile = valid_rows_in_tile - q_row_idx;
    valid_rows_in_tile = valid_rows_in_tile > 8 ? 8 : valid_rows_in_tile;
    for (int i = 0; i < valid_rows_in_tile; i++) // 遍历 i 循环
    {
        for (int col = 0; col < head_dim; col += 32) // 遍历 col 循环
        {
            const int local_col_block = col / 32;
            const int local_idx = i * headdim_warp + local_col_block;

            // 第一步：除以 l_old (e_factor 已经被并入了 l_old 的累加)
            // l_old[i] 存储了 e_factor * l_old[i] + sum(exp(S'))
            Output[local_idx] *= (1.0f / l_old[i]);
            // 第二步：写入全局 O 矩阵
            // O[global_row][q_head_idx][global_col]
            size_t o_gmem_idx = (size_t)(Q_tile_id * Br + q_row_idx + i) * num_q_heads * head_dim +
                                q_head_idx * head_dim +
                                col + (tid % 32);
            
            if (Q_tile_id * Br + (tid / 32 * 8) + i < q_seq_len)
                o_ptr[o_gmem_idx] = Output[local_idx];
        }
        
    }
    free(Output);
}

void flash_attn_gqa_cu(
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* o_ptr,
    int32_t q_seq_len,
    int32_t kv_seq_len,
    int32_t num_q_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    cudaStream_t stream)
{
    // 1. 确定启动参数
    constexpr int Br = 32;
    constexpr int THREADS_PER_BLOCK = 128; //4个warp来处理一个block
    const int N_blocks = (q_seq_len + Br - 1) / Br; // 按q的输入长度分竖着x方向的块。
    const int Tc = (kv_seq_len + q_seq_len+ Br - 1) / Br;
    const int tile_offset = Br * head_dim * num_q_heads;
    dim3 grid(N_blocks, num_q_heads); // 因为没有batch，因此按head并行。
    dim3 block(THREADS_PER_BLOCK);
    const unsigned smem_size = Br * head_dim * 3 * sizeof(float);
    // 3. 启动核函数
    flash_attn_gqa_kernel<THREADS_PER_BLOCK,Br><<<grid, block, smem_size, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        Tc,
        tile_offset,
        Br * head_dim,
        kv_seq_len + q_seq_len,
        1.0f / sqrtf((float)head_dim),
        q_seq_len,
        kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim
    );
    // 4. 检查是否有核函数启动错误
    CUDA_CHECK(cudaGetLastError());
}