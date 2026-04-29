#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float zimage_bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 zimage_float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float zimage_warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template <int BLOCK_THREADS>
__device__ __forceinline__ float zimage_block_reduce_sum(float v) {
    constexpr int NUM_WARPS = (BLOCK_THREADS + 31) / 32;
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = zimage_warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    v = (threadIdx.x < NUM_WARPS) ? warp_sums[lane] : 0.0f;
    if (warp == 0) v = zimage_warp_reduce_sum(v);
    return v;
}

__global__ void zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_kernel(
    __nv_bfloat16* __restrict__ mod_out,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int j = idx; j < dim; j += stride) {
        float scale_msa = zimage_bf16_to_float(mod_out[j]);
        float gate_msa = zimage_bf16_to_float(mod_out[dim + j]);
        float scale_mlp = zimage_bf16_to_float(mod_out[2 * dim + j]);
        float gate_mlp = zimage_bf16_to_float(mod_out[3 * dim + j]);

        mod_out[j] = zimage_float_to_bf16(scale_msa + 1.0f);
        mod_out[dim + j] = zimage_float_to_bf16(tanhf(gate_msa));
        mod_out[2 * dim + j] = zimage_float_to_bf16(scale_mlp + 1.0f);
        mod_out[3 * dim + j] = zimage_float_to_bf16(tanhf(gate_mlp));
    }
}

extern "C" void zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_forward(
    __nv_bfloat16* mod_out,
    int dim,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    zimage_fused_adaln_modulation_split_scale_add_tanh_bf16_kernel<<<blocks, threads, 0, stream>>>(
        mod_out,
        dim
    );
}

template <int BLOCK_THREADS>
__global__ void zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ qkv_out,
    const __nv_bfloat16* __restrict__ norm_q_weight,
    const __nv_bfloat16* __restrict__ norm_k_weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int n_heads,
    int head_dim,
    float eps
) {
    int s = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int dim = n_heads * head_dim;
    int half_dim = head_dim >> 1;

    int q_base = s * 3 * dim + h * head_dim;
    int k_base = s * 3 * dim + dim + h * head_dim;
    int v_base = s * 3 * dim + 2 * dim + h * head_dim;
    int out_base = s * dim + h * head_dim;

    float sum_q = 0.0f;
    float sum_k = 0.0f;
    for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
        float qv = zimage_bf16_to_float(qkv_out[q_base + d]);
        float kv = zimage_bf16_to_float(qkv_out[k_base + d]);
        sum_q = fmaf(qv, qv, sum_q);
        sum_k = fmaf(kv, kv, sum_k);
        v[out_base + d] = qkv_out[v_base + d];
    }

    sum_q = zimage_block_reduce_sum<BLOCK_THREADS>(sum_q);
    sum_k = zimage_block_reduce_sum<BLOCK_THREADS>(sum_k);

    __shared__ float inv_q;
    __shared__ float inv_k;
    if (tid == 0) {
        inv_q = rsqrtf(sum_q / float(head_dim) + eps);
        inv_k = rsqrtf(sum_k / float(head_dim) + eps);
    }
    __syncthreads();

    const float* cos_row = cos + s * half_dim;
    const float* sin_row = sin + s * half_dim;
    for (int pair = tid; pair < half_dim; pair += BLOCK_THREADS) {
        int d0 = pair << 1;
        int d1 = d0 + 1;
        float c = cos_row[pair];
        float sn = sin_row[pair];

        float q0 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(qkv_out[q_base + d0]) * inv_q * zimage_bf16_to_float(norm_q_weight[d0])
        ));
        float q1 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(qkv_out[q_base + d1]) * inv_q * zimage_bf16_to_float(norm_q_weight[d1])
        ));
        float k0 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(qkv_out[k_base + d0]) * inv_k * zimage_bf16_to_float(norm_k_weight[d0])
        ));
        float k1 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(qkv_out[k_base + d1]) * inv_k * zimage_bf16_to_float(norm_k_weight[d1])
        ));

        q[out_base + d0] = zimage_float_to_bf16(q0 * c - q1 * sn);
        q[out_base + d1] = zimage_float_to_bf16(q0 * sn + q1 * c);
        k[out_base + d0] = zimage_float_to_bf16(k0 * c - k1 * sn);
        k[out_base + d1] = zimage_float_to_bf16(k0 * sn + k1 * c);
    }
}

extern "C" void zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16_forward(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    const __nv_bfloat16* qkv_out,
    const __nv_bfloat16* norm_q_weight,
    const __nv_bfloat16* norm_k_weight,
    const float* cos,
    const float* sin,
    int seq,
    int n_heads,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    dim3 grid(seq, n_heads);
    if (head_dim <= 128) {
        zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16_kernel<128><<<grid, 128, 0, stream>>>(
            q, k, v, qkv_out, norm_q_weight, norm_k_weight, cos, sin, n_heads, head_dim, eps
        );
    } else {
        zimage_fused_qkv_split_head_rmsnorm_rope_interleaved_bf16_kernel<256><<<grid, 256, 0, stream>>>(
            q, k, v, qkv_out, norm_q_weight, norm_k_weight, cos, sin, n_heads, head_dim, eps
        );
    }
}

template <int BLOCK_THREADS>
__global__ void zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16_kernel(
    __nv_bfloat16* __restrict__ residual_mid,
    __nv_bfloat16* __restrict__ ffn_in,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ to_out_result,
    const __nv_bfloat16* __restrict__ gate_msa,
    const __nv_bfloat16* __restrict__ scale_mlp,
    const __nv_bfloat16* __restrict__ attention_norm2_weight,
    const __nv_bfloat16* __restrict__ ffn_norm1_weight,
    int dim,
    float attention_norm2_eps,
    float ffn_norm1_eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * dim;

    float sum_attn = 0.0f;
    for (int j = tid; j < dim; j += BLOCK_THREADS) {
        float y = zimage_bf16_to_float(to_out_result[base + j]);
        sum_attn = fmaf(y, y, sum_attn);
    }
    sum_attn = zimage_block_reduce_sum<BLOCK_THREADS>(sum_attn);

    __shared__ float inv_attn;
    if (tid == 0) inv_attn = rsqrtf(sum_attn / float(dim) + attention_norm2_eps);
    __syncthreads();

    float sum_mid = 0.0f;
    for (int j = tid; j < dim; j += BLOCK_THREADS) {
        float norm2 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(to_out_result[base + j]) * inv_attn * zimage_bf16_to_float(attention_norm2_weight[j])
        ));
        float gated = zimage_bf16_to_float(zimage_float_to_bf16(norm2 * zimage_bf16_to_float(gate_msa[j])));
        __nv_bfloat16 mid_b = zimage_float_to_bf16(zimage_bf16_to_float(x[base + j]) + gated);
        residual_mid[base + j] = mid_b;
        float mid = zimage_bf16_to_float(mid_b);
        sum_mid = fmaf(mid, mid, sum_mid);
    }
    sum_mid = zimage_block_reduce_sum<BLOCK_THREADS>(sum_mid);

    __shared__ float inv_mid;
    if (tid == 0) inv_mid = rsqrtf(sum_mid / float(dim) + ffn_norm1_eps);
    __syncthreads();

    for (int j = tid; j < dim; j += BLOCK_THREADS) {
        float norm1 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(residual_mid[base + j]) * inv_mid * zimage_bf16_to_float(ffn_norm1_weight[j])
        ));
        ffn_in[base + j] = zimage_float_to_bf16(norm1 * zimage_bf16_to_float(scale_mlp[j]));
    }
}

extern "C" void zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16_forward(
    __nv_bfloat16* residual_mid,
    __nv_bfloat16* ffn_in,
    const __nv_bfloat16* x,
    const __nv_bfloat16* to_out_result,
    const __nv_bfloat16* gate_msa,
    const __nv_bfloat16* scale_mlp,
    const __nv_bfloat16* attention_norm2_weight,
    const __nv_bfloat16* ffn_norm1_weight,
    int rows,
    int dim,
    float attention_norm2_eps,
    float ffn_norm1_eps,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    zimage_fused_post_attention_rmsnorm_gate_residual_and_ffn_prenorm_scale_bf16_kernel<threads><<<rows, threads, 0, stream>>>(
        residual_mid,
        ffn_in,
        x,
        to_out_result,
        gate_msa,
        scale_mlp,
        attention_norm2_weight,
        ffn_norm1_weight,
        dim,
        attention_norm2_eps,
        ffn_norm1_eps
    );
}

template <int BLOCK_THREADS>
__global__ void zimage_fused_ffn_down_rmsnorm_gate_residual_bf16_kernel(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ residual_mid,
    const __nv_bfloat16* __restrict__ ffn_out,
    const __nv_bfloat16* __restrict__ gate_mlp,
    const __nv_bfloat16* __restrict__ ffn_norm2_weight,
    int dim,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * dim;

    float sum = 0.0f;
    for (int j = tid; j < dim; j += BLOCK_THREADS) {
        float y = zimage_bf16_to_float(ffn_out[base + j]);
        sum = fmaf(y, y, sum);
    }
    sum = zimage_block_reduce_sum<BLOCK_THREADS>(sum);

    __shared__ float inv;
    if (tid == 0) inv = rsqrtf(sum / float(dim) + eps);
    __syncthreads();

    for (int j = tid; j < dim; j += BLOCK_THREADS) {
        float norm2 = zimage_bf16_to_float(zimage_float_to_bf16(
            zimage_bf16_to_float(ffn_out[base + j]) * inv * zimage_bf16_to_float(ffn_norm2_weight[j])
        ));
        float gated = zimage_bf16_to_float(zimage_float_to_bf16(norm2 * zimage_bf16_to_float(gate_mlp[j])));
        dst[base + j] = zimage_float_to_bf16(zimage_bf16_to_float(residual_mid[base + j]) + gated);
    }
}

extern "C" void zimage_fused_ffn_down_rmsnorm_gate_residual_bf16_forward(
    __nv_bfloat16* dst,
    const __nv_bfloat16* residual_mid,
    const __nv_bfloat16* ffn_out,
    const __nv_bfloat16* gate_mlp,
    const __nv_bfloat16* ffn_norm2_weight,
    int rows,
    int dim,
    float ffn_norm2_eps,
    cudaStream_t stream
) {
    constexpr int threads = 256;
    zimage_fused_ffn_down_rmsnorm_gate_residual_bf16_kernel<threads><<<rows, threads, 0, stream>>>(
        dst,
        residual_mid,
        ffn_out,
        gate_mlp,
        ffn_norm2_weight,
        dim,
        ffn_norm2_eps
    );
}
