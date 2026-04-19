/**
 * AWQ INT4 GEMV / GEMM CUDA Kernel (FP16 input/output) — Transposed Layout
 *
 * 权重在加载时已转置，使 GEMV 沿 K 方向 coalesced 访问：
 *   qweight_t: [N/8, K]  (原 [K, N/8] 转置)
 *   qzeros_t:  [N/8, num_groups]  (原 [num_groups, N/8] 转置)
 *   scales_t:  [N, num_groups]  (原 [num_groups, N] 转置)
 *
 * GEMV: 1 warp 算 8 输出 (1 packed column)，32 lanes 沿 K 方向读
 *   qweight_t[col_packed, k] → 地址 col_packed * K + k → lane 间 stride=1 → coalesced!
 *
 * AWQ packing order: [0, 4, 1, 5, 2, 6, 3, 7]
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __constant__ int AWQ_ORDER[8] = {0, 4, 1, 5, 2, 6, 3, 7};

__device__ __forceinline__ int awq_extract(int packed, int pos) {
    return (packed >> (AWQ_ORDER[pos] * 4)) & 0xF;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
//  AWQ GEMV (decode, M=1) — Transposed layout + int4 vectorized loads
//
//  Each warp computes 8 outputs (1 packed column).
//  int4 load: 4 consecutive int32 per lane per load = 32 INT4 weights.
//  Matches BF16 GEMV's float4 load granularity.
// ============================================================================

__device__ __forceinline__ void process_packed(
    int32_t qw, int32_t qz, float x_val, float scale[8], float acc[8]
) {
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        int w = awq_extract(qw, c);
        int z = awq_extract(qz, c);
        acc[c] += (float)(w - z) * scale[c] * x_val;
    }
}

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
awq_gemv_kernel(
    const half* __restrict__ input,           // [K]
    const int32_t* __restrict__ qweight_t,    // [N/8, K] (transposed)
    const int32_t* __restrict__ qzeros_t,     // [N/8, num_groups] (transposed)
    const half* __restrict__ scales_t,        // [N, num_groups] (transposed)
    half* __restrict__ output,                // [N]
    const int N, const int K, const int group_size
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int N_packed = N >> 3;
    const int num_groups = K / group_size;

    const int col_packed = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (col_packed >= N_packed) return;
    const int col_base = col_packed << 3;

    const int32_t* qw_row = qweight_t + col_packed * K;
    const int32_t* qz_row = qzeros_t + col_packed * num_groups;
    const half* sc_base = scales_t + col_base * num_groups;

    // int4 = 4 consecutive int32 = 4 K positions per load
    const int4* qw_i4 = reinterpret_cast<const int4*>(qw_row);
    const int num_int4 = K / 4;

    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int i4 = lane_id; i4 < num_int4; i4 += 32) {
        int4 qw4 = __ldg(&qw_i4[i4]);
        int k_base = i4 * 4;
        int g = k_base / group_size;

        float x0 = __half2float(__ldg(&input[k_base]));
        float x1 = __half2float(__ldg(&input[k_base + 1]));
        float x2 = __half2float(__ldg(&input[k_base + 2]));
        float x3 = __half2float(__ldg(&input[k_base + 3]));

        int32_t qz = __ldg(&qz_row[g]);
        float sc[8];
        #pragma unroll
        for (int c = 0; c < 8; c++) {
            sc[c] = __half2float(__ldg(&sc_base[c * num_groups + g]));
        }

        process_packed(qw4.x, qz, x0, sc, acc);
        process_packed(qw4.y, qz, x1, sc, acc);
        process_packed(qw4.z, qz, x2, sc, acc);
        process_packed(qw4.w, qz, x3, sc, acc);
    }

    #pragma unroll
    for (int c = 0; c < 8; c++) { acc[c] = warp_reduce_sum(acc[c]); }
    if (lane_id == 0) {
        #pragma unroll
        for (int c = 0; c < 8; c++) { output[col_base + c] = __float2half(acc[c]); }
    }
}

// ============================================================================
//  AWQ GEMM (prefill, M>1) — Also uses transposed layout
// ============================================================================
#define AWQ_GEMM_BX 16
#define AWQ_GEMM_BY 16

extern "C" __global__ void awq_gemm_kernel(
    const half* __restrict__ input,           // [M, K]
    const int32_t* __restrict__ qweight_t,    // [N/8, K] (transposed)
    const int32_t* __restrict__ qzeros_t,     // [N/8, num_groups] (transposed)
    const half* __restrict__ scales_t,        // [N, num_groups] (transposed)
    half* __restrict__ output,                // [M, N]
    int M, int N, int K, int group_size
) {
    const int row = blockIdx.y * AWQ_GEMM_BY + threadIdx.y;  // M dim
    const int col = blockIdx.x * AWQ_GEMM_BX + threadIdx.x;  // N dim
    if (row >= M || col >= N) return;

    const int col_packed = col / 8;
    const int col_in_pack = col % 8;
    const int num_groups = K / group_size;

    const int32_t* qw_row = qweight_t + col_packed * K;
    const int32_t* qz_row = qzeros_t + col_packed * num_groups;

    float acc = 0.0f;

    for (int k = 0; k < K; k++) {
        int group_idx = k / group_size;
        int32_t qw = qw_row[k];
        int w_int4 = awq_extract(qw, col_in_pack);
        int32_t qz = qz_row[group_idx];
        int zero = awq_extract(qz, col_in_pack);
        float scale_f = __half2float(scales_t[col * num_groups + group_idx]);
        acc += (float)(w_int4 - zero) * scale_f * __half2float(input[row * K + k]);
    }
    output[row * N + col] = __float2half(acc);
}

// ============================================================================
//  K-packed INT4 GEMV (compressed-tensors format, BF16 input/output)
//
//  Weight layout: weight_packed [N, K/8] int32 — 8 consecutive K-positions per int32
//  Zero point:    weight_zero_point [N/8, num_groups] int32 — packed along N
//  Scale:         weight_scale [N, num_groups] bf16
//
//  Dequant: w = (extract_k(packed, k%8) - extract_n(zeros, n%8)) * scale
//
//  Sequential INT4 packing (no AWQ_ORDER).
//  Each warp handles 1 output row, reads K/8 int32s along K with int4 vectorization.
// ============================================================================

__device__ __forceinline__ int kpack_extract(int packed, int pos) {
    return (packed >> (pos * 4)) & 0xF;
}

__device__ __forceinline__ float warp_reduce_sum_bf16(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
kpack_gemv_kernel(
    const __nv_bfloat16* __restrict__ input,        // [K]
    const int32_t* __restrict__ weight_packed,       // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,   // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,  // [N, num_groups]
    __nv_bfloat16* __restrict__ output,              // [N]
    const int N, const int K, const int group_size
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int K_packed = K >> 3;   // K / 8
    const int num_groups = K / group_size;

    // Each warp handles one output row
    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    // Pointers for this row
    const int32_t* wp_row = weight_packed + row * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + row * num_groups;

    // Zero point: packed along N, so row's zero is in weight_zero_point[row/8][g] at bit (row%8)*4
    const int zp_row_packed = row >> 3;
    const int zp_bit_offset = (row & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_row_packed * num_groups;

    // int4 vectorized loads for weight: each int4 = 4 × int32 = 32 INT4 values = 32 K positions
    const int4* wp_i4 = reinterpret_cast<const int4*>(wp_row);
    const int num_int4 = K_packed >> 2;  // K / 8 / 4 = K / 32

    // Vectorized input pointer: 8 bf16 = 16 bytes = int4
    const int4* input_i4 = reinterpret_cast<const int4*>(input);

    float acc = 0.0f;

    for (int i4 = lane_id; i4 < num_int4; i4 += 32) {
        int4 w4 = __ldg(&wp_i4[i4]);
        int k_base = i4 * 32;  // each int4 covers 32 K positions (4 int32 × 8 INT4)
        int g = k_base / group_size;

        float scale = __bfloat162float(__ldg(&sc_row[g]));
        int32_t zp_packed = __ldg(&zp_row[g]);
        int zero = (zp_packed >> zp_bit_offset) & 0xF;

        // Load input: 32 bf16 = 64 bytes = 4 int4 loads (each int4 = 8 bf16)
        // input offset: k_base/8 int4 elements from start
        int input_i4_base = k_base >> 3;  // k_base / 8, since each int4 has 8 bf16
        int4 in0 = __ldg(&input_i4[input_i4_base]);
        int4 in1 = __ldg(&input_i4[input_i4_base + 1]);
        int4 in2 = __ldg(&input_i4[input_i4_base + 2]);
        int4 in3 = __ldg(&input_i4[input_i4_base + 3]);

        // Process 4 int32 words, each with 8 INT4 weights and 8 BF16 inputs
        // word 0 -> in0 (8 bf16), word 1 -> in1, word 2 -> in2, word 3 -> in3
        const __nv_bfloat16* inp0 = reinterpret_cast<const __nv_bfloat16*>(&in0);
        const __nv_bfloat16* inp1 = reinterpret_cast<const __nv_bfloat16*>(&in1);
        const __nv_bfloat16* inp2 = reinterpret_cast<const __nv_bfloat16*>(&in2);
        const __nv_bfloat16* inp3 = reinterpret_cast<const __nv_bfloat16*>(&in3);

        int32_t word;
        float dz = (float)(-zero) * scale;

        word = w4.x;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_int4 = (word >> (j * 4)) & 0xF;
            acc += ((float)w_int4 * scale + dz) * __bfloat162float(inp0[j]);
        }
        word = w4.y;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_int4 = (word >> (j * 4)) & 0xF;
            acc += ((float)w_int4 * scale + dz) * __bfloat162float(inp1[j]);
        }
        word = w4.z;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_int4 = (word >> (j * 4)) & 0xF;
            acc += ((float)w_int4 * scale + dz) * __bfloat162float(inp2[j]);
        }
        word = w4.w;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_int4 = (word >> (j * 4)) & 0xF;
            acc += ((float)w_int4 * scale + dz) * __bfloat162float(inp3[j]);
        }
    }

    // Handle remainder (K_packed not divisible by 4)
    int k_packed_start = num_int4 * 4;
    for (int kp = k_packed_start + lane_id; kp < K_packed; kp += 32) {
        int k_base = kp * 8;
        int g = k_base / group_size;
        float scale = __bfloat162float(__ldg(&sc_row[g]));
        int32_t zp_packed = __ldg(&zp_row[g]);
        int zero = (zp_packed >> zp_bit_offset) & 0xF;
        int32_t word = __ldg(&wp_row[kp]);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_int4 = (word >> (j * 4)) & 0xF;
            float x_val = __bfloat162float(__ldg(&input[k_base + j]));
            acc += (float)(w_int4 - zero) * scale * x_val;
        }
    }

    acc = warp_reduce_sum_bf16(acc);
    if (lane_id == 0) {
        output[row] = __float2bfloat16(acc);
    }
}

// ============================================================================
//  K-packed INT4 GEMM (prefill, M>1) — BF16
// ============================================================================
#define KPACK_GEMM_BX 16
#define KPACK_GEMM_BY 16

extern "C" __global__ void kpack_gemm_kernel(
    const __nv_bfloat16* __restrict__ input,         // [M, K]
    const int32_t* __restrict__ weight_packed,        // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,    // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,   // [N, num_groups]
    __nv_bfloat16* __restrict__ output,               // [M, N]
    int M, int N, int K, int group_size
) {
    const int row = blockIdx.y * KPACK_GEMM_BY + threadIdx.y;  // M dim
    const int col = blockIdx.x * KPACK_GEMM_BX + threadIdx.x;  // N dim
    if (row >= M || col >= N) return;

    const int K_packed = K / 8;
    const int num_groups = K / group_size;

    const int32_t* wp_row = weight_packed + col * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + col * num_groups;

    const int zp_col_packed = col >> 3;
    const int zp_bit_offset = (col & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_col_packed * num_groups;

    float acc = 0.0f;

    for (int kp = 0; kp < K_packed; kp++) {
        int k_base = kp * 8;
        int g = k_base / group_size;
        float scale = __bfloat162float(sc_row[g]);
        int32_t zp_packed = zp_row[g];
        int zero = (zp_packed >> zp_bit_offset) & 0xF;
        int32_t word = wp_row[kp];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int w_int4 = (word >> (j * 4)) & 0xF;
            float x_val = __bfloat162float(input[row * K + k_base + j]);
            acc += (float)(w_int4 - zero) * scale * x_val;
        }
    }
    output[row * N + col] = __float2bfloat16(acc);
}

// ============================================================================
//  C-linkage wrappers
// ============================================================================

extern "C" void awq_gemv_cu(
    const void* input, const void* qweight_t, const void* qzeros_t,
    const void* scales_t, void* output,
    int N, int K, int group_size, cudaStream_t stream
) {
    constexpr int WARPS = 4;
    int N_packed = N / 8;
    int grid_x = (N_packed + WARPS - 1) / WARPS;

    awq_gemv_kernel<WARPS><<<grid_x, WARPS * 32, 0, stream>>>(
        (const half*)input, (const int32_t*)qweight_t,
        (const int32_t*)qzeros_t, (const half*)scales_t,
        (half*)output, N, K, group_size);
}

extern "C" void awq_gemm_cu(
    const void* input, const void* qweight_t, const void* qzeros_t,
    const void* scales_t, void* output,
    int M, int N, int K, int group_size, cudaStream_t stream
) {
    dim3 block(AWQ_GEMM_BX, AWQ_GEMM_BY);
    dim3 grid((N + AWQ_GEMM_BX - 1) / AWQ_GEMM_BX, (M + AWQ_GEMM_BY - 1) / AWQ_GEMM_BY);
    awq_gemm_kernel<<<grid, block, 0, stream>>>(
        (const half*)input, (const int32_t*)qweight_t,
        (const int32_t*)qzeros_t, (const half*)scales_t,
        (half*)output, M, N, K, group_size);
}

extern "C" void kpack_gemv_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int N, int K, int group_size, cudaStream_t stream
) {
    constexpr int WARPS = 4;
    int grid_x = (N + WARPS - 1) / WARPS;

    kpack_gemv_kernel<WARPS><<<grid_x, WARPS * 32, 0, stream>>>(
        (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
        (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
        (__nv_bfloat16*)output, N, K, group_size);
}

extern "C" void kpack_gemm_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int M, int N, int K, int group_size, cudaStream_t stream
) {
    dim3 block(KPACK_GEMM_BX, KPACK_GEMM_BY);
    dim3 grid((N + KPACK_GEMM_BX - 1) / KPACK_GEMM_BX, (M + KPACK_GEMM_BY - 1) / KPACK_GEMM_BY);
    kpack_gemm_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
        (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
        (__nv_bfloat16*)output, M, N, K, group_size);
}
