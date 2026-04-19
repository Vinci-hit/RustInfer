/**
 * INT4 GEMV / GEMM CUDA Kernel (BF16 input/output) — K-packed Layout
 *
 * Weight layout (compressed-tensors / pack-quantized format):
 *   weight_packed:     [N, K/8]        (int32) — 8 consecutive K-position INT4 per int32
 *   weight_zero_point: [N/8, num_groups] (int32) — zero points packed along N
 *   weight_scale:      [N, num_groups]   (bf16)  — per-group scale factors
 *
 * Dequant: w = (extract(packed, k%8) - extract(zeros, n%8)) * scale
 *
 * GEMV: 1 warp per output row, int4 vectorized reads along K/8.
 *       Each int4 load = 4 × int32 = 32 INT4 = 32 K positions.
 *       Input also loaded as int4 (8 bf16 = 16 bytes).
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
//  INT4 GEMV (decode, M=1) — K-packed, int4 vectorized, BF16
// ============================================================================

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
    const int K_packed = K >> 3;
    const int num_groups = K / group_size;

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int32_t* wp_row = weight_packed + row * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + row * num_groups;

    const int zp_row_packed = row >> 3;
    const int zp_bit_offset = (row & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_row_packed * num_groups;

    const int4* wp_i4 = reinterpret_cast<const int4*>(wp_row);
    const int num_int4 = K_packed >> 2;  // K / 32

    const int4* input_i4 = reinterpret_cast<const int4*>(input);

    float acc = 0.0f;

    for (int i4 = lane_id; i4 < num_int4; i4 += 32) {
        int4 w4 = __ldg(&wp_i4[i4]);
        int k_base = i4 * 32;
        int g = k_base / group_size;

        float scale = __bfloat162float(__ldg(&sc_row[g]));
        int32_t zp_packed = __ldg(&zp_row[g]);
        int zero = (zp_packed >> zp_bit_offset) & 0xF;

        int input_i4_base = k_base >> 3;
        int4 in0 = __ldg(&input_i4[input_i4_base]);
        int4 in1 = __ldg(&input_i4[input_i4_base + 1]);
        int4 in2 = __ldg(&input_i4[input_i4_base + 2]);
        int4 in3 = __ldg(&input_i4[input_i4_base + 3]);

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

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) {
        output[row] = __float2bfloat16(acc);
    }
}

// ============================================================================
//  INT4 GEMM (prefill, M>1) — K-packed, BF16
// ============================================================================
#define INT4_GEMM_BX 16
#define INT4_GEMM_BY 16

extern "C" __global__ void kpack_gemm_kernel(
    const __nv_bfloat16* __restrict__ input,         // [M, K]
    const int32_t* __restrict__ weight_packed,        // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,    // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,   // [N, num_groups]
    __nv_bfloat16* __restrict__ output,               // [M, N]
    int M, int N, int K, int group_size
) {
    const int row = blockIdx.y * INT4_GEMM_BY + threadIdx.y;  // M dim
    const int col = blockIdx.x * INT4_GEMM_BX + threadIdx.x;  // N dim
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
    dim3 block(INT4_GEMM_BX, INT4_GEMM_BY);
    dim3 grid((N + INT4_GEMM_BX - 1) / INT4_GEMM_BX, (M + INT4_GEMM_BY - 1) / INT4_GEMM_BY);
    kpack_gemm_kernel<<<grid, block, 0, stream>>>(
        (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
        (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
        (__nv_bfloat16*)output, M, N, K, group_size);
}
