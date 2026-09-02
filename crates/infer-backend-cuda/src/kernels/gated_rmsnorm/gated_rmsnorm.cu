#include <cub/block/block_reduce.cuh>

#include "gated_rmsnorm.h"

namespace {

template <typename T>
__device__ __forceinline__ float load_float(T value);

template <>
__device__ __forceinline__ float load_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float load_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float load_float<__half>(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T store_float(float value);

template <>
__device__ __forceinline__ float store_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ __half store_float<__half>(float value) {
    return __float2half_rn(value);
}

template <typename T>
__global__ void gated_rmsnorm_kernel(
    const T* input,
    const T* gate,
    const float* __restrict__ weight,
    T* output,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps) {
    constexpr int kThreads = 128;
    using BlockReduce = cub::BlockReduce<float, kThreads>;

    const int row = static_cast<int>(blockIdx.x);
    const int o0 = outer1 == 1 ? row : row / outer1;
    const int o1 = outer1 == 1 ? 0 : row % outer1;
    const int64_t input_offset = static_cast<int64_t>(o0) * input_stride0
                               + static_cast<int64_t>(o1) * input_stride1;
    const int64_t gate_offset = static_cast<int64_t>(o0) * gate_stride0
                              + static_cast<int64_t>(o1) * gate_stride1;
    const int64_t output_offset = static_cast<int64_t>(o0) * output_stride0
                                + static_cast<int64_t>(o1) * output_stride1;
    const T* input_row = input + input_offset;
    const T* gate_row = gate + gate_offset;
    T* output_row = output + output_offset;

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        const float value = load_float(input_row[col]);
        local_sum = fmaf(value, value, local_sum);
    }

    __shared__ typename BlockReduce::TempStorage reduction_storage;
    const float square_sum = BlockReduce(reduction_storage).Sum(local_sum);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(square_sum / static_cast<float>(head_dim) + eps);
    }
    // Besides broadcasting inv_rms, this barrier guarantees that every input
    // element has been consumed before an aliased input/output view is written.
    __syncthreads();

    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
        const float normalized_fp32 = load_float(input_row[col]) * inv_rms;
        // Match Qwen3.5 exactly: normalized activations are converted back to
        // the model dtype before multiplication by the fp32 norm weight.
        const T normalized_t = store_float<T>(normalized_fp32);
        const float normalized = load_float(normalized_t);
        const float gate_value = load_float(gate_row[col]);
        const float silu_gate = gate_value / (1.0f + expf(-gate_value));
        output_row[col] = store_float<T>(normalized * weight[col] * silu_gate);
    }
}

template <typename T>
void launch_gated_rmsnorm(
    const T* input,
    const T* gate,
    const float* weight,
    T* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream) {
    constexpr int kThreads = 128;
    const int rows = outer0 * outer1;
    gated_rmsnorm_kernel<T><<<rows, kThreads, 0, stream>>>(
        input,
        gate,
        weight,
        output,
        outer1,
        head_dim,
        input_stride0,
        input_stride1,
        gate_stride0,
        gate_stride1,
        output_stride0,
        output_stride1,
        eps);
}

}  // namespace

extern "C" void gated_rmsnorm_f32_forward(
    const float* input,
    const float* gate,
    const float* weight,
    float* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream) {
    launch_gated_rmsnorm(
        input, gate, weight, output, outer0, outer1, head_dim,
        input_stride0, input_stride1, gate_stride0, gate_stride1,
        output_stride0, output_stride1, eps, stream);
}

extern "C" void gated_rmsnorm_bf16_forward(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate,
    const float* weight,
    __nv_bfloat16* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream) {
    launch_gated_rmsnorm(
        input, gate, weight, output, outer0, outer1, head_dim,
        input_stride0, input_stride1, gate_stride0, gate_stride1,
        output_stride0, output_stride1, eps, stream);
}

extern "C" void gated_rmsnorm_f16_forward(
    const __half* input,
    const __half* gate,
    const float* weight,
    __half* output,
    int outer0,
    int outer1,
    int head_dim,
    int64_t input_stride0,
    int64_t input_stride1,
    int64_t gate_stride0,
    int64_t gate_stride1,
    int64_t output_stride0,
    int64_t output_stride1,
    float eps,
    cudaStream_t stream) {
    launch_gated_rmsnorm(
        input, gate, weight, output, outer0, outer1, head_dim,
        input_stride0, input_stride1, gate_stride0, gate_stride1,
        output_stride0, output_stride1, eps, stream);
}
