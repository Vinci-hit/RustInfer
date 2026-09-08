#include "causal_conv1d.h"

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
    return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ __half store_float<__half>(float value) {
    return __float2half(value);
}

template <typename T>
__global__ void causal_conv1d_silu_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ conv_state,
    const int* __restrict__ state_slots,
    const int* __restrict__ cu_seqlens,
    T* __restrict__ output,
    int num_tokens,
    int channels,
    int kernel_size,
    int num_slots)
{
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int sequence = blockIdx.y;
    if (channel >= channels) return;

    const int slot = state_slots[sequence];
    const int start = cu_seqlens[sequence];
    const int end = cu_seqlens[sequence + 1];
    if (slot < 0 || slot >= num_slots || start < 0 || end < start || end > num_tokens) return;

    const int sequence_length = end - start;
    T* state = conv_state +
        (static_cast<long long>(slot) * channels + channel) * kernel_size;
    const T* channel_weight = weight + static_cast<long long>(channel) * kernel_size;

    // One thread owns one (sequence, channel), so its caller-provided state is
    // read consistently for the whole ragged segment and written only after
    // every output has been produced. Adjacent threads walk adjacent channels,
    // giving coalesced input/output accesses at each token.
    for (int token = 0; token < sequence_length; ++token) {
        float sum = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int relative = token + tap + 1 - kernel_size;
            const T value = relative >= 0
                ? input[static_cast<long long>(start + relative) * channels + channel]
                : state[kernel_size + relative];
            sum = fmaf(load_float(value), load_float(channel_weight[tap]), sum);
        }
        const float activated = sum / (1.0f + expf(-sum));
        output[static_cast<long long>(start + token) * channels + channel] =
            store_float<T>(activated);
    }

    // Retain the last kernel_size raw inputs from [old_state, current chunk].
    // For a short chunk, source state indices are strictly ahead of their
    // destinations, so the forward copy cannot overwrite a value still needed.
    for (int state_col = 0; state_col < kernel_size; ++state_col) {
        const int concat_col = sequence_length + state_col;
        state[state_col] = concat_col < kernel_size
            ? state[concat_col]
            : input[static_cast<long long>(start + concat_col - kernel_size) * channels + channel];
    }
}

template <typename T>
void launch_causal_conv1d_silu(
    const T* input,
    const T* weight,
    T* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    T* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || batch <= 0 || channels <= 0 || kernel_size <= 0 || num_slots <= 0) {
        return;
    }
    constexpr int threads = 256;
    const dim3 grid((channels + threads - 1) / threads, batch);
    causal_conv1d_silu_kernel<<<grid, threads, 0, stream>>>(
        input,
        weight,
        conv_state,
        state_slots,
        cu_seqlens,
        output,
        num_tokens,
        channels,
        kernel_size,
        num_slots);
}

} // namespace

extern "C" void causal_conv1d_silu_f32_forward(
    const float* input,
    const float* weight,
    float* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    float* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream)
{
    launch_causal_conv1d_silu(
        input, weight, conv_state, state_slots, cu_seqlens, output,
        num_tokens, batch, channels, kernel_size, num_slots, stream);
}

extern "C" void causal_conv1d_silu_bf16_forward(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    __nv_bfloat16* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream)
{
    launch_causal_conv1d_silu(
        input, weight, conv_state, state_slots, cu_seqlens, output,
        num_tokens, batch, channels, kernel_size, num_slots, stream);
}

extern "C" void causal_conv1d_silu_f16_forward(
    const __half* input,
    const __half* weight,
    __half* conv_state,
    const int* state_slots,
    const int* cu_seqlens,
    __half* output,
    int num_tokens,
    int batch,
    int channels,
    int kernel_size,
    int num_slots,
    cudaStream_t stream)
{
    launch_causal_conv1d_silu(
        input, weight, conv_state, state_slots, cu_seqlens, output,
        num_tokens, batch, channels, kernel_size, num_slots, stream);
}
