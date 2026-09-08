#include "gated_delta_rule.h"

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

__device__ __forceinline__ float stable_sigmoid(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + expf(-value));
    }
    const float exp_value = expf(value);
    return exp_value / (1.0f + exp_value);
}

__device__ __forceinline__ float stable_softplus(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
}

template <typename T>
__global__ void gated_delta_rule_kernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ a_log,
    const T* __restrict__ dt_bias,
    float* __restrict__ recurrent_state,
    const int* __restrict__ state_slots,
    const int* __restrict__ cu_seqlens,
    T* __restrict__ output,
    int num_tokens,
    int batch,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int num_slots,
    int64_t query_row_stride,
    int64_t key_row_stride,
    int64_t value_row_stride,
    int64_t a_row_stride,
    int64_t b_row_stride)
{
    const int pair = blockIdx.x;
    const int sequence = pair / num_value_heads;
    const int value_head = pair - sequence * num_value_heads;
    if (sequence >= batch) return;

    const int slot = state_slots[sequence];
    const int start = cu_seqlens[sequence];
    const int end = cu_seqlens[sequence + 1];
    if (slot < 0 || slot >= num_slots || start < 0 || end < start || end > num_tokens) return;

    const int value_heads_per_key = num_value_heads / num_key_heads;
    const int key_head = value_head / value_heads_per_key;
    const int thread = threadIdx.x;

    extern __shared__ float shared[];
    float* query_vector = shared;
    float* key_vector = query_vector + key_head_dim;
    float* query_reduce = key_vector + key_head_dim;
    float* key_reduce = query_reduce + blockDim.x;

    float* state = recurrent_state +
        ((static_cast<int64_t>(slot) * num_value_heads + value_head) * key_head_dim) * value_head_dim;
    constexpr float qk_l2_eps = 1.0e-6f;

    for (int token = start; token < end; ++token) {
        float query_sum = 0.0f;
        float key_sum = 0.0f;
        for (int dim = thread; dim < key_head_dim; dim += blockDim.x) {
            const float query_value = load_float(
                query[static_cast<int64_t>(token) * query_row_stride +
                    static_cast<int64_t>(key_head) * key_head_dim + dim]);
            const float key_value = load_float(
                key[static_cast<int64_t>(token) * key_row_stride +
                    static_cast<int64_t>(key_head) * key_head_dim + dim]);
            query_vector[dim] = query_value;
            key_vector[dim] = key_value;
            query_sum = fmaf(query_value, query_value, query_sum);
            key_sum = fmaf(key_value, key_value, key_sum);
        }
        query_reduce[thread] = query_sum;
        key_reduce[thread] = key_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (thread < offset) {
                query_reduce[thread] += query_reduce[thread + offset];
                key_reduce[thread] += key_reduce[thread + offset];
            }
            __syncthreads();
        }

        const float query_scale =
            rsqrtf(query_reduce[0] + qk_l2_eps) * rsqrtf(float(key_head_dim));
        const float key_scale = rsqrtf(key_reduce[0] + qk_l2_eps);
        for (int dim = thread; dim < key_head_dim; dim += blockDim.x) {
            query_vector[dim] *= query_scale;
            key_vector[dim] *= key_scale;
        }
        __syncthreads();

        if (thread == 0) {
            const int64_t head_offset = static_cast<int64_t>(token) * a_row_stride + value_head;
            const float raw_a = load_float(a[head_offset]);
            const float log_decay = -expf(a_log[value_head]) *
                stable_softplus(raw_a + load_float(dt_bias[value_head]));
            query_reduce[0] = expf(log_decay);

            const float raw_b =
                load_float(b[static_cast<int64_t>(token) * b_row_stride + value_head]);
            // HF applies sigmoid in the activation dtype, then promotes beta
            // to fp32 inside the recurrence.
            key_reduce[0] = load_float(store_float<T>(stable_sigmoid(raw_b)));
        }
        __syncthreads();
        const float decay = query_reduce[0];
        const float beta = key_reduce[0];

        const T* value_row = value + static_cast<int64_t>(token) * value_row_stride +
            static_cast<int64_t>(value_head) * value_head_dim;
        T* output_row = output +
            (static_cast<int64_t>(token) * num_value_heads + value_head) * value_head_dim;

        // Threads own value columns. For each column, first decay the fp32
        // state and read its key projection, then apply the rank-one delta
        // update and read the updated state with the normalized query.
        for (int value_dim = thread; value_dim < value_head_dim; value_dim += blockDim.x) {
            float memory = 0.0f;
            for (int key_dim = 0; key_dim < key_head_dim; ++key_dim) {
                const int64_t state_index =
                    static_cast<int64_t>(key_dim) * value_head_dim + value_dim;
                const float decayed = state[state_index] * decay;
                state[state_index] = decayed;
                memory = fmaf(decayed, key_vector[key_dim], memory);
            }

            const float delta = (load_float(value_row[value_dim]) - memory) * beta;
            float result = 0.0f;
            for (int key_dim = 0; key_dim < key_head_dim; ++key_dim) {
                const int64_t state_index =
                    static_cast<int64_t>(key_dim) * value_head_dim + value_dim;
                const float updated = fmaf(key_vector[key_dim], delta, state[state_index]);
                state[state_index] = updated;
                result = fmaf(updated, query_vector[key_dim], result);
            }
            output_row[value_dim] = store_float<T>(result);
        }

        // Prevent a fast thread from overwriting the shared Q/K vectors for
        // the next token while another thread still consumes this token.
        __syncthreads();
    }
}

template <typename T>
void launch_gated_delta_rule(
    const T* query,
    const T* key,
    const T* value,
    const T* a,
    const T* b,
    const float* a_log,
    const T* dt_bias,
    float* recurrent_state,
    const int* state_slots,
    const int* cu_seqlens,
    T* output,
    int num_tokens,
    int batch,
    int num_key_heads,
    int num_value_heads,
    int key_head_dim,
    int value_head_dim,
    int num_slots,
    int64_t query_row_stride,
    int64_t key_row_stride,
    int64_t value_row_stride,
    int64_t a_row_stride,
    int64_t b_row_stride,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || batch <= 0 || num_key_heads <= 0 || num_value_heads <= 0 ||
        key_head_dim <= 0 || value_head_dim <= 0 || num_slots <= 0) {
        return;
    }
    constexpr int threads = 128;
    const int blocks = batch * num_value_heads;
    const size_t shared_bytes =
        static_cast<size_t>(2 * key_head_dim + 2 * threads) * sizeof(float);
    gated_delta_rule_kernel<<<blocks, threads, shared_bytes, stream>>>(
        query,
        key,
        value,
        a,
        b,
        a_log,
        dt_bias,
        recurrent_state,
        state_slots,
        cu_seqlens,
        output,
        num_tokens,
        batch,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        num_slots,
        query_row_stride,
        key_row_stride,
        value_row_stride,
        a_row_stride,
        b_row_stride);
}

} // namespace

#define DEFINE_GATED_DELTA_ENTRY(SUFFIX, TYPE) \
extern "C" void gated_delta_rule_##SUFFIX##_forward( \
    const TYPE* query, \
    const TYPE* key, \
    const TYPE* value, \
    const TYPE* a, \
    const TYPE* b, \
    const float* a_log, \
    const TYPE* dt_bias, \
    float* recurrent_state, \
    const int* state_slots, \
    const int* cu_seqlens, \
    TYPE* output, \
    int num_tokens, \
    int batch, \
    int num_key_heads, \
    int num_value_heads, \
    int key_head_dim, \
    int value_head_dim, \
    int num_slots, \
    int64_t query_row_stride, \
    int64_t key_row_stride, \
    int64_t value_row_stride, \
    int64_t a_row_stride, \
    int64_t b_row_stride, \
    cudaStream_t stream) \
{ \
    launch_gated_delta_rule( \
        query, key, value, a, b, a_log, dt_bias, recurrent_state, state_slots, cu_seqlens, output, \
        num_tokens, batch, num_key_heads, num_value_heads, key_head_dim, value_head_dim, num_slots, \
        query_row_stride, key_row_stride, value_row_stride, a_row_stride, b_row_stride, stream); \
}

DEFINE_GATED_DELTA_ENTRY(f32, float)
DEFINE_GATED_DELTA_ENTRY(bf16, __nv_bfloat16)
DEFINE_GATED_DELTA_ENTRY(f16, __half)

#undef DEFINE_GATED_DELTA_ENTRY
