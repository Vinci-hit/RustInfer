#include "permute.h"

// Max supported dimensions
#define MAX_DIMS 8

// Pass meta as a struct-by-value so CUDA can copy all three arrays into
// constant kernel-arg space (up to 4 KB). This avoids passing host pointers.
struct PermuteMeta {
    int64_t new_strides[MAX_DIMS];
    int64_t old_strides[MAX_DIMS];
    int perm[MAX_DIMS];
};

template<typename T>
__global__ void permute_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int ndim,
    PermuteMeta meta,
    int64_t num_elements)
{
    int64_t flat_new = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = flat_new; idx < num_elements; idx += stride) {
        int64_t old_flat = 0;
        int64_t rem = idx;
        #pragma unroll
        for (int j = 0; j < MAX_DIMS; ++j) {
            if (j >= ndim) break;
            int64_t coord = rem / meta.new_strides[j];
            rem %= meta.new_strides[j];
            old_flat += coord * meta.old_strides[meta.perm[j]];
        }
        dst[idx] = src[old_flat];
    }
}

template<typename T>
static void permute_launch(
    T* dst, const T* src,
    int ndim,
    const int64_t* new_strides_h,
    const int64_t* old_strides_h,
    const int* perm_h,
    int64_t num_elements,
    cudaStream_t stream)
{
    PermuteMeta meta;
    for (int j = 0; j < MAX_DIMS; ++j) {
        meta.new_strides[j] = (j < ndim) ? new_strides_h[j] : 0;
        meta.old_strides[j] = (j < ndim) ? old_strides_h[j] : 0;
        meta.perm[j] = (j < ndim) ? perm_h[j] : 0;
    }
    constexpr int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    permute_kernel<T><<<blocks, threads, 0, stream>>>(
        dst, src, ndim, meta, num_elements);
}

extern "C" void permute_f32_forward(
    float* dst, const float* src,
    int ndim,
    const int64_t* new_shape, const int64_t* new_strides,
    const int64_t* old_strides, const int* perm,
    int64_t num_elements, cudaStream_t stream)
{
    (void)new_shape;
    permute_launch<float>(dst, src, ndim, new_strides, old_strides, perm, num_elements, stream);
}
extern "C" void permute_bf16_forward(
    __nv_bfloat16* dst, const __nv_bfloat16* src,
    int ndim,
    const int64_t* new_shape, const int64_t* new_strides,
    const int64_t* old_strides, const int* perm,
    int64_t num_elements, cudaStream_t stream)
{
    (void)new_shape;
    permute_launch<__nv_bfloat16>(dst, src, ndim, new_strides, old_strides, perm, num_elements, stream);
}
extern "C" void permute_f16_forward(
    __half* dst, const __half* src,
    int ndim,
    const int64_t* new_shape, const int64_t* new_strides,
    const int64_t* old_strides, const int* perm,
    int64_t num_elements, cudaStream_t stream)
{
    (void)new_shape;
    permute_launch<__half>(dst, src, ndim, new_strides, old_strides, perm, num_elements, stream);
}
extern "C" void permute_i32_forward(
    int32_t* dst, const int32_t* src,
    int ndim,
    const int64_t* new_shape, const int64_t* new_strides,
    const int64_t* old_strides, const int* perm,
    int64_t num_elements, cudaStream_t stream)
{
    (void)new_shape;
    permute_launch<int32_t>(dst, src, ndim, new_strides, old_strides, perm, num_elements, stream);
}
