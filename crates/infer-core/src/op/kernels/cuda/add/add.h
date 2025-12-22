#include <cuda_runtime.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

void add_kernel_float2_forward(
    float* c,
    const float* a,
    const float* b,
    int num_elements,
    cudaStream_t stream
);

void add_inplace_kernel_float2_forward(
    float* a_and_c,
    const float* b,
    int num_elements,
    cudaStream_t stream
);

void add_kernel_bf16x8(
    __nv_bfloat16* c,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    int num_elements,
    cudaStream_t stream
);

void add_inplace_kernel_bf16x8(
    __nv_bfloat16* a_and_c,
    const __nv_bfloat16* b,
    int num_elements,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif