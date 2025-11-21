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

#ifdef __cplusplus
}
#endif