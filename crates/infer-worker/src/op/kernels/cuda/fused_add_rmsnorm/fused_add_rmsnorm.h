#include <cuda_bf16.h>
#ifdef __cplusplus
extern "C" {
#endif

void fused_add_rmsnorm_kernel_cu_bf16(
    __nv_bfloat16* norm_output,
    __nv_bfloat16* residual,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int rows, int dim, float eps,
    CUstream_st* stream
);

#ifdef __cplusplus
}
#endif
