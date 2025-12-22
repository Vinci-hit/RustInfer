#include <cuda_bf16.h>
#ifdef __cplusplus
extern "C" {
#endif

void rmsnorm_kernel_cu_dim(float*, float*, float*, int, int, float, CUstream_st*);
void rmsnorm_kernel_cu_bf16x8(__nv_bfloat16* , __nv_bfloat16* , __nv_bfloat16* , int , int , float , CUstream_st*);
#ifdef __cplusplus
}
#endif