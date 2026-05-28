// RMSNorm CUDA kernel C ABI（通用 strided 版本）。
//
// 视图模型：input/output 抽象为 `[outer0, outer1, dim]` 三维 strided 视图，
// 最后一维 dense（element stride=1），前两维 stride 由调用方提供（element 单位）。
// 启动 grid.x = outer0 * outer1。
//
// 用法约定（dispatch 层负责正确填）：
//   * dense 多维 `[..., dim]`：先 flatten 为 `[N, dim]`，再喂 outer0=N, outer1=1,
//     stride0=dim, stride1=0。
//   * strided 2-D 视图（行 stride > dim）：outer0=rows, outer1=1, stride0=row_stride,
//     stride1=0。
//   * 按 head 切的 strided 视图（如 `qkv.narrow(1, 0, q_dim)` reshape 为
//     `[T, head_num, head_dim]`）：outer0=T, outer1=head_num, stride0=cols,
//     stride1=head_dim。
//
// 对齐：dim、各 stride 都必须是 8 的倍数（half 路径）/ 4 的倍数（f32 路径）。
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifdef __cplusplus
extern "C" {
#endif

void rmsnorm_kernel_cu_dim(
    float* output, float* input, float* weight,
    int outer0, int outer1, int dim,
    long long in_stride0, long long in_stride1,
    long long out_stride0, long long out_stride1,
    float eps, CUstream_st* stream);

void rmsnorm_kernel_cu_bf16x8(
    __nv_bfloat16* output, __nv_bfloat16* input, __nv_bfloat16* weight,
    int outer0, int outer1, int dim,
    long long in_stride0, long long in_stride1,
    long long out_stride0, long long out_stride1,
    float eps, CUstream_st* stream);

void rmsnorm_kernel_cu_fp16x8(
    __half* output, __half* input, __half* weight,
    int outer0, int outer1, int dim,
    long long in_stride0, long long in_stride1,
    long long out_stride0, long long out_stride1,
    float eps, CUstream_st* stream);

#ifdef __cplusplus
}
#endif
