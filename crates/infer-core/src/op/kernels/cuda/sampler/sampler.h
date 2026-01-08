#pragma once // 防止头文件被重复包含

#include <cuda_runtime.h>
#include <cuda_bf16.h> // 需要此头文件来定义 __nv_bfloat16 类型

// 整个头文件内容都包裹在 extern "C" 中，
// 确保 C++ 编译器以 C 风格导出这些函数符号。
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在 GPU 上计算一个 F32 张量的 argmax 索引。
 *
 * @param logits_ptr      指向输入 logits 张量（F32 类型）的设备指针。
 * @param vocab_size      logits 张量中的元素数量。
 * @param result_ptr_gpu  指向用于存储结果（一个 int 类型）的设备内存的指针。
 * @param stream          执行此操作的 CUDA stream。
 * @return cudaError_t    返回最后一个 CUDA API 调用的错误码。
 */
void argmax_cu_f32_ffi(
    const float* logits_ptr,
    int vocab_size,
    int* result_ptr_gpu,
    cudaStream_t stream
);

/**
 * @brief 在 GPU 上计算一个 BF16 张量的 argmax 索引。
 *
 * @param logits_ptr      指向输入 logits 张量（__nv_bfloat16 类型）的设备指针。
 * @param vocab_size      logits 张量中的元素数量。
 * @param result_ptr_gpu  指向用于存储结果（一个 int 类型）的设备内存的指针。
 * @param stream          执行此操作的 CUDA stream。
 * @return cudaError_t    返回最后一个 CUDA API 调用的错误码。
 */
void argmax_cu_bf16_ffi(
    const __nv_bfloat16* logits_ptr,
    int vocab_size,
    int* result_ptr_gpu,
    cudaStream_t stream
);

#ifdef __cplusplus
} // extern "C"
#endif