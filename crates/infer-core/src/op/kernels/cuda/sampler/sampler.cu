// In your .cu file for CUDA kernels

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/system/cuda/execution_policy.h>

// ------------------- F32 版本 -------------------
extern "C" cudaError_t argmax_cu_f32_ffi(
    const float* logits_ptr,
    int vocab_size,
    int* result_ptr_gpu, // << 接收 GPU 指针
    cudaStream_t stream
) {
    // 使用 stream 创建 Thrust 执行策略
    auto policy = thrust::cuda::par.on(stream);

    // 将裸指针包装成 Thrust 的 device_ptr
    thrust::device_ptr<const float> d_logits(logits_ptr);
    thrust::device_ptr<int> d_result(result_ptr_gpu);

    // 使用 Thrust 找到最大元素的迭代器
    auto max_elem_it = thrust::max_element(policy, d_logits, d_logits + vocab_size);
    
    // 计算索引并将其写入到 GPU 上的结果指针
    *d_result = thrust::distance(d_logits, max_elem_it);
    
    // 返回最后一个 CUDA 错误（如果有的话）
    return cudaGetLastError();
}


// ------------------- BF16 版本 -------------------
extern "C" cudaError_t argmax_cu_bf16_ffi(
    const __nv_bfloat16* logits_ptr, // << 注意类型是 __nv_bfloat16
    int vocab_size,
    int* result_ptr_gpu, // << 接收 GPU 指针
    cudaStream_t stream
) {
    // 使用 stream 创建 Thrust 执行策略
    auto policy = thrust::cuda::par.on(stream);

    // 将裸指针包装成 Thrust 的 device_ptr
    thrust::device_ptr<const __nv_bfloat16> d_logits(logits_ptr);
    thrust::device_ptr<int> d_result(result_ptr_gpu);

    // 使用 Thrust 找到最大元素的迭代器
    auto max_elem_it = thrust::max_element(policy, d_logits, d_logits + vocab_size);
    
    // 计算索引并将其写入到 GPU 上的结果指针
    *d_result = thrust::distance(d_logits, max_elem_it);
    
    // 返回最后一个 CUDA 错误（如果有的话）
    return cudaGetLastError();
}