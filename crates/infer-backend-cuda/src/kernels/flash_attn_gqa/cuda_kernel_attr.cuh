#pragma once

#include <cuda_runtime.h>

#include <mutex>
#include <unordered_set>

namespace rustinfer::cuda {

// cudaFuncSetAttribute configures the kernel in the current CUDA context.  A
// process-wide once_flag is therefore incorrect when one process drives more
// than one GPU: only the first device gets configured.  Keep one instance of
// this guard per kernel instantiation and remember every configured device.
class PerDeviceKernelAttribute {
public:
    cudaError_t set_max_dynamic_shared_memory(const void* kernel, int bytes) {
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) return err;

        std::lock_guard<std::mutex> lock(mutex_);
        if (configured_devices_.find(device) != configured_devices_.end()) {
            return cudaSuccess;
        }

        err = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (err == cudaSuccess) configured_devices_.insert(device);
        return err;
    }

private:
    std::mutex mutex_;
    std::unordered_set<int> configured_devices_;
};

}  // namespace rustinfer::cuda
