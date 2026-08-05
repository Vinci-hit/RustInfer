#pragma once

#include <cuda_runtime.h>

namespace rustinfer::cuda {

// CUDA only requires an opt-in attribute above the default 48 KiB dynamic
// shared-memory limit. A kernel above the device limit is left unavailable,
// rather than failing initialization for models that never instantiate it.
// Setting the same attribute value repeatedly is idempotent, so callers need
// no process-global state or lock. Call this from a registered device
// initializer, never from a forward/launch path.
inline cudaError_t configure_dynamic_shared_memory(
    const void* kernel,
    int bytes,
    int max_dynamic_smem)
{
    constexpr int kDefaultDynamicSmemLimit = 48 * 1024;
    if (bytes <= kDefaultDynamicSmemLimit || bytes > max_dynamic_smem) {
        return cudaSuccess;
    }
    return cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
}

}  // namespace rustinfer::cuda
