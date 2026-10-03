#pragma once
// Optional AOT CuTe DSL backend. Modules live for the CUDA context lifetime;
// initialization runs before graph capture, and launches do no allocations.
#include <cuda.h>
#include <cstdlib>
#include <map>
#include <mutex>
#include "cute_attention_sm89.h"

namespace cute_attention_sm89 {
struct Entry { CUmodule module; CUfunction function; };
static std::mutex mutex;
static std::map<CUcontext, Entry> entries;
static thread_local CUcontext last_context = nullptr;
static thread_local CUfunction last_function = nullptr;

static bool disabled() {
    static const bool value = std::getenv("RUSTINFER_DISABLE_CUTE_ATTENTION") != nullptr;
    return value;
}
static int check(CUresult rc) {
    if (rc == CUDA_SUCCESS) return 0;
    const char* text = nullptr;
    cuGetErrorString(rc, &text);
    fprintf(stderr, "[cute_attention_sm89] CUDA driver error %d: %s\n", int(rc), text ? text : "unknown");
    return int(cudaErrorUnknown);
}
static int initialize() {
    if (disabled()) return 0;
    CUcontext context = nullptr;
    if (int rc = check(cuCtxGetCurrent(&context))) return rc;
    if (!context) return int(cudaErrorInvalidDevice);
    if (context == last_context && last_function) return 0;
    std::lock_guard<std::mutex> lock(mutex);
    auto found = entries.find(context);
    if (found == entries.end()) {
        Entry entry{};
        if (int rc = check(cuModuleLoadData(&entry.module, cute_attention_image))) return rc;
        CUresult rc = cuModuleGetFunction(&entry.function, entry.module, cute_attention_name);
        if (rc == CUDA_SUCCESS)
            rc = cuFuncSetAttribute(entry.function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 49152);
        if (rc != CUDA_SUCCESS) { cuModuleUnload(entry.module); return check(rc); }
        found = entries.emplace(context, entry).first;
        fprintf(stderr, "[cute_attention_sm89] v13 Q64/KV64 enabled; scheduler Q128 split into 2 CTAs, BF16 H32/KV8/D128 page1\n");
    }
    last_context = context;
    last_function = found->second.function;
    return 0;
}
static int launch(
    const __nv_bfloat16* q, int64_t qss, int64_t qsh,
    const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* o, int64_t oss, int64_t osh,
    const uint32_t* pages, int max_pages, int block_size,
    const int32_t* lens, const int32_t* cuq,
    const int32_t* req, const int32_t* tile, const int32_t* active,
    int tiles, int batch, int tokens, int heads, int kv_heads, int dim,
    float scale, int causal, cudaStream_t stream) {
    if (tiles <= 0 || tokens <= 0 || batch <= 0) return 0;
    if (int rc = initialize()) return rc;
    uint8_t causal_byte = causal != 0;
    void* args[] = {&q,&k,&v,&o,&pages,&lens,&cuq,&req,&tile,&active,
        &qss,&qsh,&oss,&osh,&max_pages,&block_size,&tiles,&batch,&tokens,
        &heads,&kv_heads,&dim,&scale,&causal_byte,&stream};
    return check(cuLaunchKernel(last_function, tiles*2,32,1,128,1,1,49152,
        reinterpret_cast<CUstream>(stream), args, nullptr));
}
} // namespace cute_attention_sm89
