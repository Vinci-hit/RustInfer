// One reusable row workspace. Sorting and CDF construction remain on the GPU;
// only the chosen token and its log-probability leave the device.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstddef>

static size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }

extern "C" int filtered_workspace_bytes(int n, size_t* bytes) {
    if (n <= 0) return int(cudaErrorInvalidValue);
    size_t sort_bytes = 0, scan_bytes = 0;
    auto status = cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_bytes, (float*)nullptr, (float*)nullptr,
        (int*)nullptr, (int*)nullptr, n);
    if (status != cudaSuccess) return int(status);
    status = cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
        (float*)nullptr, (float*)nullptr, n);
    if (status != cudaSuccess) return int(status);
    *bytes = 6 * aligned(size_t(n) * sizeof(float)) + std::max(sort_bytes, scan_bytes);
    return 0;
}

template<class T>
__global__ void prepare(const T* logits, float* scores, int* ids, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = float(logits[i]);
        // Stable radix sort preserves ascending token IDs on equal logits.
        scores[i] = isnan(x) ? -INFINITY : (x == 0.f ? 0.f : x);
        ids[i] = i;
    }
}

__global__ void weights(const float* scores, float* probs, int n,
                        int k, float temperature, float min_p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float max = scores[0];
    float p = max == INFINITY ? (scores[i] == INFINITY ? 1.f : 0.f)
            : max == -INFINITY ? 1.f
            : expf((scores[i] - max) / temperature);
    probs[i] = i < k && p >= min_p ? p : 0.f;
}

__global__ void choose(const int* ids, const float* probs, const float* cdf,
                       int n, float top_p, double draw, bool greedy,
                       int* out, float* logprob) {
    if (threadIdx.x || blockIdx.x) return;
    // Retain the smallest prefix whose mass reaches top_p, including the
    // boundary token. Then draw from the renormalized retained distribution.
    int lo = 0, hi = n - 1;
    double threshold = double(cdf[n - 1]) * double(top_p);
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (double(cdf[mid]) >= threshold) hi = mid;
        else lo = mid + 1;
    }
    int last = greedy ? 0 : lo;
    double target = draw * double(cdf[last]);
    lo = 0; hi = last;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (double(cdf[mid]) > target) hi = mid;
        else lo = mid + 1;
    }
    *out = ids[lo];
    *logprob = greedy ? 0.f : logf(probs[lo] / cdf[last]);
}

extern "C" int filtered_sample(const void* logits, int dtype, int n,
    float temperature, int k, float top_p, float min_p, double draw,
    int* out, float* logprob, void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
    size_t required = 0;
    int query = filtered_workspace_bytes(n, &required);
    if (query != 0) return query;
    if (workspace_bytes < required) return int(cudaErrorInvalidValue);
    size_t pitch = aligned(size_t(n) * sizeof(float));
    char* base = static_cast<char*>(workspace);
    auto scores = reinterpret_cast<float*>(base);
    auto sorted = reinterpret_cast<float*>(base + pitch);
    auto ids = reinterpret_cast<int*>(base + 2 * pitch);
    auto sorted_ids = reinterpret_cast<int*>(base + 3 * pitch);
    auto probs = reinterpret_cast<float*>(base + 4 * pitch);
    auto cdf = reinterpret_cast<float*>(base + 5 * pitch);
    void* temp = base + 6 * pitch;
    size_t capacity = workspace_bytes - 6 * pitch;
    if (dtype == 0) prepare<<<(n+255)/256,256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(logits), scores, ids, n);
    else prepare<<<(n+255)/256,256,0,stream>>>(
        static_cast<const float*>(logits), scores, ids, n);
    size_t bytes = capacity;
    auto status = cub::DeviceRadixSort::SortPairsDescending(
        temp, bytes, scores, sorted, ids, sorted_ids, n, 0, 32, stream);
    if (status != cudaSuccess) return int(status);
    bool greedy = temperature <= 0.f || k == 1 || top_p <= 0.f;
    weights<<<(n+255)/256,256,0,stream>>>(sorted, probs, n,
        greedy ? 1 : (k > 0 ? k : n), greedy ? 1.f : temperature, min_p);
    bytes = capacity;
    status = cub::DeviceScan::InclusiveSum(temp, bytes, probs, cdf, n, stream);
    if (status != cudaSuccess) return int(status);
    choose<<<1,1,0,stream>>>(sorted_ids, probs, cdf, n, top_p, draw, greedy, out, logprob);
    return int(cudaGetLastError());
}

__global__ void beam_scores(const float* sorted, const int* ids, const float* cdf,
                            int n, int k, int* out, float* logprobs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    float max = sorted[0];
    float delta = max == INFINITY ? (sorted[i] == INFINITY ? 0.f : -INFINITY)
                : max == -INFINITY ? 0.f : sorted[i] - max;
    out[i] = ids[i];
    logprobs[i] = delta - logf(cdf[n - 1]);
}

extern "C" int beam_candidates(const void* logits, int dtype, int n, int k,
    int* out, float* logprobs, void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
    size_t required = 0;
    int query = filtered_workspace_bytes(n, &required);
    if (query != 0) return query;
    if (workspace_bytes < required) return int(cudaErrorInvalidValue);
    size_t pitch = aligned(size_t(n) * sizeof(float));
    char* base = static_cast<char*>(workspace);
    auto scores = reinterpret_cast<float*>(base);
    auto sorted = reinterpret_cast<float*>(base + pitch);
    auto ids = reinterpret_cast<int*>(base + 2 * pitch);
    auto sorted_ids = reinterpret_cast<int*>(base + 3 * pitch);
    auto probs = reinterpret_cast<float*>(base + 4 * pitch);
    auto cdf = reinterpret_cast<float*>(base + 5 * pitch);
    void* temp = base + 6 * pitch;
    size_t capacity = workspace_bytes - 6 * pitch;
    if (dtype == 0) prepare<<<(n+255)/256,256,0,stream>>>(
        static_cast<const __nv_bfloat16*>(logits), scores, ids, n);
    else prepare<<<(n+255)/256,256,0,stream>>>(
        static_cast<const float*>(logits), scores, ids, n);
    size_t bytes = capacity;
    auto status = cub::DeviceRadixSort::SortPairsDescending(
        temp, bytes, scores, sorted, ids, sorted_ids, n, 0, 32, stream);
    if (status != cudaSuccess) return int(status);
    weights<<<(n+255)/256,256,0,stream>>>(sorted, probs, n, n, 1.f, 0.f);
    bytes = capacity;
    status = cub::DeviceScan::InclusiveSum(temp, bytes, probs, cdf, n, stream);
    if (status != cudaSuccess) return int(status);
    beam_scores<<<(k+255)/256,256,0,stream>>>(sorted, sorted_ids, cdf, n, k, out, logprobs);
    return int(cudaGetLastError());
}
