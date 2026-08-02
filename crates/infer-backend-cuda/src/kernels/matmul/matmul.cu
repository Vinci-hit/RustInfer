#include <cub/block/block_reduce.cuh>
#include "matmul.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <chrono>
#include <atomic>
#define CHECK_CUBLAS(func) { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS API failed at line %d with error: %d\n", \
               __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
}
// Forward declaration — defined later in this TU. Lazily creates a legacy
// cuBLAS handle used by the graph-capturable bf16 matmul path.
static cublasHandle_t get_cublas_handle();

template <int THREAD_PER_BLOCK>
__global__ void sgemv_kernel_cu_fp32x4(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K
) {
  __shared__ float sdata[THREAD_PER_BLOCK];
  const int tid = threadIdx.x;
  const int start_row = blockIdx.x;
  if (start_row >= K){
    return;
  }
  constexpr int pack_size = 4;
  const int pack_num = M / pack_size;
  const int pack_off = pack_num * pack_size;

  auto input_float4_ptr = reinterpret_cast<const float4 *>(input);
  auto weight_float4_ptr = reinterpret_cast<const float4 *>(weight + start_row * M);
  sdata[tid] = 0;
#pragma unroll
  for(int i = tid;i<pack_num;i+= blockDim.x){
    float4 input_float4 = *(input_float4_ptr + i);
    float4 weight_float4 = *(weight_float4_ptr + i);
    float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                   input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
    sdata[tid] += part_sum;
  }

  for(int i = pack_off + tid;i<M;i += blockDim.x){
    sdata[tid] += input[i] * weight[start_row * M + i];
  }
  __syncthreads();

  using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;
  float part_sum = BlockReduce(temp).Sum(sdata[tid]);
  __syncthreads();

  if (tid == 0) {
    output[start_row] = part_sum;
  }
  __syncthreads();

}

// input是一列M，weight是KxM，也就是其实是weight @ input
void sgemv_cu_fp32x4(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int K,
    cudaStream_t stream
) {
    constexpr int thread_per_block = 128;
    sgemv_kernel_cu_fp32x4<thread_per_block><<<K, thread_per_block, 0, stream>>>(input,weight,output,M,K);
}

__global__ void sgemm_naive_f32_transpose_b_kernel(
    const float *a,
    const float *b,
    float *c,
    int M, // A 的行数
    int N, // B 的行数 (也是 B^T 的列数)
    int K  // A 的列数 (也是 B 的列数)
) {
    int n_out = blockIdx.x * blockDim.x + threadIdx.x; // C 的列索引
    int m_out = blockIdx.y * blockDim.y + threadIdx.y; // C 的行索引

    // 边界检查，C 的形状是 [M, N]
    if (m_out < M && n_out < N) {
        float psum = 0.0;

        // 循环点积的长度是 K
        for (int k = 0; k < K; k++) {
            // 从 A 中获取第 m_out 行, 第 k 列的元素
            float a_val = a[m_out * K + k];

            // **核心修改**:
            // 从 B 中获取第 n_out 行, 第 k 列的元素。
            // 这等价于从 B^T 中获取第 k 行, 第 n_out 列的元素。
            float b_val = b[n_out * K + k];

            psum += a_val * b_val;
        }

        // 将结果写入 C 的 [m_out, n_out] 位置
        c[m_out * N + n_out] = psum;
    }
}

extern "C" void sgemm_naive_f32_cu(
    const float* a,
    const float* b,
    float* c,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    // 定义 block 的大小
    dim3 threads_per_block(16, 16);

    // 计算 grid 的大小
    dim3 blocks_per_grid(
        (N + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );

    sgemm_naive_f32_transpose_b_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        a, b, c, M, N, K
    );
}
// ============================================================================
// gemm_cublasLt_AxBT_RowMajor_bf16
//
// Row-major `C = A @ B^T` where A is [M, K], B is [N, K], C is [M, N]
// with row stride ldc (ldc >= N).
//
// Per (M, N, K, ldc) we cache:
//   - operationDesc / Adesc / Bdesc / Cdesc       (layout, stateless)
//   - selected cublasLtMatmulAlgo_t               (stateless handle)
//
// Selection strategy (first time a shape is seen):
//   1. Ask the heuristic for up to 32 candidates.
//   2. Benchmark the candidates whose workspace fits our budget, on a
//      private cuBLASLt handle + stream so nothing can pollute the
//      caller's stream (including an active CUDA-Graph capture).
//   3. Keep the fastest.
//
// NB: we never destroy the cached descriptors — the process keeps them
// for its whole lifetime (~10 shapes × ~100 bytes each).
// ============================================================================

#include <mutex>
#include <unordered_map>
#include <vector>

struct ZimageBf16GemmKey {
    int M, N, K, ldc;
    bool operator==(const ZimageBf16GemmKey& o) const noexcept {
        return M == o.M && N == o.N && K == o.K && ldc == o.ldc;
    }
};
struct ZimageBf16GemmKeyHash {
    size_t operator()(const ZimageBf16GemmKey& k) const noexcept {
        return (size_t(k.M) * 1315423911u) ^ (size_t(k.N) * 2654435761u)
            ^ (size_t(k.K) * 40503u) ^ (size_t(k.ldc) * 2246822519u);
    }
};
struct ZimageBf16GemmEntry {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t A = nullptr;
    cublasLtMatrixLayout_t B = nullptr;
    cublasLtMatrixLayout_t C = nullptr;
    cublasLtMatmulAlgo_t algo;
    size_t algo_ws = 0;
    bool valid = false;
};

static std::mutex g_zimage_bf16_gemm_cache_mu;
static std::unordered_map<ZimageBf16GemmKey, ZimageBf16GemmEntry, ZimageBf16GemmKeyHash>
    g_zimage_bf16_gemm_cache;

// When set (by the Rust prefill path via `zimage_set_eager_prefill_gemm`), eager
// bf16 GEMMs take the build-free `gemm_bf16_chunked_legacy` (cublasGemmEx) path
// instead of the per-shape cuBLASLt heuristic+capturability-probe cache build.
// Prefill M = num_tokens varies per request, so every distinct prompt length is
// a cold shape that otherwise pays ~9-18ms of build (heuristic + graph-capture
// probe) on the TTFT critical path — pure overhead, since eager prefill is never
// captured into a graph and so needs neither a benchmarked nor a capturable algo.
// Decode (graph capture + its eager warmup) leaves this 0 and keeps the cuBLASLt
// cached path, where the per-(capture_size) build amortizes across many replays.
static std::atomic<int> g_zimage_eager_prefill_gemm{0};
extern "C" void zimage_set_eager_prefill_gemm(int on)
{
    g_zimage_eager_prefill_gemm.store(on, std::memory_order_relaxed);
}

// Eager-prefill algo cache keyed by (N,K,ldc) — NOT exact M. The cuBLASLt heuristic's
// algo pick is robust across M for a fixed (N,K), so we run the heuristic once
// per (N,K,ldc) and reuse the (M-independent) op descriptor + algo for every prompt
// length. The cheap M-dependent matrix layouts are recreated per call. This
// gives prefill the single-pass cuBLASLt kernel (no chunked K-tile launch storm)
// with zero per-length build cost — unlike the (M,N,K)-keyed decode cache, which
// rebuilds (heuristic + capture probe) for every distinct M.
// The cuBLASLt heuristic's best algo depends on the M *regime* (small-M memory-
// bound tilings differ from large-M compute-bound ones), so a single algo cached
// per (N,K) and reused across all M mis-tiles long prefills (~2x slower). Bucket
// M into coarse regimes and cache one algo per (N,K,bucket): a new prompt length
// lands in an existing bucket → cache hit with a regime-appropriate algo, and the
// total build count stays tiny (~5 (N,K) × ~6 buckets).
static inline int zimage_m_bucket(int M)
{
    if (M <= 16) return 16;
    if (M <= 64) return 64;
    if (M <= 256) return 256;
    if (M <= 1024) return 1024;
    return ((M + 1023) / 1024) * 1024;
}
struct ZimageNkKey {
    int N, K, Mb, ldc;
    bool operator==(const ZimageNkKey& o) const {
        return N == o.N && K == o.K && Mb == o.Mb && ldc == o.ldc;
    }
};
struct ZimageNkKeyHash {
    size_t operator()(const ZimageNkKey& k) const {
        size_t h = std::hash<long long>()(((long long)k.N << 32) ^ (unsigned)k.K);
        return h * 1000003u ^ std::hash<int>()(k.Mb)
            ^ (std::hash<int>()(k.ldc) * 2246822519u);
    }
};
struct ZimageNkEntry {
    cublasLtMatmulDesc_t op = nullptr; // M-independent (transA/B + compute type)
    cublasLtMatmulAlgo_t algo;
    size_t algo_ws = 0;
    bool valid = false;
};
static std::mutex g_zimage_nk_mu;
static std::unordered_map<ZimageNkKey, ZimageNkEntry, ZimageNkKeyHash> g_zimage_nk_cache;

// Forward decl — defined later in this TU (the build-free chunked fallback).
static void gemm_bf16_chunked_legacy(
    int M, int N, int K, int ldc,
    const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, __nv_bfloat16* d_C,
    void* workspace, size_t workspaceSize, cudaStream_t stream);

// Direct cuBLASLt call with algo=nullptr: cuBLASLt's internal runtime picks the
// kernel for this exact (M,N,K). This is the pre-refactor (532f5c) path and is
// empirically faster than the heuristic-top-1 / benchmarked / capturable-cached
// algos for small-M prefill GEMMs (~2.7ms/forward on Qwen3-4B). NOT graph-
// capturable (runtime selection does private-stream probing under capture), so
// it is used ONLY on the non-capturing eager-prefill path; decode graphs keep
// the capturable cached algo. Recreates the (cheap, ~µs) descriptors per call.
static void gemm_bf16_nullptr_runtime(
    cublasLtHandle_t ltHandle, int M, int N, int K, int ldc,
    const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, __nv_bfloat16* d_C,
    void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // Row-major C[M,N]=A[M,K]·B[N,K]^T → col-major [N,M]=[N,K]·[K,M] with
    // transA=T, transB=N and swapped operands (mirrors the cached paths).
    const int m_g = N, n_g = M, k_g = K;
    cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, k_g, m_g, k_g);
    cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, k_g, n_g, k_g);
    cublasLtMatrixLayoutCreate(&C, CUDA_R_16BF, m_g, n_g, ldc);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasLtMatmul(
        ltHandle, op, &alpha,
        d_B, A, d_A, B, &beta,
        d_C, C, d_C, C,
        nullptr, workspace, workspaceSize, stream);
    cublasLtMatrixLayoutDestroy(C);
    cublasLtMatrixLayoutDestroy(B);
    cublasLtMatrixLayoutDestroy(A);
    cublasLtMatmulDescDestroy(op);
    if (status != CUBLAS_STATUS_SUCCESS) {
        gemm_bf16_chunked_legacy(
            M, N, K, ldc, d_A, d_B, d_C, workspace, workspaceSize, stream);
    }
}

// Eager bf16 GEMM via the (N,K)-cached single-pass cuBLASLt algo. Falls back to
// the chunked cublasGemmEx path on any cuBLASLt error (cold-build failure or an
// algo the cached pick can't service at this M).
[[maybe_unused]] static void gemm_bf16_eager_nk_cached(
    cublasLtHandle_t ltHandle, int M, int N, int K, int ldc,
    const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, __nv_bfloat16* d_C,
    void* workspace, size_t workspaceSize, cudaStream_t stream)
{
    // Row-major C[M,N]=A[M,K]·B[N,K]^T  ->  col-major [N,M]=[N,K]·[K,M] with
    // transA=T, transB=N and swapped operands (mirrors the decode-cache build).
    const int m_g = N, n_g = M, k_g = K;
    const ZimageNkKey nk_key{N, K, zimage_m_bucket(M), ldc};

    ZimageNkEntry entry;
    {
        std::lock_guard<std::mutex> lk(g_zimage_nk_mu);
        auto it = g_zimage_nk_cache.find(nk_key);
        if (it != g_zimage_nk_cache.end() && it->second.valid) {
            entry = it->second;
        } else {
            cublasOperation_t opA = CUBLAS_OP_T, opB = CUBLAS_OP_N;
            cublasLtMatmulDesc_t op = nullptr;
            cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr;
            cublasLtMatmulPreference_t pref = nullptr;
            cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
            cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, k_g, m_g, k_g);
            cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, k_g, n_g, k_g);
            cublasLtMatrixLayoutCreate(&C, CUDA_R_16BF, m_g, n_g, ldc);
            cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
            cublasLtMatmulHeuristicResult_t res{};
            int ret = 0;
            cublasLtHandle_t tmp_h = nullptr;
            cublasLtCreate(&tmp_h);
            cublasLtMatmulAlgoGetHeuristic(tmp_h, op, A, B, C, C, pref, 1, &res, &ret);
            cublasLtDestroy(tmp_h);
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(A);
            cublasLtMatrixLayoutDestroy(B);
            cublasLtMatrixLayoutDestroy(C);
            if (ret == 0) {
                cublasLtMatmulDescDestroy(op);
                gemm_bf16_chunked_legacy(
                    M, N, K, ldc, d_A, d_B, d_C, workspace, workspaceSize, stream);
                return;
            }
            entry.op = op;
            entry.algo = res.algo;
            entry.algo_ws = res.workspaceSize;
            entry.valid = true;
            g_zimage_nk_cache.emplace(nk_key, entry);
        }
    }

    // M-dependent layouts are cheap (~µs); recreate per call and run single-pass.
    cublasLtMatrixLayout_t A = nullptr, B = nullptr, C = nullptr;
    cublasLtMatrixLayoutCreate(&A, CUDA_R_16BF, k_g, m_g, k_g);
    cublasLtMatrixLayoutCreate(&B, CUDA_R_16BF, k_g, n_g, k_g);
    cublasLtMatrixLayoutCreate(&C, CUDA_R_16BF, m_g, n_g, ldc);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasLtMatmul(
        ltHandle, entry.op, &alpha,
        d_B, A, d_A, B, &beta,
        d_C, C, d_C, C,
        &entry.algo, workspace, workspaceSize, stream);
    cublasLtMatrixLayoutDestroy(A);
    cublasLtMatrixLayoutDestroy(B);
    cublasLtMatrixLayoutDestroy(C);
    if (st != CUBLAS_STATUS_SUCCESS) {
        gemm_bf16_chunked_legacy(
            M, N, K, ldc, d_A, d_B, d_C, workspace, workspaceSize, stream);
    }
}

// Per-shape cuBLASLt algo benchmarking is OFF by default. The cuBLASLt
// heuristic's top capturable algo matches the benchmarked pick for the decode
// shapes (measured: identical TPOT, e.g. b32 4.00ms vs 4.10ms, b64 4.87ms vs
// 4.87ms on Qwen3-4B) — both select the same large-tile single-pass kernel, so
// the benchmark added no decode speedup. But the benchmark's per-shape
// cudaDeviceSynchronize + timed sweep over ~32 algos cost ~30ms PER cold shape,
// and prefill M = num_tokens varies per request, so almost every prefill paid
// it on the TTFT critical path (TTFT 128tok 80ms -> 48.8ms with bench off).
// Opt back in with RUSTINFER_ENABLE_CUBLASLT_BF16_BENCH=1 for workloads where
// the heuristic's top algo is suboptimal (e.g. z-image DiT GEMM shapes). The
// legacy RUSTINFER_DISABLE_CUBLASLT_BF16_BENCH still force-disables.
static bool zimage_bf16_gemm_bench_disabled()
{
    static int cached = -1;
    if (cached == -1) {
        const char* en = getenv("RUSTINFER_ENABLE_CUBLASLT_BF16_BENCH");
        const char* dis = getenv("RUSTINFER_DISABLE_CUBLASLT_BF16_BENCH");
        bool force_dis = (dis != nullptr && dis[0] != '\0' && dis[0] != '0');
        bool opt_in = (en != nullptr && en[0] != '\0' && en[0] != '0');
        cached = (opt_in && !force_dis) ? 0 : 1; // disabled unless explicitly enabled
    }
    return cached == 1;
}

// Returns true if `algo` can be recorded into a CUDA graph via stream capture.
// Some cuBLASLt algorithms (notably workspace split-K reductions chosen for
// large-K shapes) issue calls that are illegal under capture and fail with
// status=13; such algos must NOT be cached for the graph decode path. We test
// by capturing a single matmul (capture records, does not execute, so the
// bench buffers are not touched here) and checking a valid graph comes back.
static bool zimage_algo_is_capturable(
    cublasLtHandle_t h,
    cublasLtMatmulDesc_t op,
    cublasLtMatrixLayout_t A,
    cublasLtMatrixLayout_t B,
    cublasLtMatrixLayout_t C,
    const cublasLtMatmulAlgo_t* algo,
    const __nv_bfloat16* bench_A,
    const __nv_bfloat16* bench_B,
    __nv_bfloat16* bench_C,
    void* ws,
    size_t wsSize)
{
    cudaStream_t ts = nullptr;
    if (cudaStreamCreate(&ts) != cudaSuccess) return false;
    float alpha = 1.0f, beta = 0.0f;
    bool capturable = false;
    // cudaStreamCaptureModeRelaxed (==2) matches the runtime decode capture.
    if (cudaStreamBeginCapture(ts, cudaStreamCaptureModeRelaxed) == cudaSuccess) {
        cublasStatus_t s = cublasLtMatmul(
            h, op, &alpha,
            bench_B, A, bench_A, B, &beta,
            bench_C, C, bench_C, C,
            algo, ws, wsSize, ts);
        cudaGraph_t g = nullptr;
        cudaError_t ce = cudaStreamEndCapture(ts, &g);
        capturable = (s == CUBLAS_STATUS_SUCCESS && ce == cudaSuccess && g != nullptr);
        if (g) cudaGraphDestroy(g);
    }
    // Clear any sticky error left by an invalidated capture attempt.
    cudaGetLastError();
    cudaStreamDestroy(ts);
    return capturable;
}

static ZimageBf16GemmEntry zimage_build_bf16_gemm_entry(
    int M, int N, int K, int ldc, size_t workspaceSize,
    // scratch buffers from the caller — we reuse these for the benchmark
    // to avoid a costly cudaMalloc of up to ~300 MiB on large FFN shapes
    // (which can race with the text encoder's allocations).
    const __nv_bfloat16* bench_A,
    const __nv_bfloat16* bench_B,
    __nv_bfloat16*       bench_C,
    void*                bench_ws)
{
    // Interpretation trick: we want row-major C[M,N] = A[M,K] @ B[N,K]^T.
    // cuBLASLt is column-major, so we ask for col-major [N,M] = [N,K]*[K,M],
    // which corresponds to transA=T, transB=N with swapped inputs.
    const int m_g = N, n_g = M, k_g = K;

    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;

    ZimageBf16GemmEntry e;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&e.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&e.A, CUDA_R_16BF, k_g, m_g, k_g));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&e.B, CUDA_R_16BF, k_g, n_g, k_g));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&e.C, CUDA_R_16BF, m_g, n_g, ldc));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

    constexpr int kRequested = 32;
    std::vector<cublasLtMatmulHeuristicResult_t> hres(kRequested);
    int returned = 0;

    const bool _gbt = (getenv("RUSTINFER_GEMM_BUILD_TRACE") != nullptr);
    auto _gbt_now = []() { return std::chrono::steady_clock::now(); };
    auto _gbt_ms = [](std::chrono::steady_clock::time_point a,
                      std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto _t_start = _gbt_now();

    // Use a private temporary cuBLASLt handle to run the heuristic. This
    // avoids touching the caller's handle (which may be mid-graph-capture).
    {
        cublasLtHandle_t tmp_h;
        cublasLtCreate(&tmp_h);
        cublasLtMatmulAlgoGetHeuristic(
            tmp_h, e.op, e.A, e.B, e.C, e.C,
            pref, kRequested, hres.data(), &returned);
        cublasLtDestroy(tmp_h);
    }
    auto _t_heur = _gbt_now();

    cublasLtMatmulPreferenceDestroy(pref);

    if (returned == 0) {
        printf("cuBLASLt BF16: no algo for M=%d N=%d K=%d\n", M, N, K);
        return e;
    }

    int best = -1;
    float best_ms = 1e30f;

    if (!zimage_bf16_gemm_bench_disabled()) {
        // Real benchmark on a private handle + stream. We reuse the
        // caller's A/B/C/workspace buffers rather than cudaMalloc'ing our
        // own ~300 MiB of scratch (which would race with the text encoder
        // allocations that happen around the first DiT layer).
        cublasLtHandle_t bench_h = nullptr;
        cudaStream_t bench_s = nullptr;
        cublasLtCreate(&bench_h);
        cudaStreamCreate(&bench_s);

        // Make sure any pending work on the default stream / caller stream
        // has landed before we start measuring.
        cudaDeviceSynchronize();

        float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        for (int i = 0; i < returned; ++i) {
            if (hres[i].state != CUBLAS_STATUS_SUCCESS) continue;
            if (hres[i].workspaceSize > workspaceSize) continue;

            // Warm up; if the algo errors out, skip it.
            cublasStatus_t s = cublasLtMatmul(
                bench_h, e.op, &alpha,
                bench_B, e.A, bench_A, e.B, &beta,
                bench_C, e.C, bench_C, e.C,
                &hres[i].algo, bench_ws, workspaceSize, bench_s);
            if (s != CUBLAS_STATUS_SUCCESS) continue;
            if (cudaStreamSynchronize(bench_s) != cudaSuccess) continue;

            // Only keep algos that are also CUDA-graph-capturable (the decode
            // path replays these inside a captured graph).
            if (!zimage_algo_is_capturable(bench_h, e.op, e.A, e.B, e.C,
                                           &hres[i].algo, bench_A, bench_B,
                                           bench_C, bench_ws, workspaceSize)) {
                continue;
            }

            bool ok = true;
            cudaEventRecord(t0, bench_s);
            for (int r = 0; r < 5; ++r) {
                if (cublasLtMatmul(
                        bench_h, e.op, &alpha,
                        bench_B, e.A, bench_A, e.B, &beta,
                        bench_C, e.C, bench_C, e.C,
                        &hres[i].algo, bench_ws, workspaceSize, bench_s) != CUBLAS_STATUS_SUCCESS) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            cudaEventRecord(t1, bench_s);
            if (cudaEventSynchronize(t1) != cudaSuccess) continue;
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, t0, t1);
            // Sanity-check: drop pathological results that indicate the
            // event clocks got corrupted by some algo state issue.
            if (ms <= 0.0f || ms > 1000.0f) continue;
            if (ms < best_ms) {
                best_ms = ms;
                best = i;
            }
        }

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaStreamDestroy(bench_s);
        cublasLtDestroy(bench_h);
    }

    auto _t_probe0 = _gbt_now();
    if (best < 0) {
        // No bench (or no benched algo captured) — take the first heuristic
        // result that is graph-capturable.
        cublasLtHandle_t test_h = nullptr;
        cublasLtCreate(&test_h);
        for (int i = 0; i < returned; ++i) {
            if (hres[i].state != CUBLAS_STATUS_SUCCESS) continue;
            if (hres[i].workspaceSize > workspaceSize) continue;
            if (zimage_algo_is_capturable(test_h, e.op, e.A, e.B, e.C,
                                          &hres[i].algo, bench_A, bench_B,
                                          bench_C, bench_ws, workspaceSize)) {
                best = i;
                break;
            }
        }
        cublasLtDestroy(test_h);
        if (best < 0) {
            // Last resort: first viable algo even if not capturable (will error
            // loudly at use under capture rather than silently misbehave).
            for (int i = 0; i < returned; ++i) {
                if (hres[i].state == CUBLAS_STATUS_SUCCESS
                    && hres[i].workspaceSize <= workspaceSize) {
                    best = i;
                    break;
                }
            }
            if (best < 0) best = 0;
        }
    }

    auto _t_probe1 = _gbt_now();
    if (_gbt) {
        fprintf(stderr, "[gemm-build] M=%d N=%d K=%d  heuristic=%.2fms  probe=%.2fms  total=%.2fms (returned=%d)\n",
               M, N, K, _gbt_ms(_t_start, _t_heur), _gbt_ms(_t_probe0, _t_probe1),
               _gbt_ms(_t_start, _t_probe1), returned);
    }

    e.algo = hres[best].algo;
    e.algo_ws = hres[best].workspaceSize;
    e.valid = true;

    if (!zimage_bf16_gemm_bench_disabled() && getenv("RUSTINFER_CUBLASLT_BF16_BENCH_VERBOSE")) {
        printf("[cublasLt-bf16-cache] M=%d N=%d K=%d picked idx=%d bench_us=%.1f ws=%zu (of %d algos)\n",
               M, N, K, best, best < 0 ? -1.0f : best_ms * 200.0f,
               e.algo_ws, returned);
    }
    return e;
}

// Fallback: legacy cuBLAS `cublasGemmEx` with K-chunked single-pass kernels.
// Used only when the benchmarked-capturable cuBLASLt algo cache has no entry
// for this shape AND we are mid graph-capture (so we cannot build one). With
// the eager warmup pass that runs before every capture this path should never
// fire in steady state; it exists purely so a cold shape seen first under
// capture still produces a correct (if slower) result instead of crashing.
//
// Row-major C[M,N] = A[M,K] @ B[N,K]^T -> column-major m=N, n=M, k=K with
// B^T (lda=K), A (ldb=K), and strided C (ldc>=N). K is tiled at K_CHUNK so cuBLAS keeps a
// single-pass (graph-capturable) kernel and accumulates chunks with beta=1.
static void gemm_bf16_chunked_legacy(
    int M, int N, int K, int ldc,
    const __nv_bfloat16 *d_A,
    const __nv_bfloat16 *d_B,
    __nv_bfloat16 *d_C,
    void *workspace, size_t workspaceSize,
    cudaStream_t stream)
{
    cublasHandle_t h = get_cublas_handle();
    if (!h) {
        printf("cuBLAS bf16 matmul: null handle (M=%d N=%d K=%d)\n", M, N, K);
        exit(EXIT_FAILURE);
    }

    cudaStreamCaptureStatus capst = cudaStreamCaptureStatusNone;
    bool capturing = (cudaStreamIsCapturing(stream, &capst) == cudaSuccess &&
                      capst != cudaStreamCaptureStatusNone);
    // cublasSetStream / cublasSetWorkspace are illegal under capture; only
    // (re)bind the handle when not capturing.
    if (!capturing) {
        cublasSetStream(h, stream);
        if (workspace != nullptr && workspaceSize > 0) {
            cublasSetWorkspace(h, workspace, workspaceSize);
        }
    }

    const int K_CHUNK = 2048;
    int kc = (K > K_CHUNK) ? K_CHUNK : K;
    float alpha = 1.0f;
    for (int k0 = 0; k0 < K; k0 += kc) {
        int this_k = (K - k0 < kc) ? (K - k0) : kc;
        float beta = (k0 == 0) ? 0.0f : 1.0f;
        cublasStatus_t status = cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, this_k, &alpha,
            d_B + k0, CUDA_R_16BF, K,
            d_A + k0, CUDA_R_16BF, K,
            &beta,
            d_C, CUDA_R_16BF, ldc,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS bf16 GemmEx fallback failed: status=%d M=%d N=%d K=%d "
                   "(chunk k0=%d kc=%d capturing=%d)\n",
                   status, M, N, K, k0, this_k, (int)capturing);
            exit(EXIT_FAILURE);
        }
    }
}

void gemm_cublasLt_AxBT_RowMajor_bf16(
    cublasLtHandle_t ltHandle,
    int M, int N, int K, int ldc,
    const __nv_bfloat16 *d_A, // shape: [M, K]
    const __nv_bfloat16 *d_B, // shape: [N, K]
    __nv_bfloat16 *d_C,       // shape: [M, N]
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream)
{
    // Fast path: a per-shape cuBLASLt algo that was (a) benchmarked as fastest
    // and (b) verified graph-capturable at warmup. This restores the large-tile,
    // single-pass cuBLASLt kernels used pre-refactor (commit 96f7b4e), which the
    // K-chunked legacy `cublasGemmEx` fallback replaced and ran ~1.6x more
    // launches / 20% slower at decode. cuBLASLt's matmul takes stream+workspace
    // as call args (no cublasSetStream/Workspace), so it is capture-safe with a
    // pre-selected algo — no handle reconfiguration inside the capture region.
    cudaStreamCaptureStatus capst = cudaStreamCaptureStatusNone;
    bool capturing = (cudaStreamIsCapturing(stream, &capst) == cudaSuccess &&
                      capst != cudaStreamCaptureStatusNone);

    // Eager prefill: single-pass cuBLASLt with an algo cached by (N,K), so each
    // distinct prompt length skips the per-(M,N,K) heuristic + capture probe
    // (~9-18ms of cold build off TTFT) while keeping the large-tile single-pass
    // kernel (no chunked K-tile launch storm).
    if (!capturing && g_zimage_eager_prefill_gemm.load(std::memory_order_relaxed) != 0) {
        // algo=nullptr runtime selection — faster than the heuristic/benchmarked
        // cached algos for small-M prefill (restores 532f5c forward speed).
        gemm_bf16_nullptr_runtime(
            ltHandle, M, N, K, ldc, d_A, d_B, d_C, workspace, workspaceSize, stream);
        return;
    }

    const ZimageBf16GemmEntry* entry = nullptr;
    ZimageBf16GemmKey key{M, N, K, ldc};
    {
        std::lock_guard<std::mutex> lk(g_zimage_bf16_gemm_cache_mu);
        auto it = g_zimage_bf16_gemm_cache.find(key);
        if (it != g_zimage_bf16_gemm_cache.end()) {
            if (it->second.valid) entry = &it->second;
        } else if (!capturing) {
            // Build (heuristic + benchmark + capturability probe) eagerly. This
            // only runs while NOT capturing — the warmup pass before each graph
            // capture exercises every decode shape, so capture always hits a
            // populated cache. Reuse the caller's d_A/d_B/d_C/workspace as bench
            // scratch (the real matmul below overwrites d_C with the answer).
            ZimageBf16GemmEntry built = zimage_build_bf16_gemm_entry(
                M, N, K, ldc, workspaceSize, d_A, d_B, d_C, workspace);
            auto ins = g_zimage_bf16_gemm_cache.emplace(key, built);
            if (ins.first->second.valid) entry = &ins.first->second;
        }
    }

    if (entry) {
        // Replay the cached capturable algo. Descriptors are handle-independent;
        // `ltHandle` is the process-wide persistent cuBLASLt handle.
        float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t status = cublasLtMatmul(
            ltHandle, entry->op, &alpha,
            d_B, entry->A, d_A, entry->B, &beta,
            d_C, entry->C, d_C, entry->C,
            &entry->algo, workspace, workspaceSize, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLASLt bf16 cached matmul failed: status=%d M=%d N=%d K=%d "
                   "(capturing=%d) — falling back to chunked legacy\n",
                   status, M, N, K, (int)capturing);
            gemm_bf16_chunked_legacy(
                M, N, K, ldc, d_A, d_B, d_C, workspace, workspaceSize, stream);
        }
        return;
    }

    // Cache miss with no chance to build (cold shape first seen under capture):
    // use the capture-safe chunked legacy path so we stay correct.
    gemm_bf16_chunked_legacy(
        M, N, K, ldc, d_A, d_B, d_C, workspace, workspaceSize, stream);
}
// ============================================================================
// BF16 GEMV kernel v3 for decode phase (M=1)
// y[n] = dot(W[n,:], x[:])  where W is [N, K], x is [1, K], y is [1, N]
//
// Design: 1 warp (32 threads) computes 1 row, 1 block has 8 warps.
// No shared memory: input vector (8KB for K=4096) fits in L1 cache (128KB)
// and is implicitly cached across warps/blocks on same SM. This avoids
// __syncthreads() overhead and saves shared memory for higher occupancy.
//
// NCU-validated improvements over v2 (N=11008, K=4096, A10 sm_86):
//   - L1/TEX hit rate: 8.3% -> 49.8% (input vector cached in L1)
//   - Achieved occupancy: 87% -> 91.6%
//   - DRAM throughput: 92.6% -> 93.9%
//   - Duration: 185us -> 181us (~2% faster)
//   - No __syncthreads() needed
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum_gemv(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
hgemv_bf16_v3_kernel(
    const __nv_bfloat16* __restrict__ input,   // [K]
    const __nv_bfloat16* __restrict__ weight,  // [N, K] row-major
    __nv_bfloat16* __restrict__ output,        // [N]
    const int N,
    const int K
) {
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);

    if (row >= N) return;

    const int pack_num = K >> 3;  // K / 8, each pack = 8 bf16 = float4
    const float4* __restrict__ input_f4 = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_f4 = reinterpret_cast<const float4*>(weight + row * K);

    float sum = 0.0f;

    for (int i = lane_id; i < pack_num; i += 32) {
        float4 x = __ldg(input_f4 + i);
        float4 w = __ldg(weight_f4 + i);

        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&x);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w);

        sum += __bfloat162float(xb[0]) * __bfloat162float(wb[0]);
        sum += __bfloat162float(xb[1]) * __bfloat162float(wb[1]);
        sum += __bfloat162float(xb[2]) * __bfloat162float(wb[2]);
        sum += __bfloat162float(xb[3]) * __bfloat162float(wb[3]);
        sum += __bfloat162float(xb[4]) * __bfloat162float(wb[4]);
        sum += __bfloat162float(xb[5]) * __bfloat162float(wb[5]);
        sum += __bfloat162float(xb[6]) * __bfloat162float(wb[6]);
        sum += __bfloat162float(xb[7]) * __bfloat162float(wb[7]);
    }

    sum = warp_reduce_sum_gemv(sum);

    if (lane_id == 0) {
        output[row] = __float2bfloat16(sum);
    }
}

extern "C" void hgemv_bf16_cu(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int N,
    int K,
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;  // 256

    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    hgemv_bf16_v3_kernel<WARPS_PER_BLOCK><<<grid, THREADS, 0, stream>>>(
        input, weight, output, N, K);
}

void gemm_cublaslt_bf16(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M,
    int N,
    int K,
    int ldc,
    cudaStream_t stream,
    cublasLtHandle_t handle,
    void* workspace, size_t workspaceSize
) {
    gemm_cublasLt_AxBT_RowMajor_bf16(
        handle, M, N, K, ldc, A, B, C, workspace, workspaceSize, stream);
}

// ============================================================================
// FP16 variants (for AWQ / float16 models)
// ============================================================================

// --- FP16 cublasLt GEMM: C = A @ B^T, row-major, all FP16 ---
void gemm_cublasLt_AxBT_RowMajor_fp16(
    cublasLtHandle_t ltHandle,
    int M, int N, int K,
    const half *d_A,
    const half *d_B,
    half *d_C,
    void *workspace,
    size_t workspaceSize,
    cudaStream_t stream)
{
    int m_gemm = N;
    int n_gemm = M;
    int k_gemm = K;

    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));

    cublasLtMatrixLayout_t Adesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, k_gemm, m_gemm, k_gemm));
    cublasLtMatrixLayout_t Bdesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, k_gemm, n_gemm, k_gemm));
    cublasLtMatrixLayout_t Cdesc = NULL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m_gemm, n_gemm, m_gemm));

    cublasLtMatmulPreference_t preference = NULL;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));
    if (returnedResults == 0) { printf("cuBLASLt FP16: No algorithm found!\n"); exit(1); }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasLtMatmul(ltHandle, operationDesc, &alpha,
                                d_B, Adesc, d_A, Bdesc, &beta,
                                d_C, Cdesc, d_C, Cdesc,
                                &heuristicResult.algo, workspace, workspaceSize, stream));

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(operationDesc);
}

// --- FP16 GEMV kernel (same structure as BF16 v3) ---
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 6)
hgemv_fp16_v3_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int N,
    const int K
) {
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;

    const int pack_num = K >> 3;
    const float4* __restrict__ input_f4 = reinterpret_cast<const float4*>(input);
    const float4* __restrict__ weight_f4 = reinterpret_cast<const float4*>(weight + row * K);

    float sum = 0.0f;
    for (int i = lane_id; i < pack_num; i += 32) {
        float4 x = __ldg(input_f4 + i);
        float4 w = __ldg(weight_f4 + i);
        const half* xh = reinterpret_cast<const half*>(&x);
        const half* wh = reinterpret_cast<const half*>(&w);
        sum += __half2float(xh[0]) * __half2float(wh[0]);
        sum += __half2float(xh[1]) * __half2float(wh[1]);
        sum += __half2float(xh[2]) * __half2float(wh[2]);
        sum += __half2float(xh[3]) * __half2float(wh[3]);
        sum += __half2float(xh[4]) * __half2float(wh[4]);
        sum += __half2float(xh[5]) * __half2float(wh[5]);
        sum += __half2float(xh[6]) * __half2float(wh[6]);
        sum += __half2float(xh[7]) * __half2float(wh[7]);
    }
    sum = warp_reduce_sum_gemv(sum);
    if (lane_id == 0) output[row] = __float2half(sum);
}

extern "C" void hgemv_fp16_cu(
    const half* input, const half* weight, half* output,
    int N, int K, cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int grid = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    hgemv_fp16_v3_kernel<WARPS_PER_BLOCK><<<grid, THREADS, 0, stream>>>(input, weight, output, N, K);
}

extern "C" void gemm_cublaslt_fp16(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    cudaStream_t stream, cublasLtHandle_t handle,
    void* workspace, size_t workspaceSize
) {
    gemm_cublasLt_AxBT_RowMajor_fp16(handle, M, N, K, A, B, C, workspace, workspaceSize, stream);
}

// ============================================================================
//  INT4 GEMV/GEMM — K-packed, BF16 magic dequant + bf16x2 FMA pipeline
//
//  Weight layout (compressed-tensors / pack-quantized format):
//    weight_packed:     [N, K/8]          (int32) — 8 consecutive K-position INT4 per int32
//    weight_zero_point: [N/8, num_groups] (int32) — zero points packed along N
//    weight_scale:      [N, num_groups]   (bf16)  — per-group scale factors
//
//  BF16 magic number dequant (zero bf16↔fp16 conversion):
//    0x4300 = bf16(128.0). OR nibble → bf16(128 + nibble).
//    bf16x2 sub(128+zp) then mul(scale) → dequant entirely in BF16 pipeline.
//    Input is already BF16, pair directly → bf16x2 FMA accumulation.
//    44 ops/word → 24 ops/word (45% reduction).
// ============================================================================

// BF16 magic dequant: extract 8 INT4 nibbles from one int32 → 4 x bf16x2.
// Output order: (n0,n4), (n1,n5), (n2,n6), (n3,n7) — interleaved pairs.
__device__ __forceinline__ void dequant_8xint4_bf16_magic(
    uint32_t word,
    __nv_bfloat162 magic_zp,  // bf162(128+zp, 128+zp)
    __nv_bfloat162 scale_bf2, // bf162(scale, scale)
    __nv_bfloat162 &out04,
    __nv_bfloat162 &out15,
    __nv_bfloat162 &out26,
    __nv_bfloat162 &out37
) {
    static constexpr uint32_t MAGIC = 0x43004300u;  // two bf16 128.0
    static constexpr uint32_t MASK  = 0x000F000Fu;

    uint32_t p04 = ((word      ) & MASK) | MAGIC;
    uint32_t p15 = ((word >>  4) & MASK) | MAGIC;
    uint32_t p26 = ((word >>  8) & MASK) | MAGIC;
    uint32_t p37 = ((word >> 12) & MASK) | MAGIC;

    out04 = __hmul2(__hsub2(*reinterpret_cast<__nv_bfloat162*>(&p04), magic_zp), scale_bf2);
    out15 = __hmul2(__hsub2(*reinterpret_cast<__nv_bfloat162*>(&p15), magic_zp), scale_bf2);
    out26 = __hmul2(__hsub2(*reinterpret_cast<__nv_bfloat162*>(&p26), magic_zp), scale_bf2);
    out37 = __hmul2(__hsub2(*reinterpret_cast<__nv_bfloat162*>(&p37), magic_zp), scale_bf2);
}

template <int WARPS_PER_BLOCK, bool GROUP_SIZE_IS_POW2>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
kpack_gemv_kernel(
    const __nv_bfloat16* __restrict__ input,        // [K]
    const int32_t* __restrict__ weight_packed,       // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,   // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,  // [N, num_groups]
    __nv_bfloat16* __restrict__ output,              // [N]
    const int N, const int K, const int group_size,
    const int group_shift
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int K_packed = K >> 3;
    const int num_groups = K / group_size;

    const int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (row >= N) return;

    const int32_t* wp_row = weight_packed + row * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + row * num_groups;

    const int zp_row_packed = row >> 3;
    const int zp_bit_offset = (row & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_row_packed * num_groups;

    const int4* input_i4 = reinterpret_cast<const int4*>(input);

    // BF16x2 accumulators — two independent chains to avoid FMA dependency stall
    __nv_bfloat162 acc_a = __float2bfloat162_rn(0.0f);
    __nv_bfloat162 acc_b = __float2bfloat162_rn(0.0f);

    int kp = lane_id;

    // Main loop with 4x unroll
    for (; kp + 3 * 32 < K_packed; kp += 4 * 32) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int kpu = kp + u * 32;
            int k_base = kpu * 8;
            int g = GROUP_SIZE_IS_POW2 ? (k_base >> group_shift) : (k_base / group_size);

            // Load scale directly as bf16 (no conversion!)
            __nv_bfloat16 scale_bf = __ldg(&sc_row[g]);
            int32_t zp_packed = __ldg(&zp_row[g]);
            int zero = (zp_packed >> zp_bit_offset) & 0xF;

            // Build magic constants in BF16
            __nv_bfloat16 magic_zp_val = __float2bfloat16(128.0f + (float)zero);
            __nv_bfloat162 scale_bf2 = __halves2bfloat162(scale_bf, scale_bf);
            __nv_bfloat162 magic_zp = __halves2bfloat162(magic_zp_val, magic_zp_val);

            int32_t word = __ldg(&wp_row[kpu]);

            // BF16 magic dequant → 4 x bf16x2
            __nv_bfloat162 d04, d15, d26, d37;
            dequant_8xint4_bf16_magic(word, magic_zp, scale_bf2, d04, d15, d26, d37);

            // Load input as int4 (8 bf16 values) — already bf16, zero conversion!
            int4 in = __ldg(&input_i4[kpu]);
            const __nv_bfloat16* inp = reinterpret_cast<const __nv_bfloat16*>(&in);

            // Pair input to match dequant interleave: (x0,x4), (x1,x5), (x2,x6), (x3,x7)
            __nv_bfloat162 x04 = __halves2bfloat162(inp[0], inp[4]);
            __nv_bfloat162 x15 = __halves2bfloat162(inp[1], inp[5]);
            __nv_bfloat162 x26 = __halves2bfloat162(inp[2], inp[6]);
            __nv_bfloat162 x37 = __halves2bfloat162(inp[3], inp[7]);

            // 4 x bf16x2 FMA = 8 multiply-adds in 4 instructions
            acc_a = __hfma2(d04, x04, acc_a);
            acc_b = __hfma2(d15, x15, acc_b);
            acc_a = __hfma2(d26, x26, acc_a);
            acc_b = __hfma2(d37, x37, acc_b);
        }
    }

    // Remainder loop
    for (; kp < K_packed; kp += 32) {
        int k_base = kp * 8;
        int g = GROUP_SIZE_IS_POW2 ? (k_base >> group_shift) : (k_base / group_size);

        __nv_bfloat16 scale_bf = __ldg(&sc_row[g]);
        int32_t zp_packed = __ldg(&zp_row[g]);
        int zero = (zp_packed >> zp_bit_offset) & 0xF;

        __nv_bfloat16 magic_zp_val = __float2bfloat16(128.0f + (float)zero);
        __nv_bfloat162 scale_bf2 = __halves2bfloat162(scale_bf, scale_bf);
        __nv_bfloat162 magic_zp = __halves2bfloat162(magic_zp_val, magic_zp_val);

        int32_t word = __ldg(&wp_row[kp]);
        __nv_bfloat162 d04, d15, d26, d37;
        dequant_8xint4_bf16_magic(word, magic_zp, scale_bf2, d04, d15, d26, d37);

        __nv_bfloat162 x04 = __halves2bfloat162(__ldg(&input[k_base+0]), __ldg(&input[k_base+4]));
        __nv_bfloat162 x15 = __halves2bfloat162(__ldg(&input[k_base+1]), __ldg(&input[k_base+5]));
        __nv_bfloat162 x26 = __halves2bfloat162(__ldg(&input[k_base+2]), __ldg(&input[k_base+6]));
        __nv_bfloat162 x37 = __halves2bfloat162(__ldg(&input[k_base+3]), __ldg(&input[k_base+7]));

        acc_a = __hfma2(d04, x04, acc_a);
        acc_b = __hfma2(d15, x15, acc_b);
        acc_a = __hfma2(d26, x26, acc_a);
        acc_b = __hfma2(d37, x37, acc_b);
    }

    // Merge bf16x2 accumulators → float for precise warp reduction
    __nv_bfloat162 sum_bf2 = __hadd2(acc_a, acc_b);
    float acc = __bfloat162float(__low2bfloat16(sum_bf2))
              + __bfloat162float(__high2bfloat16(sum_bf2));

    acc = warp_reduce_sum_gemv(acc);
    if (lane_id == 0) {
        output[row] = __float2bfloat16(acc);
    }
}

// ============================================================================
//  INT4 GEMM (M>1) — Batched-GEMV style: grid.y = M, each CTA row uses the
//  same warp-reduce GEMV pipeline as the M=1 path. Weight is broadcast across
//  all M rows (read once in L2). This gives full GEMV performance for any M.
// ============================================================================

template <int WARPS_PER_BLOCK, bool GROUP_SIZE_IS_POW2>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 4)
kpack_gemm_kernel(
    const __nv_bfloat16* __restrict__ input,         // [M, K]
    const int32_t* __restrict__ weight_packed,        // [N, K/8]
    const int32_t* __restrict__ weight_zero_point,    // [N/8, num_groups]
    const __nv_bfloat16* __restrict__ weight_scale,   // [N, num_groups]
    __nv_bfloat16* __restrict__ output,               // [M, N]
    const int M, const int N, const int K, const int group_size,
    const int group_shift
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int K_packed = K >> 3;
    const int num_groups = K / group_size;

    // blockIdx.x → which N-row (weight row), blockIdx.y → which M-row (input row)
    const int n_row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int m_row = blockIdx.y;
    if (n_row >= N || m_row >= M) return;

    const int32_t* wp_row = weight_packed + n_row * K_packed;
    const __nv_bfloat16* sc_row = weight_scale + n_row * num_groups;

    const int zp_row_packed = n_row >> 3;
    const int zp_bit_offset = (n_row & 7) * 4;
    const int32_t* zp_row = weight_zero_point + zp_row_packed * num_groups;

    const int4* input_i4 = reinterpret_cast<const int4*>(input + m_row * K);

    __nv_bfloat162 acc_a = __float2bfloat162_rn(0.0f);
    __nv_bfloat162 acc_b = __float2bfloat162_rn(0.0f);

    int kp = lane_id;

    // Main loop with 4x unroll (same as GEMV path)
    for (; kp + 3 * 32 < K_packed; kp += 4 * 32) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int kpu = kp + u * 32;
            int k_base = kpu * 8;
            int g = GROUP_SIZE_IS_POW2 ? (k_base >> group_shift) : (k_base / group_size);

            __nv_bfloat16 scale_bf = __ldg(&sc_row[g]);
            int32_t zp_packed = __ldg(&zp_row[g]);
            int zero = (zp_packed >> zp_bit_offset) & 0xF;

            __nv_bfloat16 magic_zp_val = __float2bfloat16(128.0f + (float)zero);
            __nv_bfloat162 scale_bf2 = __halves2bfloat162(scale_bf, scale_bf);
            __nv_bfloat162 magic_zp = __halves2bfloat162(magic_zp_val, magic_zp_val);

            int32_t word = __ldg(&wp_row[kpu]);
            __nv_bfloat162 d04, d15, d26, d37;
            dequant_8xint4_bf16_magic(word, magic_zp, scale_bf2, d04, d15, d26, d37);

            int4 in = __ldg(&input_i4[kpu]);
            const __nv_bfloat16* inp = reinterpret_cast<const __nv_bfloat16*>(&in);

            __nv_bfloat162 x04 = __halves2bfloat162(inp[0], inp[4]);
            __nv_bfloat162 x15 = __halves2bfloat162(inp[1], inp[5]);
            __nv_bfloat162 x26 = __halves2bfloat162(inp[2], inp[6]);
            __nv_bfloat162 x37 = __halves2bfloat162(inp[3], inp[7]);

            acc_a = __hfma2(d04, x04, acc_a);
            acc_b = __hfma2(d15, x15, acc_b);
            acc_a = __hfma2(d26, x26, acc_a);
            acc_b = __hfma2(d37, x37, acc_b);
        }
    }

    // Remainder loop
    const __nv_bfloat16* inp_base = input + m_row * K;
    for (; kp < K_packed; kp += 32) {
        int k_base = kp * 8;
        int g = GROUP_SIZE_IS_POW2 ? (k_base >> group_shift) : (k_base / group_size);

        __nv_bfloat16 scale_bf = __ldg(&sc_row[g]);
        int32_t zp_packed = __ldg(&zp_row[g]);
        int zero = (zp_packed >> zp_bit_offset) & 0xF;

        __nv_bfloat16 magic_zp_val = __float2bfloat16(128.0f + (float)zero);
        __nv_bfloat162 scale_bf2 = __halves2bfloat162(scale_bf, scale_bf);
        __nv_bfloat162 magic_zp = __halves2bfloat162(magic_zp_val, magic_zp_val);

        int32_t word = __ldg(&wp_row[kp]);
        __nv_bfloat162 d04, d15, d26, d37;
        dequant_8xint4_bf16_magic(word, magic_zp, scale_bf2, d04, d15, d26, d37);

        __nv_bfloat162 x04 = __halves2bfloat162(__ldg(&inp_base[k_base+0]), __ldg(&inp_base[k_base+4]));
        __nv_bfloat162 x15 = __halves2bfloat162(__ldg(&inp_base[k_base+1]), __ldg(&inp_base[k_base+5]));
        __nv_bfloat162 x26 = __halves2bfloat162(__ldg(&inp_base[k_base+2]), __ldg(&inp_base[k_base+6]));
        __nv_bfloat162 x37 = __halves2bfloat162(__ldg(&inp_base[k_base+3]), __ldg(&inp_base[k_base+7]));

        acc_a = __hfma2(d04, x04, acc_a);
        acc_b = __hfma2(d15, x15, acc_b);
        acc_a = __hfma2(d26, x26, acc_a);
        acc_b = __hfma2(d37, x37, acc_b);
    }

    // Warp reduce
    __nv_bfloat162 sum_bf2 = __hadd2(acc_a, acc_b);
    float acc = __bfloat162float(__low2bfloat16(sum_bf2))
              + __bfloat162float(__high2bfloat16(sum_bf2));

    acc = warp_reduce_sum_gemv(acc);
    if (lane_id == 0) {
        output[m_row * N + n_row] = __float2bfloat16(acc);
    }
}

// ============================================================================
//  INT4 C-linkage wrappers
// ============================================================================

extern "C" void kpack_gemv_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int N, int K, int group_size, cudaStream_t stream
) {
    constexpr int WARPS = 4;
    int grid_x = (N + WARPS - 1) / WARPS;
    const bool group_size_is_pow2 = group_size > 0 && ((group_size & (group_size - 1)) == 0);
    const int group_shift = group_size_is_pow2 ? __builtin_ctz(group_size) : 0;

    if (group_size_is_pow2) {
        kpack_gemv_kernel<WARPS, true><<<grid_x, WARPS * 32, 0, stream>>>(
            (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
            (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
            (__nv_bfloat16*)output, N, K, group_size, group_shift);
    } else {
        kpack_gemv_kernel<WARPS, false><<<grid_x, WARPS * 32, 0, stream>>>(
            (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
            (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
            (__nv_bfloat16*)output, N, K, group_size, 0);
    }
}

extern "C" void kpack_gemm_cu(
    const void* input, const void* weight_packed, const void* weight_zero_point,
    const void* weight_scale, void* output,
    int M, int N, int K, int group_size, cudaStream_t stream
) {
    constexpr int WARPS = 4;
    int grid_x = (N + WARPS - 1) / WARPS;
    int grid_y = M;
    dim3 grid(grid_x, grid_y);
    const bool group_size_is_pow2 = group_size > 0 && ((group_size & (group_size - 1)) == 0);
    const int group_shift = group_size_is_pow2 ? __builtin_ctz(group_size) : 0;

    if (group_size_is_pow2) {
        kpack_gemm_kernel<WARPS, true><<<grid, WARPS * 32, 0, stream>>>(
            (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
            (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
            (__nv_bfloat16*)output, M, N, K, group_size, group_shift);
    } else {
        kpack_gemm_kernel<WARPS, false><<<grid, WARPS * 32, 0, stream>>>(
            (const __nv_bfloat16*)input, (const int32_t*)weight_packed,
            (const int32_t*)weight_zero_point, (const __nv_bfloat16*)weight_scale,
            (__nv_bfloat16*)output, M, N, K, group_size, group_shift);
    }
}

// ============================================================================
//  Block-scaled E4M3FN matmul (dynamic W8A8, BF16 input/output)
//
//  Scratch is supplied by CudaConfig's address-stable workspace. The first
//  kernel quantizes every 1x128 BF16 activation block to E4M3 and writes its
//  dequant scale. Decode uses the quantized data in a warp GEMV; prefill first
//  attempts the architecture-accelerated CUTLASS blockwise Tensor Core kernel
//  and retains a tiled W8A8 CUDA fallback for unsupported configurations.
// ============================================================================

#ifdef RUSTINFER_HAS_FP8_BLOCK_ACCELERATED
extern "C" int fp8_block_accelerated_init(int device_id);
extern "C" int fp8_block_accelerated_bf16(
    const void* activation_fp8, const void* weight_fp8,
    const float* activation_scale_inv, const float* weight_scale_inv,
    void* output_bf16, int M, int N, int K, int device_id,
    void* workspace, size_t workspace_size, cudaStream_t stream);
#endif

extern "C" int fp8_block_matmul_init_cu(int device_id) {
#ifdef RUSTINFER_HAS_FP8_BLOCK_ACCELERATED
    return fp8_block_accelerated_init(device_id);
#else
    (void)device_id;
    return 0;
#endif
}

__device__ __forceinline__ float e4m3fn_to_float(const unsigned char bits) {
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;
    const unsigned int sign = (static_cast<unsigned int>(bits) & 0x80u) << 24;

    if (exponent == 0) {
        const float magnitude = static_cast<float>(mantissa) * 0x1.0p-9f;
        return sign ? -magnitude : magnitude;
    }
    if (exponent == 0x0f && mantissa == 0x07) {
        return __uint_as_float(sign | 0x7fc00000u);
    }

    // E4M3FN normal values map exactly into an f32 exponent/mantissa: bias 7
    // becomes bias 127 (+120), and the three mantissa bits occupy f32's top 3.
    return __uint_as_float(sign | ((exponent + 120u) << 23) | (mantissa << 20));
}

__device__ __forceinline__ float warp_sum_fp8(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_max_fp8(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__global__ void fp8_block_act_quant_1x128_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    float* __restrict__ scale_inv,
    const int M, const int K, const int scale_cols
) {
    const int row = blockIdx.y;
    const int scale_col = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int k = scale_col * 128 + threadIdx.x;
    const bool valid = row < M && k < K;
    const float value = valid
        ? __bfloat162float(__ldg(input + static_cast<size_t>(row) * K + k))
        : 0.0f;

    float block_max = warp_max_fp8(fabsf(value));
    __shared__ float warp_maxima[4];
    __shared__ float dequant_scale;
    if (lane == 0) warp_maxima[warp] = block_max;
    __syncthreads();
    if (warp == 0) {
        block_max = lane < 4 ? warp_maxima[lane] : 0.0f;
        block_max = warp_max_fp8(block_max);
        if (lane == 0) {
            // E4M3FN's largest finite magnitude is 448. scale_inv is the
            // multiplier used to reconstruct BF16/FP32 values from FP8.
            dequant_scale = block_max > 0.0f ? block_max * (1.0f / 448.0f) : 1.0f;
            scale_inv[static_cast<size_t>(row) * scale_cols + scale_col] = dequant_scale;
        }
    }
    __syncthreads();

    if (valid) {
        output[static_cast<size_t>(row) * K + k] = __nv_cvt_float_to_fp8(
            value / dequant_scale, __NV_SATFINITE, __NV_E4M3);
    }
}

template <int WARPS_PER_BLOCK>
__global__ void fp8_block_gemv_w8a8_bf16_kernel(
    const unsigned char* __restrict__ input,
    const float* __restrict__ input_scale_inv,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ weight_scale_inv,
    __nv_bfloat16* __restrict__ output,
    const int N, const int K, const int scale_cols,
    const int block_n, const int block_k
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (n >= N) return;

    const int scale_row = n / block_n;
    const unsigned char* weight_row = weight + static_cast<size_t>(n) * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        const float a = e4m3fn_to_float(__ldg(input + k));
        const float w = e4m3fn_to_float(__ldg(weight_row + k));
        const int scale_col = k / block_k;
        const float a_scale = __ldg(input_scale_inv + scale_col);
        const float w_scale = __ldg(
            weight_scale_inv + static_cast<size_t>(scale_row) * scale_cols + scale_col);
        acc = fmaf(a * a_scale, w * w_scale, acc);
    }
    acc = warp_sum_fp8(acc);
    if (lane == 0) output[n] = __float2bfloat16_rn(acc);
}

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void fp8_block_gemm_w8a8_bf16_kernel(
    const unsigned char* __restrict__ input,
    const float* __restrict__ input_scale_inv,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ weight_scale_inv,
    __nv_bfloat16* __restrict__ output,
    const int M, const int N, const int K, const int scale_cols,
    const int block_n, const int block_k
) {
    __shared__ float a_tile[TILE_M][TILE_K];
    __shared__ float w_tile[TILE_N][TILE_K];

    const int tid = threadIdx.x;
    const int local_m = tid / TILE_N;
    const int local_n = tid % TILE_N;
    const int global_m = blockIdx.y * TILE_M + local_m;
    const int global_n = blockIdx.x * TILE_N + local_n;
    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        for (int index = tid; index < TILE_M * TILE_K; index += blockDim.x) {
            const int row = index / TILE_K;
            const int col = index % TILE_K;
            const int m = blockIdx.y * TILE_M + row;
            const int k = k0 + col;
            if (m < M && k < K) {
                const float a = e4m3fn_to_float(
                    __ldg(input + static_cast<size_t>(m) * K + k));
                const float s = __ldg(
                    input_scale_inv + static_cast<size_t>(m) * scale_cols + k / block_k);
                a_tile[row][col] = a * s;
            } else {
                a_tile[row][col] = 0.0f;
            }
        }
        for (int index = tid; index < TILE_N * TILE_K; index += blockDim.x) {
            const int row = index / TILE_K;
            const int col = index % TILE_K;
            const int n = blockIdx.x * TILE_N + row;
            const int k = k0 + col;
            if (n < N && k < K) {
                const float w = e4m3fn_to_float(
                    __ldg(weight + static_cast<size_t>(n) * K + k));
                const float s = __ldg(
                    weight_scale_inv + static_cast<size_t>(n / block_n) * scale_cols + k / block_k);
                w_tile[row][col] = w * s;
            } else {
                w_tile[row][col] = 0.0f;
            }
        }
        __syncthreads();

        if (global_m < M && global_n < N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; ++k) {
                acc = fmaf(a_tile[local_m][k], w_tile[local_n][k], acc);
            }
        }
        __syncthreads();
    }

    if (global_m < M && global_n < N) {
        output[static_cast<size_t>(global_m) * N + global_n] = __float2bfloat16_rn(acc);
    }
}

extern "C" int fp8_block_matmul_bf16_cu(
    const void* input, const void* weight, const void* weight_scale_inv,
    void* output, int M, int N, int K, int scale_cols,
    int block_n, int block_k, int device_id,
    void* workspace, size_t workspace_size, cudaStream_t stream
) {
    if (block_n != 128 || block_k != 128 || scale_cols != K / 128) return -1;

    const size_t activation_bytes = static_cast<size_t>(M) * K;
    const size_t scale_offset = (activation_bytes + 127u) & ~size_t(127u);
    const size_t activation_scale_bytes = static_cast<size_t>(M) * scale_cols * sizeof(float);
    const size_t cutlass_offset = (scale_offset + activation_scale_bytes + 127u) & ~size_t(127u);
    if (workspace == nullptr || cutlass_offset > workspace_size) return -2;

    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    auto* activation_fp8 = workspace_bytes;
    auto* activation_scale_inv = reinterpret_cast<float*>(workspace_bytes + scale_offset);
    void* cutlass_workspace = workspace_bytes + cutlass_offset;
    const size_t cutlass_workspace_size = workspace_size - cutlass_offset;

    const dim3 quant_grid(scale_cols, M);
    fp8_block_act_quant_1x128_kernel<<<quant_grid, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), activation_fp8,
        activation_scale_inv, M, K, scale_cols);

    if (M == 1) {
        constexpr int WARPS = 4;
        const int grid_x = (N + WARPS - 1) / WARPS;
        fp8_block_gemv_w8a8_bf16_kernel<WARPS><<<grid_x, WARPS * 32, 0, stream>>>(
            activation_fp8, activation_scale_inv,
            static_cast<const unsigned char*>(weight),
            static_cast<const float*>(weight_scale_inv),
            static_cast<__nv_bfloat16*>(output),
            N, K, scale_cols, block_n, block_k);
        return 1;  // dynamic W8A8 warp GEMV
    }

#ifdef RUSTINFER_HAS_FP8_BLOCK_ACCELERATED
    const int cutlass_status = fp8_block_accelerated_bf16(
        activation_fp8, weight, activation_scale_inv,
        static_cast<const float*>(weight_scale_inv), output,
        M, N, K, device_id, cutlass_workspace, cutlass_workspace_size, stream);
    if (cutlass_status == 0) return 2;  // Hopper blockwise Tensor Core path
#else
    (void)device_id;
    (void)cutlass_workspace;
    (void)cutlass_workspace_size;
#endif

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    constexpr int TILE_K = 32;
    const dim3 block(TILE_M * TILE_N);
    const dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    fp8_block_gemm_w8a8_bf16_kernel<TILE_M, TILE_N, TILE_K><<<grid, block, 0, stream>>>(
        activation_fp8, activation_scale_inv,
        static_cast<const unsigned char*>(weight),
        static_cast<const float*>(weight_scale_inv),
        static_cast<__nv_bfloat16*>(output),
        M, N, K, scale_cols, block_n, block_k);
    return 3;  // portable CUDA W8A8 fallback
}

// ============================================================================
//  Strided-batched GEMM (BF16 / F32) — for SDPA's per-head Q@K^T and attn@V.
//
//  Computes `C[b] = A[b] @ B[b]^T` for `b in [0, batch_count)`, where each
//  batch slice is row-major `[m,k]`/`[n,k]`/`[m,n]` with batch stride =
//  `m*k`/`n*k`/`m*n` (contiguous batch dim).
//
//  We use cuBLAS classic API (`cublasGemmStridedBatchedEx`) — simpler than
//  cuBLASLt for batched, and plenty fast for SDPA of seq≤2048, n_heads≤32.
//
//  Row-major hack: cuBLAS is column-major. With row-major C = A @ B^T:
//    column-major view: C^col = (A @ B^T)^T = B @ A^T
//    so call cublasGemm with transA=N (B is col-major [K, N] viewed as row [N,K]),
//    transB=T (A^T = A in col-major when A is row-major [M,K]),
//    m=N, n=M, k=K, A_arg=B, B_arg=A, C_arg=C (row [M,N] = col [N,M]).
// ============================================================================

#include <cublas_v2.h>

static cublasHandle_t s_cublas_handle = nullptr;
static cublasHandle_t get_cublas_handle() {
    if (!s_cublas_handle) {
        cublasStatus_t s = cublasCreate(&s_cublas_handle);
        if (s != CUBLAS_STATUS_SUCCESS) {
            printf("cublasCreate failed: %d\n", s);
            return nullptr;
        }
    }
    return s_cublas_handle;
}

extern "C" void gemm_strided_batched_bf16_axbt(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    long long strideA, long long strideB, long long strideC,
    int batch_count,
    cudaStream_t stream
) {
    cublasHandle_t h = get_cublas_handle();
    if (!h) return;
    cublasSetStream(h, stream);
    float alpha = 1.0f, beta = 0.0f;
    // Row-major C[M,N] = A[M,K] @ B[N,K]^T.
    // Col-major: C_col[N,M] = B[N,K] · A^T[K,M].
    //   X = B (col-major view of [N,K] data is [K,N]), op_T → [N,K], lda=K
    //   Y = A^T (col-major view of [M,K] data is [K,M] which = A^T), op_N, ldb=K
    //   m=N, n=M, k=K, ldc=N.
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K, strideB,
        A, CUDA_R_16BF, K, strideA,
        &beta,
        C, CUDA_R_16BF, N, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (s != CUBLAS_STATUS_SUCCESS) {
        printf("gemm_strided_batched_bf16_axbt failed: %d (M=%d N=%d K=%d batch=%d)\n",
            s, M, N, K, batch_count);
    }
}

extern "C" void gemm_strided_batched_bf16_axb(
    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
    int M, int N, int K,
    long long strideA, long long strideB, long long strideC,
    int batch_count,
    cudaStream_t stream
) {
    // Row-major C[M,N] = A[M,K] @ B[K,N]
    // Col-major: cublas computes C_col = B_col @ A_col with m=N,n=M,k=K,
    //   B_arg=B (row [K,N] = col [N,K]) → CUBLAS_OP_N to get [K,N]? No:
    //   col [N,K] needs transpose to become [K,N] for the gemm = OP_T.
    //   A_arg=A (row [M,K] = col [K,M]) → no transpose to get [K,M] = OP_N.
    cublasHandle_t h = get_cublas_handle();
    if (!h) return;
    cublasSetStream(h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        h,
        CUBLAS_OP_T,   // transpose B from col [N,K] to [K,N]
        CUBLAS_OP_N,   // A is col [K,M] already
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N, strideB,    // ldb = N (col-major rows of [N,K])
        A, CUDA_R_16BF, K, strideA,    // lda = K (col-major rows of [K,M])
        &beta,
        C, CUDA_R_16BF, N, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (s != CUBLAS_STATUS_SUCCESS) {
        printf("gemm_strided_batched_bf16_axb failed: %d (M=%d N=%d K=%d batch=%d)\n",
            s, M, N, K, batch_count);
    }
}

extern "C" void gemm_strided_batched_f32_axbt(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    long long strideA, long long strideB, long long strideC,
    int batch_count,
    cudaStream_t stream
) {
    cublasHandle_t h = get_cublas_handle();
    if (!h) return;
    cublasSetStream(h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, K, strideB,
        A, CUDA_R_32F, K, strideA,
        &beta,
        C, CUDA_R_32F, N, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (s != CUBLAS_STATUS_SUCCESS) {
        printf("gemm_strided_batched_f32_axbt failed: %d (M=%d N=%d K=%d batch=%d)\n",
            s, M, N, K, batch_count);
    }
}

extern "C" void gemm_strided_batched_f32_axb(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    long long strideA, long long strideB, long long strideC,
    int batch_count,
    cudaStream_t stream
) {
    cublasHandle_t h = get_cublas_handle();
    if (!h) return;
    cublasSetStream(h, stream);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N, strideB,
        A, CUDA_R_32F, K, strideA,
        &beta,
        C, CUDA_R_32F, N, strideC,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (s != CUBLAS_STATUS_SUCCESS) {
        printf("gemm_strided_batched_f32_axb failed: %d (M=%d N=%d K=%d batch=%d)\n",
            s, M, N, K, batch_count);
    }
}

// ============================================================================
//  Standard cuBLAS sgemm wrapper for F32 (uses Tensor Cores TF32 on Hopper).
//
//  Row-major C[M,N] = A[M,K] @ B[N,K]^T  (matches our Linear: out=in@W^T).
//  Col-major equivalent: C^col[N,M] = B[N,K] @ A[K,M]^T.
//    cublas: C^col = op(X) op(Y), m=N, n=M, k=K
//    X = B (col [K,N]) → op_T → [N,K], lda=K
//    Y = A (col [K,M]) → op_N → [K,M], ldb=K
//    ldc = N
// ============================================================================

extern "C" void gemm_cublas_f32_axbt(
    const float* A, const float* B, float* C,
    int M, int N, int K, int ldc,
    cudaStream_t stream
) {
    cublasHandle_t h = get_cublas_handle();
    if (!h) return;
    cublasSetStream(h, stream);
    // Enable TF32 (default on Hopper but make sure).
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, K,
        A, CUDA_R_32F, K,
        &beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    if (s != CUBLAS_STATUS_SUCCESS) {
        printf("gemm_cublas_f32_axbt failed: %d (M=%d N=%d K=%d)\n", s, M, N, K);
    }
}

// ============================================================================
//  Standard cuBLAS sgemm wrapper for F32 (uses Tensor Cores TF32 on Hopper).
//
//  Row-major C[M,N] = A[M,K] @ B[N,K]^T  (matches our Linear: out=in@W^T).
//  Col-major equivalent: C^col[N,M] = B[N,K] @ A[K,M]^T.
//    cublas: C^col = op(X) op(Y), m=N, n=M, k=K
//    X = B (col [K,N]) → op_T → [N,K], lda=K
//    Y = A (col [K,M]) → op_N → [K,M], ldb=K
//    ldc = N
// ============================================================================
