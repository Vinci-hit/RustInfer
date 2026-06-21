// cudnn_paged_attention.cu
// -----------------------------------------------------------------------------
// cuDNN frontend SDPA path for paged decode attention.
//
// This intentionally covers decode-only first (S_q = 1). Ragged prefill needs a
// token-level causal mask; the frontend graph below only uses paged K/V
// plus KV sequence lengths, so the ragged Flash kernel remains the fallback path.
// -----------------------------------------------------------------------------

#include "flash_attn_gqa.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cudnn_paged_attention {

static constexpr int64_t UID_SCALE = 1;
static constexpr int64_t UID_V = 2;
static constexpr int64_t UID_K = 3;
static constexpr int64_t UID_PAGE_TABLE = 4;
static constexpr int64_t UID_Q = 5;
static constexpr int64_t UID_SEQ_LEN_KV = 6;
static constexpr int64_t UID_SEQ_LEN_Q = 7;
static constexpr int64_t UID_O = 8;

struct PlanKey {
    uintptr_t handle;
    int dtype;
    int batch;
    int num_blocks;
    int max_blocks_per_seq;
    int block_size;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    int64_t q_stride_b;
    int64_t q_stride_h;
    int64_t o_stride_b;
    int64_t o_stride_h;
    uint32_t scale_bits;

    bool operator==(const PlanKey& rhs) const {
        return handle == rhs.handle &&
               dtype == rhs.dtype &&
               batch == rhs.batch &&
               num_blocks == rhs.num_blocks &&
               max_blocks_per_seq == rhs.max_blocks_per_seq &&
               block_size == rhs.block_size &&
               num_q_heads == rhs.num_q_heads &&
               num_kv_heads == rhs.num_kv_heads &&
               head_dim == rhs.head_dim &&
               q_stride_b == rhs.q_stride_b &&
               q_stride_h == rhs.q_stride_h &&
               o_stride_b == rhs.o_stride_b &&
               o_stride_h == rhs.o_stride_h &&
               scale_bits == rhs.scale_bits;
    }
};

struct PlanKeyHash {
    size_t operator()(const PlanKey& k) const {
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) {
            h ^= static_cast<size_t>(v);
            h *= 1099511628211ull;
        };
        mix(k.handle);
        mix(static_cast<uint64_t>(k.dtype));
        mix(static_cast<uint64_t>(k.batch));
        mix(static_cast<uint64_t>(k.num_blocks));
        mix(static_cast<uint64_t>(k.max_blocks_per_seq));
        mix(static_cast<uint64_t>(k.block_size));
        mix(static_cast<uint64_t>(k.num_q_heads));
        mix(static_cast<uint64_t>(k.num_kv_heads));
        mix(static_cast<uint64_t>(k.head_dim));
        mix(static_cast<uint64_t>(k.q_stride_b));
        mix(static_cast<uint64_t>(k.q_stride_h));
        mix(static_cast<uint64_t>(k.o_stride_b));
        mix(static_cast<uint64_t>(k.o_stride_h));
        mix(static_cast<uint64_t>(k.scale_bits));
        return h;
    }
};

struct CudnnPlan {
    std::shared_ptr<cudnn_frontend::graph::Graph> graph;
    int64_t workspace_size = 0;
};

static std::mutex g_plan_mutex;
static std::unordered_map<PlanKey, CudnnPlan*, PlanKeyHash> g_plan_cache;

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool check(cudnnStatus_t status, const char* what) {
    if (status == CUDNN_STATUS_SUCCESS) {
        return true;
    }
    fprintf(stderr, "[cudnn_paged_attention] %s failed: %s (%d)\n",
            what, cudnnGetErrorString(status), static_cast<int>(status));
    return false;
}

static bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &capture_status);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cudnn_paged_attention] cudaStreamIsCapturing failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return capture_status != cudaStreamCaptureStatusNone;
}

static bool check_fe(cudnn_frontend::error_t status, const char* what) {
    if (status.is_good()) {
        return true;
    }
    fprintf(stderr, "[cudnn_paged_attention] %s failed: %s\n", what, status.get_message().c_str());
    return false;
}

static std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
make_tensor(cudnn_frontend::graph::Graph& graph,
            const char* name,
            cudnn_frontend::DataType_t dtype,
            int64_t uid,
            std::vector<int64_t> dims,
            std::vector<int64_t> strides,
            bool is_output = false,
            bool is_by_value = false) {
    cudnn_frontend::graph::Tensor_attributes attrs;
    attrs.set_name(name)
        .set_uid(uid)
        .set_data_type(dtype)
        .set_dim(dims)
        .set_stride(strides)
        .set_alignment(16)
        .set_is_pass_by_value(is_by_value);
    if (is_output) {
        attrs.set_output(true);
    }
    auto tensor = graph.tensor(attrs);
    return tensor;
}

static CudnnPlan* build_plan(const PlanKey& key, cudnnHandle_t handle, float softmax_scale) {
    CudnnPlan* raw_plan = new CudnnPlan();
    CudnnPlan& plan = *raw_plan;

    cudnn_frontend::DataType_t io_dtype =
        key.dtype == static_cast<int>(CUDNN_DATA_BFLOAT16) ? cudnn_frontend::DataType_t::BFLOAT16
                                                          : cudnn_frontend::DataType_t::HALF;

    std::vector<int64_t> q_dims = {key.batch, key.num_q_heads, 1, key.head_dim};
    std::vector<int64_t> q_strides = {key.q_stride_b, key.q_stride_h, key.head_dim, 1};
    std::vector<int64_t> o_dims = {key.batch, key.num_q_heads, 1, key.head_dim};
    std::vector<int64_t> o_strides = {key.o_stride_b, key.o_stride_h, key.head_dim, 1};

    std::vector<int64_t> kv_dims = {key.num_blocks, key.num_kv_heads, key.block_size, key.head_dim};
    std::vector<int64_t> kv_strides = {
        static_cast<int64_t>(key.block_size) * key.num_kv_heads * key.head_dim,
        key.head_dim,
        static_cast<int64_t>(key.num_kv_heads) * key.head_dim,
        1,
    };

    std::vector<int64_t> page_dims = {key.batch, 1, key.max_blocks_per_seq, 1};
    std::vector<int64_t> page_strides = {key.max_blocks_per_seq, key.max_blocks_per_seq, 1, 1};
    std::vector<int64_t> len_dims = {key.batch, 1, 1, 1};
    std::vector<int64_t> len_strides = {1, 1, 1, 1};

    auto graph = std::make_shared<cudnn_frontend::graph::Graph>();
    graph->set_io_data_type(io_dtype)
        .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
        .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto scale_desc = make_tensor(*graph, "scale", cudnn_frontend::DataType_t::FLOAT, UID_SCALE, {1}, {1}, false, true);
    auto v_desc = make_tensor(*graph, "V", io_dtype, UID_V, kv_dims, kv_strides);
    auto k_desc = make_tensor(*graph, "K", io_dtype, UID_K, kv_dims, kv_strides);
    auto page_desc = make_tensor(*graph, "page_table", cudnn_frontend::DataType_t::INT32, UID_PAGE_TABLE, page_dims, page_strides);
    auto q_desc = make_tensor(*graph, "Q", io_dtype, UID_Q, q_dims, q_strides);
    auto seq_len_kv_desc = make_tensor(*graph, "seq_len_kv", cudnn_frontend::DataType_t::INT32, UID_SEQ_LEN_KV, len_dims, len_strides);
    auto seq_len_q_desc = make_tensor(*graph, "seq_len_q", cudnn_frontend::DataType_t::INT32, UID_SEQ_LEN_Q, len_dims, len_strides);

    cudnn_frontend::graph::SDPA_attributes sdpa_attrs;
    sdpa_attrs.set_name("paged_decode_sdpa")
        .set_generate_stats(false)
        .set_attn_scale(scale_desc)
        .set_padding_mask(true)
        .set_seq_len_q(seq_len_q_desc)
        .set_seq_len_kv(seq_len_kv_desc)
        .set_paged_attention_k_table(page_desc)
        .set_paged_attention_v_table(page_desc)
        .set_paged_attention_max_seq_len_kv(key.max_blocks_per_seq * key.block_size)
        .set_implementation(cudnn_frontend::AttentionImplementation_t::UNIFIED);

    auto sdpa_outputs = graph->sdpa(q_desc, k_desc, v_desc, sdpa_attrs);
    auto o_desc = sdpa_outputs[0];
    o_desc->set_name("O")
        .set_uid(UID_O)
        .set_dim(o_dims)
        .set_stride(o_strides)
        .set_data_type(io_dtype)
        .set_alignment(16)
        .set_output(true);

    std::vector<cudnn_frontend::HeurMode_t> modes = {
        cudnn_frontend::HeurMode_t::A,
        cudnn_frontend::HeurMode_t::B,
        cudnn_frontend::HeurMode_t::FALLBACK,
    };
    if (!check_fe(graph->build(handle, modes, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE, false),
                  "build frontend graph")) {
        delete raw_plan;
        return nullptr;
    }

    plan.graph = graph;
    plan.workspace_size = graph->get_workspace_size();
    return raw_plan;
}

static CudnnPlan* get_or_build_plan(const PlanKey& key, cudnnHandle_t handle, float softmax_scale) {
    std::lock_guard<std::mutex> guard(g_plan_mutex);
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        return it->second;
    }
    CudnnPlan* plan = build_plan(key, handle, softmax_scale);
    if (plan != nullptr) {
        g_plan_cache.emplace(key, plan);
    }
    return plan;
}

static CudnnPlan* get_cached_plan(const PlanKey& key) {
    std::lock_guard<std::mutex> guard(g_plan_mutex);
    auto it = g_plan_cache.find(key);
    if (it == g_plan_cache.end()) {
        return nullptr;
    }
    return it->second;
}

template <typename Elem>
static int launch_decode(cudnnHandle_t handle,
                         cudnnDataType_t dtype,
                         const Elem* q,
                         int64_t qsb,
                         int64_t qsh,
                         const Elem* k_pool,
                         const Elem* v_pool,
                         Elem* o,
                         int64_t osb,
                         int64_t osh,
                         const uint32_t* block_tables,
                         int max_blocks_per_seq,
                         int block_size,
                         const int32_t* q_lens,
                         const int32_t* kv_lens,
                         int num_blocks,
                         void* workspace,
                         size_t workspace_bytes,
                         int batch,
                         int num_q_heads,
                         int num_kv_heads,
                         int head_dim,
                         float softmax_scale,
                         cudaStream_t stream) {
    if (batch <= 0) {
        return static_cast<int>(CUDNN_STATUS_SUCCESS);
    }
    if (handle == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        o == nullptr || block_tables == nullptr || q_lens == nullptr || kv_lens == nullptr) {
        fprintf(stderr, "[cudnn_paged_attention] null argument\n");
        return static_cast<int>(CUDNN_STATUS_BAD_PARAM);
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        num_q_heads % num_kv_heads != 0 || num_blocks <= 0 ||
        block_size <= 0 || max_blocks_per_seq <= 0) {
        fprintf(stderr, "[cudnn_paged_attention] invalid shape args: B=%d blocks=%d max_blocks=%d block=%d Hq=%d Hkv=%d D=%d\n",
                batch, num_blocks, max_blocks_per_seq, block_size, num_q_heads, num_kv_heads, head_dim);
        return static_cast<int>(CUDNN_STATUS_BAD_PARAM);
    }

    // cuDNN's execute() path is NOT CUDA-graph-capturable: both cudnnSetStream
    // and the frontend's internal cuStreamGetCtx call are illegal during capture
    // and invalidate it (CUDNN_STATUS_INTERNAL_ERROR / driver error 901). Decline
    // immediately while capturing — without touching the stream — so the caller
    // falls back to the capturable custom flash-decode kernel.
    if (stream_is_capturing(stream)) {
        return static_cast<int>(CUDNN_STATUS_NOT_SUPPORTED);
    }

    cudnnStatus_t status = cudnnSetStream(handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        check(status, "set stream");
        return static_cast<int>(status);
    }

    PlanKey key{};
    key.handle = reinterpret_cast<uintptr_t>(handle);
    key.dtype = static_cast<int>(dtype);
    key.batch = batch;
    key.num_blocks = num_blocks;
    key.max_blocks_per_seq = max_blocks_per_seq;
    key.block_size = block_size;
    key.num_q_heads = num_q_heads;
    key.num_kv_heads = num_kv_heads;
    key.head_dim = head_dim;
    key.q_stride_b = qsb;
    key.q_stride_h = qsh;
    key.o_stride_b = osb;
    key.o_stride_h = osh;
    key.scale_bits = float_bits(softmax_scale);

    CudnnPlan* plan = nullptr;
    if (stream_is_capturing(stream)) {
        plan = get_cached_plan(key);
        if (plan == nullptr) {
            fprintf(stderr, "[cudnn_paged_attention] missing cached cuDNN SDPA plan during CUDA graph capture\n");
            return static_cast<int>(CUDNN_STATUS_NOT_SUPPORTED);
        }
    } else {
        plan = get_or_build_plan(key, handle, softmax_scale);
    }
    if (plan == nullptr || plan->graph == nullptr) {
        return static_cast<int>(CUDNN_STATUS_NOT_SUPPORTED);
    }
    if (plan->workspace_size > static_cast<int64_t>(workspace_bytes)) {
        fprintf(stderr,
                "[cudnn_paged_attention] workspace too small: have %zu bytes, need %lld bytes\n",
                workspace_bytes,
                static_cast<long long>(plan->workspace_size));
        return static_cast<int>(CUDNN_STATUS_ALLOC_FAILED);
    }

    std::unordered_map<int64_t, void*> tensor_ptrs;
    tensor_ptrs.emplace(UID_Q, const_cast<Elem*>(q));
    tensor_ptrs.emplace(UID_K, const_cast<Elem*>(k_pool));
    tensor_ptrs.emplace(UID_V, const_cast<Elem*>(v_pool));
    tensor_ptrs.emplace(UID_O, o);
    tensor_ptrs.emplace(UID_PAGE_TABLE, const_cast<uint32_t*>(block_tables));
    tensor_ptrs.emplace(UID_SEQ_LEN_KV, const_cast<int32_t*>(kv_lens));
    tensor_ptrs.emplace(UID_SEQ_LEN_Q, const_cast<int32_t*>(q_lens));
    tensor_ptrs.emplace(UID_SCALE, &softmax_scale);

    auto fe_status = plan->graph->execute(handle, tensor_ptrs, workspace);
    if (fe_status.is_bad()) {
        fprintf(stderr, "[cudnn_paged_attention] execute failed: %s\n", fe_status.get_message().c_str());
        return static_cast<int>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    return static_cast<int>(CUDNN_STATUS_SUCCESS);
}

}  // namespace cudnn_paged_attention

extern "C" int launch_cudnn_paged_decode_bf16(
    cudnnHandle_t handle,
    const __nv_bfloat16* q, int64_t qsb, int64_t qsh,
    const __nv_bfloat16* k_pool,
    const __nv_bfloat16* v_pool,
          __nv_bfloat16* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* q_lens,
    const int32_t* kv_lens,
    int num_blocks,
    void* workspace,
    size_t workspace_bytes,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream) {
    return cudnn_paged_attention::launch_decode<__nv_bfloat16>(
        handle,
        CUDNN_DATA_BFLOAT16,
        q,
        qsb,
        qsh,
        k_pool,
        v_pool,
        o,
        osb,
        osh,
        block_tables,
        max_blocks_per_seq,
        block_size,
        q_lens,
        kv_lens,
        num_blocks,
        workspace,
        workspace_bytes,
        batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        stream);
}

extern "C" int launch_cudnn_paged_decode_fp16(
    cudnnHandle_t handle,
    const __half* q, int64_t qsb, int64_t qsh,
    const __half* k_pool,
    const __half* v_pool,
          __half* o, int64_t osb, int64_t osh,
    const uint32_t* block_tables,
    int max_blocks_per_seq,
    int block_size,
    const int32_t* q_lens,
    const int32_t* kv_lens,
    int num_blocks,
    void* workspace,
    size_t workspace_bytes,
    int batch, int num_q_heads, int num_kv_heads, int head_dim,
    float softmax_scale,
    cudaStream_t stream) {
    return cudnn_paged_attention::launch_decode<__half>(
        handle,
        CUDNN_DATA_HALF,
        q,
        qsb,
        qsh,
        k_pool,
        v_pool,
        o,
        osb,
        osh,
        block_tables,
        max_blocks_per_seq,
        block_size,
        q_lens,
        kv_lens,
        num_blocks,
        workspace,
        workspace_bytes,
        batch,
        num_q_heads,
        num_kv_heads,
        head_dim,
        softmax_scale,
        stream);
}
