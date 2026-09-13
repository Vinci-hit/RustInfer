#include <cuda_runtime.h>
#include <cstdint>

// One CTA per packed Q row. Metadata is shared by every attention layer.
// Recompute the mapping from device lengths so graph replay can vary prefixes.
__global__ void prepare_short_query_rows_kernel(
    const int32_t* tables, const int32_t* cu_q, const int32_t* kv_lens,
    int32_t* row_tables, int32_t* row_q_lens, int32_t* row_kv_lens,
    int batch, int max_blocks, int block_size) {
    const int row = blockIdx.x;
    int req = 0;
    while (req < batch && row >= cu_q[req + 1]) ++req;
    int visible = 0;
    if (req < batch && row >= cu_q[req]) {
        const int q_len = cu_q[req + 1] - cu_q[req];
        const int kv_len = kv_lens[req];
        if (q_len > 0 && kv_len >= q_len && kv_len <= max_blocks * block_size)
            visible = kv_len - q_len + (row - cu_q[req]) + 1;
    }
    if (threadIdx.x == 0) {
        row_q_lens[row] = visible > 0 ? 1 : 0;
        row_kv_lens[row] = visible;
    }
    const int pages = visible / block_size + (visible % block_size != 0);
    for (int page = threadIdx.x; page < max_blocks; page += blockDim.x) {
        // Padding in the original table need not be initialized.
        row_tables[static_cast<int64_t>(row) * max_blocks + page] =
            page < pages ? tables[static_cast<int64_t>(req) * max_blocks + page] : 0;
    }
}

extern "C" int rustinfer_prepare_short_query_rows(
    const int32_t* tables, const int32_t* cu_q, const int32_t* kv_lens,
    int32_t* row_tables, int32_t* row_q_lens, int32_t* row_kv_lens,
    int batch, int rows, int max_blocks, int block_size, cudaStream_t stream) {
    prepare_short_query_rows_kernel<<<rows, 256, 0, stream>>>(
        tables, cu_q, kv_lens, row_tables, row_q_lens, row_kv_lens,
        batch, max_blocks, block_size);
    return static_cast<int>(cudaGetLastError());
}
