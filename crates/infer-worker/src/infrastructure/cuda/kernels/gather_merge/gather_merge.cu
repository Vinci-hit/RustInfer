#include "gather_merge.h"

__global__ void append_decode_admissions_kernel(
    int* __restrict__ A,
    const int* __restrict__ B,
    int start,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    A[start + i] = B[i];
}

extern "C" void append_decode_admissions(
    int* A,
    const int* B,
    int start,
    int count,
    cudaStream_t stream)
{
    if (count <= 0) return;
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    append_decode_admissions_kernel<<<blocks, threads, 0, stream>>>(A, B, start, count);
}

__device__ __forceinline__ bool token_is_eos(int token, const int* eos_ids, int eos_len)
{
    for (int j = 0; j < eos_len; ++j) {
        if (token == eos_ids[j]) return true;
    }
    return false;
}

// Small-batch control kernel. The serving batch is capped by max_batch_seqs,
// and this launch sits behind the transformer forward, so a sequential scan
// keeps the in-place A compact unambiguous and cheap.
__global__ void merge_compact_decode_kernel(
    int* __restrict__ A,
    const int* __restrict__ C,
    const int* __restrict__ generated_counts,
    const int* __restrict__ max_tokens,
    const int* __restrict__ ignore_eos,
    const int* __restrict__ eos_ids,
    int eos_len,
    int old_batch,
    int* __restrict__ active_src_rows,
    int* __restrict__ finished_src_rows,
    int* __restrict__ finished_tokens,
    int* __restrict__ counts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int active = 0;
    int finished = 0;
    for (int i = 0; i < old_batch; ++i) {
        const int token = C[i];
        const bool hit_eos = (ignore_eos[i] == 0) && token_is_eos(token, eos_ids, eos_len);
        const bool hit_max = generated_counts[i] + 1 >= max_tokens[i];
        if (hit_eos || hit_max) {
            finished_src_rows[finished] = i;
            finished_tokens[finished] = token;
            ++finished;
        } else {
            A[active] = token;
            active_src_rows[active] = i;
            ++active;
        }
    }
    counts[0] = active;
    counts[1] = finished;
    counts[2] = old_batch;
}

extern "C" void merge_compact_decode(
    int* A,
    const int* C,
    const int* generated_counts,
    const int* max_tokens,
    const int* ignore_eos,
    const int* eos_ids,
    int eos_len,
    int old_batch,
    int* active_src_rows,
    int* finished_src_rows,
    int* finished_tokens,
    int* counts,
    cudaStream_t stream)
{
    if (old_batch <= 0) {
        return;
    }
    merge_compact_decode_kernel<<<1, 1, 0, stream>>>(
        A,
        C,
        generated_counts,
        max_tokens,
        ignore_eos,
        eos_ids,
        eos_len,
        old_batch,
        active_src_rows,
        finished_src_rows,
        finished_tokens,
        counts);
}
