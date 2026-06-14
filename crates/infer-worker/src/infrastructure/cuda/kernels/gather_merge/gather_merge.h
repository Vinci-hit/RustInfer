#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Append newly-admitted decode seed tokens from B into A at
// `A[start + i] = B[i]`. Used before a decode forward when prefill has
// produced first decode tokens that are not yet resident in A.
void append_decode_admissions(
    int* A,
    const int* B,
    int start,
    int count,
    cudaStream_t stream
);

// Merge graph output C into stable A and compact surviving rows.
//
// For each old row i:
//   token = C[i]
//   finished = (!ignore_eos[i] && token in eos_ids)
//              || generated_counts[i] + 1 >= max_tokens[i]
// Non-finished rows are written in scan order to A[active_count++] and their
// source row is written to active_src_rows. Finished rows are written to
// finished_src_rows / finished_tokens. counts = [active, finished, old_batch].
void merge_compact_decode(
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
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
