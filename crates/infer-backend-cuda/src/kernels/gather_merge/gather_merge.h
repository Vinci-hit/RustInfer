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

// Build the NEXT decode step's control plane on-device from this step's
// survivors (after merge_compact_decode). Gathers block tables/kv_lens to the
// compacted front, appends each row's next-step KV slot, advances
// position/length, rebuilds the decode tile layout, and zeroes the phantom
// tail. block_tables_out / kv_lens_out are scratch (gather cannot be in-place);
// the caller copies them back into the live buffers. Replaces the per-step host
// block-table rebuild + upload with O(batch) device work.
void compact_extend_control(
    int* block_tables,
    int* block_tables_scratch,
    int* kv_lens,
    int* kv_lens_scratch,
    int* seq_positions_out,
    int* seq_lens_step_out,
    int* rope_positions_out,
    int* cu_q_lens_out,
    int* block2req_out,
    int* block2tile_out,
    const int* active_src_rows,
    const int* counts,
    const int* new_slots,
    int mbps,
    int cap_batch,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
