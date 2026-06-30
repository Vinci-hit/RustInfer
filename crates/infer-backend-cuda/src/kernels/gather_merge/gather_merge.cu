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

constexpr int MERGE_MAX_ROWS = 1024;

static int merge_threads_for_rows(int rows)
{
    int threads = 32;
    while (threads < rows && threads < MERGE_MAX_ROWS) {
        threads <<= 1;
    }
    return threads;
}

// Fallback for unusually large batches. Normal serving batches use the
// single-block prefix-scan kernels below.
__global__ void merge_compact_decode_serial_kernel(
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
    __shared__ int active_scan[MERGE_MAX_ROWS];
    __shared__ int finished_scan[MERGE_MAX_ROWS];

    const int i = threadIdx.x;
    int active_flag = 0;
    int finished_flag = 0;
    int token = 0;
    if (i < old_batch) {
        token = C[i];
        const bool hit_eos = (ignore_eos[i] == 0) && token_is_eos(token, eos_ids, eos_len);
        const bool hit_max = generated_counts[i] + 1 >= max_tokens[i];
        finished_flag = (hit_eos || hit_max) ? 1 : 0;
        active_flag = finished_flag ? 0 : 1;
    }
    active_scan[i] = active_flag;
    finished_scan[i] = finished_flag;
    __syncthreads();

    for (int offset = 1; offset < old_batch; offset <<= 1) {
        int active_add = 0;
        int finished_add = 0;
        if (i < old_batch && i >= offset) {
            active_add = active_scan[i - offset];
            finished_add = finished_scan[i - offset];
        }
        __syncthreads();
        if (i < old_batch) {
            active_scan[i] += active_add;
            finished_scan[i] += finished_add;
        }
        __syncthreads();
    }

    if (i < old_batch) {
        if (active_flag) {
            const int dst = active_scan[i] - 1;
            A[dst] = token;
            active_src_rows[dst] = i;
        } else {
            const int dst = finished_scan[i] - 1;
            finished_src_rows[dst] = i;
            finished_tokens[dst] = token;
        }
    }
    if (i == 0) {
        counts[0] = active_scan[old_batch - 1];
        counts[1] = finished_scan[old_batch - 1];
        counts[2] = old_batch;
    }
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
    if (old_batch > MERGE_MAX_ROWS) {
        merge_compact_decode_serial_kernel<<<1, 1, 0, stream>>>(
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
        return;
    }
    const int threads = merge_threads_for_rows(old_batch);
    merge_compact_decode_kernel<<<1, threads, 0, stream>>>(
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

enum MixedRowKind {
    ROW_DECODE = 0,
    ROW_PREFILL_FINAL = 1,
    ROW_PREFILL_CONT = 2,
    ROW_PAD = 3,
};

__global__ void merge_compact_mixed_serial_kernel(
    int* __restrict__ A,
    const int* __restrict__ C,
    const int* __restrict__ row_kind,
    const int* __restrict__ generated_counts,
    const int* __restrict__ max_tokens,
    const int* __restrict__ ignore_eos,
    const int* __restrict__ eos_ids,
    int eos_len,
    int old_rows,
    int* __restrict__ active_src_rows,
    int* __restrict__ active_tokens,
    int* __restrict__ finished_src_rows,
    int* __restrict__ finished_tokens,
    int* __restrict__ prefill_final_src_rows,
    int* __restrict__ prefill_final_tokens,
    int* __restrict__ counts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int active = 0;
    int finished = 0;
    int prefill_final = 0;
    for (int i = 0; i < old_rows; ++i) {
        const int kind = row_kind[i];
        if (kind == ROW_PAD || kind == ROW_PREFILL_CONT) {
            continue;
        }
        const int token = C[i];
        const bool hit_eos = (ignore_eos[i] == 0) && token_is_eos(token, eos_ids, eos_len);
        const bool hit_max = generated_counts[i] + 1 >= max_tokens[i];
        const bool done = hit_eos || hit_max;

        if (kind == ROW_PREFILL_FINAL) {
            prefill_final_src_rows[prefill_final] = i;
            prefill_final_tokens[prefill_final] = token;
            ++prefill_final;
        }

        if (done) {
            finished_src_rows[finished] = i;
            finished_tokens[finished] = token;
            ++finished;
        } else {
            A[active] = token;
            active_src_rows[active] = i;
            active_tokens[active] = token;
            ++active;
        }
    }
    counts[0] = active;
    counts[1] = finished;
    counts[2] = prefill_final;
    counts[3] = old_rows;
}

__global__ void merge_compact_mixed_kernel(
    int* __restrict__ A,
    const int* __restrict__ C,
    const int* __restrict__ row_kind,
    const int* __restrict__ generated_counts,
    const int* __restrict__ max_tokens,
    const int* __restrict__ ignore_eos,
    const int* __restrict__ eos_ids,
    int eos_len,
    int old_rows,
    int* __restrict__ active_src_rows,
    int* __restrict__ active_tokens,
    int* __restrict__ finished_src_rows,
    int* __restrict__ finished_tokens,
    int* __restrict__ prefill_final_src_rows,
    int* __restrict__ prefill_final_tokens,
    int* __restrict__ counts)
{
    __shared__ int active_scan[MERGE_MAX_ROWS];
    __shared__ int finished_scan[MERGE_MAX_ROWS];
    __shared__ int prefill_final_scan[MERGE_MAX_ROWS];

    const int i = threadIdx.x;
    int active_flag = 0;
    int finished_flag = 0;
    int prefill_final_flag = 0;
    int token = 0;
    if (i < old_rows) {
        const int kind = row_kind[i];
        const bool emits = kind != ROW_PAD && kind != ROW_PREFILL_CONT;
        prefill_final_flag = kind == ROW_PREFILL_FINAL ? 1 : 0;
        if (emits) {
            token = C[i];
            const bool hit_eos = (ignore_eos[i] == 0) && token_is_eos(token, eos_ids, eos_len);
            const bool hit_max = generated_counts[i] + 1 >= max_tokens[i];
            finished_flag = (hit_eos || hit_max) ? 1 : 0;
            active_flag = finished_flag ? 0 : 1;
        }
    }
    active_scan[i] = active_flag;
    finished_scan[i] = finished_flag;
    prefill_final_scan[i] = prefill_final_flag;
    __syncthreads();

    for (int offset = 1; offset < old_rows; offset <<= 1) {
        int active_add = 0;
        int finished_add = 0;
        int prefill_final_add = 0;
        if (i < old_rows && i >= offset) {
            active_add = active_scan[i - offset];
            finished_add = finished_scan[i - offset];
            prefill_final_add = prefill_final_scan[i - offset];
        }
        __syncthreads();
        if (i < old_rows) {
            active_scan[i] += active_add;
            finished_scan[i] += finished_add;
            prefill_final_scan[i] += prefill_final_add;
        }
        __syncthreads();
    }

    if (i < old_rows) {
        if (prefill_final_flag) {
            const int dst = prefill_final_scan[i] - 1;
            prefill_final_src_rows[dst] = i;
            prefill_final_tokens[dst] = token;
        }
        if (active_flag) {
            const int dst = active_scan[i] - 1;
            A[dst] = token;
            active_src_rows[dst] = i;
            active_tokens[dst] = token;
        } else if (finished_flag) {
            const int dst = finished_scan[i] - 1;
            finished_src_rows[dst] = i;
            finished_tokens[dst] = token;
        }
    }
    if (i == 0) {
        counts[0] = active_scan[old_rows - 1];
        counts[1] = finished_scan[old_rows - 1];
        counts[2] = prefill_final_scan[old_rows - 1];
        counts[3] = old_rows;
    }
}

extern "C" void merge_compact_mixed(
    int* A,
    const int* C,
    const int* row_kind,
    const int* generated_counts,
    const int* max_tokens,
    const int* ignore_eos,
    const int* eos_ids,
    int eos_len,
    int old_rows,
    int* active_src_rows,
    int* active_tokens,
    int* finished_src_rows,
    int* finished_tokens,
    int* prefill_final_src_rows,
    int* prefill_final_tokens,
    int* counts,
    cudaStream_t stream)
{
    if (old_rows <= 0) {
        return;
    }
    if (old_rows > MERGE_MAX_ROWS) {
        merge_compact_mixed_serial_kernel<<<1, 1, 0, stream>>>(
            A,
            C,
            row_kind,
            generated_counts,
            max_tokens,
            ignore_eos,
            eos_ids,
            eos_len,
            old_rows,
            active_src_rows,
            active_tokens,
            finished_src_rows,
            finished_tokens,
            prefill_final_src_rows,
            prefill_final_tokens,
            counts);
        return;
    }
    const int threads = merge_threads_for_rows(old_rows);
    merge_compact_mixed_kernel<<<1, threads, 0, stream>>>(
        A,
        C,
        row_kind,
        generated_counts,
        max_tokens,
        ignore_eos,
        eos_ids,
        eos_len,
        old_rows,
        active_src_rows,
        active_tokens,
        finished_src_rows,
        finished_tokens,
        prefill_final_src_rows,
        prefill_final_tokens,
        counts);
}

// ── Device-resident decode control plane (async path) ────────────────────────
//
// After `merge_compact_decode` writes `active_src_rows` + `counts`, build the
// NEXT decode step's per-row control buffers entirely on-device: gather each
// surviving row's block table to the compacted front, append its next-step KV
// slot, advance length/position, and rebuild the (trivial) decode tile layout.
// Phantom tail rows are zeroed so they stay inert (cf. zero-pad invariant).
//
// This replaces the per-step host rebuild (`build_decode_request`) + H2D upload
// (`upload_index`) — O(batch * seq_len) host work — with O(batch) device work,
// the dominant high-QPS decode-step tail cost. Output block tables / kv_lens are
// written to scratch buffers (gather reads the live buffers; an in-place gather
// would race), then copied back to the live buffers by the caller.
//
// One block per output row; threads in a block stripe the block-table gather.
__global__ void compact_extend_control_kernel(
    const int* __restrict__ block_tables_in,   // [cap_batch, mbps] this step's order
    int*       __restrict__ block_tables_out,   // [cap_batch, mbps] scratch -> next step
    const int* __restrict__ kv_lens_in,         // [cap_batch] this step: length AFTER write
    int*       __restrict__ kv_lens_out,        // [cap_batch] scratch -> next step
    int*       __restrict__ seq_positions_out,  // [cap_batch] next step
    int*       __restrict__ seq_lens_step_out,  // [cap_batch] next step (q_len = 1)
    int*       __restrict__ rope_positions_out, // [cap_batch] next step (1 pos/row)
    int*       __restrict__ cu_q_lens_out,      // [cap_batch + 1]
    int*       __restrict__ block2req_out,      // [cap_batch]
    int*       __restrict__ block2tile_out,     // [cap_batch]
    const int* __restrict__ active_src_rows,    // [active]
    const int* __restrict__ counts,             // [active, finished, old]
    const int* __restrict__ new_slots,          // [cap_batch] next-step KV slot per output row
    int mbps,
    int cap_batch)
{
    const int active = counts[0];
    const int r = blockIdx.x;          // output row
    if (r >= cap_batch) return;
    const int t = threadIdx.x;

    if (r == 0 && t == 0) {
        cu_q_lens_out[0] = 0;          // prefix-sum base
    }

    if (r >= active) {
        // Phantom tail: inert (length 0, empty q range).
        if (t == 0) {
            kv_lens_out[r]        = 0;
            seq_positions_out[r]  = 0;
            seq_lens_step_out[r]  = 0;
            rope_positions_out[r] = 0;
            block2req_out[r]      = 0;
            block2tile_out[r]     = 0;
            cu_q_lens_out[r + 1]  = 0;
        }
        return;
    }

    const int src = active_src_rows[r];
    const int M   = kv_lens_in[src];   // entries already present: indices 0..M-1
    // Gather block_table[src][0..M] -> out[r][0..M].
    for (int j = t; j < M; j += blockDim.x) {
        block_tables_out[r * mbps + j] = block_tables_in[src * mbps + j];
    }
    if (t == 0) {
        block_tables_out[r * mbps + M] = new_slots[r]; // next-step write slot at index M
        kv_lens_out[r]        = M + 1;
        seq_positions_out[r]  = M;
        rope_positions_out[r] = M;
        seq_lens_step_out[r]  = 1;
        cu_q_lens_out[r + 1]  = r + 1;  // prefix sum of all-ones q_lens
        block2req_out[r]      = r;
        block2tile_out[r]     = 0;
    }
}

// `block_tables` / `kv_lens` are the LIVE buffers (read by the gather, then
// overwritten by the copy-back below). `*_scratch` are the gather targets — an
// in-place gather races because output row r may read source row r' > r whose
// data another block is concurrently overwriting. All other outputs are pure
// (computed from `kv_lens[src]`), so they write the live buffers directly.
extern "C" void compact_extend_control(
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
    cudaStream_t stream)
{
    if (cap_batch <= 0) {
        return;
    }
    const int threads = 256;
    compact_extend_control_kernel<<<cap_batch, threads, 0, stream>>>(
        block_tables,          // in (live)
        block_tables_scratch,  // out (scratch)
        kv_lens,               // in (live)
        kv_lens_scratch,       // out (scratch)
        seq_positions_out,
        seq_lens_step_out,
        rope_positions_out,
        cu_q_lens_out,
        block2req_out,
        block2tile_out,
        active_src_rows,
        counts,
        new_slots,
        mbps,
        cap_batch);
    // Copy the gathered scratch back into the live buffers (same stream → ordered
    // after the kernel). Full-capacity copy; phantom-tail rows are never read.
    cudaMemcpyAsync(block_tables, block_tables_scratch,
                    (size_t)cap_batch * (size_t)mbps * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(kv_lens, kv_lens_scratch,
                    (size_t)cap_batch * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);
}
