#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pipelined-decode input assembly.
//
//   A[i] = (src[i] >= 0) ? C[src[i]]        // continuing seq: previous step's argmax
//                        : B[-src[i] - 1];  // newly-admitted seq: CPU-uploaded first token
//
// All buffers are device i32 of length `batch`. This is the single
// dependency-merge point of the bubble-free decode pipeline: it folds the
// previous step's on-device output (C) and the freshly-uploaded admit tokens
// (B) into the next step's input ids (A) without any host round-trip.
//
//   A   : out, next step input_ids   [batch]
//   C   : in,  previous argmax output [>= max(src)+1]
//   B   : in,  new-admit tokens       [>= number of negative src entries]
//   src : in,  per-row source selector [batch]
void gather_merge_input(
    int* A,
    const int* C,
    const int* B,
    const int* src,
    int batch,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
