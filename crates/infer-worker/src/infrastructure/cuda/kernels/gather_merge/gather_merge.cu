#include "gather_merge.h"

// Single-dependency-merge kernel for the bubble-free decode pipeline.
//
//   A[i] = (src[i] >= 0) ? C[src[i]] : B[-src[i] - 1]
//
// `batch` is the number of active sequences this step (≤ cap_batch), so a
// single small launch suffices. One thread per row.
__global__ void gather_merge_input_kernel(
    int* __restrict__ A,
    const int* __restrict__ C,
    const int* __restrict__ B,
    const int* __restrict__ src,
    int batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    int s = src[i];
    A[i] = (s >= 0) ? C[s] : B[-s - 1];
}

extern "C" void gather_merge_input(
    int* A,
    const int* C,
    const int* B,
    const int* src,
    int batch,
    cudaStream_t stream)
{
    if (batch <= 0) return;
    const int threads = 256;
    const int blocks = (batch + threads - 1) / threads;
    gather_merge_input_kernel<<<blocks, threads, 0, stream>>>(A, C, B, src, batch);
}
