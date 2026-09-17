#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>
#include <climits>

namespace {
constexpr int Threads = 256, Tile = 2048, MaxK = 512;
struct Candidate { float score; int id; };
static_assert(sizeof(Candidate) == 2 * sizeof(int) && alignof(Candidate) == alignof(int));

__device__ Candidate empty() { return {-CUDART_INF_F, INT_MAX}; }
__device__ bool valid(int start, int tokens, int capacity) {
    return start >= 0 && start <= INT_MAX - (tokens - 1)
        && (static_cast<long long>(start) + tokens) / 4 <= capacity;
}
__device__ bool before(Candidate a, Candidate b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

// Stable 32-bit score radix sort: initial logical order is ascending absolute
// ID, so equal scores retain ascending IDs without sorting 64-bit packed keys.
// Keys are normalized first: NaN/-inf are absent, and both zero signs tie.
template<int Items, bool Direct>
__global__ void select_blocks(const float* __restrict__ scores,
                              const int* __restrict__ start_ptr,
                              Candidate* __restrict__ workspace,
                              int* __restrict__ output,
                              int tokens, int capacity, int k, int parts) {
    const int token = blockIdx.x / parts, part = blockIdx.x % parts;
    const int start = *start_ptr;
    const bool ok = valid(start, tokens, capacity);
    const int visible = ok ? static_cast<int>((static_cast<long long>(start) + token + 1) / 4) : 0;
    const int base = part * Tile;
    const size_t dst = (static_cast<size_t>(token) * parts + part) * k;
    if (base >= visible) {
        for (int j = threadIdx.x; j < k; j += Threads) {
            if constexpr (Direct) output[static_cast<size_t>(token) * k + j] = -1;
            else workspace[dst + j] = empty();
        }
        return;
    }
    float values[Items]; int ids[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int id = base + threadIdx.x * Items + i;
        const float value = id < visible ? scores[static_cast<size_t>(token) * capacity + id]
                                         : -CUDART_INF_F;
        const bool present = value > -CUDART_INF_F; // false for NaN and -inf
        values[i] = present ? (value == 0.0f ? 0.0f : value) : -CUDART_INF_F;
        ids[i] = present ? id : INT_MAX;
    }
    using Sort = cub::BlockRadixSort<float, Threads, Items, int>;
    __shared__ typename Sort::TempStorage temp;
    Sort(temp).SortDescendingBlockedToStriped(values, ids);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int rank = i * Threads + threadIdx.x;
        if (rank < k) {
            if constexpr (Direct)
                output[static_cast<size_t>(token) * k + rank] = ids[i] == INT_MAX ? -1 : ids[i];
            else workspace[dst + rank] = {values[i], ids[i]};
        }
    }
}

// Merge two sorted local top-k lists. An entry's rank is its rank within its
// list plus the number of preceding entries in the other list. Independent
// binary searches give unique output ranks; ties prefer the left sentinel.
// Dropping everything at rank>=k is exact: a discarded local item already
// has at least k preceding items, and can never enter the global top-k.
__global__ void merge(const Candidate* __restrict__ input,
                      Candidate* __restrict__ output,
                      int* __restrict__ indices,
                      int k, int parts, int out_parts, bool final) {
    const int token = blockIdx.x / out_parts, pair = blockIdx.x % out_parts;
    __shared__ Candidate items[2 * MaxK];
    for (int j = threadIdx.x; j < 2 * k; j += Threads) {
        const int part = 2 * pair + j / k;
        items[j] = part < parts ? input[(static_cast<size_t>(token) * parts + part) * k + j % k]
                                : empty();
    }
    __syncthreads();
    for (int j = threadIdx.x; j < 2 * k; j += Threads) {
        const bool right = j >= k;
        const Candidate current = items[j];
        const Candidate* other = items + (right ? 0 : k);
        int lo = 0, hi = k;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            const bool precedes = right ? !before(current, other[mid]) : before(other[mid], current);
            if (precedes) lo = mid + 1;
            else hi = mid;
        }
        const int rank = j % k + lo;
        if (rank < k) {
            if (final)
                indices[static_cast<size_t>(token) * k + rank] = current.id == INT_MAX ? -1 : current.id;
            else output[(static_cast<size_t>(token) * out_parts + pair) * k + rank] = current;
        }
    }
}
} // namespace

extern "C" int rustinfer_v4_indexer_topk(
    const float* scores, const int* start, int* workspace, int* indices,
    int tokens, int capacity, int k, cudaStream_t stream) {
    const int parts = (capacity + Tile - 1) / Tile;
    auto* bank0 = reinterpret_cast<Candidate*>(workspace);
    if (parts == 1) {
        const int count = capacity > k ? capacity : k;
        if (count <= 256)
            select_blocks<1, true><<<tokens, Threads, 0, stream>>>(scores, start, bank0, indices, tokens, capacity, k, 1);
        else if (count <= 512)
            select_blocks<2, true><<<tokens, Threads, 0, stream>>>(scores, start, bank0, indices, tokens, capacity, k, 1);
        else if (count <= 1024)
            select_blocks<4, true><<<tokens, Threads, 0, stream>>>(scores, start, bank0, indices, tokens, capacity, k, 1);
        else
            select_blocks<8, true><<<tokens, Threads, 0, stream>>>(scores, start, bank0, indices, tokens, capacity, k, 1);
        return static_cast<int>(cudaGetLastError());
    }
    select_blocks<8, false><<<static_cast<unsigned>(static_cast<size_t>(tokens) * parts), Threads, 0, stream>>>(
        scores, start, bank0, indices, tokens, capacity, k, parts);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
    const size_t bank_size = static_cast<size_t>(tokens) * parts * k;
    Candidate* source = bank0;
    Candidate* dest = bank0 + bank_size;
    for (int count = parts; count > 1; count = (count + 1) / 2) {
        const int next = (count + 1) / 2;
        merge<<<static_cast<unsigned>(static_cast<size_t>(tokens) * next), Threads, 0, stream>>>(
            source, dest, indices, k, count, next, next == 1);
        status = cudaGetLastError();
        if (status != cudaSuccess) return static_cast<int>(status);
        Candidate* swap = source; source = dest; dest = swap;
    }
    return static_cast<int>(status);
}
