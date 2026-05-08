# FlashAttnGQA Implementation Analysis: Batch Support Planning

## Executive Summary

The current FlashAttnGQA implementation is **strictly single-batch**, processing one sequence at a time with a fixed K/V cache index. To support variable-length batching (like `flash_attn_varlen`), **major architectural changes** would be needed across the Rust Op interface, CUDA kernel signatures, and kernel implementations.

---

## Current Implementation Architecture

### 1. Rust Op Structure (`flash_gqa.rs`)

**Op Definition:**
```rust
pub struct FlashAttnGQA {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
}
```

**Forward Signature:**
```rust
pub fn forward(
    &self,
    input_q: &Tensor,                    // [Q_SeqLen, Q_HiddenDim]
    input_k_cache: &Tensor,              // [Max_SeqLen, KV_HiddenDim]
    input_v_cache: &Tensor,              // [Max_SeqLen, KV_HiddenDim]
    input_kv_len: &Tensor,               // [1] - scalar KV cache length
    output_o: &mut Tensor,               // [Q_SeqLen, Q_HiddenDim]
    cuda_config: Option<&OpConfig>,
) -> Result<()>
```

### 2. Input/Output Tensor Shapes (Current)

| Tensor | Shape | Interpretation |
|--------|-------|-----------------|
| `input_q` | `[Q_SeqLen, Q_HiddenDim]` | Single sequence query (flattened heads) |
| `input_k_cache` | `[Max_SeqLen, KV_HiddenDim]` | Full K cache buffer, only first `kv_len` rows valid |
| `input_v_cache` | `[Max_SeqLen, KV_HiddenDim]` | Full V cache buffer, only first `kv_len` rows valid |
| `input_kv_len` | `[1]` or scalar | Pointer to **single i32** on device memory |
| `output_o` | `[Q_SeqLen, Q_HiddenDim]` | Single sequence output |

**Where:**
- `Q_HiddenDim = num_q_heads * head_dim`
- `KV_HiddenDim = num_kv_heads * head_dim`
- `Max_SeqLen` = allocated cache size (e.g., 4096)
- `Q_SeqLen` = query length for this forward pass (1 for decode, variable for prefill)

### 3. CUDA FFI Layer (`kernels/cuda/flash_attn_gqa/mod.rs`)

**Core FFI Signatures:**

```rust
unsafe extern "C" {
    pub fn flash_attn_gqa_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        q_seq_len: i32,                  // Length of query sequence
        kv_seq_len: *const i32,          // Pointer to device memory with KV history length
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        is_causal: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    
    pub fn flash_decoding_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        q_seq_len: i32,                  // Always 1
        kv_seq_len: *const i32,          // Pointer to device memory
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        stream: cuda::ffi::cudaStream_t,
    );
    
    // BF16 variants with multiple tile sizes
    pub fn flash_decoding_cu_bf16(...);
    pub fn flash_decoding_cu_bf16_hdim128(...);
    pub fn launch_flash_attn_cute_128x64x64_tile(...);
    pub fn launch_flash_attn_cute_bf16_hdim128(...);
    // F16 variants too
}
```

**Key Observation:** `kv_seq_len` is **a pointer to a single i32**, not an array. The kernel reads via dereference:
```cuda
int kv_seq_len = *kv_seq_len_ptr;
```

### 4. CUDA Kernel Implementations

#### A. Decode Kernel (`flash_decoding.cu`)

```cuda
__global__ void simple_gqa_decoding_native_layout_kernel(
    const float* __restrict__ Q,       
    const float* __restrict__ K_cache, 
    const float* __restrict__ V_cache, 
    float* __restrict__ Output,        
    int* kv_seq_len_ptr,               // Pointer to single i32
    int head_dim,
    int num_kv_heads,                  
    int group_size,
    float sm_scale
)
```

**Grid Layout:**
- `grid.x = num_q_heads` — one block per Q head
- `block.x = (head_dim + 31) / 32 * 32` — threads for head_dim elements

**Execution Model:**
- Each block processes 1 query head (q_seq_len=1 assumed)
- Accesses layout: `Q[q_head_idx * head_dim + tid]`
- Loops over all KV tokens in cache: `for t in 0..kv_seq_len`

**Tensor Access Pattern:**
```cuda
int kv_seq_len = *kv_seq_len_ptr + 1;  // Add 1 for current token
int q_head_idx = blockIdx.x;           // One block per Q head
int tid = threadIdx.x;                 // Thread for elements within head_dim

float q_val = Q[q_head_idx * head_dim + tid];
for (int t = 0; t < kv_seq_len; ++t) {
    int curr_kv_idx = t * stride_kv_seq + current_head_offset;
    float k_val = K_cache[curr_kv_idx];
    // Compute attention per token
}
```

#### B. Prefill Kernel (`flash_attn_gqa.cu`)

```cuda
template<const int THREADS_PER_BLOCK = 128, const int Br = 16>
__global__ void flash_attn_gqa_kernel(
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    float* __restrict__ o_ptr,
    int Tc,
    const int tile_offset,
    const int tile_size_inhead,
    const int kv_total_len,
    const float scale,
    int q_seq_len,
    int kv_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int is_causal
)
```

**Grid Layout:**
- `grid.x = ceil(q_seq_len / Br)` — blocks over query positions
- `grid.y = num_q_heads` — one layer per Q head
- Br = tile size in query dimension (typically 16)

**Execution Model:**
- Tile-based: processes `Br` query tokens and `Tc` key/value tokens at a time
- Causal masking supported
- Shared memory: Q, K, V tiles + online softmax state

### 5. CPU Reference Implementation (`kernels/cpu/flash_attn_gqa.rs`)

```rust
pub fn flash_attn_gqa(
    input_q: &Tensor,                   // [Q_SeqLen, Q_HiddenDim]
    input_k_cache: &Tensor,             // [Max_SeqLen, KV_HiddenDim]
    input_v_cache: &Tensor,             // [Max_SeqLen, KV_HiddenDim]
    output_o: &mut Tensor,              // [Q_SeqLen, Q_HiddenDim]
    q_seq_len: usize,
    current_kv_len: usize,              // Existing KV history length
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_causal: bool,
) -> Result<()>
```

**Key Detail:** Uses ndarray with shape `[Q_SeqLen, num_q_heads, head_dim]` for parallel iteration over heads. **No batch dimension**.

---

## What's Currently NOT Supported

### 1. **No Batch Dimension**
- No parameter for batch size
- No cu_seqlens (cumulative sequence lengths) for variable-length sequences
- All sequences must have same length or be processed separately

### 2. **Single KV Length Pointer**
- `input_kv_len` is `[1]` — one scalar for **all** sequences
- Flash-attn varlen requires `cu_seqlens_q` and `cu_seqlens_k` arrays

### 3. **Single Output Tensor**
- Output is `[Q_SeqLen, Hidden]`
- Would need to become `[Batch, Q_SeqLen, Hidden]` or use row offsets

### 4. **No Packed Sequences Support**
- Current layout assumes dense 2D matrices
- Varlen requires "packed" format: all sequences concatenated with cumulative indices

### 5. **No Dropout or Attention Weights**
- No option to return attention weights
- No training mode with dropout

---

## Changes Required for Batch Support

### A. Rust Op Interface Changes

**New signature (proposed):**
```rust
pub fn forward_batched(
    &self,
    input_q: &Tensor,                   // [TotalQ, Q_HiddenDim] (packed)
    input_k_cache: &Tensor,             // [TotalKV, KV_HiddenDim] (packed)
    input_v_cache: &Tensor,             // [TotalKV, KV_HiddenDim] (packed)
    cu_seqlens_q: &Tensor,              // [Batch + 1] cumulative query lengths
    cu_seqlens_k: &Tensor,              // [Batch + 1] cumulative KV lengths
    output_o: &mut Tensor,              // [TotalQ, Q_HiddenDim] (packed)
    cuda_config: Option<&OpConfig>,
) -> Result<()>
```

### B. CUDA FFI Changes

**New kernel signatures:**
```rust
unsafe extern "C" {
    pub fn flash_attn_gqa_varlen_cu(
        q_ptr: *const f32,
        k_ptr: *const f32,
        v_ptr: *const f32,
        o_ptr: *mut f32,
        cu_seqlens_q: *const i32,       // [Batch + 1]
        cu_seqlens_k: *const i32,       // [Batch + 1]
        max_seqlen_q: i32,
        max_seqlen_k: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        is_causal: i32,
        batch_size: i32,
        stream: cuda::ffi::cudaStream_t,
    );
}
```

### C. CUDA Kernel Changes

**Key changes:**
1. **Grid calculation:**
   ```cuda
   // Instead of:
   // grid.x = num_q_heads;
   
   // New batched version:
   // grid.x = num_q_heads * batch_size;  // Or use dynamic grids
   ```

2. **Sequence lookup:**
   ```cuda
   // Instead of:
   int q_head_idx = blockIdx.x;
   
   // New batched version:
   int batch_id = blockIdx.x % batch_size;
   int q_head_idx = blockIdx.x / batch_size;
   
   int q_seq_start = cu_seqlens_q[batch_id];
   int q_seq_end = cu_seqlens_q[batch_id + 1];
   int q_seq_len = q_seq_end - q_seq_start;
   
   int kv_seq_start = cu_seqlens_k[batch_id];
   int kv_seq_end = cu_seqlens_k[batch_id + 1];
   int kv_seq_len = kv_seq_end - kv_seq_start;
   ```

3. **Memory access:**
   ```cuda
   // Instead of:
   // Q[q_head_idx * head_dim + tid]
   
   // New batched version:
   // Q[(q_seq_start + q_token_id) * num_q_heads * head_dim + q_head_idx * head_dim + tid]
   ```

### D. CPU Implementation Changes

```rust
// Current: processes single sequence with parallel heads
// New: process all sequences, some in parallel (batch loop + head parallelization)

for batch_idx in 0..batch_size {
    let q_start = cu_seqlens_q[batch_idx];
    let q_end = cu_seqlens_q[batch_idx + 1];
    let kv_start = cu_seqlens_k[batch_idx];
    let kv_end = cu_seqlens_k[batch_idx + 1];
    
    // Parallel over heads (existing code)
    for q_head in 0..num_q_heads {
        // Compute attention for this head and batch
    }
}
```

---

## Current Tensor Layout Details

### Flattened Layout (Current)
```
Q: [Q_SeqLen * num_q_heads * head_dim] bytes
   Logical shape: [Q_SeqLen, num_q_heads, head_dim]
   Access: Q[seq_idx * (num_q_heads * head_dim) + head_idx * head_dim + elem_idx]

K Cache: [Max_SeqLen * num_kv_heads * head_dim] bytes
   Same layout, only first `kv_seq_len` rows are valid

Output: [Q_SeqLen * num_q_heads * head_dim] bytes
```

### Would Need to Support (for varlen batching)
```
Packed Q: [TotalQ * num_q_heads * head_dim] bytes
   cu_seqlens_q: [batch_size + 1]
   Logical: concatenated sequences from multiple batches
   
Access for batch b, seq s, head h, elem e:
   Q[(cu_seqlens_q[b] + s) * num_q_heads * head_dim + h * head_dim + e]
```

---

## Key Parameters Summary

| Parameter | Type | Current Usage | Batch Version |
|-----------|------|----------------|---------------|
| `num_q_heads` | int | Grid/block layout | Same (per head still) |
| `num_kv_heads` | int | Head grouping | Same |
| `head_dim` | int | Shared memory, block size | Same |
| `q_seq_len` | i32 | Query sequence length (1 or variable) | Max_seqlen_q |
| `kv_seq_len` | i32* | Pointer to single length | Changed to cu_seqlens_k array |
| `is_causal` | i32 | Causal masking flag | Same |
| `batch_size` | N/A | Not present | Would be added |
| `cu_seqlens_q` | N/A | Not present | [batch_size + 1] |
| `cu_seqlens_k` | N/A | Not present | [batch_size + 1] |

---

## Complexity Assessment

| Aspect | Current | Required for Batching | Complexity |
|--------|---------|----------------------|------------|
| Rust Op | Single-seq | Batched calls | **Medium** — New method or param |
| CUDA FFI | Single-seq pointer | Arrays + batch ID | **High** — New signatures needed |
| Decode kernel | ~150 lines | ~200-250 lines | **Medium** — Extra loop for batch |
| Prefill kernel | ~300-400 lines | ~500-600 lines | **High** — Grid changes, indexing |
| Tests | Single-seq | Multi-seq | **Medium** — Add batch test cases |

---

## Recommendation for Implementation Order

1. **Phase 1:** Add new CUDA kernels for varlen batching without removing old ones (backward compat)
2. **Phase 2:** Add `forward_batched()` method to Op alongside existing `forward()`
3. **Phase 3:** Update CPU reference implementation
4. **Phase 4:** Add comprehensive tests with various batch sizes and sequence lengths
5. **Phase 5:** Deprecate single-sequence path if desired

---

## References to Key Code Locations

- **Op definition:** `/data/home/vinciiliu/RustInfer/crates/infer-worker/src/op/flash_gqa.rs` lines 9-106
- **CUDA FFI wrapper:** `/data/home/vinciiliu/RustInfer/crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/mod.rs` lines 8-376
- **Decode kernel:** `/data/home/vinciiliu/RustInfer/crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/flash_decoding.cu` lines 46-108
- **Prefill kernel:** `/data/home/vinciiliu/RustInfer/crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/flash_attn_gqa.cu` lines 35-150+
- **CPU reference:** `/data/home/vinciiliu/RustInfer/crates/infer-worker/src/op/kernels/cpu/flash_attn_gqa.rs` lines 19-189
