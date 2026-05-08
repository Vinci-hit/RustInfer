# FlashAttnGQA Batch Support: Detailed Architecture Guide

## Current vs. Proposed Architecture

### Current (Single-Sequence) Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Rust Op Layer                               │
│  FlashAttnGQA::forward(input_q, input_k_cache, ...)         │
│  Input Shapes: [Q_Seq, Q_Dim] × [Max_Seq, KV_Dim]           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              CUDA FFI Layer (Rust)                           │
│  flash_attn_gqa_cu(q_ptr, k_ptr, v_ptr,                     │
│                    q_seq_len, *kv_seq_len, ...)             │
│  (kv_seq_len is single i32 pointer)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            CUDA Kernels (C/C++)                              │
│  ┌──────────────────┐       ┌──────────────────┐             │
│  │ Decode Kernel    │       │ Prefill Kernel   │             │
│  │ (q_seq_len = 1)  │       │ (q_seq_len > 1)  │             │
│  │ Grid: [N_Heads]  │       │ Grid: [Heads ×   │             │
│  │                  │       │   QSeq/Br]       │             │
│  └──────────────────┘       └──────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

**Limitations:**
- Can only process ONE sequence per forward call
- All threads work on the same KV cache index
- No built-in batch dimension
- Each sequence requires separate kernel launch

---

### Proposed (Batched Variable-Length) Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Rust Op Layer (New)                        │
│  FlashAttnGQA::forward_batched(                               │
│      input_q,              // [Total_Q, Q_Dim] (packed)       │
│      cu_seqlens_q,         // [Batch+1]                       │
│      cu_seqlens_k,         // [Batch+1]                       │
│      ...                                                      │
│  )                                                            │
│  Supports: Multiple sequences of varying lengths             │
└───────────────────┬────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│            CUDA FFI Layer (Rust) - New Signatures            │
│  flash_attn_gqa_varlen_cu(                                    │
│      q_ptr, k_ptr, v_ptr,                                    │
│      *cu_seqlens_q,        // Array of Batch+1               │
│      *cu_seqlens_k,        // Array of Batch+1               │
│      max_seqlen_q,         // Max query sequence length       │
│      max_seqlen_k,         // Max KV sequence length          │
│      batch_size,           // New parameter                   │
│      ...                                                      │
│  )                                                            │
└───────────────────┬────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│          CUDA Kernels (C/C++) - Batched Versions             │
│  ┌──────────────────────┐  ┌──────────────────────┐          │
│  │ Decode Kernel Varlen │  │ Prefill Kernel Varlen│          │
│  │ Grid: [Heads ×       │  │ Grid: [Heads ×       │          │
│  │    Batch]            │  │    Batch × QSeq/Br]  │          │
│  │                      │  │                      │          │
│  │ Lookup: cu_seqlens   │  │ Lookup: cu_seqlens   │          │
│  │ per batch item       │  │ per batch item       │          │
│  └──────────────────────┘  └──────────────────────┘          │
│                                                               │
│  Benefits:                                                    │
│  - All sequences processed in single launch                  │
│  - Variable-length support                                   │
│  - Better SM utilization with batch                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Memory Layout Transformations

### Current Layout (Single Sequence)

```
INPUT TENSORS:
┌─────────────────────────────────────────────────────────────┐
│ Q: [Q_SeqLen, Q_Dim]  where Q_Dim = N_Heads × Head_Dim      │
├─────────────────────────────────────────────────────────────┤
│ Row 0: [Head0_D0...Head0_Dn | Head1_D0...Head1_Dn | ...]    │
│ Row 1: [Head0_D0...Head0_Dn | Head1_D0...Head1_Dn | ...]    │
│ ...                                                          │
│ Row N-1: ...                                                │
└─────────────────────────────────────────────────────────────┘

K_Cache: [Max_Seq, KV_Dim]  ← Only first kv_len rows valid
V_Cache: [Max_Seq, KV_Dim]  ← Only first kv_len rows valid

OUTPUT:
O: [Q_SeqLen, Q_Dim]  ← Same shape as Q
```

**Memory Address Pattern (1 sequence):**
```
Q[seq_idx][elem_idx] → Q_buffer[seq_idx * Q_Dim + elem_idx]
Q[seq_idx][head][dim] → Q_buffer[seq_idx * N_Heads * Head_Dim 
                                 + head * Head_Dim + dim]
```

---

### Proposed Layout (Batched Variable-Length)

```
INPUT TENSORS:
┌────────────────────────────────────────────────────────────────┐
│ Q: [Total_Q, Q_Dim]  (All sequences concatenated)              │
│                                                                 │
│ cu_seqlens_q: [Batch + 1]  (Cumulative offsets)               │
│   cu_seqlens_q[0] = 0                                          │
│   cu_seqlens_q[1] = Seq_Len_0                                 │
│   cu_seqlens_q[2] = Seq_Len_0 + Seq_Len_1                     │
│   ...                                                          │
│                                                                 │
├────────────────────────────────────────────────────────────────┤
│ Q Layout (packed):                                              │
│                                                                 │
│ [Batch 0: Seq 0..N0-1] | [Batch 1: Seq 0..N1-1] | ...        │
│  ├─ Row 0: Head0|Head1|...                                     │
│  ├─ Row 1: Head0|Head1|...                                     │
│  └─ Row N0-1: Head0|Head1|...                                  │
│              ↓ (transition)                                     │
│  ├─ Row N0: Head0|Head1|...                                    │
│  ├─ Row N0+1: Head0|Head1|...                                  │
│  └─ Row N0+N1-1: Head0|Head1|...                               │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

K_Cache: [Total_KV, KV_Dim]  (All sequences' KV concatenated)
V_Cache: [Total_KV, KV_Dim]

cu_seqlens_k: [Batch + 1]  (Cumulative KV offsets)

OUTPUT:
O: [Total_Q, Q_Dim]  (Same packing as Q)
```

**Memory Address Pattern (batched):**
```
For batch b, query seq q, head h, dimension d:
  q_global_idx = cu_seqlens_q[b] + q
  Q[q_global_idx][h][d] 
    → Q_buffer[q_global_idx * Q_Dim + h * Head_Dim + d]

For batch b, KV seq k:
  kv_global_idx = cu_seqlens_k[b] + k
  K[kv_global_idx][h][d] 
    → K_buffer[kv_global_idx * KV_Dim + h * Head_Dim + d]
```

---

## Grid and Block Configuration Changes

### Decode Path

#### Current (Single Batch)
```
dim3 grid(num_q_heads);                              // Grid X
dim3 block(head_dim_rounded_to_32);                  // Block X

// Kernel logic:
int q_head_idx = blockIdx.x;                        // One block per head
int tid = threadIdx.x;                              // Thread within head
int kv_seq_len = *kv_seq_len_ptr;                   // Single pointer dereference

// Access:
Q[q_head_idx * head_dim + tid]
K[t * num_kv_heads * head_dim + kv_head_idx * head_dim + tid]
```

#### Proposed (Batched)
```
dim3 grid(num_q_heads * batch_size);                // Grid X expands with batch
dim3 block(head_dim_rounded_to_32);                 // Block X unchanged

// Kernel logic:
int global_block_idx = blockIdx.x;
int batch_id = global_block_idx % batch_size;       // Extract batch
int q_head_idx = global_block_idx / batch_size;     // Extract head
int tid = threadIdx.x;

// Cumulative offset lookup:
int q_start = cu_seqlens_q[batch_id];               // Start of this batch's Q
int kv_start = cu_seqlens_k[batch_id];              // Start of this batch's KV
int kv_len = cu_seqlens_k[batch_id + 1] - kv_start; // Length for this batch

// Access:
Q[(q_start + 0) * num_q_heads * head_dim + q_head_idx * head_dim + tid]
K[(kv_start + t) * num_kv_heads * head_dim + kv_head_idx * head_dim + tid]
```

### Prefill Path

#### Current (Single Batch)
```
dim3 grid(ceil(q_seq_len / Br), num_q_heads);       // Grid X, Y
dim3 block(THREADS_PER_BLOCK);                      // 128

// Kernel logic:
unsigned Q_tile_id = blockIdx.x;                    // Tile over query sequence
unsigned q_head_idx = blockIdx.y;                   // Head index
unsigned q_row_id = (tid / 4) + Q_tile_id * Br;    // Absolute row in Q

// Access:
Q[q_row_id * num_q_heads * head_dim + q_head_idx * head_dim + ...]
```

#### Proposed (Batched) - Option A: Grid Z Dimension
```
dim3 grid(ceil(max_seqlen_q / Br), num_q_heads, batch_size);
                                                      // Add Z dimension
// Kernel logic:
unsigned Q_tile_id = blockIdx.x;
unsigned q_head_idx = blockIdx.y;
unsigned batch_id = blockIdx.z;

// Cumulative lookup:
int q_start = cu_seqlens_q[batch_id];
int q_local_len = cu_seqlens_q[batch_id + 1] - q_start;
int q_tile_size = min(Br, q_local_len - Q_tile_id * Br);
if (Q_tile_id * Br >= q_local_len) return;          // Out of bounds for this batch

// Access:
int q_row_id = (tid / 4) + Q_tile_id * Br;         // Local tile
int q_abs_row = q_start + q_row_id;                // Global absolute row
Q[q_abs_row * num_q_heads * head_dim + ...]
```

#### Proposed (Batched) - Option B: Linear Grid (More Compatible)
```
dim3 grid(ceil(max_seqlen_q / Br) * batch_size, num_q_heads);
                                                      // Flatten batch into X
// Kernel logic:
int flat_x = blockIdx.x;
int batch_id = flat_x / ceil(max_seqlen_q / Br);
int Q_tile_id = flat_x % ceil(max_seqlen_q / Br);
int q_head_idx = blockIdx.y;

// Rest same as Option A...
```

---

## Kernel Indexing Examples

### Example 1: Batched Decode with 2 Sequences

**Inputs:**
```
Batch 0: Q_len=1, KV_len=3, Heads=8, Head_Dim=64
Batch 1: Q_len=1, KV_len=5, Heads=8, Head_Dim=64

Total_Q = 2, Total_KV = 8
cu_seqlens_q = [0, 1, 2]
cu_seqlens_k = [0, 3, 8]
```

**Grid Launch:**
```
dim3 grid(8 * 2) = 16 blocks  // 2 batches, 8 heads each
dim3 block(64);               // head_dim threads
```

**Block 0 (Batch 0, Head 0):**
```
batch_id = 0 % 2 = 0
q_head_idx = 0 / 2 = 0
q_abs_seq = cu_seqlens_q[0] + 0 = 0
kv_seq_len = cu_seqlens_k[1] - cu_seqlens_k[0] = 3

Processes: Q row 0, attends to K/V rows 0-2
```

**Block 1 (Batch 0, Head 1):**
```
batch_id = 1 % 2 = 0
q_head_idx = 1 / 2 = 0  ❌ WRONG! Should handle head offset correctly
```

**Better indexing:**
```
block_idx_in_batch = blockIdx.x / batch_size
head_idx_in_block = blockIdx.x % batch_size  ❌ Still wrong

// Correct:
global_head_block = blockIdx.x / batch_size
batch_id = blockIdx.x % batch_size
q_head_idx = global_head_block

// Then:
// Block 0: global_head_block=0, batch_id=0, q_head_idx=0
// Block 1: global_head_block=0, batch_id=1, q_head_idx=0
// Block 2: global_head_block=1, batch_id=0, q_head_idx=1
// ...
// Block 15: global_head_block=1, batch_id=1, q_head_idx=1
```

---

## Data Dependencies and Synchronization

### Current (Single)
```
┌─────────────┐
│  Rust Code  │
│  forward()  │
└──────┬──────┘
       │ Host→Device: q_ptr, k_ptr, v_ptr pointers
       │ Host→Device: kv_seq_len_ptr (single i32)
       ▼
┌──────────────────────┐
│  CUDA Kernel Launch  │
│  (Synchronous call)  │
└──────────────────────┘
       │ (implicit sync)
       ▼
┌──────────────────────┐
│  Next Op / Return    │
└──────────────────────┘
```

### Proposed (Batched)
```
┌─────────────────────────────────────────┐
│  Rust Code: forward_batched()           │
│  Validates:                             │
│  - All cu_seqlens arrays on device      │
│  - Total_Q, Total_KV match packed data  │
└──────┬──────────────────────────────────┘
       │ Host→Device: multiple cu_seqlens arrays
       │ Validate: cum_seqlens_q[-1] == Total_Q
       │ Validate: cum_seqlens_k[-1] == Total_KV
       ▼
┌──────────────────────────────────────────┐
│  CUDA Kernel Launch (Single Call)        │
│  - All batches processed in one grid     │
│  - Barriers between heads if needed      │
└──────┬───────────────────────────────────┘
       │ (implicit sync or explicit cudaDeviceSynchronize)
       ▼
┌──────────────────────────────────────────┐
│  Next Op / Return Results                │
└──────────────────────────────────────────┘
```

**Validation Requirements (Rust Layer):**
```rust
// Cu_seqlens sanity checks:
cu_seqlens_q[0] == 0
cu_seqlens_q[batch_size] == total_q_tokens
all cu_seqlens_q[i] < cu_seqlens_q[i+1]

cu_seqlens_k[0] == 0
cu_seqlens_k[batch_size] == total_kv_tokens
all cu_seqlens_k[i] < cu_seqlens_k[i+1]

// Shape checks:
input_q.shape()[0] == total_q_tokens
input_q.shape()[1] == num_q_heads * head_dim
output_o.shape() == input_q.shape()
```

---

## Implementation Checklist

### Phase 1: CUDA Kernel Implementation
- [ ] Create `flash_attn_gqa_varlen.cu` (decode path)
- [ ] Create `flash_attn_gqa_varlen_prefill.cu` (prefill path)
- [ ] Test with small batches (batch_size = 2)
- [ ] Verify indexing with printf debugging
- [ ] Benchmark against sequential launches

### Phase 2: Rust FFI Integration
- [ ] Add FFI declarations in `mod.rs`
- [ ] Implement dispatch logic in `kernels/cuda::flash_attn_gqa_varlen()`
- [ ] Handle dtype dispatch (F32, BF16, F16)
- [ ] Add parameter validation

### Phase 3: Op Layer
- [ ] Add `forward_batched()` method to `FlashAttnGQA`
- [ ] Keep old `forward()` for backward compatibility
- [ ] Document parameter semantics in docstring
- [ ] Add error messages for cu_seqlens validation

### Phase 4: CPU Reference
- [ ] Update `kernels/cpu/flash_attn_gqa.rs` for batching
- [ ] Add batch loop
- [ ] Maintain parallel head iteration
- [ ] Validate against CUDA reference

### Phase 5: Testing & Profiling
- [ ] Unit tests: varying batch sizes (1, 2, 4, 8)
- [ ] Unit tests: variable sequence lengths
- [ ] Performance benchmark: vs. sequential single-batch calls
- [ ] Numerical validation: CPU vs CUDA
- [ ] Edge cases: empty batches, single-token sequences

---

## Performance Considerations

### Expected Benefits of Batching
1. **Better GPU Utilization:** Multiple sequences share compute resources
2. **Reduced Kernel Launch Overhead:** Single launch vs. N launches
3. **Potential Tensor Optimization:** Fused operations across batch
4. **Memory Coalescing:** Better memory access patterns

### Potential Bottlenecks
1. **Imbalanced Batches:** Some sequences much longer than others → warp divergence
2. **Shared Memory:** Larger cu_seqlens arrays consume smem
3. **Occupancy:** More registers per block → fewer blocks
4. **Synchronization:** Might need thread barriers between batches

### Optimization Strategies
- Use dynamic block scheduling for imbalanced workloads
- Pre-allocate cu_seqlens in constant memory if possible
- Consider persistent kernels for very large batches
- Profile with Nsight Compute to identify bottlenecks

