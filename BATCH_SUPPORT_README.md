# FlashAttnGQA Batch Support Analysis Documentation

This directory contains comprehensive analysis and implementation planning for adding variable-length batched attention support to the FlashAttnGQA operator in RustInfer.

## Documents Included

### 1. **BATCH_SUPPORT_ANALYSIS.md**
**Purpose:** High-level overview and comprehensive reference

**Contents:**
- Executive summary of current vs. proposed architecture
- Current Rust Op structure and forward signature
- Input/output tensor shapes and parameters
- CUDA FFI layer details
- CUDA kernel implementations overview
- CPU reference implementation details
- What's currently NOT supported
- Required changes for batch support
- Tensor layout details
- Key parameters summary
- Complexity assessment
- Implementation phase recommendations

**Best for:** Understanding the big picture, planning, and decision-making

---

### 2. **BATCH_SUPPORT_ARCHITECTURE.md**
**Purpose:** Detailed architectural design document

**Contents:**
- Current vs. proposed architecture diagrams (ASCII art)
- Memory layout transformations (single vs. batched)
- Grid and block configuration changes
- Decode path: current vs. proposed
- Prefill path: current vs. proposed with two options
- Kernel indexing examples with concrete numbers
- Data dependencies and synchronization patterns
- Implementation checklist (5 phases)
- Performance considerations and optimization strategies

**Best for:** Deep technical understanding and architecture decisions

---

### 3. **BATCH_SUPPORT_CODE_EXAMPLES.md**
**Purpose:** Concrete code examples showing how to implement batch support

**Contents:**
- **Example 1:** Current single-sequence forward call and kernel structure
- **Example 2:** Proposed batched forward call and kernel structure
- **Example 3:** Memory layout with concrete byte-level numbers
- **Example 4:** CUDA kernel code snippets
  - Current decode kernel (simplified)
  - Proposed batched decode kernel (simplified)
  - Key differences highlighted
- **Example 5:** Rust FFI changes
  - Current FFI declarations
  - Proposed FFI additions
  - Full Rust wrapper implementation
- **Example 6:** Rust Op layer integration
  - New `forward_batched()` method
  - Validation logic
  - Device dispatch

**Best for:** Implementation guidance and copy-paste starting points

---

## Key Findings

### Current Implementation
- **Strictly single-batch:** Processes one sequence per forward call
- **Single KV length pointer:** No support for variable-length sequences
- **No packed sequences:** Requires separate kernel launches for each sequence
- **Grid layout:** One block per query head (e.g., 8 blocks for 8 heads)

### What Needs to Change

#### Rust Op Interface
```rust
// New method alongside existing forward()
pub fn forward_batched(
    input_q: &Tensor,           // [TotalQ, Q_HiddenDim] (packed)
    cu_seqlens_q: &Tensor,      // [Batch+1] cumulative Q offsets
    cu_seqlens_k: &Tensor,      // [Batch+1] cumulative KV offsets
    ...
) -> Result<()>
```

#### CUDA FFI
```rust
pub fn flash_attn_gqa_varlen_cu(
    q_ptr: *const f32,
    cu_seqlens_q: *const i32,   // Array (not single pointer)
    cu_seqlens_k: *const i32,   // Array (not single pointer)
    max_seqlen_q: i32,          // New parameter
    max_seqlen_k: i32,          // New parameter
    batch_size: i32,            // New parameter
    ...
)
```

#### CUDA Kernels
```cuda
// Grid now includes batch dimension
dim3 grid(num_q_heads * batch_size);  // Was just num_q_heads

// Per-block indexing changes
int batch_id = blockIdx.x % batch_size;      // Extract batch
int q_head_idx = blockIdx.x / batch_size;    // Extract head
int q_start = cu_seqlens_q[batch_id];        // Get sequence offsets
int kv_start = cu_seqlens_k[batch_id];       // Get KV offsets
```

---

## Implementation Timeline Recommendations

### Phase 1: CUDA Kernel Implementation (1-2 weeks)
- Create varlen versions of decode and prefill kernels
- Test with small batches (2-4 sequences)
- Verify indexing with printf debugging

### Phase 2: Rust FFI Integration (3-5 days)
- Add FFI declarations
- Implement dispatch logic
- Handle dtype dispatch (F32, BF16, F16)

### Phase 3: Op Layer (2-3 days)
- Add `forward_batched()` method
- Add validation for cu_seqlens
- Maintain backward compatibility with existing `forward()`

### Phase 4: CPU Reference (3-5 days)
- Update CPU implementation for batching
- Validate against CUDA reference

### Phase 5: Testing & Profiling (1 week)
- Unit tests for varying batch sizes
- Performance benchmarks
- Numerical validation

**Total Estimated Time:** 2-3 weeks for complete implementation

---

## Key Parameters Summary

| Parameter | Type | Current | Batch Version |
|-----------|------|---------|---------------|
| `q_seq_len` | i32 | Scalar per call | max_seqlen_q |
| `kv_seq_len` | i32* | Single pointer | Arrays: cu_seqlens_q/k |
| `batch_size` | N/A | Not applicable | Added parameter |
| Grid X | i32 | num_q_heads | num_q_heads × batch_size |
| Grid Y | i32 | num_q_heads | 1 (flattened into X) |
| Grid Z | i32 | N/A | Optional (batch dimension) |

---

## Critical Implementation Details

### Memory Packing
- All sequences concatenated into single tensors
- Cumulative length arrays (cu_seqlens_q, cu_seqlens_k) map batch items to packed positions
- Access pattern: `packed_buffer[cu_seqlens[batch_id] + local_idx]`

### Validation Requirements
```rust
cu_seqlens_q[0] == 0
cu_seqlens_q[batch_size] == total_q_tokens
cu_seqlens_q is strictly increasing

cu_seqlens_k[0] == 0
cu_seqlens_k[batch_size] == total_kv_tokens
cu_seqlens_k is strictly increasing
```

### Backward Compatibility
- Existing `forward()` method remains unchanged
- New `forward_batched()` added alongside it
- No breaking changes to current API
- Can deprecate single-sequence path later if desired

---

## Performance Expected Benefits

1. **Reduced Kernel Launch Overhead:** 1 launch vs. N launches
2. **Better GPU Utilization:** All heads working in parallel across batches
3. **Memory Coalescing:** Improved cache locality
4. **Scaling:** Linear improvement with batch size (up to GPU saturation)

### Potential Bottlenecks
- Imbalanced sequence lengths (warp divergence)
- Shared memory consumption of cu_seqlens arrays
- Increased register pressure per block
- Synchronization between batch items

---

## Quick Reference: File Locations in RustInfer

- **Op Definition:** `crates/infer-worker/src/op/flash_gqa.rs` (lines 9-106)
- **CUDA FFI Wrapper:** `crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/mod.rs`
- **Decode Kernel:** `crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/flash_decoding.cu`
- **Prefill Kernel:** `crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/flash_attn_gqa.cu`
- **CPU Reference:** `crates/infer-worker/src/op/kernels/cpu/flash_attn_gqa.rs`
- **Header:** `crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/flash_attn_gqa.h`

---

## Questions to Consider Before Implementation

1. **Batching Strategy:**
   - Support only decode batching, or also prefill batching?
   - Should sequences have same length or allow variable lengths?

2. **Memory Layout:**
   - Keep current flattened layout or add batch dimension?
   - How to handle padding for imbalanced batches?

3. **Performance Trade-offs:**
   - Worth the complexity for typical workloads?
   - What batch size provides good utilization?

4. **Backward Compatibility:**
   - Keep old single-sequence API forever?
   - Timeline for deprecation if needed?

5. **Testing Strategy:**
   - CPU reference for validation?
   - Numerical accuracy requirements?
   - Benchmark targets?

---

## Next Steps

1. **Review** all three analysis documents
2. **Discuss** implementation approach with team
3. **Create** CUDA kernel skeleton for varlen version
4. **Start** with Phase 1 (CUDA kernels)
5. **Iterate** and validate at each phase

---

*Analysis completed: May 2026*
*Prepared for: RustInfer Project*

