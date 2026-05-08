# FlashAttnGQA Batch Support: Implementation Roadmap

**Date:** May 2026  
**Project:** Adding variable-length batched attention to RustInfer  
**Status:** Ready for Implementation (Plan Complete)

---

## Overview

This roadmap consolidates the analysis and planning for adding batch support to the FlashAttnGQA (Grouped-Query Attention) operator in RustInfer.

### What We've Done
1. ✅ **Comprehensive Analysis** - Explored existing single-sequence implementation across 4 files
2. ✅ **Architecture Design** - Defined proposed batched architecture with grid/memory layouts
3. ✅ **Code Examples** - Provided concrete implementation snippets
4. ✅ **Implementation Plan** - Detailed 5-phase rollout with success criteria

### What Comes Next
Your team will execute the 5-phase implementation starting with CUDA kernel development.

---

## Key Artifacts

### Analysis Documents (in `/data/home/vinciiliu/RustInfer/`)
- **BATCH_SUPPORT_README.md** - Navigation guide and overview
- **BATCH_SUPPORT_ANALYSIS.md** - Current architecture, gaps, required changes
- **BATCH_SUPPORT_ARCHITECTURE.md** - Technical design with diagrams and examples
- **BATCH_SUPPORT_CODE_EXAMPLES.md** - Concrete implementation examples

### Implementation Plan
- **Location:** `/data/home/vinciiliu/.claude-internal/plans/synthetic-questing-moonbeam-agent-a6c381a7ffb9df9b8.md`
- **Contents:** 5 detailed phases with tasks, deliverables, success criteria

---

## Quick Reference: Current vs. Proposed

### Current Implementation (Single-Sequence)
```rust
// Rust Op
pub fn forward(
    input_q: &Tensor,              // [Q_SeqLen, Q_HiddenDim]
    current_kv_len_gpu: *const i32, // Pointer to single i32
    ...
) -> Result<()>

// CUDA FFI
pub fn flash_decoding_cu(
    q_ptr: *const f32,
    kv_seq_len: *const i32,        // Dereference single pointer
    ...
);

// CUDA Kernel
dim3 grid(num_q_heads);            // No batch dimension
int kv_len = *kv_seq_len_ptr + 1;  // Single KV length
```

### Proposed Implementation (Batched)
```rust
// Rust Op - NEW METHOD
pub fn forward_batched(
    input_q: &Tensor,              // [TotalQ, Q_HiddenDim] (packed)
    cu_seqlens_q: &Tensor,         // [Batch+1] cumulative offsets
    cu_seqlens_k: &Tensor,         // [Batch+1] cumulative offsets
    batch_size: usize,
    ...
) -> Result<()>

// CUDA FFI - NEW FUNCTION
pub fn flash_attn_gqa_varlen_cu(
    q_ptr: *const f32,
    cu_seqlens_q: *const i32,      // Array lookup per sequence
    cu_seqlens_k: *const i32,      // Array lookup per sequence
    batch_size: i32,
    ...
);

// CUDA Kernel - NEW GRID
dim3 grid(num_q_heads * batch_size);  // Batch × heads parallelism
int batch_id = blockIdx.x % batch_size;
int kv_len = cu_seqlens_k[batch_id+1] - cu_seqlens_k[batch_id];
```

---

## Implementation Phases

### Phase 1: CUDA Kernel Development (1-2 weeks)
**What:** Implement `flash_decoding_varlen.cu` and `flash_attn_gqa_varlen.cu`  
**Key Tasks:**
- Generalize current decode kernel to batch dimension
- Implement per-block indexing: `batch_id = blockIdx.x % batch_size`
- Access cu_seqlens arrays for per-sequence parameters
- Validate against CPU reference

**Success Criteria:**
- Small batch tests (2-4 seqs) pass numerical validation
- Indexing logic verified with printf debugging
- No crashes with edge cases

---

### Phase 2: Rust FFI Integration (3-5 days)
**What:** Add FFI declarations and Rust wrappers  
**Key Tasks:**
- Update `flash_attn_gqa.h` with new function declarations
- Add unsafe extern "C" in `mod.rs`
- Implement dtype dispatch (F32, BF16, F16)
- Create validation functions for cu_seqlens arrays

**Success Criteria:**
- FFI layer compiles without warnings
- Dtype dispatch tested for all three types
- Validation catches malformed cu_seqlens

---

### Phase 3: Rust Op Layer (2-3 days)
**What:** Add `forward_batched()` method to FlashAttnGQA Op  
**Key Tasks:**
- Implement new method alongside existing `forward()`
- Add tensor shape validation
- Device dispatch (CUDA vs. CPU)
- Extend CPU reference for batching

**Success Criteria:**
- Backward compatibility maintained
- Input validation prevents crashes
- CPU path works for validation

---

### Phase 4: Testing & Profiling (1 week)
**What:** Comprehensive test suite and benchmarks  
**Test Matrix:**
```
Batch Sizes:    1, 2, 4, 8
Q Seq Lengths:  1, 64, 128
KV Seq Lengths: 1 to 512 (variable)
Dtypes:         F32, BF16, F16
Causal:         Yes, No
```

**Success Criteria:**
- All combinations pass numerical validation
- Decode batching: ≥1.5x throughput for batch_size=4
- No regressions on single-batch path

---

### Phase 5: Documentation & Polish (2-3 days)
**What:** Finalize code, docs, and guides  
**Key Tasks:**
- Add inline code comments explaining batch indexing
- Update API documentation
- Create migration guide for users
- Performance tuning recommendations

**Success Criteria:**
- All comments explain "why" not just "what"
- API is clear and intuitive
- Users understand performance trade-offs

---

## Expected Impact

### GPU Utilization
```
Current:  1 sequence per launch
          → 8 GPU blocks for 8-head model
          
Proposed: 4 sequences per launch
          → 32 GPU blocks (8 heads × 4 seqs)
          → 4x block-level parallelism
          → ~1.5-2x throughput improvement
```

### Kernel Launch Overhead
- **Current:** N sequences = N separate CUDA kernel launches
- **Proposed:** N sequences = 1 CUDA kernel launch
- **Benefit:** Reduces launch latency from milliseconds to microseconds per batch

### Memory Efficiency
- Packed sequence format eliminates per-sequence memory overhead
- cu_seqlens arrays are tiny (typically <100 bytes for typical batches)

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| **Indexing bugs in CUDA** | Extensive printf debugging, CPU reference validation |
| **Performance regression** | Early profiling, compare against sequential launches |
| **Numerical inaccuracy** | Rigorous test suite with tolerance bands |
| **Integration issues** | Gradual rollout with feature flags if needed |

---

## File Structure

### Files to Create
```
crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/
├── flash_decoding_varlen.cu        (NEW - Batch decode kernel)
├── flash_attn_gqa_varlen.cu        (NEW - Batch prefill kernel)
```

### Files to Modify
```
crates/infer-worker/src/op/kernels/cuda/flash_attn_gqa/
├── flash_attn_gqa.h                (ADD - Function declarations)
├── mod.rs                          (ADD - FFI + Rust wrapper)

crates/infer-worker/src/op/
├── flash_gqa.rs                    (ADD - forward_batched method)
├── kernels/cpu/flash_attn_gqa.rs   (EXTEND - Batch support)

crates/infer-worker/
├── tests/test_batch_attention.rs   (NEW - Test suite)
```

---

## Decision Points

Before starting, confirm:

1. **Grid Layout:** Use flattened 1D `grid(num_q_heads * batch_size)` or 2D grid?
   - Recommendation: Start with 1D, switch if needed

2. **Max Batch Size:** Support up to 32 sequences, or higher?
   - Recommendation: 32 for typical use, scalable to higher

3. **Gradual Rollout:** Feature flag for incremental deployment?
   - Recommendation: No flag needed if tests pass

---

## Success Definition

The implementation is successful when:

✅ All parametrized tests pass (CPU vs. CUDA match within tolerance)  
✅ Decode path shows ≥1.5x speedup for batch_size=4  
✅ Prefill path shows ≥1.2x speedup for batch_size=4  
✅ No numerical regressions on existing single-batch tests  
✅ Code is well-commented and documented  
✅ API is intuitive and backward-compatible  

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 (CUDA) | 1-2 weeks | None - start here |
| Phase 2 (FFI) | 3-5 days | Phase 1 complete |
| Phase 3 (Op) | 2-3 days | Phase 2 complete |
| Phase 4 (Tests) | 1 week | Phase 3 complete |
| Phase 5 (Docs) | 2-3 days | Phase 4 complete |
| **Total** | **2-3 weeks** | |

---

## How to Use This Roadmap

1. **Planning Meetings:** Reference this roadmap and the detailed plan file
2. **Phase Kickoff:** Refer to specific phase section in the plan for tasks
3. **Development:** Use BATCH_SUPPORT_CODE_EXAMPLES.md for implementation snippets
4. **Architecture Questions:** Consult BATCH_SUPPORT_ARCHITECTURE.md for detailed design
5. **Review:** Check success criteria at end of each phase

---

## Next Steps

1. **Assign:** Designate developer(s) for each phase
2. **Setup:** Create feature branch for development
3. **Start:** Phase 1 begins with deep study of existing kernels
4. **Communicate:** Share this roadmap with team and stakeholders
5. **Execute:** Follow phase-by-phase approach, validating each step

---

## Additional Resources

### Documents
- Comprehensive analysis in `/data/home/vinciiliu/RustInfer/BATCH_SUPPORT_*.md`
- Detailed implementation plan in `.claude-internal/plans/`

### Reference Code
- Current single-sequence implementation (lines 9-106 of `flash_gqa.rs`)
- Current CUDA kernels in `flash_decoding.cu` and `flash_attn_gqa.cu`
- CPU reference in `kernels/cpu/flash_attn_gqa.rs`

---

**Document prepared:** May 2026  
**For:** RustInfer Project Team  
**Status:** Ready for Implementation

