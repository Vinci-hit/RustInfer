# RustInfer Test Inventory

## Quick Facts
- **Total inline test modules:** 40+
- **Estimated test functions:** 428+
- **Feature-gated tests:** ~30% (require `--features cuda` or others)
- **Model-dependent tests:** ~3 (marked `#[ignore]`)
- **Test files with code:** 2 dedicated files

---

## Test Modules by Subsystem

### 1. Tensor System (Core Foundation)

**File:** `crates/infer-worker/src/tensor/tests.rs`
- **Functions:** 36 test functions
- **Coverage:** Shape operations, views, materialization, dtype conversion, CUDA transfers
- **Key Tests:**
  - `strides_default_contiguous()` - Basic tensor properties
  - `transpose_flips_strides()` - View operations
  - `narrow_is_zero_copy_and_offsets_storage()` - Zero-copy views
  - `view_requires_contiguous()` - View semantics
  - `expand_broadcasts_size_one()` - Broadcasting
  - `cuda_roundtrip_preserves_data()` - CPU↔GPU transfers
  - `owns_storage_tightly_distinguishes_prefix_view()` - Regression test
  - `permute_into_cpu_matches_reference()` - Permutation correctness

**File:** `crates/infer-worker/src/tensor/dims.rs`
- **Module:** `mod tests { ... }`
- **Coverage:** Dimension validation, bounds checking

---

### 2. Memory & Buffer System

**File:** `crates/infer-worker/src/base/buffer.rs`
- **Module:** `mod tests { ... }`
- **Coverage:** Buffer allocation, CPU/CUDA transfers, async operations
- **Key Tests:**
  - `test_buffer_cpu_allocation()` - CPU buffer lifecycle
  - `test_buffer_cuda_cpu()` - CUDA ↔ CPU synchronous transfer
  - `test_buffer_cuda_to_cuda()` - CUDA ↔ CUDA transfers
  - `test_buffer_copy_from_async_d2d()` - Asynchronous D2D copy
  - `test_buffer_zero_out_async()` - Async zeroing

**File:** `crates/infer-worker/src/base/slice_utils.rs`
- **Module:** `mod tests { ... }`
- **Coverage:** Slice operations, indexing edge cases

---

### 3. Operators (26+ modules)

Each operator module follows this pattern:
```
mod tests {
    fn assert_close(...) { }  // F32 tolerance comparison
    fn assert_bf16_close(...) { }  // BF16 tolerance comparison
    
    #[test] fn op_cpu_f32() { }  // CPU F32 version
    #[test] fn op_cpu_bf16() { }  // CPU BF16 version
    #[cfg(feature = "cuda")]
    #[test] fn op_cuda_f32() { }  // GPU F32 version
    #[cfg(feature = "cuda")]
    #[test] fn op_cuda_bf16() { }  // GPU BF16 version
}
```

#### Arithmetic Operations
- **`op/add.rs`** - Element-wise addition, broadcasting, in-place
- **`op/scalar.rs`** - Scalar operations (mul, div, add by scalar)
- **`op/broadcast_mul.rs`** - Broadcasted multiplication
- **`op/ewise_mul.rs`** - Element-wise multiplication

#### Attention Mechanisms
- **`op/rope.rs`** - Rotary positional embeddings
- **`op/rope_interleaved.rs`** - Interleaved RoPE variant
- **`op/sdpa.rs`** - Scaled dot-product attention
- **`op/kv_cache.rs`** - KV cache operations
- **`op/attention/ragged.rs`** - Ragged batch attention scheduling
- **`op/attention/decode_batch.rs`** - Batch decoding kernel

#### Normalization & Activation
- **`op/rmsnorm.rs`** - RMSNorm (root mean square layer norm)
- **`op/activation.rs`** - ReLU, GELU, SiLU, SwGLU, GLU variants

#### Quantization & Conversion
- **`op/cast.rs`** - Type casting (F32 ↔ BF16 ↔ F16)
- **`op/embedding.rs`** - Token embedding lookup

#### Tensor Manipulation
- **`op/concat.rs`** - Concatenation
- **`op/pad.rs`** - Padding operations
- **`op/matmul.rs`** - Matrix multiplication

#### Sampling
- **`op/sampler.rs`** - Top-K and nucleus sampling

#### CUDA Kernels (Curated)
- **`op/kernels/cuda/matmul/mod.rs`** - Cutlass-based GEMM tests

---

### 4. Model Layer

#### LLM (Large Language Models)
- **`model/llm/mod.rs`** - Model initialization, config parsing
- **`model/runtime/kv_cache.rs`** - KV cache management and reuse

#### Common
- **`model/common/safetensor_loader.rs`** - SafeTensor weight loading and validation

#### Diffusion (Text-to-Image)
- **`model/diffusion/scheduler.rs`** - DDIM scheduler
- **`model/diffusion/vae/decoder.rs`** - VAE decoding
- **`model/diffusion/vae/state.rs`** - VAE state management
- **`model/diffusion/z_image/pipeline.rs`** - End-to-end generation pipeline
- **`model/diffusion/z_image/dit_block.rs`** - Diffusion Transformer blocks
- **`model/diffusion/z_image/text_encoder.rs`** - Text encoding
- **`model/diffusion/z_image/patchify.rs`** - Image to patches conversion
- **`model/diffusion/z_image/rope_embedder_3d.rs`** - 3D RoPE embeddings
- **`model/diffusion/z_image/state.rs`** - Pipeline state management

---

### 5. Worker & Runner (E2E Integration)

**File:** `crates/infer-worker/src/worker/runner.rs`
- **Test Modules:** 3
  - `mod tests { ... }` - Main E2E tests (18 functions)
  - `mod tests_qwen3 { ... }` - Qwen3-specific tests
  - `mod tests_perf { ... }` - Performance benchmarks

**Key Tests:**
- `runner_prefill_decode_smoke()` - **Main E2E test** (requires model, `#[ignore]`)
  - Validates thread synchronization
  - Tests prefill phase
  - Tests 8 decode iterations
  - Checks output token validity
  - Verifies no deadlocks

**Test Helpers:**
- `get_model_path()` - Model weight locator
- `fill_inputs_for_step()` - GPU tensor population
- `drive_step()` - High-level forward wrapper
- `make_prefill_meta()` - Metadata construction
- `make_single_decode_meta()` - Decode metadata

**Dummy Implementation:**
- **`worker/runner_dummy.rs`** - CPU-only mock runner
  - Always outputs token 42
  - Used for CPU-only testing without GPU

---

### 6. Protocol

**File:** `crates/infer-protocol/src/syntax_test.rs`
- **Type:** Compile-time validation
- **Function:** `test_protocol_types()`
- **Purpose:** Ensures protocol message types can be constructed
- **No runtime assertions** - just verifies compilation

---

## Test Execution Matrix

### By Feature
```
✓ no features               - CPU-only tests (~350 tests)
✓ --features cuda          - CUDA-enabled tests (~428 tests, default)
✓ --features models        - Model loading tests
✓ --features "cuda,models" - Full integration tests
```

### By Category
```bash
# Tensor & memory
cargo test --lib tensor::tests         # 36 tests
cargo test --lib base::                # buffer, slice_utils
cargo test --lib ops::                 # All operators

# Model layer
cargo test --lib model::                # All models

# Integration
cargo test runner_prefill_decode_smoke -- --ignored  # E2E (requires model)
```

### By Device
```bash
# CPU only
cargo test --no-default-features

# With CUDA
cargo test --features cuda

# Both
cargo test --all-features
```

---

## Test Quality Characteristics

### Strengths ✅
1. **Comprehensive unit test coverage** of tensor and operator layers
2. **Feature-gated CUDA tests** - builds with/without GPU support
3. **Property testing patterns** - shape, stride, storage invariants
4. **Zero-copy verification** - tests validate narrow/view implementations
5. **Multi-dtype support** - F32, BF16, F16 variants tested
6. **End-to-end integration** - runner E2E test available
7. **Thread safety** - SyncFlags synchronization validated
8. **Performance profiling** - nsys traces for optimization analysis

### Gaps ⚠️
1. **No dedicated `tests/` directory** - all colocated with source
2. **No property-based testing** (`proptest`) - only manual test cases
3. **No benchmark suite** (`criterion`) - basic perf tests only
4. **Limited error path testing** - mostly happy path
5. **No stress tests** - batching limits not thoroughly tested
6. **Model weights not in repo** - E2E tests skipped without manual setup
7. **No CI/CD integration** - tests run locally only (as of report date)
8. **Limited distributed testing** - single-GPU focus

---

## Test Dependencies & Setup

### Build Dependencies
```toml
[dependencies]
# All deps already included in main code
```

### Missing Dev Dependencies
```toml
[dev-dependencies]
# Currently EMPTY
# Recommended:
proptest = "1.0"      # Property-based testing
criterion = "0.5"     # Benchmarking framework
tempfile = "3.0"      # Test fixtures
```

### Environment Variables
```bash
LLAMA3_MODEL_PATH=/path/to/model  # For E2E tests
RUST_BACKTRACE=1                   # Backtrace on panic
RUST_LOG=debug                     # Logging
```

---

## Coverage Summary by Layer

| Layer | # Tests | Coverage | Status |
|-------|---------|----------|--------|
| Tensor | 36+ | Shape, views, materialization, dtype | ✓ Excellent |
| Memory | 15+ | Allocation, transfers, async | ✓ Good |
| Operators | 200+ | Elementwise, attention, norm, sampling | ✓ Excellent |
| Model | 40+ | Llm, diffusion, loading | ✓ Good |
| Runner | 18+ | E2E prefill/decode, threading | ⚠ Limited |
| API/Server | <5 | Health checks only | ✗ Poor |
| Protocol | 1 | Type validation | ✓ Good |

---

## Recommended Next Steps

### 1. Add Integration Tests Directory
```
tests/
├── llm_e2e.rs           # Full LLM generation pipeline
├── diffusion_e2e.rs     # Full image generation
├── batch_stress.rs      # Multi-sequence stress test
└── memory_e2e.rs        # Memory lifecycle test
```

### 2. Add Dev Dependencies
```toml
[dev-dependencies]
proptest = "1.0"
criterion = "0.5"
tempfile = "3.0"
once_cell = "1.21"
```

### 3. Add Benchmarks
```
benches/
├── attention.rs
├── matmul.rs
└── e2e_throughput.rs
```

### 4. CI/CD Integration
```yaml
# .github/workflows/test.yml
- Test on every push
- Coverage reporting
- Performance tracking
- Multi-GPU testing (when available)
```

---

## Quick Test Command Reference

```bash
# All tests
cargo test --all

# Specific subsystem
cargo test tensor::tests
cargo test op::add
cargo test model::llm
cargo test worker::runner

# By pattern
cargo test add_          # operator tests
cargo test cuda          # GPU-specific
cargo test decode        # decoding tests

# Feature combinations
cargo test --no-default-features              # CPU only
cargo test --features "cuda,models"            # Full
cargo test --all-features                      # Everything

# Debug
cargo test -- --nocapture                     # Show output
RUST_BACKTRACE=1 cargo test                   # Backtrace
cargo test -- --test-threads=1                # Serial

# E2E (requires setup)
LLAMA3_MODEL_PATH=/path cargo test runner_prefill_decode_smoke -- --ignored
```

---

## File Locations Summary

```
COMPREHENSIVE TESTING AVAILABLE IN:
├── crates/infer-worker/
│   ├── src/tensor/tests.rs           [36+ tests]
│   ├── src/base/buffer.rs            [~10 tests]
│   ├── src/base/slice_utils.rs       [inline tests]
│   ├── src/op/**/*.rs                [200+ tests]
│   ├── src/model/**/*.rs             [40+ tests]
│   ├── src/worker/runner.rs          [18+ tests]
│   └── src/worker/runner_dummy.rs    [mock impl]
├── crates/infer-protocol/
│   └── src/syntax_test.rs            [1 compile-time test]
├── crates/infer-server/
│   ├── src/                          [minimal tests]
│   └── Cargo.toml
└── crates/infer-frontend/
    ├── src/                          [no Rust tests]
    └── Cargo.toml

MISSING (Recommended):
├── tests/
│   ├── llm_e2e.rs
│   ├── diffusion_e2e.rs
│   ├── batch_stress.rs
│   └── memory_e2e.rs
└── benches/
    ├── attention.rs
    ├── matmul.rs
    └── e2e.rs
```

