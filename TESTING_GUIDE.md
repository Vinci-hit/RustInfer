# RustInfer Testing Guide - Complete Examples & Best Practices

## Table of Contents
1. [Unit Test Examples](#unit-test-examples)
2. [Integration Test Patterns](#integration-test-patterns)
3. [Conditional Compilation](#conditional-compilation)
4. [Test Fixtures](#test-fixtures)
5. [Running Tests](#running-tests)
6. [Debugging Tests](#debugging-tests)

---

## Unit Test Examples

### Example 1: Basic Operator Test (Element-wise Addition)

**File:** `crates/infer-worker/src/op/add.rs`

```rust
mod tests {
    use super::*;
    use crate::base::{DataType, DeviceType};
    use crate::base::error::Result;
    use half::bf16;

    // Helper: Float comparison with tolerance
    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "Slices have different lengths");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tol,
                "Mismatch at index {}: a = {}, b = {}", i, x, y);
        }
    }

    // Test 1: CPU F32 addition
    #[test]
    fn add_cpu_f32() -> Result<()> {
        let mut a = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
        
        let mut out = Tensor::empty(&[4], DataType::F32, DeviceType::Cpu)?;
        add(&a, &b, &mut out, None)?;
        
        assert_eq!(out.as_f32()?.as_slice()?, &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    // Test 2: In-place addition
    #[test]
    fn add_inplace_cpu_f32() -> Result<()> {
        let mut a = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        a.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[1.0, 2.0, 3.0]);
        b.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[0.5, 0.5, 0.5]);
        
        add(&mut a, &b, &mut a, None)?;  // a = a + b (in-place)
        
        assert_eq!(a.as_f32()?.as_slice()?, &[1.5, 2.5, 3.5]);
        Ok(())
    }

    // Test 3: Broadcast addition (2D + 1D)
    #[test]
    fn add_broadcast_cpu() -> Result<()> {
        let mut matrix = Tensor::empty(&[2, 3], DataType::F32, DeviceType::Cpu)?;
        let mut bias = Tensor::empty(&[3], DataType::F32, DeviceType::Cpu)?;
        
        matrix.as_f32_mut()?.as_slice_mut()?.copy_from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        bias.as_f32_mut()?.as_slice_mut()?.copy_from_slice(&[10.0, 20.0, 30.0]);
        
        let mut out = Tensor::empty(&[2, 3], DataType::F32, DeviceType::Cpu)?;
        add(&matrix, &bias, &mut out, None)?;
        
        assert_eq!(out.as_f32()?.as_slice()?,
            &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        Ok(())
    }

    // Test 4: BF16 addition
    #[test]
    fn add_cpu_bf16() -> Result<()> {
        let mut a = Tensor::empty(&[4], DataType::BF16, DeviceType::Cpu)?;
        let mut b = Tensor::empty(&[4], DataType::BF16, DeviceType::Cpu)?;
        
        let a_data: Vec<bf16> = vec![bf16::from_f32(1.0), bf16::from_f32(2.0),
                                     bf16::from_f32(3.0), bf16::from_f32(4.0)];
        let b_data: Vec<bf16> = vec![bf16::from_f32(10.0), bf16::from_f32(20.0),
                                     bf16::from_f32(30.0), bf16::from_f32(40.0)];
        
        // Copy data to device
        a.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&a_data);
        b.as_bf16_mut()?.as_slice_mut()?.copy_from_slice(&b_data);
        
        let mut out = Tensor::empty(&[4], DataType::BF16, DeviceType::Cpu)?;
        add(&a, &b, &mut out, None)?;
        
        let result = out.as_bf16()?.as_slice()?;
        assert!((result[0].to_f32() - 11.0).abs() < 0.1);
        assert!((result[1].to_f32() - 22.0).abs() < 0.1);
        Ok(())
    }

    // Test 5: CUDA-only test
    #[cfg(feature = "cuda")]
    #[test]
    fn add_cuda_f32() -> Result<()> {
        let mut a = Tensor::empty(&[4], DataType::F32, DeviceType::Cuda(0))?;
        let mut b = Tensor::empty(&[4], DataType::F32, DeviceType::Cuda(0))?;
        
        // Fill with host data
        let host_data_a = vec![1.0, 2.0, 3.0, 4.0];
        let host_data_b = vec![10.0, 20.0, 30.0, 40.0];
        a.write_from_f32_host(&host_data_a)?;
        b.write_from_f32_host(&host_data_b)?;
        
        let mut out = Tensor::empty(&[4], DataType::F32, DeviceType::Cuda(0))?;
        add(&a, &b, &mut out, None)?;
        
        // Read result back to host
        let result = out.to_cpu()?.as_f32()?.as_slice()?;
        assert_eq!(result, &[11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    // Test 6: Edge case - empty tensor
    #[test]
    fn add_empty_tensors() -> Result<()> {
        let a = Tensor::empty(&[0], DataType::F32, DeviceType::Cpu)?;
        let b = Tensor::empty(&[0], DataType::F32, DeviceType::Cpu)?;
        let mut out = Tensor::empty(&[0], DataType::F32, DeviceType::Cpu)?;
        
        add(&a, &b, &mut out, None)?;
        assert_eq!(out.as_f32()?.as_slice()?.len(), 0);
        Ok(())
    }
}
```

### Example 2: Tensor Manipulation Test

**File:** `crates/infer-worker/src/tensor/tests.rs` (excerpt)

```rust
#[test]
fn transpose_flips_strides() -> Result<()> {
    // Create 2D tensor with known shape and strides
    let t = Tensor::empty(&[2, 3], DataType::F32, DeviceType::Cpu)?;
    
    // Verify contiguous layout
    assert_eq!(t.shape(),   &[2, 3]);
    assert_eq!(t.strides(), &[3, 1]);
    assert!(t.is_contiguous());
    
    // Transpose
    let tt = t.transpose(0, 1)?;
    
    // Verify transposed properties
    assert_eq!(tt.shape(),   &[3, 2]);
    assert_eq!(tt.strides(), &[1, 3]);  // Strides flipped!
    assert!(!tt.is_contiguous());       // Now strided
    Ok(())
}

#[test]
fn narrow_is_zero_copy_and_offsets_storage() -> Result<()> {
    // Create 3×4 matrix with known values
    let mut t = Tensor::empty(&[3, 4], DataType::F32, DeviceType::Cpu)?;
    let s = t.as_f32_mut()?.as_slice_mut()?;
    for i in 0..12 { s[i] = i as f32; }
    
    // Narrow rows 1..3 (zero-copy)
    let n = t.narrow(0, 1, 2)?;
    
    // Verify properties
    assert_eq!(n.shape(), &[2, 4]);
    assert_eq!(n.strides(), &[4, 1]);
    assert_eq!(n.storage_offset(), 4);  // Offset by 4 elements
    
    // Verify values
    let dense = n.contiguous()?;
    assert_eq!(dense.as_f32()?.as_slice()?,
        &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    Ok(())
}

#[test]
fn view_requires_contiguous() -> Result<()> {
    let mut t = Tensor::empty(&[4, 5], DataType::F32, DeviceType::Cpu)?;
    
    // Transpose creates non-contiguous view
    let p = t.transpose(0, 1)?;
    
    // view() should fail on non-contiguous
    assert!(p.view(&[20]).is_err(), "view on strided must fail");
    
    // reshape() should work (materializes)
    assert!(p.reshape(&[20]).is_ok(), "reshape should densify");
    Ok(())
}

#[test]
fn expand_broadcasts_size_one() -> Result<()> {
    // Create [3, 1] tensor
    let mut t = Tensor::empty(&[3, 1], DataType::F32, DeviceType::Cpu)?;
    let s = t.as_f32_mut()?.as_slice_mut()?;
    s.copy_from_slice(&[10.0, 20.0, 30.0]);
    
    // Expand to [3, 5] (broadcast the single column)
    let e = t.expand(&[3, 5])?;
    
    // Verify broadcast structure
    assert_eq!(e.shape(),   &[3, 5]);
    assert_eq!(e.strides(), &[1, 0]);  // Stride 0 on broadcast axis!
    
    // Materialize to verify contents
    let dense = e.contiguous()?;
    let got = dense.as_f32()?.as_slice()?;
    
    // Each column should be replicated
    let expected: Vec<f32> = vec![10.0, 10.0, 10.0, 10.0, 10.0,
                                  20.0, 20.0, 20.0, 20.0, 20.0,
                                  30.0, 30.0, 30.0, 30.0, 30.0];
    assert_eq!(got, &expected[..]);
    Ok(())
}
```

---

## Integration Test Patterns

### Example 3: Runner End-to-End Test

**File:** `crates/infer-worker/src/worker/runner.rs` (excerpt)

```rust
#[test]
#[ignore = "requires LLAMA3_MODEL_PATH env or well-known model path"]
fn runner_prefill_decode_smoke() -> Result<()> {
    // 1. Locate model
    let path = match get_model_path() {
        Some(p) => p,
        None => {
            eprintln!("LLAMA3_MODEL_PATH not set; skipping");
            return Ok(());
        }
    };
    
    // 2. Initialize model and runner
    let device = DeviceType::Cuda(0);
    let model = Llama3::new(&path, device)?;
    let vocab = model.config().vocab_size;
    
    let max_batch_tokens = 512usize;
    let max_batch_seqs = 1usize;
    let runner = Arc::new(ModelRunner::new(
        model,
        device,
        max_batch_tokens,
        max_batch_seqs,
    )?);
    
    // 3. Spawn runner thread
    let runner_loop = Arc::clone(&runner);
    let runner_handle = std::thread::spawn(move || runner_loop.run());
    
    // 4. Tokenize prompt
    let prompt = "Hello, my name is";
    let prompt_tokens: Vec<i32> = runner
        .model()
        .tokenizer()
        .encode(prompt)?
        .into_iter()
        .collect();
    let prompt_len = prompt_tokens.len();
    assert!(prompt_len > 0 && prompt_len <= max_batch_tokens);
    
    // 5. Allocate KV cache
    let max_total = prompt_len + 16;
    unsafe { runner.state_mut(0).kv_cache.ensure_capacity(max_total)?; }
    
    // 6. Prefill step
    let prefill_meta = {
        let mut m = StepMeta::zeroed();
        m.num_prefill = 1;
        m.num_decode = 0;
        m.q_start_loc[0] = 0;
        m.q_start_loc[1] = prompt_len as i32;
        m.slot_indices[0] = 0;
        m.positions_start[0] = 0;
        m
    };
    
    fill_inputs_for_step(
        &runner,
        &prompt_tokens,
        &(0..prompt_len as i32).collect::<Vec<i32>>(),
        &[0i32],
        &prefill_meta,
    )?;
    
    unsafe { runner.write_meta(prefill_meta.clone()); }
    runner.set_input_ready();
    
    // 7. Wait for output
    while !runner.output_ready() {
        std::hint::spin_loop();
    }
    
    let first_token = unsafe { runner.output_tokens_dev() }
        .to_cpu()?
        .as_i32()?
        .as_slice()?[0];
    
    runner.set_output_consumed();
    
    // 8. Verify first token is valid
    assert!(
        first_token >= 0 && (first_token as usize) < vocab,
        "first_token {} out of range [0, {})", first_token, vocab
    );
    
    // 9. Generate 8 more tokens
    let mut generated = vec![first_token];
    for step_i in 0..8 {
        let pos = (prompt_len + step_i) as i32;
        let last_tok = *generated.last().unwrap();
        
        let decode_meta = {
            let mut m = StepMeta::zeroed();
            m.num_prefill = 0;
            m.num_decode = 1;
            m.q_start_loc[0] = 0;
            m.q_start_loc[1] = 1;
            m.slot_indices[0] = 0;
            m.positions_start[0] = pos;
            m
        };
        
        fill_inputs_for_step(
            &runner,
            &[last_tok],
            &[pos],
            &[pos],
            &decode_meta,
        )?;
        
        unsafe { runner.write_meta(decode_meta.clone()); }
        runner.set_input_ready();
        
        while !runner.output_ready() {
            std::hint::spin_loop();
        }
        
        let tok = unsafe { runner.output_tokens_dev() }
            .to_cpu()?
            .as_i32()?
            .as_slice()?[0];
        
        runner.set_output_consumed();
        
        assert!(
            tok >= 0 && (tok as usize) < vocab,
            "decode step {} token {} out of range", step_i, tok
        );
        
        generated.push(tok);
    }
    
    // 10. Shutdown
    runner.request_shutdown();
    let _ = runner_handle.join();
    
    eprintln!("Generated {} tokens: {:?}", generated.len(), generated);
    assert_eq!(generated.len(), 9);
    Ok(())
}
```

---

## Conditional Compilation

### Pattern 1: Feature-Gated Tests

```rust
// Only compiles when 'cuda' feature is enabled
#[cfg(feature = "cuda")]
#[test]
fn cuda_kernel_test() -> Result<()> {
    // CUDA-specific code
    let t = Tensor::empty(&[100], DataType::F32, DeviceType::Cuda(0))?;
    // ...
    Ok(())
}

// Only compiles when 'models' feature is enabled
#[cfg(feature = "models")]
#[test]
fn model_test() -> Result<()> {
    let model = Llama3::new(&path)?;
    // ...
    Ok(())
}

// Multiple conditions
#[cfg(all(feature = "cuda", feature = "models"))]
#[test]
fn cuda_model_test() -> Result<()> {
    // Only runs with both features
}
```

### Pattern 2: Ignored Tests (Requires External Setup)

```rust
// Test that requires model weights to be present
#[test]
#[ignore = "requires LLAMA3_MODEL_PATH env var"]
fn test_with_real_model() -> Result<()> {
    let path = std::env::var("LLAMA3_MODEL_PATH")?;
    let model = Llama3::new(&path)?;
    // ...
    Ok(())
}

// Test that skips if resource unavailable
#[test]
fn test_with_optional_setup() -> Result<()> {
    let Some(path) = get_model_path() else {
        eprintln!("Model not found; skipping");
        return Ok(());
    };
    // ...
    Ok(())
}
```

---

## Test Fixtures

### Pattern 1: Helper Functions

```rust
mod tests {
    use super::*;

    // Create test data
    fn create_test_tensor(shape: &[usize], value: f32) -> Result<Tensor> {
        let mut t = Tensor::empty(shape, DataType::F32, DeviceType::Cpu)?;
        t.fill_(value)?;
        Ok(t)
    }

    // Create random test data
    fn create_random_tensor(shape: &[usize], seed: u64) -> Result<Tensor> {
        Tensor::randn(shape, DataType::F32, DeviceType::Cpu, Some(seed))
    }

    // Assertion helper
    fn assert_tensor_close(a: &Tensor, b: &Tensor, tol: f32) -> Result<()> {
        let a_slice = a.as_f32()?.as_slice()?;
        let b_slice = b.as_f32()?.as_slice()?;
        assert_eq!(a_slice.len(), b_slice.len());
        for (i, (&x, &y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "Mismatch at index {}: {} vs {} (diff={})",
                i, x, y, (x-y).abs()
            );
        }
        Ok(())
    }

    #[test]
    fn test_using_fixtures() -> Result<()> {
        let a = create_test_tensor(&[10], 5.0)?;
        let b = create_random_tensor(&[10], 42)?;
        // ...
        Ok(())
    }
}
```

### Pattern 2: Test Builder Pattern

```rust
struct TensorTestBuilder {
    shape: Vec<usize>,
    device: DeviceType,
    dtype: DataType,
    seed: Option<u64>,
}

impl TensorTestBuilder {
    fn new(shape: Vec<usize>) -> Self {
        Self {
            shape,
            device: DeviceType::Cpu,
            dtype: DataType::F32,
            seed: Some(42),
        }
    }

    fn on_device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    fn with_dtype(mut self, dtype: DataType) -> Self {
        self.dtype = dtype;
        self
    }

    fn build(&self) -> Result<Tensor> {
        if let Some(seed) = self.seed {
            Tensor::randn(&self.shape, self.dtype, self.device, Some(seed))
        } else {
            Tensor::empty(&self.shape, self.dtype, self.device)
        }
    }
}

#[test]
fn test_with_builder() -> Result<()> {
    let t1 = TensorTestBuilder::new(vec![32, 64])
        .on_device(DeviceType::Cpu)
        .with_dtype(DataType::F32)
        .build()?;
    
    #[cfg(feature = "cuda")]
    let t2 = TensorTestBuilder::new(vec![32, 64])
        .on_device(DeviceType::Cuda(0))
        .with_dtype(DataType::BF16)
        .build()?;
    
    Ok(())
}
```

---

## Running Tests

### Basic Commands

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test tensor_tests::tests::transpose_flips_strides

# Run ignored tests
cargo test -- --ignored

# Run specific ignored test
cargo test runner_prefill_decode_smoke -- --ignored

# Run tests by pattern
cargo test add_          # Runs all tests starting with "add_"
cargo test cuda          # Runs all tests with "cuda" in name
```

### Feature-Specific Testing

```bash
# Test with CUDA (default)
cargo test --features cuda

# Test without CUDA
cargo test --no-default-features

# Test with all features
cargo test --all-features

# Test with specific feature combo
cargo test --features "cuda,models"
```

### Performance & Debugging

```bash
# Run with all output + test output
cargo test -- --nocapture --test-threads=1

# Run in release mode (for perf tests)
cargo test --release -- --ignored

# With backtrace on panic
RUST_BACKTRACE=1 cargo test

# With logging
RUST_LOG=debug cargo test -- --nocapture

# Run only single-threaded tests
cargo test -- --test-threads=1
```

---

## Debugging Tests

### Using println! Debugging

```rust
#[test]
fn test_with_debug_output() -> Result<()> {
    let t = Tensor::empty(&[3, 4], DataType::F32, DeviceType::Cpu)?;
    
    println!("Tensor shape: {:?}", t.shape());
    println!("Tensor strides: {:?}", t.strides());
    println!("Is contiguous: {}", t.is_contiguous());
    
    eprintln!("Error-level message: {:?}", t);
    
    Ok(())
}

// Run with:
// cargo test test_with_debug_output -- --nocapture
```

### Using dbg! Macro

```rust
#[test]
fn test_with_dbg() -> Result<()> {
    let x = 42;
    dbg!(x);  // Prints: [tests.rs:42] x = 42
    
    let t = Tensor::empty(&[10], DataType::F32, DeviceType::Cpu)?;
    dbg!(t.shape());  // Prints shape info
    
    Ok(())
}
```

### Panic Debugging

```rust
#[test]
#[should_panic(expected = "attempt to divide by zero")]
fn test_expected_panic() {
    let _ = 1 / 0;
}

#[test]
#[should_panic]
fn test_any_panic() {
    panic!("This is expected");
}
```

### Using Result Type for Better Error Messages

```rust
#[test]
fn test_with_result() -> Result<()> {
    let x = some_operation()
        .map_err(|e| println!("Operation failed: {:?}", e))
        .ok()
        .ok_or_else(|| anyhow::anyhow!("Expected value not found"))?;
    
    Ok(())
}
```

---

## Best Practices

### ✅ DO

- Use `Result<()>` return type for error handling
- Add descriptive test names: `test_narrow_is_zero_copy_and_offsets_storage()`
- Test both happy path and edge cases
- Use helper functions to reduce duplication
- Mark ignored tests with reason
- Use feature gates for device-specific tests
- Verify invariants (shape, strides, offset)
- Test error paths

### ❌ DON'T

- Don't use `.unwrap()` or `.expect()` in tests (use `?` operator)
- Don't create overly complex test setups
- Don't leave println!() statements in committed code (use eprintln!)
- Don't test implementation details
- Don't ignore test failures without documentation
- Don't create expensive tests without marking them `#[ignore]`

---

## Quick Reference: Test Anatomy

```rust
#[test]                              // Marks as test function
#[ignore = "reason"]                 // Skips by default, run with --ignored
#[cfg(feature = "cuda")]             // Conditional compilation
fn test_descriptive_name() -> Result<()> {  // Returns Result
    // 1. ARRANGE: Set up test data
    let input = create_test_data()?;
    
    // 2. ACT: Execute operation under test
    let result = operation_under_test(&input)?;
    
    // 3. ASSERT: Verify expected outcome
    assert_eq!(result.shape(), &[10]);
    assert!(result.is_contiguous());
    
    Ok(())  // ✓ Success
}
```

