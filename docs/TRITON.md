# Triton kernels

RustInfer has an optional `triton` Cargo feature for the Triton GPU kernel
language. The initial integration implements RMSNorm inside the CUDA backend.
Model code continues to call the existing `MathOps::rmsnorm` and
`MathOps::rmsnorm_inplace` interfaces.

## Build

Keep the normal CUDA build prerequisites from the [README](../README.md).
Install the pinned Triton compiler into a Python environment and select that
interpreter when building:

```bash
uv sync --inexact --extra triton
RUSTINFER_TRITON_PYTHON="$PWD/.venv/bin/python" \
  cargo build --release -p infer-worker --features triton
```

The `triton` Python extra pins `triton==3.6.0`. The build checks this version
because the generated kernel ABI is version dependent. A different compiler
version fails the build with an actionable error. Without the Cargo feature,
building RustInfer requires no Triton installation.

`CUDA_ARCH` selects the same architecture for the CUDA and Triton kernels. When
unset, the existing CUDA build script detects the local GPU. For an explicit
target, for example an NVIDIA GPU with compute capability 8.9:

```bash
CUDA_ARCH=sm_89 RUSTINFER_TRITON_PYTHON="$PWD/.venv/bin/python" \
  cargo build --release -p infer-worker --features triton
```

Triton compilation supports `sm_80` and newer targets supported by the pinned
compiler. The Python compiler accepts an explicit target and does not need a
GPU or PyTorch. A build machine still needs the project's normal CUDA toolchain
and libraries. The compiled worker needs no Python, PyTorch, or Triton package
at runtime; its kernels are embedded in the binary.

## Dispatch and lifecycle

When both AOT features are enabled, [TileLang](TILELANG.md) takes precedence.
Otherwise, RMSNorm uses Triton when the running GPU matches the compiled compute
capability, the input, weight, and output are contiguous, and the last dimension
is between 1 and 16384 inclusive. Supported dtypes are FP32, FP16, and BF16.
Input, weight, and output have the same dtype. Odd dimensions, including 37 and
1025, are supported in this path. Row count, actual dimension, and epsilon are
runtime arguments; kernels specialize only on dtype and power-of-two block size.

Strided tensors, larger dimensions, and GPUs with a different compute capability
use the existing CUDA implementation, subject to that CUDA build's GPU
architecture compatibility. Its vector layout requirements still apply:
strided views must have rank 2 or 3 with a dense last
dimension; dimensions and row addresses must be aligned to 4 elements for FP32
or 8 elements for FP16/BF16. Unsupported layouts return an error.
Epsilon must be finite and nonnegative, weight must be contiguous, and output
shape must match input shape. All tensors must belong to the operation's CUDA
configuration. Partial input/output overlap and output/weight overlap are
rejected; exact input/output aliasing is supported. Runtime kernel launch
failures are returned to the caller.

The half-precision path preserves the existing CUDA arithmetic boundaries:
accumulate in FP32, round the inverse RMS to the storage dtype, round the
normalized value, then multiply by the weight and round again. Different
parallel reduction orders can still produce small numerical differences.

CUDA configuration creation loads modules and resolves functions before graph
capture. RMSNorm uses the framework's existing device allocations and stream;
its first invocation can be captured without a separate RMSNorm warmup. Module
handles belong to each `CudaConfig` and remain alive through graph use. The
`CudaConfig::triton_available()` method reports whether that configuration has
loaded Triton kernels for its GPU.

## Validation

Run the GPU integration suite on a GPU matching the build architecture:

```bash
RUSTINFER_TRITON_PYTHON="$PWD/.venv/bin/python" \
  cargo test -p infer-backend-cuda --features triton --test triton_rmsnorm \
  -- --ignored --test-threads=1
```

Tests compare each dtype against an independent CPU oracle, cover odd dimensions
and the maximum Triton dimension, in-place operation, strided and oversized CUDA
fallback, invalid arguments, configuration lifetimes, and first-call graph
capture followed by replay with updated input. Tests assert Triton is available
so that a mismatched GPU cannot silently validate only the CUDA fallback.

This integration makes no performance claim. Benchmark the relevant shapes and
end-to-end workload before enabling it in a performance-sensitive deployment.

## Adding kernels

The source kernel is in
[`triton/rmsnorm.py`](../crates/infer-backend-cuda/triton/rmsnorm.py).
[`triton/compile.py`](../crates/infer-backend-cuda/triton/compile.py)
compiles cubins and a Rust manifest into Cargo's output directory. It verifies
the launch parameter ABI and rejects unsupported scratch, cooperative launch,
and cluster requirements. The internal
[`src/aot.rs`](../crates/infer-backend-cuda/src/aot.rs) module owns CUDA
Driver API loading and launching. Additional kernels need an explicit manifest,
validated launch contract, operator dispatch, and correctness/capture tests.
