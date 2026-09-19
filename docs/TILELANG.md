# TileLang kernels

The optional `tilelang` Cargo feature compiles RMSNorm with TileLang 0.1.14.
It uses the existing `MathOps::rmsnorm` and `MathOps::rmsnorm_inplace` interfaces.
The default build continues to use native CUDA.

## Build

Keep the normal CUDA prerequisites from the [README](../README.md). The initial
TileLang build environment is Linux x86_64 with Python 3.12 or newer:

```bash
uv sync --inexact --extra tilelang
RUSTINFER_TILELANG_PYTHON="$PWD/.venv/bin/python" \
  cargo build --release -p infer-worker --features tilelang
```

`--inexact` preserves separately installed CUDA build dependencies in the
Python environment.

Use `CUDA_ARCH=sm_89`, for example, to select an explicit target; otherwise the
build script detects the local GPU. The compiler accepts SM80 and newer targets
supported by the installed CUDA toolkit and pinned TileLang release. This
integration has been tested on SM89 only.

TileLang and its Python dependencies, including PyTorch, are needed at build
time. The compiled worker embeds cubins and needs no Python or TileLang runtime.
The compiler version is checked before code generation. The build uses TileLang
[`lower`](https://www.tilelang.com/autoapi/tilelang/tools/compile_only/index.html)
to produce CUDA source, then the CUDA toolkit's `nvcc` and `ptxas` to produce
cubins. Generated files stay in Cargo's output directory.

The `vllm-reference` Python extra pins a different TileLang compiler; install
it separately from the `tilelang` extra. To experiment
with both Rust AOT backends in one build:

```bash
uv sync --inexact --extra tilelang --extra triton
RUSTINFER_TILELANG_PYTHON="$PWD/.venv/bin/python" \
RUSTINFER_TRITON_PYTHON="$PWD/.venv/bin/python" \
  cargo build --release -p infer-worker --features tilelang,triton
```

## Dispatch and lifetime

TileLang handles contiguous FP32, FP16 and BF16 RMSNorm with dimensions 1–16384
on a GPU matching the compiled compute capability. Actual dimension and epsilon
remain runtime arguments; each dtype has power-of-two block specializations.
Indices use 64-bit arithmetic. Exact input/output aliasing is supported, and
half-precision arithmetic preserves the native CUDA rounding boundaries.

Dispatch priority with multiple features is [CuTe DSL](CUTE_DSL.md), TileLang,
Triton, then native CUDA. Unsupported shapes/layouts use the next eligible implementation.
The native CUDA alignment requirements and all validation rules described in
[the Triton guide](TRITON.md#dispatch-and-lifecycle) still apply. Runtime launch
errors propagate rather than silently retrying.

The compiler checks device parameter ordering, PTX argument types, thread count,
shared memory and alias annotations. Rust loads and resolves modules during
configuration creation, before CUDA Graph capture. The shared CUDA driver loader
in `src/aot.rs` owns each module until its configuration and graphs are finished.
`CudaConfig::tilelang_available()` reports whether TileLang modules loaded on
that GPU.

## Validation

For CUDA libraries installed as Python wheels, use the existing discovery script
before running the GPU tests:

```bash
export PATH="$PWD/.venv/bin:$PATH"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
RUSTINFER_TILELANG_PYTHON="$PWD/.venv/bin/python" \
  cargo test -p infer-backend-cuda --features tilelang --test tilelang_rmsnorm \
  -- --ignored --test-threads=1
```

The shared AOT suite compares all three dtypes with an independent CPU oracle,
and exercises odd dimensions, the largest supported dimension, in-place
execution, rounding boundaries, strided/oversized native fallback, invalid
arguments, independent configurations, and first-call graph capture/replay.
Run each backend feature separately to verify its dispatch path; a combined
build with `tilelang,triton` exercises TileLang first.

This initial integration covers RMSNorm only. End-to-end inference and
performance benchmarks are not part of this validation; no speedup is claimed.
