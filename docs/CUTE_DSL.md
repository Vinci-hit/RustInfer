# CuTe DSL kernels

The optional `cute-dsl` Cargo feature builds RMSNorm with NVIDIA CuTe DSL
(`nvidia-cutlass-dsl==4.7.1`). It implements the existing `MathOps::rmsnorm`
and `MathOps::rmsnorm_inplace` interfaces. Native CUDA remains the default.

## Build

Keep the normal CUDA build prerequisites from the [README](../README.md).
The initial Python build environment is Linux x86_64 with Python 3.12 or newer:

```bash
uv sync --inexact --extra cute-dsl
RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python" \
  cargo build --release -p infer-worker --features cute-dsl
```

`--inexact` preserves separately installed CUDA build dependencies. The
`vllm-reference` extra pins a different CuTe DSL version and must be installed
in a separate environment.

Set `CUDA_ARCH=sm_89` to select a target explicitly; otherwise the build detects
the local GPU. Targets must be SM80 or newer and supported by the pinned
compiler. Only SM89 has been tested for this integration.

The compiler calls `cute.compile` with typed pointers and runtime scalar
arguments, without executing a Python kernel. It uses CuTe DSL's
[PTX and cubin export options](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/debugging.html).
Generated artifacts stay under Cargo's output directory. The worker embeds
cubins and needs no Python or CuTe DSL installation at runtime.

## Dispatch and lifetime

CuTe DSL handles contiguous FP32, FP16 and BF16 rows with dimensions 1–16384
on a matching GPU. Each row uses one warp, a runtime loop, FP32 accumulation
and 64-bit offsets. FP16/BF16 preserve the native CUDA intermediate rounding
boundaries. Exact input/output aliasing is supported.

When multiple features are enabled, priority is CuTe DSL, TileLang, Triton,
then native CUDA. Unsupported shapes/layouts use the next eligible backend.
Existing validation and native fallback alignment rules still apply; launch
errors propagate. Enable all three with `--features cute-dsl,tilelang,triton`
and set each backend's Python interpreter environment variable.

The build checks the compiler version, target, PTX parameter types, thread
count and absence of shared-memory requirements. The Rust driver loader
resolves all functions during configuration creation before graph capture,
and keeps modules alive until their configuration and graphs are finished.
`CudaConfig::cute_dsl_available()` reports whether its kernels loaded.

## Validation

For CUDA libraries installed as Python wheels:

```bash
export PATH="$PWD/.venv/bin:$PATH"
source scripts/lib/cuda_env.sh
rustinfer_discover_cuda_libraries
RUSTINFER_CUTE_DSL_PYTHON="$PWD/.venv/bin/python" \
  cargo test -p infer-backend-cuda --features cute-dsl --test cute_dsl_rmsnorm \
  -- --ignored --test-threads=1
```

The shared AOT GPU suite checks all three dtypes against an independent CPU
oracle, dimension boundaries, in-place execution, half rounding boundaries,
strided/oversized fallback, invalid contracts, independent configurations,
and first-call CUDA Graph capture followed by replay with updated input.
Run each backend separately to exercise its dispatch path; a build with all
three features exercises CuTe DSL first.

This is an initial RMSNorm integration. The one-warp implementation has not
been tuned for performance, and end-to-end model inference has not been
validated. No speedup is claimed.
