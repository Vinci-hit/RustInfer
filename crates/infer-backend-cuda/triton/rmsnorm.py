"""Dense RMSNorm, compiled ahead of time by compile.py (no Python at runtime)."""

import triton
import triton.language as tl


@triton.jit
def rmsnorm(output, input, weight, dim, eps, BLOCK: tl.constexpr):
    # Cast before multiplying: a valid dense tensor can exceed 2**31 elements.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    offsets = row * dim + cols
    values = tl.load(input + offsets, mask=cols < dim, other=0).to(tl.float32)
    weights = tl.load(weight + cols, mask=cols < dim, other=0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / dim + eps)

    # Match the existing CUDA kernels' rounding and multiplication order. Half
    # kernels round both the scale and normalized input to the element dtype.
    if input.dtype.element_ty == tl.float32:
        result = (values * weights) * inv_rms
    else:
        scale = inv_rms.to(input.dtype.element_ty).to(tl.float32)
        normalized = (values * scale).to(input.dtype.element_ty).to(tl.float32)
        result = normalized * weights
    tl.store(output + offsets, result, mask=cols < dim)
