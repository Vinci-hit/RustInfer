"""One warp per RMSNorm row, with runtime dimension and in-place support."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def rmsnorm_kernel(output: cute.Pointer, input: cute.Pointer, weight: cute.Pointer,
                   dim: cutlass.Int32, eps: cutlass.Float32):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    # Scalar alignment is intentional: short/odd rows need no vector alignment.
    x = cute.make_tensor(input, cute.make_layout((cutlass.Int64(2147483647) * dim,)))
    y = cute.make_tensor(output, cute.make_layout((cutlass.Int64(2147483647) * dim,)))
    w = cute.make_tensor(weight, cute.make_layout((dim,)))
    base = cutlass.Int64(row) * dim
    total = cutlass.Float32(0)
    for col in range(lane, dim, 32):
        value = x[base + col].to(cutlass.Float32)
        total = total + value * value
    total = cute.arch.warp_reduction_sum(total)
    inverse = cute.math.rsqrt(total / cutlass.Float32(dim) + eps)
    # Every lane finishes reading the row for reduction before any output write.
    cute.arch.sync_warp()
    for col in range(lane, dim, 32):
        value = x[base + col].to(cutlass.Float32)
        scale = w[col].to(cutlass.Float32)
        if cutlass.const_expr(output.value_type == cutlass.Float32):
            y[base + col] = (value * scale) * inverse
        else:
            rounded_inverse = inverse.to(output.value_type).to(cutlass.Float32)
            normalized = (value * rounded_inverse).to(output.value_type).to(cutlass.Float32)
            y[base + col] = (normalized * scale).to(output.value_type)


@cute.jit
def rmsnorm(output: cute.Pointer, input: cute.Pointer, weight: cute.Pointer,
            dim: cutlass.Int32, eps: cutlass.Float32):
    rmsnorm_kernel(output, input, weight, dim, eps).launch(grid=(1, 1, 1), block=(32, 1, 1))
