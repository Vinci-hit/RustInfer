"""Single-token block-quantized Linear, four independent row warps per CTA.

Decode directly into registers; no dense weight temporary, atomics, shared
memory, or host synchronization. Reduction and bias addition use FP32.
"""
import cutlass as c
import cutlass.cute as cute
from block_embedding import decode


@cute.kernel
def gemv_kernel(output: cute.Pointer, encoded: cute.Pointer, input: cute.Pointer,
                bias: cute.Pointer, tables: cute.Pointer,
                cols: c.Int64, rows: c.Int64, row_bytes: c.Int64,
                input_stride: c.Int64, output_stride: c.Int64,
                bias_stride: c.Int64, has_bias: c.Int64,
                fmt: c.Constexpr, elements: c.Constexpr, block_bytes: c.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = thread % 32
    row = c.Int64(block) * 4 + thread // 32
    extent = cute.make_layout((c.Int64(9223372036854775807),))
    x = cute.make_tensor(input, extent)
    y = cute.make_tensor(output, extent)
    bias = cute.make_tensor(bias, extent)
    b = cute.make_tensor(encoded, cute.make_layout((rows * row_bytes,)))
    grids = cute.make_tensor(tables, cute.make_layout((17424,)))
    # Predicate is uniform within each warp, including the final partial CTA.
    if row < rows:
        total = c.Float32(0)
        for col in range(c.Int64(lane), cols, 32):
            p = row * row_bytes + (col // elements) * block_bytes
            value = decode(b, p, (col % elements).to(c.Int32), grids, fmt)
            total = total + value * x[col * input_stride].to(c.Float32)
        total = cute.arch.warp_reduction_sum(total)
        if lane == 0:
            if has_bias != 0:
                total = total + bias[row * bias_stride].to(c.Float32)
            y[row * output_stride] = total.to(output.value_type)


@cute.jit
def gemv(output: cute.Pointer, encoded: cute.Pointer, input: cute.Pointer,
         bias: cute.Pointer, tables: cute.Pointer,
         cols: c.Int64, rows: c.Int64, row_bytes: c.Int64,
         input_stride: c.Int64, output_stride: c.Int64,
         bias_stride: c.Int64, has_bias: c.Int64,
         fmt: c.Constexpr, elements: c.Constexpr, block_bytes: c.Constexpr):
    gemv_kernel(output, encoded, input, bias, tables, cols, rows, row_bytes,
                input_stride, output_stride, bias_stride, has_bias,
                fmt, elements, block_bytes).launch(grid=(1, 1, 1), block=(128, 1, 1))
