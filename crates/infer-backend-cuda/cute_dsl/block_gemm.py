"""Fused SIMT quantized GEMM: an 8-token by 4-output tile per CTA.

Each warp owns one output channel, decodes each weight once, and reuses it
across eight independent FP32 accumulators. No dense weight temporary.
"""
import cutlass as c
import cutlass.cute as cute
from block_embedding import decode


@cute.kernel
def gemm_kernel(output: cute.Pointer, encoded: cute.Pointer, input: cute.Pointer,
                bias: cute.Pointer, tables: cute.Pointer,
                cols: c.Int64, rows: c.Int64, row_bytes: c.Int64,
                input_stride: c.Int64, output_stride: c.Int64,
                bias_stride: c.Int64, has_bias: c.Int64,
                tokens: c.Int64, input_row_stride: c.Int64, output_row_stride: c.Int64,
                fmt: c.Constexpr, elements: c.Constexpr, block_bytes: c.Constexpr):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = thread % 32
    n_tiles = (rows + 3) // 4
    row = (c.Int64(block) % n_tiles) * 4 + thread // 32
    first_token = (c.Int64(block) // n_tiles) * 8
    extent = cute.make_layout((c.Int64(9223372036854775807),))
    x = cute.make_tensor(input, extent)
    y = cute.make_tensor(output, extent)
    bias = cute.make_tensor(bias, extent)
    b = cute.make_tensor(encoded, cute.make_layout((rows * row_bytes,)))
    grids = cute.make_tensor(tables, cute.make_layout((17424,)))
    # Row and token guards are warp uniform: no partial-warp reductions.
    if row < rows:
        totals = cute.make_rmem_tensor((8,), c.Float32)
        totals.fill(c.Float32(0))
        for col in range(c.Int64(lane), cols, 32):
            p = row * row_bytes + (col // elements) * block_bytes
            value = decode(b, p, (col % elements).to(c.Int32), grids, fmt)
            for t in c.range_constexpr(8):
                token = first_token + t
                if token < tokens:
                    activation = x[token * input_row_stride + col * input_stride].to(c.Float32)
                    totals[t] = totals[t] + value * activation
        for t in c.range_constexpr(8):
            token = first_token + t
            if token < tokens:
                total = cute.arch.warp_reduction_sum(totals[t])
                if lane == 0:
                    if has_bias != 0:
                        total = total + bias[row * bias_stride].to(c.Float32)
                    y[token * output_row_stride + row * output_stride] = total.to(output.value_type)


@cute.jit
def gemm(output: cute.Pointer, encoded: cute.Pointer, input: cute.Pointer,
         bias: cute.Pointer, tables: cute.Pointer,
         cols: c.Int64, rows: c.Int64, row_bytes: c.Int64,
         input_stride: c.Int64, output_stride: c.Int64,
         bias_stride: c.Int64, has_bias: c.Int64,
         tokens: c.Int64, input_row_stride: c.Int64, output_row_stride: c.Int64,
         fmt: c.Constexpr, elements: c.Constexpr, block_bytes: c.Constexpr):
    gemm_kernel(output, encoded, input, bias, tables, cols, rows, row_bytes,
                input_stride, output_stride, bias_stride, has_bias,
                tokens, input_row_stride, output_row_stride,
                fmt, elements, block_bytes).launch(grid=(1, 1, 1), block=(128, 1, 1))
