"""GGUF block embedding: one CTA per encoded block, FP32 reconstruction.

No floating-point weight matrix is materialized. Byte reads deliberately avoid
alignment assumptions (e.g. Q3_K has a 110-byte block stride).
"""
import cutlass as c
import cutlass.cute as cute

FORMATS = [
    ("Q2_K", 256, 84), ("Q3_K", 256, 110), ("Q4_K", 256, 144),
    ("Q5_K", 256, 176), ("Q6_K", 256, 210), ("Q8_0", 32, 34),
    ("IQ2_XXS", 256, 66), ("IQ2_XS", 256, 74), ("IQ2_S", 256, 82),
    ("IQ3_XXS", 256, 98), ("IQ3_S", 256, 110),
    ("IQ4_NL", 32, 18), ("IQ4_XS", 256, 136),
]
# Offsets into the shared format codebooks uploaded once per CUDA context.
GRID = {"IQ2_XXS": 0, "IQ2_XS": 2048, "IQ2_S": 6144,
        "IQ3_XXS": 14336, "IQ3_S": 15360, "IQ4": 17408}


@cute.jit
def byte(b, p):
    # CuTe pointer element inference can expose i8 even for a Uint8 pointer.
    # Mask after widening so packed bytes never inherit a sign extension.
    return b[p].to(c.Int32) & 255


@cute.jit
def word(b, p):
    return byte(b, p) | (byte(b, p + 1) << 8)


@cute.jit
def dword(b, p):
    return (word(b, p).to(c.Uint32) | (word(b, p + 2).to(c.Uint32) << 16))


@cute.jit
def half(b, p):
    return word(b, p).to(c.Uint16).bitcast(c.Float16).to(c.Float32)


@cute.jit
def signed8(v):
    return (v ^ 128) - 128


@cute.jit
def signs7(v):
    parity = v ^ (v >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    return v | ((parity & 1) << 7)


@cute.jit
def signed_value(value, mask, bit):
    return value * (1 - 2 * ((mask >> bit) & 1)).to(c.Float32)


@cute.jit
def decode(b, p, i, grids, fmt: c.Constexpr):
    value = c.Float32(0)
    if c.const_expr(fmt == "Q8_0"):
        value = half(b, p) * signed8(byte(b, p + 2 + i)).to(c.Float32)
    elif c.const_expr(fmt == "Q2_K"):
        s = byte(b, p + i // 16)
        q = (byte(b, p + 16 + (i // 128) * 32 + i % 32) >> (2 * ((i % 128) // 32))) & 3
        value = (half(b, p + 80) * (s & 15).to(c.Float32)) * q.to(c.Float32) - half(b, p + 82) * (s >> 4).to(c.Float32)
    elif c.const_expr(fmt == "Q3_K"):
        g = i // 16
        s = ((byte(b, p + 96 + g % 8) >> (4 * (g // 8))) & 15) | (((byte(b, p + 104 + g % 4) >> (2 * (g // 4))) & 3) << 4)
        lo = (byte(b, p + 32 + (i // 128) * 32 + i % 32) >> (2 * ((i % 128) // 32))) & 3
        hi = (byte(b, p + i % 32) >> (i // 32)) & 1
        value = (half(b, p + 108) * (s - 32).to(c.Float32)) * (lo - 4 * (1 - hi)).to(c.Float32)
    elif c.const_expr(fmt == "Q4_K" or fmt == "Q5_K"):
        g = i // 32
        s = c.Int32(0)
        m = c.Int32(0)
        if g < 4:
            s = byte(b, p + 4 + g) & 63
            m = byte(b, p + 8 + g) & 63
        else:
            s = (byte(b, p + 8 + g) & 15) | ((byte(b, p + g) >> 6) << 4)
            m = (byte(b, p + 8 + g) >> 4) | ((byte(b, p + 4 + g) >> 6) << 4)
        start = c.const_expr(48 if fmt == "Q5_K" else 16)
        q = (byte(b, p + start + (i // 64) * 32 + i % 32) >> (4 * ((i % 64) // 32))) & 15
        if c.const_expr(fmt == "Q5_K"):
            q = q | (((byte(b, p + 16 + i % 32) >> (i // 32)) & 1) << 4)
        value = (half(b, p) * s.to(c.Float32)) * q.to(c.Float32) - half(b, p + 2) * m.to(c.Float32)
    elif c.const_expr(fmt == "Q6_K"):
        lo = (byte(b, p + (i // 128) * 64 + i % 64) >> (4 * ((i % 128) // 64))) & 15
        hi = (byte(b, p + 128 + (i // 128) * 32 + i % 32) >> (2 * ((i % 128) // 32))) & 3
        value = (half(b, p + 208) * signed8(byte(b, p + 192 + i // 16)).to(c.Float32)) * ((lo | (hi << 4)) - 32).to(c.Float32)
    elif c.const_expr(fmt == "IQ4_NL"):
        q = (byte(b, p + 2 + i % 16) >> (4 * (i // 16))) & 15
        value = half(b, p) * signed8(byte(grids, GRID["IQ4"] + q)).to(c.Float32)
    elif c.const_expr(fmt == "IQ4_XS"):
        g = i // 32
        s = ((byte(b, p + 4 + g // 2) >> (4 * (g % 2))) & 15) | (((word(b, p + 2) >> (2 * g)) & 3) << 4)
        q = (byte(b, p + 8 + g * 16 + i % 16) >> (4 * ((i % 32) // 16))) & 15
        value = (half(b, p) * (s - 32).to(c.Float32)) * signed8(byte(grids, GRID["IQ4"] + q)).to(c.Float32)
    elif c.const_expr(fmt == "IQ2_XXS"):
        g = i // 32
        j = (i % 32) // 8
        aux = dword(b, p + 6 + g * 8)
        mask = signs7(((aux >> (7 * j)) & 127).to(c.Int32))
        grid = byte(b, p + 2 + g * 8 + j)
        d = half(b, p) * (c.Float32(0.5) + (aux >> 28).to(c.Float32)) * c.Float32(0.25)
        value = signed_value(d * byte(grids, GRID[fmt] + grid * 8 + i % 8).to(c.Float32), mask, i % 8)
    elif c.const_expr(fmt == "IQ2_XS" or fmt == "IQ2_S"):
        g = i // 8
        sg = i // 16
        if c.const_expr(fmt == "IQ2_XS"):
            q = word(b, p + 2 + 2 * g)
            grid = q & 511
            mask = signs7(q >> 9)
            s = (byte(b, p + 66 + sg // 2) >> (4 * (sg % 2))) & 15
        else:
            grid = byte(b, p + 2 + g) | (((byte(b, p + 66 + g // 4) >> (2 * (g % 4))) & 3) << 8)
            mask = byte(b, p + 34 + g)
            s = (byte(b, p + 74 + sg // 2) >> (4 * (sg % 2))) & 15
        d = half(b, p) * (c.Float32(0.5) + s.to(c.Float32)) * c.Float32(0.25)
        value = signed_value(d * byte(grids, GRID[fmt] + grid * 8 + i % 8).to(c.Float32), mask, i % 8)
    elif c.const_expr(fmt == "IQ3_XXS"):
        aux = dword(b, p + 66 + 4 * (i // 32))
        mask = signs7(((aux >> (7 * ((i % 32) // 8))) & 127).to(c.Int32))
        grid = byte(b, p + 2 + i // 4)
        d = half(b, p) * (c.Float32(0.5) + (aux >> 28).to(c.Float32)) * c.Float32(0.5)
        value = signed_value(d * byte(grids, GRID[fmt] + grid * 4 + i % 4).to(c.Float32), mask, i % 8)
    elif c.const_expr(fmt == "IQ3_S"):
        g = i // 4
        sg = i // 32
        grid = byte(b, p + 2 + g) | (((byte(b, p + 66 + g // 8) >> (g % 8)) & 1) << 8)
        s = (byte(b, p + 106 + sg // 2) >> (4 * (sg % 2))) & 15
        d = half(b, p) * (1 + 2 * s).to(c.Float32)
        value = signed_value(d * byte(grids, GRID[fmt] + grid * 4 + i % 4).to(c.Float32), byte(b, p + 74 + i // 8), i % 8)
    return value


@cute.kernel
def embedding_kernel(output: cute.Pointer, encoded: cute.Pointer, ids: cute.Pointer,
                     tables: cute.Pointer, cols: c.Int64, row_bytes: c.Int64,
                     id_stride: c.Int64, out_row_stride: c.Int64,
                     out_col_stride: c.Int64, vocab: c.Int64,
                     fmt: c.Constexpr, elements: c.Constexpr, block_bytes: c.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    # Tensor extents are descriptors only; host-side layout validation bounds
    # all actual accesses, and Int64 arithmetic supports weights larger than 4GiB.
    y = cute.make_tensor(output, cute.make_layout((c.Int64(9223372036854775807),)))
    b = cute.make_tensor(encoded, cute.make_layout((row_bytes * vocab,)))
    ids = cute.make_tensor(ids, cute.make_layout((c.Int64(9223372036854775807),)))
    grids = cute.make_tensor(tables, cute.make_layout((17424,)))
    blocks_per_row = cols // elements
    row = c.Int64(block) // blocks_per_row
    column_block = c.Int64(block) % blocks_per_row
    token = ids[row * id_stride].to(c.Int64)
    # Defensive guard as well as synchronous validation in the Rust entry point.
    if token >= 0 and token < vocab:
        p = token * row_bytes + column_block * block_bytes
        value = decode(b, p, lane, grids, fmt)
        col = column_block * elements + lane
        y[row * out_row_stride + col * out_col_stride] = value.to(output.value_type)


@cute.jit
def embedding(output: cute.Pointer, encoded: cute.Pointer, ids: cute.Pointer,
              tables: cute.Pointer, cols: c.Int64, row_bytes: c.Int64,
              id_stride: c.Int64, out_row_stride: c.Int64,
              out_col_stride: c.Int64, vocab: c.Int64,
              fmt: c.Constexpr, elements: c.Constexpr, block_bytes: c.Constexpr):
    embedding_kernel(output, encoded, ids, tables, cols, row_bytes, id_stride,
                     out_row_stride, out_col_stride, vocab, fmt, elements, block_bytes).launch(
                         grid=(1, 1, 1), block=(elements, 1, 1))
