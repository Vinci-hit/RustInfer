"""RMSNorm with runtime dimensions; compiled to CUDA at build time."""

import tilelang.language as T


def rmsnorm(dtype, block):
    @T.prim_func
    def kernel(
        output: T.handle, input: T.handle, weight: T.handle,
        dim: T.int32, eps: T.float32,
    ):
        # The launch grid supplies the actual row count. Use int64 indexing
        # so dense tensors with more than 2**31 elements remain addressable.
        X = T.match_buffer(input, (T.int64(2147483647) * dim,), dtype)
        W = T.match_buffer(weight, (dim,), dtype)
        Y = T.match_buffer(output, (T.int64(2147483647) * dim,), dtype)
        # Preserve exact input/output aliasing through device lowering.
        T.func_attr({"tl.non_restrict_params": [X.data, Y.data]})
        with T.Kernel(2147483647, threads=128) as row:
            values = T.alloc_fragment((block,), "float32")
            squares = T.alloc_fragment((block,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for col in T.Parallel(block):
                values[col] = T.if_then_else(
                    col < dim,
                    T.cast(X[T.cast(row, "int64") * dim + col], "float32"),
                    T.float32(0),
                )
                squares[col] = values[col] * values[col]
            T.reduce_sum(squares, total, dim=0)
            for col in T.Parallel(block):
                if col < dim:
                    offset = T.cast(row, "int64") * dim + col
                    inverse = T.rsqrt(total[0] / T.cast(dim, "float32") + eps)
                    w = T.cast(W[col], "float32")
                    if dtype == "float32":
                        Y[offset] = (values[col] * w) * inverse
                    else:
                        # Match native CUDA's two intermediate half roundings.
                        scale = T.cast(T.cast(inverse, dtype), "float32")
                        normalized = T.cast(values[col] * scale, dtype)
                        Y[offset] = T.cast(normalized, "float32") * w
    return kernel
