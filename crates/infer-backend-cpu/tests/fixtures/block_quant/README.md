# CPU block-quantized reference vectors

No llama.cpp executable, library, decoder, Python package or C/C++ toolchain
is used to run these tests. The backend is pure Rust.

`generate.py` uses Python's standard library to select logical quantized values,
scales and signs, computes FP32 expected results from those values, then packs
the values into GGUF-compatible bytes. It does not call the Rust decoder or
any external decoder. Each of the 13 `.bin` files has 64 cases: a little-endian
u32 case count, then repeated `(encoded block, FP32 expected block)` records.
`SHA256SUMS` records the generated files. Regenerate with:

```sh
python3 crates/infer-backend-cpu/tests/fixtures/block_quant/generate.py
cargo test -p infer-backend-cpu --test block_quant
```

Vectors include signed and unsigned bit planes, sub-block scale extrema,
negative/zero scales, half subnormals, normal and maximum finite half values,
and every IQ2/IQ3 codebook index. Other Rust tests check FP16 NaN/infinity,
unaligned bytes, multiple blocks/rows, sliced weights, strided IDs and tensors,
invalid dimensions/IDs/aliases and FP32/FP16/BF16 operators.

## Fixed codebook provenance

IQ2/IQ3 lookup tables are part of the on-disk format and must match its exact
constants. `infer-core/src/dtype/quant/codebooks.rs` contains only those constants, extracted
from `gguf-py/gguf/quants.py` at commit
`c550d2f60bde72df19fcef1fef627895095b8ba8` in the ggml-org/llama.cpp repository.
The source SHA256 is
`2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b`.
Its MIT license is preserved at `infer-core/src/dtype/quant/LICENSE-GGML`. The Rust
decoders follow the format's byte layouts; there is no runtime linkage or
comparison against llama.cpp. The test encoder shares the fixed codebook
constants, so these tests validate packing and arithmetic rather than providing
an independent verification of the codebooks themselves.

The optional maintainer script `extract_codebooks.py /path/to/quants.py` checks
the source hash and parses literal constants with Python `ast`; it does not
import or execute the source. Normal builds and vector generation need only the
checked-in tables. Fixed table payload sizes are 2,048 / 4,096 / 8,192 / 1,024 /
2,048 bytes (IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S).
