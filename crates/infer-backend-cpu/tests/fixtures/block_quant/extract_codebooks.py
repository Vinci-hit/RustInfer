#!/usr/bin/env python3
"""Extract only fixed format constants; never import/execute upstream code."""
import argparse
import ast
import hashlib
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("source", type=pathlib.Path, help="pinned gguf-py/gguf/quants.py")
args = parser.parse_args()
source = args.source.read_bytes()
assert hashlib.sha256(source).hexdigest() == "2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b"
destination = pathlib.Path(__file__).resolve().parents[4] / "infer-core/src/dtype/quant/codebooks.rs"
lines = [
    "//! Fixed codebooks are part of the GGUF encoding, not learned model weights.",
    "//! Extracted from gguf-py at c550d2f60bde72df19fcef1fef627895095b8ba8.",
    "//! See LICENSE-GGML and infer-backend-cpu/tests/fixtures/block_quant/README.md for provenance.",
]
names = ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"]
for cls in ast.parse(source).body:
    if not isinstance(cls, ast.ClassDef) or cls.name not in names:
        continue
    fields = {}
    for stmt in cls.body:
        if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in ["grid_map", "grid_hex", "grid_shape"]):
            fields[stmt.targets[0].id] = ast.literal_eval(stmt.value)
    levels = fields["grid_map"]
    bits = (len(levels) - 1).bit_length()
    per_byte = 8 // bits
    values = [levels[(byte >> (i * (8 // per_byte))) & ((1 << bits) - 1)]
              for byte in bytes.fromhex(fields["grid_hex"].decode()) for i in range(per_byte)]
    assert len(values) == fields["grid_shape"][0] * fields["grid_shape"][1]
    lines.append("#[rustfmt::skip]")
    lines.append(f"pub const {cls.name}: [u8; {len(values)}] = [")
    for i in range(0, len(values), 32):
        lines.append("    " + ", ".join(map(str, values[i:i + 32])) + ",")
    lines.append("];")
destination.write_text("\n".join(lines) + "\n")
