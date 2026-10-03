"""Reference fixtures/manifests from the pinned upstream GGUF implementation.

Run with gguf-py at llama.cpp c550d2f60bde72df19fcef1fef627895095b8ba8
on PYTHONPATH. No downloads or third-party code execution occur in Rust tests.
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np

REVISION = "c550d2f60bde72df19fcef1fef627895095b8ba8"


def manifest(path):
    reader = gguf.GGUFReader(path)
    raw = np.memmap(path, mode="r", dtype=np.uint8)
    tensors = []
    for t in reader.tensors:
        samples = []
        for offset in sorted({0, t.n_bytes // 2, max(0, t.n_bytes - 16)}):
            end = min(offset + 16, t.n_bytes)
            samples.append({"offset": offset, "bytes": raw[t.data_offset + offset:t.data_offset + end].tolist()})
        tensors.append({"name": t.name, "type": int(t.tensor_type),
                        "dimensions": t.shape.tolist(), "offset": t.data_offset,
                        "bytes": t.n_bytes, "samples": samples})
    return {"upstream": REVISION, "file": str(path.resolve()),
            "version": reader.fields["GGUF.version"].contents(),
            "alignment": reader.alignment, "data_offset": reader.data_offset,
            "metadata": [{"key": k, "types": [int(t) for t in v.types], "value": v.contents()}
                         for k, v in reader.fields.items() if not k.startswith("GGUF.")],
            "tensors": tensors,
            "layouts": [{"id": int(t), "name": t.name, "elements": b, "bytes": s}
                        for t, (b, s) in gguf.GGML_QUANT_SIZES.items()]}


def fixture(path):
    writer = gguf.GGUFWriter(path, "test")
    for typ, val in [(gguf.GGUFValueType.UINT8, 255), (gguf.GGUFValueType.INT8, -128),
                     (gguf.GGUFValueType.UINT16, 65535), (gguf.GGUFValueType.INT16, -32768),
                     (gguf.GGUFValueType.UINT32, 2**32 - 1), (gguf.GGUFValueType.INT32, -2**31),
                     (gguf.GGUFValueType.FLOAT32, 1.25), (gguf.GGUFValueType.BOOL, True),
                     (gguf.GGUFValueType.STRING, "视觉\x00GGUF"),
                     (gguf.GGUFValueType.UINT64, 2**64 - 1), (gguf.GGUFValueType.INT64, -2**63),
                     (gguf.GGUFValueType.FLOAT64, -2.5)]:
        writer.add_key_value("test." + typ.name.lower(), val, typ)
        writer.add_key_value("array." + typ.name.lower(), [val, val], gguf.GGUFValueType.ARRAY, typ)
    for typ, (block, size) in gguf.GGML_QUANT_SIZES.items():
        # Deterministic raw payload, not intended to be meaningful quantized numbers.
        data = np.arange(3 * 2 * size, dtype=np.uint8).reshape(3, 2 * size)
        writer.add_tensor("test." + typ.name.lower(), data, raw_dtype=typ)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Produce a manifest for an existing model")
    parser.add_argument("--output", type=Path, help="Output manifest JSON")
    args = parser.parse_args()
    path = args.model or Path(__file__).with_name("reference.gguf")
    if args.model is None:
        fixture(path)
    output = args.output or path.with_suffix(".json")
    reference = manifest(path)
    if args.model is None:
        reference["file"] = path.name
    output.write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    print(output)
