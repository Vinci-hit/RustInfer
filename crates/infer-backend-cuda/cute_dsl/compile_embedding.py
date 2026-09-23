"""AOT embedding compiler; imported by compile.py after version/env checks."""
import json
import re
from pathlib import Path


def compile_embedding(arch, out_dir):
    import cutlass as c
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from block_embedding import embedding, FORMATS, GRID

    source = Path(__file__).resolve().parents[2] / "infer-core/src/dtype/quant/codebooks.rs"
    constants = {name: bytes(map(int, re.findall(r"\d+", body)))
                 for name, body in re.findall(r"const (IQ\w+): \[u8; \d+\] = \[(.*?)\];", source.read_text(), re.S)}
    tables = bytearray()
    for name in ["IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S"]:
        assert len(tables) == GRID[name]
        tables += constants[name]
    assert len(tables) == GRID["IQ4"]
    tables += bytes(v & 255 for v in [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113])
    table_file = out_dir / "block_codebooks.bin"
    table_file.write_bytes(tables)
    lines = ["// Generated CuTe DSL block embedding manifest.",
             f"const TARGET_SM: i32 = {int(re.fullmatch(r'sm_([0-9]+)a?', arch)[1])};",
             f"const CODEBOOKS: &[u8] = include_bytes!({json.dumps(str(table_file))});",
             "const SPECS: &[EmbeddingSpec] = &["]
    def ptr(dtype):
        return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=dtype.width // 8)
    for name, elements, size in FORMATS:
        for dtype_id, dtype in enumerate((c.Float32, c.Float16, c.BFloat16)):
            compiled = cute.compile(
                embedding, ptr(dtype), ptr(c.Uint8), ptr(c.Int32), ptr(c.Uint8),
                *[c.Int64(1) for _ in range(6)], name, elements, size,
                options=f"--gpu-arch={arch} --keep-ptx --keep-cubin --ptxas-options=--fmad=false",
            )
            ptx = compiled.artifacts.PTX
            if not ptx.lstrip().startswith("//"):
                ptx = Path(ptx).read_text()
            entries = re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
            if len(entries) != 1:
                raise RuntimeError("Expected one embedding entry")
            kernel_name, params = entries[0]
            types = re.findall(r"\.param\s+\.(\w+)", params)
            if types != ["u64"] * 10:
                raise RuntimeError(f"Unexpected embedding ABI: {types}")
            if not re.search(rf"\.reqntid\s+{elements},\s*1,\s*1", ptx) or ".shared" in ptx:
                raise RuntimeError("Unexpected embedding launch requirements")
            if re.search(r"\.target\s+(sm_\w+)", ptx)[1] != arch:
                raise RuntimeError("Unexpected embedding target")
            image = compiled.artifacts.CUBIN
            if isinstance(image, str):
                image = Path(image).read_bytes()
            if not isinstance(image, bytes) or not image.startswith(b"\x7fELF"):
                raise RuntimeError("Expected ELF cubin")
            path = out_dir / f"embedding_{name}_{dtype_id}.cubin"
            path.write_bytes(image)
            path.with_suffix(".ptx").write_text(ptx)
            lines.append(f'    EmbeddingSpec {{ format: F::{name}, dtype: {dtype_id}, threads: {elements}, '
                         f'image: include_bytes!({json.dumps(str(path))}), name: b"{kernel_name}\\0" }},')
            print(f"Compiled embedding {name}/{dtype_id}", flush=True)
    lines.append("];")
    (out_dir / "cute_embedding_kernels.rs").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = args.out_dir.resolve() / "embedding_artifacts"
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    os.environ["CUTE_DSL_NO_CACHE"] = "1"
    compile_embedding(args.arch, args.out_dir.resolve())
