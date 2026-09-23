"""AOT gemm compiler; imported by compile.py after version/env checks."""
import json
import re
from pathlib import Path


def compile_gemm(arch, out_dir):
    import cutlass as c
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_ptr
    from block_gemm import gemm
    from block_embedding import FORMATS

    lines = ["// Generated CuTe DSL block GEMM manifest.",
             "const GEMM_SPECS: &[MatmulSpec] = &["]
    def ptr(dtype):
        return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=dtype.width // 8)
    for name, elements, size in FORMATS:
        for dtype_id, dtype in enumerate((c.Float32, c.Float16, c.BFloat16)):
            compiled = cute.compile(
                gemm, ptr(dtype), ptr(c.Uint8), ptr(dtype), ptr(dtype), ptr(c.Uint8),
                *[c.Int64(1) for _ in range(10)], name, elements, size,
                options=f"--gpu-arch={arch} --keep-ptx --keep-cubin --ptxas-options=--fmad=false",
            )
            ptx = compiled.artifacts.PTX
            if not ptx.lstrip().startswith("//"):
                ptx = Path(ptx).read_text()
            entries = re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
            if len(entries) != 1:
                raise RuntimeError("Expected one gemm entry")
            kernel_name, params = entries[0]
            types = re.findall(r"\.param\s+\.(\w+)", params)
            if types != ["u64"] * 15:
                raise RuntimeError(f"Unexpected gemm ABI: {types}")
            if not re.search(rf"\.reqntid\s+128,\s*1,\s*1", ptx) or ".shared" in ptx:
                raise RuntimeError("Unexpected gemm launch requirements")
            if re.search(r"\.target\s+(sm_\w+)", ptx)[1] != arch:
                raise RuntimeError("Unexpected gemm target")
            image = compiled.artifacts.CUBIN
            if isinstance(image, str):
                image = Path(image).read_bytes()
            if not isinstance(image, bytes) or not image.startswith(b"\x7fELF"):
                raise RuntimeError("Expected ELF cubin")
            path = out_dir / f"gemm_{name}_{dtype_id}.cubin"
            path.write_bytes(image)
            path.with_suffix(".ptx").write_text(ptx)
            lines.append(f'    MatmulSpec {{ format: F::{name}, dtype: {dtype_id}, threads: 128, '
                         f'image: include_bytes!({json.dumps(str(path))}), name: b"{kernel_name}\\0" }},')
            print(f"Compiled gemm {name}/{dtype_id}", flush=True)
    lines.append("];")
    (out_dir / "cute_gemm_kernels.rs").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = args.out_dir.resolve() / "gemm_artifacts"
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    os.environ["CUTE_DSL_NO_CACHE"] = "1"
    compile_gemm(args.arch, args.out_dir.resolve())
