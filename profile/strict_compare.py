#!/usr/bin/env python3
"""Strict per-op comparison of RustInfer vs vllm at identical DiT GEMM shapes.

Z-Image 2-step main-layer GEMMs (seq = 384 padded = 128 img + 256 cap, dim=3840,
hidden=10240):

  op           M    N       K
  to_qkv      384   11520   3840
  to_out      384   3840    3840
  w1          384   10240   3840
  w3          384   10240   3840
  w2          384   3840    10240
  adaln         1   15360   256

Each of these should appear N_CALLS = 60 = 30 main layers * 2 denoise steps,
except adaln which is (30 main + 2 refiner) * 2 = 64.

We identify each op in a profile by its launch 'block_x' (always 384) AND by
matching grid dims to output tiles typical of its shape. Then we sum total time
and compute per-call avg — the same op across runs can be directly compared.
"""
import sqlite3
import sys
from collections import defaultdict


def load_gemm(path):
    c = sqlite3.connect(path)
    q = """
        SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ,
               (k.end - k.start) AS dur
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
    """
    rows = []
    for name, gx, gy, gz, bx, by, bz, dur in c.execute(q):
        if dur is None or dur <= 0:
            continue
        lname = name.lower()
        is_gemm = (
            "gemm" in lname or "cublas" in lname or "hgemm" in lname
            or "sm90_xmma" in lname or "sm80_xmma" in lname
            or lname.startswith("nvjet_")
        )
        if not is_gemm:
            continue
        rows.append((name, gx, gy, gz, bx, by, bz, dur))
    return rows


def classify(gx, gy, tile_m, tile_n, M, N):
    """Return True if a launch with (gx, gy) plausibly corresponds to
    output tile [tile_m, tile_n] covering [M, N]. cuBLASLt stores tile
    dims in the kernel name; grid = ceil(M / tile_m) x ceil(N / tile_n)
    in some order. We accept either order."""
    import math
    want_gx1 = math.ceil(M / tile_m)
    want_gy1 = math.ceil(N / tile_n)
    want_gx2 = math.ceil(N / tile_n)
    want_gy2 = math.ceil(M / tile_m)
    return ((gx == want_gx1 and gy == want_gy1)
            or (gx == want_gx2 and gy == want_gy2))


def aggregate_by_op(rows):
    """Group GEMM launches by (gx, gy) + extract tile size from kernel name
    where possible. Return dict (gx, gy) -> (cnt, total_ns, [names])."""
    bucket = defaultdict(lambda: [0, 0, set()])
    for name, gx, gy, gz, bx, by, bz, dur in rows:
        if bz != 1 or gz != 1:
            continue
        bucket[(gx, gy)][0] += 1
        bucket[(gx, gy)][1] += dur
        bucket[(gx, gy)][2].add(name[:70])
    return bucket


def main():
    ours = load_gemm(sys.argv[1])
    ref = load_gemm(sys.argv[2])

    print(f"=== {sys.argv[1]}: {len(ours)} GEMM launches ===")
    print(f"=== {sys.argv[2]}: {len(ref)} GEMM launches ===\n")

    o = aggregate_by_op(ours)
    r = aggregate_by_op(ref)

    # Print every op bucket in `ours` sorted by total time.
    print("Our hot GEMM launch shapes (counts × grids) and what they look like in vllm:\n")
    ranked = sorted(o.items(), key=lambda kv: kv[1][1], reverse=True)
    for (gx, gy), (cnt, dur, names) in ranked[:15]:
        avg_us = dur / cnt / 1e3
        name1 = next(iter(names))
        print(f"  ours  grid=({gx},{gy})  n={cnt:4d}  avg={avg_us:7.2f} us  tot={dur/1e6:6.2f} ms  {name1}")
        # match in vllm.
        if (gx, gy) in r:
            rc, rd, rn = r[(gx, gy)]
            ravg = rd / rc / 1e3
            name2 = next(iter(rn))
            print(f"  vllm  grid=({gx},{gy})  n={rc:4d}  avg={ravg:7.2f} us  tot={rd/1e6:6.2f} ms  {name2}")
            if ravg > 0:
                print(f"         RATIO ours/vllm = {avg_us/ravg:5.2f}x")
        else:
            print(f"  vllm  grid=({gx},{gy})  — NOT PRESENT (vllm picked a different tile for this shape)")
        print()


if __name__ == "__main__":
    main()
