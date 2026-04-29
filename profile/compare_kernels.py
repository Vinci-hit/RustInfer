#!/usr/bin/env python3
"""Compare GPU kernel times between RustInfer and vllm-omni profiles.

Assumption: both profiles contain 2-step Z-Image-Turbo DiT denoise at 256x256.
We normalize every kernel's total time by dividing by the number of "inference
iterations" represented in that profile, inferred from one *marker kernel* that
fires exactly once per denoise step in the DiT main layer.

Usage:
    compare_kernels.py <rustinfer.sqlite> <vllm.sqlite>
"""
import sqlite3
import sys
from collections import defaultdict


def load_kernels(path):
    """Return dict[name] -> (count, total_ns)."""
    conn = sqlite3.connect(path)
    q = (
        "SELECT s.value, COUNT(*), SUM(k.end - k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.demangledName "
        "GROUP BY s.value"
    )
    out = {}
    for name, cnt, total in conn.execute(q):
        out[name] = (cnt, total or 0)
    return out


def classify(name):
    ln = name.lower()
    if (
        "gemm" in ln
        or "cublas" in ln
        or "hgemm" in ln
        or "sm90_xmma" in ln
        or ln.startswith("nvjet_")
        or "cutlass3x" in ln and "gemm" in ln
    ):
        return "GEMM"
    if "flash" in ln or "fmha" in ln or "mha_" in ln or ("attention" in ln and "softmax" not in ln):
        return "ATTN"
    if "rmsnorm" in ln or "rms_norm" in ln:
        return "RMSNORM"
    if "rope" in ln:
        return "ROPE"
    if "swiglu" in ln or "silu" in ln:
        return "SWIGLU"
    if "split" in ln:
        return "SPLIT"
    if "softmax" in ln:
        return "SOFTMAX"
    if "broadcast" in ln or "mul" in ln and "gemm" not in ln:
        return "ELEM"
    if "memcpy" in ln or "memset" in ln:
        return "MEMCPY"
    if "conv2d" in ln or "cudnn" in ln or "implicit_gemm" in ln:
        return "CONV"
    if "adaln" in ln or "modulation" in ln or "post_attn" in ln or "gate_up" in ln or "zimage" in ln:
        return "FUSED"
    return "OTHER"


def infer_iterations(kernels_by_name, label):
    """Use any GEMM kernel as the "iteration marker" and compute iteration count.

    For RustInfer we know DiT has 30 main layers per step × ~5 GEMMs/layer ×
    2 steps = 300 GEMM calls per one 2-step run. So the ratio helps us
    find the number of runs captured.

    For vllm-omni the structure is the same (30 layers, 5 per-layer GEMMs),
    so we can use the same heuristic. But ambiguity → we instead just
    return the total number of GEMM calls, let the caller divide by 300.
    """
    total_gemm_calls = sum(cnt for n, (cnt, _) in kernels_by_name.items() if classify(n) == "GEMM")
    print(f"  [{label}] total GEMM kernel calls = {total_gemm_calls}")
    return total_gemm_calls


def agg_by_bucket(kernels, iters):
    buckets = defaultdict(lambda: [0, 0])  # [count, total_ms]
    for name, (cnt, ns) in kernels.items():
        b = classify(name)
        buckets[b][0] += cnt
        buckets[b][1] += ns / 1e6
    # Normalize everything per-2step-run. For RustInfer we *know* 1
    # 2-step run has ~300 GEMM calls from main dit layers (30*5*2) plus
    # context_refiner (2 layers) + noise_refiner (2 layers) + final +
    # cap embedder GEMMs. We approximate: one "full 2-step" ≈ 300-350 GEMMs.
    # Instead of hand-waving, we report *per-GEMM-call-average* as the
    # primary normalization, plus absolute totals.
    return dict(buckets)


def main():
    if len(sys.argv) != 3:
        print("usage: compare_kernels.py <rustinfer.sqlite> <vllm.sqlite>")
        sys.exit(1)
    ri_path, vl_path = sys.argv[1], sys.argv[2]

    print("=== loading ===")
    ri = load_kernels(ri_path)
    vl = load_kernels(vl_path)

    print("=== iteration inference ===")
    ri_gemm_calls = infer_iterations(ri, "rustinfer")
    vl_gemm_calls = infer_iterations(vl, "vllm")

    # Heuristic: one 2-step inference ≈ 300 main-layer GEMM calls.
    # So iters ≈ total_gemm_calls / 300 (rounded).
    MAIN_GEMMS_PER_2STEP = 300
    ri_iters = max(1, round(ri_gemm_calls / MAIN_GEMMS_PER_2STEP))
    vl_iters = max(1, round(vl_gemm_calls / MAIN_GEMMS_PER_2STEP))
    print(f"  inferred 2-step iterations: rustinfer={ri_iters}, vllm={vl_iters}")

    ri_buckets = agg_by_bucket(ri, ri_iters)
    vl_buckets = agg_by_bucket(vl, vl_iters)

    # Build union of buckets
    all_b = sorted(set(ri_buckets) | set(vl_buckets))

    print("\n" + "=" * 78)
    print(f"{'bucket':<10} | {'RI total ms':>12} {'RI /run ms':>11} | "
          f"{'VL total ms':>12} {'VL /run ms':>11} | {'delta /run ms':>14}")
    print("-" * 78)
    ri_total_per_run = 0.0
    vl_total_per_run = 0.0
    for b in all_b:
        ri_c, ri_ms = ri_buckets.get(b, (0, 0.0))
        vl_c, vl_ms = vl_buckets.get(b, (0, 0.0))
        ri_per = ri_ms / ri_iters
        vl_per = vl_ms / vl_iters
        delta = ri_per - vl_per
        ri_total_per_run += ri_per
        vl_total_per_run += vl_per
        print(f"{b:<10} | {ri_ms:12.2f} {ri_per:11.2f} | "
              f"{vl_ms:12.2f} {vl_per:11.2f} | {delta:+14.2f}")
    print("-" * 78)
    print(f"{'TOTAL':<10} | {'':>12} {ri_total_per_run:11.2f} | "
          f"{'':>12} {vl_total_per_run:11.2f} | "
          f"{ri_total_per_run - vl_total_per_run:+14.2f}")
    print("\nNOTE: '/run' means per one 2-step denoise inference.")
    print("      Deltas are positive when RustInfer is slower than vllm.\n")


if __name__ == "__main__":
    main()
