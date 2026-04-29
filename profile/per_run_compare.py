#!/usr/bin/env python3
"""Normalize each profile by inferred # of 2-step denoise runs, then compare.

Inference of run-count:
    RI runs ≈ (flash_attn_gqa_* calls) / 60   (30 layers × 2 steps)
    VL runs ≈ (FlashAttnFwdSm90 calls) / 64   (32 layers × 2 steps)
"""
import sqlite3, sys
from collections import defaultdict


def load(path):
    c = sqlite3.connect(path)
    out = {}
    for name, cnt, total in c.execute(
        "SELECT s.value, COUNT(*), SUM(k.end-k.start) "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
        "GROUP BY s.value"):
        out[name] = (cnt, total or 0)
    return out


def ri_runs(kernels):
    for n, (cnt, _) in kernels.items():
        if "flash_attn_gqa_kernel_bf16_hdim128" in n:
            return cnt / 60.0
    return 1


def vl_runs(kernels):
    for n, (cnt, _) in kernels.items():
        if "FlashAttnFwdSm90" in n and cnt > 100:
            return cnt / 64.0
    return 1


def classify(name):
    ln = name.lower()
    if ("gemm" in ln or "cublas" in ln or "hgemm" in ln or "hgemv" in ln
        or "sm90_xmma" in ln or ln.startswith("nvjet_") or "splitkreduce" in ln):
        return "GEMM"
    if "flash_attn" in ln or "flashattnfwd" in ln or "fmha_cutlass" in ln:
        return "ATTN"
    if "rmsnorm" in ln or "rms_norm" in ln:
        return "RMSNORM"
    if "rope" in ln or "rotary" in ln:
        return "ROPE"
    if "swiglu" in ln or "silu" in ln:
        return "SWIGLU"
    if "split_cols" in ln:
        return "SPLIT"
    if "triton_" in ln:
        return "TRITON_EWISE"
    if "softmax" in ln:
        return "SOFTMAX"
    if "broadcast" in ln or "mul_bf16" in ln:
        return "ELEM"
    if "memcpy" in ln or "memset" in ln:
        return "MEMCPY"
    if "cudnn" in ln or "nchwtonhwc" in ln or "nhwctonchw" in ln or "implicit_gemm" in ln:
        return "CONV_CUDNN"
    if "groupnorm" in ln:
        return "GROUPNORM"
    if "tanh" in ln or "scalar_add" in ln:
        return "ELEM"
    if "fused_add_rmsnorm" in ln or "zimage" in ln:
        return "FUSED"
    return "OTHER"


def main():
    ri = load(sys.argv[1])
    vl = load(sys.argv[2])
    r_run = ri_runs(ri)
    v_run = vl_runs(vl)
    print(f"inferred runs: RI={r_run:.2f}, VL={v_run:.2f}")

    def agg(kernels):
        b = defaultdict(lambda: [0, 0.0])
        for n, (cnt, ns) in kernels.items():
            bk = classify(n)
            b[bk][0] += cnt
            b[bk][1] += ns / 1e6
        return b

    rb, vb = agg(ri), agg(vl)

    print("\n" + "=" * 78)
    print(f"{'bucket':<14} | {'RI /run ms':>11} | {'VL /run ms':>11} | {'delta ms':>10}")
    print("-" * 78)
    all_b = sorted(set(rb) | set(vb))
    r_sum = v_sum = 0.0
    for b in all_b:
        r = rb.get(b, [0, 0])[1] / r_run
        v = vb.get(b, [0, 0])[1] / v_run
        r_sum += r; v_sum += v
        print(f"{b:<14} | {r:11.3f} | {v:11.3f} | {r - v:+10.3f}")
    print("-" * 78)
    print(f"{'TOTAL':<14} | {r_sum:11.3f} | {v_sum:11.3f} | {r_sum - v_sum:+10.3f}")

    # detailed GEMM breakdown
    print("\n" + "=" * 78)
    print("GEMM kernel breakdown per-run (sorted by RI time desc)")
    print("-" * 78)
    ri_gemm = [(n, cnt, ns/1e6) for n, (cnt, ns) in ri.items() if classify(n) == "GEMM"]
    vl_gemm = [(n, cnt, ns/1e6) for n, (cnt, ns) in vl.items() if classify(n) == "GEMM"]
    ri_gemm.sort(key=lambda x: -x[2])
    # Print top RI GEMMs
    print(f"\n--- RustInfer GEMM kernels (per-run ms) ---")
    print(f"{'calls/run':>9} {'ms/run':>8} {'us/call':>8}  name")
    ri_total = 0.0
    for n, cnt, ms in ri_gemm[:15]:
        ms_r = ms / r_run
        us_c = ms * 1000.0 / cnt
        cr = cnt / r_run
        ri_total += ms_r
        print(f"{cr:9.1f} {ms_r:8.3f} {us_c:8.2f}  {n[:80]}")
    print(f"RI GEMM total (top-15): {ri_total:.2f} ms/run")
    print(f"RI GEMM total (ALL):    {sum(x[2] for x in ri_gemm)/r_run:.2f} ms/run")
    print(f"\n--- vllm GEMM kernels (per-run ms, top 15) ---")
    vl_gemm.sort(key=lambda x: -x[2])
    vl_total = 0.0
    for n, cnt, ms in vl_gemm[:15]:
        ms_r = ms / v_run
        us_c = ms * 1000.0 / cnt
        cr = cnt / v_run
        vl_total += ms_r
        print(f"{cr:9.1f} {ms_r:8.3f} {us_c:8.2f}  {n[:80]}")
    print(f"VL GEMM total (top-15): {vl_total:.2f} ms/run")
    print(f"VL GEMM total (ALL):    {sum(x[2] for x in vl_gemm)/v_run:.2f} ms/run")


if __name__ == "__main__":
    main()
