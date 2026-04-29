#!/usr/bin/env python3
"""Compare RustInfer vs vllm kernel times, normalized by flash-attention call count.

flash-attention is invoked exactly once per DiT layer.forward, so counting
those calls gives us an unambiguous measure of "how many layer-forwards did
each profile capture". We then report per-layer-forward averages — that's the
truly apples-to-apples number.

Usage: compare_per_layer.py <ri.sqlite> <vl.sqlite>
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


def attn_calls(kernels):
    """Count the *compute-bound* attention forward kernel exactly once per
    DiT-layer-forward. For our RustInfer build that's `flash_attn_gqa_*`,
    for vllm-omni that's `FlashAttnFwdSm90` (the sm90 cutlass FA3 kernel)."""
    total = 0
    for n, (cnt, _) in kernels.items():
        ln = n.lower()
        if (
            "flash_attn_gqa_kernel" in ln
            or "flashattnfwdsm90" in ln
            or "flashattnfwd" in ln
        ):
            total += cnt
    return total


def classify(name):
    ln = name.lower()
    if ("gemm" in ln or "cublas" in ln or "hgemm" in ln or "hgemv" in ln
        or "sm90_xmma" in ln or ln.startswith("nvjet_")):
        # cuBLASLt splitK reduce is epilogue to GEMM
        return "GEMM"
    if "flash_attn" in ln or "flashattnfwd" in ln:
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
        return "TRITON_FUSED"
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
    if "rope_rotate" in ln:
        return "ROPE"
    if "tanh" in ln or "scalar_add" in ln:
        return "ELEM"
    if "fused_add_rmsnorm" in ln or "fused_" in ln or "zimage" in ln:
        return "FUSED"
    return "OTHER"


def main():
    ri_p, vl_p = sys.argv[1], sys.argv[2]
    ri, vl = load(ri_p), load(vl_p)

    ri_a, vl_a = attn_calls(ri), attn_calls(vl)
    print(f"attention calls: RI={ri_a}, VL={vl_a}")

    # bucket totals
    def agg(kernels):
        b = defaultdict(lambda: [0, 0.0])
        for n, (cnt, ns) in kernels.items():
            bk = classify(n)
            b[bk][0] += cnt
            b[bk][1] += ns / 1e6
        return b

    ri_b, vl_b = agg(ri), agg(vl)

    # per-DiT-layer-forward (one attention call = one layer forward)
    print("\n" + "=" * 82)
    print(f"{'bucket':<14} | {'RI total ms':>10} {'RI/layer us':>11} | "
          f"{'VL total ms':>10} {'VL/layer us':>11} | {'delta us/layer':>14}")
    print("-" * 82)
    all_b = sorted(set(ri_b) | set(vl_b))
    ri_sum = vl_sum = 0.0
    for b in all_b:
        ri_total_ms = ri_b.get(b, [0, 0])[1]
        vl_total_ms = vl_b.get(b, [0, 0])[1]
        ri_us_per = ri_total_ms * 1000.0 / max(ri_a, 1)
        vl_us_per = vl_total_ms * 1000.0 / max(vl_a, 1)
        delta = ri_us_per - vl_us_per
        ri_sum += ri_us_per
        vl_sum += vl_us_per
        print(f"{b:<14} | {ri_total_ms:10.2f} {ri_us_per:11.2f} | "
              f"{vl_total_ms:10.2f} {vl_us_per:11.2f} | {delta:+14.2f}")
    print("-" * 82)
    print(f"{'SUM':<14} | {'':>10} {ri_sum:11.2f} | "
          f"{'':>10} {vl_sum:11.2f} | {ri_sum - vl_sum:+14.2f}")

    # assuming ~32 layer-forwards per 2-step run (30 main + 2 ctx/noise), show
    # what the per-2step-run totals would be
    LAYERS_PER_2STEP = 60  # 30 main layers × 2 steps
    print(f"\nextrapolated to 60 main-layer-forwards per 2-step run:")
    print(f"  RI sum = {ri_sum * LAYERS_PER_2STEP / 1000:.1f} ms "
          f"/ 2-step")
    print(f"  VL sum = {vl_sum * LAYERS_PER_2STEP / 1000:.1f} ms "
          f"/ 2-step")


if __name__ == "__main__":
    main()
