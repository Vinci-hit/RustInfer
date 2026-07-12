"""Plot RustInfer vs vLLM online QPS sweep — full metric grid (p50 + p99).

Reads /tmp/bench_qps_rustinfer.json and /tmp/bench_qps_vllm.json
(written by bench_online_qps.py) and renders a 3x3 grid:

  row 1: TTFT  p50 / TTFT  p99 / output throughput
  row 2: TPOT  p50 / TPOT  p99 / request throughput
  row 3: ITL   p50 / ITL   p99 / E2E p99 latency

Each curve = one target, x = QPS rate.
Output: bench/plots/ri_vs_vllm_qps_full.png  (+ a printed text table)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "ri_vs_vllm_qps_final.png")

TARGETS = [
    ("RustInfer", "/tmp/bench_qps_rustinfer.json", "#d62728", "o"),
    ("vLLM",      "/tmp/bench_qps_vllm.json",      "#1f77b4", "s"),
]


def load(path):
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    with open(path) as f:
        d = json.load(f)
    rows = [m for m in d["results"] if m]  # drop empty (all-failed) qps points
    return rows


def series(rows, key):
    xs = [r["qps"] for r in rows if key in r]
    ys = [r[key] for r in rows if key in r]
    return xs, ys


data = {name: load(path) for name, path, _, _ in TARGETS}

PANELS = [
    ("median_ttft_ms", "TTFT p50 (ms)", "lower = faster first token"),
    ("p99_ttft_ms",    "TTFT p99 (ms)", "tail first-token latency"),
    ("median_tpot_ms", "TPOT p50 (ms)", "per-output-token latency"),
    ("p99_tpot_ms",    "TPOT p99 (ms)", "tail per-token latency"),
    ("median_itl_ms",  "ITL p50 (ms)", "inter-token latency"),
    ("p99_itl_ms",     "ITL p99 (ms)", "tail inter-token latency"),
    ("median_e2e_latency_ms", "E2E p50 latency (ms)", "median end-to-end"),
    ("p99_e2e_latency_ms",    "E2E p99 latency (ms)", "tail end-to-end"),
    ("output_throughput", "Output throughput (tok/s)", "higher = better"),
]

fig, axes = plt.subplots(3, 3, figsize=(17, 13))
axes = axes.flatten()

for ax, (key, title, sub) in zip(axes, PANELS):
    for name, _, color, marker in TARGETS:
        rows = data.get(name)
        if not rows:
            continue
        xs, ys = series(rows, key)
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=name,
                    linewidth=2, markersize=7)
    ax.set_title(f"{title}\n{sub}", fontsize=11)
    ax.set_xlabel("QPS (Poisson arrival rate)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle(
    "RustInfer vs vLLM — online QPS sweep (Qwen3-4B, H200, max_tokens=512, "
    "ignore_eos, matched CUDA-graph decode capture sizes)",
    fontsize=13, y=0.997,
)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(OUT, dpi=110)
print(f"\nSaved plot: {OUT}")

# ── text table ──
def fmt(rows, key, qps):
    for r in rows or []:
        if abs(r.get("qps", -1) - qps) < 1e-6 and key in r:
            return f"{r[key]:.1f}"
    return "  -"

qps_all = sorted({r["qps"] for rows in data.values() if rows for r in rows})
metrics_tbl = [
    ("TTFT p50", "median_ttft_ms"),
    ("TTFT p99", "p99_ttft_ms"),
    ("TPOT p50", "median_tpot_ms"),
    ("TPOT p99", "p99_tpot_ms"),
    ("ITL  p50", "median_itl_ms"),
    ("ITL  p99", "p99_itl_ms"),
    ("E2E  p50", "median_e2e_latency_ms"),
    ("E2E  p99", "p99_e2e_latency_ms"),
    ("tok/s",    "output_throughput"),
]
print("\n" + "=" * 78)
print(f"{'metric':<11}{'tgt':<11}" + "".join(f"{('q'+str(int(q))):>9}" for q in qps_all))
print("-" * 78)
for label, key in metrics_tbl:
    for name, _, _, _ in TARGETS:
        rows = data.get(name)
        line = "".join(f"{fmt(rows, key, q):>9}" for q in qps_all)
        print(f"{label:<11}{name:<11}{line}")
    print("-" * 78)
