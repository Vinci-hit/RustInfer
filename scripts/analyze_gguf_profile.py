"""Aggregate synchronized prefill/decode NVTX ranges from an nsys SQLite export.

GPU active time is the union of kernel/copy/memset intervals. API times overlap
GPU execution and are reported separately, never added to GPU time.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
import statistics


GGML_TYPES = {8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
              16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS"}


def category(name):
    cute = re.search(r"_((?:IQ|Q)\d+_[A-Z0-9]+)_\d+_", name)
    llama = re.search(r"\(ggml_type\)(\d+)", name)
    if "gemv_kernel" in name or "mul_mat_vec_q<" in name:
        fmt = cute[1] if cute else GGML_TYPES.get(int(llama[1]), llama[1]) if llama else "unknown"
        return "quant_gemv/" + fmt
    if "gemm_kernel" in name or "mul_mat_q<" in name:
        fmt = cute[1] if cute else GGML_TYPES.get(int(llama[1]), llama[1]) if llama else "unknown"
        return "quant_gemm/" + fmt
    if "gated_delta" in name:
        return "gdn_recurrence"
    if "conv" in name:
        return "convolution"
    if "flash" in name or "attn" in name or "soft_max" in name:
        return "attention"
    if "rope" in name:
        return "rope"
    if "norm" in name:
        return "normalization"
    if "quantize" in name:
        return "activation_quantization"
    if "embedding" in name:
        return "embedding"
    if "get_rows" in name:
        return "row_gather"
    if "copy" in name or "cpy" in name or "contiguous" in name:
        return "layout_copy"
    return "other"


def union_ns(events):
    end, total = -1, 0
    for start, stop in sorted(events):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def coverage(events):
    merged = []
    for start, end in sorted(events):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    starts = [x[0] for x in merged]
    totals = [0]
    for start, end in merged:
        totals.append(totals[-1] + end - start)

    def until(t):
        index = bisect_right(starts, t) - 1
        if index < 0:
            return 0
        start, end = merged[index]
        return totals[index] + min(t - start, end - start)
    return lambda start, end: until(end) - until(start)


def analyze(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernels = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"))
    memory = []
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            memory.extend(dict(row, kind=table) for row in db.execute(f"SELECT * FROM {table}"))
    apis = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table in tables:
            apis.extend(db.execute(f"SELECT * FROM {table}"))
    groups = defaultdict(list)
    for span in db.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
        label = span["text"] or strings.get(span["textId"], "")
        group = "decode" if label.startswith("decode/") else "request" if label.startswith("request/") else label
        if group not in ("prefill", "decode", "host_argmax", "reset", "request"):
            continue
        start, end = span["start"], span["end"]
        selected = [k for k in kernels if start <= k["start"] and k["end"] <= end]
        copies = [k for k in memory if start <= k["start"] and k["end"] <= end]
        calls = [k for k in apis if start <= k["start"] and k["end"] <= end]
        by_name, by_category, api_times, api_overlap = Counter(), Counter(), Counter(), Counter()
        by_count = Counter()
        for k in selected:
            name = strings[k["demangledName"]]
            ns = k["end"] - k["start"]
            by_name[name] += ns
            by_category[category(name)] += ns
            by_count[category(name)] += 1
        overlap = coverage([(k["start"], k["end"]) for k in [*selected, *copies]])
        for k in calls:
            name = strings[k["nameId"]]
            api_times[name] += k["end"] - k["start"]
            api_overlap[name] += overlap(k["start"], k["end"])
        active = union_ns([(k["start"], k["end"]) for k in [*selected, *copies]])
        groups[group].append({"label": label, "wall_ns": end - start, "gpu_active_ns": active,
            "gpu_inactive_ns": end - start - active, "kernel_ns": sum(by_name.values()),
            "kernel_count": len(selected), "copy_memset_ns": sum(k["end"] - k["start"] for k in copies),
            "copy_memset_count": len(copies), "copy_memset_bytes": sum(k.get("bytes", 0) for k in copies),
            "by_kernel": by_name, "by_category": by_category, "category_count": by_count,
            "api_times": api_times, "api_gpu_overlap": api_overlap})
    result = {"source": str(path), "phases": {}}
    for group, rows in groups.items():
        n = len(rows)
        linear_counts = [sum(v for k, v in row["category_count"].items() if k.startswith("quant_")) for row in rows]
        if min(linear_counts) != max(linear_counts):
            raise ValueError(f"{group}: unequal quantized matmul counts {linear_counts}; "
                             "check for truncated CUDA trace buffers before interpreting GPU idle time")
        totals = {}
        for field in ("by_kernel", "by_category", "category_count", "api_times", "api_gpu_overlap"):
            counts = Counter()
            for row in rows:
                counts.update(row[field])
            totals[field] = {key: value / n / (1 if field == "category_count" else 1e6)
                             for key, value in counts.most_common()}
        scalar = {field.removesuffix("_ns") + "_ms": statistics.mean(r[field] for r in rows) / 1e6
                  for field in ("wall_ns", "gpu_active_ns", "gpu_inactive_ns", "kernel_ns", "copy_memset_ns")}
        scalar.update({field: statistics.mean(r[field] for r in rows)
                       for field in ("kernel_count", "copy_memset_count", "copy_memset_bytes")})
        result["phases"][group] = {"samples": n, "linear_calls_per_sample": linear_counts[0], **scalar, **totals}
    if not result["phases"].get("request") and (not result["phases"].get("decode") or not result["phases"].get("prefill")):
        raise ValueError("missing annotated phases in trace")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for phase in (p for p in ("prefill", "decode", "request") if p in result["phases"]):
        p = result["phases"][phase]
        print(phase, {k: v for k, v in p.items() if k not in ("by_kernel", "api_times", "api_gpu_overlap", "category_count")})


if __name__ == "__main__":
    main()
