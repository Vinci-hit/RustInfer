#!/usr/bin/env python3
"""Read cumulative execution summaries; keep the latest per owner/phase.

Usage: python3 scripts/summarize_execution_stats.py path/to/worker.log [--json]
Host spans can nest (Wait within Draft/CatchUp); never sum them as latency.
GPU times are sampled compute-stream spans, not a sum of kernel durations.
"""
import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r'\b(\w+)=(?:"([^"\n]*)"|([^\s,]+))')


def parse_log(text):
    owners = {}
    for line in ANSI.sub("", text).splitlines():
        if "execution_stats" not in line:
            continue
        if "execution summary" not in line and "execution tokens" not in line:
            continue
        fields = {}
        for key, quoted, bare in FIELD.findall(line):
            value = quoted or bare
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
            fields[key] = value
        owner = fields.pop("owner", None)
        if owner is None:
            continue
        record = owners.setdefault(owner, {"phases": {}})
        if "execution summary" in line:
            phase = fields.pop("phase", None)
            if phase is not None:
                record["phases"][phase] = fields
        else:
            record["tokens"] = fields
    return owners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = {str(path): parse_log(path.read_text()) for path in args.logs}
    if args.json:
        print(json.dumps(reports, indent=2, ensure_ascii=False))
        return
    for path, owners in reports.items():
        print(path)
        if not owners:
            print("  No summaries. Enable RUSTINFER_EXECUTION_STATS_EVERY and execution_stats=info.")
        for owner, report in owners.items():
            print(f"  {owner}: phase / calls / host mean ms / GPU mean ms / GPU samples / failures")
            for phase, s in report["phases"].items():
                gpu_mean = s.get("gpu_mean_ms")
                gpu_text = f"{gpu_mean:.3f}" if isinstance(gpu_mean, (float, int)) else "n/a"
                print(f"    {phase:10} {s.get('calls', 0):6}  "
                      f"{s.get('host_mean_ms', 0):9.3f}  "
                      f"{gpu_text:>12}  "
                      f"{s.get('gpu_samples', 0):6}  {s.get('failures', 0)}")
            if "tokens" in report:
                print("    tokens: " + json.dumps(report["tokens"], ensure_ascii=False))
    print("Host spans may nest; GPU spans are sampled. Do not sum phase means as request latency.")


if __name__ == "__main__":
    main()
