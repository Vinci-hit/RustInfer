#!/usr/bin/env python3
"""Compare DFlash with target CUDA graph decode and optional EAGLE3, over HTTP."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import time

import bench_qwen35_mtp as harness
from e2e_qwen35_smoke import file_sha256

PROMPTS = ["Explain how hash tables work and discuss collision resolution strategies.",
           "请解释 Rust 的所有权、借用和生命周期，并给出简单的例子。",
           "Write a Python merge sort implementation and explain its complexity."]


def summarize(rows, log):
    result = dict(throughput=sum(r["usage"]["completion_tokens"] for r in rows) / sum(r["elapsed_ms"] / 1000 for r in rows),
                  ttft_ms=statistics.mean(r["ttft_ms"] for r in rows),
                  tpot_ms=statistics.mean(r["tpot_ms"] for r in rows))
    rounds = [line for line in log.splitlines() if "speculative round" in line]
    result["rounds"] = len(rounds)
    if rounds:
        for key in ("proposed", "accepted", "emitted"):
            result[key] = sum(int(re.search(rf"\b{key}=(\d+)", line)[1]) for line in rounds)
        result["acceptance_rate"] = result["accepted"] / max(result["proposed"], 1)
        result["emitted_per_round"] = result["emitted"] / len(rounds)
        for key in ("draft_ms", "verify_ms", "catchup_ms"):
            result[key] = statistics.mean(float(re.search(rf"\b{key}=([0-9.eE+-]+)", line)[1]) for line in rounds)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--eagle-draft", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["graph", "dflash7", "dflash15"])
    parser.add_argument("--bin-dir", type=Path, default=Path(__file__).resolve().parents[1] / "target/release")
    parser.add_argument("--max-background-memory-mib", type=int, default=4096)
    parser.add_argument("--max-background-utilization", type=int, default=35)
    args = parser.parse_args()
    if args.repeats < 1 or args.tokens < 2 or len(set(args.modes)) != len(args.modes):
        parser.error("repeats must be positive, tokens >= 2, modes unique")
    if any(mode not in ("graph", "eager", "eagle3", "dflash3", "dflash7", "dflash15") for mode in args.modes):
        parser.error("unknown benchmark mode")
    if "eagle3" in args.modes and not args.eagle_draft:
        parser.error("eagle3 requires --eagle-draft")
    args.model_name = args.model.name
    args.allow_target_mismatch = False
    args.output.mkdir(parents=True, exist_ok=True)
    draft = args.draft
    # Instruct-2507 uses a direct assistant prefix, without a thinking prefix.
    harness.chat_prompt = lambda text: f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    report = dict(configuration=vars(args).copy(), prompts=PROMPTS, modes={},
                  binary_sha256={name: file_sha256(args.bin_dir / f"rustinfer-{name}") for name in ("worker", "server", "scheduler")},
                  gpu_before=harness.gpu_info(args.gpu),
                  environment={k:v for k,v in os.environ.items() if k.startswith("RUSTINFER_")})
    saved = args.output / "results.json"
    if saved.exists():
        previous = json.loads(saved.read_text())
        if previous["binary_sha256"] != report["binary_sha256"]:
            parser.error("output contains measurements from different binaries")
        report = previous
        if "error" in report:
            report.setdefault("interrupted_attempts", []).append(report.pop("error"))
    def save():
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    try:
        for mode in args.modes:
            if mode in report["modes"]:
                continue
            path = args.output / mode
            if path.exists():
                path.rename(path.with_name(f"{mode}_interrupted_{time.time_ns()}"))
            # Let nvidia-smi's rolling utilization window drop the preceding run.
            time.sleep(2)
            args.draft = args.eagle_draft if mode.startswith("eagle") else draft
            print(f"Starting {mode}", flush=True)
            with harness.Stack(args, mode) as stack:
                # Request reuse, chunked prefill, and the K=0 budget tail.
                for count in (1, 2, 9):
                    result = stack.request(PROMPTS[0], count=count)
                    assert result["usage"]["completion_tokens"] == count
                for prompt in PROMPTS:
                    stack.request(prompt, count=args.tokens)
                offset = (stack.path / "worker.log").stat().st_size
                rows = []
                for repeat in range(args.repeats):
                    for i in [(repeat + j) % len(PROMPTS) for j in range(len(PROMPTS))]:
                        result = stack.request(PROMPTS[i], count=args.tokens)
                        assert result["usage"]["completion_tokens"] == args.tokens
                        result.pop("chunks", None)
                        result.update(prompt_index=i, repeat=repeat)
                        rows.append(result)
                        print(f"{mode} repeat={repeat} prompt={i}: {result['output_tokens_per_second']:.2f} tok/s", flush=True)
                with (stack.path / "worker.log").open("rb") as log:
                    log.seek(offset)
                    measured_log = re.sub(r"\x1b\[[0-9;]*m", "", log.read().decode())
                summary = summarize(rows, measured_log)
                if mode.startswith(("dflash", "eagle")):
                    assert summary["rounds"] > 0
                    assert summary["emitted"] == len(rows) * (args.tokens - 1)
                report["modes"][mode] = dict(summary=summary, measured=rows)
                print(f"SUMMARY {mode}: {json.dumps(summary)}", flush=True)
                save()
        if "graph" in report["modes"]:
            baseline = report["modes"]["graph"]["summary"]["throughput"]
            for mode in report["modes"].values():
                mode["summary"]["speedup_vs_graph"] = mode["summary"]["throughput"] / baseline
        report["completed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["gpu_after"] = harness.gpu_info(args.gpu)
        save()


if __name__ == "__main__":
    main()
