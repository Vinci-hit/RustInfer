"""Compare Qwen3.5 weight loading with serial and double-buffered uploads.

Reuses the isolated serving stack; never drops OS caches or stops other jobs.
Reports the worker's weight-loading interval separately from total readiness.
Optional saved baseline binaries also measure the combined loader change.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import statistics
import time

from bench_qwen35_mtp import Stack, gpu_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=Path("target/release"))
    parser.add_argument("--baseline-bin-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--max-background-memory-mib", type=int, default=512)
    parser.add_argument("--max-background-utilization", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup_rounds < 0:
        parser.error("repeats must be positive and warmup-rounds nonnegative")
    args.model, args.output, args.bin_dir = args.model.resolve(), args.output.resolve(), args.bin_dir.resolve()
    if args.baseline_bin_dir:
        args.baseline_bin_dir = args.baseline_bin_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("terminated")))
    variants = ["serial", "double_buffer"]
    if args.baseline_bin_dir:
        variants.insert(0, "baseline")
    report = dict(gpu=gpu_info(args.gpu), cache_policy="OS caches retained; not a controlled cold-storage benchmark", runs=[])
    env_name = "RUSTINFER_BULK_UPLOAD_CHUNK_MIB"
    previous = os.environ.get(env_name)
    try:
        for repeat in range(-args.warmup_rounds, args.repeats):
            # Reverse order to reduce fixed ordering bias.
            for variant in variants if repeat % 2 == 0 else reversed(variants):
                current = copy.copy(args)
                current.output = args.output / f"{repeat}-{variant}"
                current.output.mkdir()
                if variant == "baseline":
                    current.bin_dir = args.baseline_bin_dir
                os.environ[env_name] = "8" if variant == "double_buffer" else "0"
                started = time.perf_counter()
                with Stack(current, "graph") as stack:
                    ready = time.perf_counter() - started
                    answers = [stack.request(prompt, count=32, stream=False, ignore_eos=False)["text"] for prompt in (
                        "What is 2 + 2? Answer with just the number.",
                        "把 hello 翻译成中文。",
                        "Write a Python function that adds two numbers.",
                    )]
                log = (current.output / "graph/worker.log").read_text()
                match = re.search(r"weights loaded in ([0-9.]+)s", log)
                if not match:
                    raise RuntimeError(f"missing weight-loading measurement: {current.output}")
                row = dict(variant=variant, repeat=repeat, warmup=repeat < 0, weights_seconds=float(match[1]), ready_seconds=ready, answers=answers)
                report["runs"].append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
    reference = report["runs"][0]["answers"]
    report["all_answers_match"] = all(row["answers"] == reference for row in report["runs"])
    report["median_weights_seconds"] = {
        variant: statistics.median(row["weights_seconds"] for row in report["runs"] if row["variant"] == variant and not row["warmup"])
        for variant in variants
    }
    (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ("all_answers_match", "median_weights_seconds")}), flush=True)
    if not report["all_answers_match"]:
        raise SystemExit("greedy response mismatch; inspect results.json")


if __name__ == "__main__":
    main()
