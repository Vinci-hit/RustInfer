"""Compare a RustInfer GGUF probe with an independently built llama.cpp probe.

Requires NumPy. See docs/GGUF_COMPARISON.md for build and run commands.
Both processes run sequentially and exit after testing; no service is started.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


CASES = [
    ("chinese", ["--prompt", "你好，请用一句话介绍自己。", "--no-thinking"], 32),
    ("arithmetic", ["--prompt", "17 加 25 等于多少？只回答数字。", "--no-thinking"], 16),
    ("thinking", ["--prompt", "What is 17 + 25?", "--reasoning-effort", "low"], 16),
]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(command, log, timeout=900):
    start = time.monotonic()
    peak = 0
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"timed out; see {log}")
                memory = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                    capture_output=True, text=True, timeout=10,
                )
                if memory.returncode == 0:
                    peak = max(peak, int(memory.stdout.strip()))
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f"exit {process.returncode}; see {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return {"command": command, "wall_seconds": time.monotonic() - start,
            "sampled_peak_total_gpu_mib": peak}


def compare_logits(actual, reference):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("incompatible or non-finite logits")
    error = a - b
    logp = a - np.max(a)
    logq = b - np.max(b)
    logp -= np.log(np.exp(logp).sum())
    logq -= np.log(np.exp(logq).sum())
    ai, bi = int(a.argmax()), int(b.argmax())
    topa, topb = set(np.argsort(a)[-5:]), set(np.argsort(b)[-5:])
    return {"relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(b), 1e-30)),
            "max_abs": float(np.max(np.abs(error))),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)),
            "kl_llama_to_rustinfer": float(np.sum(np.exp(logq) * (logq - logp))),
            "top1_matches": ai == bi, "rustinfer_top1": ai, "llama_top1": bi,
            "llama_margin_over_rustinfer_choice": float(b[bi] - b[ai]),
            "top5_overlap": len(topa & topb)}


def timing(steps):
    times = [step["elapsed_seconds"] for step in steps]
    return {"prefill_seconds": times[0], "decode_steps": len(times) - 1,
            "decode_tokens_per_second": (len(times) - 1) / sum(times[1:]) if len(times) > 1 else None}


def summarize(manifest):
    result = {"metadata": {k: v for k, v in manifest.items() if k != "cases"}, "cases": []}
    for case in manifest["cases"]:
        rust = json.loads(Path(case["rustinfer_report"]).read_text())
        llama = json.loads(Path(case["llama_report"]).read_text())
        if rust["input_ids"] != llama["input_ids"]:
            raise ValueError("prompt IDs differ")
        if len(rust["steps"]) != len(llama["teacher_forced_steps"]):
            raise ValueError("teacher-forced step counts differ")
        metrics = [compare_logits(a["logits"], b["logits"])
                   for a, b in zip(rust["steps"], llama["teacher_forced_steps"], strict=True)]
        prefix = 0
        for a, b in zip(rust["generated_ids"], llama["generated_ids"]):
            if a != b:
                break
            prefix += 1
        row = {"name": case["name"], "prompt_tokens": len(rust["input_ids"]),
               "rustinfer_process": case["rustinfer_process"],
               "tokenizer_matches": llama.get("tokenizer_matches"),
               "identical_greedy_ids": rust["generated_ids"] == llama["generated_ids"],
               "greedy_common_prefix_tokens": prefix,
               "rustinfer_text": rust.get("generated_text"), "llama_text": llama["generated_text"],
               "rustinfer_ids": rust["generated_ids"], "llama_ids": llama["generated_ids"],
               "rustinfer_timing": timing(rust["steps"]),
               "llama_teacher_forced_timing": timing(llama["teacher_forced_steps"]),
               "teacher_forced_top1_matches": sum(x["top1_matches"] for x in metrics),
               "teacher_forced_steps": len(metrics),
               "max_relative_l2": max(x["relative_l2"] for x in metrics),
               "mean_kl_llama_to_rustinfer": float(np.mean([x["kl_llama_to_rustinfer"] for x in metrics])),
               "per_step": metrics}
        result["cases"].append(row)
        print(f'{case["name"]}: tokenizer={row["tokenizer_matches"]}, '
              f'greedy_prefix={prefix}, teacher_top1={row["teacher_forced_top1_matches"]}/{len(metrics)}, '
              f'max_relative_l2={row["max_relative_l2"]:.4f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rustinfer", type=Path, default=Path("target/release/rustinfer-gguf"))
    parser.add_argument("--llama-probe", type=Path, required=True)
    parser.add_argument("--llama-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("target/gguf-comparison"))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if args.summarize_only:
        manifest = json.loads(manifest_path.read_text())
    else:
        def git(*command):
            return subprocess.check_output(["git", *command], text=True).strip()

        manifest = {"model": str(args.model.resolve()), "model_sha256": sha256(args.model),
                    "context": 2048, "rustinfer_head": git("rev-parse", "HEAD"),
                    "rustinfer_dirty": bool(git("status", "--porcelain")),
                    "rustinfer_binary_sha256": sha256(args.rustinfer),
                    "llama_probe_sha256": sha256(args.llama_probe),
                    "llama_commit": git("-C", str(args.llama_source), "rev-parse", "HEAD"),
                    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader", "-i", "0"], text=True).strip(),
                    "rustinfer_dtype": "bf16", "llama_kv_dtype": "bf16",
                    "notes": ["Release builds; sequential processes; single sequence; all decoder layers on GPU.",
                              "Same prefix at each numerical comparison: teacher-forced RustInfer IDs.",
                              "llama.cpp uses its normal mixed/F32 arithmetic; BF16 KV does not imply BF16 activations.",
                              "Timing includes launches and full-vocabulary readback; excludes load, sorting and JSON.",
                              "Single diagnostic run, no warmup on RustInfer; not an HTTP or steady-state benchmark.",
                              "GPU memory is sampled total device use, including desktop and other processes."],
                    "cases": []}
        for name, flags, steps in CASES:
            report = args.output / f"rustinfer-{name}.json"
            command = [str(args.rustinfer.resolve()), "--model", manifest["model"],
                       "--backend", "cuda", "--dtype", "bf16", "--context", "2048",
                       "--steps", str(steps), "--dump", str(report), *flags]
            print(f"Running RustInfer: {name}", flush=True)
            run_info = run(command, args.output / f"rustinfer-{name}.log")
            manifest["cases"].append({"name": name, "steps": steps,
                "rustinfer_report": str(report), "llama_report": str(args.output / f"llama-{name}.json"),
                "rustinfer_process": run_info})
            save(manifest_path, manifest)
        print("Running llama.cpp reference", flush=True)
        manifest["llama_process"] = run([str(args.llama_probe.resolve()), str(manifest_path)],
                                        args.output / "llama.log")
        save(manifest_path, manifest)
    save(args.output / "summary.json", summarize(manifest))
    print(f"Results: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
