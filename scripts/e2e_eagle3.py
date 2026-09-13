"""Compare Qwen3 eager and EAGLE3 HTTP output, stop, cancellation and reuse.

Reuses the speculative serving harness. Owns isolated processes and saves all
responses, launch configs and logs. CUDA libraries must be on LD_LIBRARY_PATH.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from bench_qwen35_mtp import Stack, chat_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-target-mismatch", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 3, 7])
    parser.add_argument("--bin-dir", type=Path, default=Path(__file__).resolve().parents[1] / "target/release")
    parser.add_argument("--max-background-memory-mib", type=int, default=4096)
    parser.add_argument("--max-background-utilization", type=int, default=30)
    args = parser.parse_args()
    if not args.widths or any(k < 1 or k > 7 for k in args.widths) or len(set(args.widths)) != len(args.widths):
        parser.error("widths must be unique and between 1 and 7")
    for attr in ("model", "draft", "output", "bin_dir"):
        setattr(args, attr, getattr(args, attr).resolve())
    args.model_name = args.model.name
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"configuration": vars(args).copy(), "modes": {}}
    cases = [
        ("What is 2 + 2? Answer with just the number.", 32, False),
        ("用一句话介绍 Rust 编程语言。", 64, False),
        ("Write a Python function that adds two numbers. Output only code.", 64, False),
        ("Explain ownership and borrowing in Rust.", 96, True),
        ("Say hello.", 1, True),
        ("Say hello.", 2, True),
    ]
    try:
        for mode in ["eager"] + [f"eagle{k}" for k in args.widths]:
            print(f"Starting {mode}", flush=True)
            with Stack(args, mode) as stack:
                responses = [stack.request(p, count=n, stream=False, ignore_eos=ignore)
                             for p, n, ignore in cases]
                result = dict(responses=responses)
                report["modes"][mode] = result
                streamed = stack.request(cases[1][0], count=64, ignore_eos=False)
                assert streamed["text"] == responses[1]["text"], (mode, "SSE mismatch")
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    queued = list(pool.map(lambda p: stack.request(p, count=32, stream=False, ignore_eos=False),
                                           [cases[0][0], cases[0][0]]))
                assert all(r["text"] == responses[0]["text"] for r in queued), "queue/reuse mismatch"
                # The current service matches token sequences; include the
                # leading space so the stop uses the token emitted in code.
                stopped = stack.request(cases[2][0], count=64, ignore_eos=False, stop=[" return"])
                result["stopped"] = stopped
                assert "return" not in stopped["text"], (mode, "stop burst escaped")
                request = urllib.request.Request(stack.url + "/v1/completions", json.dumps(dict(
                    model=args.model_name, prompt=chat_prompt(cases[3][0]), temperature=0,
                    max_tokens=512, stream=True, ignore_eos=True)).encode(),
                    {"Content-Type": "application/json"})
                with stack.opener.open(request, timeout=180) as response:
                    for line in response:
                        if line.startswith(b"data: ") and b'"text":"' in line.replace(b" ", b""):
                            break
                after_cancel = stack.request(cases[0][0], count=32, stream=False, ignore_eos=False)
                assert after_cancel["text"] == responses[0]["text"], "cancellation recovery mismatch"
                if mode != "eager":
                    invalid = urllib.request.Request(stack.url + "/v1/completions",
                        json.dumps(dict(prompt="hello", temperature=1, max_tokens=1)).encode(),
                        {"Content-Type": "application/json"})
                    try:
                        stack.opener.open(invalid, timeout=5)
                    except urllib.error.HTTPError as error:
                        assert error.code == 400, error.code
                    else:
                        raise AssertionError("EAGLE3 accepted stochastic sampling")
                    baseline = report["modes"]["eager"]["responses"]
                    result["matches_eager"] = [a["text"] == b["text"] and a["usage"] == b["usage"]
                                                for a, b in zip(responses, baseline)]
                    log = re.sub(r"\x1b\[[0-9;]*m", "", (stack.path / "worker.log").read_text())
                    rounds = [line for line in log.splitlines() if "speculative round" in line]
                    assert rounds, "speculative execution was not exercised"
                    result["rounds"] = len(rounds)
                    for key in ("proposed", "accepted", "emitted"):
                        result[key] = sum(int(re.search(rf"\b{key}=(\d+)", line)[1]) for line in rounds)
                print(f"{mode}: service checks passed; greedy matches={result.get('matches_eager')}", flush=True)
            time.sleep(2)  # Allow the previous process's GPU utilization sample to expire.
        assert all(all(m.get("matches_eager", [True])) for m in report["modes"].values()), "greedy output differs; see results.json"
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
