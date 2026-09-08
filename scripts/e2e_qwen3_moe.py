"""Smoke-test a real Qwen3 MoE checkpoint through the three-process HTTP stack."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--force-gemm", action="store_true",
                    help="Use the existing CUDA GEMM path for single-token projections too")
parser.add_argument("--require-batch-match", action="store_true",
                    help="Fail if concurrent batching changes greedy text")
parser.add_argument("--bin-dir", type=Path, default=Path("target/debug"))
args = parser.parse_args()
config = tomllib.loads(args.config.read_text())
args.output.mkdir(parents=True, exist_ok=True)
base = f"http://{config['host']}:{config['port']}"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
worker_env = os.environ.copy()
if args.force_gemm:
    worker_env["RUSTINFER_FORCE_GEMM"] = "1"
print("RUSTINFER_FORCE_GEMM:", worker_env.get("RUSTINFER_FORCE_GEMM", "unset"), flush=True)
processes = []
logs = []

def request(prompt, count=16, stream=False):
    payload = dict(model=config["model_name"], prompt=prompt, max_tokens=count,
                   temperature=0, stream=stream, ignore_eos=True)
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=600) as response:
        if not stream:
            result = json.load(response)
            assert result["usage"]["completion_tokens"] == count, result
            return result["choices"][0]["text"]
        chunks = []
        events = []
        done = False
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            result = json.loads(data)
            events.append(result)
            if result.get("choices"):
                chunks.append(result["choices"][0].get("text", ""))
        (args.output / "stream.json").write_text(json.dumps(events, indent=2))
        assert done, "stream ended without DONE"
        return "".join(chunks)

try:
    for name in ["scheduler", "worker", "server"]:
        log = (args.output / f"{name}.log").open("w")
        logs.append(log)
        processes.append(subprocess.Popen([str(args.bin_dir.resolve() / f"rustinfer-{name}"),
                                           "--config", str(args.config.resolve())],
                                          stdout=log, stderr=subprocess.STDOUT, env=worker_env))
    for _ in range(600):
        assert all(p.poll() is None for p in processes), "stack exited; inspect logs"
        if "Entering serve loop" in (args.output / "worker.log").read_text():
            try:
                opener.open(base + "/health", timeout=1).close()
                break
            except OSError:
                pass
        time.sleep(1)
    else:
        raise RuntimeError("readiness timeout")
    print("Stack ready", flush=True)
    cases = [("The capital of France is", 16), ("1 + 1 =", 8),
             ("The quick brown fox jumps over the lazy dog. " * 8 + "Summary:", 12)]
    baseline = [request(prompt, count) for prompt, count in cases]
    for case, result in zip(cases, baseline):
        assert result.strip(), (case, result)
        print(repr(case[0]), "=>", repr(result), flush=True)
    assert request(*cases[0]) == baseline[0], "repeated greedy request differs"
    streamed = request(*cases[0], stream=True)
    print("stream:", repr(streamed), flush=True)
    assert streamed == baseline[0], (streamed, baseline[0])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        actual = list(pool.map(lambda case: request(*case), cases))
    agreement = [a == b for a, b in zip(actual, baseline)]
    print(f"Batch text agreement: {sum(agreement)}/{len(agreement)}", flush=True)
    (args.output / "results.json").write_text(json.dumps([
        dict(prompt=p, max_tokens=n, text=t, concurrent_text=a, batch_match=match)
        for (p, n), t, a, match in zip(cases, baseline, actual, agreement)
    ], ensure_ascii=False, indent=2))
    if args.require_batch_match:
        assert all(agreement), (actual, baseline)
    print("PASS: HTTP chain, prefill, continuous decode, chunked prefill, repeated requests, streaming", flush=True)
finally:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for log in logs:
        log.close()
