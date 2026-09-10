"""Compare ordinary graph/eager serving with opt-in Qwen3.5 MTP over HTTP.

Uses an isolated three-process stack per mode and warm, sequential requests.
Reports wall-clock TTFT, TPOT, output tokens/s, exact response comparisons,
and raw responses. Requires an idle GPU; never stops unrelated GPU processes.
"""
import argparse
import concurrent.futures
import json
import os
import re
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from e2e_qwen35_smoke import unused_port, file_sha256


def gpu_info(index):
    data = subprocess.check_output([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip().split(", ")
    return dict(uuid=data[0], name=data[1], memory_mib=int(data[2]), utilization=int(data[3]))


def chat_prompt(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Stack:
    def __init__(self, args, mode):
        self.args, self.mode = args, mode
        self.path = args.output / mode
        self.path.mkdir()
        self.processes = []
        self.cluster = "mtp-bench-" + uuid.uuid4().hex
        self.url = f"http://127.0.0.1:{unused_port()}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        k = int(mode.removeprefix("mtp")) if mode.startswith("mtp") else 0
        arena = 0 if mode == "eager" else 256
        self.config = self.path / "config.toml"
        self.config.write_text(f'''model = {json.dumps(str(args.model))}
model_name = "Qwen3.5-4B"
cluster_id = "{self.cluster}"
device = "cuda:0"
host = "127.0.0.1"
port = {self.url.rsplit(':', 1)[1]}
request_timeout_secs = 180
max_batch_tokens = 128
max_batch_seqs = 1
max_model_len = 2048
max_inflight_requests = 8
batch_wait_ms = 0
paged_block_size = 1
chunked_prefill_size = 32
enable_prefix_caching = false
num_blocks = 2048
ignore_eos = false
worker_heartbeat_timeout_secs = 30
log_level = "debug"
capture_sizes = {"[]" if mode == "eager" else "[1]"}
mtp_num_draft_tokens = {k}
[cuda_memory]
kernel_workspace_mib = 256
graph_arena_mib = {arena}
pool_retain_mib = 256
''')

    def __enter__(self):
        info = gpu_info(self.args.gpu)
        if (info["memory_mib"] > self.args.max_background_memory_mib
                or info["utilization"] > self.args.max_background_utilization):
            raise RuntimeError(f"GPU {self.args.gpu} is not idle: {info}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.args.gpu))
        try:
            for name in ("scheduler", "server", "worker"):
                with (self.path / f"{name}.log").open("w") as log:
                    self.processes.append(subprocess.Popen([
                        str(self.args.bin_dir / f"rustinfer-{name}"), "--config", str(self.config)
                    ], stdout=log, stderr=subprocess.STDOUT, env=env))
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if any(p.poll() is not None for p in self.processes):
                    raise RuntimeError(f"{self.mode} startup failed; see {self.path}")
                try:
                    with self.opener.open(self.url + "/ready", timeout=2) as r:
                        if r.status == 200:
                            (self.path / "gpu_ready.json").write_text(json.dumps(gpu_info(self.args.gpu)))
                            return self
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.2)
            raise TimeoutError(f"{self.mode} readiness timed out")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for p in reversed(self.processes):
            if p.poll() is None:
                p.terminate()
        for p in reversed(self.processes):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        for path in Path("/tmp").glob(f"rustinfer-{self.cluster}-*.ipc"):
            path.unlink(missing_ok=True)

    def request(self, prompt, count=128, stream=True, ignore_eos=True, **extra):
        payload = dict(model="Qwen3.5-4B", prompt=chat_prompt(prompt), temperature=0,
                       max_tokens=count, ignore_eos=ignore_eos, stream=stream,
                       stream_options={"include_usage": True}, **extra)
        req = urllib.request.Request(self.url + "/v1/completions", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        started = time.perf_counter()
        first = None
        content, chunks, usage = "", [], None
        with self.opener.open(req, timeout=180) as response:
            if not stream:
                raw = json.load(response)
                return dict(text=raw["choices"][0]["text"], usage=raw["usage"], response=raw)
            done = False
            for line in response:
                line = line.decode().strip()
                if line.startswith("event: error"):
                    raise RuntimeError(line)
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    break
                raw = json.loads(line[6:])
                chunks.append(raw)
                if raw.get("error"):
                    raise RuntimeError(raw)
                usage = raw.get("usage") or usage
                for choice in raw.get("choices", []):
                    delta = choice.get("text", "")
                    if delta:
                        first = first or time.perf_counter()
                        content += delta
                    if choice.get("finish_reason") == "error":
                        raise RuntimeError(raw)
        ended = time.perf_counter()
        assert done and first is not None and usage, chunks
        tokens = usage["completion_tokens"]
        if ignore_eos:
            assert tokens == count, (tokens, count, content)
        return dict(text=content, usage=usage, ttft_ms=(first-started)*1000,
                    elapsed_ms=(ended-started)*1000,
                    tpot_ms=(ended-first)*1000/max(tokens-1, 1),
                    output_tokens_per_second=tokens/(ended-started), chunks=chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--bin-dir", type=Path, default=Path(__file__).resolve().parents[1]/"target/release")
    parser.add_argument("--modes", nargs="+", choices=["graph", "eager", "mtp1", "mtp2", "mtp3", "mtp4"], default=["graph", "eager", "mtp1", "mtp3"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--max-background-memory-mib", type=int, default=512,
                        help="Allowed pre-existing GPU memory usage (e.g. a display GPU)")
    parser.add_argument("--max-background-utilization", type=int, default=5,
                        help="Allowed pre-existing GPU utilization percentage")
    args = parser.parse_args()
    if args.repeats < 1 or args.tokens < 2 or len(set(args.modes)) != len(args.modes):
        parser.error("repeats must be positive, tokens >= 2, and modes unique")
    args.model, args.output, args.bin_dir = args.model.resolve(), args.output.resolve(), args.bin_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError("terminated")))
    prompts = [
        "Explain how a hash table works, including collisions and resizing. Give a detailed answer.",
        "请详细解释 Rust 的所有权、借用与生命周期，并给出例子。",
        "Write Python code implementing merge sort with comments and example tests.",
    ]
    checks = ["What is 2 + 2? Answer with just the number.",
              "用一句话介绍 Rust 编程语言。",
              "Write a Python function that adds two numbers. Output only code."]
    report = {"modes": {}, "gpu_before": gpu_info(args.gpu), "configuration": vars(args).copy(),
              "binary_sha256": {n: file_sha256(args.bin_dir/f"rustinfer-{n}") for n in ("worker", "server", "scheduler")}}
    def save():
        (args.output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    try:
        for mode in args.modes:
            print(f"Starting {mode}", flush=True)
            with Stack(args, mode) as stack:
                accuracy = [stack.request(p, count=32, stream=False, ignore_eos=False) for p in checks]
                streamed = stack.request(checks[1], count=32, ignore_eos=False)
                assert streamed["text"] == accuracy[1]["text"], (mode, "SSE mismatch")
                # Two HTTP requests exercise queuing at worker max_batch_seqs=1.
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    queued = list(pool.map(lambda p: stack.request(p, count=32, stream=False, ignore_eos=False), checks[:2]))
                assert [r["text"] for r in queued] == [r["text"] for r in accuracy[:2]]
                # The second half of a speculative burst must not escape a stop string.
                stopped = stack.request(checks[2], count=32, ignore_eos=False, stop=["return"])
                # Preserve failures without losing the other modes' measurements.
                stop_check = dict(passed="return" not in stopped["text"], response=stopped)
                cancel_payload = dict(prompt=chat_prompt(prompts[0]), temperature=0, max_tokens=512,
                                      ignore_eos=True, stream=True)
                cancel = urllib.request.Request(stack.url + "/v1/completions",
                    json.dumps(cancel_payload).encode(), {"Content-Type": "application/json"})
                with stack.opener.open(cancel, timeout=180) as response:
                    for line in response:
                        if line.startswith(b"data: ") and b'"text":"' in line.replace(b" ", b""):
                            break
                after_cancel = stack.request(checks[0], count=32, stream=False, ignore_eos=False)
                assert after_cancel["text"] == accuracy[0]["text"]
                if mode.startswith("mtp"):
                    invalid = urllib.request.Request(stack.url + "/v1/completions",
                        json.dumps(dict(prompt="hello", temperature=1, max_tokens=1)).encode(),
                        {"Content-Type": "application/json"})
                    try:
                        stack.opener.open(invalid, timeout=5)
                    except urllib.error.HTTPError as error:
                        assert error.code == 400, error.code
                    else:
                        raise AssertionError("MTP accepted stochastic sampling")
                for p in prompts:
                    stack.request(p, count=args.tokens)  # discard warm-up
                log_start = (stack.path / "worker.log").stat().st_size
                measured = []
                for repeat in range(args.repeats):
                    for i, p in enumerate(prompts):
                        result = stack.request(p, count=args.tokens)
                        result.update(prompt_index=i, repeat=repeat)
                        measured.append(result)
                summary = {key: statistics.median(r[key] for r in measured)
                           for key in ("ttft_ms", "tpot_ms", "elapsed_ms", "output_tokens_per_second")}
                summary["aggregate_output_tokens_per_second"] = sum(r["usage"]["completion_tokens"] for r in measured) / (sum(r["elapsed_ms"] for r in measured)/1000)
                with (stack.path / "worker.log").open("rb") as log:
                    log.seek(log_start)
                    measured_log = re.sub(r"\x1b\[[0-9;]*m", "", log.read().decode())
                (stack.path / "measured-worker.log").write_text(measured_log)
                graph_replays = measured_log.count("replaying decode CUDA graph")
                rounds = [line for line in measured_log.splitlines() if "MTP round" in line]
                if mode == "graph":
                    assert graph_replays > 0, "ordinary graph baseline did not replay graphs"
                elif mode == "eager":
                    assert graph_replays == 0, "eager baseline replayed graphs"
                else:
                    assert rounds and graph_replays == 0, "MTP serving path was not exercised"
                    for key in ("proposed", "accepted", "emitted"):
                        summary[key] = sum(int(re.search(rf"\b{key}=(\d+)", line)[1]) for line in rounds)
                    for key in ("draft_ms", "verify_ms", "catchup_ms"):
                        summary["round_median_" + key] = statistics.median(float(re.search(rf"\b{key}=([0-9.eE+-]+)", line)[1]) for line in rounds)
                report["modes"][mode] = dict(accuracy=accuracy, measured=measured, summary=summary,
                                             stop_check=stop_check,
                                             measured_graph_replays=graph_replays, measured_mtp_rounds=len(rounds))
                if "graph" in report["modes"]:
                    baseline = report["modes"]["graph"]
                    report["modes"][mode]["accuracy_matches_graph"] = [a["text"] == b["text"] for a, b in zip(accuracy, baseline["accuracy"])]
                    report["modes"][mode]["performance_text_matches_graph"] = [a["text"] == b["text"] for a, b in zip(measured, baseline["measured"])]
                print(mode, json.dumps(summary), flush=True)
                save()
            time.sleep(1)
        if "graph" in report["modes"]:
            baseline = report["modes"]["graph"]
            for result in report["modes"].values():
                result["accuracy_matches_graph"] = [a["text"] == b["text"] for a, b in zip(result["accuracy"], baseline["accuracy"])]
                result["performance_text_matches_graph"] = [a["text"] == b["text"] for a, b in zip(result["measured"], baseline["measured"])]
            report["accuracy_passed"] = all(
                all(m["accuracy_matches_graph"]) and all(m["performance_text_matches_graph"])
                for m in report["modes"].values())
            report["stop_checks_passed"] = all(m["stop_check"]["passed"] for m in report["modes"].values())
            assert report["accuracy_passed"], "greedy responses differed; see results.json"
            assert report["stop_checks_passed"], "stop string check failed; see results.json"
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
