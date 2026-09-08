"""Reproducible Qwen3.5 service regression, using only Python's standard library.

Requires built RustInfer binaries and local Qwen3.5-4B weights. Owns an isolated
three-process stack and always tears it down. Outputs JSON results, exact launch
configuration, environment metadata, and process logs, including on failure.
This is a service/semantic smoke test; HF numerical references are separate.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zlib


def red_png():
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    rows = b"".join(b"\0" + b"\xff\0\0" * 256 for _ in range(256))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {"exit_code": result.returncode, "output": result.stdout + result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bin-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "target/release")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=180)
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.bin_dir = args.bin_dir.resolve()
    output = args.output or Path(tempfile.mkdtemp(prefix="rustinfer-qwen35-"))
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    # A dedicated directory avoids overwriting results from another invocation.
    if (output / "results.json").exists():
        parser.error("output already contains results.json; choose a new directory")
    for name in ("scheduler", "worker", "server"):
        if not (args.bin_dir / f"rustinfer-{name}").is_file():
            parser.error(f"missing binary: rustinfer-{name}")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("model_type") != "qwen3_5":
        parser.error("this regression requires a Qwen3.5 multimodal checkpoint")
    port = args.port or unused_port()
    cluster = "qwen35-regression-" + uuid.uuid4().hex
    config = output / "config.toml"
    config.write_text(f'''model = {json.dumps(str(args.model))}
model_name = "Qwen3.5-4B"
cluster_id = "{cluster}"
device = {json.dumps(args.device)}
host = "127.0.0.1"
port = {port}
request_timeout_secs = {args.request_timeout}
max_batch_tokens = 256
max_batch_seqs = 4
max_model_len = 2048
max_inflight_requests = 8
batch_wait_ms = 0
paged_block_size = 1
chunked_prefill_size = 32
enable_prefix_caching = false
mem_fraction_static = 0.9
num_blocks = 8192
ignore_eos = false
mode = "llm"
worker_id = "worker-0"
worker_heartbeat_timeout_secs = 10
log_level = "debug"
capture_sizes = [1, 2, 4]
[cuda_memory]
kernel_workspace_mib = 256
graph_arena_mib = 256
pool_retain_mib = 256
''')
    metadata = {
        "revision": os.environ.get("RUSTINFER_REVISION") or command_output(
            ["git", "rev-parse", "HEAD"]),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"]),
        "rust": command_output(["rustc", "--version"]),
        "source_status": command_output(["git", "status", "--short"]),
        "binary_sha256": {name: file_sha256(args.bin_dir / f"rustinfer-{name}")
                          for name in ("scheduler", "worker", "server")},
        "model": str(args.model),
        "model_files": {},
    }
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json",
                 "model.safetensors.index.json"):
        path = args.model / name
        if path.is_file():
            metadata["model_files"][name] = file_sha256(path)
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    (output / "image.png").write_bytes(red_png())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f"http://127.0.0.1:{port}"
    processes = {}
    checks = []
    responses = []
    started = time.monotonic()
    failure = None

    def start(name):
        with (output / f"{name}.log").open("w") as log:
            processes[name] = subprocess.Popen(
                [str(args.bin_dir / f"rustinfer-{name}"), "--config", str(config)],
                stdout=log, stderr=subprocess.STDOUT)

    def status(path):
        try:
            with opener.open(base_url + path, timeout=2) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code
        except urllib.error.URLError:
            return None

    def wait_until(predicate, timeout, description, check_processes=True):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if check_processes:
                for name, process in processes.items():
                    if process.poll() is not None:
                        raise RuntimeError(f"{name} exited before {description}")
            if predicate():
                return
            time.sleep(0.2)
        raise TimeoutError(description)

    def passed(name):
        checks.append(name)
        print("PASS:", name, flush=True)

    def chat(content, **kwargs):
        return dict(model="Qwen3.5-4B", messages=[dict(role="user", content=content)],
                    max_tokens=512, temperature=0, **kwargs)

    def request(payload, endpoint="/v1/chat/completions"):
        begin = time.monotonic()
        req = urllib.request.Request(base_url + endpoint, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
        with opener.open(req, timeout=args.request_timeout) as response:
            raw = response.read().decode()
        if payload.get("stream"):
            assert "data: [DONE]" in raw and "event: error" not in raw, raw
            chunks = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith("data: {")]
            content = "".join(c["choices"][0]["delta"].get("content", "")
                              for c in chunks if c.get("choices"))
            data = {"stream": chunks}
        else:
            data = json.loads(raw)
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content", choice.get("text"))
        responses.append({"request": payload, "response": data,
                          "elapsed_seconds": time.monotonic() - begin})
        return content

    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    print("Artifacts:", output, flush=True)
    try:
        start("scheduler")
        start("server")
        wait_until(lambda: status("/health") == 200, 30, "HTTP liveness")
        # Allow at least one periodic Pong; an old transport-only /ready will fail.
        for _ in range(20):
            assert status("/ready") == 503, "ready before Worker/model bootstrap"
            time.sleep(0.2)
        passed("loading instance stays unready while HTTP is alive")
        start("worker")
        wait_until(lambda: status("/ready") == 200, args.startup_timeout,
                   "model/worker/engine readiness")
        passed("startup self-check and model readiness")
        text = chat("What is 2 + 2? Answer briefly.")
        answer = request(text)
        assert answer and "4" in answer.split("</think>")[-1], answer
        assert request(dict(text, stream=True)) == answer
        passed("text generation and SSE agree, with output budget above batch budget")
        image = {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(red_png()).decode()}}
        visual = chat([image, {"type": "text", "text": "What color is this image? Answer briefly."}])
        color = request(visual)
        assert color and "red" in color.split("</think>")[-1].lower(), color
        assert request(dict(visual, stream=True)) == color
        passed("image recognition and SSE agree")
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            actual = list(pool.map(request, [text, visual, text, visual]))
        assert actual == [answer, color, answer, color], actual
        passed("mixed text/image concurrency preserves results")
        completion = request(dict(prompt="The capital of France is", temperature=0,
                                  max_tokens=16), "/v1/completions")
        assert completion and "Paris" in completion, completion
        passed("text completion endpoint")
        for payload in [chat("x" * (1024 * 1024 + 1)),
                        chat([{"type": "image_url", "image_url": {
                            "url": "data:image/png;base64,AAAA"}}])]:
            try:
                request(payload)
            except urllib.error.HTTPError as error:
                assert error.code == 400, (error.code, error.read())
            else:
                raise AssertionError("invalid request accepted")
        passed("oversized text and invalid image return 400")
        cancel = urllib.request.Request(base_url + "/v1/chat/completions",
            json.dumps(dict(visual, stream=True)).encode(), {"Content-Type": "application/json"})
        with opener.open(cancel, timeout=args.request_timeout) as response:
            response.readline()
        assert request(text) == answer
        passed("inference works after streaming client disconnect")
        with opener.open(base_url + "/metrics", timeout=5) as response:
            metrics = response.read().decode()
            assert "text/plain" in response.headers.get("Content-Type", "")
        (output / "metrics.prom").write_text(metrics)
        assert "# TYPE rustinfer_http_requests_total counter" in metrics, metrics
        assert 'status="400"' in metrics and 'status="200"' in metrics, metrics
        assert "rustinfer_scheduler_metrics_available 1" in metrics, metrics
        for prefix in ("rustinfer_scheduler_completions_total ",
                       'rustinfer_scheduler_kv_tokens{state="capacity"} '):
            samples = [float(line.split()[-1]) for line in metrics.splitlines()
                       if line.startswith(prefix)]
            assert samples and samples[0] > 0, (prefix, metrics)
        passed("Prometheus export includes HTTP outcomes and live scheduler/KV metrics")
        worker_log = (output / "worker.log").read_text()
        graph_replays = sum("replaying decode CUDA graph" in line and "multimodal=true" in line
                            for line in worker_log.splitlines())
        assert graph_replays > 0, "no image decode CUDA graph replay"
        passed(f"image decode CUDA Graph replay ({graph_replays} launches)")
        processes["worker"].terminate()
        wait_until(lambda: status("/ready") == 503, 30, "readiness after Worker loss",
                   check_processes=False)
        assert status("/health") == 200
        passed("worker loss withdraws readiness while HTTP stays alive")
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        (output / "results.json").write_text(json.dumps({
            "passed": failure is None, "checks": checks, "error": failure,
            "elapsed_seconds": time.monotonic() - started, "responses": responses,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
