"""Start one serving stack, warm it, measure identical SSE requests, then stop it.

Run under nsys to capture the HTTP client, server/scheduler/worker process tree.
NVTX request ranges enclose the whole chain. No persistent service is left behind.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["rustinfer", "llama"], required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvtx-library", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, default=Path("/tmp/rustinfer-llama-reference/build/bin/llama-server"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", type=Path, default=Path("rustinfer.toml"))
    parser.add_argument("--chat", action="store_true", help="Include embedded chat-template rendering and tokenization")
    parser.add_argument("--llama-checkpoints", type=int, help="Override llama-server's recurrent checkpoint policy")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workload = json.loads(args.workload.read_text())
    nvtx = ctypes.CDLL(str(args.nvtx_library.resolve()))
    nvtx.ri_nvtx_push.argtypes = [ctypes.c_char_p]
    nvtx.ri_nvtx_push.restype = None
    nvtx.ri_nvtx_pop.argtypes = []
    nvtx.ri_nvtx_pop.restype = None
    processes, files = [], []
    result = {"engine": args.engine, "chat": args.chat, "requests": [], "commands": []}
    base = f"http://127.0.0.1:{args.port}"

    def start(name, command):
        output = (args.output / f"{name}.log").open("w")
        files.append(output)
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        result["commands"].append(command)

    def request():
        body = {"model": model_id, "prompt": workload["input_ids"], "temperature": 0,
                "max_tokens": 32, "stream": True, "stream_options": {"include_usage": True},
                "cache_prompt": False}
        endpoint = "/v1/completions"
        if args.chat:
            endpoint = "/v1/chat/completions"
            del body["prompt"]
            body["messages"] = [{"role": "user", "content": workload["text_prompt"]}]
            if args.engine == "rustinfer":
                body["enable_thinking"] = False
            else:
                body["chat_template_kwargs"] = {"enable_thinking": False}
        begun = time.perf_counter()
        first = None
        pieces, usage, finish, done = [], None, None, False
        data = json.dumps(body).encode()
        req = urllib.request.Request(base + endpoint, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise RuntimeError(chunk)
                for choice in chunk.get("choices", []):
                    text = (choice.get("delta", {}).get("content") if args.chat else choice.get("text")) or ""
                    if text:
                        if first is None:
                            first = time.perf_counter() - begun
                        pieces.append(text)
                    finish = choice.get("finish_reason") or finish
                usage = chunk.get("usage") or usage
        elapsed = time.perf_counter() - begun
        if not (done and first and usage and finish):
            raise RuntimeError("incomplete SSE response")
        return {"ttft_seconds": first, "wall_seconds": elapsed, "usage": usage,
                "text": "".join(pieces), "finish_reason": finish,
                "decode_tokens_per_second": (usage["completion_tokens"] - 1) / (elapsed - first)}

    try:
        # Never take over another running service.
        import socket
        with socket.socket() as check:
            if check.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"port {args.port} is already in use")
        if args.engine == "rustinfer":
            import tomllib
            config = tomllib.loads(args.config.read_text())
            if config["port"] != args.port or config["model"] != workload["model"]:
                raise RuntimeError("root config must match workload model and port")
            for name in ("scheduler", "worker", "server"):
                start(name, [str(Path(f"target/release/rustinfer-{name}").resolve()), "--config", str(args.config.resolve())])
            health = "/ready"
        else:
            start("llama-server", [str(args.llama_server.resolve()), "-m", workload["model"],
                "--host", "127.0.0.1", "--port", str(args.port), "-ngl", "999", "-c", str(workload["context"]),
                "-b", "64", "-ub", "64", "-np", "1", "-t", "4", "-tb", "4",
                "-ctk", "bf16", "-ctv", "bf16", "-fa", "on", "--cache-ram", "0", "--no-cache-prompt",
                *([] if args.llama_checkpoints is None else ["--ctx-checkpoints", str(args.llama_checkpoints)])])
            health = "/health"
        deadline = time.monotonic() + 180
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("server process exited; inspect logs")
            try:
                with urllib.request.urlopen(base + health, timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("service did not become ready")
            time.sleep(0.5)
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
            model_id = json.load(response)["data"][0]["id"]
        result["warmup"] = request()
        nvtx.ri_nvtx_push(b"measured")
        try:
            for i in range(workload["repeats"]):
                nvtx.ri_nvtx_push(f"request/{i}".encode())
                try:
                    row = request()
                finally:
                    nvtx.ri_nvtx_pop()
                if row["text"] != result["warmup"]["text"]:
                    raise RuntimeError("greedy response changed between runs")
                if row["usage"]["prompt_tokens"] != len(workload["input_ids"]):
                    raise RuntimeError("prompt token count changed")
                result["requests"].append(row)
                print(args.engine, i, row, flush=True)
        finally:
            nvtx.ri_nvtx_pop()
            # End the nsys capture while workers are alive, allowing CUPTI to flush.
            # Killing an active worker first can silently lose its last GPU buffers.
            time.sleep(2)
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for file in files:
            file.close()
        (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
