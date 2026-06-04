"""Online serving benchmark: burst arrival, 60s duration, no max_tokens limit.

Simulates real traffic with burst arrival pattern against both RustInfer
and vLLM servers. Measures throughput, latency distribution, and quality.

Usage:
    # Against RustInfer (default port 8000)
    python3 bench/bench_online_burst.py --target rustinfer --duration 60

    # Against vLLM (default port 8001)
    python3 bench/bench_online_burst.py --target vllm --port 8001 --duration 60
"""
import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, asdict
from typing import List

import aiohttp


@dataclass
class RequestResult:
    prompt: str
    completion_tokens: int
    prompt_tokens: int
    latency_s: float
    ttft_s: float  # time to first token (0 if non-streaming)
    text: str
    success: bool
    error: str


RUSTINFER_MODEL = "llama3.2-1b"
VLLM_MODEL = "default"


async def send_rustinfer(session, url, prompt):
    """Send chat completion to RustInfer server."""
    payload = {
        "model": RUSTINFER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
        elapsed = time.perf_counter() - start
        if resp.status != 200:
            return RequestResult(prompt[:100], 0, 0, elapsed, 0, "", False, str(body))
        usage = body["usage"]
        text = body["choices"][0]["message"]["content"]
        return RequestResult(
            prompt[:100],
            usage["completion_tokens"],
            usage.get("prompt_tokens", 0),
            elapsed, 0, text, True, "",
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(prompt[:100], 0, 0, elapsed, 0, "", False, str(e))


async def send_vllm(session, url, prompt):
    """Send chat completion to vLLM server (server applies its own chat template)."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
        elapsed = time.perf_counter() - start
        if resp.status != 200:
            return RequestResult(prompt[:100], 0, 0, elapsed, 0, "", False, str(body))
        usage = body["usage"]
        text = body["choices"][0]["message"]["content"]
        return RequestResult(
            prompt[:100],
            usage["completion_tokens"],
            usage.get("prompt_tokens", 0),
            elapsed, 0, text, True, "",
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(prompt[:100], 0, 0, elapsed, 0, "", False, str(e))


async def run_burst(url, prompts, duration_s, concurrency, target):
    """Send requests in bursts for `duration_s` seconds."""
    send_fn = send_rustinfer if target == "rustinfer" else send_vllm
    results: List[RequestResult] = []
    total_sent = 0
    start_time = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        pending = set()
        prompt_idx = 0

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration_s and len(pending) == 0:
                break

            # Fill up to concurrency slots (burst: send as fast as possible)
            while len(pending) < concurrency and (time.perf_counter() - start_time) < duration_s:
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                task = asyncio.create_task(send_fn(session, url, prompt))
                pending.add(task)
                total_sent += 1

            if not pending:
                break

            # Wait for at least one to complete
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                results.append(task.result())

        # Drain remaining
        if pending:
            done, _ = await asyncio.wait(pending)
            for task in done:
                results.append(task.result())

    return results, total_sent


def compute_stats(results: List[RequestResult], duration_s: float, label: str):
    """Compute and print statistics."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        print(f"  {label}: ALL REQUESTS FAILED ({len(failed)} errors)")
        for r in failed[:3]:
            print(f"    {r.error[:100]}")
        return {}

    total_completion = sum(r.completion_tokens for r in successful)
    total_prompt = sum(r.prompt_tokens for r in successful)
    latencies = sorted([r.latency_s for r in successful])
    tok_per_req = [r.completion_tokens / r.latency_s for r in successful if r.latency_s > 0]

    wall_time = max(r.latency_s for r in successful)  # approximate
    actual_duration = latencies[-1] if latencies else duration_s

    stats = {
        "label": label,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_completion_tokens": total_completion,
        "total_prompt_tokens": total_prompt,
        "throughput_tok_s": total_completion / duration_s,
        "requests_per_s": len(successful) / duration_s,
        "latency_p50": latencies[len(latencies) // 2],
        "latency_p90": latencies[int(len(latencies) * 0.9)],
        "latency_p99": latencies[int(len(latencies) * 0.99)],
        "latency_mean": sum(latencies) / len(latencies),
        "latency_max": latencies[-1],
        "per_req_tok_s_mean": sum(tok_per_req) / len(tok_per_req) if tok_per_req else 0,
        "avg_completion_tokens": total_completion / len(successful),
    }

    print(f"\n{'═' * 60}")
    print(f"  {label} — Online Serving Benchmark (burst, {duration_s}s)")
    print(f"{'═' * 60}")
    print(f"  Requests:    {stats['successful']} ok / {stats['failed']} failed / {stats['total_requests']} total")
    print(f"  Throughput:  {stats['throughput_tok_s']:.0f} tok/s (completion)")
    print(f"  Requests/s:  {stats['requests_per_s']:.1f}")
    print(f"  Avg output:  {stats['avg_completion_tokens']:.0f} tokens/req")
    print(f"  Latency:")
    print(f"    p50:  {stats['latency_p50']:.3f}s")
    print(f"    p90:  {stats['latency_p90']:.3f}s")
    print(f"    p99:  {stats['latency_p99']:.3f}s")
    print(f"    mean: {stats['latency_mean']:.3f}s")
    print(f"    max:  {stats['latency_max']:.3f}s")
    print(f"  Per-req:     {stats['per_req_tok_s_mean']:.0f} tok/s/req")

    if failed:
        print(f"  Errors: {failed[0].error[:80]}")

    return stats


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["rustinfer", "vllm"], required=True)
    ap.add_argument("--url", default=None, help="Override URL (default: localhost:8000 for RI, :8001 for vLLM)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--prompts", default="/root/RustInfer/bench/bench_prompts.json")
    ap.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    ap.add_argument("--concurrency", type=int, default=32, help="Max concurrent requests")
    ap.add_argument("--output", default=None, help="Save results JSON")
    ap.add_argument("--model", default=None, help="Model name sent to RustInfer (selects chat template)")
    args = ap.parse_args()

    if args.model:
        global RUSTINFER_MODEL, VLLM_MODEL
        RUSTINFER_MODEL = args.model
        VLLM_MODEL = args.model

    if args.url:
        url = args.url
    elif args.port:
        url = f"http://localhost:{args.port}"
    else:
        url = "http://localhost:8000" if args.target == "rustinfer" else "http://localhost:8001"

    with open(args.prompts) as f:
        pool = json.load(f)
    # Filter reasonable length prompts
    prompts = [p for p in pool if isinstance(p, str) and 40 <= len(p) <= 500]
    random.seed(42)
    random.shuffle(prompts)
    print(f"Loaded {len(prompts)} prompts (filtered from {len(pool)})")
    print(f"Target: {args.target} @ {url}")
    print(f"Duration: {args.duration}s, Concurrency: {args.concurrency}")

    # Warmup
    print("Warming up...")
    send_fn = send_rustinfer if args.target == "rustinfer" else send_vllm
    async with aiohttp.ClientSession() as session:
        for _ in range(3):
            await send_fn(session, url, "Hello, how are you?")
    print("Warmup done.")

    # Run
    print(f"\nRunning {args.duration}s burst benchmark...")
    results, total_sent = await run_burst(url, prompts, args.duration, args.concurrency, args.target)

    # Stats
    label = f"{args.target}-c{args.concurrency}"
    stats = compute_stats(results, args.duration, label)

    # Save
    out_path = args.output or f"/tmp/bench_online_{args.target}.json"
    output_data = {
        "config": {
            "target": args.target,
            "url": url,
            "duration": args.duration,
            "concurrency": args.concurrency,
        },
        "stats": stats,
        "samples": [asdict(r) for r in results[:50]],  # first 50 for quality check
    }
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
