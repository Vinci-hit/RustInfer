"""Online benchmark: same prompts, one tag at a time.

Usage:
    python3 bench/bench_online_compare.py --tag rustinfer   # bench :8000 as rustinfer
    python3 bench/bench_online_compare.py --tag vllm        # bench :8000 as vllm
"""

import argparse
import asyncio
import json
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
    ttft_s: float
    text: str
    success: bool
    error: str


RUSTINFER_MODEL = "llama3.2-1b"
URL = "http://127.0.0.1:8000"
DURATION = 60
CONCURRENCY = 32
PROMPTS_FILE = "/root/RustInfer/bench/bench_prompts.json"


async def send_rustinfer(session, url, prompt):
    payload = {
        "model": RUSTINFER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            body = await resp.json()
        elapsed = time.perf_counter() - start
        if resp.status != 200:
            return RequestResult(prompt[:100], 0, 0, elapsed, 0, "", False, str(body))
        usage = body["usage"]
        text = body["choices"][0]["message"]["content"]
        return RequestResult(
            prompt[:100], usage["completion_tokens"], usage.get("prompt_tokens", 0),
            elapsed, 0, text, True, "",
        )
    except Exception as e:
        return RequestResult(prompt[:100], 0, 0, time.perf_counter() - start, 0, "", False, str(e))


async def send_vllm(session, url, prompt):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "ignore_eos": True,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            body = await resp.json()
        elapsed = time.perf_counter() - start
        if resp.status != 200:
            return RequestResult(prompt[:100], 0, 0, elapsed, 0, "", False, str(body))
        usage = body["usage"]
        text = body["choices"][0]["message"]["content"]
        return RequestResult(
            prompt[:100], usage["completion_tokens"], usage.get("prompt_tokens", 0),
            elapsed, 0, text, True, "",
        )
    except Exception as e:
        return RequestResult(prompt[:100], 0, 0, time.perf_counter() - start, 0, "", False, str(e))


async def run_bench(session, url, prompts, duration_s, concurrency, send_fn):
    results: List[RequestResult] = []
    start_time = time.perf_counter()
    pending = set()
    prompt_idx = 0

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= duration_s and not pending:
            break
        while len(pending) < concurrency and time.perf_counter() - start_time < duration_s:
            prompt = prompts[prompt_idx % len(prompts)]
            prompt_idx += 1
            pending.add(asyncio.create_task(send_fn(session, url, prompt)))
        if not pending:
            break
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            results.append(t.result())
    if pending:
        done, _ = await asyncio.wait(pending)
        for t in done:
            results.append(t.result())
    return results


def compute_stats(results, duration_s, label):
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    if not successful:
        print(f"  {label}: ALL FAILED ({len(failed)} errors)")
        for r in failed[:3]:
            print(f"    {r.error[:100]}")
        return {}

    total_completion = sum(r.completion_tokens for r in successful)
    total_prompt = sum(r.prompt_tokens for r in successful)
    latencies = sorted(r.latency_s for r in successful)
    tok_per_req = [r.completion_tokens / r.latency_s for r in successful if r.latency_s > 0]

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

    print(f"\n{'=' * 60}")
    print(f"  {label} — {duration_s}s, c={len(results)}")
    print(f"{'=' * 60}")
    print(f"  Requests:    {stats['successful']} ok / {stats['failed']} failed")
    print(f"  Throughput:  {stats['throughput_tok_s']:.0f} tok/s")
    print(f"  Requests/s:  {stats['requests_per_s']:.1f}")
    print(f"  Avg output:  {stats['avg_completion_tokens']:.0f} tokens/req")
    print(f"  Latency:     p50={stats['latency_p50']:.3f}s  p90={stats['latency_p90']:.3f}s  p99={stats['latency_p99']:.3f}s")
    print(f"  Latency:     mean={stats['latency_mean']:.3f}s  max={stats['latency_max']:.3f}s")
    print(f"  Per-req:     {stats['per_req_tok_s_mean']:.0f} tok/s/req")
    if failed:
        print(f"  Errors: {failed[0].error[:80]}")
    return stats


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["rustinfer", "vllm"],
                     help="Which target is running on :8000")
    ap.add_argument("--url", default=URL, help="Target URL (default: " + URL + ")")
    ap.add_argument("--duration", type=int, default=DURATION)
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    with open(PROMPTS_FILE) as f:
        pool = json.load(f)
    prompts = [p for p in pool if isinstance(p, str) and 40 <= len(p) <= 500]
    print(f"Loaded {len(prompts)} prompts")
    print(f"Tag: {args.tag}, URL: {args.url}, Duration: {args.duration}s, Concurrency: {args.concurrency}")

    send_fn = send_rustinfer if args.tag == "rustinfer" else send_vllm
    async with aiohttp.ClientSession() as session:
        print("Warming up...")
        for _ in range(3):
            await send_fn(session, args.url, "Hello, how are you?")
        print("Warmup done.")
        results = await run_bench(session, args.url, prompts, args.duration,
                                  args.concurrency, send_fn)

    label = f"{args.tag}-c{args.concurrency}"
    stats = compute_stats(results, args.duration, label)
    out = f"/tmp/bench_online_{args.tag}.json"
    with open(out, "w") as f:
        json.dump({
            "config": {"target": args.tag, "url": args.url,
                       "duration": args.duration, "concurrency": args.concurrency},
            "stats": stats,
            "samples": [asdict(r) for r in results[:50]],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    asyncio.run(main())
