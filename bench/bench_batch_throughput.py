"""Benchmark RustInfer worker throughput at varying concurrency.

Measures wall-clock throughput (tokens generated / total seconds) for
batch sizes 1, 2, 4, 8, 16, 32 by sending N completion requests to the
HTTP server with `concurrency=N` (so all requests are in-flight at once).

For each (model, concurrency) pair we send `concurrency` requests with
`max_tokens=128` and a fixed long prompt; total tokens generated is
divided by the wall-clock from "first request submitted" to "last response
received" to obtain end-to-end throughput.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import List

import aiohttp


@dataclass
class Result:
    completion_tokens: int
    latency_s: float
    text: str


async def send_one(session, url, prompt, max_tokens):
    payload = {
        "model": "llama3.2-1b",  # ignored by server, model_name is set at start
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.perf_counter()
    async with session.post(
        f"{url}/v1/completions", json=payload,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        body = await resp.json()
    elapsed = time.perf_counter() - start
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {body}")
    usage = body["usage"]
    text = body["choices"][0]["text"]
    return Result(usage["completion_tokens"], elapsed, text)


async def run_batch(url, prompts, max_tokens):
    """Send len(prompts) requests concurrently, return list of Result."""
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        tasks = [send_one(session, url, p, max_tokens) for p in prompts]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return wall, results


def pick_prompts(pool, n, min_chars=80):
    """Pick n distinct prompts that are reasonably long."""
    long_ones = [p for p in pool if isinstance(p, str) and len(p) >= min_chars]
    if len(long_ones) < n:
        long_ones = [p for p in pool if isinstance(p, str)]
    # Stable selection so runs are repeatable.
    return long_ones[:n]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--prompts", default="/root/RustInfer/bench/bench_prompts.json")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", default="llama3.2-1b")
    args = ap.parse_args()

    with open(args.prompts) as f:
        pool = json.load(f)
    print(f"loaded {len(pool)} prompts")

    # Warmup: a few small requests to JIT any first-time CUDA paths.
    if args.warmup > 0:
        async with aiohttp.ClientSession() as session:
            warm = [send_one(session, args.url, "Hello", 8) for _ in range(args.warmup)]
            await asyncio.gather(*warm)
        print(f"warmup: {args.warmup} requests done")

    batch_sizes = [int(b) for b in args.batches.split(",")]
    rows: List[dict] = []
    for bs in batch_sizes:
        prompts = pick_prompts(pool, bs)
        wall, results = await run_batch(args.url, prompts, args.max_tokens)
        total_completion = sum(r.completion_tokens for r in results)
        per_req_lat = [r.latency_s for r in results]
        per_req_tok = [r.completion_tokens / r.latency_s for r in results]
        agg_thru = total_completion / wall
        row = dict(
            label=args.label,
            batch=bs,
            max_tokens=args.max_tokens,
            wall_s=wall,
            total_completion=total_completion,
            agg_throughput_tok_s=agg_thru,
            per_req_min_tok_s=min(per_req_tok),
            per_req_mean_tok_s=statistics.mean(per_req_tok),
            per_req_max_tok_s=max(per_req_tok),
            per_req_lat_s_p50=statistics.median(per_req_lat),
            per_req_lat_s_max=max(per_req_lat),
        )
        rows.append(row)
        print(
            f"  batch={bs:>2}  wall={wall:6.2f}s  total_tok={total_completion:>5d}"
            f"  agg={agg_thru:7.1f} tok/s  per-req mean={row['per_req_mean_tok_s']:6.1f}"
            f"  p50_lat={row['per_req_lat_s_p50']:.2f}s"
        )

    print()
    print(f"=== Summary: {args.label} (max_tokens={args.max_tokens}) ===")
    print(f"{'batch':>6} {'agg tok/s':>11} {'mean tok/s':>11} {'p50 lat':>9}")
    for r in rows:
        print(
            f"{r['batch']:>6} {r['agg_throughput_tok_s']:>11.1f}"
            f" {r['per_req_mean_tok_s']:>11.1f} {r['per_req_lat_s_p50']:>8.2f}s"
        )

    # Save JSON for later aggregation.
    out = f"/tmp/bench_{args.label}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    asyncio.run(main())
