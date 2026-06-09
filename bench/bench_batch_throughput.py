"""Benchmark RustInfer worker throughput at varying concurrency.

Measures wall-clock throughput (tokens generated / total seconds) for
batch sizes 1, 2, 4, 8, 16, 32 by sending N completion requests to the
HTTP server with `concurrency=N` (so all requests are in-flight at once).

For each (model, concurrency) pair we send `concurrency` requests with
`max_tokens=512` and a fixed long prompt; total tokens generated is
divided by the wall-clock from "first request submitted" to "last response
received" to obtain end-to-end throughput.

Saves full outputs to JSON for manual quality inspection.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List

import aiohttp


@dataclass
class Result:
    prompt: str
    completion_tokens: int
    latency_s: float
    text: str


async def send_one(session, url, prompt, max_tokens, model="llama3.2-1b", ignore_eos=False):
    payload = {
        # "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if ignore_eos:
        # Force exactly max_tokens of generation on both engines so the
        # throughput comparison measures equal work (vLLM + RustInfer both
        # accept this field). Without it, greedy decode stops at EOS and
        # short-answer prompts produce wildly different token counts.
        payload["ignore_eos"] = True
        payload["min_tokens"] = max_tokens
    start = time.perf_counter()
    async with session.post(
        f"{url}/v1/chat/completions", json=payload,
        timeout=aiohttp.ClientTimeout(total=600),
    ) as resp:
        body = await resp.json()
    elapsed = time.perf_counter() - start
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {body}")
    usage = body["usage"]
    text = body["choices"][0]["message"]["content"]
    return Result(prompt, usage["completion_tokens"], elapsed, text)


async def run_batch(url, prompts, max_tokens, model="llama3.2-1b", ignore_eos=False):
    """Send len(prompts) requests concurrently, return list of Result."""
    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        tasks = [send_one(session, url, p, max_tokens, model, ignore_eos) for p in prompts]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return wall, results


def pick_prompts(pool, n, min_chars=80):
    """Pick n distinct prompts that are reasonably long."""
    long_ones = [p for p in pool if isinstance(p, str) and len(p) >= min_chars]
    if len(long_ones) < n:
        long_ones = [p for p in pool if isinstance(p, str)]
    return long_ones[:n]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--prompts", default="/root/RustInfer/bench/bench_prompts.json")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--label", default="llama3.2-1b")
    ap.add_argument("--model", default="llama3.2-1b",
                    help="Model name sent in the API request payload")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Force exactly max_tokens of generation on both engines "
                         "(sends ignore_eos+min_tokens) for an equal-work comparison")
    ap.add_argument("--output", default="/tmp/bench_results.json",
                    help="Save full results (with generated text) for quality check")
    args = ap.parse_args()

    with open(args.prompts) as f:
        pool = json.load(f)
    print(f"loaded {len(pool)} prompts")

    # Warmup — need longer outputs to stabilize cuBLASLt algo selection
    if args.warmup > 0:
        async with aiohttp.ClientSession() as session:
            warm = [send_one(session, args.url, "Write a detailed explanation of how computers work.", 100, args.model, args.ignore_eos) for _ in range(5)]
            await asyncio.gather(*warm)
        print(f"warmup: 5 requests x 100 tokens done")

    batch_sizes = [int(b) for b in args.batches.split(",")]
    all_results: List[dict] = []
    rows: List[dict] = []

    for bs in batch_sizes:
        prompts = pick_prompts(pool, bs)
        wall, results = await run_batch(args.url, prompts, args.max_tokens, args.model, args.ignore_eos)
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

        # Save individual results for quality inspection
        for r in results:
            all_results.append(asdict(r))

    print()
    print(f"=== Summary: {args.label} (max_tokens={args.max_tokens}) ===")
    print(f"{'batch':>6} {'agg tok/s':>11} {'mean tok/s':>11} {'p50 lat':>9}")
    for r in rows:
        print(
            f"{r['batch']:>6} {r['agg_throughput_tok_s']:>11.1f}"
            f" {r['per_req_mean_tok_s']:>11.1f} {r['per_req_lat_s_p50']:>8.2f}s"
        )

    # Save full output for quality check
    output_data = {
        "summary": rows,
        "samples": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {args.output}")
    print(f"  - {len(rows)} batch configs")
    print(f"  - {len(all_results)} generation samples (check 'samples' for text quality)")

    # Print a few sample outputs for quick eyeball check
    print(f"\n=== Sample outputs (first 3) ===")
    for i, r in enumerate(all_results[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {r['prompt'][:100]}")
        print(f"Output ({r['completion_tokens']} tokens): {r['text'][:300]}...")


if __name__ == "__main__":
    asyncio.run(main())
