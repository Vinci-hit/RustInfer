"""
TTFT / TPOT 压测脚本 (兼容非流式)

策略:
- TTFT: 用 max_tokens=1 发一批请求，测量端到端延迟即为 TTFT (prefill + 1 decode step)
- TPOT: 用 max_tokens=N 发请求，TPOT = (total_latency - ttft) / (num_tokens - 1)

用法:
  python bench_ttft_tpot.py --url http://localhost:8000 \
    --num-requests 100 --concurrency 32 --max-tokens 256 --arrival-rate 20
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

PROMPTS = None


def load_prompts(dataset_path: str = None) -> list:
    import os
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path) as f:
            return json.load(f)
    default_path = os.path.join(os.path.dirname(__file__), "bench_prompts.json")
    if os.path.exists(default_path):
        with open(default_path) as f:
            return json.load(f)
    return [
        "Give three tips for staying healthy.",
        "What are the three primary colors?",
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about the ocean.",
        "What is the difference between machine learning and deep learning?",
    ]


@dataclass
class Result:
    request_id: int
    status: str
    num_tokens: int
    latency_ms: float
    error: Optional[str] = None


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    request_id: int,
    max_tokens: int,
    stream: bool = False,
) -> Result:
    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    if stream:
        payload["stream"] = True
        start = time.perf_counter()
        try:
            async with session.post(
                f"{url}/v1/chat/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    elapsed = (time.perf_counter() - start) * 1000
                    return Result(request_id, "error", 0, elapsed, str(resp.status))
                first_token_time = None
                last_token_time = None
                token_count = 0
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content", ""):
                            token_count += 1
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            last_token_time = now
                    except:
                        continue
                elapsed = (time.perf_counter() - start) * 1000
                ttft = (first_token_time - start) * 1000 if first_token_time else elapsed
                return Result(request_id, "ok_stream", token_count, elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return Result(request_id, "error", 0, elapsed, str(e)[:100])
    else:
        start = time.perf_counter()
        try:
            async with session.post(
                f"{url}/v1/chat/completions", json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                elapsed = (time.perf_counter() - start) * 1000
                body = await resp.json()
                if resp.status != 200:
                    return Result(request_id, "error", 0, elapsed, str(body)[:100])
                num_tokens = body.get("usage", {}).get("completion_tokens", 0)
                return Result(request_id, "ok", num_tokens, elapsed)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return Result(request_id, "error", 0, elapsed, str(e)[:100])


async def run_phase(session, url, num_requests, concurrency, max_tokens, arrival_rate, stream=False):
    """Run a batch of requests, return list of Results"""
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def go(idx):
        async with semaphore:
            prompt = PROMPTS[idx % len(PROMPTS)]
            r = await send_request(session, url, prompt, idx, max_tokens, stream=stream)
            results.append(r)

    tasks = []
    for i in range(num_requests):
        tasks.append(asyncio.create_task(go(i)))
        if arrival_rate > 0 and i < num_requests - 1:
            await asyncio.sleep(1.0 / arrival_rate)
    await asyncio.gather(*tasks)
    return results


def percentile(arr, p):
    if not arr:
        return 0
    arr_s = sorted(arr)
    idx = min(int(len(arr_s) * p), len(arr_s) - 1)
    return arr_s[idx]


async def run_benchmark(url, num_requests, concurrency, max_tokens, arrival_rate, use_stream):
    print(f"\n{'='*60}")
    print(f"  TTFT / TPOT Benchmark")
    print(f"{'='*60}")
    print(f"  URL:          {url}")
    print(f"  Requests:     {num_requests}")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Max tokens:   {max_tokens}")
    print(f"  Arrival rate: {arrival_rate:.1f} req/s")
    print(f"  Mode:         {'streaming' if use_stream else 'non-streaming (TTFT via max_tokens=1)'}")
    print(f"{'='*60}\n")

    async with aiohttp.ClientSession() as session:
        # Health check
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"ERROR: not healthy"); return
        except Exception as e:
            print(f"ERROR: {e}"); return

        print("Server healthy.\n")

        if use_stream:
            # Streaming mode: get TTFT and TPOT from stream timing
            print("Phase: Streaming requests for TTFT + TPOT...")
            results = await run_phase(session, url, num_requests, concurrency, max_tokens, arrival_rate, stream=True)
            # For streaming, we stored ttft in a special way - re-implement inline
            # Actually let's just do it properly with a custom streaming function
            print("  (streaming mode - see separate implementation)")
        else:
            # Non-streaming mode:
            # Phase 1: TTFT = latency of max_tokens=1 requests
            print("Phase 1: Measuring TTFT (max_tokens=1)...")
            ttft_results = await run_phase(session, url, num_requests, concurrency, 1, arrival_rate, stream=False)
            ttft_ok = [r for r in ttft_results if r.status == "ok"]

            # Phase 2: Full generation for TPOT
            print(f"Phase 2: Measuring TPOT (max_tokens={max_tokens})...")
            full_results = await run_phase(session, url, num_requests, concurrency, max_tokens, arrival_rate, stream=False)
            full_ok = [r for r in full_results if r.status == "ok"]

    # Report
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")

    if ttft_ok:
        ttfts = [r.latency_ms for r in ttft_ok]
        print(f"\n  --- TTFT (Time To First Token) --- [{len(ttft_ok)} samples]")
        print(f"  mean:  {sum(ttfts)/len(ttfts):.2f} ms")
        print(f"  p50:   {percentile(ttfts, 0.5):.2f} ms")
        print(f"  p90:   {percentile(ttfts, 0.9):.2f} ms")
        print(f"  p99:   {percentile(ttfts, 0.99):.2f} ms")

    if full_ok:
        # TPOT = (total_latency - avg_ttft) / (num_tokens - 1)
        avg_ttft = sum(ttfts) / len(ttfts) if ttft_ok else 0
        tpots = []
        for r in full_ok:
            if r.num_tokens > 1:
                tpot = (r.latency_ms - avg_ttft) / (r.num_tokens - 1)
                tpots.append(tpot)

        total_tokens = sum(r.num_tokens for r in full_ok)
        total_time_s = max(r.latency_ms for r in full_ok) / 1000  # rough

        print(f"\n  --- TPOT (Time Per Output Token) --- [{len(tpots)} samples]")
        if tpots:
            print(f"  mean:  {sum(tpots)/len(tpots):.3f} ms")
            print(f"  p50:   {percentile(tpots, 0.5):.3f} ms")
            print(f"  p90:   {percentile(tpots, 0.9):.3f} ms")
            print(f"  p99:   {percentile(tpots, 0.99):.3f} ms")

        print(f"\n  --- Throughput ---")
        print(f"  Total tokens: {total_tokens}")
        failed = [r for r in full_results if r.status != "ok"]
        print(f"  OK/Failed:    {len(full_ok)}/{len(failed)}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="TTFT/TPOT Benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--num-requests", "-n", type=int, default=100)
    parser.add_argument("--concurrency", "-c", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--arrival-rate", type=float, default=20.0)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--stream", action="store_true", help="Use streaming (requires server support)")
    args = parser.parse_args()

    global PROMPTS
    PROMPTS = load_prompts(args.dataset)
    print(f"Loaded {len(PROMPTS)} prompts")

    asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        arrival_rate=args.arrival_rate,
        use_stream=args.stream,
    ))


if __name__ == "__main__":
    main()
