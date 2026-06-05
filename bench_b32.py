"""Benchmark RustInfer with b=32 concurrency.

Sends 32 concurrent requests to /v1/chat/completions, measures
throughput and latency. Warms up first.
"""
import asyncio
import json
import time
import aiohttp


URL = "http://127.0.0.1:8014"
MODEL = "llama3.2-1b"
PROMPT = "Explain the theory of relativity in simple terms. " * 5
MAX_TOKENS = 128


async def send_one(session, payload, sem):
    async with sem:
        start = time.perf_counter()
        async with session.post(
            f"{URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
        elapsed = time.perf_counter() - start
        return elapsed, body


async def run_bench(num_requests, max_concurrency, warmup=False):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }

    sem = asyncio.Semaphore(max_concurrency)
    async with aiohttp.ClientSession() as session:
        if warmup:
            # warmup: send 1 request
            print("Warming up (1 request)...")
            elapsed, body = await send_one(session, payload, asyncio.Semaphore(1))
            print(f"  warmup done: {elapsed:.2f}s")
            return

        print(f"Running benchmark: {num_requests} requests, concurrency={max_concurrency}")
        t0 = time.perf_counter()
        tasks = [send_one(session, payload, sem) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    # collect stats
    latencies = [r[0] for r in results]
    total_tokens = 0
    for _, body in results:
        usage = body.get("usage", {})
        total_tokens += usage.get("completion_tokens", 0)

    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    throughput = total_tokens / wall

    print(f"\n{'='*50}")
    print(f"Benchmark Results (b={num_requests}, concurrency={max_concurrency})")
    print(f"{'='*50}")
    print(f"Total requests:     {num_requests}")
    print(f"Total wall time:    {wall:.2f}s")
    print(f"Total tokens:       {total_tokens}")
    print(f"Throughput:         {throughput:.1f} tok/s")
    print(f"Avg latency:        {avg_latency:.2f}s")
    print(f"P50 latency:        {p50:.2f}s")
    print(f"P95 latency:        {p95:.2f}s")
    print(f"P99 latency:        {p99:.2f}s")
    print(f"{'='*50}")


async def main():
    # warmup
    await run_bench(1, 1, warmup=True)
    # benchmark: b=32
    await run_bench(32, 32, warmup=False)


if __name__ == "__main__":
    asyncio.run(main())
