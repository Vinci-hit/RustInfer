"""Benchmark RustInfer with b=32 concurrency, 300 rounds.

Sends 32 concurrent requests per round for 300 rounds.
Reports overall and per-round statistics.
"""
import asyncio
import json
import time
import aiohttp
import statistics


URL = "http://127.0.0.1:8014"
MODEL = "llama3.2-1b"
PROMPT = "Explain the theory of relativity in simple terms. " * 5
MAX_TOKENS = 128
ROUNDS = 300
B = 32


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
        usage = body.get("usage", {})
        return elapsed, usage.get("completion_tokens", 0)


async def run_round(session, round_id, sem):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    t0 = time.perf_counter()
    tasks = [send_one(session, payload, sem) for _ in range(B)]
    results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)
    throughput = total_tokens / wall

    return {
        "round": round_id,
        "wall_s": wall,
        "total_tokens": total_tokens,
        "throughput_tok_s": throughput,
        "latencies": latencies,
    }


async def main():
    sem = asyncio.Semaphore(B)
    all_rounds = []

    async with aiohttp.ClientSession() as session:
        # warmup
        print("Warming up (1 request)...")
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
        start = time.perf_counter()
        async with session.post(
            f"{URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            await resp.json()
        print(f"  warmup done: {time.perf_counter() - start:.2f}s")

        print(f"\nRunning {ROUNDS} rounds, b={B}...")
        t_start = time.perf_counter()
        for i in range(ROUNDS):
            r = await run_round(session, i + 1, sem)
            all_rounds.append(r)
            if (i + 1) % 10 == 0:
                print(f"  round {i+1}/{ROUNDS} done, throughput={r['throughput_tok_s']:.0f} tok/s")
        total_wall = time.perf_counter() - t_start

    # aggregate stats
    all_throughputs = [r["throughput_tok_s"] for r in all_rounds]
    all_latencies = [l for r in all_rounds for l in r["latencies"]]
    total_tokens_all = sum(r["total_tokens"] for r in all_rounds)

    print(f"\n{'='*60}")
    print(f"300 Rounds Benchmark Summary (b={B})")
    print(f"{'='*60}")
    print(f"Total rounds:        {ROUNDS}")
    print(f"Total wall time:     {total_wall:.1f}s")
    print(f"Total tokens:        {total_tokens_all}")
    print(f"Overall throughput:  {total_tokens_all / total_wall:.1f} tok/s")
    print(f"")
    print(f"Per-round throughput:")
    print(f"  mean:    {statistics.mean(all_throughputs):.1f} tok/s")
    print(f"  median:  {statistics.median(all_throughputs):.1f} tok/s")
    print(f"  stdev:   {statistics.stdev(all_throughputs):.1f} tok/s")
    print(f"  min:     {min(all_throughputs):.1f} tok/s")
    print(f"  max:     {max(all_throughputs):.1f} tok/s")
    print(f"")
    print(f"Per-request latency:")
    print(f"  mean:    {statistics.mean(all_latencies):.3f}s")
    print(f"  median:  {statistics.median(all_latencies):.3f}s")
    print(f"  stdev:   {statistics.stdev(all_latencies):.3f}s")
    print(f"  min:     {min(all_latencies):.3f}s")
    print(f"  max:     {max(all_latencies):.3f}s")
    print(f"  p50:     {sorted(all_latencies)[len(all_latencies)//2]:.3f}s")
    print(f"  p95:     {sorted(all_latencies)[int(len(all_latencies)*0.95)]:.3f}s")
    print(f"  p99:     {sorted(all_latencies)[int(len(all_latencies)*0.99)]:.3f}s")
    print(f"{'='*60}")

    # save raw data
    with open("/root/RustInfer/bench_b32_300_results.json", "w") as f:
        json.dump({"rounds": all_rounds, "config": {"b": B, "rounds": ROUNDS, "max_tokens": MAX_TOKENS}}, f)
    print("Raw data saved to bench_b32_300_results.json")


if __name__ == "__main__":
    asyncio.run(main())
