"""
真实到达时间 online benchmark。

特点：
- 不同 prompt 长度（短/中/长混合，固定 seed 可复现）
- 不同请求到达时间（指数分布 inter-arrival，模拟真实流量）
- warmup 请求不计入统计
- 对 OpenAI-compatible `/v1/chat/completions` 服务通用，可用于 RustInfer / vLLM 对比
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp


BASE_SNIPPETS = [
    "Solve a difficult distributed systems design problem: build a globally replicated low-latency LLM serving platform and analyze scheduler correctness under failures.",
    "Prove or disprove a performance claim about continuous batching with stochastic arrivals, variable prompt lengths, and bounded GPU memory.",
    "Design an asynchronous CUDA inference pipeline and reason carefully about stream ordering, buffer ownership, and request completion races.",
    "Analyze a production incident where long prefill requests, short decode requests, and client disconnects interact in a high-concurrency model server.",
    "Give a rigorous engineering proposal for minimizing tail latency while preserving throughput in an online transformer serving system.",
    "Compare two architectures for request scheduling: a CPU-driven pipeline and a device-resident metadata pipeline, including failure modes and tradeoffs.",
    "Explain how to validate that double buffering actually overlaps CPU work with GPU decode when request arrivals are irregular and prompts have mixed lengths.",
    "Write a detailed technical review of CUDA Graph based decode serving, including graph bucketization, buffer reuse, synchronization hazards, and metrics.",
]

FILLER = (
    "The answer should be long, concrete, and analytical. Include assumptions, invariants, edge cases, "
    "failure scenarios, pseudo-code, complexity analysis, metrics, experimental methodology, and tradeoffs. "
    "Discuss queueing effects, stochastic arrivals, prompt length variance, GPU streams, CUDA events, H2D and D2H transfers, "
    "double buffering, prefill/decode interference, cancellation, EOS handling, backpressure, and tail latency. "
)


@dataclass
class RequestSpec:
    request_id: int
    arrival_s: float
    prompt: str
    max_tokens: int


@dataclass
class RequestResult:
    request_id: int
    arrival_s: float
    prompt_words: int
    status: str
    output_tokens: int
    latency_ms: float
    ttft_ms: Optional[float]
    error: Optional[str] = None


def percentile(values, p):
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(int(len(xs) * p), len(xs) - 1)
    return xs[idx]


def make_prompt(target_words: int, rng: random.Random) -> str:
    text = rng.choice(BASE_SNIPPETS) + " "
    while len(text.split()) < target_words:
        text += FILLER
    words = text.split()[:target_words]
    return " ".join(words)


def make_specs(num_requests: int, arrival_rate: float, max_tokens: int, seed: int) -> list[RequestSpec]:
    rng = random.Random(seed)
    # 覆盖明显不同的 prompt 长度；不是 token 精确长度，但足够形成 prefill 差异。
    prompt_word_buckets = [8, 16, 32, 64, 96, 128, 192, 256]
    specs = []
    t = 0.0
    for i in range(num_requests):
        if i > 0 and arrival_rate > 0:
            # Poisson arrival: exponential inter-arrival，固定 seed 可复现。
            t += rng.expovariate(arrival_rate)
        words = prompt_word_buckets[i % len(prompt_word_buckets)]
        # 加一点扰动，避免完全周期化。
        words = max(4, int(words * rng.uniform(0.85, 1.15)))
        specs.append(RequestSpec(i, t, make_prompt(words, rng), max_tokens))
    return specs


async def health_check(url: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"health check failed: HTTP {resp.status}")


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    spec: RequestSpec,
    stream: bool,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": spec.max_tokens,
        "min_tokens": spec.max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": stream,
    }
    start = time.perf_counter()
    prompt_words = len(spec.prompt.split())
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if stream:
                if resp.status != 200:
                    body = await resp.text()
                    return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "error", 0,
                                         (time.perf_counter() - start) * 1000.0, None, body[:200])
                first_token_time = None
                output_tokens = 0
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        output_tokens += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                end = time.perf_counter()
                ttft_ms = (first_token_time - start) * 1000.0 if first_token_time else None
                return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "ok", output_tokens,
                                     (end - start) * 1000.0, ttft_ms)

            body = await resp.json()
            end = time.perf_counter()
            if resp.status != 200:
                return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "error", 0,
                                     (end - start) * 1000.0, None, str(body)[:200])
            usage = body.get("usage") or {}
            output_tokens = int(usage.get("completion_tokens") or 0)
            return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "ok", output_tokens,
                                 (end - start) * 1000.0, None)
    except Exception as e:
        return RequestResult(spec.request_id, spec.arrival_s, prompt_words, "error", 0,
                             (time.perf_counter() - start) * 1000.0, None, repr(e)[:200])


async def run_specs(url: str, model: str, specs: list[RequestSpec], concurrency: int, stream: bool, verbose: bool):
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []
    start = time.perf_counter()

    async def one(spec: RequestSpec):
        delay = start + spec.arrival_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                r = await send_request(session, url, model, spec, stream)
            results.append(r)
            if verbose:
                mark = "ok" if r.status == "ok" else "ERR"
                print(f"  [{mark}] req-{r.request_id:03d} arrive={r.arrival_s:7.3f}s "
                      f"prompt_words={r.prompt_words:3d} out={r.output_tokens:3d} "
                      f"lat={r.latency_ms:8.1f}ms")

    await asyncio.gather(*(asyncio.create_task(one(s)) for s in specs))
    wall_s = time.perf_counter() - start
    return sorted(results, key=lambda r: r.request_id), wall_s


def summarize(label: str, results: list[RequestResult], wall_s: float):
    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    lat = [r.latency_ms for r in ok]
    toks = sum(r.output_tokens for r in ok)
    tpots = [(r.latency_ms / r.output_tokens) for r in ok if r.output_tokens > 0]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]

    print("\n" + "=" * 72)
    print(f"  {label} results")
    print("=" * 72)
    print(f"  wall time:       {wall_s:.3f} s")
    print(f"  requests:        {len(results)} total, {len(ok)} ok, {len(failed)} failed")
    print(f"  output tokens:   {toks}")
    print(f"  throughput:      {(toks / wall_s) if wall_s > 0 else 0:.1f} tok/s")
    if ok:
        print(f"  prompt words:    mean={statistics.mean(r.prompt_words for r in ok):.1f}, "
              f"min={min(r.prompt_words for r in ok)}, max={max(r.prompt_words for r in ok)}")
        print(f"  latency ms:      mean={statistics.mean(lat):.1f}, p50={percentile(lat, 0.50):.1f}, "
              f"p90={percentile(lat, 0.90):.1f}, p99={percentile(lat, 0.99):.1f}")
        print(f"  req tok/s:       mean={statistics.mean((r.output_tokens / (r.latency_ms / 1000.0)) for r in ok if r.latency_ms > 0):.1f}")
        if tpots:
            print(f"  latency/out tok: mean={statistics.mean(tpots):.3f} ms, p50={percentile(tpots, 0.50):.3f}, "
                  f"p90={percentile(tpots, 0.90):.3f}")
        if ttfts:
            print(f"  TTFT ms:         mean={statistics.mean(ttfts):.1f}, p50={percentile(ttfts, 0.50):.1f}, "
                  f"p90={percentile(ttfts, 0.90):.1f}")
    if failed:
        print("  failures:")
        for r in failed[:5]:
            print(f"    req-{r.request_id}: {r.error}")
    print("=" * 72)
    return {
        "ok": len(ok),
        "failed": len(failed),
        "tokens": toks,
        "throughput": toks / wall_s if wall_s > 0 else 0,
        "lat_mean": statistics.mean(lat) if lat else 0,
        "lat_p50": percentile(lat, 0.50),
        "lat_p90": percentile(lat, 0.90),
        "lat_p99": percentile(lat, 0.99),
    }


async def main_async(args):
    await health_check(args.url)

    warmup_specs = make_specs(args.warmup_requests, args.arrival_rate, args.max_tokens, args.seed + 1000)
    bench_specs = make_specs(args.num_requests, args.arrival_rate, args.max_tokens, args.seed)

    print(f"Service:        {args.label}")
    print(f"URL:            {args.url}")
    print(f"Model:          {args.model}")
    print(f"Warmup:         {args.warmup_requests} requests")
    print(f"Bench:          {args.num_requests} requests")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Arrival rate:   {args.arrival_rate:.2f} req/s, poisson inter-arrival")
    print(f"Max tokens:     {args.max_tokens}")
    print(f"Stream:         {args.stream}")
    print(f"Seed:           {args.seed}")

    if args.warmup_requests > 0:
        print("\nWarmup phase...")
        warmup_results, warmup_wall = await run_specs(
            args.url, args.model, warmup_specs, args.concurrency, args.stream, args.verbose
        )
        summarize(f"{args.label} warmup (not counted)", warmup_results, warmup_wall)

    print("\nBenchmark phase...")
    results, wall_s = await run_specs(
        args.url, args.model, bench_specs, args.concurrency, args.stream, args.verbose
    )
    summary = summarize(args.label, results, wall_s)
    print("JSON_SUMMARY " + json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Real-arrival OpenAI-compatible online benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3-0.6B")
    parser.add_argument("--label", default="server")
    parser.add_argument("--num-requests", type=int, default=80)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--arrival-rate", type=float, default=24.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
