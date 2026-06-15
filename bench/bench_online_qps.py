"""Online serving benchmark with Poisson arrival (QPS rate).

Aligns with vLLM / SGLang official `benchmark_serving.py`:
  - Poisson arrival process (exponential inter-arrival times)
  - Sweeps QPS rates: --qps 1,2,4,8,16,32
  - Each request: ignore_eos=True, max_tokens=512 (fixed work)
  - Records: TTFT, TPOT (per-token latency), ITL (inter-token latency)
  - Same prompt sequence for both targets (no shuffle, round-robin)

Usage:
    python3 bench/bench_online_qps.py --tag rustinfer --qps 1,2,4,8,16,32
    python3 bench/bench_online_qps.py --tag vllm --qps 1,2,4,8,16,32

Outputs:
    /tmp/bench_qps_rustinfer.json
    /tmp/bench_qps_vllm.json
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import aiohttp
import numpy as np


@dataclass
class RequestResult:
    """Matches sglang RequestFuncOutput structure."""
    prompt: str
    generated_text: str = ""
    output_len: List[int] = field(default_factory=list)  # len per turn
    prompt_len: List[int] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)  # e2e per turn
    ttft: List[float] = field(default_factory=list)  # time to first token per turn
    itl: List[float] = field(default_factory=list)  # inter-token latencies
    success: bool = False
    error: str = ""


RUSTINFER_MODEL = "llama3.2-1b"
URL = "http://127.0.0.1:8000"
DURATION = 60
PROMPTS_FILE = "/root/RustInfer/bench/bench_prompts.json"
MAX_TOKENS = 512


async def _send_streaming(session, url, prompt, payload):
    """Shared streaming logic for both rustinfer and vllm."""
    start = time.perf_counter()
    result = RequestResult(prompt=prompt[:100])
    generated_text = ""
    output_token_count = 0
    usage_seen = False
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                try:
                    body = await resp.json()
                except Exception:
                    body = await resp.text()
                result.success = False
                result.error = f"HTTP {resp.status}: {body}"
                return result
            first_token_time = None
            most_recent_timestamp = start
            async for chunk_bytes in resp.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = chunk_bytes.decode("utf-8")
                if chunk.startswith("data: "):
                    chunk = chunk[6:]
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                # Usage may arrive in last chunk (only if server supports include_usage)
                if data.get("usage"):
                    usage = data["usage"]
                    result.prompt_len = [usage.get("prompt_tokens", 0)]
                    result.output_len = [usage.get("completion_tokens", output_token_count)]
                    result.latency = [now - start]
                    result.success = True
                    usage_seen = True
                # Process content delta
                if data.get("choices"):
                    delta = data["choices"][0].get("delta", {}) or {}
                    content = delta.get("content")
                    if content:
                        if first_token_time is None:
                            first_token_time = now
                            result.ttft = [now - start]
                        else:
                            # ITL = time since previous content token
                            result.itl.append(now - most_recent_timestamp)
                        most_recent_timestamp = now
                        generated_text += content
                        output_token_count += 1
            # Fallback when server didn't emit usage chunk: use observed token count
            if not usage_seen and first_token_time is not None:
                result.prompt_len = [0]
                result.output_len = [output_token_count]
                result.latency = [time.perf_counter() - start]
                result.success = True
            elif not result.success:
                result.latency = [time.perf_counter() - start]
                if first_token_time is None:
                    result.error = result.error or "no tokens received"
            result.generated_text = generated_text
    except Exception as e:
        result.success = False
        result.error = f"{type(e).__name__}: {e}"
        if not result.latency:
            result.latency = [time.perf_counter() - start]
    return result


async def send_rustinfer(session, url, prompt):
    payload = {
        "model": RUSTINFER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "ignore_eos": True,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return await _send_streaming(session, url, prompt, payload)


async def send_vllm(session, url, prompt):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "ignore_eos": True,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return await _send_streaming(session, url, prompt, payload)


async def get_requests(prompts, qps, num_requests):
    """Async generator: yield (prompt, idx) at Poisson intervals."""
    for i in range(num_requests):
        if i > 0:
            interval = np.random.exponential(1.0 / qps)
            await asyncio.sleep(interval)
        yield prompts[i % len(prompts)], i


async def benchmark(url, prompts, qps, duration_s, send_fn, max_concurrency=None):
    """Run one QPS rate benchmark. Returns list of RequestResult.

    max_concurrency=None → unlimited (sglang default). Set to an int to cap.
    """
    results: List[RequestResult] = []
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_send(prompt, idx):
        if semaphore is None:
            result = await send_fn(session, url, prompt)
        else:
            async with semaphore:
                result = await send_fn(session, url, prompt)
        results.append(result)

    num_requests = int(qps * duration_s * 1.5)  # headroom

    async with aiohttp.ClientSession() as session:
        # Warmup: 2 requests
        print("  Warming up (2 requests)...")
        warmup_prompt = prompts[0]
        await send_fn(session, url, warmup_prompt)
        await send_fn(session, url, warmup_prompt)
        print(f"  Warmup done. Starting benchmark (qps={qps})...")

        start_time = time.perf_counter()
        tasks = []
        async for prompt, idx in get_requests(prompts, qps, num_requests):
            elapsed = time.perf_counter() - start_time
            if elapsed > duration_s:
                break
            tasks.append(asyncio.create_task(limited_send(prompt, idx)))
        # Wait for all launched tasks
        if tasks:
            await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - start_time

    return results, benchmark_duration


def calculate_metrics(results, duration_s, qps):
    """Mirror sglang calculate_metrics."""
    output_lens = []
    total_input = 0
    completed = 0
    failed = 0
    itls = []
    tpots = []
    ttfts = []
    e2e_latencies = []

    for r in results:
        if r.success:
            completed += 1
            # One iteration per turn (single-turn → 1 iteration)
            num_turns = max(len(r.output_len), len(r.latency))
            for j in range(num_turns):
                output_len = r.output_len[j] if j < len(r.output_len) else 0
                output_lens.append(output_len)
                total_input += r.prompt_len[j] if j < len(r.prompt_len) else 0
                if output_len > 1 and j < len(r.latency) and j < len(r.ttft):
                    # TPOT = (latency - ttft) / (output_len - 1)
                    tpot = (r.latency[j] - r.ttft[j]) / (output_len - 1)
                    tpots.append(tpot)
            itls.extend(r.itl)
            ttfts.extend(r.ttft)
            e2e_latencies.extend(r.latency)
        else:
            failed += 1
            output_lens.append(0)

    if completed == 0:
        # Surface the first few errors so failures are diagnosable
        sample_errors = [r.error for r in results if not r.success][:3]
        print(f"  WARNING: All requests failed! ({failed} failures)")
        for err in sample_errors:
            print(f"    err: {err}")
        return {}

    def _pct(xs, p):
        return float(np.percentile(xs, p)) if xs else 0.0

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    def _median(xs):
        return float(np.median(xs)) if xs else 0.0

    def _std(xs):
        return float(np.std(xs)) if xs else 0.0

    metrics = {
        "qps": qps,
        "completed": completed,
        "failed": failed,
        "total_input": total_input,
        "total_output": sum(output_lens),
        "request_throughput": completed / duration_s,
        "input_throughput": total_input / duration_s,
        "output_throughput": sum(output_lens) / duration_s,
        "mean_ttft_ms": _mean(ttfts) * 1000,
        "median_ttft_ms": _median(ttfts) * 1000,
        "std_ttft_ms": _std(ttfts) * 1000,
        "p90_ttft_ms": _pct(ttfts, 90) * 1000,
        "p99_ttft_ms": _pct(ttfts, 99) * 1000,
        "mean_tpot_ms": _mean(tpots) * 1000,
        "median_tpot_ms": _median(tpots) * 1000,
        "std_tpot_ms": _std(tpots) * 1000,
        "p90_tpot_ms": _pct(tpots, 90) * 1000,
        "p99_tpot_ms": _pct(tpots, 99) * 1000,
        "mean_itl_ms": _mean(itls) * 1000,
        "median_itl_ms": _median(itls) * 1000,
        "std_itl_ms": _std(itls) * 1000,
        "p90_itl_ms": _pct(itls, 90) * 1000,
        "p99_itl_ms": _pct(itls, 99) * 1000,
        "mean_e2e_latency_ms": _mean(e2e_latencies) * 1000,
        "median_e2e_latency_ms": _median(e2e_latencies) * 1000,
        "std_e2e_latency_ms": _std(e2e_latencies) * 1000,
        "p99_e2e_latency_ms": _pct(e2e_latencies, 99) * 1000,
        "concurrency": (sum(e2e_latencies) / duration_s) if e2e_latencies else 0.0,
    }

    print(f"  QPS={qps}: completed={completed} failed={failed} "
          f"throughput={metrics['output_throughput']:.0f} tok/s "
          f"TTFTp50={_pct(ttfts, 50)*1000:.1f}ms "
          f"TPOTp50={_pct(tpots, 50)*1000:.1f}ms "
          f"ITLp50={_pct(itls, 50)*1000:.1f}ms")
    return metrics


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["rustinfer", "vllm"],
                     help="Which target is running on :8000")
    ap.add_argument("--url", default=URL, help="Target URL")
    ap.add_argument("--qps", default="1,2,4,8,16,32",
                     help="Comma-separated QPS rates to sweep")
    ap.add_argument("--duration", type=int, default=DURATION)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--max-concurrency", type=int, default=None,
                    help="Cap in-flight requests. Default unlimited (sglang default).")
    args = ap.parse_args()

    qps_rates = [float(x) for x in args.qps.split(",")]
    with open(PROMPTS_FILE) as f:
        pool = json.load(f)
    prompts = [p for p in pool if isinstance(p, str) and 40 <= len(p) <= 500]
    print(f"Loaded {len(prompts)} prompts")
    print(f"Tag: {args.tag}, URL: {args.url}, QPS sweep: {qps_rates}, Duration: {args.duration}s")

    send_fn = send_rustinfer if args.tag == "rustinfer" else send_vllm
    all_metrics = []

    for qps in qps_rates:
        print(f"\n{'=' * 60}")
        print(f"  QPS={qps} — {args.tag}")
        print(f"{'=' * 60}")
        results, bench_dur = await benchmark(
            args.url, prompts, qps, args.duration, send_fn,
            max_concurrency=args.max_concurrency,
        )
        metrics = calculate_metrics(results, bench_dur, qps)
        all_metrics.append(metrics)

    # Save all results
    out = f"/tmp/bench_qps_{args.tag}.json"
    with open(out, "w") as f:
        json.dump({
            "config": {
                "tag": args.tag, "url": args.url,
                "qps_rates": qps_rates, "duration": args.duration,
                "max_tokens": args.max_tokens,
            },
            "results": all_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: {args.tag}")
    print(f"{'=' * 80}")
    print(f"{'QPS':>6} {'req/s':>8} {'tok/s':>10} {'TTFTp50':>10} {'TPOTp50':>10} {'ITLp50':>10}")
    for qps, m in zip(qps_rates, all_metrics):
        if not m:
            continue
        print(f"{qps:>6.1f} {m['request_throughput']:>8.1f} {m['output_throughput']:>10.0f} "
              f"{m['median_ttft_ms']:>10.1f} {m['median_tpot_ms']:>10.1f} {m['median_itl_ms']:>10.1f}")


if __name__ == "__main__":
    asyncio.run(main())
