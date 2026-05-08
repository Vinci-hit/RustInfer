"""
RustInfer Online Continuous Batching 压测脚本

使用方式:
  1. 启动三进程:
     # Terminal 1: Worker
     cargo run -p infer-worker --features cuda,models --bin rustinfer-worker -- \
       --model ~/models/Llama-3.2-1B-Instruct --device cuda:0 \
       --worker-pull-endpoint ipc:///tmp/rustinfer-worker-in.ipc \
       --worker-push-endpoint ipc:///tmp/rustinfer-worker-out.ipc

     # Terminal 2: Scheduler
     cargo run -p infer-scheduler -- \
       --frontend-endpoint ipc:///tmp/rustinfer.ipc \
       --worker-push-endpoint ipc:///tmp/rustinfer-worker-in.ipc \
       --worker-pull-endpoint ipc:///tmp/rustinfer-worker-out.ipc

     # Terminal 3: HTTP Server
     cargo run -p infer-server -- \
       --tokenizer ~/models/Llama-3.2-1B-Instruct \
       --engine-endpoint ipc:///tmp/rustinfer.ipc

  2. 运行压测:
     python bench/bench_online.py --url http://localhost:8000 --num-requests 20 --concurrency 4
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

# 多样化的 prompts（从数据集加载或使用默认）
PROMPTS = None  # 在 main 中加载


def load_prompts(dataset_path: str = None) -> list:
    """从 JSON 文件加载 prompts，或使用内置默认"""
    import os
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path) as f:
            return json.load(f)

    # 默认 fallback
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
        "How do you make a cup of coffee?",
        "Describe the process of photosynthesis.",
        "What are the benefits of regular exercise?",
        "Explain how a computer works to a five year old.",
        "What is the capital of Japan?",
    ]


@dataclass
class RequestResult:
    request_id: int
    prompt: str
    status: str
    num_tokens: int
    latency_ms: float
    tokens_per_second: float
    error: Optional[str] = None


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    request_id: int,
    max_tokens: int,
) -> RequestResult:
    """发送单个请求并测量延迟"""
    start = time.perf_counter()

    payload = {
        "model": "llama3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000
            body = await resp.json()

            if resp.status != 200:
                return RequestResult(
                    request_id=request_id,
                    prompt=prompt[:50],
                    status="error",
                    num_tokens=0,
                    latency_ms=elapsed_ms,
                    tokens_per_second=0,
                    error=str(body),
                )

            num_tokens = body.get("usage", {}).get("completion_tokens", 0)
            tps = num_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

            return RequestResult(
                request_id=request_id,
                prompt=prompt[:50],
                status="ok",
                num_tokens=num_tokens,
                latency_ms=elapsed_ms,
                tokens_per_second=tps,
            )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id,
            prompt=prompt[:50],
            status="error",
            num_tokens=0,
            latency_ms=elapsed_ms,
            tokens_per_second=0,
            error=str(e),
        )


async def run_benchmark(
    url: str,
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    arrival_rate: float,
):
    """运行 online 压测: 模拟请求以一定速率到达"""
    print(f"\n{'='*60}")
    print(f"  RustInfer Online Continuous Batching Benchmark")
    print(f"{'='*60}")
    print(f"  URL:          {url}")
    print(f"  Requests:     {num_requests}")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Max tokens:   {max_tokens}")
    print(f"  Arrival rate: {arrival_rate:.1f} req/s")
    print(f"{'='*60}\n")

    # 检查服务健康
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"ERROR: Server not healthy (status={resp.status})")
                    return
        except Exception as e:
            print(f"ERROR: Cannot connect to server: {e}")
            return

    print("Server is healthy, starting benchmark...\n")

    results: list = []
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.perf_counter()

    async def throttled_request(session, idx):
        async with semaphore:
            prompt = PROMPTS[idx % len(PROMPTS)]
            result = await send_request(session, url, prompt, idx, max_tokens)
            results.append(result)
            # 实时打印
            status_char = "✓" if result.status == "ok" else "✗"
            print(
                f"  [{status_char}] req-{idx:03d} | {result.num_tokens:3d} tokens | "
                f"{result.latency_ms:7.1f}ms | {result.tokens_per_second:5.1f} tok/s"
            )

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            task = asyncio.create_task(throttled_request(session, i))
            tasks.append(task)
            # 模拟 Poisson-like 到达间隔
            if arrival_rate > 0 and i < num_requests - 1:
                await asyncio.sleep(1.0 / arrival_rate)

        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time

    # ── 统计结果 ──
    successful = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Requests:         {len(results)} total, {len(successful)} ok, {len(failed)} failed")

    if successful:
        latencies = [r.latency_ms for r in successful]
        tokens = [r.num_tokens for r in successful]
        total_tokens = sum(tokens)
        throughput = total_tokens / total_time

        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p90 = latencies_sorted[int(len(latencies_sorted) * 0.9)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]

        print(f"  Total tokens:     {total_tokens}")
        print(f"  Throughput:       {throughput:.1f} tokens/s (system-wide)")
        print(f"  Avg tokens/req:   {total_tokens / len(successful):.1f}")
        print(f"  Latency (ms):     p50={p50:.0f}  p90={p90:.0f}  p99={p99:.0f}")
        print(f"  Avg latency:      {sum(latencies) / len(latencies):.0f}ms")
        print(f"  Avg per-req tps:  {sum(r.tokens_per_second for r in successful) / len(successful):.1f} tok/s")

    if failed:
        print(f"\n  Failures:")
        for r in failed[:5]:
            print(f"    req-{r.request_id}: {r.error}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="RustInfer Online Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--num-requests", "-n", type=int, default=100, help="Total requests")
    parser.add_argument("--concurrency", "-c", type=int, default=8, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens per request")
    parser.add_argument("--arrival-rate", type=float, default=20.0, help="Requests per second arrival rate (0=burst)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to prompts JSON file")
    args = parser.parse_args()

    global PROMPTS
    PROMPTS = load_prompts(args.dataset)
    print(f"Loaded {len(PROMPTS)} prompts from dataset")

    asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        arrival_rate=args.arrival_rate,
    ))


if __name__ == "__main__":
    main()
