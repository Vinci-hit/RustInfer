"""High-concurrency KV-cache pressure test for RustInfer.

Use this after lowering the KV cache size. The script sends waves of concurrent
long-prompt requests with long fixed-length decoding so the worker has to handle
KV pressure, preemption/relief, and recovery without hanging or crashing.

Usage:
    python3 bench/bench_kv_pressure.py --url http://127.0.0.1:8000
    python3 bench/bench_kv_pressure.py --concurrency 128 --waves 3
    python3 bench/bench_kv_pressure.py --dump-json /tmp/kv_pressure.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

try:
    import aiohttp
except ImportError as e:
    raise SystemExit("aiohttp is required; install it or run in the existing bench environment") from e


DEFAULT_URL = "http://127.0.0.1:8000"


@dataclass
class RequestResult:
    index: int
    wave: int
    status: int | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    finish_reason: str
    text_preview: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress RustInfer under KV-cache pressure")
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument(
        "--endpoint",
        choices=["completions", "chat"],
        default="completions",
        help="API endpoint to exercise. completions avoids chat-template variance.",
    )
    parser.add_argument("--model", default="qwen3", help="Model field sent in requests")
    parser.add_argument("--concurrency", type=int, default=64, help="Concurrent requests per wave")
    parser.add_argument("--waves", type=int, default=2, help="Number of pressure waves")
    parser.add_argument("--wave-gap", type=float, default=1.0, help="Seconds to sleep between waves")
    parser.add_argument("--prompt-lines", type=int, default=80, help="Long-prompt filler lines")
    parser.add_argument("--max-tokens", type=int, default=128, help="Decode tokens per request")
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force decode to max_tokens. Use --no-ignore-eos to allow early EOS.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout seconds")
    parser.add_argument("--warmup", type=int, default=1, help="Short warmup requests before pressure")
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help="Exit non-zero if failures/total is greater than this ratio",
    )
    parser.add_argument("--dump-json", default="", help="Write raw results to this JSON file")
    return parser.parse_args()


def endpoint_path(endpoint: str) -> str:
    if endpoint == "chat":
        return "/v1/chat/completions"
    return "/v1/completions"


def build_prompt(index: int, prompt_lines: int) -> str:
    # Put the unique marker first so prefix caching cannot hide KV pressure.
    req_id = f"kv-pressure-{index:06d}"
    rows = [
        f"{req_id} begins here.",
        "This request intentionally uses a unique long prompt to consume KV cache.",
        "Do not summarize; continue only when asked at the end.",
        "",
    ]
    for i in range(prompt_lines):
        rows.append(
            f"{req_id} filler row {i:04d}: this deterministic text is part of "
            "a high-concurrency KV-cache pressure scenario for RustInfer."
        )
    rows.append("")
    rows.append(f"{req_id} final instruction: answer with a short deterministic sentence.")
    return "\n".join(rows)


def make_payload(args: argparse.Namespace, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "stream": False,
        "ignore_eos": args.ignore_eos,
    }
    if args.endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    return payload


def extract_text_and_finish(endpoint: str, body: dict[str, Any]) -> tuple[str, str]:
    choices = body.get("choices") or []
    if not choices:
        return "", ""
    choice = choices[0] or {}
    finish_reason = str(choice.get("finish_reason") or "")
    if endpoint == "chat":
        text = ((choice.get("message") or {}).get("content") or "").strip()
    else:
        text = (choice.get("text") or "").strip()
    return text.replace("\n", "\\n")[:160], finish_reason


async def response_body(resp: aiohttp.ClientResponse) -> Any:
    try:
        return await resp.json(content_type=None)
    except Exception:
        return await resp.text()


async def send_one(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    index: int,
    wave: int,
    max_tokens: int,
) -> RequestResult:
    prompt = build_prompt(index, args.prompt_lines)
    payload = make_payload(args, prompt, max_tokens)
    start = time.perf_counter()
    try:
        async with session.post(endpoint_path(args.endpoint), json=payload) as resp:
            body = await response_body(resp)
        elapsed = time.perf_counter() - start
        if resp.status != 200:
            return RequestResult(
                index=index,
                wave=wave,
                status=resp.status,
                latency_s=elapsed,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                success=False,
                finish_reason="",
                text_preview="",
                error=str(body)[:500],
            )

        if not isinstance(body, dict):
            return RequestResult(index, wave, resp.status, elapsed, 0, 0, 0, False, "", "", str(body)[:500])

        usage = body.get("usage") or {}
        text_preview, finish_reason = extract_text_and_finish(args.endpoint, body)
        return RequestResult(
            index=index,
            wave=wave,
            status=resp.status,
            latency_s=elapsed,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            success=True,
            finish_reason=finish_reason,
            text_preview=text_preview,
            error="",
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return RequestResult(
            index=index,
            wave=wave,
            status=None,
            latency_s=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            success=False,
            finish_reason="",
            text_preview="",
            error=repr(e)[:500],
        )


async def run_health_check(session: aiohttp.ClientSession, args: argparse.Namespace, label: str) -> RequestResult:
    saved_ignore_eos = args.ignore_eos
    saved_prompt_lines = args.prompt_lines
    args.ignore_eos = False
    args.prompt_lines = 1
    try:
        result = await send_one(session, args, index=-1, wave=-1, max_tokens=1)
    finally:
        args.ignore_eos = saved_ignore_eos
        args.prompt_lines = saved_prompt_lines
    status = "OK" if result.success else "FAIL"
    print(f"{label} health: {status} status={result.status} latency={result.latency_s:.3f}s")
    if result.error:
        print(f"  health error: {result.error}")
    return result


async def run(args: argparse.Namespace) -> tuple[list[RequestResult], RequestResult, RequestResult]:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(
        base_url=args.url.rstrip("/"),
        timeout=timeout,
        connector=connector,
        trust_env=False,
    ) as session:
        before = await run_health_check(session, args, "Before")

        for i in range(args.warmup):
            result = await send_one(session, args, index=-(i + 2), wave=-1, max_tokens=1)
            print(
                f"warmup #{i:02d}: status={result.status} "
                f"success={result.success} latency={result.latency_s:.3f}s"
            )
            if not result.success:
                break

        results: list[RequestResult] = []
        next_index = 0
        for wave in range(args.waves):
            print()
            print(f"Starting wave {wave + 1}/{args.waves}: concurrency={args.concurrency}")
            wave_start = time.perf_counter()
            tasks = [
                asyncio.create_task(send_one(session, args, next_index + i, wave, args.max_tokens))
                for i in range(args.concurrency)
            ]
            wave_results = await asyncio.gather(*tasks)
            results.extend(wave_results)
            next_index += args.concurrency
            elapsed = time.perf_counter() - wave_start
            ok = sum(1 for r in wave_results if r.success)
            print(f"Wave {wave + 1} done: {ok}/{len(wave_results)} ok, wall={elapsed:.2f}s")
            if wave + 1 < args.waves and args.wave_gap > 0:
                await asyncio.sleep(args.wave_gap)

        after = await run_health_check(session, args, "After")
        return results, before, after


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, int((len(values) - 1) * pct))
    return sorted(values)[idx]


def summarize(results: list[RequestResult], before: RequestResult, after: RequestResult) -> dict[str, Any]:
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_s for r in ok]
    completion = [r.completion_tokens for r in ok]
    prompt = [r.prompt_tokens for r in ok]
    errors = Counter((r.status, r.error[:120]) for r in failed)

    stats: dict[str, Any] = {
        "total": len(results),
        "successful": len(ok),
        "failed": len(failed),
        "failure_rate": (len(failed) / len(results)) if results else 1.0,
        "health_before": before.success,
        "health_after": after.success,
    }
    if latencies:
        stats.update(
            {
                "latency_s_min": min(latencies),
                "latency_s_p50": percentile(latencies, 0.50),
                "latency_s_p90": percentile(latencies, 0.90),
                "latency_s_p99": percentile(latencies, 0.99),
                "latency_s_max": max(latencies),
                "latency_s_mean": statistics.mean(latencies),
                "completion_tokens_mean": statistics.mean(completion),
                "prompt_tokens_mean": statistics.mean(prompt),
            }
        )

    print()
    print("=" * 72)
    print("KV pressure summary")
    print("=" * 72)
    print(f"Requests:      {len(ok)} ok / {len(failed)} failed / {len(results)} total")
    print(f"Failure rate:  {stats['failure_rate']:.2%}")
    print(f"Health check:  before={before.success} after={after.success}")
    if latencies:
        print(
            "Latency:       "
            f"p50={stats['latency_s_p50']:.2f}s "
            f"p90={stats['latency_s_p90']:.2f}s "
            f"p99={stats['latency_s_p99']:.2f}s "
            f"max={stats['latency_s_max']:.2f}s"
        )
        print(
            "Tokens:        "
            f"prompt_avg={stats['prompt_tokens_mean']:.0f} "
            f"completion_avg={stats['completion_tokens_mean']:.0f}"
        )
    if failed:
        print()
        print("Top failures:")
        for (status, error), count in errors.most_common(5):
            print(f"  count={count} status={status} error={error}")

    return stats


def validate_args(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.waves < 1:
        raise SystemExit("--waves must be at least 1")
    if args.prompt_lines < 1:
        raise SystemExit("--prompt-lines must be at least 1")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be at least 1")
    if not (0.0 <= args.max_failure_rate <= 1.0):
        raise SystemExit("--max-failure-rate must be between 0 and 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"Target:       {args.url}{endpoint_path(args.endpoint)}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Waves:        {args.waves}, gap={args.wave_gap}s")
    print(f"Prompt lines: {args.prompt_lines}")
    print(f"Max tokens:   {args.max_tokens}, ignore_eos={args.ignore_eos}")
    print(f"Timeout:      {args.timeout}s")

    started = time.perf_counter()
    results, before, after = asyncio.run(run(args))
    wall = time.perf_counter() - started
    stats = summarize(results, before, after)
    stats["wall_s"] = wall
    print(f"Wall time:     {wall:.2f}s")

    if args.dump_json:
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": vars(args),
                    "stats": stats,
                    "health_before": asdict(before),
                    "health_after": asdict(after),
                    "results": [asdict(r) for r in results],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nWrote JSON: {args.dump_json}")

    if not before.success or not after.success:
        raise SystemExit(1)
    if stats["failure_rate"] > args.max_failure_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
